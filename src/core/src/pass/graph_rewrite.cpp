// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/graph_rewrite.hpp"

#include <algorithm>
#include <deque>
#include <iostream>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <regex>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "perf_counters.hpp"
#include "ngraph/pass/visualize_tree.hpp"

/* GraphRewrite algorithm:
 * GraphRewrite processes an input graph in an topological order(i.e. args before users)
 * Given the following graph:          Abs2
 *                                   /       \
 *                         Constant1         Add4 - Result5
 *                                   \      /
 *                                    Neg3
 *
 * The topological order would be : `Constant1`, `Abs2`, `Neg3`, `Add4`, `Result5`
 * Note, `Abs2` comes before `Neg3` as `Abs2`'s id = 2 is *less* than `Neg3`'s one (id = 3)
 * Next, GraphRewrite will invoke matchers passes registered in add_matcher order.
 * For example:
 *     ov::pass::GraphRewrite pass;
 *     pass.add_matcher<m1>();
 *     pass.add_matcher<m2>();
 *     pass.add_matcher<m3>();
 * Matcher passes will be called as follows: `m1`, `m2`, `m3`
 * Matchers should only replace nodes in the graph that come before the current root
 * node in the topological order. For example, if Matcher matches Neg3, it should only
 * replace nodes `Abs2` and `Constant1` if needed
 * This gives Matchers a nice cascading property. For example, if m1 folds `Abs2(Constant1)`
 * and `m2` folds `Neg3(Constant1)` when `m3` is called on `Add4` it will discover that
 * both `Abs2` and `Neg3` were already replaced by constants, so `Add4` will also be folded into
 * one.
 * If any matcher passes succeeds the rest of the matchers will **not** be called.
 * E.g. if `m1` succeeds and replaces `Abs2` with a new constant, nor `m2` or `m3` will be called
 * However, sometimes, you will need more than one fusion occur on the same node.
 * In this case, you need to register nodes in MatcherPass manually using register_new_node method.
 * GraphRewrite will automatically add this nodes in the beginning of execution queue.
 * If MatcherPass register more than one node make sure that this nodes are registered in
 * topological order. */

namespace ov {
namespace pass {
namespace {
PerfCounters& perf_counters_graph_rewrite() {
    static PerfCounters counters;
    return counters;
}
}  // namespace
}  // namespace pass
}  // namespace ov

bool ov::pass::BackwardGraphRewrite::run_on_model(const std::shared_ptr<ov::Model>& f) {
    // Initialize execution queue with nodes in topological order
    std::deque<std::weak_ptr<Node>> nodes_to_run;
    for (auto& node : f->get_ordered_ops()) {
        nodes_to_run.emplace_front(node);
    }
    return apply_matcher_passes(f, std::move(nodes_to_run));
}

bool ov::pass::GraphRewrite::run_on_model(const std::shared_ptr<ov::Model>& f) {
    // Initialize execution queue with nodes in topological order
    std::deque<std::weak_ptr<Node>> nodes_to_run;
    for (auto& node : f->get_ordered_ops()) {
        nodes_to_run.emplace_back(node);
    }
    return apply_matcher_passes(f, std::move(nodes_to_run));
}

// static size_t g_index = 0;
bool ov::pass::GraphRewrite::apply_matcher_passes(std::shared_ptr<Model> f,
                                                  std::deque<std::weak_ptr<Node>> nodes_to_run) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::nGraph, "pass::GraphRewrite::run_on_function");

    bool rewritten = false;
    const auto& pass_config = get_pass_config();
    std::vector<std::string> key_array;
    key_array.push_back("Multiply_26858");
    key_array.push_back("Multiply_29378");
    key_array.push_back("GlobalAveragePool_201/reduce");
    key_array.push_back("Conv_202/WithoutBiases/fq_input_0");

    std::vector<std::string> find_key;
    find_key.push_back("Multiply_26858");
    find_key.push_back("Multiply_29378");

    // Check that all Matchers in MatcherPasses has type bases root node
    bool all_roots_has_type = true;
    std::unordered_map<NodeTypeInfo, std::vector<size_t>> type_to_matcher;
    for (size_t matcher_index = 0; matcher_index < m_matchers.size(); ++matcher_index) {
        // Skip passes that are disabled
        if (pass_config->is_disabled(m_matchers[matcher_index]->get_type_info()))
            continue;

        auto matcher = m_matchers[matcher_index]->get_matcher();
        if (!matcher) {
            all_roots_has_type = false;
            break;
        }

        auto root = matcher->get_pattern_value().get_node_shared_ptr();
        // pattern::op::AnyOutput operation automatically appends for multi output operations inside
        // Matcher and to gen actual root node we need to take it's parent.
        if (auto any_type = std::dynamic_pointer_cast<pattern::op::AnyOutput>(root)) {
            root = any_type->input_value(0).get_node_shared_ptr();
        }

        // if root is an operation from opset or has pattern::op::WrapType type then we can extract
        // it's type
        // and use it in unordered_map as key for fast MatcherPass search. Otherwise type is unknown
        // and default algorithm is used.
        if (auto p = std::dynamic_pointer_cast<pattern::op::Pattern>(root)) {
            if (auto any_type = std::dynamic_pointer_cast<pattern::op::WrapType>(p)) {
                for (const auto& root_type_info : any_type->get_wrapped_types()) {
                    type_to_matcher[root_type_info].push_back(matcher_index);
                }
            } else {
                all_roots_has_type = false;
                break;
            }
        } else {
            type_to_matcher[root->get_type_info()].push_back(matcher_index);
        }

        // TODO: traverse parents for root_type_info in order to register complete list of matchers
        // including ones triggered by parent type info.
    }

    auto find_node = [&](std::shared_ptr<MatcherPass> m_pass, std::shared_ptr<Node> node, std::string title) -> bool {
        // {
        //     for (auto key : find_key) {
        //         if (node->get_friendly_name() == key) {
        //             std::cout << title << "] Found " << key << " in pass: " << m_pass->get_name() << std::endl;
        //             return true;
        //         }
        //     }
        // }

        // if (node->get_friendly_name() == "GlobalAveragePool_201/reduce") {
        //     for (auto n : f->get_ordered_ops()) {
        //         for (auto key : find_key) {
        //             if (n->get_friendly_name() == key) {
        //                 std::cout << title << "] Found " << key << " in pass: " << m_pass->get_name() << std::endl;
        //                 return true;
        //             }
        //         }
        //     }
        // }
        return false;
    };

    auto find_node_simple = [&](std::shared_ptr<MatcherPass> m_pass, bool is_tracking, std::string title) -> bool {
        // if (is_tracking) {
        //     for (auto n : f->get_ordered_ops()) {
        //         for (auto key : find_key) {
        //             if (n->get_friendly_name() == key) {
        //                 // std::cout << title << "] Found " << key << " in pass: " << m_pass->get_name() << std::endl;
        //                 return true;
        //             }
        //         }
        //     }
        //     // std::cout << title << "] No Found in pass: " << m_pass->get_name() << std::endl;
        // }
        return false;
    };

    auto find_qt = [&](std::shared_ptr<MatcherPass> m_pass) -> bool {
        // for (auto n : f->get_ordered_ops()) {
        //     if (n->get_friendly_name() == "Conv_202/WithoutBiases/fq_input_0") {
        //         return true;
        //     }
        // }
        return false;
    };

    auto dump_matcher = [&](std::shared_ptr<MatcherPass> m_pass, std::shared_ptr<Node> node, std::string title) {
        // std::vector<std::string> dump_array;
        // dump_array.push_back("GlobalAveragePool_201/reduce");
        // dump_array.push_back("Conv_202/WithoutBiases/fq_input_0");
        // bool is_found = false;
        // for (auto key_node : dump_array) {
        //     if (node->get_friendly_name() == key_node) {
        //         is_found = true;
        //         break;
        //     }
        // }
        // // FakeQuantizeTransformation][Conv_202/WithoutBiases/fq_input_0
        // std::vector<std::string> pass_key_array ={
        //     "ReduceMeanTransformation",
        //     "FakeQuantizeTransformation"
        // };

        // bool is_found_pass = false;
        // for (auto key_pass_name : pass_key_array) {
        //     if (m_pass->get_name() == key_pass_name) {
        //         is_found_pass = true;
        //         break;
        //     }
        // }

        // if (is_found && is_found_pass) {
        //     const size_t num_digits_in_pass_index = 3;
        //     std::string index_str = std::to_string(g_index++);
        //     index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
        //     auto name_str = std::regex_replace(node->get_friendly_name(), std::regex("/"), "_");
        //     auto base_filename = name_str + std::string("_") + index_str + std::string("_");
        //     base_filename += title  + std::string("_") + m_pass->get_name() + std::string(".svg");
        //     ov::pass::VisualizeTree vt(base_filename);
        //     // vt.run_on_model(f);
        //     std::cout << "Dump matcher : " << base_filename << std::endl;
        // }
    };


    // This lambda preforms execution of particular MatcherPass on given node.
    // It automatically handles nodes registered by MatcherPass during transformation and set
    // transformation callback.
    auto run_matcher_pass = [&](std::shared_ptr<MatcherPass> m_pass, std::shared_ptr<Node> node) -> bool {
        // Keep this property check for backward compatibility. In future transformation property
        // will be deprecated and removed.
        if (m_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && f->is_dynamic()) {
            NGRAPH_DEBUG << "matcher callback requires static shape but the "
                            "function is dynamic, skipping this "
                            "optimization till the shapes are fully "
                            "materialized";
            return false;
        }

        // Apply MatcherPass. In case if it returns true no other MatcherPasses will apply
        // to this node
        bool is_found = find_node(m_pass, node, "before pass is applied");
        dump_matcher(m_pass, node, "before");
        bool status = m_pass->apply(node);
        dump_matcher(m_pass, node, "after");
        if (is_found != find_node(m_pass, node, "after pass is applied")) {
            if (is_found)
                std::cout << m_pass->get_name() << " removed Multiply_* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            else
                std::cout << m_pass->get_name() << " created Multiply_* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        }

        // In case if MatcherPass registered nodes they will be added to the beginning of execution
        // queue
        const auto& new_nodes = m_pass->get_new_nodes();
        if (!new_nodes.empty()) {
            // Need to push nodes in reverse order as we expect that nodes in new_nodes
            // vector are in topological order
            for (auto it = new_nodes.rbegin(); it != new_nodes.rend(); it++) {
                nodes_to_run.emplace_front(*it);
            }
            m_pass->clear_new_nodes();
        }
        find_node(m_pass, node, "final checking ");
        return status;
    };

    // list of matchers to run for a node; define here to keep memory allocated
    std::vector<size_t> matcher_passes_to_run;

    bool is_tracking = false;
    while (!nodes_to_run.empty()) {
        auto weak_node = nodes_to_run.front();
        nodes_to_run.pop_front();

        auto node = weak_node.lock();
        if (!node)
            continue;
        {
            is_tracking = false;
            for (auto key : key_array) {
                if (node->get_friendly_name() == key) {
                    is_tracking = true;
                    break;
                }
            }
        }

        // Recursive apply Matchers for sub-graph based nodes
        if (auto sub_graph_node = std::dynamic_pointer_cast<ngraph::op::util::MultiSubGraphOp>(node)) {
            size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
            for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                auto sub_graph = sub_graph_node->get_function(sub_graph_ind);
                run_on_model(sub_graph);
            }
        }
        // Temporary keep this GraphRewrite property for backward compatibility
        if (m_enable_shape_inference) {
            node->revalidate_and_infer_types();
        }
        // If all Matchers in MatcherPasses has type based root node then we apply efficient
        // algorithm for finding matchers
        if (all_roots_has_type) {
            const DiscreteTypeInfo* node_type_info = &node->get_type_info();
            matcher_passes_to_run.clear();
            while (node_type_info) {
                auto matchers = type_to_matcher.find(*node_type_info);
                if (matchers != type_to_matcher.end()) {
                    // do not run found matchers immediately, need to collect all matchers for
                    // parents
                    // and sort them in order of the registration
                    matcher_passes_to_run.insert(matcher_passes_to_run.end(),
                                                 matchers->second.begin(),
                                                 matchers->second.end());
                }
                node_type_info = node_type_info->parent;
            }

            std::sort(matcher_passes_to_run.begin(), matcher_passes_to_run.end());

            // TODO: type_to_matcher with just collected list of matchers to enable
            // fast processing at the next time when node with the same type will be processed

            for (size_t matcher_index : matcher_passes_to_run) {
                if ("GlobalAveragePool_201/reduce" == node->get_friendly_name()) {
                    std::cout << "Call matcher case01 : " << m_matchers[matcher_index]->get_name() << std::endl;
                }
                if (run_matcher_pass(m_matchers[matcher_index], node)) {
                    rewritten = true;
                    break;
                }
            }
        }
        // Otherwise we use default algorithm that iterates over all registered matcher passes
        else {
            for (auto& m_pass : m_matchers) {
                // Skip passes that are disabled
                if (pass_config->is_disabled(m_pass->get_type_info()))
                    continue;

                // if (is_tracking) {
                //     std::cout << " [[" << node->get_friendly_name() << "]] Before call matcher case02 : " << m_pass->get_name() << std::endl;
                // }
                bool ret = find_node_simple(m_pass, is_tracking, "before matcher ");
                bool is_found = find_qt(m_pass);
                if (run_matcher_pass(m_pass, node)) {
                    rewritten = true;
                    // if (is_tracking) {
                    //     std::cout << " [[" << node->get_friendly_name() << "]] After call matcher case02 return true and break : " << m_pass->get_name() << std::endl;
                    // }
                    if (ret != find_node_simple(m_pass, is_tracking, "after matcher and breaking ")) {
                        if (ret) {
                            std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Mul is removed after matcher is running and break" << std::endl;
                        } else {
                            std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Mul is created after matcher is running and break" << std::endl;
                        }
                    }
                    if (is_found != find_qt(m_pass)) {
                        if (ret) {
                            std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Conv_202/WithoutBiases/fq_input_0 is removed after matcher is running and break" << std::endl;
                        } else {
                            std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Conv_202/WithoutBiases/fq_input_0 is created after matcher is running and break" << std::endl;
                        }
                    }
                    break;
                }
                // if (is_tracking) {
                //     std::cout << " [[" << node->get_friendly_name() << "]]  After call matcher and continue loop : " << m_pass->get_name() << std::endl;
                // }
                // find_node_simple(m_pass, is_tracking, "after matcher ");
                if (ret != find_node_simple(m_pass, is_tracking, "after matcher")) {
                    if (ret) {
                        std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Mul is removed after matcher is running " << std::endl;
                    } else {
                        std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Mul is created after matcher is running" << std::endl;
                    }
                }
                if (is_found != find_qt(m_pass)) {
                    if (ret) {
                        std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Conv_202/WithoutBiases/fq_input_0 is removed after matcher is running" << std::endl;
                    } else {
                        std::cout << "[" << m_pass->get_name() << "][" << node->get_friendly_name() << "] Conv_202/WithoutBiases/fq_input_0 is created after matcher is running" << std::endl;
                    }
                }
            }
        }
    }
    return rewritten;
}

void ov::pass::GraphRewrite::add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                                         const graph_rewrite_callback& callback,
                                         const PassPropertyMask& property) {
    m_matchers.push_back(std::make_shared<MatcherPass>(
        m->get_name(),
        m,
        [m, callback](const std::shared_ptr<Node>& node) -> bool {
            NGRAPH_DEBUG << "Running matcher " << m->get_name() << " on " << node;
            if (m->match(node->output(0))) {
                NGRAPH_DEBUG << "Matcher " << m->get_name() << " matched " << node;
                NGRAPH_PASS_CALLBACK(m);
                bool status = callback(*m.get());
                // explicitly clear Matcher state because it holds pointers to matched nodes
                m->clear_state();
                return status;
            }
            m->clear_state();
            return false;
        },
        property));
}

void ov::pass::GraphRewrite::add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                                         const graph_rewrite_callback& callback) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    // TODO: before deprecate this function, by default expect the
    // callback require static shape.
    add_matcher(m, callback, {PassProperty::REQUIRE_STATIC_SHAPE});
    NGRAPH_SUPPRESS_DEPRECATED_END
}

void ov::pass::GraphRewrite::set_pass_config(const std::shared_ptr<PassConfig>& rhs) {
    auto pass_config = get_pass_config();
    // We have to preserve disabled passes because in case when we register matchers inside
    // GraphRewrite c-tor we work with local PassConfig instance.
    // For example:
    //
    // class ExampleGraphRewrite: public pass::GraphRewrite {
    //      ExampleGraphRewrite() {
    //          add_mather<TestMatcher1, false /* disabled by default */>();
    //          add_mather<TestMatcher2>();
    //      }
    // };
    //
    // When we call add_matcher inside c-tor we automatically work with locally created PassConfig
    // instance that is not shared. So when instance of this pass is being created in pass::Manager
    // we set shared PassConfig but we will override already existing rules inside local config. To
    // resolve this we have to copy disabled passes from local PassConfig to shared but we take into
    // account that if passes were manually enabled we do not add them.
    rhs->add_disabled_passes(*pass_config);
    PassBase::set_pass_config(rhs);

    // update nested transformations with new shared pass_config
    for (auto& pass : m_matchers) {
        pass->set_pass_config(rhs);
    }
}

void ov::pass::RecurrentGraphRewrite::add_matcher(const std::shared_ptr<pattern::RecurrentMatcher>& m,
                                                  const ov::recurrent_graph_rewrite_callback& callback,
                                                  const PassPropertyMask& property) {
    m_matchers.push_back(std::make_shared<MatcherPass>(
        "Recurrent matcher",
        nullptr,
        [m, callback](const std::shared_ptr<Node>& node) {
            NGRAPH_DEBUG << "Running recurrent matcher on " << node;
            if (m->match(node->output(0))) {
                NGRAPH_DEBUG << "Recurrent matcher matched " << m.get();
                return callback(*m.get());
            }
            return false;
        },
        property));
}

void ov::pass::RecurrentGraphRewrite::add_matcher(const std::shared_ptr<pattern::RecurrentMatcher>& m,
                                                  const ov::recurrent_graph_rewrite_callback& callback) {
    // TODO: before deprecate this function, by default expect the
    // callback require static shape.
    add_matcher(m, callback, {PassProperty::REQUIRE_STATIC_SHAPE});
}

bool ov::pass::RecurrentGraphRewrite::run_on_model(const std::shared_ptr<Model>& f) {
    bool changed = false;
    size_t i = 0;

    // This check is very expensive and is only needed for experimental features, so we will hide
    // it behind an environment variable for now. TODO: Find a less expensive way to handle this.
    static bool s_rerun_dynamic_check = ngraph::getenv_bool("NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK");

    auto run_matchers = [&]() -> bool {
        bool is_dyn_func = s_rerun_dynamic_check && f->is_dynamic();
        for (const auto& node : f->get_ops()) {
            for (auto& m_pass : m_matchers) {
                if (is_dyn_func && m_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE)) {
                    NGRAPH_DEBUG << "matcher callback requires static shape but the "
                                    "function is dynamic, skipping this "
                                    "optimization till the shapes are fully "
                                    "materialized";
                    continue;
                }
                if (m_pass->apply(node)) {
                    // If call back may change function's is_dynamic state, we need to
                    // update the cached value.
                    if (m_pass->get_property(PassProperty::CHANGE_DYNAMIC_STATE)) {
                        is_dyn_func = s_rerun_dynamic_check && f->is_dynamic();
                    }
                    return true;
                }
            }
        }
        return false;
    };

    do {
        changed = run_matchers();
        i++;
    } while (changed && i < m_num_iters);
    return changed;
}

void ov::pass::MatcherPass::register_matcher(const std::shared_ptr<ov::pass::pattern::Matcher>& m,
                                             const ov::graph_rewrite_callback& callback,
                                             const PassPropertyMask& property) {
    set_name(m->get_name());
    set_property(property, true);
    m_matcher = m;
    m_handler = [m, callback](const std::shared_ptr<Node>& node) -> bool {
        if (m->match(node->output(0))) {
            NGRAPH_DEBUG << "Matcher " << m->get_name() << " matched " << node;
            NGRAPH_PASS_CALLBACK(m);
            bool status = callback(*m.get());
            // explicitly clear Matcher state because it holds pointers to matched nodes
            m->clear_state();
            return status;
        }
        m->clear_state();
        return false;
    };
}

bool ov::pass::MatcherPass::apply(std::shared_ptr<ov::Node> node) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::nGraph, pass::perf_counters_graph_rewrite()[get_type_info()]);
    m_new_nodes.clear();
    if (m_handler)
        return m_handler(node);
    return false;
}
