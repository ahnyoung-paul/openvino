// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/constant_folding.hpp"
#include <ngraph/op/constant.hpp>
#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/rt_info.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConstantFolding, "ConstantFolding", 0);

static size_t check_idx = 0;
static size_t check_loops(std::shared_ptr<ngraph::Function> func, std::string comments, size_t idx) {
    size_t count = 0;
    for (auto& op : func->get_ops()) {
        std::string type_str(op->get_type_name());
        if (type_str == "Loop" || type_str == "TensorIterator") {
            count++;
        }
        // std::cout << op->get_name() << " : " << op->get_type_name() << std::endl;
    }
    size_t count2 = 0;
    for (const auto& op : func->get_ordered_ops()) {
        std::string type_str(op->get_type_name());
        if (type_str == "Loop" || type_str == "TensorIterator") {
            count2++;
        }
    }
    std::cout << "[" << std::to_string(idx) << "-" << comments << "] Number of Loop: (" << count << "/" << count2 << ") sizeof ops : " << func->get_ops().size() << std::endl;
    return count;
}

bool ngraph::pass::ConstantFolding::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    size_t curr_idx = check_idx++;
    bool rewritten = pre_calculated_values_folding(f);
    std::cout << "ConstantFolding " << f->get_name() << std::endl;
    size_t count = check_loops(f, "ConstantFolding START .. rewritten("+std::to_string(rewritten)+")", curr_idx);

    for (const auto& node : f->get_ordered_ops())
    {
        // if (count > 0)
        //     std::cout << "ConstantFolding::Node:: " << node->get_type_name() << std::endl;
        std::string type_str(node->get_type_name());
        if (type_str == "Loop" || type_str == "TensorIterator") {
            std::cout << node->get_name() << " : " << type_str << " in ConstantFolding" << std::endl;
        }


        if (rewritten)
        {
            node->validate_and_infer_types();
        }

        bool is_constant_fold = false;
        OutputVector replacements(node->get_output_size());
        if ((type_str != "Loop" && type_str != "TensorIterator") && node->constant_fold(replacements, node->input_values()))
        // if (node->constant_fold(replacements, node->input_values()))
        {
            is_constant_fold = true;
            NGRAPH_CHECK(replacements.size() == node->get_output_size(),
                         "constant_fold_default returned incorrect number of replacements for ",
                         node);

            for (size_t i = 0; i < replacements.size(); ++i)
            {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement))
                {
                    if (replacements.size() == 1)
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name());
                    }
                    else
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name() + "." + std::to_string(i));
                    }
                    node_output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(node, replacement);

                    rewritten = true;
                }
            }
        }
        else
        {
            is_constant_fold = false;
            // recursively constant fold operators containing subgraphs (ie: TensorIterator, Loop)
            if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
            {
                if (const auto& sub_graph = sub_graph_node->get_function())
                {
                    rewritten |= run_on_function(sub_graph);
                }
            }
        }
        if (rewritten) {
            std::string type_str2(node->get_type_name());
            if (type_str2 == "Loop" || type_str == "TensorIterator") {
                std::cout << "replacement " << type_str2 << " is constant_fold: " << (is_constant_fold? "True" : "False") << std::endl;
            }
        }

    }

    check_loops(f, "ConstantFolding END .. rewritten("+std::to_string(rewritten)+")", curr_idx);
    return rewritten;
}

void ngraph::pass::ConstantFolding::copy_runtime_info_to_target_inputs(
    const std::shared_ptr<Node>& node, const Output<Node>& replacement)
{
    for (auto& input : replacement.get_target_inputs())
    {
        auto consumer = input.get_node()->shared_from_this();
        copy_runtime_info({node, consumer}, consumer);
    }
}

bool ngraph::pass::ConstantFolding::pre_calculated_values_folding(
    const std::shared_ptr<ngraph::Function>& f)
{
    deque<shared_ptr<Node>> nodes;
    set<shared_ptr<Node>> visited;
    for (auto& r : f->get_results())
        nodes.push_back(r);
    for (auto& r : f->get_sinks())
        nodes.emplace_back(r);

    bool rewritten = false;
    while (!nodes.empty())
    {
        auto curr_node = nodes.front();
        nodes.pop_front();
        if (visited.count(curr_node) || is_type<op::Constant>(curr_node))
            continue;
        visited.insert(curr_node);

        for (auto& input_value : curr_node->input_values())
        {
            if (input_value.get_tensor().has_and_set_bound())
            {
                auto input_node = input_value.get_node_shared_ptr();
                auto replacement =
                    std::make_shared<op::Constant>(input_value.get_tensor().get_lower_value());
                if (replacement && !is_type<op::Constant>(input_node))
                {
                    if (input_node->get_output_size() == 1)
                    {
                        replacement->set_friendly_name(input_node->get_friendly_name());
                    }
                    else
                    {
                        replacement->set_friendly_name(input_node->get_friendly_name() + "." +
                                                       std::to_string(input_value.get_index()));
                    }
                    input_value.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(input_node, replacement);

                    rewritten = true;
                }
            }
            else
            {
                // continue searching
                const auto& input_node = input_value.get_node_shared_ptr();
                nodes.push_front(input_node);
            }
        }
    }
    return rewritten;
}
