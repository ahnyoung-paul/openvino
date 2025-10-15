// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_rms.hpp"

#include "ov_ops/rms.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/select.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"
#include <openvino/pass/manager.hpp>
#include <openvino/pass/visualize_tree.hpp>

#include <memory>

namespace ov::intel_gpu {

DisableFP16CompForRMS::DisableFP16CompForRMS() {
    using namespace ov::pass::pattern;

    auto add_m = wrap_type<ov::op::v1::Add>({any_input(), any_input()}, type_matches(element::f32));
    auto rms_post_m = wrap_type<ov::op::internal::RMS>({any_input(), wrap_type<ov::op::v0::Constant>()}, type_matches(element::f32));
    auto add_1_m = wrap_type<ov::op::v1::Add>({add_m, rms_post_m}, type_matches(element::f32));
    auto rms_m = wrap_type<ov::op::internal::RMS>({add_1_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
        if (!rms || transformation_callback(rms)) {
            return false;
        }
        if (pattern_map.count(rms_post_m) > 0) {
            auto rms_post = pattern_map.at(rms_post_m).get_node_shared_ptr();
            if (rms_post) {
                ov::disable_fp16_compression(rms_post);
            }
        }
        ov::disable_fp16_compression(rms);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_m, "DisableFP16CompForRMS");
    this->register_matcher(m, callback);
}

static bool has_names(const ov::Output<ov::Node>& output) {
#if 0
    std::vector<std::string> issued_names = {
        "__module.model.layers.6.input_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.6.self_attn.q_norm/aten::mul/Multiply_1",
        "__module.model.layers.6.self_attn.k_norm/aten::mul/Multiply_1",
        "__module.model.layers.6.pre_feedforward_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.6.post_attention_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.6.post_feedforward_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.7.input_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.7.self_attn.q_norm/aten::mul/Multiply_1",
        "__module.model.layers.7.self_attn.k_norm/aten::mul/Multiply_1",
        "__module.model.layers.7.pre_feedforward_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.7.post_attention_layernorm/aten::mul/Multiply_1",
        "__module.model.layers.7.post_feedforward_layernorm/aten::mul/Multiply_1"
    };
    auto node_name = output.get_node_shared_ptr()->get_friendly_name();
    return std::find(issued_names.begin(), issued_names.end(), node_name) != issued_names.end();
#else
    auto node_name = output.get_node_shared_ptr()->get_friendly_name();
    // std::cout << node_name << std::endl;
    return node_name.find("__module.model.layers.6") != std::string::npos ||
           node_name.find("__module.model.layers.6") != std::string::npos;
#endif
}

DebugRMSFusion::DebugRMSFusion(const ov::intel_gpu::ExecutionConfig& config,
    std::string tag, bool dump_graph) : m_tag(tag), m_dump_graph(dump_graph), m_config(config) {}

bool DebugRMSFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (auto& node : model->get_ordered_ops()) {
        bool is_debug = has_names(node);
        if (is_debug) {
            std::cout << m_tag << ";" << node->get_friendly_name()
                        << ";" << node->get_type_name()
                        << ";" << node->get_output_element_type(0)
                        << ";" << node->get_output_partial_shape(0).to_string() << std::endl;
        }
    }
    if (!m_dump_graph)
        return true;

    std::string path = m_config.get_dump_graphs_path();
    if (!path.empty()) {
        ov::pass::Manager manager;
        if (path.back() != '/' && path.back() != '\\') {
            path += "/";
        }
        std::string dump_file_name = path + "debug_ngraph_" + m_tag + ".svg";
        // Serialize ov::Model to before.svg file before transformation
        manager.register_pass<ov::pass::VisualizeTree>(dump_file_name);
        manager.run_passes(model);
        std::cout << "Dump " << dump_file_name << std::endl;
    }
    return true;
}

}  // namespace ov::intel_gpu
