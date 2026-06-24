// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_pooler_precision.hpp"

#include <cstdlib>
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/file_util.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

IncreasePrecisionForVisionPooler::IncreasePrecisionForVisionPooler() {}

bool IncreasePrecisionForVisionPooler::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Dump model graph before transformation if OV_GPU_DUMP_GRAPHS_PATH is set
    GPU_DEBUG_CODE({
        const char* dump_path_env = std::getenv("OV_GPU_DUMP_GRAPHS_PATH");
        if (dump_path_env != nullptr && dump_path_env[0] != '\0') {
            auto dump_path = ov::util::make_path(dump_path_env);
            auto path_before = dump_path / (model->get_name() + "_before_increase_pooler_precision.svg");
            ov::pass::VisualizeTree(path_before).run_on_model(model);
            GPU_DEBUG_COUT << "IncreasePrecisionForVisionPooler: Dumped model graph before transformation to: "
                           << path_before << std::endl;
        }
    });

    const float scale_factor = 4.0f;

    std::shared_ptr<ov::Node> pooler_matmul = nullptr;
    std::shared_ptr<ov::Node> rms_mul = nullptr;

    // Pattern-based matching for Vision Pooler MatMul
    // Pattern: MatMul → Multiply (with constant ~33.9375) → GatherND
    for (const auto& node : model->get_ordered_ops()) {
        if (!pooler_matmul && ov::is_type<ov::op::v0::MatMul>(node)) {
            // Check if this MatMul has the expected output pattern
            auto matmul_output = node->output(0);
            bool has_expected_pattern = false;

            for (auto& target_input : matmul_output.get_target_inputs()) {
                auto consumer = target_input.get_node()->shared_from_this();

                // Check for Multiply consumer
                if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(consumer)) {
                    // Check if Multiply has a constant input with value ~33.9375 (tolerance for FP16)
                    for (size_t i = 0; i < multiply->get_input_size(); ++i) {
                        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(
                                multiply->get_input_node_shared_ptr(i))) {
                            auto values = constant->cast_vector<float>();
                            if (!values.empty() && values[0] > 30.0f && values[0] < 40.0f) {
                                // Check if this Multiply feeds into GatherND
                                for (auto& mul_target : multiply->output(0).get_target_inputs()) {
                                    auto mul_consumer = mul_target.get_node()->shared_from_this();
                                    if (mul_consumer->get_type_info().name == std::string("GatherND")) {
                                        has_expected_pattern = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if (has_expected_pattern) break;
                    }
                }
                if (has_expected_pattern) break;
            }

            if (has_expected_pattern) {
                pooler_matmul = node;
            }
        }

        // Pattern-based matching for RMS Norm end
        // Pattern: Multiply (RMS norm) → Convert (f32→f16) → FullyConnectedCompressed → Result
        if (!rms_mul && ov::is_type<ov::op::v1::Multiply>(node)) {
            auto multiply_output = node->output(0);
            bool has_rms_norm_pattern = false;

            for (auto& target_input : multiply_output.get_target_inputs()) {
                auto consumer = target_input.get_node()->shared_from_this();

                // Check for Convert (f32 → f16)
                if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(consumer)) {
                    if (convert->get_input_element_type(0) == ov::element::f32 &&
                        convert->get_output_element_type(0) == ov::element::f16) {

                        // Check if Convert feeds into FullyConnectedCompressed
                        for (auto& conv_target : convert->output(0).get_target_inputs()) {
                            auto fc_consumer = conv_target.get_node()->shared_from_this();
                            if (fc_consumer->get_type_info().name == std::string("FullyConnectedCompressed")) {
                                // Check if FullyConnectedCompressed feeds into Result
                                for (auto& fc_target : fc_consumer->output(0).get_target_inputs()) {
                                    if (ov::is_type<ov::op::v0::Result>(fc_target.get_node()->shared_from_this())) {
                                        has_rms_norm_pattern = true;
                                        break;
                                    }
                                }
                            }
                            if (has_rms_norm_pattern) break;
                        }
                    }
                }
                if (has_rms_norm_pattern) break;
            }

            if (has_rms_norm_pattern) {
                rms_mul = node;
            }
        }
    }

    if (!pooler_matmul || !rms_mul)
        return false;

    GPU_DEBUG_COUT << "IncreasePrecisionForVisionPooler: scale_down on " << pooler_matmul->get_friendly_name()
                   << ", scale_up after " << rms_mul->get_friendly_name() << ", scale_factor=" << scale_factor << std::endl;

    // 1. Insert scale_down (÷4096) on input(1) of pooler MatMul
    auto input1_et = pooler_matmul->input(1).get_element_type();
    auto scale_down_const = std::make_shared<ov::op::v0::Constant>(input1_et, ov::Shape{}, std::vector<float>{1.0f / scale_factor});
    auto scale_down = std::make_shared<ov::op::v1::Multiply>(pooler_matmul->input(1).get_source_output(), scale_down_const);
    scale_down->set_friendly_name(pooler_matmul->get_friendly_name() + "_asf_scale_down_in1");
    ov::copy_runtime_info(pooler_matmul, scale_down);
    pooler_matmul->input(1).replace_source_output(scale_down->output(0));

    // 2. Insert scale_up (×4096) after RMSNorm Multiply output
    //    f16→f32 convert before scale_up, then f32→f16 convert after (avoid f16 overflow on ×4096)
    auto output_et = rms_mul->get_output_element_type(0);

    // TODO: scale_up commented out for testing.
    // RMSNorm may auto-cancel the scale_down (normalization divides by scaled magnitude),
    // so scale_up might not be needed. Testing without it to verify.
    //
    // // Collect consumers before modification
    // std::vector<ov::Input<ov::Node>> consumers;
    // for (auto& input : rms_mul->get_output_target_inputs(0)) {
    //     consumers.push_back(input);
    // }
    //
    // // Build: rms_mul → Convert(f32) → ×4096 → Convert(f16)
    // std::shared_ptr<ov::Node> scale_up_input = rms_mul;
    // if (output_et != ov::element::f32) {
    //     auto to_f32 = std::make_shared<ov::op::v0::Convert>(rms_mul->output(0), ov::element::f32);
    //     to_f32->set_friendly_name(rms_mul->get_friendly_name() + "_to_f32");
    //     ov::copy_runtime_info(rms_mul, to_f32);
    //     scale_up_input = to_f32;
    // }
    //
    // auto scale_up_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_factor});
    // auto scale_up = std::make_shared<ov::op::v1::Multiply>(scale_up_input->output(0), scale_up_const);
    // scale_up->set_friendly_name(rms_mul->get_friendly_name() + "_asf_scale_up");
    // ov::copy_runtime_info(rms_mul, scale_up);
    //
    // std::shared_ptr<ov::Node> final_output = scale_up;
    // if (output_et != ov::element::f32) {
    //     auto to_original = std::make_shared<ov::op::v0::Convert>(scale_up->output(0), output_et);
    //     to_original->set_friendly_name(rms_mul->get_friendly_name() + "_to_f16");
    //     ov::copy_runtime_info(rms_mul, to_original);
    //     final_output = to_original;
    // }
    //
    // // Replace original consumers
    // for (auto& input : consumers) {
    //     input.replace_source_output(final_output->output(0));
    // }

    // Dump model graph after transformation if OV_GPU_DUMP_GRAPHS_PATH is set
    GPU_DEBUG_CODE({
        const char* dump_path_env = std::getenv("OV_GPU_DUMP_GRAPHS_PATH");
        if (dump_path_env != nullptr && dump_path_env[0] != '\0') {
            auto dump_path = ov::util::make_path(dump_path_env);
            auto path_after = dump_path / (model->get_name() + "_after_increase_pooler_precision.svg");
            ov::pass::VisualizeTree(path_after).run_on_model(model);
            GPU_DEBUG_COUT << "IncreasePrecisionForVisionPooler: Dumped model graph after transformation to: "
                           << path_after << std::endl;
        }
    });

    return true;
}

}  // namespace ov::intel_gpu
