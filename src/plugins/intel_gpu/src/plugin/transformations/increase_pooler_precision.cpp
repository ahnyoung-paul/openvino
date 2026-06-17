// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_pooler_precision.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

IncreasePrecisionForVisionPooler::IncreasePrecisionForVisionPooler() {}

bool IncreasePrecisionForVisionPooler::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const std::string scale_down_target = "vision_tower.pooler";
    const std::string scale_down_target2 = "matmul";
    const std::string scale_up_target = "embedding_pre_projection_norm";
    const std::string scale_up_target2 = "mul";
    const float scale_factor = 4096.0f;

    std::shared_ptr<ov::Node> pooler_matmul = nullptr;
    std::shared_ptr<ov::Node> rms_mul = nullptr;

    for (const auto& node : model->get_ordered_ops()) {
        const auto& name = node->get_friendly_name();

        if (!pooler_matmul &&
            name.find(scale_down_target) != std::string::npos &&
            name.find(scale_down_target2) != std::string::npos &&
            ov::is_type<ov::op::v0::MatMul>(node)) {
            pooler_matmul = node;
        }

        if (!rms_mul &&
            name.find(scale_up_target) != std::string::npos &&
            name.find(scale_up_target2) != std::string::npos &&
            ov::is_type<ov::op::v1::Multiply>(node)) {
            rms_mul = node;
        }
    }

    if (!pooler_matmul || !rms_mul)
        return false;

    GPU_DEBUG_COUT << "IncreasePrecisionForVisionPooler: scale_down on " << pooler_matmul->get_friendly_name()
                   << ", scale_up after " << rms_mul->get_friendly_name() << std::endl;

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

    return true;
}

}  // namespace ov::intel_gpu
