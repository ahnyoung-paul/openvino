// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_pooler_precision.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

IncreasePrecisionForVisionPooler::IncreasePrecisionForVisionPooler() {
    using namespace ov::pass::pattern;
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;
    namespace v8 = ov::op::v8;

    // Subgraph pattern from openvino_vision_embeddings_model (pre-ConvertPrecision, all f32):
    //
    // MatMul(pooler) → Multiply(×scalar) → GatherND → Multiply → Add(residual)
    // → Power(x²) → ReduceMean → Add(ε) → Power(rsqrt) → Multiply(RMSNorm)
    // → MatMul(FC) → Result

    auto pooler_matmul = wrap_type<v0::MatMul>({any_input(), any_input()});
    auto scalar_multiply = wrap_type<v1::Multiply>({pooler_matmul, any_input()});
    auto gather_nd = wrap_type<v8::GatherND>({scalar_multiply, any_input()});
    auto post_gather_multiply = wrap_type<v1::Multiply>({gather_nd, any_input()});
    auto residual_add = wrap_type<v1::Add>({post_gather_multiply, any_input()});
    auto power = wrap_type<v1::Power>({residual_add, any_input()});
    auto reduce_mean = wrap_type<v1::ReduceMean>({power, any_input()});
    auto add_eps = wrap_type<v1::Add>({reduce_mean, any_input()});
    auto power_rsqrt = wrap_type<v1::Power>({add_eps, any_input()});
    auto rms_multiply = wrap_type<v1::Multiply>({any_input(), power_rsqrt});
    auto fc = wrap_type<v0::MatMul>({rms_multiply, any_input()});
    auto result = wrap_type<v0::Result>({fc});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = ov::as_type_ptr<v0::MatMul>(pattern_map.at(pooler_matmul).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto rms_mul = pattern_map.at(rms_multiply).get_node_shared_ptr();

        const float scale_factor = 4.0f;

        GPU_DEBUG_COUT << "IncreasePrecisionForVisionPooler: scale_down on " << matmul->get_friendly_name()
                       << ", scale_up after " << rms_mul->get_friendly_name() << ", scale_factor=" << scale_factor << std::endl;

        // Insert scale_down (÷scale_factor) on input(1) of pooler MatMul
        auto input1_et = matmul->input(1).get_element_type();
        auto scale_down_const = std::make_shared<v0::Constant>(input1_et, ov::Shape{}, std::vector<float>{1.0f / scale_factor});
        auto scale_down = std::make_shared<v1::Multiply>(matmul->input(1).get_source_output(), scale_down_const);
        scale_down->set_friendly_name(matmul->get_friendly_name() + "_asf_scale_down_in1");
        ov::copy_runtime_info(matmul, scale_down);
        matmul->input(1).replace_source_output(scale_down->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, "IncreasePrecisionForVisionPooler");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
