// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_pooler_precision.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
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
    using namespace ov::op;
    using namespace ov::pass::pattern;

    // Pattern: pooler MatMul → GatherND → Multiply(×scalar) → Power(²) → ReduceMean → ... → FC → Result
    //
    // This matches the specific subgraph in gemma-4 vision encoder that terminates at the
    // program output (Result). The full exec graph path:
    //   gemm:__module.model.model.vision_tower.pooler/aten::matmul/MatMul
    //   → gathernd:__module.model.model.vision_tower/aten::index/GatherND
    //   → multiply:Multiply_76358 (×33.9375 = √1152)
    //   → power:embed_vision.embedding_pre_projection_norm/aten::pow/Power
    //   → reducemean:embed_vision.embedding_pre_projection_norm/aten::mean/ReduceMean
    //   → (decomposed RMSNorm: Add → Power → Multiply)
    //   → fullyconnectedcompressed:embed_vision.embedding_projection/ov_ext::linear/MatMul.0
    //   → result:Result_49143
    //
    // The ×33.9375 scalar causes FP16 overflow when vision_features are large (±3074).
    // We disable fp16 compression on pooler MatMul through decomposed RMSNorm so
    // ConvertPrecision keeps this subgraph in f32.

    // Match from pooler MatMul up to ReduceMean (structural pattern)
    auto pooler_matmul = wrap_type<v0::MatMul>({any_input(), any_input()});
    auto gather_nd = wrap_type<v8::GatherND>({pooler_matmul, any_input()});
    auto scalar_multiply = wrap_type<v1::Multiply>({gather_nd, any_input()});
    auto power = wrap_type<v1::Power>({scalar_multiply, any_input()});
    auto reduce_mean = wrap_type<v1::ReduceMean>({power, any_input()});

    // Continue matching through decomposed RMSNorm to FC and finally Result
    auto add_eps = wrap_type<v1::Add>({reduce_mean, any_input()});
    auto power_rsqrt = wrap_type<v1::Power>({add_eps, any_input()});
    auto rms_multiply = wrap_type<v1::Multiply>({any_input(), power_rsqrt});
    auto fc = wrap_type<v0::MatMul>({rms_multiply, any_input()});
    auto result = wrap_type<v0::Result>({fc});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul_node = pattern_map.at(pooler_matmul).get_node_shared_ptr();

        if (!matmul_node || transformation_callback(matmul_node))
            return false;

        if (matmul_node->get_output_element_type(0) == ov::element::f32)
            return false;

        // Marking reduce_mean is sufficient — ConvertPrecision propagates f32
        // to connected nodes (pooler MatMul, GatherND, Multiply, Power upstream
        // and Add, Power, Multiply downstream in decomposed RMSNorm).
        pattern_map.at(reduce_mean).get_node_shared_ptr()->get_rt_info()["disable_fp16_compression"] = true;

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, "IncreasePrecisionForVisionPooler");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
