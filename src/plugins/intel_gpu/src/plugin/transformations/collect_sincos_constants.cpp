// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collect_sincos_constants.hpp"

#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/sin.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"

namespace ov::intel_gpu {

CollectSinCosConstants::CollectSinCosConstants(const std::shared_ptr<ConstantSet>& constants) : m_constants(constants) {
    OPENVINO_ASSERT(m_constants, "CollectSinCosConstants requires a shared Constant set");
}

bool CollectSinCosConstants::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_constants->clear();
    std::unordered_set<const ov::Node*> visited;
    ov::OutputVector pending;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type_any_of<ov::op::v0::Sin, ov::op::v0::Cos>(node)) {
            pending.push_back(node->input_value(0));
        }
    }

    while (!pending.empty()) {
        const auto output = pending.back();
        pending.pop_back();
        if (!output.get_element_type().is_real()) {
            continue;
        }
        const auto node = output.get_node_shared_ptr();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            if (output.get_element_type() == ov::element::f32) {
                m_constants->insert(node.get());
            }
            continue;
        }

        // Follow floating-point dependencies without an op allowlist, including through MatMul.
        // Integer/bool inputs (e.g. shapes, axes and indices) end the path even if derived from floats.
        // Only outer-graph input edges are followed; subgraph bodies are not traversed.
        for (const auto& input : node->input_values()) {
            if (input.get_element_type().is_real()) {
                pending.push_back(input);
            }
        }
    }

    return false;
}

// Shared by the GPU pipeline and accuracy regression tests to avoid duplicating the preservation policy.
bool skip_constant_precision_preservation(const std::shared_ptr<const ov::Node>& node,
                                          const CollectSinCosConstants::ConstantSet& sincos_constants) {
    const auto targets = node->get_output_target_inputs(0);
    const ov::Node* next_node = node.get();
    if (!targets.empty()) {
        next_node = targets.begin()->get_node();
    }
    if (ov::is_type<ov::op::v0::Convert>(next_node)) {
        const auto convert_targets = next_node->get_output_target_inputs(0);
        if (!convert_targets.empty()) {
            next_node = convert_targets.begin()->get_node();
        }
    }
    if (ov::is_type_any_of<ov::op::v0::MatMul,
                           ov::op::internal::MOE,
                           ov::op::internal::GatherMatmulCompressed,
                           ov::op::internal::GroupedMatMulCompressed>(next_node)) {
        return false;
    }
    return sincos_constants.count(node.get()) == 0;
}

}  // namespace ov::intel_gpu
