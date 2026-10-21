// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collect_sincos_constants.hpp"

#include <filesystem>
#include <vector>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/visualize_tree.hpp"

namespace ov::intel_gpu {

// Temporary debugging only: remove this implementation after investigation.
DumpNetwork::DumpNetwork(const std::string& path, const std::string& tag) : m_path(path), m_tag(tag) {}

bool DumpNetwork::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (m_path.empty()) {
        return false;
    }
    std::filesystem::create_directories(m_path);
    const auto filename = std::filesystem::path(m_path) / (model->get_name() + "_" + m_tag + ".svg");
    ov::pass::VisualizeTree(filename).run_on_model(model);
    GPU_DEBUG_COUT << "Dumped network to " << filename << std::endl;
    return false;
}

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

        // Preserve coefficients on supported angle-value paths, not every Constant upstream of Sin/Cos.
        // These arithmetic ops combine operand values; Concat carries values from every input.
        // Follow all operands so coefficients on either side of an expression are included.
        if (ov::is_type_any_of<ov::op::v1::Add,
                               ov::op::v1::Subtract,
                               ov::op::v1::Multiply,
                               ov::op::v1::Divide,
                               ov::op::v0::Concat>(node)) {
            for (const auto& input : node->input_values()) {
                pending.push_back(input);
            }
        } else if (ov::is_type_any_of<ov::op::v0::Convert,
                                      ov::op::v1::Reshape,
                                      ov::op::v1::Transpose,
                                      ov::op::v0::Squeeze,
                                      ov::op::v0::Unsqueeze,
                                      ov::op::v1::Broadcast,
                                      ov::op::v3::Broadcast,
                                      ov::op::v1::Gather,
                                      ov::op::v7::Gather,
                                      ov::op::v8::Gather,
                                      ov::op::v1::StridedSlice,
                                      ov::op::v1::Split,
                                      ov::op::v1::VariadicSplit>(node)) {
            // These versions convert, rearrange or select values from input 0. Other inputs describe
            // shapes, axes, indices or slice bounds; they are not floating-point angle coefficients.
            pending.push_back(node->input_value(0));
        }
        // Keep an explicit allowlist: floating-point type alone does not identify an angle-value path.
        // Stop at MatMul (which can lead to large weight/activation subgraphs) and unlisted operations;
        // do not enter subgraph bodies. Add an op/version only after checking all value-input ports
        // and whether traversing it would preserve unrelated Constants.
    }

    return false;
}

}  // namespace ov::intel_gpu
