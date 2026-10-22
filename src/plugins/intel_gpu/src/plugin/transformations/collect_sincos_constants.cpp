// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collect_sincos_constants.hpp"

#include <filesystem>
#include <vector>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
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

}  // namespace ov::intel_gpu
