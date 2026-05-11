// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "force_fp32_selective.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "plugin/transformations/utils.hpp"

namespace ov {
namespace intel_gpu {

namespace {

// Pass-through ops: no arithmetic, just data movement.
// These are automatically absorbed into f32 regions when they bridge two f32 ops,
// preventing f32->f16->f32 roundtrip truncation.
bool is_passthrough_op(const std::string& type_name) {
    static const std::unordered_set<std::string> passthrough_types = {
        "Reshape", "Transpose", "Squeeze", "Unsqueeze",
        "StridedSlice", "Gather", "Concat", "Split", "VariadicSplit",
        "ShapeOf", "Broadcast"
    };
    return passthrough_types.count(type_name) > 0;
}

bool matches_type(const std::vector<std::string>& forced_types, const std::string& type_name) {
    return std::find(forced_types.begin(), forced_types.end(), type_name) != forced_types.end();
}

bool matches_name(const std::vector<std::string>& forced_names, const std::string& node_name) {
    for (const auto& pattern : forced_names) {
        if (node_name.find(pattern) != std::string::npos)
            return true;
    }
    return false;
}

}  // namespace

bool ForceFP32Selective::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const bool has_types = !m_forced_types.empty();
    const bool has_names = !m_forced_names.empty();

    if (!has_types && !has_names) {
        return false;
    }

    GPU_DEBUG_COUT << "[ForceFP32Selective] Filter mode: "
                   << (has_types && has_names ? "type AND name" :
                       has_types ? "type only" : "name only")
                   << std::endl;
    if (has_types) {
        GPU_DEBUG_COUT << "[ForceFP32Selective] Types: ";
        for (const auto& t : m_forced_types) GPU_DEBUG_COUT << t << " ";
        GPU_DEBUG_COUT << std::endl;
    }
    if (has_names) {
        GPU_DEBUG_COUT << "[ForceFP32Selective] Name patterns: ";
        for (const auto& n : m_forced_names) GPU_DEBUG_COUT << n << " ";
        GPU_DEBUG_COUT << std::endl;
    }

    // ====================================================================
    // Phase 1: Seed Collection — identify ops that match the user filter
    // ====================================================================
    std::unordered_set<ov::Node*> f32_region;     // nodes that will run in f32
    std::vector<std::string> skipped_types;

    for (const auto& node : model->get_ops()) {
        const std::string node_type = node->get_type_name();
        const std::string node_name = node->get_friendly_name();

        bool type_match = !has_types || matches_type(m_forced_types, node_type);
        bool name_match = !has_names || matches_name(m_forced_names, node_name);

        if (!type_match &&
            std::find(skipped_types.begin(), skipped_types.end(), node_type) == skipped_types.end()) {
            skipped_types.push_back(node_type);
        }

        if (!type_match || !name_match)
            continue;

        // Skip already-f32, Placeholder, and Result/Parameter boundary ops
        const auto et = node->get_output_element_type(0);
        if (et == ov::element::f32)
            continue;
        if (et != ov::element::f16)
            continue;
        if (node_name.find("Placeholder") != std::string::npos)
            continue;

        f32_region.insert(node.get());
        GPU_DEBUG_COUT << "[ForceFP32Selective] Phase1 seed: type=" << node_type
                       << "  name=" << node_name << std::endl;
    }

    GPU_DEBUG_COUT << "[ForceFP32Selective] Phase1 seeds: " << f32_region.size() << std::endl;

    // ====================================================================
    // Phase 2: Region Expansion — absorb pass-through ops connected to
    //          the f32 region via flood-fill, preventing f32->f16->f32
    //          roundtrip truncation through chains of data-movement ops.
    //          A passthrough op is absorbed if it has at least one
    //          f32-region neighbor (producer OR consumer). Iterate until
    //          no new ops are absorbed.
    // ====================================================================
    bool expanded = true;
    int expansion_rounds = 0;
    while (expanded) {
        expanded = false;
        expansion_rounds++;
        for (const auto& node : model->get_ops()) {
            if (f32_region.count(node.get()))
                continue;

            const std::string node_type = node->get_type_name();
            if (!is_passthrough_op(node_type))
                continue;

            const auto et = node->get_output_element_type(0);
            if (et != ov::element::f16)
                continue;

            // Check: does this node have at least one f32-region neighbor?
            // (producer OR consumer — either direction triggers absorption)
            bool has_f32_neighbor = false;
            for (const auto& input : node->inputs()) {
                auto* producer = input.get_source_output().get_node();
                if (f32_region.count(producer)) {
                    has_f32_neighbor = true;
                    break;
                }
            }
            if (!has_f32_neighbor) {
                for (const auto& output : node->outputs()) {
                    for (const auto& target : output.get_target_inputs()) {
                        if (f32_region.count(target.get_node())) {
                            has_f32_neighbor = true;
                            break;
                        }
                    }
                    if (has_f32_neighbor)
                        break;
                }
            }
            if (!has_f32_neighbor)
                continue;

            // This pass-through op is connected to the f32 region — absorb it
            f32_region.insert(node.get());
            expanded = true;
            GPU_DEBUG_COUT << "[ForceFP32Selective] Phase2 absorbed: type=" << node_type
                           << "  name=" << node->get_friendly_name() << std::endl;
        }
    }

    GPU_DEBUG_COUT << "[ForceFP32Selective] Phase2 done: region_size=" << f32_region.size()
                   << "  rounds=" << expansion_rounds << std::endl;

    // ====================================================================
    // Phase 3a: Input Promotion — insert f16->f32 boundary Converts at
    //           region entry points and validate type inference.
    // ====================================================================
    bool changed = false;
    int forced_count = 0;
    size_t global_convert_idx = 0;
    const auto desired_et = ov::element::f32;

    for (const auto& node : model->get_ops()) {
        if (!f32_region.count(node.get()))
            continue;

        const std::string node_type = node->get_type_name();
        const std::string node_name = node->get_friendly_name();

        for (const auto& input : node->inputs()) {
            const auto& incoming_output = input.get_source_output();
            const auto& incoming_node = incoming_output.get_node_shared_ptr();
            const auto input_et = incoming_output.get_element_type();

            if (input_et == desired_et)
                continue;
            if (input_et != ov::element::f16)
                continue;

            // Skip Placeholder inputs
            if (incoming_node->get_friendly_name().find("Placeholder") != std::string::npos ||
                incoming_node->get_type_info().name == std::string("gpu_opset::Placeholder")) {
                continue;
            }

            // If producer is inside f32 region and already processed (topological),
            // its output is already f32 — the input_et check handles this.
            if (f32_region.count(incoming_output.get_node()))
                continue;

            // Boundary: producer is outside region — insert Convert f16->f32
            auto in_convert = ov::as_type_ptr<ov::op::v0::Convert>(incoming_node);
            if (in_convert && in_convert->get_users().size() == 1 &&
                input_et.bitwidth() <= desired_et.bitwidth()) {
                auto convert = std::make_shared<ov::op::v0::Convert>(
                    incoming_node->input_value(0), desired_et);
                convert->set_friendly_name(
                    in_convert->get_friendly_name() + "_increase_precision_" +
                    std::to_string(global_convert_idx));
                ov::copy_runtime_info(incoming_node, convert);
                ov::replace_node(incoming_node, convert);
            } else {
                auto convert = std::make_shared<ov::op::v0::Convert>(incoming_output, desired_et);
                convert->set_friendly_name(
                    incoming_node->get_friendly_name() + "_increase_precision_" +
                    std::to_string(global_convert_idx));
                ov::copy_runtime_info(incoming_node, convert);
                input.replace_source_output(convert);
            }
            global_convert_idx++;
        }

        // Force output type to f32. Skips validate_and_infer_types() because:
        // 1. TypeRelaxed ops (common in GPU plugin) have overridden types that
        //    prevent normal validation from propagating input type changes.
        // 2. dynamic_cast<TypeRelaxedBase*> fails across shared library boundaries,
        //    so we can't reliably detect TypeRelaxed for set_overridden_output_type.
        // 3. The GPU backend will see f32 inputs and f32 output metadata, which is
        //    sufficient for correct precision-promoted execution.
        if (node->get_output_element_type(0) != desired_et) {
            node->set_output_type(0, desired_et, node->get_output_partial_shape(0));
        }

        forced_count++;
        changed = true;
        GPU_DEBUG_COUT << "[ForceFP32Selective] FORCED: type=" << node_type
                       << "  name=" << node_name
                       << "  output_et=" << node->get_output_element_type(0) << std::endl;
    }

    // ====================================================================
    // Phase 3b: Output Restore — insert f32->f16 Converts at region exit
    //           points. Done separately after all promotions so that
    //           region-internal edges are never accidentally restored.
    // ====================================================================
    for (const auto& node : model->get_ops()) {
        if (!f32_region.count(node.get()))
            continue;

        // After promotion, output type should be f32; restore to f16 for outside consumers
        const auto current_et = node->get_output_element_type(0);
        if (current_et != desired_et)
            continue;

        for (const auto& output : node->outputs()) {
            std::vector<ov::Input<ov::Node>> outside_consumers;
            for (const auto& target_input : output.get_target_inputs()) {
                if (!f32_region.count(target_input.get_node())) {
                    outside_consumers.push_back(target_input);
                }
            }
            for (auto& target_input : outside_consumers) {
                auto convert = std::make_shared<ov::op::v0::Convert>(output, ov::element::f16);
                convert->set_friendly_name(
                    target_input.get_node()->shared_from_this()->get_friendly_name() +
                    "_restore_precision_" + std::to_string(global_convert_idx));
                ov::copy_runtime_info(node, convert);
                target_input.replace_source_output(convert);
                global_convert_idx++;
            }
        }
    }

    GPU_DEBUG_COUT << "[ForceFP32Selective] Done. Region=" << f32_region.size()
                   << "  Forced=" << forced_count << std::endl;

    if (!skipped_types.empty()) {
        std::stringstream ss;
        ss << "[ForceFP32Selective] Skipped types: ";
        for (const auto& t : skipped_types) ss << t << ",";
        GPU_DEBUG_COUT << ss.str() << std::endl;
    }

    return changed;
}

}  // namespace intel_gpu
}  // namespace ov
