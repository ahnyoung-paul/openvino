// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"

#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations/utils/utils.hpp"

#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/graph/topology.hpp"

#include <vector>
#include <algorithm>

using Loop = ov::op::v5::Loop;
using TensorIterator = ov::op::v0::TensorIterator;

namespace ov {
namespace intel_gpu {

template<class DATA_TYPE>
static DATA_TYPE CreateScalarData(ProgramBuilder &p, const cldnn::primitive_id& id, int64_t num) {
    auto mem = p.get_engine().allocate_memory({ cldnn::data_types::i64, cldnn::format::bfyx, { 1, 1, 1, 1 } });
    cldnn::mem_lock<int64_t> ptr{mem, p.get_engine().get_service_stream()};
    *ptr.begin() = num;
    return {id, mem};
}

static cldnn::mutable_data CreateAdditionalOutputData(ProgramBuilder &p, const std::shared_ptr<ov::Node>& op,
                                                        const cldnn::primitive_id& id, const cldnn::primitive_id& input,
                                                        const int32_t output_idx) {
    const auto precision = cldnn::element_type_to_data_type(op->get_output_element_type(output_idx));
    const auto format = cldnn::format::get_default_format(op->get_output_shape(output_idx).size());
    const auto tensor = tensor_from_dims(op->get_output_shape(output_idx));
    cldnn::layout output_layout = cldnn::layout(precision, format, tensor);
    auto mem = p.get_engine().allocate_memory(output_layout);
    auto md = cldnn::mutable_data(id, {cldnn::input_info(input)}, std::move(mem)); // cldnn::data cannot set dependency
    return md;
}

static void SetLoopInputOutputMap(ProgramBuilder& p,
                                    const std::shared_ptr<ov::op::util::SubGraphOp>& op,
                                    cldnn::primitive::input_info_arr& inputs,
                                    std::vector<cldnn::loop::io_primitive_map>& input_primitive_maps,
                                    std::vector<cldnn::loop::io_primitive_map>& output_primitive_maps,
                                    std::vector<cldnn::loop::backedge_mapping>& back_edges_maps) {
    const std::string layerName = layer_type_name_ID(op);
    const auto& loop_input_descs = op->get_input_descriptions();
    const auto& loop_output_descs = op->get_output_descriptions();
    const auto& body_inputs = op->get_function()->get_parameters();
    const auto& body_outputs = op->get_function()->get_results();

    auto config = p.get_config();
    bool use_new_shape_infer = config.get_property(ov::intel_gpu::allow_new_shape_infer);

    // set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        auto external_id = inputs.at(loop_input_desc->m_input_index);
        auto& body_input = body_inputs.at(loop_input_desc->m_body_parameter_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_input);

        // set input mapping
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            input_primitive_maps.emplace_back(external_id, internal_id, sliceInfo->m_axis,
                sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
        } else {
            // input without slicing
            input_primitive_maps.emplace_back(external_id, internal_id);
        }

        // set back edges
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::MergedInputDescription>(loop_input_desc)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);

            cldnn::primitive_id to_id = layer_type_name_ID(to);
            cldnn::primitive_id from_id = layer_type_name_ID(from);

            back_edges_maps.emplace_back(from_id, to_id);
        }
    }

    use_new_shape_infer = false;
    if (use_new_shape_infer) {
        std::cout << "use_new_shape_infer is true" << std::endl;
    } else {
        // set output mapping
        for (const auto& loop_output_desc : loop_output_descs) {
            const uint64_t output_idx = loop_output_desc->m_output_index;
            const auto body_idx = loop_output_desc->m_body_value_index;

            // Add additional mutable_data for multiple outputs
            // primitive ID should be <TI primitive ID>.<output_idx> if output_idx > 0
            // otherwise primitive ID should be equals to TI primitive ID
            const std::string layerNameWithIndex = layerName + ".out" + std::to_string(output_idx);
            std::string external_id;
            if (output_idx > 0) {
                cldnn::mutable_data output_data = CreateAdditionalOutputData(p, op, layerNameWithIndex, layerName, output_idx);
                p.add_primitive(*op, std::move(output_data));
                external_id = layerNameWithIndex;
                // TODO: Why this makes issue ?
                // Error has occured for: reshape:TensorIterator_293.2
                // Output layout count(=2) is not equal to: input layout count(=4)
                // Output layout of reshape primitive changes size of input buffer
                // p.primitive_ids[layerNameWithIndex] = layerName;
                // p.primitive_ids[layerName] = layerName;
            } else {
                p.primitive_ids[layerNameWithIndex] = layerName;
                p.primitive_ids[layerName] = layerName;
                external_id = layerName;
            }
            const auto& body_output = body_outputs.at(loop_output_desc->m_body_value_index);
            cldnn::primitive_id internal_id = layer_type_name_ID(body_output);

            std::cout << "loop_output_descs = [output_idx:" << output_idx << "=> output idx of "
                        << layerName << ", body_idx:" << body_idx
                        << "=> internal_id: " << internal_id << "]" << std::endl;

            // update primitive_map
            if (const auto& concatOutput =
                std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(loop_output_desc)) {
                // output which requires concatenation
                output_primitive_maps.emplace_back(external_id, internal_id, concatOutput->m_axis,
                    concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
            }
            if (std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(loop_output_desc)) {
                // output which requires no concatenation
                output_primitive_maps.emplace_back(external_id, internal_id);
            }
        }
    }
}

static std::vector<cldnn::primitive_id> GetOutputNames(const cldnn::primitive_id id,
                                                        const cldnn::primitive_id body_execution_condition_id,
                                                        const std::vector<cldnn::loop::io_primitive_map>& output_primitive_maps,
                                                        const std::vector<cldnn::loop::backedge_mapping>& back_edges) {
    std::vector<cldnn::primitive_id> output_names;
    OPENVINO_ASSERT(!output_primitive_maps.empty(), "[GPU] Output primitive map should have at least 1 mapping in primitive ", id);
    for (auto out_map : output_primitive_maps) {
        output_names.push_back(out_map.internal_id.pid);
    }

    // setup outputs for backedges
    for (auto& back_edge : back_edges) {
        output_names.push_back(back_edge.from);
    }

    // if execution_condition_id is specified, we need to add the id in build_option::outputs
    if (!body_execution_condition_id.empty()) {
        output_names.push_back(body_execution_condition_id);
    }

    return output_names;
}


static void CreateCommonLoopOp(ProgramBuilder& p, const std::shared_ptr<ov::op::util::SubGraphOp>& op, bool is_loop_op) {
    const std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);

    auto ov_model = op->get_function();
    // Set special body ports: current_iteration input , execution condition output
    cldnn::primitive_id body_current_iteration_id;
    cldnn::primitive_id body_execution_condition_id;
    cldnn::primitive_id trip_count_id;
    cldnn::primitive_id first_execution_condition_id;

    if (is_loop_op) {
        auto loop_op = std::dynamic_pointer_cast<Loop>(op);
        auto special_body_ports = loop_op->get_special_body_ports();
        if (special_body_ports.current_iteration_input_idx >= 0) {
            const auto& body_inputs = loop_op->get_function()->get_parameters();
            auto current_iteration_input = body_inputs.at(special_body_ports.current_iteration_input_idx);
            body_current_iteration_id = layer_type_name_ID(current_iteration_input);
        }

        if (special_body_ports.body_condition_output_idx >= 0) {
            const auto& body_outputs = loop_op->get_function()->get_results();
            auto body_condition_output = body_outputs.at(special_body_ports.body_condition_output_idx)->get_input_node_shared_ptr(0);
            body_execution_condition_id = layer_type_name_ID(body_condition_output);
        }

        trip_count_id = layer_type_name_ID(loop_op->get_input_node_shared_ptr(0));
        first_execution_condition_id = layer_type_name_ID(loop_op->get_input_node_shared_ptr(1));
    }

    // setup input_primitive_maps/ output_primitive_maps and back_edges
    std::vector<cldnn::loop::io_primitive_map> input_primitive_maps;
    std::vector<cldnn::loop::io_primitive_map> output_primitive_maps;
    std::vector<cldnn::loop::backedge_mapping> back_edges;

    SetLoopInputOutputMap(p, op, inputs, input_primitive_maps, output_primitive_maps, back_edges);

    auto output_names_vec = GetOutputNames(layerName, body_execution_condition_id, output_primitive_maps, back_edges);

    auto config = p.get_config();
    config.set_property(ov::intel_gpu::custom_outputs(output_names_vec));
    config.set_property(ov::intel_gpu::max_dynamic_batch(1));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(op->is_dynamic()));

    // get body program from ov::Model
    ProgramBuilder prog(ov_model, p.get_engine(), config, false, false, p.get_task_executor(), true);
    auto body_program = prog.get_compiled_program();

    // set trip count, initial execution condition, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    const int64_t num_iterations = op->get_num_iterations();
    const cldnn::primitive_id num_iteration_id = layerName + "_numIteration";
    {
        cldnn::mutable_data num_iteration = CreateScalarData<cldnn::mutable_data>(p, num_iteration_id, 0);
        p.add_primitive(*op, std::move(num_iteration));
    }

    const cldnn::loop loopPrimitive(
        layerName,                      /* layer name of this primitive (output id) */
        inputs,                         /* inputs of this layer */
        body_program,                  /* body network */
        trip_count_id,                  /* trip_count data in outer network, always same as num_iterations in TI */
        first_execution_condition_id,   /* initial_execution_condition data in outer network, always true in TI */
        num_iteration_id,               /* actual number of iteration data in body network */
        input_primitive_maps,           /* input mappings connecting outer network and inner network */
        output_primitive_maps,          /* output mappings connecting outer network and inner network */
        back_edges,                     /* back edge mapping */
        num_iterations,                 /* max iteration, i.e. length of iteration axis */
        body_current_iteration_id,
        body_execution_condition_id);

    p.add_primitive(*op, loopPrimitive);
}

static void CreateLoopOp(ProgramBuilder& p, const std::shared_ptr<Loop>& op) {
    CreateCommonLoopOp(p, op, true);
}

/* The above code is a comment in C++ programming language. It is not doing anything in terms of code
execution. It is simply providing information or documentation about the code. */
static void CreateTensorIteratorOp(ProgramBuilder& p, const std::shared_ptr<TensorIterator>& op) {
    CreateCommonLoopOp(p, op, false);
}

REGISTER_FACTORY_IMPL(v5, Loop);
REGISTER_FACTORY_IMPL(v0, TensorIterator);

}  // namespace intel_gpu
}  // namespace ov
