// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "primitive_inst.h"
#include <string>
#include <memory>
#include <vector>

namespace cldnn {
template<>
struct typed_program_node<loop> : public typed_program_node_base<loop> {
private:
    using parent = typed_program_node_base<loop>;

    std::vector<loop::io_primitive_map>& input_primitive_maps;
    std::vector<loop::io_primitive_map>& output_primitive_maps;
    std::vector<loop::backedge_mapping>& back_edges;

public:
    typed_program_node(std::shared_ptr<loop> prim, program& prog) :
        parent(prim, prog),
        input_primitive_maps(prim->input_primitive_maps),
        output_primitive_maps(prim->output_primitive_maps),
        back_edges(prim->back_edges) {}

    int64_t get_max_iteration() const { return get_primitive()->get_max_num_iteration(); }
    program::ptr get_body_program() const { return get_primitive()->body_program; }

    const primitive_id& get_trip_count_id() const { return get_primitive()->trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return get_primitive()->first_execution_condition_id; }
    const primitive_id& get_current_iteration_id() const { return get_primitive()->body_current_iteration_id; }
    const primitive_id& get_execution_condition_id() const { return get_primitive()->body_execution_condition_id; }
    const primitive_id& get_num_iteration_id() const { return get_primitive()->num_iteration_id; }

    const std::vector<loop::io_primitive_map>& get_input_primitive_maps() const { return input_primitive_maps; }
    const std::vector<loop::io_primitive_map>& get_output_primitive_maps() const { return output_primitive_maps; }
    const std::vector<loop::backedge_mapping>& get_back_edges() const { return back_edges;}

    void update_primitive_map(const primitive_id& prevID, const primitive_id& newID, bool external_id = true) {
        if (external_id) {
            for (auto& pm : input_primitive_maps) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                }
            }
        } else {
            for (auto& pm : input_primitive_maps) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                }
            }
            for (auto& back_edge : back_edges) {
                if (back_edge.from == prevID) {
                    back_edge.from = newID;
                }
                if (back_edge.to == prevID) {
                    back_edge.to = newID;
                }
            }
        }
    }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->inner_progs = { get_primitive()->body_program };
        return params;
    }
};

using loop_node = typed_program_node<loop>;

template <>
class typed_primitive_inst<loop> : public typed_primitive_inst_base<loop> {
    using parent = typed_primitive_inst_base<loop>;
    using parent::parent;

public:
    struct backedge_memory_mapping {
        enum backedge_type {
            // output memory(from_primitive) of body network needs to be concatenated
            CONCAT_OUTPUT,
            // output memory(from_primitive) of body network does not need to be concateneated
            // input memory is shared by output memory
            SINGLE_SHARED,
            // output memory(from_primitive) of body network does not need to be concateneated
            // input memory is not shared by output memroy
            // each iteration input memory and output memory are swapped
            SINGLE,
        };
        std::shared_ptr<primitive_inst> from_primitive;
        std::shared_ptr<primitive_inst> to_primitive;
        std::vector<memory::ptr> from_mems;
        memory::ptr initial_mem;
        cldnn::stream& stream;
        backedge_type type;
        size_t total_bytes;

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            std::vector<memory::ptr> _from_mems, memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = CONCAT_OUTPUT):
            from_primitive(_from_primitive),
            to_primitive(std::move(_to_primitive)),
            from_mems(_from_mems),
            initial_mem(std::move(_initial_mem)),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            memory::ptr _from_mem, memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = SINGLE_SHARED):
            from_primitive(_from_primitive),
            to_primitive(std::move(_to_primitive)),
            from_mems{std::move(_from_mem)},
            initial_mem(std::move(_initial_mem)),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = SINGLE):
            from_primitive(_from_primitive),
            to_primitive(std::move(_to_primitive)),
            initial_mem(std::move(_initial_mem)),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        void setup_iteration(int64_t iter) const {
            if (type == CONCAT_OUTPUT) {
                if (iter == 0) {
                    to_primitive->set_output_memory(initial_mem);
                } else if (iter > 0) {
                    to_primitive->set_output_memory(from_mems.at(iter - 1));
                } else {
                    throw std::runtime_error("Invalid iteraton count" + std::to_string(iter));
                }
            } else if (type == SINGLE_SHARED && iter == 0) {
                from_mems.front()->copy_from(stream, *initial_mem);
            } else if (type == SINGLE) {
                memory::ptr mem1 = to_primitive->output_memory_ptr();
                if (iter == 0) {
                    mem1->copy_from(stream, *initial_mem);
                } else {
                    memory::ptr mem2 = from_primitive->output_memory_ptr();
                    to_primitive->set_output_memory(std::move(mem2));
                    from_primitive->set_output_memory(mem1);
                }
            }
        }

private:
        void validate_backedge_memory() {
            for (const auto& from_mem : from_mems) {
                const size_t from_mem_bytes = from_mem->get_layout().bytes_count();
                if (from_mem_bytes != total_bytes) {
                    throw std::runtime_error("Invalid backedge memory layout: "
                        "size not matched with that of initial_mem");
                }
            }
        }
    };

    struct concatenated_memory_mapping {
        concatenated_memory_mapping(int64_t axis,
                                    memory::ptr concatenated_mem,
                                    std::vector<memory::ptr> sliced_mems,
                                    stream& stream,
                                    int64_t iteration_elements = 0,
                                    int64_t stride = 0,
                                    int64_t initial_offset = 0) :
            axis(axis),
            concatenated_mem(concatenated_mem),
            sliced_mems(sliced_mems),
            stream(stream),
            bytes_per_element(data_type_traits::size_of(concatenated_mem->get_layout().data_type)),
            batch_size(get_batch_size(concatenated_mem->get_layout(), axis)),
            bytes_batch_stride((static_cast<int64_t>(concatenated_mem->get_layout().count()) / batch_size) * bytes_per_element),
            bytes_iteration(iteration_elements * bytes_per_element),
            bytes_iteration_stride(stride * bytes_iteration),
            bytes_iteration_initial_offset(initial_offset * bytes_iteration) {}

        static int64_t get_batch_size(layout mem_layout, int64_t axis) {
            if (axis < 0) {
                throw std::runtime_error("axis should be positive integer or zero");
            }

            int64_t batch_size = 1;
            for (int64_t i = 0; i < axis; ++i) {
                batch_size *= mem_layout.get_tensor().raw[i];
            }
            for (int64_t i = axis-1; i >= 2; --i) {
                batch_size *= mem_layout.get_tensor().raw[i];
            }
            return batch_size;
        }

        void restore_concatenated_mem() const {
            mem_lock<uint8_t> concat_mem_lock{ concatenated_mem, stream };
            int64_t iteration_offset = bytes_iteration_initial_offset;
            for (const auto& sliced_mem : sliced_mems) {
                for (int64_t batch = 0; batch < batch_size; ++batch) {
                    const int64_t src_offset = batch * bytes_iteration;
                    const int64_t dst_offset = batch * bytes_batch_stride + iteration_offset;
                    mem_lock<uint8_t> sliced_mem_lock{ sliced_mem, stream };
                    uint8_t* src = sliced_mem_lock.data() + src_offset;
                    uint8_t* dst = concat_mem_lock.data() + dst_offset;
                    std::copy(src, src + bytes_iteration, dst);
                }
                iteration_offset += bytes_iteration_stride;
            }
        }

        void setup_sliced_output_memory(uint64_t iteration) const {
            const auto& sliced_output_mem = sliced_mems.at(iteration);
            sliced_data_prim->set_output_memory(sliced_output_mem);
        }

        memory::ptr get_sliced_mem(int64_t iteration) const {
            mem_lock<uint8_t, mem_lock_type::read> from_lock{ concatenated_mem, stream };
            int64_t batch_offset = 0;
            const int64_t iteration_offset = bytes_iteration_initial_offset +
                bytes_iteration_stride * iteration;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                const int64_t src_offset = batch_offset + iteration_offset;
                const int64_t dst_offset = batch * bytes_iteration;
                mem_lock<uint8_t> to_lock{ sliced_mems.at(iteration), stream };
                const auto src = from_lock.begin() + src_offset;
                const auto dst = to_lock.begin() + dst_offset;
                std::copy(src, src + bytes_iteration, dst);
                batch_offset += bytes_batch_stride;
            }
            return sliced_mems.at(iteration);
        }

        const int64_t axis;
        std::shared_ptr<primitive_inst> concat_data_prim;
        std::shared_ptr<primitive_inst> sliced_data_prim;
        memory::ptr concatenated_mem;
        std::vector<memory::ptr> sliced_mems;
        cldnn::stream& stream;
        // element size
        const int64_t bytes_per_element;
        // number of higher level of dimension of slicing axis
        const int64_t batch_size;
        // stride of batch in concatanated memory
        const int64_t bytes_batch_stride;
        // byte size of each iteration per batch in a sliced memory
        const int64_t bytes_iteration;
        // byte size of each iteration (bytes_iteration * batch_size) in a sliced memory
        const int64_t bytes_iteration_stride;
        // byte offset of 1st iteration in a batch in a sliced memory
        const int64_t bytes_iteration_initial_offset;
    };

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(loop_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(const loop_node& node, kernel_impl_params const& impl_param);
    bool preproc_memories_done = false;
    std::vector<backedge_memory_mapping> backedge_memory_mappings;
    std::vector<concatenated_memory_mapping> concatenated_input_mem_mappings;
    std::vector<concatenated_memory_mapping> concatenated_output_mem_mappings;

    static std::string to_string(const loop_node& node);

public:
    typed_primitive_inst(network& network, const loop_node& node);
    network::ptr get_body_network() const { return body_network; }
    void preprocess_input_memory();
    void preprocess_output_memory();
    void preprocess_backedge_memory();
    void update_mapped_memory();
    event::ptr set_output_memory(memory::ptr mem, bool check = true, size_t idx = 0) override;

    void save(BinaryOutputBuffer& ob) const override;
    void load(BinaryInputBuffer& ib) override;

private:
    network::ptr body_network;
    memory::ptr get_external_memory(const primitive_id& external_id) const;
    std::vector<memory::ptr> get_sliced_mem(const primitive_id& internal_id) const;
    std::vector<loop::io_primitive_map> _input_primitive_maps;
    std::vector<loop::io_primitive_map> _output_primitive_maps;
    std::vector<loop::backedge_mapping> _back_edges;
    primitive_id _trip_count_id;
    primitive_id _initial_execution_id;
    primitive_id _current_iteration_id;
    primitive_id _condition_id;
    primitive_id _num_iteration_id;
    int64_t _max_iteration = 0;
};

using loop_inst = typed_primitive_inst<loop>;
}  // namespace cldnn
