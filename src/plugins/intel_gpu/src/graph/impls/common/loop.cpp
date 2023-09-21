// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "loop_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace common {

// read scala value from data primitive
static int64_t read_scalar_value(memory::ptr mem, stream& stream) {
    int64_t trip_count = 0;
    const layout& prim_layout = mem->get_layout();

    switch (prim_layout.data_type) {
    case data_types::u8: {
        mem_lock<uint8_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i8: {
        mem_lock<int8_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i32: {
        mem_lock<int32_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i64: {
        mem_lock<int64_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    default:
        throw std::runtime_error("Invalid data type : " + data_type_traits::name(prim_layout.data_type));
    }
    return trip_count;
}

template<typename T>
static inline void validate_input_value(int64_t input) {
    if (input < std::numeric_limits<T>::min() || input > std::numeric_limits<T>::max()) {
        throw std::runtime_error("Invalid data value : " + std::to_string(input));
    }
}

static void write_scalar_value(memory::ptr mem, stream& stream, int64_t input) {
    const layout& prim_layout = mem->get_layout();

    switch (prim_layout.data_type) {
    case data_types::u8: {
        validate_input_value<uint8_t>(input);
        mem_lock<uint8_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<uint8_t>(input);
        break;
    }
    case data_types::i8: {
        validate_input_value<int8_t>(input);
        mem_lock<int8_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<int8_t>(input);
        break;
    }
    case data_types::i32: {
        validate_input_value<int32_t>(input);
        mem_lock<int32_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<int32_t>(input);
        break;
    }
    case data_types::i64: {
        mem_lock<int64_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = input;
        break;
    }
    default:
        throw std::runtime_error("Invalid data type : " + data_type_traits::name(prim_layout.data_type));
    }
}

struct loop_impl : typed_primitive_impl<loop> {
    using parent = typed_primitive_impl<loop>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::loop_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<loop_impl>(*this);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    loop_impl() : parent() {}

    loop_impl(const loop_impl& other) : typed_primitive_impl<loop>(other),
        _back_edges(other._back_edges) {}

    explicit loop_impl(const loop_node& node) {
        set_node_params(node);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<loop>());
        const auto& node = arg.as<loop>();
        _back_edges = node.get_back_edges();
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, loop_inst& instance) override {
        const auto& impl_params = instance.get_impl_params();
        const auto& primitive = impl_params->typed_desc<loop>();
        auto& outer_network = instance.get_network();
        auto& stream = outer_network.get_stream();

        const auto max_num_iteration = primitive->get_max_num_iteration();
        auto body_network = instance.get_body_network();

        auto ev = stream.create_user_event(false);

        if (!instance.is_dynamic()) {
            if (!instance.preproc_memories_done) {
                instance.preprocess_output_memory();
                instance.preprocess_input_memory();
                instance.preprocess_backedge_memory();
                instance.preproc_memories_done = true;
            }
        } else {
            instance.preproc_memories_done = false;
            instance.preprocess_input_memory();
        }

        //////////////////////////////////////////
        // memory pointers for outer network
        //////////////////////////////////////////
        // read trip_count from outer network
        int64_t trip_count = -1;
        if (!primitive->trip_count_id.empty()) {
            memory::ptr trip_count_mem = outer_network.get_primitive(primitive->trip_count_id)->output_memory_ptr();
            trip_count = read_scalar_value(std::move(trip_count_mem), stream);
        }
        trip_count = (trip_count > 0)? trip_count : max_num_iteration;

        std::cout << "trip_count : " << trip_count << std::endl;

        // read initial execution condition from outer network
        int64_t execution_condition = 1;
        if (!primitive->first_execution_condition_id.empty()) {
            memory::ptr first_execution_condition_mem = outer_network.get_primitive(primitive->first_execution_condition_id)->output_memory_ptr();
            execution_condition = read_scalar_value(first_execution_condition_mem, stream);
        }

        //////////////////////////////////////////
        // memory pointers for body network
        //////////////////////////////////////////
        // shortcut of execution_condition memory in body network
        memory::ptr body_execution_condition_mem = nullptr;
        if (!primitive->body_execution_condition_id.empty()) {
            body_execution_condition_mem = body_network->get_primitive(primitive->body_execution_condition_id)->output_memory_ptr();
        }

        // shortcut of current_iteration memory in body network
        memory::ptr body_current_iteration_mem = nullptr;
        if (!primitive->body_current_iteration_id.empty()) {
            body_current_iteration_mem = body_network->get_primitive(primitive->body_current_iteration_id)->output_memory_ptr();
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;

        // If there are concatenated_output_mem_mappings or backedge_memory_mappings we need to wait for
        // previous tasks before accessing memory in get_sliced_mem() and setup_iteration() functions
        if (!concatenated_input_mem_mappings.empty() || !instance.backedge_memory_mappings.empty()) {
            for (auto& e : events) {
                e->wait();
            }
        }

        // Set sliced input data
        for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
            const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
            memory::ptr mem = concatenated_input.get_sliced_mem(0);
            if (mem) {
                body_network->set_input_data(concatenated_input.sliced_data_prim->id(), mem);
            } else {
                CLDNN_ERROR_MESSAGE(instance.id(), "sliced input memory of loop is not allocated properly");
            }
        }

        std::vector<event::ptr> all_events;
        std::vector<event::ptr> loop_carried_dep(events.begin(), events.end());
        int64_t current_iteration_idx = 0;
        while (current_iteration_idx < trip_count && execution_condition) {
            if (body_current_iteration_mem != nullptr) {
                write_scalar_value(body_current_iteration_mem, stream, current_iteration_idx);
                body_network->set_input_data(primitive->body_current_iteration_id, body_current_iteration_mem);
            }

            // Copy & Set sliced input memory
            for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
                const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
                memory::ptr mem = concatenated_input.get_sliced_mem(current_iteration_idx);
                if (mem) {
                    concatenated_input.sliced_data_prim->set_output_memory(mem);
                } else {
                    CLDNN_ERROR_MESSAGE(instance.id(), "sliced input memory of loop is not allocated properly");
                }
            }

            // Set backedges
            for (const auto& backedge_memory_mapping : instance.backedge_memory_mappings) {
                backedge_memory_mapping.setup_iteration(current_iteration_idx);
            }

            // Set sliced output memory
            for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
                concat_output_mem_mapping.setup_sliced_output_memory(current_iteration_idx);
            }

            // execute body network
            body_network->execute(loop_carried_dep);

            loop_carried_dep.clear();
            for (const auto& backedge : _back_edges) {
                event::ptr body_event;
                if (body_network->has_event(backedge.from))
                    body_event = body_network->get_primitive_event(backedge.from);
                loop_carried_dep.emplace_back(body_event);
            }

            // Collect output events for waiting for all iterations finishing
            for (auto& out : body_network->get_outputs()) {
                auto output_id = out->id();
                if (body_network->has_event(output_id)) {
                    auto output_event = body_network->get_primitive_event(output_id);
                    all_events.push_back(output_event);
                }
            }

            // execution condition is the result of body network execution
            if (body_execution_condition_mem != nullptr) {
                execution_condition = read_scalar_value(body_execution_condition_mem, body_network->get_stream());
            }

            // update index & execution condition for the next iteration
            ++current_iteration_idx;
            if (!loop_carried_dep.empty())
                stream.wait_for_events(loop_carried_dep);

            if (instance.is_dynamic()) {
                if (!instance.preproc_memories_done) {
                    instance.update_output_layout();
                    instance.update_output_mapped_memory();
                    instance.preprocess_output_memory();
                    instance.preprocess_backedge_memory();
                    instance.preproc_memories_done = true;
                }
            }
        }

        // Reset network and wait for all collected events
        body_network->reset_execution(false);
        stream.wait_for_events(all_events);

        // Concatenate sliced output to the outer network
        for (size_t i = 0; i < concatenated_output_mem_mappings.size(); ++i) {
            const auto& concat_output = concatenated_output_mem_mappings.at(i);
            concat_output.restore_concatenated_mem();
        }

        // Update actual num iteration
        if (!primitive->num_iteration_id.empty()) {
            // update num_iterations (actual number of iterations)
            memory::ptr num_actual_iterations_mem = outer_network.get_primitive(primitive->num_iteration_id)->output_memory_ptr();
            write_scalar_value(num_actual_iterations_mem, stream, current_iteration_idx);
        }

        instance.update_output_layout();

        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const loop_node& arg, const kernel_impl_params&) {
        return make_unique<loop_impl>(arg);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _back_edges;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _back_edges;
    }

private:
    std::vector<cldnn::loop::backedge_mapping> _back_edges;
};

namespace detail {
attach_loop_common::attach_loop_common() {
    implementation_map<loop>::add(impl_types::common,
                                    shape_types::dynamic_shape,
                                    loop_impl::create,
                                    {},
                                    {});
    implementation_map<loop>::add(impl_types::common, loop_impl::create, {});
}
}  // namespace detail

}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::loop_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::loop)
