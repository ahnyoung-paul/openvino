// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <cstdlib>
#include <memory>
#include <chrono>
#include <iostream>

namespace ov::intel_gpu {

namespace {

// CVS-187607: LoRA switch time optimization.
//
// The default set_state() path uploads the incoming tensor into the state's device
// buffer with a blocking convert_and_copy() on the service stream. For LoRA adapter
// switching this happens 756 times per switch (252 layers x {alpha, A, B}) and
// accounts for ~95% of the total switch time.
//
// When the caller (GenAI) already produced the state contents in device memory, the
// copy is redundant: the state can simply adopt the incoming device buffer, which is
// O(1). This is the same no-copy principle already used by set_memory().
//
// Gated at runtime so a single binary can A/B the two paths:
//   OV_LORA_NO_COPY_SWAP=1  -> adopt RemoteTensor memory (no copy)
//   unset / 0               -> original host->device copy path
//
// Precondition for the swap: the caller must keep the source device tensor alive for
// as long as the state is in use (resident adapter tensors).
bool no_copy_swap_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("OV_LORA_NO_COPY_SWAP");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return enabled;
}

// CVS-187607: diagnostic-only wall-time vs GPU-copy-event breakdown for set_state().
// Gate: OV_LORA_SET_STATE_PROFILE=1. Accumulates instead of printing per call (252
// calls/switch would otherwise interleave), dumps totals once at process exit.
bool set_state_profile_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("OV_LORA_SET_STATE_PROFILE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return enabled;
}

struct SetStateProfileStats {
    uint64_t calls = 0, bytes = 0;
    uint64_t layout_alloc_ns = 0;    // update_device_buffer() + layout bookkeeping
    uint64_t copy_enqueue_ns = 0;    // host-side cost of issuing the copy
    uint64_t copy_wait_ns = 0;       // host-side wait (queue wait + GPU exec, from host view)
    uint64_t gpu_submission_ns = 0;  // OpenCL profiling: queued -> submitted
    uint64_t gpu_starting_ns = 0;    // OpenCL profiling: submitted -> start
    uint64_t gpu_executing_ns = 0;   // OpenCL profiling: start -> end
    uint64_t gpu_profiled_calls = 0;

    ~SetStateProfileStats() {
        if (calls == 0)
            return;
        auto ms = [](uint64_t ns) { return ns / 1.0e6; };
        std::cerr << "[CVS-187607] set_state profile: calls=" << calls << " bytes=" << bytes
                   << " layout/alloc=" << ms(layout_alloc_ns) << "ms"
                   << " copy_enqueue=" << ms(copy_enqueue_ns) << "ms"
                   << " copy_wait(host)=" << ms(copy_wait_ns) << "ms";
        if (gpu_profiled_calls > 0) {
            std::cerr << " gpu[queued->submit]=" << ms(gpu_submission_ns) << "ms"
                       << " gpu[submit->start]=" << ms(gpu_starting_ns) << "ms"
                       << " gpu[start->end]=" << ms(gpu_executing_ns) << "ms"
                       << " (profiled_calls=" << gpu_profiled_calls << ")";
        } else {
            std::cerr << " gpu profiling unavailable (need ov::enable_profiling(true))";
        }
        std::cerr << std::endl;
    }
};

SetStateProfileStats& set_state_profile_stats() {
    static SetStateProfileStats stats;
    return stats;
}

}  // namespace

VariableState::VariableState(const VariableStateInfo& info, RemoteContextImpl::Ptr context, std::shared_ptr<cldnn::ShapePredictor> shape_predictor)
    : VariableStateBase{info.m_id, context}
    , m_layout(info.m_layout)
    , m_user_specified_type(info.m_user_specified_type)
    , m_shape_predictor(shape_predictor)
    , m_prim_inst(info.m_release_variable_inst)
    , m_transpose_required(info.transpose_required)
    , m_initial_layout(info.m_layout) {
    update_device_buffer();
}

void VariableState::reset() {
    m_is_set = false;
    set_layout(m_initial_layout);
    for (auto& user : m_prim_inst) {
        if (const auto prim = user.lock(); prim) {
            prim->release_variable();
        }
    }
}

cldnn::memory::ptr VariableState::get_memory() const {
    return m_memory;
}

const cldnn::layout& VariableState::get_layout() const {
    return m_layout;
}

void VariableState::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    GPU_DEBUG_TRACE_DETAIL << m_name << " : Update memory (Ptr : " << new_mem->buffer_ptr()
                           << ", layout : " << actual_layout.to_short_string() << ")" << std::endl;
    m_memory = new_mem;
    m_layout = actual_layout;
    actual_size = m_memory->size();
    update_device_buffer();
}

void VariableState::set_layout(const cldnn::layout& new_layout) {
    if (m_layout == new_layout)
        return;
    m_layout = new_layout;
    GPU_DEBUG_TRACE_DETAIL << m_name << " : " << "Update state layout to " << new_layout.to_short_string() << std::endl;
    update_device_buffer();
}

void VariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
    // CVS-187607: no-copy fast path. If the incoming tensor already lives in device
    // memory, adopt its buffer instead of copying into ours. Falls through to the
    // regular copy path for host tensors, so mixed callers stay correct.
    if (no_copy_swap_enabled()) {
        if (auto remote = dynamic_cast<const RemoteTensorImpl*>(state._ptr.get())) {
            m_memory = remote->get_original_memory();
            m_layout.set_partial_shape(state->get_shape());
            actual_size = m_memory->size();
            GPU_DEBUG_TRACE_DETAIL << m_name << " : LoRA no-copy swap (Ptr : " << m_memory->buffer_ptr()
                                   << ", layout : " << m_layout.to_short_string() << ")" << std::endl;
            set();
            return;
        }
    }

    const bool profile = set_state_profile_enabled();
    auto t_begin = std::chrono::steady_clock::now();

    auto src_shape = state->get_shape();
    size_t src_rank = src_shape.size();
    cldnn::padding::DynamicDimsMask dynamic_pad_dims;
    for (size_t i = 0; i < src_rank; i++) {
        dynamic_pad_dims[i] = m_layout.data_padding._dynamic_dims_mask[i];
    }
    m_layout.data_padding = cldnn::padding(std::vector<ov::Dimension::value_type>(src_rank, 0),
                                           std::vector<ov::Dimension::value_type>(src_rank, 0),
                                           dynamic_pad_dims);
    auto src_stride = state->get_strides();
    for (size_t i = 0; i < src_rank; ++i) {
        src_stride[i] /= state->get_element_type().bitwidth() / 8;
    }
    m_layout.set_partial_shape(src_shape);
    update_device_buffer();

    auto t_layout_done = std::chrono::steady_clock::now();
    if (profile) {
        set_state_profile_stats().layout_alloc_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_layout_done - t_begin).count();
    }

    if (actual_size == 0) {
        set();
        return;
    }

    // check whether the src tensor is padded
    std::vector<size_t> src_stride_no_pad(src_rank, 1);
    std::vector<ov::Dimension::value_type> upper_pad(src_rank, 0);
    std::vector<ov::Dimension::value_type> lower_pad(src_rank, 0);
    for (int32_t i = static_cast<int32_t>(src_stride.size()) - 1; i >= 0; --i) {
        if (i <= static_cast<int32_t>(src_stride.size()) - 2)
            src_stride_no_pad[i] = src_stride_no_pad[i + 1] * src_shape[i + 1];
        if (src_stride[i] != src_stride_no_pad[i]) {
            OPENVINO_ASSERT(src_stride[i] > src_stride_no_pad[i]);
            size_t padded_size = src_stride[i] / src_stride[i + 1];
            size_t non_padded_size = src_stride_no_pad[i] / src_stride_no_pad[i + 1];
            ov::Dimension::value_type pad_dim = i + 1;
            upper_pad[pad_dim] = static_cast<ov::Dimension::value_type>(padded_size) - static_cast<ov::Dimension::value_type>(non_padded_size);
        }
    }
    cldnn::padding src_padd = cldnn::padding(lower_pad, upper_pad);
    auto src_fmt = cldnn::format::get_default_format(src_rank);
    auto src_layout = cldnn::layout(ov::PartialShape(src_shape), state->get_element_type(), src_fmt, src_padd);

    auto& stream = m_context->get_engine().get_service_stream();

    // Fast-path condition mirrors convert_and_copy(ITensor*, memory::ptr, ...): same dtype,
    // no transpose, host tensor -> plain device copy. Bypass the helper only when profiling
    // is on, so we can request a non-blocking copy and read the real OpenCL event timings
    // (queued->submit->start->end) instead of one opaque blocking wall-clock number.
    if (profile && !m_transpose_required && state->get_element_type() == m_memory->get_layout().data_type &&
        !dynamic_cast<const RemoteTensorImpl*>(state._ptr.get())) {
        auto& stats = set_state_profile_stats();
        auto t_enqueue_start = std::chrono::steady_clock::now();
        auto event = m_memory->copy_from(stream, state->data(), /*blocking=*/false);
        auto t_enqueue_done = std::chrono::steady_clock::now();
        if (event)
            event->wait();
        auto t_wait_done = std::chrono::steady_clock::now();

        stats.calls++;
        stats.bytes += m_memory->size();
        stats.copy_enqueue_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_enqueue_done - t_enqueue_start).count();
        stats.copy_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_wait_done - t_enqueue_done).count();

        if (event) {
            for (const auto& interval : event->get_profiling_info()) {
                auto ns = static_cast<uint64_t>(interval.value->value().count());
                if (interval.stage == cldnn::instrumentation::profiling_stage::submission)
                    stats.gpu_submission_ns += ns;
                else if (interval.stage == cldnn::instrumentation::profiling_stage::starting)
                    stats.gpu_starting_ns += ns;
                else if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                    stats.gpu_executing_ns += ns;
                    stats.gpu_profiled_calls++;
                }
            }
        }
    } else {
        auto t_copy_start = std::chrono::steady_clock::now();
        convert_and_copy(state._ptr.get(), m_memory, stream, src_layout, m_transpose_required);
        if (profile) {
            auto& stats = set_state_profile_stats();
            stats.calls++;
            stats.bytes += m_memory->size();
            stats.copy_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_copy_start).count();
        }
    }
    set();
}

void VariableState::update_device_buffer() {
    OPENVINO_ASSERT(m_context != nullptr, "m_context should not be null.");
    if (m_layout.is_dynamic() || m_layout.bytes_count() == 0) {
        m_shape_predictor->reset();
        m_memory.reset();
        actual_size = 0;
        return;
    }

    if (actual_size < m_layout.bytes_count()) {
        const auto alloc_type = m_context->get_engine().use_unified_shared_memory() ? cldnn::allocation_type::usm_device : cldnn::allocation_type::cl_mem;
        const auto current_buf_size = m_layout.get_padded_dims();
        ov::Shape current_shape(current_buf_size.begin(), current_buf_size.end());
        const auto alloc_shape = predict_shape(m_name, cldnn::layout(current_shape, m_layout.data_type, m_layout.format), *m_shape_predictor);
        const auto alloc_layout = cldnn::layout(alloc_shape, m_layout.data_type, m_layout.format);
        m_memory = m_context->get_engine().allocate_memory(alloc_layout, alloc_type, false);
        actual_size = std::max(actual_size, alloc_layout.bytes_count());
    }

    OPENVINO_ASSERT(m_memory != nullptr, "m_memory is nullptr!!!");
    m_memory = m_context->get_engine().reinterpret_buffer(*m_memory, m_layout);
}

ov::element::Type VariableState::get_user_specified_type() const {
    return m_user_specified_type != ov::element::dynamic ? m_user_specified_type : ov::element::Type(m_layout.data_type);
}

ov::SoPtr<ov::ITensor> VariableState::get_state() const {
    if (m_memory == nullptr) {
        const auto& pshape = m_layout.get_partial_shape();
        const auto& shape = get_tensor_shape(pshape);
        return m_context->create_host_tensor(get_user_specified_type(), shape);
    }

    auto tensor = m_context->create_host_tensor(get_user_specified_type(), m_memory->get_layout().get_shape());

    convert_and_copy(m_memory, tensor._ptr.get(), m_context->get_engine().get_service_stream());

    return tensor;
}

}  // namespace ov::intel_gpu
