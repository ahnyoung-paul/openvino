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
#include <memory>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

namespace ov::intel_gpu {

// [LORA-DIAG] Accumulators to split VariableState::set_state cost into
// buffer (re)allocation vs host->device convert_and_copy. Enabled via env
// OV_LORA_DIAG=1. Reported by AdapterController through stderr on the genai side;
// here we just accumulate and dump periodically.
namespace {
struct SetStateDiag {
    // per-window accumulators (window = one reported line = one full LoRA switch)
    double update_buffer_us = 0.0;   // sum of update_device_buffer() time
    double convert_copy_us = 0.0;    // sum of convert_and_copy() time (H2D copy path)
    size_t copy_count = 0;           // number of convert_and_copy() calls in window
    size_t copy_bytes = 0;           // total bytes copied H2D in window
    double copy_min_us = 0.0;        // fastest single copy in window
    double copy_max_us = 0.0;        // slowest single copy in window
    double swap_us = 0.0;            // sum of no-copy pointer-swap time (RemoteTensor path)
    size_t swap_count = 0;           // number of no-copy swaps in window
    size_t calls = 0;                // total set_state calls (monotonic)
    // Window size = one full LoRA switch. Default 756 = 252 layers x 3 tensors (alpha/A/B).
    // Override via OV_LORA_DIAG_WINDOW if the layer/tensor count differs for a given model.
    size_t window = []{
        const char* w = std::getenv("OV_LORA_DIAG_WINDOW");
        if (w && w[0] != '\0') { long v = std::atol(w); if (v > 0) return static_cast<size_t>(v); }
        return static_cast<size_t>(756);
    }();
    bool enabled = []{
        const char* e = std::getenv("OV_LORA_DIAG");
        return e && e[0] == '1';
    }();
};
SetStateDiag& set_state_diag() {
    static SetStateDiag d;
    return d;
}

// [LORA-DIAG] Emit one line per full switch, distinguishing the no-copy (RemoteTensor swap)
// portion from the H2D-copy portion so mixed switches are unambiguous. MODE classifies the
// whole switch: SWAP = all no-copy, H2D = all copy, MIXED = both (partial fallback).
void set_state_diag_maybe_report(SetStateDiag& diag) {
    if (!diag.enabled || diag.window == 0 || (diag.calls % diag.window) != 0)
        return;
    const char* mode = (diag.copy_count == 0) ? "SWAP(no-copy)"
                     : (diag.swap_count == 0) ? "H2D(copy)"
                                              : "MIXED";
    const double swap_ms = diag.swap_us / 1000.0;
    const double copy_ms = diag.convert_copy_us / 1000.0;
    const double mb = diag.copy_bytes / (1024.0 * 1024.0);
    const double avg_copy_us = diag.copy_count ? diag.convert_copy_us / diag.copy_count : 0.0;
    const double eff_gbps = copy_ms > 0.0
        ? (diag.copy_bytes / (1024.0 * 1024.0 * 1024.0)) / (copy_ms / 1000.0)
        : 0.0;
    std::cerr << "[LORA-DIAG][GPU] switch calls=" << diag.calls
              << "  MODE=" << mode
              << "  | no-copy swaps=" << diag.swap_count
              << " swap_time=" << swap_ms << "ms"
              << "  | H2D copies=" << diag.copy_count
              << " copy_time=" << copy_ms << "ms"
              << " avg=" << avg_copy_us << "us"
              << " min=" << diag.copy_min_us << "us"
              << " max=" << diag.copy_max_us << "us"
              << " bytes=" << mb << "MB"
              << " eff_bw=" << eff_gbps << "GB/s"
              << "  | alloc=" << diag.update_buffer_us / 1000.0 << "ms"
              << std::endl;
    diag.update_buffer_us = 0.0;
    diag.convert_copy_us = 0.0;
    diag.copy_count = 0;
    diag.copy_bytes = 0;
    diag.copy_min_us = 0.0;
    diag.copy_max_us = 0.0;
    diag.swap_us = 0.0;
    diag.swap_count = 0;
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
    // [LORA-DIAG] report the runtime type of the FIRST non-remote tensor seen after any remote
    // ones, so we can identify which tensors fall back to the host->copy path.
    if (set_state_diag().enabled) {
        const ov::ITensor* p = state._ptr.get();
        bool is_remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(p) != nullptr;
        static int shown = 0;
        static bool seen_remote = false;
        if (is_remote) seen_remote = true;
        if (seen_remote && !is_remote && shown < 5) {
            shown++;
            std::cerr << "[LORA-DIAG][SETSTATE] NON-REMOTE after remote: type=" << (p ? typeid(*p).name() : "null")
                      << "  shape_rank=" << state->get_shape().size()
                      << "  bytes=" << state->get_byte_size() << std::endl;
        }
    }
    // PoC (CVS-187607): if the incoming tensor already lives in this device's memory
    // (a RemoteTensor), skip the host->device convert_and_copy entirely and just adopt
    // its memory buffer (pointer swap). This turns the ~95% LoRA-switch bottleneck into
    // an O(1) rebind, provided the caller keeps the source tensor alive (resident adapter).
    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(state._ptr.get())) {
        auto& diag = set_state_diag();
        auto t0 = std::chrono::steady_clock::now();
        auto mem = remote->get_original_memory();
        m_layout.set_partial_shape(state->get_shape());
        m_memory = mem;
        actual_size = m_memory->size();
        set();
        auto t1 = std::chrono::steady_clock::now();
        if (diag.enabled) {
            diag.swap_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            diag.swap_count += 1;
            ++diag.calls;
            set_state_diag_maybe_report(diag);
        }
        return;
    }

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

    auto& diag = set_state_diag();
    auto t_alloc0 = std::chrono::steady_clock::now();
    update_device_buffer();
    auto t_alloc1 = std::chrono::steady_clock::now();
    if (diag.enabled) {
        diag.update_buffer_us +=
            std::chrono::duration<double, std::micro>(t_alloc1 - t_alloc0).count();
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

    auto t_copy0 = std::chrono::steady_clock::now();
    convert_and_copy(state._ptr.get(), m_memory, m_context->get_engine().get_service_stream(), src_layout, m_transpose_required);
    auto t_copy1 = std::chrono::steady_clock::now();
    if (diag.enabled) {
        double this_copy_us = std::chrono::duration<double, std::micro>(t_copy1 - t_copy0).count();
        diag.convert_copy_us += this_copy_us;
        diag.copy_count += 1;
        diag.copy_bytes += state->get_byte_size();
        if (diag.copy_min_us == 0.0 || this_copy_us < diag.copy_min_us) diag.copy_min_us = this_copy_us;
        if (this_copy_us > diag.copy_max_us) diag.copy_max_us = this_copy_us;
        ++diag.calls;
        set_state_diag_maybe_report(diag);
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
