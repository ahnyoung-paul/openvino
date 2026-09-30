// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/state_conversion_executor.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <memory>
#include <limits>
#include <utility>

namespace ov::intel_gpu {

namespace {

bool try_gpu_conversion(const ov::ITensor* src, const cldnn::memory::ptr& dst,
                        cldnn::stream& stream, const cldnn::layout& src_layout,
                        const std::shared_ptr<cldnn::program>& program) {
    if (!program || program->get_engine().runtime_type() != cldnn::runtime_types::ocl ||
        src_layout.data_padding || dst->get_layout().data_padding ||
        !cldnn::format::is_default_format(src_layout.format) ||
        src_layout.format != dst->get_layout().format ||
        src->get_shape() != dst->get_layout().get_shape())
        return false;

    const cldnn::state_conversion_key key{static_cast<ov::element::Type_t>(src->get_element_type()), dst->get_layout().data_type};
    const auto& device_info = program->get_engine().get_device_info();
    if (!cldnn::state_conversion_executor::supports(key) ||
        ((key.first == cldnn::data_types::f64 || key.second == cldnn::data_types::f64) && !device_info.supports_fp64) ||
        ((key.first == cldnn::data_types::f16 || key.second == cldnn::data_types::f16) && !device_info.supports_fp16))
        return false;

    const auto& shape = src->get_shape();
    const auto& strides = src->get_strides();
    if (shape.empty() || strides.size() != shape.size())
        return false;
    size_t expected_stride = src->get_element_type().size();
    for (size_t i = shape.size(); i-- > 0;) {
        if (strides[i] != expected_stride ||
            shape[i] > std::numeric_limits<size_t>::max() / expected_stride)
            return false;
        expected_stride *= shape[i];
    }

    cldnn::memory::ptr src_memory;
    if (const auto* host = dynamic_cast<const USMHostTensor*>(src))
        src_memory = host->get_impl()->get_original_memory();
    else if (const auto* remote = dynamic_cast<const RemoteTensorImpl*>(src))
        src_memory = remote->get_original_memory();
    else
        return false;

    if (!src_memory || (src_memory->get_allocation_type() != cldnn::allocation_type::usm_host &&
                        src_memory->get_allocation_type() != cldnn::allocation_type::usm_device) ||
        (dst->get_allocation_type() != cldnn::allocation_type::usm_device &&
         dst->get_allocation_type() != cldnn::allocation_type::cl_mem))
        return false;
    auto* src_engine = src_memory->get_engine();
    auto* dst_engine = dst->get_engine();
    if (!src_engine || !dst_engine || src_engine->runtime_type() != cldnn::runtime_types::ocl ||
        dst_engine->runtime_type() != cldnn::runtime_types::ocl ||
        src_engine->get_user_context(cldnn::runtime_types::ocl) !=
            dst_engine->get_user_context(cldnn::runtime_types::ocl) ||
        program->get_engine().get_user_context(cldnn::runtime_types::ocl) !=
            dst_engine->get_user_context(cldnn::runtime_types::ocl))
        return false;

    auto executor = program->get_state_conversion_executor();
    OPENVINO_ASSERT(executor && executor->has_kernel(key), "[GPU] State conversion kernel was not prepared");
    executor->execute(key, src_memory, dst, stream, ov::shape_size(shape));
    return true;
}

}  // namespace

VariableState::VariableState(const VariableStateInfo& info, RemoteContextImpl::Ptr context,
                             std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                             std::shared_ptr<cldnn::program> program)
    : VariableStateBase{info.m_id, context}
    , m_layout(info.m_layout)
    , m_user_specified_type(info.m_user_specified_type)
    , m_shape_predictor(shape_predictor)
    , m_prim_inst(info.m_release_variable_inst)
    , m_transpose_required(info.transpose_required)
    , m_program(std::move(program))
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
    if (!m_transpose_required && state->get_element_type() == get_user_specified_type() &&
        state->get_element_type() != m_layout.data_type &&
        try_gpu_conversion(state._ptr.get(), m_memory, stream, src_layout, m_program)) {
        set();
        return;
    }
    convert_and_copy(state._ptr.get(), m_memory, stream, src_layout, m_transpose_required);
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
