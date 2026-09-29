// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "openvino/reference/convert.hpp"

#include <memory>
#include <vector>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

template <typename Src, typename Dst>
void check_usm_host_to_device_conversion(const ov::Shape& shape,
                                         const ov::element::Type& src_type,
                                         const ov::element::Type& dst_type) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    auto src = std::make_shared<USMHostTensor>(context, src_type, shape);
    auto src_mem = src->get_impl()->get_original_memory();
    ASSERT_EQ(src_mem->get_allocation_type(), allocation_type::usm_host);

    const auto count = ov::shape_size(shape);
    std::vector<Src> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = static_cast<Src>(static_cast<float>(static_cast<int>(i % 257) - 128) / 7.f);
    set_values(src_mem, values);

    std::vector<Dst> expected(count);
    ov::reference::convert(values.data(), expected.data(), count);

    const auto fmt = format::get_default_format(shape.size());
    layout src_layout{shape, src_type, fmt};
    layout dst_layout{shape, dst_type, fmt};
    auto dst_mem = engine.allocate_memory(dst_layout, allocation_type::usm_device);
    ASSERT_EQ(dst_mem->get_allocation_type(), allocation_type::usm_device);

    OV_ASSERT_NO_THROW(convert_and_copy(src.get(), dst_mem, stream, src_layout, false));

    cldnn::mem_lock<Dst, mem_lock_type::read> actual(dst_mem, stream);
    for (size_t i = 0; i < count; ++i)
        ASSERT_EQ(actual[i], expected[i]) << "element " << i;
}

}  // namespace

// IRemoteTensor::data() always throws, so if the fast-path `return` is ever dropped again and
// execution falls through to the fallback path, this call throws instead of silently passing.
TEST(convert_and_copy_test, remote_tensor_fast_path_does_not_fall_through) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    const ov::Shape shape{1, 2, 2, 2};
    const ov::element::Type et = ov::element::f32;

    auto src_remote = std::make_shared<RemoteTensorImpl>(context, shape, et);
    auto src_mem = src_remote->get_original_memory();
    std::vector<float> src_values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    set_values(src_mem, src_values);

    layout dst_layout{shape, et, format::bfyx};
    auto dst_mem = engine.allocate_memory(dst_layout);

    OV_ASSERT_NO_THROW(convert_and_copy(src_remote.get(), dst_mem, stream, dst_layout, false));

    cldnn::mem_lock<float, mem_lock_type::read> dst_ptr(dst_mem, stream);
    for (size_t i = 0; i < src_values.size(); ++i) {
        ASSERT_EQ(dst_ptr[i], src_values[i]);
    }
}

TEST(convert_and_copy_test_paul, usm_host_bf16_to_f16_small_and_large) {
    if (!get_test_engine().supports_allocation(allocation_type::usm_host) ||
        !get_test_engine().supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "USM host or device allocation is not supported";

    check_usm_host_to_device_conversion<ov::bfloat16, ov::float16>({1, 2, 2, 3}, ov::element::bf16, ov::element::f16);
    check_usm_host_to_device_conversion<ov::bfloat16, ov::float16>({1, 1, 256, 256}, ov::element::bf16, ov::element::f16);
}

TEST(convert_and_copy_test_paul, usm_host_f32_to_f16) {
    if (!get_test_engine().supports_allocation(allocation_type::usm_host) ||
        !get_test_engine().supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "USM host or device allocation is not supported";

    check_usm_host_to_device_conversion<float, ov::float16>({1, 2, 3, 4, 5}, ov::element::f32, ov::element::f16);
}
