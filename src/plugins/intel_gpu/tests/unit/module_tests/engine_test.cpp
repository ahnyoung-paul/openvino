// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include <memory>

using namespace cldnn;
using namespace ::tests;

template<typename T>
static void run_zero_mem_test(cldnn::engine& engine, cldnn::stream& stream, cldnn::data_types d_type, cldnn::allocation_type type, T sample_data) {
    std::shared_ptr<memory> mem = nullptr;
    layout zero_layout = {{}, d_type, format::bfyx};
    ASSERT_NO_THROW(mem = engine.allocate_memory(zero_layout, type));
    ASSERT_NE(mem, nullptr);

    mem->copy_from(stream, &sample_data, true);
    ASSERT_EQ(mem->get_layout(), zero_layout);
    ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_usm>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(engine));

    T *p_data = reinterpret_cast<T*>(mem->lock(stream, mem_lock_type::read));
    std::cout << "mem " << mem->get_layout().to_short_string() << " - " << *p_data << std::endl;
    ASSERT_EQ(*p_data, sample_data);
}

TEST(engine, memory_creation) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    tests::random_generator rg(GET_SUITE_NAME);

    std::shared_ptr<memory> mem = nullptr;
    layout layout_to_allocate = {{2, 4}, data_types::u8, format::bfyx};
    ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_TRUE(mem->is_allocated_by(engine));

    ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::cl_mem));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_buffer>(mem), nullptr);
    ASSERT_TRUE(mem->is_allocated_by(engine));

    if (engine.supports_allocation(allocation_type::usm_host)) {
        ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::usm_host));
        ASSERT_NE(mem, nullptr);
        ASSERT_EQ(mem->get_layout(), layout_to_allocate);
        ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_usm>(mem), nullptr);
        ASSERT_TRUE(mem->is_allocated_by(engine));

        int64_t sample_data1 = rg.generate_random_val<int64_t>(-3000, 3000, 1);
        run_zero_mem_test<int16_t>(engine, stream, data_types::i64, allocation_type::usm_host, sample_data1);

        float sample_data2 = rg.generate_random_val<float>(3, 1, 10);
        run_zero_mem_test<float>(engine, stream, data_types::f32, allocation_type::usm_host, sample_data2);
    }

    if (engine.supports_allocation(allocation_type::usm_device)) {
        ASSERT_NO_THROW(mem = engine.allocate_memory(layout_to_allocate, allocation_type::usm_device));
        ASSERT_NE(mem, nullptr);
        ASSERT_EQ(mem->get_layout(), layout_to_allocate);
        ASSERT_NE(std::dynamic_pointer_cast<ocl::gpu_usm>(mem), nullptr);
        ASSERT_TRUE(mem->is_allocated_by(engine));

        int64_t sample_data1 = rg.generate_random_val<int64_t>(-3000, 3000, 1);
        run_zero_mem_test<int16_t>(engine, stream, data_types::i64, allocation_type::usm_device, sample_data1);

        float sample_data2 = rg.generate_random_val<float>(3, 1, 10);
        run_zero_mem_test<float>(engine, stream, data_types::f32, allocation_type::usm_device, sample_data2);
    }

    std::vector<uint8_t> host_data(2*4);
    ASSERT_NO_THROW(mem = engine.attach_memory(layout_to_allocate, host_data.data()));
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(mem->get_layout(), layout_to_allocate);
    ASSERT_NE(std::dynamic_pointer_cast<simple_attached_memory>(mem), nullptr);
    ASSERT_FALSE(mem->is_allocated_by(engine));
    ASSERT_EQ(std::dynamic_pointer_cast<simple_attached_memory>(mem)->lock(get_test_stream(), mem_lock_type::read), host_data.data());
}
