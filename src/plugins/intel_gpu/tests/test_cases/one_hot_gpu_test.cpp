// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/one_hot.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template <typename T>
VF<T> one_hot_cpu(VF<T> &indices, std::vector<tensor::value_type> &indices_shape,
        const size_t depth, const int64_t one_hot_axis) {
    // Step 1: Set off_value(0) to the output
    const size_t num_ind = [&] {
        size_t size = 1;
        for (auto d : indices_shape) {
            size *= d;
        }
        return size;
    }();

    VF<T> out(num_ind * depth);
    std::fill(out.begin(), out.end(), 0);

    // Step 2: Write on_value(1) at needed positions
    const size_t inner_block = [&] {
        size_t mul = 1;
        for (size_t i = one_hot_axis; i < indices_shape.size(); ++i)
            mul *= indices_shape[i];
        return mul;
    }();

    for (size_t outer_i = 0; outer_i < num_ind; outer_i += inner_block) {
        for (size_t inner_i = 0; inner_i < inner_block; inner_i++) {
            auto input_val = indices[outer_i + inner_i];
            // Negative indices are ignored
            if ((input_val >= 0) && (static_cast<size_t>(input_val) < depth)) {
                auto oh_index = static_cast<size_t>(input_val);
                size_t output_idx = (outer_i * depth + inner_i + oh_index * inner_block);
                out[output_idx] = 1;
            }
        }
    }
    return out;
}


template <typename T>
void generic_one_hot_test_int(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, tensor shape,
    uint16_t one_hot_axis, int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0, int output_padding_x = 0) {
    std::vector<tensor::value_type> output_dims = shape.sizes(format::get_default_format(test_input_fmt.dimension()+1));
    int32_t one_hot_limit = output_dims[one_hot_axis];

    int min_random = 0, max_random = one_hot_limit + 2;
    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
    auto& engine = get_test_engine();
    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = engine.allocate_memory({ type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input, input_rnd_vec);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", "input", shape, one_hot_axis, one_hot_limit));

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    auto indice_shape = input_tensor.sizes(format::get_default_format(test_input_fmt.dimension()));
    VF<T> output_cpu_vec = one_hot_cpu<T>(input_rnd_vec, indice_shape, one_hot_limit, one_hot_axis);
    EXPECT_EQ(output_layout.format.value, format::get_default_format(test_input_fmt.dimension()+1).value);
    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[0];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[2];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];

    EXPECT_EQ(z_size, (int)output_dims[4]);
    EXPECT_EQ(y_size, (int)output_dims[3]);
    EXPECT_EQ(x_size, (int)output_dims[2]);
    EXPECT_EQ(f_size, (int)output_dims[1]);
    EXPECT_EQ(b_size, (int)output_dims[0]);

    bool test_is_correct = true;

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
            break;
        }
    }
    EXPECT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "input_b = " << input_b << std::endl
        << "input_f = " << input_f << std::endl
        << "input_y = " << input_y << std::endl
        << "input_x = " << input_x << std::endl
        << "one_hot_limit = " << one_hot_limit << std::endl
        << "one_hot_axis = " << one_hot_axis << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl;
}

TEST(one_hot_gpu_i32, generic) {
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(5, 2, 1, 1, 2), 0);
    generic_one_hot_test_int<int32_t>(format::bfyx, 1, 2, 3, 1, tensor(1, 5, 1, 3, 2), 1);
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 1, 4), 2);
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 4, 1), 3);
}

TEST(one_hot_gpu_i64, generic) {
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(5, 2, 1, 1, 2), 0);
    generic_one_hot_test_int<int64_t>(format::bfyx, 1, 2, 3, 1, tensor(1, 5, 1, 3, 2), 1);
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 1, 4), 2);
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 4, 1), 3);
}

TEST(one_hot_gpu_i32, bfzyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 2);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 2);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32_to_f32, bfyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, data_types::f32, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(z_size, 2);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    std::vector<float> output_cpu_vec = {1.f, 0.f, 0.f, 0.f, 0.f,
                                         0.f, 1.f, 0.f, 0.f, 0.f};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

TEST(one_hot_gpu_i64_to_f32, bfyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, data_types::f32, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 2);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    std::vector<float> output_cpu_vec = {1.f, 0.f, 0.f, 0.f, 0.f,
                                         0.f, 1.f, 0.f, 0.f, 0.f};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

TEST(one_hot_gpu_i32, bfzyx_ax0) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(3, in_b, in_x, in_y, in_f);
    uint16_t one_hot_axis = 0;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 3);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax0) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(3, in_b, in_x, in_y, in_f);
    uint16_t one_hot_axis = 0;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 3);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32, bfzyx_ax1) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, 3, in_x, in_y, in_f);
    uint16_t one_hot_axis = 1;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 3);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax1) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, 3, in_x, in_y, in_f);
    uint16_t one_hot_axis = 1;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 3);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32, bfzyx_ax2) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, in_y, 3);
    uint16_t one_hot_axis = 2;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax2) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, in_y, 3);
    uint16_t one_hot_axis = 2;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 1);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32, bfzyx_ax3) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, 3, in_y);
    uint16_t one_hot_axis = 3;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    tensor output_tensor = output_layout.get_buffer_size();
    int z_size = output_tensor.spatial[2];
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax3) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, 3, in_y);
    uint16_t one_hot_axis = 3;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output","input", shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    EXPECT_EQ(test_is_correct, true);
}

TEST(one_hot_error, basic_error_wrong_axis) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", "input", tensor(1, 1, 1, 50), 5, 2));

    std::string msg_to_find = "Incorrect parameters configuration: one_hot_axis should be less or equal to 4.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

// One hot is not modified yet.
// openvino/src/plugins/intel_gpu/src/graph/one_hot.cpp
// one_hot_inst::typed_primitive_inst is fully disabled
// TODO : Not used this function because of one_hot_inst::typed_primitive_inst is disabled.
// TEST(one_hot_error, basic_error_bad_shape) {
//     auto& engine = get_test_engine();
//     auto input = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 1, 1 } });

//     topology topology;
//     topology.add(input_layout("input", input->get_layout()));
//     topology.add(one_hot("output", "input", tensor(1, 5, 1, 50), 2, 2));

//     std::string msg_to_find = "Incorrect parameters configuration: shape does not fit input size.";
//     EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
// }
