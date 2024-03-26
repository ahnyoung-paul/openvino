// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "openvino/reference/softmax.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>

#include <softmax_inst.h>

using namespace cldnn;
using namespace std;
using namespace ::tests;

class softmax_gpu_xb_f32_test_fixture: public ::testing::Test {
public:
    static const int32_t
        output_x  = 10, output_b  = 2,  // size of whole output buffer
        input_x   = 10, input_b   = 2,  // size of whole input buffer
        in_size   = input_x*input_b,
        out_size  = output_x*output_b;

    float in_buffer[in_size];
    float out_buffer[out_size];
    float expected_buffer[out_size];

    cldnn::engine& engine;
    cldnn::memory::ptr input;

    //neural::primitive output = memory::allocate({ memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    softmax_gpu_xb_f32_test_fixture()
        : engine(get_test_engine())
        , input(engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, 1, input_x, 1}}))
    {}

    void compare_out_buffer_with_expected() {
        for(size_t i = 0; i < out_size; ++i) {
            // does output have expected values
            ASSERT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    }

    void compare_out_buffer_with_expected_batch_wise() {
        for(size_t b = 0; b < output_b; ++b) {
            float batch_wise_sum = 0;
            for(size_t x = 0; x < output_x; ++x) {
                auto idx = b+x*output_b;
                batch_wise_sum += out_buffer[idx];
                // does output have expected values
                ASSERT_TRUE(are_equal(out_buffer[idx], expected_buffer[idx]))
                    << "At ["<< idx <<  "] Expected : " << expected_buffer[idx] << " actual : " << out_buffer[idx];
            }
            // does it sum to 1 batch wise
            ASSERT_TRUE(are_equal(batch_wise_sum, 1.0f))
                << "Expected : " << 1.0f << " actual : " << batch_wise_sum;
        }
    }

    void test_input_same_values(bool is_caching_test) {
    // in_buffer filled with same value == 1.0f
        for(uint32_t i = 0; i < out_size; ++i) {
                in_buffer[i] = 1.0f;
            expected_buffer[i] = 0.1f;
        }
        std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));

        set_values(input, in_b);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(softmax("softmax", input_info("input"), 3));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "softmax");

        auto output_prim = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            out_buffer[i] = output_ptr[i];
        }
        compare_out_buffer_with_expected();

    }

    void test_input_same_values_batch_wise(bool is_caching_test) {
    // in_buffer filled with same value == 1..2 each batch accordingly (softmax can only xb_f32 )
        for(size_t i = 0; i < output_x; ++i) {
            for(size_t j = 0; j < output_b; ++j)
                in_buffer[j+i*output_b] = (j+i*output_b) % 2 +1.0f;
        }

        std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));
        set_values(input, in_b);
        // fill buffer with the expected 0.1f value
        for(size_t i = 0; i < out_size; ++i)
            expected_buffer[i] = 0.1f;

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(softmax("softmax", input_info("input"), 3));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "softmax");

        auto output_prim = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            out_buffer[i] = output_ptr[i];
        }
        compare_out_buffer_with_expected_batch_wise();
    }

    void test_values_batch_wise(bool is_caching_test) {
        float in_buf[in_size] = {
        //b0  b1
            2.0f, 2.0f, //x0
            2.0f, 2.0f, //x1
            2.0f, 2.0f, //x2
            3.0f, 3.0f, //x3
            5.0f, 5.0f, //x4
            4.0f, 4.0f, //x5
            3.0f, 3.0f, //x6
            2.0f, 2.0f, //x7
            2.0f, 2.0f, //x8
            2.0f, 2.0f  //x9
        };

        float exp_buf[out_size] = {
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f,
            0.069858674f,    0.069858674f,
            0.516189665f,    0.516189665f,
            0.189895565f,    0.189895565f,
            0.069858674f,    0.069858674f,
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f

        };

        std::vector<float> in_b(std::begin(in_buf), std::end(in_buf));
        set_values(input, in_b);
        std::copy(exp_buf, exp_buf+in_size, expected_buffer);

        // out_buffer filled with non-signaling NaN
        for(size_t i = 0; i < out_size; ++i)
            out_buffer[i] = NAN;

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(softmax("softmax", input_info("input"), 3));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "softmax");

        auto output_prim = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            out_buffer[i] = output_ptr[i];
        }
        compare_out_buffer_with_expected_batch_wise();
    }
};

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values) {
    this->test_input_same_values(false);
}

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_batch_wise) {
    this->test_input_same_values_batch_wise(false);
}

TEST_F(softmax_gpu_xb_f32_test_fixture, values_batch_wise) {
    this->test_values_batch_wise(false);
}

TEST(softmax_gpu_bfyx_f32, normalize_y) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 2));

    vector<float> input_vec = {
              //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[12] = {
        0.689974481f,   //b=0, f=0, x=0
        0.832018385f,   //b=0, f=0, x=1

        0.999962831f,   //b=0, f=1, x=0
        0.993307149f,   //b=0, f=1, x=1

        0.999962831f,   //b=0, f=2, x=0
        0.993307149f,   //b=0, f=2, x=1

        0.98201379f,    //b=1, f=0, x=0
        0.99998987f,    //b=1, f=0, x=1

        0.98201379f,    //b=1, f=1, x=0
        0.999547378f,   //b=1, f=1, x=1

        0.999962831f,   //b=1, f=2, x=0
        0.993307149f    //b=1, f=2, x=1
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t l = 0; l < feature_num; l++) {
            for (uint32_t k = 0; k < x_size; k++) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < y_size; j++) {
                    int index = i * feature_num * x_size * y_size +
                                l * x_size * y_size +
                                j * x_size +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_f) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 1));

    vector<float> input_vec = {
        //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //b=0, y=0, x=0
        0.364854551f, //b=0, y=0, x=1

        0.999963085f, //b=0, y=1, x=0
        0.493894592f, //b=0, y=1, x=1

        0.719294981f, //b=1, y=0, x=0
        0.364854551f, //b=1, y=0, x=1

        0.73105857f, //b=1, y=1, x=0
        0.977054322f //b=1, y=1, x=1
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output->count(); i++) {
        std::cerr << "i = " << i << " v = " << output_ptr[i] << std::endl;
    }

    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y_size; j++) {
            for (uint32_t k = 0; k < x_size; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < feature_num; l++) {
                    int index = i * feature_num * x_size * y_size +
                                l * x_size * y_size +
                                j * x_size +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_bfzyx_f32, normalize_z) {
    //  Input  : 2x3x2x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size  *y_size * z_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { batch_num, feature_num, x_size, y_size, z_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 2));

    vector<float> input_vec = {
        //    z0y0x0 z0y0x1 z0y1x0 z0y1x1 z1y0x0 z1y0x1 z1y1x0 z1y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f, 0.2f, -0.2f, 0.9f,  2.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f, 0.3f, 0.1f,  -11.f, 6.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f, 0.1f, 0.3f,  -9.f,  4.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f, 5.f,  0.1f,  6.f,   22.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f, 2.2f,  0.3f,  6.f,  5.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f, 1.2f, 0.3f,  -12.f,  2.2f
    };
    set_values(input, input_vec);

    float expected_max_values[24] = {
        0.524979f, 0.524979f,
        0.5f,      0.731059f,
        0.524979f, 0.524979f,
        0.731059f, 0.731059f,
        0.524979f, 0.524979f,
        0.731059f, 0.731059f,
        0.880797f, 0.598688f,
        0.731059f, 0.999955f,
        0.858149f, 0.549834f,
        0.880797f, 0.952574f,
        0.731059f, 0.524979f,
        0.880797f, 0.952574f,
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) {
        for (uint32_t l = 0; l < feature_num; l++) {
            for (uint32_t j = 0; j < y_size; j++) {
                for (uint32_t k = 0; k < x_size; k++) {
                    float sum = 0.0f;
                    for (uint32_t m = 0; m < z_size; m++) {
                        int index = i * feature_num * x_size * y_size * z_size +
                                    l * x_size * y_size * z_size +
                                    m * x_size * y_size +
                                    j * x_size +
                                    k;
                        if (out_buffer[index] >= temp_max) {
                            temp_max = out_buffer[index];
                        }
                        sum += out_buffer[index];
                    }
                    ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                    temp_max = 0;
                    max_value_buffer_index++;
                    ASSERT_EQ(true, are_equal(sum, expected_sum));
                    sum = 0.0f;
                }
            }
        }
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_b) {
    //  Input  : 3x2x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 2,
            batch_num = 3, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 0));

    vector<float> input_vec = {
        //      y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/3.f,  0.5f,  7.f,   12.f,

        /*b1f0*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,

        /*b2f0*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b2f1*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //f=0, y=0, x=0
        0.364854551f, //f=0, y=0, x=1

        0.999963085f, //f=0, y=1, x=0
        0.493894592f, //f=0, y=1, x=1

        0.719294981f, //f=1, y=0, x=0
        0.364854551f, //f=1, y=0, x=1

        0.73105857f, //f=1, y=1, x=0
        0.977054322f //f=1, y=1, x=1
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < feature_num; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y_size; j++) {
            for (uint32_t k = 0; k < x_size; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < batch_num; l++) {
                    int index = l * feature_num * x_size * y_size +
                                i * x_size * y_size +
                                j * x_size +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO:
//TEST(NegativeSoftmaxTest, DISABLED_TestAll) {
//}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class softmax_test : public tests::generic_test
{

public:
    softmax_test() : tests::generic_test() {}

    void SetUp() override {
        max_ulps_diff_allowed = 6;
    }

    static void TearDownTestCase() {
        all_layer_params.clear();
        all_generic_params.clear();
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params() {
        all_layer_params.emplace_back(new softmax("softmax", input_info("input0"), 1));

        //The test checks only valid combinations.
        //TODO: add more combinations.

        return all_layer_params;
    }

    static std::vector<std::shared_ptr<tests::test_params>> generate_generic_test_params() {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    bool is_format_supported(cldnn::format format) override {
        return
            format == cldnn::format::yxfb ||
            format == cldnn::format::bfyx;
    }

    template<typename Type>
    memory::ptr generate_reference_typed(const std::vector<memory::ptr>& inputs) {
        assert(inputs.size() == 1);
        const memory::ptr input = inputs[0];

        //Output is bfyx
        auto output = engine.allocate_memory(cldnn::layout(input->get_layout().data_type, input->get_layout().format, input->get_layout().get_tensor()));

        cldnn::mem_lock<Type> in0_mem(input, get_test_stream());
        cldnn::mem_lock<Type> out_mem(output, get_test_stream());

        const int in0_b = input->get_layout().get_tensor().sizes()[0];
        const int in0_f = input->get_layout().get_tensor().sizes()[1];
        const int in0_h = input->get_layout().get_tensor().sizes()[3];
        const int in0_w = input->get_layout().get_tensor().sizes()[2];

//        const int out_b = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[0];
//        const int out_f = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[1];
//        const int out_h = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[2];
//        const int out_w = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[3];

//        assert(in0_b == out_b);
//        assert(in0_f == out_f);
//        assert(in0_h == out_h);
//        assert(in0_w == out_w);

        std::vector<float> cached_exp_vals;
        cached_exp_vals.resize(in0_f);

        const auto input_desc = get_linear_memory_desc(input->get_layout());

        for (int n = 0; n < in0_b; ++n)
        for (int y = 0; y < in0_h; ++y)
        for (int x = 0; x < in0_w; ++x) {
            float max_val = -std::numeric_limits<float>::infinity();

            for (int c = 0; c < in0_f; ++c) {
                const size_t in0_idx = get_linear_index(input->get_layout(), n, c, y, x, input_desc);

                max_val = std::max(max_val, static_cast<float>(in0_mem[in0_idx]));
            }

            float Z = 0;

            for (int c = 0; c < in0_f; ++c) {
                const size_t in0_idx = get_linear_index(input->get_layout(), n, c, y, x, input_desc);

                float tmp = static_cast<float>((Type)std::exp(static_cast<float>(in0_mem[in0_idx]) - max_val));
                Z += tmp;
                cached_exp_vals[c] = tmp;
            }

            for (int c = 0; c < in0_f; ++c) {
                const size_t out_idx = get_linear_index(output->get_layout(), n, c, y, x, input_desc);
                out_mem[out_idx] = (Type)(cached_exp_vals[c] / Z);
            }
        }

        return output;
    }

    virtual memory::ptr generate_reference(const std::vector<memory::ptr>& inputs) override {
        if (generic_params->data_type == data_types::f32) {
            return generate_reference_typed<float>(inputs);
        } else {
            return generate_reference_typed<ov::float16>(inputs);
        }
    }

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>>& info) {
        std::stringstream res;

        const auto& p = std::get<0>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i) {
            const auto chans = format::traits(p->fmt).order;

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].get_tensor().sizes(p->fmt).size(); ++j) {
                res << chans[j] << p->input_layouts[i].get_tensor().sizes(p->fmt)[j];
            }
        }

        return res.str();
    }

private:
    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;
};

std::vector<std::shared_ptr<cldnn::primitive>> softmax_test::all_layer_params = {};
std::vector<std::shared_ptr<tests::test_params>> softmax_test::all_generic_params = {};

TEST_P(softmax_test, SOFTMAX) {
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_SOFTMAX,
    softmax_test,
    ::testing::Combine(::testing::ValuesIn(softmax_test::generate_generic_test_params()), ::testing::ValuesIn(softmax_test::generate_specific_test_params())),
    softmax_test::custom_param_name);



namespace {
template<typename T>
struct SoftmaxParams {
    int64_t axis;
    tensor input_tensor;
    std::vector<T> input;
    std::vector<T> expected;
};

template<typename T>
using SoftmaxParamsWithFormat = std::tuple<
    SoftmaxParams<T>,
    format::type,     // source (plain) layout
    format::type      // target (blocked) layout
>;

const std::vector<format::type> formats2D{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
};

const std::vector<format::type> formats3D{
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::bs_fs_zyx_bsv16_fsv16
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<SoftmaxParams<T>> generateSoftmaxParams2D() {
    const std::vector<SoftmaxParams<T>> result = {
        {
            0,
            tensor(3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 3.f, 0.5f, 7.f, 12.f, 0.2f, 0.2f, -10.f, 5.2f,
                4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.311493f, 0.270291f, 0.999963f, 0.0122108f,
                0.264614f, 0.364855f, 0.268941f, 0.977054f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.719295f, 0.364855f, 0.731059f, 0.0218575f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.0160912f, 0.270291f, 1.1134e-08f, 0.00108822f})
        },
        {
            1,
            tensor(2, 3, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f,
                3.f, 0.5f, 7.f, 12.f, 4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.311493f, 0.270291f, 0.999963f, 0.0122108f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.264614f, 0.364855f, 0.268941f, 0.977054f,
                0.719295f, 0.364855f, 0.731059f, 0.0218575f,
                0.0160912f, 0.270291f, 1.1134e-08f, 0.00108822f})
        },
        {
            2,
            tensor(2, 3, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f,
                3.f, 0.5f, 7.f, 12.f, 4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.310026f, 0.167982f, 0.689974f, 0.832018f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f,
                0.0179862f, 1.013e-05f, 0.982014f, 0.99999f,
                0.0179862f, 0.000452622f, 0.982014f, 0.999547f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f})
        },
        {
            3,
            tensor(2, 3, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f,
                3.f, 0.5f, 7.f, 12.f, 4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.549834f, 0.450166f, 0.354344f, 0.645656f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f,
                0.924142f, 0.0758582f, 0.00669285f, 0.993307f,
                0.970688f, 0.0293122f, 0.450166f, 0.549834f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f})
        },

    };
    return result;
}

template<typename T>
std::vector<SoftmaxParams<T>> generateSoftmaxParams3D() {
    const std::vector<SoftmaxParams<T>> result = {
        {
            0,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.0521536f, 0.354344f, 0.00223785f, 2.75357e-05f, 0.00816257f, 0.425557f, 0.0060598f, 3.39827e-09f,
                0.0218813f, 0.425557f, 1.523e-08f, 0.0474259f, 0.130108f, 0.450166f, 4.13994e-08f, 0.731059f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.24974f, 0.5f, 0.952574f, 0.880797f,
                0.947846f, 0.645656f, 0.997762f, 0.999972f, 0.991837f, 0.574443f, 0.99394f, 1.0f,
                0.978119f, 0.574443f, 1.0f, 0.952574f, 0.869892f, 0.549834f, 1.0f, 0.268941f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.75026f, 0.5f, 0.0474259f, 0.119203f})
        },
        {
            1,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.311493f, 0.270291f, 0.999963f, 0.0122108f, 0.332225f, 0.250089f, 0.999943f, 0.0213123f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f, 0.367165f, 0.337585f, 6.79002e-06f, 0.862025f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f, 0.30061f, 0.412327f, 5.01718e-05f, 0.116662f,
                0.264614f, 0.364855f, 0.268941f, 0.977054f, 0.923207f, 0.290461f, 0.5f, 1.0f,
                0.719295f, 0.364855f, 0.731059f, 0.0218575f, 0.0561403f, 0.35477f, 0.5f, 5.05653e-08f,
                0.0160912f, 0.270291f, 1.1134e-08f, 0.00108822f, 0.0206528f, 0.35477f, 7.615e-09f, 2.5175e-09f})
        },
        {
            2,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.475021f, 0.524979f, 0.5f, 0.268941f, 0.524979f, 0.475021f, 0.5f, 0.731059f,
                0.475021f, 0.524979f, 0.731059f, 0.268941f, 0.524979f, 0.475021f, 0.268941f, 0.731059f,
                0.524979f, 0.475021f, 0.268941f, 0.731059f, 0.475021f, 0.524979f, 0.731059f, 0.268941f,
                0.119203f, 0.598688f, 0.731059f, 4.53979e-05f, 0.880797f, 0.401312f, 0.268941f, 0.999955f,
                0.858149f, 0.549834f, 0.880797f, 0.952574f, 0.141851f, 0.450166f, 0.119203f, 0.0474259f,
                0.268941f, 0.475021f, 0.880797f, 0.952574f, 0.731059f, 0.524979f, 0.119203f, 0.0474259f})
        },
        {
            3,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.310026f, 0.167982f, 0.689974f, 0.832018f, 0.331812f, 0.0629734f, 0.668188f, 0.937027f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f, 0.999988f, 0.00223785f, 1.23728e-05f, 0.997762f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f, 0.999888f, 0.0198403f, 0.000111653f, 0.98016f,
                0.0179862f, 1.013e-05f, 0.982014f, 0.99999f, 0.268941f, 3.08284e-10f, 0.731059f, 1.0f,
                0.0179862f, 0.000452622f, 0.982014f, 0.999547f, 0.0218813f, 0.00739154f, 0.978119f, 0.992609f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f, 0.999998f, 0.130108f, 1.8506e-06f, 0.869892f})
        },
        {
            4,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.549834f, 0.450166f, 0.354344f, 0.645656f, 0.598688f, 0.401312f, 0.167982f, 0.832018f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f, 0.549834f, 0.450166f, 3.38949e-08f, 1.0f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f, 0.450166f, 0.549834f, 1.8506e-06f, 0.999998f,
                0.924142f, 0.0758582f, 0.00669285f, 0.993307f, 0.992609f, 0.00739154f, 1.12535e-07f, 1.0f,
                0.970688f, 0.0293122f, 0.450166f, 0.549834f, 0.869892f, 0.130108f, 0.689974f, 0.310025f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f, 0.710949f, 0.28905f, 6.80798e-07f, 0.999999f})
        }
    };
    return result;
}

template<typename T>
float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<ov::float16>() {
    return 0.2;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<SoftmaxParamsWithFormat<T> > &param) {
        std::stringstream buf;
        SoftmaxParams<T> p;
        format::type plain_format;
        format::type target_format;
        std::tie(p, plain_format, target_format) = param.param;
        buf << "_inputTensor=" << p.input_tensor.to_string()
            << "_axis=" << p.axis
            << "_plainFormat=" << fmt_to_str(plain_format)
            << "_targetFormat=" << fmt_to_str(target_format);
        return buf.str();
    }
};
}; // namespace



template<typename T>
struct softmax_gpu_formats_test
        : public ::testing::TestWithParam<SoftmaxParamsWithFormat<T> > {
public:
    void test(bool is_caching_test) {
        const auto data_type = ov::element::from<T>();
        SoftmaxParams<T> params;
        format::type plain_format;
        format::type target_format;

        std::tie(params, plain_format, target_format) = this->GetParam();

        auto& engine = get_test_engine();
        const auto input = engine.allocate_memory({data_type, plain_format, params.input_tensor});

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("reordered_input", input_info("input"), target_format, data_type));
        topology.add(softmax("blocked_softmax", input_info("reordered_input"), params.axis));
        topology.add(reorder("softmax", input_info("blocked_softmax"), plain_format, data_type));

        set_values(input, params.input);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(false));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);
        const auto outputs = network->execute();
        const auto output = outputs.at("softmax").get_memory();
        const cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.input_tensor.count(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_NEAR(output_ptr[i], params.expected[i], getError<T>()) << "target_format=" << target_format << ", i=" << i;
        }
    }
};

using softmax_gpu_formats_test_f32 = softmax_gpu_formats_test<float>;
using softmax_gpu_formats_test_f16 = softmax_gpu_formats_test<ov::float16>;

TEST_P(softmax_gpu_formats_test_f32, softmax_gpu_formats_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(softmax_gpu_formats_test_f16, softmax_gpu_formats_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f32_2d,
                         softmax_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams2D<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                                 ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f16_2d,
                         softmax_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams2D<ov::float16>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                                 ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f32_3d,
                         softmax_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                                 ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f16_3d,
                         softmax_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams3D<ov::float16>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                                 ),
                         PrintToStringParamName());

TEST(softmax_gpu_bfyx_f32, normalize_f_dynamic) {
    auto& engine = get_test_engine();

    const int64_t x = 2, y = 2, f = 3, b = 2;
    const int64_t buf_size = b*f*y*x;
    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax("softmax", input_info("input"), 1));

    vector<float> input_vec = {
        //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //b=0, y=0, x=0
        0.364854551f, //b=0, y=0, x=1

        0.999963085f, //b=0, y=1, x=0
        0.493894592f, //b=0, y=1, x=1

        0.719294981f, //b=1, y=0, x=0
        0.364854551f, //b=1, y=0, x=1

        0.73105857f, //b=1, y=1, x=0
        0.977054322f //b=1, y=1, x=1
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("softmax");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < b; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y; j++) {
            for (uint32_t k = 0; k < x; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < f; l++) {
                    int index = i * f * x * y +
                                l * x * y +
                                j * x +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_TRUE(are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_TRUE(are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_cached) {
    this->test_input_same_values(true);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_batch_wise_cached) {
    this->test_input_same_values_batch_wise(true);
}

TEST_F(softmax_gpu_xb_f32_test_fixture, values_batch_wise_cached) {
    this->test_values_batch_wise(true);
}

TEST_P(softmax_test, SOFTMAX_cached) {
    run_single_test(true);
}

TEST_P(softmax_gpu_formats_test_f32, softmax_gpu_formats_test_f32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(softmax_gpu_formats_test_f16, softmax_gpu_formats_test_f16_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
#endif

TEST(softmax_gpu_bfyx_f32, bf_opt_normalize_f_dynamic) {
    auto& engine = get_test_engine();

    const int64_t x = 1, y = 1, f = 3, b = 2;
    const int64_t buf_size = b*f*y*x;
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), y, x},
                                       data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax("softmax", input_info("input"), 1));

    vector<float> input_vec = {
              //y0x0
        /*b0f0*/0.1f,
        /*b0f1*/0.2f,
        /*b0f2*/0.2f,
        /*b1f0*/3.f,
        /*b1f1*/4.f,
        /*b1f2*/0.2f,
    };
    set_values(input, input_vec);

    float expected_max_values[2] = {
        0.344253346f, //b=0, y=0, x=0
        0.719294981f  //b=1, y=0, x=0
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("softmax");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < b; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y; j++) {
            for (uint32_t k = 0; k < x; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < f; l++) {
                    int index = i * f * x * y +
                                l * x * y +
                                j * x +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_TRUE(are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_TRUE(are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

std::vector<ov::float16> output_ref = {
0.009529f,0.005257f,0.000920f,0.000404f,0.000267f,0.000328f,0.000324f,0.001319f,0.001319f,0.000371f,0.000528f,0.000165f,0.000348f,0.000576f,0.000782f,0.000423f,0.000559f,0.001471f,0.000525f,0.000572f,0.000478f,0.000735f,0.000238f,0.000555f,0.001288f,0.001029f,0.000096f,0.000124f,0.000345f,0.001156f,
0.000928f,0.001330f,0.000581f,0.001590f,0.000453f,0.001629f,0.000299f,0.001556f,0.000091f,0.000165f,0.001352f,0.000131f,0.000251f,0.000141f,0.001156f,0.000770f,0.001042f,0.000453f,0.001127f,0.002298f,0.000751f,0.000371f,0.000648f,0.003674f,0.003189f,0.000770f,0.002058f,0.001137f,0.001679f,0.000292f,
0.000800f,0.001877f,0.001119f,0.001519f,0.000139f,0.000562f,0.001068f,0.000294f,0.001665f,0.000396f,0.001832f,0.000122f,0.001544f,0.000156f,0.000300f,0.000513f,0.000658f,0.000872f,0.001428f,0.000402f,0.000879f,0.001933f,0.002840f,0.001416f,0.000689f,0.001965f,0.000300f,0.001845f,0.000913f,0.000144f,
0.000253f,0.000304f,0.000906f,0.001777f,0.001679f,0.000373f,0.000844f,0.000638f,0.000826f,0.001451f,0.000388f,0.001184f,0.000739f,0.001531f,0.001370f,0.001556f,0.000453f,0.001556f,0.000150f,0.002012f,0.000608f,0.001802f,0.000559f,0.000145f,0.000361f,0.001877f,0.000648f,0.000478f,0.000996f,0.000324f,
0.000844f,0.000819f,0.001933f,0.000186f,0.000723f,0.000826f,0.000262f,0.000739f,0.002407f,0.000307f,0.001352f,0.001906f,0.000150f,0.001544f,0.000294f,0.000276f,0.002075f,0.003265f,0.000156f,0.002682f,0.000430f,0.001458f,0.001982f,0.003395f,0.000211f,0.000240f,0.000776f,0.003046f,0.001604f,0.004604f,
0.003698f,0.005054f,0.000408f,0.001085f,0.003876f,0.007191f,0.006546f,0.000404f,0.000840f,0.004681f,0.005180f,0.001845f,0.001483f,0.000685f,0.002243f,0.000546f,0.000324f,0.001370f,0.003061f,0.004982f,0.001404f,0.002645f,0.000685f,0.001906f,0.000987f,0.004940f,0.000241f,0.003729f,0.004604f,0.001629f,
0.005733f,0.001507f,0.002337f,0.001250f,0.004711f,0.007351f,0.001616f,0.007587f,0.002748f,0.006142f,0.004391f,0.003582f,0.001862f,0.002058f,0.000474f,0.001288f,0.005054f,0.001917f,0.004463f,0.000496f,0.000680f,0.004787f,0.000770f,0.000360f,0.000572f,0.002949f,0.005379f,0.001982f,0.001802f,0.003265f,
0.002726f,0.008141f,0.001341f,0.002604f,0.002668f,0.000375f,0.000763f,0.005684f,0.000482f,0.000581f,0.001544f,0.000987f,0.001763f,0.004032f,0.005951f,0.001816f,0.008537f,0.001471f,0.003502f,0.004982f,0.000844f,0.011040f,0.000613f,0.000680f,0.005733f,0.007412f,0.003237f,0.007828f,0.009155f,0.001270f,
0.000634f,0.003876f,0.004391f,0.001604f,0.015327f,0.012039f,0.001103f,0.011223f,0.000586f,0.002426f,0.011292f,0.004681f,0.018341f,0.011391f,0.017639f,0.014740f,0.001862f,0.001051f,0.003214f,0.009224f,0.016968f,0.031708f,0.020950f,0.002337f,0.001352f,0.003788f,0.023361f,0.021774f,0.007023f,0.029083f,
0.001451f,0.004124f,0.014626f,0.029999f,0.004288f,0.023010f,0.016968f,0.006798f,0.025253f,0.002337f,0.005913f,0.001917f,0.031952f,0.000906f,0.027954f,0.002840f,0.003910f,0.001777f,0.051025f,0.000057f,0.000067f,0.000030f,0.000009f,0.000027f,0.000009f,0.000020f,0.000017f,0.000014f,0.000007f,0.000011f,
0.000022f,0.000026f,0.000042f,0.000060f,0.000017f,0.000023f,0.000020f,0.000016f,0.000044f,0.000025f,0.000022f,0.000039f,0.000041f,0.000056f,0.000051f,0.000044f,0.000055f,0.000076f,0.000091f,0.000073f,0.000065f,0.000178f,0.000072f,0.000111f,0.000060f,0.000386f,0.000099f,0.000105f,0.000257f,0.000088f,
0.000167f,0.000128f,0.000151f,0.000045f,0.000072f,0.000051f,0.000142f,0.000033f,0.000027f,0.000035f,0.000025f,0.000019f,0.000037f,0.000014f,0.000015f,0.000021f,0.000107f,0.000020f,0.000069f,0.000068f,0.000018f,0.000025f,0.000024f,0.000070f,0.000045f,0.000048f,0.000078f,0.000040f,0.000034f,0.000081f,
0.000050f,0.000045f,0.000103f,0.000098f,0.000344f,0.000061f,0.000088f,0.000043f,0.000056f,0.000058f,0.000189f,0.000051f,0.000051f,0.000173f,0.000029f,0.000050f,0.000038f,0.000053f,0.000055f,0.000057f,0.000051f,0.000032f,0.000031f,0.000052f,0.000094f,0.000025f,0.000146f,0.000025f,0.000030f,0.000114f,
0.000058f,0.000031f,0.000039f,0.000022f,0.000028f,0.000041f,0.000021f,0.000021f,0.000025f,0.000073f,0.000004f,0.000028f,0.000044f,0.000007f,0.000035f,0.000004f,0.000017f,0.000005f,0.000008f,0.000007f,0.000013f,0.000004f,0.000027f,0.000007f,0.000020f,0.000036f,0.000026f,0.000008f,0.000026f,0.000006f,
0.000027f,0.000041f,0.000010f,0.000160f,0.000095f,0.000016f,0.000017f,0.000151f,0.000043f,0.000162f,0.000172f,0.000238f,0.000058f,0.000258f,0.000487f,0.000166f,0.000915f,0.000609f,0.000276f,0.000487f,0.000156f,0.000553f,0.000526f,0.000136f,0.000115f,0.000106f,0.000383f,0.000478f,0.000104f,0.000091f,
0.000304f,0.000092f,0.000202f,0.000072f,0.000299f,0.000529f,0.000500f,0.000088f,0.000099f,0.000279f,0.000056f,0.000299f,0.000418f,0.000289f,0.000138f,0.000265f,0.000156f,0.000122f,0.000249f,0.000129f,0.000371f,0.000388f,0.000151f,0.000247f,0.000105f,0.000109f,0.000072f,0.000164f,0.000069f,0.000051f,
0.000147f,0.000228f,0.000182f,0.000267f,0.000416f,0.000086f,0.000178f,0.000075f,0.000526f,0.000463f,0.000214f,0.000640f,0.000432f,0.000929f,0.000286f,0.000375f,0.000458f,0.000455f,0.001087f,0.001061f,0.000359f,0.000895f,0.000267f,0.001472f,0.001920f,0.004234f,0.000365f,0.003222f,0.002352f,0.000603f,
0.003851f,0.004097f,0.002077f,0.000438f,0.002283f,0.000692f,0.001606f,0.000461f,0.000794f,0.001968f,0.000371f,0.001820f,0.001430f,0.000329f,0.000487f,0.000480f,0.000557f,0.000505f,0.001894f,0.007660f,0.001121f,0.001594f,0.003767f,0.002352f,0.000895f,0.001212f,0.001546f,0.002607f,0.001910f,0.009682f,
0.007660f,0.005020f,0.005146f,0.002230f,0.001546f,0.009605f,0.018814f,0.012535f,0.006104f,0.005386f,0.008820f,0.004875f,0.008965f,0.014526f,0.012619f,0.006351f,0.004913f,0.010559f,0.008095f,0.022354f,0.024338f,0.010979f,0.015106f,0.030304f,0.015961f,0.014412f,0.021317f,0.026321f,0.053162f,0.032227f,
0.055695f,0.053558f,0.108215f,0.041412f,0.092590f,0.062683f,0.049530f,0.011406f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000001f,0.000001f,0.000019f,0.000005f,0.000000f,0.000000f,0.000000f,0.000000f,
0.000000f,0.000001f,0.000001f,0.000001f,0.000000f,0.000000f,0.000000f,0.000002f,0.000000f,0.000000f,0.000000f,0.000001f,0.000000f,0.000002f,0.000008f,0.000002f,0.000000f,0.000001f,0.000008f,0.000040f,0.000054f,0.000017f,0.000007f,0.000001f,0.000001f,0.000004f,0.000007f,0.000012f,0.000003f,0.000000f,
0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000001f,0.000000f,0.000000f,0.000000f,0.000001f,0.000003f,0.000006f,0.000025f,0.000001f,0.000006f,0.000000f,0.000010f,0.000005f,0.000009f,0.000015f,0.000002f,0.000000f,0.000000f,0.000000f,0.000000f,0.000002f,
0.000001f,0.000000f,0.000003f,0.000001f,0.000000f,0.000001f,0.000000f,0.000000f,0.000000f,0.000002f,0.000017f,0.000036f,0.000094f,0.000005f,0.000003f,0.000016f,0.000014f,0.000038f,0.000015f,0.000008f,0.000002f,0.000008f,0.000003f,0.000010f,0.000001f,0.000002f,0.000000f,0.000001f,0.000001f,0.000000f,
0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000001f,0.000001f,0.000001f,0.000000f,0.000000f,0.000000f,0.000001f,0.000001f,0.000001f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000002f,
0.000010f,0.000003f,0.000001f,0.000000f,0.000000f,0.000001f,0.000009f,0.000022f,0.000119f,0.000054f,0.000009f,0.000008f,0.000005f,0.000001f,0.000000f,0.000000f,0.000000f,0.000000f,0.000001f,0.000003f,0.000000f,0.000000f,0.000000f,0.000000f,0.000001f,0.000007f,0.000007f,0.000002f,0.000001f,0.000000f,
0.000002f,0.000033f,0.000049f,0.000032f,0.000006f,0.000005f,0.000004f,0.000005f,0.000003f,0.000009f,0.000010f,0.000001f,0.000004f,0.000003f,0.000002f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000002f,0.000000f,0.000001f,0.000000f,0.000001f,0.000006f,0.000005f,0.000012f,0.000031f,
0.000005f,0.000003f,0.000009f,0.000112f,0.000131f,0.000023f,0.000042f,0.000030f,0.000046f,0.000097f,0.000004f,0.000001f,0.000003f,0.000000f,0.000007f,0.000017f,0.000046f,0.000025f,0.000002f,0.000001f,0.000003f,0.000034f,0.000139f,0.000740f,0.000504f,0.000266f,0.000102f,0.000393f,0.000131f,0.000157f,
0.000022f,0.000010f,0.000003f,0.000016f,0.000014f,0.000011f,0.000002f,0.000002f,0.000001f,0.000001f,0.000044f,0.000039f,0.000026f,0.000004f,0.000000f,0.000002f,0.000002f,0.000005f,0.000083f,0.000240f,0.000081f,0.000108f,0.000157f,0.000168f,0.000164f,0.000146f,0.000190f,0.000029f,0.000162f,0.000122f,
0.000121f,0.000080f,0.000010f,0.000007f,0.000013f,0.000031f,0.000099f,0.000096f,0.000009f,0.000022f,0.000045f,0.000140f,0.001059f,0.002859f,0.004257f,0.002010f,0.002642f,0.003473f,0.008606f,0.023758f,0.058746f,0.023026f,0.208252f,0.289062f,0.263184f,0.082886f,0.020294f,0.000000f,0.000000f,0.000000f,
0.000000f,0.000001f,0.000000f,0.000000f,0.000000f,0.000000f,0.000003f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000034f,0.000024f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000003f,0.000041f,0.000010f,0.000000f,0.000000f,
0.000000f,0.000001f,0.000028f,0.000442f,0.000008f,0.000000f,0.000000f,0.000000f,0.000024f,0.000285f,0.000087f,0.000002f,0.000000f,0.000000f,0.000001f,0.000001f,0.000003f,0.000002f,0.000000f,0.000000f,0.000002f,0.000003f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000009f,0.000067f,0.000016f,
0.000000f,0.000000f,0.000000f,0.000002f,0.000023f,0.000010f,0.000000f,0.000000f,0.000000f,0.000000f,0.000006f,0.000008f,0.000006f,0.000000f,0.000000f,0.000003f,0.000011f,0.000001f,0.000000f,0.000000f,0.000000f,0.000001f,0.000021f,0.000181f,0.000013f,0.000000f,0.000000f,0.000000f,0.000000f,0.000005f,
0.000044f,0.000024f,0.000002f,0.000000f,0.000010f,0.000005f,0.000007f,0.000001f,0.000000f,0.000000f,0.000002f,0.000025f,0.000079f,0.000003f,0.000000f,0.000000f,0.000003f,0.000222f,0.000055f,0.000011f,0.000000f,0.000000f,0.000000f,0.000000f,0.000004f,0.000010f,0.000002f,0.000000f,0.000000f,0.000000f,
0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000003f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000005f,0.000003f,0.000000f,0.000000f,0.000000f,0.000000f,0.000037f,0.000031f,0.000003f,0.000000f,0.000000f,0.000000f,0.000001f,0.000002f,0.000005f,0.000001f,0.000001f,
0.000021f,0.000043f,0.000035f,0.000000f,0.000000f,0.000000f,0.000000f,0.000001f,0.000093f,0.000027f,0.000001f,0.000000f,0.000000f,0.000000f,0.000015f,0.000066f,0.000009f,0.000001f,0.000000f,0.000001f,0.000014f,0.000061f,0.000010f,0.000000f,0.000000f,0.000002f,0.000018f,0.000030f,0.000008f,0.000000f,
0.000000f,0.000000f,0.000008f,0.000819f,0.000120f,0.000002f,0.000000f,0.000000f,0.000000f,0.000000f,0.000006f,0.000001f,0.000000f,0.000000f,0.000000f,0.000001f,0.000003f,0.000001f,0.000000f,0.000000f,0.000000f,0.000002f,0.000014f,0.000009f,0.000000f,0.000000f,0.000000f,0.000023f,0.000723f,0.000249f,
0.000003f,0.000000f,0.000000f,0.000000f,0.000002f,0.000456f,0.000214f,0.000001f,0.000000f,0.000000f,0.000002f,0.000004f,0.000010f,0.000001f,0.000001f,0.000002f,0.000307f,0.000833f,0.000100f,0.000000f,0.000000f,0.000000f,0.000000f,0.000016f,0.000053f,0.000004f,0.000000f,0.000000f,0.000000f,0.000044f,
0.000645f,0.000304f,0.000005f,0.000000f,0.000000f,0.000000f,0.000000f,0.000003f,0.000001f,0.000000f,0.000002f,0.000046f,0.000267f,0.000084f,0.000001f,0.000000f,0.000000f,0.000001f,0.000012f,0.000851f,0.000099f,0.000000f,0.000000f,0.000000f,0.000001f,0.000037f,0.000073f,0.000001f,0.000001f,0.000001f,
0.000012f,0.000274f,0.000317f,0.000058f,0.000014f,0.000014f,0.000112f,0.000898f,0.000561f,0.000097f,0.000000f,0.000021f,0.000793f,0.125000f,0.790039f,0.072449f,0.001782f,0.002493f,0.000298f,0.000720f,0.000264f,0.000753f,0.000612f,0.000138f,0.001378f,0.000041f,0.000236f,0.001285f,0.000181f,0.004406f,
0.000325f,0.000750f,0.000547f,0.000101f,0.000889f,0.000760f,0.000055f,0.000263f,0.001334f,0.000176f,0.001386f,0.000151f,0.000125f,0.000125f,0.000106f,0.001274f,0.000447f,0.000205f,0.000241f,0.001768f,0.000126f,0.001467f,0.002396f,0.002268f,0.000172f,0.000145f,0.001621f,0.000135f,0.000332f,0.000164f,
0.001178f,0.000410f,0.001897f,0.000160f,0.001587f,0.001726f,0.000346f,0.000185f,0.000896f,0.000282f,0.002132f,0.000179f,0.003199f,0.001303f,0.001523f,0.000071f,0.000063f,0.001443f,0.000575f,0.000250f,0.000283f,0.001124f,0.000197f,0.000269f,0.002380f,0.000350f,0.003775f,0.000182f,0.001621f,0.000122f,
0.000303f,0.000660f,0.001049f,0.000100f,0.000515f,0.002396f,0.000350f,0.001712f,0.000261f,0.000288f,0.006126f,0.002325f,0.000293f,0.003305f,0.000437f,0.000166f,0.000374f,0.000278f,0.001944f,0.000350f,0.004272f,0.000350f,0.002018f,0.000332f,0.001836f,0.002287f,0.000215f,0.002474f,0.002150f,0.004379f,
0.000129f,0.004238f,0.000194f,0.003744f,0.000255f,0.000714f,0.000303f,0.002434f,0.000335f,0.000169f,0.001023f,0.000245f,0.001386f,0.000083f,0.000587f,0.003105f,0.000499f,0.000798f,0.000348f,0.000124f,0.001295f,0.000332f,0.000208f,0.000380f,0.002720f,0.000209f,0.001431f,0.000287f,0.000126f,0.000904f,
0.000146f,0.000216f,0.000978f,0.000423f,0.000119f,0.002287f,0.000450f,0.000923f,0.010017f,0.000430f,0.000082f,0.000499f,0.001851f,0.000323f,0.000208f,0.003857f,0.000570f,0.004879f,0.000181f,0.000166f,0.017975f,0.000392f,0.003227f,0.000157f,0.000139f,0.003490f,0.000656f,0.001944f,0.001912f,0.000523f,
0.002611f,0.000267f,0.000366f,0.002756f,0.002895f,0.004440f,0.000791f,0.001609f,0.000766f,0.001662f,0.000535f,0.003653f,0.000434f,0.004440f,0.000552f,0.000343f,0.003683f,0.000566f,0.000766f,0.001955f,0.000660f,0.005154f,0.000556f,0.008430f,0.003252f,0.004082f,0.000714f,0.000341f,0.000978f,0.004585f,
0.000720f,0.002871f,0.004082f,0.001303f,0.000539f,0.000430f,0.000495f,0.005482f,0.000303f,0.000502f,0.000335f,0.003376f,0.005836f,0.000241f,0.003405f,0.001478f,0.007206f,0.007324f,0.000309f,0.003405f,0.021515f,0.000583f,0.001141f,0.004879f,0.000469f,0.001224f,0.003199f,0.000374f,0.000326f,0.001141f,
0.000962f,0.005665f,0.009773f,0.001264f,0.005836f,0.014122f,0.001106f,0.010658f,0.000607f,0.001295f,0.005527f,0.008430f,0.000356f,0.008560f,0.000978f,0.000630f,0.001254f,0.005440f,0.002590f,0.001345f,0.024368f,0.013672f,0.001955f,0.018417f,0.001058f,0.004311f,0.000804f,0.000636f,0.015137f,0.001478f,
0.016235f,0.001636f,0.000911f,0.001106f,0.006886f,0.043488f,0.001089f,0.015869f,0.001187f,0.001058f,0.001233f,0.007500f,0.020218f,0.002087f,0.010170f,0.022385f,0.000625f,0.009186f,0.014008f,0.026169f,0.003132f,0.014221f,0.002895f,0.004475f,0.019897f,0.005394f,0.006264f,0.003803f,0.044861f,0.003574f,
0.102661f,0.002848f,0.004082f,0.005196f,0.061798f,0.001128f,0.000308f,0.000140f,0.000052f,0.000314f,0.000043f,0.000030f,0.000528f,0.000183f,0.000358f,0.000156f,0.000375f,0.000186f,0.000084f,0.000156f,0.000069f,0.000355f,0.001099f,0.000078f,0.000060f,0.000527f,0.000171f,0.000442f,0.000235f,0.000278f,
0.000737f,0.000205f,0.000459f,0.000819f,0.001420f,0.001099f,0.001004f,0.000845f,0.000312f,0.000804f,0.000292f,0.001467f,0.000583f,0.000294f,0.001156f,0.000270f,0.000699f,0.001078f,0.001309f,0.000300f,0.000931f,0.000135f,0.001288f,0.000125f,0.000247f,0.000658f,0.000384f,0.000164f,0.000957f,0.000308f,
0.000573f,0.000349f,0.000678f,0.000249f,0.000630f,0.001092f,0.000348f,0.000972f,0.001146f,0.000520f,0.000119f,0.001371f,0.000826f,0.000689f,0.000743f,0.000758f,0.000336f,0.000412f,0.000809f,0.001188f,0.001467f,0.000264f,0.001043f,0.000371f,0.000804f,0.000437f,0.000712f,0.001007f,0.000928f,0.000911f,
0.000442f,0.001207f,0.000781f,0.001156f,0.000558f,0.000858f,0.000733f,0.000222f,0.000845f,0.000688f,0.001867f,0.000215f,0.002069f,0.000228f,0.000509f,0.001428f,0.001240f,0.000198f,0.000548f,0.001156f,0.000548f,0.001182f,0.000570f,0.000293f,0.000706f,0.001004f,0.000257f,0.000733f,0.000798f,0.000112f,
0.001120f,0.000081f,0.000887f,0.000330f,0.000700f,0.000398f,0.000781f,0.000815f,0.000831f,0.000089f,0.000764f,0.000843f,0.000898f,0.000511f,0.000804f,0.000167f,0.001240f,0.000743f,0.000252f,0.001635f,0.000986f,0.000384f,0.001676f,0.001338f,0.000423f,0.001210f,0.002522f,0.001264f,0.002245f,0.001023f,
0.001512f,0.000442f,0.001110f,0.001360f,0.000406f,0.004978f,0.000931f,0.002132f,0.003153f,0.000437f,0.002428f,0.000303f,0.001349f,0.001648f,0.000576f,0.002522f,0.002871f,0.000267f,0.001072f,0.000304f,0.002201f,0.002394f,0.002388f,0.002779f,0.000954f,0.003729f,0.000561f,0.001812f,0.003557f,0.004017f,
0.001125f,0.000986f,0.000988f,0.002998f,0.002563f,0.000936f,0.003670f,0.003599f,0.000425f,0.002676f,0.000776f,0.002613f,0.001052f,0.001802f,0.000651f,0.001736f,0.000914f,0.001188f,0.000892f,0.001622f,0.002394f,0.000845f,0.002802f,0.002817f,0.002684f,0.002172f,0.001617f,0.003714f,0.001455f,0.002848f,
0.004955f,0.001817f,0.004147f,0.000428f,0.002409f,0.005554f,0.002563f,0.003572f,0.000384f,0.002771f,0.001595f,0.005135f,0.001063f,0.004189f,0.006069f,0.001013f,0.005508f,0.007469f,0.008263f,0.005016f,0.005177f,0.002052f,0.006512f,0.000556f,0.002447f,0.009148f,0.003225f,0.004337f,0.004978f,0.000695f,
0.001622f,0.004864f,0.002167f,0.007046f,0.008018f,0.005821f,0.000607f,0.005749f,0.008110f,0.008110f,0.003654f,0.005318f,0.003462f,0.003254f,0.001635f,0.004921f,0.005772f,0.001713f,0.018341f,0.004189f,0.011116f,0.016052f,0.012802f,0.007671f,0.001191f,0.008018f,0.001940f,0.011299f,0.016449f,0.013306f,
0.008774f,0.004654f,0.013107f,0.015564f,0.007858f,0.017914f,0.014290f,0.018631f,0.008560f,0.035339f,0.008736f,0.020630f,0.014854f,0.010376f,0.025848f,0.025070f,0.050232f,0.018051f,0.018204f,0.010704f,0.032684f,0.040375f,0.037323f,0.014854f,0.019104f,0.015236f,0.005878f,0.000551f,0.000593f,0.000482f,
0.000295f,0.009460f,0.001723f,0.003429f,0.002798f,0.000713f,0.002396f,0.000652f,0.005306f,0.000553f,0.001050f,0.001301f,0.000551f,0.000345f,0.003269f,0.003168f,0.000935f,0.003429f,0.001968f,0.004234f,0.000114f,0.000404f,0.000515f,0.002129f,0.000657f,0.005028f,0.000647f,0.001298f,0.000405f,0.001431f,
0.000609f,0.002451f,0.000114f,0.000784f,0.001239f,0.000488f,0.000262f,0.000505f,0.000739f,0.000466f,0.001031f,0.000625f,0.001136f,0.001465f,0.000484f,0.000405f,0.000587f,0.015717f,0.002670f,0.000367f,0.001738f,0.010399f,0.001256f,0.001559f,0.003857f,0.001673f,0.000543f,0.005524f,0.000167f,0.000650f,
0.000905f,0.000243f,0.003002f,0.000213f,0.002758f,0.000115f,0.001850f,0.000470f,0.000234f,0.000937f,0.000665f,0.004074f,0.005787f,0.001607f,0.005569f,0.005974f,0.000930f,0.005974f,0.002001f,0.002018f,0.000438f,0.003351f,0.001186f,0.000370f,0.000192f,0.000175f,0.001355f,0.001795f,0.001245f,0.000463f,
0.001256f,0.000694f,0.001344f,0.001795f,0.000532f,0.002798f,0.001318f,0.001395f,0.002586f,0.001572f,0.000451f,0.003429f,0.000139f,0.005650f,0.000557f,0.002001f,0.000536f,0.000318f,0.000478f,0.002001f,0.001145f,0.004074f,0.004795f,0.001283f,0.004471f,0.000439f,0.005268f,0.000625f,0.000663f,0.000496f,
0.000279f,0.000585f,0.003124f,0.000505f,0.001497f,0.000765f,0.000211f,0.000968f,0.000580f,0.000543f,0.001232f,0.005268f,0.000739f,0.001850f,0.000526f,0.009460f,0.002451f,0.007545f,0.000333f,0.000322f,0.000780f,0.002977f,0.007973f,0.004543f,0.017014f,0.003826f,0.002651f,0.006611f,0.001738f,0.025330f,
0.004368f,0.002266f,0.004948f,0.003456f,0.015961f,0.010399f,0.001252f,0.000981f,0.002018f,0.000598f,0.000844f,0.005306f,0.002863f,0.003826f,0.001318f,0.001318f,0.000482f,0.010483f,0.000557f,0.002571f,0.000193f,0.001936f,0.007973f,0.002321f,0.002712f,0.000792f,0.001418f,0.000945f,0.018814f,0.004139f,
0.000854f,0.003429f,0.014320f,0.002247f,0.005478f,0.017136f,0.003098f,0.003098f,0.000355f,0.004505f,0.002230f,0.000757f,0.007721f,0.000935f,0.000749f,0.003708f,0.000603f,0.000232f,0.000631f,0.003073f,0.004612f,0.003223f,0.001497f,0.005745f,0.002911f,0.006657f,0.001239f,0.001594f,0.004105f,0.000267f,
0.000553f,0.003456f,0.001202f,0.000504f,0.001384f,0.003914f,0.008820f,0.001910f,0.009995f,0.005829f,0.003223f,0.000830f,0.001497f,0.005478f,0.001039f,0.005611f,0.001014f,0.000398f,0.002491f,0.001752f,0.003456f,0.002001f,0.010559f,0.001199f,0.001895f,0.001619f,0.004436f,0.001076f,0.007904f,0.005745f,
0.000514f,0.005146f,0.000361f,0.001696f,0.005398f,0.013878f,0.007092f,0.021164f,0.005745f,0.012253f,0.001202f,0.001497f,0.006210f,0.001723f,0.030319f,0.008102f,0.036835f,0.001431f,0.001766f,0.007034f,0.005878f,0.013351f,0.008621f,0.005352f,0.001256f,0.007664f,0.004105f,0.006504f,0.003195f,0.003052f,
0.012352f,0.008102f,0.002888f,0.001229f,0.014206f,0.001344f,0.005569f,0.000663f,0.003708f,0.003679f,0.003567f,0.001186f,0.005028f,0.000333f,0.000378f,0.000033f,0.000030f,0.000058f,0.000048f,0.000026f,0.000277f,0.000213f,0.000021f,0.000074f,0.000161f,0.000128f,0.000193f,0.000074f,0.000036f,0.000093f,
0.000243f,0.000029f,0.000022f,0.000033f,0.000111f,0.000218f,0.000145f,0.000476f,0.000608f,0.000044f,0.000036f,0.000104f,0.000711f,0.000195f,0.000381f,0.000256f,0.000302f,0.000158f,0.000195f,0.000280f,0.000504f,0.000071f,0.000116f,0.000361f,0.000074f,0.000115f,0.000053f,0.000224f,0.000326f,0.000138f,
0.000161f,0.000141f,0.000397f,0.000182f,0.000068f,0.000137f,0.000583f,0.000618f,0.000282f,0.001311f,0.001668f,0.001014f,0.000054f,0.000076f,0.000414f,0.000365f,0.001192f,0.000072f,0.000255f,0.000466f,0.000103f,0.001025f,0.000306f,0.001062f,0.000052f,0.000392f,0.000064f,0.000119f,0.000088f,0.000067f,
0.000072f,0.000145f,0.000231f,0.000208f,0.001355f,0.001815f,0.001067f,0.000888f,0.000454f,0.000132f,0.000621f,0.000806f,0.000196f,0.000353f,0.000325f,0.000131f,0.000507f,0.000544f,0.000212f,0.000144f,0.000235f,0.000152f,0.000423f,0.000277f,0.000998f,0.000170f,0.000747f,0.000541f,0.000604f,0.000151f,
0.000640f,0.000035f,0.000324f,0.000068f,0.000232f,0.000158f,0.000057f,0.000177f,0.000801f,0.000088f,0.000057f,0.000099f,0.000121f,0.000089f,0.000236f,0.000406f,0.000081f,0.000085f,0.000203f,0.000104f,0.000305f,0.000279f,0.000076f,0.000043f,0.000419f,0.000074f,0.000129f,0.000186f,0.000114f,0.000125f,
0.000355f,0.000123f,0.000460f,0.000505f,0.001534f,0.001073f,0.000974f,0.000063f,0.000232f,0.000564f,0.000494f,0.000149f,0.000488f,0.000947f,0.000747f,0.000037f,0.000068f,0.000423f,0.000548f,0.000451f,0.000136f,0.000355f,0.001612f,0.001921f,0.001497f,0.000156f,0.000236f,0.000215f,0.000566f,0.000186f,
0.000867f,0.001986f,0.000665f,0.001009f,0.000409f,0.000719f,0.001864f,0.001705f,0.000476f,0.000141f,0.000371f,0.000511f,0.000816f,0.000374f,0.001659f,0.002165f,0.000476f,0.001431f,0.000711f,0.000541f,0.000823f,0.001395f,0.001723f,0.002451f,0.000812f,0.000844f,0.001344f,0.000297f,0.001297f,0.001232f,
0.001682f,0.003246f,0.000315f,0.000636f,0.004265f,0.001418f,0.000455f,0.000635f,0.007488f,0.003197f,0.002230f,0.000488f,0.001062f,0.002062f,0.001623f,0.000630f,0.000398f,0.005432f,0.000876f,0.002607f,0.002193f,0.000857f,0.001458f,0.000778f,0.000741f,0.001895f,0.008476f,0.003532f,0.003790f,0.003143f,
0.003485f,0.000394f,0.001602f,0.001266f,0.002451f,0.001287f,0.002525f,0.000713f,0.001937f,0.004471f,0.001800f,0.003740f,0.005432f,0.000837f,0.000846f,0.006508f,0.001395f,0.009308f,0.006710f,0.004948f,0.011238f,0.002470f,0.003428f,0.005310f,0.002665f,0.002752f,0.009171f,0.003914f,0.004799f,0.005653f,
0.003561f,0.011406f,0.008553f,0.007973f,0.005966f,0.006252f,0.009605f,0.005604f,0.017273f,0.013969f,0.015472f,0.017929f,0.013237f,0.006813f,0.021988f,0.056580f,0.022690f,0.027573f,0.015121f,0.030777f,0.034851f,0.023590f,0.030060f,0.049927f,0.060699f,0.066650f,0.020325f,0.046204f,0.013237f,0.010307f,
0.007774f,0.046906f,0.044373f,0.249268f,0.003044f,0.002012f,0.001054f,0.002012f,0.003365f,0.001937f,0.007603f,0.000497f,0.000990f,0.001037f,0.000719f,0.009590f,0.002909f,0.001953f,0.000879f,0.000860f,0.001820f,0.003145f,0.000492f,0.001019f,0.001085f,0.000719f,0.006203f,0.002687f,0.000943f,0.000995f,
0.000719f,0.002159f,0.001517f,0.002567f,0.001163f,0.003506f,0.000814f,0.003563f,0.001937f,0.004566f,0.000990f,0.001404f,0.003874f,0.000751f,0.000922f,0.000874f,0.002504f,0.001028f,0.005646f,0.001068f,0.005737f,0.003506f,0.000990f,0.000912f,0.004124f,0.001340f,0.006004f,0.000674f,0.005348f,0.004745f,
0.003420f,0.001037f,0.000860f,0.003162f,0.001568f,0.002445f,0.001183f,0.002445f,0.001249f,0.001028f,0.004402f,0.000897f,0.007126f,0.001193f,0.003506f,0.000832f,0.001019f,0.004681f,0.003609f,0.000576f,0.000851f,0.000897f,0.000582f,0.004066f,0.002142f,0.002392f,0.002405f,0.003448f,0.001493f,0.003906f,
0.001656f,0.000860f,0.001112f,0.001340f,0.004902f,0.000711f,0.001921f,0.001734f,0.005390f,0.003347f,0.005600f,0.003214f,0.001995f,0.002464f,0.005093f,0.002062f,0.000618f,0.001995f,0.001219f,0.003971f,0.001415f,0.001963f,0.002159f,0.003145f,0.001319f,0.001710f,0.004002f,0.001995f,0.005600f,0.000832f,
0.000893f,0.000912f,0.000625f,0.002125f,0.002567f,0.001077f,0.002485f,0.001980f,0.001462f,0.001642f,0.004566f,0.001616f,0.005955f,0.001905f,0.001493f,0.002884f,0.001820f,0.001517f,0.002792f,0.006104f,0.000995f,0.003265f,0.001602f,0.001963f,0.003294f,0.002645f,0.000981f,0.001517f,0.004498f,0.007088f,
0.001193f,0.009163f,0.001363f,0.004822f,0.001719f,0.001530f,0.006443f,0.001239f,0.008873f,0.001790f,0.001602f,0.004902f,0.003294f,0.001719f,0.002213f,0.002296f,0.005093f,0.002230f,0.002316f,0.002884f,0.001415f,0.004261f,0.002602f,0.002769f,0.002230f,0.002424f,0.001875f,0.003590f,0.001953f,0.002884f,
0.002142f,0.001415f,0.002710f,0.002195f,0.001790f,0.004158f,0.000958f,0.004425f,0.001269f,0.004192f,0.002769f,0.003365f,0.005859f,0.001228f,0.001555f,0.001259f,0.002645f,0.002373f,0.002354f,0.002464f,0.002028f,0.003044f,0.002296f,0.003265f,0.002666f,0.002213f,0.002316f,0.001239f,0.003790f,0.000731f,
0.002108f,0.001163f,0.002546f,0.003069f,0.000628f,0.001963f,0.003019f,0.002108f,0.003145f,0.002373f,0.001602f,0.001860f,0.003534f,0.002373f,0.002096f,0.003294f,0.001820f,0.001438f,0.002769f,0.001173f,0.002108f,0.001829f,0.002947f,0.003094f,0.001594f,0.001953f,0.004604f,0.001340f,0.000685f,0.001203f,
0.001581f,0.002125f,0.002792f,0.002012f,0.002243f,0.002947f,0.011475f,0.002687f,0.001351f,0.004784f,0.002230f,0.003448f,0.006496f,0.001054f,0.006290f,0.000974f,0.002930f,0.001473f,0.002567f,0.002581f,0.001890f,0.003420f,0.000734f,0.006237f,0.000700f,0.002546f,0.002602f,0.001829f,0.002687f,0.001340f,
0.001517f,0.002045f,0.001937f,0.001530f,0.000711f,0.002464f,0.002316f,0.001682f,0.000432f,0.000937f,0.001669f,0.001790f,0.001173f,0.001462f,0.002028f,0.001748f,0.000974f,0.001581f,0.001581f,0.001804f,0.001517f,0.000011f,0.000006f,0.000000f,0.000000f,0.000006f,0.000000f,0.000000f,0.000002f,0.000002f,
0.000003f,0.000003f,0.000008f,0.000001f,0.000002f,0.000001f,0.000001f,0.000030f,0.000306f,0.000022f,0.000006f,0.000011f,0.000001f,0.000001f,0.000000f,0.000005f,0.000003f,0.000003f,0.000004f,0.000066f,0.000154f,0.000739f,0.000092f,0.000127f,0.000015f,0.000017f,0.000016f,0.000219f,0.000120f,0.000012f,
0.000008f,0.000007f,0.000011f,0.000213f,0.000276f,0.000197f,0.000276f,0.000009f,0.000059f,0.000004f,0.000015f,0.000134f,0.000037f,0.000022f,0.000027f,0.000077f,0.000040f,0.000039f,0.000009f,0.000004f,0.000000f,0.000003f,0.000104f,0.000484f,0.000172f,0.000032f,0.000004f,0.000048f,0.000083f,0.000252f,
0.000319f,0.000285f,0.000012f,0.000072f,0.000024f,0.000209f,0.000428f,0.000143f,0.000018f,0.000007f,0.000010f,0.000004f,0.000013f,0.000013f,0.000001f,0.000042f,0.000029f,0.000106f,0.000635f,0.000570f,0.000016f,0.000018f,0.000021f,0.000028f,0.000811f,0.000823f,0.000659f,0.000076f,0.000220f,0.000050f,
0.000094f,0.000232f,0.000110f,0.000016f,0.000039f,0.000130f,0.000119f,0.000410f,0.000080f,0.000016f,0.000050f,0.000107f,0.000026f,0.000131f,0.000009f,0.000001f,0.000003f,0.000000f,0.000002f,0.000020f,0.000123f,0.000033f,0.000050f,0.000005f,0.000006f,0.000001f,0.000018f,0.000015f,0.000035f,0.000014f,
0.000039f,0.000030f,0.000414f,0.000012f,0.000006f,0.000010f,0.000010f,0.000011f,0.000017f,0.000021f,0.000006f,0.000012f,0.000007f,0.000059f,0.000044f,0.000143f,0.000346f,0.000023f,0.000034f,0.000014f,0.000044f,0.000631f,0.000659f,0.000087f,0.000130f,0.000360f,0.000139f,0.000093f,0.000002f,0.000001f,
0.000005f,0.000007f,0.000112f,0.000072f,0.000377f,0.000124f,0.000201f,0.000045f,0.000190f,0.000267f,0.000692f,0.001716f,0.000104f,0.000184f,0.000302f,0.001097f,0.000811f,0.000195f,0.000520f,0.000149f,0.000291f,0.000100f,0.000656f,0.000444f,0.000003f,0.000009f,0.000020f,0.001017f,0.000324f,0.000392f,
0.000034f,0.000004f,0.000001f,0.000010f,0.000045f,0.000157f,0.000514f,0.000168f,0.000193f,0.000051f,0.000074f,0.000168f,0.000188f,0.000095f,0.000027f,0.000129f,0.000323f,0.001392f,0.001423f,0.000163f,0.000337f,0.001333f,0.002083f,0.002718f,0.000191f,0.000301f,0.000005f,0.000015f,0.000012f,0.000051f,
0.000870f,0.000263f,0.000120f,0.000080f,0.000528f,0.000099f,0.001312f,0.000384f,0.002439f,0.000084f,0.000948f,0.001312f,0.003412f,0.000487f,0.000263f,0.000069f,0.000083f,0.000144f,0.000469f,0.000399f,0.000739f,0.000080f,0.000007f,0.000459f,0.002102f,0.005707f,0.001056f,0.001073f,0.000146f,0.000032f,
0.000047f,0.000479f,0.000466f,0.001370f,0.004620f,0.004696f,0.001976f,0.004181f,0.000739f,0.000328f,0.000016f,0.000016f,0.000033f,0.000414f,0.005276f,0.001781f,0.000870f,0.000430f,0.000157f,0.004147f,0.003128f,0.002399f,0.004047f,0.001097f,0.001408f,0.008118f,0.003206f,0.004620f,0.012283f,0.004772f,
0.011536f,0.014565f,0.050873f,0.013397f,0.005318f,0.005802f,0.036926f,0.151733f,0.412598f,0.156616f,0.004745f,0.003899f,0.001677f,0.001939f,0.005878f,0.002089f,0.004120f,0.001963f,0.000613f,0.004349f,0.000334f,0.000519f,0.000383f,0.000793f,0.001505f,0.001499f,0.000941f,0.006405f,0.001666f,0.003664f,
0.004421f,0.000318f,0.000462f,0.000340f,0.000487f,0.000404f,0.003077f,0.003899f,0.006088f,0.000444f,0.004761f,0.000440f,0.004387f,0.000678f,0.004471f,0.000344f,0.003077f,0.000547f,0.003338f,0.008316f,0.000369f,0.002821f,0.004303f,0.004669f,0.000573f,0.005249f,0.000483f,0.009033f,0.000493f,0.000485f,
0.003872f,0.006065f,0.000443f,0.001246f,0.000653f,0.003338f,0.000592f,0.000499f,0.000730f,0.005028f,0.005165f,0.000365f,0.004932f,0.000403f,0.005306f,0.000272f,0.007458f,0.008583f,0.000414f,0.002041f,0.000533f,0.003338f,0.000257f,0.003563f,0.005188f,0.010391f,0.000306f,0.005112f,0.000227f,0.000368f,
0.000274f,0.000620f,0.008453f,0.000281f,0.000485f,0.000525f,0.008896f,0.000428f,0.003914f,0.003963f,0.005009f,0.004932f,0.000323f,0.001134f,0.000397f,0.016678f,0.000397f,0.014099f,0.000401f,0.000341f,0.010979f,0.001418f,0.000397f,0.000448f,0.001493f,0.000515f,0.005501f,0.000630f,0.003994f,0.000875f,
0.013718f,0.000403f,0.004814f,0.009499f,0.000486f,0.010193f,0.000338f,0.003914f,0.000252f,0.000443f,0.000302f,0.005901f,0.000409f,0.003195f,0.000325f,0.007904f,0.010765f,0.005459f,0.000486f,0.007233f,0.000445f,0.006428f,0.006638f,0.000424f,0.009689f,0.006138f,0.000470f,0.000555f,0.003414f,0.000468f,
0.004581f,0.000496f,0.001103f,0.000605f,0.010681f,0.012062f,0.000774f,0.013939f,0.006580f,0.001411f,0.002798f,0.000772f,0.007729f,0.006454f,0.001418f,0.001397f,0.001390f,0.008484f,0.006279f,0.000582f,0.000507f,0.000863f,0.000388f,0.012535f,0.000504f,0.008003f,0.015358f,0.001781f,0.000535f,0.000527f,
0.004814f,0.000450f,0.003664f,0.000774f,0.003414f,0.000795f,0.004471f,0.000438f,0.000537f,0.001109f,0.000537f,0.006428f,0.008553f,0.000679f,0.001880f,0.000861f,0.004028f,0.001038f,0.001267f,0.001424f,0.000847f,0.001159f,0.002008f,0.000780f,0.005520f,0.002548f,0.000809f,0.005009f,0.000898f,0.009247f,
0.005520f,0.001029f,0.006405f,0.004089f,0.008896f,0.000828f,0.000941f,0.001302f,0.000504f,0.001505f,0.002041f,0.000938f,0.001915f,0.000501f,0.001595f,0.004856f,0.016541f,0.000556f,0.004833f,0.006329f,0.000600f,0.008751f,0.007458f,0.007427f,0.000625f,0.001134f,0.001150f,0.004314f,0.000485f,0.000844f,
0.010727f,0.001163f,0.005806f,0.006016f,0.000774f,0.000751f,0.001553f,0.000875f,0.001029f,0.004856f,0.021576f,0.000722f,0.004303f,0.018387f,0.007549f,0.001570f,0.002590f,0.001781f,0.005249f,0.000811f,0.014488f,0.010681f,0.002283f,0.004040f,0.001397f,0.001226f,0.007484f,0.017273f,0.003248f,0.002106f,
0.002155f,0.002190f,0.002491f,0.007729f,0.017960f,0.003593f,0.001469f,0.001267f,0.002649f,0.001822f,0.004745f,0.003456f,0.001505f,0.001704f,0.004250f,0.001570f,0.001046f,0.001752f,0.001698f,0.003351f,0.001588f,0.002329f,0.002934f,0.003794f,0.002329f,0.002701f,0.003248f,0.003389f,0.003050f,0.000939f,
0.000847f,0.000490f,0.000056f,0.000037f,0.000049f,0.000045f,0.000736f,0.000429f,0.000037f,0.000112f,0.000052f,0.000156f,0.000384f,0.000754f,0.000106f,0.000115f,0.000066f,0.000065f,0.000060f,0.000059f,0.000144f,0.000074f,0.000202f,0.000754f,0.000631f,0.000026f,0.000034f,0.000071f,0.001204f,0.000172f,
0.000709f,0.000070f,0.000543f,0.000084f,0.000569f,0.000552f,0.001224f,0.000033f,0.000090f,0.000748f,0.000078f,0.000102f,0.000061f,0.000345f,0.000119f,0.000343f,0.000073f,0.000350f,0.000535f,0.000117f,0.000034f,0.000146f,0.001557f,0.000522f,0.000028f,0.000502f,0.003714f,0.000436f,0.000044f,0.000093f,
0.000620f,0.000147f,0.000736f,0.000037f,0.000221f,0.000107f,0.000052f,0.001463f,0.000079f,0.001764f,0.000037f,0.001038f,0.000074f,0.000109f,0.000204f,0.000320f,0.000169f,0.000443f,0.000156f,0.000426f,0.003626f,0.000156f,0.001204f,0.001254f,0.000677f,0.000027f,0.001346f,0.000159f,0.000034f,0.000070f,
0.000051f,0.000626f,0.000587f,0.001764f,0.000055f,0.000424f,0.000091f,0.000424f,0.000772f,0.000047f,0.000364f,0.000490f,0.001557f,0.000368f,0.001364f,0.000067f,0.001254f,0.000026f,0.000772f,0.000039f,0.000255f,0.000038f,0.000023f,0.000086f,0.000164f,0.000210f,0.000104f,0.000263f,0.000088f,0.000206f,
0.000077f,0.000401f,0.000034f,0.000108f,0.000074f,0.000019f,0.000079f,0.000829f,0.000033f,0.000214f,0.000081f,0.000022f,0.000242f,0.000053f,0.000063f,0.000345f,0.000480f,0.000058f,0.000645f,0.000176f,0.004688f,0.003353f,0.001245f,0.000039f,0.000060f,0.000384f,0.000461f,0.000779f,0.001712f,0.007442f,
0.002842f,0.000092f,0.000147f,0.001023f,0.002214f,0.000560f,0.000039f,0.000088f,0.001755f,0.003569f,0.006935f,0.000349f,0.000055f,0.000556f,0.000066f,0.000095f,0.005531f,0.001884f,0.002115f,0.000325f,0.000587f,0.000220f,0.013687f,0.000157f,0.001432f,0.000067f,0.001500f,0.002234f,0.000354f,0.001536f,
0.000255f,0.000350f,0.000552f,0.007038f,0.002548f,0.000110f,0.001334f,0.005116f,0.000573f,0.000682f,0.001292f,0.000229f,0.001150f,0.000075f,0.004948f,0.001303f,0.000335f,0.001657f,0.000109f,0.000161f,0.003048f,0.000103f,0.000094f,0.000195f,0.003767f,0.004948f,0.001292f,0.000902f,0.000772f,0.000975f,
0.004948f,0.000320f,0.000403f,0.004726f,0.000134f,0.000370f,0.002474f,0.000284f,0.000327f,0.000809f,0.000151f,0.000308f,0.000931f,0.004040f,0.012360f,0.003176f,0.000472f,0.000954f,0.003735f,0.000156f,0.004551f,0.000163f,0.000225f,0.001195f,0.004074f,0.001245f,0.004688f,0.004139f,0.000368f,0.000417f,
0.001262f,0.001764f,0.000279f,0.002592f,0.005577f,0.000345f,0.006367f,0.000217f,0.001740f,0.002396f,0.003538f,0.007793f,0.024750f,0.011787f,0.007095f,0.000417f,0.000498f,0.020370f,0.006115f,0.023254f,0.009117f,0.021332f,0.000410f,0.000408f,0.019135f,0.016876f,0.014206f,0.036011f,0.011978f,0.000464f,
0.035461f,0.020370f,0.034637f,0.004990f,0.014206f,0.053650f,0.067322f,0.016754f,0.003656f,0.106689f,0.002958f,0.038635f,0.002455f,0.068909f,0.007378f,0.006763f,0.002132f,0.017975f,0.001050f,0.001033f,0.000108f,0.000048f,0.001208f,0.000047f,0.000101f,0.000123f,0.000373f,0.000282f,0.000065f,0.000163f,
0.000074f,0.000129f,0.000091f,0.000046f,0.000337f,0.000724f,0.000055f,0.000120f,0.000333f,0.000063f,0.000155f,0.000075f,0.000288f,0.000071f,0.000282f,0.001121f,0.001139f,0.000128f,0.000566f,0.000080f,0.001771f,0.000511f,0.001242f,0.000218f,0.002043f,0.000343f,0.000426f,0.003296f,0.000191f,0.001407f,
0.001445f,0.003948f,0.000397f,0.000696f,0.000357f,0.003023f,0.000375f,0.000319f,0.001136f,0.001307f,0.000501f,0.000317f,0.000669f,0.000895f,0.000963f,0.000319f,0.000895f,0.003456f,0.002247f,0.000553f,0.000705f,0.000252f,0.002729f,0.000669f,0.003096f,0.002525f,0.000952f,0.000631f,0.001623f,0.000501f,
0.000618f,0.001505f,0.001538f,0.008408f,0.000553f,0.000853f,0.000308f,0.000515f,0.000358f,0.000526f,0.000952f,0.000197f,0.000730f,0.001228f,0.002729f,0.000937f,0.001030f,0.001551f,0.001452f,0.001880f,0.000705f,0.000518f,0.000767f,0.003242f,0.000843f,0.008087f,0.000905f,0.000880f,0.004330f,0.001242f,
0.000989f,0.000968f,0.000374f,0.001061f,0.002228f,0.001493f,0.000700f,0.000445f,0.011955f,0.001100f,0.001756f,0.003851f,0.000614f,0.001297f,0.000686f,0.001021f,0.000385f,0.000634f,0.000491f,0.000805f,0.000270f,0.002016f,0.000624f,0.001834f,0.002043f,0.001422f,0.000895f,0.001785f,0.000819f,0.001266f,
0.003048f,0.000843f,0.003222f,0.001820f,0.001024f,0.000431f,0.002502f,0.000947f,0.001558f,0.000814f,0.001900f,0.000542f,0.003168f,0.004471f,0.001593f,0.001935f,0.001186f,0.001422f,0.001199f,0.002285f,0.008545f,0.005219f,0.001473f,0.001380f,0.001864f,0.007786f,0.003971f,0.001952f,0.000654f,0.000705f,
0.001722f,0.003023f,0.001864f,0.003948f,0.008476f,0.001418f,0.000739f,0.001849f,0.003763f,0.001372f,0.001984f,0.001369f,0.002411f,0.002180f,0.001151f,0.001525f,0.000395f,0.002060f,0.001180f,0.003242f,0.004234f,0.001593f,0.000805f,0.001322f,0.003618f,0.002111f,0.000922f,0.002043f,0.000411f,0.000762f,
0.004574f,0.000676f,0.007721f,0.001300f,0.001044f,0.002180f,0.000361f,0.008949f,0.003269f,0.001380f,0.006809f,0.001623f,0.004795f,0.000526f,0.001452f,0.001100f,0.000929f,0.001915f,0.003536f,0.001682f,0.002266f,0.001021f,0.001966f,0.002077f,0.006504f,0.000874f,0.005924f,0.006504f,0.001144f,0.020020f,
0.011223f,0.003618f,0.000277f,0.001920f,0.001766f,0.004330f,0.000759f,0.001952f,0.012924f,0.001497f,0.007538f,0.007034f,0.001322f,0.000947f,0.001445f,0.001151f,0.000566f,0.007141f,0.038879f,0.001357f,0.004612f,0.025513f,0.015717f,0.003479f,0.004398f,0.005188f,0.003733f,0.002321f,0.006454f,0.004166f,
0.003197f,0.003323f,0.004719f,0.001144f,0.016724f,0.033539f,0.004131f,0.002111f,0.001780f,0.003399f,0.002094f,0.020645f,0.041718f,0.004795f,0.004948f,0.001126f,0.004330f,0.008820f,0.037109f,0.005779f,0.002129f,0.006554f,0.019257f,0.006397f,0.002111f,0.004990f,0.007965f,0.014763f,0.004833f,0.013870f,
0.016983f,0.012428f,0.007488f,0.014313f,0.020645f,0.037109f,0.027161f,0.002615f,0.001070f,0.000146f,0.000134f,0.002205f,0.000133f,0.000416f,0.000463f,0.000111f,0.001090f,0.000032f,0.000067f,0.000032f,0.000125f,0.000163f,0.000179f,0.000360f,0.004524f,0.000174f,0.000478f,0.001100f,0.000030f,0.000070f,
0.000036f,0.000095f,0.000095f,0.000857f,0.002405f,0.003843f,0.000143f,0.002792f,0.000132f,0.004593f,0.000175f,0.003111f,0.000093f,0.000995f,0.000128f,0.000915f,0.005051f,0.000096f,0.002260f,0.002657f,0.003492f,0.000144f,0.002136f,0.000150f,0.006329f,0.000177f,0.000125f,0.001986f,0.002329f,0.000249f,
0.000320f,0.000195f,0.001322f,0.000173f,0.000108f,0.000214f,0.003901f,0.003929f,0.000168f,0.002792f,0.000210f,0.003366f,0.000134f,0.006626f,0.004738f,0.000198f,0.001212f,0.000293f,0.000963f,0.000151f,0.002188f,0.002726f,0.007698f,0.000294f,0.001581f,0.000060f,0.000119f,0.000060f,0.000210f,0.004490f,
0.000150f,0.000253f,0.000287f,0.005718f,0.000198f,0.002499f,0.001651f,0.002071f,0.003414f,0.000268f,0.001203f,0.000178f,0.009949f,0.000262f,0.011284f,0.000273f,0.000154f,0.009064f,0.000603f,0.000228f,0.000156f,0.001466f,0.000178f,0.004219f,0.000213f,0.001414f,0.000288f,0.011642f,0.000138f,0.002747f,
0.003338f,0.000211f,0.005764f,0.000177f,0.001980f,0.000078f,0.000154f,0.000072f,0.002260f,0.000142f,0.001835f,0.000098f,0.003551f,0.004280f,0.003036f,0.000163f,0.003134f,0.000262f,0.005245f,0.003635f,0.000147f,0.006584f,0.003036f,0.000184f,0.000328f,0.002054f,0.000144f,0.002224f,0.000216f,0.000413f,
0.000305f,0.007820f,0.006088f,0.000445f,0.014595f,0.001719f,0.000618f,0.001742f,0.000352f,0.007874f,0.006329f,0.000582f,0.000423f,0.000450f,0.004562f,0.003551f,0.000198f,0.000192f,0.000249f,0.000181f,0.006531f,0.000316f,0.008713f,0.013496f,0.000484f,0.000272f,0.000249f,0.005627f,0.000184f,0.002947f,
0.000374f,0.006790f,0.000405f,0.002329f,0.000211f,0.000218f,0.002123f,0.000173f,0.008255f,0.011017f,0.000402f,0.000601f,0.000271f,0.005543f,0.000311f,0.000245f,0.000288f,0.000270f,0.000320f,0.004780f,0.000217f,0.010765f,0.000538f,0.000222f,0.004452f,0.000288f,0.011734f,0.005051f,0.000270f,0.010033f,
0.002808f,0.008858f,0.000354f,0.000406f,0.002792f,0.000243f,0.001322f,0.003256f,0.000451f,0.002829f,0.000201f,0.000461f,0.002792f,0.018311f,0.000284f,0.007580f,0.007339f,0.000711f,0.014595f,0.010941f,0.010117f,0.000448f,0.000721f,0.000676f,0.006187f,0.000343f,0.000650f,0.018753f,0.000647f,0.006531f,
0.005375f,0.000700f,0.000383f,0.002602f,0.000459f,0.000771f,0.010849f,0.039062f,0.000492f,0.006683f,0.049774f,0.009209f,0.000910f,0.003664f,0.001170f,0.004387f,0.001441f,0.044952f,0.007820f,0.002680f,0.006187f,0.001385f,0.001256f,0.016037f,0.024445f,0.001318f,0.001373f,0.001738f,0.002171f,0.002260f,
0.018463f,0.028580f,0.001594f,0.001318f,0.001493f,0.002560f,0.002308f,0.020752f,0.001963f,0.001534f,0.001915f,0.015068f,0.002260f,0.000830f,0.001567f,0.002224f,0.008125f,0.001795f,0.010521f,0.003723f,0.008583f,0.002346f,0.013290f,0.016800f,0.016800f,0.004089f,0.000977f,0.001307f,0.000220f,0.000165f,
0.000384f,0.000077f,0.000049f,0.000026f,0.000119f,0.000083f,0.000077f,0.000121f,0.000057f,0.000100f,0.000120f,0.000237f,0.001629f,0.001880f,0.000337f,0.000204f,0.000386f,0.000121f,0.000214f,0.000122f,0.000362f,0.000165f,0.000286f,0.000643f,0.000876f,0.000981f,0.001047f,0.000237f,0.001003f,0.000801f,
0.000960f,0.000846f,0.002235f,0.001121f,0.000464f,0.000900f,0.000709f,0.001064f,0.001531f,0.001814f,0.001411f,0.000778f,0.000631f,0.001145f,0.000560f,0.000757f,0.000747f,0.000453f,0.000185f,0.000085f,0.000185f,0.000105f,0.000171f,0.000150f,0.000380f,0.000565f,0.001659f,0.002148f,0.001686f,0.000412f,
0.000568f,0.000559f,0.000698f,0.000920f,0.001573f,0.000494f,0.002068f,0.000348f,0.001633f,0.000915f,0.001319f,0.004482f,0.001590f,0.000797f,0.000915f,0.001273f,0.000554f,0.000422f,0.000243f,0.000367f,0.001249f,0.002323f,0.000430f,0.002033f,0.000158f,0.000103f,0.000167f,0.000422f,0.002062f,0.000782f,
0.004189f,0.001766f,0.001177f,0.001219f,0.000942f,0.001590f,0.000805f,0.002157f,0.001796f,0.002398f,0.000326f,0.002644f,0.000755f,0.001870f,0.000258f,0.000591f,0.001249f,0.000410f,0.000194f,0.000114f,0.000042f,0.000086f,0.000239f,0.000280f,0.000957f,0.001700f,0.000656f,0.000330f,0.000273f,0.000445f,
0.000669f,0.000346f,0.000485f,0.000346f,0.001615f,0.000321f,0.000825f,0.000192f,0.000089f,0.000743f,0.000515f,0.000834f,0.003052f,0.001081f,0.001747f,0.001615f,0.000920f,0.000907f,0.003698f,0.001037f,0.000453f,0.000715f,0.000643f,0.000386f,0.000915f,0.003355f,0.001219f,0.005753f,0.003948f,0.004189f,
0.001031f,0.000245f,0.000387f,0.000435f,0.000448f,0.000776f,0.000776f,0.001555f,0.003204f,0.000966f,0.001905f,0.000795f,0.000927f,0.001845f,0.001547f,0.003069f,0.001633f,0.001534f,0.001121f,0.001273f,0.002182f,0.005615f,0.001115f,0.005596f,0.001805f,0.001974f,0.004448f,0.002359f,0.002378f,0.001143f,
0.000576f,0.002724f,0.001056f,0.001762f,0.000342f,0.000355f,0.000072f,0.000091f,0.000638f,0.002445f,0.001505f,0.002001f,0.001911f,0.001319f,0.000464f,0.001362f,0.001273f,0.001963f,0.001615f,0.000977f,0.001766f,0.002552f,0.005463f,0.000741f,0.004375f,0.004375f,0.007103f,0.007149f,0.001781f,0.001366f,
0.002483f,0.000574f,0.001350f,0.001911f,0.003044f,0.002924f,0.002594f,0.001569f,0.000992f,0.001174f,0.001790f,0.004078f,0.011703f,0.003111f,0.007435f,0.014458f,0.002531f,0.004543f,0.000661f,0.000453f,0.001040f,0.002073f,0.000511f,0.006073f,0.003002f,0.001845f,0.003212f,0.004982f,0.004543f,0.004509f,
0.011444f,0.004620f,0.000626f,0.002747f,0.000465f,0.001294f,0.001239f,0.003319f,0.010658f,0.002491f,0.013573f,0.004307f,0.005375f,0.003937f,0.003611f,0.002861f,0.003002f,0.011078f,0.005520f,0.003551f,0.001674f,0.001855f,0.006073f,0.003265f,0.004459f,0.032837f,0.003204f,0.008759f,0.007732f,0.021698f,
0.009262f,0.029404f,0.010086f,0.006020f,0.033081f,0.008629f,0.011345f,0.014229f,0.089905f,0.010338f,0.092041f,0.037781f,0.041504f,0.026779f,0.029648f,0.016403f,0.005898f,0.002535f,0.000793f,0.002535f,0.000711f,0.001458f,0.005943f,0.001434f,0.001745f,0.000759f,0.000844f,0.000837f,0.002121f,0.002422f,
0.000853f,0.000543f,0.000806f,0.000837f,0.001577f,0.001423f,0.000739f,0.000940f,0.000979f,0.001225f,0.001220f,0.000741f,0.001051f,0.001200f,0.002007f,0.001627f,0.001337f,0.001119f,0.000927f,0.001256f,0.000759f,0.004700f,0.001171f,0.000793f,0.002024f,0.000597f,0.001200f,0.001564f,0.001552f,0.000458f,
0.001161f,0.000637f,0.002104f,0.000751f,0.000822f,0.001479f,0.001220f,0.001236f,0.006840f,0.001664f,0.000987f,0.001652f,0.004559f,0.000853f,0.003439f,0.002598f,0.000858f,0.001627f,0.001704f,0.002680f,0.000598f,0.002104f,0.002619f,0.001600f,0.000816f,0.002441f,0.001042f,0.001109f,0.001209f,0.001458f,
0.005714f,0.001337f,0.002188f,0.001411f,0.001678f,0.001745f,0.005630f,0.001526f,0.001884f,0.003838f,0.001348f,0.002518f,0.001652f,0.001858f,0.000940f,0.001315f,0.000987f,0.000844f,0.001081f,0.001704f,0.002556f,0.000765f,0.004631f,0.000778f,0.000955f,0.002041f,0.001587f,0.000696f,0.001664f,0.001411f,
0.001884f,0.002291f,0.001745f,0.001479f,0.002743f,0.004490f,0.001315f,0.002367f,0.002943f,0.000979f,0.004082f,0.000774f,0.003553f,0.001109f,0.001315f,0.001423f,0.002258f,0.001479f,0.001690f,0.000477f,0.002831f,0.002069f,0.002899f,0.001469f,0.002024f,0.000757f,0.002151f,0.003035f,0.000477f,0.002478f,
0.001664f,0.000533f,0.000940f,0.001423f,0.000658f,0.001308f,0.002808f,0.002943f,0.001392f,0.004284f,0.003664f,0.000806f,0.006786f,0.006840f,0.002207f,0.010033f,0.001870f,0.006134f,0.005203f,0.004597f,0.008720f,0.003012f,0.004818f,0.003496f,0.001978f,0.003334f,0.005810f,0.000963f,0.003721f,0.001171f,
0.002121f,0.006947f,0.006897f,0.003311f,0.002789f,0.003553f,0.001068f,0.002239f,0.006378f,0.001771f,0.002138f,0.001771f,0.001932f,0.003086f,0.001539f,0.001932f,0.002535f,0.003933f,0.001828f,0.008652f,0.003334f,0.003061f,0.004082f,0.007111f,0.002104f,0.002121f,0.003553f,0.003838f,0.001961f,0.005585f,
0.006325f,0.001526f,0.003439f,0.002556f,0.006042f,0.002518f,0.002151f,0.002876f,0.002459f,0.003691f,0.002943f,0.002619f,0.002291f,0.001136f,0.003107f,0.006634f,0.003780f,0.001786f,0.001411f,0.006275f,0.002720f,0.007225f,0.001993f,0.003635f,0.003838f,0.002121f,0.008926f,0.006786f,0.004559f,0.003132f,
0.005630f,0.002720f,0.002808f,0.001516f,0.004353f,0.008453f,0.003607f,0.004284f,0.004284f,0.002069f,0.002966f,0.002766f,0.003086f,0.003496f,0.003750f,0.023331f,0.001577f,0.005047f,0.017334f,0.015533f,0.003576f,0.003208f,0.005001f,0.004383f,0.003311f,0.015419f,0.009956f,0.005810f,0.012779f,0.004559f,
0.004818f,0.006481f,0.014488f,0.008186f,0.004597f,0.008003f,0.005810f,0.008514f,0.008514f,0.017609f,0.008064f,0.003387f,0.003286f,0.009956f,0.003607f,0.012390f,0.007454f,0.003286f,0.002899f,0.011459f,0.001613f,0.001552f,0.004631f,0.001359f,0.007572f,0.004353f,0.007572f,0.002556f,0.007404f,0.002188f,
0.012093f,0.014717f,0.016800f,0.002831f,0.000015f,0.000024f,0.000002f,0.000002f,0.000016f,0.000001f,0.000002f,0.000035f,0.000004f,0.000032f,0.000002f,0.000003f,0.000001f,0.000001f,0.000001f,0.000002f,0.000024f,0.000070f,0.000003f,0.000004f,0.000039f,0.000002f,0.000005f,0.000003f,0.000006f,0.000005f,
0.000018f,0.000039f,0.000037f,0.000024f,0.000069f,0.000012f,0.000057f,0.000043f,0.000067f,0.000035f,0.000032f,0.000049f,0.000040f,0.000160f,0.000035f,0.000147f,0.000316f,0.000296f,0.000106f,0.000350f,0.000037f,0.000571f,0.000054f,0.000119f,0.000745f,0.000547f,0.000030f,0.000294f,0.000049f,0.000848f,
0.000031f,0.000192f,0.000064f,0.000595f,0.001647f,0.000144f,0.000685f,0.000067f,0.000365f,0.000015f,0.000674f,0.000435f,0.000110f,0.000326f,0.000115f,0.000133f,0.000060f,0.000234f,0.000427f,0.000309f,0.000026f,0.000485f,0.000037f,0.000056f,0.000026f,0.000093f,0.000772f,0.000022f,0.000466f,0.000130f,
0.001263f,0.000091f,0.000473f,0.000273f,0.000496f,0.000559f,0.000030f,0.000652f,0.000182f,0.000998f,0.000029f,0.001109f,0.000025f,0.000068f,0.001060f,0.000160f,0.000035f,0.000176f,0.000561f,0.000260f,0.000437f,0.000239f,0.000169f,0.000386f,0.000344f,0.000072f,0.000339f,0.000174f,0.000004f,0.000247f,
0.000008f,0.000261f,0.000053f,0.000102f,0.000038f,0.000237f,0.000036f,0.000369f,0.000016f,0.000217f,0.000307f,0.000238f,0.000097f,0.000288f,0.000034f,0.000399f,0.000155f,0.000038f,0.000453f,0.000316f,0.000109f,0.000044f,0.000600f,0.000060f,0.000490f,0.000097f,0.000369f,0.000105f,0.000871f,0.000445f,
0.000029f,0.000200f,0.000361f,0.000130f,0.000996f,0.000440f,0.001158f,0.001948f,0.000138f,0.000204f,0.000018f,0.000490f,0.000953f,0.000045f,0.000101f,0.000586f,0.000074f,0.000562f,0.000038f,0.000523f,0.000533f,0.000270f,0.000215f,0.000260f,0.000595f,0.000124f,0.000247f,0.000412f,0.000739f,0.000710f,
0.000384f,0.000474f,0.000227f,0.001319f,0.000373f,0.001534f,0.001880f,0.000113f,0.001242f,0.000581f,0.004734f,0.000633f,0.001355f,0.000231f,0.000035f,0.000559f,0.000729f,0.000490f,0.002739f,0.001427f,0.000288f,0.002316f,0.000208f,0.004307f,0.002165f,0.000274f,0.002644f,0.001177f,0.001896f,0.000527f,
0.000652f,0.001560f,0.000205f,0.000485f,0.000508f,0.001034f,0.001816f,0.000101f,0.000268f,0.000401f,0.000857f,0.000169f,0.001392f,0.002111f,0.000110f,0.001701f,0.002413f,0.003651f,0.000257f,0.001470f,0.000961f,0.004879f,0.000314f,0.000611f,0.006409f,0.000739f,0.002207f,0.003029f,0.000066f,0.000447f,
0.004047f,0.001019f,0.000628f,0.008270f,0.008202f,0.000296f,0.004787f,0.013519f,0.007675f,0.001051f,0.004620f,0.000674f,0.002342f,0.000209f,0.003403f,0.004536f,0.000979f,0.011887f,0.002541f,0.001408f,0.008293f,0.006660f,0.004341f,0.000399f,0.002056f,0.000757f,0.005234f,0.015854f,0.010406f,0.005554f,
0.001308f,0.001438f,0.010872f,0.007381f,0.019669f,0.009293f,0.004204f,0.006359f,0.030472f,0.007980f,0.008018f,0.013206f,0.009583f,0.026245f,0.026459f,0.041016f,0.039734f,0.025269f,0.074768f,0.082092f,0.107117f,0.082764f,0.100647f,0.001614f,0.002131f,0.000236f,0.000196f,0.001658f,0.000208f,0.000500f,
0.001713f,0.001258f,0.003477f,0.000968f,0.001304f,0.000916f,0.000468f,0.000345f,0.000434f,0.001043f,0.001215f,0.000243f,0.000578f,0.003084f,0.000722f,0.001019f,0.000730f,0.001417f,0.001043f,0.001166f,0.002121f,0.001858f,0.001282f,0.002121f,0.000899f,0.002226f,0.001554f,0.002707f,0.001093f,0.001410f,
0.001379f,0.001623f,0.003765f,0.001183f,0.002575f,0.003819f,0.002707f,0.000788f,0.002563f,0.000260f,0.003620f,0.000235f,0.000581f,0.002090f,0.002388f,0.000108f,0.001429f,0.000606f,0.002321f,0.000760f,0.001228f,0.000981f,0.002861f,0.004436f,0.000722f,0.001425f,0.000500f,0.001111f,0.000193f,0.001460f,
0.000932f,0.000916f,0.000816f,0.001161f,0.000709f,0.000748f,0.000883f,0.001279f,0.000970f,0.000135f,0.001932f,0.000476f,0.000695f,0.000476f,0.000438f,0.002403f,0.000816f,0.000838f,0.002142f,0.001984f,0.001793f,0.001448f,0.001183f,0.001874f,0.001793f,0.000896f,0.000847f,0.001943f,0.002254f,0.000746f,
0.002113f,0.000720f,0.001566f,0.002148f,0.000720f,0.000705f,0.001695f,0.001379f,0.001464f,0.001835f,0.001300f,0.000995f,0.001914f,0.001709f,0.000973f,0.001448f,0.001525f,0.000259f,0.002472f,0.000589f,0.003363f,0.001007f,0.001247f,0.000684f,0.002226f,0.000756f,0.002195f,0.000353f,0.002077f,0.001993f,
0.001835f,0.001881f,0.002054f,0.000638f,0.002745f,0.001816f,0.001448f,0.003344f,0.002834f,0.001745f,0.000968f,0.003586f,0.001698f,0.002941f,0.003538f,0.004311f,0.001989f,0.004311f,0.002399f,0.000622f,0.001722f,0.004135f,0.001005f,0.003603f,0.002346f,0.004650f,0.007645f,0.000538f,0.000868f,0.000294f,
0.003460f,0.006207f,0.002018f,0.002073f,0.002533f,0.000574f,0.003084f,0.000578f,0.003363f,0.003130f,0.001893f,0.002403f,0.002762f,0.003061f,0.002180f,0.002308f,0.002197f,0.003620f,0.004051f,0.002354f,0.002899f,0.001833f,0.004658f,0.002138f,0.004242f,0.005100f,0.000583f,0.004593f,0.003477f,0.005772f,
0.003603f,0.003294f,0.003477f,0.001513f,0.005199f,0.001745f,0.002090f,0.003902f,0.003483f,0.003853f,0.006634f,0.002609f,0.004192f,0.003084f,0.004761f,0.004089f,0.002054f,0.002613f,0.004906f,0.005707f,0.002409f,0.001021f,0.002171f,0.001136f,0.003210f,0.002249f,0.000574f,0.001793f,0.001745f,0.002956f,
0.003435f,0.002539f,0.003584f,0.001069f,0.005386f,0.008865f,0.009125f,0.002707f,0.002645f,0.005684f,0.005428f,0.001410f,0.001681f,0.006535f,0.004433f,0.003061f,0.004738f,0.001698f,0.005428f,0.004452f,0.005936f,0.003214f,0.005848f,0.006569f,0.001843f,0.004040f,0.006748f,0.011948f,0.006863f,0.004475f,
0.010376f,0.004009f,0.003273f,0.007881f,0.012840f,0.007458f,0.006317f,0.013329f,0.004715f,0.006748f,0.004860f,0.003828f,0.005039f,0.004692f,0.007446f,0.005722f,0.006927f,0.004883f,0.003971f,0.019348f,0.008392f,0.005360f,0.017685f,0.007957f,0.005020f,0.021011f,0.028931f,0.010139f,0.020599f,0.016052f,
0.003435f,0.021591f,0.008163f,0.006535f,0.010460f,0.021011f,0.007317f,0.021088f,0.014153f,0.010361f,0.010506f,0.008621f,0.000401f,0.001920f,0.000038f,0.000018f,0.001348f,0.000018f,0.000091f,0.001267f,0.000101f,0.003498f,0.000234f,0.000244f,0.000171f,0.000035f,0.000044f,0.000032f,0.001092f,0.001982f,
0.000027f,0.000112f,0.003401f,0.000268f,0.000358f,0.000272f,0.000121f,0.000230f,0.002316f,0.004879f,0.003757f,0.000184f,0.001098f,0.000261f,0.004070f,0.000240f,0.004410f,0.000255f,0.002436f,0.000256f,0.002674f,0.007679f,0.000291f,0.006054f,0.006294f,0.007099f,0.000302f,0.005146f,0.000142f,0.008316f,
0.000173f,0.000349f,0.003929f,0.007706f,0.000167f,0.001784f,0.000179f,0.006218f,0.000205f,0.001683f,0.000209f,0.005455f,0.009689f,0.000236f,0.001378f,0.000270f,0.003223f,0.000092f,0.004539f,0.003757f,0.000240f,0.002956f,0.000242f,0.002859f,0.000226f,0.004650f,0.004578f,0.003815f,0.000103f,0.004978f,
0.000383f,0.000437f,0.000315f,0.000704f,0.002956f,0.000281f,0.003204f,0.000294f,0.007637f,0.000260f,0.003805f,0.004692f,0.004902f,0.005764f,0.000135f,0.001640f,0.000299f,0.012604f,0.000120f,0.011086f,0.000120f,0.000256f,0.009521f,0.001346f,0.000113f,0.000295f,0.002325f,0.000311f,0.004902f,0.000299f,
0.002897f,0.001874f,0.004467f,0.000253f,0.003006f,0.004524f,0.000083f,0.003126f,0.000103f,0.004833f,0.000501f,0.000552f,0.000377f,0.001200f,0.000298f,0.004505f,0.000121f,0.002234f,0.004559f,0.003328f,0.000328f,0.007050f,0.000145f,0.002205f,0.003929f,0.000225f,0.007858f,0.007648f,0.000262f,0.000257f,
0.005585f,0.000234f,0.003847f,0.001110f,0.000943f,0.000474f,0.008339f,0.004711f,0.000156f,0.000513f,0.004021f,0.000123f,0.002893f,0.000362f,0.007290f,0.011589f,0.000141f,0.000700f,0.000072f,0.005535f,0.009369f,0.000247f,0.000730f,0.002949f,0.000215f,0.005520f,0.000156f,0.012886f,0.008804f,0.003309f,
0.000371f,0.000314f,0.004520f,0.000404f,0.003578f,0.004021f,0.006775f,0.000566f,0.004742f,0.000426f,0.000634f,0.006294f,0.000477f,0.008003f,0.008598f,0.000254f,0.002996f,0.000473f,0.010368f,0.000644f,0.003027f,0.000393f,0.000375f,0.002913f,0.001667f,0.001213f,0.004440f,0.004009f,0.000426f,0.002378f,
0.000568f,0.011353f,0.009483f,0.000428f,0.015945f,0.005512f,0.009743f,0.000438f,0.000393f,0.003195f,0.000281f,0.003983f,0.001982f,0.000610f,0.003941f,0.000193f,0.001608f,0.003830f,0.008080f,0.000519f,0.007473f,0.007313f,0.000258f,0.007145f,0.011307f,0.005146f,0.000793f,0.006763f,0.000667f,0.009117f,
0.000470f,0.000826f,0.011978f,0.000813f,0.008484f,0.008583f,0.000295f,0.000714f,0.004829f,0.000745f,0.000879f,0.010033f,0.008965f,0.000368f,0.003963f,0.009064f,0.007515f,0.000704f,0.006413f,0.000751f,0.007214f,0.000417f,0.001224f,0.009590f,0.000281f,0.004768f,0.000730f,0.001146f,0.020218f,0.012566f,
0.005638f,0.000243f,0.001655f,0.000355f,0.002174f,0.025864f,0.014915f,0.006378f,0.000811f,0.001532f,0.008240f,0.001200f,0.015053f,0.006714f,0.001052f,0.001048f,0.011528f,0.001532f,0.001731f,0.004841f,0.001544f,0.010414f,0.009392f,0.016068f,0.001578f,0.011574f,0.002251f,0.013527f,0.016068f,0.017365f,
0.001768f,0.002691f,0.001913f,0.000142f,0.000152f,0.000660f,0.000138f,0.000537f,0.003147f,0.000678f,0.001817f,0.000143f,0.000251f,0.000108f,0.000462f,0.000084f,0.000109f,0.000161f,0.000428f,0.000068f,0.000322f,0.001078f,0.000075f,0.000127f,0.000055f,0.000434f,0.000136f,0.000262f,0.000520f,0.000449f,
0.000206f,0.001106f,0.000103f,0.000962f,0.000451f,0.000681f,0.000148f,0.000230f,0.000283f,0.000311f,0.001331f,0.000162f,0.000449f,0.000774f,0.000613f,0.000135f,0.001150f,0.000063f,0.001671f,0.000064f,0.000138f,0.001081f,0.000592f,0.000087f,0.000708f,0.000381f,0.001455f,0.000539f,0.000482f,0.000784f,
0.001213f,0.002218f,0.000295f,0.001274f,0.000135f,0.000476f,0.000156f,0.000518f,0.000337f,0.000660f,0.000449f,0.000923f,0.000242f,0.000391f,0.000325f,0.000576f,0.000454f,0.000213f,0.001141f,0.000105f,0.000179f,0.000078f,0.000190f,0.004662f,0.000234f,0.000479f,0.001348f,0.001536f,0.001040f,0.000962f,
0.000510f,0.000944f,0.001198f,0.000344f,0.000982f,0.000687f,0.001224f,0.000292f,0.001435f,0.000289f,0.000479f,0.001887f,0.000141f,0.000243f,0.000558f,0.001491f,0.000479f,0.001081f,0.000561f,0.000479f,0.001324f,0.000774f,0.000273f,0.001451f,0.001150f,0.000213f,0.003242f,0.000206f,0.002024f,0.000178f,
0.000254f,0.000117f,0.002640f,0.000229f,0.001852f,0.000224f,0.001131f,0.001595f,0.000916f,0.001178f,0.001257f,0.000375f,0.004272f,0.001451f,0.000813f,0.003536f,0.001707f,0.000831f,0.000374f,0.003233f,0.000815f,0.003683f,0.000842f,0.003359f,0.000618f,0.003368f,0.001712f,0.000541f,0.003517f,0.003323f,
0.000969f,0.005932f,0.001282f,0.004162f,0.007732f,0.001006f,0.001866f,0.000659f,0.003857f,0.007408f,0.001782f,0.000764f,0.002127f,0.000479f,0.001443f,0.000487f,0.002132f,0.001797f,0.000728f,0.001224f,0.002029f,0.003443f,0.001106f,0.001707f,0.001831f,0.002855f,0.002974f,0.001532f,0.002132f,0.000726f,
0.005108f,0.001048f,0.005600f,0.006020f,0.000543f,0.004059f,0.001959f,0.008080f,0.002533f,0.003014f,0.003225f,0.000678f,0.011116f,0.000802f,0.000567f,0.002800f,0.001040f,0.001467f,0.007435f,0.000930f,0.002832f,0.001675f,0.002758f,0.002172f,0.001274f,0.001630f,0.001582f,0.002514f,0.003775f,0.000551f,
0.000916f,0.000625f,0.001694f,0.003147f,0.000378f,0.000820f,0.001630f,0.002155f,0.001048f,0.002008f,0.003826f,0.000581f,0.006275f,0.011871f,0.018616f,0.000856f,0.001424f,0.002771f,0.011612f,0.000586f,0.000578f,0.009735f,0.002480f,0.003260f,0.006390f,0.001015f,0.002863f,0.009399f,0.003023f,0.001487f,
0.013725f,0.016815f,0.001168f,0.005035f,0.014214f,0.010445f,0.003878f,0.007881f,0.005596f,0.005043f,0.001582f,0.013893f,0.009628f,0.003574f,0.023087f,0.003857f,0.001846f,0.009773f,0.007858f,0.003672f,0.003450f,0.011703f,0.003199f,0.012947f,0.010994f,0.009148f,0.004555f,0.005814f,0.003105f,0.008598f,
0.007587f,0.020203f,0.006119f,0.004162f,0.006306f,0.028061f,0.003910f,0.003910f,0.016571f,0.003941f,0.016373f,0.025650f,0.027405f,0.007408f,0.018906f,0.005619f,0.067810f,0.050018f,0.038055f,0.003122f,0.002871f,0.002077f,0.000722f,0.000595f,0.001399f,0.000439f,0.000497f,0.001027f,0.001517f,0.002411f,
0.000832f,0.000907f,0.000719f,0.000777f,0.000999f,0.001663f,0.000609f,0.000866f,0.000816f,0.000775f,0.002045f,0.000631f,0.000647f,0.000496f,0.001915f,0.000707f,0.000633f,0.001044f,0.000981f,0.000912f,0.001572f,0.000616f,0.001297f,0.000648f,0.001310f,0.000732f,0.000707f,0.000759f,0.000754f,0.001655f,
0.000892f,0.001262f,0.001879f,0.001407f,0.000493f,0.001380f,0.000230f,0.002895f,0.000248f,0.000803f,0.001391f,0.001441f,0.000160f,0.001192f,0.001358f,0.001489f,0.000767f,0.000801f,0.000744f,0.002378f,0.003363f,0.001111f,0.001628f,0.000557f,0.000850f,0.000215f,0.000970f,0.000767f,0.000792f,0.000595f,
0.001064f,0.000511f,0.000944f,0.000659f,0.000925f,0.000868f,0.000185f,0.001509f,0.000525f,0.000498f,0.000344f,0.000263f,0.001580f,0.000511f,0.000505f,0.000899f,0.001238f,0.001047f,0.000963f,0.000664f,0.001079f,0.001195f,0.000548f,0.000485f,0.001441f,0.001968f,0.000467f,0.001362f,0.000462f,0.001366f,
0.001751f,0.000317f,0.000521f,0.001469f,0.001019f,0.001343f,0.001437f,0.001179f,0.000836f,0.002089f,0.001232f,0.000659f,0.001027f,0.001129f,0.000154f,0.002100f,0.000295f,0.002583f,0.001232f,0.001056f,0.000677f,0.001920f,0.000726f,0.001472f,0.000328f,0.001358f,0.001628f,0.001563f,0.001407f,0.001242f,
0.000436f,0.002037f,0.000984f,0.000722f,0.002945f,0.001770f,0.000905f,0.000855f,0.002443f,0.001189f,0.002005f,0.003216f,0.002628f,0.001452f,0.003382f,0.002222f,0.000382f,0.003052f,0.002197f,0.002537f,0.002590f,0.005199f,0.005585f,0.007385f,0.001208f,0.000754f,0.000607f,0.003437f,0.005165f,0.003220f,
0.002283f,0.002590f,0.000673f,0.002346f,0.000401f,0.003290f,0.002806f,0.002205f,0.001551f,0.003637f,0.002169f,0.000821f,0.002066f,0.001388f,0.002882f,0.001993f,0.001799f,0.002157f,0.001814f,0.003313f,0.001931f,0.003834f,0.003904f,0.000417f,0.003010f,0.004379f,0.004940f,0.002026f,0.002579f,0.001158f,
0.000905f,0.003208f,0.001272f,0.001559f,0.004082f,0.004349f,0.002916f,0.006104f,0.002216f,0.004009f,0.003035f,0.002476f,0.004681f,0.002071f,0.003389f,0.003132f,0.008469f,0.002089f,0.001070f,0.002640f,0.000950f,0.002321f,0.002340f,0.000382f,0.001421f,0.001641f,0.003077f,0.001719f,0.002411f,0.003300f,
0.000463f,0.004204f,0.005611f,0.006004f,0.001915f,0.002857f,0.002222f,0.005322f,0.001064f,0.001456f,0.006432f,0.002583f,0.003202f,0.004749f,0.000864f,0.003452f,0.003851f,0.004326f,0.003437f,0.006844f,0.007957f,0.001597f,0.004253f,0.012177f,0.003265f,0.002943f,0.005791f,0.003876f,0.005684f,0.001192f,
0.011848f,0.006439f,0.008278f,0.007175f,0.013908f,0.004807f,0.014748f,0.010498f,0.009789f,0.003702f,0.003323f,0.006756f,0.004253f,0.014580f,0.009789f,0.009331f,0.014809f,0.006634f,0.012321f,0.006874f,0.015396f,0.013268f,0.008644f,0.023483f,0.016266f,0.006397f,0.012177f,0.007465f,0.006905f,0.018631f,
0.012321f,0.024033f,0.011894f,0.019836f,0.025192f,0.040710f,0.034027f,0.031586f,0.008095f,0.006710f,0.002388f,0.001178f,0.000714f,0.002626f,0.000604f,0.000575f,0.001471f,0.000843f,0.002270f,0.001518f,0.001400f,0.001505f,0.000875f,0.001020f,0.000868f,0.001465f,0.002306f,0.001239f,0.001052f,0.002586f,
0.001576f,0.001207f,0.001377f,0.001086f,0.001377f,0.001537f,0.002455f,0.002424f,0.001069f,0.001812f,0.001673f,0.003633f,0.000902f,0.002970f,0.001062f,0.002535f,0.001147f,0.001995f,0.003883f,0.001058f,0.002415f,0.002769f,0.002922f,0.000896f,0.001817f,0.000498f,0.003462f,0.000478f,0.001278f,0.001878f,
0.001966f,0.000916f,0.001695f,0.001585f,0.002058f,0.001524f,0.001504f,0.001165f,0.003075f,0.003235f,0.001842f,0.001945f,0.002092f,0.002092f,0.000700f,0.002533f,0.001781f,0.001099f,0.001838f,0.001055f,0.001680f,0.000991f,0.002270f,0.002581f,0.003214f,0.000738f,0.002762f,0.001628f,0.001382f,0.001505f,
0.001160f,0.002100f,0.001228f,0.001451f,0.000849f,0.002819f,0.001224f,0.002310f,0.002083f,0.002277f,0.002106f,0.000430f,0.000988f,0.001029f,0.003719f,0.000471f,0.003332f,0.000458f,0.001119f,0.003847f,0.001570f,0.000430f,0.000882f,0.001491f,0.000947f,0.003351f,0.001089f,0.002260f,0.001898f,0.004135f,
0.001055f,0.002703f,0.003922f,0.000617f,0.003851f,0.000410f,0.003523f,0.002710f,0.002392f,0.002771f,0.002327f,0.002378f,0.003208f,0.000923f,0.003288f,0.003462f,0.004417f,0.001495f,0.003139f,0.000748f,0.003529f,0.004147f,0.001035f,0.006016f,0.003780f,0.001141f,0.002794f,0.004669f,0.001510f,0.003420f,
0.003330f,0.001470f,0.002413f,0.006241f,0.004532f,0.000742f,0.002115f,0.001609f,0.000671f,0.003229f,0.001388f,0.006466f,0.006477f,0.001324f,0.001984f,0.000835f,0.006096f,0.005936f,0.001273f,0.002750f,0.003883f,0.001307f,0.005840f,0.000818f,0.005962f,0.007069f,0.004520f,0.001340f,0.001545f,0.004803f,
0.000856f,0.004673f,0.003868f,0.006016f,0.001283f,0.004730f,0.001266f,0.002642f,0.004784f,0.001471f,0.005741f,0.005695f,0.000682f,0.002132f,0.001083f,0.005066f,0.001437f,0.004055f,0.001119f,0.001844f,0.002075f,0.003284f,0.002970f,0.005596f,0.005371f,0.001755f,0.002789f,0.002636f,0.006977f,0.004551f,
0.001410f,0.006241f,0.003983f,0.005146f,0.001116f,0.001301f,0.002838f,0.000798f,0.006023f,0.003952f,0.001331f,0.005619f,0.000809f,0.004929f,0.005314f,0.007767f,0.001089f,0.007660f,0.008354f,0.000913f,0.008102f,0.007439f,0.004730f,0.002623f,0.008247f,0.001388f,0.006828f,0.001026f,0.002911f,0.009666f,
0.001910f,0.008530f,0.008751f,0.000770f,0.001651f,0.004128f,0.002029f,0.004147f,0.009438f,0.011276f,0.001335f,0.006763f,0.011368f,0.003620f,0.002003f,0.008430f,0.001573f,0.007797f,0.000785f,0.003136f,0.002670f,0.000983f,0.006977f,0.001826f,0.004059f,0.016342f,0.019363f,0.010971f,0.001266f,0.002813f,
0.000916f,0.003302f,0.015114f,0.017807f,0.010696f,0.001826f,0.003702f,0.013138f,0.002058f,0.017151f,0.012955f,0.001860f,0.001942f,0.011909f,0.001549f,0.003405f,0.014061f,0.001485f,0.012627f,0.011192f,0.014137f,0.001890f,0.011703f,0.001925f,0.011520f,0.012390f,0.013924f,0.001844f,0.005733f,0.001142f,
0.001073f,0.004757f,0.000335f,0.004162f,0.003574f,0.000588f,0.001501f,0.000156f,0.001350f,0.001108f,0.001931f,0.007629f,0.001017f,0.005241f,0.000361f,0.000184f,0.003910f,0.003687f,0.000150f,0.001167f,0.000810f,0.001314f,0.001530f,0.002365f,0.000082f,0.000088f,0.000090f,0.002628f,0.000256f,0.002062f,
0.000142f,0.002739f,0.000133f,0.002235f,0.000156f,0.002169f,0.000106f,0.000150f,0.002094f,0.000083f,0.000114f,0.000077f,0.001174f,0.000090f,0.001470f,0.000067f,0.001314f,0.001418f,0.000106f,0.000053f,0.001005f,0.000476f,0.000635f,0.000061f,0.001384f,0.000253f,0.002296f,0.000117f,0.000118f,0.002573f,
0.000222f,0.001606f,0.000095f,0.003145f,0.000121f,0.000067f,0.002836f,0.000086f,0.003517f,0.000069f,0.003750f,0.000054f,0.000076f,0.000102f,0.004047f,0.000133f,0.001388f,0.000989f,0.001593f,0.000285f,0.000270f,0.003119f,0.000060f,0.005558f,0.000090f,0.005733f,0.000103f,0.000060f,0.000086f,0.000065f,
0.008606f,0.000092f,0.004280f,0.000034f,0.007683f,0.000047f,0.007397f,0.006317f,0.000029f,0.000324f,0.007114f,0.003372f,0.000063f,0.002953f,0.000087f,0.003910f,0.000069f,0.000278f,0.000057f,0.003105f,0.000090f,0.000086f,0.003672f,0.000124f,0.003853f,0.000118f,0.001225f,0.000810f,0.001189f,0.000290f,
0.002146f,0.000065f,0.004463f,0.000306f,0.000058f,0.000093f,0.003990f,0.000038f,0.004940f,0.000120f,0.000084f,0.003757f,0.000046f,0.000044f,0.004398f,0.004417f,0.000076f,0.007084f,0.000097f,0.001369f,0.001255f,0.004047f,0.000075f,0.000071f,0.008675f,0.001900f,0.001433f,0.009789f,0.000434f,0.003258f,
0.000112f,0.000118f,0.011765f,0.002438f,0.005386f,0.000105f,0.000118f,0.003145f,0.002913f,0.000221f,0.006783f,0.000127f,0.007843f,0.000031f,0.000075f,0.000171f,0.006756f,0.003820f,0.000156f,0.004829f,0.000147f,0.000173f,0.000097f,0.006863f,0.000118f,0.007473f,0.005081f,0.000129f,0.008774f,0.000098f,
0.000083f,0.009346f,0.001232f,0.003805f,0.000092f,0.005119f,0.000361f,0.006199f,0.005939f,0.000857f,0.000411f,0.001490f,0.000160f,0.000311f,0.013275f,0.000642f,0.007629f,0.000110f,0.000069f,0.011574f,0.000042f,0.000106f,0.000078f,0.012238f,0.007629f,0.000092f,0.014084f,0.000591f,0.000243f,0.009872f,
0.000084f,0.010674f,0.001033f,0.000129f,0.000134f,0.012718f,0.000083f,0.000120f,0.015411f,0.000186f,0.000205f,0.000448f,0.008209f,0.000184f,0.009827f,0.000103f,0.011497f,0.002739f,0.000162f,0.008408f,0.000098f,0.000146f,0.014641f,0.006626f,0.000134f,0.007160f,0.008751f,0.000112f,0.000275f,0.016846f,
0.000472f,0.000136f,0.003067f,0.010391f,0.000187f,0.012184f,0.000183f,0.019943f,0.005688f,0.003605f,0.031219f,0.000792f,0.008812f,0.010590f,0.000078f,0.000161f,0.000406f,0.048553f,0.010750f,0.034302f,0.011131f,0.000077f,0.000159f,0.000408f,0.011810f,0.015900f,0.000469f,0.017456f,0.000303f,0.000524f,
0.023666f,0.015762f,0.000472f,0.020248f,0.008186f,0.000417f,0.020004f,0.000448f,0.000566f,0.000290f,0.020172f,0.000414f,0.017517f,0.000347f,0.000375f,0.000251f,0.012039f,0.004349f,0.001957f,0.000849f,0.000502f,0.003077f,0.000535f,0.000721f,0.001393f,0.001131f,0.004204f,0.001984f,0.001816f,0.001640f,
0.000570f,0.000658f,0.000495f,0.001196f,0.001437f,0.000546f,0.000687f,0.003460f,0.001746f,0.001655f,0.001571f,0.000913f,0.001007f,0.001666f,0.002953f,0.002666f,0.000702f,0.001207f,0.000969f,0.002508f,0.000648f,0.002659f,0.000676f,0.002003f,0.000653f,0.001730f,0.003313f,0.000665f,0.002277f,0.002720f,
0.002510f,0.000531f,0.001478f,0.000572f,0.002764f,0.000630f,0.000884f,0.001433f,0.002274f,0.000590f,0.001719f,0.001376f,0.001716f,0.001162f,0.001162f,0.000891f,0.003489f,0.004108f,0.000956f,0.001443f,0.001244f,0.002066f,0.000465f,0.003113f,0.001978f,0.000996f,0.001784f,0.001023f,0.001918f,0.000941f,
0.002474f,0.002914f,0.002863f,0.000618f,0.004349f,0.002157f,0.002056f,0.001925f,0.001098f,0.002136f,0.001492f,0.001281f,0.001062f,0.003536f,0.001223f,0.002489f,0.002708f,0.003155f,0.002684f,0.000884f,0.000760f,0.001060f,0.004799f,0.000800f,0.004349f,0.000764f,0.000961f,0.004238f,0.001725f,0.000590f,
0.000822f,0.001354f,0.000819f,0.003746f,0.000808f,0.002590f,0.002077f,0.004276f,0.000978f,0.002453f,0.004303f,0.000540f,0.003778f,0.000714f,0.004982f,0.002295f,0.002148f,0.002180f,0.002010f,0.001577f,0.003174f,0.000600f,0.003559f,0.003683f,0.004166f,0.001288f,0.004074f,0.001051f,0.002611f,0.004063f,
0.000990f,0.005192f,0.004536f,0.000946f,0.001663f,0.004154f,0.001238f,0.002926f,0.004276f,0.001773f,0.002031f,0.005768f,0.004204f,0.000691f,0.001484f,0.002268f,0.000613f,0.002708f,0.001615f,0.006611f,0.007904f,0.001220f,0.001585f,0.001001f,0.006729f,0.007820f,0.001944f,0.002825f,0.003431f,0.000764f,
0.006702f,0.001228f,0.008461f,0.008217f,0.004860f,0.001699f,0.001852f,0.004536f,0.001113f,0.004635f,0.003386f,0.005657f,0.001311f,0.004711f,0.001196f,0.002157f,0.004082f,0.001465f,0.004635f,0.004108f,0.000797f,0.003199f,0.001995f,0.004471f,0.002195f,0.003300f,0.001714f,0.002338f,0.003555f,0.002432f,
0.002752f,0.003672f,0.004673f,0.001792f,0.003042f,0.002550f,0.006695f,0.005409f,0.001803f,0.008949f,0.004528f,0.006577f,0.001580f,0.001626f,0.002432f,0.000663f,0.007423f,0.003763f,0.001608f,0.003828f,0.000739f,0.004974f,0.005062f,0.008659f,0.001597f,0.006348f,0.007526f,0.000879f,0.008476f,0.009666f,
0.004272f,0.002666f,0.007412f,0.001628f,0.005203f,0.000988f,0.003536f,0.008781f,0.002405f,0.007835f,0.009201f,0.001836f,0.002329f,0.003559f,0.002226f,0.003229f,0.006851f,0.009003f,0.000988f,0.005356f,0.008408f,0.004108f,0.002386f,0.007191f,0.002382f,0.008331f,0.000990f,0.002350f,0.004028f,0.001012f,
0.004906f,0.002474f,0.003887f,0.016922f,0.015961f,0.009758f,0.001483f,0.002886f,0.001382f,0.003105f,0.018692f,0.017731f,0.011208f,0.003286f,0.004993f,0.011627f,0.002577f,0.015587f,0.012383f,0.002953f,0.002949f,0.010826f,0.002207f,0.004784f,0.010269f,0.002062f,0.012970f,0.010185f,0.015076f,0.002010f,
0.014137f,0.002716f,0.011719f,0.012169f,0.014503f,0.002316f,0.001262f,0.000671f,0.000210f,0.000133f,0.000654f,0.000106f,0.000237f,0.001849f,0.000194f,0.001521f,0.000331f,0.000525f,0.000472f,0.000433f,0.000386f,0.000356f,0.000790f,0.000769f,0.000376f,0.000792f,0.003353f,0.000675f,0.000788f,0.000629f,
0.000662f,0.000744f,0.000771f,0.001384f,0.001350f,0.001521f,0.002254f,0.001064f,0.001832f,0.001611f,0.002327f,0.000878f,0.001457f,0.001354f,0.001489f,0.003544f,0.001123f,0.001767f,0.002594f,0.001815f,0.001118f,0.002029f,0.001445f,0.002188f,0.001132f,0.000692f,0.001798f,0.001691f,0.000389f,0.002644f,
0.000452f,0.001824f,0.000738f,0.001418f,0.001126f,0.003239f,0.005299f,0.000841f,0.002012f,0.000890f,0.001403f,0.000373f,0.001738f,0.001016f,0.000736f,0.001203f,0.000929f,0.000989f,0.000596f,0.001231f,0.001912f,0.001642f,0.000461f,0.005276f,0.000942f,0.001039f,0.000874f,0.001014f,0.002909f,0.000673f,
0.000992f,0.000859f,0.002851f,0.000751f,0.001530f,0.001215f,0.001834f,0.001301f,0.000937f,0.001199f,0.001154f,0.001862f,0.001030f,0.001863f,0.001073f,0.000782f,0.001136f,0.000995f,0.001728f,0.001534f,0.001215f,0.001501f,0.001517f,0.001422f,0.000966f,0.001163f,0.001399f,0.000569f,0.001648f,0.001982f,
0.000475f,0.003157f,0.001279f,0.005619f,0.001993f,0.002514f,0.002001f,0.003139f,0.001551f,0.002665f,0.000668f,0.003475f,0.002293f,0.002092f,0.001538f,0.002165f,0.001612f,0.003042f,0.002800f,0.001255f,0.002937f,0.002674f,0.001343f,0.001183f,0.003202f,0.001228f,0.002495f,0.002071f,0.002100f,0.001307f,
0.002638f,0.001863f,0.000803f,0.002081f,0.003223f,0.000947f,0.006466f,0.001262f,0.004730f,0.006962f,0.000962f,0.002327f,0.000366f,0.004345f,0.006989f,0.001002f,0.001691f,0.002882f,0.000947f,0.003315f,0.002552f,0.002356f,0.002871f,0.001735f,0.002575f,0.001711f,0.003517f,0.001655f,0.002216f,0.002573f,
0.002644f,0.003412f,0.002863f,0.003208f,0.003056f,0.003925f,0.002174f,0.003284f,0.002953f,0.001653f,0.006824f,0.001489f,0.003614f,0.001626f,0.003016f,0.001567f,0.000966f,0.004429f,0.002443f,0.002337f,0.002655f,0.002638f,0.001854f,0.004292f,0.002155f,0.004219f,0.001982f,0.002272f,0.002356f,0.001915f,
0.001912f,0.003239f,0.002407f,0.002398f,0.001580f,0.002396f,0.002188f,0.002459f,0.002787f,0.000874f,0.003252f,0.003181f,0.004940f,0.001616f,0.004002f,0.005844f,0.001525f,0.009087f,0.011894f,0.007736f,0.003397f,0.004623f,0.003658f,0.006336f,0.001485f,0.003630f,0.009979f,0.002407f,0.004528f,0.006824f,
0.002674f,0.003586f,0.004391f,0.004410f,0.004566f,0.006119f,0.014114f,0.002775f,0.005703f,0.006268f,0.013954f,0.003460f,0.005360f,0.004177f,0.004654f,0.002110f,0.007553f,0.013062f,0.004654f,0.025772f,0.005215f,0.006748f,0.009979f,0.011894f,0.008530f,0.006569f,0.021042f,0.004547f,0.022125f,0.010170f,
0.011436f,0.008270f,0.006805f,0.008942f,0.010132f,0.010086f,0.013741f,0.010292f,0.014748f,0.009819f,0.017303f,0.009743f,0.011620f,0.009048f,0.010658f,0.008331f,0.009369f,0.007355f,0.014969f,0.007385f,0.016312f,0.007385f,0.007385f,0.005619f,0.006443f,0.003550f,0.002253f,0.000434f,0.000265f,0.002821f,
0.000272f,0.000395f,0.001405f,0.000322f,0.003502f,0.001031f,0.001224f,0.001178f,0.000368f,0.000402f,0.000231f,0.001448f,0.002232f,0.000529f,0.000844f,0.004074f,0.001478f,0.001285f,0.001198f,0.000552f,0.000778f,0.002562f,0.004017f,0.003614f,0.000583f,0.001467f,0.001004f,0.004189f,0.000613f,0.003502f,
0.000578f,0.002771f,0.000665f,0.002668f,0.004463f,0.000570f,0.002644f,0.003006f,0.003122f,0.000610f,0.002117f,0.000495f,0.003761f,0.000463f,0.000926f,0.002066f,0.002977f,0.000879f,0.001966f,0.001274f,0.002514f,0.001428f,0.001478f,0.000854f,0.003134f,0.003677f,0.000815f,0.001459f,0.001168f,0.002504f,
0.000450f,0.002565f,0.001974f,0.000594f,0.002090f,0.000638f,0.002317f,0.000529f,0.002497f,0.002705f,0.002676f,0.000525f,0.003164f,0.001246f,0.001083f,0.001001f,0.000835f,0.001636f,0.000607f,0.000865f,0.000381f,0.002420f,0.000603f,0.001809f,0.002262f,0.002277f,0.002028f,0.000285f,0.000535f,0.000620f,
0.003815f,0.000481f,0.003252f,0.000493f,0.000889f,0.003399f,0.001408f,0.000489f,0.000730f,0.001364f,0.000789f,0.003696f,0.001124f,0.003122f,0.002083f,0.003832f,0.001049f,0.003002f,0.005573f,0.000951f,0.003914f,0.000427f,0.003693f,0.001553f,0.001654f,0.001754f,0.002010f,0.001636f,0.002798f,0.000671f,
0.003122f,0.003336f,0.003717f,0.001068f,0.003490f,0.000648f,0.002705f,0.004410f,0.000998f,0.004921f,0.003778f,0.000804f,0.001478f,0.003910f,0.001101f,0.003069f,0.002542f,0.001095f,0.001143f,0.006027f,0.004528f,0.000539f,0.001237f,0.002062f,0.000310f,0.002493f,0.000787f,0.006355f,0.007355f,0.000985f,
0.001532f,0.000991f,0.007118f,0.007694f,0.000934f,0.001670f,0.002937f,0.000587f,0.005981f,0.000665f,0.007244f,0.007099f,0.004089f,0.000954f,0.000865f,0.004311f,0.000660f,0.004234f,0.003315f,0.004757f,0.000730f,0.004684f,0.000783f,0.001553f,0.003693f,0.001078f,0.004410f,0.004036f,0.000753f,0.002184f,
0.000835f,0.004311f,0.001303f,0.003414f,0.001309f,0.001775f,0.002453f,0.002411f,0.002052f,0.004452f,0.005142f,0.001519f,0.002913f,0.002022f,0.007812f,0.006191f,0.001424f,0.008614f,0.005665f,0.006897f,0.001201f,0.001156f,0.003139f,0.000886f,0.006542f,0.003315f,0.001495f,0.004574f,0.001309f,0.005280f,
0.006504f,0.007896f,0.001127f,0.005867f,0.006245f,0.001086f,0.006062f,0.006905f,0.004482f,0.001641f,0.005165f,0.001040f,0.004898f,0.000868f,0.001951f,0.007576f,0.002016f,0.007935f,0.007812f,0.001512f,0.001795f,0.003126f,0.001846f,0.003078f,0.006699f,0.007084f,0.001295f,0.003815f,0.007736f,0.006302f,
0.002089f,0.008743f,0.001754f,0.009628f,0.000944f,0.002167f,0.004326f,0.000756f,0.005028f,0.001345f,0.002974f,0.017914f,0.017746f,0.011238f,0.001532f,0.002316f,0.000906f,0.002146f,0.018707f,0.018951f,0.012688f,0.001921f,0.003109f,0.011322f,0.001704f,0.018845f,0.014511f,0.002434f,0.002058f,0.012726f,
0.001846f,0.003744f,0.013206f,0.002045f,0.017227f,0.013367f,0.019791f,0.002176f,0.023270f,0.002911f,0.016342f,0.020294f,0.025665f,0.004005f,0.000128f,0.000194f,0.000047f,0.000004f,0.000239f,0.000004f,0.000004f,0.000362f,0.000007f,0.000196f,0.000016f,0.000036f,0.000029f,0.000013f,0.000026f,0.000001f,
0.000107f,0.000857f,0.000043f,0.000071f,0.000626f,0.000104f,0.000070f,0.000049f,0.000057f,0.000048f,0.000150f,0.000305f,0.000356f,0.000057f,0.000455f,0.000210f,0.000876f,0.000170f,0.000532f,0.000063f,0.000424f,0.000100f,0.000427f,0.001357f,0.000057f,0.000391f,0.000593f,0.000823f,0.000174f,0.001041f,
0.000137f,0.001550f,0.000083f,0.000212f,0.000876f,0.000786f,0.000203f,0.000792f,0.000852f,0.001562f,0.000571f,0.000488f,0.000243f,0.000703f,0.000932f,0.000169f,0.000721f,0.000472f,0.000574f,0.000125f,0.000467f,0.000325f,0.000061f,0.000343f,0.000066f,0.000186f,0.000029f,0.000219f,0.000292f,0.000377f,
0.000045f,0.000425f,0.000088f,0.000071f,0.000056f,0.000080f,0.000336f,0.000023f,0.000080f,0.000016f,0.000288f,0.000039f,0.000269f,0.000174f,0.000158f,0.000137f,0.000007f,0.000109f,0.000054f,0.000519f,0.000043f,0.000496f,0.000045f,0.000098f,0.000660f,0.000100f,0.000037f,0.000069f,0.000195f,0.000102f,
0.000395f,0.000225f,0.000255f,0.000240f,0.000694f,0.000126f,0.000602f,0.000773f,0.000062f,0.000890f,0.000010f,0.000251f,0.000070f,0.000169f,0.000204f,0.000700f,0.000419f,0.000588f,0.000117f,0.000448f,0.000472f,0.000500f,0.000149f,0.000440f,0.000114f,0.001704f,0.000803f,0.000325f,0.001266f,0.000528f,
0.000189f,0.000362f,0.001760f,0.000356f,0.001204f,0.000221f,0.000166f,0.000182f,0.002565f,0.001978f,0.000039f,0.000533f,0.000068f,0.000010f,0.001899f,0.000118f,0.002811f,0.003647f,0.000240f,0.000746f,0.000273f,0.004253f,0.004398f,0.000084f,0.000126f,0.000652f,0.000046f,0.001485f,0.000108f,0.001432f,
0.001225f,0.000405f,0.000096f,0.000147f,0.001225f,0.000057f,0.000583f,0.000843f,0.001173f,0.000094f,0.000751f,0.000109f,0.000230f,0.001566f,0.000143f,0.001800f,0.002857f,0.000053f,0.000398f,0.000085f,0.002222f,0.000190f,0.002199f,0.000455f,0.000201f,0.000551f,0.000616f,0.000188f,0.002066f,0.001061f,
0.000318f,0.001537f,0.000557f,0.002836f,0.001394f,0.000321f,0.002098f,0.000883f,0.001286f,0.000153f,0.000275f,0.001526f,0.000202f,0.000701f,0.000955f,0.000386f,0.004108f,0.000744f,0.001133f,0.001489f,0.001623f,0.000089f,0.001347f,0.001819f,0.000137f,0.002857f,0.003353f,0.003956f,0.000272f,0.000924f,
0.000269f,0.003656f,0.000132f,0.000386f,0.005283f,0.000832f,0.004665f,0.004349f,0.000476f,0.000700f,0.002123f,0.000932f,0.001809f,0.009232f,0.011673f,0.000919f,0.003204f,0.010918f,0.001369f,0.001814f,0.011482f,0.000857f,0.004665f,0.000109f,0.002613f,0.000413f,0.000127f,0.016129f,0.000798f,0.002333f,
0.027985f,0.027115f,0.011101f,0.001432f,0.002489f,0.000184f,0.002022f,0.028442f,0.028442f,0.013542f,0.002352f,0.002094f,0.011124f,0.001984f,0.029785f,0.016388f,0.003017f,0.003893f,0.031982f,0.002462f,0.005661f,0.045929f,0.003256f,0.033234f,0.045776f,0.062073f,0.004860f,0.038849f,0.005550f,0.073730f,
0.074036f,0.089233f,0.025970f,0.000499f,0.002090f,0.000660f,0.000094f,0.001633f,0.000110f,0.000081f,0.001645f,0.000140f,0.000886f,0.000295f,0.000364f,0.000440f,0.000176f,0.000895f,0.000082f,0.002115f,0.006233f,0.000321f,0.000268f,0.002199f,0.001058f,0.001208f,0.001505f,0.000488f,0.001018f,0.002506f,
0.004845f,0.004635f,0.000688f,0.002262f,0.001859f,0.005554f,0.001187f,0.005417f,0.001047f,0.006176f,0.001100f,0.003645f,0.009354f,0.000883f,0.004627f,0.005470f,0.006371f,0.001224f,0.005405f,0.000809f,0.005871f,0.000758f,0.001337f,0.004448f,0.003811f,0.000694f,0.002401f,0.000898f,0.006706f,0.000719f,
0.004951f,0.000554f,0.004734f,0.004333f,0.000452f,0.002323f,0.001511f,0.005791f,0.000448f,0.009842f,0.006725f,0.000415f,0.005630f,0.000401f,0.003666f,0.000319f,0.004662f,0.005070f,0.006981f,0.000268f,0.003164f,0.001123f,0.001219f,0.001673f,0.003489f,0.002859f,0.000782f,0.004154f,0.000323f,0.005089f,
0.000319f,0.004662f,0.003622f,0.003431f,0.002586f,0.000123f,0.003712f,0.000426f,0.005848f,0.000199f,0.006210f,0.000224f,0.000518f,0.004387f,0.005302f,0.000240f,0.000562f,0.001615f,0.000734f,0.003614f,0.000839f,0.002123f,0.000936f,0.005230f,0.001018f,0.003483f,0.003300f,0.000443f,0.003422f,0.000221f,
0.001837f,0.000761f,0.001139f,0.001589f,0.002094f,0.001675f,0.003956f,0.000738f,0.003778f,0.003540f,0.005798f,0.000571f,0.003254f,0.000333f,0.004246f,0.004604f,0.000536f,0.004688f,0.002840f,0.000539f,0.000788f,0.004021f,0.000726f,0.003031f,0.000901f,0.000711f,0.001033f,0.005001f,0.004944f,0.000387f,
0.002323f,0.000315f,0.000211f,0.005180f,0.000662f,0.004345f,0.003588f,0.001195f,0.002270f,0.000783f,0.006222f,0.004662f,0.000580f,0.001516f,0.003803f,0.000572f,0.007416f,0.000508f,0.006432f,0.007717f,0.004639f,0.000973f,0.000813f,0.006886f,0.000832f,0.004543f,0.005859f,0.005535f,0.000878f,0.003941f,
0.000825f,0.001822f,0.004379f,0.001176f,0.004475f,0.008537f,0.000836f,0.002243f,0.000913f,0.005089f,0.001269f,0.005928f,0.001219f,0.001036f,0.001101f,0.005768f,0.002668f,0.006470f,0.005344f,0.000709f,0.002743f,0.001691f,0.010338f,0.004856f,0.000922f,0.009087f,0.005260f,0.006874f,0.000946f,0.000769f,
0.003632f,0.000964f,0.004322f,0.009583f,0.001659f,0.007217f,0.002466f,0.007092f,0.005424f,0.007450f,0.001296f,0.006676f,0.007591f,0.000895f,0.008133f,0.006294f,0.005688f,0.003189f,0.007881f,0.001871f,0.005928f,0.001601f,0.005901f,0.008408f,0.002430f,0.008324f,0.008636f,0.000823f,0.001679f,0.004021f,
0.001770f,0.003571f,0.008255f,0.010544f,0.001595f,0.009270f,0.008797f,0.006809f,0.002674f,0.007591f,0.002455f,0.005260f,0.000864f,0.004196f,0.001240f,0.000535f,0.010483f,0.001927f,0.004932f,0.010376f,0.012939f,0.008896f,0.003269f,0.005283f,0.001312f,0.005180f,0.012062f,0.014046f,0.009178f,0.002693f,
0.004589f,0.007519f,0.003033f,0.009888f,0.007462f,0.003656f,0.003048f,0.009071f,0.003237f,0.004333f,0.008461f,0.003269f,0.006115f,0.005394f,0.005939f,0.003593f,0.003336f,0.003601f,0.003138f,0.002890f,0.003067f,0.008377f,0.000888f,0.000784f,0.000689f,0.000046f,0.001216f,0.000045f,0.000050f,0.003777f,
0.000130f,0.000723f,0.000048f,0.000090f,0.000059f,0.000231f,0.000590f,0.000035f,0.000284f,0.002205f,0.000035f,0.000041f,0.000603f,0.000052f,0.000124f,0.000090f,0.000156f,0.000124f,0.000450f,0.001065f,0.000857f,0.000108f,0.000618f,0.000136f,0.001539f,0.000226f,0.001040f,0.000080f,0.000490f,0.000120f,
0.000438f,0.003262f,0.000084f,0.000886f,0.001355f,0.001825f,0.000106f,0.001715f,0.000144f,0.004265f,0.000180f,0.000142f,0.002254f,0.000874f,0.000155f,0.000630f,0.000298f,0.003134f,0.000175f,0.001118f,0.000305f,0.002638f,0.002617f,0.000101f,0.000987f,0.000201f,0.002266f,0.000122f,0.002575f,0.002056f,
0.000246f,0.002096f,0.000266f,0.000757f,0.000171f,0.001115f,0.001599f,0.002514f,0.000167f,0.001427f,0.000140f,0.000263f,0.000189f,0.000745f,0.002466f,0.000260f,0.001656f,0.000414f,0.004230f,0.000236f,0.002024f,0.001478f,0.002096f,0.002159f,0.000222f,0.002653f,0.000220f,0.002348f,0.000214f,0.005074f,
0.000223f,0.000184f,0.005436f,0.000735f,0.000231f,0.000256f,0.001856f,0.000267f,0.002380f,0.000326f,0.000898f,0.000537f,0.002432f,0.000163f,0.002035f,0.001284f,0.000151f,0.003090f,0.000141f,0.001355f,0.000115f,0.000219f,0.000163f,0.001034f,0.000238f,0.002785f,0.000145f,0.001085f,0.002304f,0.002520f,
0.000254f,0.001206f,0.000224f,0.003206f,0.002298f,0.000155f,0.005367f,0.001133f,0.000147f,0.000167f,0.002972f,0.000141f,0.003059f,0.000147f,0.000336f,0.000331f,0.004978f,0.003149f,0.000229f,0.005318f,0.000110f,0.000496f,0.013153f,0.000608f,0.004807f,0.004364f,0.001491f,0.001811f,0.000422f,0.004295f,
0.004135f,0.000496f,0.000459f,0.002514f,0.000208f,0.001933f,0.000242f,0.001866f,0.003393f,0.000903f,0.000344f,0.000590f,0.002939f,0.000229f,0.001197f,0.002188f,0.001442f,0.000658f,0.001264f,0.000401f,0.000425f,0.003645f,0.000250f,0.003506f,0.007721f,0.000313f,0.001724f,0.000640f,0.008125f,0.000393f,
0.003607f,0.000603f,0.000250f,0.001805f,0.003843f,0.000586f,0.008270f,0.001323f,0.000173f,0.002079f,0.000338f,0.009331f,0.001715f,0.000306f,0.003029f,0.002329f,0.002726f,0.000433f,0.000819f,0.003004f,0.000504f,0.001003f,0.001959f,0.000590f,0.007412f,0.000474f,0.000946f,0.001917f,0.001922f,0.000347f,
0.003244f,0.004581f,0.000390f,0.008125f,0.007545f,0.005943f,0.000745f,0.002419f,0.001181f,0.011536f,0.000776f,0.001527f,0.012794f,0.000984f,0.005161f,0.007156f,0.001040f,0.000832f,0.005184f,0.000826f,0.001088f,0.018478f,0.047546f,0.000848f,0.009087f,0.045563f,0.003149f,0.001057f,0.012505f,0.001178f,
0.005356f,0.001051f,0.018906f,0.000541f,0.001664f,0.040039f,0.001616f,0.001277f,0.007576f,0.013153f,0.005661f,0.004372f,0.007851f,0.001767f,0.009132f,0.011978f,0.019669f,0.007950f,0.002316f,0.002136f,0.011887f,0.002666f,0.019974f,0.008064f,0.001724f,0.002956f,0.021271f,0.001345f,0.002218f,0.030457f,
0.001560f,0.016052f,0.021179f,0.015266f,0.005276f,0.020691f,0.004711f,0.038513f,0.032166f,0.033722f,0.006618f,0.001303f,0.004017f,0.000161f,0.000024f,0.002392f,0.000018f,0.000100f,0.006454f,0.000264f,0.005661f,0.000191f,0.000237f,0.000151f,0.000085f,0.000155f,0.000043f,0.000924f,0.003145f,0.000018f,
0.000080f,0.003819f,0.000125f,0.000199f,0.000141f,0.000165f,0.000240f,0.001703f,0.004360f,0.003117f,0.000151f,0.001594f,0.000178f,0.004395f,0.000242f,0.003868f,0.000145f,0.001610f,0.000165f,0.001543f,0.008720f,0.000162f,0.003960f,0.004143f,0.005226f,0.000167f,0.004169f,0.000037f,0.008682f,0.000049f,
0.000185f,0.003487f,0.003439f,0.000078f,0.001247f,0.000184f,0.005634f,0.000211f,0.001296f,0.000416f,0.006866f,0.010864f,0.000310f,0.001986f,0.000260f,0.002731f,0.000079f,0.003689f,0.002710f,0.000348f,0.002249f,0.000377f,0.001671f,0.000299f,0.003290f,0.003510f,0.003660f,0.000081f,0.004375f,0.000171f,
0.000259f,0.000159f,0.000498f,0.007004f,0.000377f,0.003214f,0.000864f,0.008751f,0.000481f,0.003466f,0.003201f,0.003960f,0.005325f,0.000156f,0.003433f,0.000473f,0.009995f,0.000098f,0.012314f,0.000092f,0.000289f,0.013977f,0.000730f,0.000086f,0.000429f,0.004509f,0.000390f,0.005486f,0.000329f,0.002508f,
0.002197f,0.006355f,0.000224f,0.004169f,0.004772f,0.000068f,0.007706f,0.000093f,0.007401f,0.000360f,0.000400f,0.000248f,0.002525f,0.000376f,0.006760f,0.000111f,0.002188f,0.005154f,0.003340f,0.000566f,0.005268f,0.000114f,0.005360f,0.004543f,0.000308f,0.012703f,0.006256f,0.000385f,0.000323f,0.007618f,
0.000220f,0.006168f,0.000660f,0.001019f,0.000566f,0.011772f,0.004696f,0.000122f,0.001733f,0.001645f,0.000154f,0.009613f,0.000429f,0.010803f,0.016434f,0.000169f,0.001292f,0.000051f,0.007141f,0.011909f,0.000400f,0.000875f,0.003578f,0.000209f,0.003466f,0.000095f,0.008400f,0.007572f,0.001657f,0.000348f,
0.000387f,0.005611f,0.000383f,0.002659f,0.003914f,0.005753f,0.000738f,0.002939f,0.000407f,0.000513f,0.008064f,0.000281f,0.008926f,0.013878f,0.000095f,0.002316f,0.000369f,0.013977f,0.000398f,0.002537f,0.000385f,0.000255f,0.003820f,0.001479f,0.000522f,0.006180f,0.001614f,0.000316f,0.003036f,0.000443f,
0.009666f,0.004299f,0.000329f,0.008415f,0.003073f,0.005135f,0.000301f,0.000354f,0.004417f,0.000115f,0.001727f,0.001402f,0.000374f,0.006866f,0.000054f,0.000618f,0.002792f,0.005226f,0.000319f,0.006779f,0.006931f,0.000128f,0.010513f,0.016724f,0.010918f,0.000588f,0.003246f,0.000816f,0.015686f,0.000252f,
0.000363f,0.014534f,0.000582f,0.005253f,0.006027f,0.000146f,0.000680f,0.007053f,0.000730f,0.000688f,0.015060f,0.023148f,0.000214f,0.004974f,0.019226f,0.002680f,0.000473f,0.005852f,0.000555f,0.004753f,0.000245f,0.003036f,0.002199f,0.000198f,0.013580f,0.000504f,0.000610f,0.012337f,0.010017f,0.002357f,
0.000147f,0.002022f,0.000174f,0.002804f,0.012535f,0.009613f,0.002266f,0.000397f,0.000699f,0.004509f,0.000789f,0.011909f,0.002598f,0.000323f,0.000411f,0.010941f,0.000440f,0.000463f,0.006855f,0.000418f,0.005966f,0.008667f,0.011169f,0.000677f,0.007572f,0.000577f,0.024063f,0.018127f,0.016022f,0.000350f,
0.000878f,0.000612f,0.000180f,0.000721f,0.000458f,0.001024f,0.000730f,0.000473f,0.000692f,0.000960f,0.000866f,0.000744f,0.000619f,0.000342f,0.000168f,0.000245f,0.000597f,0.000812f,0.002014f,0.001575f,0.002234f,0.002239f,0.002060f,0.002192f,0.001349f,0.001060f,0.000947f,0.001727f,0.001718f,0.002192f,
0.002140f,0.002422f,0.001521f,0.002365f,0.001583f,0.002537f,0.001357f,0.004364f,0.001005f,0.001452f,0.001775f,0.000902f,0.001828f,0.001286f,0.003075f,0.001384f,0.002390f,0.001271f,0.002350f,0.004009f,0.001741f,0.001516f,0.002319f,0.004688f,0.006001f,0.002745f,0.004040f,0.002796f,0.001343f,0.002396f,
0.004097f,0.001600f,0.003716f,0.003445f,0.003090f,0.001070f,0.005817f,0.003075f,0.002174f,0.003685f,0.002069f,0.001925f,0.000904f,0.001984f,0.003557f,0.002163f,0.000871f,0.003933f,0.003519f,0.003107f,0.003859f,0.002087f,0.002789f,0.001139f,0.001296f,0.000994f,0.002613f,0.001813f,0.002722f,0.001432f,
0.001925f,0.001147f,0.000285f,0.000740f,0.001571f,0.002234f,0.000653f,0.001833f,0.000766f,0.001775f,0.001654f,0.002642f,0.000653f,0.001957f,0.001053f,0.003578f,0.002613f,0.005577f,0.001347f,0.004063f,0.001951f,0.008514f,0.003075f,0.003281f,0.002699f,0.003344f,0.000978f,0.002865f,0.003399f,0.004131f,
0.004906f,0.003798f,0.004108f,0.002508f,0.002476f,0.004150f,0.001914f,0.002960f,0.002804f,0.002075f,0.001460f,0.002834f,0.002121f,0.001332f,0.001413f,0.001321f,0.001623f,0.001627f,0.001987f,0.002768f,0.001321f,0.003290f,0.003246f,0.002333f,0.002192f,0.001636f,0.001335f,0.001533f,0.002571f,0.000968f,
0.001873f,0.003889f,0.002481f,0.003416f,0.006058f,0.004852f,0.008575f,0.004242f,0.004608f,0.002384f,0.002607f,0.002882f,0.001732f,0.004288f,0.001672f,0.002607f,0.002707f,0.003151f,0.004665f,0.004162f,0.003025f,0.001987f,0.002020f,0.002272f,0.002150f,0.002384f,0.001488f,0.002371f,0.002357f,0.001789f,
0.003159f,0.001496f,0.001410f,0.001596f,0.003298f,0.003922f,0.001858f,0.009262f,0.003757f,0.005077f,0.001704f,0.002283f,0.001060f,0.002014f,0.001868f,0.003435f,0.002228f,0.003666f,0.003075f,0.004368f,0.003416f,0.004131f,0.005039f,0.003445f,0.004627f,0.004349f,0.003616f,0.002937f,0.001920f,0.004951f,
0.005264f,0.009094f,0.004604f,0.012100f,0.011581f,0.002960f,0.004459f,0.004730f,0.003281f,0.005928f,0.004120f,0.006001f,0.007332f,0.006535f,0.005444f,0.004444f,0.005817f,0.002436f,0.003254f,0.008148f,0.003967f,0.013428f,0.003578f,0.004921f,0.001813f,0.004063f,0.002087f,0.005215f,0.005444f,0.003201f,
0.003716f,0.002134f,0.003557f,0.003408f,0.008850f,0.014198f,0.004631f,0.009987f,0.002642f,0.001863f,0.002882f,0.008041f,0.001488f,0.005234f,0.007450f,0.008171f,0.004543f,0.004753f,0.005405f,0.005951f,0.004780f,0.002768f,0.005775f,0.006107f,0.006165f,0.006767f,0.006084f,0.004368f,0.005165f,0.005848f,
0.004753f,0.005398f,0.008369f,0.007195f,0.004131f,0.005669f,0.008644f,0.002455f,0.005543f,0.003635f,0.002943f,0.003201f,0.008202f,0.001789f,0.010666f,0.001695f,0.002037f,0.002628f,0.067139f,0.000012f,0.000010f,0.000002f,0.000000f,0.000035f,0.000000f,0.000000f,0.000020f,0.000001f,0.000025f,0.000006f,
0.000005f,0.000003f,0.000000f,0.000001f,0.000000f,0.000019f,0.000047f,0.000000f,0.000000f,0.000007f,0.000001f,0.000001f,0.000001f,0.000000f,0.000001f,0.000002f,0.000005f,0.000005f,0.000004f,0.000003f,0.000004f,0.000005f,0.000002f,0.000004f,0.000001f,0.000004f,0.000002f,0.000002f,0.000011f,0.000001f,
0.000009f,0.000018f,0.000023f,0.000006f,0.000012f,0.000003f,0.000030f,0.000004f,0.000006f,0.000027f,0.000030f,0.000012f,0.000033f,0.000009f,0.000023f,0.000004f,0.000006f,0.000006f,0.000026f,0.000056f,0.000028f,0.000021f,0.000020f,0.000012f,0.000002f,0.000022f,0.000009f,0.000007f,0.000007f,0.000006f,
0.000003f,0.000005f,0.000004f,0.000006f,0.000007f,0.000006f,0.000007f,0.000003f,0.000003f,0.000002f,0.000001f,0.000004f,0.000002f,0.000003f,0.000010f,0.000014f,0.000019f,0.000009f,0.000005f,0.000007f,0.000009f,0.000010f,0.000007f,0.000022f,0.000019f,0.000016f,0.000029f,0.000011f,0.000008f,0.000030f,
0.000011f,0.000006f,0.000012f,0.000022f,0.000020f,0.000035f,0.000015f,0.000014f,0.000015f,0.000033f,0.000006f,0.000014f,0.000012f,0.000001f,0.000009f,0.000001f,0.000009f,0.000013f,0.000015f,0.000009f,0.000006f,0.000005f,0.000007f,0.000001f,0.000009f,0.000006f,0.000010f,0.000008f,0.000009f,0.000015f,
0.000014f,0.000009f,0.000006f,0.000019f,0.000014f,0.000014f,0.000024f,0.000036f,0.000006f,0.000032f,0.000020f,0.000018f,0.000054f,0.000145f,0.000080f,0.000009f,0.000012f,0.000021f,0.000008f,0.000201f,0.000029f,0.000264f,0.000360f,0.000034f,0.000057f,0.000002f,0.000056f,0.000082f,0.000002f,0.000035f,
0.000074f,0.000030f,0.000125f,0.000012f,0.000083f,0.000114f,0.000036f,0.000027f,0.000013f,0.000054f,0.000008f,0.000023f,0.000049f,0.000066f,0.000049f,0.000031f,0.000031f,0.000053f,0.000082f,0.000024f,0.000095f,0.000128f,0.000009f,0.000074f,0.000032f,0.000337f,0.000100f,0.000182f,0.000060f,0.000041f,
0.000064f,0.000261f,0.000314f,0.000352f,0.000249f,0.000191f,0.000233f,0.000196f,0.000500f,0.000186f,0.000057f,0.000301f,0.000109f,0.000249f,0.000160f,0.000101f,0.000095f,0.000072f,0.000171f,0.000072f,0.000129f,0.000129f,0.000020f,0.000027f,0.000016f,0.000039f,0.000011f,0.000050f,0.000068f,0.000048f,
0.000053f,0.000064f,0.000059f,0.000036f,0.000068f,0.000040f,0.000103f,0.000050f,0.000104f,0.000222f,0.000121f,0.000104f,0.000114f,0.000039f,0.000069f,0.000181f,0.000198f,0.000312f,0.000652f,0.000489f,0.000095f,0.000735f,0.001298f,0.000482f,0.000199f,0.000969f,0.000066f,0.000528f,0.000111f,0.000283f,
0.000341f,0.000245f,0.001642f,0.000301f,0.000823f,0.001916f,0.001594f,0.000595f,0.000027f,0.000179f,0.000044f,0.000802f,0.003710f,0.002954f,0.001155f,0.000191f,0.001103f,0.003620f,0.001354f,0.005375f,0.002769f,0.001486f,0.000953f,0.009140f,0.002411f,0.004189f,0.017746f,0.003145f,0.022644f,0.039703f,
0.053406f,0.006641f,0.047546f,0.010353f,0.139648f,0.293213f,0.267090f,0.032654f,};

std::vector<ov::float16> input_data = {
11.039062f,10.445312f,8.703125f,7.878906f,7.464844f,7.671875f,7.660156f,9.062500f,9.062500f,7.792969f,8.148438f,6.984375f,7.730469f,8.234375f,8.539062f,7.925781f,8.203125f,9.171875f,8.140625f,8.226562f,8.046875f,8.476562f,7.347656f,8.195312f,9.039062f,8.812500f,6.449219f,6.699219f,7.722656f,
8.929688f,8.710938f,9.070312f,8.242188f,9.250000f,7.992188f,9.273438f,7.578125f,9.226562f,6.390625f,6.980469f,9.085938f,6.757812f,7.402344f,6.824219f,8.929688f,8.523438f,8.828125f,7.992188f,8.906250f,9.617188f,8.500000f,7.792969f,8.351562f,10.085938f,9.945312f,8.523438f,9.507812f,8.914062f,9.304688f,
7.554688f,8.562500f,9.414062f,8.898438f,9.203125f,6.816406f,8.210938f,8.851562f,7.562500f,9.296875f,7.859375f,9.390625f,6.683594f,9.218750f,6.929688f,7.582031f,8.117188f,8.367188f,8.648438f,9.140625f,7.875000f,8.656250f,9.445312f,9.828125f,9.132812f,8.414062f,9.460938f,7.582031f,9.398438f,8.695312f,
6.847656f,7.410156f,7.593750f,8.687500f,9.359375f,9.304688f,7.796875f,8.617188f,8.335938f,8.593750f,9.156250f,7.839844f,8.953125f,8.484375f,9.210938f,9.101562f,9.226562f,7.992188f,9.226562f,6.882812f,9.484375f,8.289062f,9.375000f,8.203125f,6.851562f,7.765625f,9.414062f,8.351562f,8.046875f,8.781250f,
7.660156f,8.617188f,8.585938f,9.445312f,7.101562f,8.460938f,8.593750f,7.445312f,8.484375f,9.664062f,7.605469f,9.085938f,9.429688f,6.886719f,9.218750f,7.562500f,7.496094f,9.515625f,9.968750f,6.929688f,9.773438f,7.941406f,9.164062f,9.468750f,10.007812f,7.230469f,7.359375f,8.531250f,9.898438f,9.257812f,
10.312500f,10.093750f,10.406250f,7.890625f,8.867188f,10.140625f,10.757812f,10.664062f,7.878906f,8.609375f,10.328125f,10.429688f,9.398438f,9.179688f,8.406250f,9.593750f,8.179688f,7.660156f,9.101562f,9.906250f,10.390625f,9.125000f,9.757812f,8.406250f,9.429688f,8.773438f,10.382812f,7.363281f,10.101562f,10.312500f,
9.273438f,10.531250f,9.195312f,9.632812f,9.007812f,10.335938f,10.781250f,9.265625f,10.812500f,9.796875f,10.601562f,10.265625f,10.062500f,9.406250f,9.507812f,8.039062f,9.039062f,10.406250f,9.437500f,10.281250f,8.085938f,8.398438f,10.351562f,8.523438f,7.761719f,8.226562f,9.867188f,10.468750f,9.468750f,9.375000f,
9.968750f,9.789062f,10.882812f,9.078125f,9.742188f,9.765625f,7.804688f,8.515625f,10.523438f,8.054688f,8.242188f,9.218750f,8.773438f,9.351562f,10.179688f,10.570312f,9.382812f,10.929688f,9.171875f,10.039062f,10.390625f,8.617188f,11.187500f,8.296875f,8.398438f,10.531250f,10.789062f,9.960938f,10.843750f,11.000000f,
9.023438f,8.328125f,10.140625f,10.265625f,9.257812f,11.515625f,11.273438f,8.882812f,11.203125f,8.250000f,9.671875f,11.210938f,10.328125f,11.695312f,11.218750f,11.656250f,11.476562f,9.406250f,8.835938f,9.953125f,11.007812f,11.617188f,12.242188f,11.828125f,9.632812f,9.085938f,10.117188f,11.937500f,11.867188f,10.734375f,
12.156250f,9.156250f,10.203125f,11.468750f,12.187500f,10.242188f,11.921875f,11.617188f,10.703125f,12.015625f,9.632812f,10.562500f,9.437500f,12.250000f,8.687500f,12.117188f,9.828125f,10.148438f,9.359375f,12.718750f,5.625000f,5.796875f,4.976562f,3.742188f,4.886719f,3.833984f,4.582031f,4.390625f,4.199219f,3.542969f,
3.962891f,4.675781f,4.847656f,5.332031f,5.691406f,4.425781f,4.714844f,4.566406f,4.386719f,5.367188f,4.792969f,4.664062f,5.250000f,5.296875f,5.617188f,5.515625f,5.371094f,5.601562f,5.914062f,6.097656f,5.878906f,5.761719f,6.765625f,5.867188f,6.296875f,5.687500f,7.539062f,6.183594f,6.238281f,7.132812f,
6.062500f,6.707031f,6.437500f,6.605469f,5.402344f,5.867188f,5.507812f,6.539062f,5.093750f,4.871094f,5.152344f,4.820312f,4.550781f,5.195312f,4.191406f,4.289062f,4.621094f,6.257812f,4.570312f,5.812500f,5.800781f,4.476562f,4.816406f,4.761719f,5.828125f,5.394531f,5.460938f,5.949219f,5.269531f,5.117188f,
5.984375f,5.488281f,5.402344f,6.218750f,6.175781f,7.425781f,5.695312f,6.070312f,5.347656f,5.605469f,5.640625f,6.828125f,5.507812f,5.519531f,6.738281f,4.964844f,5.503906f,5.230469f,5.550781f,5.585938f,5.632812f,5.507812f,5.058594f,5.023438f,5.531250f,6.132812f,4.796875f,6.574219f,4.824219f,5.000000f,
6.320312f,5.652344f,5.011719f,5.253906f,4.656250f,4.921875f,5.304688f,4.652344f,4.648438f,4.808594f,5.871094f,3.062500f,4.914062f,5.371094f,3.525391f,5.144531f,3.044922f,4.429688f,3.273438f,3.695312f,3.460938f,4.117188f,2.912109f,4.898438f,3.480469f,4.593750f,5.175781f,4.847656f,3.648438f,4.835938f,
3.333984f,4.867188f,5.289062f,3.931641f,6.664062f,6.140625f,4.359375f,4.402344f,6.605469f,5.339844f,6.675781f,6.734375f,7.058594f,5.652344f,7.140625f,7.773438f,6.695312f,8.406250f,8.000000f,7.207031f,7.773438f,6.636719f,7.902344f,7.851562f,6.500000f,6.328125f,6.253906f,7.535156f,7.753906f,6.230469f,
6.097656f,7.300781f,6.113281f,6.898438f,5.855469f,7.289062f,7.859375f,7.800781f,6.070312f,6.179688f,7.214844f,5.617188f,7.289062f,7.621094f,7.253906f,6.511719f,7.167969f,6.636719f,6.386719f,7.101562f,6.445312f,7.503906f,7.546875f,6.605469f,7.097656f,6.238281f,6.277344f,5.855469f,6.687500f,5.820312f,
5.527344f,6.578125f,7.015625f,6.789062f,7.171875f,7.617188f,6.046875f,6.765625f,5.902344f,7.851562f,7.726562f,6.949219f,8.046875f,7.656250f,8.421875f,7.242188f,7.511719f,7.714844f,7.707031f,8.578125f,8.554688f,7.468750f,8.382812f,7.171875f,8.882812f,9.148438f,9.937500f,7.484375f,9.664062f,9.351562f,
7.988281f,9.843750f,9.906250f,9.226562f,7.667969f,9.320312f,8.125000f,8.968750f,7.718750f,8.265625f,9.171875f,7.503906f,9.093750f,8.851562f,7.386719f,7.773438f,7.761719f,7.910156f,7.812500f,9.132812f,10.531250f,8.609375f,8.960938f,9.820312f,9.351562f,8.382812f,8.687500f,8.929688f,9.453125f,9.140625f,
10.765625f,10.531250f,10.109375f,10.132812f,9.296875f,8.929688f,10.757812f,11.429688f,11.023438f,10.304688f,10.179688f,10.671875f,10.078125f,10.687500f,11.171875f,11.031250f,10.343750f,10.085938f,10.851562f,10.585938f,11.601562f,11.687500f,10.890625f,11.210938f,11.906250f,11.265625f,11.164062f,11.554688f,11.765625f,12.468750f,
11.968750f,12.515625f,12.476562f,13.179688f,12.218750f,13.023438f,12.632812f,12.398438f,10.929688f,2.771484f,1.462891f,0.598145f,-0.208374f,1.766602f,0.018265f,0.545410f,2.181641f,2.751953f,2.072266f,1.876953f,1.875977f,1.919922f,2.074219f,4.625000f,5.308594f,8.203125f,6.886719f,3.863281f,3.011719f,2.720703f,
3.017578f,4.398438f,4.785156f,5.652344f,5.261719f,2.306641f,2.291016f,3.029297f,5.957031f,4.273438f,4.457031f,3.375000f,4.667969f,4.343750f,5.953125f,7.343750f,6.082031f,3.693359f,5.417969f,7.339844f,8.960938f,9.265625f,8.125000f,7.238281f,5.531250f,5.503906f,6.640625f,7.195312f,7.757812f,6.480469f,
3.734375f,3.224609f,3.275391f,2.404297f,0.289062f,1.258789f,2.277344f,1.824219f,1.748047f,3.431641f,4.773438f,3.849609f,4.492188f,2.746094f,5.605469f,6.488281f,7.062500f,8.476562f,5.273438f,7.007812f,3.546875f,7.535156f,6.941406f,7.441406f,8.000000f,5.933594f,4.371094f,3.746094f,3.894531f,3.400391f,
6.035156f,4.703125f,4.531250f,6.453125f,5.339844f,4.351562f,4.785156f,3.330078f,1.880859f,3.839844f,5.867188f,8.109375f,8.859375f,9.804688f,6.777344f,6.488281f,8.015625f,7.914062f,8.906250f,7.957031f,7.281250f,5.894531f,7.367188f,6.421875f,7.546875f,4.960938f,5.796875f,3.207031f,5.433594f,5.203125f,
3.341797f,2.062500f,0.307861f,-1.014648f,2.707031f,2.105469f,5.187500f,5.281250f,4.699219f,3.130859f,2.679688f,3.761719f,4.667969f,4.597656f,4.957031f,4.132812f,3.931641f,3.388672f,2.218750f,1.461914f,2.265625f,-0.371338f,-0.137329f,1.388672f,1.732422f,1.398438f,0.873047f,1.032227f,1.011719f,3.339844f,
5.742188f,7.546875f,6.445312f,5.074219f,4.480469f,3.398438f,5.660156f,7.457031f,8.367188f,10.046875f,9.250000f,7.496094f,7.398438f,6.800781f,5.457031f,3.958984f,3.724609f,3.859375f,3.990234f,5.035156f,6.242188f,2.998047f,2.677734f,2.011719f,3.429688f,4.945312f,7.222656f,7.257812f,5.937500f,5.460938f,
4.167969f,6.031250f,8.773438f,9.156250f,8.742188f,7.144531f,6.871094f,6.527344f,6.890625f,6.484375f,7.453125f,7.554688f,4.949219f,6.652344f,6.210938f,5.882812f,4.167969f,3.023438f,0.500000f,0.293945f,1.728516f,4.578125f,5.871094f,3.814453f,5.156250f,4.347656f,4.617188f,7.003906f,6.816406f,7.773438f,
8.687500f,6.792969f,6.265625f,7.417969f,9.992188f,10.140625f,8.406250f,9.015625f,8.671875f,9.101562f,9.835938f,6.589844f,4.976562f,6.175781f,3.734375f,7.242188f,8.117188f,9.085938f,8.500000f,5.859375f,5.484375f,6.234375f,8.804688f,10.203125f,11.875000f,11.492188f,10.851562f,9.890625f,11.242188f,10.140625f,
10.328125f,8.367188f,7.562500f,6.500000f,8.031250f,7.902344f,7.660156f,6.003906f,5.941406f,5.511719f,5.582031f,9.046875f,8.929688f,8.539062f,6.578125f,4.582031f,5.863281f,5.781250f,6.949219f,9.687500f,10.750000f,9.664062f,9.945312f,10.320312f,10.390625f,10.367188f,10.250000f,10.515625f,8.617188f,10.351562f,
10.070312f,10.062500f,9.656250f,7.578125f,7.210938f,7.800781f,8.718750f,9.859375f,9.828125f,7.453125f,8.343750f,9.062500f,10.210938f,12.234375f,13.226562f,13.625000f,12.875000f,13.148438f,13.421875f,14.328125f,15.343750f,16.250000f,15.312500f,17.515625f,17.843750f,17.750000f,16.593750f,15.187500f,4.203125f,4.351562f,
6.652344f,7.449219f,8.054688f,2.210938f,-0.302979f,1.694336f,5.304688f,9.585938f,6.996094f,3.185547f,-1.752930f,-3.103516f,0.598633f,5.046875f,11.875000f,11.515625f,6.281250f,3.771484f,3.677734f,3.785156f,5.320312f,5.441406f,5.378906f,3.503906f,1.890625f,5.160156f,9.484375f,12.046875f,10.625000f,6.457031f,
4.277344f,4.882812f,8.484375f,11.656250f,14.429688f,10.453125f,4.746094f,4.707031f,5.472656f,11.500000f,13.992188f,12.804688f,9.039062f,6.058594f,4.824219f,8.593750f,8.414062f,9.273438f,8.976562f,7.140625f,5.933594f,9.156250f,9.304688f,7.210938f,3.097656f,0.336670f,0.339355f,3.882812f,10.539062f,12.546875f,
11.101562f,6.589844f,3.292969f,4.437500f,8.968750f,11.492188f,10.585938f,7.117188f,5.613281f,3.845703f,7.042969f,10.195312f,10.367188f,10.156250f,7.285156f,6.511719f,9.414062f,10.773438f,8.578125f,6.937500f,4.593750f,2.804688f,8.406250f,11.375000f,13.539062f,10.898438f,5.261719f,-0.164917f,1.286133f,6.109375f,
9.898438f,12.132812f,11.507812f,8.890625f,6.945312f,10.593750f,9.914062f,10.226562f,8.328125f,6.476562f,5.027344f,8.882812f,11.546875f,12.710938f,9.546875f,7.250000f,4.304688f,9.367188f,13.742188f,12.343750f,10.757812f,4.789062f,-1.913086f,2.464844f,2.394531f,9.835938f,10.632812f,9.039062f,5.296875f,4.757812f,
4.980469f,7.343750f,5.824219f,5.468750f,4.121094f,4.585938f,6.113281f,9.414062f,7.363281f,6.425781f,0.515137f,-0.647949f,2.617188f,7.296875f,9.929688f,9.421875f,6.160156f,2.583984f,4.484375f,7.453125f,11.960938f,11.781250f,9.515625f,5.937500f,2.343750f,4.566406f,7.718750f,8.859375f,10.007812f,8.601562f,
7.949219f,11.375000f,12.093750f,11.890625f,7.511719f,2.640625f,0.248657f,2.457031f,8.187500f,12.875000f,11.632812f,7.773438f,4.300781f,3.769531f,6.644531f,11.054688f,12.531250f,10.507812f,7.968750f,5.175781f,7.750000f,11.015625f,12.445312f,10.625000f,7.152344f,7.402344f,9.078125f,11.203125f,11.734375f,10.406250f,
7.007812f,3.156250f,6.718750f,10.429688f,15.046875f,13.132812f,8.937500f,1.840820f,-1.144531f,0.924805f,6.812500f,10.109375f,8.132812f,7.632812f,5.468750f,6.621094f,8.562500f,9.523438f,8.171875f,6.171875f,4.359375f,5.218750f,9.140625f,10.960938f,10.492188f,6.156250f,5.109375f,6.851562f,11.492188f,14.921875f,
13.859375f,9.390625f,6.187500f,1.543945f,6.703125f,9.148438f,14.460938f,13.703125f,8.273438f,6.605469f,6.328125f,9.164062f,9.789062f,10.601562f,8.734375f,8.179688f,9.062500f,14.070312f,15.062500f,12.945312f,7.449219f,3.488281f,2.087891f,6.558594f,11.093750f,12.312500f,9.781250f,6.121094f,3.406250f,6.152344f,
12.132812f,14.812500f,14.054688f,10.039062f,5.742188f,5.464844f,6.160156f,6.496094f,9.296875f,8.046875f,7.195312f,9.078125f,12.179688f,13.929688f,12.773438f,8.039062f,5.457031f,2.640625f,7.804688f,10.828125f,15.085938f,12.937500f,7.128906f,4.496094f,5.085938f,8.328125f,11.953125f,12.632812f,7.851562f,8.156250f,
7.859375f,10.796875f,13.953125f,14.101562f,12.406250f,11.007812f,10.945312f,13.062500f,15.140625f,14.671875f,12.914062f,7.183594f,11.359375f,15.015625f,20.078125f,21.921875f,19.531250f,10.343750f,10.679688f,8.554688f,9.437500f,8.437500f,9.484375f,9.273438f,7.785156f,10.085938f,6.582031f,8.320312f,10.015625f,8.054688f,
11.250000f,8.640625f,9.476562f,9.164062f,7.476562f,9.648438f,9.492188f,6.859375f,8.429688f,10.054688f,8.031250f,10.093750f,7.875000f,7.679688f,7.687500f,7.527344f,10.007812f,8.960938f,8.179688f,8.343750f,10.335938f,7.691406f,10.148438f,10.640625f,10.585938f,8.007812f,7.832031f,10.250000f,7.761719f,8.664062f,
7.957031f,9.929688f,8.875000f,10.406250f,7.929688f,10.226562f,10.312500f,8.703125f,8.078125f,9.656250f,8.500000f,10.523438f,8.046875f,10.929688f,10.031250f,10.187500f,7.125000f,6.992188f,10.132812f,9.210938f,8.382812f,8.507812f,9.882812f,8.140625f,8.453125f,10.632812f,8.718750f,11.093750f,8.062500f,10.250000f,
7.664062f,8.570312f,9.351562f,9.812500f,7.464844f,9.101562f,10.640625f,8.718750f,10.304688f,8.421875f,8.523438f,11.578125f,10.609375f,8.539062f,10.960938f,8.937500f,7.968750f,8.781250f,8.484375f,10.429688f,8.718750f,11.218750f,8.718750f,10.468750f,8.664062f,10.375000f,10.593750f,8.226562f,10.671875f,10.531250f,
11.242188f,7.718750f,11.210938f,8.125000f,11.085938f,8.398438f,9.429688f,8.570312f,10.656250f,8.671875f,7.984375f,9.789062f,8.359375f,10.093750f,7.281250f,9.234375f,10.898438f,9.070312f,9.539062f,8.710938f,7.675781f,10.023438f,8.664062f,8.195312f,8.796875f,10.765625f,8.203125f,10.125000f,8.515625f,7.695312f,
9.664062f,7.839844f,8.234375f,9.742188f,8.906250f,7.636719f,10.593750f,8.968750f,9.687500f,12.070312f,8.921875f,7.257812f,9.070312f,10.382812f,8.632812f,8.195312f,11.117188f,9.203125f,11.351562f,8.054688f,7.968750f,12.656250f,8.828125f,10.937500f,7.914062f,7.789062f,11.015625f,9.343750f,10.429688f,10.414062f,
9.117188f,10.726562f,8.445312f,8.757812f,10.781250f,10.828125f,11.257812f,9.531250f,10.242188f,9.500000f,10.273438f,9.140625f,11.062500f,8.929688f,11.257812f,9.171875f,8.695312f,11.070312f,9.195312f,9.500000f,10.437500f,9.351562f,11.406250f,9.179688f,11.898438f,10.945312f,11.171875f,9.429688f,8.687500f,9.742188f,
11.289062f,9.437500f,10.820312f,11.171875f,10.031250f,9.148438f,8.921875f,9.062500f,11.468750f,8.570312f,9.078125f,8.671875f,10.984375f,11.531250f,8.343750f,10.992188f,10.156250f,11.742188f,11.757812f,8.593750f,10.992188f,12.835938f,9.226562f,9.898438f,11.351562f,9.007812f,9.968750f,10.929688f,8.781250f,8.648438f,
9.898438f,9.726562f,11.500000f,12.046875f,10.000000f,11.531250f,12.414062f,9.867188f,12.132812f,9.265625f,10.023438f,11.476562f,11.898438f,8.734375f,11.914062f,9.742188f,9.304688f,9.992188f,11.460938f,10.718750f,10.062500f,12.960938f,12.382812f,10.437500f,12.679688f,9.820312f,11.226562f,9.546875f,9.312500f,12.484375f,
10.156250f,12.554688f,10.257812f,9.671875f,9.867188f,11.695312f,13.539062f,9.851562f,12.531250f,9.937500f,9.820312f,9.976562f,11.781250f,12.773438f,10.500000f,12.085938f,12.875000f,9.296875f,11.984375f,12.406250f,13.031250f,10.906250f,12.421875f,10.828125f,11.265625f,12.757812f,11.453125f,11.601562f,11.101562f,13.570312f,
11.039062f,14.398438f,10.812500f,11.171875f,11.414062f,13.890625f,5.789062f,4.492188f,3.701172f,2.707031f,4.507812f,2.515625f,2.160156f,5.031250f,3.970703f,4.640625f,3.802734f,4.687500f,3.986328f,3.197266f,3.802734f,2.992188f,4.632812f,5.761719f,3.121094f,2.857422f,5.027344f,3.906250f,4.851562f,4.218750f,
4.386719f,5.363281f,4.085938f,4.890625f,5.468750f,6.019531f,5.761719f,5.671875f,5.500000f,4.503906f,5.449219f,4.437500f,6.050781f,5.128906f,4.445312f,5.812500f,4.359375f,5.308594f,5.742188f,5.937500f,4.464844f,5.597656f,3.662109f,5.921875f,3.582031f,4.273438f,5.250000f,4.710938f,3.861328f,5.625000f,
4.488281f,5.113281f,4.617188f,5.281250f,4.277344f,5.207031f,5.757812f,4.613281f,5.640625f,5.804688f,5.015625f,3.535156f,5.984375f,5.476562f,5.296875f,5.371094f,5.390625f,4.578125f,4.781250f,5.457031f,5.839844f,6.050781f,4.335938f,5.710938f,4.675781f,5.449219f,4.839844f,5.328125f,5.675781f,5.593750f,
5.574219f,4.851562f,5.855469f,5.421875f,5.812500f,5.085938f,5.515625f,5.359375f,4.164062f,5.500000f,5.292969f,6.292969f,4.132812f,6.394531f,4.191406f,4.992188f,6.023438f,5.882812f,4.050781f,5.066406f,5.812500f,5.066406f,5.835938f,5.105469f,4.441406f,5.320312f,5.671875f,4.308594f,5.359375f,5.441406f,
3.474609f,5.781250f,3.154297f,5.546875f,4.558594f,5.312500f,4.746094f,5.421875f,5.464844f,5.484375f,3.244141f,5.398438f,5.496094f,5.562500f,4.996094f,5.449219f,3.876953f,5.882812f,5.371094f,4.289062f,6.160156f,5.652344f,4.710938f,6.183594f,5.960938f,4.808594f,5.859375f,6.593750f,5.902344f,6.476562f,
5.691406f,6.082031f,4.851562f,5.773438f,5.976562f,4.765625f,7.273438f,5.597656f,6.425781f,6.816406f,4.839844f,6.554688f,4.472656f,5.968750f,6.167969f,5.117188f,6.593750f,6.722656f,4.347656f,5.738281f,4.476562f,6.457031f,6.542969f,6.539062f,6.691406f,5.621094f,6.984375f,5.089844f,6.261719f,6.937500f,
7.058594f,5.785156f,5.652344f,5.656250f,6.765625f,6.609375f,5.601562f,6.968750f,6.949219f,4.812500f,6.652344f,5.414062f,6.628906f,5.718750f,6.257812f,5.238281f,6.218750f,5.578125f,5.839844f,5.554688f,6.152344f,6.542969f,5.500000f,6.699219f,6.703125f,6.656250f,6.445312f,6.148438f,6.980469f,6.042969f,
6.714844f,7.269531f,6.265625f,7.089844f,4.820312f,6.546875f,7.382812f,6.609375f,6.941406f,4.710938f,6.687500f,6.136719f,7.304688f,5.730469f,7.101562f,7.472656f,5.679688f,7.375000f,7.679688f,7.781250f,7.281250f,7.312500f,6.386719f,7.542969f,5.082031f,6.562500f,7.882812f,6.839844f,7.136719f,7.273438f,
5.304688f,6.152344f,7.250000f,6.441406f,7.621094f,7.750000f,7.429688f,5.167969f,7.417969f,7.761719f,7.761719f,6.964844f,7.339844f,6.910156f,6.847656f,6.160156f,7.261719f,7.421875f,6.207031f,8.578125f,7.101562f,8.078125f,8.445312f,8.218750f,7.707031f,5.843750f,7.750000f,6.332031f,8.093750f,8.468750f,
8.257812f,7.839844f,7.207031f,8.242188f,8.414062f,7.730469f,8.554688f,8.328125f,8.593750f,7.816406f,9.234375f,7.835938f,8.695312f,8.367188f,8.007812f,8.921875f,8.890625f,9.585938f,8.562500f,8.570312f,8.039062f,9.156250f,9.367188f,9.289062f,8.367188f,10.664062f,10.437500f,9.484375f,7.117188f,7.191406f,
6.980469f,6.492188f,9.960938f,8.257812f,8.945312f,8.742188f,7.375000f,8.585938f,7.285156f,9.382812f,7.121094f,7.761719f,7.976562f,7.117188f,6.648438f,8.898438f,8.867188f,7.644531f,8.945312f,8.390625f,9.156250f,5.542969f,6.804688f,7.050781f,8.468750f,7.292969f,9.328125f,7.277344f,7.972656f,6.808594f,
8.070312f,7.218750f,8.609375f,5.539062f,7.468750f,7.925781f,6.996094f,6.375000f,7.031250f,7.410156f,6.949219f,7.742188f,7.242188f,7.839844f,8.093750f,6.988281f,6.808594f,7.179688f,10.468750f,8.695312f,6.710938f,8.265625f,10.054688f,7.941406f,8.156250f,9.062500f,8.226562f,7.101562f,9.421875f,5.925781f,
7.281250f,7.613281f,6.296875f,8.812500f,6.167969f,8.726562f,5.546875f,8.328125f,6.957031f,6.257812f,7.648438f,7.304688f,9.117188f,9.468750f,8.187500f,9.429688f,9.500000f,7.640625f,9.500000f,8.406250f,8.414062f,6.886719f,8.921875f,7.882812f,6.718750f,6.062500f,5.972656f,8.015625f,8.296875f,7.933594f,
6.941406f,7.941406f,7.347656f,8.007812f,8.296875f,7.082031f,8.742188f,7.988281f,8.046875f,8.664062f,8.164062f,6.917969f,8.945312f,5.742188f,9.445312f,7.128906f,8.406250f,7.089844f,6.566406f,6.972656f,8.406250f,7.847656f,9.117188f,9.281250f,7.960938f,9.210938f,6.890625f,9.375000f,7.242188f,7.300781f,
7.011719f,6.433594f,7.175781f,8.851562f,7.031250f,8.117188f,7.445312f,6.156250f,7.679688f,7.167969f,7.101562f,7.921875f,9.375000f,7.410156f,8.328125f,7.070312f,9.960938f,8.609375f,9.734375f,6.613281f,6.578125f,7.464844f,8.804688f,9.789062f,9.226562f,10.546875f,9.054688f,8.687500f,9.601562f,8.265625f,
10.945312f,9.187500f,8.531250f,9.312500f,8.953125f,10.484375f,10.054688f,7.937500f,7.695312f,8.414062f,7.199219f,7.542969f,9.382812f,8.765625f,9.054688f,7.988281f,7.988281f,6.980469f,10.062500f,7.128906f,8.656250f,6.070312f,8.375000f,9.789062f,8.554688f,8.710938f,7.480469f,8.062500f,7.656250f,10.648438f,
9.132812f,7.554688f,8.945312f,10.375000f,8.523438f,9.414062f,10.554688f,8.843750f,8.843750f,6.675781f,9.218750f,8.515625f,7.433594f,9.757812f,7.644531f,7.421875f,9.023438f,7.207031f,6.253906f,7.253906f,8.835938f,9.242188f,8.882812f,8.117188f,9.460938f,8.781250f,9.609375f,7.925781f,8.179688f,9.125000f,
6.390625f,7.121094f,8.953125f,7.898438f,7.027344f,8.039062f,9.078125f,9.890625f,8.359375f,10.015625f,9.476562f,8.882812f,7.527344f,8.117188f,9.414062f,7.750000f,9.437500f,7.726562f,6.792969f,8.625000f,8.273438f,8.953125f,8.406250f,10.070312f,7.894531f,8.351562f,8.195312f,9.203125f,7.785156f,9.781250f,
9.460938f,7.046875f,9.351562f,6.695312f,8.242188f,9.398438f,10.343750f,9.671875f,10.765625f,9.460938f,10.218750f,7.898438f,8.117188f,9.539062f,8.257812f,11.125000f,9.804688f,11.320312f,8.070312f,8.281250f,9.664062f,9.484375f,10.304688f,9.867188f,9.390625f,7.941406f,9.750000f,9.125000f,9.585938f,8.875000f,
8.828125f,10.226562f,9.804688f,8.773438f,7.917969f,10.367188f,8.007812f,9.429688f,7.300781f,9.023438f,9.015625f,8.984375f,7.882812f,9.328125f,6.277344f,6.406250f,3.976562f,3.884766f,4.539062f,4.332031f,3.710938f,6.093750f,5.832031f,3.490234f,4.769531f,5.550781f,5.324219f,5.734375f,4.777344f,4.046875f,
5.003906f,5.960938f,3.847656f,3.578125f,3.974609f,5.179688f,5.855469f,5.449219f,6.636719f,6.878906f,4.257812f,4.046875f,5.109375f,7.035156f,5.746094f,6.414062f,6.015625f,6.179688f,5.531250f,5.746094f,6.105469f,6.691406f,4.730469f,5.218750f,6.359375f,4.773438f,5.214844f,4.445312f,5.878906f,6.257812f,
5.398438f,5.550781f,5.421875f,6.453125f,5.671875f,4.683594f,5.386719f,6.839844f,6.894531f,6.109375f,7.648438f,7.890625f,7.390625f,4.468750f,4.804688f,6.496094f,6.371094f,7.554688f,4.746094f,6.011719f,6.613281f,5.101562f,7.402344f,6.195312f,7.437500f,4.429688f,6.441406f,4.621094f,5.246094f,4.949219f,
4.667969f,4.742188f,5.445312f,5.914062f,5.808594f,7.679688f,7.972656f,7.441406f,7.257812f,6.585938f,5.355469f,6.902344f,7.160156f,5.750000f,6.335938f,6.253906f,5.343750f,6.699219f,6.769531f,5.828125f,5.433594f,5.929688f,5.488281f,6.515625f,6.093750f,7.375000f,5.605469f,7.085938f,6.761719f,6.875000f,
5.484375f,6.929688f,4.011719f,6.250000f,4.695312f,5.917969f,5.535156f,4.515625f,5.648438f,7.156250f,4.941406f,4.511719f,5.070312f,5.261719f,4.953125f,5.933594f,6.476562f,4.863281f,4.910156f,5.785156f,5.113281f,6.187500f,6.101562f,4.800781f,4.222656f,6.507812f,4.769531f,5.332031f,5.695312f,5.210938f,
5.296875f,6.343750f,5.277344f,6.601562f,6.695312f,7.804688f,7.449219f,7.351562f,4.617188f,5.917969f,6.804688f,6.671875f,5.476562f,6.660156f,7.324219f,7.085938f,4.082031f,4.691406f,6.515625f,6.777344f,6.582031f,5.382812f,6.343750f,7.855469f,8.031250f,7.781250f,5.515625f,5.933594f,5.843750f,6.808594f,
5.695312f,7.234375f,8.062500f,6.968750f,7.386719f,6.484375f,7.046875f,8.000000f,7.910156f,6.636719f,5.414062f,6.386719f,6.707031f,7.175781f,6.394531f,7.882812f,8.148438f,6.636719f,7.734375f,7.035156f,6.761719f,7.183594f,7.710938f,7.921875f,8.273438f,7.167969f,7.207031f,7.671875f,6.164062f,7.636719f,
7.585938f,7.898438f,8.554688f,6.222656f,6.925781f,8.828125f,7.726562f,6.589844f,6.921875f,9.390625f,8.539062f,8.179688f,6.660156f,7.437500f,8.101562f,7.863281f,6.914062f,6.457031f,9.070312f,7.246094f,8.335938f,8.164062f,7.222656f,7.753906f,7.125000f,7.078125f,8.015625f,9.515625f,8.640625f,8.710938f,
8.523438f,8.625000f,6.445312f,7.847656f,7.613281f,8.273438f,7.628906f,8.304688f,7.039062f,8.039062f,8.875000f,7.964844f,8.695312f,9.070312f,7.199219f,7.210938f,9.250000f,7.710938f,9.609375f,9.281250f,8.976562f,9.796875f,8.281250f,8.609375f,9.046875f,8.359375f,8.390625f,9.593750f,8.742188f,8.945312f,
9.109375f,8.648438f,9.812500f,9.523438f,9.453125f,9.164062f,9.210938f,9.640625f,9.101562f,10.226562f,10.015625f,10.117188f,10.265625f,9.960938f,9.296875f,10.468750f,11.414062f,10.500000f,10.695312f,10.093750f,10.804688f,10.929688f,10.539062f,10.781250f,11.289062f,11.484375f,11.578125f,10.390625f,11.210938f,9.960938f,
9.710938f,9.429688f,11.226562f,15.304688f,17.031250f,12.625000f,12.210938f,11.562500f,12.210938f,12.726562f,12.171875f,13.539062f,10.812500f,11.500000f,11.546875f,11.179688f,13.773438f,12.578125f,12.179688f,11.382812f,11.359375f,12.109375f,12.656250f,10.804688f,11.531250f,11.593750f,11.179688f,13.335938f,12.500000f,11.453125f,
11.507812f,11.179688f,12.281250f,11.929688f,12.453125f,11.664062f,12.765625f,11.304688f,12.781250f,12.171875f,13.031250f,11.500000f,11.851562f,12.867188f,11.226562f,11.429688f,11.375000f,12.429688f,11.539062f,13.242188f,11.578125f,13.257812f,12.765625f,11.500000f,11.421875f,12.929688f,11.804688f,13.304688f,11.117188f,13.187500f,
13.070312f,12.742188f,11.546875f,11.359375f,12.664062f,11.960938f,12.406250f,11.679688f,12.406250f,11.734375f,11.539062f,12.992188f,11.406250f,13.476562f,11.687500f,12.765625f,11.328125f,11.531250f,13.054688f,12.796875f,10.960938f,11.351562f,11.406250f,10.968750f,12.914062f,12.273438f,12.382812f,12.390625f,12.750000f,11.914062f,
12.875000f,12.015625f,11.359375f,11.617188f,11.804688f,13.101562f,11.171875f,12.164062f,12.062500f,13.195312f,12.718750f,13.234375f,12.679688f,12.203125f,12.414062f,13.140625f,12.234375f,11.031250f,12.203125f,11.710938f,12.890625f,11.859375f,12.187500f,12.281250f,12.656250f,11.789062f,12.046875f,12.898438f,12.203125f,13.234375f,
11.328125f,11.398438f,11.421875f,11.039062f,12.265625f,12.453125f,11.585938f,12.421875f,12.195312f,11.890625f,12.007812f,13.031250f,11.992188f,13.296875f,12.156250f,11.914062f,12.570312f,12.109375f,11.929688f,12.539062f,13.320312f,11.507812f,12.695312f,11.984375f,12.187500f,12.703125f,12.484375f,11.492188f,11.929688f,13.015625f,
13.468750f,11.687500f,13.726562f,11.820312f,13.085938f,12.054688f,11.937500f,13.375000f,11.726562f,13.695312f,12.093750f,11.984375f,13.101562f,12.703125f,12.054688f,12.304688f,12.343750f,13.140625f,12.312500f,12.351562f,12.570312f,11.859375f,12.960938f,12.468750f,12.531250f,12.312500f,12.398438f,12.140625f,12.789062f,12.179688f,
12.570312f,12.273438f,11.859375f,12.507812f,12.296875f,12.093750f,12.937500f,11.468750f,13.000000f,11.750000f,12.945312f,12.531250f,12.726562f,13.281250f,11.718750f,11.953125f,11.742188f,12.484375f,12.375000f,12.367188f,12.414062f,12.218750f,12.625000f,12.343750f,12.695312f,12.492188f,12.304688f,12.351562f,11.726562f,12.843750f,
11.195312f,12.257812f,11.664062f,12.445312f,12.632812f,11.046875f,12.187500f,12.617188f,12.257812f,12.656250f,12.375000f,11.984375f,12.132812f,12.773438f,12.375000f,12.250000f,12.703125f,12.109375f,11.875000f,12.531250f,11.671875f,12.257812f,12.117188f,12.593750f,12.640625f,11.976562f,12.179688f,13.039062f,11.804688f,11.132812f,
11.695312f,11.968750f,12.265625f,12.539062f,12.210938f,12.320312f,12.593750f,13.953125f,12.500000f,11.812500f,13.078125f,12.312500f,12.750000f,13.382812f,11.562500f,13.351562f,11.484375f,12.585938f,11.898438f,12.453125f,12.460938f,12.148438f,12.742188f,11.203125f,13.343750f,11.156250f,12.445312f,12.468750f,12.117188f,12.500000f,
11.804688f,11.929688f,12.226562f,12.171875f,11.937500f,11.171875f,12.414062f,12.351562f,12.031250f,10.671875f,11.445312f,12.023438f,12.093750f,11.671875f,11.890625f,12.218750f,12.070312f,11.484375f,11.968750f,11.968750f,12.101562f,11.929688f,7.761719f,7.253906f,3.812500f,3.197266f,7.234375f,3.623047f,3.390625f,5.828125f,
6.066406f,6.382812f,6.402344f,7.417969f,5.402344f,5.996094f,5.128906f,5.316406f,8.773438f,11.101562f,8.468750f,7.253906f,7.750000f,5.476562f,5.578125f,4.648438f,6.960938f,6.484375f,6.300781f,6.843750f,9.570312f,10.414062f,11.984375f,9.906250f,10.226562f,8.085938f,8.226562f,8.140625f,10.765625f,10.171875f,
7.886719f,7.503906f,7.292969f,7.808594f,10.742188f,11.000000f,10.664062f,11.000000f,7.605469f,9.460938f,6.773438f,8.070312f,10.273438f,8.992188f,8.484375f,8.695312f,9.726562f,9.062500f,9.054688f,7.546875f,6.664062f,3.822266f,6.546875f,10.023438f,11.562500f,10.531250f,8.843750f,6.843750f,9.250000f,9.796875f,
10.906250f,11.148438f,11.031250f,7.867188f,9.656250f,8.578125f,10.726562f,11.437500f,10.343750f,8.250000f,7.343750f,7.718750f,6.652344f,7.902344f,7.953125f,5.492188f,9.109375f,8.734375f,10.046875f,11.835938f,11.726562f,8.125000f,8.242188f,8.414062f,8.710938f,12.078125f,12.093750f,11.875000f,9.710938f,10.773438f,
9.296875f,9.921875f,10.828125f,10.078125f,8.171875f,9.039062f,10.250000f,10.156250f,11.398438f,9.757812f,8.125000f,9.296875f,10.054688f,8.648438f,10.257812f,7.609375f,5.539062f,6.503906f,4.500000f,5.953125f,8.390625f,10.195312f,8.890625f,9.296875f,6.917969f,7.093750f,5.031250f,8.289062f,8.062500f,8.929688f,
7.980469f,9.054688f,8.781250f,11.406250f,7.851562f,7.195312f,7.660156f,7.644531f,7.781250f,8.218750f,8.437500f,7.125000f,7.871094f,7.269531f,9.460938f,9.156250f,10.343750f,11.226562f,8.523438f,8.906250f,7.980469f,9.171875f,11.828125f,11.875000f,9.843750f,10.250000f,11.265625f,10.320312f,9.914062f,6.281250f,
4.796875f,6.988281f,7.273438f,10.101562f,9.664062f,11.312500f,10.203125f,10.687500f,9.187500f,10.625000f,10.968750f,11.921875f,12.828125f,10.023438f,10.593750f,11.093750f,12.382812f,12.078125f,10.656250f,11.632812f,10.382812f,11.054688f,9.984375f,11.867188f,11.476562f,6.320312f,7.558594f,8.375000f,12.304688f,11.164062f,
11.351562f,8.898438f,6.632812f,4.953125f,7.636719f,9.179688f,10.437500f,11.625000f,10.507812f,10.640625f,9.304688f,9.687500f,10.507812f,10.617188f,9.937500f,8.671875f,10.234375f,11.156250f,12.617188f,12.640625f,10.476562f,11.203125f,12.578125f,13.023438f,13.289062f,10.632812f,11.085938f,6.968750f,8.117188f,7.871094f,
9.312500f,12.148438f,10.953125f,10.171875f,9.765625f,11.648438f,9.976562f,12.562500f,11.328125f,13.179688f,9.812500f,12.234375f,12.562500f,13.515625f,11.570312f,10.953125f,9.617188f,9.804688f,10.351562f,11.531250f,11.367188f,11.984375f,9.765625f,7.257812f,11.507812f,13.031250f,14.031250f,12.343750f,12.359375f,10.367188f,
8.851562f,9.242188f,11.554688f,11.523438f,12.601562f,13.820312f,13.835938f,12.968750f,13.718750f,11.984375f,11.171875f,8.148438f,8.156250f,8.867188f,11.406250f,13.953125f,12.867188f,12.148438f,11.445312f,10.437500f,13.710938f,13.429688f,13.164062f,13.687500f,12.382812f,12.632812f,14.382812f,13.453125f,13.820312f,14.796875f,
13.851562f,14.734375f,14.968750f,16.218750f,14.882812f,13.960938f,14.046875f,15.898438f,17.312500f,18.312500f,17.343750f,6.121094f,5.925781f,5.082031f,5.226562f,6.335938f,5.300781f,5.980469f,5.238281f,4.074219f,6.035156f,3.462891f,3.906250f,3.607422f,4.332031f,4.972656f,4.968750f,4.503906f,6.421875f,5.074219f,
5.863281f,6.050781f,3.417969f,3.791016f,3.484375f,3.845703f,3.658203f,5.687500f,5.925781f,6.371094f,3.751953f,6.125000f,3.742188f,6.042969f,4.175781f,6.062500f,3.496094f,5.687500f,3.960938f,5.769531f,6.683594f,3.564453f,5.601562f,6.023438f,6.105469f,4.007812f,6.222656f,3.837891f,6.765625f,3.857422f,
3.839844f,5.917969f,6.367188f,3.750000f,4.785156f,4.136719f,5.769531f,4.039062f,3.869141f,4.250000f,6.179688f,6.207031f,3.554688f,6.160156f,3.656250f,6.234375f,3.261719f,6.574219f,6.714844f,3.681641f,5.277344f,3.933594f,5.769531f,3.208984f,5.835938f,6.210938f,6.906250f,3.376953f,6.195312f,3.080078f,
3.562500f,3.271484f,4.085938f,6.699219f,3.292969f,3.839844f,3.921875f,6.750000f,3.714844f,5.929688f,5.941406f,6.175781f,6.160156f,3.435547f,4.691406f,3.640625f,7.378906f,3.640625f,7.210938f,3.650391f,3.490234f,6.960938f,4.914062f,3.638672f,3.761719f,4.964844f,3.900391f,6.269531f,4.101562f,5.949219f,
4.429688f,7.183594f,3.656250f,6.136719f,6.816406f,3.843750f,6.886719f,3.478516f,5.929688f,3.185547f,3.750000f,3.367188f,6.339844f,3.669922f,5.726562f,3.439453f,6.632812f,6.941406f,6.261719f,3.843750f,6.542969f,3.755859f,6.425781f,6.457031f,3.707031f,6.835938f,6.378906f,3.810547f,3.974609f,5.792969f,
3.804688f,6.085938f,3.863281f,4.664062f,4.062500f,6.933594f,7.054688f,4.308594f,7.199219f,6.449219f,4.910156f,5.593750f,4.304688f,6.609375f,6.429688f,4.914062f,4.898438f,4.894531f,6.703125f,6.402344f,4.023438f,3.884766f,4.417969f,3.617188f,7.093750f,3.876953f,6.644531f,7.296875f,5.140625f,3.939453f,
3.923828f,6.136719f,3.765625f,5.863281f,4.308594f,5.792969f,4.335938f,6.062500f,3.740234f,3.943359f,4.667969f,3.941406f,6.425781f,6.710938f,4.179688f,5.195312f,4.414062f,5.957031f,4.601562f,4.800781f,4.917969f,4.398438f,4.710938f,5.261719f,4.316406f,6.273438f,5.500000f,4.351562f,6.175781f,4.457031f,
6.789062f,6.273438f,4.593750f,6.421875f,5.972656f,6.750000f,4.375000f,4.503906f,4.828125f,3.878906f,4.972656f,5.277344f,4.500000f,5.214844f,3.873047f,5.031250f,6.144531f,7.371094f,3.978516f,6.140625f,6.410156f,4.054688f,6.734375f,6.574219f,6.570312f,4.093750f,4.691406f,4.703125f,6.027344f,3.839844f,
4.394531f,6.937500f,4.714844f,6.324219f,6.359375f,4.308594f,4.277344f,5.003906f,4.429688f,4.593750f,6.144531f,7.636719f,4.238281f,6.023438f,7.476562f,6.585938f,5.015625f,5.515625f,5.140625f,6.222656f,4.355469f,7.238281f,6.933594f,5.390625f,5.960938f,4.898438f,4.769531f,6.578125f,7.414062f,5.742188f,
5.308594f,5.332031f,5.347656f,5.476562f,6.609375f,7.453125f,5.843750f,4.949219f,4.800781f,5.539062f,5.164062f,6.121094f,5.804688f,4.972656f,5.097656f,6.011719f,5.015625f,4.609375f,5.125000f,5.093750f,5.773438f,5.027344f,5.410156f,5.640625f,5.898438f,5.410156f,5.558594f,5.742188f,5.785156f,5.679688f,
9.476562f,9.375000f,8.828125f,6.660156f,6.242188f,6.511719f,6.445312f,9.234375f,8.695312f,6.238281f,7.351562f,6.582031f,7.683594f,8.585938f,9.257812f,7.296875f,7.382812f,6.820312f,6.808594f,6.722656f,6.703125f,7.605469f,6.929688f,7.941406f,9.257812f,9.078125f,5.886719f,6.164062f,6.890625f,9.726562f,
7.781250f,9.195312f,6.878906f,8.929688f,7.062500f,8.976562f,8.945312f,9.742188f,6.121094f,7.136719f,9.250000f,6.984375f,7.261719f,6.742188f,8.476562f,7.414062f,8.468750f,6.921875f,8.492188f,8.914062f,7.390625f,6.160156f,7.613281f,9.984375f,8.890625f,5.953125f,8.851562f,10.851562f,8.710938f,6.421875f,
7.171875f,9.062500f,7.621094f,9.234375f,6.234375f,8.031250f,7.304688f,6.578125f,9.921875f,6.996094f,10.109375f,6.230469f,9.578125f,6.941406f,7.324219f,7.949219f,8.398438f,7.761719f,8.726562f,7.683594f,8.687500f,10.828125f,7.679688f,9.726562f,9.765625f,9.148438f,5.910156f,9.835938f,7.699219f,6.152344f,
6.875000f,6.562500f,9.070312f,9.007812f,10.109375f,6.644531f,8.679688f,7.144531f,8.679688f,9.281250f,6.484375f,8.531250f,8.828125f,9.984375f,8.539062f,9.851562f,6.839844f,9.765625f,5.863281f,9.281250f,6.285156f,8.171875f,6.265625f,5.753906f,7.085938f,7.734375f,7.976562f,7.277344f,8.203125f,7.109375f,
7.960938f,6.972656f,8.625000f,6.156250f,7.308594f,6.937500f,5.597656f,7.000000f,9.351562f,6.125000f,7.996094f,7.027344f,5.703125f,8.125000f,6.605469f,6.777344f,8.476562f,8.804688f,6.691406f,9.101562f,7.800781f,11.085938f,10.750000f,9.757812f,6.289062f,6.718750f,8.585938f,8.765625f,9.289062f,10.078125f,
11.546875f,10.585938f,7.152344f,7.617188f,9.562500f,10.335938f,8.960938f,6.300781f,7.109375f,10.101562f,10.812500f,11.476562f,8.484375f,6.640625f,8.953125f,6.820312f,7.187500f,11.250000f,10.171875f,10.289062f,8.414062f,9.007812f,8.023438f,12.156250f,7.691406f,9.898438f,6.835938f,9.945312f,10.343750f,8.500000f,
9.968750f,8.171875f,8.492188f,8.945312f,11.492188f,10.476562f,7.332031f,9.828125f,11.171875f,8.984375f,9.156250f,9.796875f,8.062500f,9.679688f,6.953125f,11.140625f,9.804688f,8.445312f,10.046875f,7.320312f,7.710938f,10.656250f,7.265625f,7.175781f,7.906250f,10.867188f,11.140625f,9.796875f,9.437500f,9.281250f,
9.515625f,11.140625f,8.398438f,8.632812f,11.093750f,7.535156f,8.546875f,10.445312f,8.281250f,8.421875f,9.328125f,7.652344f,8.359375f,9.468750f,10.937500f,12.054688f,10.695312f,8.789062f,9.492188f,10.859375f,7.679688f,11.054688f,7.726562f,8.046875f,9.718750f,10.945312f,9.757812f,11.085938f,10.960938f,8.539062f,
8.664062f,9.773438f,10.109375f,8.265625f,10.492188f,11.257812f,8.476562f,11.390625f,8.015625f,10.093750f,10.414062f,10.804688f,11.593750f,12.750000f,12.007812f,11.500000f,8.664062f,8.843750f,12.554688f,11.351562f,12.687500f,11.750000f,12.601562f,8.648438f,8.640625f,12.492188f,12.367188f,12.195312f,13.125000f,12.023438f,
8.773438f,13.109375f,12.554688f,13.085938f,11.148438f,12.195312f,13.523438f,13.750000f,12.359375f,10.835938f,14.210938f,10.625000f,13.195312f,10.437500f,13.773438f,11.539062f,11.453125f,10.296875f,12.429688f,7.394531f,7.378906f,5.121094f,4.320312f,7.535156f,4.285156f,5.050781f,5.250000f,6.359375f,6.082031f,4.613281f,
5.535156f,4.734375f,5.300781f,4.949219f,4.257812f,6.257812f,7.023438f,4.453125f,5.222656f,6.246094f,4.574219f,5.480469f,4.757812f,6.101562f,4.703125f,6.078125f,7.460938f,7.476562f,5.289062f,6.777344f,4.816406f,7.917969f,6.675781f,7.562500f,5.820312f,8.062500f,6.277344f,6.492188f,8.539062f,5.687500f,
7.687500f,7.714844f,8.718750f,6.421875f,6.984375f,6.316406f,8.453125f,6.367188f,6.203125f,7.472656f,7.613281f,6.656250f,6.199219f,6.945312f,7.234375f,7.308594f,6.203125f,7.234375f,8.585938f,8.156250f,6.753906f,6.996094f,5.968750f,8.351562f,6.945312f,8.476562f,8.273438f,7.296875f,6.886719f,7.832031f,
6.656250f,6.863281f,7.753906f,7.777344f,9.476562f,6.753906f,7.187500f,6.167969f,6.683594f,6.320312f,6.703125f,7.296875f,5.722656f,7.031250f,7.550781f,8.351562f,7.281250f,7.375000f,7.785156f,7.718750f,7.976562f,6.996094f,6.687500f,7.082031f,8.523438f,7.175781f,9.437500f,7.246094f,7.218750f,8.812500f,
7.562500f,7.335938f,7.312500f,6.363281f,7.406250f,8.148438f,7.746094f,6.988281f,6.535156f,9.828125f,7.441406f,7.910156f,8.695312f,6.859375f,7.605469f,6.968750f,7.367188f,6.390625f,6.890625f,6.632812f,7.128906f,6.035156f,8.046875f,6.875000f,7.953125f,8.062500f,7.699219f,7.234375f,7.925781f,7.144531f,
7.582031f,8.460938f,7.175781f,8.515625f,7.945312f,7.371094f,6.503906f,8.265625f,7.292969f,7.789062f,7.140625f,7.988281f,6.734375f,8.500000f,8.843750f,7.812500f,8.007812f,7.515625f,7.699219f,7.527344f,8.171875f,9.492188f,9.000000f,7.734375f,7.667969f,7.968750f,9.398438f,8.726562f,8.015625f,6.921875f,
6.996094f,7.890625f,8.453125f,7.968750f,8.718750f,9.484375f,7.695312f,7.042969f,7.960938f,8.671875f,7.664062f,8.031250f,7.660156f,8.226562f,8.125000f,7.488281f,7.769531f,6.417969f,8.070312f,7.511719f,8.523438f,8.789062f,7.812500f,7.128906f,7.625000f,8.632812f,8.093750f,7.265625f,8.062500f,6.457031f,
7.074219f,8.867188f,6.953125f,9.390625f,7.609375f,7.390625f,8.125000f,6.328125f,9.539062f,8.531250f,7.667969f,9.265625f,7.832031f,8.914062f,6.703125f,7.718750f,7.441406f,7.273438f,7.996094f,8.609375f,7.867188f,8.164062f,7.367188f,8.023438f,8.078125f,9.218750f,7.210938f,9.125000f,9.218750f,7.480469f,
10.343750f,9.765625f,8.632812f,6.062500f,8.000000f,7.914062f,8.812500f,7.070312f,8.015625f,9.906250f,7.750000f,9.367188f,9.296875f,7.625000f,7.292969f,7.714844f,7.488281f,6.777344f,9.312500f,11.007812f,7.652344f,8.875000f,10.585938f,10.101562f,8.593750f,8.828125f,8.992188f,8.664062f,8.187500f,9.210938f,
8.773438f,8.507812f,8.546875f,8.898438f,7.480469f,10.164062f,10.859375f,8.765625f,8.093750f,7.921875f,8.570312f,8.085938f,10.375000f,11.078125f,8.914062f,8.945312f,7.464844f,8.812500f,9.523438f,10.960938f,9.101562f,8.101562f,9.226562f,10.304688f,9.203125f,8.093750f,8.953125f,9.421875f,10.039062f,8.921875f,
9.976562f,10.179688f,9.867188f,9.359375f,10.007812f,10.375000f,10.960938f,10.648438f,8.273438f,7.378906f,5.386719f,5.296875f,8.101562f,5.292969f,6.433594f,6.542969f,5.113281f,7.398438f,3.849609f,4.609375f,3.851562f,5.230469f,5.496094f,5.593750f,6.289062f,8.820312f,5.558594f,6.570312f,7.406250f,3.800781f,
4.652344f,4.000000f,4.953125f,4.953125f,7.156250f,8.187500f,8.656250f,5.363281f,8.335938f,5.285156f,8.835938f,5.562500f,8.445312f,4.929688f,7.304688f,5.253906f,7.222656f,8.929688f,4.960938f,8.125000f,8.289062f,8.562500f,5.375000f,8.070312f,5.414062f,9.156250f,5.582031f,5.226562f,7.996094f,8.156250f,
5.921875f,6.171875f,5.675781f,7.589844f,5.550781f,5.082031f,5.769531f,8.671875f,8.679688f,5.523438f,8.335938f,5.750000f,8.523438f,5.296875f,9.203125f,8.867188f,5.691406f,7.503906f,6.082031f,7.273438f,5.417969f,8.093750f,8.312500f,9.351562f,6.085938f,7.769531f,4.500000f,5.187500f,4.500000f,5.750000f,
8.812500f,5.410156f,5.937500f,6.062500f,9.054688f,5.691406f,8.226562f,7.812500f,8.039062f,8.539062f,5.992188f,7.496094f,5.585938f,9.609375f,5.972656f,9.734375f,6.011719f,5.441406f,9.515625f,6.804688f,5.832031f,5.449219f,7.691406f,5.585938f,8.750000f,5.765625f,7.656250f,6.066406f,9.765625f,5.332031f,
8.320312f,8.515625f,5.753906f,9.062500f,5.578125f,7.992188f,4.761719f,5.437500f,4.675781f,8.125000f,5.359375f,7.917969f,4.992188f,8.578125f,8.765625f,8.421875f,5.492188f,8.453125f,5.972656f,8.968750f,8.601562f,5.390625f,9.195312f,8.421875f,5.617188f,6.195312f,8.031250f,5.375000f,8.109375f,5.777344f,
6.425781f,6.125000f,9.367188f,9.117188f,6.500000f,9.992188f,7.851562f,6.828125f,7.867188f,6.265625f,9.375000f,9.156250f,6.769531f,6.449219f,6.511719f,8.828125f,8.578125f,5.691406f,5.664062f,5.921875f,5.601562f,9.187500f,6.156250f,9.476562f,9.914062f,6.585938f,6.007812f,5.917969f,9.039062f,5.617188f,
8.390625f,6.328125f,9.226562f,6.406250f,8.156250f,5.753906f,5.785156f,8.062500f,5.554688f,9.421875f,9.710938f,6.398438f,6.800781f,6.003906f,9.023438f,6.144531f,5.902344f,6.066406f,6.000000f,6.171875f,8.875000f,5.781250f,9.687500f,6.691406f,5.808594f,8.804688f,6.066406f,9.773438f,8.929688f,6.000000f,
9.617188f,8.343750f,9.492188f,6.269531f,6.410156f,8.335938f,5.898438f,7.589844f,8.492188f,6.515625f,8.351562f,5.707031f,6.535156f,8.335938f,10.218750f,6.050781f,9.335938f,9.304688f,6.968750f,9.992188f,9.703125f,9.625000f,6.507812f,6.984375f,6.917969f,9.132812f,6.242188f,6.878906f,10.242188f,6.875000f,
9.187500f,8.992188f,6.953125f,6.351562f,8.265625f,6.531250f,7.050781f,9.695312f,10.976562f,6.601562f,9.210938f,11.218750f,9.531250f,7.214844f,8.609375f,7.468750f,8.789062f,7.675781f,11.117188f,9.367188f,8.296875f,9.132812f,7.636719f,7.539062f,10.085938f,10.507812f,7.585938f,7.628906f,7.863281f,8.085938f,
8.125000f,10.226562f,10.664062f,7.777344f,7.585938f,7.710938f,8.250000f,8.148438f,10.343750f,7.984375f,7.738281f,7.960938f,10.023438f,8.125000f,7.125000f,7.761719f,8.109375f,9.406250f,7.894531f,9.664062f,8.625000f,9.460938f,8.164062f,9.898438f,10.132812f,10.132812f,8.718750f,6.250000f,6.542969f,4.757812f,
4.468750f,5.316406f,3.707031f,3.248047f,2.623047f,4.144531f,3.785156f,3.714844f,4.160156f,3.406250f,3.964844f,4.152344f,4.835938f,6.761719f,6.906250f,5.183594f,4.683594f,5.320312f,4.164062f,4.730469f,4.167969f,5.257812f,4.472656f,5.023438f,5.832031f,6.140625f,6.253906f,6.320312f,4.832031f,6.277344f,
6.050781f,6.234375f,6.105469f,7.078125f,6.386719f,5.503906f,6.167969f,5.929688f,6.335938f,6.699219f,6.871094f,6.617188f,6.023438f,5.812500f,6.410156f,5.695312f,5.996094f,5.980469f,5.484375f,4.582031f,3.806641f,4.582031f,4.023438f,4.507812f,4.378906f,5.304688f,5.703125f,6.781250f,7.039062f,6.796875f,
5.386719f,5.707031f,5.691406f,5.914062f,6.191406f,6.726562f,5.566406f,7.000000f,5.218750f,6.765625f,6.183594f,6.550781f,7.773438f,6.738281f,6.046875f,6.183594f,6.515625f,5.683594f,5.410156f,4.859375f,5.273438f,6.496094f,7.117188f,5.429688f,6.984375f,4.425781f,4.003906f,4.484375f,5.410156f,6.996094f,
6.027344f,7.707031f,6.843750f,6.437500f,6.472656f,6.214844f,6.738281f,6.058594f,7.042969f,6.859375f,7.148438f,5.152344f,7.246094f,5.992188f,6.898438f,4.917969f,5.750000f,6.496094f,5.382812f,4.632812f,4.101562f,3.109375f,3.818359f,4.843750f,5.000000f,6.230469f,6.804688f,5.851562f,5.164062f,4.976562f,
5.464844f,5.871094f,5.210938f,5.550781f,5.210938f,6.753906f,5.136719f,6.082031f,4.621094f,3.851562f,5.976562f,5.609375f,6.093750f,7.390625f,6.351562f,6.832031f,6.753906f,6.191406f,6.175781f,7.582031f,6.308594f,5.480469f,5.937500f,5.832031f,5.320312f,6.183594f,7.484375f,6.472656f,8.023438f,7.648438f,
7.707031f,6.304688f,4.863281f,5.324219f,5.441406f,5.468750f,6.019531f,6.019531f,6.714844f,7.437500f,6.238281f,6.917969f,6.042969f,6.199219f,6.886719f,6.710938f,7.394531f,6.765625f,6.703125f,6.386719f,6.515625f,7.054688f,8.000000f,6.382812f,7.996094f,6.863281f,6.953125f,7.765625f,7.132812f,7.140625f,
6.406250f,5.722656f,7.277344f,6.328125f,6.839844f,5.203125f,5.242188f,3.636719f,3.882812f,5.824219f,7.167969f,6.683594f,6.968750f,6.921875f,6.550781f,5.503906f,6.582031f,6.515625f,6.949219f,6.753906f,6.250000f,6.843750f,7.210938f,7.972656f,5.972656f,7.750000f,7.750000f,8.234375f,8.242188f,6.851562f,
6.585938f,7.183594f,5.718750f,6.574219f,6.921875f,7.386719f,7.347656f,7.226562f,6.722656f,6.265625f,6.433594f,6.855469f,7.679688f,8.734375f,7.410156f,8.281250f,8.945312f,7.203125f,7.789062f,5.859375f,5.480469f,6.312500f,7.003906f,5.601562f,8.078125f,7.375000f,6.886719f,7.441406f,7.878906f,7.789062f,
7.781250f,8.710938f,7.804688f,5.804688f,7.285156f,5.507812f,6.531250f,6.488281f,7.472656f,8.640625f,7.187500f,8.882812f,7.734375f,7.957031f,7.644531f,7.558594f,7.324219f,7.375000f,8.679688f,7.984375f,7.542969f,6.789062f,6.890625f,8.078125f,7.457031f,7.769531f,9.765625f,7.437500f,8.445312f,8.320312f,
9.351562f,8.500000f,9.656250f,8.585938f,8.070312f,9.773438f,8.429688f,8.703125f,8.929688f,10.773438f,8.609375f,10.796875f,9.906250f,10.000000f,9.562500f,9.664062f,10.953125f,9.929688f,9.085938f,7.921875f,9.085938f,7.812500f,8.531250f,9.937500f,8.515625f,8.710938f,7.878906f,7.984375f,7.976562f,8.906250f,
9.039062f,7.996094f,7.542969f,7.937500f,7.976562f,8.609375f,8.507812f,7.851562f,8.093750f,8.132812f,8.359375f,8.351562f,7.855469f,8.203125f,8.335938f,8.851562f,8.640625f,8.445312f,8.265625f,8.078125f,8.382812f,7.878906f,9.703125f,8.312500f,7.921875f,8.859375f,7.636719f,8.335938f,8.601562f,8.593750f,
7.371094f,8.304688f,7.703125f,8.898438f,7.867188f,7.957031f,8.546875f,8.351562f,8.367188f,10.078125f,8.664062f,8.140625f,8.656250f,9.671875f,7.996094f,9.390625f,9.109375f,8.000000f,8.640625f,8.687500f,9.140625f,7.640625f,8.898438f,9.117188f,8.625000f,7.953125f,9.046875f,8.195312f,8.257812f,8.343750f,
8.531250f,9.898438f,8.445312f,8.937500f,8.500000f,8.671875f,8.710938f,9.882812f,8.578125f,8.789062f,9.500000f,8.453125f,9.078125f,8.656250f,8.773438f,8.093750f,8.429688f,8.140625f,7.984375f,8.234375f,8.687500f,9.093750f,7.886719f,9.687500f,7.902344f,8.109375f,8.867188f,8.617188f,7.792969f,8.664062f,
8.500000f,8.789062f,8.984375f,8.710938f,8.546875f,9.164062f,9.656250f,8.429688f,9.015625f,9.234375f,8.132812f,9.562500f,7.898438f,9.421875f,8.257812f,8.429688f,8.507812f,8.968750f,8.546875f,8.679688f,7.414062f,9.195312f,8.882812f,9.218750f,8.539062f,8.859375f,7.875000f,8.921875f,9.265625f,7.414062f,
9.062500f,8.664062f,7.523438f,8.093750f,8.507812f,7.734375f,8.421875f,9.187500f,9.234375f,8.484375f,9.609375f,9.453125f,7.937500f,10.070312f,10.078125f,8.945312f,10.460938f,8.781250f,9.968750f,9.804688f,9.679688f,10.320312f,9.257812f,9.726562f,9.406250f,8.835938f,9.359375f,9.914062f,8.117188f,9.468750f,
8.312500f,8.906250f,10.093750f,10.085938f,9.351562f,9.179688f,9.421875f,8.218750f,8.960938f,10.007812f,8.726562f,8.914062f,8.726562f,8.812500f,9.281250f,8.585938f,8.812500f,9.085938f,9.523438f,8.757812f,10.312500f,9.359375f,9.273438f,9.562500f,10.117188f,8.898438f,8.906250f,9.421875f,9.500000f,8.828125f,
9.875000f,10.000000f,8.578125f,9.390625f,9.093750f,9.953125f,9.078125f,8.921875f,9.210938f,9.054688f,9.460938f,9.234375f,9.117188f,8.984375f,8.281250f,9.289062f,10.046875f,9.484375f,8.734375f,8.500000f,9.992188f,9.156250f,10.132812f,8.843750f,9.445312f,9.500000f,8.906250f,10.343750f,10.070312f,9.671875f,
9.296875f,9.882812f,9.156250f,9.187500f,8.570312f,9.625000f,10.289062f,9.437500f,9.609375f,9.609375f,8.882812f,9.242188f,9.171875f,9.281250f,9.406250f,9.476562f,11.304688f,8.609375f,9.773438f,11.007812f,10.898438f,9.429688f,9.320312f,9.765625f,9.632812f,9.351562f,10.890625f,10.453125f,9.914062f,10.703125f,
9.671875f,9.726562f,10.023438f,10.828125f,10.257812f,9.679688f,10.234375f,9.914062f,10.296875f,10.296875f,11.023438f,10.242188f,9.375000f,9.343750f,10.453125f,9.437500f,10.671875f,10.164062f,9.343750f,9.218750f,10.593750f,8.632812f,8.593750f,9.687500f,8.460938f,10.179688f,9.625000f,10.179688f,9.093750f,10.156250f,
8.937500f,10.648438f,10.843750f,10.976562f,9.195312f,-0.231445f,0.230103f,-2.464844f,-2.410156f,-0.214111f,-3.117188f,-2.263672f,0.586914f,-1.509766f,0.511230f,-2.300781f,-1.910156f,-2.886719f,-2.746094f,-2.894531f,-2.402344f,0.225952f,1.278320f,-2.037109f,-1.657227f,0.686523f,-2.177734f,-1.362305f,-2.042969f,-1.166992f,
-1.400391f,-0.081238f,0.702148f,0.634766f,0.204834f,1.265625f,-0.506836f,1.064453f,0.780762f,1.230469f,0.590820f,0.487305f,0.922363f,0.710449f,2.105469f,0.584473f,2.019531f,2.781250f,2.720703f,1.686523f,2.886719f,0.643066f,3.375000f,1.011719f,1.807617f,3.638672f,3.332031f,0.414062f,2.710938f,0.916504f,
3.769531f,0.463135f,2.285156f,1.177734f,3.416016f,4.433594f,1.995117f,3.556641f,1.226562f,2.925781f,-0.297607f,3.537109f,3.101562f,1.730469f,2.812500f,1.775391f,1.916016f,1.128906f,2.486328f,3.087891f,2.759766f,0.284180f,3.210938f,0.627930f,1.044922f,0.279297f,1.557617f,3.675781f,0.132568f,3.169922f,
1.895508f,4.167969f,1.536133f,3.183594f,2.636719f,3.232422f,3.353516f,0.421875f,3.505859f,2.226562f,3.933594f,0.396729f,4.039062f,0.254150f,1.243164f,3.994141f,2.101562f,0.573730f,2.195312f,3.355469f,2.587891f,3.105469f,2.505859f,2.156250f,2.982422f,2.865234f,1.305664f,2.853516f,2.189453f,-1.484375f,
2.539062f,-0.888672f,2.593750f,0.998535f,1.648438f,0.666504f,2.490234f,0.601562f,2.939453f,-0.205566f,2.410156f,2.753906f,2.500000f,1.599609f,2.687500f,0.560547f,3.015625f,2.074219f,0.652344f,3.142578f,2.779297f,1.713867f,0.813477f,3.423828f,1.112305f,3.216797f,1.603516f,2.935547f,1.682617f,3.794922f,
3.126953f,0.402344f,2.328125f,2.914062f,1.893555f,3.929688f,3.113281f,4.082031f,4.601562f,1.955078f,2.343750f,-0.055634f,3.216797f,3.886719f,0.826172f,1.639648f,3.396484f,1.323242f,3.357422f,0.675293f,3.285156f,3.302734f,2.623047f,2.396484f,2.583984f,3.412109f,1.847656f,2.541016f,3.046875f,3.634766f,
3.589844f,2.974609f,3.189453f,2.451172f,4.210938f,2.949219f,4.363281f,4.566406f,1.756836f,4.152344f,3.392578f,5.488281f,3.478516f,4.238281f,2.464844f,0.591309f,3.353516f,3.617188f,3.218750f,4.941406f,4.289062f,2.685547f,4.773438f,2.363281f,5.394531f,4.707031f,2.640625f,4.906250f,4.097656f,4.574219f,
3.292969f,3.505859f,4.378906f,2.347656f,3.210938f,3.255859f,3.966797f,4.531250f,1.635742f,2.615234f,3.019531f,3.781250f,2.158203f,4.265625f,4.683594f,1.724609f,4.464844f,4.816406f,5.230469f,2.576172f,4.320312f,3.894531f,5.519531f,2.777344f,3.441406f,5.792969f,3.630859f,4.726562f,5.042969f,1.212891f,
3.128906f,5.332031f,3.951172f,3.466797f,6.046875f,6.039062f,2.716797f,5.500000f,6.539062f,5.972656f,3.984375f,5.464844f,3.537109f,4.785156f,2.371094f,5.160156f,5.445312f,3.914062f,6.410156f,4.867188f,4.277344f,6.050781f,5.832031f,5.402344f,3.013672f,4.656250f,3.656250f,5.589844f,6.699219f,6.277344f,
5.648438f,4.203125f,4.296875f,6.320312f,5.933594f,6.914062f,6.164062f,5.371094f,5.785156f,7.351562f,6.011719f,6.015625f,6.515625f,6.195312f,7.203125f,7.210938f,7.648438f,7.617188f,7.164062f,8.250000f,8.343750f,8.609375f,8.351562f,8.546875f,1.875977f,2.152344f,-0.048279f,-0.235840f,1.903320f,-0.174316f,
0.703125f,1.934570f,1.625977f,2.642578f,1.364258f,1.662109f,1.308594f,0.638184f,0.330811f,0.562500f,1.436523f,1.589844f,-0.018387f,0.849121f,2.521484f,1.069336f,1.415039f,1.080078f,1.744141f,1.437500f,1.551758f,2.148438f,2.015625f,1.643555f,2.148438f,1.289062f,2.195312f,1.837891f,2.392578f,1.483398f,
1.740234f,1.716797f,1.878906f,2.722656f,1.563477f,2.341797f,2.736328f,2.392578f,1.158203f,2.337891f,0.045837f,2.683594f,-0.050476f,0.853516f,2.132812f,2.265625f,-0.832031f,1.753906f,0.895508f,2.238281f,1.123047f,1.601562f,1.376953f,2.447266f,2.886719f,1.070312f,1.749023f,0.704590f,1.499023f,-0.249878f,
1.773438f,1.325195f,1.307617f,1.190430f,1.544922f,1.052734f,1.104492f,1.268555f,1.640625f,1.365234f,-0.603516f,2.054688f,0.654297f,1.030273f,0.650391f,0.568848f,2.273438f,1.192383f,1.218750f,2.158203f,2.080078f,1.979492f,1.766602f,1.564453f,2.023438f,1.980469f,1.287109f,1.229492f,2.060547f,2.208984f,
1.103516f,2.144531f,1.068359f,1.844727f,2.160156f,1.066406f,1.046875f,1.923828f,1.718750f,1.777344f,2.003906f,1.658203f,1.390625f,2.044922f,1.931641f,1.366211f,1.766602f,1.818359f,0.043518f,2.300781f,0.867188f,2.609375f,1.403320f,1.617188f,1.017578f,2.195312f,1.118164f,2.181641f,0.354248f,2.126953f,
2.085938f,2.003906f,2.027344f,2.115234f,0.944824f,2.406250f,1.993164f,1.767578f,2.603516f,2.437500f,1.953125f,1.364258f,2.673828f,1.925781f,2.474609f,2.660156f,2.857422f,2.083984f,2.857422f,2.271484f,0.920898f,1.939453f,2.816406f,1.400391f,2.677734f,2.248047f,2.933594f,3.429688f,0.774902f,1.253906f,
0.173340f,2.636719f,3.222656f,2.097656f,2.125000f,2.326172f,0.840332f,2.521484f,0.849609f,2.609375f,2.537109f,2.035156f,2.273438f,2.412109f,2.515625f,2.175781f,2.232422f,2.183594f,2.683594f,2.794922f,2.251953f,2.460938f,2.001953f,2.935547f,2.156250f,2.841797f,3.025391f,0.854492f,2.919922f,2.642578f,
3.148438f,2.677734f,2.587891f,2.642578f,1.809570f,3.044922f,1.952148f,2.132812f,2.757812f,2.644531f,2.746094f,3.289062f,2.355469f,2.830078f,2.521484f,2.957031f,2.804688f,2.115234f,2.357422f,2.986328f,3.138672f,2.275391f,1.417969f,2.171875f,1.522461f,2.562500f,2.207031f,0.840332f,1.979492f,1.952148f,
2.480469f,2.630859f,2.328125f,2.671875f,1.462891f,3.080078f,3.578125f,3.607422f,2.392578f,2.369141f,3.134766f,3.087891f,1.737305f,1.913086f,3.273438f,2.884766f,2.515625f,2.951172f,1.926758f,3.087891f,2.888672f,3.177734f,2.564453f,3.162109f,3.279297f,2.007812f,2.792969f,3.306641f,3.876953f,3.322266f,
2.894531f,3.736328f,2.785156f,2.582031f,3.460938f,3.949219f,3.406250f,3.240234f,3.986328f,2.947266f,3.306641f,2.976562f,2.738281f,3.013672f,2.941406f,3.404297f,3.140625f,3.332031f,2.982422f,2.775391f,4.359375f,3.523438f,3.076172f,4.269531f,3.470703f,3.009766f,4.441406f,4.761719f,3.712891f,4.421875f,
4.171875f,2.630859f,4.468750f,3.496094f,3.273438f,3.744141f,4.441406f,3.386719f,4.445312f,4.046875f,3.734375f,3.748047f,3.550781f,0.027130f,1.594727f,-2.335938f,-3.050781f,1.240234f,-3.085938f,-1.458984f,1.179688f,-1.349609f,2.193359f,-0.506348f,-0.467285f,-0.822754f,-2.406250f,-2.169922f,-2.515625f,1.029297f,
1.625000f,-2.669922f,-1.247070f,2.166016f,-0.375244f,-0.086426f,-0.362305f,-1.170898f,-0.527832f,1.781250f,2.527344f,2.265625f,-0.748047f,1.035156f,-0.403320f,2.345703f,-0.488770f,2.425781f,-0.424561f,1.832031f,-0.421387f,1.924805f,2.980469f,-0.293213f,2.742188f,2.781250f,2.902344f,-0.256836f,2.580078f,-1.005859f,
3.060547f,-0.812988f,-0.109131f,2.310547f,2.984375f,-0.846680f,1.521484f,-0.777344f,2.769531f,-0.640625f,1.462891f,-0.625488f,2.638672f,3.212891f,-0.499023f,1.261719f,-0.368164f,2.113281f,-1.448242f,2.455078f,2.265625f,-0.483398f,2.025391f,-0.476318f,1.991211f,-0.544922f,2.478516f,2.462891f,2.281250f,-1.337891f,
2.546875f,-0.019928f,0.114868f,-0.214233f,0.591309f,2.025391f,-0.326904f,2.105469f,-0.285889f,2.974609f,-0.405273f,2.277344f,2.488281f,2.531250f,2.693359f,-1.059570f,1.436523f,-0.264893f,3.476562f,-1.182617f,3.347656f,-1.182617f,-0.420166f,3.195312f,1.239258f,-1.233398f,-0.276611f,1.784180f,-0.225220f,2.531250f,
-0.264404f,2.005859f,1.570312f,2.439453f,-0.432373f,2.042969f,2.451172f,-1.546875f,2.082031f,-1.334961f,2.517578f,0.250488f,0.346680f,-0.037079f,1.125000f,-0.268311f,2.447266f,-1.176758f,1.747070f,2.458984f,2.144531f,-0.170410f,2.894531f,-0.993164f,1.732422f,2.310547f,-0.549805f,3.003906f,2.976562f,-0.398926f,
-0.419189f,2.662109f,-0.507812f,2.289062f,1.045898f,0.881836f,0.196045f,3.062500f,2.492188f,-0.918945f,0.272461f,2.333984f,-1.154297f,2.003906f,-0.075806f,2.927734f,3.392578f,-1.018555f,0.583496f,-1.687500f,2.652344f,3.179688f,-0.457031f,0.625488f,2.023438f,-0.597168f,2.650391f,-0.916016f,3.498047f,3.117188f,
2.138672f,-0.049377f,-0.217041f,2.449219f,0.036865f,2.216797f,2.333984f,2.855469f,0.370117f,2.498047f,0.087585f,0.486328f,2.781250f,0.200928f,3.021484f,3.093750f,-0.430664f,2.039062f,0.191528f,3.281250f,0.502441f,2.048828f,0.009476f,-0.039917f,2.011719f,1.454102f,1.134766f,2.431641f,2.330078f,0.087891f,
1.808594f,0.376709f,3.371094f,3.191406f,0.095337f,3.710938f,2.648438f,3.218750f,0.116638f,0.007637f,2.103516f,-0.329102f,2.324219f,1.625977f,0.446533f,2.312500f,-0.707520f,1.416016f,2.285156f,3.031250f,0.285400f,2.953125f,2.931641f,-0.410645f,2.908203f,3.367188f,2.580078f,0.709473f,2.853516f,0.536621f,
3.152344f,0.187134f,0.749023f,3.425781f,0.734863f,3.080078f,3.091797f,-0.279053f,0.603516f,2.515625f,0.648926f,0.813477f,3.248047f,3.134766f,-0.057739f,2.318359f,3.146484f,2.958984f,0.590332f,2.800781f,0.655762f,2.917969f,0.065674f,1.144531f,3.203125f,-0.327881f,2.503906f,0.627441f,1.079102f,3.949219f,
3.472656f,2.671875f,-0.472412f,1.445312f,-0.094666f,1.719727f,4.195312f,3.644531f,2.794922f,0.732422f,1.368164f,3.050781f,1.123047f,3.654297f,2.845703f,0.989746f,0.988770f,3.386719f,1.368164f,1.490234f,2.519531f,1.374023f,3.285156f,3.181641f,3.718750f,1.399414f,3.390625f,1.751953f,3.546875f,3.718750f,
3.796875f,1.512695f,3.142578f,2.802734f,0.197388f,0.270264f,1.738281f,0.174194f,1.531250f,3.298828f,1.766602f,2.750000f,0.207153f,0.770508f,-0.074524f,1.377930f,-0.322998f,-0.063232f,0.327148f,1.304688f,-0.525391f,1.018555f,2.226562f,-0.436279f,0.089233f,-0.741211f,1.316406f,0.158081f,0.814941f,1.500977f,
1.350586f,0.578613f,2.255859f,-0.119751f,2.115234f,1.357422f,1.770508f,0.239136f,0.682129f,0.890137f,0.984375f,2.439453f,0.332520f,1.350586f,1.898438f,1.665039f,0.149170f,2.291016f,-0.616211f,2.666016f,-0.595703f,0.171509f,2.232422f,1.626953f,-0.284668f,1.808594f,1.186523f,2.527344f,1.534180f,1.421875f,
1.912109f,2.345703f,2.951172f,0.934570f,2.394531f,0.149292f,1.410156f,0.294922f,1.496094f,1.067383f,1.740234f,1.350586f,2.072266f,0.732422f,1.212891f,1.027344f,1.602539f,1.364258f,0.608887f,2.283203f,-0.101807f,0.435547f,-0.399658f,0.491211f,3.693359f,0.701172f,1.416992f,2.453125f,2.582031f,2.191406f,
2.113281f,1.479492f,2.093750f,2.332031f,1.084961f,2.138672f,1.776367f,2.357422f,0.922852f,2.513672f,0.911133f,1.419922f,2.787109f,0.189941f,0.739258f,1.571289f,2.552734f,1.419922f,2.232422f,1.573242f,1.417969f,2.433594f,1.898438f,0.854980f,2.525391f,2.294922f,0.609863f,3.330078f,0.573730f,2.859375f,
0.423340f,0.780273f,0.009018f,3.125000f,0.680176f,2.769531f,0.656250f,2.275391f,2.619141f,2.066406f,2.314453f,2.380859f,1.170898f,3.605469f,2.525391f,1.946289f,3.416016f,2.687500f,1.967773f,1.166016f,3.326172f,1.950195f,3.457031f,1.979492f,3.365234f,1.672852f,3.367188f,2.689453f,1.538086f,3.410156f,
3.353516f,2.123047f,3.933594f,2.400391f,3.580078f,4.199219f,2.158203f,2.777344f,1.735352f,3.501953f,4.156250f,2.730469f,1.883789f,2.906250f,1.417969f,2.519531f,1.431641f,2.910156f,2.738281f,1.835938f,2.357422f,2.861328f,3.390625f,2.253906f,2.687500f,2.757812f,3.201172f,3.244141f,2.580078f,2.910156f,
1.830078f,3.785156f,2.199219f,3.876953f,3.949219f,1.543945f,3.554688f,2.826172f,4.242188f,3.083984f,3.255859f,3.324219f,1.765625f,4.562500f,1.931641f,1.585938f,3.183594f,2.193359f,2.537109f,4.160156f,2.082031f,3.193359f,2.669922f,3.167969f,2.929688f,2.396484f,2.642578f,2.613281f,3.074219f,3.480469f,
1.553711f,2.064453f,1.683594f,2.679688f,3.298828f,1.179688f,1.952148f,2.640625f,2.921875f,2.197266f,2.849609f,3.496094f,1.609375f,3.990234f,4.628906f,5.078125f,1.997070f,2.505859f,3.171875f,4.605469f,1.616211f,1.605469f,4.429688f,3.060547f,3.335938f,4.007812f,2.166016f,3.205078f,4.394531f,3.259766f,
2.550781f,4.773438f,4.976562f,2.308594f,3.769531f,4.808594f,4.500000f,3.509766f,4.218750f,3.875000f,3.771484f,2.613281f,4.785156f,4.417969f,3.427734f,5.292969f,3.503906f,2.765625f,4.433594f,4.214844f,3.453125f,3.392578f,4.613281f,3.316406f,4.714844f,4.550781f,4.367188f,3.669922f,3.914062f,3.285156f,
4.304688f,4.179688f,5.160156f,3.964844f,3.580078f,3.994141f,5.488281f,3.517578f,3.515625f,4.960938f,3.523438f,4.949219f,5.398438f,5.464844f,4.156250f,5.093750f,3.878906f,6.371094f,6.066406f,5.792969f,3.291016f,2.941406f,2.617188f,1.562500f,1.367188f,2.222656f,1.062500f,1.189453f,1.914062f,2.302734f,
2.767578f,1.702148f,1.789062f,1.554688f,1.634766f,1.886719f,2.396484f,1.389648f,1.741211f,1.684570f,1.633789f,2.601562f,1.425781f,1.449219f,1.184570f,2.535156f,1.538086f,1.427734f,1.929688f,1.868164f,1.793945f,2.339844f,1.401367f,2.146484f,1.455078f,2.156250f,1.574219f,1.539062f,1.609375f,1.605469f,
2.388672f,1.771484f,2.119141f,2.517578f,2.228516f,1.179688f,2.208984f,0.419678f,2.949219f,0.493408f,1.666992f,2.216797f,2.251953f,0.053711f,2.062500f,2.191406f,2.285156f,1.620117f,1.665039f,1.590820f,2.751953f,3.099609f,1.992188f,2.373047f,1.301758f,1.721680f,0.347656f,1.854492f,1.621094f,1.651367f,
1.367188f,1.950195f,1.213867f,1.828125f,1.467773f,1.808594f,1.745117f,0.194336f,2.296875f,1.242188f,1.191406f,0.819824f,0.551758f,2.343750f,1.214844f,1.202148f,1.780273f,2.099609f,1.934570f,1.848633f,1.475586f,1.960938f,2.064453f,1.286133f,1.162109f,2.251953f,2.562500f,1.124023f,2.195312f,1.114258f,
2.197266f,2.447266f,0.738770f,1.232422f,2.271484f,1.904297f,2.181641f,2.248047f,2.050781f,1.706055f,2.623047f,2.095703f,1.469727f,1.912109f,2.007812f,0.016846f,2.626953f,0.665039f,2.835938f,2.093750f,1.939453f,1.496094f,2.539062f,1.565430f,2.273438f,0.769531f,2.193359f,2.373047f,2.333984f,2.228516f,
2.101562f,1.056641f,2.599609f,1.871094f,1.562500f,2.966797f,2.458984f,1.786133f,1.730469f,2.781250f,2.060547f,2.582031f,3.054688f,2.853516f,2.259766f,3.105469f,2.685547f,0.926270f,3.001953f,2.673828f,2.818359f,2.837891f,3.535156f,3.607422f,3.886719f,2.076172f,1.604492f,1.385742f,3.121094f,3.529297f,
3.056641f,2.710938f,2.837891f,1.492188f,2.740234f,0.971680f,3.078125f,2.917969f,2.677734f,2.324219f,3.177734f,2.660156f,1.686523f,2.611328f,2.212891f,2.945312f,2.578125f,2.472656f,2.656250f,2.482422f,3.083984f,2.544922f,3.230469f,3.248047f,1.010742f,2.988281f,3.363281f,3.484375f,2.593750f,2.833984f,
2.031250f,1.786133f,3.052734f,2.126953f,2.330078f,3.292969f,3.357422f,2.957031f,3.695312f,2.683594f,3.275391f,2.996094f,2.792969f,3.429688f,2.615234f,3.107422f,3.027344f,4.023438f,2.623047f,1.952148f,2.857422f,1.833984f,2.728516f,2.736328f,0.925293f,2.238281f,2.382812f,3.009766f,2.427734f,2.765625f,
3.080078f,1.117188f,3.322266f,3.611328f,3.679688f,2.535156f,2.937500f,2.685547f,3.558594f,1.948242f,2.261719f,3.748047f,2.835938f,3.050781f,3.445312f,1.740234f,3.125000f,3.234375f,3.351562f,3.121094f,3.810547f,3.960938f,2.353516f,3.333984f,4.386719f,3.070312f,2.964844f,3.642578f,3.242188f,3.625000f,
2.062500f,4.359375f,3.750000f,4.000000f,3.857422f,4.519531f,3.457031f,4.578125f,4.238281f,4.167969f,3.195312f,3.087891f,3.796875f,3.333984f,4.566406f,4.167969f,4.121094f,4.582031f,3.779297f,4.398438f,3.814453f,4.621094f,4.472656f,4.042969f,5.042969f,4.675781f,3.742188f,4.386719f,3.896484f,3.818359f,
4.812500f,4.398438f,5.066406f,4.363281f,4.875000f,5.113281f,5.593750f,5.414062f,5.339844f,3.978516f,2.667969f,1.635742f,0.927734f,0.427246f,1.730469f,0.260742f,0.210327f,1.151367f,0.591797f,1.583984f,1.180664f,1.098633f,1.172852f,0.631836f,0.785156f,0.621094f,1.147461f,1.598633f,0.978027f,0.814453f,
1.714844f,1.218750f,0.952637f,1.084961f,0.848145f,1.084961f,1.194336f,1.661133f,1.651367f,0.830566f,1.359375f,1.280273f,2.054688f,0.659668f,1.852539f,0.825195f,1.695312f,0.902832f,1.455078f,2.121094f,0.819824f,1.647461f,1.782227f,1.836914f,0.655273f,1.361328f,0.067505f,2.005859f,0.024933f,1.008789f,
1.394531f,1.438477f,0.676270f,1.290039f,1.224609f,1.485352f,1.184570f,1.171875f,0.917480f,1.886719f,1.938477f,1.375000f,1.429688f,1.500977f,1.500977f,0.407715f,1.694336f,1.341797f,0.857910f,1.373047f,0.819336f,1.284180f,0.755859f,1.583008f,1.713867f,1.931641f,0.461670f,1.779297f,1.250977f,1.086914f,
1.172852f,0.912109f,1.506836f,0.969727f,1.136719f,0.601562f,1.799805f,0.967285f,1.601562f,1.499023f,1.586914f,1.508789f,-0.079285f,0.751953f,0.792480f,2.078125f,0.008965f,1.967773f,-0.015564f,0.877930f,2.111328f,1.214844f,-0.079285f,0.639648f,1.164062f,0.708008f,1.973633f,0.849609f,1.579102f,1.405273f,
2.183594f,0.819336f,1.758789f,2.130859f,0.280762f,2.113281f,-0.126465f,2.023438f,1.760742f,1.636719f,1.784180f,1.609375f,1.631836f,1.930664f,0.686523f,1.955078f,2.005859f,2.250000f,1.166016f,1.908203f,0.473877f,2.025391f,2.185547f,0.799316f,2.558594f,2.093750f,0.894531f,1.791992f,2.304688f,1.175781f,
1.994141f,1.966797f,1.148438f,1.644531f,2.595703f,2.275391f,0.464111f,1.513672f,1.240234f,0.364990f,1.936523f,1.092773f,2.630859f,2.632812f,1.043945f,1.449219f,0.584961f,2.572266f,2.544922f,1.005859f,1.775391f,2.121094f,1.031250f,2.529297f,0.562988f,2.550781f,2.720703f,2.273438f,1.057617f,1.199219f,
2.333984f,0.607910f,2.306641f,2.117188f,2.558594f,1.012695f,2.318359f,1.000000f,1.735352f,2.330078f,1.151367f,2.511719f,2.503906f,0.380859f,1.520508f,0.844238f,2.386719f,1.127930f,2.164062f,0.875977f,1.375977f,1.493164f,1.953125f,1.853516f,2.486328f,2.445312f,1.325195f,1.790039f,1.733398f,2.707031f,
2.279297f,1.107422f,2.595703f,2.146484f,2.402344f,0.874023f,1.027344f,1.806641f,0.538574f,2.560547f,2.138672f,1.050781f,2.490234f,0.551758f,2.359375f,2.435547f,2.814453f,0.850586f,2.800781f,2.886719f,0.673828f,2.857422f,2.771484f,2.318359f,1.727539f,2.875000f,1.090820f,2.685547f,0.791504f,1.833008f,
3.033203f,1.412109f,2.908203f,2.933594f,0.503906f,1.265625f,2.181641f,1.472656f,2.185547f,3.009766f,3.187500f,1.052734f,2.675781f,3.195312f,2.050781f,1.458008f,2.896484f,1.216797f,2.818359f,0.522461f,1.907227f,1.746094f,0.745605f,2.707031f,1.367188f,2.166016f,3.558594f,3.728516f,3.160156f,1.000000f,
1.798828f,0.676758f,1.958984f,3.480469f,3.644531f,3.134766f,1.367188f,2.074219f,3.339844f,1.487305f,3.607422f,3.326172f,1.383789f,1.426758f,3.242188f,1.200195f,1.990234f,3.408203f,1.160156f,3.300781f,3.179688f,3.414062f,1.399414f,3.224609f,1.419922f,3.208984f,3.281250f,3.398438f,1.376953f,5.925781f,
4.312500f,4.250000f,5.738281f,3.085938f,5.605469f,5.453125f,3.650391f,4.585938f,2.322266f,4.480469f,4.281250f,4.835938f,6.210938f,4.195312f,5.835938f,3.160156f,2.484375f,5.542969f,5.484375f,2.283203f,4.332031f,3.970703f,4.453125f,4.605469f,5.039062f,1.682617f,1.754883f,1.772461f,5.144531f,2.816406f,
4.902344f,2.224609f,5.187500f,2.162109f,4.984375f,2.318359f,4.953125f,1.932617f,2.281250f,4.917969f,1.685547f,2.007812f,1.613281f,4.339844f,1.771484f,4.562500f,1.472656f,4.453125f,4.527344f,1.927734f,1.236328f,4.183594f,3.439453f,3.722656f,1.378906f,4.503906f,2.806641f,5.011719f,2.029297f,2.042969f,
5.125000f,2.675781f,4.652344f,1.828125f,5.324219f,2.066406f,1.477539f,5.222656f,1.724609f,5.437500f,1.508789f,5.500000f,1.258789f,1.597656f,1.893555f,5.578125f,2.162109f,4.507812f,4.167969f,4.644531f,2.919922f,2.865234f,5.316406f,1.362305f,5.894531f,1.770508f,5.925781f,1.908203f,1.361328f,1.726562f,
1.438477f,6.332031f,1.790039f,5.632812f,0.808105f,6.218750f,1.112305f,6.179688f,6.023438f,0.635742f,3.050781f,6.140625f,5.394531f,1.410156f,5.261719f,1.740234f,5.542969f,1.502930f,2.896484f,1.319336f,5.312500f,1.769531f,1.724609f,5.480469f,2.087891f,5.527344f,2.042969f,4.382812f,3.966797f,4.351562f,
2.941406f,4.941406f,1.441406f,5.675781f,2.996094f,1.333008f,1.804688f,5.562500f,0.912598f,5.777344f,2.062500f,1.699219f,5.503906f,1.103516f,1.066406f,5.660156f,5.664062f,1.599609f,6.136719f,1.850586f,4.492188f,4.406250f,5.578125f,1.589844f,1.529297f,6.339844f,4.820312f,4.539062f,6.460938f,3.341797f,
5.359375f,1.989258f,2.041016f,6.644531f,5.070312f,5.863281f,1.921875f,2.037109f,5.324219f,5.250000f,2.667969f,6.093750f,2.115234f,6.238281f,0.708984f,1.589844f,2.416016f,6.089844f,5.519531f,2.322266f,5.753906f,2.261719f,2.421875f,1.851562f,6.105469f,2.044922f,6.191406f,5.804688f,2.130859f,6.351562f,
1.856445f,1.688477f,6.414062f,4.386719f,5.515625f,1.792969f,5.812500f,3.160156f,6.003906f,5.960938f,4.023438f,3.289062f,4.578125f,2.343750f,3.011719f,6.765625f,3.732422f,6.210938f,1.974609f,1.496094f,6.628906f,1.001953f,1.939453f,1.629883f,6.683594f,6.210938f,1.791992f,6.824219f,3.652344f,2.761719f,
6.468750f,1.704102f,6.546875f,4.210938f,2.134766f,2.173828f,6.722656f,1.687500f,2.056641f,6.914062f,2.492188f,2.591797f,3.376953f,6.285156f,2.482422f,6.464844f,1.902344f,6.621094f,5.187500f,2.357422f,6.308594f,1.854492f,2.251953f,6.863281f,6.070312f,2.173828f,6.148438f,6.347656f,1.992188f,2.886719f,
7.003906f,3.429688f,2.187500f,5.300781f,6.519531f,2.498047f,6.679688f,2.480469f,7.171875f,5.917969f,5.460938f,7.621094f,3.943359f,6.355469f,6.539062f,1.630859f,2.353516f,3.277344f,8.062500f,6.554688f,7.714844f,6.589844f,1.612305f,2.337891f,3.281250f,6.648438f,6.945312f,3.423828f,7.039062f,2.984375f,
3.533203f,7.343750f,6.937500f,3.427734f,7.187500f,6.281250f,3.304688f,7.175781f,3.373047f,3.607422f,2.941406f,7.183594f,3.296875f,7.042969f,3.121094f,3.199219f,2.798828f,6.667969f,2.541016f,1.742188f,0.907227f,0.382324f,2.195312f,0.446777f,0.747070f,1.402344f,1.193359f,2.507812f,1.755859f,1.668945f,
1.566406f,0.507812f,0.653320f,0.368652f,1.250977f,1.433594f,0.467529f,0.695312f,2.312500f,1.628906f,1.573242f,1.524414f,0.980469f,1.077148f,1.581055f,2.154297f,2.052734f,0.717773f,1.259766f,1.040039f,1.990234f,0.636230f,2.048828f,0.678711f,1.765625f,0.644043f,1.619141f,2.269531f,0.665039f,1.895508f,
2.072266f,1.991211f,0.439941f,1.462891f,0.511230f,2.087891f,0.608887f,0.949707f,1.431641f,1.892578f,0.544922f,1.613281f,1.391602f,1.611328f,1.223633f,1.220703f,0.957031f,2.320312f,2.484375f,1.026367f,1.437500f,1.288086f,1.797852f,0.307373f,2.207031f,1.753906f,1.067383f,1.650391f,1.093750f,1.721680f,
1.009766f,1.977539f,2.140625f,2.123047f,0.590820f,2.541016f,1.838867f,1.793945f,1.726562f,1.164062f,1.830078f,1.470703f,1.318359f,1.133789f,2.333984f,1.274414f,1.985352f,2.068359f,2.220703f,2.058594f,0.948242f,0.797852f,1.128906f,2.640625f,0.848145f,2.541016f,0.799805f,1.031250f,2.515625f,1.617188f,
0.545410f,0.874512f,1.375977f,0.870117f,2.392578f,0.860352f,2.023438f,1.802734f,2.525391f,1.047852f,1.967773f,2.531250f,0.455322f,2.400391f,0.735352f,2.677734f,1.901367f,1.834961f,1.851562f,1.768555f,1.527344f,2.226562f,0.560059f,2.341797f,2.375000f,2.498047f,1.324219f,2.476562f,1.122070f,2.031250f,
2.474609f,1.062500f,2.718750f,2.583984f,1.016602f,1.580078f,2.496094f,1.284180f,2.144531f,2.525391f,1.644531f,1.781250f,2.824219f,2.507812f,0.701172f,1.466797f,1.891602f,0.582520f,2.068359f,1.549805f,2.960938f,3.138672f,1.270508f,1.532227f,1.072266f,2.978516f,3.128906f,1.736328f,2.109375f,2.304688f,
0.799805f,2.974609f,1.278320f,3.207031f,3.177734f,2.652344f,1.602539f,1.686523f,2.583984f,1.177734f,2.605469f,2.291016f,2.804688f,1.341797f,2.621094f,1.250977f,1.838867f,2.478516f,1.454102f,2.605469f,2.484375f,0.842773f,2.234375f,1.762695f,2.568359f,1.857422f,2.265625f,1.610352f,1.921875f,2.339844f,
1.961914f,2.083984f,2.373047f,2.613281f,1.654297f,2.183594f,2.007812f,2.972656f,2.759766f,1.662109f,3.263672f,2.582031f,2.955078f,1.529297f,1.556641f,1.960938f,0.662598f,3.076172f,2.396484f,1.546875f,2.414062f,0.769531f,2.675781f,2.693359f,3.230469f,1.539062f,2.919922f,3.089844f,0.941406f,3.208984f,
3.339844f,2.523438f,2.052734f,3.074219f,1.557617f,2.720703f,1.057617f,2.333984f,3.244141f,1.949219f,3.130859f,3.291016f,1.678711f,1.918945f,2.341797f,1.872070f,2.244141f,2.996094f,3.269531f,1.058594f,2.750000f,3.201172f,2.484375f,1.941406f,3.044922f,1.939453f,3.191406f,1.062500f,1.924805f,2.464844f,
1.083984f,2.662109f,1.975586f,2.429688f,3.900391f,3.841797f,3.349609f,1.463867f,2.130859f,1.393555f,2.205078f,4.000000f,3.947266f,3.488281f,2.261719f,2.679688f,3.525391f,2.017578f,3.818359f,3.587891f,2.154297f,2.152344f,3.453125f,1.863281f,2.636719f,3.400391f,1.794922f,3.634766f,3.392578f,3.785156f,
1.769531f,3.720703f,2.070312f,3.533203f,3.570312f,3.746094f,1.911133f,2.992188f,2.359375f,1.196289f,0.741211f,2.333984f,0.514160f,1.318359f,3.373047f,1.118164f,3.177734f,1.652344f,2.113281f,2.007812f,1.921875f,1.806641f,1.724609f,2.521484f,2.496094f,1.782227f,2.525391f,3.968750f,2.365234f,2.519531f,
2.294922f,2.347656f,2.462891f,2.498047f,3.083984f,3.058594f,3.177734f,3.572266f,2.820312f,3.363281f,3.234375f,3.603516f,2.628906f,3.134766f,3.060547f,3.156250f,4.023438f,2.875000f,3.328125f,3.710938f,3.353516f,2.869141f,3.466797f,3.126953f,3.541016f,2.882812f,2.390625f,3.343750f,3.283203f,1.810547f,
3.730469f,1.965820f,3.359375f,2.455078f,3.107422f,2.876953f,3.933594f,4.425781f,2.585938f,3.457031f,2.640625f,3.097656f,1.772461f,3.310547f,2.775391f,2.451172f,2.941406f,2.683594f,2.746094f,2.240234f,2.966797f,3.406250f,3.253906f,1.984375f,4.421875f,2.699219f,2.794922f,2.623047f,2.771484f,3.826172f,
2.363281f,2.750000f,2.605469f,3.806641f,2.470703f,3.183594f,2.953125f,3.365234f,3.021484f,2.693359f,2.939453f,2.902344f,3.378906f,2.789062f,3.380859f,2.828125f,2.511719f,2.884766f,2.753906f,3.304688f,3.185547f,2.953125f,3.164062f,3.173828f,3.111328f,2.722656f,2.910156f,3.093750f,2.195312f,3.257812f,
3.441406f,2.013672f,3.908203f,3.003906f,4.484375f,3.447266f,3.679688f,3.451172f,3.902344f,3.197266f,3.738281f,2.355469f,4.003906f,3.587891f,3.496094f,3.187500f,3.531250f,3.236328f,3.871094f,3.789062f,2.984375f,3.835938f,3.742188f,3.052734f,2.925781f,3.921875f,2.964844f,3.671875f,3.486328f,3.500000f,
3.027344f,3.728516f,3.380859f,2.539062f,3.490234f,3.929688f,2.703125f,4.625000f,2.990234f,4.312500f,4.699219f,2.718750f,3.603516f,1.753906f,4.226562f,4.703125f,2.761719f,3.283203f,3.816406f,2.705078f,3.957031f,3.695312f,3.615234f,3.812500f,3.308594f,3.705078f,3.294922f,4.015625f,3.261719f,3.554688f,
3.703125f,3.730469f,3.986328f,3.810547f,3.923828f,3.875000f,4.125000f,3.535156f,3.947266f,3.841797f,3.259766f,4.679688f,3.156250f,4.042969f,3.244141f,3.861328f,3.207031f,2.722656f,4.246094f,3.652344f,3.607422f,3.734375f,3.728516f,3.375000f,4.214844f,3.525391f,4.199219f,3.441406f,3.578125f,3.615234f,
3.408203f,3.406250f,3.933594f,3.636719f,3.632812f,3.216797f,3.630859f,3.541016f,3.658203f,3.783203f,2.623047f,3.937500f,3.916016f,4.355469f,3.238281f,4.144531f,4.523438f,3.179688f,4.964844f,5.234375f,4.804688f,3.980469f,4.289062f,4.054688f,4.605469f,3.152344f,4.046875f,5.058594f,3.636719f,4.269531f,
4.679688f,3.742188f,4.035156f,4.238281f,4.242188f,4.277344f,4.570312f,5.406250f,3.779297f,4.500000f,4.593750f,5.394531f,4.000000f,4.437500f,4.187500f,4.296875f,3.505859f,4.781250f,5.328125f,4.296875f,6.007812f,4.410156f,4.667969f,5.058594f,5.234375f,4.902344f,4.640625f,5.804688f,4.273438f,5.855469f,
5.078125f,5.195312f,4.871094f,4.675781f,4.949219f,5.074219f,5.070312f,5.378906f,5.089844f,5.449219f,5.042969f,5.609375f,5.035156f,5.210938f,4.960938f,5.125000f,4.878906f,4.996094f,4.753906f,5.464844f,4.757812f,5.550781f,4.757812f,4.757812f,4.484375f,4.621094f,2.263672f,1.809570f,0.160400f,-0.333740f,
2.033203f,-0.309570f,0.066101f,1.336914f,-0.138184f,2.250000f,1.026367f,1.199219f,1.159180f,-0.004223f,0.087280f,-0.468506f,1.366211f,1.798828f,0.358643f,0.828613f,2.402344f,1.388672f,1.247070f,1.176758f,0.403076f,0.746094f,1.938477f,2.386719f,2.281250f,0.455566f,1.378906f,1.000977f,2.429688f,0.508301f,
2.250000f,0.449951f,2.015625f,0.590820f,1.978516f,2.492188f,0.434326f,1.969727f,2.097656f,2.134766f,0.501953f,1.747070f,0.293945f,2.322266f,0.226318f,0.916992f,1.721680f,2.087891f,0.868164f,1.672852f,1.239258f,1.918945f,1.352539f,1.388672f,0.837402f,2.138672f,2.298828f,0.791016f,1.375977f,1.151367f,
1.914062f,0.199463f,1.939453f,1.675781f,0.474609f,1.734375f,0.547363f,1.837891f,0.359375f,1.912109f,1.992188f,1.980469f,0.348877f,2.148438f,1.216797f,1.076172f,0.998047f,0.816406f,1.487305f,0.498535f,0.850586f,0.029755f,1.880859f,0.490479f,1.589844f,1.812500f,1.821289f,1.704102f,-0.257324f,0.373779f,
0.518066f,2.335938f,0.262939f,2.175781f,0.289062f,0.879395f,2.220703f,1.337891f,0.281250f,0.680176f,1.306641f,0.760254f,2.304688f,1.112305f,2.134766f,1.730469f,2.339844f,1.042969f,2.095703f,2.714844f,0.944336f,2.361328f,0.143799f,2.302734f,1.438477f,1.500000f,1.557617f,1.696289f,1.488281f,2.025391f,
0.595703f,2.134766f,2.201172f,2.310547f,1.061523f,2.246094f,0.561035f,1.991211f,2.480469f,0.996582f,2.589844f,2.326172f,0.778809f,1.388672f,2.359375f,1.094727f,2.117188f,1.928711f,1.085938f,1.130859f,2.792969f,2.507812f,0.377441f,1.208984f,1.720703f,-0.177002f,1.911133f,0.755371f,2.845703f,2.992188f,
0.981934f,1.423828f,0.986816f,2.958984f,3.037109f,0.927734f,1.509766f,2.074219f,0.464355f,2.785156f,0.587891f,2.976562f,2.957031f,2.404297f,0.948242f,0.851562f,2.457031f,0.580566f,2.439453f,2.195312f,2.556641f,0.682129f,2.541016f,0.749023f,1.435547f,2.302734f,1.071289f,2.480469f,2.392578f,0.712891f,
1.776367f,0.816895f,2.457031f,1.261719f,2.224609f,1.266602f,1.571289f,1.894531f,1.876953f,1.713867f,2.490234f,2.634766f,1.414062f,2.066406f,1.701172f,3.052734f,2.820312f,1.349609f,3.150391f,2.730469f,2.927734f,1.180664f,1.141602f,2.140625f,0.874023f,2.875000f,2.195312f,1.397461f,2.517578f,1.266602f,
2.660156f,2.869141f,3.064453f,1.117188f,2.765625f,2.828125f,1.079102f,2.798828f,2.929688f,2.496094f,1.493164f,2.638672f,1.035156f,2.585938f,0.854980f,1.666016f,3.021484f,1.697266f,3.068359f,3.052734f,1.408203f,1.581055f,2.136719f,1.609375f,2.121094f,2.898438f,2.955078f,1.252930f,2.335938f,3.042969f,
2.837891f,1.732422f,3.166016f,1.558594f,3.261719f,0.937500f,1.768555f,2.460938f,0.714355f,2.611328f,1.291992f,2.085938f,3.882812f,3.873047f,3.416016f,1.422852f,1.835938f,0.895996f,1.759766f,3.925781f,3.939453f,3.537109f,1.649414f,2.130859f,3.423828f,1.529297f,3.933594f,3.671875f,1.885742f,1.718750f,
3.541016f,1.608398f,2.316406f,3.578125f,1.711914f,3.843750f,3.589844f,3.982422f,1.774414f,4.144531f,2.064453f,3.791016f,4.007812f,4.242188f,2.384766f,1.018555f,1.435547f,0.012131f,-2.548828f,1.640625f,-2.408203f,-2.507812f,2.056641f,-1.939453f,1.445312f,-1.032227f,-0.247070f,-0.467041f,-1.289062f,-0.558594f,
-3.683594f,0.831055f,2.919922f,-0.070862f,0.421143f,2.603516f,0.806152f,0.411377f,0.052216f,0.219116f,0.034210f,1.177734f,1.883789f,2.044922f,0.209106f,2.287109f,1.513672f,2.941406f,1.299805f,2.443359f,0.299561f,2.216797f,0.773926f,2.222656f,3.376953f,0.209473f,2.134766f,2.552734f,2.878906f,1.324219f,
3.115234f,1.087891f,3.511719f,0.591309f,1.522461f,2.943359f,2.832031f,1.482422f,2.839844f,2.914062f,3.521484f,2.509766f,2.357422f,1.656250f,2.724609f,3.005859f,1.291992f,2.748047f,2.322266f,2.521484f,0.992676f,2.312500f,1.949219f,0.279541f,2.005859f,0.353027f,1.391602f,-0.464600f,1.555664f,1.844727f,
2.099609f,-0.029755f,2.218750f,0.639648f,0.431641f,0.189453f,0.549805f,1.983398f,-0.693848f,0.537598f,-1.074219f,1.829102f,-0.162231f,1.761719f,1.324219f,1.227539f,1.088867f,-1.926758f,0.855469f,0.153442f,2.419922f,-0.073608f,2.373047f,-0.026443f,0.749023f,2.662109f,0.774902f,-0.213135f,0.400635f,1.439453f,
0.793945f,2.146484f,1.583984f,1.708008f,1.644531f,2.705078f,1.005859f,2.566406f,2.814453f,0.290283f,2.957031f,-1.555664f,1.688477f,0.412598f,1.293945f,1.484375f,2.712891f,2.203125f,2.544922f,0.933105f,2.271484f,2.322266f,2.380859f,1.166992f,2.251953f,0.902832f,3.607422f,2.857422f,1.947266f,3.306641f,
2.431641f,1.412109f,2.058594f,3.640625f,2.042969f,3.263672f,1.565430f,1.279297f,1.369141f,4.015625f,3.755859f,-0.175659f,2.445312f,0.378662f,-1.545898f,3.714844f,0.937500f,4.109375f,4.367188f,1.642578f,2.781250f,1.776367f,4.523438f,4.554688f,0.598145f,1.001953f,2.644531f,-0.003372f,3.468750f,0.852539f,
3.435547f,3.277344f,2.169922f,0.731445f,1.154297f,3.277344f,0.200073f,2.537109f,2.900391f,3.234375f,0.711426f,2.787109f,0.857910f,1.604492f,3.523438f,1.128906f,3.660156f,4.125000f,0.133667f,2.152344f,0.613281f,3.873047f,1.415039f,3.861328f,2.287109f,1.468750f,2.476562f,2.587891f,1.401367f,3.798828f,
3.132812f,1.929688f,3.505859f,2.490234f,4.117188f,3.406250f,1.936523f,3.814453f,2.951172f,3.324219f,1.198242f,1.782227f,3.496094f,1.474609f,2.718750f,3.029297f,2.121094f,4.488281f,2.779297f,3.197266f,3.474609f,3.560547f,0.647461f,3.369141f,3.671875f,1.083984f,4.125000f,4.285156f,4.449219f,1.774414f,
2.998047f,1.756836f,4.371094f,1.051758f,2.121094f,4.738281f,2.890625f,4.613281f,4.542969f,2.333984f,2.714844f,3.826172f,3.003906f,3.666016f,5.296875f,5.531250f,2.990234f,4.238281f,5.464844f,3.388672f,3.669922f,5.515625f,2.919922f,4.613281f,0.854492f,4.035156f,2.191406f,1.013672f,5.855469f,2.849609f,
3.921875f,6.406250f,6.375000f,5.480469f,3.433594f,3.986328f,1.378906f,3.777344f,6.421875f,6.421875f,5.679688f,3.929688f,3.812500f,5.484375f,3.757812f,6.468750f,5.871094f,4.179688f,4.433594f,6.539062f,3.974609f,4.808594f,6.902344f,4.253906f,6.578125f,6.898438f,7.203125f,4.656250f,6.734375f,4.789062f,
7.375000f,7.378906f,7.566406f,6.332031f,0.024353f,1.458008f,0.303955f,-1.641602f,1.210938f,-1.487305f,-1.795898f,1.218750f,-1.247070f,0.600098f,-0.500977f,-0.290771f,-0.101501f,-1.016602f,0.608887f,-1.779297f,1.468750f,2.550781f,-0.416748f,-0.596191f,1.507812f,0.774902f,0.909180f,1.129883f,0.002480f,0.737793f,
1.639648f,2.298828f,2.253906f,0.345459f,1.537109f,1.339844f,2.435547f,0.892578f,2.410156f,0.766113f,2.541016f,0.814941f,2.013672f,2.957031f,0.596191f,2.251953f,2.419922f,2.572266f,0.921387f,2.408203f,0.507324f,2.490234f,0.442627f,1.011719f,2.212891f,2.058594f,0.355225f,1.595703f,0.612305f,2.623047f,
0.391357f,2.320312f,0.129517f,2.275391f,2.187500f,-0.073792f,1.563477f,1.131836f,2.476562f,-0.084839f,3.007812f,2.626953f,-0.159790f,2.449219f,-0.190796f,2.019531f,-0.422607f,2.259766f,2.343750f,2.664062f,-0.599121f,1.872070f,0.835449f,0.918945f,1.234375f,1.969727f,1.770508f,0.473389f,2.144531f,-0.410889f,
2.347656f,-0.422852f,2.259766f,2.007812f,1.954102f,1.670898f,-1.379883f,2.031250f,-0.133545f,2.486328f,-0.895996f,2.546875f,-0.775879f,0.062561f,2.199219f,2.388672f,-0.708496f,0.145508f,1.199219f,0.411621f,2.005859f,0.545410f,1.472656f,0.654297f,2.375000f,0.738770f,1.968750f,1.915039f,-0.093628f,1.951172f,
-0.789551f,1.329102f,0.445557f,0.850098f,1.182617f,1.459961f,1.236328f,2.095703f,0.417236f,2.050781f,1.984375f,2.478516f,0.158447f,1.901367f,-0.379639f,2.166016f,2.248047f,0.096985f,2.265625f,1.763672f,0.102417f,0.479980f,2.111328f,0.399658f,1.829102f,0.615234f,0.377930f,0.752441f,2.330078f,2.318359f,
-0.227905f,1.563477f,-0.433838f,-0.835449f,2.365234f,0.307617f,2.189453f,1.998047f,0.898438f,1.541016f,0.477295f,2.548828f,2.259766f,0.175659f,1.135742f,2.056641f,0.162231f,2.724609f,0.042389f,2.582031f,2.763672f,2.255859f,0.693359f,0.513672f,2.650391f,0.535645f,2.234375f,2.488281f,2.431641f,0.588867f,
2.091797f,0.526855f,1.321289f,2.197266f,0.881836f,2.218750f,2.865234f,0.541016f,1.528320f,0.629883f,2.347656f,0.958496f,2.500000f,0.917480f,0.756348f,0.816406f,2.472656f,1.702148f,2.587891f,2.396484f,0.374023f,1.729492f,1.247070f,3.056641f,2.300781f,0.638184f,2.927734f,2.380859f,2.648438f,0.664062f,
0.456055f,2.009766f,0.682617f,2.183594f,2.980469f,1.227539f,2.697266f,1.623047f,2.679688f,2.412109f,2.728516f,0.980957f,2.619141f,2.748047f,0.609863f,2.816406f,2.560547f,2.458984f,1.879883f,2.785156f,1.347656f,2.500000f,1.191406f,2.496094f,2.849609f,1.608398f,2.839844f,2.876953f,0.525879f,1.237305f,
2.111328f,1.291016f,1.993164f,2.832031f,3.076172f,1.187500f,2.947266f,2.894531f,2.638672f,1.704102f,2.748047f,1.618164f,2.380859f,0.573730f,2.154297f,0.935547f,0.092896f,3.070312f,1.376953f,2.316406f,3.060547f,3.281250f,2.906250f,1.905273f,2.384766f,0.991699f,2.365234f,3.210938f,3.363281f,2.937500f,
1.711914f,2.244141f,2.738281f,1.831055f,3.011719f,2.730469f,2.017578f,1.834961f,2.925781f,1.895508f,2.187500f,2.855469f,1.904297f,2.531250f,2.406250f,2.501953f,2.000000f,1.925781f,2.001953f,1.864258f,1.782227f,1.840820f,2.845703f,1.429688f,1.303711f,1.175781f,-1.539062f,1.744141f,-1.559570f,-1.451172f,
2.876953f,-0.493408f,1.224609f,-1.494141f,-0.854492f,-1.287109f,0.081909f,1.019531f,-1.795898f,0.289795f,2.337891f,-1.803711f,-1.640625f,1.041992f,-1.413086f,-0.541992f,-0.864746f,-0.309570f,-0.537598f,0.751465f,1.610352f,1.393555f,-0.680176f,1.068359f,-0.450439f,1.978516f,0.058563f,1.586914f,-0.977051f,0.832520f,
-0.570801f,0.722656f,2.730469f,-0.934082f,1.426758f,1.852539f,2.148438f,-0.697266f,2.085938f,-0.391602f,2.998047f,-0.169556f,-0.407471f,2.361328f,1.415039f,-0.314941f,1.085938f,0.335449f,2.689453f,-0.196899f,1.660156f,0.357666f,2.517578f,2.509766f,-0.744629f,1.534180f,-0.055756f,2.365234f,-0.550293f,2.492188f,
2.267578f,0.146118f,2.287109f,0.224365f,1.269531f,-0.216309f,1.656250f,2.017578f,2.468750f,-0.245850f,1.902344f,-0.417725f,0.209229f,-0.116272f,1.251953f,2.449219f,0.203857f,2.050781f,0.663086f,2.990234f,0.100830f,2.251953f,1.939453f,2.287109f,2.316406f,0.043152f,2.523438f,0.036957f,2.400391f,0.004890f,
3.171875f,0.045105f,-0.145752f,3.240234f,1.240234f,0.082153f,0.185059f,2.166016f,0.224854f,2.416016f,0.424805f,1.439453f,0.924805f,2.435547f,-0.267090f,2.257812f,1.795898f,-0.345215f,2.675781f,-0.409668f,1.851562f,-0.618164f,0.025406f,-0.266846f,1.580078f,0.109375f,2.572266f,-0.383545f,1.630859f,2.382812f,
2.470703f,0.175537f,1.734375f,0.051880f,2.712891f,2.378906f,-0.317139f,3.228516f,1.670898f,-0.372803f,-0.242065f,2.636719f,-0.413818f,2.666016f,-0.366699f,0.458984f,0.441406f,3.152344f,2.695312f,0.072632f,3.218750f,-0.659668f,0.846191f,4.125000f,1.048828f,3.117188f,3.021484f,1.947266f,2.142578f,0.685547f,
3.005859f,2.966797f,0.845703f,0.770020f,2.468750f,-0.023041f,2.207031f,0.129150f,2.171875f,2.769531f,1.446289f,0.478760f,1.019531f,2.626953f,0.075317f,1.727539f,2.330078f,1.913086f,1.129883f,1.782227f,0.633789f,0.689453f,2.841797f,0.158813f,2.802734f,3.591797f,0.387695f,2.091797f,1.100586f,3.642578f,
0.613770f,2.830078f,1.041016f,0.159180f,2.138672f,2.894531f,1.009766f,3.660156f,1.826172f,-0.207153f,2.279297f,0.460449f,3.781250f,2.085938f,0.363037f,2.656250f,2.392578f,2.550781f,0.710938f,1.345703f,2.648438f,0.861816f,1.549805f,2.220703f,1.017578f,3.550781f,0.799316f,1.492188f,2.197266f,2.201172f,
0.489990f,2.724609f,3.070312f,0.607422f,3.642578f,3.568359f,3.330078f,1.254883f,2.431641f,1.712891f,3.994141f,1.291992f,1.970703f,4.097656f,1.531250f,3.189453f,3.515625f,1.585938f,1.361328f,3.193359f,1.357422f,1.632812f,4.464844f,5.410156f,1.383789f,3.755859f,5.367188f,2.695312f,1.603516f,4.074219f,
1.711914f,3.226562f,1.596680f,4.488281f,0.933105f,2.058594f,5.238281f,2.027344f,1.792969f,3.572266f,4.125000f,3.281250f,3.023438f,3.609375f,2.117188f,3.759766f,4.031250f,4.527344f,3.621094f,2.388672f,2.306641f,4.023438f,2.527344f,4.542969f,3.634766f,2.091797f,2.632812f,4.605469f,1.843750f,2.345703f,
4.964844f,1.992188f,4.324219f,4.601562f,4.273438f,3.210938f,4.578125f,3.097656f,5.199219f,5.019531f,5.066406f,3.437500f,0.697266f,1.823242f,-1.395508f,-3.296875f,1.303711f,-3.609375f,-1.872070f,2.296875f,-0.902832f,2.166016f,-1.221680f,-1.007812f,-1.458008f,-2.033203f,-1.433594f,-2.710938f,0.354248f,1.578125f,
-3.578125f,-2.099609f,1.771484f,-1.651367f,-1.182617f,-1.525391f,-1.369141f,-0.993652f,0.965820f,1.904297f,1.568359f,-1.460938f,0.897949f,-1.291992f,1.912109f,-0.987793f,1.785156f,-1.498047f,0.907715f,-1.366211f,0.864746f,2.597656f,-1.390625f,1.808594f,1.854492f,2.085938f,-1.353516f,1.859375f,-2.855469f,2.593750f,
-2.580078f,-1.255859f,1.680664f,1.667969f,-2.125000f,0.654785f,-1.260742f,2.162109f,-1.125977f,0.690918f,-0.446533f,2.359375f,2.818359f,-0.740234f,1.116211f,-0.915527f,1.436523f,-2.107422f,1.738281f,1.430664f,-0.619629f,1.242188f,-0.542969f,0.946289f,-0.777344f,1.623047f,1.687500f,1.729492f,-2.083984f,1.908203f,
-1.333008f,-0.919922f,-1.412109f,-0.265381f,2.378906f,-0.542480f,1.599609f,0.285156f,2.601562f,-0.299561f,1.675781f,1.595703f,1.807617f,2.105469f,-1.427734f,1.666016f,-0.315674f,2.734375f,-1.886719f,2.943359f,-1.956055f,-0.808594f,3.070312f,0.118042f,-2.017578f,-0.414795f,1.938477f,-0.510254f,2.134766f,-0.680664f,
1.350586f,1.219727f,2.281250f,-1.062500f,1.860352f,1.995117f,-2.250000f,2.474609f,-1.940430f,2.433594f,-0.590820f,-0.483154f,-0.964355f,1.358398f,-0.547363f,2.343750f,-1.765625f,1.215820f,2.072266f,1.637695f,-0.136841f,2.093750f,-1.739258f,2.111328f,1.946289f,-0.747070f,2.974609f,2.265625f,-0.522461f,-0.701172f,
2.462891f,-1.083008f,2.251953f,0.015297f,0.448730f,-0.137451f,2.898438f,1.978516f,-1.675781f,0.982910f,0.929199f,-1.437500f,2.695312f,-0.412842f,2.812500f,3.232422f,-1.346680f,0.687988f,-2.548828f,2.398438f,2.910156f,-0.483643f,0.298828f,1.707031f,-1.133789f,1.675781f,-1.926758f,2.560547f,2.457031f,0.936523f,
-0.621582f,-0.515625f,2.158203f,-0.527344f,1.409180f,1.795898f,2.181641f,0.126831f,1.509766f,-0.468750f,-0.236084f,2.519531f,-0.841309f,2.621094f,3.062500f,-1.920898f,1.271484f,-0.566895f,3.070312f,-0.488525f,1.364258f,-0.524902f,-0.936035f,1.773438f,0.823242f,-0.218506f,2.253906f,0.911133f,-0.718262f,1.541992f,
-0.380615f,2.701172f,1.890625f,-0.678223f,2.562500f,1.555664f,2.068359f,-0.767578f,-0.607422f,1.917969f,-1.732422f,0.978027f,0.770020f,-0.549805f,2.359375f,-2.486328f,-0.046936f,1.458984f,2.085938f,-0.710938f,2.345703f,2.369141f,-1.621094f,2.785156f,3.250000f,2.822266f,-0.099060f,1.609375f,0.228149f,3.185547f,
-0.946777f,-0.581055f,3.109375f,-0.111755f,2.091797f,2.228516f,-1.491211f,0.047821f,2.386719f,0.116211f,0.059479f,3.144531f,3.574219f,-1.112305f,2.037109f,3.388672f,1.417969f,-0.316650f,2.199219f,-0.157227f,1.991211f,-0.976074f,1.543945f,1.220703f,-1.188477f,3.041016f,-0.254150f,-0.062988f,2.945312f,2.736328f,
1.290039f,-1.488281f,1.137695f,-1.318359f,1.462891f,2.960938f,2.695312f,1.250977f,-0.490967f,0.073669f,1.938477f,0.195923f,2.910156f,1.387695f,-0.697754f,-0.455811f,2.824219f,-0.390625f,-0.338623f,2.357422f,-0.443115f,2.218750f,2.591797f,2.845703f,0.040405f,2.457031f,-0.119873f,3.613281f,3.330078f,3.207031f,
-0.616699f,1.718750f,1.358398f,0.140625f,1.524414f,1.070312f,1.875000f,1.535156f,1.100586f,1.479492f,1.808594f,1.708984f,1.555664f,1.372070f,0.779297f,0.066406f,0.441406f,1.334961f,1.640625f,2.552734f,2.304688f,2.654297f,2.656250f,2.574219f,2.634766f,2.150391f,1.911133f,1.796875f,2.398438f,2.392578f,
2.634766f,2.611328f,2.734375f,2.269531f,2.710938f,2.310547f,2.781250f,2.156250f,3.324219f,1.853516f,2.224609f,2.423828f,1.745117f,2.455078f,2.101562f,2.972656f,2.175781f,2.722656f,2.091797f,2.707031f,3.238281f,2.406250f,2.267578f,2.693359f,3.396484f,3.642578f,2.861328f,3.248047f,2.878906f,2.144531f,
2.724609f,3.259766f,2.322266f,3.164062f,3.087891f,2.980469f,1.918945f,3.611328f,2.972656f,2.626953f,3.156250f,2.578125f,2.505859f,1.750977f,2.535156f,3.119141f,2.621094f,1.711914f,3.220703f,3.109375f,2.984375f,3.201172f,2.585938f,2.876953f,1.979492f,2.111328f,1.844727f,2.810547f,2.445312f,2.851562f,
2.210938f,2.505859f,1.988281f,0.594238f,1.551758f,2.302734f,2.654297f,1.427734f,2.457031f,1.584961f,2.423828f,2.353516f,2.822266f,1.423828f,2.521484f,1.900391f,3.126953f,2.812500f,3.570312f,2.148438f,3.253906f,2.519531f,3.994141f,2.972656f,3.039062f,2.843750f,3.056641f,1.827148f,2.902344f,3.074219f,
3.269531f,3.441406f,3.185547f,3.263672f,2.771484f,2.757812f,3.275391f,2.500000f,2.935547f,2.880859f,2.582031f,2.230469f,2.892578f,2.603516f,2.136719f,2.197266f,2.128906f,2.333984f,2.337891f,2.539062f,2.867188f,2.128906f,3.042969f,3.027344f,2.697266f,2.634766f,2.341797f,2.140625f,2.279297f,2.794922f,
1.818359f,2.478516f,3.208984f,2.759766f,3.080078f,3.652344f,3.429688f,4.000000f,3.296875f,3.378906f,2.718750f,2.808594f,2.908203f,2.400391f,3.306641f,2.365234f,2.808594f,2.847656f,2.998047f,3.390625f,3.277344f,2.957031f,2.539062f,2.554688f,2.671875f,2.617188f,2.720703f,2.248047f,2.712891f,2.708984f,
2.433594f,3.001953f,2.253906f,2.195312f,2.318359f,3.044922f,3.218750f,2.470703f,4.078125f,3.173828f,3.476562f,2.382812f,2.675781f,1.912109f,2.550781f,2.474609f,3.085938f,2.652344f,3.150391f,2.972656f,3.326172f,3.080078f,3.269531f,3.468750f,3.087891f,3.382812f,3.320312f,3.136719f,2.927734f,2.501953f,
3.451172f,3.511719f,4.058594f,3.376953f,4.343750f,4.300781f,2.935547f,3.345703f,3.404297f,3.039062f,3.630859f,3.267578f,3.642578f,3.843750f,3.728516f,3.544922f,3.341797f,3.611328f,2.740234f,3.031250f,3.949219f,3.228516f,4.449219f,3.126953f,3.445312f,2.445312f,3.251953f,2.585938f,3.501953f,3.544922f,
3.013672f,3.164062f,2.607422f,3.119141f,3.078125f,4.031250f,4.503906f,3.384766f,4.152344f,2.822266f,2.472656f,2.910156f,3.935547f,2.248047f,3.505859f,3.859375f,3.951172f,3.363281f,3.410156f,3.539062f,3.634766f,3.416016f,2.869141f,3.605469f,3.660156f,3.669922f,3.763672f,3.656250f,3.326172f,3.492188f,
3.617188f,3.410156f,3.537109f,3.976562f,3.824219f,3.269531f,3.585938f,4.007812f,2.750000f,3.562500f,3.142578f,2.931641f,3.013672f,3.955078f,2.431641f,4.218750f,2.378906f,2.562500f,2.816406f,6.058594f,2.156250f,2.023438f,0.524902f,-1.704102f,3.251953f,-2.259766f,-1.584961f,2.718750f,-0.467529f,2.925781f,
1.528320f,1.386719f,0.850586f,-1.099609f,-0.822266f,-1.865234f,2.652344f,3.562500f,-1.367188f,-1.245117f,1.600586f,-0.114014f,-0.019318f,-0.086365f,-1.634766f,-0.019302f,0.364990f,1.383789f,1.390625f,0.995117f,0.680176f,0.967285f,1.334961f,0.187134f,1.202148f,-0.019348f,1.035156f,0.340820f,0.421875f,2.095703f,
0.006859f,1.860352f,2.568359f,2.835938f,1.472656f,2.220703f,0.667969f,3.107422f,0.987305f,1.551758f,2.986328f,3.089844f,2.150391f,3.208984f,1.881836f,2.835938f,1.190430f,1.523438f,1.559570f,2.966797f,3.738281f,3.054688f,2.742188f,2.703125f,2.185547f,0.501465f,2.773438f,1.912109f,1.650391f,1.651367f,
1.459961f,0.759766f,1.277344f,1.207031f,1.536133f,1.601562f,1.411133f,1.702148f,0.947266f,0.829590f,0.401123f,-0.117554f,1.013672f,0.354248f,0.676270f,2.015625f,2.375000f,2.634766f,1.877930f,1.336914f,1.658203f,1.901367f,2.000000f,1.712891f,2.822266f,2.660156f,2.490234f,3.082031f,2.080078f,1.837891f,
3.097656f,2.085938f,1.458008f,2.207031f,2.800781f,2.683594f,3.263672f,2.425781f,2.339844f,2.441406f,3.195312f,1.574219f,2.345703f,2.166016f,-0.030945f,1.937500f,-0.216187f,1.950195f,2.246094f,2.410156f,1.897461f,1.484375f,1.356445f,1.668945f,-0.251953f,1.893555f,1.561523f,2.029297f,1.765625f,1.931641f,
2.402344f,2.328125f,1.884766f,1.439453f,2.660156f,2.367188f,2.353516f,2.878906f,3.277344f,1.528320f,3.191406f,2.697266f,2.597656f,3.705078f,4.679688f,4.093750f,1.876953f,2.197266f,2.742188f,1.723633f,5.011719f,3.082031f,5.281250f,5.593750f,3.230469f,3.746094f,0.371582f,3.730469f,4.105469f,0.601074f,
3.257812f,4.015625f,3.109375f,4.535156f,2.160156f,4.125000f,4.441406f,3.291016f,2.982422f,2.294922f,3.693359f,1.832031f,2.845703f,3.593750f,3.894531f,3.601562f,3.156250f,3.142578f,3.685547f,4.109375f,2.873047f,4.261719f,4.554688f,1.953125f,4.015625f,3.164062f,5.527344f,4.312500f,4.906250f,3.802734f,
3.429688f,3.869141f,5.269531f,5.457031f,5.570312f,5.222656f,4.957031f,5.156250f,4.984375f,5.921875f,4.929688f,3.742188f,5.414062f,4.398438f,5.222656f,4.777344f,4.324219f,4.269531f,3.974609f,4.851562f,3.984375f,4.570312f,4.566406f,2.679688f,3.019531f,2.451172f,3.376953f,2.111328f,3.611328f,3.919922f,
3.582031f,3.683594f,3.869141f,3.783203f,3.279297f,3.923828f,3.386719f,4.339844f,3.628906f,4.351562f,5.109375f,4.500000f,4.351562f,4.445312f,3.367188f,3.933594f,4.902344f,4.996094f,5.453125f,6.187500f,5.902344f,4.257812f,6.308594f,6.875000f,5.882812f,5.000000f,6.582031f,3.900391f,5.976562f,4.417969f,
5.355469f,5.535156f,5.207031f,7.109375f,5.410156f,6.421875f,7.265625f,7.082031f,6.093750f,2.986328f,4.894531f,3.482422f,6.394531f,7.925781f,7.699219f,6.757812f,4.957031f,6.714844f,7.902344f,6.917969f,8.296875f,7.632812f,7.011719f,6.566406f,8.828125f,7.496094f,8.046875f,9.492188f,7.761719f,9.734375f,
10.296875f,10.593750f,8.507812f,10.476562f,8.953125f,11.554688f,12.296875f,12.203125f,10.101562f,};


static void run_softmax_bfyx_opt(const int64_t b, const int64_t f, const int64_t y, const int64_t x, const uint64_t axis) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc softmax_bf_kernel = {format::bfyx, "softmax_gpu_bf"};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"softmax", softmax_bf_kernel}}));

    const int64_t buf_size = b * f * y * x;
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), f, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                        data_types::f16, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f16, format::bfyx};

    std::string softmax_id = "softmax";
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax(softmax_id, input_info("input"), axis));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    auto input_mem = engine.allocate_memory(input_layout_static);

    // auto input_data = rg.generate_random_1d<ov::float16>(buf_size, -20, 20);
    set_values(input_mem, input_data);

    std::map<cldnn::primitive_id, cldnn::network_output> outputs;
    cldnn::memory::ptr output = nullptr;

    network->set_input_data("input", input_mem);
    outputs = network->execute();
    output = outputs.at(softmax_id).get_memory();
    ASSERT_NE(output, nullptr);

    // std::vector<ov::float16> output_ref(buf_size);
    // ov::reference::softmax<ov::float16>(input_data.data(), output_ref.data(), input_layout_static.get_shape(), ov::AxisSet{axis});
    ASSERT_NE(output, nullptr);
    const float threshold_fp16 = 0.0005f;
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());
    for (size_t idx = 0; idx < static_cast<size_t>(buf_size); idx++) {
        if ((std::abs(float(output_ptr[idx])) - float(output_ref[idx])) > threshold_fp16) {
            std::cout << idx << ", " << std::fixed << setprecision(6) << output_ptr[idx] << " vs " << output_ref[idx];
            ASSERT_FALSE(1);
        }
        // ASSERT_NEAR(float(output_ptr[idx]), float(output_ref[idx]), threshold_fp16) << idx << ", " << std::fixed << setprecision(8) << output_ptr[idx] << " vs " << output_ref[idx];
    }
}

TEST(softmax_gpu_bfyx_f16, opt_softmax_bf_axis_3) {
    run_softmax_bfyx_opt(1, 4, 2, 3083, 3);
}

TEST(softmax_gpu_bfyx_f16, opt_softmax_bf_axis_3_01) {
    run_softmax_bfyx_opt(1, 32, 1, 289, 3);
}
