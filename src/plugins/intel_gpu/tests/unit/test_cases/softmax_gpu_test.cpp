// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

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

std::vector<ov::float16> input_data = {
1.892578,1.790039,0.380371,0.936035,
2.115234,1.270508,0.835449,0.274658,1.670898,1.614258,-0.825684,2.117188,-0.067139,1.004883,1.163086,-0.656250,1.956055,2.271484,2.658203,1.503906,1.643555,1.382812,1.022461,1.482422,
0.037262,1.689453,1.521484,1.229492,0.105042,0.665039,1.093750,0.885742,0.331299,0.910156,1.163086,0.052338,0.830566,1.935547,0.446777,0.432617,-1.138672,0.402588,-0.905762,0.367432,
1.407227,1.034180,1.019531,0.803223,-0.474365,1.024414,1.427734,1.098633,-0.045837,1.449219,1.598633,0.855469,1.429688,1.002930,0.944336,1.552734,2.210938,1.975586,1.556641,1.842773,
0.467041,0.358643,0.567871,1.002930,1.691406,1.958008,1.992188,1.793945,1.176758,1.400391,-0.767578,2.482422,1.987305,1.586914,0.921875,0.116272,1.916992,2.048828,1.045898,1.593750,
0.245361,0.157837,1.507812,0.522949,0.602051,-1.426758,0.007580,1.144531,0.831543,2.359375,1.207031,2.470703,1.616211,1.351562,1.208008,1.728516,-0.638672,0.455811,0.461426,0.820801,
0.539062,1.071289,1.500977,0.745605,0.976562,0.570312,1.158203,3.087891,3.253906,-0.047791,0.944336,2.341797,-0.756836,2.156250,0.420410,2.697266,2.144531,-1.208984,1.065430,1.600586,
0.044891,-0.009232,1.501953,0.325928,1.814453,0.016769,1.419922,0.151611,0.286133,1.075195,1.466797,-0.548828,1.044922,2.109375,1.767578,1.474609,1.357422,-0.250000,0.082764,0.965820,
1.447266,1.939453,-0.558105,1.410156,-1.143555,2.257812,1.796875,2.687500,2.658203,0.183228,0.971680,0.634277,3.099609,1.275391,2.333984,2.736328,0.642578,2.507812,1.286133,1.485352,
1.381836,0.067627,0.866699,0.476807,1.962891,2.167969,2.160156,0.925781,1.805664,1.339844,-0.391846,1.574219,0.683594,1.456055,0.757324,0.277344,0.423828,1.030273,0.784180,0.312744,
0.427979,1.216797,1.099609,0.929199,0.890625,0.433105,1.540039,-1.565430,1.442383,1.960938,1.029297,2.531250,1.452148,1.167969,1.138672,-0.187622,0.912598,1.306641,0.981934,1.094727,
1.220703,0.201172,1.597656,1.643555,0.682617,1.473633,0.326416,1.411133,2.789062,2.654297,2.072266,1.262695,1.341797,-0.513184,1.249023,0.581543,1.262695,0.887695,-0.857422,0.491699,
-0.670898,-0.586914,0.238647,-0.494141,1.096680,0.798828,-0.310791,0.655273,-0.844238,1.197266,0.127686,1.237305,1.784180,2.417969,1.775391,1.600586,1.085938,0.833008,1.369141,2.156250,
2.654297,2.785156,2.142578,1.094727,0.403076,-0.497314,2.283203,2.783203,2.880859,3.312500,2.410156,1.886719,2.392578,2.931641,3.177734,2.902344,3.208984,1.237305,2.646484,2.636719,
3.380859,1.932617,1.175781,2.189453,2.039062,2.253906,3.326172,1.377930,2.628906,2.267578,2.673828,2.220703,2.353516,0.251221,0.843262,-0.270264,2.337891,1.987305,2.630859,2.519531,
2.101562,2.089844,0.341309,2.230469,1.096680,0.687012,2.343750,1.240234,1.541016,2.240234,2.535156,2.533203,2.634766,2.173828,0.964355,0.299805,2.855469,2.316406,2.357422,2.369141,
1.600586,2.912109,1.682617,1.479492,0.690430,-0.313477,0.350830,0.489258,1.203125,1.317383,0.431885,1.074219,1.260742,0.277832,0.001854,1.239258,0.804688,0.263672,0.104065,-0.435059,
-1.243164,0.945312,0.155396,0.346191,0.260498,-0.029999,1.057617,0.428467,0.887695,-0.608887,-0.354004,0.490234,0.020142,0.041809,1.338867,0.621094,-0.031708,0.571777,0.003557,0.520508,
0.557129,1.569336,1.303711,0.373291,0.590820,0.112915,1.152344,0.738770,1.231445,0.005173,0.029419,0.664551,-0.398438,-0.853516,-0.491699,-0.665039,-1.242188,0.274658,1.083008,1.874023,
1.499023,0.974121,-0.054291,-1.423828,-1.107422,-1.057617,-0.484863,0.921875,0.051483,0.895020,-0.389648,0.625977,0.895020,-0.409180,-1.227539,-0.320312,-0.520508,0.130615,1.369141,0.781250,
0.706543,-0.555664,1.526367,1.519531,1.182617,0.938965,0.580566,0.236938,0.262451,0.867676,1.686523,0.797852,2.234375,1.998047,0.114685,-0.628418,-0.127808,-0.042206,-0.432129,0.136353,
0.942871,0.932617,0.792969,2.462891,1.788086,0.271729,0.461914,1.187500,1.215820,1.701172,1.556641,1.177734,0.765625,0.205933,0.640137,0.561523,1.369141,1.680664,0.703125,0.286377,
-0.916504,0.724609,-0.426758,1.269531,0.240723,1.725586,0.293213,1.424805,1.612305,1.231445,1.091797,0.181152,1.309570,1.478516,2.212891,1.985352,1.319336,0.911621,-0.382812,0.513184,
-0.796387,0.995605,1.250000,0.934082,0.427246,0.681152,0.979004,0.949707,1.026367,1.697266,1.265625,0.518066,0.381348,0.532227,-0.137207,-0.740234,-0.362061,0.528320,-0.484619,0.593262,
0.818359,0.306152,-0.104431,-0.539062,-0.534668,0.244629,1.014648,0.576172,0.408447,0.359619,1.212891,1.152344,0.367188,0.490967,0.873535,0.706543,1.102539,0.860352,1.880859,0.682617,
1.168945,1.272461,0.038574,1.506836,2.183594,1.264648,1.264648,1.363281,0.535156,1.442383,1.557617,2.119141,2.566406,2.009766,1.891602,0.800293,0.204224,0.547363,1.220703,0.898926,
1.169922,0.326416,1.361328,1.051758,1.361328,1.039062,0.528809,0.424805,0.424805,1.501953,2.000000,1.849609,1.021484,0.377197,1.116211,0.925293,1.268555,1.502930,0.801758,1.022461,
1.208008,0.891602,0.499023,2.261719,2.035156,2.859375,2.587891,3.091797,2.939453,2.242188,1.897461,1.490234,-0.754883,2.062500,2.953125,2.289062,2.574219,2.052734,2.105469,1.862305,
1.564453,1.365234,2.333984,1.461914,1.498047,-0.982422,1.786133,1.339844,2.117188,1.375977,1.239258,1.759766,1.502930,1.588867,1.169922,0.941895,0.442383,1.073242,-0.065247,1.794922,
2.146484,1.248047,1.909180,0.906250,0.587402,-1.332031,2.484375,2.195312,2.015625,1.377930,2.269531,2.181641,0.428223,0.171997,0.515137,-0.391357,2.050781,1.503906,1.789062,2.285156,
2.753906,2.306641,0.969238,1.562500,1.456055,1.775391,2.070312,0.213745,0.387695,-1.054688,-1.824219,2.697266,2.238281,2.443359,1.705078,1.415039,1.048828,0.924316,-0.625488,1.574219,
0.920410,-0.596191,0.910156,1.037109,1.315430,0.709473,0.475342,0.699219,-0.880859,1.206055,0.620117,2.224609,1.249023,0.829590,0.340820,1.405273,1.282227,1.450195,1.024414,1.032227,
-0.788574,0.095642,-0.572266,2.429688,1.980469,2.207031,1.705078,2.546875,2.230469,1.353516,1.011719,1.424805,-0.362061,0.990234,0.239746,3.435547,3.080078,2.158203,2.796875,1.800781,
0.731934,1.405273,2.636719,3.021484,2.162109,1.118164,2.359375,0.198730,1.230469,1.830078,2.314453,2.357422,2.232422,1.919922,1.496094,1.647461,1.591797,2.015625,1.380859,2.208984,
2.708984,3.035156,1.930664,1.234375,0.960938,-1.070312,1.622070,1.155273,3.900391,4.148438,2.531250,2.292969,2.263672,2.585938,1.902344,2.107422,1.969727,-0.197754,1.645508,1.209961,
2.347656,2.564453,3.787109,1.325195,2.576172,3.025391,3.166016,2.736328,1.650391,2.095703,0.732910,2.589844,2.435547,3.812500,3.787109,3.056641,1.419922,1.462891,1.756836,1.179688,
2.537109,1.642578,2.714844,3.603516,2.824219,3.605469,2.162109,3.910156,4.082031,4.460938,1.973633,3.771484,3.019531,3.029297,2.937500,3.000000,3.074219,4.191406,3.865234,3.720703,
2.769531,2.490234,0.794434,0.117554,4.488281,4.097656,4.632812,4.554688,4.679688,4.554688,3.650391,2.453125,2.130859,1.773438,2.224609,3.216797,3.462891,2.775391,1.409180,1.536133,
1.464844,0.503418,2.335938,2.097656,1.891602,3.214844,2.529297,3.017578,3.039062,3.517578,3.154297,2.755859,2.445312,1.693359,2.458984,1.654297,1.043945,1.398438,0.748535,2.197266,
2.214844,1.591797,4.140625,0.752441,1.465820,1.371094,0.765625,0.959961,1.876953,1.731445,2.949219,2.089844,1.939453,2.005859,1.742188,0.742188,0.577148,0.641113,0.007713,0.896973,
0.701660,-0.057953,-0.057678,-0.273682,-0.421143,0.740234,0.640137,-0.247437,-0.011627,-0.597656,-0.305420,0.539062,1.790039,1.622070,1.628906,2.574219,-0.480225,0.027481,-0.748535,0.361084,
-0.197388,0.869629,0.881348,1.312500,2.035156,2.341797,2.287109,1.509766,0.914551,1.842773,0.717773,2.269531,1.620117,0.696289,0.894043,0.950684,1.768555,1.482422,2.488281,2.808594,
2.572266,1.892578,0.533203,1.708008,1.170898,2.115234,1.443359,2.183594,1.602539,1.426758,1.671875,1.427734,0.357178,0.309814,0.384277,1.257812,1.847656,2.294922,1.684570,1.880859,
0.488037,0.175903,-0.232056,1.938477,2.625000,1.612305,2.048828,1.154297,2.519531,2.820312,2.822266,2.058594,2.458984,1.454102,2.857422,2.638672,0.389404,1.870117,1.759766,-0.087585,
3.160156,2.724609,3.775391,2.589844,1.586914,1.165039,2.304688,2.615234,2.654297,2.279297,1.744141,2.595703,1.521484,1.686523,0.767578,1.548828,1.538086,0.610352,1.083008,0.482422,
1.466797,1.623047,1.822266,0.358154,0.728027,-0.137939,0.936523,0.502441,0.578613,1.532227,0.316650,0.542480,1.420898,1.963867,2.003906,2.398438,0.771484,1.305664,-0.242798,0.576172,
2.005859,1.586914,0.791504,1.811523,2.351562,2.449219,2.310547,1.701172,0.031799,1.345703,0.257812,1.297852,1.962891,2.035156,1.243164,0.018204,-0.058929,0.017410,0.843750,1.174805,
1.242188,0.972168,1.250000,-0.273438,0.606445,0.420898,0.685059,1.620117,1.927734,1.133789,1.469727,1.009766,0.391113,-0.536621,-0.165039,0.486328,0.711914,2.027344,1.517578,0.968262,
1.039062,0.516602,0.770508,0.197021,0.816895,0.098816,0.070312,0.920410,0.246948,0.069824,1.361328,2.732422,2.158203,1.796875,1.994141,0.982422,0.929199,0.290771,0.333740,-1.363281,
0.543457,1.038086,1.324219,3.044922,2.001953,0.174805,0.856934,1.400391,-0.020645,1.735352,1.370117,-0.411377,1.206055,-0.297852,0.160034,1.040039,1.003906,-0.771484,-0.440918,-0.953613,
-0.676270,-0.306396,-0.651367,0.466797,0.000113,-0.305176,0.323242,1.904297,1.938477,1.625977,-1.610352,-0.828613,-1.728516,0.203735,-0.519531,1.467773,0.415283,-0.070679,2.275391,1.789062,
-0.982910,0.653320,-0.667969,2.066406,2.447266,2.685547,1.114258,2.054688,0.707031,0.842285,1.337891,1.321289,0.157227,0.919434,0.930664,-0.170654,0.493164,-0.376953,0.038330,2.460938,
-0.914062,-2.015625,-2.304688,1.053711,0.677734,1.117188,1.552734,0.173584,0.542969,1.612305,1.156250,1.853516,-0.433594,0.499023,-0.222534,0.702637,-1.903320,0.107727,1.651367,0.703125,
-0.227173,0.441406,0.929688,1.035156,-0.014336,1.907227,0.823242,1.716797,0.572266,1.793945,2.070312,1.651367,1.490234,-0.300537,0.459229,1.159180,1.613281,1.104492,1.416992,-0.872070,
1.160156,1.433594,0.382324,2.234375,1.883789,0.579590,-2.191406,-1.021484,0.454590,0.453613,0.676758,0.542480,0.226807,0.426514,0.241821,0.068481,0.454102,-0.451660,0.450195,-0.227783,
-0.993652,0.502930,1.094727,0.343994,0.152222,1.140625,0.598145,1.614258,1.736328,0.686523,-0.777832,-0.926270,0.962891,0.706055,0.276611,1.533203,1.504883,-0.000359,-2.166016,-0.426758,
1.179688,0.926758,1.493164,1.376953,1.358398,0.756836,1.761719,0.872559,1.083984,-0.582031,-1.596680,0.031830,-0.599609,0.358887,0.326416,-0.465332,-1.029297,-0.015060,-0.151001,-0.100952,
0.340576,-0.139282,0.426758,0.516602,0.107788,-0.003796,-0.330566,0.941895,0.838379,1.022461,1.101562,0.707031,0.151978,0.881836,-0.049377,0.873047,-1.517578,2.056641,0.137695,2.291016,
2.236328,0.972656,-1.366211,-0.350830,-0.683594,2.019531,2.001953,1.310547,0.075562,0.594727,1.116211,2.109375,2.017578,1.147461,1.286133,0.185913,-0.810059,0.935059,0.527344,-1.177734,
0.979004,0.970215,1.672852,2.009766,1.554688,1.313477,0.541992,0.518555,-0.510742,1.144531,1.530273,1.041992,0.664062,0.423096,0.640625,-0.274658,1.032227,0.782715,-0.373291,0.132202,
-1.600586,1.646484,2.082031,2.125000,1.143555,1.333008,1.557617,1.514648,1.511719,1.614258,1.379883,0.216797,-0.265381,1.110352,1.839844,1.089844,1.789062,1.217773,1.028320,0.955566,
0.344238,-2.074219,1.856445,1.133789,-1.089844,0.951172,0.852539,0.616211,1.184570,-0.593750,-1.572266,-1.736328,-0.704590,-0.294922,0.177368,0.065552,0.174194,-0.401367,-1.029297,0.287842,
0.381592,0.466309,0.394043,0.207764,-0.444824,-0.219238,0.201172,-0.094421,0.612305,0.351318,0.708984,0.942383,0.379639,-0.329102,0.764160,1.035156,-0.338135,-0.792969,0.692383,0.811523,
1.303711,1.440430,0.695801,1.414062,1.516602,1.375977,1.355469,0.038940,-0.620117,-0.720703,0.412109,-0.268555,-0.909180,0.144287,-0.560059,-0.760742,-0.306152,-1.050781,-1.635742,0.985352,
-0.272949,0.929199,1.208008,1.812500,2.095703,1.659180,1.173828,0.091125,-0.660156,-0.454590,0.003380,1.208984,0.830566,-0.015541,-0.043549,-0.598633,0.973633,0.866211,0.635254,0.954102,
1.121094,-0.359131,1.659180,0.890625,1.180664,3.310547,2.816406,0.613281,1.419922,0.741211,1.748047,1.992188,1.556641,0.853027,2.074219,2.566406,3.107422,2.234375,2.160156,2.023438,
1.714844,1.401367,-0.808105,1.984375,0.700684,1.164062,2.154297,1.901367,1.372070,0.468262,0.496582,0.752441,1.424805,1.096680,-0.208130,1.399414,1.151367,1.241211,-0.232178,1.746094,
1.764648,1.052734,2.058594,1.985352,0.191772,1.003906,1.101562,-0.013260,0.661133,0.556152,0.921387,1.635742,1.494141,0.682617,-0.163330,-0.662598,0.617676,-0.924316,-2.599609,1.826172,
1.932617,0.683594,0.413574,-1.120117,-0.998047,1.025391,0.735352,-0.570801,-0.028076,-1.669922,-0.596191,-0.515137,-1.070312,-1.859375,-1.792969,-2.390625,-1.828125,-0.489014,-1.198242,-0.922363,
-2.406250,-2.179688,-0.095459,-1.001953,0.678711,-0.416748,-1.709961,-2.785156,-3.449219,-2.261719,-0.644531,-0.412354,0.018509,-0.189575,-0.986328,-0.008270,-0.599121,-1.710938,-0.450195,-1.612305,
-0.408691,-1.286133,-0.973145,-2.005859,-1.834961,-1.645508,-2.328125,-1.070312,-0.355469,-1.024414,-1.253906,-1.549805,-3.492188,-1.721680,-2.646484,-1.313477,-1.291016,-0.604004,-0.853027,-2.027344,
-2.087891,-1.637695,-1.212891,-1.760742,-2.449219,-2.751953,-2.699219,-0.748047,-0.860840,-1.884766,-2.693359,-1.409180,-2.238281,-1.790039,-3.537109,-1.948242,-0.769531,-0.408447,-3.412109,-0.592285,
-0.290771,-1.425781,-2.152344,-2.556641,-1.617188,-2.777344,-3.019531,-2.189453,-2.222656,-2.304688,-1.695312,-1.235352,-0.459229,-1.715820,-2.302734,-1.514648,-1.252930,-2.251953,-4.531250,-1.780273,
-0.683594,-1.541992,-1.197266,-1.656250,-3.267578,-2.884766,-3.330078,-1.709961,-1.156250,-1.638672,-3.312500,-4.890625,-3.451172,-2.466797,-2.625000,-3.328125,-2.839844,-2.894531,-4.933594,-3.179688,
-3.789062,-3.191406,-1.931641,-2.173828,-1.934570,-1.241211,-2.205078,-2.257812,-3.132812,-3.357422,-4.019531,-2.951172,-1.745117,-0.661133,-0.873535,-1.766602,-2.679688,-2.437500,-2.558594,-1.508789,
-0.625488,-0.701660,-1.500977,-0.190186,-0.559570,-2.851562,-0.869629,-1.937500,-2.117188,-0.553223,-1.088867,-1.952148,-1.613281,-1.910156,-2.355469,-4.566406,-0.525879,-1.044922,-1.863281,-1.382812,
-3.314453,-1.975586,-1.485352,-2.220703,-1.965820,-1.416992,-1.360352,-1.656250,-1.103516,-1.563477,-1.493164,-2.412109,-1.764648,-1.983398,-2.558594,-0.700195,-1.025391,-1.428711,-3.349609,-1.191406,
-1.034180,-0.826660,0.450684,-0.743652,-0.896484,-2.630859,-1.445312,-3.580078,-1.717773,-1.570312,-1.327148,-1.968750,-2.205078,-1.716797,-1.307617,-1.632812,-2.447266,-4.957031,-2.453125,-1.939453,
-1.695312,-1.558594,-2.492188,-2.386719,-3.984375,-3.884766,-3.646484,-2.380859,-2.253906,-3.158203,-3.855469,-4.035156,-2.009766,-3.207031,-3.458984,-3.857422,-2.589844,-1.772461,-2.781250,-1.283203,
-1.322266,-1.794922,-3.505859,-1.669922,-1.349609,-0.498779,-1.411133,-1.996094,-1.469727,-2.054688,-3.218750,-1.177734,-1.349609,-2.597656,-3.291016,-2.248047,-3.126953,-1.426758,-2.705078,-3.283203,
-2.228516,-2.921875,-3.386719,-2.617188,-2.935547,-4.113281,-3.412109,-3.884766,-2.339844,-2.320312,-1.249023,-1.367188,-3.240234,-1.757812,-1.665039,-2.906250,-1.465820,-3.011719,-3.054688,-3.314453,
-5.035156,-1.297852,-0.532715,-1.346680,-2.392578,-0.241943,-1.290039,-0.756836,-2.449219,-1.380859,-0.327881,0.040070,-0.558105,-0.007519,-3.777344,-0.149536,0.240723,-1.795898,-2.126953,-1.471680,
-1.907227,-1.681641,-0.633301,-1.334961,-3.234375,-2.873047,-3.273438,-1.902344,-1.662109,-0.498535,-0.391602,-1.836914,-2.335938,-2.312500,-1.487305,-2.080078,-2.822266,-2.771484,-2.759766,-4.757812,
-2.042969,-1.592773,-0.249390,-0.984375,-1.127930,-1.215820,-1.190430,-2.832031,-0.562988,-1.904297,-1.566406,-1.875000,-1.102539,-2.609375,-2.033203,-0.251465,-1.698242,-0.025864,-0.129028,-3.099609,
0.146973,0.019958,-0.595215,-0.156738,0.819336,-1.646484,1.030273,-0.032135,-0.472412,-0.554688,-0.975098,-2.291016,-2.054688,-1.492188,-0.044922,-2.128906,-2.134766,-1.187500,-1.830078,-2.626953,
-2.302734,-0.179810,-1.808594,-2.085938,0.096741,-0.896484,-1.145508,-0.988770,0.062012,-0.759277,-1.681641,-1.025391,-0.232544,-2.667969,-0.371338,-2.914062,-1.751953,-0.047577,0.968262,-0.089233,
1.212891,1.025391,-1.930664,0.564453,-1.306641,0.563477,-1.146484,-1.289062,-1.107422,-0.978516,0.480469,-0.576172,-1.456055,-0.766113,-1.188477,-2.742188,-0.791504,-0.471436,0.042816,0.780273,
-1.026367,-0.472656,0.162964,0.930176,1.167969,0.639648,-0.432617,-1.958984,-0.948730,-0.988770,-0.883789,-0.800293,0.314209,0.217285,-0.572754,-1.813477,-0.965332,0.944824,0.787109,-0.441895,
0.961426,1.930664,0.542969,0.838867,0.300293,1.081055,0.897461,0.605957,0.048615,0.764648,-0.599121,-0.625977,0.629883,-0.587402,-0.269043,1.452148,-0.383057,-0.679688,-1.505859,-2.072266,
1.385742,0.154785,-0.736328,-3.734375,-3.416016,1.063477,1.800781,0.332520,0.536133,-1.111328,-4.261719,-1.949219,-2.857422,-2.583984,-1.148438,0.422852,-0.297607,0.125610,-1.925781,-0.651367,
-0.061646,-2.341797,-2.734375,-2.064453,-2.865234,-1.651367,-0.843262,-1.838867,-1.859375,-1.816406,-0.615234,-0.805664,-1.069336,-0.824707,-3.277344,-0.892578,-1.178711,-1.080078,-0.397461,0.086548,
-1.111328,-1.065430,-0.210327,-1.288086,-0.995117,-1.177734,-0.386475,-2.507812,-0.976074,-1.486328,-1.663086,-2.974609,-2.833984,-1.917969,-1.150391,-2.992188,-0.898926,-1.845703,-1.807617,-2.146484,
-2.228516,-1.170898,-1.684570,-1.492188,-1.141602,-1.478516,-2.816406,-2.955078,-2.894531,-2.072266,-1.904297,-1.144531,-1.547852,-1.183594,-0.462158,-1.221680,-0.629883,-1.639648,-1.761719,-0.839844,
-0.829102,-2.208984,-2.236328,-1.698242,-1.975586,-2.203125,-2.685547,-1.480469,-1.716797,-1.535156,-2.064453,-2.253906,-1.587891,-2.265625,-1.860352,-2.058594,-1.661133,-2.326172,-2.738281,-1.316406,
0.008347,-2.332031,-1.470703,-4.265625,-2.595703,-1.761719,-0.097778,-1.769531,-0.323730,-2.568359,-5.019531,-4.578125,-5.597656,-3.060547,-4.054688,-3.052734,-1.749023,-2.767578,-1.408203,-3.158203,
-3.123047,-3.093750,-1.681641,-4.359375,-4.289062,-2.353516,-1.689453,-2.519531,-1.916016,-2.777344,-1.852539,-3.035156,-1.501953,-0.641113,-1.928711,-2.009766,-3.054688,-1.649414,0.083374,1.050781,
0.013542,-0.173950,-1.395508,-2.380859,-1.253906,-1.981445,-1.708984,-3.769531,-2.212891,-0.950195,-1.653320,-1.333984,-2.880859,-2.142578,-1.655273,-0.469727,-2.976562,-0.021210,-2.580078,-2.386719,
-3.828125,-1.741211,-2.431641,-1.780273,-1.527344,-1.102539,-0.548340,-1.599609,-2.521484,-3.615234,-0.393311,-2.609375,0.009430,-0.598145,0.074646,-1.368164,-1.141602,-1.335938,-0.437744,0.021301,
-2.375000,-2.748047,-0.132935,-1.059570,-1.028320,-0.218262,-2.318359,-0.585449,-4.140625,0.064087,-2.279297,-2.544922,-2.347656,-0.426514,-0.281982,-1.436523,-1.279297,-1.468750,-0.054596,-1.205078,
-0.957520,-0.073792,-2.652344,-3.138672,-2.603516,-2.804688,-2.542969,-0.854004,-0.246948,-2.460938,-2.224609,-1.912109,-2.904297,-2.787109,-4.007812,-2.009766,-0.966309,-1.479492,-1.424805,-1.900391,
-0.091125,-1.377930,-1.798828,-1.952148,-2.015625,-2.072266,-0.826660,-1.781250,-2.173828,-4.640625,-2.904297,-1.006836,-1.504883,-1.541016,-2.498047,-2.318359,-1.392578,-0.190796,-2.044922,-0.724609,
-0.311035,-0.384766,0.298340,-2.591797,-1.491211,-3.605469,-2.701172,-0.355469,-1.665039,-0.618164,-0.614258,-1.241211,-0.666992,-1.950195,-0.178467,-0.999512,-3.111328,-4.050781,-3.509766,0.549805,
-2.710938,-2.779297,-4.472656,-0.624512,-1.601562,-0.976562,-1.166992,-2.656250,0.021301,-2.775391,-2.292969,-3.179688,0.333740,-1.195312,-0.446289,-0.355469,0.896973,1.834961,-1.003906,1.622070,
1.127930,0.644531,1.378906,-0.575684,1.387695,1.463867,0.650391,-4.066406,-0.932617,-2.066406,-2.101562,-1.511719,-0.524414,-1.033203,-4.035156,-0.151367,1.036133,-1.554688,-1.762695,-2.091797,
1.415039,0.687012,0.535156,-0.846680,-1.684570,-3.550781,-0.798828,-0.919434,-0.298096,0.044983,-1.327148,-1.832031,-1.674805,-0.845215,-0.357422,-1.541992,0.387695,-1.272461,0.833496,-2.062500,
-1.922852,-1.347656,0.495605,0.660156,1.759766,0.279785,0.056061,-0.419189,0.465088,0.480713,-0.815430,-0.524902,0.491699,-1.529297,-0.030075,0.692383,-1.309570,-1.462891,0.943359,-0.173340,
1.130859,1.186523,0.066223,0.066895,0.016830,-0.386963,0.048889,-1.083984,-0.875000,-0.551758,-1.675781,0.539062,-0.399658,-0.122192,-2.337891,0.044617,-1.825195,-1.183594,-1.216797,0.502930,
-0.939941,-2.050781,-2.281250,-0.847656,0.426514,-1.251953,-0.441162,-0.305664,-1.700195,-0.659668,-3.625000,0.203491,-1.391602,-1.556641,-0.092163,-1.476562,-0.455811,0.997559,-0.483154,-0.157471,
-0.854004,-0.926758,-0.862305,-0.587402,0.241943,-0.747070,-0.097778,-2.119141,-1.005859,-0.194214,-1.941406,-1.741211,-3.197266,-1.385742,-2.589844,-2.199219,-1.836914,0.032410,0.340820,-1.280273,
-0.028519,-0.781250,-0.215210,-2.076172,-1.326172,-0.194214,-2.150391,-1.464844,-0.872559,0.306152,-1.019531,-1.772461,-2.173828,-0.765625,-0.526855,0.398193,0.304688,-0.076355,0.522949,0.832031,
-0.143921,-0.310547,0.994629,-0.426514,0.418457,-0.654297,-0.114807,-0.941406,-1.361328,-1.395508,-1.303711,-0.316162,-1.119141,-1.989258,0.687500,0.522949,-0.594238,-2.507812,-4.550781,-1.575195,
0.792969,0.555664,-1.065430,-1.221680,-2.392578,-2.642578,-2.566406,-2.203125,-1.371094,-2.175781,-2.261719,-0.786133,-0.611816,-0.120239,-0.755371,-0.596680,-3.277344,-0.719727,0.093018,0.371338,
0.594727,0.072083,-0.757324,-1.848633,-0.681152,0.035583,0.822266,-1.023438,-1.041016,-1.796875,-1.903320,0.361816,-1.107422,-0.972168,-0.578125,0.560059,-0.648438,-0.061462,0.283447,-1.028320,
-3.111328,-2.177734,-2.896484,-2.468750,-1.176758,-0.698242,-2.306641,-3.496094,-2.027344,-2.357422,-1.413086,-1.339844,-1.462891,-0.960938,-2.710938,-1.808594,-0.907715,-0.143555,-0.646973,-0.819824,
-1.018555,-0.881348,-1.295898,-0.812012,-2.130859,-0.426758,-2.439453,-1.207031,-0.616211,0.001805,-0.402832,-0.902832,-0.700684,-1.037109,-0.419678,-0.745117,0.942383,0.642090,0.801270,-0.197021,
0.463867,0.224243,0.490234,0.757324,-0.067871,-0.874512,-2.087891,-2.224609,-0.015961,-1.743164,-2.191406,-2.107422,-0.703125,-0.113281,0.819824,0.384277,-0.866699,-1.761719,-0.616699,-1.060547,
-0.545410,0.031082,-0.232178,-1.574219,-2.048828,-0.427246,0.403564,-0.631348,-0.165649,0.216064,0.103821,0.767090,0.490234,-0.405029,0.000749,0.184082,-0.553711,-0.375000,-1.860352,0.668457,
0.665527,-0.246460,-0.061951,-0.524902,0.229004,0.783691,0.734863,1.482422,-1.076172,-0.127197,-1.246094,-1.298828,-3.542969,-0.315186,-1.290039,-0.169189,1.212891,0.988770,0.713867,-0.325928,
-1.848633,-1.301758,-1.900391,-1.338867,-1.397461,0.583984,-0.330811,-0.643555,-2.554688,-1.374023,-1.335938,0.058624,0.187866,0.183105,-1.738281,-0.696289,0.845215,-0.159912,-0.891602,-1.153320,
-0.199463,-1.451172,-1.750977,-1.429688,-0.724121,-1.905273,-1.825195,0.194336,0.991699,0.218018,1.818359,1.217773,0.063232,0.779297,-1.798828,0.544434,0.994629,-0.726562,-0.090942,0.816895,
-1.632812,0.033386,-0.949707,-0.423096,0.074768,-0.750000,-1.680664,-0.160522,-0.794434,-0.642090,-0.390869,-1.420898,-2.000000,-0.468994,-0.893066,-0.839355,0.170776,-2.744141,-1.083008,-0.472168,
-1.370117,-0.823730,-0.971680,-0.437256,-0.343018,-1.991211,-1.981445,0.402832,-3.501953,-0.706543,0.387939,0.406982,0.891113,0.870605,0.921875,-1.235352,-0.886230,0.705566,-0.441895,-1.295898,
-1.356445,-0.752441,-1.745117,-2.601562,-0.272705,-0.258301,-1.554688,-1.064453,-0.900391,-0.486572,-1.698242,-2.876953,-0.612305,-1.581055,-0.666504,-2.218750,-1.566406,-2.013672,-2.685547,-1.749023,
-0.981445,-0.437744,-2.460938,-0.839844,-1.726562,-1.324219,0.178711,-2.111328,-2.636719,-2.017578,-1.064453,-0.115479,0.251709,-1.221680,-1.865234,-1.038086,0.367676,-0.618164,0.853027,0.344727,
0.028824,-0.031891,0.021835,0.612305,0.249390,-1.085938,-3.232422,-1.708008,-1.424805,1.060547,0.260742,-0.345459,0.786133,-1.496094,0.613281,-2.632812,0.667480,0.825195,1.787109,1.904297,
-0.009392,0.071167,0.582031,-0.028152,-0.800293,-0.221924,-2.703125,0.018707,0.230957,-0.354004,0.386475,0.723145,0.907227,0.347412,0.334717,1.607422,0.640137,0.489258,-0.958008,-1.108398,
-0.019150,1.675781,0.851562,2.095703,2.017578,1.389648,0.563477,-0.698730,0.137085,0.294434,-0.036377,1.958008,0.909180,1.333008,2.269531,0.950195,0.805664,1.626953,0.255371,-0.870605,
1.577148,1.194336,0.714844,-0.178955,-2.574219,-2.208984,-1.398438,0.545898,-0.553711,0.789551,-0.682129,-1.642578,-0.328125,-2.453125,-1.031250,-0.746094,0.344238,-0.280273,0.965332,0.686523,
-0.200562,-1.196289,-0.529297,-0.503906,-0.192993,-0.669922,-0.771973,-0.038177,-0.657227,-0.833984,-1.515625,-1.351562,-1.946289,-0.182617,1.207031,-2.720703,-1.031250,-2.183594,-0.399414,-0.042297,
-0.293701,-1.307617,-0.877441,-0.322021,0.349854,-2.025391,-1.230469,-0.716309,0.289307,-0.682129,-0.216919,-2.509766,-1.533203,-1.691406,-2.609375,-2.695312,-3.160156,-0.890137,-1.172852,-1.392578,
-0.782715,-2.203125,-1.728516,-1.125000,-3.207031,-2.355469,-3.146484,-2.277344,-2.396484,0.720703,-0.626465,0.764160,-0.023193,0.292236,-0.281006,1.048828,2.300781,1.606445,1.442383,-0.018036,
-0.345703,-2.277344,0.145386,1.202148,0.736328,1.302734,-2.072266,-0.663086,-2.074219,1.254883,1.291016,-0.004925,0.865234,-0.691406,2.773438,2.138672,0.404541,0.729980,-0.787598,0.151855,
0.859863,1.903320,0.325439,0.034637,-0.049408,1.841797,-0.190430,-0.265625,0.825195,0.997070,-0.098083,1.341797,-2.943359,-0.366211,0.709961,0.922852,1.325195,1.349609,0.791016,1.094727,
2.421875,2.492188,1.332031,-0.268311,0.412354,0.504395,1.252930,1.576172,1.251953,0.469238,3.187500,2.515625,1.219727,2.134766,3.644531,2.861328,0.403564,1.532227,3.917969,1.520508,
2.468750,0.808594,2.103516,-1.599609,0.053650,1.417969,2.359375,0.628418,0.749023,0.206421,0.933594,1.754883,1.803711,0.710449,-0.409180,-2.042969,-0.227539,1.736328,-0.524414,1.026367,
1.330078,2.265625,0.688477,1.204102,0.897949,-0.152344,1.377930,0.312988,3.007812,2.109375,-0.008896,1.363281,2.154297,2.279297,3.091797,1.123047,1.487305,2.003906,2.253906,0.900879,
-0.565918,0.448730,1.436523,0.527344,1.383789,0.203979,1.351562,0.921387,0.036407,-0.780762,0.531250,1.311523,-0.652832,0.794434,0.701660,2.179688,2.929688,1.755859,-0.585938,-1.920898,
-0.922363,-0.078308,0.211548,0.376465,-0.349365,-2.005859,-1.354492,0.597656,1.041992,2.103516,1.547852,0.119202,-0.706055,-0.374023,0.495361,0.588379,2.267578,2.095703,2.808594,3.417969,
1.216797,1.140625,2.269531,0.943848,0.234009,1.187500,2.662109,2.080078,3.632812,1.254883,2.728516,2.669922,2.939453,4.273438,3.185547,2.097656,0.903809,1.434570,1.910156,0.025681,
0.988770,0.160034,3.019531,1.502930,1.116211,2.105469,0.516602,2.398438,1.711914,-0.326172,0.833984,3.451172,3.222656,3.439453,3.177734,1.850586,0.392334,0.039124,-0.763184,1.335938,
1.310547,1.001953,0.843750,0.660645,0.839355,1.788086,2.722656,0.968262,1.616211,3.378906,2.339844,4.492188,0.482422,0.976562,3.054688,3.890625,4.449219,4.359375,4.054688,2.474609,
0.693848,1.720703,2.892578,3.478516,3.054688,3.398438,1.930664,3.173828,2.058594,2.070312,-0.467529,1.934570,1.060547,0.035492,1.890625,2.484375,0.832031,0.767578,-0.200562,1.395508,
2.697266,1.704102,2.126953,1.968750,0.487549,2.365234,1.679688,1.415039,3.414062,1.876953,3.279297,3.306641,2.316406,2.162109,2.736328,4.207031,3.097656,2.873047,3.332031,3.335938,
2.697266,1.737305,1.072266,2.287109,2.187500,2.093750,3.302734,2.257812,3.769531,3.148438,3.392578,0.439697,0.614746,1.509766,2.724609,2.777344,2.349609,2.611328,0.526855,0.687988,
1.937500,0.910156,2.443359,1.583008,1.977539,1.626953,2.074219,0.728516,2.189453,0.337891,-0.738770,1.450195,3.056641,2.138672,1.340820,1.740234,1.607422,2.718750,0.599121,2.718750,
2.962891,2.820312,3.220703,0.367920,2.060547,3.644531,1.475586,2.623047,3.359375,2.394531,2.732422,4.144531,2.748047,3.582031,1.414062,1.385742,2.771484,1.191406,1.473633,1.261719,
2.207031,0.657715,2.011719,1.351562,1.881836,3.855469,3.548828,1.281250,3.296875,2.810547,2.421875,1.165039,2.201172,3.490234,0.836914,2.949219,3.876953,2.808594,2.292969,2.021484,
2.550781,3.685547,1.931641,3.466797,1.953125,1.030273,2.189453,3.978516,3.869141,3.656250,4.527344,3.332031,1.938477,3.130859,4.535156,1.878906,3.880859,4.718750,4.574219,4.054688,
4.761719,5.136719,4.359375,2.980469,0.558594,3.445312,4.656250,4.523438,4.148438,5.035156,4.300781,3.351562,4.410156,4.339844,3.503906,4.144531,3.445312,3.664062,2.941406,2.093750,
3.087891,4.214844,1.957031,3.281250,4.492188,4.457031,3.703125,3.990234,4.449219,3.382812,2.699219,3.746094,5.550781,3.484375,4.367188,4.355469,4.070312,1.287109,3.248047,3.914062,
5.250000,4.574219,4.804688,5.113281,5.261719,4.035156,5.078125,2.246094,3.294922,5.050781,3.734375,5.109375,2.886719,4.121094,4.660156,3.755859,4.121094,4.722656,3.982422,4.214844,
3.349609,1.551758,3.748047,2.441406,1.577148,4.324219,3.576172,3.853516,2.855469,4.472656,2.330078,4.117188,2.480469,3.886719,4.761719,4.898438,4.726562,4.855469,4.882812,3.667969,
3.771484,4.730469,4.421875,5.777344,5.398438,5.460938,5.515625,4.875000,5.359375,4.324219,5.765625,5.582031,6.089844,4.910156,5.414062,5.535156,3.916016,5.980469,5.718750,4.582031,
5.726562,7.070312,6.484375,5.871094,3.570312,5.265625,5.535156,6.277344,4.042969,6.839844,6.230469,4.699219,5.640625,6.632812,6.480469,5.511719,7.300781,6.269531,6.355469,4.750000,
5.667969,7.183594,6.492188,7.539062,6.070312,7.246094,8.335938,7.570312,7.992188,8.257812,5.136719,8.593750,6.785156,9.484375,8.468750,8.562500,-65504.000000,-65504.000000,-65504.000000,1.930664,
1.867188,-0.239990,0.409180,1.827148,0.987305,1.543945,1.230469,1.702148,1.648438,-0.436768,1.821289,0.262451,1.018555,1.681641,-0.257568,1.700195,1.967773,2.072266,1.086914,2.230469,
1.624023,1.212891,1.600586,-0.058929,1.360352,1.362305,1.025391,0.275635,1.234375,1.286133,1.088867,0.452148,1.068359,0.909180,-0.201538,0.310303,1.653320,0.590820,1.048828,0.349609,
0.897949,-0.831543,0.466553,1.392578,1.390625,1.336914,0.858887,-0.245850,0.848145,1.256836,1.101562,-0.023651,1.426758,1.742188,1.193359,1.513672,1.291016,1.270508,1.391602,1.757812,
1.568359,1.456055,2.068359,1.264648,0.716797,0.629395,0.586914,1.127930,1.497070,1.541016,1.611328,1.359375,1.543945,-0.420166,2.156250,1.824219,1.872070,1.155273,-0.137817,1.609375,
1.961914,1.425781,1.500977,0.995605,0.735840,1.370117,1.002930,1.157227,0.014832,0.325439,1.259766,0.782227,1.818359,0.473877,2.279297,1.793945,1.661133,1.236328,1.916992,-0.057159,
1.196289,0.415527,0.515137,0.173218,0.832031,1.929688,0.844238,1.581055,0.437988,1.406250,2.896484,2.746094,0.153564,1.688477,2.406250,0.133789,2.105469,0.537598,2.480469,2.435547,
-0.013550,1.466797,1.654297,-0.266602,-0.573242,1.350586,0.328613,1.559570,0.205200,1.641602,0.943359,0.995117,1.268555,1.438477,-0.278564,0.264160,1.889648,1.769531,1.880859,1.837891,
0.596191,-0.086731,1.314453,1.514648,1.956055,0.454346,1.686523,-0.639648,2.160156,1.570312,2.287109,2.484375,0.403320,0.899902,0.361084,2.916016,1.488281,2.439453,2.472656,0.495361,
1.944336,0.885254,1.676758,2.126953,1.240234,1.153320,0.518066,1.419922,1.487305,1.936523,0.702148,1.754883,1.419922,0.041504,1.415039,1.195312,1.560547,1.036133,0.817383,0.809570,
1.104492,0.829590,0.404297,0.996094,1.063477,1.008789,1.182617,1.350586,0.903320,1.804688,-0.566895,1.632812,1.737305,0.236938,2.316406,1.863281,1.482422,1.501953,0.475342,1.068359,
1.147461,0.961914,1.186523,1.166992,0.547363,1.480469,1.734375,1.002930,1.421875,0.884277,0.960449,2.011719,2.173828,1.958984,1.697266,1.432617,-0.161377,1.059570,0.415771,1.612305,
1.444336,0.297363,0.872070,-0.667480,-0.921875,-0.165283,-0.940430,0.802734,0.505371,-0.434570,0.690918,-0.445557,1.500977,0.362305,0.995605,1.477539,1.877930,1.569336,1.648438,1.077148,
1.094727,1.278320,1.679688,1.882812,2.148438,2.017578,0.660156,0.248535,-0.494141,1.644531,2.082031,2.291016,2.648438,2.181641,1.650391,2.005859,2.427734,2.687500,2.283203,2.326172,
0.218872,2.177734,2.253906,2.984375,1.925781,1.458984,1.798828,1.451172,1.147461,2.425781,0.442871,2.363281,1.975586,2.402344,1.759766,1.871094,0.721191,0.715332,0.087036,1.894531,
1.705078,2.166016,2.046875,1.854492,1.663086,0.228882,1.289062,0.425537,0.678223,2.271484,1.166992,0.773438,1.014648,1.625000,1.593750,2.083984,2.355469,1.217773,0.840332,2.611328,
1.991211,2.052734,1.820312,1.050781,2.332031,0.660156,1.254883,0.664062,0.272705,-0.044556,0.596191,0.482178,0.317139,0.211914,0.895020,1.528320,0.215942,-0.412842,0.390381,0.608887,
0.823242,1.177734,0.517090,-0.217285,0.935059,0.229126,0.022766,-0.049225,-0.125122,0.957520,0.365479,0.697266,-0.395508,0.211670,0.923828,-0.059723,-0.311523,0.551758,0.668945,0.708984,
1.375000,0.584961,0.733887,0.218872,1.378906,1.265625,0.364746,0.522949,-0.138306,0.924316,0.351807,0.844238,0.468262,0.588867,1.114258,-0.128052,-0.802246,-0.412598,-0.434082,-1.192383,
-0.202759,0.570801,1.110352,1.022461,1.336914,0.622070,-0.282715,-1.090820,-1.364258,-0.705566,0.626953,0.177368,0.645020,-0.137329,0.677246,1.099609,-0.168701,-1.056641,-0.322510,-0.317871,
0.152466,1.291016,1.085938,1.413086,0.381592,1.511719,1.326172,1.129883,0.762207,0.769531,0.365479,0.608398,0.289307,0.326904,0.324219,1.510742,2.140625,0.641113,-0.471436,-0.073059,
0.039154,-0.368652,0.108276,0.863770,0.310547,0.172852,2.332031,1.951172,0.978027,1.092773,1.353516,1.223633,1.667969,1.682617,1.659180,1.531250,0.829102,1.012695,0.391357,1.341797,
1.571289,1.222656,0.912598,0.152100,1.130859,0.189087,1.377930,0.622559,1.997070,0.898926,1.773438,1.945312,1.882812,2.046875,1.142578,1.520508,1.717773,2.080078,2.060547,1.738281,
1.959961,0.482422,0.927246,0.122253,1.133789,1.507812,1.490234,0.997070,1.211914,1.261719,0.620117,1.253906,1.891602,1.765625,1.084961,1.325195,1.332031,0.661621,0.224243,0.493164,
0.992188,-0.157104,0.787109,1.104492,1.084961,0.816406,0.850098,0.129150,0.346924,0.945801,0.665039,0.783691,0.675293,1.817383,1.753906,0.982422,1.075195,0.797363,1.249023,1.575195,
1.168945,1.672852,0.564453,1.463867,1.537109,0.267090,1.778320,2.080078,1.504883,1.608398,1.940430,1.289062,1.627930,1.263672,1.794922,1.602539,1.989258,2.142578,1.291016,1.126953,
1.023438,1.529297,1.084961,1.481445,0.635254,1.500977,1.020508,1.416992,1.578125,1.066406,0.870605,0.837402,1.397461,2.017578,1.509766,1.647461,0.991211,1.610352,1.417969,1.162109,
1.667969,1.125000,1.988281,1.552734,0.960449,0.877930,1.918945,1.993164,2.248047,2.531250,2.890625,2.806641,2.429688,2.177734,2.378906,0.290283,1.851562,2.501953,1.925781,2.070312,
2.414062,2.523438,2.048828,1.340820,1.618164,1.965820,1.642578,1.641602,-0.053925,1.859375,1.252930,1.967773,1.213867,1.497070,2.167969,1.831055,2.246094,1.541992,1.268555,0.927246,
1.666992,0.376953,1.657227,1.927734,1.323242,2.421875,1.719727,1.116211,-0.333496,2.041016,1.522461,1.587891,1.393555,2.414062,2.554688,0.914062,0.495117,0.839355,0.426270,2.072266,
1.600586,1.681641,2.003906,2.654297,2.181641,1.056641,1.275391,0.820312,1.259766,2.000000,1.117188,1.261719,-0.123840,-1.001953,2.314453,1.811523,2.275391,1.868164,1.785156,1.545898,
1.048828,0.092957,1.415039,0.713867,-0.568359,0.434082,0.957520,1.209961,0.845215,0.486572,1.069336,-0.304199,1.132812,0.376953,2.097656,1.303711,1.084961,0.195679,1.480469,0.933594,
1.015625,0.707031,1.460938,0.591797,0.784668,0.144409,2.171875,1.351562,1.403320,1.088867,2.117188,1.813477,1.000977,0.810547,1.772461,0.761230,1.276367,0.347656,2.871094,2.705078,
2.031250,2.798828,2.089844,1.730469,1.261719,2.125000,2.449219,2.212891,1.584961,2.406250,0.119263,1.402344,1.955078,2.298828,2.509766,2.333984,2.029297,1.193359,1.791016,1.708984,
1.942383,1.025391,1.668945,1.876953,1.646484,1.902344,1.101562,1.533203,-0.034180,1.450195,0.820801,3.167969,3.625000,2.171875,2.001953,1.873047,2.300781,1.536133,1.358398,1.858398,
0.360352,1.590820,1.132812,2.320312,2.318359,3.210938,0.983398,1.620117,2.183594,2.369141,2.277344,1.316406,2.173828,0.887207,2.099609,1.695312,3.150391,3.505859,2.931641,2.095703,
1.115234,0.913086,1.101562,1.764648,0.957520,1.633789,2.650391,1.842773,3.085938,2.177734,3.421875,3.833984,3.730469,1.311523,2.673828,2.187500,2.642578,2.462891,2.316406,1.860352,
2.968750,2.597656,3.273438,2.867188,2.220703,0.827637,0.195801,3.554688,3.136719,3.779297,3.779297,3.916016,3.777344,2.886719,2.130859,1.816406,1.516602,1.314453,2.478516,1.911133,
2.166016,1.099609,1.343750,1.188477,0.499512,1.797852,1.319336,1.171875,3.103516,2.529297,2.728516,2.619141,3.103516,2.878906,2.767578,2.494141,1.910156,2.605469,1.535156,0.987305,
1.236328,0.316406,1.588867,1.491211,1.010742,2.703125,0.809570,1.913086,1.931641,0.765137,0.402344,0.945312,1.328125,2.476562,2.332031,1.932617,2.093750,1.565430,0.663086,0.585938,
0.779297,-0.308594,0.801758,0.541992,0.000153,0.033356,-0.364014,-0.189697,0.797363,0.518555,-0.156128,0.133301,-0.180054,-0.171387,0.342041,1.004883,1.519531,1.592773,2.185547,0.138428,
0.786133,0.004429,0.519531,-0.013481,0.871094,0.838379,1.045898,1.770508,2.134766,2.138672,1.513672,1.301758,1.658203,0.729980,1.925781,1.083984,0.683105,0.874512,0.935547,1.412109,
0.815918,1.767578,2.324219,2.386719,1.864258,0.756348,1.435547,0.810059,1.536133,0.791992,1.788086,1.413086,1.288086,1.537109,1.686523,0.713379,0.781738,0.520508,0.975098,0.845703,
1.743164,1.500977,2.046875,1.018555,0.118713,-0.039581,1.356445,2.160156,1.559570,2.064453,1.256836,2.339844,2.445312,2.435547,1.943359,2.300781,1.708984,2.580078,2.406250,0.936035,
1.834961,1.663086,0.110657,2.375000,1.981445,3.105469,2.429688,1.721680,1.613281,2.152344,2.240234,2.011719,1.529297,1.356445,1.986328,1.635742,1.683594,1.016602,1.542969,1.667969,
0.778809,1.107422,0.422852,1.388672,1.477539,1.722656,0.442139,0.890137,0.027954,0.578125,0.301270,0.878906,2.171875,1.069336,0.858398,1.229492,0.487793,1.169922,1.526367,0.875977,
1.311523,-0.146240,0.745117,1.995117,1.502930,0.846191,1.674805,2.048828,1.370117,1.898438,1.425781,-0.029984,1.194336,-0.079651,0.861816,1.787109,1.851562,1.139648,0.481934,0.266846,
-0.440430,0.112549,0.759766,1.203125,1.324219,1.424805,-0.331055,0.316650,-0.181885,0.367188,1.113281,1.935547,1.302734,1.601562,1.438477,0.883789,0.463135,0.444092,0.610840,0.399658,
0.857422,1.412109,1.231445,1.257812,0.761230,0.701172,0.145996,0.313477,-0.122681,0.364502,1.104492,0.099609,-0.337646,0.946289,2.484375,2.021484,1.761719,1.811523,0.909180,0.979980,
0.551270,0.298096,-0.963379,0.367676,0.285156,0.765137,2.771484,2.302734,1.043945,1.210938,1.296875,-0.682617,0.843262,0.453369,-0.495361,1.150391,-0.096863,0.344727,1.096680,1.021484,
-0.299561,-0.087219,-0.571289,-0.947266,-0.582031,-0.613770,0.528809,-0.053284,-0.595703,0.206055,1.612305,2.015625,1.915039,-0.393066,0.061951,-1.077148,-0.389404,-0.942871,0.919434,0.217285,
0.298340,2.232422,1.702148,-0.739258,0.342773,-0.712402,1.788086,1.982422,2.197266,1.244141,1.916016,1.375977,1.362305,1.372070,1.136719,-0.282959,0.914551,1.045898,0.376221,0.362549,
0.206787,0.128540,2.060547,-0.371826,-1.079102,-1.156250,1.320312,0.655762,0.826660,1.208984,0.044128,0.424805,1.550781,0.975098,1.662109,0.252441,1.021484,0.145752,0.820312,-1.403320,
-0.242432,1.102539,0.770020,0.148315,0.646484,0.834961,0.555664,-0.348145,1.488281,0.791992,1.594727,0.850098,1.435547,1.585938,1.189453,1.370117,-0.047424,0.639648,0.966309,1.516602,
0.904297,1.158203,-0.625488,0.576172,0.613770,-0.313477,1.810547,1.643555,0.924805,-1.125977,-0.557617,0.268311,0.097778,0.537598,0.625000,-0.294678,0.820312,0.316162,0.046600,0.363281,
-0.307373,0.585449,-0.203247,-0.718750,0.351074,1.014648,0.319092,0.219482,1.129883,0.632812,1.411133,1.508789,0.771484,-0.366943,-0.114807,1.044922,0.293701,-0.115295,0.988770,1.266602,
0.401367,-0.892090,-0.056641,1.300781,0.537598,1.322266,1.065430,1.119141,0.645996,1.372070,0.791016,1.242188,0.099609,-0.947754,0.094543,-0.236450,0.325684,0.510254,-0.092468,-0.398682,
0.269043,-0.044250,-0.270752,-0.145508,-0.331787,0.700684,1.013672,0.175171,-0.138428,0.045105,0.807617,0.720703,0.895508,1.096680,0.509277,-0.051117,1.009766,0.743164,0.757812,-1.568359,
1.444336,-0.902832,1.740234,1.936523,1.298828,-0.655273,-0.657227,-0.833984,1.699219,1.698242,0.895020,-0.123840,0.437988,0.627930,1.702148,1.548828,0.719238,1.141602,0.210083,-0.333496,
0.926270,0.276123,-0.833984,0.659668,0.527344,1.011719,1.144531,0.777344,1.381836,0.942871,0.794434,0.159302,0.873047,1.190430,0.902832,0.821289,0.850586,0.990234,-0.178223,0.987305,
0.829590,-0.218994,-0.114929,-1.595703,1.392578,1.947266,1.915039,1.181641,1.488281,1.449219,1.353516,1.235352,1.370117,1.528320,0.672363,0.493652,1.250977,1.500000,0.477783,1.431641,
1.344727,1.225586,1.201172,0.405029,-1.004883,1.780273,1.233398,-0.839844,0.914062,0.803711,0.637207,1.309570,-0.110046,-1.113281,-0.961914,-0.407959,-0.144043,0.374023,0.484863,0.811035,
0.153076,-0.214722,0.487061,0.279785,0.508301,0.520996,0.680664,0.107788,0.224487,0.390625,-0.056396,0.589844,0.630859,0.821777,1.130859,0.464355,0.372070,1.242188,1.305664,0.054749,
-0.927734,-0.335938,-0.690430,0.074707,1.502930,0.955566,1.267578,1.189453,1.145508,1.144531,-0.067566,-0.534180,-0.590332,0.480225,-0.590332,-1.250000,-0.454590,-0.724121,-0.490967,-0.084290,
-0.895996,-1.077148,1.027344,-0.094116,0.674805,0.722656,1.101562,1.509766,1.657227,1.339844,0.749512,-0.474121,-0.965820,-0.666992,0.648926,0.422852,-0.293457,-0.341553,-0.073242,1.073242,
0.544922,0.095825,0.418701,0.544922,-0.841797,1.030273,-0.075500,0.325928,2.951172,2.443359,0.352783,0.877930,0.481689,1.746094,1.975586,1.615234,0.866211,1.675781,2.021484,1.789062,
1.677734,1.911133,1.927734,1.682617,1.151367,-0.155518,1.736328,0.882812,1.118164,1.820312,1.699219,1.335938,0.903320,1.052734,1.029297,1.496094,0.756348,-0.175537,1.403320,1.408203,
1.435547,0.236694,1.672852,1.316406,1.209961,2.136719,2.410156,1.047852,1.543945,1.349609,-0.173828,0.082336,0.341553,1.072266,1.521484,1.362305,1.150391,0.411865,0.265137,1.192383,
-0.616211,-2.558594,1.580078,1.702148,0.935059,0.943359,-0.670410,0.132935,1.077148,0.921387,0.110596,0.611328,-1.032227,0.154175,0.133545,-0.623535,-1.256836,-1.162109,-1.578125,-0.952148,
0.021255,-0.457275,-0.281494,-1.635742,-0.787598,0.152588,-0.614746,0.854980,0.247925,-0.489990,-1.231445,-1.380859,-1.266602,-0.305664,-0.168701,0.058441,-0.363037,0.139771,0.389648,-0.148315,
-0.826172,-0.102600,-0.527832,0.181763,-0.573242,-0.255127,-1.320312,-1.407227,-1.235352,-1.599609,-0.408691,0.195923,-0.428711,-0.605957,-0.397461,-1.529297,-0.646973,-1.905273,-0.821289,-0.558594,
-0.176147,-0.086121,-0.971191,-0.682617,-0.663574,-0.349365,-1.046875,-1.489258,-1.826172,-1.830078,-0.030609,-0.022705,-0.842285,-1.125000,-0.114319,-1.120117,-0.566406,-2.505859,-1.278320,-0.026093,
0.458008,-2.300781,0.049469,0.274414,-0.934570,-1.425781,-1.549805,-0.142944,-1.413086,-2.169922,-1.723633,-1.683594,-1.854492,-0.916504,-0.930176,0.462402,-0.547363,-0.919434,-0.329346,-0.142700,
-1.285156,-3.121094,-1.058594,-0.122375,-0.517578,0.024567,-0.309082,-1.072266,-1.583008,-2.330078,-0.723145,-0.017639,-0.363281,-1.968750,-3.072266,-2.605469,-1.185547,-1.562500,-2.316406,-1.873047,
-1.849609,-3.289062,-1.699219,-2.472656,-1.406250,-0.975098,-1.355469,-0.718262,-0.380615,-0.975586,-0.205078,-0.859375,-1.376953,-2.150391,-1.663086,-1.155273,0.000142,0.114502,-0.435303,-1.802734,
-1.853516,-1.857422,-0.661133,0.201660,0.283691,-0.574219,0.616699,0.259277,-1.935547,0.133667,-0.799316,-1.008789,0.296631,-0.239258,-0.579102,-0.327881,-0.574219,-1.186523,-3.373047,0.066711,
-0.463379,-0.775879,-0.151733,-1.912109,-0.358154,-0.221558,-1.131836,-0.799805,-0.325928,-0.573730,-0.912109,-0.219727,-0.427490,-0.276123,-0.673340,-0.485840,-0.757812,-1.416992,0.211182,-0.128296,
-0.376465,-1.902344,-0.141113,-0.366211,-0.337891,0.989746,0.011673,0.230347,-1.123047,-0.056976,-1.819336,-0.693359,-0.525391,-0.300781,-0.923340,-1.053711,-0.736328,0.039764,-0.137573,-1.067383,
-2.748047,-1.232422,-0.636230,-1.032227,-0.550293,-0.870117,-0.780273,-1.626953,-1.881836,-1.791016,-1.216797,-1.042969,-1.458984,-1.857422,-2.292969,-1.104492,-1.846680,-2.478516,-2.814453,-1.421875,
-0.622559,-1.493164,-0.366455,-0.034607,-0.427734,-1.823242,-0.730469,-0.765137,0.004047,-0.702637,-0.563477,-0.295898,-0.868164,-1.724609,-0.303955,-0.843262,-1.755859,-2.398438,-1.254883,-1.667969,
-0.541992,-1.905273,-2.570312,-1.112305,-1.933594,-2.669922,-2.246094,-2.222656,-2.806641,-1.893555,-2.488281,-1.321289,-1.431641,-0.572754,-0.634766,-1.742188,-0.761230,-0.825195,-1.939453,-1.150391,
-2.222656,-2.603516,-2.251953,-3.119141,-0.190308,0.301758,-0.641113,-1.370117,0.530762,-0.513184,-0.046173,-2.113281,-0.870605,0.169922,0.781250,0.235107,0.563965,-2.638672,0.223999,0.651855,
-1.136719,-1.523438,-0.341309,-1.247070,-1.580078,-0.906738,-0.996582,-1.866211,-1.262695,-1.548828,-0.937012,-0.954102,-0.167236,0.163574,-1.269531,-1.917969,-1.859375,-0.565430,-1.177734,-2.003906,
-1.910156,-2.119141,-4.195312,-1.826172,-1.274414,0.454590,0.186890,0.010880,-0.094543,-0.360840,-1.880859,0.234619,-0.333496,-0.674805,-0.827637,-0.488770,-2.060547,-1.033203,0.182129,-0.610352,
0.508789,0.302246,-2.158203,0.533691,0.702637,0.180176,0.338623,1.221680,-1.904297,1.238281,0.715332,0.500000,0.443848,-0.027237,-1.665039,-1.922852,-1.437500,0.343262,-0.804199,-1.167969,
-0.535156,-0.900879,-1.896484,-1.429688,0.585938,-0.985352,-0.958984,0.586914,-0.268555,-0.335693,-0.248535,0.534668,-0.247681,-0.728027,-0.468994,0.226074,-0.692871,0.687500,-2.111328,-1.729492,
-0.267090,0.824219,0.196655,1.416992,1.532227,-0.857910,1.013672,-0.107422,1.055664,-0.737305,-1.218750,-1.125000,-1.002930,0.719238,-0.225464,-1.166992,-0.650879,-1.142578,-2.529297,-0.770996,
-0.103943,0.409668,1.102539,0.021652,-0.156494,0.236816,0.960938,1.593750,1.161133,0.322998,-0.947754,-0.256348,-0.351562,-0.575195,-0.426270,0.420410,0.255371,-0.060760,-1.088867,-0.117920,
1.293945,1.149414,0.010277,1.014648,1.921875,0.617188,1.283203,0.337891,1.669922,1.200195,0.693359,0.819824,1.084961,0.366455,0.675781,1.162109,-0.025314,0.120605,1.492188,0.238403,
-0.328369,-0.827148,-1.191406,1.495117,0.940918,0.409668,-2.160156,-2.164062,1.203125,1.748047,1.025391,1.278320,0.076477,-2.078125,-1.050781,-1.947266,-1.968750,-0.472900,0.935059,0.468506,
0.898438,-0.662109,0.140381,0.754395,-0.848633,-0.833984,-0.846680,-1.478516,-0.427002,0.138916,-0.636230,-0.717773,-0.479004,-0.114624,-0.261963,0.003792,0.104797,-1.581055,0.212402,-0.371582,
-0.251953,-0.104248,0.626465,-0.421143,0.212646,0.809570,-0.202637,-0.254395,-0.150391,0.211182,-1.739258,-0.395020,-0.253418,-0.971191,-1.621094,-1.177734,-0.749023,-0.184937,-1.756836,-0.494141,
-1.108398,-1.737305,-1.214844,-1.287109,-0.272461,-0.892578,-0.770996,-0.367188,-0.194214,-1.019531,-0.977539,-1.585938,-1.095703,-1.042969,-0.306396,-0.635742,-0.492920,0.045990,-0.314697,0.104980,
-0.677734,-0.601074,0.072266,-0.160278,-1.318359,-2.478516,-1.527344,-0.928711,-0.940430,-1.420898,-0.959473,-1.294922,-1.047852,-1.312500,-1.182617,-0.833008,-1.500977,-1.215820,-1.382812,-0.937500,
-1.177734,-1.794922,-0.133911,0.678711,-1.139648,-0.180908,-2.792969,-1.735352,-0.653320,0.232788,-1.099609,0.300781,-0.756348,-2.439453,-2.757812,-4.320312,-2.763672,-3.082031,-2.531250,-0.919922,
-1.901367,-0.673828,-2.212891,-2.060547,-1.490234,-0.616211,-2.578125,-2.705078,-1.476562,-0.336670,-1.165039,-0.846680,-1.618164,-1.192383,-2.300781,-0.921387,-0.111084,-0.838867,-0.903320,-1.827148,
-1.146484,0.192139,1.207031,0.501953,0.566406,-0.127197,-0.897949,-0.507812,-1.099609,-1.132812,-2.552734,-1.422852,-0.671875,-1.155273,-0.776855,-1.490234,-1.040039,-0.605957,0.279541,-1.294922,
0.480957,-1.579102,-1.305664,-2.335938,-0.801758,-1.590820,-1.326172,-1.055664,-0.163086,0.480713,-0.294678,-1.333008,-2.142578,0.244629,-1.477539,0.541016,-0.062012,0.654785,-0.889648,-0.108643,
-0.177612,0.764648,0.859863,-1.240234,-2.087891,0.033447,-0.297607,0.015717,0.570312,-0.356934,0.167847,-2.580078,0.536133,-1.343750,-1.424805,-0.878906,0.211426,0.068604,-0.591797,-0.472656,
-0.311279,0.647461,-0.256348,0.020660,0.921875,-1.339844,-1.513672,-1.405273,-1.802734,-1.970703,-0.275635,0.418213,-0.903809,-0.161621,-0.494629,-1.790039,-1.940430,-2.871094,-1.060547,-0.318115,
-0.404053,-0.281006,-0.321777,0.528809,-0.376465,-0.387207,-0.831055,-1.249023,-1.080078,-0.291748,-0.791016,-1.233398,-2.599609,-1.611328,-0.629883,-0.962402,-0.931641,-1.090820,-1.172852,-0.352539,
0.451904,-1.076172,-0.375000,0.088440,0.256836,1.145508,-0.809570,-0.635254,-1.869141,-1.254883,0.184326,-0.766113,-0.069702,-0.117676,-0.167725,-0.065796,-0.710449,0.425293,-0.426758,-1.635742,
-2.925781,-2.238281,1.209961,-0.974121,-1.058594,-2.656250,-0.202637,-0.859375,-0.700195,-0.596191,-0.933594,0.850586,-1.446289,-1.354492,-2.398438,0.259033,-0.630859,-0.196167,-0.097351,1.147461,
2.023438,0.350830,1.919922,1.255859,0.406494,1.000000,-0.788574,1.583008,1.814453,1.203125,-2.843750,-0.694336,-1.716797,-1.676758,-0.840820,-0.220093,-0.741699,-3.050781,-0.269043,0.579102,
-1.485352,-1.333008,-1.391602,1.737305,0.968262,0.980469,0.384033,-0.735352,-2.951172,-0.695312,-1.090820,-0.343994,0.209351,-0.531738,-0.820312,-0.729980,-0.615723,-0.593750,-1.503906,0.333252,
-0.390869,1.127930,-1.027344,-1.077148,-0.467041,1.163086,1.032227,1.925781,1.155273,0.344238,-0.405762,0.893555,0.659180,-0.183594,0.092224,0.714844,-1.075195,0.862305,1.195312,0.332764,
-0.204346,1.143555,-0.216431,1.005859,1.316406,0.575684,0.779785,0.993652,0.399414,0.623535,-0.447998,-0.325928,-0.133179,-0.856445,0.972656,0.058807,0.672852,-1.198242,0.750000,-0.937012,
-0.650391,-1.076172,0.771973,0.098389,-1.011719,-1.300781,-0.246460,0.746094,-0.569336,0.157837,0.312744,-0.082336,0.078247,-2.373047,0.465332,-1.106445,-1.205078,0.061432,-1.246094,-0.029175,
1.200195,0.462646,0.691406,0.093262,0.194702,-0.538574,-0.486084,0.433594,-0.328857,0.465088,-1.359375,-0.531738,0.083740,-1.286133,-1.029297,-1.732422,-0.688477,-1.462891,-1.676758,-1.504883,
0.309814,0.755859,-0.568848,0.461426,-0.369873,0.385498,-1.371094,-0.310547,0.400879,-1.475586,-1.162109,-1.212891,0.310059,-0.206543,-0.770020,-1.130859,-0.130371,-0.439453,0.301270,0.401123,
0.128296,1.005859,0.973633,0.772461,0.373779,1.174805,0.374268,0.862793,-0.265625,0.189209,-0.347412,-1.144531,-0.693848,-0.616211,-0.101074,-0.891602,-1.501953,0.778320,1.070312,0.328857,
-1.302734,-2.804688,-1.066406,0.881348,0.789062,-0.254395,0.025070,-0.665039,-1.541016,-1.198242,-1.039062,-0.615234,-1.428711,-1.234375,0.157471,-0.137207,0.493896,0.132690,0.258789,-1.685547,
0.530762,0.772461,0.830078,1.102539,0.988281,0.143433,-0.910645,-0.048248,0.293701,1.084961,0.293457,0.197876,-0.382080,-0.250244,0.897461,-0.486572,-0.536133,0.030746,0.976074,-0.292236,
0.611816,1.064453,0.478760,-1.827148,-1.113281,-2.125000,-2.074219,-1.213867,0.231445,-1.235352,-1.888672,-0.580566,-1.294922,-0.885254,-0.755859,-0.869629,0.035431,-1.603516,-0.664062,-0.203125,
0.431885,0.085632,0.156616,0.332031,0.071106,-0.061340,0.062042,-1.225586,0.743164,-1.760742,-0.619141,-0.366211,-0.115479,0.210205,0.206421,0.582031,0.068237,-0.012611,-0.410400,1.184570,
1.129883,0.849609,0.503906,0.763672,0.846680,1.080078,1.718750,0.592285,0.458008,-1.038086,-0.880371,0.654297,-0.751465,-1.425781,-1.370117,0.362061,0.543457,0.939941,0.917480,0.328613,
-0.409668,1.022461,-0.424072,-0.130615,-0.154297,0.404785,-0.618652,-0.771484,1.238281,1.356445,0.083313,0.551758,0.761719,0.573730,1.196289,0.713867,0.016220,0.672852,1.034180,0.703125,
0.437012,-0.620117,1.008789,1.206055,0.558594,0.916016,0.425537,1.105469,0.789551,0.632324,1.749023,0.229370,0.696777,0.169556,-0.534180,-2.113281,0.057281,-0.339600,0.402832,1.429688,
1.533203,1.299805,1.033203,-0.317139,-0.031403,-0.533691,-0.727539,-0.800293,1.355469,0.427002,0.486328,-0.828613,-0.524902,-0.729004,0.526367,0.824707,0.862793,-0.784668,0.597168,1.340820,
0.329102,-0.246704,-0.446289,0.620117,-0.424316,-0.722168,0.092590,0.253906,-0.283447,-0.743164,0.740723,1.548828,0.124573,1.847656,1.533203,1.115234,1.425781,-0.076294,0.718262,1.076172,
-0.125000,0.670898,1.496094,-0.722168,0.906738,0.131592,0.411865,0.872559,0.075195,-0.539551,0.316650,0.310303,0.007702,0.414551,0.058960,-1.080078,0.081726,0.163208,-0.075134,1.069336,
-0.999512,0.385986,0.328369,-0.200317,-0.564453,-0.591309,0.380371,1.041992,-0.256348,-0.526855,1.084961,-1.835938,0.476807,0.986328,0.924805,1.210938,1.115234,1.361328,-0.386475,0.371826,
1.384766,0.138672,-0.249268,-0.308350,0.159790,-0.242554,-1.104492,0.575195,0.385498,-0.747070,-0.428711,0.295898,0.457031,-0.681152,-1.442383,0.057190,-0.419678,0.087219,-1.360352,-0.577148,
-0.950195,-1.308594,-0.156250,0.555664,0.998047,-0.806152,0.075562,-0.528320,-0.738770,0.986816,-0.642578,-0.784668,-0.645996,-0.165649,0.280762,1.060547,0.027451,-0.523926,0.047821,0.959473,
0.263184,1.399414,1.192383,0.989746,1.271484,0.850098,1.402344,1.168945,0.226196,-1.086914,0.034393,-0.719238,0.747559,0.445068,0.322510,1.692383,0.256592,1.552734,-1.027344,0.957031,
1.059570,2.017578,2.490234,0.870117,0.837402,1.139648,0.638672,0.380127,0.586914,-1.363281,0.695312,0.482422,0.151855,1.139648,1.367188,1.383789,1.088867,0.789551,1.713867,1.347656,
1.451172,0.652344,0.113159,0.652832,1.753906,0.841309,2.169922,2.277344,2.037109,1.062500,0.423828,0.492432,0.729004,0.438965,2.181641,1.135742,1.837891,2.246094,1.460938,1.589844,
2.298828,1.111328,-0.121216,1.987305,1.833008,1.656250,0.874512,-0.698242,-1.266602,-1.087891,1.069336,0.348145,1.873047,0.743652,-0.267334,0.370605,-1.359375,-0.623535,-0.261475,1.083984,
0.420166,1.875977,1.447266,1.175781,0.488770,0.823730,0.656250,0.694824,0.093567,-0.410156,0.596680,0.629883,0.340088,-0.229492,-0.650879,-1.069336,0.634766,1.950195,-1.064453,0.303223,
-0.852051,0.363037,0.536133,0.455811,-0.504395,0.007965,0.572754,1.164062,-0.954102,0.002274,0.151001,0.881348,0.258789,0.033020,-2.111328,-0.569336,0.312988,-1.058594,-1.403320,-2.064453,
-0.467529,-0.958008,-0.699219,0.047180,-1.126953,-0.218506,-0.370361,-1.814453,-1.352539,-1.391602,-1.390625,-1.547852,1.147461,-0.139160,1.252930,0.666016,0.819824,-0.166626,1.403320,2.410156,
2.234375,2.259766,0.996094,0.336914,-1.719727,0.384277,1.532227,1.149414,1.808594,-0.796875,0.308105,-0.806641,1.401367,1.617188,0.321533,0.791016,-0.296631,2.576172,2.353516,1.339844,
1.297852,0.278076,1.067383,1.373047,2.093750,0.431641,0.944336,0.808594,2.193359,0.007793,-0.366211,1.145508,1.660156,0.887695,1.983398,-1.618164,0.303223,1.347656,0.970215,1.315430,
1.344727,0.793945,1.195312,2.611328,2.751953,1.971680,0.029053,1.462891,0.561035,1.037109,1.544922,1.299805,1.226562,3.189453,2.345703,1.611328,2.232422,3.527344,2.904297,1.079102,
2.158203,3.656250,1.684570,2.474609,1.334961,2.445312,-0.872070,0.669922,1.701172,2.583984,1.288086,0.813965,0.127441,0.260498,1.321289,1.888672,1.543945,0.634766,-0.112488,0.675293,
2.041016,-0.568848,1.303711,1.766602,2.576172,1.317383,1.419922,1.093750,0.594238,1.671875,1.292969,2.912109,2.183594,0.260986,1.521484,2.345703,2.470703,3.123047,1.130859,1.186523,
1.767578,2.656250,1.750000,0.561035,0.941895,1.259766,0.020020,1.256836,0.662598,2.191406,1.379883,0.788086,-0.042175,0.846680,1.309570,-0.626465,0.731445,0.476318,2.105469,2.908203,
2.572266,0.411133,-1.210938,-0.607910,0.374023,0.191528,0.503906,0.311035,-1.291992,-1.308594,0.187500,0.664551,1.880859,1.718750,0.771484,0.134766,0.290527,0.683594,0.484619,2.351562,
1.694336,2.925781,3.234375,1.274414,1.548828,2.867188,1.238281,0.637207,1.202148,2.134766,1.836914,3.119141,1.649414,3.115234,2.501953,2.664062,3.871094,2.869141,2.294922,1.390625,
1.957031,2.001953,0.099121,0.963379,0.534180,3.007812,2.111328,1.833984,2.246094,1.235352,2.568359,1.916992,0.128174,1.143555,2.949219,2.763672,3.314453,3.435547,2.542969,1.302734,
0.426270,-0.791504,1.486328,1.799805,1.699219,1.720703,0.994629,1.279297,2.480469,2.695312,1.815430,2.255859,3.035156,2.273438,3.880859,1.069336,1.847656,2.996094,3.537109,3.902344,
3.974609,4.136719,2.925781,1.861328,1.926758,2.630859,2.958984,2.410156,3.062500,2.312500,3.177734,2.617188,2.058594,0.520508,2.042969,1.418945,0.219971,1.950195,2.382812,0.968262,
1.103516,0.595703,1.768555,2.705078,1.724609,1.851562,1.911133,1.297852,2.660156,1.802734,1.759766,2.892578,1.306641,3.025391,3.203125,2.494141,2.648438,2.562500,3.785156,3.052734,
2.744141,3.275391,3.144531,2.634766,2.308594,1.851562,2.980469,2.359375,2.585938,2.746094,1.518555,3.050781,2.970703,3.359375,0.858887,0.726562,1.580078,2.763672,2.832031,2.601562,
2.789062,0.785156,0.683105,1.631836,0.802734,2.447266,1.919922,2.576172,2.150391,2.335938,1.586914,2.568359,0.518066,-0.240112,1.026367,2.761719,2.052734,1.683594,2.351562,2.361328,
3.023438,0.266113,2.345703,2.419922,2.859375,3.332031,0.910645,2.312500,3.349609,1.908203,2.779297,3.359375,2.375000,2.296875,3.646484,2.462891,3.742188,1.441406,1.294922,2.318359,
1.232422,1.435547,2.044922,2.509766,1.361328,1.754883,1.075195,0.968750,3.082031,3.138672,1.339844,3.060547,2.587891,2.121094,1.451172,2.148438,3.269531,0.639160,2.609375,3.460938,
2.585938,2.310547,2.585938,3.015625,3.195312,1.248047,2.671875,1.512695,1.491211,2.705078,3.287109,2.910156,2.783203,3.474609,3.076172,2.191406,3.048828,4.039062,1.084961,3.044922,
3.798828,3.578125,3.357422,3.744141,4.242188,3.751953,2.757812,1.515625,3.029297,3.552734,3.205078,3.142578,4.093750,3.570312,3.410156,3.785156,3.464844,3.158203,3.402344,3.308594,
3.261719,2.595703,2.181641,2.697266,3.572266,1.469727,2.898438,3.763672,3.587891,3.273438,3.408203,3.871094,3.041016,2.789062,3.033203,4.488281,2.603516,4.089844,4.128906,3.845703,
1.902344,2.847656,3.531250,4.054688,3.777344,4.250000,4.136719,4.308594,3.001953,3.962891,2.103516,2.929688,4.308594,3.087891,4.253906,2.013672,3.513672,4.062500,2.921875,3.222656,
3.666016,3.162109,4.003906,3.314453,1.988281,3.248047,2.408203,1.108398,2.919922,2.847656,3.177734,2.775391,3.845703,2.171875,3.279297,2.058594,3.130859,3.765625,3.847656,3.667969,
3.859375,4.042969,3.548828,3.775391,3.712891,3.419922,4.699219,4.195312,4.488281,4.453125,3.636719,3.773438,3.279297,4.335938,4.765625,5.035156,4.058594,4.480469,4.144531,3.421875,
4.277344,4.234375,3.259766,4.789062,5.429688,5.027344,4.738281,3.732422,4.292969,4.050781,4.355469,2.785156,5.070312,4.953125,4.140625,4.503906,5.226562,4.972656,4.242188,5.750000,
4.976562,4.792969,3.835938,4.628906,5.386719,5.246094,5.652344,5.066406,5.714844,6.359375,5.601562,6.144531,6.308594,3.947266,6.175781,4.464844,5.730469,5.988281,6.867188,6.699219,
-65504.000000,-65504.000000,0.369141,1.112305,-0.810059,-0.866699,1.511719,-0.021210,0.387939,0.915527,1.184570,1.280273,-1.449219,0.608398,-1.528320,0.738770,0.450195,-1.246094,1.391602,0.844238,
1.367188,-0.483154,1.082031,1.560547,1.432617,1.075195,-0.968262,1.000000,0.908203,0.692383,-0.654785,-0.092224,0.438477,0.129272,0.172485,0.078003,0.016571,-1.454102,-0.993652,0.357178,
-0.954590,-0.063416,-0.819336,-0.238037,-1.928711,-0.881836,-0.114685,-0.174438,0.353760,0.466797,-1.492188,-0.654297,-0.298096,-0.369141,-0.452148,0.912598,0.317139,-0.259033,0.527344,0.526855,
0.497070,0.955078,1.229492,0.650391,0.896484,1.674805,0.971680,1.277344,1.092773,0.316406,0.442139,0.803711,1.813477,2.187500,1.017578,1.116211,-1.248047,1.908203,1.594727,0.788574,
0.256592,-1.207031,1.574219,2.242188,0.592773,1.767578,-0.156616,-0.562012,0.394287,-0.038269,0.370361,-2.232422,0.379883,0.234741,-0.914062,0.639648,-2.017578,1.597656,1.695312,1.569336,
1.239258,0.847168,-1.993164,-1.294922,-0.122375,0.502930,0.099426,0.406982,-0.027283,0.700195,1.371094,0.666992,0.202515,1.882812,2.400391,-1.599609,0.414062,2.394531,-1.508789,1.355469,
-2.119141,2.193359,2.449219,-2.265625,1.688477,1.665039,-0.259521,-1.264648,0.238281,-1.282227,1.498047,-0.781250,0.714355,-0.706543,-0.200928,1.015625,0.865723,-2.386719,-0.763672,-0.496582,
0.983398,0.452637,0.817871,0.036377,-0.301758,-0.452148,-0.397461,0.826172,-1.593750,0.662109,-1.666992,1.008789,0.280273,1.262695,1.738281,-0.970703,0.446289,0.113708,1.623047,0.336670,
1.591797,2.042969,-0.923340,1.297852,0.381836,1.317383,1.290039,0.433594,0.823730,0.049530,0.581055,0.352783,0.823242,-0.399414,1.059570,0.288330,-0.897949,0.389648,-0.555664,0.698242,
0.225708,-1.030273,-1.408203,-0.290283,-0.027176,-0.801758,-0.398193,0.327393,0.008354,-0.573730,-0.286865,-0.692871,0.678223,-2.328125,-0.215942,-0.059875,-2.417969,0.583496,0.843262,0.428223,
0.304932,-1.250977,0.446777,0.654297,0.030380,0.672852,0.380859,-1.193359,1.347656,0.691406,1.173828,1.335938,-0.213013,0.410156,1.410156,1.551758,1.056641,1.027344,0.634277,-1.388672,
0.600586,-1.350586,0.597656,1.469727,0.146729,0.134155,-1.439453,-1.719727,-1.293945,-1.963867,-0.227173,0.164185,-1.310547,0.270996,-1.453125,0.722656,-0.208130,0.670898,0.360596,0.619141,
-0.009445,0.751465,-0.122314,0.680664,1.302734,1.461914,1.627930,1.949219,1.654297,1.330078,0.659668,-1.492188,1.447266,1.697266,2.246094,3.359375,2.232422,1.312500,1.866211,2.824219,
3.386719,3.646484,3.447266,-0.402832,1.234375,1.300781,3.798828,3.072266,1.673828,3.285156,1.160156,0.820312,2.392578,-1.139648,1.501953,2.382812,2.275391,1.734375,2.144531,-1.164062,
0.846191,-0.962891,1.895508,1.844727,2.316406,2.449219,2.482422,1.994141,-0.768555,1.373047,-0.089905,1.041992,2.062500,1.609375,1.199219,0.953613,1.273438,0.797363,1.270508,2.615234,
1.383789,0.754883,2.656250,1.909180,1.617188,2.248047,0.201904,2.371094,0.437500,0.688965,1.174805,0.844727,0.051056,-0.396729,0.060516,-0.961914,-0.119995,-0.488770,0.314453,-0.204590,
-1.118164,-0.779297,-0.712402,0.297852,1.016602,0.059082,-1.776367,-0.594727,-2.400391,-1.766602,-1.559570,-1.762695,-1.394531,-1.114258,-0.607422,-1.840820,-1.448242,-1.299805,-1.772461,-2.027344,
-1.462891,-1.231445,-0.103088,1.034180,-0.162598,-0.139282,-1.435547,-0.444336,-1.209961,-0.614258,0.349609,-0.678711,-0.913086,-1.075195,-0.301514,-0.817871,-1.295898,-1.037109,-1.691406,-2.027344,
-1.889648,-1.880859,-2.224609,-1.261719,-0.831055,-0.698242,-0.971191,-0.078369,-0.520508,-1.747070,-2.302734,-2.710938,-2.669922,-2.416016,-1.410156,-0.534668,-1.334961,-1.095703,-1.463867,-1.948242,
-2.318359,-1.547852,-1.725586,-0.880859,0.105591,0.042389,0.487305,-0.818359,0.407715,-0.043121,-0.939941,-0.346924,0.317139,0.156372,-0.125854,-0.010162,0.053131,-0.357910,0.211914,0.824219,
0.430176,-0.562988,-0.249023,-0.830566,-1.209961,-0.797363,0.258301,-0.024551,-1.133789,1.596680,1.217773,-0.164062,0.668945,0.486084,0.260254,0.313477,-0.464844,0.535645,0.728027,-0.218750,
0.118896,-1.157227,-0.377930,-1.060547,0.073730,-0.387695,-0.781738,-0.685059,-1.820312,-0.596191,-1.960938,-0.034119,-1.573242,0.019302,0.199341,0.096741,0.240845,-0.801758,0.325928,-0.147949,
0.546387,0.643555,0.784180,1.523438,0.031464,-0.196899,-1.843750,-0.182495,-0.085144,0.022507,0.145508,0.022659,-0.445801,-1.082031,-0.741211,0.109863,0.378662,-0.502930,-0.751953,-0.719727,
-1.297852,-1.481445,-1.741211,-0.821289,-2.871094,-1.489258,-1.576172,-0.831543,-1.208008,-0.594727,-1.466797,-1.627930,-1.966797,-2.376953,-1.863281,-1.142578,-0.126099,-0.443604,-1.399414,-1.548828,
-0.457031,-0.470703,-0.723145,-0.679199,-0.020721,-0.360596,-0.515625,0.187134,0.287598,0.282715,0.418945,-0.522461,-0.152710,0.833984,0.375732,0.442139,-0.049805,-0.000732,-0.347412,1.164062,
1.307617,0.389893,0.928223,0.433838,-0.219604,-0.580566,-0.054565,-1.169922,-0.270020,-0.694824,-0.682129,-0.030334,-0.363770,-0.402832,-0.896484,-0.203369,0.020752,-0.238281,0.317871,0.233032,
0.258545,-0.615723,-0.250488,0.255615,0.038086,0.095825,0.683594,0.395020,-0.424316,0.599609,0.536621,1.768555,1.882812,2.609375,2.107422,1.632812,2.001953,1.640625,-1.134766,1.878906,
1.959961,0.816406,-0.042175,2.335938,2.027344,2.078125,1.767578,1.217773,1.678711,1.113281,1.867188,-1.605469,1.463867,0.694336,1.302734,-1.138672,0.858887,1.406250,1.122070,0.659668,
0.976074,0.524902,0.330322,0.914062,-1.574219,0.505859,0.802734,-0.275879,1.107422,0.975098,0.781738,-2.097656,0.819824,-0.091370,0.847656,0.974121,1.671875,2.074219,-0.663574,0.008606,
-0.336426,-1.427734,1.979492,1.499023,1.637695,1.699219,2.361328,2.451172,0.406494,1.485352,1.190430,1.320312,1.736328,0.101196,0.998047,-0.695312,-2.638672,1.806641,0.905273,1.511719,
1.672852,1.732422,1.091797,1.266602,-1.070312,0.873047,0.273926,-0.944824,0.426025,-0.590332,0.596191,0.414551,0.583008,0.738770,-1.009766,0.071594,-2.080078,1.132812,0.957031,1.141602,
0.853516,0.738770,0.733887,0.880371,1.088867,1.532227,0.809082,1.025391,-0.856934,1.727539,1.276367,1.862305,2.019531,1.807617,2.636719,2.027344,1.929688,2.712891,1.764648,1.894531,
-0.506836,3.082031,2.681641,1.018555,3.279297,2.683594,1.492188,1.465820,1.850586,2.302734,2.166016,1.518555,2.349609,0.461914,0.472900,1.064453,1.955078,2.083984,2.492188,2.007812,
1.163086,1.890625,1.657227,2.160156,1.469727,1.831055,1.872070,2.800781,2.253906,2.234375,2.341797,0.501465,1.294922,-0.417969,3.037109,4.144531,1.991211,3.566406,3.277344,2.259766,
2.923828,3.251953,3.314453,1.239258,2.679688,0.665039,2.210938,3.449219,4.679688,1.565430,3.078125,3.369141,1.939453,4.214844,3.572266,4.050781,1.916016,2.955078,0.872070,4.187500,
4.031250,4.460938,2.560547,3.263672,2.710938,0.824219,2.871094,2.033203,3.507812,4.417969,2.990234,4.542969,2.644531,4.957031,4.855469,5.300781,2.001953,4.699219,4.480469,3.429688,
4.761719,4.671875,4.328125,5.343750,4.421875,5.519531,3.703125,5.093750,3.355469,1.815430,5.328125,4.828125,5.710938,6.222656,6.378906,6.257812,4.183594,4.332031,4.570312,4.203125,
4.062500,3.685547,2.644531,3.449219,3.017578,3.195312,3.308594,2.535156,2.767578,1.611328,0.831543,3.443359,3.488281,3.384766,3.164062,3.814453,3.134766,2.945312,3.603516,2.337891,
2.919922,1.917969,2.152344,2.300781,1.175781,1.512695,1.761719,1.497070,3.796875,1.733398,2.949219,2.103516,1.319336,0.793457,0.916504,1.885742,2.945312,3.246094,2.076172,2.263672,
2.328125,0.863281,0.852539,2.345703,0.160645,0.437500,0.514648,1.066406,1.180664,-0.097351,-0.065491,-0.359375,-0.017181,-0.282715,-0.108032,0.399170,1.392578,0.884766,0.855957,0.306152,
0.724121,2.222656,0.117493,1.129883,-0.026306,-0.526367,-0.824707,-0.027115,0.033508,0.600098,0.852539,1.286133,1.598633,1.536133,1.845703,1.843750,1.003906,1.794922,0.937988,2.099609,
1.277344,2.175781,2.058594,1.581055,1.938477,2.351562,2.539062,2.800781,2.212891,1.955078,1.248047,2.095703,0.458252,2.484375,2.224609,1.980469,1.073242,1.909180,1.666992,0.849121,
1.251953,1.150391,0.560547,1.895508,1.806641,2.867188,2.136719,1.353516,0.578125,1.347656,1.745117,1.469727,2.326172,1.315430,2.302734,2.541016,2.550781,2.205078,2.832031,2.039062,
3.134766,3.089844,0.932129,2.884766,2.716797,0.339111,3.189453,2.687500,3.328125,2.976562,2.220703,3.173828,3.416016,2.789062,2.605469,1.587891,2.873047,3.414062,2.583984,2.574219,
1.842773,2.001953,1.908203,1.453125,1.842773,0.780273,1.028320,1.336914,1.621094,1.489258,1.328125,0.315430,0.880371,0.489746,0.156250,1.569336,1.083984,0.785645,0.741699,0.679199,
1.042969,1.225586,1.498047,3.587891,0.948242,0.866699,1.175781,0.813477,0.136108,1.240234,1.542969,1.541992,1.628906,1.216797,0.630371,2.880859,1.128906,0.719238,1.150391,1.755859,
2.007812,1.540039,0.903320,0.288574,0.210327,0.109863,0.794922,1.027344,1.212891,0.697754,0.348877,-0.434326,-0.176270,0.442627,1.271484,0.904785,1.000000,0.641113,0.298340,0.075317,
-0.103760,-0.257080,-0.233643,-0.461426,0.181641,0.746094,0.908203,2.128906,0.978516,0.191406,-0.013580,-0.192749,0.209961,0.666992,0.704102,-0.029175,0.570801,2.070312,2.328125,2.273438,
2.238281,2.181641,2.169922,1.064453,0.808594,0.118896,0.643555,0.880371,0.490479,2.843750,2.960938,1.838867,2.265625,1.884766,-0.207397,1.134766,0.092773,0.507324,1.218750,0.739258,
0.629883,1.203125,1.257812,0.026855,0.411133,-0.786621,-0.820312,-0.415771,-1.204102,-0.194702,0.738770,-0.172729,-0.239624,1.516602,1.785156,2.103516,-0.338135,0.749023,-1.463867,-0.196899,
-2.011719,-0.466064,1.292969,-0.127441,2.751953,2.357422,-0.622070,0.620605,-1.131836,1.998047,2.732422,2.775391,0.068115,2.794922,0.254395,1.724609,1.850586,1.691406,-1.074219,-0.304932,
1.614258,0.967285,1.388672,0.262451,0.078979,1.790039,-1.856445,-0.913574,-2.052734,1.327148,0.589355,0.558105,0.749023,-1.351562,-0.196777,0.842773,1.291992,1.771484,-0.675781,1.097656,
0.666992,1.206055,-2.476562,-0.709473,0.838379,-0.393311,-0.872559,1.245117,1.185547,0.849609,-1.155273,1.300781,0.564941,2.509766,1.408203,2.070312,1.997070,1.572266,1.645508,-0.479004,
0.753418,1.308594,1.390625,1.072266,1.327148,-0.236572,1.057617,1.040039,-0.565918,1.261719,1.735352,1.468750,-0.514648,-0.207275,0.213501,-0.278320,-1.005859,0.059448,-0.569824,0.352783,
0.128662,-0.247559,-0.210205,-0.404053,-0.236084,-0.502930,-1.414062,-0.635254,-0.056915,-0.389404,-0.819336,0.259277,-0.110901,0.493652,0.595703,-0.169800,-0.154785,-0.122375,0.816406,0.231079,
-0.544434,0.440674,0.901855,0.570312,-0.441406,0.141479,0.614746,0.273682,0.735840,0.687012,0.679688,-0.076111,1.114258,0.815918,1.388672,0.730469,-0.777344,0.253418,-0.930664,-0.301270,
-0.159424,-0.598633,-0.541016,-0.205200,-0.311523,-1.077148,-0.705566,-1.610352,0.205322,0.201050,-0.070068,-0.507812,-1.467773,-0.022446,0.238403,0.498535,0.479004,0.563965,-0.066711,0.799805,
0.187012,0.734375,-0.694824,1.113281,-0.945312,1.531250,2.066406,1.771484,-0.417236,0.441895,-0.977051,1.592773,1.974609,1.473633,0.013847,1.462891,1.943359,2.046875,2.353516,1.957031,
2.328125,1.322266,-0.372070,1.300781,1.157227,-1.108398,1.144531,0.788086,1.114258,1.338867,-0.003632,1.758789,1.861328,1.688477,0.676270,0.781738,0.859375,0.545410,0.050293,0.045685,
1.057617,0.532715,0.790527,0.875488,-0.456055,0.775391,-1.105469,1.435547,1.504883,1.855469,1.673828,1.736328,1.888672,1.599609,1.485352,0.347412,2.085938,1.883789,0.903809,1.471680,
1.608398,0.089966,1.547852,1.560547,1.835938,1.537109,1.138672,-1.373047,1.428711,0.770996,-1.826172,0.647461,0.496826,0.518555,1.320312,-0.050873,-0.173706,-0.435303,0.196411,-0.690430,
-0.370605,-0.211914,-0.233276,-0.470703,-0.902832,-0.888672,-1.120117,-0.557617,-0.487305,-0.046234,-0.098572,0.174927,-0.030106,-1.047852,0.430420,0.265137,0.537598,0.628418,0.530762,-0.070068,
0.664551,0.704590,0.446777,-0.607422,-0.017853,0.060089,0.617676,1.777344,1.951172,1.623047,1.486328,1.206055,1.269531,-0.700195,0.510254,0.103455,0.589844,-0.160645,-0.488037,0.100830,
0.362549,0.808105,1.011719,-0.141846,-0.881836,1.058594,0.357178,0.806641,0.928711,1.033203,1.299805,1.476562,2.398438,1.452148,0.497559,-0.119507,0.008888,-0.316895,1.195312,0.993164,
1.053711,0.296387,0.957520,1.257812,1.403320,1.557617,2.210938,0.274658,2.435547,0.743164,1.228516,4.261719,4.343750,1.786133,2.537109,0.718750,3.634766,4.027344,4.152344,3.453125,
2.804688,2.503906,3.421875,2.865234,3.984375,3.558594,3.181641,2.511719,-0.039764,2.695312,1.417969,2.068359,2.771484,2.390625,2.169922,0.228516,2.222656,1.985352,1.868164,1.240234,
-0.731445,1.395508,2.037109,2.123047,1.069336,1.646484,1.465820,-0.151733,1.893555,2.347656,0.273926,1.348633,1.345703,-1.293945,0.540527,-0.508789,1.547852,1.975586,2.019531,1.729492,
1.039062,0.623535,0.621094,-0.547363,-3.222656,0.638184,1.042969,0.764160,0.513672,0.335205,-1.011719,0.321533,-0.105652,-1.194336,-0.215698,-1.466797,-0.662109,-1.165039,-1.809570,-4.097656,
-2.529297,-3.199219,-2.382812,-1.687500,-2.412109,-2.263672,-2.498047,-3.585938,-1.509766,-2.640625,-1.702148,-2.427734,-3.550781,-2.089844,-2.568359,-2.654297,-2.117188,-2.642578,-2.140625,-1.092773,
-2.199219,-0.778809,-1.470703,-2.449219,-1.572266,-2.566406,-1.128906,-2.175781,-2.214844,-3.664062,-2.585938,-2.658203,-3.074219,-1.907227,-1.475586,-1.964844,-1.611328,-2.003906,-2.890625,-1.844727,
-2.935547,-2.675781,-3.566406,-2.283203,-2.205078,-2.488281,-2.816406,-2.708984,-2.666016,-3.253906,-3.859375,-3.849609,-4.437500,-2.160156,-2.328125,-2.855469,-3.142578,-2.482422,-3.093750,-3.306641,
-5.515625,-3.833984,-2.287109,-1.819336,-3.179688,-1.727539,-2.150391,-3.064453,-3.142578,-2.980469,-2.958984,-2.824219,-3.458984,-3.474609,-3.169922,-4.906250,-2.869141,-2.769531,-2.041016,-2.046875,
-2.419922,-1.798828,-1.646484,-2.515625,-5.519531,-3.769531,-2.558594,-3.238281,-1.914062,-1.977539,-4.546875,-4.324219,-5.195312,-3.550781,-2.775391,-3.078125,-3.974609,-5.839844,-4.902344,-5.050781,
-4.359375,-4.679688,-4.394531,-4.171875,-7.007812,-4.871094,-5.332031,-5.273438,-3.314453,-4.195312,-5.023438,-3.529297,-3.246094,-2.960938,-2.806641,-3.337891,-5.640625,-5.375000,-3.814453,-2.250000,
-2.220703,-3.058594,-3.046875,-3.318359,-3.966797,-3.031250,-1.947266,-2.111328,-3.093750,-1.999023,-2.126953,-4.699219,-2.222656,-3.320312,-3.601562,-2.619141,-3.031250,-4.199219,-3.029297,-3.117188,
-3.650391,-5.515625,-3.345703,-3.962891,-4.683594,-3.224609,-4.515625,-4.156250,-3.578125,-4.578125,-4.191406,-3.601562,-3.753906,-4.628906,-3.640625,-3.402344,-3.271484,-3.531250,-3.921875,-3.791016,
-4.617188,-3.523438,-3.623047,-3.855469,-4.589844,-3.361328,-3.554688,-3.791016,-2.162109,-3.025391,-2.318359,-3.976562,-3.072266,-5.304688,-3.736328,-3.480469,-3.128906,-3.783203,-3.849609,-3.556641,
-3.857422,-3.658203,-4.140625,-6.070312,-4.703125,-4.832031,-4.875000,-4.449219,-4.492188,-4.066406,-5.328125,-5.812500,-6.234375,-5.015625,-5.511719,-5.484375,-5.808594,-5.464844,-4.476562,-5.527344,
-6.082031,-6.343750,-5.003906,-4.562500,-4.910156,-3.720703,-3.962891,-4.179688,-6.027344,-3.837891,-4.046875,-3.283203,-4.738281,-3.324219,-2.287109,-2.484375,-3.773438,-3.521484,-3.767578,-5.250000,
-6.132812,-3.402344,-4.777344,-3.451172,-4.617188,-5.003906,-4.835938,-4.484375,-4.968750,-4.734375,-4.617188,-7.007812,-4.882812,-5.636719,-3.978516,-3.937500,-3.572266,-3.697266,-4.777344,-2.980469,
-2.736328,-4.441406,-3.974609,-6.589844,-6.558594,-6.742188,-7.472656,-2.576172,-1.937500,-2.703125,-4.214844,-2.148438,-2.802734,-3.255859,-6.261719,-3.000000,-1.620117,-1.177734,-1.342773,-0.998535,
-4.578125,-2.013672,-1.773438,-3.175781,-2.634766,-2.691406,-2.699219,-2.919922,-2.796875,-2.597656,-5.281250,-3.242188,-4.468750,-1.943359,-2.345703,-1.904297,-2.001953,-3.093750,-3.390625,-3.472656,
-3.433594,-3.332031,-3.734375,-3.708984,-3.982422,-7.085938,-5.007812,-4.878906,-2.361328,-1.833008,-1.853516,-2.162109,-2.632812,-4.687500,-2.882812,-4.398438,-2.986328,-2.853516,-2.718750,-4.250000,
-4.375000,-2.406250,-3.777344,-1.609375,-1.780273,-3.923828,-1.681641,-1.496094,-1.629883,-1.488281,-1.022461,-4.468750,-1.047852,-1.311523,-1.174805,-1.508789,-1.878906,-3.705078,-4.046875,-4.289062,
-2.417969,-4.671875,-3.271484,-2.632812,-3.990234,-4.550781,-4.437500,-1.885742,-3.373047,-4.699219,-1.986328,-2.900391,-2.662109,-2.396484,-1.581055,-2.554688,-3.750000,-3.267578,-2.080078,-4.570312,
-1.936523,-4.253906,-3.855469,-2.960938,-1.779297,-2.789062,-0.279297,-0.247070,-2.136719,-0.459473,-2.708984,-0.884766,-2.330078,-2.093750,-1.955078,-1.820312,-1.447266,-1.333008,-1.561523,-0.744629,
-1.598633,-3.642578,-2.228516,-1.868164,-0.473877,0.792480,-1.461914,-0.915039,-0.944336,-0.114990,-0.019455,0.658203,0.160645,-1.545898,-1.066406,-1.820312,-1.668945,-1.409180,-0.437012,-0.147949,
-0.987305,-1.854492,-1.129883,0.879883,0.736816,-0.948242,-0.206299,0.515137,-2.205078,-0.336914,-0.204712,1.079102,0.876953,0.399170,-0.947266,0.393555,-0.594238,-1.324219,1.081055,-0.658691,
-1.204102,0.857910,-1.695312,-0.743164,-1.971680,-3.019531,1.183594,-0.207153,-1.015625,-3.447266,-3.857422,0.648926,1.128906,-0.172974,0.647461,0.035645,-3.455078,-1.776367,-3.542969,-4.519531,
-3.484375,-0.565430,-0.932617,-0.523926,-2.312500,-1.455078,-1.168945,-3.642578,-4.023438,-2.748047,-4.097656,-3.939453,-1.939453,-4.082031,-2.476562,-2.761719,-2.064453,-2.361328,-2.679688,-2.201172,
-2.585938,-1.708984,-2.287109,-2.853516,-2.367188,-1.973633,-2.912109,-2.302734,-0.966309,-1.951172,-1.892578,-2.134766,-1.416016,-2.939453,-1.945312,-2.759766,-2.437500,-3.333984,-3.107422,-2.388672,
-1.943359,-3.744141,-2.451172,-3.732422,-3.447266,-2.876953,-2.830078,-2.257812,-3.003906,-3.009766,-2.648438,-3.060547,-3.093750,-3.416016,-3.498047,-3.392578,-3.425781,-3.099609,-3.492188,-2.818359,
-2.322266,-2.906250,-2.279297,-2.693359,-2.667969,-2.128906,-2.552734,-4.320312,-6.070312,-5.441406,-2.667969,-2.949219,-2.175781,-1.992188,-2.451172,-2.917969,-2.937500,-1.977539,-1.456055,-2.087891,
-2.666016,-2.904297,-2.554688,-3.443359,-2.847656,-2.375000,-0.671387,-3.027344,-2.656250,-4.007812,-2.750000,-2.707031,-1.170898,-2.865234,-0.881836,-2.337891,-3.726562,-3.562500,-5.777344,-4.597656,
-6.570312,-5.195312,-2.597656,-3.972656,-1.641602,-3.812500,-3.960938,-4.195312,-1.747070,-4.328125,-4.691406,-3.972656,-3.576172,-3.072266,-1.805664,-2.621094,-2.396484,-4.628906,-2.734375,-1.297852,
-1.677734,-1.580078,-2.556641,-2.019531,-1.291016,-0.307861,-1.239258,-0.599609,-1.737305,-1.813477,-1.107422,-2.056641,-2.308594,-4.878906,-2.759766,-1.526367,-2.181641,-2.023438,-2.998047,-2.156250,
-1.986328,-1.264648,-4.382812,-1.592773,-4.640625,-4.144531,-5.230469,-2.035156,-3.742188,-3.062500,-3.000000,-2.261719,-1.425781,-2.365234,-2.511719,-4.117188,-1.918945,-5.187500,-1.456055,-2.107422,
-1.393555,-3.066406,-3.148438,-2.490234,-1.664062,-0.679199,-3.269531,-4.835938,-1.718750,-2.787109,-2.568359,-0.256348,-2.445312,-1.081055,-5.250000,-0.954590,-3.486328,-4.183594,-3.226562,-0.891602,
-1.249023,-2.837891,-2.625000,-2.787109,-0.793457,-2.103516,-1.885742,-0.889648,-3.771484,-3.689453,-2.775391,-3.324219,-3.904297,-2.642578,-1.624023,-2.949219,-3.117188,-2.150391,-3.386719,-4.105469,
-5.132812,-4.441406,-2.662109,-2.529297,-2.472656,-3.359375,-1.491211,-2.867188,-3.304688,-2.630859,-3.039062,-2.666016,-1.538086,-1.989258,-2.810547,-4.667969,-3.056641,-1.850586,-2.679688,-2.779297,
-3.250000,-2.380859,-1.415039,-0.708008,-3.794922,-2.755859,-2.503906,-0.866211,0.232544,-2.125000,-1.512695,-3.953125,-3.738281,-0.692871,-2.390625,-1.330078,-2.021484,-3.164062,-2.166016,-3.193359,
-0.722168,-1.186523,-3.898438,-6.277344,-6.203125,-0.564453,-3.562500,-3.425781,-5.761719,-1.448242,-3.500000,-2.181641,-1.763672,-4.062500,0.362549,-2.537109,-2.611328,-4.878906,-0.224976,-2.148438,
-0.714355,-0.557617,0.491211,1.733398,-2.150391,2.095703,1.622070,0.775879,0.790527,-2.958984,0.001598,1.509766,2.294922,-3.138672,-0.291260,-2.001953,-3.269531,-1.511719,-0.072083,-0.340820,
-4.359375,-0.150269,1.007812,-2.503906,-1.949219,-2.958984,1.225586,0.951172,0.493164,-1.791992,-0.864258,-4.214844,-0.975586,-1.882812,-3.460938,-0.385742,-0.792480,-1.862305,-0.659668,-0.804199,
-1.131836,-3.570312,-0.315918,-2.703125,1.781250,-1.554688,-1.958984,-2.613281,-0.099731,0.785645,1.932617,-0.563965,-0.036896,-0.489258,0.349854,1.044922,-0.094360,-0.546875,-0.023224,-2.542969,
-1.277344,0.629883,-1.868164,-1.637695,0.405029,-1.705078,-0.256836,0.378906,-0.252197,0.392334,-0.438232,-0.679199,-0.599121,-1.771484,-1.370117,-1.376953,-3.085938,-1.199219,-2.107422,-2.050781,
-3.201172,-0.945801,-2.869141,-2.873047,-4.035156,-1.305664,-2.138672,-2.728516,-2.884766,-2.185547,-1.280273,-3.824219,-2.027344,-1.412109,-2.617188,-1.812500,-4.777344,-1.161133,-3.066406,-3.675781,
-1.686523,-2.791016,-2.097656,0.009422,-0.986816,-0.451416,-1.333984,-1.997070,-2.578125,-3.097656,-1.165039,-1.361328,-0.840820,-2.521484,-1.762695,-1.226562,-3.296875,-3.269531,-4.207031,-1.918945,
-3.347656,-3.503906,-3.560547,-1.882812,-1.729492,-3.218750,-1.417969,-1.817383,-1.717773,-3.335938,-2.507812,-0.708008,-2.904297,-2.867188,-3.750000,-1.118164,-2.070312,-2.173828,-2.306641,-1.411133,
-1.967773,-1.661133,-2.501953,-2.222656,-0.492432,0.233032,-0.881348,-1.296875,0.603027,-0.630859,0.246094,-0.433105,-0.875488,-2.185547,-2.140625,-2.894531,-1.554688,-0.731445,-1.886719,-3.925781,
-0.598633,-0.232788,-0.750488,-2.304688,-5.429688,-2.982422,-0.591797,-0.779785,-2.433594,-1.714844,-3.603516,-2.964844,-3.320312,-3.085938,-2.421875,-3.357422,-3.798828,-2.822266,-2.144531,-1.653320,
-1.917969,-1.456055,-3.496094,-2.445312,-1.598633,-1.500977,-1.560547,-0.697754,-1.337891,-1.954102,-2.091797,-1.479492,-0.653809,-2.355469,-1.139648,-1.936523,-2.095703,-0.963867,-2.808594,-2.857422,
-2.349609,-1.235352,-1.870117,-1.546875,-1.052734,-2.478516,-3.363281,-2.078125,-4.074219,-4.425781,-4.144531,-2.904297,-3.697266,-3.158203,-2.685547,-3.685547,-3.773438,-3.777344,-3.255859,-2.949219,
-4.117188,-3.679688,-3.298828,-2.789062,-3.111328,-2.964844,-3.005859,-2.265625,-2.847656,-2.607422,-3.990234,-2.423828,-3.441406,-2.927734,-2.986328,-2.904297,-1.765625,-1.743164,-1.484375,-1.479492,
-1.862305,-2.939453,-1.267578,-1.311523,-1.445312,-0.880371,-0.762695,-1.110352,-0.784668,-0.056152,-1.048828,-1.068359,-2.429688,-2.867188,-1.371094,-2.451172,-3.244141,-3.099609,-1.187500,-1.620117,
-1.528320,-1.469727,-1.263672,-1.283203,-1.037109,-1.659180,-1.895508,-2.677734,-1.449219,-2.552734,-2.416016,-0.229736,-0.633789,-2.037109,-1.661133,-1.128906,-1.359375,-0.909180,-0.479492,-1.347656,
-1.060547,0.026062,-0.393066,-0.360596,-2.687500,-0.314941,-0.393311,-0.243530,-0.079102,0.212769,-0.197388,-0.248169,-0.174438,0.454346,-1.455078,0.288818,-0.186035,-1.420898,-4.265625,-1.433594,
-3.535156,-1.000977,0.472656,-0.110535,-0.290039,-1.279297,-2.398438,-0.876953,-2.076172,-2.144531,-3.847656,-1.187500,-1.071289,-0.544434,-2.080078,-1.762695,-2.923828,-1.999023,-1.916992,-1.159180,
-3.259766,-2.679688,-0.659668,-1.844727,-2.492188,-2.044922,-0.321045,-2.300781,-2.533203,-2.699219,-1.094727,-2.386719,-1.980469,-1.208984,-0.586426,-1.748047,0.119019,0.209106,0.014015,0.955566,
-1.577148,-0.482178,-0.509766,-2.787109,-2.023438,0.089294,-3.017578,-0.881836,-2.074219,-1.320312,-0.730957,-1.353516,-2.083984,-1.311523,-1.861328,-2.078125,-1.487305,-2.365234,-3.166016,-2.246094,
-2.906250,-3.072266,-1.485352,-3.685547,-2.000000,-1.349609,-2.773438,-2.908203,-3.294922,-1.844727,-1.186523,-1.606445,-2.177734,-0.984863,-3.947266,-2.302734,-1.441406,-1.455078,-0.903320,-1.035156,
-0.891113,-2.478516,-1.689453,-0.139893,-1.172852,-2.882812,-3.042969,-1.462891,-2.378906,-2.132812,-0.918457,-1.706055,-3.539062,-2.773438,-2.568359,-1.539062,-2.027344,-3.062500,-2.203125,-3.345703,
-2.658203,-3.990234,-3.740234,-3.580078,-4.093750,-3.693359,-2.142578,-2.148438,-3.267578,-2.785156,-3.994141,-4.003906,-1.948242,-3.980469,-2.910156,-2.582031,-2.460938,-2.248047,-1.847656,-3.740234,
-3.503906,-2.134766,-0.962402,-2.183594,-0.907715,-1.577148,-1.530273,-1.489258,-0.260010,-0.258301,-0.799805,-1.544922,-3.419922,-2.121094,-2.189453,-0.347656,-1.680664,-1.658203,0.209839,-1.275391,
0.237305,-2.833984,-0.511230,-0.718750,0.723633,0.665039,0.127563,0.021500,0.030914,-0.573730,-1.796875,-0.701660,-3.472656,-1.098633,-0.715820,-1.549805,-1.142578,-0.235352,0.079956,-0.538086,
-0.971680,0.208862,-0.733887,0.504883,-1.051758,-0.578613,-0.343750,0.132690,-0.809082,0.787598,1.161133,0.885254,1.345703,0.131958,0.151245,-0.015518,-0.376709,1.242188,0.163940,0.162964,
1.640625,0.110657,1.106445,1.998047,0.523438,-1.682617,0.339844,0.032623,0.457275,0.830078,-2.830078,-1.944336,-2.277344,-0.705078,-2.363281,-0.145020,-0.595215,-1.270508,-0.738281,-4.156250,
-2.542969,-2.062500,-1.204102,-1.634766,-0.344727,-0.815918,-0.643555,-1.129883,-1.084961,-1.419922,-1.727539,-2.423828,-2.935547,-2.062500,-0.694336,-0.475098,-1.089844,-2.458984,-3.105469,-1.214844,
0.044678,-2.087891,-0.800293,-3.179688,-1.159180,-0.756836,-0.880859,-1.951172,-1.122070,-0.245605,-0.134766,-1.542969,-0.697266,-1.155273,-0.749512,-1.685547,-1.670898,-5.304688,-2.498047,-1.597656,
-2.919922,-3.363281,-4.835938,-2.689453,-3.562500,-3.187500,-1.806641,-2.755859,-1.609375,-1.269531,-3.343750,-3.558594,-3.001953,-3.541016,-3.494141,-1.153320,-2.255859,-0.899414,-1.105469,-1.218750,
-1.263672,-0.954590,0.670410,1.066406,1.242188,0.146973,-0.323486,-2.976562,-1.062500,-1.157227,0.053345,1.089844,-0.672852,0.275879,-2.042969,0.503906,0.477539,-0.119019,0.658203,-1.567383,
1.991211,1.642578,0.090393,1.662109,0.175293,0.286377,0.203857,1.025391,-0.934570,-0.088928,0.877930,1.497070,-0.436523,-0.874023,-0.259766,0.167236,0.449951,1.681641,-2.054688,-0.372314,
-0.352051,-0.009399,1.082031,1.040039,0.797363,0.168823,2.406250,2.212891,2.683594,0.718750,0.948242,0.488770,0.636719,1.164062,1.732422,0.871094,3.539062,2.513672,1.125977,1.465820,
3.416016,2.794922,0.832520,2.240234,3.906250,1.663086,2.330078,1.378906,2.705078,-0.620605,0.657715,1.510742,2.396484,0.955566,1.045898,-0.053345,0.289551,0.427734,1.037109,0.845215,
0.271729,-0.445312,-0.205566,0.820801,-2.908203,-0.639648,0.522461,1.513672,0.578125,0.104614,0.216553,-0.504395,0.440674,-0.243042,1.730469,0.670410,-0.774414,-0.069702,1.038086,1.179688,
1.867188,-0.146729,0.108521,0.055481,2.242188,1.421875,0.971680,0.562500,0.669434,-1.321289,0.038757,-0.036591,1.086914,1.144531,0.080566,-1.105469,-0.484131,0.232788,-1.529297,-0.189331,
-0.246704,1.017578,1.912109,0.918945,0.236816,-1.831055,-1.414062,-0.782227,-1.118164,-1.311523,-0.297119,-1.272461,-1.496094,0.614746,-0.900879,1.054688,1.083008,-0.157471,-0.319092,0.604004,
-0.612305,-0.441650,0.853027,1.227539,1.360352,2.351562,0.482910,0.799316,2.636719,1.619141,0.845703,0.619141,1.641602,1.374023,2.478516,1.989258,2.962891,2.753906,2.576172,3.427734,
2.279297,1.431641,0.764160,1.639648,2.025391,-0.453369,0.281982,0.061951,2.822266,1.330078,1.007812,1.766602,0.193726,2.287109,1.366211,-0.420654,-0.875977,1.930664,1.432617,2.259766,
2.820312,2.324219,0.770020,-0.557617,-2.625000,-0.197144,0.216919,0.931152,0.683594,0.189819,0.309814,1.199219,2.091797,0.574219,0.947266,2.681641,1.542969,3.839844,0.404541,0.785156,
3.292969,3.732422,3.931641,3.849609,3.880859,3.478516,1.254883,2.234375,2.570312,2.914062,2.302734,2.996094,2.332031,3.781250,2.375000,1.949219,-0.739258,1.358398,0.036713,-1.603516,
0.621582,1.694336,-0.447266,0.252441,0.401611,1.465820,2.203125,1.063477,0.655762,1.302734,0.377197,1.819336,1.167969,0.835938,1.902344,0.643555,1.935547,3.125000,2.615234,2.126953,
2.060547,3.388672,2.072266,1.959961,2.714844,2.685547,2.630859,1.738281,0.752441,2.158203,2.521484,2.146484,2.416016,0.572266,2.568359,2.611328,2.949219,1.194336,0.277588,1.334961,
1.517578,1.708984,2.146484,1.768555,-0.189331,-0.329346,1.252930,-0.471924,1.293945,1.868164,1.672852,1.349609,1.372070,0.602539,1.096680,-0.723145,-0.649414,-0.588867,1.070312,0.436523,
0.678223,1.533203,2.230469,2.458984,-0.185181,1.631836,0.992676,2.492188,2.978516,1.361328,2.404297,2.876953,1.117188,2.419922,2.708984,1.956055,1.862305,3.103516,2.001953,2.628906,
1.119141,0.962891,2.558594,0.549805,0.602051,1.375977,2.529297,1.357422,1.711914,0.597168,-0.150146,2.669922,3.042969,1.493164,3.044922,2.257812,1.621094,1.080078,1.380859,2.429688,
-0.406982,1.703125,2.779297,1.874023,1.786133,1.870117,2.408203,2.777344,0.508789,1.823242,1.245117,1.150391,2.837891,3.648438,3.044922,2.218750,3.621094,2.785156,1.972656,3.593750,
4.523438,1.302734,3.371094,4.457031,4.492188,4.050781,4.914062,5.445312,4.937500,3.869141,1.866211,3.933594,4.593750,3.929688,2.492188,4.664062,4.300781,3.734375,4.808594,4.437500,
3.164062,3.783203,2.851562,4.425781,3.597656,2.085938,3.173828,4.062500,1.613281,3.492188,4.394531,4.328125,3.748047,3.677734,4.449219,3.880859,3.544922,3.568359,4.808594,2.248047,
4.417969,5.605469,5.375000,3.121094,3.900391,3.853516,5.046875,4.187500,4.929688,5.878906,5.910156,4.644531,5.218750,3.605469,4.109375,5.570312,4.007812,4.968750,2.359375,4.304688,
5.378906,4.246094,4.316406,4.656250,3.806641,4.089844,4.574219,3.480469,4.355469,2.996094,1.361328,3.246094,2.648438,3.826172,3.408203,4.597656,3.015625,3.675781,2.390625,3.695312,
4.585938,4.441406,4.613281,4.710938,5.160156,4.507812,4.734375,5.167969,4.414062,5.363281,5.285156,6.105469,6.257812,5.691406,5.832031,4.421875,5.679688,5.562500,7.164062,6.796875,
6.601562,6.593750,4.167969,6.890625,6.593750,5.207031,6.382812,7.820312,7.546875,7.410156,5.707031,7.082031,6.875000,6.664062,3.431641,7.406250,7.042969,5.304688,7.535156,8.046875,
7.382812,5.308594,8.281250,8.437500,8.398438,7.015625,7.363281,8.640625,7.625000,9.320312,7.585938,8.953125,10.132812,9.492188,9.898438,10.296875,7.511719,10.382812,8.156250,10.429688,
9.156250,11.828125,12.046875,8.773438,-65504.000000,0.515625,1.060547,0.387451,-0.837402,1.350586,0.206787,0.015701,0.961914,1.089844,1.205078,-0.563477,0.429443,-1.781250,0.504395,0.124390,
-1.302734,1.443359,0.658203,0.795410,-1.273438,0.553711,1.518555,1.683594,1.031250,-0.818848,0.906250,0.429932,0.686035,-0.808105,-0.135864,0.161865,-0.010605,0.049011,-0.196533,-0.259766,
-1.498047,-1.537109,-0.453125,-1.715820,-0.522949,-0.613770,-0.356201,-1.493164,-1.042969,-0.549316,-0.606445,-0.475098,0.126587,-2.455078,-0.997559,-1.051758,-0.878906,-0.710449,0.721191,0.012581,
-0.336182,0.430420,0.560059,0.345215,1.023438,1.087891,0.451416,0.555664,1.516602,1.266602,1.631836,1.702148,0.800293,0.303711,0.611328,1.432617,2.015625,1.369141,1.313477,-0.310791,
1.689453,1.589844,0.761230,0.364258,-0.965820,1.547852,2.072266,0.595215,1.614258,0.214355,-0.068237,0.467285,0.156616,0.310547,-2.070312,0.481689,0.054779,-0.821777,-0.012177,-2.884766,
0.679199,1.245117,1.438477,1.441406,0.794922,-1.642578,-1.497070,-0.147827,0.468018,0.271240,0.458740,-0.397217,0.855957,1.538086,1.114258,0.431641,1.266602,1.834961,-1.529297,0.552734,
2.250000,-0.957031,1.370117,-1.788086,1.984375,2.248047,-1.606445,2.261719,1.776367,0.084717,-1.304688,-0.023727,-1.777344,1.367188,-0.476562,0.821777,-0.517578,0.189819,1.004883,0.719238,
-2.660156,-0.833008,-1.048828,0.416992,-0.105652,0.423096,1.001953,0.117188,-0.234497,-0.512695,0.161987,-1.451172,0.249634,-0.905762,0.393555,-0.152344,0.559082,1.050781,-1.559570,0.582031,
0.339844,1.434570,0.205688,1.326172,1.916016,-1.280273,1.310547,0.562988,1.068359,1.430664,1.129883,1.083984,0.582520,0.394287,-0.284424,0.142334,-0.698730,0.742188,0.342529,0.026108,
0.270752,-0.682129,0.362305,0.189941,-1.134766,-1.368164,-0.619141,-0.337891,-1.005859,-0.632812,-0.246338,-0.091309,-0.590820,-0.381348,-1.006836,0.009804,-2.433594,-0.660645,-0.555176,-2.658203,
-0.491943,0.346924,0.251465,0.225220,-1.534180,0.490967,0.243896,-0.426758,0.387207,0.037415,-1.761719,1.141602,0.330566,1.151367,1.190430,-0.037231,0.457031,0.612793,0.856445,0.364746,
0.615234,0.427490,-1.588867,0.912109,-0.894043,0.627930,1.036133,0.416504,0.049744,-1.310547,-1.854492,-1.792969,-2.878906,-1.155273,-0.433105,-1.253906,-0.089600,-1.210938,0.087341,-0.284180,
0.287598,-0.292969,-0.226440,-0.912109,0.195068,-0.623535,0.495361,1.196289,0.945801,0.637695,1.065430,1.003906,0.990234,0.751465,-0.506348,1.243164,1.375000,1.678711,2.851562,2.007812,
1.332031,1.764648,2.417969,2.808594,3.097656,3.152344,-0.348633,0.752930,1.046875,3.140625,2.830078,2.142578,3.482422,1.281250,0.523926,1.659180,-2.402344,1.219727,2.058594,2.054688,
1.861328,1.770508,-0.974121,0.753418,-0.750000,1.392578,1.420898,1.733398,1.917969,1.816406,1.880859,-0.701172,1.117188,-0.479004,0.630859,1.818359,1.714844,1.372070,1.708984,1.084961,
0.280518,1.046875,2.478516,1.630859,1.585938,2.798828,2.095703,1.787109,1.991211,-0.015541,1.919922,0.848145,1.517578,1.496094,1.519531,0.371094,0.844727,0.157227,-1.014648,-0.314209,
-1.051758,0.455078,-0.092102,-1.102539,-0.341797,-1.062500,0.429688,2.050781,0.558105,-0.765137,-0.951660,-1.640625,-2.349609,-1.956055,-2.244141,-1.379883,-1.561523,-0.892578,-2.240234,-1.626953,
-1.320312,-2.171875,-2.617188,-1.714844,-2.062500,-0.603516,1.263672,-0.146362,0.086060,-1.149414,-1.553711,-1.334961,-1.445312,-0.134399,-0.721191,-0.664062,-1.510742,-0.666992,-0.982910,-1.487305,
-0.973633,-1.980469,-2.482422,-1.692383,-2.505859,-2.679688,-1.415039,-1.809570,-2.044922,-2.435547,-1.151367,-1.293945,-0.943359,-2.318359,-2.558594,-3.677734,-3.148438,-2.458984,-1.751953,-1.806641,
-1.709961,-1.891602,-2.738281,-3.101562,-1.643555,-2.447266,-1.060547,0.223999,-0.603516,0.022049,-1.010742,-0.542480,-0.802246,-0.694336,-1.250977,-0.264648,-0.406738,0.226562,-0.365967,-0.149292,
-1.149414,-0.684082,0.365967,0.196045,-0.569336,0.701172,-0.919922,-1.022461,-0.642090,-0.342529,-0.542480,-2.041016,0.798828,0.648438,-0.493164,0.458008,0.034515,-0.190063,-0.464111,-1.128906,
-0.261475,-0.087463,-0.705566,-0.230957,-1.424805,-1.066406,-2.140625,-0.810547,-0.976074,-0.024521,-1.180664,-1.733398,-1.509766,-2.539062,-1.123047,-2.089844,-0.693848,-0.726562,-0.427490,-0.222168,
-0.943848,-0.297607,-0.742188,-0.604004,-0.232300,-0.020737,1.005859,0.316406,-0.250732,-1.236328,-0.515625,-0.828613,-0.720215,-0.541016,-0.372803,-0.804199,-0.996582,-1.425781,-0.567383,-0.233276,
-1.140625,-0.577148,-1.357422,-1.528320,-1.135742,-2.009766,-1.548828,-2.849609,-2.164062,-2.601562,-1.840820,-2.015625,-0.822266,-1.646484,-2.146484,-3.019531,-3.384766,-2.693359,-2.068359,-0.939941,
-1.125977,-1.512695,-1.861328,-0.589355,-0.830078,-1.075195,-1.245117,-0.451172,-0.721191,-0.968750,0.445557,0.836426,0.298096,-0.380615,-0.269775,-0.280762,0.306641,0.442383,0.214233,0.009613,
-0.772949,-1.118164,0.289307,0.810059,0.226562,1.931641,1.201172,-0.129395,-0.632812,-0.507324,-1.056641,-0.627441,-0.778809,-0.887207,-0.351318,-0.605469,-0.335205,-0.489014,-0.653809,-0.764648,
-0.419922,-0.299561,-0.077393,-0.176147,-0.216309,-0.757324,-0.273682,-0.621582,-0.155029,0.157959,0.134766,-0.079590,-0.292236,0.111267,1.370117,1.474609,2.236328,1.750000,1.439453,1.835938,
2.033203,-0.236572,1.879883,1.813477,0.639648,-0.676270,2.138672,2.138672,2.082031,1.997070,1.820312,1.425781,0.852051,1.624023,-1.367188,1.442383,1.047852,1.032227,-1.581055,0.947754,
1.241211,1.002930,0.733398,1.120117,0.364502,0.233887,1.122070,-1.032227,-0.047058,-0.032928,0.086792,0.892578,0.809570,0.805176,-1.179688,0.078308,-0.468750,0.139038,0.379395,1.262695,
2.062500,-0.830566,0.943848,0.229492,-0.984863,1.733398,1.642578,1.615234,1.434570,1.954102,2.517578,0.386963,1.718750,1.396484,1.589844,1.830078,0.729980,1.373047,0.152954,-1.706055,
1.747070,0.583008,0.782227,1.262695,1.649414,1.574219,1.417969,-0.320312,0.625977,0.073303,-1.076172,0.358398,-0.187134,0.507812,0.435791,0.708008,0.966797,-0.050842,0.079224,-2.289062,
0.592285,0.616211,1.027344,1.147461,1.501953,0.925293,0.975586,1.033203,1.614258,1.845703,1.550781,-0.072021,1.479492,1.156250,1.566406,2.111328,2.267578,3.029297,2.449219,2.406250,
3.474609,3.699219,2.837891,0.602051,2.976562,2.652344,1.063477,3.169922,3.136719,2.224609,1.953125,1.686523,1.879883,1.707031,1.628906,2.279297,1.551758,1.059570,1.135742,2.253906,
2.123047,2.433594,2.136719,1.800781,1.739258,1.574219,1.944336,2.470703,1.742188,1.587891,4.605469,2.263672,2.513672,3.041016,2.667969,1.889648,0.450684,2.615234,3.576172,2.396484,
3.744141,3.623047,3.343750,3.455078,3.511719,3.671875,2.695312,3.140625,1.391602,2.349609,3.550781,4.597656,2.574219,3.435547,3.585938,2.474609,4.320312,4.011719,4.625000,3.417969,
3.496094,1.452148,3.988281,3.671875,4.281250,3.949219,3.781250,2.923828,1.778320,2.654297,1.731445,3.343750,4.191406,3.423828,4.414062,3.115234,4.648438,4.714844,5.136719,3.673828,
4.589844,4.449219,3.662109,4.890625,4.996094,4.687500,5.734375,5.179688,5.671875,4.101562,5.773438,4.914062,3.496094,5.265625,4.792969,5.539062,6.296875,6.300781,6.781250,4.625000,
4.777344,5.128906,5.285156,4.781250,4.238281,2.955078,3.324219,3.146484,4.238281,3.896484,4.667969,3.343750,2.058594,0.806641,2.923828,3.324219,3.105469,2.818359,3.416016,2.763672,
3.025391,3.408203,3.224609,2.724609,3.187500,2.167969,2.484375,0.958496,1.915039,1.341797,1.021484,3.208984,1.543945,3.148438,2.667969,1.674805,0.618652,1.260742,1.326172,2.808594,
3.066406,2.277344,1.961914,3.537109,0.614746,0.524902,2.851562,-0.148804,1.056641,0.121582,1.079102,1.211914,-0.390137,0.011856,-0.317139,-0.492432,-0.842285,-0.087219,0.018829,1.646484,
0.588867,-0.057770,-0.804688,-0.326904,1.344727,-0.351807,1.268555,0.094238,-1.230469,-1.425781,-1.149414,-1.151367,-0.723145,-0.101440,0.474854,0.515625,0.325684,1.783203,1.053711,1.457031,
0.887207,0.922852,1.820312,0.869629,1.586914,1.589844,1.308594,1.018555,1.609375,2.003906,2.167969,2.328125,1.375000,1.113281,1.422852,0.110535,1.714844,1.608398,1.445312,0.892578,
1.398438,1.151367,0.762695,0.830078,0.483887,-0.056061,0.635254,0.597168,2.054688,2.251953,1.403320,1.091797,0.369141,0.628418,0.388184,1.384766,0.735352,1.909180,1.622070,1.413086,
1.307617,1.991211,1.890625,2.994141,2.134766,1.196289,2.265625,2.337891,0.293945,2.771484,2.251953,2.466797,2.033203,1.474609,3.310547,3.226562,2.433594,2.447266,0.662109,2.328125,
2.882812,2.208984,2.474609,2.230469,1.805664,1.525391,1.158203,1.867188,1.095703,0.663574,0.300781,0.809570,2.191406,1.157227,0.669434,0.704590,0.076660,0.291260,1.031250,0.842773,
0.326416,0.271484,0.859863,-0.029907,0.156006,0.961914,3.623047,1.380859,1.597656,0.720703,0.044830,-0.456299,0.314941,0.909668,1.819336,1.064453,0.779785,0.624512,3.160156,1.534180,
1.216797,0.337402,1.021484,2.427734,1.380859,1.156250,0.391113,0.043976,0.291260,0.177979,0.148804,0.877930,1.617188,0.116394,-0.620605,-0.732910,-0.405273,0.616699,0.503906,0.562012,
0.180908,-0.024353,-0.188232,-0.081543,-0.141235,-0.867188,-1.175781,-0.761719,-0.548340,0.133179,2.789062,0.804688,0.490479,-0.546387,-0.795410,0.154419,0.971191,0.540039,-0.258545,0.709473,
1.428711,1.735352,1.575195,1.886719,3.349609,2.310547,1.398438,0.654297,0.963379,1.545898,0.754395,0.745605,2.378906,2.515625,1.851562,2.353516,2.078125,0.733887,0.730957,-0.618164,
0.410400,1.388672,0.926270,1.278320,0.937500,1.068359,1.114258,0.418457,-0.509766,-0.804688,-0.730469,-1.329102,-0.032074,0.424561,-0.320801,0.163330,1.187500,0.831543,1.562500,0.806641,
1.046875,-0.845215,-0.386963,-1.963867,-0.588867,0.730957,0.008690,2.535156,2.523438,0.061188,0.471436,-1.156250,1.763672,2.179688,2.519531,-0.101013,2.757812,0.171387,2.248047,2.519531,
1.867188,-0.584473,-0.647461,1.197266,0.612793,1.508789,0.476807,0.392822,1.410156,-2.341797,-1.056641,-1.967773,1.323242,0.598145,0.027969,0.208740,-1.411133,-0.795898,0.582520,1.135742,
1.477539,-0.672852,1.068359,0.885254,1.217773,-1.999023,-0.711914,0.308105,-1.044922,-1.645508,1.244141,1.403320,1.093750,-1.056641,0.961426,0.178833,2.312500,1.240234,1.985352,2.205078,
1.647461,1.681641,-0.804688,1.594727,1.583008,1.882812,1.259766,1.394531,0.284912,1.269531,0.977051,-0.064697,1.103516,1.343750,1.516602,1.020508,1.166992,0.703125,-0.231445,-0.925781,
-0.384766,-0.705566,0.267578,0.251221,-0.392090,-0.284668,-0.278320,-0.436768,-0.675293,-0.760254,-0.864258,-0.485352,-0.844238,-0.797852,0.086426,0.196655,0.425537,0.580566,0.158936,-0.011124,
0.561523,0.973633,0.309326,-0.358643,-0.355469,0.469971,0.489502,0.196045,0.884766,1.189453,0.261475,0.991699,0.512695,0.633789,-0.221191,1.040039,0.770020,1.301758,1.399414,0.440430,
0.718262,-0.426758,-0.469971,-0.232178,-0.661133,-0.560547,-0.083252,-0.078857,-1.249023,-1.145508,-1.938477,-0.145508,0.355957,-0.083374,-0.742676,-1.261719,-0.552246,-0.691406,-0.040070,-0.027649,
0.095886,0.531250,0.508301,0.379639,0.484131,0.718750,0.545410,-0.159180,0.877930,1.559570,1.845703,0.738770,1.178711,0.007950,1.165039,1.656250,1.110352,-0.303711,1.775391,2.333984,
2.333984,2.078125,1.764648,2.345703,1.626953,0.573242,1.273438,0.980957,-0.695312,1.009766,0.576660,0.945801,0.673828,-0.362061,1.407227,1.761719,1.881836,1.445312,0.506836,0.541992,
0.077209,-0.303467,-0.328369,0.737305,0.643066,0.633789,0.828613,-0.778320,0.478271,-1.114258,1.163086,1.437500,1.259766,1.415039,1.430664,1.860352,1.652344,1.307617,0.341064,1.815430,
1.875977,1.447266,1.461914,1.411133,0.421875,1.022461,1.006836,1.718750,1.578125,1.387695,0.072205,1.510742,0.681152,-1.704102,0.832031,0.237061,0.444336,1.080078,0.059662,0.151855,
0.279541,0.744629,-1.049805,-0.752441,-0.608398,-0.335449,-1.032227,-0.800781,-1.113281,-1.831055,-1.383789,-0.972656,-0.659180,-0.342285,0.303223,-0.076965,-0.791504,0.037994,-0.611328,0.105408,
0.237061,0.182739,0.344971,0.274658,0.209717,0.592773,-0.747559,-0.410156,-0.292236,-0.233398,1.254883,2.107422,1.877930,1.595703,1.204102,1.465820,-0.400879,0.490234,0.164429,1.015625,
-0.147461,-0.519531,0.552734,0.227295,1.523438,1.275391,0.611816,-0.145752,0.754395,0.314453,0.753418,0.601562,0.280518,0.713379,1.340820,1.927734,2.384766,0.773926,-0.056152,-0.204224,
0.062408,0.682129,0.795898,1.067383,0.732910,1.537109,1.264648,1.331055,2.214844,2.082031,0.733887,2.228516,0.404541,0.630371,4.109375,4.554688,2.449219,2.947266,1.033203,3.949219,
4.585938,4.710938,4.015625,3.341797,2.802734,3.625000,2.400391,3.773438,3.783203,3.585938,2.921875,1.261719,2.722656,1.600586,2.294922,3.027344,2.365234,2.058594,0.304443,2.541016,
2.222656,2.015625,1.139648,-0.290039,0.949707,1.429688,2.000000,1.443359,1.618164,1.325195,-0.133911,1.967773,2.064453,-0.047516,1.614258,1.690430,-0.997559,0.401855,-0.740723,1.562500,
2.132812,2.023438,1.854492,1.251953,1.185547,1.040039,-0.136841,-2.505859,0.231567,0.472900,0.490723,0.605957,0.867188,-0.717773,0.176270,-0.479004,-1.457031,-0.435059,-1.129883,-0.665039,
-1.190430,-1.991211,-4.585938,-3.134766,-3.941406,-2.558594,-1.836914,-2.898438,-2.484375,-2.447266,-4.175781,-1.899414,-2.994141,-2.546875,-3.441406,-4.421875,-2.339844,-2.728516,-2.535156,-2.271484,
-3.320312,-2.869141,-1.822266,-3.083984,-1.211914,-1.814453,-2.353516,-2.017578,-2.669922,-1.558594,-2.248047,-2.423828,-4.429688,-3.009766,-3.322266,-3.740234,-1.879883,-1.913086,-1.923828,-1.596680,
-1.871094,-2.656250,-1.843750,-2.835938,-2.761719,-3.738281,-3.138672,-2.861328,-3.001953,-3.195312,-2.931641,-2.970703,-3.810547,-4.550781,-4.304688,-4.792969,-2.798828,-2.931641,-3.482422,-3.351562,
-2.828125,-3.380859,-3.164062,-5.238281,-4.648438,-3.238281,-2.466797,-3.273438,-2.074219,-2.636719,-3.441406,-3.519531,-3.207031,-2.917969,-2.714844,-3.285156,-2.687500,-3.431641,-5.515625,-3.541016,
-3.273438,-2.154297,-2.072266,-2.269531,-1.870117,-1.641602,-2.476562,-4.636719,-3.974609,-2.943359,-3.968750,-2.390625,-2.097656,-4.699219,-4.156250,-4.871094,-3.919922,-3.128906,-3.615234,-4.183594,
-5.601562,-5.101562,-5.007812,-4.843750,-5.015625,-4.046875,-4.445312,-6.648438,-5.171875,-5.417969,-5.792969,-3.767578,-4.621094,-5.585938,-4.207031,-3.640625,-3.238281,-2.507812,-2.667969,-5.656250,
-5.281250,-4.281250,-2.894531,-2.732422,-3.589844,-2.935547,-3.048828,-3.880859,-2.699219,-1.928711,-2.115234,-2.775391,-2.050781,-2.261719,-4.453125,-2.296875,-3.048828,-3.267578,-2.681641,-3.253906,
-4.382812,-3.099609,-3.042969,-3.564453,-4.503906,-3.427734,-4.363281,-5.339844,-3.742188,-4.410156,-3.978516,-3.644531,-4.812500,-4.941406,-4.300781,-3.953125,-5.542969,-4.128906,-3.611328,-3.666016,
-3.419922,-3.945312,-3.490234,-4.332031,-3.734375,-4.218750,-4.121094,-4.339844,-3.388672,-3.605469,-3.812500,-2.761719,-3.365234,-2.712891,-3.574219,-2.945312,-5.125000,-3.380859,-3.677734,-3.320312,
-3.921875,-3.734375,-3.642578,-3.945312,-3.669922,-4.156250,-5.457031,-4.683594,-5.261719,-5.218750,-5.121094,-4.957031,-4.265625,-5.128906,-5.761719,-5.687500,-5.410156,-5.835938,-6.054688,-6.007812,
-5.710938,-4.800781,-5.980469,-6.632812,-6.832031,-5.105469,-4.906250,-4.988281,-4.250000,-4.285156,-4.550781,-6.042969,-4.375000,-4.457031,-3.953125,-5.660156,-3.603516,-2.314453,-2.542969,-3.384766,
-3.603516,-3.917969,-5.527344,-6.109375,-3.798828,-4.871094,-3.693359,-4.746094,-5.203125,-4.761719,-4.781250,-5.117188,-4.265625,-4.921875,-6.738281,-4.992188,-5.171875,-3.955078,-4.078125,-4.027344,
-4.144531,-5.164062,-3.289062,-3.220703,-4.851562,-4.402344,-7.410156,-7.761719,-7.636719,-8.460938,-2.988281,-2.294922,-2.970703,-4.429688,-2.570312,-3.343750,-3.658203,-7.171875,-3.560547,-2.173828,
-1.731445,-1.411133,-1.003906,-3.570312,-1.895508,-2.259766,-3.468750,-2.638672,-2.359375,-2.500000,-2.800781,-2.193359,-2.884766,-5.292969,-3.236328,-4.660156,-1.566406,-2.271484,-1.877930,-2.363281,
-3.292969,-3.337891,-3.373047,-3.027344,-3.267578,-3.585938,-2.798828,-3.843750,-6.417969,-5.351562,-5.808594,-2.958984,-2.503906,-1.936523,-1.805664,-2.503906,-3.355469,-3.023438,-4.554688,-3.205078,
-3.232422,-2.886719,-3.628906,-4.441406,-2.878906,-4.089844,-2.195312,-2.332031,-2.845703,-1.806641,-1.825195,-1.992188,-1.521484,-0.904297,-4.062500,-1.397461,-1.941406,-1.274414,-1.419922,-1.310547,
-3.222656,-3.941406,-3.527344,-2.587891,-5.046875,-3.261719,-2.064453,-3.064453,-4.414062,-3.630859,-2.210938,-3.710938,-5.089844,-2.466797,-3.617188,-3.515625,-2.910156,-1.864258,-2.957031,-4.070312,
-3.480469,-2.955078,-4.902344,-2.142578,-4.253906,-4.132812,-3.099609,-2.546875,-3.791016,-0.923340,-0.755859,-1.328125,-0.588867,-2.363281,-1.288086,-2.617188,-2.234375,-2.066406,-1.921875,-1.451172,
-1.559570,-1.418945,0.145142,-1.405273,-2.730469,-2.431641,-2.443359,-1.016602,0.618652,-1.401367,-0.578613,-0.851562,-0.283691,-0.166016,0.581055,0.198486,-0.504395,-0.480957,-1.418945,-1.899414,
-2.000000,-0.960938,-0.621094,-1.116211,-1.007812,-1.181641,0.818359,0.491455,-1.173828,-0.275879,0.051666,-3.029297,-1.137695,-0.618652,0.987305,1.110352,0.347656,-0.821289,0.198120,-0.784668,
-1.444336,1.010742,-0.412109,-1.269531,0.528809,-2.267578,-0.967285,-1.823242,-2.824219,1.082031,-0.057709,-1.034180,-2.574219,-3.386719,0.478027,0.586426,-0.766113,0.354004,-0.006111,-2.222656,
-1.472656,-3.177734,-4.468750,-3.933594,-1.361328,-1.582031,-1.000000,-2.132812,-1.611328,-1.617188,-4.144531,-4.207031,-3.042969,-4.429688,-4.886719,-2.607422,-5.128906,-3.113281,-3.335938,-2.300781,
-3.216797,-3.419922,-2.562500,-2.414062,-2.171875,-2.515625,-3.033203,-2.935547,-2.972656,-4.332031,-2.886719,-1.382812,-2.361328,-2.056641,-2.199219,-1.861328,-2.648438,-2.033203,-3.138672,-2.773438,
-3.433594,-3.001953,-2.259766,-2.095703,-3.533203,-2.521484,-4.277344,-4.039062,-2.966797,-3.099609,-2.582031,-3.160156,-3.621094,-3.439453,-3.468750,-3.060547,-3.287109,-3.384766,-3.533203,-4.058594,
-3.759766,-4.265625,-3.775391,-3.371094,-3.880859,-3.150391,-3.488281,-2.814453,-2.646484,-3.080078,-4.410156,-6.308594,-6.828125,-3.308594,-3.628906,-1.998047,-1.861328,-2.712891,-3.314453,-3.580078,
-2.261719,-1.453125,-2.062500,-2.599609,-2.869141,-2.730469,-3.529297,-2.630859,-2.458984,-0.679688,-3.121094,-2.939453,-4.097656,-2.783203,-2.781250,-1.406250,-3.339844,-1.383789,-2.431641,-2.757812,
-3.070312,-3.332031,-4.558594,-7.027344,-6.351562,-3.263672,-3.529297,-1.828125,-3.556641,-3.718750,-4.316406,-1.859375,-3.789062,-4.070312,-3.080078,-3.626953,-3.072266,-1.865234,-2.658203,-2.466797,
-4.121094,-2.933594,-1.741211,-2.130859,-1.523438,-2.066406,-1.737305,-1.458984,-0.781250,-1.726562,-0.934570,-2.134766,-1.112305,-0.540527,-1.752930,-2.236328,-4.386719,-2.753906,-1.626953,-2.394531,
-2.078125,-2.779297,-2.017578,-1.793945,-1.269531,-4.039062,-1.796875,-5.078125,-4.742188,-6.062500,-1.979492,-4.066406,-3.162109,-3.314453,-2.779297,-1.919922,-2.537109,-2.408203,-3.734375,-2.242188,
-5.242188,-2.128906,-2.416016,-2.001953,-3.119141,-3.449219,-2.798828,-1.784180,-0.859863,-3.017578,-4.312500,-2.195312,-3.228516,-3.445312,-0.433838,-2.138672,-0.879395,-3.785156,-0.972656,-3.650391,
-4.582031,-3.619141,-1.084961,-1.457031,-3.187500,-2.763672,-2.958984,-1.052734,-1.707031,-1.594727,-0.987793,-3.126953,-3.435547,-2.550781,-3.248047,-3.984375,-3.173828,-2.093750,-3.289062,-3.330078,
-1.831055,-3.240234,-4.007812,-4.050781,-4.910156,-3.593750,-3.292969,-2.779297,-3.625000,-1.713867,-2.925781,-3.443359,-3.146484,-2.535156,-2.445312,-1.566406,-2.076172,-2.890625,-3.724609,-2.494141,
-1.585938,-2.671875,-2.900391,-3.267578,-2.386719,-1.122070,-0.569824,-3.912109,-2.968750,-3.021484,-1.368164,0.040649,-1.655273,-0.812012,-2.564453,-3.113281,-1.007812,-2.755859,-1.673828,-2.328125,
-3.580078,-2.468750,-3.322266,-1.148438,-1.352539,-3.839844,-5.722656,-5.980469,-1.079102,-3.722656,-3.392578,-5.351562,-1.301758,-3.802734,-2.673828,-2.259766,-4.503906,0.230225,-2.265625,-2.308594,
-4.726562,-0.563477,-2.626953,-0.833984,-0.518066,0.419434,1.599609,-2.095703,2.210938,1.682617,1.073242,0.854492,-2.884766,-0.444092,1.407227,2.449219,-0.347412,0.489014,-1.094727,-1.585938,
-1.586914,0.020050,-0.010025,-3.898438,0.148682,0.928223,-2.798828,-2.048828,-2.220703,1.231445,1.338867,0.894531,-1.369141,-0.379883,-1.901367,-0.599121,-1.504883,-3.738281,-0.497314,-0.550781,
-0.450928,0.097717,-0.186157,-0.660156,-2.912109,-0.511719,-2.716797,2.005859,-0.405762,-0.667480,-1.578125,0.422852,1.065430,2.140625,-0.194946,0.468994,-0.157593,0.731445,1.263672,0.671875,
0.500488,0.802246,-1.098633,-0.802734,1.177734,-1.245117,-0.645508,0.712402,-1.157227,-0.590820,0.151001,-0.304199,0.584473,-0.203369,-0.339600,-0.542969,-1.794922,-1.672852,-1.484375,-3.464844,
-1.531250,-2.388672,-2.126953,-2.890625,-1.107422,-2.529297,-2.542969,-3.644531,-1.730469,-2.361328,-2.919922,-2.824219,-2.154297,-1.605469,-3.496094,-2.613281,-1.834961,-2.472656,-1.921875,-3.275391,
-1.324219,-3.187500,-3.884766,-1.881836,-2.466797,-2.041016,-0.578613,-1.214844,-0.524902,-1.101562,-1.759766,-2.339844,-3.023438,-1.742188,-1.983398,-1.566406,-1.981445,-1.614258,-1.208984,-2.773438,
-2.941406,-3.630859,-2.029297,-3.095703,-3.666016,-3.880859,-2.386719,-2.390625,-3.548828,-2.234375,-2.435547,-2.183594,-2.929688,-2.486328,-1.112305,-2.808594,-2.789062,-3.710938,-1.678711,-2.537109,
-2.519531,-2.341797,-1.176758,-2.052734,-2.132812,-3.140625,-2.771484,-1.128906,0.017868,-0.903320,-0.992188,-0.070190,-0.989258,-0.131592,-0.652832,-0.919922,-2.667969,-2.447266,-3.162109,-1.691406,
-0.719727,-2.011719,-3.537109,-1.444336,-1.022461,-1.071289,-2.164062,-3.800781,-2.833984,-1.181641,-1.590820,-3.144531,-2.169922,-3.992188,-2.910156,-2.484375,-2.947266,-2.548828,-3.863281,-3.888672,
-3.134766,-3.074219,-2.611328,-2.580078,-2.054688,-2.804688,-2.154297,-2.187500,-2.199219,-2.441406,-1.662109,-1.771484,-1.419922,-2.162109,-2.050781,-1.315430,-2.039062,-1.619141,-1.947266,-1.454102,
-1.479492,-3.105469,-2.978516,-2.753906,-1.586914,-2.705078,-1.712891,-1.451172,-2.130859,-2.953125,-1.698242,-3.703125,-4.308594,-4.074219,-3.341797,-4.148438,-2.914062,-2.259766,-3.339844,-4.238281,
-4.164062,-3.050781,-3.792969,-4.308594,-3.714844,-3.962891,-3.634766,-3.876953,-3.640625,-3.218750,-2.705078,-2.794922,-2.701172,-4.453125,-2.777344,-3.550781,-3.105469,-3.271484,-3.044922,-2.324219,
-1.977539,-1.644531,-1.468750,-1.945312,-2.593750,-1.798828,-2.003906,-2.125000,-1.167969,-1.099609,-1.142578,-0.916992,0.009354,-1.195312,-0.240723,-1.831055,-1.743164,-1.478516,-2.488281,-3.156250,
-2.960938,-1.262695,-1.518555,-2.257812,-1.976562,-1.194336,-1.264648,-0.339600,-1.087891,-1.381836,-2.425781,-1.716797,-2.585938,-2.136719,-0.005161,-0.680664,-1.460938,-1.895508,-1.587891,-1.778320,
-1.263672,-1.008789,-1.244141,-0.834473,-0.084595,0.439453,0.416260,-1.077148,-0.226685,-0.253662,-0.305176,0.073792,0.810059,0.542969,-0.072327,-0.097717,0.204712,0.028152,0.537109,1.067383,
-0.515625,-1.880859,-1.012695,-2.525391,-0.951172,0.401123,-0.142944,-0.145264,-0.950195,-1.503906,-0.535645,-1.133789,-1.659180,-3.453125,-1.711914,-1.291992,-0.902344,-1.550781,-1.409180,-2.556641,
-1.911133,-2.169922,-1.643555,-3.351562,-2.662109,-0.892578,-1.775391,-2.525391,-1.950195,-0.185425,-2.070312,-1.986328,-2.781250,-1.022461,-1.760742,-1.590820,-1.324219,-0.612793,-1.772461,-0.418945,
-0.323730,-0.074219,1.028320,-0.744629,-0.310303,-0.444092,-2.667969,-2.298828,0.049011,-2.859375,-0.409180,-1.767578,-1.026367,-0.576660,-1.145508,-1.530273,-1.174805,-1.665039,-2.082031,-1.781250,
-2.007812,-2.759766,-1.950195,-2.414062,-2.871094,-1.732422,-3.128906,-1.938477,-1.106445,-2.246094,-2.613281,-3.458984,-2.123047,-1.146484,-0.724121,-1.304688,-0.965820,-2.238281,-1.667969,-1.404297,
-1.405273,-0.990723,-1.106445,-0.924316,-2.050781,-1.080078,0.149292,-0.521484,-1.681641,-3.005859,-1.306641,-1.754883,-1.252930,-0.344238,-1.271484,-2.630859,-2.425781,-2.480469,-1.400391,-1.572266,
-2.046875,-1.642578,-2.607422,-2.316406,-3.525391,-3.044922,-3.093750,-3.279297,-3.404297,-1.909180,-1.745117,-2.498047,-2.642578,-3.488281,-3.886719,-2.476562,-3.830078,-2.400391,-2.115234,-2.179688,
-2.210938,-1.960938,-3.554688,-3.242188,-1.999023,-0.752441,-1.756836,-0.992188,-1.362305,-1.194336,-1.092773,0.414062,0.063110,-0.130859,-0.571777,-1.789062,-1.125000,-1.119141,-0.063049,-1.373047,
-1.541992,0.308838,-0.473145,0.939453,-2.164062,0.043579,-0.320068,0.480713,0.763672,0.611816,0.724121,0.727051,-0.070496,-0.959473,-0.279785,-2.341797,-0.144775,-0.299072,-1.164062,-0.981445,
-0.151489,0.028793,-0.137817,-0.066284,0.536621,-0.296875,0.658203,-0.094849,0.489746,0.581055,0.328125,-0.341064,0.605469,1.182617,1.195312,1.875000,1.703125,1.196289,1.042969,0.466064,
1.558594,1.010742,0.763672,1.914062,0.934570,1.943359,2.656250,1.942383,0.152466,1.304688,0.832520,1.284180,1.576172,-0.759277,-0.347900,-1.421875,-0.104492,-1.594727,0.208984,0.185425,
-0.302734,-0.063904,-2.437500,-2.023438,-1.745117,-1.078125,-1.269531,0.035980,-0.386719,0.095032,-0.085754,-0.635742,-0.784180,-1.467773,-2.345703,-2.847656,-2.062500,-0.437744,0.116272,0.214600,
-1.534180,-1.547852,-0.763184,0.013771,-0.916504,-0.030823,-1.924805,-0.422607,-0.385986,-0.318604,-1.439453,-0.345703,0.427979,0.288330,-0.249268,0.072144,-0.695312,-0.636719,-0.730469,-1.223633,
-3.027344,-2.250000,-0.862793,-2.324219,-2.638672,-2.744141,-2.347656,-3.332031,-3.554688,-1.707031,-2.458984,-0.623047,-0.823730,-1.976562,-3.246094,-1.836914,-3.296875,-2.472656,-1.560547,-2.441406,
-1.483398,-1.162109,-1.326172,-0.518066,-0.605469,0.014977,0.392822,0.899414,0.231567,0.083496,-1.457031,-0.758301,-1.155273,-0.212036,0.778809,0.163208,0.823242,-0.120483,0.412354,0.575684,
0.016968,0.579102,-1.222656,1.654297,1.403320,0.346191,1.860352,1.060547,0.988281,0.777832,1.055664,-0.543457,0.143433,1.167969,1.481445,0.012421,-0.650391,-0.216675,0.610352,0.719238,
1.750977,-0.486328,0.240601,-0.123413,-0.069092,0.755371,0.986328,0.960938,1.217773,2.859375,2.503906,2.980469,1.458008,1.911133,1.375977,1.195312,1.416992,1.846680,1.312500,3.650391,
2.996094,2.214844,2.449219,3.392578,3.011719,0.936523,3.058594,4.507812,2.544922,2.773438,1.939453,3.052734,0.683105,1.352539,1.914062,2.562500,0.922363,1.209961,0.301025,1.070312,
0.399658,0.753418,1.151367,1.000977,1.250000,0.717773,1.120117,-2.443359,-0.986328,0.183716,1.121094,0.640137,0.133789,0.086121,-0.141479,-0.034363,0.040161,1.409180,0.589355,0.120850,
0.356934,1.069336,1.102539,1.730469,-0.763184,0.235596,-0.148682,1.950195,1.592773,2.091797,1.324219,0.898926,-0.839844,-0.137451,0.073242,1.505859,1.397461,0.743652,0.192749,-0.215820,
-0.033966,-1.451172,-0.288574,-0.304688,0.650391,1.406250,0.886230,0.668945,0.909668,-0.610352,-0.079163,-1.327148,-0.452881,-0.565430,-0.766602,-0.841797,1.095703,-1.058594,0.875977,0.615723,
-0.034149,0.329834,1.733398,-0.432129,0.213135,0.458252,1.449219,0.810547,2.339844,0.421875,1.248047,2.339844,1.662109,1.331055,1.228516,1.751953,0.723145,2.005859,2.029297,2.974609,
3.152344,3.187500,3.525391,1.895508,1.265625,0.914551,1.931641,2.193359,0.322510,0.249878,0.285400,2.486328,1.821289,1.297852,1.645508,0.365967,1.964844,1.243164,0.104248,-1.066406,
1.615234,0.360107,1.363281,2.216797,2.380859,1.150391,-0.260986,-0.506836,-0.645996,-0.142700,0.461426,0.592773,0.113403,0.719238,0.897461,1.685547,0.384521,0.605469,2.164062,1.254883,
3.158203,0.684570,0.675293,3.082031,3.660156,3.662109,3.560547,3.474609,3.357422,1.235352,2.519531,2.832031,2.671875,1.823242,2.396484,2.091797,3.751953,2.792969,2.455078,-0.995117,
1.182617,-0.181763,-1.815430,0.324951,1.363281,-0.861328,0.021515,0.578613,1.543945,1.996094,1.177734,0.458740,0.758301,0.265381,1.585938,0.870605,1.095703,1.692383,0.875977,1.338867,
2.630859,2.582031,2.310547,2.687500,3.164062,2.121094,1.725586,2.802734,2.804688,2.853516,2.275391,2.066406,2.675781,2.751953,2.593750,2.527344,0.559570,2.123047,2.296875,3.146484,
2.068359,1.223633,2.994141,2.029297,1.971680,2.134766,2.154297,0.166992,0.126953,1.381836,-0.332275,1.324219,2.416016,2.035156,2.013672,1.397461,1.227539,1.446289,-0.492188,0.043335,
-0.617676,0.946777,-0.709961,0.549316,1.361328,2.623047,2.777344,0.771484,1.689453,1.060547,2.339844,2.882812,2.539062,3.152344,2.935547,2.925781,3.197266,3.117188,2.498047,2.103516,
2.996094,2.929688,3.226562,1.461914,1.461914,3.134766,1.894531,1.047852,2.615234,2.685547,2.460938,2.197266,1.269531,-0.118408,2.089844,2.361328,1.710938,2.919922,2.466797,1.800781,
1.215820,1.675781,2.285156,0.262451,1.790039,2.457031,2.005859,1.738281,2.509766,2.808594,2.908203,0.688965,2.203125,1.074219,1.871094,2.771484,3.574219,3.406250,2.398438,3.158203,
2.750000,1.605469,3.951172,4.710938,2.173828,3.500000,4.183594,4.656250,4.507812,4.996094,5.468750,5.175781,4.273438,3.521484,4.890625,4.511719,4.332031,2.435547,4.289062,4.597656,
3.828125,4.968750,4.742188,3.458984,3.804688,3.210938,4.507812,3.970703,1.993164,3.677734,4.136719,1.976562,3.568359,4.167969,4.289062,4.367188,4.082031,4.070312,4.273438,3.898438,
4.105469,4.769531,2.724609,4.492188,5.902344,5.789062,4.851562,5.179688,4.542969,4.957031,4.250000,4.726562,6.351562,6.105469,5.406250,5.843750,5.105469,5.109375,6.058594,4.660156,
5.289062,2.996094,4.496094,5.496094,4.644531,4.777344,4.964844,5.070312,4.714844,5.218750,4.292969,4.679688,4.082031,3.019531,3.640625,2.792969,3.437500,3.671875,4.968750,3.058594,
4.082031,3.324219,4.269531,4.921875,4.613281,4.621094,4.851562,5.527344,5.507812,5.519531,5.660156,4.960938,5.277344,4.835938,5.847656,6.101562,5.789062,5.917969,5.898438,6.355469,
5.878906,7.390625,7.820312,7.308594,7.558594,4.441406,7.355469,6.875000,5.695312,6.390625,8.148438,8.015625,7.703125,6.117188,7.613281,7.394531,6.773438,4.191406,7.273438,7.292969,
5.328125,8.296875,8.406250,7.628906,5.914062,8.406250,8.414062,8.664062,8.234375,8.046875,9.093750,8.085938,9.406250,8.015625,9.218750,10.234375,9.921875,9.898438,10.515625,7.675781,
10.875000,9.593750,13.179688,10.875000,12.101562,12.875000,12.742188,10.687500};

std::vector<ov::float16> output_ref = {
0.000083,0.000075,0.000018,0.000032,
0.000104,0.000045,0.000029,0.000016,0.000067,0.000063,0.000005,0.000104,0.000012,0.000034,0.000040,0.000006,0.000088,0.000121,0.000177,0.000056,0.000065,0.000050,0.000035,0.000055,
0.000013,0.000068,0.000057,0.000042,0.000014,0.000024,0.000037,0.000030,0.000017,0.000031,0.000040,0.000013,0.000029,0.000087,0.000019,0.000019,0.000004,0.000019,0.000005,0.000018,
0.000051,0.000035,0.000034,0.000028,0.000008,0.000035,0.000052,0.000037,0.000012,0.000053,0.000062,0.000030,0.000052,0.000034,0.000032,0.000059,0.000114,0.000090,0.000059,0.000079,
0.000020,0.000018,0.000022,0.000034,0.000068,0.000088,0.000092,0.000075,0.000041,0.000050,0.000006,0.000150,0.000091,0.000061,0.000031,0.000014,0.000085,0.000097,0.000036,0.000061,
0.000016,0.000015,0.000056,0.000021,0.000023,0.000003,0.000013,0.000039,0.000029,0.000132,0.000042,0.000148,0.000063,0.000048,0.000042,0.000070,0.000007,0.000020,0.000020,0.000028,
0.000021,0.000036,0.000056,0.000026,0.000033,0.000022,0.000040,0.000273,0.000324,0.000012,0.000032,0.000130,0.000006,0.000108,0.000019,0.000185,0.000107,0.000004,0.000036,0.000062,
0.000013,0.000012,0.000056,0.000017,0.000077,0.000013,0.000052,0.000014,0.000017,0.000037,0.000054,0.000007,0.000036,0.000103,0.000073,0.000055,0.000049,0.000010,0.000014,0.000033,
0.000053,0.000087,0.000007,0.000051,0.000004,0.000120,0.000075,0.000183,0.000177,0.000015,0.000033,0.000024,0.000278,0.000045,0.000129,0.000192,0.000024,0.000153,0.000045,0.000055,
0.000050,0.000013,0.000030,0.000020,0.000089,0.000109,0.000108,0.000031,0.000076,0.000048,0.000008,0.000060,0.000025,0.000053,0.000027,0.000017,0.000019,0.000035,0.000027,0.000017,
0.000019,0.000042,0.000037,0.000032,0.000030,0.000019,0.000058,0.000003,0.000053,0.000089,0.000035,0.000157,0.000053,0.000040,0.000039,0.000010,0.000031,0.000046,0.000033,0.000037,
0.000042,0.000015,0.000062,0.000065,0.000025,0.000055,0.000017,0.000051,0.000203,0.000177,0.000099,0.000044,0.000048,0.000007,0.000044,0.000022,0.000044,0.000030,0.000005,0.000020,
0.000006,0.000007,0.000016,0.000008,0.000037,0.000028,0.000009,0.000024,0.000005,0.000041,0.000014,0.000043,0.000075,0.000140,0.000073,0.000062,0.000037,0.000029,0.000049,0.000108,
0.000177,0.000202,0.000106,0.000037,0.000019,0.000008,0.000122,0.000201,0.000223,0.000344,0.000139,0.000082,0.000136,0.000234,0.000300,0.000228,0.000310,0.000043,0.000176,0.000175,
0.000366,0.000086,0.000040,0.000111,0.000096,0.000119,0.000347,0.000049,0.000173,0.000120,0.000180,0.000115,0.000131,0.000016,0.000029,0.000009,0.000129,0.000091,0.000174,0.000155,
0.000102,0.000101,0.000018,0.000116,0.000037,0.000025,0.000130,0.000043,0.000058,0.000118,0.000157,0.000157,0.000174,0.000110,0.000033,0.000017,0.000217,0.000127,0.000132,0.000133,
0.000062,0.000230,0.000067,0.000055,0.000025,0.000009,0.000018,0.000020,0.000042,0.000047,0.000019,0.000037,0.000044,0.000017,0.000013,0.000043,0.000028,0.000016,0.000014,0.000008,
0.000004,0.000032,0.000015,0.000018,0.000016,0.000012,0.000036,0.000019,0.000030,0.000007,0.000009,0.000020,0.000013,0.000013,0.000048,0.000023,0.000012,0.000022,0.000013,0.000021,
0.000022,0.000060,0.000046,0.000018,0.000023,0.000014,0.000040,0.000026,0.000043,0.000013,0.000013,0.000024,0.000008,0.000005,0.000008,0.000006,0.000004,0.000016,0.000037,0.000081,
0.000056,0.000033,0.000012,0.000003,0.000004,0.000004,0.000008,0.000031,0.000013,0.000031,0.000008,0.000023,0.000031,0.000008,0.000004,0.000009,0.000007,0.000014,0.000049,0.000027,
0.000025,0.000007,0.000058,0.000057,0.000041,0.000032,0.000022,0.000016,0.000016,0.000030,0.000068,0.000028,0.000116,0.000092,0.000014,0.000007,0.000011,0.000012,0.000008,0.000014,
0.000032,0.000032,0.000028,0.000146,0.000075,0.000016,0.000020,0.000041,0.000042,0.000068,0.000059,0.000041,0.000027,0.000015,0.000024,0.000022,0.000049,0.000067,0.000025,0.000017,
0.000005,0.000026,0.000008,0.000044,0.000016,0.000070,0.000017,0.000052,0.000063,0.000043,0.000037,0.000015,0.000046,0.000055,0.000114,0.000091,0.000047,0.000031,0.000009,0.000021,
0.000006,0.000034,0.000044,0.000032,0.000019,0.000025,0.000033,0.000032,0.000035,0.000068,0.000044,0.000021,0.000018,0.000021,0.000011,0.000006,0.000009,0.000021,0.000008,0.000023,
0.000028,0.000017,0.000011,0.000007,0.000007,0.000016,0.000034,0.000022,0.000019,0.000018,0.000042,0.000040,0.000018,0.000020,0.000030,0.000025,0.000037,0.000030,0.000082,0.000025,
0.000040,0.000045,0.000013,0.000056,0.000111,0.000044,0.000044,0.000049,0.000021,0.000053,0.000059,0.000104,0.000163,0.000093,0.000083,0.000028,0.000015,0.000022,0.000042,0.000031,
0.000040,0.000017,0.000049,0.000036,0.000049,0.000035,0.000021,0.000019,0.000019,0.000056,0.000092,0.000080,0.000035,0.000018,0.000038,0.000031,0.000044,0.000056,0.000028,0.000035,
0.000042,0.000030,0.000021,0.000120,0.000096,0.000218,0.000165,0.000275,0.000235,0.000118,0.000083,0.000055,0.000006,0.000098,0.000239,0.000123,0.000163,0.000097,0.000103,0.000080,
0.000059,0.000049,0.000129,0.000054,0.000056,0.000005,0.000075,0.000048,0.000104,0.000049,0.000043,0.000072,0.000056,0.000061,0.000040,0.000032,0.000019,0.000036,0.000012,0.000075,
0.000107,0.000044,0.000084,0.000031,0.000022,0.000003,0.000150,0.000112,0.000094,0.000049,0.000121,0.000110,0.000019,0.000015,0.000021,0.000008,0.000097,0.000056,0.000075,0.000123,
0.000196,0.000125,0.000033,0.000059,0.000053,0.000073,0.000099,0.000015,0.000018,0.000004,0.000002,0.000185,0.000117,0.000144,0.000068,0.000051,0.000036,0.000031,0.000007,0.000060,
0.000031,0.000007,0.000031,0.000035,0.000046,0.000025,0.000020,0.000025,0.000005,0.000042,0.000023,0.000116,0.000044,0.000029,0.000018,0.000051,0.000045,0.000053,0.000035,0.000035,
0.000006,0.000014,0.000007,0.000142,0.000091,0.000113,0.000068,0.000159,0.000116,0.000048,0.000034,0.000052,0.000009,0.000034,0.000016,0.000389,0.000271,0.000108,0.000204,0.000075,
0.000026,0.000051,0.000175,0.000256,0.000108,0.000038,0.000132,0.000015,0.000043,0.000077,0.000126,0.000132,0.000116,0.000085,0.000056,0.000065,0.000061,0.000094,0.000050,0.000114,
0.000187,0.000259,0.000086,0.000043,0.000033,0.000004,0.000063,0.000040,0.000617,0.000791,0.000157,0.000123,0.000120,0.000165,0.000084,0.000103,0.000089,0.000010,0.000065,0.000042,
0.000130,0.000162,0.000553,0.000047,0.000164,0.000256,0.000295,0.000192,0.000065,0.000102,0.000026,0.000166,0.000143,0.000565,0.000553,0.000265,0.000052,0.000054,0.000072,0.000041,
0.000158,0.000064,0.000188,0.000458,0.000210,0.000460,0.000108,0.000623,0.000739,0.001080,0.000090,0.000544,0.000255,0.000259,0.000235,0.000251,0.000271,0.000826,0.000597,0.000515,
0.000199,0.000151,0.000028,0.000014,0.001109,0.000751,0.001285,0.001186,0.001344,0.001186,0.000480,0.000145,0.000106,0.000073,0.000116,0.000312,0.000398,0.000200,0.000051,0.000058,
0.000054,0.000021,0.000129,0.000102,0.000083,0.000310,0.000157,0.000255,0.000261,0.000420,0.000293,0.000197,0.000144,0.000068,0.000146,0.000066,0.000036,0.000050,0.000026,0.000112,
0.000114,0.000061,0.000784,0.000026,0.000054,0.000049,0.000027,0.000033,0.000081,0.000070,0.000238,0.000101,0.000087,0.000093,0.000071,0.000026,0.000022,0.000024,0.000013,0.000031,
0.000025,0.000012,0.000012,0.000009,0.000008,0.000026,0.000024,0.000010,0.000012,0.000007,0.000009,0.000021,0.000075,0.000063,0.000064,0.000163,0.000008,0.000013,0.000006,0.000018,
0.000010,0.000030,0.000030,0.000046,0.000096,0.000130,0.000123,0.000056,0.000031,0.000079,0.000026,0.000121,0.000063,0.000025,0.000030,0.000032,0.000073,0.000055,0.000150,0.000206,
0.000163,0.000083,0.000021,0.000069,0.000040,0.000104,0.000053,0.000111,0.000062,0.000052,0.000067,0.000052,0.000018,0.000017,0.000018,0.000044,0.000079,0.000124,0.000067,0.000082,
0.000020,0.000015,0.000010,0.000087,0.000173,0.000063,0.000097,0.000040,0.000155,0.000210,0.000210,0.000098,0.000146,0.000053,0.000218,0.000175,0.000018,0.000081,0.000072,0.000011,
0.000293,0.000190,0.000544,0.000166,0.000061,0.000040,0.000125,0.000171,0.000177,0.000122,0.000071,0.000167,0.000057,0.000068,0.000027,0.000059,0.000058,0.000023,0.000037,0.000020,
0.000054,0.000063,0.000077,0.000018,0.000026,0.000011,0.000032,0.000021,0.000022,0.000058,0.000017,0.000021,0.000052,0.000089,0.000093,0.000138,0.000027,0.000046,0.000010,0.000022,
0.000093,0.000061,0.000028,0.000077,0.000131,0.000144,0.000126,0.000068,0.000013,0.000048,0.000016,0.000046,0.000089,0.000096,0.000043,0.000013,0.000012,0.000013,0.000029,0.000040,
0.000043,0.000033,0.000044,0.000009,0.000023,0.000019,0.000025,0.000063,0.000086,0.000039,0.000054,0.000034,0.000018,0.000007,0.000011,0.000020,0.000025,0.000095,0.000057,0.000033,
0.000035,0.000021,0.000027,0.000015,0.000028,0.000014,0.000013,0.000031,0.000016,0.000013,0.000049,0.000192,0.000108,0.000075,0.000092,0.000033,0.000032,0.000017,0.000017,0.000003,
0.000022,0.000035,0.000047,0.000262,0.000092,0.000015,0.000030,0.000050,0.000012,0.000071,0.000049,0.000008,0.000042,0.000009,0.000015,0.000035,0.000034,0.000006,0.000008,0.000005,
0.000006,0.000009,0.000006,0.000020,0.000013,0.000009,0.000017,0.000084,0.000087,0.000063,0.000003,0.000005,0.000002,0.000015,0.000007,0.000054,0.000019,0.000012,0.000121,0.000075,
0.000005,0.000024,0.000006,0.000098,0.000144,0.000183,0.000038,0.000097,0.000025,0.000029,0.000048,0.000047,0.000015,0.000031,0.000032,0.000010,0.000020,0.000009,0.000013,0.000146,
0.000005,0.000002,0.000001,0.000036,0.000025,0.000038,0.000059,0.000015,0.000022,0.000063,0.000040,0.000080,0.000008,0.000021,0.000010,0.000025,0.000002,0.000014,0.000065,0.000025,
0.000010,0.000019,0.000032,0.000035,0.000012,0.000084,0.000028,0.000069,0.000022,0.000075,0.000099,0.000065,0.000055,0.000009,0.000020,0.000040,0.000063,0.000037,0.000051,0.000005,
0.000040,0.000052,0.000018,0.000116,0.000082,0.000022,0.000001,0.000004,0.000020,0.000020,0.000025,0.000021,0.000016,0.000019,0.000016,0.000013,0.000020,0.000008,0.000020,0.000010,
0.000005,0.000021,0.000037,0.000018,0.000014,0.000039,0.000023,0.000063,0.000071,0.000025,0.000006,0.000005,0.000033,0.000025,0.000016,0.000058,0.000056,0.000013,0.000001,0.000008,
0.000041,0.000032,0.000055,0.000049,0.000049,0.000027,0.000073,0.000030,0.000037,0.000007,0.000003,0.000013,0.000007,0.000018,0.000017,0.000008,0.000004,0.000012,0.000011,0.000011,
0.000018,0.000011,0.000019,0.000021,0.000014,0.000013,0.000009,0.000032,0.000029,0.000035,0.000037,0.000025,0.000014,0.000030,0.000012,0.000030,0.000003,0.000097,0.000014,0.000123,
0.000116,0.000033,0.000003,0.000009,0.000006,0.000094,0.000092,0.000046,0.000013,0.000023,0.000038,0.000103,0.000094,0.000039,0.000045,0.000015,0.000006,0.000032,0.000021,0.000004,
0.000033,0.000033,0.000067,0.000093,0.000059,0.000046,0.000021,0.000021,0.000008,0.000039,0.000058,0.000035,0.000024,0.000019,0.000024,0.000009,0.000035,0.000027,0.000009,0.000014,
0.000003,0.000065,0.000100,0.000104,0.000039,0.000048,0.000059,0.000057,0.000057,0.000063,0.000050,0.000015,0.000010,0.000038,0.000079,0.000037,0.000075,0.000042,0.000035,0.000032,
0.000018,0.000002,0.000080,0.000039,0.000004,0.000032,0.000029,0.000023,0.000041,0.000007,0.000003,0.000002,0.000006,0.000009,0.000015,0.000013,0.000015,0.000008,0.000004,0.000017,
0.000018,0.000020,0.000018,0.000015,0.000008,0.000010,0.000015,0.000011,0.000023,0.000018,0.000025,0.000032,0.000018,0.000009,0.000027,0.000035,0.000009,0.000006,0.000025,0.000028,
0.000046,0.000052,0.000025,0.000051,0.000057,0.000049,0.000049,0.000013,0.000007,0.000006,0.000019,0.000010,0.000005,0.000014,0.000007,0.000006,0.000009,0.000004,0.000002,0.000033,
0.000009,0.000032,0.000042,0.000077,0.000102,0.000066,0.000040,0.000014,0.000006,0.000008,0.000013,0.000042,0.000029,0.000012,0.000012,0.000007,0.000033,0.000030,0.000024,0.000032,
0.000039,0.000009,0.000066,0.000030,0.000041,0.000344,0.000209,0.000023,0.000052,0.000026,0.000072,0.000092,0.000059,0.000029,0.000099,0.000163,0.000280,0.000116,0.000108,0.000095,
0.000069,0.000050,0.000006,0.000091,0.000025,0.000040,0.000108,0.000084,0.000049,0.000020,0.000021,0.000026,0.000052,0.000037,0.000010,0.000050,0.000039,0.000043,0.000010,0.000071,
0.000073,0.000036,0.000098,0.000091,0.000015,0.000034,0.000037,0.000012,0.000024,0.000022,0.000031,0.000064,0.000055,0.000025,0.000011,0.000006,0.000023,0.000005,0.000001,0.000077,
0.000086,0.000025,0.000019,0.000004,0.000005,0.000035,0.000026,0.000007,0.000012,0.000002,0.000007,0.000007,0.000004,0.000002,0.000002,0.000001,0.000002,0.000008,0.000004,0.000005,
0.000001,0.000001,0.000011,0.000005,0.000025,0.000008,0.000002,0.000001,0.000000,0.000001,0.000007,0.000008,0.000013,0.000010,0.000005,0.000012,0.000007,0.000002,0.000008,0.000003,
0.000008,0.000003,0.000005,0.000002,0.000002,0.000002,0.000001,0.000004,0.000009,0.000004,0.000004,0.000003,0.000000,0.000002,0.000001,0.000003,0.000003,0.000007,0.000005,0.000002,
0.000002,0.000002,0.000004,0.000002,0.000001,0.000001,0.000001,0.000006,0.000005,0.000002,0.000001,0.000003,0.000001,0.000002,0.000000,0.000002,0.000006,0.000008,0.000000,0.000007,
0.000009,0.000003,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001,0.000002,0.000004,0.000008,0.000002,0.000001,0.000003,0.000004,0.000001,0.000000,0.000002,
0.000006,0.000003,0.000004,0.000002,0.000000,0.000001,0.000000,0.000002,0.000004,0.000002,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000001,
0.000000,0.000001,0.000002,0.000001,0.000002,0.000004,0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,0.000002,0.000006,0.000005,0.000002,0.000001,0.000001,0.000001,0.000003,
0.000007,0.000006,0.000003,0.000010,0.000007,0.000001,0.000005,0.000002,0.000001,0.000007,0.000004,0.000002,0.000003,0.000002,0.000001,0.000000,0.000007,0.000004,0.000002,0.000003,
0.000000,0.000002,0.000003,0.000001,0.000002,0.000003,0.000003,0.000002,0.000004,0.000003,0.000003,0.000001,0.000002,0.000002,0.000001,0.000006,0.000004,0.000003,0.000000,0.000004,
0.000004,0.000005,0.000020,0.000006,0.000005,0.000001,0.000003,0.000000,0.000002,0.000003,0.000003,0.000002,0.000001,0.000002,0.000003,0.000002,0.000001,0.000000,0.000001,0.000002,
0.000002,0.000003,0.000001,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000002,0.000001,0.000000,0.000000,0.000001,0.000002,0.000001,0.000003,
0.000003,0.000002,0.000000,0.000002,0.000003,0.000008,0.000003,0.000002,0.000003,0.000002,0.000000,0.000004,0.000003,0.000001,0.000000,0.000001,0.000001,0.000003,0.000001,0.000000,
0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000004,0.000003,0.000000,0.000002,0.000002,0.000001,0.000003,0.000001,0.000001,0.000000,
0.000000,0.000003,0.000007,0.000003,0.000001,0.000010,0.000003,0.000006,0.000001,0.000003,0.000009,0.000013,0.000007,0.000012,0.000000,0.000011,0.000016,0.000002,0.000001,0.000003,
0.000002,0.000002,0.000007,0.000003,0.000000,0.000001,0.000000,0.000002,0.000002,0.000008,0.000008,0.000002,0.000001,0.000001,0.000003,0.000002,0.000001,0.000001,0.000001,0.000000,
0.000002,0.000003,0.000010,0.000005,0.000004,0.000004,0.000004,0.000001,0.000007,0.000002,0.000003,0.000002,0.000004,0.000001,0.000002,0.000010,0.000002,0.000012,0.000011,0.000001,
0.000014,0.000013,0.000007,0.000011,0.000028,0.000002,0.000035,0.000012,0.000008,0.000007,0.000005,0.000001,0.000002,0.000003,0.000012,0.000001,0.000001,0.000004,0.000002,0.000001,
0.000001,0.000010,0.000002,0.000002,0.000014,0.000005,0.000004,0.000005,0.000013,0.000006,0.000002,0.000004,0.000010,0.000001,0.000009,0.000001,0.000002,0.000012,0.000033,0.000011,
0.000042,0.000035,0.000002,0.000022,0.000003,0.000022,0.000004,0.000003,0.000004,0.000005,0.000020,0.000007,0.000003,0.000006,0.000004,0.000001,0.000006,0.000008,0.000013,0.000027,
0.000004,0.000008,0.000015,0.000032,0.000040,0.000024,0.000008,0.000002,0.000005,0.000005,0.000005,0.000006,0.000017,0.000015,0.000007,0.000002,0.000005,0.000032,0.000028,0.000008,
0.000033,0.000086,0.000022,0.000029,0.000017,0.000037,0.000031,0.000023,0.000013,0.000027,0.000007,0.000007,0.000024,0.000007,0.000010,0.000053,0.000009,0.000006,0.000003,0.000002,
0.000050,0.000015,0.000006,0.000000,0.000000,0.000036,0.000075,0.000017,0.000021,0.000004,0.000000,0.000002,0.000001,0.000001,0.000004,0.000019,0.000009,0.000014,0.000002,0.000006,
0.000012,0.000001,0.000001,0.000002,0.000001,0.000002,0.000005,0.000002,0.000002,0.000002,0.000007,0.000006,0.000004,0.000005,0.000000,0.000005,0.000004,0.000004,0.000008,0.000014,
0.000004,0.000004,0.000010,0.000003,0.000005,0.000004,0.000009,0.000001,0.000005,0.000003,0.000002,0.000001,0.000001,0.000002,0.000004,0.000001,0.000005,0.000002,0.000002,0.000001,
0.000001,0.000004,0.000002,0.000003,0.000004,0.000003,0.000001,0.000001,0.000001,0.000002,0.000002,0.000004,0.000003,0.000004,0.000008,0.000004,0.000007,0.000002,0.000002,0.000005,
0.000005,0.000001,0.000001,0.000002,0.000002,0.000001,0.000001,0.000003,0.000002,0.000003,0.000002,0.000001,0.000003,0.000001,0.000002,0.000002,0.000002,0.000001,0.000001,0.000003,
0.000013,0.000001,0.000003,0.000000,0.000001,0.000002,0.000011,0.000002,0.000009,0.000001,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000002,0.000001,0.000003,0.000001,
0.000001,0.000001,0.000002,0.000000,0.000000,0.000001,0.000002,0.000001,0.000002,0.000001,0.000002,0.000001,0.000003,0.000007,0.000002,0.000002,0.000001,0.000002,0.000014,0.000036,
0.000013,0.000010,0.000003,0.000001,0.000004,0.000002,0.000002,0.000000,0.000001,0.000005,0.000002,0.000003,0.000001,0.000001,0.000002,0.000008,0.000001,0.000012,0.000001,0.000001,
0.000000,0.000002,0.000001,0.000002,0.000003,0.000004,0.000007,0.000003,0.000001,0.000000,0.000008,0.000001,0.000013,0.000007,0.000013,0.000003,0.000004,0.000003,0.000008,0.000013,
0.000001,0.000001,0.000011,0.000004,0.000004,0.000010,0.000001,0.000007,0.000000,0.000013,0.000001,0.000001,0.000001,0.000008,0.000009,0.000003,0.000003,0.000003,0.000012,0.000004,
0.000005,0.000012,0.000001,0.000001,0.000001,0.000001,0.000001,0.000005,0.000010,0.000001,0.000001,0.000002,0.000001,0.000001,0.000000,0.000002,0.000005,0.000003,0.000003,0.000002,
0.000011,0.000003,0.000002,0.000002,0.000002,0.000002,0.000005,0.000002,0.000001,0.000000,0.000001,0.000005,0.000003,0.000003,0.000001,0.000001,0.000003,0.000010,0.000002,0.000006,
0.000009,0.000009,0.000017,0.000001,0.000003,0.000000,0.000001,0.000009,0.000002,0.000007,0.000007,0.000004,0.000006,0.000002,0.000010,0.000005,0.000001,0.000000,0.000000,0.000022,
0.000001,0.000001,0.000000,0.000007,0.000003,0.000005,0.000004,0.000001,0.000013,0.000001,0.000001,0.000001,0.000017,0.000004,0.000008,0.000009,0.000031,0.000078,0.000005,0.000063,
0.000039,0.000024,0.000049,0.000007,0.000050,0.000054,0.000024,0.000000,0.000005,0.000002,0.000002,0.000003,0.000007,0.000004,0.000000,0.000011,0.000035,0.000003,0.000002,0.000002,
0.000051,0.000025,0.000021,0.000005,0.000002,0.000000,0.000006,0.000005,0.000009,0.000013,0.000003,0.000002,0.000002,0.000005,0.000009,0.000003,0.000018,0.000004,0.000029,0.000002,
0.000002,0.000003,0.000020,0.000024,0.000072,0.000017,0.000013,0.000008,0.000020,0.000020,0.000006,0.000007,0.000020,0.000003,0.000012,0.000025,0.000003,0.000003,0.000032,0.000010,
0.000039,0.000041,0.000013,0.000013,0.000013,0.000008,0.000013,0.000004,0.000005,0.000007,0.000002,0.000021,0.000008,0.000011,0.000001,0.000013,0.000002,0.000004,0.000004,0.000021,
0.000005,0.000002,0.000001,0.000005,0.000019,0.000004,0.000008,0.000009,0.000002,0.000006,0.000000,0.000015,0.000003,0.000003,0.000011,0.000003,0.000008,0.000034,0.000008,0.000011,
0.000005,0.000005,0.000005,0.000007,0.000016,0.000006,0.000011,0.000001,0.000005,0.000010,0.000002,0.000002,0.000001,0.000003,0.000001,0.000001,0.000002,0.000013,0.000018,0.000003,
0.000012,0.000006,0.000010,0.000002,0.000003,0.000010,0.000001,0.000003,0.000005,0.000017,0.000005,0.000002,0.000001,0.000006,0.000007,0.000019,0.000017,0.000012,0.000021,0.000029,
0.000011,0.000009,0.000034,0.000008,0.000019,0.000006,0.000011,0.000005,0.000003,0.000003,0.000003,0.000009,0.000004,0.000002,0.000025,0.000021,0.000007,0.000001,0.000000,0.000003,
0.000028,0.000022,0.000004,0.000004,0.000001,0.000001,0.000001,0.000001,0.000003,0.000001,0.000001,0.000006,0.000007,0.000011,0.000006,0.000007,0.000000,0.000006,0.000014,0.000018,
0.000023,0.000013,0.000006,0.000002,0.000006,0.000013,0.000028,0.000004,0.000004,0.000002,0.000002,0.000018,0.000004,0.000005,0.000007,0.000022,0.000006,0.000012,0.000017,0.000004,
0.000001,0.000001,0.000001,0.000001,0.000004,0.000006,0.000001,0.000000,0.000002,0.000001,0.000003,0.000003,0.000003,0.000005,0.000001,0.000002,0.000005,0.000011,0.000006,0.000005,
0.000005,0.000005,0.000003,0.000006,0.000001,0.000008,0.000001,0.000004,0.000007,0.000013,0.000008,0.000005,0.000006,0.000004,0.000008,0.000006,0.000032,0.000024,0.000028,0.000010,
0.000020,0.000016,0.000020,0.000027,0.000012,0.000005,0.000002,0.000001,0.000012,0.000002,0.000001,0.000002,0.000006,0.000011,0.000028,0.000018,0.000005,0.000002,0.000007,0.000004,
0.000007,0.000013,0.000010,0.000003,0.000002,0.000008,0.000019,0.000007,0.000011,0.000015,0.000014,0.000027,0.000020,0.000008,0.000013,0.000015,0.000007,0.000009,0.000002,0.000024,
0.000024,0.000010,0.000012,0.000007,0.000016,0.000027,0.000026,0.000055,0.000004,0.000011,0.000004,0.000003,0.000000,0.000009,0.000003,0.000010,0.000042,0.000034,0.000025,0.000009,
0.000002,0.000003,0.000002,0.000003,0.000003,0.000022,0.000009,0.000007,0.000001,0.000003,0.000003,0.000013,0.000015,0.000015,0.000002,0.000006,0.000029,0.000011,0.000005,0.000004,
0.000010,0.000003,0.000002,0.000003,0.000006,0.000002,0.000002,0.000015,0.000034,0.000015,0.000077,0.000042,0.000013,0.000027,0.000002,0.000022,0.000034,0.000006,0.000011,0.000028,
0.000002,0.000013,0.000005,0.000008,0.000013,0.000006,0.000002,0.000011,0.000006,0.000007,0.000008,0.000003,0.000002,0.000008,0.000005,0.000005,0.000015,0.000001,0.000004,0.000008,
0.000003,0.000005,0.000005,0.000008,0.000009,0.000002,0.000002,0.000019,0.000000,0.000006,0.000018,0.000019,0.000030,0.000030,0.000031,0.000004,0.000005,0.000025,0.000008,0.000003,
0.000003,0.000006,0.000002,0.000001,0.000009,0.000010,0.000003,0.000004,0.000005,0.000008,0.000002,0.000001,0.000007,0.000003,0.000006,0.000001,0.000003,0.000002,0.000001,0.000002,
0.000005,0.000008,0.000001,0.000005,0.000002,0.000003,0.000015,0.000002,0.000001,0.000002,0.000004,0.000011,0.000016,0.000004,0.000002,0.000004,0.000018,0.000007,0.000029,0.000018,
0.000013,0.000012,0.000013,0.000023,0.000016,0.000004,0.000000,0.000002,0.000003,0.000036,0.000016,0.000009,0.000028,0.000003,0.000023,0.000001,0.000024,0.000029,0.000075,0.000084,
0.000012,0.000013,0.000022,0.000012,0.000006,0.000010,0.000001,0.000013,0.000016,0.000009,0.000018,0.000026,0.000031,0.000018,0.000017,0.000062,0.000024,0.000020,0.000005,0.000004,
0.000012,0.000067,0.000029,0.000102,0.000094,0.000050,0.000022,0.000006,0.000014,0.000017,0.000012,0.000088,0.000031,0.000048,0.000121,0.000032,0.000028,0.000063,0.000016,0.000005,
0.000060,0.000041,0.000026,0.000010,0.000001,0.000001,0.000003,0.000022,0.000007,0.000028,0.000006,0.000002,0.000009,0.000001,0.000004,0.000006,0.000018,0.000009,0.000033,0.000025,
0.000010,0.000004,0.000007,0.000008,0.000010,0.000006,0.000006,0.000012,0.000006,0.000005,0.000003,0.000003,0.000002,0.000010,0.000042,0.000001,0.000004,0.000001,0.000008,0.000012,
0.000009,0.000003,0.000005,0.000009,0.000018,0.000002,0.000004,0.000006,0.000017,0.000006,0.000010,0.000001,0.000003,0.000002,0.000001,0.000001,0.000001,0.000005,0.000004,0.000003,
0.000006,0.000001,0.000002,0.000004,0.000001,0.000001,0.000001,0.000001,0.000001,0.000026,0.000007,0.000027,0.000012,0.000017,0.000009,0.000036,0.000125,0.000062,0.000053,0.000012,
0.000009,0.000001,0.000014,0.000042,0.000026,0.000046,0.000002,0.000006,0.000002,0.000044,0.000045,0.000012,0.000030,0.000006,0.000200,0.000106,0.000019,0.000026,0.000006,0.000014,
0.000030,0.000084,0.000017,0.000013,0.000012,0.000079,0.000010,0.000010,0.000029,0.000034,0.000011,0.000048,0.000001,0.000009,0.000025,0.000031,0.000047,0.000048,0.000028,0.000037,
0.000141,0.000151,0.000047,0.000010,0.000019,0.000021,0.000044,0.000060,0.000044,0.000020,0.000302,0.000154,0.000042,0.000106,0.000478,0.000218,0.000019,0.000058,0.000627,0.000057,
0.000148,0.000028,0.000102,0.000003,0.000013,0.000052,0.000132,0.000023,0.000026,0.000015,0.000032,0.000072,0.000076,0.000025,0.000008,0.000002,0.000010,0.000071,0.000007,0.000035,
0.000047,0.000120,0.000025,0.000042,0.000031,0.000011,0.000049,0.000017,0.000252,0.000103,0.000012,0.000049,0.000108,0.000122,0.000275,0.000039,0.000055,0.000093,0.000119,0.000031,
0.000007,0.000019,0.000052,0.000021,0.000050,0.000015,0.000048,0.000031,0.000013,0.000006,0.000021,0.000046,0.000006,0.000028,0.000025,0.000110,0.000234,0.000072,0.000007,0.000002,
0.000005,0.000012,0.000015,0.000018,0.000009,0.000002,0.000003,0.000023,0.000035,0.000102,0.000059,0.000014,0.000006,0.000009,0.000020,0.000022,0.000120,0.000102,0.000206,0.000381,
0.000042,0.000039,0.000121,0.000032,0.000016,0.000041,0.000179,0.000100,0.000473,0.000044,0.000190,0.000180,0.000235,0.000896,0.000302,0.000102,0.000031,0.000052,0.000084,0.000013,
0.000034,0.000015,0.000255,0.000056,0.000038,0.000103,0.000021,0.000138,0.000069,0.000009,0.000029,0.000393,0.000313,0.000389,0.000300,0.000080,0.000018,0.000013,0.000006,0.000048,
0.000046,0.000034,0.000029,0.000024,0.000029,0.000075,0.000190,0.000033,0.000063,0.000366,0.000130,0.001116,0.000020,0.000033,0.000265,0.000610,0.001068,0.000977,0.000719,0.000148,
0.000025,0.000069,0.000225,0.000404,0.000265,0.000373,0.000086,0.000298,0.000098,0.000099,0.000008,0.000086,0.000036,0.000013,0.000083,0.000150,0.000029,0.000027,0.000010,0.000050,
0.000185,0.000068,0.000104,0.000089,0.000020,0.000133,0.000067,0.000051,0.000379,0.000081,0.000333,0.000340,0.000127,0.000108,0.000192,0.000837,0.000277,0.000221,0.000349,0.000351,
0.000185,0.000071,0.000036,0.000123,0.000111,0.000102,0.000340,0.000120,0.000541,0.000290,0.000371,0.000019,0.000023,0.000056,0.000190,0.000201,0.000131,0.000170,0.000021,0.000025,
0.000087,0.000031,0.000144,0.000061,0.000090,0.000063,0.000099,0.000026,0.000111,0.000017,0.000006,0.000053,0.000265,0.000106,0.000048,0.000071,0.000062,0.000189,0.000023,0.000189,
0.000242,0.000210,0.000312,0.000018,0.000098,0.000478,0.000055,0.000173,0.000359,0.000137,0.000192,0.000786,0.000196,0.000448,0.000051,0.000050,0.000200,0.000041,0.000055,0.000044,
0.000113,0.000024,0.000093,0.000048,0.000082,0.000590,0.000434,0.000045,0.000338,0.000208,0.000141,0.000040,0.000113,0.000411,0.000029,0.000238,0.000600,0.000206,0.000123,0.000095,
0.000160,0.000499,0.000086,0.000400,0.000088,0.000035,0.000111,0.000665,0.000597,0.000483,0.001155,0.000349,0.000087,0.000286,0.001164,0.000082,0.000607,0.001397,0.001210,0.000719,
0.001458,0.002125,0.000977,0.000246,0.000022,0.000391,0.001311,0.001149,0.000791,0.001917,0.000920,0.000357,0.001025,0.000956,0.000415,0.000786,0.000391,0.000486,0.000236,0.000102,
0.000273,0.000844,0.000088,0.000333,0.001116,0.001077,0.000507,0.000676,0.001068,0.000366,0.000185,0.000530,0.003216,0.000406,0.000982,0.000972,0.000731,0.000045,0.000322,0.000627,
0.002373,0.001210,0.001523,0.002073,0.002405,0.000706,0.002003,0.000118,0.000338,0.001948,0.000521,0.002068,0.000224,0.000770,0.001319,0.000535,0.000770,0.001404,0.000670,0.000844,
0.000357,0.000059,0.000530,0.000144,0.000060,0.000943,0.000448,0.000587,0.000217,0.001092,0.000128,0.000765,0.000149,0.000610,0.001458,0.001675,0.001408,0.001603,0.001647,0.000488,
0.000544,0.001415,0.001040,0.004025,0.002756,0.002932,0.003103,0.001635,0.002653,0.000943,0.003983,0.003311,0.005512,0.001692,0.002800,0.003164,0.000627,0.004932,0.003801,0.001220,
0.003824,0.014664,0.008156,0.004425,0.000443,0.002413,0.003164,0.006641,0.000711,0.011650,0.006329,0.001370,0.003515,0.009468,0.008141,0.003086,0.018463,0.006588,0.007183,0.001443,
0.003611,0.016418,0.008224,0.023422,0.005390,0.017487,0.052002,0.024170,0.036865,0.048065,0.002125,0.067322,0.011032,0.163940,0.059387,0.065186,0.000000,0.000000,0.000000,0.000296,
0.000278,0.000034,0.000065,0.000267,0.000116,0.000201,0.000147,0.000236,0.000223,0.000028,0.000265,0.000056,0.000119,0.000231,0.000033,0.000235,0.000308,0.000341,0.000128,0.000400,
0.000218,0.000144,0.000213,0.000040,0.000167,0.000168,0.000119,0.000057,0.000148,0.000155,0.000128,0.000068,0.000125,0.000107,0.000035,0.000058,0.000224,0.000077,0.000122,0.000061,
0.000106,0.000019,0.000068,0.000173,0.000173,0.000164,0.000102,0.000034,0.000101,0.000151,0.000130,0.000042,0.000179,0.000246,0.000142,0.000196,0.000156,0.000153,0.000173,0.000249,
0.000207,0.000184,0.000341,0.000152,0.000088,0.000080,0.000077,0.000133,0.000192,0.000200,0.000215,0.000167,0.000201,0.000028,0.000371,0.000266,0.000279,0.000137,0.000037,0.000215,
0.000305,0.000179,0.000192,0.000116,0.000090,0.000170,0.000117,0.000137,0.000044,0.000059,0.000151,0.000094,0.000265,0.000069,0.000421,0.000258,0.000226,0.000148,0.000293,0.000040,
0.000142,0.000065,0.000072,0.000051,0.000099,0.000296,0.000100,0.000209,0.000067,0.000176,0.000779,0.000669,0.000050,0.000232,0.000477,0.000049,0.000353,0.000074,0.000513,0.000491,
0.000042,0.000187,0.000225,0.000033,0.000024,0.000166,0.000060,0.000204,0.000053,0.000221,0.000111,0.000116,0.000153,0.000181,0.000033,0.000056,0.000284,0.000252,0.000283,0.000269,
0.000078,0.000039,0.000159,0.000196,0.000304,0.000068,0.000232,0.000023,0.000373,0.000207,0.000424,0.000515,0.000064,0.000106,0.000061,0.000793,0.000190,0.000491,0.000509,0.000071,
0.000301,0.000104,0.000229,0.000360,0.000148,0.000136,0.000072,0.000178,0.000190,0.000298,0.000087,0.000248,0.000178,0.000045,0.000177,0.000142,0.000205,0.000121,0.000097,0.000096,
0.000130,0.000098,0.000064,0.000116,0.000124,0.000118,0.000141,0.000166,0.000106,0.000262,0.000024,0.000220,0.000244,0.000055,0.000436,0.000277,0.000189,0.000192,0.000069,0.000125,
0.000135,0.000113,0.000141,0.000138,0.000074,0.000189,0.000244,0.000117,0.000178,0.000104,0.000113,0.000321,0.000377,0.000305,0.000234,0.000180,0.000037,0.000124,0.000065,0.000216,
0.000182,0.000058,0.000103,0.000022,0.000017,0.000037,0.000017,0.000096,0.000071,0.000028,0.000086,0.000028,0.000192,0.000062,0.000116,0.000188,0.000281,0.000207,0.000223,0.000126,
0.000128,0.000154,0.000231,0.000283,0.000369,0.000323,0.000083,0.000055,0.000026,0.000223,0.000345,0.000424,0.000607,0.000380,0.000223,0.000320,0.000487,0.000632,0.000421,0.000441,
0.000054,0.000380,0.000410,0.000849,0.000294,0.000185,0.000259,0.000184,0.000135,0.000486,0.000067,0.000457,0.000310,0.000474,0.000249,0.000279,0.000089,0.000088,0.000047,0.000286,
0.000236,0.000374,0.000333,0.000274,0.000227,0.000054,0.000156,0.000066,0.000085,0.000417,0.000138,0.000093,0.000119,0.000218,0.000212,0.000346,0.000453,0.000145,0.000099,0.000584,
0.000315,0.000335,0.000265,0.000123,0.000442,0.000083,0.000151,0.000084,0.000057,0.000041,0.000078,0.000069,0.000059,0.000053,0.000105,0.000198,0.000053,0.000028,0.000063,0.000079,
0.000098,0.000140,0.000072,0.000035,0.000110,0.000054,0.000044,0.000041,0.000038,0.000112,0.000062,0.000086,0.000029,0.000053,0.000108,0.000040,0.000031,0.000075,0.000084,0.000087,
0.000170,0.000077,0.000090,0.000054,0.000170,0.000152,0.000062,0.000073,0.000037,0.000108,0.000061,0.000100,0.000069,0.000077,0.000131,0.000038,0.000019,0.000028,0.000028,0.000013,
0.000035,0.000076,0.000130,0.000119,0.000164,0.000080,0.000032,0.000014,0.000011,0.000021,0.000080,0.000051,0.000082,0.000037,0.000085,0.000130,0.000036,0.000015,0.000031,0.000031,
0.000050,0.000156,0.000128,0.000177,0.000063,0.000195,0.000162,0.000133,0.000092,0.000093,0.000062,0.000079,0.000058,0.000060,0.000059,0.000195,0.000366,0.000081,0.000027,0.000040,
0.000045,0.000030,0.000048,0.000102,0.000059,0.000051,0.000442,0.000303,0.000114,0.000128,0.000166,0.000146,0.000228,0.000231,0.000226,0.000199,0.000098,0.000118,0.000063,0.000164,
0.000207,0.000146,0.000107,0.000050,0.000133,0.000052,0.000170,0.000080,0.000316,0.000106,0.000253,0.000301,0.000283,0.000333,0.000135,0.000196,0.000240,0.000344,0.000338,0.000244,
0.000305,0.000070,0.000108,0.000049,0.000133,0.000194,0.000191,0.000116,0.000144,0.000152,0.000080,0.000151,0.000284,0.000251,0.000128,0.000161,0.000163,0.000083,0.000054,0.000070,
0.000116,0.000037,0.000095,0.000130,0.000128,0.000097,0.000101,0.000049,0.000061,0.000111,0.000084,0.000094,0.000085,0.000264,0.000248,0.000115,0.000125,0.000095,0.000150,0.000208,
0.000138,0.000229,0.000075,0.000186,0.000200,0.000056,0.000254,0.000344,0.000193,0.000215,0.000299,0.000156,0.000219,0.000152,0.000259,0.000213,0.000313,0.000366,0.000156,0.000132,
0.000119,0.000199,0.000128,0.000189,0.000081,0.000192,0.000119,0.000178,0.000208,0.000125,0.000103,0.000099,0.000174,0.000323,0.000194,0.000223,0.000116,0.000215,0.000178,0.000138,
0.000228,0.000132,0.000313,0.000203,0.000113,0.000103,0.000293,0.000315,0.000407,0.000540,0.000774,0.000710,0.000487,0.000380,0.000464,0.000058,0.000274,0.000523,0.000294,0.000341,
0.000481,0.000536,0.000333,0.000164,0.000216,0.000307,0.000221,0.000221,0.000041,0.000276,0.000151,0.000308,0.000144,0.000192,0.000376,0.000268,0.000406,0.000201,0.000153,0.000108,
0.000228,0.000062,0.000225,0.000296,0.000161,0.000484,0.000240,0.000131,0.000031,0.000330,0.000197,0.000210,0.000173,0.000481,0.000553,0.000107,0.000071,0.000099,0.000066,0.000341,
0.000213,0.000231,0.000319,0.000611,0.000380,0.000124,0.000153,0.000098,0.000151,0.000318,0.000131,0.000152,0.000038,0.000016,0.000434,0.000263,0.000417,0.000278,0.000256,0.000202,
0.000122,0.000047,0.000177,0.000088,0.000024,0.000066,0.000112,0.000144,0.000100,0.000070,0.000125,0.000032,0.000133,0.000062,0.000350,0.000158,0.000128,0.000052,0.000189,0.000110,
0.000119,0.000087,0.000185,0.000078,0.000094,0.000050,0.000377,0.000166,0.000175,0.000128,0.000357,0.000263,0.000117,0.000097,0.000253,0.000092,0.000154,0.000061,0.000758,0.000640,
0.000327,0.000704,0.000347,0.000242,0.000152,0.000360,0.000498,0.000392,0.000210,0.000477,0.000049,0.000175,0.000303,0.000427,0.000527,0.000444,0.000327,0.000142,0.000257,0.000238,
0.000299,0.000119,0.000228,0.000280,0.000223,0.000288,0.000130,0.000199,0.000042,0.000183,0.000098,0.001020,0.001613,0.000377,0.000318,0.000280,0.000429,0.000199,0.000167,0.000276,
0.000061,0.000211,0.000133,0.000437,0.000437,0.001065,0.000115,0.000217,0.000382,0.000458,0.000419,0.000160,0.000377,0.000104,0.000351,0.000234,0.001003,0.001431,0.000806,0.000348,
0.000131,0.000107,0.000130,0.000251,0.000112,0.000220,0.000607,0.000272,0.000941,0.000380,0.001315,0.001986,0.001792,0.000159,0.000622,0.000383,0.000602,0.000504,0.000436,0.000276,
0.000838,0.000576,0.001134,0.000755,0.000396,0.000098,0.000052,0.001503,0.000991,0.001882,0.001882,0.002153,0.001877,0.000770,0.000363,0.000264,0.000196,0.000159,0.000512,0.000291,
0.000374,0.000130,0.000164,0.000141,0.000071,0.000259,0.000161,0.000139,0.000956,0.000540,0.000657,0.000589,0.000956,0.000763,0.000682,0.000520,0.000291,0.000581,0.000199,0.000116,
0.000148,0.000059,0.000211,0.000191,0.000118,0.000640,0.000096,0.000291,0.000296,0.000092,0.000064,0.000111,0.000162,0.000512,0.000442,0.000297,0.000348,0.000206,0.000084,0.000077,
0.000094,0.000032,0.000096,0.000074,0.000043,0.000045,0.000030,0.000036,0.000095,0.000072,0.000037,0.000049,0.000036,0.000036,0.000061,0.000117,0.000196,0.000212,0.000383,0.000049,
0.000094,0.000043,0.000072,0.000042,0.000103,0.000099,0.000122,0.000252,0.000363,0.000366,0.000196,0.000157,0.000225,0.000089,0.000294,0.000128,0.000085,0.000103,0.000110,0.000177,
0.000097,0.000251,0.000440,0.000468,0.000277,0.000092,0.000181,0.000096,0.000199,0.000095,0.000257,0.000177,0.000156,0.000200,0.000232,0.000088,0.000094,0.000072,0.000114,0.000100,
0.000246,0.000192,0.000333,0.000119,0.000048,0.000041,0.000167,0.000373,0.000204,0.000338,0.000151,0.000446,0.000495,0.000491,0.000301,0.000429,0.000238,0.000566,0.000477,0.000110,
0.000269,0.000227,0.000048,0.000462,0.000312,0.000959,0.000487,0.000240,0.000216,0.000370,0.000404,0.000321,0.000199,0.000167,0.000313,0.000221,0.000231,0.000119,0.000201,0.000228,
0.000094,0.000130,0.000066,0.000173,0.000188,0.000240,0.000067,0.000105,0.000044,0.000077,0.000058,0.000103,0.000377,0.000125,0.000102,0.000147,0.000070,0.000139,0.000198,0.000103,
0.000159,0.000037,0.000091,0.000316,0.000193,0.000101,0.000229,0.000333,0.000170,0.000287,0.000179,0.000042,0.000142,0.000040,0.000102,0.000257,0.000274,0.000135,0.000069,0.000056,
0.000028,0.000048,0.000092,0.000143,0.000161,0.000179,0.000031,0.000059,0.000036,0.000062,0.000131,0.000298,0.000158,0.000213,0.000181,0.000104,0.000068,0.000067,0.000079,0.000064,
0.000102,0.000177,0.000147,0.000151,0.000092,0.000087,0.000050,0.000059,0.000038,0.000062,0.000130,0.000048,0.000031,0.000111,0.000515,0.000324,0.000250,0.000263,0.000107,0.000114,
0.000075,0.000058,0.000016,0.000062,0.000057,0.000092,0.000687,0.000430,0.000122,0.000144,0.000157,0.000022,0.000100,0.000068,0.000026,0.000135,0.000039,0.000061,0.000129,0.000119,
0.000032,0.000039,0.000024,0.000017,0.000024,0.000023,0.000073,0.000041,0.000024,0.000053,0.000216,0.000323,0.000291,0.000029,0.000046,0.000015,0.000029,0.000017,0.000108,0.000054,
0.000058,0.000401,0.000236,0.000021,0.000061,0.000021,0.000257,0.000313,0.000386,0.000148,0.000291,0.000170,0.000168,0.000170,0.000134,0.000032,0.000107,0.000122,0.000062,0.000062,
0.000053,0.000049,0.000338,0.000030,0.000015,0.000014,0.000161,0.000083,0.000098,0.000144,0.000045,0.000066,0.000203,0.000114,0.000227,0.000055,0.000119,0.000050,0.000098,0.000011,
0.000034,0.000130,0.000093,0.000050,0.000082,0.000099,0.000075,0.000030,0.000190,0.000095,0.000212,0.000101,0.000181,0.000210,0.000141,0.000170,0.000041,0.000081,0.000113,0.000196,
0.000106,0.000137,0.000023,0.000077,0.000079,0.000031,0.000263,0.000223,0.000108,0.000014,0.000025,0.000056,0.000047,0.000074,0.000080,0.000032,0.000098,0.000059,0.000045,0.000062,
0.000032,0.000077,0.000035,0.000021,0.000061,0.000119,0.000059,0.000054,0.000133,0.000081,0.000176,0.000194,0.000093,0.000030,0.000038,0.000122,0.000058,0.000038,0.000116,0.000152,
0.000064,0.000018,0.000041,0.000157,0.000074,0.000161,0.000125,0.000131,0.000082,0.000170,0.000095,0.000148,0.000048,0.000017,0.000047,0.000034,0.000059,0.000072,0.000039,0.000029,
0.000056,0.000041,0.000033,0.000037,0.000031,0.000086,0.000119,0.000051,0.000037,0.000045,0.000096,0.000088,0.000105,0.000129,0.000071,0.000041,0.000118,0.000090,0.000092,0.000009,
0.000182,0.000017,0.000246,0.000298,0.000157,0.000022,0.000022,0.000019,0.000235,0.000235,0.000105,0.000038,0.000067,0.000080,0.000236,0.000202,0.000088,0.000135,0.000053,0.000031,
0.000108,0.000057,0.000019,0.000083,0.000073,0.000118,0.000135,0.000094,0.000171,0.000110,0.000095,0.000051,0.000103,0.000141,0.000106,0.000098,0.000101,0.000116,0.000036,0.000116,
0.000098,0.000035,0.000038,0.000009,0.000173,0.000301,0.000291,0.000140,0.000190,0.000183,0.000166,0.000148,0.000170,0.000198,0.000084,0.000070,0.000150,0.000192,0.000069,0.000179,
0.000164,0.000147,0.000143,0.000064,0.000016,0.000255,0.000148,0.000019,0.000107,0.000096,0.000081,0.000159,0.000039,0.000014,0.000016,0.000029,0.000037,0.000062,0.000070,0.000097,
0.000050,0.000035,0.000070,0.000057,0.000071,0.000072,0.000085,0.000048,0.000054,0.000063,0.000041,0.000077,0.000081,0.000098,0.000133,0.000068,0.000062,0.000148,0.000158,0.000045,
0.000017,0.000031,0.000021,0.000046,0.000193,0.000112,0.000152,0.000141,0.000135,0.000135,0.000040,0.000025,0.000024,0.000069,0.000024,0.000012,0.000027,0.000021,0.000026,0.000039,
0.000018,0.000015,0.000120,0.000039,0.000085,0.000089,0.000130,0.000194,0.000225,0.000164,0.000091,0.000027,0.000016,0.000022,0.000082,0.000066,0.000032,0.000031,0.000040,0.000125,
0.000074,0.000047,0.000065,0.000074,0.000018,0.000121,0.000040,0.000059,0.000822,0.000495,0.000061,0.000103,0.000069,0.000246,0.000310,0.000216,0.000102,0.000229,0.000324,0.000257,
0.000231,0.000291,0.000296,0.000231,0.000136,0.000037,0.000244,0.000104,0.000131,0.000265,0.000235,0.000164,0.000106,0.000124,0.000121,0.000192,0.000092,0.000036,0.000175,0.000176,
0.000181,0.000055,0.000229,0.000160,0.000144,0.000364,0.000478,0.000122,0.000201,0.000166,0.000036,0.000047,0.000060,0.000125,0.000197,0.000168,0.000135,0.000065,0.000056,0.000141,
0.000023,0.000003,0.000208,0.000236,0.000110,0.000111,0.000022,0.000049,0.000126,0.000108,0.000048,0.000079,0.000015,0.000050,0.000049,0.000023,0.000012,0.000013,0.000009,0.000017,
0.000044,0.000027,0.000032,0.000008,0.000019,0.000050,0.000023,0.000101,0.000055,0.000026,0.000013,0.000011,0.000012,0.000032,0.000036,0.000046,0.000030,0.000049,0.000063,0.000037,
0.000019,0.000039,0.000025,0.000052,0.000024,0.000033,0.000011,0.000011,0.000013,0.000009,0.000028,0.000052,0.000028,0.000023,0.000029,0.000009,0.000022,0.000006,0.000019,0.000025,
0.000036,0.000039,0.000016,0.000022,0.000022,0.000030,0.000015,0.000010,0.000007,0.000007,0.000042,0.000042,0.000018,0.000014,0.000038,0.000014,0.000024,0.000004,0.000012,0.000042,
0.000068,0.000004,0.000045,0.000057,0.000017,0.000010,0.000009,0.000037,0.000010,0.000005,0.000008,0.000008,0.000007,0.000017,0.000017,0.000068,0.000025,0.000017,0.000031,0.000037,
0.000012,0.000002,0.000015,0.000038,0.000026,0.000044,0.000032,0.000015,0.000009,0.000004,0.000021,0.000042,0.000030,0.000006,0.000002,0.000003,0.000013,0.000009,0.000004,0.000007,
0.000007,0.000002,0.000008,0.000004,0.000011,0.000016,0.000011,0.000021,0.000029,0.000016,0.000035,0.000018,0.000011,0.000005,0.000008,0.000014,0.000043,0.000048,0.000028,0.000007,
0.000007,0.000007,0.000022,0.000053,0.000057,0.000024,0.000080,0.000056,0.000006,0.000049,0.000019,0.000016,0.000058,0.000034,0.000024,0.000031,0.000024,0.000013,0.000001,0.000046,
0.000027,0.000020,0.000037,0.000006,0.000030,0.000034,0.000014,0.000019,0.000031,0.000024,0.000017,0.000035,0.000028,0.000033,0.000022,0.000027,0.000020,0.000010,0.000053,0.000038,
0.000030,0.000006,0.000037,0.000030,0.000031,0.000116,0.000043,0.000054,0.000014,0.000040,0.000007,0.000021,0.000026,0.000032,0.000017,0.000015,0.000021,0.000045,0.000037,0.000015,
0.000003,0.000013,0.000023,0.000015,0.000025,0.000018,0.000020,0.000008,0.000007,0.000007,0.000013,0.000015,0.000010,0.000007,0.000004,0.000014,0.000007,0.000004,0.000003,0.000010,
0.000023,0.000010,0.000030,0.000042,0.000028,0.000007,0.000021,0.000020,0.000043,0.000021,0.000024,0.000032,0.000018,0.000008,0.000032,0.000018,0.000007,0.000004,0.000012,0.000008,
0.000025,0.000006,0.000003,0.000014,0.000006,0.000003,0.000005,0.000005,0.000003,0.000006,0.000004,0.000011,0.000010,0.000024,0.000023,0.000008,0.000020,0.000019,0.000006,0.000014,
0.000005,0.000003,0.000005,0.000002,0.000036,0.000058,0.000023,0.000011,0.000073,0.000026,0.000041,0.000005,0.000018,0.000051,0.000094,0.000054,0.000075,0.000003,0.000054,0.000083,
0.000014,0.000009,0.000031,0.000012,0.000009,0.000017,0.000016,0.000007,0.000012,0.000009,0.000017,0.000017,0.000036,0.000051,0.000012,0.000006,0.000007,0.000024,0.000013,0.000006,
0.000006,0.000005,0.000001,0.000007,0.000012,0.000068,0.000052,0.000043,0.000039,0.000030,0.000007,0.000054,0.000031,0.000022,0.000019,0.000026,0.000005,0.000015,0.000052,0.000023,
0.000071,0.000058,0.000005,0.000073,0.000087,0.000051,0.000060,0.000146,0.000006,0.000148,0.000088,0.000071,0.000067,0.000042,0.000008,0.000006,0.000010,0.000061,0.000019,0.000013,
0.000025,0.000017,0.000006,0.000010,0.000077,0.000016,0.000016,0.000077,0.000033,0.000031,0.000033,0.000073,0.000034,0.000021,0.000027,0.000054,0.000021,0.000085,0.000005,0.000008,
0.000033,0.000098,0.000052,0.000178,0.000199,0.000018,0.000119,0.000039,0.000124,0.000021,0.000013,0.000014,0.000016,0.000088,0.000034,0.000013,0.000022,0.000014,0.000003,0.000020,
0.000039,0.000065,0.000130,0.000044,0.000037,0.000055,0.000113,0.000212,0.000137,0.000059,0.000017,0.000033,0.000030,0.000024,0.000028,0.000066,0.000055,0.000040,0.000014,0.000038,
0.000157,0.000135,0.000043,0.000119,0.000294,0.000080,0.000155,0.000060,0.000229,0.000143,0.000086,0.000098,0.000128,0.000062,0.000085,0.000138,0.000042,0.000049,0.000191,0.000055,
0.000031,0.000019,0.000013,0.000192,0.000110,0.000065,0.000005,0.000005,0.000143,0.000247,0.000119,0.000154,0.000047,0.000005,0.000015,0.000006,0.000006,0.000027,0.000110,0.000069,
0.000106,0.000022,0.000049,0.000091,0.000018,0.000019,0.000018,0.000010,0.000028,0.000049,0.000023,0.000021,0.000027,0.000038,0.000033,0.000043,0.000048,0.000009,0.000053,0.000030,
0.000033,0.000039,0.000080,0.000028,0.000053,0.000096,0.000035,0.000033,0.000037,0.000053,0.000008,0.000029,0.000033,0.000016,0.000009,0.000013,0.000020,0.000036,0.000007,0.000026,
0.000014,0.000008,0.000013,0.000012,0.000033,0.000018,0.000020,0.000030,0.000035,0.000015,0.000016,0.000009,0.000014,0.000015,0.000032,0.000023,0.000026,0.000045,0.000031,0.000048,
0.000022,0.000024,0.000046,0.000037,0.000011,0.000004,0.000009,0.000017,0.000017,0.000010,0.000016,0.000012,0.000015,0.000012,0.000013,0.000019,0.000010,0.000013,0.000011,0.000017,
0.000013,0.000007,0.000038,0.000085,0.000014,0.000036,0.000003,0.000008,0.000022,0.000054,0.000014,0.000058,0.000020,0.000004,0.000003,0.000001,0.000003,0.000002,0.000003,0.000017,
0.000006,0.000022,0.000005,0.000005,0.000010,0.000023,0.000003,0.000003,0.000010,0.000031,0.000013,0.000018,0.000009,0.000013,0.000004,0.000017,0.000039,0.000019,0.000017,0.000007,
0.000014,0.000052,0.000144,0.000071,0.000076,0.000038,0.000017,0.000026,0.000014,0.000014,0.000003,0.000010,0.000022,0.000014,0.000020,0.000010,0.000015,0.000023,0.000057,0.000012,
0.000069,0.000009,0.000012,0.000004,0.000019,0.000009,0.000011,0.000015,0.000037,0.000069,0.000032,0.000011,0.000005,0.000055,0.000010,0.000074,0.000040,0.000083,0.000018,0.000039,
0.000036,0.000092,0.000102,0.000012,0.000005,0.000045,0.000032,0.000044,0.000076,0.000030,0.000051,0.000003,0.000073,0.000011,0.000010,0.000018,0.000053,0.000046,0.000024,0.000027,
0.000031,0.000082,0.000033,0.000044,0.000108,0.000011,0.000009,0.000011,0.000007,0.000006,0.000033,0.000065,0.000017,0.000037,0.000026,0.000007,0.000006,0.000002,0.000015,0.000031,
0.000029,0.000032,0.000031,0.000073,0.000030,0.000029,0.000019,0.000012,0.000015,0.000032,0.000019,0.000013,0.000003,0.000009,0.000023,0.000016,0.000017,0.000014,0.000013,0.000030,
0.000068,0.000015,0.000030,0.000047,0.000056,0.000135,0.000019,0.000023,0.000007,0.000012,0.000052,0.000020,0.000040,0.000038,0.000036,0.000040,0.000021,0.000066,0.000028,0.000008,
0.000002,0.000005,0.000144,0.000016,0.000015,0.000003,0.000035,0.000018,0.000021,0.000024,0.000017,0.000101,0.000010,0.000011,0.000004,0.000056,0.000023,0.000035,0.000039,0.000135,
0.000324,0.000061,0.000294,0.000151,0.000064,0.000117,0.000019,0.000209,0.000263,0.000143,0.000003,0.000021,0.000008,0.000008,0.000019,0.000035,0.000020,0.000002,0.000033,0.000077,
0.000010,0.000011,0.000011,0.000244,0.000113,0.000114,0.000063,0.000021,0.000002,0.000021,0.000014,0.000031,0.000053,0.000025,0.000019,0.000021,0.000023,0.000024,0.000010,0.000060,
0.000029,0.000133,0.000015,0.000015,0.000027,0.000138,0.000121,0.000294,0.000137,0.000061,0.000029,0.000105,0.000083,0.000036,0.000047,0.000088,0.000015,0.000102,0.000142,0.000060,
0.000035,0.000135,0.000035,0.000118,0.000160,0.000076,0.000094,0.000116,0.000064,0.000080,0.000027,0.000031,0.000038,0.000018,0.000114,0.000046,0.000084,0.000013,0.000091,0.000017,
0.000022,0.000015,0.000093,0.000047,0.000016,0.000012,0.000034,0.000091,0.000024,0.000050,0.000059,0.000040,0.000047,0.000004,0.000068,0.000014,0.000013,0.000046,0.000012,0.000042,
0.000143,0.000068,0.000086,0.000047,0.000052,0.000025,0.000027,0.000066,0.000031,0.000068,0.000011,0.000025,0.000047,0.000012,0.000015,0.000008,0.000022,0.000010,0.000008,0.000010,
0.000058,0.000092,0.000024,0.000068,0.000030,0.000063,0.000011,0.000031,0.000064,0.000010,0.000013,0.000013,0.000058,0.000035,0.000020,0.000014,0.000038,0.000028,0.000058,0.000064,
0.000049,0.000118,0.000114,0.000093,0.000062,0.000139,0.000062,0.000102,0.000033,0.000052,0.000030,0.000014,0.000021,0.000023,0.000039,0.000018,0.000010,0.000094,0.000125,0.000060,
0.000012,0.000003,0.000015,0.000104,0.000095,0.000033,0.000044,0.000022,0.000009,0.000013,0.000015,0.000023,0.000010,0.000013,0.000050,0.000037,0.000070,0.000049,0.000056,0.000008,
0.000073,0.000093,0.000098,0.000130,0.000116,0.000050,0.000017,0.000041,0.000058,0.000128,0.000058,0.000052,0.000029,0.000033,0.000106,0.000026,0.000025,0.000044,0.000114,0.000032,
0.000079,0.000124,0.000069,0.000007,0.000014,0.000005,0.000005,0.000013,0.000054,0.000013,0.000006,0.000024,0.000012,0.000018,0.000020,0.000018,0.000045,0.000009,0.000022,0.000035,
0.000066,0.000047,0.000050,0.000060,0.000046,0.000040,0.000046,0.000013,0.000090,0.000007,0.000023,0.000030,0.000038,0.000053,0.000053,0.000077,0.000046,0.000042,0.000028,0.000141,
0.000133,0.000101,0.000071,0.000092,0.000101,0.000126,0.000240,0.000078,0.000068,0.000015,0.000018,0.000083,0.000020,0.000010,0.000011,0.000062,0.000074,0.000110,0.000108,0.000060,
0.000028,0.000119,0.000028,0.000038,0.000037,0.000064,0.000023,0.000020,0.000148,0.000167,0.000047,0.000075,0.000092,0.000076,0.000142,0.000088,0.000044,0.000084,0.000121,0.000087,
0.000067,0.000023,0.000118,0.000144,0.000075,0.000107,0.000066,0.000130,0.000095,0.000081,0.000247,0.000054,0.000086,0.000051,0.000025,0.000005,0.000046,0.000031,0.000064,0.000179,
0.000199,0.000157,0.000121,0.000031,0.000042,0.000025,0.000021,0.000019,0.000167,0.000066,0.000070,0.000019,0.000026,0.000021,0.000073,0.000098,0.000102,0.000020,0.000078,0.000164,
0.000060,0.000034,0.000028,0.000080,0.000028,0.000021,0.000047,0.000055,0.000032,0.000020,0.000090,0.000202,0.000049,0.000273,0.000199,0.000131,0.000179,0.000040,0.000088,0.000126,
0.000038,0.000084,0.000192,0.000021,0.000106,0.000049,0.000065,0.000103,0.000046,0.000025,0.000059,0.000058,0.000043,0.000065,0.000046,0.000015,0.000047,0.000051,0.000040,0.000125,
0.000016,0.000063,0.000060,0.000035,0.000024,0.000024,0.000063,0.000122,0.000033,0.000025,0.000128,0.000007,0.000069,0.000115,0.000108,0.000144,0.000131,0.000167,0.000029,0.000062,
0.000171,0.000049,0.000033,0.000032,0.000051,0.000034,0.000014,0.000076,0.000063,0.000020,0.000028,0.000058,0.000068,0.000022,0.000010,0.000046,0.000028,0.000047,0.000011,0.000024,
0.000017,0.000012,0.000037,0.000075,0.000117,0.000019,0.000046,0.000025,0.000021,0.000116,0.000023,0.000020,0.000023,0.000037,0.000057,0.000124,0.000044,0.000026,0.000045,0.000113,
0.000056,0.000174,0.000141,0.000116,0.000153,0.000101,0.000175,0.000138,0.000054,0.000015,0.000045,0.000021,0.000091,0.000067,0.000059,0.000233,0.000056,0.000203,0.000015,0.000112,
0.000124,0.000323,0.000520,0.000103,0.000099,0.000135,0.000081,0.000063,0.000077,0.000011,0.000086,0.000070,0.000050,0.000135,0.000169,0.000171,0.000128,0.000095,0.000238,0.000165,
0.000184,0.000083,0.000048,0.000083,0.000248,0.000099,0.000377,0.000419,0.000330,0.000124,0.000066,0.000070,0.000089,0.000067,0.000380,0.000134,0.000269,0.000406,0.000185,0.000211,
0.000427,0.000130,0.000038,0.000313,0.000268,0.000225,0.000103,0.000021,0.000012,0.000015,0.000125,0.000061,0.000280,0.000090,0.000033,0.000062,0.000011,0.000023,0.000033,0.000128,
0.000066,0.000280,0.000182,0.000139,0.000070,0.000098,0.000083,0.000086,0.000047,0.000028,0.000078,0.000080,0.000060,0.000034,0.000022,0.000015,0.000081,0.000302,0.000015,0.000058,
0.000018,0.000062,0.000073,0.000068,0.000026,0.000043,0.000076,0.000138,0.000017,0.000043,0.000050,0.000104,0.000056,0.000044,0.000005,0.000024,0.000059,0.000015,0.000011,0.000005,
0.000027,0.000016,0.000021,0.000045,0.000014,0.000035,0.000030,0.000007,0.000011,0.000011,0.000011,0.000009,0.000135,0.000037,0.000151,0.000084,0.000098,0.000036,0.000175,0.000478,
0.000401,0.000411,0.000116,0.000060,0.000008,0.000063,0.000199,0.000135,0.000262,0.000019,0.000058,0.000019,0.000175,0.000216,0.000059,0.000095,0.000032,0.000566,0.000452,0.000164,
0.000157,0.000057,0.000125,0.000170,0.000348,0.000066,0.000111,0.000096,0.000386,0.000043,0.000030,0.000135,0.000226,0.000104,0.000313,0.000009,0.000058,0.000165,0.000113,0.000160,
0.000164,0.000095,0.000142,0.000584,0.000672,0.000309,0.000044,0.000185,0.000075,0.000121,0.000202,0.000157,0.000147,0.001042,0.000448,0.000215,0.000401,0.001463,0.000785,0.000126,
0.000371,0.001661,0.000231,0.000512,0.000164,0.000495,0.000018,0.000084,0.000236,0.000570,0.000156,0.000097,0.000049,0.000056,0.000161,0.000284,0.000201,0.000081,0.000038,0.000085,
0.000330,0.000024,0.000158,0.000251,0.000566,0.000160,0.000178,0.000128,0.000078,0.000229,0.000157,0.000791,0.000382,0.000056,0.000197,0.000448,0.000507,0.000975,0.000133,0.000141,
0.000251,0.000611,0.000247,0.000075,0.000110,0.000151,0.000044,0.000151,0.000084,0.000384,0.000170,0.000095,0.000041,0.000101,0.000159,0.000023,0.000089,0.000069,0.000353,0.000787,
0.000561,0.000065,0.000013,0.000023,0.000062,0.000052,0.000071,0.000059,0.000012,0.000012,0.000052,0.000084,0.000283,0.000240,0.000093,0.000049,0.000058,0.000085,0.000070,0.000452,
0.000234,0.000802,0.001092,0.000153,0.000202,0.000755,0.000148,0.000081,0.000143,0.000363,0.000269,0.000971,0.000223,0.000969,0.000523,0.000617,0.002062,0.000758,0.000427,0.000173,
0.000304,0.000318,0.000047,0.000113,0.000073,0.000870,0.000354,0.000269,0.000406,0.000148,0.000561,0.000293,0.000049,0.000135,0.000819,0.000682,0.001181,0.001334,0.000546,0.000158,
0.000066,0.000019,0.000189,0.000260,0.000235,0.000240,0.000116,0.000155,0.000513,0.000637,0.000264,0.000411,0.000894,0.000417,0.002081,0.000125,0.000273,0.000860,0.001474,0.002125,
0.002287,0.002689,0.000802,0.000276,0.000294,0.000597,0.000828,0.000478,0.000918,0.000434,0.001031,0.000589,0.000336,0.000072,0.000332,0.000178,0.000054,0.000302,0.000465,0.000113,
0.000130,0.000078,0.000252,0.000640,0.000242,0.000274,0.000291,0.000157,0.000615,0.000262,0.000249,0.000774,0.000158,0.000884,0.001057,0.000520,0.000607,0.000556,0.001891,0.000911,
0.000667,0.001137,0.000996,0.000597,0.000433,0.000274,0.000846,0.000454,0.000570,0.000669,0.000196,0.000908,0.000838,0.001236,0.000102,0.000089,0.000208,0.000682,0.000730,0.000579,
0.000699,0.000094,0.000085,0.000220,0.000096,0.000495,0.000294,0.000566,0.000369,0.000444,0.000210,0.000561,0.000072,0.000034,0.000120,0.000680,0.000335,0.000231,0.000452,0.000454,
0.000884,0.000056,0.000448,0.000484,0.000750,0.001204,0.000107,0.000434,0.001223,0.000289,0.000693,0.001236,0.000462,0.000427,0.001648,0.000504,0.001811,0.000182,0.000157,0.000437,
0.000148,0.000181,0.000333,0.000527,0.000167,0.000248,0.000125,0.000113,0.000936,0.000991,0.000164,0.000916,0.000570,0.000359,0.000184,0.000369,0.001131,0.000081,0.000584,0.001367,
0.000570,0.000434,0.000570,0.000876,0.001048,0.000150,0.000622,0.000195,0.000191,0.000640,0.001149,0.000789,0.000693,0.001386,0.000931,0.000384,0.000906,0.002439,0.000128,0.000903,
0.001918,0.001539,0.001233,0.001816,0.002985,0.001831,0.000678,0.000196,0.000888,0.001499,0.001060,0.000996,0.002575,0.001527,0.001298,0.001891,0.001374,0.001009,0.001291,0.001175,
0.001122,0.000575,0.000380,0.000637,0.001527,0.000187,0.000781,0.001851,0.001552,0.001134,0.001298,0.002062,0.000898,0.000699,0.000891,0.003819,0.000579,0.002562,0.002668,0.002007,
0.000288,0.000741,0.001467,0.002472,0.001877,0.003012,0.002689,0.003189,0.000865,0.002262,0.000351,0.000804,0.003189,0.000941,0.003021,0.000323,0.001443,0.002495,0.000797,0.001077,
0.001679,0.001015,0.002356,0.001181,0.000313,0.001106,0.000477,0.000130,0.000795,0.000741,0.001031,0.000687,0.002007,0.000377,0.001140,0.000336,0.000982,0.001856,0.002014,0.001684,
0.002035,0.002445,0.001494,0.001871,0.001759,0.001313,0.004715,0.002850,0.003819,0.003687,0.001630,0.001871,0.001140,0.003283,0.005039,0.006599,0.002487,0.003788,0.002708,0.001315,
0.003092,0.002966,0.001119,0.005157,0.009781,0.006546,0.004906,0.001792,0.003141,0.002466,0.003345,0.000697,0.006836,0.006077,0.002697,0.003881,0.007988,0.006203,0.002985,0.013489,
0.006229,0.005180,0.001991,0.004398,0.009384,0.008148,0.012238,0.006805,0.013016,0.024811,0.011627,0.020004,0.023590,0.002224,0.020645,0.003733,0.013229,0.017120,0.041199,0.034821,
0.000000,0.000000,0.000003,0.000005,0.000001,0.000001,0.000008,0.000002,0.000003,0.000004,0.000006,0.000006,0.000000,0.000003,0.000000,0.000004,0.000003,0.000000,0.000007,0.000004,
0.000007,0.000001,0.000005,0.000008,0.000007,0.000005,0.000001,0.000005,0.000004,0.000003,0.000001,0.000002,0.000003,0.000002,0.000002,0.000002,0.000002,0.000000,0.000001,0.000003,
0.000001,0.000002,0.000001,0.000001,0.000000,0.000001,0.000002,0.000001,0.000002,0.000003,0.000000,0.000001,0.000001,0.000001,0.000001,0.000004,0.000002,0.000001,0.000003,0.000003,
0.000003,0.000005,0.000006,0.000003,0.000004,0.000009,0.000005,0.000006,0.000005,0.000002,0.000003,0.000004,0.000011,0.000015,0.000005,0.000005,0.000000,0.000012,0.000009,0.000004,
0.000002,0.000001,0.000008,0.000016,0.000003,0.000010,0.000001,0.000001,0.000003,0.000002,0.000003,0.000000,0.000003,0.000002,0.000001,0.000003,0.000000,0.000009,0.000009,0.000008,
0.000006,0.000004,0.000000,0.000000,0.000002,0.000003,0.000002,0.000003,0.000002,0.000004,0.000007,0.000003,0.000002,0.000011,0.000019,0.000000,0.000003,0.000019,0.000000,0.000007,
0.000000,0.000016,0.000020,0.000000,0.000009,0.000009,0.000001,0.000000,0.000002,0.000000,0.000008,0.000001,0.000004,0.000001,0.000001,0.000005,0.000004,0.000000,0.000001,0.000001,
0.000005,0.000003,0.000004,0.000002,0.000001,0.000001,0.000001,0.000004,0.000000,0.000003,0.000000,0.000005,0.000002,0.000006,0.000010,0.000001,0.000003,0.000002,0.000009,0.000002,
0.000009,0.000013,0.000001,0.000006,0.000003,0.000007,0.000006,0.000003,0.000004,0.000002,0.000003,0.000002,0.000004,0.000001,0.000005,0.000002,0.000001,0.000003,0.000001,0.000003,
0.000002,0.000001,0.000000,0.000001,0.000002,0.000001,0.000001,0.000002,0.000002,0.000001,0.000001,0.000001,0.000003,0.000000,0.000001,0.000002,0.000000,0.000003,0.000004,0.000003,
0.000002,0.000000,0.000003,0.000003,0.000002,0.000003,0.000003,0.000001,0.000007,0.000003,0.000006,0.000007,0.000001,0.000003,0.000007,0.000008,0.000005,0.000005,0.000003,0.000000,
0.000003,0.000000,0.000003,0.000008,0.000002,0.000002,0.000000,0.000000,0.000000,0.000000,0.000001,0.000002,0.000000,0.000002,0.000000,0.000004,0.000001,0.000003,0.000003,0.000003,
0.000002,0.000004,0.000002,0.000003,0.000006,0.000008,0.000009,0.000012,0.000009,0.000007,0.000003,0.000000,0.000007,0.000009,0.000016,0.000050,0.000016,0.000006,0.000011,0.000029,
0.000052,0.000067,0.000054,0.000001,0.000006,0.000006,0.000078,0.000038,0.000009,0.000046,0.000006,0.000004,0.000019,0.000001,0.000008,0.000019,0.000017,0.000010,0.000015,0.000001,
0.000004,0.000001,0.000012,0.000011,0.000018,0.000020,0.000021,0.000013,0.000001,0.000007,0.000002,0.000005,0.000014,0.000009,0.000006,0.000005,0.000006,0.000004,0.000006,0.000024,
0.000007,0.000004,0.000025,0.000012,0.000009,0.000016,0.000002,0.000019,0.000003,0.000003,0.000006,0.000004,0.000002,0.000001,0.000002,0.000001,0.000002,0.000001,0.000002,0.000001,
0.000001,0.000001,0.000001,0.000002,0.000005,0.000002,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000002,0.000005,0.000001,0.000001,0.000000,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000002,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000001,0.000002,0.000002,0.000003,0.000001,0.000003,0.000002,0.000001,0.000001,0.000002,0.000002,0.000002,0.000002,0.000002,0.000001,0.000002,0.000004,
0.000003,0.000001,0.000001,0.000001,0.000001,0.000001,0.000002,0.000002,0.000001,0.000009,0.000006,0.000001,0.000003,0.000003,0.000002,0.000002,0.000001,0.000003,0.000004,0.000001,
0.000002,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000000,0.000001,0.000000,0.000002,0.000000,0.000002,0.000002,0.000002,0.000002,0.000001,0.000002,0.000001,
0.000003,0.000003,0.000004,0.000008,0.000002,0.000001,0.000000,0.000001,0.000002,0.000002,0.000002,0.000002,0.000001,0.000001,0.000001,0.000002,0.000003,0.000001,0.000001,0.000001,
0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000002,0.000001,0.000000,0.000000,
0.000001,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,0.000002,0.000002,0.000002,0.000003,0.000001,0.000001,0.000004,0.000003,0.000003,0.000002,0.000002,0.000001,0.000006,
0.000006,0.000003,0.000004,0.000003,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000001,0.000002,0.000001,0.000002,0.000002,
0.000002,0.000001,0.000001,0.000002,0.000002,0.000002,0.000003,0.000003,0.000001,0.000003,0.000003,0.000010,0.000011,0.000024,0.000014,0.000009,0.000013,0.000009,0.000001,0.000011,
0.000012,0.000004,0.000002,0.000018,0.000013,0.000014,0.000010,0.000006,0.000009,0.000005,0.000011,0.000000,0.000008,0.000003,0.000006,0.000001,0.000004,0.000007,0.000005,0.000003,
0.000005,0.000003,0.000002,0.000004,0.000000,0.000003,0.000004,0.000001,0.000005,0.000005,0.000004,0.000000,0.000004,0.000002,0.000004,0.000005,0.000009,0.000014,0.000001,0.000002,
0.000001,0.000000,0.000013,0.000008,0.000009,0.000010,0.000018,0.000020,0.000003,0.000008,0.000006,0.000007,0.000010,0.000002,0.000005,0.000001,0.000000,0.000011,0.000004,0.000008,
0.000009,0.000010,0.000005,0.000006,0.000001,0.000004,0.000002,0.000001,0.000003,0.000001,0.000003,0.000003,0.000003,0.000004,0.000001,0.000002,0.000000,0.000005,0.000005,0.000005,
0.000004,0.000004,0.000004,0.000004,0.000005,0.000008,0.000004,0.000005,0.000001,0.000010,0.000006,0.000011,0.000013,0.000011,0.000024,0.000013,0.000012,0.000026,0.000010,0.000012,
0.000001,0.000038,0.000025,0.000005,0.000046,0.000026,0.000008,0.000008,0.000011,0.000017,0.000015,0.000008,0.000018,0.000003,0.000003,0.000005,0.000012,0.000014,0.000021,0.000013,
0.000006,0.000012,0.000009,0.000015,0.000008,0.000011,0.000011,0.000028,0.000016,0.000016,0.000018,0.000003,0.000006,0.000001,0.000036,0.000110,0.000013,0.000061,0.000046,0.000017,
0.000032,0.000045,0.000048,0.000006,0.000025,0.000003,0.000016,0.000055,0.000187,0.000008,0.000038,0.000051,0.000012,0.000118,0.000062,0.000100,0.000012,0.000033,0.000004,0.000115,
0.000098,0.000150,0.000023,0.000046,0.000026,0.000004,0.000031,0.000013,0.000058,0.000144,0.000035,0.000164,0.000024,0.000247,0.000223,0.000350,0.000013,0.000190,0.000153,0.000054,
0.000203,0.000185,0.000132,0.000363,0.000144,0.000434,0.000071,0.000283,0.000050,0.000011,0.000359,0.000217,0.000525,0.000877,0.001026,0.000906,0.000114,0.000133,0.000168,0.000116,
0.000101,0.000070,0.000024,0.000055,0.000036,0.000042,0.000048,0.000022,0.000028,0.000009,0.000004,0.000054,0.000057,0.000051,0.000041,0.000079,0.000040,0.000033,0.000064,0.000018,
0.000032,0.000012,0.000015,0.000017,0.000006,0.000008,0.000010,0.000008,0.000078,0.000010,0.000033,0.000014,0.000007,0.000004,0.000004,0.000011,0.000033,0.000045,0.000014,0.000017,
0.000018,0.000004,0.000004,0.000018,0.000002,0.000003,0.000003,0.000005,0.000006,0.000002,0.000002,0.000001,0.000002,0.000001,0.000002,0.000003,0.000007,0.000004,0.000004,0.000002,
0.000004,0.000016,0.000002,0.000005,0.000002,0.000001,0.000001,0.000002,0.000002,0.000003,0.000004,0.000006,0.000009,0.000008,0.000011,0.000011,0.000005,0.000010,0.000004,0.000014,
0.000006,0.000015,0.000014,0.000008,0.000012,0.000018,0.000022,0.000028,0.000016,0.000012,0.000006,0.000014,0.000003,0.000021,0.000016,0.000013,0.000005,0.000012,0.000009,0.000004,
0.000006,0.000005,0.000003,0.000012,0.000011,0.000031,0.000015,0.000007,0.000003,0.000007,0.000010,0.000008,0.000018,0.000006,0.000017,0.000022,0.000022,0.000016,0.000029,0.000013,
0.000040,0.000038,0.000004,0.000031,0.000026,0.000002,0.000042,0.000026,0.000048,0.000034,0.000016,0.000041,0.000053,0.000028,0.000024,0.000009,0.000031,0.000053,0.000023,0.000023,
0.000011,0.000013,0.000012,0.000007,0.000011,0.000004,0.000005,0.000007,0.000009,0.000008,0.000007,0.000002,0.000004,0.000003,0.000002,0.000008,0.000005,0.000004,0.000004,0.000003,
0.000005,0.000006,0.000008,0.000063,0.000004,0.000004,0.000006,0.000004,0.000002,0.000006,0.000008,0.000008,0.000009,0.000006,0.000003,0.000031,0.000005,0.000004,0.000005,0.000010,
0.000013,0.000008,0.000004,0.000002,0.000002,0.000002,0.000004,0.000005,0.000006,0.000003,0.000002,0.000001,0.000001,0.000003,0.000006,0.000004,0.000005,0.000003,0.000002,0.000002,
0.000002,0.000001,0.000001,0.000001,0.000002,0.000004,0.000004,0.000015,0.000005,0.000002,0.000002,0.000001,0.000002,0.000003,0.000004,0.000002,0.000003,0.000014,0.000018,0.000017,
0.000016,0.000015,0.000015,0.000005,0.000004,0.000002,0.000003,0.000004,0.000003,0.000030,0.000034,0.000011,0.000017,0.000011,0.000001,0.000005,0.000002,0.000003,0.000006,0.000004,
0.000003,0.000006,0.000006,0.000002,0.000003,0.000001,0.000001,0.000001,0.000001,0.000001,0.000004,0.000001,0.000001,0.000008,0.000010,0.000014,0.000001,0.000004,0.000000,0.000001,
0.000000,0.000001,0.000006,0.000002,0.000027,0.000018,0.000001,0.000003,0.000001,0.000013,0.000027,0.000028,0.000002,0.000028,0.000002,0.000010,0.000011,0.000009,0.000001,0.000001,
0.000009,0.000005,0.000007,0.000002,0.000002,0.000010,0.000000,0.000001,0.000000,0.000007,0.000003,0.000003,0.000004,0.000000,0.000001,0.000004,0.000006,0.000010,0.000001,0.000005,
0.000003,0.000006,0.000000,0.000001,0.000004,0.000001,0.000001,0.000006,0.000006,0.000004,0.000001,0.000006,0.000003,0.000021,0.000007,0.000014,0.000013,0.000008,0.000009,0.000001,
0.000004,0.000006,0.000007,0.000005,0.000007,0.000001,0.000005,0.000005,0.000001,0.000006,0.000010,0.000008,0.000001,0.000001,0.000002,0.000001,0.000001,0.000002,0.000001,0.000002,
0.000002,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000002,0.000001,0.000001,0.000002,0.000002,0.000003,0.000003,0.000001,0.000001,0.000002,0.000004,0.000002,
0.000001,0.000003,0.000004,0.000003,0.000001,0.000002,0.000003,0.000002,0.000004,0.000003,0.000003,0.000002,0.000005,0.000004,0.000007,0.000004,0.000001,0.000002,0.000001,0.000001,
0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000002,0.000002,0.000002,0.000001,0.000000,0.000002,0.000002,0.000003,0.000003,0.000003,0.000002,0.000004,
0.000002,0.000004,0.000001,0.000005,0.000001,0.000008,0.000014,0.000010,0.000001,0.000003,0.000001,0.000009,0.000013,0.000008,0.000002,0.000008,0.000012,0.000013,0.000018,0.000012,
0.000018,0.000007,0.000001,0.000006,0.000006,0.000001,0.000005,0.000004,0.000005,0.000007,0.000002,0.000010,0.000011,0.000009,0.000003,0.000004,0.000004,0.000003,0.000002,0.000002,
0.000005,0.000003,0.000004,0.000004,0.000001,0.000004,0.000001,0.000007,0.000008,0.000011,0.000009,0.000010,0.000012,0.000009,0.000008,0.000002,0.000014,0.000011,0.000004,0.000008,
0.000009,0.000002,0.000008,0.000008,0.000011,0.000008,0.000005,0.000000,0.000007,0.000004,0.000000,0.000003,0.000003,0.000003,0.000007,0.000002,0.000001,0.000001,0.000002,0.000001,
0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000002,0.000002,0.000002,0.000002,0.000001,0.000003,0.000002,0.000003,0.000003,0.000003,0.000002,
0.000003,0.000004,0.000003,0.000001,0.000002,0.000002,0.000003,0.000010,0.000012,0.000009,0.000008,0.000006,0.000006,0.000001,0.000003,0.000002,0.000003,0.000001,0.000001,0.000002,
0.000003,0.000004,0.000005,0.000001,0.000001,0.000005,0.000003,0.000004,0.000004,0.000005,0.000006,0.000008,0.000019,0.000007,0.000003,0.000002,0.000002,0.000001,0.000006,0.000005,
0.000005,0.000002,0.000005,0.000006,0.000007,0.000008,0.000016,0.000002,0.000020,0.000004,0.000006,0.000124,0.000134,0.000010,0.000022,0.000004,0.000066,0.000098,0.000110,0.000055,
0.000029,0.000021,0.000053,0.000031,0.000093,0.000061,0.000042,0.000021,0.000002,0.000026,0.000007,0.000014,0.000028,0.000019,0.000015,0.000002,0.000016,0.000013,0.000011,0.000006,
0.000001,0.000007,0.000013,0.000015,0.000005,0.000009,0.000008,0.000001,0.000012,0.000018,0.000002,0.000007,0.000007,0.000000,0.000003,0.000001,0.000008,0.000013,0.000013,0.000010,
0.000005,0.000003,0.000003,0.000001,0.000000,0.000003,0.000005,0.000004,0.000003,0.000002,0.000001,0.000002,0.000002,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,
0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,
0.000000,0.000000,0.000000,0.000000,0.000001,0.000004,0.000000,0.000001,0.000001,0.000002,0.000002,0.000003,0.000002,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,
0.000001,0.000000,0.000001,0.000004,0.000004,0.000001,0.000001,0.000003,0.000000,0.000001,0.000001,0.000005,0.000004,0.000003,0.000001,0.000003,0.000001,0.000000,0.000005,0.000001,
0.000001,0.000004,0.000000,0.000001,0.000000,0.000000,0.000006,0.000001,0.000001,0.000000,0.000000,0.000003,0.000005,0.000001,0.000003,0.000002,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,
0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000002,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000001,0.000001,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000003,0.000000,0.000000,0.000000,0.000001,0.000000,
0.000001,0.000001,0.000003,0.000010,0.000000,0.000014,0.000009,0.000004,0.000004,0.000000,0.000002,0.000008,0.000017,0.000000,0.000001,0.000000,0.000000,0.000000,0.000002,0.000001,
0.000000,0.000001,0.000005,0.000000,0.000000,0.000000,0.000006,0.000005,0.000003,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,
0.000001,0.000000,0.000001,0.000000,0.000010,0.000000,0.000000,0.000000,0.000002,0.000004,0.000012,0.000001,0.000002,0.000001,0.000002,0.000005,0.000002,0.000001,0.000002,0.000000,
0.000000,0.000003,0.000000,0.000000,0.000003,0.000000,0.000001,0.000003,0.000001,0.000003,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000002,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000001,0.000002,0.000001,0.000000,0.000003,0.000001,0.000002,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,
0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000001,0.000000,0.000001,0.000001,0.000000,
0.000001,0.000002,0.000001,0.000001,0.000000,0.000001,0.000001,0.000001,0.000002,0.000002,0.000001,0.000001,0.000001,0.000003,0.000000,0.000002,0.000001,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000003,0.000002,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,
0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000001,0.000001,0.000000,0.000002,0.000002,0.000002,0.000005,
0.000000,0.000001,0.000001,0.000000,0.000000,0.000002,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,
0.000001,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000002,0.000000,
0.000002,0.000000,0.000001,0.000001,0.000004,0.000003,0.000002,0.000002,0.000002,0.000001,0.000000,0.000001,0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,0.000002,0.000001,
0.000001,0.000002,0.000001,0.000003,0.000001,0.000001,0.000001,0.000002,0.000001,0.000004,0.000006,0.000004,0.000007,0.000002,0.000002,0.000002,0.000001,0.000006,0.000002,0.000002,
0.000009,0.000002,0.000005,0.000013,0.000003,0.000000,0.000002,0.000002,0.000003,0.000004,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,
0.000000,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,
0.000002,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000000,0.000001,0.000001,0.000002,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,
0.000000,0.000001,0.000003,0.000005,0.000006,0.000002,0.000001,0.000000,0.000001,0.000001,0.000002,0.000005,0.000001,0.000002,0.000000,0.000003,0.000003,0.000002,0.000003,0.000000,
0.000013,0.000009,0.000002,0.000009,0.000002,0.000002,0.000002,0.000005,0.000001,0.000002,0.000004,0.000008,0.000001,0.000001,0.000001,0.000002,0.000003,0.000009,0.000000,0.000001,
0.000001,0.000002,0.000005,0.000005,0.000004,0.000002,0.000019,0.000016,0.000026,0.000004,0.000004,0.000003,0.000003,0.000006,0.000010,0.000004,0.000060,0.000021,0.000005,0.000008,
0.000053,0.000028,0.000004,0.000016,0.000086,0.000009,0.000018,0.000007,0.000026,0.000001,0.000003,0.000008,0.000019,0.000005,0.000005,0.000002,0.000002,0.000003,0.000005,0.000004,
0.000002,0.000001,0.000001,0.000004,0.000000,0.000001,0.000003,0.000008,0.000003,0.000002,0.000002,0.000001,0.000003,0.000001,0.000010,0.000003,0.000001,0.000002,0.000005,0.000006,
0.000011,0.000001,0.000002,0.000002,0.000016,0.000007,0.000005,0.000003,0.000003,0.000000,0.000002,0.000002,0.000005,0.000005,0.000002,0.000001,0.000001,0.000002,0.000000,0.000001,
0.000001,0.000005,0.000012,0.000004,0.000002,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,0.000000,0.000003,0.000001,0.000005,0.000005,0.000001,0.000001,0.000003,
0.000001,0.000001,0.000004,0.000006,0.000007,0.000018,0.000003,0.000004,0.000024,0.000009,0.000004,0.000003,0.000009,0.000007,0.000021,0.000013,0.000034,0.000027,0.000023,0.000054,
0.000017,0.000007,0.000004,0.000009,0.000013,0.000001,0.000002,0.000002,0.000029,0.000007,0.000005,0.000010,0.000002,0.000017,0.000007,0.000001,0.000001,0.000012,0.000007,0.000017,
0.000029,0.000018,0.000004,0.000001,0.000000,0.000001,0.000002,0.000004,0.000003,0.000002,0.000002,0.000006,0.000014,0.000003,0.000004,0.000025,0.000008,0.000081,0.000003,0.000004,
0.000047,0.000073,0.000088,0.000082,0.000085,0.000056,0.000006,0.000016,0.000023,0.000032,0.000017,0.000035,0.000018,0.000076,0.000019,0.000012,0.000001,0.000007,0.000002,0.000000,
0.000003,0.000009,0.000001,0.000002,0.000003,0.000008,0.000016,0.000005,0.000003,0.000006,0.000003,0.000011,0.000006,0.000004,0.000012,0.000003,0.000012,0.000040,0.000024,0.000015,
0.000014,0.000052,0.000014,0.000012,0.000026,0.000026,0.000024,0.000010,0.000004,0.000015,0.000022,0.000015,0.000019,0.000003,0.000023,0.000024,0.000033,0.000006,0.000002,0.000007,
0.000008,0.000010,0.000015,0.000010,0.000001,0.000001,0.000006,0.000001,0.000006,0.000011,0.000009,0.000007,0.000007,0.000003,0.000005,0.000001,0.000001,0.000001,0.000005,0.000003,
0.000003,0.000008,0.000016,0.000020,0.000001,0.000009,0.000005,0.000021,0.000034,0.000007,0.000019,0.000031,0.000005,0.000020,0.000026,0.000012,0.000011,0.000039,0.000013,0.000024,
0.000005,0.000005,0.000023,0.000003,0.000003,0.000007,0.000022,0.000007,0.000010,0.000003,0.000001,0.000025,0.000037,0.000008,0.000037,0.000017,0.000009,0.000005,0.000007,0.000020,
0.000001,0.000010,0.000028,0.000011,0.000010,0.000011,0.000019,0.000028,0.000003,0.000011,0.000006,0.000005,0.000030,0.000067,0.000037,0.000016,0.000065,0.000028,0.000012,0.000063,
0.000160,0.000006,0.000051,0.000150,0.000155,0.000100,0.000237,0.000402,0.000242,0.000083,0.000011,0.000089,0.000172,0.000088,0.000021,0.000184,0.000128,0.000073,0.000213,0.000147,
0.000041,0.000076,0.000030,0.000145,0.000063,0.000014,0.000041,0.000101,0.000009,0.000057,0.000141,0.000132,0.000074,0.000069,0.000148,0.000085,0.000060,0.000062,0.000213,0.000016,
0.000144,0.000473,0.000375,0.000040,0.000086,0.000082,0.000271,0.000115,0.000240,0.000620,0.000641,0.000180,0.000320,0.000064,0.000106,0.000456,0.000096,0.000250,0.000018,0.000129,
0.000377,0.000122,0.000130,0.000183,0.000078,0.000104,0.000168,0.000057,0.000135,0.000035,0.000007,0.000045,0.000025,0.000080,0.000053,0.000173,0.000036,0.000068,0.000019,0.000070,
0.000171,0.000148,0.000176,0.000194,0.000304,0.000158,0.000198,0.000305,0.000144,0.000371,0.000344,0.000778,0.000906,0.000516,0.000594,0.000144,0.000507,0.000453,0.002245,0.001554,
0.001278,0.001267,0.000112,0.001707,0.001267,0.000317,0.001026,0.004322,0.003288,0.002872,0.000525,0.002069,0.001680,0.001360,0.000054,0.002857,0.001987,0.000350,0.003252,0.005424,
0.002796,0.000351,0.006863,0.008011,0.007713,0.001934,0.002741,0.009819,0.003557,0.019363,0.003424,0.013443,0.043640,0.022995,0.034515,0.051483,0.003174,0.056061,0.006046,0.058777,
0.016464,0.237915,0.295898,0.011208,0.000000,0.000001,0.000002,0.000001,0.000000,0.000002,0.000001,0.000001,0.000001,0.000002,0.000002,0.000000,0.000001,0.000000,0.000001,0.000001,
0.000000,0.000002,0.000001,0.000001,0.000000,0.000001,0.000002,0.000003,0.000002,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,
0.000000,0.000001,0.000001,0.000001,0.000002,0.000002,0.000001,0.000001,0.000002,0.000002,0.000003,0.000003,0.000001,0.000001,0.000001,0.000002,0.000004,0.000002,0.000002,0.000000,
0.000003,0.000003,0.000001,0.000001,0.000000,0.000003,0.000004,0.000001,0.000003,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,
0.000001,0.000002,0.000002,0.000002,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000001,0.000003,0.000002,0.000001,0.000002,0.000003,0.000000,0.000001,
0.000005,0.000000,0.000002,0.000000,0.000004,0.000005,0.000000,0.000005,0.000003,0.000001,0.000000,0.000001,0.000000,0.000002,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,
0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000001,0.000000,0.000001,0.000002,0.000000,0.000001,
0.000001,0.000002,0.000001,0.000002,0.000004,0.000000,0.000002,0.000001,0.000002,0.000002,0.000002,0.000002,0.000001,0.000001,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,
0.000001,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000002,0.000001,0.000002,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001,
0.000001,0.000001,0.000000,0.000001,0.000000,0.000001,0.000002,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,
0.000001,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000002,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000000,0.000002,0.000002,0.000003,0.000009,0.000004,
0.000002,0.000003,0.000006,0.000009,0.000012,0.000013,0.000000,0.000001,0.000002,0.000013,0.000009,0.000005,0.000018,0.000002,0.000001,0.000003,0.000000,0.000002,0.000004,0.000004,
0.000004,0.000003,0.000000,0.000001,0.000000,0.000002,0.000002,0.000003,0.000004,0.000003,0.000004,0.000000,0.000002,0.000000,0.000001,0.000003,0.000003,0.000002,0.000003,0.000002,
0.000001,0.000002,0.000006,0.000003,0.000003,0.000009,0.000004,0.000003,0.000004,0.000001,0.000004,0.000001,0.000002,0.000002,0.000003,0.000001,0.000001,0.000001,0.000000,0.000000,
0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000004,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000002,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,
0.000000,0.000000,0.000001,0.000001,0.000001,0.000004,0.000002,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000001,0.000002,0.000002,0.000005,0.000003,0.000002,0.000003,
0.000004,0.000000,0.000004,0.000003,0.000001,0.000000,0.000005,0.000005,0.000004,0.000004,0.000003,0.000002,0.000001,0.000003,0.000000,0.000002,0.000002,0.000002,0.000000,0.000001,
0.000002,0.000001,0.000001,0.000002,0.000001,0.000001,0.000002,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000000,0.000001,0.000001,0.000002,
0.000004,0.000000,0.000001,0.000001,0.000000,0.000003,0.000003,0.000003,0.000002,0.000004,0.000007,0.000001,0.000003,0.000002,0.000003,0.000003,0.000001,0.000002,0.000001,0.000000,
0.000003,0.000001,0.000001,0.000002,0.000003,0.000003,0.000002,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,
0.000001,0.000001,0.000002,0.000002,0.000002,0.000001,0.000001,0.000002,0.000003,0.000003,0.000003,0.000001,0.000002,0.000002,0.000003,0.000004,0.000005,0.000011,0.000006,0.000006,
0.000018,0.000022,0.000009,0.000001,0.000011,0.000008,0.000002,0.000013,0.000013,0.000005,0.000004,0.000003,0.000004,0.000003,0.000003,0.000005,0.000003,0.000002,0.000002,0.000005,
0.000005,0.000006,0.000005,0.000003,0.000003,0.000003,0.000004,0.000006,0.000003,0.000003,0.000054,0.000005,0.000007,0.000011,0.000008,0.000004,0.000001,0.000007,0.000019,0.000006,
0.000023,0.000020,0.000015,0.000017,0.000018,0.000021,0.000008,0.000013,0.000002,0.000006,0.000019,0.000054,0.000007,0.000017,0.000020,0.000006,0.000041,0.000030,0.000056,0.000017,
0.000018,0.000002,0.000030,0.000021,0.000039,0.000028,0.000024,0.000010,0.000003,0.000008,0.000003,0.000015,0.000036,0.000017,0.000045,0.000012,0.000057,0.000060,0.000092,0.000021,
0.000053,0.000046,0.000021,0.000073,0.000080,0.000059,0.000168,0.000097,0.000158,0.000033,0.000176,0.000074,0.000018,0.000106,0.000066,0.000138,0.000295,0.000297,0.000481,0.000056,
0.000064,0.000092,0.000107,0.000065,0.000038,0.000010,0.000015,0.000013,0.000038,0.000027,0.000058,0.000015,0.000004,0.000001,0.000010,0.000015,0.000012,0.000009,0.000017,0.000009,
0.000011,0.000016,0.000014,0.000008,0.000013,0.000005,0.000006,0.000001,0.000004,0.000002,0.000002,0.000014,0.000003,0.000013,0.000008,0.000003,0.000001,0.000002,0.000002,0.000009,
0.000012,0.000005,0.000004,0.000019,0.000001,0.000001,0.000009,0.000000,0.000002,0.000001,0.000002,0.000002,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000003,
0.000001,0.000001,0.000000,0.000000,0.000002,0.000000,0.000002,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000003,0.000002,0.000002,
0.000001,0.000001,0.000003,0.000001,0.000003,0.000003,0.000002,0.000001,0.000003,0.000004,0.000005,0.000006,0.000002,0.000002,0.000002,0.000001,0.000003,0.000003,0.000002,0.000001,
0.000002,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000004,0.000005,0.000002,0.000002,0.000001,0.000001,0.000001,0.000002,0.000001,0.000004,0.000003,0.000002,
0.000002,0.000004,0.000004,0.000011,0.000005,0.000002,0.000005,0.000006,0.000001,0.000009,0.000005,0.000006,0.000004,0.000002,0.000015,0.000014,0.000006,0.000006,0.000001,0.000006,
0.000010,0.000005,0.000006,0.000005,0.000003,0.000003,0.000002,0.000004,0.000002,0.000001,0.000001,0.000001,0.000005,0.000002,0.000001,0.000001,0.000001,0.000001,0.000002,0.000001,
0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000020,0.000002,0.000003,0.000001,0.000001,0.000000,0.000001,0.000001,0.000003,0.000002,0.000001,0.000001,0.000013,0.000003,
0.000002,0.000001,0.000002,0.000006,0.000002,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000003,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,
0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000009,0.000001,0.000001,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000001,
0.000002,0.000003,0.000003,0.000004,0.000015,0.000005,0.000002,0.000001,0.000001,0.000003,0.000001,0.000001,0.000006,0.000007,0.000003,0.000006,0.000004,0.000001,0.000001,0.000000,
0.000001,0.000002,0.000001,0.000002,0.000001,0.000002,0.000002,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000002,0.000001,0.000003,0.000001,
0.000002,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000007,0.000007,0.000001,0.000001,0.000000,0.000003,0.000005,0.000007,0.000000,0.000009,0.000001,0.000005,0.000007,
0.000004,0.000000,0.000000,0.000002,0.000001,0.000002,0.000001,0.000001,0.000002,0.000000,0.000000,0.000000,0.000002,0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,0.000002,
0.000002,0.000000,0.000002,0.000001,0.000002,0.000000,0.000000,0.000001,0.000000,0.000000,0.000002,0.000002,0.000002,0.000000,0.000001,0.000001,0.000005,0.000002,0.000004,0.000005,
0.000003,0.000003,0.000000,0.000003,0.000003,0.000004,0.000002,0.000002,0.000001,0.000002,0.000001,0.000001,0.000002,0.000002,0.000002,0.000002,0.000002,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,
0.000001,0.000001,0.000001,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,0.000001,0.000001,0.000000,0.000002,0.000001,0.000002,0.000002,0.000001,
0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,
0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000003,0.000003,0.000001,0.000002,0.000001,0.000002,0.000003,0.000002,0.000000,0.000003,0.000006,
0.000006,0.000004,0.000003,0.000006,0.000003,0.000001,0.000002,0.000001,0.000000,0.000001,0.000001,0.000001,0.000001,0.000000,0.000002,0.000003,0.000004,0.000002,0.000001,0.000001,
0.000001,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000000,0.000002,0.000002,0.000002,0.000002,0.000002,0.000004,0.000003,0.000002,0.000001,0.000003,
0.000004,0.000002,0.000002,0.000002,0.000001,0.000002,0.000001,0.000003,0.000003,0.000002,0.000001,0.000002,0.000001,0.000000,0.000001,0.000001,0.000001,0.000002,0.000001,0.000001,
0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000001,
0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000002,0.000004,0.000004,0.000003,0.000002,0.000002,0.000000,0.000001,0.000001,0.000001,
0.000000,0.000000,0.000001,0.000001,0.000003,0.000002,0.000001,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000002,0.000004,0.000006,0.000001,0.000001,0.000000,
0.000001,0.000001,0.000001,0.000002,0.000001,0.000003,0.000002,0.000002,0.000005,0.000004,0.000001,0.000005,0.000001,0.000001,0.000033,0.000052,0.000006,0.000010,0.000002,0.000028,
0.000053,0.000060,0.000030,0.000015,0.000009,0.000020,0.000006,0.000024,0.000024,0.000020,0.000010,0.000002,0.000008,0.000003,0.000005,0.000011,0.000006,0.000004,0.000001,0.000007,
0.000005,0.000004,0.000002,0.000000,0.000001,0.000002,0.000004,0.000002,0.000003,0.000002,0.000000,0.000004,0.000004,0.000001,0.000003,0.000003,0.000000,0.000001,0.000000,0.000003,
0.000005,0.000004,0.000003,0.000002,0.000002,0.000002,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000001,0.000002,0.000001,0.000000,0.000001,0.000000,
0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000002,0.000001,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000003,0.000000,0.000005,0.000003,0.000002,0.000001,0.000000,0.000000,0.000002,0.000006,0.000000,0.000001,0.000000,0.000000,
0.000000,0.000001,0.000001,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000002,0.000002,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000004,0.000000,0.000000,0.000000,0.000001,0.000002,0.000005,0.000000,0.000001,0.000000,0.000001,0.000002,0.000001,
0.000001,0.000001,0.000000,0.000000,0.000002,0.000000,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000000,0.000001,0.000001,0.000001,0.000002,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000002,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,
0.000000,0.000001,0.000000,0.000001,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000001,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000000,0.000001,0.000002,0.000002,0.000004,0.000003,0.000002,0.000002,0.000001,
0.000003,0.000001,0.000001,0.000004,0.000001,0.000004,0.000008,0.000004,0.000001,0.000002,0.000001,0.000002,0.000003,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,
0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,
0.000000,0.000000,0.000000,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000000,0.000001,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
0.000000,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000000,0.000001,0.000001,
0.000001,0.000001,0.000000,0.000003,0.000002,0.000001,0.000004,0.000002,0.000001,0.000001,0.000002,0.000000,0.000001,0.000002,0.000002,0.000001,0.000000,0.000000,0.000001,0.000001,
0.000003,0.000000,0.000001,0.000000,0.000001,0.000001,0.000001,0.000001,0.000002,0.000010,0.000007,0.000011,0.000002,0.000004,0.000002,0.000002,0.000002,0.000003,0.000002,0.000021,
0.000011,0.000005,0.000006,0.000016,0.000011,0.000001,0.000012,0.000049,0.000007,0.000009,0.000004,0.000012,0.000001,0.000002,0.000004,0.000007,0.000001,0.000002,0.000001,0.000002,
0.000001,0.000001,0.000002,0.000001,0.000002,0.000001,0.000002,0.000000,0.000000,0.000001,0.000002,0.000001,0.000001,0.000001,0.000000,0.000001,0.000001,0.000002,0.000001,0.000001,
0.000001,0.000002,0.000002,0.000003,0.000000,0.000001,0.000000,0.000004,0.000003,0.000004,0.000002,0.000001,0.000000,0.000000,0.000001,0.000002,0.000002,0.000001,0.000001,0.000000,
0.000001,0.000000,0.000000,0.000000,0.000001,0.000002,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000002,0.000000,0.000001,0.000001,
0.000001,0.000001,0.000003,0.000000,0.000001,0.000001,0.000002,0.000001,0.000006,0.000001,0.000002,0.000006,0.000003,0.000002,0.000002,0.000003,0.000001,0.000004,0.000004,0.000011,
0.000013,0.000013,0.000018,0.000004,0.000002,0.000001,0.000004,0.000005,0.000001,0.000001,0.000001,0.000006,0.000003,0.000002,0.000003,0.000001,0.000004,0.000002,0.000001,0.000000,
0.000003,0.000001,0.000002,0.000005,0.000006,0.000002,0.000000,0.000000,0.000000,0.000000,0.000001,0.000001,0.000001,0.000001,0.000001,0.000003,0.000001,0.000001,0.000005,0.000002,
0.000013,0.000001,0.000001,0.000012,0.000021,0.000021,0.000019,0.000018,0.000016,0.000002,0.000007,0.000009,0.000008,0.000003,0.000006,0.000004,0.000023,0.000009,0.000006,0.000000,
0.000002,0.000000,0.000000,0.000001,0.000002,0.000000,0.000001,0.000001,0.000003,0.000004,0.000002,0.000001,0.000001,0.000001,0.000003,0.000001,0.000002,0.000003,0.000001,0.000002,
0.000008,0.000007,0.000005,0.000008,0.000013,0.000005,0.000003,0.000009,0.000009,0.000009,0.000005,0.000004,0.000008,0.000009,0.000007,0.000007,0.000001,0.000005,0.000005,0.000013,
0.000004,0.000002,0.000011,0.000004,0.000004,0.000005,0.000005,0.000001,0.000001,0.000002,0.000000,0.000002,0.000006,0.000004,0.000004,0.000002,0.000002,0.000002,0.000000,0.000001,
0.000000,0.000001,0.000000,0.000001,0.000002,0.000008,0.000009,0.000001,0.000003,0.000002,0.000006,0.000010,0.000007,0.000013,0.000010,0.000010,0.000013,0.000012,0.000007,0.000004,
0.000011,0.000010,0.000014,0.000002,0.000002,0.000013,0.000004,0.000002,0.000007,0.000008,0.000006,0.000005,0.000002,0.000000,0.000004,0.000006,0.000003,0.000010,0.000006,0.000003,
0.000002,0.000003,0.000005,0.000001,0.000003,0.000006,0.000004,0.000003,0.000007,0.000009,0.000010,0.000001,0.000005,0.000002,0.000004,0.000009,0.000019,0.000016,0.000006,0.000013,
0.000009,0.000003,0.000028,0.000060,0.000005,0.000018,0.000036,0.000057,0.000049,0.000080,0.000129,0.000097,0.000039,0.000018,0.000073,0.000049,0.000042,0.000006,0.000040,0.000054,
0.000025,0.000078,0.000062,0.000017,0.000024,0.000014,0.000049,0.000029,0.000004,0.000022,0.000034,0.000004,0.000019,0.000035,0.000040,0.000043,0.000032,0.000032,0.000039,0.000027,
0.000033,0.000064,0.000008,0.000049,0.000199,0.000178,0.000070,0.000097,0.000051,0.000078,0.000038,0.000061,0.000312,0.000244,0.000121,0.000188,0.000090,0.000090,0.000233,0.000058,
0.000108,0.000011,0.000049,0.000133,0.000057,0.000064,0.000078,0.000087,0.000060,0.000100,0.000040,0.000059,0.000032,0.000011,0.000021,0.000009,0.000017,0.000021,0.000078,0.000012,
0.000032,0.000015,0.000039,0.000075,0.000055,0.000055,0.000070,0.000137,0.000135,0.000136,0.000157,0.000078,0.000107,0.000069,0.000188,0.000243,0.000178,0.000202,0.000199,0.000313,
0.000195,0.000882,0.001355,0.000813,0.001042,0.000046,0.000854,0.000527,0.000162,0.000325,0.001882,0.001648,0.001207,0.000247,0.001101,0.000886,0.000476,0.000036,0.000787,0.000800,
0.000112,0.002184,0.002434,0.001119,0.000202,0.002434,0.002455,0.003155,0.002052,0.001702,0.004841,0.001767,0.006626,0.001648,0.005486,0.015129,0.011078,0.010818,0.020096,0.001174,
0.028748,0.007988,0.288086,0.028748,0.098022,0.212280,0.186035,0.023849};

static void run_softmax_bfyx_opt(const int64_t b, const int64_t f, const int64_t y, const int64_t x) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::enable_profiling(true));
    ov::intel_gpu::ImplementationDesc softmax_bf_kernel = {format::bfyx, "softmax_gpu_bf"};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"softmax", softmax_bf_kernel}}));

    const int64_t buf_size = b * f * y * x;
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), f, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                        data_types::f16, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f16, format::bfyx};

    std::string softmax_id = "softmax";
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax(softmax_id, input_info("input"), 3));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    auto input_mem = engine.allocate_memory(input_layout_static);

    set_values(input_mem, input_data);

    std::map<cldnn::primitive_id, cldnn::network_output> outputs;
    cldnn::memory::ptr output = nullptr;

    std::vector<int64_t> time_records;
    const size_t num_tests = 1;
    for (size_t i = 0; i < num_tests; i++) {
        network->set_input_data("input", input_mem);
        outputs = network->execute();
        output = outputs.at(softmax_id).get_memory();
        ASSERT_NE(output, nullptr);

        auto executed_primitives = network->get_executed_primitives();
        ASSERT_NE(executed_primitives.find(softmax_id), executed_primitives.end());
        auto ev = executed_primitives[softmax_id];
        if (ev != nullptr) {
            auto intervals = ev->get_profiling_info();
            for (const auto &interval : intervals) {
                if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                    time_records.push_back(std::chrono::duration_cast<std::chrono::microseconds>(interval.value->value()).count());
                }
            }
        }
    }

    if (num_tests > 1) {
        auto output_layout = output->get_layout();
        auto sum = std::accumulate(time_records.begin(), time_records.end(), 0);
        auto max = *std::max_element(time_records.begin(), time_records.end());
        auto min = *std::min_element(time_records.begin(), time_records.end());
        auto avg = (static_cast<float>(sum - max - min) / (time_records.size() - 2)) / 1000.f;
        auto min_ms =  (static_cast<float>(min) / 1000.f);
        auto max_ms =  (static_cast<float>(max) / 1000.f);
        std::cout << "latency[kernel : " << network->get_primitive(softmax_id)->get_implementation_name() << "]"
                    << "[num:" << std::setfill('0') << std::setw(3) << time_records.size() << "]"
                    << "[output: " << output_layout.to_short_string() << "] avg: "
                    << std::setfill(' ') << std::setw(8) << avg << " ms, max: " << max_ms
                    << " ms, min: " << min_ms << " ms, min io throughput : "
                    << (static_cast<float>(output_layout.count() * 2) / min / 1e3f) << std::endl;
    } else {
        std::cout << "latency[kernel : " << network->get_primitive(softmax_id)->get_implementation_name() << "] "
                    << (static_cast<float>(time_records.front()) / 1000.f) << " ms " << std::endl;
    }

    ASSERT_NE(output, nullptr);
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());
    size_t not_matched = 0;
    for (size_t idx = 0; idx < static_cast<size_t>(buf_size); idx++) {

        if (std::fabs(static_cast<float>(output_ptr[idx]) - static_cast<float>(output_ref[idx])) > 0.000005f) {
            std::cout << "Checking " << std::fixed << setprecision(8) << output_ptr[idx] << " vs " << output_ref[idx] << std::endl;
            not_matched++;
        }
    }
    std::cout << "not matched: " << not_matched << ", pass_rate: " << (static_cast<float>(buf_size - not_matched) * 100 / buf_size) << std::endl;
    ASSERT_EQ(not_matched, 0);
}

TEST(softmax_gpu_bfyx_f16, opt_softmax_bf) {
    run_softmax_bfyx_opt(1, 2, 2, 3083);
}