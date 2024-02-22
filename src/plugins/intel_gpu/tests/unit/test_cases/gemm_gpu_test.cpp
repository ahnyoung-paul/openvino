// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/transpose.hpp"

#include "intel_gpu/runtime/compilation_context.hpp"
#include "gemm_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace  {

const std::vector<cldnn::format> f_blocked_4d_formats = {
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
};

const std::vector<cldnn::format> f_blocked_5d_formats = {
    format::b_fs_zyx_fsv16,
    format::b_fs_zyx_fsv32,
};

const std::vector<cldnn::format> b_blocked_4d_formats = {
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
};

const std::vector<cldnn::format> b_blocked_5d_formats = {
    format::bs_fs_zyx_bsv16_fsv32,
    format::bs_fs_zyx_bsv16_fsv16,
    format::bs_fs_zyx_bsv32_fsv32,
    format::bs_fs_zyx_bsv32_fsv16,
};

// TODO: uncomment in scope of CVS-85940
const std::vector<cldnn::format> planar_formats = {
    format::bfyx,
    /*
    format::bfzyx,
    format::bfwzyx,
     */
};


const std::vector<data_types> float_types = {
    data_types::f16, data_types::f32 };

const std::vector<data_types> all_types = {
    data_types::f16, data_types::f32 , data_types::i8, data_types::u8, data_types::i32
};

typedef std::tuple<
std::vector<std::vector<int32_t>>,
std::vector<std::vector<float>>,
format,
data_types,
std::vector<float>,
bool,
bool,
float,
float
>
GemmParams;

class GemmGPUTest : public ::testing::TestWithParam<GemmParams> {
protected:
    std::vector<std::vector<float>> input_data;
    std::vector<float> out_data;
    std::vector<std::vector<int32_t>> shapes;
    format fmt{format::bfyx};
    data_types type;
    bool transpose_input0;
    bool transpose_input1;
    float alpha;
    float beta;

    virtual void fill_gemm_params() {
        GemmParams params = testing::TestWithParam<GemmParams>::GetParam();
        std::tie(shapes, input_data, fmt, type, out_data, transpose_input0,
                 transpose_input1, alpha, beta) = params;
    }

    virtual void process_program(program::ptr) {
    }

public:
    virtual ~GemmGPUTest() {}
    void test(bool is_caching_test = false) {

        fill_gemm_params();

        topology tp;

        std::vector<std::pair<primitive_id, memory_ptr>> network_inputs;
        std::vector<input_info> gemm_inputs;

        auto &engine = get_test_engine();
        for (size_t i = 0; i < shapes.size(); ++i) {
            tensor t{shapes[i]};
            layout l{data_types::f32, format::bfyx, t};
            auto input = engine.allocate_memory(l);
            set_values(input, input_data[i]);
            primitive_id prim_id = std::string("input") + std::to_string(i);
            network_inputs.emplace_back(prim_id, input);

            primitive_id prim_id_reordered = prim_id + "_reordered";
            // tp.add(data(prim_id, input));
            tp.add(input_layout(prim_id, input->get_layout()));
            tp.add(reorder(prim_id_reordered, prim_id, fmt, type));
            gemm_inputs.push_back(input_info(prim_id_reordered));
        }

        auto g = gemm("gemm_output", gemm_inputs, type, transpose_input0, transpose_input1, alpha, beta);
        tp.add(g);
        tp.add(reorder("output", input_info("gemm_output"), format::bfyx, data_types::f32));

        cldnn::network::ptr network;
        if (is_caching_test) {
            membuf mem_buf;
            {
                std::ostream out_mem(&mem_buf);
                BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
                ob.set_stream(get_test_stream_ptr().get());
                program::build_program(engine, tp, get_test_default_config(engine))->save(ob);
            }
            {
                std::istream in_mem(&mem_buf);
                BinaryInputBuffer ib = BinaryInputBuffer(in_mem, engine);
                auto imported_prog = std::make_shared<cldnn::program>(engine, get_test_default_config(engine));
                imported_prog->load(ib);
                network = std::make_shared<cldnn::network>(imported_prog);
            }
        } else {
            network = std::make_shared<cldnn::network>(engine, tp, get_test_default_config(engine));
        }
        process_program(network->get_program());

        for (auto &input : network_inputs) {
            network->set_input_data(input.first, input.second);
        }
        auto outputs = network->execute();
        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), out_data.size());
        const auto abs_error = type == data_types::f16 ? 0.1 : 0.0001;
        for (uint32_t i = 0; i < out_data.size(); ++i) {
            ASSERT_NEAR(output_ptr[i], out_data[i], abs_error);
        }
    }
};

class GemmGPUTestRandom : public GemmGPUTest {

    ov::Shape input0_shape;
    ov::Shape input1_shape;
    ov::Shape output_shape;

    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void generated_inputs() {
        for (size_t i = 0; i < shapes.size(); ++i) {
            size_t size = ov::shape_size(shapes[i]);
            input_data[i] = rg.generate_random_1d<float>(size, -1, 1, 10);
        }
    }

    void process_program(program::ptr p) override {
        std::vector<program_node*>& prog_nodes = p->get_outputs();
        auto inputs = p->get_inputs();
        auto input_it = inputs.begin();
        input0_shape = (*input_it)->get_output_layout().get_shape();
        ++input_it;
        input1_shape = (*input_it)->get_output_layout().get_shape();
        layout output_layout = prog_nodes[0]->get_output_layout();
        output_shape = output_layout.get_shape();
        out_data.resize(ov::shape_size(output_shape));
        calculate_output_data();
    }

    void calculate_output_data() {
        ov::reference::matmul<float>(input_data[0].data(),
                                     input_data[1].data(),
                                     out_data.data(),
                                     input0_shape,
                                     input1_shape,
                                     output_shape,
                                     transpose_input0,
                                     transpose_input1);
    }

protected:

    void fill_gemm_params() override {
        GemmGPUTest::fill_gemm_params();
        // this class support only simple gemm case: 2 inputs, alpha eq 1.f and beta eq 0.f
        ASSERT_THAT(input_data.size(), 2ul);
        ASSERT_THAT(alpha, 1.f);
        ASSERT_THAT(beta, 0.f);
        generated_inputs();
    }

};

TEST_P(GemmGPUTest, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(GemmGPUTestRandom, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(
    GemmGPUTest_basic_t1, GemmGPUTestRandom,
    ::testing::Combine(
        ::testing::Values(std::vector<std::vector<int32_t>>{{1, 1, 3, 4},
                                                            {1, 1, 1, 4}}),
        ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
        ::testing::ValuesIn(planar_formats), ::testing::ValuesIn(float_types),
        ::testing::Values(std::vector<float>{}),
        ::testing::Values(true), ::testing::Values(false),
        ::testing::Values(1.0f), ::testing::Values(0.0f)));

INSTANTIATE_TEST_SUITE_P(
    GemmGPUTest_basic_t2, GemmGPUTestRandom,
    ::testing::Combine(
        ::testing::Values(std::vector<std::vector<int32_t>>{{1, 1, 4, 3},
                                                            {1, 1, 4, 1}}),
        ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
        ::testing::ValuesIn(planar_formats), ::testing::ValuesIn(float_types),
        ::testing::Values(std::vector<float>{}),
        ::testing::Values(false), ::testing::Values(true),
        ::testing::Values(1.0f), ::testing::Values(0.0f)));

class gemm_gpu_tests: public ::testing::Test {
public:
    void test_basic_bfyx_t2_inplace_crop_with_pad(bool is_caching_test) {
        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 3 } });
        auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 1 } });

        std::vector<float> input_data = {
            1.f, -2.f,  3.f, -4.f,
            5.f,  6.f, 1.f, 2.f,
            3.f, 3.f, 2.f, -1.f,

            1.f, -2.f,  3.f, -4.f,
            5.f,  6.f, 1.f, 2.f,
            3.f, 3.f, 2.f, -1.f,
        };

        std::vector<float> input_data2 = {
            2.f, 5.f, -4.f, -7.f,
        };
        set_values(input, input_data);
        set_values(input2, input_data2);

        std::vector<float> out_data = {
            8.f, 22.f, 20.f
        };

        topology topology;
        topology.add(
            input_layout("input", input->get_layout())
        );
        topology.add(
            input_layout("input2", input2->get_layout())
        );
        topology.add(
            crop("crop.1", input_info("input"), { 1, 1, 4, 3 }, { 0, 1, 0, 0 })
        );
        topology.add(
            gemm("output", { input_info("crop.1"), input_info("input2") }, data_types::f32, false, true)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        network->set_input_data("input2", input2);
        auto outputs = network->execute();

        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), (uint32_t)3);
        for (uint32_t i = 0; i < out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
        }
    }

    void test_dynamic(bool is_caching_test) {
        auto& engine = get_test_engine();
        ov::Shape in1_shape = { 1, 1, 3, 4 };
        ov::Shape in2_shape = { 1, 4 };
        auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
        auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
        auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx});
        auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx});

        std::vector<float> input1_data = {
            1.f, -2.f, 3.f, -4.f,
            5.f, 6.f, 1.f, 2.f,
            3.f, 3.f, 2.f, -1.f,
        };

        std::vector<float> input2_data = {
            2.f, 5.f, -4.f, -7.f,
        };
        set_values(input1, input1_data);
        set_values(input2, input2_data);

        std::vector<float> out_data = {
            8.f, 22.f, 20.f
        };

        topology topology;
        topology.add(input_layout("input1", in1_layout),
                    input_layout("input2", in2_layout),
                    gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, true, 1.0f, 0.0f, 4, 2)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input1);
        network->set_input_data("input2", input2);

        auto inst = network->get_primitive("gemm");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network->execute();

        auto output = outputs.at("gemm").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), (uint32_t)3);
        for (uint32_t i = 0; i < out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
        }
    }

    void test_dynamic_padding(bool is_caching_test) {
        tests::random_generator rg;
        rg.set_seed(GET_SUITE_NAME);

        auto& engine = get_test_engine();

        const unsigned long BATCH_SIZE = 31;
        const unsigned long M_SIZE = 11;
        const unsigned long K_SIZE = 37;
        const unsigned long N_SIZE = 49;

        auto fill_mem = [&](cldnn::memory_ptr mem, std::vector<ov::float16>& data) {
            cldnn::mem_lock<ov::float16> mem_ptr(mem, get_test_stream());
            auto&& l = mem->get_layout();
            auto data_idx = 0;
            for (cldnn::tensor::value_type b = 0; b < l.batch(); ++b) {
                for (cldnn::tensor::value_type f = 0; f < l.feature(); ++f) {
                    for (cldnn::tensor::value_type y = 0; y < l.spatial(1); ++y) {
                        for (cldnn::tensor::value_type x = 0; x < l.spatial(0); ++x) {
                            auto tensor_coord = cldnn::tensor{{b, f, x, y}, 0};
                            auto buffer_idx = l.get_linear_offset(tensor_coord);
                            mem_ptr[buffer_idx] = data[data_idx++];
                        }
                    }
                }
            }
        };

        const auto align_size_m = 13;
        const auto align_size_k = 16;
        const auto align_size_n = 15;
        const auto align_size_b1 = 3;
        const auto align_size_b2 = 19;

        const auto aligned_batch1_size = align_to(1ul, align_size_b1);
        auto padding_size_batch1 = static_cast<int>(aligned_batch1_size - 1);

        const auto aligned_batch2_size = align_to(BATCH_SIZE, align_size_b2);
        auto padding_size_batch2 = static_cast<int>(aligned_batch2_size - BATCH_SIZE);

        const auto aligned_m_size = align_to(M_SIZE, align_size_m);
        auto padding_size_m = static_cast<int>(aligned_m_size - M_SIZE);
        const auto aligned_k_size = align_to(K_SIZE, align_size_k);
        auto padding_size_k = static_cast<int>(aligned_k_size - K_SIZE);
        const auto aligned_n_size = align_to(N_SIZE, align_size_n);
        auto padding_size_n = static_cast<int>(aligned_n_size - N_SIZE);

        ov::Shape in1_shape = { 1, BATCH_SIZE, M_SIZE, K_SIZE };
        ov::Shape in2_shape = { 1, BATCH_SIZE, K_SIZE, N_SIZE };
        ov::Shape in1_shape_aligned = { aligned_batch1_size, aligned_batch2_size, aligned_m_size, aligned_k_size };
        ov::Shape in2_shape_aligned = { aligned_batch1_size, aligned_batch2_size, aligned_k_size, aligned_n_size };

        // Use dynamic padding for all BFYX dimensions
        tensor dyn_pad_dims_input({1, 1, 1, 1}, 0);

        auto in1_layout = layout{ {-1, -1, -1, -1}, data_types::f16, format::bfyx, padding({0, 0, 0, 0}, {0, 0, 0, 0}, 0.0f, dyn_pad_dims_input)};
        auto in2_layout = layout{ {-1, -1, -1, -1}, data_types::f16, format::bfyx, padding({0, 0, 0, 0}, {0, 0, 0, 0}, 0.0f, dyn_pad_dims_input)};

        auto aligned_input1_mem = engine.allocate_memory({ov::PartialShape(in1_shape_aligned), data_types::f16, format::bfyx});
        auto aligned_input2_mem = engine.allocate_memory({ov::PartialShape(in2_shape_aligned), data_types::f16, format::bfyx});

        auto input1_mem = engine.reinterpret_buffer(*aligned_input1_mem, layout{ov::PartialShape(in1_shape),
                                                                                data_types::f16,
                                                                                format::bfyx,
                                                                                padding({padding_size_batch1, 0, 0, 0},
                                                                                        {0, padding_size_batch2, padding_size_k, padding_size_m}, 0.0f, dyn_pad_dims_input)});

        auto input2_mem = engine.reinterpret_buffer(*aligned_input2_mem, layout{ov::PartialShape(in2_shape),
                                                                                data_types::f16,
                                                                                format::bfyx,
                                                                                padding({0, padding_size_batch2, 0, 0},
                                                                                        {padding_size_batch1, 0, padding_size_n, padding_size_k}, 0.0f, dyn_pad_dims_input)});

        auto input_1_data = rg.generate_random_1d<ov::float16>(ov::shape_size(in1_shape), -2, 2);
        auto input_2_data = rg.generate_random_1d<ov::float16>(ov::shape_size(in2_shape), -2, 2);

        fill_mem(input1_mem, input_1_data);
        fill_mem(input2_mem, input_2_data);

        auto get_ref_results = [&]() {
            ov::Shape in1_shape = { 1, BATCH_SIZE, M_SIZE, K_SIZE };
            ov::Shape in2_shape = { 1, BATCH_SIZE, K_SIZE, N_SIZE };
            auto in1_layout = layout{ {-1, -1, -1, -1}, data_types::f16, format::bfyx};
            auto in2_layout = layout{ {-1, -1, -1, -1}, data_types::f16, format::bfyx};

            auto input1_mem = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f16, format::bfyx});
            auto input2_mem = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f16, format::bfyx});

            fill_mem(input1_mem, input_1_data);
            fill_mem(input2_mem, input_2_data);

            topology topology;
            topology.add(input_layout("input1", in1_layout),
                        input_layout("input2", in2_layout),
                        gemm("gemm_ref", { input_info("input1"), input_info("input2") }, data_types::f16, false, false, 1.0f, 0.0f, 4, 4)
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::optimize_data(true));
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

            network network(engine, topology, config);
            network.set_input_data("input1", input1_mem);
            network.set_input_data("input2", input2_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "gemm_ref");

            auto inst = network.get_primitive("gemm_ref");

            auto output_mem = outputs.at("gemm_ref").get_memory();
            auto output_layout = outputs.at("gemm_ref").get_layout();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology;
        topology.add(input_layout("input1", in1_layout),
                     input_layout("input2", in2_layout),
                     gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f16, false, false, 1.0f, 0.0f, 4, 4)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input1_mem);
        network->set_input_data("input2", input2_mem);

        auto inst = network->get_primitive("gemm");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network->execute();

        auto output_mem = outputs.at("gemm").get_memory();
        auto output_layout = outputs.at("gemm").get_layout();

        auto res = engine.reinterpret_buffer(*output_mem, output_layout);

        auto ref_res = get_ref_results();

        mem_lock<ov::float16> res_lock(res, get_test_stream());
        mem_lock<ov::float16> res_ref_lock(ref_res, get_test_stream());
        for (size_t i = 0; i < res->count(); i++) {
            ASSERT_EQ(res_lock[i], res_ref_lock[i]) << i;
        }
    }

    void test_dynamic_multi_inference_same_shape(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto in1_dyn_layout = layout{ ov::PartialShape{ 1, 1, ov::Dimension(1, 10), 4 }, data_types::f32, format::bfyx };
        auto in1_actual_layout = layout{ ov::PartialShape{ 1, 1, 3, 4 }, data_types::f32, format::bfyx };
        auto in2_dyn_layout = layout{ ov::PartialShape{ 4, ov::Dimension(1, 10) }, data_types::f32, format::bfyx };
        auto in2_actual_layout = layout{ ov::PartialShape{ 4, 1 }, data_types::f32, format::bfyx };
        auto input1_1 = engine.allocate_memory(in1_actual_layout);
        auto input1_2 = engine.allocate_memory(in1_actual_layout);
        auto input2_1 = engine.allocate_memory(in2_actual_layout);
        auto input2_2 = engine.allocate_memory(in2_actual_layout);

        std::vector<float> input1_data1 = {
            1.f, -2.f, 3.f, -4.f,
            5.f, 6.f, 1.f, 2.f,
            3.f, 3.f, 2.f, -1.f,
        };
        std::vector<float> input1_data2 = {
            -1.f, 2.f, -3.f, 4.f,
            5.f, 6.f, -1.f, 2.f,
            3.f, -3.f, 2.f, 1.f,
        };
        std::vector<float> input2_data1 = {
            2.f, 5.f, -4.f, -7.f,
        };
        std::vector<float> input2_data2 = {
            4.f, 7.f, 2.f, 5.f,
        };
        set_values(input1_1, input1_data1);
        set_values(input1_2, input1_data2);
        set_values(input2_1, input2_data1);
        set_values(input2_2, input2_data2);

        std::vector<float> out_data1 = {
            8.f, 22.f, 20.f
        };
        std::vector<float> out_data2 = {
            24.f, 70.f, 0.f
        };

        topology topology;
        topology.add(input_layout("input1", in1_dyn_layout),
                    input_layout("input2", in2_dyn_layout),
                    gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, false, 1.0f, 0.0f, 4, 2)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        {
            network->set_input_data("input1", input1_1);
            network->set_input_data("input2", input2_1);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "gemm");

            auto prog = network->get_program();
            auto& node = prog->get_node("gemm");
            auto impl = node.get_selected_impl();
            ASSERT_TRUE(impl != nullptr);
            ASSERT_TRUE(impl->is_dynamic());

            auto output_prim_mem = outputs.begin()->second.get_memory();
            cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

            ASSERT_EQ(output_ptr.size(), (uint32_t)3);
            for (uint32_t i = 0; i < out_data1.size(); ++i) {
                ASSERT_FLOAT_EQ(output_ptr[i], out_data1[i]);
            }
        }

        {
            network->set_input_data("input1", input1_2);
            network->set_input_data("input2", input2_2);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "gemm");

            auto output_prim_mem = outputs.begin()->second.get_memory();
            cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

            ASSERT_EQ(output_ptr.size(), (uint32_t)3);
            for (uint32_t i = 0; i < out_data2.size(); ++i) {
                ASSERT_FLOAT_EQ(output_ptr[i], out_data2[i]);
            }
        }
    }

    void test_dynamic_multi_inference_different_shape(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto in1_dyn_layout = layout{ ov::PartialShape{ 1, 1, ov::Dimension(1, 10), 4 }, data_types::f32, format::bfyx };
        auto in1_actual_layout1 = layout{ ov::PartialShape{ 1, 1, 3, 4 }, data_types::f32, format::bfyx };
        auto in1_actual_layout2 = layout{ ov::PartialShape{ 1, 1, 4, 4 }, data_types::f32, format::bfyx };
        auto in2_dyn_layout = layout{ ov::PartialShape{ 4, ov::Dimension(1, 10) }, data_types::f32, format::bfyx };
        auto in2_actual_layout = layout{ ov::PartialShape{ 4, 1 }, data_types::f32, format::bfyx };
        auto input1_1 = engine.allocate_memory(in1_actual_layout1);
        auto input1_2 = engine.allocate_memory(in1_actual_layout2);
        auto input2 = engine.allocate_memory(in2_actual_layout);

        std::vector<float> input1_data1 = {
            1.f, -2.f, 3.f, -4.f,
            5.f, 6.f, 1.f, 2.f,
            3.f, 3.f, 2.f, -1.f,
        };
        std::vector<float> input1_data2 = {
            -1.f, 2.f, -3.f, 4.f,
            5.f, 6.f, -1.f, 2.f,
            3.f, -3.f, 2.f, 1.f,
            1.f, 2.f, -5.f, 6.f,
        };
        std::vector<float> input2_data = {
            2.f, 5.f, -4.f, -7.f,
        };
        set_values(input1_1, input1_data1);
        set_values(input1_2, input1_data2);
        set_values(input2, input2_data);

        std::vector<float> out_data1 = {
            8.f, 22.f, 20.f
        };
        std::vector<float> out_data2 = {
            -8.f, 30.f, -24.f, -10.f
        };

        topology topology;
        topology.add(input_layout("input1", in1_dyn_layout),
                    input_layout("input2", in2_dyn_layout),
                    gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, false, 1.0f, 0.0f, 4, 2)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        {
            network->set_input_data("input1", input1_1);
            network->set_input_data("input2", input2);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "gemm");

            auto prog = network->get_program();
            auto& node = prog->get_node("gemm");
            auto impl = node.get_selected_impl();
            ASSERT_TRUE(impl != nullptr);
            ASSERT_TRUE(impl->is_dynamic());

            auto output_prim_mem = outputs.begin()->second.get_memory();
            cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

            ASSERT_EQ(output_ptr.size(), (uint32_t)3);
            for (uint32_t i = 0; i < out_data1.size(); ++i) {
                ASSERT_FLOAT_EQ(output_ptr[i], out_data1[i]);
            }
        }

        {
            network->set_input_data("input1", input1_2);
            network->set_input_data("input2", input2);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "gemm");

            auto output_prim_mem = outputs.begin()->second.get_memory();
            cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

            ASSERT_EQ(output_ptr.size(), (uint32_t)4);
            for (uint32_t i = 0; i < out_data2.size(); ++i) {
                ASSERT_FLOAT_EQ(output_ptr[i], out_data2[i]);
            }
        }
    }

    void test_transpose_matmul(size_t num_dims, bool is_input_dynamic, bool is_caching_test) {
        tests::random_generator rg;
        rg.set_seed(GET_SUITE_NAME);

        const unsigned long BATCH_SIZE = 19;
        const unsigned long M_SIZE = 37;
        const unsigned long K_SIZE = 23;
        const unsigned long N_SIZE = 29;

        auto fill_mem = [&](cldnn::memory_ptr mem, std::vector<float>& data) {
            cldnn::mem_lock<float> mem_ptr(mem, get_test_stream());
            auto&& l = mem->get_layout();
            auto data_idx = 0;
            for (cldnn::tensor::value_type b = 0; b < l.batch(); ++b) {
                for (cldnn::tensor::value_type f = 0; f < l.feature(); ++f) {
                    for (cldnn::tensor::value_type y = 0; y < l.spatial(1); ++y) {
                        for (cldnn::tensor::value_type x = 0; x < l.spatial(0); ++x) {
                            auto tensor_coord = cldnn::tensor{{b, f, x, y}, 0};
                            auto buffer_idx = l.get_linear_offset(tensor_coord);
                            mem_ptr[buffer_idx] = data[data_idx++];
                        }
                    }
                }
            }
        };

        auto& engine = get_test_engine();
        ov::Shape input0_shape;
        ov::Shape input1_shape;
        std::vector<int64_t> input0_order;
        std::vector<int64_t> input1_order;
        cldnn::layout input0_layout;
        cldnn::layout input1_layout;

        if (num_dims == 1) {
            input0_shape = { K_SIZE };
            input1_shape = { N_SIZE, K_SIZE };
            input0_order = { 0 };
            input1_order = { 1, 0 };
        } else if (num_dims == 2) {
            input0_shape = { K_SIZE, M_SIZE };
            input1_shape = { N_SIZE, K_SIZE };
            input0_order = { 1, 0 };
            input1_order = { 1, 0 };
        } else if (num_dims == 3) {
            input0_shape = { BATCH_SIZE, K_SIZE, M_SIZE };
            input1_shape = { N_SIZE, BATCH_SIZE, K_SIZE };
            input0_order = { 0, 2, 1 };
            input1_order = { 1, 2, 0 };
        } else if (num_dims == 4) {
            input0_shape = { BATCH_SIZE, K_SIZE, 1, M_SIZE };
            input1_shape = { N_SIZE, BATCH_SIZE, 1, K_SIZE };
            input0_order = { 0, 2, 3, 1 };
            input1_order = { 1, 2, 3, 0 };
        }

        if (is_input_dynamic) {
            input0_layout = layout{ov::PartialShape::dynamic(input0_shape.size()), data_types::f32, format::bfyx};
            input1_layout = layout{ov::PartialShape::dynamic(input1_shape.size()), data_types::f32, format::bfyx};
        } else {
            input0_layout = layout{ov::PartialShape(input0_shape), data_types::f32, format::bfyx};
            input1_layout = layout{ov::PartialShape(input1_shape), data_types::f32, format::bfyx};
        }

        auto input0_mem = engine.allocate_memory(layout{ov::PartialShape(input0_shape), data_types::f32, format::bfyx});
        auto input1_mem = engine.allocate_memory(layout{ov::PartialShape(input1_shape), data_types::f32, format::bfyx});

        auto input_0_data = rg.generate_random_1d<float>(ov::shape_size(input0_shape), -2, 2);
        auto input_1_data = rg.generate_random_1d<float>(ov::shape_size(input1_shape), -2, 2);

        fill_mem(input0_mem, input_0_data);
        fill_mem(input1_mem, input_1_data);

        topology topology;
        topology.add(input_layout("input0", input0_layout),
                     input_layout("input1", input1_layout),
                     gemm("gemm", { input_info("input0"), input_info("input1") }, data_types::f32, input0_order, input1_order)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input0", input0_mem);
        network->set_input_data("input1", input1_mem);

        auto inst = network->get_primitive("gemm");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic() == is_input_dynamic);

        auto outputs = network->execute();

        auto output_mem = outputs.at("gemm").get_memory();
        cldnn::mem_lock<float> output_ptr(output_mem, get_test_stream());

        ov::Shape ref_input0_shape;
        ov::Shape ref_input1_shape;
        ov::Shape ref_output_shape;
        if (num_dims == 1) {
            ref_input0_shape = { K_SIZE };
            ref_input1_shape = { K_SIZE, N_SIZE };
            ref_output_shape = { 1, N_SIZE };
        } else if (num_dims == 2) {
            ref_input0_shape = { M_SIZE, K_SIZE };
            ref_input1_shape = { K_SIZE, N_SIZE };
            ref_output_shape = { M_SIZE, N_SIZE };
        } else if (num_dims == 3) {
            ref_input0_shape = { BATCH_SIZE, M_SIZE, K_SIZE };
            ref_input1_shape = { BATCH_SIZE, K_SIZE, N_SIZE };
            ref_output_shape = { BATCH_SIZE, M_SIZE, N_SIZE };
        } else if (num_dims == 4) {
            ref_input0_shape = { BATCH_SIZE, 1, M_SIZE, K_SIZE };
            ref_input1_shape = { BATCH_SIZE, 1, K_SIZE, N_SIZE };
            ref_output_shape = { BATCH_SIZE, 1, M_SIZE, N_SIZE };
        }

        std::vector<float> ref_out_data;
        ref_out_data.resize(ov::shape_size(ref_output_shape));

        std::vector<float> ref_input_0_data(input_0_data.size());
        std::vector<float> ref_input_1_data(input_1_data.size());

        ov::reference::transpose((const char *)(input_0_data.data()),
                                 (char *)(ref_input_0_data.data()),
                                 input0_shape,
                                 sizeof(float),
                                 input0_order,
                                 ref_input0_shape);

        ov::reference::transpose((const char *)(input_1_data.data()),
                                 (char *)(ref_input_1_data.data()),
                                 input1_shape,
                                 sizeof(float),
                                 input1_order,
                                 ref_input1_shape);

        ov::reference::matmul<float>(ref_input_0_data.data(),
                                     ref_input_1_data.data(),
                                     ref_out_data.data(),
                                     ref_input0_shape,
                                     ref_input1_shape,
                                     ref_output_shape,
                                     false,
                                     false);

        ASSERT_EQ(output_ptr.size(), ref_out_data.size());

        const auto abs_error = 0.0001;
        for (uint32_t i = 0; i < ref_out_data.size(); ++i) {
            ASSERT_NEAR(output_ptr[i], ref_out_data[i], abs_error);
        }
    }

    void test_transpose_matmul_transpose(size_t num_dims, bool is_input_dynamic, bool is_caching_test) {
        tests::random_generator rg;
        rg.set_seed(GET_SUITE_NAME);

        const unsigned long BATCH_SIZE = 19;
        const unsigned long M_SIZE = 17;
        const unsigned long K_SIZE = 22;
        const unsigned long N_SIZE = 32;

        auto fill_mem = [&](cldnn::memory_ptr mem, std::vector<ov::float16>& data) {
            cldnn::mem_lock<ov::float16> mem_ptr(mem, get_test_stream());
            auto&& l = mem->get_layout();
            auto data_idx = 0;
            for (cldnn::tensor::value_type b = 0; b < l.batch(); ++b) {
                for (cldnn::tensor::value_type f = 0; f < l.feature(); ++f) {
                    for (cldnn::tensor::value_type y = 0; y < l.spatial(1); ++y) {
                        for (cldnn::tensor::value_type x = 0; x < l.spatial(0); ++x) {
                            auto tensor_coord = cldnn::tensor{{b, f, x, y}, 0};
                            auto buffer_idx = l.get_linear_offset(tensor_coord);
                            mem_ptr[buffer_idx] = data[data_idx++];
                        }
                    }
                }
            }
        };

        auto& engine = get_test_engine();
        ov::Shape input0_shape;
        ov::Shape input1_shape;
        std::vector<int64_t> input0_order;
        std::vector<int64_t> input1_order;
        std::vector<int64_t> output_order;
        cldnn::layout input0_layout;
        cldnn::layout input1_layout;

        if (num_dims == 1) {
            input0_shape = { K_SIZE };
            input1_shape = { N_SIZE, K_SIZE };
            input0_order = { 0 };
            input1_order = { 1, 0 };
            output_order = { 0 };
        } else if (num_dims == 2) {
            input0_shape = { K_SIZE, M_SIZE };
            input1_shape = { N_SIZE, K_SIZE };
            input0_order = { 1, 0 };
            input1_order = { 1, 0 };
            output_order = { 1, 0 };
        } else if (num_dims == 3) {
            input0_shape = { BATCH_SIZE, K_SIZE, M_SIZE };
            input1_shape = { N_SIZE, BATCH_SIZE, K_SIZE };
            input0_order = { 0, 2, 1 };
            input1_order = { 1, 2, 0 };
            output_order = { 1, 0, 2 };
        } else if (num_dims == 4) {
            input0_shape = { M_SIZE, K_SIZE, 1, BATCH_SIZE };
            input1_shape = { N_SIZE, 1, BATCH_SIZE, K_SIZE };
            input0_order = {3, 2, 0, 1};
            input1_order = {2, 1, 3, 0};
            output_order = {1, 0, 3, 2};
        }

        if (is_input_dynamic) {
            input0_layout = layout{ov::PartialShape::dynamic(input0_shape.size()), data_types::f16, format::bfyx};
            input1_layout = layout{ov::PartialShape::dynamic(input1_shape.size()), data_types::f16, format::bfyx};
        } else {
            input0_layout = layout{ov::PartialShape(input0_shape), data_types::f16, format::bfyx};
            input1_layout = layout{ov::PartialShape(input1_shape), data_types::f16, format::bfyx};
        }

        auto input0_mem = engine.allocate_memory(layout{ov::PartialShape(input0_shape), data_types::f16, format::bfyx});
        auto input1_mem = engine.allocate_memory(layout{ov::PartialShape(input1_shape), data_types::f16, format::bfyx});

        auto input_0_data = rg.generate_random_1d<ov::float16>(ov::shape_size(input0_shape), -2, 2);
        auto input_1_data = rg.generate_random_1d<ov::float16>(ov::shape_size(input1_shape), -2, 2);

        fill_mem(input0_mem, input_0_data);
        fill_mem(input1_mem, input_1_data);

        topology topology;
        topology.add(input_layout("input0", input0_layout),
                     input_layout("input1", input1_layout),
                     gemm("gemm", { input_info("input0"), input_info("input1") }, data_types::f16, input0_order, input1_order, output_order)
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input0", input0_mem);
        network->set_input_data("input1", input1_mem);

        auto inst = network->get_primitive("gemm");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic() == is_input_dynamic);

        auto outputs = network->execute();

        auto output_mem = outputs.at("gemm").get_memory();
        cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

        ov::Shape ref_input0_shape;
        ov::Shape ref_input1_shape;
        ov::Shape ref_output_shape;
        ov::Shape transposed_output_shape;
        if (num_dims == 1) {
            ref_input0_shape = { K_SIZE };
            ref_input1_shape = { K_SIZE, N_SIZE };
            ref_output_shape = { 1, N_SIZE };
            transposed_output_shape = { N_SIZE, 1 };
        } else if (num_dims == 2) {
            ref_input0_shape = { M_SIZE, K_SIZE };
            ref_input1_shape = { K_SIZE, N_SIZE };
            ref_output_shape = { M_SIZE, N_SIZE };
            transposed_output_shape = { N_SIZE, M_SIZE };
        } else if (num_dims == 3) {
            ref_input0_shape = { BATCH_SIZE, M_SIZE, K_SIZE };
            ref_input1_shape = { BATCH_SIZE, K_SIZE, N_SIZE };
            ref_output_shape = { BATCH_SIZE, M_SIZE, N_SIZE };
            transposed_output_shape = { M_SIZE, BATCH_SIZE, N_SIZE };
        } else if (num_dims == 4) {
            ref_input0_shape = { BATCH_SIZE, 1, M_SIZE, K_SIZE };
            ref_input1_shape = { BATCH_SIZE, 1, K_SIZE, N_SIZE };
            ref_output_shape = { BATCH_SIZE, 1, M_SIZE, N_SIZE };
            transposed_output_shape = { 1, BATCH_SIZE, N_SIZE, M_SIZE };
        }

        std::vector<ov::float16> ref_out_data;
        ref_out_data.resize(ov::shape_size(ref_output_shape));
        std::vector<ov::float16> transposed_out_data;
        transposed_out_data.resize(ov::shape_size(ref_output_shape));

        std::vector<ov::float16> ref_input_0_data(input_0_data.size());
        std::vector<ov::float16> ref_input_1_data(input_1_data.size());

        ov::reference::transpose((const char *)(input_0_data.data()),
                                 (char *)(ref_input_0_data.data()),
                                 input0_shape,
                                 sizeof(ov::float16),
                                 input0_order,
                                 ref_input0_shape);

        ov::reference::transpose((const char *)(input_1_data.data()),
                                 (char *)(ref_input_1_data.data()),
                                 input1_shape,
                                 sizeof(ov::float16),
                                 input1_order,
                                 ref_input1_shape);

        ov::reference::matmul<ov::float16>(ref_input_0_data.data(),
                                           ref_input_1_data.data(),
                                           ref_out_data.data(),
                                           ref_input0_shape,
                                           ref_input1_shape,
                                           ref_output_shape,
                                           false,
                                           false);

        ov::reference::transpose((const char *)(ref_out_data.data()),
                                 (char *)(transposed_out_data.data()),
                                 ref_output_shape,
                                 sizeof(ov::float16),
                                 output_order,
                                 transposed_output_shape);

        ASSERT_EQ(output_ptr.size(), transposed_out_data.size());

        const auto abs_error = 0.0001;
        for (uint32_t i = 0; i < transposed_out_data.size(); ++i) {
            ASSERT_NEAR(output_ptr[i], transposed_out_data[i], abs_error);
        }
    }
};

TEST_F(gemm_gpu_tests, basic_bfyx_t2_inplace_crop_with_pad) {
    this->test_basic_bfyx_t2_inplace_crop_with_pad(false);
}

TEST_F(gemm_gpu_tests, dynamic) {
    this->test_dynamic(false);
}

TEST_F(gemm_gpu_tests, dynamic_padding) {
    this->test_dynamic_padding(false);
}

TEST_F(gemm_gpu_tests, dynamic_multi_inference_same_shape) {
    this->test_dynamic_multi_inference_same_shape(false);
}

TEST_F(gemm_gpu_tests, dynamic_multi_inference_different_shape) {
    this->test_dynamic_multi_inference_different_shape(false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_dynamic_1d) {
    this->test_transpose_matmul(1, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_static_1d) {
    this->test_transpose_matmul(1, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_dynamic_2d) {
    this->test_transpose_matmul(2, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_static_2d) {
    this->test_transpose_matmul(2, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_dynamic_3d) {
    this->test_transpose_matmul(3, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_static_3d) {
    this->test_transpose_matmul(3, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_dynamic_4d) {
    this->test_transpose_matmul(4, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_static_4d) {
    this->test_transpose_matmul(4, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_dynamic_1d) {
    this->test_transpose_matmul_transpose(1, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_static_1d) {
    this->test_transpose_matmul_transpose(1, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_dynamic_2d) {
    this->test_transpose_matmul_transpose(2, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_static_2d) {
    this->test_transpose_matmul_transpose(2, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_dynamic_3d) {
    this->test_transpose_matmul_transpose(3, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_static_3d) {
    this->test_transpose_matmul_transpose(3, false, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_dynamic_4d) {
    this->test_transpose_matmul_transpose(4, true, false);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_static_4d) {
    this->test_transpose_matmul_transpose(4, false, false);
}

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_t1t2,
        GemmGPUTestRandom,
        ::testing::Combine(
            ::testing::Values(std::vector<std::vector<int32_t>>{{2, 1, 3, 4}, {2, 1, 4, 1}}),
            ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
            ::testing::ValuesIn(planar_formats),
            ::testing::ValuesIn(float_types),
            ::testing::Values(std::vector<float>{}),
            ::testing::Values(true),
            ::testing::Values(true),
            ::testing::Values(1.0f),
            ::testing::Values(0.0f)
            )
        );

INSTANTIATE_TEST_SUITE_P(
    GemmGPUTest_basic_input3, GemmGPUTest,
    ::testing::Combine(
        ::testing::Values(std::vector<std::vector<int32_t>>{
            {1, 1, 3, 2}, {1, 1, 2, 3}, {1, 1, 2, 2}}),
        ::testing::Values(std::vector<std::vector<float>>{
            {1.0f, 2.0f, 3.0f, 1.0f, 0.0f, 1.0f},
            {
                3.0f,
                3.0f,
                1.0f,
                2.0f,
                1.0f,
                2.0f,
            },
            {
                1.0f,
                0.0f,
                2.0f,
                0.0f,
            }}),
        ::testing::ValuesIn(planar_formats), ::testing::ValuesIn(all_types),
        ::testing::Values(std::vector<float>{26.0f, 26.0f, 28.0f, 10.0f}),
        ::testing::Values(false), ::testing::Values(false),
        ::testing::Values(2.0f), ::testing::Values(10.0f)));

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_t1t2,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 4, 3 }, { 1, 1, 3, 2 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f
                        },
                        {
                            3.0f, 3.0f, 1.0f,
                            2.0f, 1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(true),
                    ::testing::Values(true),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_1,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 3, 4 }, { 1, 1, 2, 3 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 1.0f, 0.0f,
                            2.0f, 0.0f, 0.0f,
                            3.0f, 1.0f, 0.0f,
                            4.0f, 0.0f, 0.0f

                        },
                        {
                            3.0f, 2.0f,
                            3.0f, 1.0f,
                            1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_t2,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 3, 4 }, { 1, 1, 3, 2 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 1.0f, 0.0f,
                            2.0f, 0.0f, 0.0f,
                            3.0f, 1.0f, 0.0f,
                            4.0f, 0.0f, 0.0f

                        },
                        {
                            3.0f, 3.0f, 1.0f,
                            2.0f, 1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(false),
                    ::testing::Values(true),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_t1,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 4, 3 }, { 1, 1, 2, 3 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f
                        },
                        {
                            3.0f, 2.0f,
                            3.0f, 1.0f,
                            1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(true),
                    ::testing::Values(false),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_basic,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 2, 1, 4, 3 }, { 2, 1, 1, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);


INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_basic3_bfyx,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 5, 1, 500, 9 }, { 5, 1, 1, 500 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_basic_smarcink2,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 2, 1, 3, 2 }, { 2, 1, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_f_block_4d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 32, 3, 2 }, { 1, 32, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(f_blocked_4d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_b_block_4d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 32, 1, 3, 2 }, { 32, 1, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(b_blocked_4d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);
// TODO: enable in scope of CVS-85940
INSTANTIATE_TEST_SUITE_P(
        DISABLED_GemmGPUTest_f_block_5d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 16, 2, 3, 2 }, { 1, 16, 2, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(f_blocked_5d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_GemmGPUTest_b_block_5d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 16, 1, 2, 3, 2 }, { 16, 1, 2, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(b_blocked_5d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

struct gemm_base_test_params {
    size_t m_size;
    size_t n_size;
    size_t k_size;
    size_t b0_num;
    size_t f0_num;
    size_t b1_num;
    size_t f1_num;
    size_t b2_num;
    size_t f2_num;
    size_t b_out_num;
    size_t f_out_num;
    bool transpose_input0;
    bool transpose_input1;
    float alpha;
    float beta;
    cldnn::data_types allocate0_type;
    cldnn::data_types allocate1_type;
    cldnn::data_types allocate2_type;
    cldnn::data_types output_type;
    std::vector <int> range0;
    std::vector <int> range1;
    std::vector <int> range2;
    std::string kernel_name;
};

#ifdef ENABLE_ONEDNN_FOR_GPU

#define CASE_GEMM_INT8_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_ONEDNN_2 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_ONEDNN_3 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_ONEDNN_4 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_ONEDNN_1 1, 64, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_ONEDNN_2 64, 1, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_ONEDNN_3 1, 1, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_ONEDNN_4 64, 64, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_ONEDNN_2 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_ONEDNN_3 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_ONEDNN_4 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_ONEDNN_2 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_ONEDNN_3 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_ONEDNN_4 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_NN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_NT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_NN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_NT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_NN_TRANSPOSITION_ONEDNN 32, 16, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_NT_TRANSPOSITION_ONEDNN 16, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TN_TRANSPOSITION_ONEDNN 32, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TT_TRANSPOSITION_ONEDNN 32, 64, 96, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_NN_TRANSPOSITION_ONEDNN 32, 16, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_NT_TRANSPOSITION_ONEDNN 16, 64, 128, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TN_TRANSPOSITION_ONEDNN 32, 64, 96, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TT_TRANSPOSITION_ONEDNN 32, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_1 32, 32, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_3 64, 32, 64, 1, 2, 1, 1, 1, 2, 1, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_4 32, 64, 64, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_1 32, 32, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_3 64, 32, 64, 1, 2, 1, 1, 1, 2, 1, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_4 32, 64, 64, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_1 32, 32, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_3 64, 32, 64, 1, 2, 1, 1, 1, 2, 1, 2, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_4 32, 64, 64, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_COMBO_ONEDNN_1 5, 18, 99, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_ONEDNN_2 1, 32, 65, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_ONEDNN_3 13, 4, 64, 1, 2, 1, 1, 1, 2, 1, 2, true, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_ONEDNN_4 128, 126, 127, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_COMBO_ONEDNN_1 11, 16, 65, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_COMBO_ONEDNN_2 13, 14, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_COMBO_ONEDNN_3 16, 16, 99, 1, 2, 1, 2, 1, 2, 1, 2, true, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_COMBO_ONEDNN_4 3, 1, 77, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

// Currently broadcasting support wasn't implemented for f16 cases with biases
#define CASE_GEMM_FP16_COMBO_ONEDNN_1 5, 7, 65, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_COMBO_ONEDNN_2 32, 8, 128, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_COMBO_ONEDNN_3 14, 2, 69, 1, 2, 1, 1, 1, 2, 1, 2, true, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_COMBO_ONEDNN_4 1, 1, 64, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_COMBO_ONEDNN_1 7, 17, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_COMBO_ONEDNN_2 26, 22, 79, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_COMBO_ONEDNN_3 5, 7, 81, 1, 2, 1, 1, 1, 2, 1, 2, true, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_COMBO_ONEDNN_4 61, 1, 99, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#endif  // ENABLE_ONEDNN_FOR_GPU

#define CASE_GEMM_INT8_NN_TRANSPOSITION 64, 64, 64, 1, 2, 1, 2, 1, 2, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_NT_TRANSPOSITION 32, 64, 32, 2, 1, 2, 1, 2, 1, 2, 1, false, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TN_TRANSPOSITION 128, 64, 32, 2, 2, 2, 2, 2, 2, 2, 2, true, false, \
1.0f, 0.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TT_TRANSPOSITION 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.2f, 0.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_BROADCAST_1 32, 32, 32, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCAST_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCAST_3 64, 32, 32, 1, 2, 2, 1, 1, 2, 2, 2, false, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCAST_4 32, 64, 32, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.2f, 0.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_LEFTOVERS_1 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_2 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.6f, 1.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_3 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_4 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_5 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_6 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.6f, 1.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_7 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_8 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_9 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_10 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.6f, 1.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_11 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_12 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_COMBO_1 8, 8, 32, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_2 16, 16, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.7f, 0.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_3 11, 31, 21, 7, 15, 7, 15, 7, 15, 7, 15, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_4 32, 32, 32, 3, 6, 3, 6, 3, 6, 3, 6, true, true, \
1.2f, 4.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_SLM_COMBO_1 64, 64, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_SLM_COMBO_2 384, 384, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 0.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_SLM_COMBO_3 128, 128, 64, 2, 3, 2, 3, 2, 3, 2, 3, false, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_SLM_COMBO_4 256, 64, 64, 3, 6, 3, 6, 3, 6, 3, 6, false, false, \
1.2f, 4.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_NN_1 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_2 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_3 31, 47, 65, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_4 65, 31, 47, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_NT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_TN_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TN_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TN_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TN_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_TT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_NN_BROADCAST_1 64, 96, 32, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_BROADCAST_2 32, 16, 16, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_BROADCAST_3 5, 1, 3, 1, 2, 2, 1, 1, 2, 2, 2, false, false, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_BROADCAST_4 64, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_TILED_NN_1 64, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_2 128, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_3 131, 17, 15, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_4 33, 17, 17, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_NT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_TN_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TN_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TN_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TN_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_TT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_NN_BROADCAST_1 64, 96, 128, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_BROADCAST_2 64, 16, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_BROADCAST_3 1, 2, 3, 1, 2, 2, 1, 1, 2, 2, 2, false, false, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_BROADCAST_4 8, 8, 8, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

template <typename gemm_params, typename input0_type, typename input1_type, typename input2_type, typename output_type, typename accumulator_type>
class GemmBaseTest : public ::testing::TestWithParam<gemm_params> {
public:
    virtual ov::intel_gpu::ImplementationDesc getImplementationDesc(gemm_params& p) {
         return { format::bfyx, p.kernel_name };
    }

    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    inline size_t getGemmIndex(size_t x, size_t y, size_t f, size_t b, size_t x_size, size_t y_size, size_t f_num, size_t b_num,
                               size_t x_pitch, size_t y_pitch, size_t f_pitch, size_t b_pitch) {
        return (x % x_size) * x_pitch + (y % y_size) * y_pitch + (f % f_num) * f_pitch + (b % b_num) * b_pitch;
    }

    void execute(gemm_params& p, bool is_caching_test = false) {
        auto y0_size = p.m_size;
        auto y0_pitch = p.k_size;
        auto x0_size = p.k_size;
        auto x0_pitch = 1;
        auto f0_pitch = y0_size * x0_size;
        auto b0_pitch = p.f0_num * f0_pitch;

        auto y1_size = p.k_size;
        auto y1_pitch = p.n_size;
        auto x1_size = p.n_size;
        auto x1_pitch = 1;
        auto f1_pitch = y1_size * x1_size;
        auto b1_pitch = p.f1_num * f1_pitch;

        auto y2_size = p.m_size;
        auto y2_pitch = p.n_size;
        auto x2_size = p.n_size;
        auto x2_pitch = 1;
        auto f2_pitch = y2_size * x2_size;
        auto b2_pitch = p.f2_num * f2_pitch;

        auto y_out_size = p.m_size;
        auto y_out_pitch = p.n_size;
        auto x_out_size = p.n_size;
        auto x_out_pitch = 1;
        auto f_out_pitch = y_out_size * x_out_size;
        auto b_out_pitch = p.f_out_num * f_out_pitch;

        if (p.transpose_input0) {
            y0_size = p.k_size;
            y0_pitch = p.m_size;
            x0_size = p.m_size;
            x0_pitch = 1;
        }

        if (p.transpose_input1) {
            y1_size = p.n_size;
            y1_pitch = p.k_size;
            x1_size = p.k_size;
            x1_pitch = 1;
        }

        auto& engine = get_test_engine();
        auto input0_size = tensor((int)p.b0_num, (int)p.f0_num, (int)x0_size, (int)y0_size);
        VVVVF<input0_type> input0_data = rg.generate_random_4d<input0_type>(p.b0_num, p.f0_num, x0_size, y0_size, p.range0[0], p.range0[1], p.range0[2]);
        auto input0_data_bfyx = flatten_4d(format::bfyx, input0_data);
        auto input0_mem = engine.allocate_memory({ p.allocate0_type, format::bfyx, input0_size });
        set_values(input0_mem, input0_data_bfyx);

        auto input1_size = tensor((int)p.b1_num, (int)p.f1_num, (int)x1_size, (int)y1_size);
        VVVVF<input1_type> input1_data = rg.generate_random_4d<input1_type>(p.b1_num, p.f1_num, x1_size, y1_size, p.range1[0], p.range1[1], p.range1[2]);
        auto input1_data_bfyx = flatten_4d(format::bfyx, input1_data);
        auto input1_mem = engine.allocate_memory({ p.allocate1_type, format::bfyx, input1_size });
        set_values(input1_mem, input1_data_bfyx);

        auto input2_size = tensor((int)p.b2_num, (int)p.f2_num, (int)x2_size, (int)y2_size);
        VVVVF<input2_type> input2_data = rg.generate_random_4d<input2_type>(p.b2_num, p.f2_num, x2_size, y2_size, p.range2[0], p.range2[1], p.range2[2]);
        auto input2_data_bfyx = flatten_4d(format::bfyx, input2_data);
        auto input2_mem = engine.allocate_memory({ p.allocate2_type, format::bfyx, input2_size });
        set_values(input2_mem, input2_data_bfyx);

        std::vector<output_type> out_data(p.b_out_num * p.f_out_num * p.m_size * p.n_size);

        for (size_t b = 0; b < p.b_out_num; ++b) {
            for (size_t f = 0; f < p.f_out_num; ++f) {
                for (size_t y = 0; y < p.m_size; ++y) {
                    for (size_t x = 0; x < p.n_size; ++x) {
                        size_t input2_data_index = getGemmIndex(x, y, f, b, x2_size, y2_size, p.f2_num, p.b2_num, x2_pitch, y2_pitch, f2_pitch, b2_pitch);
                        size_t out_data_index = getGemmIndex(x, y, f, b, x_out_size, y_out_size, p.f_out_num, p.b_out_num,
                                                             x_out_pitch, y_out_pitch, f_out_pitch, b_out_pitch);
                        accumulator_type acc = 0;

                        for (size_t k = 0; k < p.k_size; ++k) {
                            size_t input0_data_index = getGemmIndex(k * (!p.transpose_input0) + y * p.transpose_input0, y * (!p.transpose_input0) +
                            k * p.transpose_input0, f, b, x0_size, y0_size, p.f0_num, p.b0_num, x0_pitch, y0_pitch, f0_pitch, b0_pitch);
                            size_t input1_data_index = getGemmIndex(x * (!p.transpose_input1) + k * p.transpose_input1, k * (!p.transpose_input1) +
                            x * p.transpose_input1, f, b, x1_size, y1_size, p.f1_num, p.b1_num, x1_pitch, y1_pitch, f1_pitch, b1_pitch);
                            acc += (accumulator_type)input0_data_bfyx[input0_data_index] * (accumulator_type)input1_data_bfyx[input1_data_index];
                        }

                        out_data[out_data_index] = (output_type)acc;
                        out_data[out_data_index] *= (output_type)p.alpha;
                        if (p.beta)
                            out_data[out_data_index] += (output_type)p.beta * (output_type)input2_data_bfyx[input2_data_index];
                    }
                }
            }
        }

        topology topology;
        topology.add(input_layout("input0", input0_mem->get_layout()));
        topology.add(input_layout("input1", input1_mem->get_layout()));
        if (p.beta != 0) {
            topology.add(input_layout("input2", input2_mem->get_layout()));
            topology.add(gemm("gemm_bfyx", { input_info("input0"), input_info("input1"), input_info("input2") }, p.output_type, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
        } else {
            topology.add(gemm("gemm_bfyx", { input_info("input0"), input_info("input1") }, p.output_type, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
        }
        topology.add(reorder("reorder_bfyx", input_info("gemm_bfyx"), format::bfyx, data_types::f32));

        ov::intel_gpu::ImplementationDesc gemm_impl = getImplementationDesc(p);

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_bfyx", gemm_impl} }));

        cldnn::network::ptr network = get_network(engine, topology, cfg, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input0", input0_mem);
        network->set_input_data("input1", input1_mem);
        if (p.beta != 0) {
            network->set_input_data("input2", input2_mem);
        }
        auto outputs = network->execute();
        auto output = outputs.at("reorder_bfyx").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        const float threshold_int8 = 1.f;
        const float threshold_fp16 = 1e-1;
        const float threshold_fp32 = 3e-4;

        ASSERT_EQ(output_ptr.size(), (size_t)(p.b_out_num * p.f_out_num * p.m_size * p.n_size));
        if (sizeof(input0_type) == 1) {
            for (size_t i = 0; i < out_data.size(); ++i) {
                ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_int8) << "index = " << i;
            }
        } else if (sizeof(input0_type) == 2) {
            for (size_t i = 0; i < out_data.size(); ++i) {
                ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_fp16) << "index = " << i;
            }
        } else {
            for (size_t i = 0; i < out_data.size(); ++i) {
                ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_fp32) << "index = " << i;
            }
        }
    }
};

#ifdef ENABLE_ONEDNN_FOR_GPU
struct gemm_onednn_test_params {
    std::vector<tensor> in_shapes;
    tensor out_shape;
    tensor kernel;
    tensor pad;
    data_types data_type_in0;
    data_types data_type_in1;
    data_types data_type_in2;
    format input_format;
    data_types default_type;
    format default_format;
};

template <typename T>
class GemmOneDNNTest : public ::testing::TestWithParam<T> {
public:
    tests::random_generator rg;
    cldnn::engine& engine = get_test_engine();
    topology topology_ocl;
    topology topology_onednn;

    ExecutionConfig config_ocl;
    ExecutionConfig config_onednn;

    float tolerance = 0.0f;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
        config_ocl.set_property(ov::intel_gpu::optimize_data(true));
        config_ocl.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        if (engine.get_device_info().supports_immad) {
            config_onednn.set_property(ov::intel_gpu::optimize_data(true));
            config_onednn.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        }
    }

    void execute(T& p) {
        auto input0_prim = get_generated_random_1d_mem(engine, get_input_layout(p, 0));
        auto input1_prim = get_generated_random_1d_mem(engine, get_input_layout(p, 1));

        network network_ocl(engine, topology_ocl, config_ocl);
        network network_onednn(engine, topology_onednn, config_onednn);

        network_ocl.set_input_data("input0", input0_prim);
        network_ocl.set_input_data("input1", input1_prim);
        network_onednn.set_input_data("input0", input0_prim);
        network_onednn.set_input_data("input1", input1_prim);

        compare(network_ocl, network_onednn, p);
    }

    void compare(network& network_ocl, network& network_onednn, T& p) {
        auto outputs_ocl = network_ocl.execute();
        auto outputs_onednn = network_onednn.execute();

        ASSERT_EQ(outputs_ocl.size(), outputs_onednn.size());
        ASSERT_EQ(outputs_ocl.size(), size_t(1));

        auto val_ocl = get_output_values_to_float(network_ocl, outputs_ocl.begin()->second);
        auto val_onednn = get_output_values_to_float(network_onednn, outputs_onednn.begin()->second);

        ASSERT_EQ(val_ocl.size(), val_onednn.size());

        for (size_t i = 0; i < val_ocl.size(); i++) {
            ASSERT_NEAR(val_ocl[i], val_onednn[i], tolerance)
                << "tolerance = " << tolerance
                << "\ni = " << i
                << "\nocl[i] = " << val_ocl[i]
                << "\nonednn[i] = " << val_onednn[i];
        }
    }

    layout get_input_layout(T& p, int in_no) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        if (in_no == 0)
            return layout{ p.data_type_in0, p.input_format, p.in_shapes.at(0), padding{ pad_ } };
        else if (in_no == 1)
            return layout{ p.data_type_in1, p.input_format, p.in_shapes.at(1), padding{ pad_ } };
        else
            return layout{ p.data_type_in2, p.input_format, p.in_shapes.at(2), padding{ pad_ } };
    }

    cldnn::memory::ptr get_generated_random_1d_mem(cldnn::engine& engine, cldnn::layout l) {
        auto prim = engine.allocate_memory(l);
        cldnn::tensor s = l.get_tensor();
        if (l.data_type == cldnn::data_types::i8 || l.data_type == cldnn::data_types::u8) {
            VF<uint8_t> rnd_vec = rg.generate_random_1d<uint8_t>(s.count(), -200, 200);
            set_values(prim, rnd_vec);
        } else if (l.data_type == cldnn::data_types::f16) {
            VF<ov::float16> rnd_vec = rg.generate_random_1d<ov::float16>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = rg.generate_random_1d<float>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        }

        return prim;
    }
};

class gemm_onednn_ndims : public GemmOneDNNTest<gemm_onednn_test_params> {};
TEST_P(gemm_onednn_ndims, basic) {
    if (!engine.get_device_info().supports_immad)
        return;

    auto p = GetParam();

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    topology_ocl.add(input_layout("input0", in_layout0));
    topology_ocl.add(input_layout("input1", in_layout1));
    topology_ocl.add(gemm("gemm0_ocl", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()));
    topology_ocl.add(reorder("reorder0", input_info("gemm0_ocl"), p.default_format, data_types::f32));

    topology_onednn.add(input_layout("input0", get_input_layout(p, 0)));
    topology_onednn.add(input_layout("input1", get_input_layout(p, 1)));
    topology_onednn.add(gemm("gemm0_onednn", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()));
    topology_onednn.add(reorder("reorder0", input_info("gemm0_onednn"), p.default_format, data_types::f32));

    ov::intel_gpu::ImplementationDesc gemm_impl_ocl = { p.default_format, "", impl_types::ocl };
    config_ocl.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm0_ocl", gemm_impl_ocl } }));

    ov::intel_gpu::ImplementationDesc gemm_impl_onednn = { p.default_format, "", impl_types::onednn };
    config_onednn.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm0_onednn", gemm_impl_onednn } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}
#define CASE_GEMM_ONEDNN_FP16_4D { { 2, 3, 2, 2 }, { 2, 3, 2, 2 } }, { 2, 3, 2, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ONEDNN_FP16_5D { { 1, 3, 4, 4, 4 }, { 1, 3, 4, 4, 4 } }, { 1, 3, 4, 4, 4 }, tensor{ 1 }, tensor{ 0 }, \
data_types::f16, data_types::f16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GEMM_ONEDNN_FP16_6D { { 2, 3, 5, 4, 3, 2 }, { 2, 3, 4, 5, 3, 2 } }, { 2, 3, 5, 5, 3, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::f16, data_types::f16, data_types::f16, format::bfwzyx, data_types::f16, format::bfwzyx
#define CASE_GEMM_ONEDNN_I8_4D { { 2, 3, 2, 2 }, { 2, 3, 2, 2 } }, { 2, 3, 2, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::i8, format::bfyx
#define CASE_GEMM_ONEDNN_I8_5D { { 1, 3, 4, 4, 4 }, { 1, 3, 4, 4, 4 } }, { 1, 3, 4, 4, 4 }, tensor{ 1 }, tensor{ 0 }, \
data_types::i8, data_types::i8, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx
#define CASE_GEMM_ONEDNN_I8_6D { { 2, 3, 5, 4, 3, 2 }, { 2, 3, 4, 5, 3, 2 } }, { 2, 3, 5, 5, 3, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::i8, data_types::i8, data_types::i8, format::bfwzyx, data_types::i8, format::bfwzyx

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_onednn_ndims, ::testing::ValuesIn(std::vector<gemm_onednn_test_params>{
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_FP16_4D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_FP16_5D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_FP16_6D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_I8_4D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_I8_5D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_I8_6D },
}));

TEST(gemm_onednn, impl_replacement_with_cldnn) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    ov::Shape in1_shape = { 1, 1, 3, 4 };
    ov::Shape in2_shape = { 1, 4 };
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx});
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx});

    std::vector<float> input1_data = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };

    std::vector<float> input2_data = {
        2.f, 5.f, -4.f, -7.f,
    };
    set_values(input1, input1_data);
    set_values(input2, input2_data);

    std::vector<float> out_data = {
        8.f, 22.f, 20.f
    };

    topology topology;
    topology.add(input_layout("input1", in1_layout),
                 input_layout("input2", in2_layout),
                 gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, true, 1.0f, 0.0f, 4, 2)
    );

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
    ExecutionConfig cfg{ ov::intel_gpu::queue_type(QueueTypes::in_order),
                         ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm", fc_impl} }),
                         ov::intel_gpu::optimize_data(true),
                         ov::intel_gpu::allow_new_shape_infer(true) };

    network network(engine, topology, cfg);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("gemm");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gemm").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)3);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
    }

    // WA: Call wait_all() to wait for all queued kernels compilation finish
    network.get_program()->get_compilation_context().wait_all();

    // Check if OneDNN's impl is used for the next execute() call
    network.execute();
    inst = network.get_primitive("gemm");
    impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_FALSE(impl->is_dynamic());
}

template <typename gemm_params, typename input0_type, typename input1_type, typename input2_type, typename output_type, typename accumulator_type>
class GemmBaseOneDNNTest : public ::GemmBaseTest<gemm_params, input0_type, input1_type, input2_type, output_type, accumulator_type> {
public:
    virtual ov::intel_gpu::ImplementationDesc getImplementationDesc(gemm_params& p) {
        return { format::bfyx, "", impl_types::onednn };
    }

    void execute(gemm_params& p, bool is_caching_test = false) {
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        GemmBaseTest<gemm_params, input0_type, input1_type, input2_type, output_type, accumulator_type>::execute(p, is_caching_test);
    }
};

class gemm_int8_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_4, "" },
}));

class gemm_uint8_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, uint8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_uint8_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_uint8_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_4, "" },
}));

class gemm_fp16_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_4, "" },
}));

class gemm_fp32_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_4, "" },
}));

class gemm_int8_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TT_TRANSPOSITION_ONEDNN, "" },

    gemm_base_test_params{ CASE_GEMM_INT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
}));

class gemm_uint8_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, uint8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_uint8_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_uint8_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_UINT8_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TT_TRANSPOSITION_ONEDNN, "" },

    gemm_base_test_params{ CASE_GEMM_UINT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
}));

class gemm_fp16_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_TT_TRANSPOSITION_ONEDNN, "" },
}));

class gemm_fp32_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_TT_TRANSPOSITION_ONEDNN, "" },
}));

class gemm_int8_broadcasting_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_broadcasting_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_broadcasting_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_4, "" },
}));

class gemm_fp16_broadcasting_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_broadcasting_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_broadcasting_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_4, "" },
}));

class gemm_fp32_broadcasting_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, int32_t> {};
TEST_P(gemm_fp32_broadcasting_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_broadcasting_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_4, "" },
}));

class gemm_int8_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_4, "" },
}));

class gemm_uint8_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, uint8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_uint8_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_uint8_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_4, "" },
}));

class gemm_fp16_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_4, "" },
}));

class gemm_fp32_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_4, "" },
}));

#endif // ENABLE_ONEDNN_FOR_GPU

class gemm_int8_transposition_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_transposition_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_transposition_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_NN_TRANSPOSITION, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_NT_TRANSPOSITION, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_TN_TRANSPOSITION, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_TT_TRANSPOSITION, "gemm_mmad_int8" },
}));

class gemm_int8_broadcast_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_broadcast_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_broadcast_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_1, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_2, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_3, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_4, "gemm_mmad_int8" },
}));

class gemm_int8_leftovers_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_leftovers_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_leftovers_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_1, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_2, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_3, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_4, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_5, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_6, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_7, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_8, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_9, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_10, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_11, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_12, "gemm_mmad_int8" },
}));

class gemm_int8_combo_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_combo_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_combo_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_1, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_2, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_3, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_4, "gemm_mmad_int8" },
}));

class gemm_int8_slm_combo_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_slm_combo_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_slm_combo_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_1, "gemm_mmad_int8_slm" },
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_2, "gemm_mmad_int8_slm" },
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_3, "gemm_mmad_int8_slm" },
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_4, "gemm_mmad_int8_slm" },
}));

class gemm_fp32_tiled_nn_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_nn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_nn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_nt_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_nt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_nt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_tn_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_tn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_tn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_tt_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_tt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_tt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_nn_broadcast_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_nn_broadcast_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_nn_broadcast_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_nn_tests : public ::GemmBaseTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_tiled_nn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_nn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_nt_tests : public ::GemmBaseTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_tiled_nt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_nt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_tn_tests : public ::GemmBaseTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_tiled_tn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_tn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_tt_tests : public ::GemmBaseTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_tiled_tt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_tt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_nn_broadcast_tests : public ::GemmBaseTest<gemm_base_test_params, ov::float16, ov::float16, ov::float16, ov::float16, ov::float16> {};
TEST_P(gemm_fp16_tiled_nn_broadcast_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_nn_broadcast_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_4, "gemm_tiled_opt" },
}));

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(GemmGPUTest, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(GemmGPUTestRandom, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST_P(gemm_int8_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_uint8_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_uint8_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_broadcasting_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_broadcasting_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_broadcasting_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_uint8_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
#endif // ENABLE_ONEDNN_FOR_GPU

TEST_P(gemm_int8_transposition_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_broadcast_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_leftovers_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_combo_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_slm_combo_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_nn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_nt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_tn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_tt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_nn_broadcast_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_nn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_nt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_tn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_tt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_nn_broadcast_tests, basic_cached) { auto p = GetParam(); execute(p); }

TEST_F(gemm_gpu_tests, dynamic_cached) {
    this->test_dynamic(true);
}

TEST_F(gemm_gpu_tests, dynamic_multi_inference_same_shape_cached) {
    this->test_dynamic_multi_inference_same_shape(true);
}

TEST_F(gemm_gpu_tests, dynamic_multi_inference_different_shape_cached) {
    this->test_dynamic_multi_inference_different_shape(true);
}
#endif // RUN_ALL_MODEL_CACHING_TESTS

TEST_F(gemm_gpu_tests, basic_bfyx_t2_inplace_crop_with_pad_cached) {
    this->test_basic_bfyx_t2_inplace_crop_with_pad(true);
}

TEST_F(gemm_gpu_tests, transpose_matmul_dynamic_4d_cached) {
    this->test_transpose_matmul(4, true, true);
}

TEST_F(gemm_gpu_tests, transpose_matmul_transpose_dynamic_4d_cached) {
    this->test_transpose_matmul_transpose(4, true, true);
}


std::vector<float> input1_data1={
-0.204631f,-0.121582f,-1.892062f,-0.041277f,-0.674334f,0.534690f,-1.252063f,0.453808f,0.423076f,0.726113f,-0.912417f,-0.241269f,2.005219f,-0.136383f,-0.295427f,-0.522324f,1.252073f,-0.029389f,0.068339f,1.085615f,
-0.924208f,0.071915f,0.195635f,0.224738f,1.131527f,-0.195937f,0.009962f,-1.357137f,0.632310f,0.113157f,-0.181886f,0.295696f,-0.455646f,0.402452f,-0.877722f,-0.472235f,0.412148f,1.327806f,1.285834f,-0.385866f,
-0.074691f,0.298028f,-0.790274f,-0.662098f,0.096340f,-0.379695f,0.752443f,0.721383f,-0.788055f,-0.511133f,-0.328075f,-0.619234f,0.886190f,0.258011f,-1.678177f,-0.143632f,0.666455f,-0.143924f,0.933182f,-1.676170f,
0.341168f,-0.410647f,-0.325941f,-0.108051f,0.355150f,0.943707f,-0.120721f,-1.538862f,-0.905325f,0.406958f,-0.048290f,-1.142317f,-0.798710f,-1.868605f,0.968584f,-0.440271f,0.972596f,-0.913654f,-0.705945f,0.115010f,
-0.847259f,-1.537169f,-0.875255f,-2.858124f,3.731263f,-1.902563f,1.314405f,0.859289f,-0.249838f,1.517014f,-0.901951f,0.390541f,-1.156970f,-0.570347f,-0.405358f,-1.800451f,2.600536f,0.404824f,1.367401f,0.319975f,
-0.035515f,-1.028374f,-0.037465f,0.531297f,-0.982694f,1.488958f,0.633100f,0.602502f,0.039319f,-1.158374f,-0.661321f,0.019868f,1.079783f,1.195598f,-0.533965f,-0.126539f,0.685000f,-0.939331f,-0.484355f,1.407340f,
0.708869f,0.726644f,1.822406f,-0.948087f,0.352710f,-0.819566f,1.712037f,-1.349522f,-0.352949f,-1.871281f,0.016442f,-0.336664f,0.301883f,2.984765f,1.318438f,-1.610883f,-0.558480f,-0.256588f,2.072463f,-1.285055f,
-1.042503f,0.251528f,-1.239464f,-0.202283f,-0.565635f,-0.031900f,-0.552119f,-1.570521f,2.383812f,-2.035691f,2.397705f,-0.124004f,-0.045654f,0.888523f,0.972283f,1.224618f,-0.726720f,-1.721287f,-1.346529f,-0.912096f,
0.888486f,0.699885f,1.384940f,-1.809804f,0.733760f,-2.254208f,-0.484082f,1.690980f,-0.400365f,-0.202531f,0.837283f,-0.203283f,0.499031f,0.636948f,-1.062837f,-0.279775f,1.799950f,0.520717f,-0.002183f,-1.362916f,
0.006636f,-1.855627f,-0.542210f,0.747300f,0.869437f,1.882588f,2.828919f,1.043533f,2.940565f,-0.022565f,-0.462118f,-0.246070f,-0.567894f,-0.078971f,-1.111095f,-0.851947f,-1.053061f,1.631609f,1.084648f,-0.401269f,
-0.509984f,-1.276055f,0.361804f,0.690590f,-2.159471f,0.375799f,-0.606834f,1.068587f,-0.907706f,-1.894222f,0.759531f,-1.574776f,2.872773f,-1.938751f,2.394896f,0.983135f,0.269507f,0.424329f,0.974536f,0.247430f,
-0.623000f,-0.522890f,-1.954751f,-0.693193f,0.871726f,-0.184196f,0.891654f,0.562723f,-0.217110f,-0.415152f,0.625060f,-0.193400f,-2.099214f,-0.382353f,2.199698f,-0.148989f,0.341407f,-0.873971f,-1.055771f,-0.750040f,
1.966426f,-0.743194f,-0.713226f,0.243448f,0.848127f,-1.123557f,1.359153f,-0.606091f,-0.210089f,1.546869f,0.133445f,2.332562f,0.236079f,0.968399f,-1.113102f,-1.940277f,0.120434f,-0.263833f,-3.112742f,0.348462f,
-0.010618f,2.021881f,0.094364f,-0.800740f,-0.818139f,-3.344273f,0.717900f,-0.898762f,-1.414467f,0.728080f,-1.523094f,1.040100f,-1.164684f,-1.024821f,0.144754f,-0.126973f,1.355560f,-3.233693f,1.106011f,-0.592354f,
0.286175f,1.695512f,-1.058198f,0.177665f,-0.120113f,-1.951829f,-0.541762f,-0.460539f,1.734901f,1.423976f,0.857547f,2.103116f,0.831155f,-0.089665f,0.534431f,-0.079589f,0.178384f,-0.020420f,0.555855f,0.506746f,
0.773670f,0.432172f,0.347747f,-0.567043f,1.737170f,0.147583f,-1.030579f,-0.059261f,1.403112f,-0.248147f,0.065735f,2.454981f,-0.441329f,3.113305f,0.856980f,2.836091f,0.220305f,1.580270f,-0.298613f,0.097317f,
-1.426903f,-0.248848f,-1.367034f,-0.287178f,-1.323465f,0.299417f,0.861624f,0.008704f,-1.075076f,-1.096197f,0.330351f,0.895568f,-0.819286f,0.741472f,-0.977140f,0.335815f,-1.177800f,-1.271896f,1.761490f,-1.822564f,
1.494791f,-1.252261f,-0.261009f,1.022651f,1.314262f,-0.704251f,0.705873f,1.873536f,-0.483957f,-1.215214f,-0.492220f,-0.209988f,1.180433f,-0.357232f,0.756983f,1.249305f,0.481511f,-0.668698f,1.072277f,-1.874061f,
0.272798f,-0.716611f,0.963295f,-0.086289f,-0.050899f,-0.997508f,-0.770930f,-0.522883f,1.741569f,-0.400494f,-2.249887f,1.116130f,1.715476f,-0.626960f,-0.530299f,2.070428f,-0.767388f,0.221946f,-0.009122f,3.715770f,
1.285051f,1.660926f,-0.561403f,-0.751622f,-0.530133f,0.501861f,-0.465690f,-1.615761f,-0.166627f,-0.453325f,0.075008f,0.794913f,-1.043066f,-0.853225f,-0.092356f,0.268175f,-1.441051f,0.818937f,-1.691483f,2.005301f,
-1.653402f,-1.513045f,1.800489f,-1.603181f,2.321302f,-0.521007f,-0.476178f,0.146664f,0.330880f,1.345440f,0.210404f,0.913511f,0.339041f,-2.130182f,-0.643137f,-0.606291f,2.127570f,0.058629f,2.381715f,0.816676f,
0.053433f,0.015653f,0.535496f,-1.803916f,0.570825f,0.379315f,0.261685f,0.917406f,-1.318969f,-1.087916f,-0.196653f,-0.681477f,1.653568f,0.939841f,-1.972420f,-0.198760f,0.668941f,-1.096862f,-0.527964f,3.851149f,
-0.000679f,-0.643324f,0.503644f,1.171909f,0.912054f,-0.171160f,-0.102423f,-1.625998f,-2.033978f,0.681143f,-0.308105f,-0.429215f,-0.227274f,0.716384f,1.369145f,-0.906898f,-0.637303f,-1.031408f,-1.084180f,1.395892f,
-1.100896f,-0.174149f,-0.925733f,1.293436f,-0.816929f,1.878420f,1.307231f,0.059228f,2.330189f,-0.791243f,-0.024409f,-1.106349f,0.735974f,-1.430485f,-0.127795f,-0.065008f,0.290540f,-2.431614f,1.938394f,1.486769f,
1.054230f,0.383727f,0.360849f,1.196777f,2.262561f,-0.311985f,-0.747437f,-0.461922f,0.305520f,-0.904760f,-0.138027f,0.651899f,-0.891778f,-0.939113f,1.024274f,-3.609183f,1.613984f,1.262020f,-2.632707f,-0.144958f,
-0.668082f,-2.121397f,-0.188272f,2.967816f,0.517732f,0.993344f,-0.199858f,0.353302f,2.159609f,0.794254f,-0.272626f,-0.993496f,-1.422857f,-1.253383f,0.783654f,-1.284170f,0.089131f,-1.385330f,2.234484f,-1.463259f,
-0.895761f,-2.080028f,-0.725622f,1.640192f,-0.716277f,1.018931f,-1.142937f,0.905331f,1.109877f,-1.286359f,1.167920f,-1.918445f,2.705729f,-1.093814f,0.041699f,-1.199952f,0.630258f,-1.500281f,0.579911f,0.505586f,
-0.256061f,-1.169584f,0.823237f,0.710731f,1.624878f,-1.488178f,-0.512769f,0.229141f,1.198385f,-0.005714f,-1.354435f,-1.641862f,-0.247286f,1.105234f,-1.629192f,0.636948f,-2.444488f,-1.029048f,0.257958f,-2.696749f,
0.707512f,0.570762f,-1.125334f,-0.945877f,-1.086880f,-2.367006f,-0.327985f,1.711972f,-0.073466f,0.175467f,0.459805f,0.871223f,0.630723f,2.591256f,0.355947f,-0.317292f,-1.668851f,1.093427f,0.625300f,-0.326993f,
-0.326407f,-0.138560f,1.219214f,-1.241809f,-0.357804f,-1.752421f,-1.786726f,0.555107f,-0.488205f,-0.812120f,-0.460454f,1.149078f,-0.152310f,-0.921947f,0.994766f,-1.525438f,1.820946f,0.019513f,0.302900f,-0.376742f,
0.630621f,-0.855746f,0.345685f,1.879544f,-0.292592f,-0.171736f,-0.922004f,-0.676850f,0.822031f,-0.801980f,0.931638f,1.127409f,1.127394f,-1.214157f,0.239931f,-1.496169f,-0.238787f,0.740244f,-0.649013f,0.064001f,
-1.487350f,0.048685f,-1.691290f,-1.032854f,2.159554f,-0.212682f,-0.236192f,-0.528485f,-0.350767f,-1.716631f,-0.524706f,1.547380f,0.001325f,1.284017f,0.105453f,-0.285790f,0.031326f,1.518265f,0.194157f,-0.834646f,
-1.851081f,-0.901666f,0.869195f,1.274271f,1.719452f,1.695706f,0.481758f,-1.343174f,-0.861305f,-2.178129f,0.166011f,0.915589f,-0.805276f,-0.884024f,0.237348f,1.412980f,0.305656f,-0.033802f,1.296372f,-2.761031f,
2.475461f,-1.806245f,0.574880f,-0.925038f,0.042034f,-0.265476f,0.570354f,2.287245f,0.510178f,-0.995775f,-0.267171f,1.500121f,0.968785f,-1.679146f,1.897052f,0.325275f,1.580361f,-1.422486f,-1.127020f,-0.684691f,
-0.863568f,1.085837f,-0.650077f,0.623143f,-2.197670f,0.152597f,-1.917714f,-3.030506f,1.440493f,0.144870f,1.811878f,0.582804f,-2.094681f,-2.296125f,0.450787f,1.764765f,0.101019f,0.590208f,0.403557f,-0.602270f,
-0.310035f,1.114771f,0.413703f,-0.360537f,-1.068717f,-0.076265f,0.772197f,0.752167f,1.579895f,1.264633f,0.009861f,-2.779324f,-1.167009f,-1.314120f,0.535447f,1.633754f,-0.959455f,-0.719713f,0.945840f,-0.124963f,
-1.015418f,-0.548106f,0.308698f,-1.915112f,2.054541f,-1.700524f,0.190903f,0.024502f,0.581029f,-1.264338f,0.495331f,1.255298f,2.112021f,0.581690f,0.384532f,0.730463f,0.340263f,-0.165518f,3.920743f,-0.982811f,
2.294359f,-1.386587f,0.799429f,-0.741672f,0.870650f,-1.088192f,-1.481095f,-0.310060f,-1.422502f,0.596324f,-1.512971f,-1.608276f,0.562816f,-0.334972f,3.544264f,0.219344f,-2.024154f,-2.299127f,0.571566f,2.134293f,
-0.709602f,0.359256f,0.712163f,0.944832f,-1.661464f,1.520553f,0.444577f,0.134770f,-1.027646f,-0.104342f,0.647223f,0.518711f,1.217212f,1.535883f,-0.045703f,-1.931488f,-1.001327f,-1.588028f,1.639326f,2.185794f,
-1.422642f,-0.430749f,1.417037f,0.135039f,-0.990208f,-0.057946f,0.390178f,-1.973692f,1.657312f,-1.798392f,0.517660f,0.283989f,0.223753f,-1.042423f,0.774899f,0.953730f,2.310414f,0.681796f,0.732741f,0.808200f,
0.073557f,0.279038f,3.970247f,-0.940517f,2.241556f,-1.783044f,0.869148f,-0.932348f,1.024252f,-1.095355f,-1.463077f,0.290684f,-1.135695f,0.469625f,-1.500014f,-0.812155f,0.157337f,0.056156f,3.980973f,0.562259f,
-1.270186f,-1.568840f,0.784153f,2.264591f,-0.725632f,0.035664f,0.839954f,1.379412f,-1.646394f,1.135530f,0.877789f,0.417379f,-1.252321f,-0.106217f,0.571736f,0.568022f,0.963011f,1.733658f,0.034740f,-1.145598f,
-0.746099f,-1.334449f,2.360804f,2.481438f,-2.060700f,0.142690f,1.298436f,0.047326f,-0.691848f,0.313959f,0.286531f,-1.528882f,1.526866f,-1.249939f,0.571618f,0.445586f,-0.179389f,-0.834871f,1.278887f,0.767349f,
2.608517f,0.773632f,1.456535f,0.433995f,0.151496f,0.675113f,3.536217f,-0.719732f,2.148798f,-2.071020f,0.908570f,-1.129648f,1.320467f,-1.001096f,-1.434291f,0.595181f,-0.578614f,0.320329f,-1.332104f,-0.243480f,
-0.060568f,0.908801f,4.008699f,0.712177f,-0.499922f,-0.874629f,0.771212f,2.308040f,-0.663161f,-0.286911f,1.060234f,1.659837f,-1.545317f,0.920024f,1.187203f,0.925924f,-1.387197f,-0.408595f,1.040512f,0.805118f,
0.834710f,1.790607f,0.388656f,-0.817431f,-0.615661f,-0.918273f,2.441058f,2.824877f,-2.177778f,1.034823f,0.556824f,0.099064f,-0.085941f,0.616072f,-0.031925f,-1.349308f,1.905951f,-0.592235f,0.384774f,0.291668f,
-0.262773f,-0.999272f,1.349567f,0.839788f,2.555569f,0.789352f,2.266919f,0.198417f,0.667991f,0.926465f,3.109390f,-0.521045f,1.741074f,-2.202552f,0.988321f,-1.485149f,1.536705f,-0.951508f,-1.544385f,1.098872f,
-0.078192f,0.283311f,-1.334668f,0.333311f,-0.234625f,1.689820f,3.541823f,0.592428f,0.140166f,-0.858611f,0.922400f,2.438535f,-0.642865f,-0.433151f,1.121093f,1.320955f,-1.402675f,0.355885f,1.351006f,1.072676f,
-1.492465f,-0.886588f,1.447796f,0.973550f,1.082013f,1.857382f,0.853813f,-0.659049f,-0.665208f,-0.142195f,2.010168f,2.726565f,-2.000071f,1.902533f,-0.229865f,0.012278f,0.734933f,0.531668f,-0.591763f,-1.030752f,
2.308655f,-0.303157f,0.004438f,-0.107155f,-0.174816f,-1.171487f,1.322356f,1.179238f,2.382853f,1.029155f,2.641913f,-0.188096f,0.964155f,0.790561f,2.777591f,-0.115209f,1.312226f,-2.327025f,0.911152f,-1.396662f,
1.411843f,-0.688367f,-1.263709f,1.192939f,0.519782f,0.014068f,-1.277480f,0.485765f,0.040620f,2.477837f,2.990798f,0.199910f,0.382088f,-1.100934f,1.381952f,2.059054f,-0.509314f,-0.503327f,1.104040f,0.823194f,
-1.202053f,0.045265f,1.399113f,1.362181f,-1.523026f,-1.122297f,1.606802f,1.226216f,1.647989f,2.202868f,1.409532f,-0.574321f,-0.688567f,0.383806f,1.302606f,2.278521f,-1.573006f,2.254043f,-1.022381f,-0.057931f,
1.048730f,0.195629f,-1.257904f,-0.989171f,2.714721f,-0.385238f,-0.514356f,-0.153020f,-0.013812f,-1.504746f,0.833558f,1.359140f,2.219686f,1.093260f,2.543728f,-0.136003f,1.120006f,0.543128f,2.841312f,-0.000710f,
1.126850f,-2.169153f,0.870390f,-1.076285f,1.212891f,-0.259281f,-1.256520f,1.000148f,0.856113f,-0.252152f,-1.133303f,0.042597f,0.509869f,2.821522f,2.670339f,-0.297044f,-0.001371f,-1.487556f,1.880972f,1.450873f,
-0.296671f,-0.485151f,0.679984f,0.236995f,-1.408024f,-0.060472f,1.065697f,1.299220f,-1.218928f,-1.368228f,1.339739f,1.368641f,1.989177f,2.422303f,1.800866f,-0.699313f,-0.839765f,0.913081f,1.128920f,1.715017f,
-1.259650f,2.271106f,-1.469450f,-0.328437f,0.767881f,-0.213108f,-1.980774f,-0.899190f,2.724394f,-0.821629f,-0.859627f,0.114749f,0.348773f,-1.765844f,0.344402f,1.370933f,1.897698f,1.399281f,1.720982f,0.021140f,
1.079580f,0.290087f,3.180120f,0.110306f,0.940775f,-2.106092f,0.863474f,-0.721776f,0.837028f,0.260767f,-1.114264f,0.418569f,1.092698f,-0.414471f,-0.823729f,-0.150865f,1.124456f,2.715491f,2.551745f,-0.519452f,
-0.055617f,-1.762351f,2.225120f,0.403658f,-0.252130f,-0.270322f,0.208499f,-0.109564f,-1.567426f,0.112606f,0.836904f,1.243585f,-0.935525f,-1.463318f,0.765656f,1.589326f,2.289493f,2.545734f,2.117502f,-0.706411f,
-1.153999f,1.315411f,1.221609f,1.081928f,-1.109499f,2.060446f,-1.514252f,-0.608983f,0.115879f,-0.829797f,-2.345365f,-0.726941f,2.257890f,-1.431871f,-1.263103f,0.416693f,0.502403f,-1.849174f,-0.224648f,1.204704f,
1.852006f,1.524165f,0.856579f,0.466342f,0.683242f,0.206928f,3.809727f,0.008922f,0.731843f,-1.931984f,0.684053f,-0.486287f,0.643533f,0.821204f,-1.043785f,0.009518f,1.242359f,-0.426576f,-0.388914f,-0.646851f,
1.604172f,2.174891f,2.341283f,-0.487913f,-0.389222f,-1.755502f,2.441820f,-0.335208f,-0.296018f,-0.097799f,-0.194104f,0.274746f,-1.711198f,0.323788f,0.768753f,0.991698f,-0.479796f,-1.810670f,0.179883f,1.638595f,
2.329450f,2.417031f,2.000580f,-0.511823f,-1.336976f,1.130945f,1.509623f,0.714234f,-1.697433f,1.808208f,-1.328205f,-0.664035f,-0.894699f,-1.072769f,-2.415717f,-0.650610f,1.474616f,-2.012546f,-1.547572f,1.117167f,
0.463214f,-1.876878f,-0.389845f,0.722245f,2.034456f,1.341540f,0.051443f,0.888340f,0.183065f,0.339559f,4.310627f,0.079674f,0.845138f,-1.681725f,0.839869f,-0.293152f,0.667473f,1.347295f,-1.030518f,-0.517029f,
1.030922f,-0.421064f,0.328411f,-0.852576f,1.948894f,1.480738f,2.068269f,-0.315911f,-0.478404f,-1.081270f,2.380581f,-0.438910f,-0.482123f,-0.138504f,-0.468664f,1.105263f,-1.729459f,0.376031f,0.752694f,0.926085f,
-0.256733f,-1.785311f,-0.111180f,1.676591f,2.264935f,2.100311f,2.154793f,-0.063428f,-1.471358f,1.068612f,1.678726f,0.565289f,-2.367138f,1.881698f,-1.074896f,-0.624664f,-1.882704f,-1.010065f,-1.628671f,-0.455739f,
0.639412f,-2.077865f,-1.952823f,1.403090f,0.558315f,-1.767553f,-0.472502f,0.111195f,2.165587f,0.876171f,-0.186197f,1.259844f,0.021155f,0.253460f,4.589600f,0.078450f,0.880102f,-1.615833f,0.972635f,-0.641134f,
0.851459f,1.357115f,-1.088309f,-0.455440f,0.574360f,-0.300135f,1.039667f,-1.255092f,1.828044f,1.020312f,1.372400f,0.029073f,-0.591442f,-0.201166f,1.880240f,-0.051902f,-0.736152f,-0.383129f,-0.473555f,2.165945f,
-1.568775f,0.544353f,0.818117f,0.890687f,-0.474358f,-1.776326f,0.156419f,1.695860f,2.188431f,1.910266f,2.274947f,0.190177f,-1.368470f,0.963457f,1.609439f,0.861635f,-2.953955f,2.040335f,-0.873908f,-0.480548f,
-2.419439f,-0.907395f,-0.683713f,-0.239681f,0.001523f,-1.874767f,-2.438940f,1.214152f,0.499544f,-1.790202f,-0.218587f,-0.114588f,2.496012f,0.150016f,0.169381f,1.454862f,0.085781f,0.254571f,4.325623f,0.377993f,
1.044751f,-1.467856f,1.187319f,-1.152528f,1.417666f,1.262085f,-1.306102f,-0.148193f,0.014557f,-0.162625f,1.328887f,-1.282668f,1.285021f,0.611732f,0.492544f,0.142703f,-0.847159f,0.560217f,1.666550f,0.545723f,
-0.854949f,-0.551820f,-0.089027f,2.644166f,-1.389320f,0.421367f,1.109221f,0.882036f,-0.631404f,-1.594087f,1.001730f,1.535707f,2.276436f,1.746134f,2.492224f,0.057254f,-1.236509f,0.858395f,1.034915f,1.156134f,
-2.859848f,2.352356f,-0.996742f,-0.304813f,-2.712166f,-0.596613f,0.178717f,-0.197907f,-0.041427f,-1.051451f,-2.563249f,0.541946f,0.661537f,-1.878389f,-0.002345f,-0.219812f,2.670377f,-0.118660f,0.606509f,1.234129f,
0.245559f,-0.177030f,3.843251f,0.448095f,1.170280f,-1.439597f,1.110310f,-1.757108f,1.988238f,0.987357f,-1.528941f,0.259497f,-0.554423f,-0.091702f,1.359152f,-1.385581f,0.926770f,0.747207f,-0.236039f,0.053761f,
-1.154418f,1.083066f,1.270001f,1.202125f,-0.910896f,-0.810533f,0.386277f,2.526768f,-1.355702f,0.158041f,1.507502f,1.199630f,-0.866015f,-1.059337f,1.808825f,1.599901f,2.314398f,1.640823f,2.680252f,-0.719637f,
-0.927778f,0.788788f,0.491349f,1.386391f,-2.374412f,2.209724f,-1.087662f,-0.537989f,-2.666571f,-0.644841f,0.669445f,-0.216223f,0.167152f,-0.286178f,-2.333694f,-0.146134f,0.759090f,-2.178404f,0.153170f,-0.399232f,
2.677982f,-0.075549f,0.862924f,0.907604f,0.813591f,-0.390189f,3.212236f,0.717978f,1.228697f,-1.420869f,0.955476f,-2.196082f,2.138464f,0.794871f,-1.741461f,0.477617f,-1.164025f,-0.157122f,1.205719f,-1.516191f,
0.743852f,0.876998f,-0.590256f,-0.216351f,-1.856877f,1.122667f,1.133281f,1.425088f,-0.968273f,-0.932534f,0.597838f,1.828088f,-1.206447f,-0.189440f,1.555628f,1.473627f,-0.738479f,-0.690983f,2.608248f,1.530550f,
2.542573f,1.751775f,2.587222f,-1.728407f,-0.756967f,0.820840f,-0.044723f,1.405544f,-1.681076f,1.891716f,-1.041725f,-0.652919f,-2.749199f,-0.778884f,0.607513f,-0.246357f,0.692835f,0.117931f,-1.989109f,-0.398538f,
0.632796f,-2.517323f,-0.029870f,-0.181498f,2.587531f,0.209112f,0.621222f,0.379739f,0.997165f,-0.978663f,2.653024f,0.686196f,1.360380f,-1.467234f,0.700900f,-2.113007f,2.059360f,0.612685f,-1.910557f,0.488039f,
-1.498057f,-0.024218f,0.709589f,-1.596547f,0.638268f,0.846041f,-0.621875f,-0.577362f,-2.516529f,0.898227f,1.031084f,1.537311f,-1.001600f,-0.818577f,0.774664f,0.702990f,-1.522268f,-0.447177f,1.591720f,1.400522f,
-0.670421f,-0.380316f,3.098692f,1.507172f,2.812853f,2.016264f,2.537152f,-2.678850f,-0.680898f,0.697971f,-0.122023f,1.307459f,-0.877587f,1.219592f,-0.698758f,-0.824325f,-2.942655f,-0.842577f,0.436542f,-0.469654f,
1.185080f,0.048600f,-1.888757f,-0.726737f,0.551726f,-2.607261f,-0.249649f,-0.119531f,2.324004f,0.745919f,0.173995f,0.327830f,0.935690f,-1.187184f,2.330147f,0.515289f,1.484358f,-1.455623f,0.506443f,-1.756672f,
1.617598f,0.402441f,-2.095164f,0.234275f,-1.999977f,0.054941f,0.283771f,-1.615592f,0.736352f,0.648499f,-0.047404f,-0.695228f,-3.092392f,0.555824f,1.115817f,1.068915f,-1.050437f,-0.525772f,0.854905f,0.001173f,
-1.852966f,-0.539487f,1.173562f,1.356849f,-0.644518f,-0.148083f,3.068914f,1.440205f,2.753316f,2.196599f,2.147702f,-3.604105f,-0.465503f,0.334960f,0.403152f,1.217325f,-0.341604f,0.394049f,0.060936f,-0.886545f,
-3.179464f,-1.009694f,0.023279f,-0.725397f,1.488040f,-0.271647f,-1.398935f,-0.519724f,0.404521f,-2.347939f,-0.610519f,-0.055629f,2.294487f,1.132935f,-0.132790f,0.318876f,0.590310f,-1.212873f,2.152623f,-0.063513f,
1.805905f,-1.592676f,0.328850f,-1.252066f,1.297596f,0.253561f,-1.839871f,0.135590f,-2.407394f,0.264102f,0.001658f,-1.367311f,0.953848f,0.526802f,0.865214f,-0.276685f,-3.345703f,0.405385f,1.334125f,0.659837f,
-1.095457f,-0.320881f,1.070830f,-0.377099f,-2.068469f,-0.281540f,1.072227f,1.354795f,-0.435700f,0.108342f,2.537161f,1.269046f,2.416578f,2.459622f,1.700298f,-3.830868f,-0.249922f,-0.272835f,1.390972f,1.451584f,
-0.225725f,-0.202114f,0.801422f,-0.940077f,-3.621652f,-0.953185f,-0.005910f,-0.941657f,1.572975f,-0.995308f,-0.810787f,-0.272528f,0.172182f,-2.299829f,-0.664198f,-0.258608f,2.008301f,1.351140f,-0.495548f,0.492245f,
0.334908f,-0.866209f,2.164005f,-0.403499f,1.742108f,-1.683972f,0.217752f,-0.831671f,0.921385f,-0.145047f,-1.611039f,-0.019622f,-2.621369f,0.191557f,-0.120452f,-0.883998f,0.873544f,0.392810f,1.674244f,0.037414f,
-2.897103f,0.663658f,1.574972f,0.229939f,-1.217802f,-0.426114f,1.337530f,-0.278832f,-2.048539f,-0.040260f,1.123233f,1.362957f,-0.327816f,0.166441f,1.924659f,1.146738f,2.040128f,2.553111f,1.412328f,-3.385080f,
-0.019105f,-0.569531f,2.415543f,1.589459f,-0.322062f,-0.321721f,1.385753f,-0.829491f,-4.071938f,-0.744584f,-0.000227f,-0.949380f,1.448184f,-1.328079f,-0.394529f,0.180412f,-0.352656f,-1.960254f,-0.556348f,-0.376959f,
1.850992f,1.597432f,-0.263164f,0.708437f,-0.152250f,-0.342488f,2.066089f,-0.670180f,1.862835f,-1.953464f,0.343978f,-0.512032f,0.821190f,-0.493533f,-1.316790f,0.198603f,-2.395785f,0.454707f,-0.148368f,-0.173284f,
0.678988f,0.353644f,2.180207f,0.566448f,-2.136232f,1.157081f,1.632369f,-0.029234f,-1.257620f,-0.655136f,1.554562f,0.393795f,-2.111443f,0.038250f,1.277334f,1.361744f,-0.322606f,0.243132f,1.635215f,1.207432f,
1.728427f,2.476777f,1.202201f,-2.878294f,0.208137f,-0.783475f,3.241560f,1.804808f,-0.944381f,-0.202428f,1.470366f,-0.816951f,-4.021357f,-0.623245f,0.143437f,-0.997998f,1.361639f,-1.291157f,0.036288f,0.439136f,
-0.828875f,-1.589803f,-0.189974f,-0.811950f,1.841320f,1.491091f,0.344932f,0.485201f,-0.189094f,0.327376f,1.638961f,-0.680080f,1.694974f,-2.005987f,0.356153f,-0.439928f,1.004693f,-0.802200f,-1.104148f,0.644886f,
-2.152634f,0.462147f,-0.198620f,0.506775f,0.305833f,0.669264f,2.234118f,0.826688f,-1.160450f,1.794765f,1.683236f,0.138919f,-1.292161f,-0.967072f,1.865919f,0.939617f,-1.801104f,-0.232047f,1.473923f,1.563545f,
-0.422224f,0.205165f,1.598145f,1.599000f,1.387416f,2.453147f,1.119861f,-2.210898f,0.284370f,-0.568154f,3.655152f,1.983270f,-1.365787f,0.299082f,1.256023f,-0.762752f,-3.551996f,-0.482884f,0.205089f,-0.935140f,
1.553072f,-0.898589f,0.112332f,0.422785f,-1.112740f,-1.444216f,0.248896f,-0.961935f,1.832873f,1.415699f,1.308901f,0.062110f,0.019882f,0.836942f,1.236758f,-0.675068f,1.354785f,-2.309908f,0.400671f,-0.563198f,
1.153950f,-1.171080f,-0.805849f,1.128425f,-1.674060f,0.564822f,-0.096796f,1.130836f,0.080545f,1.344888f,1.809874f,1.034984f,-0.296724f,2.126181f,1.620270f,0.373846f,-1.057914f,-1.319784f,2.296619f,1.315265f,
-1.684276f,-0.630253f,1.612762f,1.849253f,-0.469301f,-0.237559f,2.015716f,2.023717f,1.463288f,2.276958f,1.337652f,-1.888571f,0.386187f,-0.226349f,3.537189f,2.046903f,-1.466442f,0.965958f,0.483536f,-0.757569f,
-3.145481f,-0.526936f,0.115280f,-0.744877f,1.933210f,-0.372203f,-0.049016f,0.120246f,-1.298301f,-1.328714f,0.385390f,-0.711772f,1.662473f,1.523283f,2.250390f,-0.154089f,0.597668f,0.997211f,0.499279f,-0.513896f,
1.076944f,-2.333190f,0.379613f,-0.489742f,1.193406f,-1.292966f,-0.677512f,1.474525f,-1.276434f,0.368165f,-0.186012f,1.432408f,0.000530f,1.931809f,1.252159f,0.754506f,0.186710f,1.995707f,1.765615f,0.532798f,
-1.038522f,-1.462129f,2.332226f,1.127407f,-1.532861f,-0.926870f,1.592771f,1.959009f,-0.445455f,-0.347421f,2.216309f,2.560303f,1.651892f,2.095655f,1.594787f,-1.821453f,0.270401f,0.380262f,2.813519f,1.807533f,
-1.458302f,1.672198f,-0.299062f,-0.907787f,-2.217119f,-0.774100f,-0.427611f,-0.882390f,2.525696f,-0.067058f,-0.276369f,-0.194899f,-1.240176f,-1.317577f,0.162103f,-0.634782f,1.249450f,1.420602f,2.861885f,-0.623773f,
0.790878f,0.992735f,0.036616f,-0.421447f,0.796868f,-2.379133f,0.335185f,-0.278313f,0.901125f,-1.427834f,-0.604872f,1.359665f,-1.001233f,0.173479f,-0.437762f,1.391429f,0.295293f,2.613889f,0.743726f,0.395029f,
0.388318f,1.491557f,1.938542f,0.508325f,-0.909564f,-1.324332f,2.425269f,0.680310f,-1.488158f,-0.969775f,1.165980f,2.015535f,-0.354794f,-0.672489f,2.324579f,2.918409f,2.171382f,2.196230f,1.922124f,-1.845073f,
0.101445f,0.914591f,2.210799f,1.204013f,-1.141134f,1.633430f,-1.230020f,-1.127789f,-1.906343f,-1.341242f,-1.056183f,-0.942809f,2.960189f,-0.289691f,-0.706710f,-0.482565f,-1.113548f,-1.481582f,-0.167109f,-0.482967f,
0.627025f,1.405716f,2.820032f,-0.539938f,0.997594f,0.918106f,-0.067198f,-0.395816f,0.491827f,-2.225174f,-0.065340f,-0.021391f,0.640011f,-1.345174f,-0.253067f,1.221293f,-0.577867f,-0.088337f,-0.434794f,1.073719f,
0.933356f,2.771219f,0.470076f,-0.063122f,0.226240f,0.896083f,2.364205f,-0.240526f,-0.798465f,-1.138965f,2.080108f,0.181996f,-1.381967f,-0.877444f,0.829166f,1.747179f,-0.162119f,-0.719672f,1.842978f,3.226248f,
2.459800f,2.161394f,1.923744f,-1.997611f,-0.150613f,1.335231f,2.006499f,0.567801f,-0.889353f,1.283463f,-1.577851f,-1.244727f,-2.143524f,-1.985775f,-1.665385f,-1.206689f,2.976365f,-0.598052f,-0.998556f,-0.214199f,
-0.944767f,-1.555454f,-0.763703f,-0.466535f,0.211940f,1.596531f,2.256877f,-0.487013f,0.932594f,0.872722f,0.148103f,-0.557564f,0.186871f,-1.949827f,0.038303f,0.443160f,0.217779f,-1.070494f,0.006566f,0.576377f,
-0.400103f,-0.219721f,-0.122492f,0.725398f,1.504099f,2.358949f,0.133174f,-0.444628f,0.042403f,0.087849f,2.736188f,-0.791668f,-0.747436f,-0.910333f,1.648678f,0.030187f,-1.622439f,-0.659488f,0.460674f,1.596828f,
0.292580f,-0.780211f,1.222921f,3.655316f,2.824706f,1.970973f,1.672949f,-1.692764f,-0.604932f,1.325468f,2.013480f,-0.005824f,-1.087521f,1.113416f,-1.741457f,-1.433155f,-2.618579f,-2.384290f,-1.933812f,-1.414264f,
2.594623f,-1.341808f,-1.318988f,0.244808f,-0.782928f,-1.693474f,-1.077724f,-0.786553f,0.161636f,1.593364f,1.445802f,0.029094f,0.695925f,0.950889f,0.489318f,-0.716667f,0.045930f,-1.726680f,-0.231331f,0.883597f,
0.157660f,-0.792620f,0.290185f,0.029775f,-0.437403f,-0.115698f,0.184234f,0.271695f,1.938270f,1.649192f,0.073694f,-0.382402f,-0.091593f,-0.186492f,2.723432f,-1.218234f,-0.764009f,-0.691210f,1.067059f,0.775379f,
-1.661766f,-0.305072f,0.216928f,1.132912f,0.738148f,-0.976617f,0.756522f,3.932630f,2.743973f,1.578308f,1.466100f,-1.450986f,-0.734447f,1.305215f,2.255455f,-0.423906f,-1.720290f,0.771046f,-1.519828f,-1.478897f,
-3.481405f,-2.714934f,-1.647024f,-1.484507f,1.845940f,-1.762128f,-1.626428f,0.744611f,-0.532556f,-1.549525f,-1.188994f,-1.087205f,0.232301f,1.085460f,1.001901f,0.362509f,0.326505f,1.095999f,0.978995f,-0.968955f,
0.064880f,-1.511711f,0.034429f,0.867940f,0.222747f,-0.613337f,0.332176f,-0.341424f,-0.740584f,-0.209060f,0.881742f,-0.147871f,2.074741f,0.921834f,-0.384217f,-0.315349f,-0.514467f,-0.129499f,2.588509f,-0.917222f,
-0.735707f,-0.630427f,0.693609f,1.666563f,-1.663904f,0.110076f,0.138115f,0.858409f,0.851763f,-1.078211f,0.566832f,3.983779f,2.712678f,1.005998f,1.203881f,-1.073999f,-0.979728f,0.965170f,2.345375f,-0.535553f,
-2.447679f,0.718266f,-1.217314f,-1.590715f,-3.988835f,-2.562553f,-0.774078f,-1.524099f,1.093510f,-1.817959f,-1.807223f,0.932900f,-0.396221f,-1.355970f,-0.930983f,-1.364600f,0.505012f,0.523951f,0.866383f,0.538033f,
0.184594f,1.333226f,1.244877f,-1.014605f,0.224234f,-1.410164f,0.129135f,0.487220f,0.476764f,-0.733755f,0.199183f,-0.262531f,-1.171553f,0.022813f,1.224745f,-0.448494f,1.769898f,0.327655f,-0.985186f,-0.077200f,
-0.537589f,0.115745f,1.911220f,-0.159948f,-1.173444f,-0.599429f,0.681792f,2.599987f,-1.494071f,0.193585f,0.104841f,0.812268f,0.691789f,-0.891738f,0.745690f,3.962285f,2.466455f,0.694598f,0.764655f,-0.732190f,
-1.142545f,0.666345f,2.091355f,-0.133605f,-2.977252f,0.669352f,-1.024404f,-1.254274f,-4.324019f,-2.176558f,0.192805f,-1.587227f,0.621917f,-1.367545f,-2.109483f,0.596888f,-0.406237f,-1.330296f,-0.554784f,-1.679430f,
0.582352f,0.049411f,1.411495f,0.670365f,0.141479f,1.219049f,0.941783f,-0.859473f,0.190049f,-1.297656f,0.334077f,-0.008028f,0.928111f,-0.987504f,-0.106912f,-0.055218f,-1.640183f,0.138045f,1.309012f,-0.733444f,
1.319974f,-0.012644f,-1.764865f,-0.027037f,-0.860221f,0.355398f,1.552629f,0.603296f,-1.146100f,-0.679969f,1.018562f,3.199866f,-1.382881f,0.413748f,0.257061f,0.675807f,0.509672f,-0.710728f,1.434473f,4.027882f,
2.485932f,0.178830f,0.666474f,-0.692042f,-0.860569f,0.582165f,1.702894f,0.131928f,-3.176407f,0.875660f,-1.114929f,-1.007387f,-4.070000f,-1.862443f,0.881705f,-1.766084f,0.297376f,-0.535151f,-2.046483f,0.131440f,
-0.353021f,-1.245563f,-0.158238f,-1.651923f,0.827272f,-0.338547f,1.950231f,0.358393f,0.529596f,1.148393f,0.504312f,-0.729441f,0.389849f,-1.172890f,0.201759f,-0.482444f,1.581055f,-1.319381f,-0.156385f,0.395481f,
-2.041411f,0.391379f,1.127235f,-1.072707f,0.760503f,-0.077219f,-2.174374f,-0.185270f,-1.318456f,0.412828f,1.133196f,1.520119f,-1.173567f,-0.742255f,1.304792f,3.109912f,-1.328296f,0.314045f,0.393903f,0.809213f,
0.331805f,-0.268339f,2.304969f,3.771995f,2.434485f,-0.041436f,0.481297f,-1.317838f,-0.722591f,0.540845f,0.990913f,0.649686f,-2.793291f,0.632128f,-1.212171f,-1.072334f,-3.675081f,-1.600387f,1.426489f,-1.939906f,
0.734974f,0.246123f,-1.899892f,-0.420292f,-0.328062f,-1.471956f,-0.088802f,-1.466951f,0.712631f,-0.414655f,2.117717f,-0.131665f,0.817928f,0.720364f,0.079941f,-0.753325f,0.501576f,-1.107961f,0.223361f,-1.008809f,
1.821851f,-1.647024f,-0.290323f,0.633716f,-2.329855f,0.517403f,0.749911f,-1.159500f,0.585951f,0.243125f,-2.250171f,-0.441185f,-1.904621f,-0.086538f,0.885347f,2.062854f,-1.084238f,-0.685430f,1.226442f,2.302299f,
-1.398230f,0.175628f,0.640570f,0.937866f,0.171000f,0.166544f,3.093360f,3.721104f,2.682473f,0.151731f,0.358817f,-1.957385f,-0.655474f,0.342916f,0.489326f,0.554757f,-2.147677f,0.334452f,-1.100355f,-1.048104f,
-3.123719f,-1.468669f,1.343007f,-2.236674f,1.141954f,0.812178f,-1.657611f,-0.917459f,-0.314783f,-1.699108f,0.026650f,-1.186707f,0.770335f,-0.002498f,2.143530f,-0.633536f,1.156663f,0.354584f,-0.437078f,-0.796480f,
0.438200f,-1.192916f,0.051612f,-0.914623f,1.891342f,-1.724088f,-0.669940f,0.706142f,-2.599920f,0.614651f,0.188104f,-1.229954f,0.366833f,0.301170f,-1.954008f,-0.942058f,-2.601211f,-0.897049f,0.896853f,2.271345f,
-1.144910f,-0.501905f,1.176959f,1.380429f,-1.718075f,0.017720f,0.454409f,0.753431f,0.291417f,0.339309f,3.436745f,3.525571f,2.783785f,0.206991f,-0.104531f,-2.735290f,-0.661961f,0.275235f,0.483314f,0.728788f,
-1.376703f,-0.161817f,-0.628673f,-1.392699f,-2.911246f,-1.484475f,0.847742f,-2.542699f,1.600531f,0.693971f,-1.219384f,-1.101356f,-0.230533f,-1.783638f,-0.234829f,-0.831249f,0.418487f,0.488491f,1.672794f,-0.880535f,
1.038999f,-0.073877f,-0.497232f,-1.093779f,0.723209f,-1.282942f,-0.122731f,-0.561247f,1.478296f,-1.749605f,-0.840442f,0.370314f,-2.643915f,0.701999f,-0.276219f,-1.330549f,0.552164f,0.083622f,-1.220657f,-0.964429f,
-3.235212f,-1.913571f,0.991044f,2.024775f,-1.138153f,-0.106880f,0.938686f,0.452011f,-1.969625f,0.199661f,0.306418f,0.686137f,0.378761f,0.580502f,3.292194f,3.324650f,2.720241f,0.554012f,-0.603969f,-3.398588f,
-0.589608f,0.048011f,1.082781f,0.870683f,-0.965590f,-0.877413f,-0.090896f,-1.223489f,-2.612005f,-1.276766f,0.330522f,-2.829125f,1.745992f,0.274583f,-0.765890f,-0.816626f,-0.406650f,-1.866841f,-0.287705f,-0.729819f,
0.480353f,1.009724f,1.117173f,-0.772823f,0.680416f,0.094676f,-0.453731f,-1.506712f,0.839488f,-1.401195f,-0.276550f,-0.253181f,1.284498f,-1.772246f,-0.735304f,0.122892f,-2.644229f,1.006763f,-0.757491f,-1.174634f,
0.578057f,0.115157f,-0.078301f,-0.748051f,-3.336506f,-2.476716f,1.278673f,1.660608f,-1.072285f,0.183241f,0.828369f,0.295502f,-2.317231f,0.613861f,0.012974f,0.477846f,0.494287f,0.581017f,2.715508f,3.048617f,
2.312561f,0.554695f,-1.150678f,-3.276106f,-0.522704f,-0.339542f,1.962355f,0.991539f,-0.885444f,-1.411784f,0.773153f,-1.289149f,-2.691655f,-0.985411f,0.141207f,-3.125061f,1.630714f,-0.333361f,-0.225693f,-0.401032f,
-0.480331f,-1.511879f,-0.263966f,-0.785065f,0.405346f,1.335743f,0.812409f,-0.772139f,0.225595f,0.452467f,-0.071384f,-1.850659f,1.049415f,-1.465381f,-0.190946f,0.113137f,1.032519f,-1.710334f,-0.517276f,0.126495f,
-2.444813f,1.156340f,-0.951436f,-1.097684f,0.512950f,-0.101377f,0.856186f,-0.389587f,-3.198277f,-2.548596f,1.468020f,1.393267f,-1.127441f,0.316475f,0.508909f,0.433985f,-2.400863f,1.046594f,0.201081f,0.451511f,
0.549433f,0.655042f,2.139922f,2.901626f,1.746972f,0.819360f,-1.601102f,-2.833266f,-0.399337f,-0.745275f,3.048630f,1.141637f,-1.230743f,-1.681858f,1.458128f,-0.983177f,-2.656947f,-0.506517f,0.021896f,-3.201449f,
1.305660f,-0.801561f,0.288393f,0.208004f,-0.851965f,-1.414449f,-0.054035f,-0.775215f,0.307296f,1.364518f,0.768150f,-0.586396f,-0.196567f,0.972266f,0.130647f,-1.999228f,1.012397f,-1.673151f,-0.101870f,0.493679f,
1.196206f,-1.880300f,-0.592240f,0.294765f,-2.131566f,1.279551f,-0.844965f,-0.576702f,0.155388f,-0.117991f,1.637695f,-0.004314f,-2.361168f,-2.475736f,1.598882f,1.155102f,-1.086100f,0.067093f,0.441759f,1.053354f,
-2.465433f,1.308857f,0.339192f,0.365954f,0.605051f,0.550141f,1.713235f,2.827257f,1.259342f,0.660362f,-1.770568f,-1.808389f,-0.228274f,-0.865007f,3.847268f,1.393338f,-1.777073f,-1.334995f,1.646587f,-0.780215f,
-2.453650f,0.001556f,0.069846f,-3.099064f,0.923436f,-0.674543f,0.592549f,0.523332f,-1.270605f,-1.036220f,0.313074f,-0.953583f,0.526969f,1.289634f,1.131072f,-0.685259f,-0.246433f,1.531790f,0.296587f,-1.931439f,
0.672163f,-1.755637f,0.274801f,0.521236f,1.449983f,-1.897557f,-0.370842f,0.703480f,-1.401793f,1.530034f,-0.738282f,0.063275f,-0.194363f,0.074701f,1.789781f,0.310597f,-1.520913f,-2.129719f,1.774486f,1.386252f,
-0.976298f,-0.156916f,0.439634f,1.898317f,-2.283659f,1.215563f,0.374642f,0.596411f,0.601424f,0.373537f,1.859671f,2.818980f,0.813423f,0.665422f,-1.850381f,-0.982382f,-0.255746f,-0.692823f,4.357995f,1.749312f,
-2.265379f,-0.585212f,1.428736f,-0.590151f,-1.613972f,0.446912f,-0.005899f,-3.026285f,0.839614f,-0.325468f,0.798127f,0.548701f,-1.569710f,-0.872066f,0.885858f,-1.041185f,0.691395f,1.196612f,1.867032f,-0.980421f,
-0.132951f,2.022746f,-0.008289f,-1.884897f,0.415768f,-1.988302f,0.442562f,0.154091f,1.824381f,-2.026201f,-0.295772f,1.131886f,-0.647429f,1.613555f,-0.567880f,0.407458f,-0.627784f,0.881225f,1.618645f,0.326381f,
-0.713990f,-1.951105f,1.572742f,1.740496f,-0.767149f,-0.504173f,0.406480f,2.152870f,-2.053551f,0.928754f,0.819159f,0.672216f,0.407147f,-0.011774f,2.400072f,3.035228f,0.673370f,0.631766f,-1.498853f,-0.673246f,
-0.216975f,-0.089469f,4.005520f,1.875039f,-2.400255f,0.440858f,0.637475f,-0.583287f,-0.658214f,0.760568f,-0.178995f,-2.801305f,1.128800f,0.264685f,0.458135f,0.228848f,-1.520717f,-0.944103f,1.153990f,-0.655964f,
0.660817f,1.171610f,2.695509f,-1.329445f,0.499713f,1.968133f,-0.273140f,-1.435202f,-0.029093f,-2.087933f,0.344221f,0.182453f,2.000021f,-1.928185f,-0.496264f,1.527527f,0.000168f,1.455338f,-0.669415f,0.505365f,
-0.902362f,1.594734f,1.014232f,0.078899f,-0.287509f,-2.349441f,1.699673f,1.984293f,-0.690969f,-0.772308f,0.208914f,1.958684f,-2.012089f,0.769882f,0.859034f,0.824753f,0.337809f,-0.231219f,2.814823f,3.394526f,
0.753153f,0.537731f,-1.127229f,-0.477127f,-0.274627f,0.466383f,3.352088f,1.674313f,-2.163072f,1.068396f,-0.232683f,-0.675059f,0.351681f,0.779959f,-0.630957f,-2.604685f,1.676000f,0.472992f,0.178178f,0.027067f,
-1.434737f,-1.055719f,0.981169f,-0.292413f,0.388852f,1.167140f,3.070194f,-1.768752f,0.792267f,1.879032f,-0.436392f,-1.298496f,-0.375934f,-1.932946f,0.339159f,0.198890f,1.992698f,-1.652005f,-0.341615f,1.535536f,
0.653769f,1.363630f,-0.846736f,0.312146f,-0.579324f,2.292726f,0.509664f,-0.427437f,-0.227255f,-2.953209f,2.141778f,1.855687f,-0.502667f,-0.573888f,0.047138f,1.494974f,-1.944462f,0.534773f,0.546148f,0.915953f,
0.219204f,-0.603810f,2.780549f,3.600241f,1.204006f,0.661883f,-0.724745f,-0.591666f,-0.610676f,1.115274f,2.710512f,1.196408f,-1.708110f,1.409624f,-0.891632f,-0.886911f,0.843822f,0.368626f,-1.505403f,-2.609661f,
1.934744f,0.325408f,-0.081270f,-0.057109f,-1.184655f,-1.317383f,0.504741f,-0.111932f,0.055463f,1.309288f,2.590455f,-1.739052f,0.920207f,1.594848f,-0.205430f,-1.151093f,-0.611954f,-1.784422f,0.203217f,0.535320f,
1.629539f,-1.309788f,-0.148485f,1.297853f,1.217359f,1.181047f,-0.754475f,-0.145935f,-0.105896f,2.571179f,0.377406f,-0.857992f,-0.406115f,-3.629519f,2.522081f,1.128712f,-0.301702f,-0.452787f,-0.344630f,1.006833f,
-1.840034f,0.775365f,0.319104f,0.594788f,0.703210f,-0.783629f,2.444974f,3.554566f,1.624743f,0.764243f,-0.454352f,-0.614684f,-0.759863f,1.507440f,2.436598f,0.462286f,-1.466666f,1.257851f,-1.122028f,-1.117288f,
0.823852f,-0.049109f,-2.102408f,-2.732873f,1.765737f,-0.233799f,-0.361098f,0.174566f,-1.002438f,-1.577860f,-0.080612f,-0.109774f,-0.195622f,1.355404f,1.636438f,-1.509173f,0.867095f,1.458020f,0.485595f,-1.129920f,
-0.977234f,-1.573553f,0.269495f,0.948881f,1.276658f,-0.731137f,-0.040881f,0.808983f,1.519475f,1.145339f,-0.602598f,-0.715668f,0.294639f,2.029243f,0.150974f,-1.155324f,-0.702674f,-4.107337f,2.943235f,0.244654f,
-0.223471f,-0.218434f,-1.097998f,0.840371f,-1.945081f,0.915935f,0.017664f,0.490929f,1.016003f,-1.007429f,1.971407f,3.576834f,1.577135f,0.836241f,-0.281138f,-0.614748f,-1.114147f,1.721503f,2.453974f,-0.075587f,
-1.421262f,0.981562f,-1.180271f,-1.074892f,0.245718f,-0.522895f,-2.374733f,-2.717199f,1.185074f,-0.936014f,-0.518831f,0.838928f,-0.618866f,-1.608127f,-0.410610f,-0.273411f,-0.061414f,1.382983f,0.682480f,-1.106697f,
0.522558f,1.251140f,1.153748f,-1.304739f,-1.108503f,-1.445771f,0.345593f,1.432280f,1.236224f,-0.244160f,0.092349f,0.260082f,1.733848f,1.101885f,0.176677f,-1.202529f,0.646668f,1.450219f,-0.003673f,-1.023147f,
-1.028611f,-4.168572f,2.933845f,-0.143496f,-0.206748f,-0.026490f,-1.644852f,1.500265f,-2.140604f,1.286423f,-0.143046f,0.105786f,1.353989f,-1.181520f,1.477877f,3.517198f,1.512418f,0.704618f,-0.374454f,-0.300869f,
-1.331121f,1.641446f,2.754391f,-0.245938f,-1.952278f,0.939166f,-0.868066f,-1.070055f,-0.646809f,-0.472925f,-2.119174f,-2.511748f,0.350081f,-1.321329f,-1.026348f,1.290032f,-0.553300f,-1.583822f,-0.394562f,-0.483092f,
0.258336f,1.118285f,-0.057611f,-0.582074f,0.119982f,1.356785f,1.937436f,-1.402329f,-0.951035f,-1.155669f,0.401080f,1.189698f,1.473492f,0.206192f,0.025256f,-0.146477f,1.589324f,1.193970f,0.864788f,-1.752219f,
0.755855f,0.798405f,-0.213079f,-0.837300f,-1.213379f,-3.827135f,2.647468f,-0.071881f,-0.373942f,-0.003630f,-2.045083f,2.407559f,-2.077626f,1.530853f,0.077981f,-0.027650f,1.499349f,-1.434183f,1.290131f,3.384998f,
1.281609f,0.399643f,-0.197527f,0.156612f,-1.506039f,1.412049f,2.748007f,-0.127875f,-2.673311f,1.162037f,-0.444242f,-0.965989f,-1.140309f,-0.107037f,-1.288974f,-2.359679f,-0.472152f,-1.304634f,-1.173322f,1.549408f,
-0.320706f,-1.644151f,-0.237251f,-0.888624f,0.523483f,0.646972f,-0.217704f,-0.203395f,0.001396f,1.314675f,2.348067f,-1.166702f,-0.720956f,-1.100420f,0.637707f,0.890879f,1.898810f,0.411987f,-0.227899f,-0.057998f,
1.314049f,1.431496f,1.402298f,-1.975520f,0.462333f,0.420946f,-0.794007f,-0.512195f,-1.314232f,-3.000735f,2.143165f,0.432336f,-0.643667f,-0.200093f,-2.143321f,3.467522f,-1.926201f,1.649251f,0.242760f,-0.188402f,
1.243601f,-1.399115f,1.604622f,3.216873f,1.091750f,0.042216f,-0.142727f,0.517936f,-1.494856f,1.374480f,2.598239f,0.099186f,-3.115226f,1.355277f,-0.323345f,-0.728774f,-1.515594f,0.368718f,-0.411842f,-2.059611f,
-1.091215f,-0.978224f,-1.535218f,1.261642f,-0.319398f,-1.564746f,0.053057f,-1.026754f,0.936867f,0.093504f,-0.066642f,-0.297319f,0.125415f,1.266747f,2.257771f,-0.722322f,-0.699740f,-1.006494f,0.875282f,0.194161f,
2.493711f,0.287581f,-0.512879f,0.265398f,0.983203f,1.545360f,1.554211f,-2.150380f,-0.218825f,0.190652f,-1.561823f,-0.418590f,-1.509797f,-2.352310f,1.827568f,1.131575f,-0.732674f,-0.363764f,-1.970129f,3.854820f,
-1.729466f,1.605537f,0.546300f,0.040988f,1.034108f,-1.202436f,2.523530f,2.956655f,1.034438f,-0.017113f,0.039496f,0.436792f,-1.171210f,1.416556f,1.994578f,0.510742f,-3.142414f,1.811762f,-0.416287f,-0.694210f,
-1.354750f,0.680191f,0.392160f,-1.872529f,-1.271830f,-0.137800f,-1.708762f,0.779761f,-0.176211f,-1.721758f,0.326994f,-1.185333f,1.173902f,-0.304946f,0.417223f,-0.525810f,0.354915f,0.884814f,2.011786f,-0.430843f,
-0.587483f,-0.965107f,0.807614f,-0.420580f,3.148742f,0.278526f,-0.905360f,0.620453f,0.650600f,1.787802f,1.538951f,-2.558998f,-0.719176f,0.304060f,-2.179956f,-0.614891f,-2.062215f,-2.076867f,1.538928f,1.861544f,
-0.593807f,-0.701218f,-1.606610f,3.595576f,-1.593616f,1.366052f,0.805679f,0.318145f,0.631055f,-0.825014f,3.293969f,2.767097f,0.926645f,-0.116826f,0.300591f,-0.425963f,-1.143250f,1.346743f,1.134842f,0.734984f,
-2.577939f,1.815832f,-0.513414f,-0.729496f,-1.044885f,0.899827f,0.715475f,-1.809956f,-0.975962f,0.525647f,-1.454494f,0.060405f,-0.006470f,-2.080550f,0.361966f,-1.025999f,1.218199f,-0.301700f,0.409540f,-0.878232f,
0.979258f,0.279754f,1.659782f,-0.347332f,-0.366532f,-0.872517f,0.748449f,-0.772133f,3.324979f,0.106829f,-1.230686f,0.924567f,0.327838f,1.814873f,1.206360f,-2.725890f,-0.936572f,0.484885f,-2.303935f,-0.923516f,
-2.679482f,-2.170907f,1.353153f,2.180700f,-0.516590f,-0.732147f,-1.607281f,2.812826f,-1.739180f,1.096990f,1.001325f,0.416282f,0.580651f,-0.448794f,4.136707f,2.432769f,1.220887f,0.117957f,0.309516f,-1.176887f,
-0.941925f,1.362269f,0.605818f,0.938176f,-1.640771f,1.635781f,-0.455505f,-0.830918f,-0.811750f,1.061829f,0.580477f,-1.836833f,-0.447782f,0.887460f,-1.226807f,-0.404322f,-0.093221f,-2.339101f,0.411786f,-0.612683f,
1.393092f,0.119444f,0.108777f,-1.373317f,1.185450f,-0.324585f,1.360617f,-0.279625f,-0.286570f,-0.960348f,0.619609f,-0.758260f,3.180967f,0.136257f,-1.459547f,0.912041f,0.024463f,1.923303f,0.670535f,-2.696182f,
-0.984266f,0.509165f,-2.157164f,-1.232117f,-3.522569f,-2.664799f,1.289894f,2.110517f,-0.541134f,-0.580640f,-1.481033f,1.677513f,-2.042572f,0.737326f,1.070709f,0.468726f,0.699173f,-0.248306f,4.607334f,2.175069f,
1.289587f,0.501041f,0.269001f,-2.294642f,-0.762011f,1.197275f,0.376127f,0.861089f,-0.731861f,1.174535f,-0.050445f,-0.823964f,-0.753203f,1.056546f,0.298383f,-1.914988f,-0.059388f,0.847148f,-0.748803f,-0.412424f,
-0.090679f,-2.511015f,0.158917f,-0.334070f,1.285792f,0.685989f,-0.453453f,-1.398434f,1.017748f,-0.746149f,1.366192f,-0.515934f,-0.063602f,-0.958227f,0.505984f,-0.478687f,2.916473f,0.124096f,-1.700620f,0.701455f,
-0.145867f,2.076255f,0.321788f,-2.923495f,-0.868939f,0.523296f,-1.315889f,-1.398820f,-4.050355f,-3.050454f,1.519959f,1.770003f,-0.721870f,-0.336728f,-1.702028f,0.932140f,-2.229950f,0.746602f,0.884212f,0.296032f,
0.773301f,-0.120723f,4.643683f,1.766815f,1.134978f,0.777585f,0.047444f,-3.097034f,-0.545882f,0.838093f,0.854554f,1.000973f,-0.247798f,0.552268f,0.825459f,-0.843479f,-0.844684f,1.147567f,-0.011437f,-2.032243f,
-0.013266f,0.418968f,-0.296603f,-0.226036f,-0.072580f,-2.552639f,-0.176425f,-0.238827f,1.359939f,1.189473f,-1.129716f,-1.349510f,0.879555f,-0.809366f,1.585608f,-0.646952f,0.259990f,-1.100414f,0.341017f,-0.068482f,
2.556643f,0.168301f,-1.761067f,0.401657f,-0.278015f,2.241530f,-0.091344f,-2.854675f,-0.917233f,0.401444f,-0.161375f,-1.145234f,-4.161661f,-2.963059f,1.660443f,1.208687f,-0.722085f,-0.315778f,-1.729407f,0.563427f,
-2.428633f,0.984200f,0.803277f,0.184687f,0.823926f,0.116521f,4.105046f,1.517663f,0.647664f,1.209539f,-0.109651f,-3.387489f,-0.233584f,0.536877f,1.682497f,1.153184f,0.174876f,-0.013825f,1.608153f,-0.809885f,
-1.362504f,1.285274f,-0.209048f,-1.944551f,0.017233f,-0.299835f,0.083677f,0.127264f,-0.128197f,-2.339034f,-0.217411f,-0.228016f,1.302303f,1.534062f,-1.540562f,-1.209895f,0.356020f,-0.628206f,2.183814f,-1.017137f,
0.393742f,-1.228159f,0.527434f,0.406400f,2.342773f,0.216859f,-1.676001f,0.468007f,-0.305820f,2.509084f,-0.076072f,-2.527652f,-0.886791f,0.208304f,0.834043f,-0.554505f,-3.837634f,-2.643819f,1.898097f,0.749127f,
-0.653289f,-0.436032f,-1.731276f,0.754376f,-2.552166f,1.209997f,0.797941f,0.180449f,0.894807f,0.124565f,3.545172f,1.093663f,0.032508f,1.452768f,-0.165614f,-3.120274f,-0.101518f,0.131523f,2.666976f,1.317060f,
0.132428f,-0.131653f,2.346173f,-0.658529f,-1.618864f,1.631069f,-0.165699f,-1.900843f,-0.185600f,-0.782214f,0.709976f,0.398893f,-0.654300f,-2.204140f,-0.084016f,-0.499332f,1.477378f,1.454076f,-1.695835f,-0.886550f,
0.057626f,-0.279931f,2.386774f,-1.080157f,0.414902f,-1.317054f,0.656908f,0.797757f,2.329932f,0.116195f,-1.481701f,0.527266f,-0.000859f,2.633304f,-0.255969f,-1.912867f,-1.406852f,0.199608f,1.355385f,0.027080f,
-3.003386f,-1.869817f,2.105430f,0.381946f,-0.665584f,-0.606116f,-1.458897f,1.379274f,-2.438007f,1.288565f,1.192496f,0.419955f,0.882520f,0.089386f,3.170620f,1.027630f,-0.291780f,1.534245f,-0.085723f,-2.685698f,
0.174923f,-0.008934f,3.263207f,1.534844f,-0.142034f,0.382380f,2.696785f,-0.451329f,-1.716418f,1.890819f,0.146300f,-1.596849f,-0.262639f,-0.776815f,0.997043f,0.645004f,-1.069352f,-1.758692f,0.313086f,-0.646176f,
1.538219f,1.569245f,-1.321451f,-1.053106f,0.029979f,0.376182f,2.394110f,-1.049422f,0.271702f,-1.498698f,0.728891f,0.756310f,2.461496f,-0.278742f,-1.450548f,0.887247f,0.545123f,2.846736f,-0.050851f,-1.275524f,
-1.844540f,0.557091f,1.307947f,0.455431f,-2.079658f,-0.943258f,1.917793f,0.264804f,-0.669984f,-1.137110f,-1.100362f,1.944540f,-2.255702f,1.005985f,1.460578f,0.500739f,0.888674f,-0.142026f,3.370790f,0.908694f,
-0.901673f,1.505980f,0.127662f,-2.250324f,0.243946f,0.221255f,3.669612f,1.710073f,-0.431818f,0.911727f,2.511331f,-0.386684f,-1.123604f,2.028720f,0.098734f,-1.241825f,-0.090876f,-0.627847f,1.374841f,0.839883f,
-1.370266f,-1.615162f,0.647586f,-0.718587f,1.529797f,1.386969f,-0.752568f,-1.317420f,0.242776f,0.654302f,2.236655f,-0.790351f,0.041377f,-1.582784f,0.782588f,0.851998f,2.496361f,-0.421354f,-1.192765f,1.356427f,
0.907152f,2.679836f,-0.049267f,-0.732472f,-2.163484f,1.171779f,1.235993f,0.579825f,-1.325367f,-0.328446f,1.954826f,0.439468f,-0.482441f,-1.541915f,-0.809628f,2.335109f,-2.057082f,0.435680f,1.629574f,0.721622f,
0.764403f,-0.357095f,3.628105f,1.235095f,-1.137263f,1.547372f,0.580441f,-1.988187f,0.293930f,0.652303f,3.367876f,1.664677f,-0.446735f,1.896737f,1.802953f,-0.445196f,-0.507058f,2.303153f,-0.026807f,-0.998685f,
0.343555f,-0.236471f,1.260069f,0.630771f,-1.524227f,-1.577302f,0.652679f,-0.355102f,1.570563f,1.273232f,-0.061476f,-1.614940f,0.821722f,0.621429f,1.896354f,-0.656617f,-0.455953f,-1.696422f,0.754905f,0.677827f,
2.576208f,-0.569816f,-0.982829f,1.713648f,1.593734f,2.692180f,-0.074232f,-0.532692f,-2.279892f,1.720160f,0.719826f,0.236797f,-0.493283f,-0.234149f,1.948572f,0.553767f,-0.433369f,-1.839985f,-0.674989f,2.255076f,
-1.630378f,0.207604f,1.761841f,0.964242f,0.763590f,-0.756893f,4.057428f,1.425369f,-0.966497f,1.657001f,1.146980f,-1.965417f,0.330942f,1.265258f,2.714398f,1.485800f,-0.144386f,2.679012f,1.089375f,-0.315770f,
0.066336f,2.166984f,-0.318550f,-0.652245f,1.042226f,-0.187266f,1.153281f,0.341579f,-1.625632f,-1.604228f,0.553924f,-0.131362f,1.297707f,1.209987f,0.429664f,-2.047349f,1.281738f,0.539901f,1.842668f,-0.375744f,
-0.598556f,-1.737379f,0.800280f,1.002463f,2.444941f,-0.547015f,-1.018663f,1.935978f,1.993183f,2.545403f,-0.238279f,-0.469694f,-2.274142f,2.368894f,0.228568f,0.141299f,-0.298944f,-0.313914f,2.253869f,0.336458f,
-0.100227f,-1.951811f,-0.605540f,1.803737f,-1.486481f,-0.006974f,1.484510f,0.896873f,0.892564f,-1.148688f,4.155890f,1.685159f,-0.809706f,1.697708f,1.680678f,-2.067974f,0.180653f,1.888079f,1.874166f,0.913036f,
0.200556f,3.105625f,0.218146f,-0.490943f,0.501222f,1.539672f,-0.808172f,-0.542553f,1.254164f,-0.480990f,0.828378f,0.214191f,-1.408463f,-1.834550f,0.219802f,0.058216f,0.899512f,1.131479f,0.356654f,-2.000381f,
1.399688f,0.414459f,1.934035f,-0.278604f,-0.784524f,-1.315530f,0.672318f,1.287479f,2.032631f,-0.564472f,-0.814831f,1.624391f,2.469641f,2.350798f,-0.204596f,-1.107173f,-1.836153f,2.616031f,-0.483713f,-0.348539f,
-0.216462f,-0.538866f,2.479761f,-0.298081f,-0.113343f,-2.007872f,-0.773405f,1.571453f,-1.285864f,-0.014709f,1.256265f,0.660993f,1.015629f,-1.211926f,3.844179f,1.872050f,-0.616407f,1.648102f,2.101845f,-2.395737f,
0.117676f,2.178928f,1.252252f,0.128690f,0.733794f,3.165249f,-0.086023f,-0.559633f,0.325915f,0.862427f,-1.246124f,-0.589296f,1.619304f,-0.872835f,0.565422f,0.149691f,-1.208927f,-1.835230f,-0.531359f,0.071019f,
0.523884f,0.926667f,-0.220377f,-1.735257f,1.570432f,-0.003276f,2.478015f,-0.332108f,-1.183643f,-1.160969f,0.644173f,1.938722f,1.660229f,-0.398683f,-0.460888f,1.064929f,2.602937f,2.243284f,-0.148781f,-1.503142f,
-1.229297f,2.238569f,-0.742642f,-0.546972f,-0.341978f,-0.804236f,2.664437f,-1.022990f,-0.071188f,-1.977671f,-1.166929f,1.542481f,-1.325416f,-0.003083f,0.893388f,0.424634f,1.478962f,-1.241672f,3.293118f,2.082018f,
-0.194509f,1.719185f,2.315284f,-2.655856f,-0.141488f,2.356479f,0.955320f,-0.453577f,0.809710f,2.757319f,-0.247498f,-0.801291f,-0.318818f,0.363799f,-1.417779f,-0.552754f,1.424065f,-1.627052f,0.286765f,0.511276f,
-0.949605f,-1.864321f,-1.040283f,0.004492f,0.427211f,0.909151f,-1.140216f,-1.384863f,1.286644f,0.062687f,3.056992f,-0.494657f,-1.202477f,-0.824629f,0.474579f,2.180766f,1.335958f,-0.267872f,-0.296648f,0.517764f,
2.526711f,2.178722f,0.152339f,-2.002771f,-0.950461f,1.708982f,-0.897674f,-0.527471f,-0.592157f,-0.839537f,2.751917f,-1.654159f,-0.020379f,-1.898500f,-1.363284f,1.991742f,-1.343215f,0.249806f,0.684890f,0.061874f,
1.886739f,-1.511863f,2.648273f,2.228848f,-0.391411f,1.409232f,2.327065f,-2.545199f,-0.436410f,2.315899f,0.892791f,-0.944045f,0.579534f,2.432297f,-0.126638f,-0.943408f,-1.202590f,-0.079427f,-1.292210f,-0.469902f,
0.836733f,-2.214521f,0.004141f,1.002427f,-0.895645f,-1.914922f,-1.080674f,-0.263086f,0.571612f,0.622370f,-1.714390f,-0.919933f,0.945298f,-0.023614f,3.608466f,-0.594820f,-1.284017f,-0.412054f,0.426756f,2.379588f,
1.254869f,-0.021379f,-0.240612f,-0.059616f,2.091252f,2.199359f,0.643403f,-2.407190f,-0.582775f,0.722544f,-1.078629f,-0.440740f,-0.806750f,-0.457184f,2.454977f,-1.457894f,-0.212303f,-1.821739f,-1.545812f,2.951466f,
-1.236946f,0.331088f,0.551365f,-0.160599f,2.130010f,-1.389647f,2.346600f,2.116004f,-0.397699f,0.931764f,2.189359f,-2.437168f,-0.542729f,2.138318f,0.894515f,-1.051198f,0.013606f,2.141923f,0.229563f,-0.848746f,
-2.025250f,-0.120811f,-0.370126f,-0.480715f,-0.051295f,-2.327523f,-0.373141f,1.099784f,-0.941135f,-1.749202f,-0.985342f,-0.720966f,0.739192f,0.087403f,-1.747802f,-0.450086f,0.684148f,0.186873f,4.036021f,-0.791527f,
-1.013974f,-0.136851f,0.437816f,2.250574f,1.426556f,-0.086114f,-0.396736f,-0.152075f,1.578995f,2.414054f,0.982216f,-2.702846f,-0.887097f,-0.027539f,-1.524138f,-0.052824f,-0.836950f,0.138235f,1.928317f,-1.139438f,
-0.362129f,-2.043887f,-1.510628f,3.756309f,-1.142663f,0.757132f,0.504399f,-0.258511f,1.822547f,-1.421545f,2.267032f,2.076922f,-0.662726f,0.447406f,2.140006f,-2.114631f,-0.557226f,1.700175f,0.643322f,-0.861203f,
-0.462193f,2.287057f,0.553240f,-0.600192f,-2.573535f,0.159347f,0.550295f,-0.306869f,-0.443615f,-2.200974f,-0.486207f,0.682973f,-0.809588f,-1.622724f,-0.769841f,-0.958515f,1.063966f,-0.577739f,-1.380075f,-0.218607f,
0.749272f,0.013838f,3.882076f,-0.420931f,-0.826823f,-0.012921f,0.637293f,1.658515f,1.782078f,-0.318699f,-0.457327f,0.037665f,0.896211f,2.442713f,1.148115f,-2.881871f,-1.266264f,-0.386941f,-2.159338f,-0.000195f,
-0.856229f,0.941152f,1.501942f,-0.349557f,-0.665027f,-2.221760f,-1.020718f,4.571728f,-0.935016f,0.726872f,0.796282f,-0.226032f,1.630112f,-1.374198f,2.673323f,2.057980f,-0.520561f,0.235728f,2.492206f,-2.303193f,
-0.432727f,1.651230f,-0.048186f,-0.477111f,-0.562060f,2.502337f,0.455542f,-0.280436f,-2.673604f,0.400796f,1.547033f,-0.153981f,-0.456838f,-1.566848f,-0.640406f,0.088229f,-0.778226f,-1.565489f,-0.401145f,-0.740431f,
1.307244f,-1.229209f,-0.590236f,-0.369726f,0.953583f,-0.132875f,3.629098f,-0.364648f,-0.480019f,0.062100f,0.596420f,1.118440f,2.183951f,-0.804953f,-0.707929f,0.324860f,0.304983f,2.570435f,0.862136f,-3.082254f,
-1.663555f,-0.530775f,-2.640478f,0.055979f,-1.071353f,1.381853f,1.085780f,0.301849f,-0.655644f,-2.530948f,-0.508778f,4.546233f,-0.712099f,0.453611f,0.937922f,-0.297086f,1.298442f,-1.015303f,3.276341f,1.863171f,
-0.543935f,0.023442f,2.386270f,-2.815482f,-0.175692f,1.625906f,-0.936860f,-0.206414f,-0.172525f,2.423182f,0.251458f,-0.289837f,-2.517236f,0.502050f,2.084278f,-0.310517f,-0.290111f,-0.923929f,-0.461052f,-0.586239f,
-0.721319f,-1.502356f,-0.227054f,-0.576108f,1.314476f,-1.382381f,-0.194593f,-0.720057f,1.530307f,-0.414655f,3.255614f,-0.139477f,-0.341965f,0.409476f,0.475538f,0.792248f,2.449942f,-1.227267f,-0.892822f,0.611243f,
-0.058063f,2.612729f,0.429622f,-3.082939f,-2.023150f,-0.407199f,-2.605342f,-0.139261f,-1.555379f,1.475277f,0.919384f,0.854176f,-0.673985f,-2.652006f,-0.118035f,4.021867f,-0.676536f,0.273209f,0.968351f,-0.078440f,
1.037613f,-0.718075f,3.812984f,1.653403f,-0.321208f,-0.054264f,2.451112f,-3.463739f,-0.148041f,1.448917f,-1.866446f,-0.080834f,0.524575f,2.241969f,0.115711f,-0.340439f,-2.075676f,0.419264f,2.169960f,-0.386056f,
0.397889f,-0.285892f,-0.367253f,-1.358363f,-0.638385f,-1.773647f,0.007483f,-0.241347f,1.299194f,-1.213204f,0.121144f,-1.015133f,1.753869f,-0.905155f,2.867905f,-0.079090f,-0.373354f,0.307472f,0.269201f,0.635021f,
2.365276f,-1.461307f,-1.160932f,0.647428f,-0.536103f,2.521667f,-0.289986f,-3.323069f,-2.073958f,-0.402092f,-2.275776f,-0.559313f,-1.884309f,0.994492f,0.608432f,1.137774f,-0.579061f,-2.766587f,-0.107835f,3.192510f,
-0.822986f,-0.024304f,0.840618f,-0.170167f,1.016135f,-0.348990f,4.084664f,1.661742f,0.011979f,0.112583f,2.392040f,-4.345778f,0.070131f,1.520858f,-2.301804f,-0.080049f,1.237564f,1.836528f,0.151019f,-0.353073f,
-1.882046f,0.212354f,1.932003f,-0.528863f,1.008701f,-0.198466f,0.036451f,-1.624376f,-0.611323f,-1.862869f,-0.199092f,0.223417f,1.245695f,-0.750309f,0.046171f,-1.236189f,1.897504f,-1.161367f,2.858923f,-0.323582f,
-0.134446f,0.465867f,0.046290f,0.864622f,1.945337f,-1.548843f,-1.265377f,0.496130f,-0.829535f,2.452763f,-0.791736f,-3.444961f,-1.875428f,-0.498045f,-1.560131f,-0.786071f,-2.432549f,0.372210f,0.762568f,0.951676f,
-0.494180f,-2.440596f,-0.014477f,2.398075f,-1.010045f,-0.045618f,0.687953f,-0.264908f,0.938039f,-0.274356f,4.118298f,1.385518f,0.060174f,0.332805f,2.197342f,-5.111532f,0.143270f,1.152771f,-2.168669f,-0.049816f,
1.915759f,1.163389f,0.577226f,-0.465544f,-1.662178f,0.257916f,1.498850f,-0.830728f,1.302741f,-0.529982f,0.278296f,-1.790652f,-0.762479f,-1.751885f,-0.253923f,0.586088f,1.099214f,-0.502247f,-0.308917f,-1.027209f,
1.564750f,-1.247752f,2.811537f,-0.601945f,0.242156f,0.324661f,-0.243307f,1.313504f,1.458912f,-1.675505f,-1.314644f,0.107601f,-1.253664f,2.554072f,-1.481903f,-3.216136f,-1.696774f,-0.530131f,-0.521315f,-0.648300f,
-2.752811f,-0.107498f,0.834402f,0.622340f,-0.714971f,-2.367133f,0.137180f,1.883994f,-1.309034f,0.142183f,0.435661f,-0.342568f,1.058093f,-0.033811f,3.508172f,1.086123f,-0.176221f,0.645087f,1.686204f,-5.608122f,
0.258493f,0.819966f,-1.629785f,0.031116f,2.250704f,0.429638f,1.251768f,-0.310098f,-2.060481f,0.244734f,1.219525f,-0.875611f,1.609981f,-1.000736f,0.733483f,-1.560572f,-0.894025f,-1.653392f,-0.390237f,0.690638f,
1.079499f,-0.178153f,-0.514233f,-0.732375f,1.089440f,-1.308957f,3.292779f,-0.905753f,0.465500f,0.337462f,-0.340361f,1.552680f,1.003481f,-1.769111f,-1.305993f,-0.060075f,-1.485224f,2.567592f,-1.805102f,-3.152098f,
-1.610649f,-0.747518f,0.808014f,-0.354402f,-2.486581f,0.158406f,0.953009f,0.174229f,-0.623466f,-2.363430f,0.195178f,1.702113f,-1.457849f,0.466226f,0.269208f,-0.396778f,0.954307f,0.140508f,2.775237f,0.757217f,
-0.335749f,0.795831f,1.497963f,-5.245789f,0.525431f,0.363484f,-0.727760f,0.208402f,2.292591f,0.167834f,2.030657f,-0.157607f,-2.456399f,0.499722f,1.116520f,-1.038144f,1.461316f,-1.507709f,1.120193f,-1.356476f,
-1.051174f,-1.364164f,-0.380390f,0.750306f,1.012291f,-0.035956f,-0.532268f,-0.520302f,0.690696f,-0.927201f,3.390524f,-1.114926f,0.723377f,0.253473f,-0.320281f,1.849438f,0.910161f,-1.982695f,-1.104137f,-0.102504f,
-1.465644f,2.655816f,-1.917927f,-2.455557f,-1.671902f,-0.942847f,1.720698f,0.319966f,-1.902222f,0.522428f,1.207278f,-0.065098f,-0.610455f,-2.559530f,0.541476f,2.029607f,-1.470840f,0.585138f,0.428360f,-0.321294f,
0.833214f,0.122035f,1.998896f,0.603862f,-0.592008f,0.938796f,1.382802f,-4.848375f,0.788835f,0.049448f,0.174168f,0.688818f,1.959906f,-0.076498f,2.395638f,0.079274f,-2.736755f,0.613096f,1.189112f,-0.976324f,
1.322687f,-1.841797f,1.417368f,-1.217060f,-1.371031f,-1.075720f,-0.090834f,0.484648f,1.082079f,-0.059539f,-0.139123f,-0.364818f,0.340124f,-0.438408f,3.596383f,-1.151795f,0.743542f,0.065959f,-0.135664f,1.769628f,
0.834678f,-2.234403f,-1.085031f,0.046305f,-1.179713f,2.789315f,-2.064127f,-1.822096f,-1.859839f,-1.005786f,2.220449f,0.779438f,-1.039214f,1.336872f,1.282115f,-0.151216f,-0.748151f,-2.965714f,0.861359f,2.636915f,
-1.296568f,0.371419f,0.631441f,-0.204360f,0.802342f,-0.070791f,1.629453f,0.661498f,-0.773228f,0.912202f,1.472522f,-4.118188f,0.995174f,-0.025227f,0.753512f,0.829565f,1.553999f,0.248065f,2.316039f,0.106302f,
-2.588836f,0.869138f,1.272416f,-0.643088f,1.372785f,-1.550614f,1.643064f,-0.934440f,-1.686570f,-0.631823f,0.371162f,0.409642f,1.262473f,-0.143901f,0.667924f,-0.532284f,0.451082f,0.106799f,3.481252f,-0.994658f,
0.678920f,-0.062381f,-0.332585f,1.931081f,0.867677f,-2.433436f,-0.850481f,0.502195f,-1.077895f,2.580849f,-1.999528f,-1.146210f,-2.113701f,-0.495255f,2.315469f,1.033666f,-0.000176f,2.100675f,1.408969f,-0.094068f,
-0.598070f,-3.413432f,1.073959f,3.109562f,-1.233019f,-0.047208f,0.786800f,0.256015f,0.539034f,-0.171107f,1.753655f,0.945335f,-0.984408f,0.937499f,1.690861f,-3.636137f,1.166728f,0.201364f,0.850011f,1.092020f,
1.108976f,1.067953f,1.778115f,0.227723f,-2.143697f,1.087824f,1.277181f,-0.361142f,1.582204f,-1.083859f,1.523377f,-1.226497f,-2.155714f,-0.493802f,0.755899f,0.601683f,1.347531f,-0.245836f,1.618156f,-0.637290f,
0.595239f,0.396808f,2.860511f,-0.897481f,0.533657f,-0.191818f,-0.354433f,1.603729f,0.995897f,-2.475936f,-0.884640f,0.840904f,-0.799729f,2.393408f,-2.095659f,-0.404021f,-2.291129f,0.166140f,2.140560f,1.084823f,
0.774227f,2.621593f,1.362368f,0.131599f,-0.619972f,-3.827467f,1.387045f,3.283106f,-0.921443f,-0.454234f,1.083567f,0.450940f,0.327021f,-0.405335f,1.970479f,1.293542f,-0.832683f,0.803709f,2.194087f,-3.134142f,
1.172008f,0.795595f,0.545606f,1.150489f,1.073941f,1.746544f,1.019570f,0.255343f,-1.497346f,1.160645f,0.914975f,-0.197434f,1.982133f,-0.637661f,1.197597f,-1.583345f,-2.138515f,-0.606468f,0.785029f,0.847748f,
1.268740f,-0.207055f,2.690223f,-0.846481f,1.053869f,0.370669f,2.528719f,-0.615076f,0.124874f,-0.184191f,-0.316718f,1.425662f,0.949181f,-2.555958f,-0.695178f,0.978869f,-0.428323f,2.158096f,-2.249498f,-0.083759f,
-2.121070f,0.865908f,1.765112f,0.697941f,1.402985f,2.560868f,1.575028f,0.273056f,-0.463423f,-3.962978f,1.615211f,2.859513f,-0.648917f,-0.895848f,1.049194f,0.736640f,0.346581f,-0.708591f,2.025289f,1.593133f,
-0.375260f,0.862352f,2.682558f,-3.100795f,1.079072f,1.463824f,-0.428589f,0.774550f,1.193605f,2.302695f,-0.134347f,0.276539f,-0.966073f,0.622376f,0.409200f,-0.150667f,2.496104f,-0.411610f,0.696841f,-1.984043f,
-2.040958f,-0.600383f,0.568848f,1.072550f,0.988488f,-0.408982f,3.128721f,-0.885097f,1.189264f,0.280998f,2.149934f,-0.336388f,-0.121713f,-0.162710f,-0.444806f,1.513526f,0.571781f,-2.330786f,-0.651408f,1.007389f,
-0.011849f,1.825276f,-2.487381f,-0.400865f,-1.722147f,1.359965f,1.320439f,0.452692f,1.434801f,2.336546f,2.017501f,-0.211638f,-0.369911f,-3.963582f,1.689993f,2.414647f,-0.704013f,-1.063664f,0.840842f,0.681743f,
0.369991f,-0.908319f,1.766739f,2.090589f,0.181070f,0.931139f,3.169197f,-3.090030f,0.970114f,1.978742f,-0.933800f,0.339900f,1.693882f,2.344919f,-0.756830f,-0.010783f,-0.989557f,-0.015203f,-0.347931f,-0.251914f,
2.866764f,-0.697252f,0.194866f,-2.238764f,-1.732856f,-0.809477f,0.049059f,1.097925f,0.402836f,-0.221347f,2.984613f,-0.823180f,1.305598f,0.156489f,2.406718f,-0.246873f,-0.202511f,0.053485f,-0.886621f,1.797447f,
0.090462f,-2.102935f,-0.370345f,0.443811f,-0.153558f,1.500061f,-2.421437f,-0.620703f,-0.970820f,1.566445f,1.318095f,0.029130f,1.394269f,1.967475f,2.478433f,-1.046819f,-0.221611f,-4.002314f,1.459654f,1.827816f,
-0.736617f,-1.005655f,0.338081f,0.576194f,0.568185f,-1.192126f,1.219591f,2.360794f,0.643293f,1.077330f,3.341320f,-3.259763f,0.809646f,2.407033f,-1.128781f,-0.384780f,1.823221f,2.003708f,-1.222448f,-0.126164f,
-1.271623f,-0.480549f,-1.064485f,-0.259168f,2.879333f,-1.071904f,-0.145805f,-2.101686f,-1.542279f,-0.946121f,-0.555792f,1.278682f,0.228453f,-0.031827f,2.328865f,-0.533826f,0.956071f,0.046076f,2.738253f,-0.326939f,
-0.515517f,0.351839f,-0.988627f,2.398795f,-0.254422f,-1.765168f,-0.119264f,-0.203132f,-0.001801f,1.106896f,-2.144366f,-0.940037f,-0.253285f,1.019378f,1.236557f,-0.150282f,1.311231f,1.667417f,2.777900f,-1.896508f,
-0.297833f,-3.851243f,0.925032f,1.873188f,-1.009149f,-0.944719f,0.154064f,0.378482f,0.982707f,-1.321329f,0.356121f,2.635063f,1.102957f,0.971496f,3.274259f,-2.967896f,0.458733f,2.357054f,-0.821625f,-0.922149f,
1.490052f,1.577597f,-1.317816f,-0.226484f,-2.002326f,-1.045840f,-1.410719f,-0.403404f,2.169271f,-1.704852f,-0.706135f,-1.566620f,-1.447043f,-0.893467f,-0.884075f,0.926548f,0.135354f,-0.153351f,1.573926f,-0.020804f,
0.486951f,0.167712f,3.338669f,-0.500319f,-0.659655f,0.560740f,-1.152843f,2.483090f,-0.543768f,-1.125949f,0.012270f,-0.821990f,-0.270064f,1.047941f,-1.414024f,-1.381241f,0.219367f,0.319050f,1.247219f,-0.060418f,
0.923289f,1.548132f,2.764113f,-2.292692f,-0.322627f,-3.863583f,0.628779f,2.633816f,-1.062356f,-0.716897f,-0.129184f,0.187212f,1.314037f,-1.363663f,-0.206069f,2.625768f,1.270073f,0.662637f,2.992217f,-2.513376f,
0.182518f,2.200859f,-0.643700f,-1.069105f,0.675145f,1.389215f,-1.120104f,-0.399965f,-3.126367f,-1.245889f,-1.087420f,-0.552564f,1.115684f,-1.955338f,-1.154760f,-1.182685f,-1.136390f,-0.955027f,-0.918960f,0.483270f,
0.282283f,-0.548759f,1.138835f,0.682648f,0.011412f,0.237873f,3.641111f,-0.645643f,-0.438744f,0.779962f,-0.954635f,2.236428f,-0.371373f,-0.808125f,-0.151966f,-1.224396f,-0.823273f,0.946617f,-0.950518f,-1.550604f,
0.273676f,-0.403950f,1.106237f,0.226857f,0.732625f,2.117160f,2.587532f,-2.120532f,-0.473135f,-3.729323f,0.388237f,3.347711f,-1.148988f,-0.472529f,-0.112867f,0.219144f,1.422233f,-1.444827f,-0.638542f,2.749846f,
1.329855f,0.248928f,2.776179f,-1.730635f,0.094549f,2.075594f,-0.210181f,-0.773353f,-0.217209f,1.198882f,-0.850240f,-0.115744f,-3.819246f,-0.988662f,-0.276905f,-0.464113f,0.307426f,-1.931894f,-1.748300f,-1.101322f,
-0.998204f,-0.746154f,-0.485181f,0.248268f,0.705354f,-1.015208f,1.428563f,0.984080f,-0.346632f,0.383087f,3.575176f,-0.600168f,-0.360228f,0.800972f,-0.889620f,1.686128f,-0.004928f,-0.724850f,-0.170015f,-1.076217f,
-1.270372f,1.044766f,-0.544386f,-1.614442f,0.089200f,-0.815632f,0.462005f,0.478220f,0.639437f,2.603343f,2.340473f,-1.515086f,-0.753389f,-3.911646f,0.535350f,4.211712f,-1.013058f,-0.145372f,0.051944f,0.208383f,
0.966232f,-1.508149f,-0.323382f,2.877118f,1.512556f,-0.054366f,2.732039f,-1.284529f,0.230464f,1.979907f,-0.294567f,-0.514224f,-0.850269f,1.434992f,-0.673741f,0.035741f,-4.169994f,-0.687347f,0.548266f,-0.568428f,
-0.327824f,-1.173379f,-2.178982f,-1.665061f,-0.945103f,-0.675480f,-0.341592f,-0.186018f,0.920300f,-1.533643f,2.177465f,1.112836f,-0.318181f,0.425283f,3.130629f,-0.186075f,-0.161291f,0.606953f,-0.800814f,0.920119f,
0.608793f,-0.891729f,-0.585635f,-0.700186f,-1.939384f,1.094818f,-0.366981f,-1.591780f,-0.126322f,-0.925941f,-0.158932f,0.474621f,0.360039f,3.255247f,2.040068f,-0.683592f,-1.026914f,-3.992281f,0.881534f,4.612109f,
-0.892131f,-0.351655f,0.129252f,0.351565f,0.683963f,-1.155034f,0.465625f,3.078321f,1.699874f,-0.266218f,2.644431f,-1.125443f,0.419452f,1.897156f,-0.651446f,-0.030847f,-0.894643f,1.476295f,-1.188016f,0.119058f,
-4.112823f,-0.385753f,1.113621f,-0.617406f,-0.377543f,-0.013377f,-2.383809f,-2.259298f,-1.102587f,-0.782744f,-0.129157f,-0.046045f,0.977179f,-1.628729f,2.796177f,0.722051f,-0.013192f,0.321601f,2.363497f,0.036385f,
-0.006188f,0.816450f,-0.837194f,0.351303f,1.048598f,-0.994587f,-0.720814f,-0.256240f,-2.507436f,1.076339f,-0.510326f,-1.519861f,-0.504747f,-0.761892f,-0.548475f,0.307473f,-0.058143f,3.263961f,2.033977f,0.009352f,
-0.885562f,-4.086033f,1.080041f,4.040551f,-1.113739f,-0.584345f,0.476601f,0.893249f,0.541496f,-0.635996f,1.294508f,3.197742f,2.232263f,-0.378293f,2.528222f,-1.794826f,0.572419f,1.966184f,-1.316538f,0.277913f,
-0.622079f,1.375342f,-1.526797f,-0.100870f,-3.766771f,-0.229165f,1.143387f,-0.911076f,-0.074193f,0.805788f,-2.290639f,-2.827165f,-0.905744f,-1.033386f,0.013089f,0.005566f,0.995645f,-1.550292f,3.171565f,0.278608f,
0.449208f,-0.060354f,1.508977f,-0.043396f,-0.133643f,0.792538f,-1.044904f,-0.070266f,1.127114f,-1.170696f,-0.823695f,-0.126851f,-2.965086f,0.938683f,-0.771863f,-1.698628f,-0.481612f,-0.345957f,-0.681311f,-0.196177f,
-0.765073f,2.796578f,1.949618f,0.371520f,-0.871794f,-4.034062f,1.120056f,3.066658f,-1.295498f,-0.878678f,0.592408f,1.040287f,0.519347f,-0.269449f,1.821997f,3.138264f,2.677078f,-0.296143f,2.172665f,-2.314821f,
0.563011f,1.985327f,-1.634847f,0.409959f,-0.077892f,0.831508f,-1.592465f,-0.218626f,-3.672461f,-0.418653f,0.752034f,-1.135795f,0.343792f,1.200836f,-2.409240f,-3.343852f,-0.801263f,-1.373459f,0.097019f,0.146118f,
0.883813f,-1.004448f,3.136789f,-0.043217f,0.494466f,-0.542385f,0.949292f,-0.087617f,0.076036f,0.669703f,-1.367216f,-0.104865f,1.227618f,-1.078846f,-0.987011f,-0.180322f,-3.319212f,0.922799f,-1.040100f,-1.672955f,
-0.129761f,-0.390589f,-0.357541f,-0.640367f,-1.486801f,1.764102f,1.862891f,0.318417f,-0.884439f,-3.603079f,1.075572f,1.907657f,-1.566625f,-0.955683f,0.410996f,1.063573f,0.558095f,-0.070626f,2.075641f,3.205458f,
3.060691f,-0.158422f,1.792990f,-3.031482f,0.617189f,1.836634f,-1.530234f,0.241522f,0.443018f,0.070697f,-1.228215f,-0.461299f,-3.566893f,-0.529652f,0.405787f,-1.578962f,0.745927f,1.342514f,-2.137799f,-3.594759f,
-0.760399f,-1.478123f,-0.201755f,0.309472f,0.641789f,-0.451061f,2.673138f,-0.068651f,0.209263f,-0.573998f,0.687678f,-0.419598f,0.302951f,0.550821f,-1.441445f,0.044999f,0.803335f,-1.100408f,-1.116752f,-0.520894f,
-3.763179f,0.816904f,-1.540525f,-1.729156f,0.113851f,-0.467432f,0.347310f,-0.796612f,-2.288605f,0.983502f,2.158991f,0.079082f,-1.009865f,-3.278716f,1.015784f,1.194027f,-2.121908f,-0.829208f,0.115600f,0.948344f,
0.772583f,0.152521f,1.918853f,3.188375f,3.053364f,-0.051564f,1.171375f,-3.520478f,0.656160f,1.426940f,-0.942796f,0.496777f,0.685569f,-0.866275f,-0.711614f,-0.525027f,-3.679577f,-0.462563f,-0.064038f,-2.042251f,
0.891003f,0.918942f,-1.703540f,-3.283925f,-0.761871f,-1.451017f,-0.324048f,0.532156f,0.568944f,-0.054445f,2.323423f,0.006181f,-0.122548f,-0.494090f,0.680699f,-0.704171f,0.619699f,0.393561f,-1.603513f,0.270936f,
0.526282f,-1.078730f,-0.954556f,-0.824853f,-4.013605f,0.994782f,-1.881095f,-1.635518f,0.311584f,-0.761240f,1.571878f,-0.644974f,-2.619282f,0.400147f,2.330693f,-0.189379f,-1.030713f,-2.967338f,0.914514f,0.807262f,
-2.335063f,-0.537246f,-0.166436f,0.970811f,0.606035f,0.491065f,1.220042f,3.040976f,2.902472f,0.228759f,0.553130f,-3.352689f,0.849950f,1.010514f,0.017114f,0.493436f,0.835028f,-1.538903f,-0.002829f,-0.470904f,
-3.862345f,-0.237799f,-0.385712f,-2.333148f,0.751660f,0.381677f,-1.384881f,-2.961552f,-0.912712f,-1.320970f,-0.388915f,0.525321f,0.426522f,0.192301f,2.176339f,0.398479f,-0.647845f,-0.169806f,0.745974f,-1.225847f,
0.723496f,0.211595f,-1.610110f,0.762502f,0.293444f,-1.199195f,-0.936408f,-0.848304f,-4.171042f,0.985080f,-1.971646f,-1.017906f,0.485236f,-0.843032f,2.563774f,-0.152293f,-2.446609f,0.223287f,2.632447f,-0.502988f,
-1.224126f,-2.821841f,0.905611f,1.016083f,-2.447821f,0.083429f,-0.153279f,0.847441f,0.670923f,0.593643f,0.636512f,2.864622f,2.840575f,0.318566f,0.048397f,-3.083388f,0.928490f,0.455394f,1.303204f,0.884682f,
0.416884f,-1.895212f,0.777027f,-0.316213f,-4.016254f,-0.059427f,-0.344771f,-2.568096f,0.522793f,0.029778f,-1.063929f,-2.574755f,-1.241541f,-0.873397f,-0.126454f,0.145333f,0.437361f,0.280185f,2.250359f,0.405360f,
-1.221722f,0.289070f,0.862872f,-1.366111f,0.819092f,0.024672f,-1.560944f,0.905306f,0.305259f,-1.301326f,-0.791537f,-0.714734f,-4.065038f,1.138373f,-2.089085f,-0.580565f,0.327690f,-1.056778f,3.221549f,0.377183f,
-1.734306f,0.412597f,2.698177f,-0.569050f,-1.152317f,-2.806768f,0.913550f,1.425047f,-2.499487f,0.153899f,0.066039f,0.821966f,0.626696f,0.499241f,0.149031f,2.742856f,2.202630f,0.186328f,-0.378048f,-2.170943f,
1.115540f,0.403481f,2.112630f,1.281500f,-0.242496f,-1.807045f,0.917867f,-0.142161f,-3.772349f,0.270176f,-0.338194f,-2.679509f,0.194244f,0.150377f,-0.573204f,-2.114596f,-1.565827f,-0.455032f,0.303129f,0.087562f,
0.493530f,0.311864f,2.731731f,0.379556f,-1.326055f,0.992855f,0.653717f,-1.296916f,0.704518f,-0.213688f,-1.447965f,0.810291f,0.530772f,-1.521643f,-0.654877f,-0.527257f,-3.324836f,1.256364f,-1.938693f,-0.001851f,
-0.181796f,-0.725136f,3.486066f,0.768021f,-0.879704f,0.705180f,2.557586f,-0.342904f,-1.071718f,-3.115819f,0.951695f,2.113025f,-2.331245f,0.024401f,0.277146f,0.956574f,0.581158f,0.275438f,0.031922f,3.039275f,
1.697013f,-0.025834f,-0.554381f,-1.356057f,1.153411f,0.302278f,2.789467f,1.675275f,-0.948387f,-1.146512f,0.783525f,-0.064710f,-3.264628f,0.860232f,-0.192219f,-2.549635f,0.152604f,0.460487f,-0.497597f,-1.874842f,
-1.891959f,-0.334995f,0.656826f,-0.050936f,0.727199f,0.079656f,3.593665f,0.034543f,-1.084598f,1.442798f,0.301546f,-1.377089f,0.362309f,-0.369826f,-1.253792f,0.579036f,0.765771f,-1.686526f,-0.518300f,0.152285f,
-2.949915f,1.157387f,-1.938127f,0.569686f,-0.455845f,-0.248122f,3.315796f,0.740284f,-0.125059f,0.757970f,2.436410f,0.244289f,-0.862955f,-3.271384f,1.033661f,2.758496f,-2.179110f,-0.105403f,0.432905f,1.306782f,
0.579687f,0.016932f,0.405985f,3.433827f,1.628840f,-0.145493f,-0.583399f,-0.669985f,0.970801f,0.820647f,2.672547f,1.752665f,-1.295786f,-0.415204f,0.061110f,0.208736f,-2.280874f,1.130579f,-0.203074f,-2.546717f,
0.490880f,1.039920f,-0.649394f,-2.198895f,-1.977195f,-0.169888f,1.277021f,0.276839f,0.594911f,0.117643f,4.552702f,-0.461732f,-0.647190f,1.756364f,-0.108532f,-1.218338f,0.018265f,-0.516067f,-1.221591f,0.360858f,
0.968132f,-1.738775f,-0.346630f,0.566026f,-2.401317f,1.037427f,-2.195397f,1.006606f,-0.547633f,0.409439f,2.778180f,0.568107f,0.613560f,0.504212f,2.631951f,0.769946f,-0.915270f,-3.592901f,0.979955f,2.520531f,
-2.009477f,-0.337079f,0.540868f,1.480329f,0.457795f,-0.143034f,0.897941f,3.833674f,1.829378f,-0.371989f,-0.295664f,-0.404480f,0.998691f,1.311996f,2.096698f,1.592842f,-1.152159f,0.127347f,-0.798717f,0.090360f,
-1.255901f,1.000826f,-0.716346f,-2.584712f,1.101179f,1.453393f,-0.859337f,-2.620041f,-2.039626f,-0.201383f,1.148016f,0.605478f,0.238479f,0.087037f,5.206833f,-0.648347f,-0.230636f,1.932923f,-0.523786f,-1.028689f,
-0.272885f,-0.559198f,-1.269986f,0.437947f,0.769885f,-1.740930f,-0.310215f,0.718265f,-1.880050f,1.118311f,-2.439565f,0.886407f,-0.475147f,1.180173f,2.414847f,0.138554f,0.815366f,-0.192732f,2.864137f,0.821129f,
-0.736150f,-3.432693f,0.952951f,2.118817f,-1.910948f,-0.500921f,0.210901f,1.469260f,0.340179f,-0.499095f,0.972290f,4.273096f,2.250381f,-0.220008f,-0.073631f,-0.525962f,0.789188f,1.891348f,1.319435f,1.088312f,
-0.904327f,0.625649f,-1.812896f,-0.067022f,-0.609513f,0.662592f,-1.340355f,-2.687430f,1.435130f,1.294547f,-1.346637f,-2.746041f,-1.588950f,-0.566284f,0.842282f,0.735543f,-0.248952f,0.160137f,5.327435f,-0.755308f,
0.028085f,1.603520f,-0.664919f,-1.059451f,-0.523320f,-0.331309f,-1.541040f,0.832153f,0.575408f,-1.696044f,-0.037816f,0.346599f,-1.405308f,0.829612f,-2.436318f,0.448468f,0.098948f,1.479975f,1.937716f,-0.416932f,
0.595499f,-1.009580f,3.264985f,0.394337f,-0.416441f,-3.234207f,0.570600f,1.683120f,-1.863093f,-0.348500f,-0.135841f,1.154019f,0.708959f,-0.593506f,0.702084f,4.441512f,2.558446f,-0.301043f,-0.010167f,-0.584069f,
0.405217f,2.332976f,0.924402f,0.384360f,-0.510962f,0.525336f,-2.175349f,-0.162296f,-0.444141f,-0.005443f,-1.974564f,-2.948007f,1.638133f,0.851314f,-1.513090f,-2.608961f,-1.357912f,-0.724380f,0.234095f,0.913569f,
-0.728644f,0.136749f,4.628799f,-0.708098f,-0.078247f,1.328427f,-0.092943f,-0.925132f,-0.889543f,-0.112554f,-1.543821f,1.187511f,0.240076f,-1.248196f,0.101449f,-0.102250f,-0.929221f,0.600336f,-2.331993f,-0.183217f,
0.603250f,1.375553f,1.780805f,-0.526038f,0.087274f,-1.958042f,3.614745f,-0.612995f,-0.362997f,-2.794758f,-0.013523f,1.154118f,-2.207382f,0.020816f,-0.495762f,0.886372f,0.912156f,-0.761918f,0.270793f,4.614452f,
2.898420f,-0.392095f,0.035857f,-0.572064f,-0.018465f,2.565697f,0.840581f,-0.163192f,-0.491506f,0.154845f,-2.335523f,-0.388690f,-0.696722f,-0.432540f,-2.457009f,-3.060503f,1.412683f,0.289127f,-1.893624f,-2.302324f,
-1.187225f,-0.936404f,-0.185364f,0.787076f,-0.743878f,0.251297f,3.513058f,-0.357567f,-0.333980f,1.421806f,0.446661f,-1.189113f,-1.180673f,0.110796f,-1.566011f,1.549258f,0.166847f,-0.810341f,0.133437f,-0.753545f,
-0.745217f,0.664387f,-1.836398f,-0.772629f,0.982670f,0.519171f,1.545828f,-0.963486f,0.076809f,-2.501331f,3.745407f,-1.013399f,-0.342999f,-2.491077f,-0.683322f,1.711623f,-2.239083f,0.487193f,-0.854742f,0.514228f,
1.655841f,-0.924380f,-0.553212f,4.636106f,2.789750f,-0.490992f,-0.197838f,-0.217216f,-0.388353f,2.538130f,0.981209f,-0.584986f,-1.241250f,-0.206301f,-2.039074f,-0.475903f,-1.246863f,-0.632312f,-2.538312f,-3.214680f,
0.457138f,-0.362815f,-2.157857f,-1.576480f,-0.890682f,-0.749380f,-0.352689f,0.761906f,-0.601512f,0.074493f,2.714587f,0.141425f,-0.688292f,1.271663f,1.235837f,-1.480729f,-1.235561f,0.437016f,-1.385857f,1.591549f,
0.244973f,-0.358199f,0.178999f,-1.227595f,-0.962973f,0.680280f,-1.421991f,-1.355975f,1.242459f,-0.289293f,1.661275f,-0.740861f,-0.486107f,-2.584653f,3.542792f,-0.967234f,-0.425537f,-2.335855f,-1.260888f,2.521090f,
-2.178468f,0.895452f,-0.792649f,0.142939f,1.676539f,-1.193126f,-1.030200f,4.666825f,2.555955f,-0.805862f,-0.630863f,0.334654f,-0.526995f,2.273659f,1.389948f,-0.525317f,-1.870084f,-0.401382f,-1.668568f,-0.474730f,
-2.029544f,-0.381530f,-1.953347f,-3.142336f,-0.503076f,-0.620382f,-2.306184f,-1.133321f,-0.691063f,-0.716500f,-0.122301f,0.275186f,-0.145325f,-0.399444f,2.202090f,0.555993f,-1.171287f,1.542469f,1.798023f,-1.447259f,
-1.034606f,0.473510f,-1.411736f,1.354890f,0.514983f,-0.017057f,0.017354f,-1.314564f,-1.076292f,0.853193f,-0.604305f,-1.598800f,1.089662f,-0.944749f,1.338635f,-0.652562f,-0.633216f,-2.340803f,3.166643f,-0.268834f,
-0.544751f,-2.198280f,-1.573609f,3.590861f,-2.127071f,1.183178f,-0.722976f,-0.064441f,1.603864f,-1.252921f,-0.864857f,4.354602f,2.410637f,-1.283257f,-0.623368f,0.869848f,-0.677081f,1.963955f,1.434000f,-0.307373f,
-2.599895f,-0.185931f,-1.285374f,-0.091998f,-2.255283f,0.128256f,-1.055574f,-3.070423f,-1.230944f,-0.427206f,-2.662571f,-1.051053f,-0.793010f,-0.511229f,0.166470f,0.197306f,0.150035f,-1.070789f,2.443322f,0.690311f,
-1.115430f,1.600732f,1.697810f,-1.233953f,-0.875057f,0.628054f,-0.934104f,0.922952f,1.171834f,-0.106155f,-0.239536f,-1.176803f,-1.192762f,1.103108f,-0.389621f,-1.932975f,0.485840f,-1.265031f,0.625112f,-0.331897f,
-0.660826f,-2.081861f,2.665100f,0.418613f,-0.758579f,-2.514305f,-1.405596f,4.424435f,-2.106982f,1.343117f,-0.425399f,-0.138824f,1.369795f,-1.141878f,-0.396916f,4.278188f,2.138095f,-1.633301f,-0.746120f,1.127965f,
-0.615285f,1.775561f,1.181001f,0.116162f,-2.950857f,0.208083f,-1.324039f,0.040108f,-2.181160f,0.678339f,-0.207839f,-2.940577f,-1.721937f,0.363968f,-2.886661f,-1.640495f,-0.435339f,-0.658667f,0.595409f,-0.116313f,
0.428234f,-1.486934f,3.018867f,0.498577f,-0.757921f,1.466614f,1.695050f,-0.855573f,-0.642951f,0.710389f,-0.770098f,0.112404f,1.919968f,-0.268655f,-0.528021f,-0.627856f,-1.490454f,1.328203f,-0.290963f,-2.252399f,
-0.019745f,-1.224687f,-0.026814f,-0.496215f,-1.070185f,-1.669453f,2.367757f,1.381492f,-0.674681f,-2.691129f,-1.349616f,4.628320f,-1.749266f,1.199176f,-0.209931f,0.069094f,0.985482f,-0.774780f,0.616672f,3.927210f,
2.077878f,-1.731199f,-0.687573f,0.852976f,-0.556006f,1.836048f,0.387788f,0.557871f,-2.717265f,0.274757f,-1.400642f,0.286209f,-1.641390f,1.116194f,0.415540f,-2.920643f,-1.590445f,1.248011f,-2.727966f,-2.115368f,
-0.521044f,-0.887101f,0.966896f,-0.018243f,0.526197f,-1.813002f,3.245429f,0.182610f,-0.484188f,0.943139f,1.315671f,-0.713932f,-0.597690f,0.621231f,-0.874334f,-0.297360f,2.328222f,-0.347253f,-0.752418f,-0.146759f,
-1.719613f,1.491584f,-0.484695f,-2.471534f,-0.581399f,-0.913368f,-0.374587f,-0.604853f,-1.548146f,-1.889714f,2.007983f,2.031061f,-0.654289f,-2.706102f,-1.283365f,4.015550f,-1.844528f,0.871359f,-0.116462f,0.328774f,
0.766276f,-0.474033f,1.473881f,3.790795f,2.262522f,-1.749866f,-0.611473f,0.053660f,-0.218214f,1.996527f,-0.416331f,0.760916f,-1.990681f,0.458532f,-1.498728f,0.031806f,-0.973288f,1.309794f,0.439671f,-3.043602f,
-1.166994f,1.771951f,-2.364655f,-2.663883f,-0.234100f,-1.025933f,1.073094f,0.446648f,0.499878f,-1.419804f,3.169395f,-0.434017f,-0.075749f,0.346501f,0.986243f,-0.726882f,-0.591006f,0.610635f,-0.896703f,-0.503683f,
2.452811f,-0.503854f,-1.100293f,-0.040604f,-1.941281f,1.360505f,-1.043951f,-2.636932f,-0.799470f,-0.692157f,-0.209650f,-1.002664f,-2.252584f,-2.503864f,1.927053f,2.044916f,-0.504649f,-2.594451f,-1.322568f,2.883398f,
-2.124768f,0.894349f,0.119281f,0.099701f,0.761293f,-0.267553f,2.084237f,3.444858f,2.500091f,-1.462085f,-0.710459f,-0.648479f,-0.285096f,1.838196f,-0.819822f,0.741907f,-1.133665f,-0.012948f,-1.468610f,-0.013630f,
-0.603186f,1.296859f,0.147859f,-3.202714f,-0.867342f,1.943665f,-2.076713f,-2.984862f,-0.065779f,-1.433246f,0.967146f,0.735681f,0.523304f,-0.950885f,2.681130f,-0.812594f,0.142981f,0.057324f,0.840480f,-0.678157f,
-0.650531f,0.634354f,-1.029729f,-0.437437f,2.381639f,-0.225708f,-1.271041f,-0.210519f,-1.970012f,1.527928f,-1.579391f,-2.906820f,-0.697754f,-0.756625f,0.180374f,-1.476655f,-3.096086f,-3.502819f,2.017835f,2.181256f,
-0.474283f,-2.173156f,-1.611733f,1.997244f,-2.288635f,0.914887f,-0.061284f,0.084874f,0.802044f,-0.050235f,2.399850f,3.266363f,2.476244f,-1.222571f,-0.990804f,-1.475494f,-0.391468f,1.725942f,-0.588121f,0.724234f,
-0.430081f,-0.568832f,-0.819721f,-0.147331f,-0.351802f,1.471766f,-0.314192f,-3.515282f,-0.602106f,1.709235f,-1.742474f,-2.917816f,-0.163855f,-1.405400f,0.688384f,0.920849f,0.565158f,-0.399900f,2.015486f,-0.785419f,
-0.123109f,-0.259074f,0.951922f,-1.023490f,-0.167557f,0.408475f,-1.186152f,-0.061022f,1.958425f,-0.147589f,-1.409361f,-0.532248f,-1.941820f,1.761009f,-1.904079f,-3.032390f,-0.613892f,-0.774554f,1.198085f,-1.393334f,
-3.492778f,-4.103634f,2.155972f,1.826634f,-0.443833f,-1.899260f,-1.928344f,1.387084f,-2.699950f,0.988586f,-0.360604f,0.039946f,1.084687f,0.055908f,2.158593f,2.898452f,2.157865f,-0.920942f,-1.475014f,-1.905582f,
-0.212579f,1.320070f,0.020261f,0.950572f,-0.119515f,-1.411014f,-0.103558f,0.057610f,-0.465406f,1.886317f,-0.650927f,-3.623327f,-0.546151f,1.249012f,-1.274106f,-2.433307f,-0.232244f,-1.358870f,0.494029f,1.135460f,
0.477134f,0.047485f,1.441766f,-0.904024f,-0.507561f,-0.211700f,1.278604f,-1.249002f,-0.030011f,0.320278f,-1.263181f,0.458127f,1.650431f,0.076637f,-1.606047f,-0.770303f,-1.796484f,1.907091f,-2.130211f,-2.845921f,
-0.497223f,-0.945306f,2.410427f,-1.072625f,-3.540381f,-4.407775f,2.588820f,1.294328f,-0.614280f,-1.719770f,-1.984859f,1.184956f,-2.881676f,1.391938f,-0.430563f,-0.066281f,1.150841f,0.193547f,1.689468f,2.540226f,
1.749703f,-0.462640f,-1.767696f,-1.841151f,-0.094909f,1.028285f,1.087975f,1.109347f,-0.059548f,-1.627886f,0.735363f,0.099574f,-0.657922f,1.969764f,-1.050948f,-3.796517f,-0.829749f,0.541100f,-0.746146f,-1.979619f,
-0.488589f,-1.266459f,0.490288f,1.237926f,0.305501f,0.391805f,0.983739f,-0.519829f,-0.926229f,0.084170f,1.826814f,-1.556108f,0.156573f,0.187471f,-1.045328f,0.815126f,1.808286f,-0.029427f,-1.205353f,-0.735625f,
-1.628103f,2.318746f,-1.972641f,-2.552112f,-0.623269f,-0.893316f,3.222579f,-0.571598f,-3.236855f,-4.111458f,2.721055f,0.853224f,-0.353254f,-1.717724f,-2.154186f,1.408571f,-2.984536f,1.843340f,-0.260247f,-0.077390f,
1.346919f,0.090493f,0.921823f,2.122856f,1.125203f,-0.374113f,-1.922825f,-1.458612f,0.168726f,0.580552f,2.114465f,1.299016f,-0.222583f,-1.709043f,1.548175f,0.301652f,-0.851640f,2.545049f,-1.113884f,-3.686351f,
-1.023734f,0.101525f,-0.156124f,-1.496867f,-0.730085f,-1.021438f,0.704952f,0.912868f,0.573871f,0.439880f,0.755306f,-0.463339f,-1.375772f,0.562505f,1.971686f,-1.554511f,0.126117f,-0.066575f,-0.889470f,0.958163f,
1.860985f,0.019691f,-1.265872f,-0.484736f,-1.016158f,2.307876f,-2.005797f,-1.953226f,-0.966972f,-0.943944f,3.583584f,-0.103551f,-2.537005f,-3.667413f,2.750025f,0.691330f,-0.484407f,-1.734625f,-2.225974f,2.019528f,
-3.033890f,1.898133f,0.022181f,-0.192228f,1.374771f,0.041007f,0.645869f,1.986331f,0.596293f,-0.168746f,-2.035864f,-0.666787f,0.225101f,0.636869f,3.138620f,1.711815f,-0.916992f,-1.244724f,1.612941f,0.487972f,
-0.560772f,2.774688f,-1.000833f,-3.348092f,-1.199937f,0.025566f,0.113835f,-1.005828f,-1.162434f,-0.637508f,1.088380f,0.724114f,0.753861f,0.305662f,0.963139f,-0.472630f,-1.258093f,1.062412f,1.981361f,-1.472282f,
-0.166109f,-0.220795f,-0.714333f,1.064836f,2.065327f,0.005004f,-1.110145f,-0.021855f,-0.445863f,2.561841f,-1.741771f,-1.339867f,-1.441818f,-0.629338f,3.876242f,0.226696f,-1.675463f,-2.974020f,2.745931f,0.587050f,
-0.326521f,-2.279591f,-2.025241f,2.760811f,-2.599287f,1.608624f,0.330261f,0.058784f,1.092476f,-0.178642f,0.737423f,2.009060f,0.166451f,-0.268835f,-1.802152f,-0.080773f,0.244596f,0.792043f,3.513873f,1.963867f,
-1.345979f,-0.570335f,1.525086f,0.395800f,0.063811f,3.453561f,-0.891812f,-3.048509f,-1.357687f,0.411513f,0.238892f,-0.996092f,-1.410909f,-0.535326f,1.577640f,0.613954f,0.954014f,0.429601f,1.576507f,-0.820499f,
-1.143945f,1.274034f,2.033430f,-1.147564f,-0.382256f,-0.401871f,-0.569057f,0.740071f,2.351769f,0.087909f,-1.173805f,0.383709f,0.087774f,2.460459f,-1.551220f,-0.963189f,-1.853743f,0.060508f,3.533320f,0.358497f,
-0.921888f,-2.394482f,2.707126f,1.138494f,-0.333050f,-2.741444f,-2.079527f,3.123630f,-2.478171f,1.313102f,0.852821f,0.417359f,0.880274f,-0.560303f,1.314487f,2.220143f,0.162149f,-0.185358f,-1.357249f,0.593599f,
0.368331f,1.294696f,3.052116f,2.081593f,-1.642844f,0.663777f,0.830812f,0.515914f,0.724093f,3.870692f,-0.970329f,-2.602365f,-1.185668f,0.887814f,-0.128602f,-1.259460f,-1.494563f,-0.594072f,1.777392f,1.034602f,
1.212899f,0.183479f,2.463353f,-1.017282f,-0.674178f,1.302025f,1.649860f,-0.928800f,-0.667275f,-0.561793f,-0.433325f,0.548484f,2.519567f,0.189216f,-1.095076f,0.847926f,0.685630f,2.629259f,-1.675557f,-0.498205f,
-2.035980f,0.987465f,2.830531f,0.220970f,-0.099198f,-2.342836f,2.776087f,1.292145f,-0.269791f,-2.863072f,-1.951861f,2.943460f,-2.170686f,0.874299f,0.905789f,0.663356f,0.745942f,-0.850300f,1.836120f,2.365774f,
0.327872f,-0.164202f,-0.724027f,0.695719f,0.261143f,2.172264f,2.499375f,1.926551f,-1.469584f,1.568305f,0.044257f,0.413843f,1.798757f,3.594389f,-1.241142f,-2.311251f,-0.758570f,1.301501f,-0.280827f,-1.404076f,
-1.393269f,-0.695868f,1.717826f,1.267933f,1.007162f,0.181677f,2.835279f,-1.468439f,-0.204931f,1.098796f,1.376800f,-0.350251f,-1.125074f,-0.653984f,-0.560731f,0.227752f,2.512451f,0.352224f,-1.148981f,1.096788f,
1.364277f,2.093409f,-1.492779f,-0.704707f,-1.783330f,1.564512f,2.357286f,-0.163205f,-0.041335f,-2.568297f,3.013580f,1.152882f,0.070396f,-2.965445f,-2.084179f,2.425515f,-2.058796f,0.602932f,0.786804f,0.755965f,
0.759274f,-1.230752f,2.086352f,2.434320f,0.528744f,0.057768f,-0.067139f,0.341859f,0.131062f,2.775565f,1.627384f,1.531873f,-0.910770f,1.960719f,-0.911687f,0.402569f,2.022348f,3.207890f,-1.955707f,-2.097691f,
-0.388290f,1.150741f,-0.777661f,-1.888728f,-0.988336f,-1.039760f,1.158350f,1.523342f,0.617517f,0.184841f,2.654290f,-1.320776f,0.188920f,0.794078f,1.679247f,-0.261267f,-1.298305f,-0.256721f,-0.521541f,0.574194f,
2.272498f,0.567433f,-1.044089f,0.779034f,1.520049f,1.926803f,-1.519991f,-1.298282f,-1.445885f,1.888894f,1.657115f,-0.628251f,-0.247878f,-2.873124f,3.433298f,0.572529f,0.098111f,-2.874332f,-2.391339f,1.883951f,
-2.047972f,0.407641f,0.633288f,0.716686f,0.849623f,-1.244116f,1.929715f,2.552782f,1.045172f,0.241893f,0.592207f,0.164110f,0.080972f,3.146627f,0.862401f,0.889874f,-0.298819f,1.845518f,-1.348239f,0.192060f,
1.960151f,2.654607f,-2.414525f,-2.121115f,-0.256710f,0.857610f,-1.275965f,-1.837196f,-0.634472f,-1.319409f,0.636366f,1.459757f,0.512883f,0.199549f,1.924359f,-1.303469f,0.149845f,0.419313f,2.253854f,-0.062084f,
-1.481413f,-0.026554f,-0.551772f,0.892318f,1.902181f,1.072453f,-0.957058f,0.104288f,1.859616f,1.735542f,-1.277680f,-1.840135f,-0.817040f,1.646904f,1.585769f,-0.815129f,-0.676398f,-3.099463f,3.726607f,-0.372310f,
0.259224f,-2.674835f,-2.719924f,1.445394f,-2.101195f,0.592991f,0.337965f,0.436111f,1.210897f,-1.512262f,1.361382f,2.507501f,1.415473f,0.464701f,0.858977f,-0.059936f,-0.280037f,3.467301f,0.717612f,0.344982f,
-0.164033f,1.671431f,-1.504221f,-0.047793f,1.366757f,2.087490f,-2.900816f,-2.005320f,-0.366699f,0.281432f,-1.470543f,-1.538353f,-0.379129f,-1.449714f,-0.052333f,1.427269f,0.435665f,0.385601f,0.913170f,-0.840145f,
0.047876f,0.097046f,2.796402f,-0.007411f,-1.587887f,0.002396f,-0.604445f,1.207822f,1.474012f,1.652440f,-0.893636f,-0.539226f,1.894731f,1.438412f,-1.023198f,-2.184883f,-0.344751f,1.118330f,1.400948f,-1.028680f,
-1.098301f,-3.087261f,3.902269f,-1.136550f,0.250036f,-2.474576f,-3.238712f,1.601065f,-2.063343f,0.838820f,0.212450f,0.317283f,1.556020f,-1.485632f,0.964612f,2.360764f,1.442613f,0.421405f,0.881939f,-0.209939f,
-0.510937f,3.375963f,0.748776f,-0.334798f,-0.064298f,1.090770f,-1.165418f,-0.030656f,0.478640f,1.767668f,-2.739782f,-1.998815f,-1.107808f,-0.452864f,-1.635523f,-1.003486f,-0.174186f,-1.522695f,-0.186804f,1.207331f,
0.493140f,0.144999f,-0.222874f,-0.233562f,-0.496643f,0.073676f,3.609217f,-0.202806f,-1.451022f,0.246088f,-0.603424f,1.362722f,1.577634f,2.090307f,-0.879202f,-1.097977f,1.494175f,1.640767f,-0.391689f,-2.697430f,
-0.145288f,0.240298f,1.568114f,-0.943833f,-1.556574f,-2.728057f,3.646966f,-1.366815f,-0.128026f,-2.392371f,-3.545567f,2.334149f,-2.289442f,1.176538f,0.158233f,-0.040182f,1.951048f,-1.749927f,0.429292f,2.357294f,
0.952119f,0.355613f,0.838771f,-0.028418f,-0.429495f,3.076561f,1.233523f,-0.299497f,-0.772998f,1.209842f,-0.574273f,-0.184959f,-0.364960f,1.918535f,-2.187605f,-1.848767f,-1.908603f,-0.902788f,-1.788273f,-0.445442f,
-0.094681f,-1.650601f,-0.117239f,0.954072f,1.059666f,0.054326f,-0.686901f,0.247481f,-0.831650f,0.196489f,4.337635f,-0.349424f,-1.316218f,0.495413f,-0.278384f,1.213749f,1.683379f,2.289916f,-1.016611f,-1.136641f,
1.098837f,1.797354f,0.131460f,-3.081089f,-0.186820f,-0.400288f,1.371779f,-0.542202f,-1.562895f,-2.091071f,3.293477f,-0.948614f,-0.279915f,-2.523901f,-3.582878f,3.213394f,-2.196579f,1.198886f,0.436067f,0.015546f,
1.926955f,-1.614940f,0.504109f,2.051097f,0.827620f,-0.021385f,0.748872f,0.289355f,-0.592173f,2.575249f,1.133049f,0.074996f,-1.331059f,1.123577f,-0.026591f,0.202817f,-1.060283f,2.291692f,-1.156798f,-1.610650f,
-2.608759f,-0.824685f,-1.867636f,-0.562028f,-0.069890f,-1.344451f,0.265331f,0.528309f,1.555784f,-0.637663f,-0.675441f,0.419468f,-0.980600f,0.094121f,4.443370f,-0.188098f,-0.895777f,0.378142f,-0.220506f,0.659704f,
2.113017f,2.115476f,-1.245904f,-0.848169f,0.576668f,2.049040f,0.525147f,-3.104277f,-0.680133f,-0.821544f,1.021392f,-0.127969f,-1.648816f,-1.190935f,2.852194f,-0.406272f,-0.455697f,-2.721068f,-3.255098f,4.210104f,
-2.123509f,1.242028f,0.659113f,-0.114942f,1.504583f,-1.638051f,0.798391f,1.775580f,0.532255f,-0.050058f,0.922501f,0.395346f,-0.444418f,2.210764f,0.858919f,0.659591f,-1.638923f,1.434422f,0.247193f,0.433347f,
-1.286531f,2.581034f,-0.303811f,-1.418752f,-2.953948f,-0.314045f,-1.928038f,-0.677040f,0.009508f,-1.481300f,0.582254f,0.461405f,1.981324f,-1.124217f,-0.042939f,0.553109f,-0.646953f,-0.035439f,4.367580f,0.019698f,
-0.693063f,0.415664f,0.050491f,-0.093079f,2.731276f,1.989692f,-1.606562f,-0.419021f,0.220400f,2.229404f,0.513167f,-3.159207f,-1.226849f,-0.829060f,0.402539f,-0.128350f,-1.808280f,-0.574765f,2.338414f,0.413459f,
-0.458203f,-2.894416f,-2.855503f,4.449203f,-1.886721f,1.189012f,1.071548f,0.160975f,1.121881f,-1.275609f,1.562966f,1.480440f,0.607674f,-0.365419f,1.111090f,0.078551f,-0.157653f,2.188785f,0.330315f,1.083872f,
-1.403549f,1.708374f,0.165015f,0.589617f,-1.104272f,2.740557f,0.564516f,-1.239356f,-2.831029f,0.527278f,-1.958396f,-1.420131f,0.176329f,-1.621040f,0.911030f,0.559530f,2.151397f,-1.349874f,0.488514f,0.327815f,
-0.352224f,-0.399758f,3.953106f,0.304895f,-0.427653f,0.423664f,0.019161f,-0.421012f,3.136018f,1.571480f,-1.818517f,-0.035910f,-0.052118f,2.189329f,0.192568f,-3.285937f,-1.648830f,-0.678472f,0.062811f,-0.029155f,
-2.049654f,-0.173936f,2.034174f,0.950653f,-0.421226f,-3.171606f,-2.360625f,4.186234f,-2.088637f,0.676991f,1.349772f,0.393094f
};
std::vector<float> input1_data2={
-0.082356f,-0.282336f,-0.059442f,-0.060488f,-0.250684f,-0.111670f,-0.275121f,-0.182875f,-0.271705f,-0.195441f,-0.276596f,-0.042281f,-0.070622f,-0.082708f,-0.108323f,-0.122374f,-0.125447f,-0.096911f,-0.036306f,0.007458f,
0.036428f,0.056826f,0.008089f,-0.013319f,-0.032795f,-0.023781f,0.017309f,0.025204f,0.052908f,0.052227f,0.053296f,0.040591f,0.030536f,0.048142f,0.101641f,0.152852f,0.208406f,0.244973f,0.241214f,0.210360f,
0.162787f,0.159638f,0.167024f,0.194899f,0.204305f,0.238391f,0.233994f,0.214710f,0.200458f,0.186915f,0.201731f,0.235945f,0.291974f,0.338469f,0.347507f,0.334722f,0.294201f,0.277242f,0.232374f,0.244192f,
0.248963f,0.275719f,0.281326f,0.279040f,0.261530f,0.261661f,0.239484f,0.250111f,0.295128f,0.328810f,0.385023f,0.418028f,0.405382f,0.370987f,0.315897f,0.276649f,0.260504f,0.271964f,0.253270f,0.255543f,
0.238337f,0.211003f,0.193798f,0.173703f,0.154594f,0.146867f,0.200896f,0.240775f,0.271680f,0.272413f,0.279890f,0.228227f,0.195799f,0.154307f,0.173422f,0.176744f,0.218916f,0.215956f,0.221244f,0.209872f,
0.175083f,0.177856f,0.195554f,0.208893f,0.280657f,0.312005f,0.363425f,0.369934f,0.330872f,0.285323f,0.251416f,0.245414f,0.270857f,0.269769f,0.308196f,0.294266f,0.292012f,0.284923f,0.259316f,0.231741f,
0.244958f,0.243122f,0.288212f,0.354783f,0.376992f,0.363197f,0.332615f,0.288738f,-0.126164f,-0.049956f,0.003127f,0.032219f,-0.046234f,-0.134013f,-0.067247f,0.066222f,0.116443f,0.030593f,0.180139f,0.132993f,
0.143412f,0.123831f,0.106930f,0.058609f,0.002467f,-0.024841f,-0.049584f,-0.059419f,-0.088573f,-0.099707f,-0.062229f,-0.009158f,0.058071f,0.105261f,0.156301f,0.190850f,0.207885f,0.230935f,0.206604f,0.184059f,
0.131772f,0.095185f,0.081346f,0.066072f,0.056403f,0.050343f,0.053517f,0.096331f,0.147775f,0.197800f,0.252029f,0.277445f,0.303414f,0.305592f,0.302795f,0.280933f,0.236182f,0.170075f,0.126740f,0.088235f,
0.069878f,0.052995f,0.027349f,0.017690f,0.040881f,0.085064f,0.142939f,0.181454f,0.212993f,0.230129f,0.236288f,0.231131f,0.218487f,0.160979f,0.113177f,0.059518f,0.025617f,-0.008003f,-0.007239f,-0.010533f,
-0.012296f,0.010282f,0.061384f,0.108818f,0.134168f,0.164973f,0.180391f,0.204317f,0.207414f,0.224149f,0.188962f,0.139403f,0.080034f,0.051265f,0.038998f,0.015192f,0.015392f,-0.006753f,0.024862f,0.057522f,
0.096487f,0.161160f,0.223315f,0.249164f,0.263169f,0.307001f,0.298631f,0.305970f,0.270870f,0.213105f,0.169561f,0.120674f,0.115958f,0.081863f,0.077853f,0.060586f,0.068076f,0.097560f,0.152132f,0.201284f,
0.232558f,0.262626f,0.255853f,0.267062f,0.257224f,0.221600f,0.172710f,0.105136f,0.059631f,0.009580f,-0.004318f,-0.002640f,-0.013509f,0.002524f,-0.006057f,0.021367f,0.083373f,-0.062661f,-0.172032f,-0.207567f,
-0.247666f,-0.201509f,-0.142504f,-0.112123f,-0.183787f,0.061621f,-0.080701f,-0.042326f,-0.076358f,-0.093285f,-0.047990f,0.001251f,0.017129f,0.018760f,-0.028445f,-0.109527f,-0.140557f,-0.116035f,-0.033360f,0.077135f,
0.184324f,0.226509f,0.231803f,0.187141f,0.119517f,0.075720f,0.060720f,0.086469f,0.116624f,0.111068f,0.090990f,0.003263f,-0.057440f,-0.078669f,-0.058137f,0.006349f,0.121707f,0.216208f,0.268612f,0.262314f,
0.231459f,0.169863f,0.106829f,0.087212f,0.144778f,0.199562f,0.211835f,0.201748f,0.139262f,0.076707f,0.072191f,0.119140f,0.204086f,0.318428f,0.422841f,0.477305f,0.491737f,0.457427f,0.404763f,0.365470f,
0.341855f,0.371181f,0.413579f,0.414446f,0.399273f,0.334209f,0.231754f,0.197514f,0.185056f,0.219282f,0.286526f,0.338285f,0.367068f,0.340334f,0.284076f,0.199115f,0.093039f,0.026564f,-0.000886f,0.014369f,
0.000413f,-0.022444f,-0.084675f,-0.208673f,-0.283693f,-0.338252f,-0.314638f,-0.230048f,-0.137510f,-0.069440f,-0.051293f,-0.063648f,-0.120657f,-0.198571f,-0.255440f,-0.284094f,-0.275437f,-0.229307f,-0.216582f,-0.207322f,
-0.263961f,-0.351345f,-0.397866f,-0.395114f,-0.324400f,-0.224317f,-0.095774f,-0.009309f,0.005547f,-0.000780f,-0.034101f,-0.071086f,-0.125704f,-0.114440f,-0.090425f,-0.020683f,0.008487f,0.005565f,-0.019502f,-0.109339f,
-0.146078f,-0.164726f,-0.111348f,-0.032059f,0.090732f,0.253275f,-0.139521f,0.163478f,-0.271035f,0.117486f,0.247700f,-0.085015f,-0.004459f,0.034771f,-0.088502f,-0.492348f,-0.525423f,-0.530995f,-0.502803f,-0.474916f,
-0.453128f,-0.437414f,-0.423469f,-0.403783f,-0.372441f,-0.371254f,-0.389540f,-0.399652f,-0.388842f,-0.390322f,-0.414728f,-0.414447f,-0.429385f,-0.405718f,-0.362372f,-0.289770f,-0.225385f,-0.152869f,-0.102768f,-0.048618f,
-0.009170f,0.015827f,0.031343f,0.035110f,0.016521f,-0.002772f,-0.017775f,-0.060302f,-0.083743f,-0.114279f,-0.148257f,-0.148498f,-0.118422f,-0.089236f,-0.068016f,-0.052611f,-0.037741f,-0.060541f,-0.067362f,-0.099032f,
-0.141089f,-0.179344f,-0.225203f,-0.268051f,-0.317011f,-0.378762f,-0.428635f,-0.471499f,-0.458947f,-0.437722f,-0.416254f,-0.364906f,-0.333242f,-0.314041f,-0.299005f,-0.275825f,-0.282035f,-0.310348f,-0.326830f,-0.343405f,
-0.374041f,-0.396650f,-0.428721f,-0.466809f,-0.494940f,-0.486957f,-0.472804f,-0.425449f,-0.372413f,-0.321704f,-0.282864f,-0.243416f,-0.204535f,-0.167785f,-0.163868f,-0.158156f,-0.132151f,-0.104888f,-0.113592f,-0.108332f,
-0.139515f,-0.142874f,-0.158011f,-0.145491f,-0.082350f,-0.030759f,0.029044f,0.054946f,0.072291f,0.104236f,0.084241f,0.074631f,0.038727f,0.015205f,-0.022983f,-0.053927f,-0.103423f,-0.149978f,-0.197967f,-0.256215f,
-0.294328f,-0.303999f,-0.283495f,-0.274812f,-0.263799f,-0.254356f,-0.245945f,-0.254473f,-0.294114f,-0.321834f,-0.373313f,-0.409294f,0.130128f,-0.156792f,-0.129972f,-0.133448f,-0.057645f,-0.104832f,-0.153742f,0.160025f,
0.185014f,0.081426f,0.032990f,-0.009224f,-0.070570f,-0.077600f,-0.092404f,-0.070031f,-0.024072f,0.028748f,0.084031f,0.102242f,0.093398f,0.081692f,0.099211f,0.127495f,0.135273f,0.167319f,0.162892f,0.139462f,
0.098410f,0.054349f,0.018269f,0.016383f,0.068030f,0.103299f,0.167905f,0.185934f,0.204649f,0.174037f,0.178836f,0.161072f,0.160212f,0.155145f,0.172149f,0.148627f,0.100541f,0.045058f,-0.032110f,-0.105180f,
-0.120951f,-0.113422f,-0.085586f,-0.066152f,-0.051574f,-0.065094f,-0.099258f,-0.118169f,-0.125071f,-0.122638f,-0.122451f,-0.136394f,-0.147542f,-0.192589f,-0.258001f,-0.351748f,-0.399155f,-0.433987f,-0.461434f,-0.431076f,
-0.401018f,-0.380745f,-0.350613f,-0.366495f,-0.383929f,-0.370028f,-0.366758f,-0.362332f,-0.330434f,-0.322072f,-0.321871f,-0.348162f,-0.389949f,-0.415599f,-0.420211f,-0.389192f,-0.330204f,-0.249356f,-0.183682f,-0.130377f,
-0.081113f,-0.064662f,-0.050043f,0.013187f,0.054156f,0.127625f,0.172192f,0.191487f,0.183562f,0.143268f,0.091944f,0.047148f,0.015818f,0.018451f,0.071295f,0.126781f,0.133862f,0.157192f,0.119196f,0.076302f,
0.072694f,0.045514f,0.043252f,0.039258f,0.027017f,-0.009775f,-0.078246f,-0.135924f,-0.215345f,-0.285060f,-0.323996f,-0.301448f,-0.265307f,-0.202391f,-0.160480f,-0.180060f,-0.186976f,-0.242170f,-0.255096f,-0.272791f,
0.269250f,0.040452f,0.120401f,-0.043380f,-0.040485f,-0.118796f,-0.165495f,-0.192039f,0.122581f,-0.137382f,-0.049656f,-0.078425f,-0.047370f,-0.025051f,-0.002243f,0.038602f,0.055766f,0.097832f,0.124322f,0.118602f,
0.078027f,0.027128f,-0.003400f,0.000084f,0.013961f,0.021457f,0.058393f,0.088119f,0.108380f,0.106013f,0.093087f,0.084875f,0.056024f,0.052762f,0.047701f,0.007712f,-0.047999f,-0.103602f,-0.198899f,-0.269286f,
-0.280861f,-0.304286f,-0.264634f,-0.248443f,-0.220778f,-0.211829f,-0.221209f,-0.221895f,-0.232872f,-0.227378f,-0.207999f,-0.180831f,-0.171559f,-0.197790f,-0.238298f,-0.278547f,-0.326997f,-0.306456f,-0.272494f,-0.224318f,
-0.171446f,-0.132293f,-0.084219f,-0.064782f,-0.039940f,-0.037788f,-0.034494f,-0.034510f,-0.035943f,-0.033453f,-0.085182f,-0.145484f,-0.193496f,-0.260341f,-0.283736f,-0.304286f,-0.302385f,-0.266450f,-0.234800f,-0.207979f,
-0.181711f,-0.183369f,-0.185039f,-0.188013f,-0.184736f,-0.178963f,-0.158714f,-0.193394f,-0.207604f,-0.295757f,-0.345408f,-0.398760f,-0.409581f,-0.390864f,-0.395870f,-0.363472f,-0.350559f,-0.338257f,-0.354333f,-0.362624f,
-0.402425f,-0.436505f,-0.442070f,-0.414381f,-0.445437f,-0.478792f,-0.519304f,-0.563491f,-0.634047f,-0.663046f,-0.644383f,-0.636324f,-0.583986f,-0.545927f,-0.498522f,-0.453750f,-0.435402f,-0.416439f,-0.407798f,-0.381041f,
-0.369191f,-0.323511f,-0.298263f,-0.309718f,-0.310002f,-0.355517f,-0.368513f,-0.367858f,0.287210f,-0.034476f,-0.030634f,-0.091017f,-0.080509f,-0.081722f,0.006855f,-0.096538f,-0.177083f,-0.038042f,-0.082128f,-0.221479f,
-0.244357f,-0.236030f,-0.198507f,-0.114150f,-0.052131f,0.009122f,0.053317f,0.091594f,0.100166f,0.106352f,0.128066f,0.140091f,0.166639f,0.145711f,0.087694f,0.049130f,-0.009373f,-0.042156f,-0.050779f,-0.037576f,
-0.032080f,-0.009054f,-0.009758f,-0.015160f,-0.052980f,-0.101847f,-0.130272f,-0.182952f,-0.207769f,-0.257514f,-0.293630f,-0.382247f,-0.452847f,-0.518188f,-0.568105f,-0.573422f,-0.546992f,-0.521179f,-0.470197f,-0.419758f,
-0.393528f,-0.379387f,-0.374226f,-0.361457f,-0.346264f,-0.303131f,-0.275606f,-0.276615f,-0.292422f,-0.309258f,-0.313854f,-0.306866f,-0.256828f,-0.197712f,-0.117736f,-0.045631f,0.025186f,0.074042f,0.089668f,0.061416f,
0.074533f,0.076448f,0.098993f,0.111343f,0.100188f,0.063733f,0.050074f,0.006657f,-0.008117f,-0.007739f,0.036292f,0.062802f,0.130181f,0.167963f,0.223375f,0.214066f,0.222435f,0.160710f,0.120460f,0.103691f,
0.073302f,0.048677f,-0.027198f,-0.107883f,-0.203491f,-0.295631f,-0.341762f,-0.390082f,-0.405305f,-0.407171f,-0.380853f,-0.362205f,-0.364214f,-0.404773f,-0.434770f,-0.469493f,-0.483518f,-0.487558f,-0.469810f,-0.487129f,
-0.526393f,-0.590927f,-0.633758f,-0.664348f,-0.649845f,-0.628788f,-0.561206f,-0.490513f,-0.393785f,-0.332747f,-0.233202f,-0.232130f,-0.204172f,-0.212824f,-0.213022f,-0.164833f,-0.192984f,-0.092809f,-0.006621f,-0.026059f,
0.137814f,0.155664f,0.232537f,0.175360f,0.153117f,-0.071456f,-0.056977f,-0.096726f,-0.027311f,0.089847f,0.177564f,0.224784f,0.222494f,0.216535f,0.212771f,0.249125f,0.268520f,0.288970f,0.260949f,0.189876f,
0.059703f,-0.066726f,-0.163601f,-0.229824f,-0.210585f,-0.137831f,-0.054890f,0.002311f,0.047938f,0.049282f,0.032305f,0.068867f,0.106549f,0.150562f,0.186312f,0.203593f,0.141071f,0.056428f,-0.056817f,-0.106888f,
-0.109320f,-0.060167f,0.040728f,0.157982f,0.255436f,0.293522f,0.293828f,0.282146f,0.283045f,0.334111f,0.372636f,0.407183f,0.395371f,0.318776f,0.180381f,0.062383f,-0.030537f,-0.088886f,-0.093542f,-0.051448f,
-0.001615f,0.060139f,0.056089f,0.040047f,-0.016790f,-0.041272f,-0.053151f,-0.022231f,-0.018882f,-0.037595f,-0.102170f,-0.175489f,-0.279889f,-0.366377f,-0.437050f,-0.455843f,-0.410070f,-0.331343f,-0.242472f,-0.155750f,
-0.131879f,-0.143571f,-0.128965f,-0.076430f,-0.034959f,0.051964f,0.117111f,0.128465f,0.094242f,0.036275f,-0.037288f,-0.095228f,-0.116151f,-0.066728f,0.024939f,0.167112f,0.273273f,0.314528f,0.357013f,0.321470f,
0.341437f,0.392014f,0.452108f,0.498973f,0.559342f,0.507964f,0.434903f,0.339186f,0.253469f,0.195393f,0.202536f,0.210997f,0.301343f,0.405351f,0.493229f,0.533896f,0.526168f,0.460738f,0.443376f,0.400815f,
0.431315f,0.432834f,0.441104f,0.398302f,0.019328f,-0.131473f,-0.200267f,-0.212126f,-0.189267f,-0.170122f,-0.158075f,-0.255541f,-0.084184f,-0.131613f,-0.130573f,-0.217785f,-0.174028f,-0.152502f,-0.153806f,-0.137930f,
-0.132800f,-0.162013f,-0.195937f,-0.221814f,-0.233277f,-0.218617f,-0.175907f,-0.159181f,-0.120369f,-0.095618f,-0.060666f,-0.027316f,0.010791f,0.023779f,0.047181f,0.058426f,0.049806f,0.046419f,0.011892f,-0.046884f,
-0.109658f,-0.137185f,-0.127250f,-0.137304f,-0.122042f,-0.110534f,-0.110102f,-0.114612f,-0.091430f,-0.079355f,-0.076789f,-0.063792f,-0.076653f,-0.083571f,-0.076760f,-0.118008f,-0.185857f,-0.228579f,-0.237993f,-0.220452f,
-0.194441f,-0.174512f,-0.138551f,-0.128853f,-0.109983f,-0.057290f,-0.011240f,0.009497f,0.048926f,0.055198f,0.077629f,0.065762f,0.046098f,0.001415f,-0.044434f,-0.044100f,-0.028598f,-0.022425f,0.008813f,0.035726f,
0.043246f,0.061876f,0.102207f,0.135450f,0.156234f,0.180801f,0.199379f,0.216366f,0.223450f,0.223760f,0.199370f,0.142985f,0.112870f,0.096929f,0.113552f,0.161789f,0.151889f,0.146814f,0.162563f,0.150787f,
0.168606f,0.175979f,0.170678f,0.168524f,0.168908f,0.167708f,0.128220f,0.103326f,0.034268f,-0.034289f,-0.099002f,-0.106593f,-0.077612f,-0.069316f,-0.066365f,-0.073915f,-0.070691f,-0.048112f,-0.022323f,-0.015731f,
0.007026f,0.033699f,0.034075f,0.029898f,0.049427f,0.012279f,-0.011145f,-0.040849f,-0.094533f,-0.077126f,-0.072620f,-0.019712f,-0.107461f,-0.162892f,-0.320152f,-0.234761f,-0.115589f,-0.211470f,-0.225991f,-0.223228f,
-0.226961f,-0.163506f,-0.244777f,-0.218050f,-0.248981f,-0.238532f,-0.164309f,-0.079036f,0.025317f,0.094531f,0.148209f,0.163201f,0.150129f,0.138152f,0.129319f,0.119253f,0.094924f,0.070514f,0.013715f,-0.043902f,
-0.080448f,-0.113061f,-0.120861f,-0.047397f,0.000034f,0.071758f,0.139689f,0.152994f,0.157686f,0.109094f,0.079076f,0.053135f,0.017146f,0.007680f,-0.013809f,-0.058321f,-0.116486f,-0.160144f,-0.157607f,-0.150942f,
-0.081944f,0.018062f,0.082877f,0.163179f,0.200588f,0.211094f,0.201952f,0.175457f,0.192417f,0.171389f,0.160443f,0.155374f,0.104507f,0.063417f,0.019213f,0.001715f,0.006703f,0.066220f,0.130992f,0.227009f,
0.293191f,0.315206f,0.309259f,0.290955f,0.271124f,0.232088f,0.193999f,0.181214f,0.160796f,0.130576f,0.071551f,0.015726f,-0.025034f,-0.024286f,0.000203f,0.056599f,0.149483f,0.236547f,0.292624f,0.327891f,
0.315608f,0.279976f,0.267000f,0.269611f,0.252894f,0.244749f,0.218900f,0.175613f,0.100917f,0.034651f,0.028799f,0.015867f,0.043046f,0.128228f,0.204585f,0.277688f,0.319280f,0.298694f,0.293655f,0.257249f,
0.265271f,0.235921f,0.225359f,0.239103f,0.207137f,0.164082f,0.127242f,0.102040f,0.071496f,0.092119f,0.159808f,0.226561f,0.335088f,0.428003f,0.470777f,0.461480f,0.442287f,0.388606f,0.350651f,0.338227f,
0.279727f,-0.072135f,-0.110317f,-0.085990f,-0.044458f,-0.156965f,-0.294494f,-0.301641f,-0.390359f,-0.172917f,-0.149238f,-0.286034f,-0.152909f,-0.041240f,-0.027931f,-0.052771f,-0.118395f,-0.141203f,-0.152839f,-0.112775f,
-0.107484f,-0.128066f,-0.195970f,-0.268220f,-0.309203f,-0.336057f,-0.278781f,-0.176714f,-0.040227f,0.063758f,0.144896f,0.138181f,0.092085f,0.013208f,-0.008511f,-0.006454f,0.011183f,-0.002796f,-0.015699f,-0.094331f,
-0.153524f,-0.196581f,-0.205226f,-0.142099f,-0.047769f,0.097633f,0.208133f,0.283763f,0.249749f,0.195213f,0.113078f,0.072208f,0.063813f,0.073419f,0.079792f,0.039595f,-0.035673f,-0.128097f,-0.182839f,-0.203371f,
-0.169026f,-0.084396f,0.025055f,0.120572f,0.169549f,0.133920f,0.063630f,-0.034454f,-0.128433f,-0.161278f,-0.193056f,-0.219111f,-0.254613f,-0.366499f,-0.459379f,-0.571917f,-0.623981f,-0.621789f,-0.591206f,-0.493226f,
-0.402661f,-0.327978f,-0.283181f,-0.328008f,-0.413061f,-0.489390f,-0.533396f,-0.522570f,-0.476254f,-0.450309f,-0.475838f,-0.495370f,-0.549434f,-0.566258f,-0.567804f,-0.501494f,-0.375045f,-0.262209f,-0.131611f,-0.042440f,
-0.029692f,-0.085446f,-0.177649f,-0.225692f,-0.258394f,-0.241713f,-0.189580f,-0.202855f,-0.275266f,-0.331826f,-0.420798f,-0.467953f,-0.438844f,-0.371645f,-0.261716f,-0.119182f,-0.014236f,0.045561f,0.033220f,-0.035150f,
-0.144715f,-0.231138f,-0.274841f,-0.288685f,-0.244194f,-0.240182f,-0.288799f,-0.377497f,0.065991f,-0.193877f,-0.077512f,-0.185408f,-0.011371f,-0.147334f,-0.115957f,-0.230212f,-0.093622f,0.026385f,0.096781f,0.126974f,
0.196820f,0.268293f,0.280893f,0.279785f,0.225487f,0.156258f,0.073667f,0.009265f,-0.022622f,-0.012695f,0.043490f,0.061383f,0.078140f,0.059961f,0.068042f,0.051270f,0.086444f,0.107933f,0.131527f,0.132492f,
0.110897f,0.036949f,-0.059332f,-0.141324f,-0.214529f,-0.211342f,-0.189446f,-0.140733f,-0.102057f,-0.049962f,-0.057347f,-0.050861f,-0.018993f,0.023826f,0.057131f,0.091629f,0.108525f,0.095719f,0.036124f,-0.047949f,
-0.145802f,-0.180975f,-0.181561f,-0.144331f,-0.107680f,-0.056120f,-0.036708f,-0.025707f,-0.017457f,-0.005182f,0.025613f,0.065228f,0.081626f,0.082131f,0.058879f,-0.035461f,-0.132204f,-0.233883f,-0.315429f,-0.352716f,
-0.318758f,-0.282361f,-0.240218f,-0.218181f,-0.210261f,-0.199710f,-0.180140f,-0.138061f,-0.127366f,-0.080537f,-0.040801f,-0.033309f,-0.071127f,-0.132104f,-0.224080f,-0.314490f,-0.348494f,-0.351704f,-0.278416f,-0.201359f,
-0.171391f,-0.148816f,-0.137551f,-0.126536f,-0.095870f,-0.057830f,-0.024395f,0.024458f,0.057000f,0.038334f,-0.013538f,-0.093353f,-0.189770f,-0.248926f,-0.272125f,-0.240520f,-0.186047f,-0.121216f,-0.086531f,-0.074325f,
-0.057365f,-0.044703f,-0.031419f,0.004248f,0.022792f,0.085757f,0.121199f,0.121746f,0.040615f,-0.037004f,-0.130968f,-0.178790f,-0.203854f,-0.174849f,-0.112433f,-0.055561f,-0.138356f,0.067581f,0.326282f,0.042522f,
-0.080650f,-0.227080f,-0.199146f,0.083063f,0.115191f,-0.012682f,0.044176f,0.000698f,-0.064025f,-0.097714f,-0.140699f,-0.119168f,-0.099444f,-0.040334f,-0.027259f,-0.070915f,-0.146236f,-0.196731f,-0.212406f,-0.164082f,
-0.059012f,0.026042f,0.101204f,0.125957f,0.100034f,0.056739f,-0.008956f,-0.036616f,-0.032955f,-0.007458f,0.004374f,-0.025257f,-0.095880f,-0.189248f,-0.258786f,-0.290677f,-0.253284f,-0.180096f,-0.105159f,-0.046926f,
-0.039958f,-0.059367f,-0.128741f,-0.196247f,-0.198775f,-0.183284f,-0.146914f,-0.087705f,-0.100152f,-0.165806f,-0.220252f,-0.277001f,-0.256070f,-0.181972f,-0.092830f,0.021091f,0.119610f,0.175455f,0.192814f,0.165681f,
0.154732f,0.161332f,0.198948f,0.245444f,0.314319f,0.341293f,0.316615f,0.270813f,0.210917f,0.207516f,0.241061f,0.324128f,0.416467f,0.503635f,0.562245f,0.565225f,0.534421f,0.494037f,0.458805f,0.439054f,
0.471711f,0.503437f,0.543482f,0.491172f,0.412880f,0.292228f,0.205462f,0.190534f,0.200409f,0.241455f,0.293168f,0.327815f,0.299968f,0.263954f,0.201344f,0.088263f,0.048937f,0.049666f,0.068686f,0.099685f,
0.102458f,0.054510f,-0.049475f,-0.122182f,-0.184922f,-0.164789f,-0.071927f,-0.004504f,0.075711f,0.156009f,0.179499f,0.147689f,0.109271f,0.065786f,0.041868f,0.047802f,0.116587f,0.174148f,0.212462f,0.217600f,
0.165031f,0.104677f,0.056688f,0.073009f,0.140718f,-0.072681f,-0.039178f,-0.005528f,0.003314f,0.060038f,0.075277f,0.142505f,-0.139879f,-0.098039f,-0.223095f,-0.153989f,-0.156932f,-0.085627f,0.024933f,0.135203f,
0.201825f,0.217028f,0.205440f,0.181697f,0.189787f,0.211772f,0.231057f,0.213264f,0.184326f,0.118876f,0.008772f,-0.094573f,-0.140051f,-0.132128f,-0.066771f,0.013323f,0.072875f,0.109087f,0.082268f,0.033199f,
-0.005700f,-0.020473f,-0.030240f,-0.045976f,-0.057901f,-0.092532f,-0.171566f,-0.243331f,-0.330368f,-0.359885f,-0.334979f,-0.251619f,-0.120290f,-0.025012f,0.034591f,0.034163f,0.031039f,0.013373f,0.037761f,0.075804f,
0.117139f,0.121164f,0.103377f,0.060417f,-0.013697f,-0.075202f,-0.100161f,-0.061252f,0.034580f,0.138492f,0.245233f,0.320775f,0.322109f,0.287427f,0.278872f,0.229170f,0.232131f,0.247519f,0.241276f,0.216524f,
0.156212f,0.069514f,-0.016213f,-0.096199f,-0.110683f,-0.064930f,-0.003023f,0.097933f,0.172999f,0.167262f,0.153137f,0.097613f,0.047119f,0.021965f,0.015010f,0.016621f,-0.018635f,-0.068165f,-0.166236f,-0.280510f,
-0.380364f,-0.443387f,-0.435169f,-0.384744f,-0.288878f,-0.207532f,-0.158317f,-0.160895f,-0.184872f,-0.230630f,-0.252018f,-0.230235f,-0.205241f,-0.169261f,-0.187649f,-0.206413f,-0.274790f,-0.348001f,-0.405255f,-0.407329f,
-0.360164f,-0.257905f,-0.135797f,-0.005109f,0.076076f,0.105694f,0.090896f,0.042259f,0.023398f,0.031484f,0.066458f,0.080635f,-0.065683f,-0.046567f,-0.207773f,-0.041086f,-0.018419f,-0.102322f,-0.113756f,-0.183259f,
-0.178343f,-0.068924f,-0.006882f,0.182491f,0.259091f,0.249119f,0.186657f,0.061070f,-0.039654f,-0.106800f,-0.154420f,-0.117806f,-0.094334f,-0.066247f,-0.054796f,-0.047814f,-0.027571f,0.001612f,0.079883f,0.180733f,
0.253800f,0.296491f,0.252125f,0.172718f,0.032206f,-0.078206f,-0.154071f,-0.189360f,-0.183946f,-0.139994f,-0.117266f,-0.083937f,-0.068309f,-0.077750f,-0.051774f,0.041306f,0.141461f,0.235179f,0.271667f,0.263928f,
0.170079f,0.063203f,-0.044558f,-0.105545f,-0.097540f,-0.068806f,-0.022070f,0.025572f,0.041222f,0.052095f,0.064126f,0.120250f,0.223462f,0.334893f,0.438366f,0.475417f,0.477221f,0.399946f,0.304715f,0.197882f,
0.121649f,0.089414f,0.085920f,0.143991f,0.166292f,0.156733f,0.182673f,0.149846f,0.144830f,0.198717f,0.298759f,0.365374f,0.433397f,0.439362f,0.375543f,0.266634f,0.135140f,0.004475f,-0.066654f,-0.102338f,
-0.078649f,-0.050966f,-0.070049f,-0.062451f,-0.080172f,-0.112431f,-0.105676f,-0.032685f,0.047435f,0.135836f,0.191284f,0.164477f,0.089354f,-0.032258f,-0.147449f,-0.230987f,-0.285949f,-0.246068f,-0.205161f,-0.161130f,
-0.127435f,-0.112586f,-0.113857f,-0.101317f,-0.045103f,0.063401f,0.176361f,0.274077f,0.320971f,0.313278f,0.226851f,0.109843f,0.007054f,-0.079843f,-0.115099f,-0.066183f,0.010560f,0.077024f,0.129165f,0.142381f,
0.195089f,0.008997f,0.081678f,0.016751f,-0.143073f,0.002937f,0.039940f,0.141242f,0.070658f,0.048948f,0.137391f,-0.029418f,-0.013528f,-0.004681f,-0.004625f,-0.025633f,-0.051703f,-0.065268f,-0.092960f,-0.125216f,
-0.126560f,-0.134432f,-0.135046f,-0.128249f,-0.150484f,-0.178529f,-0.201663f,-0.191040f,-0.189129f,-0.161009f,-0.159450f,-0.196269f,-0.191330f,-0.218458f,-0.264820f,-0.280050f,-0.297408f,-0.285330f,-0.281539f,-0.260020f,
-0.248235f,-0.263335f,-0.256131f,-0.271383f,-0.238876f,-0.213487f,-0.200274f,-0.179640f,-0.154225f,-0.172487f,-0.167270f,-0.187954f,-0.213966f,-0.217374f,-0.199400f,-0.194047f,-0.175871f,-0.149725f,-0.147489f,-0.179529f,
-0.172057f,-0.157788f,-0.131987f,-0.125448f,-0.105812f,-0.098119f,-0.112962f,-0.122857f,-0.156262f,-0.157838f,-0.175130f,-0.160839f,-0.146893f,-0.114207f,-0.088080f,-0.077858f,-0.082683f,-0.076610f,-0.057710f,-0.027157f,
-0.006709f,0.018674f,0.013302f,0.020152f,0.003987f,-0.000322f,-0.040958f,-0.065181f,-0.056771f,-0.057775f,-0.056305f,-0.060504f,-0.063386f,-0.091002f,-0.102748f,-0.110938f,-0.090583f,-0.088540f,-0.073473f,-0.050893f,
-0.049189f,-0.043136f,-0.053868f,-0.093414f,-0.122531f,-0.115100f,-0.101239f,-0.098294f,-0.050929f,-0.041167f,-0.018398f,-0.040621f,-0.040892f,-0.046923f,-0.027246f,-0.005008f,0.009511f,0.052168f,0.057151f,0.056591f,
0.027422f,0.018017f,-0.012649f,-0.011442f,-0.005074f,-0.000245f,0.041504f,0.057629f,-0.210086f,0.266030f,0.296233f,0.144242f,0.178400f,0.169100f,0.200384f,0.281001f,0.264333f,0.093060f,0.113639f,0.162721f,
0.152675f,0.177571f,0.261078f,0.365703f,0.417457f,0.403401f,0.311379f,0.187425f,0.056513f,-0.033069f,-0.080789f,-0.124756f,-0.131603f,-0.175882f,-0.218977f,-0.281234f,-0.317959f,-0.331072f,-0.297594f,-0.216189f,
-0.130092f,-0.073698f,-0.092563f,-0.142914f,-0.233646f,-0.302739f,-0.334858f,-0.318778f,-0.271761f,-0.209674f,-0.161937f,-0.133829f,-0.109089f,-0.093504f,-0.040744f,0.063766f,0.207249f,0.334137f,0.427673f,0.414612f,
0.360303f,0.277080f,0.195339f,0.170982f,0.158606f,0.182805f,0.210031f,0.222599f,0.204101f,0.173936f,0.144977f,0.116137f,0.146562f,0.258488f,0.343452f,0.385399f,0.372415f,0.278349f,0.165295f,0.053177f,
-0.032485f,-0.059004f,-0.036022f,0.004298f,0.032397f,0.039947f,0.018789f,-0.008825f,-0.049161f,-0.045171f,-0.016828f,0.079304f,0.149410f,0.152282f,0.117071f,0.010046f,-0.110407f,-0.196886f,-0.271679f,-0.270915f,
-0.246045f,-0.223525f,-0.214026f,-0.220174f,-0.230761f,-0.251242f,-0.214947f,-0.145144f,-0.002489f,0.132328f,0.225932f,0.277732f,0.242758f,0.189946f,0.125442f,0.099789f,0.118163f,0.181094f,0.239991f,0.330668f,
0.346950f,0.380643f,0.370548f,0.362314f,0.385249f,0.434121f,0.557138f,0.655403f,0.706771f,0.698143f,0.624318f,0.515271f,0.410571f,0.321093f,0.295746f,0.284226f,0.093497f,-0.188340f,-0.175652f,-0.198527f,
-0.255060f,-0.190388f,-0.179376f,-0.235777f,-0.066830f,-0.031445f,-0.080552f,-0.206259f,-0.138318f,-0.085406f,-0.052879f,-0.052371f,-0.100972f,-0.156582f,-0.213609f,-0.259217f,-0.287297f,-0.280733f,-0.242242f,-0.229718f,
-0.240313f,-0.280366f,-0.309919f,-0.316361f,-0.300673f,-0.286607f,-0.270747f,-0.261038f,-0.304062f,-0.386449f,-0.452774f,-0.517485f,-0.570769f,-0.569724f,-0.544160f,-0.489773f,-0.429259f,-0.403696f,-0.396885f,-0.367334f,
-0.318425f,-0.245091f,-0.187118f,-0.097067f,-0.051147f,-0.051495f,-0.079860f,-0.119779f,-0.153934f,-0.158930f,-0.149124f,-0.086858f,-0.015316f,0.022127f,0.029547f,0.036718f,0.053469f,0.093563f,0.145191f,0.167650f,
0.202830f,0.202821f,0.204768f,0.140101f,0.064143f,-0.028089f,-0.096883f,-0.107605f,-0.108116f,-0.094679f,-0.066162f,-0.044602f,-0.081291f,-0.092264f,-0.083075f,-0.074372f,-0.040645f,-0.020923f,0.006024f,-0.022770f,
-0.053861f,-0.148574f,-0.227325f,-0.289729f,-0.325777f,-0.307703f,-0.293134f,-0.230875f,-0.233071f,-0.232824f,-0.254740f,-0.261283f,-0.243836f,-0.207585f,-0.134234f,-0.096672f,-0.051892f,-0.044746f,-0.111331f,-0.141434f,
-0.191112f,-0.205790f,-0.197387f,-0.135721f,-0.063048f,0.012922f,0.062080f,0.092263f,0.097485f,0.140269f,0.192877f,0.243160f,0.320223f,0.395301f,0.423303f,0.449705f,0.380197f,0.324297f,0.254391f,0.207223f,
0.201759f,0.230939f,0.284008f,0.351076f,0.041083f,0.096316f,-0.043180f,0.110873f,0.188276f,0.224421f,0.284186f,0.347330f,0.368573f,0.115072f,0.094038f,0.071835f,0.045416f,0.033139f,0.002752f,-0.080354f,
-0.183022f,-0.298054f,-0.362841f,-0.342069f,-0.250582f,-0.135654f,-0.028459f,0.047141f,0.073485f,0.050243f,0.031368f,0.007302f,-0.005057f,0.012158f,0.025558f,-0.002601f,-0.056156f,-0.143993f,-0.229621f,-0.265053f,
-0.214514f,-0.112692f,0.005438f,0.127525f,0.171846f,0.172062f,0.133862f,0.081531f,0.041486f,0.012370f,0.008030f,0.002173f,-0.021529f,-0.100747f,-0.196091f,-0.301851f,-0.332669f,-0.297648f,-0.204681f,-0.083341f,
0.015370f,0.078985f,0.083800f,0.047207f,0.006856f,-0.023631f,-0.013255f,-0.011209f,0.015126f,0.002461f,-0.043520f,-0.098318f,-0.154702f,-0.178779f,-0.124857f,-0.026682f,0.113647f,0.221300f,0.312454f,0.340147f,
0.321018f,0.286643f,0.246641f,0.210842f,0.221438f,0.221210f,0.227084f,0.170817f,0.110378f,0.000701f,-0.089971f,-0.127360f,-0.087848f,-0.019423f,0.093678f,0.143525f,0.180643f,0.149889f,0.114197f,0.068198f,
0.014396f,-0.004991f,-0.001947f,0.015451f,-0.009789f,-0.062331f,-0.154309f,-0.273492f,-0.315209f,-0.295461f,-0.227989f,-0.123987f,-0.006188f,0.074578f,0.073994f,0.050314f,0.003489f,-0.068406f,-0.099018f,-0.121038f,
-0.115479f,-0.101450f,-0.125970f,-0.187150f,-0.241777f,-0.331140f,-0.394437f,-0.366445f,-0.302220f,-0.168267f,-0.032847f,0.046182f,-0.569003f,0.045195f,0.032338f,0.136315f,0.098469f,0.201719f,0.080093f,0.141620f,
0.234048f,0.084716f,-0.024093f,0.105433f,0.140176f,0.164415f,0.202676f,0.247283f,0.282749f,0.290853f,0.311229f,0.326617f,0.359917f,0.356796f,0.394637f,0.402264f,0.387571f,0.363506f,0.331850f,0.306659f,
0.299416f,0.304591f,0.304438f,0.319512f,0.311276f,0.298668f,0.266081f,0.234840f,0.216730f,0.198564f,0.172111f,0.155079f,0.133653f,0.110458f,0.065604f,0.017067f,-0.016039f,-0.042408f,-0.030972f,-0.014222f,
0.009639f,0.033265f,0.033500f,0.046276f,0.052684f,0.065106f,0.095309f,0.118151f,0.149570f,0.169949f,0.170285f,0.170755f,0.166271f,0.168662f,0.199959f,0.235091f,0.267345f,0.316252f,0.351724f,0.385538f,
0.400878f,0.406996f,0.410790f,0.407540f,0.434201f,0.449225f,0.453346f,0.432020f,0.417082f,0.387209f,0.357317f,0.368169f,0.389026f,0.410476f,0.433662f,0.469235f,0.505062f,0.506513f,0.508066f,0.496739f,
0.483617f,0.477940f,0.480054f,0.464581f,0.451601f,0.388764f,0.344698f,0.267738f,0.237127f,0.201295f,0.177974f,0.171710f,0.183016f,0.161022f,0.176461f,0.144751f,0.109544f,0.103080f,0.076084f,0.101051f,
0.111750f,0.099604f,0.100216f,0.073003f,0.058556f,0.029565f,0.032423f,0.041257f,0.067378f,0.102391f,0.155681f,0.180597f,0.226868f,0.252988f,0.290116f,0.297872f,0.308779f,0.336732f,0.371817f,0.390248f,
0.234935f,0.310261f,0.239714f,0.240906f,0.176098f,0.136119f,0.085857f,0.169038f,0.126604f,0.253157f,0.236810f,0.012674f,-0.012263f,-0.026692f,-0.006317f,0.047918f,0.061595f,0.066119f,0.015529f,-0.085702f,
-0.188971f,-0.280601f,-0.316887f,-0.287012f,-0.234990f,-0.175078f,-0.118812f,-0.099433f,-0.103597f,-0.108642f,-0.071216f,-0.022465f,0.063674f,0.108000f,0.118134f,0.059319f,-0.045754f,-0.130909f,-0.214511f,-0.256436f,
-0.229097f,-0.193190f,-0.128702f,-0.103318f,-0.135800f,-0.161188f,-0.204644f,-0.207792f,-0.178656f,-0.137732f,-0.088757f,-0.125231f,-0.171886f,-0.299507f,-0.413725f,-0.502744f,-0.553449f,-0.517217f,-0.466561f,-0.416878f,
-0.374957f,-0.383209f,-0.404854f,-0.398216f,-0.343742f,-0.294151f,-0.223626f,-0.175741f,-0.148552f,-0.149979f,-0.224296f,-0.322364f,-0.397147f,-0.400577f,-0.367975f,-0.298896f,-0.203637f,-0.134584f,-0.128332f,-0.138001f,
-0.146190f,-0.124483f,-0.088728f,-0.036362f,0.013301f,0.075557f,0.061645f,-0.019496f,-0.146267f,-0.268311f,-0.347080f,-0.379601f,-0.344699f,-0.307803f,-0.267643f,-0.246257f,-0.262416f,-0.286714f,-0.307434f,-0.321360f,
-0.289476f,-0.218428f,-0.177588f,-0.172902f,-0.228024f,-0.329273f,-0.458974f,-0.573740f,-0.642832f,-0.661304f,-0.584974f,-0.544874f,-0.511078f,-0.523602f,-0.518924f,-0.567354f,-0.585740f,-0.601698f,-0.570624f,-0.494136f,
-0.479681f,-0.471783f,-0.481661f,-0.582922f,-0.662062f,-0.765044f,-0.803595f,-0.800292f,-0.101155f,-0.304690f,-0.307480f,-0.481849f,-0.341799f,-0.257601f,-0.197713f,0.095583f,0.075283f,-0.048769f,-0.020896f,0.029977f,
0.043190f,0.081070f,0.147248f,0.189778f,0.187822f,0.139400f,0.064044f,0.010079f,-0.015175f,0.037258f,0.112989f,0.200296f,0.261920f,0.275396f,0.228139f,0.165148f,0.115270f,0.117818f,0.146708f,0.206983f,
0.239301f,0.227540f,0.169735f,0.123945f,0.068299f,0.054363f,0.107716f,0.201729f,0.284082f,0.373664f,0.370538f,0.339200f,0.273946f,0.234254f,0.223975f,0.277287f,0.324476f,0.360438f,0.330766f,0.280542f,
0.187548f,0.147949f,0.119900f,0.186284f,0.285176f,0.355133f,0.428869f,0.418666f,0.363895f,0.283015f,0.206823f,0.195490f,0.207000f,0.221809f,0.240338f,0.208596f,0.129662f,0.056297f,-0.004438f,-0.032868f,
-0.013885f,0.046768f,0.136527f,0.195714f,0.207355f,0.189464f,0.114138f,0.079258f,0.044330f,0.040440f,0.105886f,0.155994f,0.169742f,0.155681f,0.092241f,0.045038f,0.027784f,0.054724f,0.134096f,0.253857f,
0.380951f,0.452951f,0.466244f,0.439889f,0.396028f,0.357020f,0.358255f,0.374212f,0.444352f,0.493154f,0.494751f,0.450445f,0.365967f,0.292858f,0.266508f,0.289372f,0.378296f,0.481318f,0.552000f,0.585491f,
0.575512f,0.513260f,0.442593f,0.385639f,0.360252f,0.398125f,0.470489f,0.493521f,0.486251f,0.431402f,0.357110f,0.257127f,0.219716f,0.221025f,0.258429f,0.348821f,0.029886f,0.379394f,0.355434f,0.246785f,
0.197919f,0.157379f,0.119758f,-0.002401f,0.057058f,0.101845f,0.148049f,-0.071041f,-0.047022f,-0.045015f,-0.051158f,-0.094203f,-0.180867f,-0.224090f,-0.288479f,-0.338046f,-0.387721f,-0.416392f,-0.448388f,-0.457594f,
-0.435489f,-0.380847f,-0.319623f,-0.248693f,-0.185737f,-0.139898f,-0.117369f,-0.141273f,-0.162834f,-0.226591f,-0.265437f,-0.304289f,-0.343553f,-0.375072f,-0.381623f,-0.393302f,-0.366486f,-0.318817f,-0.275652f,-0.191978f,
-0.128660f,-0.059589f,-0.024431f,0.009812f,0.002372f,-0.036349f,-0.096606f,-0.128766f,-0.169028f,-0.201415f,-0.238484f,-0.266608f,-0.283617f,-0.264146f,-0.223547f,-0.182649f,-0.123503f,-0.046968f,0.038029f,0.087103f,
0.116326f,0.120614f,0.123799f,0.070463f,0.037496f,0.003708f,-0.028057f,-0.046376f,-0.086548f,-0.085425f,-0.086001f,-0.069530f,-0.045025f,0.016223f,0.072294f,0.116336f,0.167677f,0.172612f,0.167937f,0.125402f,
0.050627f,-0.017393f,-0.077830f,-0.149279f,-0.228402f,-0.272237f,-0.354528f,-0.388854f,-0.415798f,-0.414323f,-0.394046f,-0.345716f,-0.305418f,-0.238753f,-0.202855f,-0.166492f,-0.170492f,-0.183481f,-0.236361f,-0.293196f,
-0.318935f,-0.360436f,-0.381945f,-0.402985f,-0.423301f,-0.401655f,-0.386070f,-0.366578f,-0.296034f,-0.221717f,-0.162960f,-0.085004f,-0.022727f,-0.025232f,-0.030619f,-0.098682f,-0.157494f,-0.216341f,-0.269693f,-0.286930f,
-0.301928f,-0.328653f,-0.330214f,-0.337291f,-0.155527f,0.069576f,0.202709f,0.147001f,0.189922f,0.070930f,0.053780f,0.073834f,0.034305f,-0.091154f,-0.024368f,0.241281f,0.290780f,0.300643f,0.289379f,0.277150f,
0.245699f,0.279186f,0.333014f,0.390958f,0.418642f,0.403501f,0.334099f,0.265661f,0.191242f,0.180509f,0.187290f,0.218500f,0.260540f,0.280339f,0.298336f,0.270627f,0.227606f,0.216087f,0.222629f,0.295210f,
0.328404f,0.365899f,0.337665f,0.261156f,0.192566f,0.138056f,0.137361f,0.160917f,0.207074f,0.272674f,0.315826f,0.353198f,0.343145f,0.278242f,0.289227f,0.326672f,0.400350f,0.463140f,0.493759f,0.471884f,
0.414378f,0.330974f,0.280945f,0.251468f,0.270133f,0.319615f,0.365036f,0.395030f,0.406847f,0.382777f,0.348231f,0.332667f,0.337474f,0.361924f,0.389566f,0.394879f,0.369797f,0.278660f,0.171787f,0.093630f,
0.026556f,0.013355f,0.006335f,0.040927f,0.069813f,0.048520f,0.025496f,-0.017952f,-0.087817f,-0.124003f,-0.096245f,-0.044364f,-0.008014f,-0.007748f,-0.042210f,-0.139461f,-0.214077f,-0.259461f,-0.272777f,-0.254345f,
-0.229904f,-0.167455f,-0.122289f,-0.122086f,-0.130344f,-0.169823f,-0.191212f,-0.165670f,-0.101323f,-0.039413f,0.020039f,0.037338f,-0.008411f,-0.079768f,-0.156430f,-0.173337f,-0.154868f,-0.128690f,-0.045390f,0.038707f,
0.084494f,0.091139f,0.067959f,0.044003f,0.011824f,0.003214f,0.054979f,0.098261f,0.158133f,0.167730f,0.129665f,0.038507f,-0.209444f,0.059492f,-0.007604f,0.153723f,0.143376f,0.158163f,0.240061f,0.174935f,
0.264214f,0.161040f,0.261953f,0.280322f,0.235602f,0.200910f,0.175690f,0.176341f,0.215469f,0.242059f,0.266413f,0.284756f,0.294767f,0.301846f,0.307347f,0.292184f,0.301737f,0.288336f,0.251679f,0.191217f,
0.144886f,0.080119f,0.025012f,-0.007253f,-0.003132f,0.009449f,0.039707f,0.064654f,0.084530f,0.088283f,0.110604f,0.112285f,0.135379f,0.149450f,0.146407f,0.143184f,0.101580f,0.074097f,0.029464f,0.000139f,
0.000689f,0.001216f,0.039087f,0.094518f,0.115566f,0.166886f,0.171321f,0.181570f,0.195002f,0.205918f,0.234716f,0.230581f,0.202704f,0.166873f,0.120106f,0.065287f,0.013818f,-0.006433f,-0.013894f,0.002573f,
0.018443f,0.053702f,0.070010f,0.076940f,0.083545f,0.102996f,0.092176f,0.091696f,0.106035f,0.098263f,0.084004f,0.036123f,0.003710f,-0.054145f,-0.073449f,-0.089638f,-0.072775f,-0.044516f,-0.024243f,0.018565f,
0.031882f,0.040504f,0.052779f,0.084470f,0.087055f,0.108607f,0.112072f,0.100893f,0.071910f,0.050176f,-0.001613f,-0.038573f,-0.045138f,-0.056035f,-0.016546f,0.023504f,0.074453f,0.116049f,0.121836f,0.155943f,
0.172088f,0.195712f,0.202843f,0.218127f,0.246133f,0.221553f,0.203915f,0.163734f,0.112364f,0.089840f,0.078897f,0.105276f,0.124554f,0.183233f,0.208044f,0.239427f,0.247417f,0.255640f,0.284074f,0.250011f,
-0.047946f,0.041860f,0.157537f,0.142537f,-0.034788f,0.221256f,0.031736f,0.254996f,0.135895f,0.071647f,0.006809f,-0.048976f,-0.040111f,-0.038998f,-0.018695f,-0.039650f,-0.088813f,-0.135058f,-0.175968f,-0.168608f,
-0.184771f,-0.168152f,-0.190344f,-0.209826f,-0.246941f,-0.264104f,-0.262413f,-0.221886f,-0.188878f,-0.141598f,-0.097751f,-0.070150f,-0.085267f,-0.096612f,-0.120476f,-0.111158f,-0.118503f,-0.097201f,-0.079771f,-0.088955f,
-0.101426f,-0.106860f,-0.146455f,-0.138338f,-0.133736f,-0.076374f,-0.059271f,-0.024245f,-0.014250f,-0.027576f,-0.067822f,-0.104027f,-0.145586f,-0.147827f,-0.129182f,-0.140163f,-0.165863f,-0.195098f,-0.229641f,-0.267925f,
-0.273440f,-0.267592f,-0.217439f,-0.197206f,-0.161612f,-0.139846f,-0.146946f,-0.158277f,-0.177194f,-0.190961f,-0.173683f,-0.145821f,-0.119848f,-0.106231f,-0.113284f,-0.092902f,-0.107151f,-0.107744f,-0.073379f,-0.045698f,
0.013754f,0.042349f,0.089124f,0.106435f,0.095945f,0.083778f,0.056891f,0.058216f,0.070838f,0.079703f,0.083336f,0.067563f,0.035901f,0.000043f,-0.022627f,-0.018375f,0.015476f,0.030314f,0.092670f,0.111076f,
0.141820f,0.175600f,0.153677f,0.121613f,0.093057f,0.078847f,0.094273f,0.105860f,0.107438f,0.088021f,0.023867f,0.006535f,-0.021291f,-0.030216f,-0.026300f,-0.001111f,0.040672f,0.057925f,0.074968f,0.050343f,
0.023611f,-0.025075f,-0.058860f,-0.088575f,-0.054784f,-0.054147f,-0.052074f,-0.057626f,0.157198f,-0.074233f,-0.195588f,-0.084381f,0.050026f,0.087158f,0.090541f,0.086881f,0.007921f,0.071484f,0.111292f,-0.126691f,
-0.067108f,-0.038118f,0.000787f,-0.021125f,-0.067901f,-0.161042f,-0.222208f,-0.263630f,-0.252164f,-0.227550f,-0.215088f,-0.216165f,-0.238268f,-0.284032f,-0.291135f,-0.309578f,-0.288889f,-0.254780f,-0.217442f,-0.204683f,
-0.203530f,-0.269347f,-0.328357f,-0.372828f,-0.392836f,-0.368136f,-0.331188f,-0.285696f,-0.242109f,-0.244815f,-0.259353f,-0.256121f,-0.249619f,-0.221311f,-0.171591f,-0.108887f,-0.079741f,-0.081027f,-0.133338f,-0.193522f,
-0.256661f,-0.274171f,-0.244028f,-0.221191f,-0.186666f,-0.165702f,-0.192050f,-0.202391f,-0.210163f,-0.236437f,-0.226551f,-0.208547f,-0.158803f,-0.130424f,-0.158820f,-0.215299f,-0.261893f,-0.329345f,-0.362925f,-0.353372f,
-0.316698f,-0.276870f,-0.242264f,-0.227532f,-0.245294f,-0.261306f,-0.264900f,-0.244787f,-0.215558f,-0.190587f,-0.135229f,-0.109412f,-0.148849f,-0.216563f,-0.291495f,-0.340751f,-0.329784f,-0.320102f,-0.263764f,-0.205295f,
-0.213110f,-0.250974f,-0.260429f,-0.270105f,-0.266289f,-0.244548f,-0.197119f,-0.120977f,-0.086429f,-0.066047f,-0.122309f,-0.165411f,-0.238395f,-0.244784f,-0.229956f,-0.171731f,-0.126199f,-0.099573f,-0.089629f,-0.090273f,
-0.098480f,-0.107809f,-0.128178f,-0.109578f,-0.066182f,-0.013036f,0.030829f,0.022827f,-0.036910f,-0.104825f,-0.178429f,-0.234785f,-0.216452f,-0.197796f,-0.131877f,-0.090323f,0.232293f,-0.003602f,-0.195289f,0.032484f,
-0.030386f,-0.006727f,0.042713f,0.157109f,0.124833f,0.226821f,0.263851f,0.013131f,-0.005326f,-0.028065f,-0.029840f,-0.008666f,0.030165f,0.027514f,-0.007011f,-0.059811f,-0.130255f,-0.194177f,-0.226648f,-0.230482f,
-0.248789f,-0.242553f,-0.255415f,-0.268911f,-0.307075f,-0.337847f,-0.364050f,-0.377281f,-0.348974f,-0.313472f,-0.323957f,-0.351918f,-0.401999f,-0.457804f,-0.499475f,-0.498708f,-0.467663f,-0.464293f,-0.422250f,-0.392422f,
-0.382620f,-0.380453f,-0.396645f,-0.399378f,-0.372397f,-0.318072f,-0.271354f,-0.258679f,-0.294439f,-0.337402f,-0.369344f,-0.402124f,-0.404202f,-0.365033f,-0.348289f,-0.326050f,-0.286677f,-0.278239f,-0.292791f,-0.324189f,
-0.321062f,-0.317858f,-0.271317f,-0.246364f,-0.222318f,-0.238968f,-0.278297f,-0.301832f,-0.324978f,-0.338907f,-0.303519f,-0.234853f,-0.200363f,-0.149605f,-0.105778f,-0.108859f,-0.105360f,-0.125159f,-0.129231f,-0.098498f,
-0.039645f,-0.000226f,-0.001707f,-0.041771f,-0.090598f,-0.147983f,-0.185627f,-0.213166f,-0.208007f,-0.181592f,-0.183748f,-0.162910f,-0.165299f,-0.180939f,-0.185397f,-0.215292f,-0.187983f,-0.168773f,-0.120830f,-0.093009f,
-0.111395f,-0.138121f,-0.164041f,-0.213956f,-0.218328f,-0.204634f,-0.147907f,-0.117155f,-0.067958f,-0.055557f,-0.066338f,-0.067333f,-0.099608f,-0.072574f,-0.084453f,-0.046069f,-0.002772f,0.029265f,0.031862f,-0.012082f,
-0.059651f,-0.086642f,-0.104304f,-0.091935f,-0.064300f,-0.090560f,-0.148312f,-0.058687f,0.025667f,0.026536f,-0.025753f,0.022423f,-0.003199f,0.009575f,0.029040f,0.255785f,0.284452f,0.289270f,0.297159f,0.281478f,
0.247578f,0.237000f,0.214670f,0.236329f,0.278126f,0.316005f,0.331865f,0.333564f,0.300518f,0.286338f,0.245992f,0.225725f,0.216655f,0.187562f,0.154915f,0.120886f,0.075533f,0.031368f,-0.027930f,-0.073247f,
-0.076266f,-0.050393f,-0.027481f,-0.025314f,-0.022122f,-0.042033f,-0.034669f,-0.050395f,-0.062415f,-0.052642f,-0.050646f,-0.050875f,-0.041837f,-0.079298f,-0.108537f,-0.127677f,-0.146375f,-0.093497f,-0.026915f,0.019894f,
0.049984f,0.074843f,0.064029f,0.073511f,0.095063f,0.104758f,0.115366f,0.121320f,0.123636f,0.098387f,0.085500f,0.044100f,-0.003507f,-0.031661f,-0.032151f,0.007281f,0.038947f,0.059952f,0.065269f,0.057834f,
0.040622f,0.034631f,0.039311f,0.048726f,0.077598f,0.062439f,0.063672f,0.044069f,0.011306f,-0.031616f,-0.075104f,-0.091041f,-0.034453f,-0.021861f,0.020876f,0.022548f,0.006114f,-0.029509f,-0.026302f,-0.043421f,
-0.060313f,-0.051225f,-0.064987f,-0.064854f,-0.085100f,-0.113641f,-0.165363f,-0.192311f,-0.236266f,-0.198268f,-0.174573f,-0.134613f,-0.065360f,-0.046816f,-0.042007f,-0.056136f,-0.056679f,-0.033228f,-0.014654f,-0.019695f,
-0.000717f,0.027808f,0.034491f,0.030207f,0.033278f,0.003997f,-0.000696f,0.023703f,0.074296f,0.131347f,0.188353f,0.231811f,-0.133473f,0.044561f,-0.022356f,-0.135431f,-0.281579f,-0.266795f,-0.220973f,-0.273635f,
-0.282761f,-0.157026f,-0.092702f,0.262263f,0.298191f,0.316043f,0.348360f,0.361464f,0.401358f,0.434454f,0.437521f,0.417878f,0.335388f,0.297794f,0.245688f,0.222466f,0.267066f,0.310966f,0.384748f,0.419678f,
0.430159f,0.435668f,0.423066f,0.412521f,0.417783f,0.432510f,0.459685f,0.420175f,0.396628f,0.313035f,0.234566f,0.182500f,0.192497f,0.218912f,0.282353f,0.347600f,0.383975f,0.409069f,0.402574f,0.395707f,
0.382762f,0.409409f,0.422893f,0.453650f,0.433939f,0.391011f,0.314827f,0.261207f,0.228509f,0.209600f,0.257800f,0.305501f,0.373740f,0.408349f,0.425553f,0.401917f,0.409837f,0.381167f,0.389495f,0.371346f,
0.376361f,0.349752f,0.288356f,0.201041f,0.116470f,0.048633f,0.013840f,0.002521f,0.037601f,0.088080f,0.118066f,0.133906f,0.141293f,0.131287f,0.122689f,0.104157f,0.107506f,0.119483f,0.138391f,0.108499f,
0.078476f,0.009041f,-0.034062f,-0.058086f,-0.035228f,0.018629f,0.082916f,0.151605f,0.177440f,0.208863f,0.202641f,0.209542f,0.167534f,0.175953f,0.170089f,0.215511f,0.209191f,0.172414f,0.120476f,0.045365f,
-0.010013f,-0.032677f,-0.017807f,0.031982f,0.101002f,0.171490f,0.203810f,0.230765f,0.234009f,0.209122f,0.204700f,0.204437f,0.210269f,0.220740f,0.235308f,0.200799f,0.152886f,0.095998f,0.043982f,-0.010155f,
0.121959f,-0.255223f,-0.126671f,-0.222339f,-0.123299f,-0.156635f,-0.075276f,-0.243182f,-0.148488f,-0.188004f,-0.295841f,0.032625f,0.070782f,0.151915f,0.228332f,0.273050f,0.263048f,0.178694f,0.066018f,-0.041828f,
-0.099667f,-0.077624f,-0.027870f,0.001131f,-0.011929f,-0.053490f,-0.102922f,-0.116583f,-0.103225f,-0.021331f,0.080199f,0.176313f,0.272434f,0.288425f,0.239183f,0.159594f,0.065572f,0.064640f,0.097835f,0.157077f,
0.199974f,0.212259f,0.155000f,0.108547f,0.055681f,0.023237f,0.079870f,0.142508f,0.218638f,0.259801f,0.244108f,0.132694f,-0.018989f,-0.123753f,-0.185820f,-0.165039f,-0.146375f,-0.128127f,-0.151354f,-0.234187f,
-0.308568f,-0.382560f,-0.409800f,-0.367574f,-0.306170f,-0.239380f,-0.195830f,-0.180407f,-0.238149f,-0.323686f,-0.418272f,-0.440475f,-0.411830f,-0.326822f,-0.256125f,-0.180723f,-0.188369f,-0.193215f,-0.233774f,-0.210466f,
-0.175613f,-0.076007f,0.036428f,0.146693f,0.234271f,0.237421f,0.191042f,0.092929f,0.052492f,0.055501f,0.136624f,0.229608f,0.292529f,0.302665f,0.283015f,0.236934f,0.213856f,0.214584f,0.274847f,0.375487f,
0.505158f,0.582210f,0.610310f,0.522512f,0.426663f,0.312385f,0.236401f,0.245219f,0.275116f,0.291220f,0.305226f,0.263745f,0.171257f,0.094235f,0.014806f,-0.027225f,0.003468f,0.046679f,0.144412f,0.188429f,
0.178310f,0.101019f,-0.016220f,-0.161488f,-0.253220f,-0.260222f,-0.227873f,-0.163622f,-0.013757f,0.102097f,0.180874f,0.084662f,0.254999f,0.200181f,0.133060f,0.062982f,-0.024777f,0.115342f,0.076758f,0.111248f,
0.115556f,0.107627f,0.038031f,0.009383f,-0.012412f,0.014558f,0.069309f,0.151880f,0.192132f,0.221342f,0.178187f,0.128923f,0.088207f,0.064626f,0.053214f,0.064280f,0.064695f,0.053026f,0.021016f,-0.037066f,
-0.069427f,-0.091396f,-0.066272f,-0.028258f,0.026537f,0.080717f,0.071598f,0.039275f,-0.022203f,-0.081318f,-0.106957f,-0.128894f,-0.100939f,-0.116498f,-0.110076f,-0.153853f,-0.208924f,-0.257155f,-0.278770f,-0.261148f,
-0.196815f,-0.128746f,-0.088775f,-0.063487f,-0.112712f,-0.164575f,-0.203706f,-0.228697f,-0.234985f,-0.207290f,-0.206090f,-0.196501f,-0.231648f,-0.277338f,-0.311448f,-0.324117f,-0.313308f,-0.255243f,-0.170197f,-0.117435f,
-0.088374f,-0.081261f,-0.124068f,-0.158092f,-0.165809f,-0.168706f,-0.134548f,-0.100949f,-0.069921f,-0.053350f,-0.074799f,-0.111419f,-0.113755f,-0.087604f,-0.061409f,0.047784f,0.123546f,0.178711f,0.189806f,0.134729f,
0.082416f,0.057259f,0.035705f,0.043987f,0.046356f,0.095477f,0.080333f,0.050079f,-0.000725f,-0.062193f,-0.092373f,-0.078546f,-0.033238f,0.033307f,0.109812f,0.116922f,0.094529f,0.023722f,-0.034062f,-0.063690f,
-0.086310f,-0.059866f,-0.078620f,-0.049034f,-0.067479f,-0.079209f,-0.137325f,-0.170410f,-0.198516f,-0.149632f,-0.092050f,-0.029100f,0.031701f,0.092559f,0.076331f,0.036016f,0.163007f,0.024370f,0.079316f,0.048277f,
0.027762f,0.059375f,0.056471f,0.414936f,0.190912f,0.103793f,0.034534f,-0.123584f,-0.155151f,-0.169292f,-0.135233f,-0.086904f,-0.061110f,-0.072774f,-0.100398f,-0.149134f,-0.182563f,-0.189406f,-0.144538f,-0.124517f,
-0.070206f,-0.081723f,-0.121177f,-0.164892f,-0.204704f,-0.225063f,-0.196431f,-0.174382f,-0.106704f,-0.087965f,-0.095119f,-0.144174f,-0.181809f,-0.184307f,-0.187005f,-0.169868f,-0.114710f,-0.092433f,-0.088877f,-0.118319f,
-0.174376f,-0.224118f,-0.245629f,-0.212149f,-0.162075f,-0.105037f,-0.077152f,-0.108174f,-0.121389f,-0.147345f,-0.181502f,-0.172090f,-0.132291f,-0.087936f,-0.046007f,-0.024033f,-0.072850f,-0.125493f,-0.163970f,-0.162530f,
-0.141159f,-0.075458f,-0.019326f,0.036337f,0.048521f,0.017632f,-0.011846f,-0.035131f,-0.041697f,-0.000948f,0.010542f,0.061714f,0.074716f,0.049760f,-0.001909f,-0.069983f,-0.108378f,-0.124902f,-0.113521f,-0.084728f,
-0.036107f,-0.064170f,-0.082794f,-0.147617f,-0.222791f,-0.264675f,-0.276653f,-0.287196f,-0.240283f,-0.243681f,-0.255259f,-0.322941f,-0.392818f,-0.430800f,-0.482087f,-0.454211f,-0.415500f,-0.365219f,-0.335479f,-0.331787f,
-0.347399f,-0.395628f,-0.429942f,-0.443724f,-0.430932f,-0.361163f,-0.303096f,-0.287581f,-0.296932f,-0.354642f,-0.406622f,-0.428391f,-0.445822f,-0.429284f,-0.361486f,-0.305890f,-0.263168f,-0.246710f,-0.274015f,-0.326901f,
-0.361876f,-0.365851f,-0.355750f,-0.322201f,-0.175302f,-0.031804f,-0.051126f,0.044946f,-0.112488f,0.014163f,-0.032254f,-0.102629f,-0.019812f,-0.124555f,-0.192313f,0.035665f,0.086173f,0.147297f,0.159162f,0.164545f,
0.122258f,0.104337f,0.082899f,0.075498f,0.093214f,0.072421f,0.033682f,-0.042933f,-0.084540f,-0.126359f,-0.119805f,-0.086833f,-0.022714f,0.055922f,0.124821f,0.161321f,0.197822f,0.182688f,0.192684f,0.209917f,
0.224546f,0.230501f,0.258273f,0.235255f,0.193754f,0.149953f,0.125876f,0.133850f,0.170571f,0.224518f,0.291278f,0.347718f,0.358116f,0.366077f,0.344802f,0.308183f,0.297390f,0.275158f,0.272877f,0.245812f,
0.185435f,0.116400f,0.045180f,-0.013355f,-0.038330f,-0.028924f,-0.001117f,0.029923f,0.079221f,0.109725f,0.101913f,0.076367f,0.060656f,0.020252f,0.013199f,0.015035f,0.028733f,-0.015916f,-0.056046f,-0.104193f,
-0.145982f,-0.175873f,-0.151677f,-0.107019f,-0.070669f,-0.002663f,0.043057f,0.073727f,0.062370f,0.058718f,0.045771f,0.044897f,0.056909f,0.104053f,0.088736f,0.083060f,0.034171f,0.017008f,-0.016583f,0.005558f,
0.045950f,0.109191f,0.194253f,0.272433f,0.317964f,0.316119f,0.342558f,0.319960f,0.317207f,0.319688f,0.322446f,0.330928f,0.294110f,0.256713f,0.182459f,0.123256f,0.094524f,0.090681f,0.122514f,0.143079f,
0.216421f,0.237849f,0.268414f,0.242397f,0.193455f,0.133921f,0.105117f,0.040545f,0.032872f,0.043826f,0.007047f,-0.024717f,0.174612f,-0.093316f,-0.172953f,0.001987f,-0.105976f,0.049643f,0.076576f,-0.005408f,
0.103086f,0.030233f,0.059679f,0.286922f,0.288741f,0.249962f,0.186327f,0.134140f,0.151403f,0.198751f,0.265668f,0.325384f,0.339124f,0.331823f,0.277355f,0.182585f,0.111474f,0.063979f,0.032291f,0.001106f,
0.000712f,-0.056301f,-0.135579f,-0.215165f,-0.276049f,-0.307792f,-0.302175f,-0.246911f,-0.205785f,-0.201751f,-0.228349f,-0.279120f,-0.337086f,-0.399878f,-0.420653f,-0.419067f,-0.381276f,-0.334220f,-0.331164f,-0.351118f,
-0.378723f,-0.416115f,-0.375507f,-0.309413f,-0.202793f,-0.125125f,-0.059832f,-0.053612f,-0.080337f,-0.099613f,-0.153575f,-0.132759f,-0.098780f,-0.052505f,-0.007453f,-0.007625f,-0.019054f,-0.055501f,-0.074630f,-0.048167f,
-0.004060f,0.063754f,0.133992f,0.155948f,0.155368f,0.156367f,0.102337f,0.084551f,0.062656f,0.058746f,0.099850f,0.127681f,0.146630f,0.114076f,0.065120f,0.003775f,-0.025285f,-0.024486f,0.024689f,0.056028f,
0.100572f,0.095390f,0.037008f,-0.063868f,-0.157187f,-0.235280f,-0.300504f,-0.303753f,-0.311077f,-0.312746f,-0.305829f,-0.338488f,-0.417504f,-0.472089f,-0.493446f,-0.448394f,-0.373017f,-0.262524f,-0.207403f,-0.177849f,
-0.195393f,-0.228590f,-0.246149f,-0.260928f,-0.240738f,-0.217767f,-0.167376f,-0.105076f,-0.092661f,-0.105865f,-0.137880f,-0.152160f,-0.131244f,-0.060143f,0.021651f,0.123769f,0.204589f,0.223312f,0.246067f,0.204203f,
0.111535f,-0.048287f,-0.109477f,-0.081334f,-0.238494f,-0.027535f,0.036543f,-0.110853f,-0.042829f,0.026014f,0.036883f,-0.089496f,-0.064024f,-0.034545f,0.010824f,0.050953f,0.096407f,0.105604f,0.104985f,0.094918f,
0.104758f,0.122211f,0.161780f,0.174697f,0.171217f,0.118989f,0.088980f,0.045567f,0.001221f,-0.014288f,-0.013676f,-0.003670f,-0.000432f,0.007254f,-0.015077f,-0.041212f,-0.068109f,-0.087051f,-0.106228f,-0.094716f,
-0.080559f,-0.092562f,-0.134312f,-0.184794f,-0.220665f,-0.252132f,-0.263890f,-0.229900f,-0.205469f,-0.183326f,-0.161821f,-0.152218f,-0.140795f,-0.139789f,-0.125402f,-0.122838f,-0.065360f,-0.040397f,-0.030087f,-0.050147f,
-0.065355f,-0.086182f,-0.108527f,-0.092419f,-0.089190f,-0.050124f,-0.015518f,0.010355f,0.010290f,0.017014f,-0.008326f,-0.025936f,-0.028678f,0.015444f,0.022191f,0.021570f,0.008668f,-0.036217f,-0.066008f,-0.091630f,
-0.091703f,-0.090174f,-0.049768f,-0.017785f,0.016559f,0.032544f,0.024354f,0.025415f,0.007515f,0.015440f,0.022435f,0.039249f,0.066030f,0.039666f,-0.000693f,-0.063606f,-0.115609f,-0.151669f,-0.162525f,-0.151882f,
-0.153414f,-0.140016f,-0.120296f,-0.127892f,-0.132554f,-0.142981f,-0.156773f,-0.161274f,-0.143586f,-0.097863f,-0.081274f,-0.094041f,-0.122333f,-0.175428f,-0.210727f,-0.215183f,-0.195876f,-0.171006f,-0.105844f,-0.072087f,
-0.012927f,0.002821f,0.017818f,0.007675f,0.025106f,0.013932f,0.062892f,0.108489f,-0.032200f,0.199733f,-0.125977f,0.186606f,0.006565f,0.135738f,0.239824f,0.094615f,0.274719f,0.195231f,0.259545f,0.309690f,
0.309894f,0.270473f,0.255722f,0.189914f,0.153176f,0.113838f,0.099559f,0.078813f,0.092054f,0.115884f,0.129676f,0.160242f,0.170389f,0.194924f,0.215583f,0.250349f,0.241254f,0.221878f,0.188182f,0.129690f,
0.076287f,0.025546f,-0.011161f,-0.042075f,-0.068368f,-0.061952f,-0.024627f,-0.028077f,0.001112f,0.015034f,0.018777f,0.052136f,0.044671f,0.055448f,0.033245f,-0.021478f,-0.070692f,-0.128949f,-0.167964f,-0.207315f,
-0.224479f,-0.226781f,-0.225225f,-0.213570f,-0.181776f,-0.156593f,-0.142475f,-0.107265f,-0.079385f,-0.036752f,-0.036692f,-0.045406f,-0.065371f,-0.110071f,-0.160698f,-0.217981f,-0.245289f,-0.261011f,-0.256868f,-0.237264f,
-0.201741f,-0.162046f,-0.141973f,-0.103211f,-0.089653f,-0.047144f,-0.000544f,0.034512f,0.061137f,0.039541f,0.017798f,-0.005216f,-0.050293f,-0.083838f,-0.114738f,-0.118143f,-0.099349f,-0.088151f,-0.070478f,-0.066475f,
-0.045326f,-0.027230f,-0.008789f,0.019060f,0.051765f,0.063259f,0.041406f,0.019362f,-0.027617f,-0.097923f,-0.144499f,-0.182371f,-0.215102f,-0.210809f,-0.212015f,-0.201115f,-0.206340f,-0.183766f,-0.157304f,-0.115470f,
-0.134914f,-0.104232f,-0.084398f,-0.093779f,-0.107355f,-0.125115f,-0.180754f,-0.214772f,-0.257506f,-0.255562f,-0.268725f,-0.263939f,-0.235035f,-0.200404f,-0.169139f,-0.137388f,-0.207053f,-0.060055f,0.039316f,-0.025529f,
0.035039f,0.037616f,0.026899f,0.001904f,0.073981f,-0.044261f,-0.048960f,-0.040155f,-0.084805f,-0.126064f,-0.174680f,-0.194898f,-0.179347f,-0.162267f,-0.136193f,-0.112886f,-0.082109f,-0.071492f,-0.047948f,-0.049544f,
-0.059772f,-0.064054f,-0.086017f,-0.118541f,-0.128571f,-0.167044f,-0.170757f,-0.200799f,-0.197338f,-0.177862f,-0.157885f,-0.116479f,-0.072827f,-0.044842f,-0.016746f,0.003367f,-0.014634f,-0.009518f,-0.031914f,-0.041683f,
-0.062728f,-0.074985f,-0.079768f,-0.122857f,-0.134347f,-0.145211f,-0.123460f,-0.077710f,-0.052841f,-0.027382f,-0.005021f,0.013359f,0.054782f,0.041002f,0.032193f,0.034917f,0.018034f,0.011215f,-0.007650f,-0.024268f,
-0.047573f,-0.073992f,-0.051987f,-0.027660f,0.034427f,0.075191f,0.124249f,0.167559f,0.205908f,0.245984f,0.236923f,0.283234f,0.281571f,0.290887f,0.266307f,0.251636f,0.262957f,0.250503f,0.218225f,0.207323f,
0.226274f,0.245928f,0.304574f,0.337320f,0.375237f,0.391145f,0.411792f,0.395116f,0.384027f,0.357408f,0.363414f,0.333080f,0.308937f,0.301565f,0.269766f,0.221664f,0.208025f,0.189978f,0.218528f,0.251744f,
0.288822f,0.324563f,0.345301f,0.358007f,0.377773f,0.369049f,0.385921f,0.375496f,0.349107f,0.329141f,0.291924f,0.293835f,0.247144f,0.207815f,0.203440f,0.193166f,0.201932f,0.257738f,0.282495f,0.335959f,
0.338907f,0.377742f,0.367090f,0.386327f,-0.253346f,0.121857f,0.484309f,0.124161f,0.210323f,0.123988f,0.079174f,0.152944f,0.278792f,0.034833f,0.092850f,0.131834f,0.178598f,0.205020f,0.194976f,0.188906f,
0.157457f,0.159130f,0.183657f,0.176294f,0.187132f,0.194971f,0.198167f,0.204080f,0.172812f,0.135664f,0.100032f,0.095124f,0.100696f,0.114789f,0.114535f,0.098565f,0.061689f,0.041674f,0.030021f,0.052684f,
0.040107f,0.059289f,0.052550f,0.073692f,0.072126f,0.062308f,0.034783f,0.022397f,0.034995f,0.064362f,0.076878f,0.104575f,0.120841f,0.088509f,0.105614f,0.102190f,0.122796f,0.157011f,0.172027f,0.183732f,
0.220449f,0.219006f,0.208956f,0.181070f,0.174040f,0.163821f,0.194488f,0.202206f,0.217098f,0.230378f,0.180173f,0.191612f,0.167708f,0.173012f,0.167041f,0.164175f,0.152266f,0.147836f,0.140757f,0.107659f,
0.070197f,0.030220f,0.011002f,0.011998f,0.016030f,0.029644f,0.002607f,-0.026940f,-0.064226f,-0.081523f,-0.100497f,-0.093754f,-0.119587f,-0.119785f,-0.112909f,-0.120556f,-0.151142f,-0.156082f,-0.205577f,-0.210734f,
-0.237509f,-0.214130f,-0.182053f,-0.169126f,-0.178083f,-0.203557f,-0.212885f,-0.208232f,-0.202726f,-0.193850f,-0.144725f,-0.135089f,-0.091868f,-0.076227f,-0.098834f,-0.077490f,-0.100045f,-0.090939f,-0.074210f,-0.047558f,
-0.006642f,0.022460f,0.022040f,0.047658f,0.003473f,0.002586f,-0.012335f,0.008777f,0.022432f,0.062576f,0.088675f,0.082663f,-0.006453f,0.091726f,-0.007830f,0.043329f,-0.157658f,-0.124123f,-0.123347f,-0.322264f,
-0.239029f,-0.237250f,-0.275641f,-0.100369f,-0.124332f,-0.160068f,-0.175862f,-0.180172f,-0.173374f,-0.128256f,-0.082937f,-0.090401f,-0.108073f,-0.178213f,-0.263725f,-0.309728f,-0.291616f,-0.269869f,-0.197738f,-0.128248f,
-0.057817f,-0.043330f,-0.042049f,-0.051431f,-0.019337f,0.009054f,0.063741f,0.108821f,0.113214f,0.081612f,0.020828f,-0.046942f,-0.107781f,-0.109889f,-0.082494f,-0.020115f,0.039840f,0.083444f,0.099268f,0.091787f,
0.069393f,0.066020f,0.108839f,0.176110f,0.194262f,0.180708f,0.134500f,0.056404f,-0.028674f,-0.075574f,-0.094147f,-0.059050f,-0.009530f,0.061466f,0.110858f,0.139260f,0.148596f,0.176090f,0.183973f,0.226017f,
0.301128f,0.360376f,0.365896f,0.364166f,0.297251f,0.233354f,0.195290f,0.156695f,0.186820f,0.217752f,0.273282f,0.319247f,0.336103f,0.326015f,0.310267f,0.296135f,0.297017f,0.327094f,0.368098f,0.407414f,
0.367000f,0.305381f,0.188144f,0.098260f,0.044747f,0.012283f,0.040249f,0.072842f,0.108022f,0.143341f,0.161643f,0.137915f,0.121506f,0.115215f,0.145040f,0.211798f,0.237984f,0.254645f,0.207822f,0.152365f,
0.054771f,-0.017503f,-0.053142f,-0.052206f,-0.008750f,0.034919f,0.110417f,0.136794f,0.160723f,0.128070f,0.098479f,0.106710f,0.109047f,0.131277f,0.153972f,0.189966f,0.167546f,0.107539f,0.001135f,-0.063311f,
-0.022956f,-0.135516f,-0.387117f,-0.168629f,-0.147075f,-0.112660f,-0.009249f,0.076991f,0.021362f,-0.091723f,-0.085347f,0.002794f,0.035707f,0.061018f,0.083761f,0.086320f,0.057902f,0.029558f,-0.000116f,-0.004201f,
0.042980f,0.101406f,0.145687f,0.181228f,0.184406f,0.143496f,0.081146f,0.046035f,0.018451f,0.012880f,0.025136f,0.020166f,0.003658f,-0.066988f,-0.090105f,-0.119009f,-0.115139f,-0.055924f,0.019733f,0.075056f,
0.120579f,0.134388f,0.119071f,0.073778f,0.075753f,0.060662f,0.090309f,0.138486f,0.182782f,0.169986f,0.141707f,0.122491f,0.113374f,0.139362f,0.195950f,0.275659f,0.340505f,0.383954f,0.393175f,0.355008f,
0.336693f,0.284562f,0.275891f,0.273274f,0.285355f,0.278033f,0.250282f,0.230350f,0.162449f,0.116749f,0.101873f,0.115746f,0.171571f,0.177961f,0.191153f,0.181644f,0.141290f,0.089465f,0.018903f,-0.016336f,
-0.042380f,-0.056957f,-0.067465f,-0.076205f,-0.109389f,-0.183567f,-0.250693f,-0.270146f,-0.255528f,-0.195426f,-0.132223f,-0.072387f,-0.042657f,-0.054834f,-0.068650f,-0.116068f,-0.147979f,-0.160738f,-0.129970f,-0.089863f,
-0.059267f,-0.076631f,-0.106756f,-0.140289f,-0.153164f,-0.152748f,-0.069883f,-0.013689f,0.080155f,0.147840f,0.179134f,0.173421f,0.166890f,0.144395f,0.113698f,0.129650f,0.134033f,0.212003f,0.228896f,0.226430f,
0.199268f,0.162838f,0.135976f,0.111125f,0.161422f,0.212492f,0.258824f,0.303044f,0.185655f,-0.055474f,-0.136943f,-0.006563f,-0.087621f,-0.005891f,-0.009282f,0.026947f,-0.095392f,0.028704f,0.106331f,-0.336179f,
-0.331012f,-0.317873f,-0.319703f,-0.270561f,-0.205970f,-0.126074f,-0.072920f,-0.021105f,0.000895f,-0.004635f,-0.021077f,-0.067528f,-0.097471f,-0.103895f,-0.155223f,-0.191125f,-0.237738f,-0.298755f,-0.343401f,-0.371652f,
-0.387255f,-0.370514f,-0.351997f,-0.310124f,-0.300522f,-0.321695f,-0.347993f,-0.398948f,-0.423228f,-0.442393f,-0.441632f,-0.439708f,-0.456910f,-0.467977f,-0.474833f,-0.467817f,-0.453423f,-0.433582f,-0.365565f,-0.307906f,
-0.241465f,-0.174332f,-0.148482f,-0.141445f,-0.140722f,-0.155635f,-0.153348f,-0.147900f,-0.138118f,-0.144805f,-0.162973f,-0.185708f,-0.208993f,-0.213444f,-0.210332f,-0.191390f,-0.166820f,-0.143287f,-0.127235f,-0.154128f,
-0.192979f,-0.235074f,-0.290185f,-0.320984f,-0.355994f,-0.370035f,-0.398445f,-0.397197f,-0.432898f,-0.461375f,-0.487876f,-0.470134f,-0.470450f,-0.428782f,-0.370512f,-0.315118f,-0.281454f,-0.265429f,-0.272435f,-0.305801f,
-0.320022f,-0.323523f,-0.320457f,-0.324480f,-0.340353f,-0.365096f,-0.379140f,-0.416453f,-0.432721f,-0.448700f,-0.413363f,-0.369450f,-0.320644f,-0.255908f,-0.212010f,-0.224997f,-0.219251f,-0.236213f,-0.241497f,-0.213357f,
-0.194384f,-0.183560f,-0.168349f,-0.176301f,-0.177001f,-0.155307f,-0.150066f,-0.094629f,-0.057322f,0.008122f,0.086620f,0.119486f,0.164009f,0.149334f,0.118844f,0.093655f,-0.022963f,0.170534f,0.232882f,0.165132f,
0.203619f,0.063687f,0.056858f,-0.209282f,-0.172315f,-0.011850f,-0.081864f,-0.164518f,-0.173047f,-0.169807f,-0.153904f,-0.150183f,-0.146184f,-0.116641f,-0.112847f,-0.101891f,-0.115523f,-0.135110f,-0.176267f,-0.200365f,
-0.212147f,-0.204285f,-0.210653f,-0.159365f,-0.146323f,-0.103918f,-0.064503f,-0.035304f,0.020444f,0.056298f,0.096935f,0.126773f,0.129028f,0.141017f,0.137303f,0.086595f,0.049052f,0.032833f,0.022318f,0.013455f,
0.017890f,0.055094f,0.058179f,0.062335f,0.061235f,0.080127f,0.098504f,0.108789f,0.109748f,0.094066f,0.067658f,0.041586f,-0.026777f,-0.065398f,-0.120566f,-0.112911f,-0.147377f,-0.133202f,-0.128914f,-0.100809f,
-0.093718f,-0.065581f,-0.031668f,0.001646f,0.030115f,0.059510f,0.066036f,0.084445f,0.049950f,0.016140f,-0.007076f,-0.045093f,-0.049620f,-0.070247f,-0.070671f,-0.047599f,-0.033359f,-0.009778f,-0.000260f,0.029821f,
0.045992f,0.079852f,0.109831f,0.117624f,0.125314f,0.097067f,0.080310f,0.036404f,0.000184f,-0.015933f,-0.009640f,-0.026413f,0.001610f,0.010574f,0.016593f,0.032915f,0.067997f,0.095565f,0.136122f,0.155404f,
0.188457f,0.163157f,0.155687f,0.131758f,0.078250f,0.028755f,-0.017881f,-0.040345f,-0.055154f,-0.067023f,-0.083294f,-0.060655f,-0.045286f,-0.060229f,-0.053454f,-0.049689f,-0.048479f,-0.028112f,-0.032080f,-0.040865f,
-0.056016f,-0.072081f,-0.137317f,-0.201013f,0.212487f,-0.007529f,-0.047240f,-0.112994f,-0.138670f,0.027489f,-0.032262f,0.022100f,0.005817f,-0.023087f,-0.041725f,-0.133340f,-0.100955f,-0.011147f,0.044730f,0.055173f,
0.026073f,-0.023392f,-0.105137f,-0.145475f,-0.160826f,-0.120966f,-0.081834f,-0.043052f,-0.027248f,-0.063624f,-0.094359f,-0.085080f,-0.073567f,-0.020136f,0.054904f,0.089385f,0.097756f,0.049483f,-0.012827f,-0.090167f,
-0.129083f,-0.151054f,-0.118144f,-0.068739f,-0.028157f,-0.018967f,-0.051562f,-0.059448f,-0.076564f,-0.056926f,-0.006315f,0.080950f,0.120679f,0.123245f,0.078483f,-0.002146f,-0.051248f,-0.089984f,-0.083399f,-0.055096f,
0.007917f,0.043416f,0.051791f,0.012936f,0.010352f,-0.017103f,0.018463f,0.079692f,0.134593f,0.174464f,0.190414f,0.156432f,0.097203f,0.011614f,-0.045361f,-0.077548f,-0.060751f,-0.033629f,-0.001449f,-0.004034f,
-0.047775f,-0.076427f,-0.114644f,-0.124560f,-0.098807f,-0.050866f,0.004899f,0.039987f,0.002430f,-0.065566f,-0.145891f,-0.233981f,-0.289596f,-0.280145f,-0.248065f,-0.213079f,-0.169821f,-0.199825f,-0.244085f,-0.261737f,
-0.292167f,-0.272614f,-0.214009f,-0.154915f,-0.108687f,-0.098837f,-0.120773f,-0.207364f,-0.271885f,-0.326545f,-0.353473f,-0.332447f,-0.274973f,-0.228566f,-0.186614f,-0.207689f,-0.244985f,-0.261366f,-0.254018f,-0.214232f,
-0.174380f,-0.079976f,-0.042360f,-0.031611f,-0.050161f,-0.115945f,-0.188658f,-0.260908f,-0.271703f,-0.258903f,-0.216477f,-0.175751f,-0.144221f,0.043455f,0.111865f,0.150839f,0.125992f,0.030969f,-0.016450f,-0.113430f,
-0.084800f,-0.053683f,-0.073842f,0.013360f,0.051512f,0.111836f,0.178008f,0.272039f,0.334227f,0.367611f,0.368435f,0.350326f,0.285649f,0.210196f,0.129091f,0.061900f,-0.004180f,-0.072181f,-0.132765f,-0.145870f,
-0.145484f,-0.133259f,-0.096232f,-0.055052f,-0.001878f,0.036424f,0.055698f,0.063968f,0.038985f,-0.017443f,-0.084588f,-0.122988f,-0.175836f,-0.187860f,-0.206773f,-0.195226f,-0.165994f,-0.117745f,-0.018055f,0.080989f,
0.178821f,0.285251f,0.360657f,0.427344f,0.445943f,0.438112f,0.409914f,0.369306f,0.353226f,0.304420f,0.271442f,0.251024f,0.239697f,0.241560f,0.267654f,0.291337f,0.381309f,0.448385f,0.528832f,0.588960f,
0.600335f,0.583037f,0.542731f,0.457312f,0.373010f,0.280076f,0.216016f,0.155178f,0.123591f,0.074094f,0.035310f,0.013890f,-0.015704f,0.025136f,0.038105f,0.095784f,0.125283f,0.135474f,0.141756f,0.124186f,
0.037962f,-0.053149f,-0.129230f,-0.192282f,-0.258135f,-0.316588f,-0.375307f,-0.411653f,-0.420750f,-0.426965f,-0.383456f,-0.304773f,-0.225470f,-0.150566f,-0.063980f,-0.023133f,0.021407f,0.036696f,0.005921f,-0.012834f,
-0.049604f,-0.056737f,-0.067501f,-0.061503f,-0.076187f,-0.065219f,0.003965f,0.036869f,0.120946f,0.201845f,0.300625f,0.396199f,0.460140f,0.471904f,0.479481f,0.444266f,0.422194f,0.339982f,0.303212f,0.226485f,
0.215040f,-0.125882f,-0.127762f,-0.161144f,-0.140871f,-0.133376f,-0.124766f,0.005134f,-0.040888f,-0.146863f,-0.134278f,-0.241486f,-0.212129f,-0.223159f,-0.251700f,-0.274636f,-0.311313f,-0.356121f,-0.372250f,-0.368587f,
-0.353188f,-0.350019f,-0.332854f,-0.319595f,-0.298355f,-0.299594f,-0.292905f,-0.255660f,-0.254900f,-0.243658f,-0.249043f,-0.265388f,-0.293359f,-0.316674f,-0.361020f,-0.370089f,-0.339135f,-0.336391f,-0.282301f,-0.265250f,
-0.253590f,-0.228121f,-0.200970f,-0.185350f,-0.134019f,-0.094665f,-0.073945f,-0.059262f,-0.041394f,-0.070695f,-0.088108f,-0.110290f,-0.092413f,-0.091747f,-0.061656f,-0.026920f,-0.017371f,0.009111f,0.022715f,0.046507f,
0.077959f,0.093663f,0.134357f,0.142081f,0.156198f,0.151315f,0.134267f,0.117253f,0.088661f,0.064956f,0.070672f,0.080290f,0.086393f,0.090878f,0.098899f,0.086824f,0.090869f,0.090804f,0.092935f,0.090420f,
0.073451f,0.062490f,0.054444f,-0.005890f,-0.060589f,-0.124520f,-0.161012f,-0.189827f,-0.203789f,-0.207646f,-0.210021f,-0.213826f,-0.207265f,-0.230342f,-0.236225f,-0.238272f,-0.228169f,-0.211408f,-0.203289f,-0.176733f,
-0.192645f,-0.226369f,-0.242283f,-0.265431f,-0.286074f,-0.256120f,-0.236832f,-0.195127f,-0.182138f,-0.148967f,-0.099160f,-0.098781f,-0.071893f,-0.044731f,-0.033798f,0.022558f,0.035373f,0.067957f,0.024453f,0.024020f,
-0.009187f,-0.060348f,-0.077449f,-0.081402f,-0.061719f,-0.008566f,0.009808f,0.010562f,-0.089337f,-0.019759f,0.152230f,0.078263f,-0.054841f,0.006322f,-0.004198f,-0.054096f,-0.071065f,-0.084144f,-0.129962f,-0.174959f,
-0.152551f,-0.131682f,-0.109294f,-0.104027f,-0.085853f,-0.028263f,0.058537f,0.156678f,0.245885f,0.298551f,0.305509f,0.266178f,0.240954f,0.188264f,0.126624f,0.088786f,0.103987f,0.117124f,0.098875f,0.089590f,
0.082710f,0.074153f,0.110995f,0.170124f,0.242521f,0.297315f,0.305971f,0.286560f,0.232732f,0.168414f,0.096036f,0.040719f,0.024736f,0.014696f,0.026755f,0.025355f,0.019501f,0.007855f,0.028188f,0.085570f,
0.155147f,0.234371f,0.310382f,0.361772f,0.362906f,0.313375f,0.270876f,0.205196f,0.163918f,0.133994f,0.147449f,0.136638f,0.149847f,0.136228f,0.115529f,0.128615f,0.143950f,0.197440f,0.237136f,0.296299f,
0.309416f,0.274556f,0.192089f,0.101236f,0.011919f,-0.077196f,-0.138095f,-0.177739f,-0.185328f,-0.201224f,-0.225737f,-0.249812f,-0.247809f,-0.246115f,-0.211478f,-0.152749f,-0.074130f,-0.013112f,0.037310f,0.005984f,
-0.046386f,-0.091384f,-0.138576f,-0.211561f,-0.233391f,-0.228876f,-0.240268f,-0.241841f,-0.243801f,-0.283747f,-0.268544f,-0.233410f,-0.185269f,-0.104588f,-0.044517f,0.001564f,0.026571f,-0.015432f,-0.073999f,-0.119573f,
-0.167116f,-0.222822f,-0.219956f,-0.186114f,-0.174843f,-0.156643f,-0.131920f,-0.136786f,-0.116105f,-0.071263f,-0.002494f,0.042639f,0.119065f,0.142638f,0.182430f,0.150191f,-0.305880f,0.141772f,0.016603f,0.080673f,
0.035101f,-0.061182f,-0.073711f,0.040833f,-0.095440f,-0.177904f,-0.054110f,0.088477f,0.189786f,0.314496f,0.370906f,0.391723f,0.362169f,0.320993f,0.260975f,0.233564f,0.193901f,0.176864f,0.166717f,0.166740f,
0.154653f,0.150729f,0.203119f,0.263647f,0.345432f,0.472151f,0.551066f,0.614681f,0.623959f,0.593442f,0.532447f,0.470435f,0.411787f,0.349011f,0.314002f,0.296904f,0.272565f,0.241508f,0.212349f,0.235999f,
0.268019f,0.312282f,0.362517f,0.448899f,0.450366f,0.438232f,0.376645f,0.297148f,0.211397f,0.155891f,0.092820f,0.059483f,0.043534f,-0.000245f,-0.027505f,-0.041736f,-0.026174f,0.010762f,0.066982f,0.148880f,
0.231067f,0.266865f,0.284167f,0.232719f,0.157088f,0.093456f,0.023278f,-0.007172f,-0.025352f,-0.033545f,-0.073914f,-0.088911f,-0.083384f,-0.086881f,-0.038508f,0.019467f,0.117766f,0.237925f,0.319315f,0.369352f,
0.376278f,0.362806f,0.330590f,0.281323f,0.239497f,0.234508f,0.248408f,0.249722f,0.262598f,0.257474f,0.259453f,0.280068f,0.321928f,0.394983f,0.485940f,0.551076f,0.589847f,0.614397f,0.552639f,0.469167f,
0.382325f,0.300203f,0.240720f,0.193260f,0.166147f,0.121154f,0.090587f,0.039618f,0.028963f,0.043857f,0.074132f,0.114846f,0.203751f,0.270065f,0.293209f,0.307339f,0.240571f,0.161394f,0.104203f,0.020875f,
-0.000839f,-0.034310f,-0.056358f,-0.046288f,0.258410f,-0.028100f,0.097060f,0.155158f,0.307169f,0.131778f,0.005535f,0.140294f,0.244588f,0.096123f,0.262358f,-0.098076f,-0.132309f,-0.161028f,-0.181275f,-0.158764f,
-0.103521f,-0.014747f,0.054761f,0.092701f,0.076710f,0.040275f,-0.005206f,-0.049774f,-0.068059f,-0.036894f,-0.047617f,-0.033888f,-0.047328f,-0.076747f,-0.113567f,-0.114524f,-0.092287f,-0.017826f,0.059111f,0.119633f,
0.156179f,0.136030f,0.085600f,0.007141f,-0.060186f,-0.084389f,-0.072835f,-0.074548f,-0.094526f,-0.124403f,-0.180054f,-0.256847f,-0.283416f,-0.275196f,-0.211823f,-0.157033f,-0.116292f,-0.099179f,-0.160268f,-0.199849f,
-0.294566f,-0.330971f,-0.352445f,-0.358934f,-0.377366f,-0.400821f,-0.416351f,-0.475524f,-0.511055f,-0.535983f,-0.546828f,-0.499431f,-0.444563f,-0.382720f,-0.335727f,-0.366840f,-0.405154f,-0.453723f,-0.494081f,-0.514606f,
-0.488926f,-0.478215f,-0.468438f,-0.474900f,-0.471180f,-0.497141f,-0.507018f,-0.510694f,-0.437454f,-0.338690f,-0.264820f,-0.168575f,-0.109011f,-0.120460f,-0.167417f,-0.185756f,-0.202496f,-0.179240f,-0.131212f,-0.119471f,
-0.101497f,-0.119750f,-0.160880f,-0.212829f,-0.234746f,-0.230458f,-0.175241f,-0.103815f,-0.041474f,0.015848f,-0.019185f,-0.094430f,-0.166861f,-0.252999f,-0.292907f,-0.281093f,-0.302806f,-0.318666f,-0.337599f,-0.381627f,
-0.424189f,-0.474495f,-0.503682f,-0.525866f,-0.464424f,-0.414382f,-0.298532f,-0.265452f,-0.294073f,-0.348360f,-0.408659f,-0.478201f,0.060957f,0.071123f,0.027436f,0.062427f,-0.088442f,-0.033335f,-0.023404f,-0.193309f,
-0.095002f,-0.032949f,-0.105722f,-0.210219f,-0.173626f,-0.094116f,0.046008f,0.165736f,0.212801f,0.221463f,0.156169f,0.073224f,-0.023808f,-0.050308f,-0.056897f,-0.054594f,-0.041548f,-0.055944f,-0.070836f,-0.094152f,
-0.088641f,-0.057991f,0.012615f,0.103742f,0.180806f,0.211771f,0.183682f,0.107082f,-0.008578f,-0.100106f,-0.159501f,-0.169126f,-0.165543f,-0.149017f,-0.148883f,-0.156361f,-0.178827f,-0.179760f,-0.140733f,-0.066851f,
0.051919f,0.136644f,0.176301f,0.161263f,0.074897f,-0.028959f,-0.102529f,-0.118110f,-0.113696f,-0.098723f,-0.077240f,-0.087005f,-0.103755f,-0.122313f,-0.121398f,-0.094831f,-0.015325f,0.090735f,0.172704f,0.200491f,
0.167586f,0.082866f,-0.019651f,-0.154530f,-0.214463f,-0.218695f,-0.254658f,-0.243209f,-0.243893f,-0.277116f,-0.305497f,-0.315932f,-0.314427f,-0.270190f,-0.181102f,-0.074273f,-0.008086f,-0.013842f,-0.044590f,-0.140456f,
-0.255920f,-0.314495f,-0.346825f,-0.314526f,-0.298696f,-0.280020f,-0.309076f,-0.329928f,-0.362800f,-0.364303f,-0.323911f,-0.264188f,-0.154256f,-0.085652f,-0.021209f,-0.042246f,-0.107795f,-0.224954f,-0.313449f,-0.353319f,
-0.369285f,-0.368331f,-0.337789f,-0.322670f,-0.331663f,-0.337072f,-0.348442f,-0.334841f,-0.296394f,-0.227319f,-0.105229f,0.025318f,0.055091f,0.031847f,-0.011316f,-0.150671f,-0.231487f,-0.308479f,-0.307572f,-0.299123f,
-0.052543f,0.012897f,-0.029512f,-0.043128f,-0.300113f,-0.130892f,-0.135780f,-0.037825f,-0.184430f,0.072208f,0.158088f,0.383793f,0.451481f,0.466843f,0.422268f,0.366431f,0.316981f,0.288280f,0.247724f,0.204193f,
0.105661f,0.000999f,-0.083122f,-0.131540f,-0.140247f,-0.079053f,0.008267f,0.096703f,0.169996f,0.169457f,0.143770f,0.080061f,0.018344f,-0.041501f,-0.084941f,-0.130667f,-0.202406f,-0.268504f,-0.320702f,-0.397265f,
-0.408108f,-0.373006f,-0.262139f,-0.141488f,-0.030516f,0.046334f,0.082657f,0.070892f,0.021803f,-0.040632f,-0.078483f,-0.110114f,-0.142387f,-0.179987f,-0.246217f,-0.324882f,-0.402833f,-0.429470f,-0.377061f,-0.302992f,
-0.179968f,-0.079900f,0.002853f,0.006810f,-0.007434f,-0.064387f,-0.119843f,-0.191913f,-0.257305f,-0.286085f,-0.328489f,-0.365391f,-0.426432f,-0.467966f,-0.482310f,-0.438243f,-0.355441f,-0.209918f,-0.069025f,0.052672f,
0.137479f,0.173990f,0.168811f,0.148748f,0.093883f,0.073869f,0.069359f,0.041042f,0.011992f,-0.016552f,-0.097062f,-0.140775f,-0.176006f,-0.123365f,-0.032193f,0.083549f,0.216004f,0.308639f,0.348662f,0.331991f,
0.323244f,0.271584f,0.198633f,0.171760f,0.122233f,0.118459f,0.080342f,0.020047f,-0.063726f,-0.091157f,-0.072896f,-0.018316f,0.093671f,0.218727f,0.315398f,0.390826f,0.391715f,0.388770f,0.345834f,0.258024f,
0.196628f,0.166291f,0.151061f,0.121390f,0.099936f,0.043132f,0.000678f,-0.027280f,0.146008f,0.024606f,0.102068f,0.009913f,0.016809f,0.014684f,-0.041795f,0.026817f,0.023444f,-0.026538f,-0.011646f,-0.039614f,
0.017773f,0.044575f,0.019778f,-0.011420f,-0.070955f,-0.091066f,-0.122363f,-0.092715f,-0.065459f,-0.054663f,-0.060362f,-0.065915f,-0.079349f,-0.099359f,-0.084886f,-0.003939f,0.046058f,0.103067f,0.105299f,0.067933f,
0.018829f,-0.022723f,-0.079597f,-0.102895f,-0.101315f,-0.082835f,-0.087792f,-0.096281f,-0.126146f,-0.166655f,-0.202495f,-0.185610f,-0.134193f,-0.082223f,-0.057791f,-0.030701f,-0.088627f,-0.140022f,-0.182354f,-0.232013f,
-0.261368f,-0.234983f,-0.201965f,-0.187836f,-0.210487f,-0.227166f,-0.253930f,-0.271704f,-0.223389f,-0.170921f,-0.092772f,-0.040929f,-0.012940f,-0.009644f,-0.053376f,-0.095488f,-0.134634f,-0.149230f,-0.134713f,-0.110468f,
-0.079934f,-0.078947f,-0.087393f,-0.115283f,-0.140204f,-0.129899f,-0.058032f,-0.011700f,0.049415f,0.097686f,0.103647f,0.078054f,0.022722f,-0.001745f,-0.049121f,-0.056779f,-0.044691f,-0.020086f,-0.005020f,-0.028165f,
-0.075359f,-0.114004f,-0.145457f,-0.121145f,-0.079023f,-0.028264f,0.014268f,0.032393f,-0.010291f,-0.054505f,-0.121046f,-0.193847f,-0.205527f,-0.237600f,-0.179426f,-0.195406f,-0.180122f,-0.202664f,-0.238494f,-0.290985f,
-0.308472f,-0.268790f,-0.213577f,-0.150195f,-0.095874f,-0.074276f,-0.102022f,-0.149258f,-0.206511f,-0.234084f,-0.261711f,-0.250853f,-0.195010f,-0.161365f,-0.127419f,-0.168673f,-0.071941f,0.162627f,0.258992f,0.257218f,
0.247494f,0.130351f,0.072328f,0.096612f,-0.139151f,-0.177070f,-0.082423f,-0.209923f,-0.121704f,-0.012115f,0.053681f,0.099332f,0.081717f,0.038586f,0.021089f,0.008675f,-0.008868f,-0.034110f,-0.099624f,-0.171005f,
-0.260478f,-0.327812f,-0.345568f,-0.310383f,-0.215060f,-0.109692f,0.023514f,0.091925f,0.134083f,0.137662f,0.098907f,0.062621f,0.037482f,-0.002555f,-0.051273f,-0.095281f,-0.181691f,-0.270680f,-0.348523f,-0.371338f,
-0.357384f,-0.293424f,-0.171614f,-0.077281f,-0.029386f,-0.000162f,-0.015146f,-0.060906f,-0.097005f,-0.127778f,-0.149315f,-0.183037f,-0.248953f,-0.336263f,-0.402067f,-0.490157f,-0.524557f,-0.483290f,-0.390727f,-0.262126f,
-0.130499f,-0.056673f,0.024012f,0.049793f,0.040586f,0.019835f,0.004904f,-0.029709f,-0.036351f,-0.050406f,-0.082336f,-0.147716f,-0.191538f,-0.211930f,-0.201528f,-0.132273f,-0.019673f,0.118708f,0.240155f,0.310839f,
0.365328f,0.370446f,0.339739f,0.321726f,0.271249f,0.263122f,0.226323f,0.168735f,0.085982f,-0.036641f,-0.119198f,-0.152547f,-0.126765f,-0.083049f,0.026591f,0.135026f,0.213106f,0.256055f,0.239618f,0.202704f,
0.135658f,0.109404f,0.052114f,0.017712f,-0.017021f,-0.097701f,-0.176133f,-0.301770f,-0.348879f,-0.355825f,-0.328256f,-0.254026f,-0.129120f,-0.030210f,0.041474f,0.078936f,0.056784f,0.009651f,-0.002115f,-0.061223f,
-0.095767f,-0.117942f,-0.139377f,-0.166387f,-0.146010f,-0.029665f,-0.123052f,-0.072184f,-0.060005f,0.056923f,0.024743f,-0.087560f,-0.155528f,-0.167774f,-0.229284f,-0.168256f,-0.078215f,-0.005233f,0.023686f,0.018449f,
-0.011540f,-0.037585f,-0.005019f,0.042623f,0.128717f,0.246916f,0.334446f,0.368401f,0.346091f,0.330625f,0.322881f,0.337041f,0.403335f,0.455854f,0.500383f,0.509559f,0.429266f,0.353059f,0.268091f,0.206534f,
0.197472f,0.225574f,0.230485f,0.217583f,0.154835f,0.072637f,-0.035339f,-0.130185f,-0.192010f,-0.194502f,-0.169924f,-0.169678f,-0.190456f,-0.266008f,-0.344619f,-0.412472f,-0.419772f,-0.381075f,-0.296681f,-0.217906f,
-0.164481f,-0.173252f,-0.197858f,-0.230305f,-0.230411f,-0.176077f,-0.096038f,0.025814f,0.096036f,0.139952f,0.154302f,0.133725f,0.117543f,0.127844f,0.158018f,0.236300f,0.317928f,0.405884f,0.410024f,0.366726f,
0.328412f,0.279144f,0.272640f,0.338196f,0.419574f,0.529518f,0.601516f,0.629783f,0.602248f,0.578883f,0.550713f,0.544609f,0.575464f,0.628416f,0.698159f,0.703715f,0.640324f,0.544259f,0.460002f,0.364571f,
0.318131f,0.319970f,0.315769f,0.315201f,0.283884f,0.211450f,0.102400f,-0.023360f,-0.108099f,-0.155589f,-0.144667f,-0.096958f,-0.088498f,-0.089741f,-0.169992f,-0.268293f,-0.334131f,-0.383959f,-0.393837f,-0.363479f,
-0.267740f,-0.212286f,-0.159031f,-0.172660f,-0.197573f,-0.218368f,-0.217827f,-0.181500f,-0.098282f,0.010375f,0.092819f,0.156016f,0.298114f,-0.158334f,-0.202575f,-0.302927f,-0.195973f,-0.277228f,-0.370861f,-0.417866f,
-0.332081f,-0.215033f,-0.235626f,-0.279725f,-0.275154f,-0.242848f,-0.203468f,-0.160027f,-0.113667f,-0.016811f,0.013224f,0.015533f,-0.032765f,-0.121146f,-0.149522f,-0.176753f,-0.180592f,-0.164465f,-0.158705f,-0.141187f,
-0.110914f,-0.124298f,-0.097798f,-0.110151f,-0.086269f,-0.014799f,0.032088f,0.063189f,-0.001317f,-0.081606f,-0.141016f,-0.184676f,-0.218874f,-0.234252f,-0.212374f,-0.188988f,-0.175581f,-0.144304f,-0.149292f,-0.125473f,
-0.123422f,-0.094508f,-0.022441f,0.019201f,0.038285f,0.039458f,-0.043071f,-0.103118f,-0.162576f,-0.179040f,-0.173302f,-0.164902f,-0.152219f,-0.125754f,-0.109371f,-0.098013f,-0.102143f,-0.098398f,-0.076562f,-0.059026f,
-0.007465f,0.013691f,-0.017216f,-0.085789f,-0.173175f,-0.231389f,-0.281406f,-0.292459f,-0.292780f,-0.274617f,-0.262263f,-0.247063f,-0.247675f,-0.227798f,-0.212366f,-0.199111f,-0.153839f,-0.086266f,-0.025070f,0.009973f,
-0.042991f,-0.087783f,-0.143444f,-0.155950f,-0.159812f,-0.135999f,-0.115527f,-0.103557f,-0.076571f,-0.060325f,-0.057369f,-0.059353f,-0.070383f,-0.042069f,0.009740f,0.067530f,0.092544f,0.066661f,0.017580f,-0.063281f,
-0.118037f,-0.137946f,-0.142867f,-0.156096f,-0.139105f,-0.115645f,-0.086305f,-0.064781f,-0.045730f,-0.037115f,-0.024224f,-0.024059f,0.036607f,0.093946f,0.137727f,0.102689f,0.043839f,-0.037777f,-0.085278f,-0.152713f,
-0.007963f,0.078594f,0.326025f,0.096677f,-0.026553f,0.272936f,0.266944f,0.233004f,0.402852f,0.312335f,0.300780f,0.214489f,0.196955f,0.194388f,0.189777f,0.144764f,0.053110f,-0.050291f,-0.147064f,-0.210102f,
-0.190036f,-0.113481f,-0.032642f,-0.002866f,-0.008809f,-0.048189f,-0.093236f,-0.132772f,-0.175117f,-0.170753f,-0.140163f,-0.135020f,-0.150619f,-0.198323f,-0.276332f,-0.327393f,-0.300560f,-0.212985f,-0.085820f,0.021187f,
0.104994f,0.136140f,0.133772f,0.102467f,0.071365f,0.060835f,0.066579f,0.126164f,0.142921f,0.121075f,0.043017f,-0.050069f,-0.109245f,-0.108529f,-0.051363f,0.022889f,0.100607f,0.141765f,0.152253f,0.108008f,
0.048369f,-0.026997f,-0.077656f,-0.086242f,-0.094395f,-0.090323f,-0.147620f,-0.200687f,-0.296437f,-0.342902f,-0.378513f,-0.336730f,-0.248203f,-0.128275f,-0.072528f,-0.037883f,-0.039286f,-0.052306f,-0.116176f,-0.147437f,
-0.174500f,-0.192386f,-0.165317f,-0.186826f,-0.219312f,-0.328179f,-0.427403f,-0.479249f,-0.455711f,-0.383684f,-0.304101f,-0.218437f,-0.171359f,-0.140598f,-0.170165f,-0.161151f,-0.196022f,-0.205048f,-0.165640f,-0.107157f,
-0.063560f,-0.063268f,-0.112876f,-0.162714f,-0.230782f,-0.220293f,-0.177696f,-0.054676f,0.044474f,0.151800f,0.175530f,0.178916f,0.147463f,0.098764f,0.057697f,0.015630f,0.030874f,0.026442f,0.044836f,0.035768f,
-0.052604f,-0.152719f,-0.250460f,-0.287595f,-0.279726f,-0.187828f,-0.079578f,-0.010026f,-0.023873f,0.063970f,0.119565f,0.046957f,0.064736f,0.051162f,0.033536f,0.183078f,0.188720f,0.019485f,0.029564f,-0.060041f,
-0.070452f,-0.086892f,-0.066193f,-0.049654f,-0.017886f,-0.020159f,-0.019708f,-0.028111f,-0.063460f,-0.089080f,-0.111946f,-0.130587f,-0.140128f,-0.133756f,-0.146059f,-0.155361f,-0.163494f,-0.166637f,-0.164702f,-0.153289f,
-0.128102f,-0.106037f,-0.088492f,-0.090228f,-0.109643f,-0.106041f,-0.134608f,-0.133556f,-0.147742f,-0.139273f,-0.116321f,-0.114094f,-0.107066f,-0.078725f,-0.093158f,-0.065912f,-0.033974f,0.004501f,0.017601f,0.030873f,
0.034746f,0.027269f,0.017181f,-0.002850f,-0.027159f,-0.028991f,-0.027381f,-0.013197f,-0.030215f,-0.022763f,-0.040132f,-0.031998f,-0.042566f,-0.000510f,0.013042f,0.048364f,0.074792f,0.059041f,0.044785f,0.013897f,
0.003481f,-0.017332f,-0.018920f,-0.037772f,-0.016677f,-0.023031f,-0.020557f,-0.045098f,-0.042265f,-0.056843f,-0.047348f,-0.029956f,-0.022769f,-0.004981f,0.022419f,-0.002625f,-0.043335f,-0.055303f,-0.075496f,-0.115268f,
-0.102495f,-0.122345f,-0.114816f,-0.124115f,-0.108436f,-0.116640f,-0.093063f,-0.118706f,-0.072570f,-0.038008f,-0.018215f,0.008356f,0.038836f,0.020328f,0.004471f,0.014379f,-0.017777f,0.000168f,-0.002760f,0.015549f,
0.013968f,0.030891f,0.029259f,0.029441f,0.030482f,0.057538f,0.075281f,0.088535f,0.107543f,0.123939f,0.117637f,0.124656f,0.095709f,0.072247f,0.045002f,0.020717f,-0.145025f,0.254868f,0.138201f,0.192559f,
0.194612f,0.266259f,0.261507f,0.154336f,0.173295f,0.149169f,0.297000f,0.377841f,0.357024f,0.314731f,0.278620f,0.248362f,0.279463f,0.294856f,0.303923f,0.308103f,0.288282f,0.257741f,0.234276f,0.224906f,
0.234680f,0.239863f,0.263186f,0.268612f,0.249115f,0.212430f,0.166283f,0.148887f,0.129757f,0.165866f,0.208100f,0.238002f,0.254354f,0.257674f,0.262307f,0.243176f,0.259047f,0.309519f,0.341621f,0.375662f,
0.401016f,0.403814f,0.353872f,0.340565f,0.309405f,0.303734f,0.339082f,0.361489f,0.386557f,0.382250f,0.364136f,0.326052f,0.314501f,0.300298f,0.311369f,0.329697f,0.344671f,0.324788f,0.275252f,0.213457f,
0.148088f,0.103103f,0.069340f,0.074233f,0.063949f,0.075810f,0.053961f,0.032468f,-0.009089f,-0.057718f,-0.078365f,-0.068885f,-0.068640f,-0.046716f,-0.057790f,-0.090234f,-0.121022f,-0.186297f,-0.264599f,-0.301423f,
-0.305824f,-0.306621f,-0.297389f,-0.263264f,-0.270002f,-0.289043f,-0.292882f,-0.307588f,-0.275065f,-0.229064f,-0.176083f,-0.130189f,-0.115867f,-0.095579f,-0.117635f,-0.141754f,-0.178898f,-0.169472f,-0.137524f,-0.091985f,
-0.051658f,-0.013327f,-0.006177f,-0.011519f,-0.030345f,-0.028782f,-0.020184f,0.026572f,0.063138f,0.094378f,0.098872f,0.071911f,0.040183f,0.006017f,-0.060960f,-0.053505f,-0.071462f,-0.064484f,-0.032565f,-0.034509f,
-0.053057f,-0.064879f,-0.089671f,-0.131951f,-0.074344f,0.294981f,0.178201f,0.140325f,0.128611f,0.208204f,0.180251f,0.125598f,0.059824f,0.061529f,0.090520f,0.080978f,0.102026f,0.123421f,0.154664f,0.124305f,
0.075029f,0.026733f,-0.040253f,-0.059284f,-0.049065f,-0.015044f,0.030957f,0.089870f,0.107014f,0.144454f,0.168395f,0.216984f,0.243655f,0.305742f,0.345101f,0.373736f,0.366236f,0.338500f,0.279834f,0.214308f,
0.170168f,0.169926f,0.165920f,0.178850f,0.208626f,0.178936f,0.153228f,0.120082f,0.081811f,0.062460f,0.037910f,0.030822f,0.005743f,-0.044307f,-0.117844f,-0.207566f,-0.284640f,-0.325551f,-0.344934f,-0.338199f,
-0.327998f,-0.306161f,-0.318186f,-0.303479f,-0.296541f,-0.291417f,-0.264622f,-0.208658f,-0.168595f,-0.151155f,-0.136782f,-0.163456f,-0.175544f,-0.205212f,-0.231195f,-0.204913f,-0.154574f,-0.086944f,-0.047213f,0.001225f,
0.004488f,0.042220f,0.078818f,0.085964f,0.161263f,0.219001f,0.295704f,0.321062f,0.320735f,0.307517f,0.264480f,0.199705f,0.192594f,0.185214f,0.224273f,0.251187f,0.272247f,0.243905f,0.241979f,0.232301f,
0.214652f,0.234653f,0.234185f,0.232104f,0.218314f,0.208479f,0.160460f,0.051205f,-0.043977f,-0.116613f,-0.174306f,-0.187184f,-0.195150f,-0.180029f,-0.220158f,-0.257148f,-0.285340f,-0.317501f,-0.345204f,-0.354570f,
-0.341366f,-0.334087f,-0.334671f,-0.346273f,-0.371846f,-0.442859f,-0.482900f,-0.525579f,-0.539513f,-0.515188f,-0.458250f,-0.399417f,0.364908f,-0.287712f,-0.107659f,-0.051298f,0.258691f,0.096271f,0.053090f,-0.135041f,
-0.260126f,-0.211881f,-0.270300f,-0.255700f,-0.216773f,-0.192369f,-0.220648f,-0.292070f,-0.373838f,-0.414983f,-0.386006f,-0.269696f,-0.133145f,-0.065702f,-0.085583f,-0.188017f,-0.291717f,-0.396374f,-0.457035f,-0.449689f,
-0.393936f,-0.322367f,-0.288921f,-0.300292f,-0.358667f,-0.406251f,-0.428049f,-0.358118f,-0.228583f,-0.095323f,-0.025045f,-0.046470f,-0.119258f,-0.246810f,-0.339387f,-0.379229f,-0.377990f,-0.321620f,-0.215758f,-0.168543f,
-0.196822f,-0.254049f,-0.320389f,-0.323201f,-0.262429f,-0.147346f,-0.022665f,0.044106f,0.016819f,-0.066216f,-0.198697f,-0.308527f,-0.363236f,-0.362663f,-0.298207f,-0.219951f,-0.177856f,-0.182743f,-0.218593f,-0.243914f,
-0.254137f,-0.210780f,-0.084176f,0.044108f,0.133263f,0.149771f,0.091783f,-0.008282f,-0.116866f,-0.184384f,-0.210555f,-0.187559f,-0.134952f,-0.070567f,-0.052754f,-0.100261f,-0.155066f,-0.227955f,-0.242835f,-0.175952f,
-0.054333f,0.060142f,0.078552f,0.023712f,-0.105168f,-0.217972f,-0.338125f,-0.389018f,-0.396008f,-0.351537f,-0.268562f,-0.193049f,-0.200969f,-0.256041f,-0.331898f,-0.370775f,-0.318749f,-0.203322f,-0.063487f,0.047296f,
0.069090f,0.002804f,-0.122903f,-0.228773f,-0.331604f,-0.360241f,-0.340810f,-0.278420f,-0.191061f,-0.144542f,-0.149951f,-0.199166f,-0.273110f,-0.326194f,-0.318174f,-0.235227f,-0.110265f,-0.007737f,0.023675f,-0.033142f,
-0.050578f,0.178034f,0.215273f,0.237921f,0.374344f,0.264267f,0.156769f,0.190655f,0.174414f,0.150494f,0.102106f,-0.095475f,-0.077401f,-0.078255f,-0.065222f,-0.061483f,-0.067265f,-0.080327f,-0.091371f,-0.100806f,
-0.084059f,-0.069079f,-0.067629f,-0.085462f,-0.110285f,-0.131259f,-0.159427f,-0.177953f,-0.159461f,-0.144132f,-0.096509f,-0.075707f,-0.063261f,-0.063096f,-0.080240f,-0.084580f,-0.081033f,-0.055902f,-0.060334f,-0.044440f,
-0.063860f,-0.085735f,-0.137323f,-0.178763f,-0.197210f,-0.183246f,-0.154108f,-0.135842f,-0.118000f,-0.099260f,-0.124015f,-0.144149f,-0.122483f,-0.140454f,-0.109373f,-0.094514f,-0.105831f,-0.112150f,-0.140338f,-0.182839f,
-0.210044f,-0.214830f,-0.186422f,-0.154883f,-0.115841f,-0.085887f,-0.051822f,-0.038170f,-0.018593f,-0.019903f,0.005409f,0.026071f,0.052960f,0.070813f,0.077688f,0.079613f,0.055889f,0.037126f,0.010282f,0.022340f,
0.033420f,0.071902f,0.090434f,0.102004f,0.140374f,0.118877f,0.116834f,0.097791f,0.082890f,0.083389f,0.083658f,0.083953f,0.040464f,-0.024642f,-0.067760f,-0.118987f,-0.137631f,-0.149301f,-0.116934f,-0.092336f,
-0.065647f,-0.052380f,-0.054702f,-0.050066f,-0.078643f,-0.091664f,-0.094683f,-0.063423f,-0.067556f,-0.079762f,-0.093453f,-0.140978f,-0.177804f,-0.216562f,-0.210295f,-0.198689f,-0.183237f,-0.147428f,-0.116274f,-0.117875f,
-0.099510f,-0.111131f,-0.128956f,-0.113446f,-0.118562f,-0.098078f,-0.077084f,-0.048937f,0.001134f,0.082224f,-0.005432f,0.093809f,0.060358f,0.087260f,0.112675f,0.059084f,-0.142652f,0.140453f,0.165434f,0.060112f,
0.065370f,-0.013985f,-0.068511f,-0.123803f,-0.133667f,-0.145231f,-0.131149f,-0.119001f,-0.122527f,-0.135331f,-0.162147f,-0.201111f,-0.233912f,-0.250327f,-0.236466f,-0.215393f,-0.213201f,-0.235269f,-0.257379f,-0.304108f,
-0.325616f,-0.337873f,-0.284267f,-0.244474f,-0.198895f,-0.140384f,-0.119824f,-0.113577f,-0.101398f,-0.109425f,-0.068115f,-0.028605f,0.024610f,0.060498f,0.073561f,0.055699f,-0.004466f,0.002977f,-0.010835f,0.049387f,
0.074787f,0.101874f,0.111897f,0.106218f,0.073758f,0.049332f,0.035239f,0.034070f,0.037912f,0.058477f,0.057204f,0.025966f,-0.014305f,-0.087109f,-0.121075f,-0.136977f,-0.112134f,-0.094572f,-0.065116f,-0.064569f,
-0.058304f,-0.067570f,-0.101556f,-0.124951f,-0.131672f,-0.132812f,-0.110959f,-0.094750f,-0.095950f,-0.135873f,-0.213213f,-0.272177f,-0.317874f,-0.330678f,-0.291252f,-0.275026f,-0.268460f,-0.249480f,-0.264834f,-0.277179f,
-0.307005f,-0.309890f,-0.286056f,-0.224620f,-0.204722f,-0.160380f,-0.135065f,-0.138048f,-0.168958f,-0.202496f,-0.181052f,-0.148102f,-0.079461f,-0.022415f,0.035977f,0.061673f,0.072255f,0.058924f,0.036743f,0.055624f,
0.082868f,0.117045f,0.136510f,0.175345f,0.174122f,0.128547f,0.052325f,0.015627f,-0.010242f,-0.007528f,0.001514f,0.019528f,0.065310f,0.060798f,0.055808f,0.019409f,0.091866f,-0.108674f,-0.057296f,-0.097587f,
-0.049045f,-0.140242f,0.008388f,0.232759f,-0.034529f,-0.112236f,-0.155889f,-0.028033f,0.029797f,0.077082f,0.107891f,0.107891f,0.101734f,0.088745f,0.067579f,0.068076f,0.093608f,0.112478f,0.150193f,0.182929f,
0.174098f,0.172800f,0.139709f,0.118809f,0.119131f,0.143494f,0.153290f,0.153819f,0.129945f,0.055199f,0.014252f,-0.041133f,-0.069732f,-0.076523f,-0.059992f,-0.038527f,-0.036268f,-0.042430f,-0.064101f,-0.092505f,
-0.107138f,-0.100177f,-0.052351f,-0.038058f,-0.022057f,-0.032761f,-0.071691f,-0.102626f,-0.107488f,-0.118863f,-0.080723f,-0.034736f,0.006548f,0.042549f,0.039401f,0.049035f,0.038497f,0.056865f,0.079421f,0.118711f,
0.152977f,0.153989f,0.167211f,0.109681f,0.071785f,0.037231f,0.002797f,-0.009451f,0.008906f,0.016439f,0.044076f,0.031577f,-0.004018f,-0.037748f,-0.051088f,-0.047464f,-0.028785f,-0.005586f,0.028250f,0.014640f,
-0.021678f,-0.058423f,-0.104396f,-0.141931f,-0.136078f,-0.131510f,-0.089852f,-0.071137f,-0.062882f,-0.071680f,-0.090470f,-0.132288f,-0.156033f,-0.137896f,-0.121806f,-0.095881f,-0.099099f,-0.119850f,-0.179218f,-0.209754f,
-0.241708f,-0.293935f,-0.274231f,-0.254734f,-0.195555f,-0.151656f,-0.146385f,-0.161129f,-0.168405f,-0.177741f,-0.162295f,-0.126789f,-0.078277f,-0.050774f,-0.006119f,-0.005504f,-0.029295f,-0.045739f,-0.057657f,-0.078471f,
-0.053461f,-0.015093f,0.021212f,0.055707f,-0.011359f,-0.020188f,-0.285765f,-0.146238f,-0.260501f,-0.228823f,-0.102964f,-0.078276f,-0.038815f,-0.095665f,-0.121290f,-0.085138f,-0.021419f,0.039485f,0.097906f,0.115875f,
0.123936f,0.117916f,0.103253f,0.083748f,0.077657f,0.116204f,0.144895f,0.155628f,0.174242f,0.176557f,0.159813f,0.148573f,0.191059f,0.208697f,0.236556f,0.277498f,0.262007f,0.251312f,0.196648f,0.150756f,
0.123126f,0.090180f,0.063779f,0.067864f,0.068811f,0.075962f,0.036368f,0.029139f,-0.007639f,0.006327f,0.011679f,0.036314f,0.055982f,0.042840f,0.014416f,-0.035214f,-0.064386f,-0.085905f,-0.093144f,-0.081276f,
-0.065606f,-0.052757f,-0.032023f,-0.049706f,-0.055603f,-0.057020f,-0.048191f,-0.002309f,0.016987f,0.033752f,0.050867f,0.010301f,-0.043416f,-0.076852f,-0.121416f,-0.152699f,-0.158042f,-0.137426f,-0.136495f,-0.144809f,
-0.140087f,-0.159867f,-0.168964f,-0.147598f,-0.120312f,-0.080618f,-0.023199f,0.002229f,0.011359f,-0.001033f,-0.018975f,-0.045975f,-0.052047f,-0.050096f,0.001679f,0.042286f,0.071373f,0.081245f,0.070633f,0.040021f,
0.052893f,0.077973f,0.079510f,0.101396f,0.119984f,0.111204f,0.092493f,0.020846f,-0.012893f,-0.072807f,-0.097966f,-0.106978f,-0.096380f,-0.102665f,-0.072508f,-0.104955f,-0.122441f,-0.138003f,-0.134966f,-0.112755f,
-0.089204f,-0.057438f,-0.029374f,-0.017715f,-0.032162f,-0.041790f,-0.085911f,-0.105874f,-0.124580f,-0.103646f,-0.084990f,-0.070943f
};
std::vector<float> input2_data2={
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,
-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f,-10000.000000f
};
std::vector<float> exp_output_data={
-10006.040039f,-9997.950195f,-9996.833008f,-9997.969727f,-9998.177734f,-9998.441406f,-9998.433594f,-9997.742188f,-9996.877930f,-9999.922852f,-9999.191406f,-9997.909180f,-9997.899414f,-9997.839844f,-9997.830078f,-9997.709961f,-9997.614258f,-9997.663086f,-9997.562500f,-9997.598633f,
-9997.566406f,-9997.462891f,-9997.436523f,-9997.403320f,-9997.467773f,-9997.471680f,-9997.502930f,-9997.715820f,-9997.725586f,-9997.735352f,-9997.804688f,-9997.678711f,-9997.702148f,-9997.674805f,-9997.732422f,-9997.757812f,-9997.790039f,-9997.840820f,-9997.942383f,-9997.837891f,
-9997.904297f,-9997.801758f,-9997.984375f,-9998.050781f,-9998.152344f,-9998.279297f,-9998.230469f,-9998.237305f,-9998.068359f,-9998.027344f,-9998.008789f,-9997.904297f,-9997.924805f,-9997.879883f,-9997.848633f,-9997.835938f,-9997.562500f,-9997.671875f,-9997.622070f,-9997.565430f,
-9997.641602f,-9997.627930f,-9997.636719f,-9997.639648f,-9997.593750f,-9997.428711f,-9997.329102f,-9997.177734f,-9997.096680f,-9997.104492f,-9997.101562f,-9997.021484f,-9996.923828f,-9996.825195f,-9996.849609f,-9996.768555f,-9996.729492f,-9996.677734f,-9996.717773f,-9996.703125f,
-9996.539062f,-9996.470703f,-9996.445312f,-9996.277344f,-9996.085938f,-9995.993164f,-9995.864258f,-9995.803711f,-9995.899414f,-9995.910156f,-9995.754883f,-9995.793945f,-9995.834961f,-9996.019531f,-9995.963867f,-9996.114258f,-9996.287109f,-9996.256836f,-9996.362305f,-9996.501953f,
-9996.333984f,-9996.283203f,-9996.194336f,-9996.194336f,-9996.254883f,-9996.223633f,-9996.413086f,-9996.271484f,-9996.197266f,-9996.183594f,-9996.297852f,-9996.262695f,-9996.253906f,-9996.158203f,-9996.191406f,-9996.382812f,-9996.366211f,-9996.363281f,-9996.162109f,-9996.076172f,
-9995.968750f,-9995.843750f,-9995.906250f,-9995.691406f,-9995.849609f,-9995.720703f,-9995.708984f,-9995.522461f,-9996.310547f,-9995.840820f,-9993.902344f,-9995.285156f,-9997.321289f,-9997.780273f,-9998.635742f,-9996.962891f,-9997.916016f,-9997.593750f,-9996.436523f,-9999.296875f,
-9999.742188f,-10000.158203f,-10000.378906f,-10000.405273f,-10000.524414f,-10000.291992f,-10000.632812f,-10001.373047f,-10002.452148f,-10003.439453f,-10004.164062f,-10004.218750f,-10003.785156f,-10002.970703f,-10002.050781f,-10001.293945f,-10001.110352f,-10001.030273f,-10000.876953f,-10000.959961f,
-10000.509766f,-10000.198242f,-9999.907227f,-10000.057617f,-10000.769531f,-10001.530273f,-10002.420898f,-10003.145508f,-10003.006836f,-10002.697266f,-10001.885742f,-10001.166016f,-10001.000977f,-10000.789062f,-10001.208984f,-10001.394531f,-10001.631836f,-10001.650391f,-10001.331055f,-10001.500977f,
-10001.652344f,-10002.582031f,-10003.652344f,-10004.882812f,-10005.752930f,-10005.657227f,-10005.277344f,-10004.541992f,-10003.943359f,-10003.537109f,-10003.354492f,-10003.205078f,-10003.060547f,-10002.868164f,-10002.599609f,-10002.327148f,-10001.903320f,-10001.728516f,-10002.194336f,-10003.093750f,
-10003.987305f,-10004.327148f,-10004.451172f,-10003.921875f,-10003.164062f,-10002.339844f,-10001.774414f,-10001.618164f,-10001.538086f,-10001.727539f,-10001.773438f,-10001.862305f,-10001.916016f,-10001.659180f,-10001.614258f,-10002.255859f,-10003.314453f,-10004.585938f,-10005.811523f,-10006.731445f,
-10006.702148f,-10006.414062f,-10005.886719f,-10005.191406f,-10004.865234f,-10004.758789f,-10004.777344f,-10005.018555f,-10005.212891f,-10004.907227f,-10004.740234f,-10004.506836f,-10004.473633f,-10005.177734f,-10006.108398f,-10007.250977f,-10008.462891f,-10008.633789f,-10008.375000f,-10007.920898f,
-10007.196289f,-10006.857422f,-10006.399414f,-10006.405273f,-10006.393555f,-10006.966797f,-10007.179688f,-10007.030273f,-10007.065430f,-10007.105469f,-10006.992188f,-10007.419922f,-10008.016602f,-10009.031250f,-10009.740234f,-10010.125977f,-9994.264648f,-9995.075195f,-9996.117188f,-9995.026367f,
-9995.409180f,-9996.142578f,-9997.780273f,-9998.573242f,-9998.341797f,-9997.520508f,-9997.151367f,-10001.456055f,-10001.830078f,-10002.395508f,-10002.559570f,-10002.617188f,-10002.668945f,-10002.437500f,-10002.411133f,-10002.717773f,-10003.338867f,-10004.419922f,-10005.657227f,-10006.368164f,
-10006.547852f,-10006.177734f,-10005.615234f,-10004.816406f,-10004.426758f,-10004.363281f,-10004.154297f,-10004.092773f,-10003.689453f,-10003.309570f,-10002.769531f,-10002.537109f,-10002.689453f,-10003.143555f,-10003.907227f,-10005.110352f,-10005.433594f,-10005.702148f,-10005.083984f,-10004.495117f,
-10004.118164f,-10003.608398f,-10003.801758f,-10003.863281f,-10004.012695f,-10003.833984f,-10003.544922f,-10003.316406f,-10003.009766f,-10003.595703f,-10004.302734f,-10005.502930f,-10006.992188f,-10007.349609f,-10007.609375f,-10007.191406f,-10006.807617f,-10006.405273f,-10005.975586f,-10005.881836f,
-10005.597656f,-10005.375977f,-10005.120117f,-10004.670898f,-10004.065430f,-10003.644531f,-10003.644531f,-10004.201172f,-10004.923828f,-10005.800781f,-10006.304688f,-10006.395508f,-10006.031250f,-10005.398438f,-10004.768555f,-10004.613281f,-10004.386719f,-10004.621094f,-10004.649414f,-10004.824219f,
-10004.667969f,-10004.505859f,-10004.314453f,-10004.509766f,-10005.034180f,-10006.128906f,-10007.745117f,-10008.891602f,-10009.615234f,-10009.778320f,-10009.730469f,-10009.068359f,-10008.567383f,-10008.450195f,-10008.195312f,-10008.225586f,-10008.324219f,-10008.029297f,-10007.765625f,-10007.084961f,
-10006.943359f,-10006.899414f,-10007.472656f,-10008.168945f,-10009.725586f,-10010.446289f,-10010.785156f,-10010.675781f,-10010.166992f,-10009.840820f,-10009.254883f,-10008.960938f,-10008.825195f,-10009.051758f,-10009.457031f,-10009.253906f,-10009.260742f,-10009.127930f,-10008.833984f,-10008.970703f,
-10009.271484f,-10010.173828f,-10010.969727f,-10011.727539f,-9994.912109f,-9996.267578f,-9994.809570f,-9995.047852f,-9993.342773f,-9995.392578f,-9996.916992f,-9999.019531f,-9999.246094f,-9997.902344f,-9996.456055f,-10000.312500f,-10000.628906f,-10001.289062f,-10001.977539f,-10002.506836f,
-10002.752930f,-10002.379883f,-10001.898438f,-10001.410156f,-10001.424805f,-10002.285156f,-10003.453125f,-10004.495117f,-10005.138672f,-10005.198242f,-10004.791016f,-10004.083008f,-10003.661133f,-10003.552734f,-10003.496094f,-10004.050781f,-10004.057617f,-10003.647461f,-10003.002930f,-10002.044922f,
-10001.539062f,-10001.331055f,-10001.922852f,-10003.017578f,-10003.811523f,-10004.575195f,-10004.398438f,-10004.012695f,-10003.555664f,-10002.970703f,-10003.004883f,-10003.208008f,-10003.939453f,-10004.247070f,-10003.873047f,-10003.400391f,-10002.602539f,-10002.231445f,-10002.489258f,-10003.458984f,
-10004.914062f,-10005.696289f,-10006.360352f,-10006.401367f,-10006.267578f,-10005.819336f,-10005.326172f,-10005.117188f,-10005.067383f,-10005.272461f,-10005.387695f,-10005.057617f,-10004.289062f,-10003.278320f,-10002.461914f,-10002.451172f,-10002.888672f,-10003.589844f,-10004.306641f,-10004.845703f,
-10004.877930f,-10004.584961f,-10004.048828f,-10003.761719f,-10003.364258f,-10003.443359f,-10003.901367f,-10004.447266f,-10004.718750f,-10004.369141f,-10003.882812f,-10003.230469f,-10003.155273f,-10003.711914f,-10004.931641f,-10006.457031f,-10007.579102f,-10008.115234f,-10008.263672f,-10007.791992f,
-10007.333008f,-10006.969727f,-10006.871094f,-10006.804688f,-10007.440430f,-10007.582031f,-10007.438477f,-10006.852539f,-10005.930664f,-10005.191406f,-10004.946289f,-10005.505859f,-10006.670898f,-10007.698242f,-10008.585938f,-10008.869141f,-10008.685547f,-10008.520508f,-10007.979492f,-10007.496094f,
-10007.208008f,-10007.699219f,-10008.332031f,-10008.713867f,-10008.864258f,-10008.567383f,-10007.847656f,-10007.137695f,-10006.815430f,-10007.258789f,-10007.853516f,-10009.098633f,-9996.037109f,-9996.248047f,-9994.654297f,-9993.548828f,-9994.947266f,-9993.321289f,-9994.778320f,-9996.588867f,
-9996.405273f,-9997.703125f,-9996.935547f,-10000.617188f,-10000.625000f,-10000.869141f,-10001.437500f,-10002.106445f,-10002.658203f,-10002.780273f,-10002.420898f,-10001.737305f,-10000.958984f,-10000.989258f,-10001.622070f,-10002.743164f,-10003.846680f,-10004.694336f,-10005.024414f,-10004.553711f,
-10004.126953f,-10003.679688f,-10003.253906f,-10003.613281f,-10003.785156f,-10003.738281f,-10003.506836f,-10002.679688f,-10001.680664f,-10000.756836f,-10000.567383f,-10001.286133f,-10002.127930f,-10003.346680f,-10004.027344f,-10004.336914f,-10004.257812f,-10003.765625f,-10003.457031f,-10003.230469f,
-10003.774414f,-10004.365234f,-10004.460938f,-10004.509766f,-10003.693359f,-10003.048828f,-10002.517578f,-10002.840820f,-10003.835938f,-10004.779297f,-10005.992188f,-10006.949219f,-10007.416016f,-10007.476562f,-10007.005859f,-10006.555664f,-10006.268555f,-10006.362305f,-10006.714844f,-10006.478516f,
-10006.280273f,-10005.405273f,-10004.333984f,-10003.792969f,-10003.403320f,-10003.412109f,-10004.254883f,-10005.037109f,-10005.701172f,-10006.066406f,-10006.042969f,-10005.821289f,-10005.147461f,-10004.875000f,-10004.948242f,-10005.375977f,-10005.535156f,-10005.776367f,-10005.632812f,-10004.812500f,
-10004.166016f,-10003.660156f,-10004.089844f,-10005.301758f,-10006.428711f,-10007.445312f,-10008.298828f,-10008.447266f,-10008.249023f,-10007.754883f,-10007.305664f,-10006.684570f,-10006.961914f,-10007.456055f,-10007.443359f,-10007.521484f,-10006.818359f,-10005.648438f,-10004.916992f,-10004.538086f,
-10005.231445f,-10006.008789f,-10007.299805f,-10008.379883f,-10009.009766f,-10009.475586f,-10009.285156f,-10008.875977f,-10008.137695f,-10008.215820f,-10008.638672f,-10008.935547f,-10009.468750f,-10009.637695f,-10009.210938f,-10008.514648f,-10007.724609f,-10007.236328f,-10007.157227f,-10008.036133f,
-9995.282227f,-9998.327148f,-9996.766602f,-9995.916016f,-9993.127930f,-9994.664062f,-9994.622070f,-9996.045898f,-9996.714844f,-9997.083984f,-9996.142578f,-10000.223633f,-10000.324219f,-10000.532227f,-10000.936523f,-10001.490234f,-10002.120117f,-10002.662109f,-10002.701172f,-10002.189453f,
-10001.214844f,-10000.611328f,-10000.567383f,-10001.150391f,-10002.113281f,-10003.187500f,-10004.079102f,-10004.407227f,-10004.588867f,-10004.332031f,-10003.941406f,-10004.157227f,-10004.273438f,-10004.435547f,-10004.639648f,-10004.121094f,-10003.166016f,-10001.860352f,-10001.181641f,-10000.965820f,
-10001.228516f,-10002.329102f,-10003.257812f,-10004.063477f,-10004.516602f,-10004.522461f,-10004.199219f,-10003.919922f,-10004.148438f,-10004.478516f,-10004.808594f,-10005.100586f,-10004.689453f,-10003.921875f,-10002.980469f,-10002.484375f,-10002.602539f,-10002.972656f,-10003.987305f,-10005.136719f,
-10006.080078f,-10006.697266f,-10006.740234f,-10006.370117f,-10006.044922f,-10006.057617f,-10006.261719f,-10006.167969f,-10006.193359f,-10005.667969f,-10004.662109f,-10003.867188f,-10002.804688f,-10002.200195f,-10002.293945f,-10002.725586f,-10003.474609f,-10004.208008f,-10004.713867f,-10004.988281f,
-10004.741211f,-10004.514648f,-10004.406250f,-10004.742188f,-10004.839844f,-10005.281250f,-10005.632812f,-10005.189453f,-10004.461914f,-10003.734375f,-10003.356445f,-10003.721680f,-10004.500977f,-10005.381836f,-10006.572266f,-10007.098633f,-10007.667969f,-10007.706055f,-10007.314453f,-10006.646484f,
-10006.649414f,-10006.919922f,-10007.201172f,-10007.482422f,-10007.024414f,-10006.189453f,-10005.072266f,-10004.246094f,-10003.855469f,-10004.007812f,-10004.838867f,-10005.760742f,-10006.742188f,-10007.690430f,-10008.098633f,-10008.031250f,-10007.624023f,-10007.401367f,-10007.494141f,-10007.590820f,
-10008.070312f,-10008.371094f,-10008.359375f,-10007.806641f,-10006.923828f,-10005.940430f,-10005.177734f,-10005.409180f,-9995.470703f,-9999.483398f,-9996.970703f,-9997.009766f,-9995.702148f,-9994.668945f,-9995.197266f,-9993.499023f,-9994.218750f,-9995.841797f,-9995.499023f,-9999.638672f,
-10000.258789f,-10000.524414f,-10000.617188f,-10000.929688f,-10001.469727f,-10002.052734f,-10002.458984f,-10002.454102f,-10001.729492f,-10000.882812f,-10000.231445f,-10000.168945f,-10000.791016f,-10001.630859f,-10002.777344f,-10003.690430f,-10004.423828f,-10004.516602f,-10004.166992f,-10004.219727f,
-10003.941406f,-10004.124023f,-10004.500977f,-10004.357422f,-10003.766602f,-10002.599609f,-10001.573242f,-10000.794922f,-10000.493164f,-10001.047852f,-10001.807617f,-10002.912109f,-10003.907227f,-10004.424805f,-10004.589844f,-10004.352539f,-10004.068359f,-10004.229492f,-10004.451172f,-10005.111328f,
-10005.072266f,-10004.613281f,-10003.722656f,-10002.961914f,-10002.435547f,-10002.136719f,-10002.718750f,-10003.735352f,-10004.925781f,-10006.139648f,-10006.776367f,-10006.798828f,-10006.659180f,-10006.421875f,-10006.586914f,-10006.400391f,-10006.617188f,-10006.420898f,-10005.750000f,-10005.090820f,
-10003.873047f,-10002.577148f,-10002.207031f,-10001.920898f,-10002.383789f,-10003.151367f,-10004.116211f,-10004.848633f,-10005.011719f,-10005.104492f,-10004.773438f,-10004.703125f,-10004.617188f,-10005.002930f,-10005.554688f,-10005.543945f,-10004.992188f,-10004.207031f,-10003.388672f,-10003.196289f,
-10003.171875f,-10003.682617f,-10004.605469f,-10005.389648f,-10006.318359f,-10006.934570f,-10006.829102f,-10006.322266f,-10006.071289f,-10005.929688f,-10006.000000f,-10006.397461f,-10006.363281f,-10005.902344f,-10005.080078f,-10004.173828f,-10003.162109f,-10002.673828f,-10002.843750f,-10003.419922f,
-10004.416016f,-10005.871094f,-10006.813477f,-10007.364258f,-10007.464844f,-10007.281250f,-10007.000000f,-10006.758789f,-10006.961914f,-10007.322266f,-10007.540039f,-10007.474609f,-10006.854492f,-10005.976562f,-10004.809570f,-10004.280273f,-9995.210938f,-9998.807617f,-9997.978516f,-9997.727539f,
-9996.463867f,-9995.875000f,-9995.198242f,-9997.213867f,-9994.106445f,-9994.833984f,-9995.283203f,-9999.493164f,-10000.474609f,-10001.353516f,-10001.541016f,-10001.685547f,-10001.878906f,-10002.374023f,-10002.925781f,-10003.496094f,-10003.144531f,-10002.566406f,-10001.491211f,-10000.896484f,
-10000.622070f,-10001.113281f,-10002.161133f,-10003.161133f,-10004.351562f,-10005.100586f,-10005.329102f,-10005.483398f,-10005.062500f,-10005.104492f,-10005.254883f,-10005.509766f,-10005.499023f,-10004.540039f,-10003.559570f,-10002.423828f,-10001.403320f,-10001.150391f,-10001.354492f,-10002.330078f,
-10003.258789f,-10004.317383f,-10004.975586f,-10005.286133f,-10005.083984f,-10004.917969f,-10004.843750f,-10005.260742f,-10005.448242f,-10005.580078f,-10005.157227f,-10004.285156f,-10003.478516f,-10002.416992f,-10002.171875f,-10002.658203f,-10003.653320f,-10004.874023f,-10005.963867f,-10006.611328f,
-10006.990234f,-10006.985352f,-10007.000977f,-10006.761719f,-10006.800781f,-10006.826172f,-10006.715820f,-10006.470703f,-10005.437500f,-10004.070312f,-10002.989258f,-10002.124023f,-10002.144531f,-10002.593750f,-10003.464844f,-10004.426758f,-10005.196289f,-10005.888672f,-10006.067383f,-10006.192383f,
-10005.731445f,-10006.023438f,-10006.625977f,-10006.966797f,-10006.859375f,-10006.500977f,-10005.789062f,-10004.930664f,-10004.255859f,-10004.019531f,-10004.450195f,-10005.137695f,-10006.186523f,-10007.032227f,-10007.639648f,-10007.656250f,-10007.423828f,-10007.396484f,-10007.021484f,-10007.012695f,
-10007.115234f,-10006.998047f,-10006.860352f,-10005.958008f,-10004.941406f,-10003.895508f,-10003.061523f,-10002.862305f,-10003.689453f,-10004.835938f,-10005.954102f,-10006.952148f,-10007.698242f,-10007.963867f,-10007.826172f,-10007.412109f,-10007.346680f,-10007.263672f,-10007.562500f,-10007.750977f,
-10007.818359f,-10007.100586f,-10005.874023f,-10004.824219f,-9993.662109f,-9998.705078f,-9998.729492f,-9998.718750f,-9997.045898f,-9996.883789f,-9995.586914f,-9995.273438f,-9996.683594f,-9994.428711f,-9994.543945f,-9998.949219f,-10000.374023f,-10001.610352f,-10002.025391f,-10002.094727f,
-10001.908203f,-10002.254883f,-10002.875000f,-10003.642578f,-10003.674805f,-10003.323242f,-10002.388672f,-10001.688477f,-10000.994141f,-10000.886719f,-10001.536133f,-10002.730469f,-10004.274414f,-10005.538086f,-10006.075195f,-10006.398438f,-10005.834961f,-10005.622070f,-10005.617188f,-10005.964844f,
-10006.088867f,-10005.532227f,-10004.736328f,-10003.777344f,-10002.393555f,-10001.738281f,-10001.292969f,-10001.819336f,-10002.883789f,-10004.290039f,-10005.414062f,-10006.167969f,-10006.045898f,-10005.650391f,-10005.340820f,-10005.590820f,-10005.883789f,-10006.219727f,-10006.152344f,-10005.551758f,
-10004.793945f,-10003.501953f,-10002.660156f,-10002.583984f,-10003.196289f,-10004.462891f,-10005.955078f,-10007.033203f,-10007.810547f,-10007.982422f,-10007.805664f,-10007.575195f,-10007.347656f,-10007.365234f,-10007.346680f,-10007.516602f,-10006.826172f,-10005.477539f,-10004.310547f,-10003.026367f,
-10002.446289f,-10002.391602f,-10003.051758f,-10004.225586f,-10005.392578f,-10006.530273f,-10006.972656f,-10007.024414f,-10006.592773f,-10006.616211f,-10006.947266f,-10007.559570f,-10007.703125f,-10007.692383f,-10007.246094f,-10006.401367f,-10005.354492f,-10004.600586f,-10004.402344f,-10004.640625f,
-10005.666992f,-10006.796875f,-10007.875000f,-10008.385742f,-10008.371094f,-10008.081055f,-10007.612305f,-10007.352539f,-10007.391602f,-10007.465820f,-10007.487305f,-10007.107422f,-10006.177734f,-10004.971680f,-10003.745117f,-10002.939453f,-10003.156250f,-10004.017578f,-10005.377930f,-10006.617188f,
-10007.987305f,-10008.643555f,-10008.745117f,-10008.214844f,-10007.842773f,-10007.496094f,-10007.671875f,-10007.960938f,-10008.190430f,-10007.940430f,-10006.975586f,-10005.984375f,-9995.948242f,-9997.749023f,-9998.669922f,-9998.017578f,-9998.098633f,-9997.582031f,-9996.320312f,-9995.251953f,
-9994.908203f,-9994.590820f,-9992.727539f,-9996.813477f,-9997.846680f,-9999.213867f,-10000.008789f,-10000.525391f,-10000.541016f,-10000.783203f,-10001.249023f,-10001.953125f,-10002.418945f,-10002.625000f,-10002.154297f,-10001.442383f,-10000.665039f,-10000.039062f,-9999.905273f,-10000.245117f,
-10001.455078f,-10002.565430f,-10003.548828f,-10004.246094f,-10004.316406f,-10004.197266f,-10004.208008f,-10004.415039f,-10004.641602f,-10004.610352f,-10004.313477f,-10003.699219f,-10002.458008f,-10001.622070f,-10000.568359f,-10000.406250f,-10000.678711f,-10001.522461f,-10002.805664f,-10003.723633f,
-10004.170898f,-10004.295898f,-10004.060547f,-10004.182617f,-10004.331055f,-10004.687500f,-10005.016602f,-10005.037109f,-10004.623047f,-10003.440430f,-10002.376953f,-10001.713867f,-10001.475586f,-10002.005859f,-10002.969727f,-10004.126953f,-10005.041016f,-10005.696289f,-10005.976562f,-10005.996094f,
-10005.903320f,-10005.832031f,-10005.924805f,-10006.218750f,-10006.060547f,-10005.290039f,-10004.201172f,-10002.944336f,-10002.006836f,-10001.350586f,-10001.183594f,-10001.772461f,-10002.566406f,-10003.768555f,-10004.593750f,-10005.158203f,-10005.227539f,-10005.127930f,-10005.369141f,-10005.766602f,
-10006.014648f,-10006.408203f,-10006.487305f,-10006.075195f,-10005.074219f,-10004.032227f,-10003.341797f,-10002.786133f,-10003.175781f,-10003.808594f,-10004.782227f,-10005.584961f,-10006.216797f,-10006.342773f,-10006.210938f,-10005.902344f,-10005.871094f,-10005.824219f,-10006.052734f,-10006.208008f,
-10005.806641f,-10004.839844f,-10003.569336f,-10002.472656f,-10001.917969f,-10001.970703f,-10002.622070f,-10003.530273f,-10004.731445f,-10005.631836f,-10006.462891f,-10006.519531f,-10006.447266f,-10006.253906f,-10006.117188f,-10006.166992f,-10006.460938f,-10006.485352f,-10006.012695f,-10005.353516f,
-9990.874023f,-9997.565430f,-9999.846680f,-9998.739258f,-9999.780273f,-9998.035156f,-9997.177734f,-9996.948242f,-9995.892578f,-9993.664062f,-9992.732422f,-9997.822266f,-9998.925781f,-10000.485352f,-10001.729492f,-10002.674805f,-10003.027344f,-10003.208984f,-10003.779297f,-10004.477539f,
-10005.194336f,-10005.785156f,-10005.787109f,-10005.343750f,-10004.616211f,-10003.727539f,-10003.200195f,-10003.008789f,-10003.940430f,-10005.286133f,-10006.563477f,-10007.817383f,-10008.113281f,-10008.270508f,-10008.297852f,-10008.588867f,-10008.932617f,-10008.934570f,-10008.791016f,-10008.581055f,
-10007.385742f,-10006.610352f,-10004.994141f,-10004.228516f,-10003.901367f,-10004.317383f,-10005.713867f,-10006.987305f,-10007.825195f,-10008.341797f,-10008.261719f,-10008.531250f,-10008.613281f,-10009.039062f,-10009.441406f,-10009.736328f,-10009.976562f,-10008.840820f,-10007.838867f,-10006.776367f,
-10005.880859f,-10005.893555f,-10006.548828f,-10007.891602f,-10009.164062f,-10010.276367f,-10011.086914f,-10011.532227f,-10011.612305f,-10011.520508f,-10011.663086f,-10011.943359f,-10012.036133f,-10011.521484f,-10010.584961f,-10009.250000f,-10008.038086f,-10006.651367f,-10005.665039f,-10005.785156f,
-10006.345703f,-10007.728516f,-10008.766602f,-10009.647461f,-10010.058594f,-10010.067383f,-10010.332031f,-10010.689453f,-10010.660156f,-10011.085938f,-10011.571289f,-10011.411133f,-10010.556641f,-10009.203125f,-10008.155273f,-10006.792969f,-10006.469727f,-10006.652344f,-10007.595703f,-10008.647461f,
-10009.596680f,-10010.100586f,-10010.289062f,-10009.888672f,-10009.966797f,-10009.619141f,-10009.739258f,-10010.094727f,-10010.139648f,-10009.394531f,-10008.104492f,-10006.838867f,-10005.834961f,-10005.200195f,-10005.378906f,-10005.869141f,-10007.348633f,-10008.506836f,-10009.907227f,-10010.367188f,
-10010.688477f,-10010.327148f,-10010.331055f,-10010.291992f,-10010.433594f,-10010.650391f,-10010.294922f,-10010.006836f,-9993.941406f,-9999.565430f,-10001.846680f,-10000.172852f,-10001.333008f,-10000.245117f,-9999.332031f,-9999.655273f,-9998.319336f,-9995.991211f,-9994.750977f,-9993.211914f,
-9993.539062f,-9994.875977f,-9996.487305f,-9998.173828f,-9999.178711f,-9999.617188f,-10000.011719f,-10000.362305f,-10000.738281f,-10001.206055f,-10001.411133f,-10001.307617f,-10000.836914f,-9999.969727f,-9999.045898f,-9998.390625f,-9998.599609f,-9999.624023f,-10001.088867f,-10002.792969f,
-10003.949219f,-10004.880859f,-10005.119141f,-10005.395508f,-10005.448242f,-10005.228516f,-10004.936523f,-10004.984375f,-10004.043945f,-10003.481445f,-10001.779297f,-10000.581055f,-9999.681641f,-9999.476562f,-10000.352539f,-10001.647461f,-10003.068359f,-10004.413086f,-10004.982422f,-10005.322266f,
-10005.264648f,-10005.247070f,-10005.339844f,-10005.578125f,-10005.923828f,-10005.183594f,-10004.387695f,-10003.211914f,-10001.834961f,-10001.296875f,-10001.302734f,-10002.303711f,-10003.492188f,-10005.282227f,-10006.834961f,-10007.943359f,-10008.525391f,-10008.515625f,-10008.489258f,-10008.490234f,
-10008.401367f,-10008.143555f,-10007.669922f,-10006.811523f,-10005.780273f,-10004.240234f,-10002.857422f,-10002.358398f,-10002.453125f,-10003.605469f,-10004.899414f,-10006.543945f,-10007.843750f,-10008.511719f,-10008.982422f,-10009.277344f,-10008.824219f,-10008.980469f,-10009.376953f,-10009.392578f,
-10009.060547f,-10007.849609f,-10006.797852f,-10005.003906f,-10004.166992f,-10003.604492f,-10004.083008f,-10004.930664f,-10006.442383f,-10007.795898f,-10008.839844f,-10008.771484f,-10008.857422f,-10008.128906f,-10007.765625f,-10007.783203f,-10007.931641f,-10007.435547f,-10006.543945f,-10005.402344f,
-10004.125000f,-10002.900391f,-10002.395508f,-10002.178711f,-10003.227539f,-10004.419922f,-10006.307617f,-10007.478516f,-10008.616211f,-10008.685547f,-10008.740234f,-10008.339844f,-10007.922852f,-10007.644531f,-10007.170898f,-10007.161133f,-9993.759766f,-10000.055664f,-10002.191406f,-10000.653320f,
-10001.591797f,-10000.535156f,-9999.839844f,-10000.654297f,-9999.725586f,-9997.335938f,-9996.046875f,-9993.867188f,-9993.254883f,-9993.666016f,-9994.847656f,-9996.563477f,-9998.083984f,-9998.991211f,-9999.806641f,-10000.164062f,-10000.460938f,-10000.736328f,-10000.977539f,-10001.133789f,
-10001.105469f,-10000.676758f,-9999.816406f,-9998.875000f,-9998.401367f,-9998.634766f,-9999.438477f,-10000.949219f,-10002.362305f,-10003.900391f,-10004.792969f,-10005.471680f,-10005.733398f,-10005.464844f,-10005.107422f,-10005.178711f,-10004.491211f,-10004.333008f,-10003.082031f,-10001.885742f,
-10000.680664f,-9999.811523f,-9999.844727f,-10000.347656f,-10001.588867f,-10003.182617f,-10004.328125f,-10005.301758f,-10005.613281f,-10005.592773f,-10005.474609f,-10005.505859f,-10005.883789f,-10005.446289f,-10005.062500f,-10004.295898f,-10002.925781f,-10002.060547f,-10001.360352f,-10001.541016f,
-10002.064453f,-10003.616211f,-10005.263672f,-10006.945312f,-10008.249023f,-10008.764648f,-10009.049805f,-10009.042969f,-10008.811523f,-10008.599609f,-10008.332031f,-10007.915039f,-10007.356445f,-10006.040039f,-10004.551758f,-10003.535156f,-10002.923828f,-10003.254883f,-10003.922852f,-10005.407227f,
-10007.086914f,-10008.360352f,-10009.393555f,-10010.018555f,-10009.569336f,-10009.437500f,-10009.615234f,-10009.594727f,-10009.607422f,-10008.859375f,-10008.246094f,-10006.543945f,-10005.390625f,-10004.235352f,-10003.871094f,-10003.916992f,-10005.074219f,-10006.574219f,-10008.236328f,-10008.927734f,
-10009.572266f,-10009.069336f,-10008.544922f,-10008.315430f,-10008.323242f,-10007.924805f,-10007.465820f,-10006.861328f,-10005.815430f,-10004.477539f,-10003.530273f,-10002.620117f,-10002.732422f,-10003.161133f,-10004.645508f,-10005.941406f,-10007.689453f,-10008.503906f,-10009.084961f,-10009.028320f,
-10008.384766f,-10007.840820f,-10007.099609f,-10007.080078f,-9993.992188f,-10000.595703f,-10002.890625f,-10001.311523f,-10001.974609f,-10000.799805f,-10000.128906f,-10001.250000f,-10000.728516f,-9998.551758f,-9997.573242f,-9995.236328f,-9993.850586f,-9993.308594f,-9993.649414f,-9994.900391f,
-9996.532227f,-9997.814453f,-9999.093750f,-9999.686523f,-10000.040039f,-10000.136719f,-10000.253906f,-10000.523438f,-10000.843750f,-10000.917969f,-10000.432617f,-9999.577148f,-9998.734375f,-9998.293945f,-9998.293945f,-9999.187500f,-10000.406250f,-10002.189453f,-10003.622070f,-10004.796875f,
-10005.456055f,-10005.300781f,-10004.934570f,-10004.935547f,-10004.373047f,-10004.496094f,-10003.792969f,-10002.900391f,-10001.812500f,-10000.655273f,-9999.959961f,-9999.561523f,-10000.194336f,-10001.545898f,-10002.955078f,-10004.469727f,-10005.312500f,-10005.528320f,-10005.382812f,-10005.203125f,
-10005.442383f,-10005.177734f,-10005.102539f,-10004.844727f,-10003.819336f,-10003.060547f,-10002.045898f,-10001.547852f,-10001.291992f,-10002.247070f,-10003.579102f,-10005.449219f,-10007.322266f,-10008.458008f,-10009.246094f,-10009.482422f,-10009.171875f,-10008.921875f,-10008.746094f,-10008.613281f,
-10008.513672f,-10007.640625f,-10006.450195f,-10005.264648f,-10004.216797f,-10003.758789f,-10003.580078f,-10004.452148f,-10006.020508f,-10007.621094f,-10009.134766f,-10010.225586f,-10010.003906f,-10009.739258f,-10009.616211f,-10009.346680f,-10009.458984f,-10009.075195f,-10008.958008f,-10007.683594f,
-10006.600586f,-10005.227539f,-10004.201172f,-10003.382812f,-10003.791992f,-10004.990234f,-10006.820312f,-10008.106445f,-10009.377930f,-10009.375977f,-10008.983398f,-10008.595703f,-10008.371094f,-10007.937500f,-10007.743164f,-10007.595703f,-10006.994141f,-10005.927734f,-10004.946289f,-10003.708008f,
-10003.029297f,-10002.610352f,-10003.238281f,-10004.172852f,-10006.109375f,-10007.502930f,-10008.625000f,-10009.261719f,-10008.702148f,-10008.103516f,-10007.122070f,-10006.929688f,-9993.887695f,-10000.881836f,-10003.058594f,-10002.018555f,-10002.537109f,-10001.240234f,-10000.389648f,-10001.224609f,
-10001.027344f,-9999.119141f,-9998.639648f,-9996.663086f,-9995.068359f,-9993.827148f,-9993.269531f,-9993.636719f,-9994.911133f,-9996.280273f,-9998.007812f,-9999.172852f,-9999.914062f,-10000.051758f,-9999.960938f,-10000.083984f,-10000.393555f,-10000.796875f,-10000.799805f,-10000.398438f,
-9999.713867f,-9999.012695f,-9998.355469f,-9998.455078f,-9998.988281f,-10000.519531f,-10002.201172f,-10003.905273f,-10005.186523f,-10005.450195f,-10005.204102f,-10005.033203f,-10004.298828f,-10004.338867f,-10003.946289f,-10003.460938f,-10002.804688f,-10001.852539f,-10000.889648f,-9999.772461f,
-9999.490234f,-10000.104492f,-10001.251953f,-10003.054688f,-10004.455078f,-10005.222656f,-10005.401367f,-10005.166016f,-10005.125977f,-10004.693359f,-10004.561523f,-10004.616211f,-10004.054688f,-10003.780273f,-10002.979492f,-10002.221680f,-10001.367188f,-10001.521484f,-10002.145508f,-10003.712891f,
-10005.795898f,-10007.491211f,-10008.968750f,-10009.789062f,-10009.636719f,-10009.281250f,-10008.971680f,-10008.722656f,-10008.773438f,-10008.299805f,-10007.653320f,-10006.780273f,-10005.838867f,-10004.934570f,-10004.020508f,-10004.039062f,-10004.989258f,-10006.500000f,-10008.301758f,-10010.011719f,
-10010.412109f,-10010.410156f,-10010.152344f,-10009.512695f,-10009.332031f,-10008.961914f,-10009.189453f,-10008.431641f,-10007.801758f,-10006.717773f,-10005.471680f,-10004.081055f,-10003.604492f,-10004.034180f,-10005.465820f,-10006.905273f,-10008.692383f,-10009.431641f,-10009.575195f,-10009.312500f,
-10008.869141f,-10008.147461f,-10007.719727f,-10007.680664f,-10007.446289f,-10006.848633f,-10006.225586f,-10005.164062f,-10004.196289f,-10003.102539f,-10002.730469f,-10002.819336f,-10004.422852f,-10005.969727f,-10007.500977f,-10009.050781f,-10009.048828f,-10008.760742f,-10007.684570f,-10007.185547f,
-9993.816406f,-10000.835938f,-10003.134766f,-10002.364258f,-10002.916992f,-10001.767578f,-10000.762695f,-10001.383789f,-10001.343750f,-9999.399414f,-9999.440430f,-9998.125000f,-9996.755859f,-9995.227539f,-9993.887695f,-9993.211914f,-9993.623047f,-9994.539062f,-9996.276367f,-9997.889648f,
-9999.164062f,-9999.651367f,-9999.604492f,-9999.682617f,-9999.896484f,-10000.417969f,-10000.847656f,-10000.994141f,-10000.728516f,-10000.191406f,-9999.257812f,-9998.655273f,-9998.333008f,-9999.138672f,-10000.554688f,-10002.460938f,-10004.287109f,-10005.206055f,-10005.425781f,-10005.387695f,
-10004.622070f,-10004.532227f,-10004.230469f,-10004.074219f,-10003.941406f,-10003.422852f,-10002.577148f,-10001.159180f,-10000.027344f,-9999.646484f,-10000.025391f,-10001.523438f,-10003.121094f,-10004.395508f,-10005.118164f,-10005.146484f,-10005.069336f,-10004.500977f,-10004.215820f,-10004.329102f,
-10004.139648f,-10004.357422f,-10004.018555f,-10003.382812f,-10002.242188f,-10001.646484f,-10001.400391f,-10002.231445f,-10003.950195f,-10005.780273f,-10007.764648f,-10009.318359f,-10009.661133f,-10009.457031f,-10009.155273f,-10008.757812f,-10008.732422f,-10008.536133f,-10008.373047f,-10007.988281f,
-10007.440430f,-10006.465820f,-10005.124023f,-10004.277344f,-10004.278320f,-10005.160156f,-10006.702148f,-10008.702148f,-10009.746094f,-10010.321289f,-10010.308594f,-10009.557617f,-10009.129883f,-10008.632812f,-10008.992188f,-10008.691406f,-10008.585938f,-10008.071289f,-10007.020508f,-10005.521484f,
-10004.375000f,-10003.915039f,-10004.466797f,-10005.458984f,-10007.304688f,-10008.613281f,-10009.461914f,-10009.669922f,-10009.347656f,-10008.563477f,-10007.842773f,-10007.669922f,-10007.627930f,-10007.444336f,-10007.249023f,-10006.606445f,-10005.736328f,-10004.353516f,-10003.128906f,-10002.225586f,
-10002.960938f,-10004.041016f,-10005.475586f,-10007.676758f,-10008.357422f,-10008.771484f,-10007.962891f,-10007.434570f,-9993.561523f,-10000.816406f,-10003.054688f,-10002.463867f,-10003.270508f,-10002.304688f,-10001.180664f,-10001.617188f,-10001.227539f,-9999.310547f,-9999.594727f,-9998.846680f,
-9998.087891f,-9996.842773f,-9995.324219f,-9993.938477f,-9993.424805f,-9993.502930f,-9994.709961f,-9996.364258f,-9998.000977f,-9998.952148f,-9999.151367f,-9999.283203f,-9999.344727f,-9999.717773f,-10000.345703f,-10000.917969f,-10001.195312f,-10001.249023f,-10000.602539f,-9999.808594f,
-9998.892578f,-9998.855469f,-9999.572266f,-10001.168945f,-10003.146484f,-10004.578125f,-10005.327148f,-10005.664062f,-10005.022461f,-10004.822266f,-10004.316406f,-10004.301758f,-10004.528320f,-10004.539062f,-10004.205078f,-10003.100586f,-10001.621094f,-10000.553711f,-9999.976562f,-10000.685547f,
-10001.936523f,-10003.347656f,-10004.532227f,-10004.982422f,-10005.102539f,-10004.492188f,-10004.001953f,-10003.924805f,-10003.855469f,-10004.393555f,-10004.618164f,-10004.505859f,-10003.600586f,-10002.758789f,-10001.994141f,-10001.999023f,-10002.966797f,-10004.390625f,-10006.394531f,-10008.447266f,
-10009.353516f,-10009.495117f,-10009.375000f,-10008.855469f,-10008.594727f,-10008.429688f,-10008.516602f,-10008.601562f,-10008.626953f,-10008.061523f,-10006.889648f,-10005.641602f,-10004.866211f,-10004.863281f,-10005.660156f,-10007.380859f,-10008.686523f,-10009.795898f,-10010.243164f,-10009.699219f,
-10009.148438f,-10008.357422f,-10008.565430f,-10008.416992f,-10008.699219f,-10008.766602f,-10008.301758f,-10007.267578f,-10006.037109f,-10005.082031f,-10004.693359f,-10004.805664f,-10006.145508f,-10007.476562f,-10008.779297f,-10009.551758f,-10009.614258f,-10009.018555f,-10008.042969f,-10007.549805f,
-10007.442383f,-10007.438477f,-10007.558594f,-10007.430664f,-10007.088867f,-10005.966797f,-10004.520508f,-10002.956055f,-10002.737305f,-10002.904297f,-10003.691406f,-10005.916992f,-10007.031250f,-10008.147461f,-10007.865234f,-10007.584961f,-9993.672852f,-10000.644531f,-10002.853516f,-10002.000000f,
-10003.117188f,-10002.704102f,-10001.783203f,-10002.259766f,-10001.876953f,-9999.611328f,-9999.762695f,-9999.325195f,-9998.990234f,-9998.127930f,-9996.805664f,-9995.114258f,-9993.851562f,-9992.985352f,-9993.293945f,-9994.497070f,-9996.208008f,-9997.621094f,-9998.360352f,-9998.854492f,
-9998.968750f,-9999.148438f,-9999.702148f,-10000.339844f,-10000.822266f,-10001.376953f,-10001.192383f,-10000.555664f,-9999.465820f,-9998.769531f,-9998.652344f,-9999.590820f,-10001.312500f,-10003.089844f,-10004.445312f,-10005.482422f,-10005.340820f,-10005.300781f,-10004.613281f,-10004.515625f,
-10004.766602f,-10004.975586f,-10005.048828f,-10004.450195f,-10003.119141f,-10001.770508f,-10000.426758f,-10000.152344f,-10000.649414f,-10001.794922f,-10003.229492f,-10004.206055f,-10004.847656f,-10004.572266f,-10004.111328f,-10003.800781f,-10003.614258f,-10004.061523f,-10004.491211f,-10004.781250f,
-10004.229492f,-10003.510742f,-10002.562500f,-10001.937500f,-10002.077148f,-10002.772461f,-10004.386719f,-10006.663086f,-10008.127930f,-10008.911133f,-10009.339844f,-10009.083008f,-10008.705078f,-10008.482422f,-10008.507812f,-10008.748047f,-10009.081055f,-10008.933594f,-10008.182617f,-10006.979492f,
-10005.791016f,-10004.970703f,-10004.838867f,-10005.828125f,-10007.022461f,-10008.484375f,-10009.528320f,-10009.576172f,-10009.312500f,-10008.448242f,-10008.470703f,-10008.241211f,-10008.547852f,-10008.868164f,-10008.882812f,-10008.439453f,-10007.578125f,-10006.564453f,-10005.540039f,-10004.729492f,
-10005.246094f,-10006.164062f,-10007.552734f,-10008.874023f,-10009.597656f,-10009.588867f,-10008.772461f,-10008.133789f,-10007.791992f,-10007.679688f,-10007.732422f,-10007.755859f,-10007.804688f,-10007.112305f,-10005.892578f,-10004.124023f,-10003.202148f,-10002.415039f,-10002.251953f,-10003.971680f,
-10005.114258f,-10006.784180f,-10007.189453f,-10007.539062f,-9993.614258f,-10000.972656f,-10003.018555f,-10001.597656f,-10002.652344f,-10002.655273f,-10002.076172f,-10002.906250f,-10002.594727f,-10000.091797f,-10000.174805f,-9999.683594f,-9999.668945f,-9999.245117f,-9998.418945f,-9996.869141f,
-9995.219727f,-9993.571289f,-9992.878906f,-9993.199219f,-9994.480469f,-9996.014648f,-9997.211914f,-9998.176758f,-9998.562500f,-9998.700195f,-9999.232422f,-9999.812500f,-10000.353516f,-10001.291016f,-10001.646484f,-10001.437500f,-10000.573242f,-9999.543945f,-9998.701172f,-9998.771484f,
-9999.773438f,-10001.416016f,-10003.034180f,-10004.683594f,-10005.202148f,-10005.596680f,-10004.937500f,-10004.837891f,-10005.008789f,-10005.266602f,-10005.594727f,-10005.577148f,-10004.721680f,-10003.523438f,-10001.783203f,-10000.616211f,-10000.131836f,-10000.577148f,-10001.755859f,-10002.968750f,
-10004.103516f,-10004.321289f,-10004.184570f,-10003.831055f,-10003.547852f,-10003.812500f,-10004.255859f,-10004.832031f,-10004.707031f,-10004.392578f,-10003.638672f,-10002.653320f,-10002.087891f,-10001.903320f,-10002.705078f,-10004.686523f,-10006.310547f,-10007.629883f,-10008.669922f,-10008.915039f,
-10008.684570f,-10008.508789f,-10008.434570f,-10008.689453f,-10009.158203f,-10009.389648f,-10009.190430f,-10008.380859f,-10007.170898f,-10005.828125f,-10004.840820f,-10004.815430f,-10005.367188f,-10006.715820f,-10008.105469f,-10008.808594f,-10009.065430f,-10008.383789f,-10008.397461f,-10008.089844f,
-10008.334961f,-10008.721680f,-10009.080078f,-10009.193359f,-10008.970703f,-10008.322266f,-10007.038086f,-10005.531250f,-10005.103516f,-10005.180664f,-10006.121094f,-10007.591797f,-10008.800781f,-10009.463867f,-10009.135742f,-10008.617188f,-10008.198242f,-10007.977539f,-10007.862305f,-10007.853516f,
-10008.172852f,-10007.947266f,-10007.236328f,-10005.698242f,-10004.391602f,-10002.839844f,-10001.690430f,-10002.484375f,-10003.148438f,-10004.878906f,-10005.761719f,-10006.787109f,-9993.755859f,-10001.634766f,-10003.327148f,-10001.206055f,-10001.959961f,-10001.996094f,-10001.835938f,-10003.454102f,
-10003.272461f,-10000.796875f,-10000.821289f,-9999.985352f,-9999.865234f,-9999.615234f,-9999.316406f,-9998.285156f,-9996.802734f,-9994.851562f,-9993.344727f,-9992.635742f,-9993.005859f,-9994.167969f,-9995.570312f,-9997.020508f,-9997.947266f,-9998.311523f,-9998.916992f,-9999.346680f,
-9999.634766f,-10000.583984f,-10001.222656f,-10001.516602f,-10001.233398f,-10000.391602f,-9999.272461f,-9998.566406f,-9998.629883f,-9999.624023f,-10001.071289f,-10003.077148f,-10004.296875f,-10005.376953f,-10005.122070f,-10005.165039f,-10005.188477f,-10005.230469f,-10005.466797f,-10005.775391f,
-10005.527344f,-10004.937500f,-10003.356445f,-10001.782227f,-10000.436523f,-9999.908203f,-10000.347656f,-10001.363281f,-10002.792969f,-10003.621094f,-10004.114258f,-10004.068359f,-10003.826172f,-10003.835938f,-10004.009766f,-10004.496094f,-10004.621094f,-10004.786133f,-10004.567383f,-10003.693359f,
-10002.896484f,-10002.009766f,-10001.879883f,-10003.101562f,-10004.395508f,-10005.943359f,-10007.585938f,-10008.563477f,-10008.828125f,-10008.927734f,-10008.802734f,-10008.837891f,-10009.095703f,-10009.416016f,-10009.581055f,-10009.328125f,-10008.537109f,-10007.213867f,-10005.805664f,-10004.772461f,
-10004.333008f,-10005.002930f,-10006.262695f,-10007.462891f,-10008.398438f,-10008.227539f,-10008.506836f,-10008.253906f,-10008.308594f,-10008.459961f,-10008.800781f,-10009.113281f,-10009.520508f,-10009.561523f,-10008.588867f,-10006.935547f,-10005.842773f,-10004.932617f,-10004.986328f,-10006.080078f,
-10007.409180f,-10008.670898f,-10009.157227f,-10009.131836f,-10008.911133f,-10008.698242f,-10008.307617f,-10008.004883f,-10008.194336f,-10008.176758f,-10007.991211f,-10006.990234f,-10005.889648f,-10004.108398f,-10002.316406f,-10002.080078f,-10001.858398f,-10003.055664f,-10003.961914f,-10005.468750f,
-9993.733398f,-10002.712891f,-10003.984375f,-10001.346680f,-10001.315430f,-10001.025391f,-10000.954102f,-10003.117188f,-10003.218750f,-10001.180664f,-10001.550781f,-10000.548828f,-10000.269531f,-10000.004883f,-9999.981445f,-9999.470703f,-9998.422852f,-9996.689453f,-9994.844727f,-9993.322266f,
-9992.619141f,-9992.913086f,-9993.928711f,-9995.455078f,-9996.783203f,-9997.593750f,-9998.587891f,-9999.135742f,-9999.325195f,-10000.104492f,-10000.759766f,-10001.319336f,-10001.592773f,-10001.222656f,-10000.282227f,-9999.214844f,-9998.440430f,-9998.501953f,-9999.300781f,-10001.061523f,
-10002.582031f,-10004.252930f,-10004.692383f,-10005.256836f,-10005.464844f,-10005.488281f,-10005.541016f,-10005.909180f,-10006.018555f,-10006.032227f,-10004.976562f,-10003.522461f,-10001.731445f,-10000.348633f,-9999.784180f,-10000.073242f,-10001.224609f,-10002.289062f,-10003.333008f,-10003.843750f,
-10003.983398f,-10004.094727f,-10004.109375f,-10004.358398f,-10004.507812f,-10004.937500f,-10005.188477f,-10004.581055f,-10003.936523f,-10002.738281f,-10001.823242f,-10002.101562f,-10002.565430f,-10003.750977f,-10005.520508f,-10007.005859f,-10007.915039f,-10008.572266f,-10008.745117f,-10008.796875f,
-10008.870117f,-10009.099609f,-10009.323242f,-10009.482422f,-10009.144531f,-10008.256836f,-10006.938477f,-10005.352539f,-10003.977539f,-10003.720703f,-10004.291992f,-10005.457031f,-10006.768555f,-10007.162109f,-10007.999023f,-10008.165039f,-10008.406250f,-10008.485352f,-10008.714844f,-10008.922852f,
-10009.668945f,-10010.313477f,-10009.874023f,-10008.618164f,-10007.302734f,-10005.726562f,-10004.827148f,-10005.100586f,-10005.929688f,-10007.285156f,-10008.388672f,-10008.972656f,-10009.309570f,-10009.511719f,-10009.206055f,-10008.745117f,-10008.716797f,-10008.613281f,-10008.645508f,-10008.082031f,
-10007.350586f,-10005.766602f,-10003.839844f,-10002.887695f,-10001.744141f,-10001.983398f,-10002.325195f,-10003.759766f,-9993.666016f,-10003.707031f,-10005.097656f,-10002.165039f,-10001.230469f,-10000.422852f,-10000.064453f,-10002.402344f,-10002.595703f,-10001.206055f,-10002.020508f,-10001.003906f,
-10000.554688f,-10000.109375f,-10000.048828f,-9999.822266f,-9999.203125f,-9997.992188f,-9996.353516f,-9994.599609f,-9993.113281f,-9992.522461f,-9992.750977f,-9993.892578f,-9995.303711f,-9996.495117f,-9997.943359f,-9998.785156f,-9999.076172f,-9999.662109f,-10000.149414f,-10000.706055f,
-10001.291992f,-10001.428711f,-10000.997070f,-10000.118164f,-9999.041992f,-9998.381836f,-9998.420898f,-9999.523438f,-10000.833984f,-10002.729492f,-10003.773438f,-10004.961914f,-10005.550781f,-10005.785156f,-10005.737305f,-10005.967773f,-10006.102539f,-10006.481445f,-10006.005859f,-10005.091797f,
-10003.436523f,-10001.688477f,-10000.349609f,-9999.796875f,-10000.251953f,-10001.078125f,-10002.324219f,-10003.296875f,-10003.881836f,-10004.333984f,-10004.406250f,-10004.420898f,-10004.467773f,-10004.902344f,-10005.395508f,-10005.115234f,-10004.889648f,-10003.866211f,-10002.725586f,-10002.320312f,
-10001.891602f,-10002.375977f,-10003.782227f,-10005.320312f,-10006.685547f,-10007.863281f,-10008.463867f,-10008.693359f,-10008.704102f,-10008.760742f,-10008.785156f,-10009.030273f,-10008.971680f,-10008.620117f,-10007.791992f,-10006.268555f,-10004.400391f,-10003.403320f,-10003.051758f,-10003.688477f,
-10004.857422f,-10005.560547f,-10006.873047f,-10007.569336f,-10008.203125f,-10008.388672f,-10008.548828f,-10008.518555f,-10009.272461f,-10010.218750f,-10010.279297f,-10009.682617f,-10008.701172f,-10007.053711f,-10005.603516f,-10005.084961f,-10005.087891f,-10006.014648f,-10007.252930f,-10008.252930f,
-10009.172852f,-10009.930664f,-10009.997070f,-10009.613281f,-10009.438477f,-10009.042969f,-10008.972656f,-10008.564453f,-10008.171875f,-10007.017578f,-10005.434570f,-10004.359375f,-10002.698242f,-10002.025391f,-10001.546875f,-10002.375977f,-9993.500977f,-10004.424805f,-10006.093750f,-10003.352539f,
-10001.977539f,-10000.624023f,-9999.651367f,-10001.381836f,-10001.557617f,-10000.723633f,-10001.960938f,-10001.201172f,-10000.847656f,-10000.314453f,-9999.996094f,-9999.744141f,-9999.327148f,-9998.622070f,-9997.485352f,-9996.055664f,-9994.331055f,-9993.042969f,-9992.332031f,-9992.674805f,
-9993.655273f,-9994.861328f,-9996.627930f,-9997.903320f,-9998.582031f,-9999.192383f,-9999.573242f,-9999.922852f,-10000.507812f,-10000.965820f,-10001.071289f,-10000.791016f,-9999.928711f,-9999.016602f,-9998.477539f,-9998.778320f,-9999.423828f,-10001.010742f,-10002.279297f,-10003.973633f,
-10005.145508f,-10005.883789f,-10006.062500f,-10006.249023f,-10006.143555f,-10006.549805f,-10006.461914f,-10006.135742f,-10005.019531f,-10003.458008f,-10001.829102f,-10000.568359f,-10000.118164f,-10000.221680f,-10001.100586f,-10002.167969f,-10003.075195f,-10004.013672f,-10004.418945f,-10004.430664f,
-10004.395508f,-10004.664062f,-10005.070312f,-10004.978516f,-10005.147461f,-10004.554688f,-10003.653320f,-10003.046875f,-10001.994141f,-10001.626953f,-10002.291016f,-10003.371094f,-10004.740234f,-10006.190430f,-10007.251953f,-10007.884766f,-10008.118164f,-10008.184570f,-10007.936523f,-10008.035156f,
-10007.980469f,-10008.011719f,-10007.772461f,-10006.823242f,-10005.008789f,-10003.767578f,-10002.621094f,-10002.429688f,-10002.974609f,-10003.534180f,-10005.005859f,-10006.171875f,-10007.366211f,-10007.959961f,-10008.342773f,-10008.254883f,-10008.848633f,-10009.739258f,-10010.074219f,-10010.073242f,
-10009.697266f,-10008.552734f,-10007.146484f,-10006.160156f,-10005.332031f,-10005.470703f,-10006.223633f,-10007.196289f,-10008.446289f,-10009.735352f,-10010.390625f,-10010.411133f,-10010.390625f,-10009.793945f,-10009.433594f,-10008.857422f,-10008.539062f,-10007.731445f,-10006.658203f,-10005.922852f,
-10004.273438f,-10003.080078f,-10001.812500f,-10001.738281f,-9993.528320f,-10004.451172f,-10006.653320f,-10004.283203f,-10003.005859f,-10001.345703f,-9999.829102f,-10000.831055f,-10000.588867f,-9999.936523f,-10001.375000f,-10001.062500f,-10001.037109f,-10000.694336f,-10000.255859f,-9999.896484f,
-9999.455078f,-9998.986328f,-9998.274414f,-9997.359375f,-9995.814453f,-9994.277344f,-9992.859375f,-9992.392578f,-9992.592773f,-9993.400391f,-9995.089844f,-9996.621094f,-9997.729492f,-9998.595703f,-9999.088867f,-9999.293945f,-9999.740234f,-10000.224609f,-10000.560547f,-10000.817383f,
-10000.375000f,-9999.612305f,-9998.849609f,-9998.580078f,-9998.467773f,-9999.414062f,-10000.431641f,-10002.272461f,-10003.875000f,-10005.187500f,-10005.884766f,-10006.299805f,-10006.090820f,-10006.391602f,-10006.415039f,-10006.438477f,-10005.822266f,-10004.726562f,-10003.291992f,-10001.781250f,
-10000.675781f,-9999.999023f,-10000.159180f,-10000.893555f,-10001.822266f,-10003.101562f,-10003.945312f,-10004.228516f,-10004.344727f,-10004.543945f,-10004.795898f,-10004.745117f,-10005.041016f,-10004.792969f,-10004.282227f,-10003.817383f,-10002.588867f,-10001.653320f,-10001.593750f,-10001.993164f,
-10002.962891f,-10004.304688f,-10005.581055f,-10006.584961f,-10007.226562f,-10007.557617f,-10007.291016f,-10007.293945f,-10007.121094f,-10007.233398f,-10007.321289f,-10006.968750f,-10005.502930f,-10004.451172f,-10002.946289f,-10002.062500f,-10001.819336f,-10001.852539f,-10003.033203f,-10004.321289f,
-10005.906250f,-10006.930664f,-10007.801758f,-10007.967773f,-10008.538086f,-10009.273438f,-10009.629883f,-10009.861328f,-10009.955078f,-10009.446289f,-10008.473633f,-10007.491211f,-10006.261719f,-10005.671875f,-10005.675781f,-10006.226562f,-10007.401367f,-10008.917969f,-10010.078125f,-10010.614258f,
-10011.060547f,-10010.631836f,-10010.153320f,-10009.454102f,-10008.989258f,-10008.267578f,-10007.539062f,-10007.187500f,-10005.916992f,-10004.663086f,-10003.038086f,-10002.217773f,-9993.680664f,-10003.984375f,-10006.733398f,-10004.788086f,-10004.164062f,-10002.432617f,-10000.600586f,-10000.718750f,
-10000.011719f,-9999.128906f,-10000.453125f,-10000.261719f,-10000.712891f,-10000.820312f,-10000.497070f,-10000.085938f,-9999.480469f,-9998.960938f,-9998.448242f,-9998.048828f,-9996.961914f,-9995.610352f,-9993.895508f,-9992.733398f,-9992.112305f,-9992.208008f,-9993.438477f,-9994.943359f,
-9996.429688f,-9997.720703f,-9998.584961f,-9998.848633f,-9999.188477f,-9999.512695f,-9999.788086f,-10000.405273f,-10000.404297f,-10000.070312f,-9999.439453f,-9998.958008f,-9998.210938f,-9998.404297f,-9998.752930f,-10000.328125f,-10002.056641f,-10003.800781f,-10005.167969f,-10006.062500f,
-10005.945312f,-10006.140625f,-10006.038086f,-10006.063477f,-10005.750000f,-10005.180664f,-10004.226562f,-10002.898438f,-10001.486328f,-10000.137695f,-9999.470703f,-9999.505859f,-10000.058594f,-10001.360352f,-10002.569336f,-10003.292969f,-10003.758789f,-10004.083984f,-10004.228516f,-10004.118164f,
-10004.285156f,-10004.163086f,-10004.045898f,-10003.923828f,-10002.933594f,-10001.833008f,-10001.243164f,-10000.976562f,-10001.283203f,-10002.144531f,-10003.312500f,-10004.556641f,-10005.653320f,-10006.488281f,-10006.513672f,-10006.593750f,-10006.356445f,-10006.319336f,-10006.457031f,-10006.555664f,
-10005.576172f,-10005.070312f,-10003.693359f,-10002.487305f,-10001.546875f,-10000.901367f,-10001.458984f,-10002.460938f,-10004.146484f,-10005.524414f,-10006.992188f,-10007.722656f,-10008.528320f,-10009.141602f,-10009.378906f,-10009.476562f,-10009.797852f,-10009.810547f,-10009.425781f,-10008.819336f,
-10007.622070f,-10006.549805f,-10005.748047f,-10005.622070f,-10006.291016f,-10007.629883f,-10008.983398f,-10009.959961f,-10011.026367f,-10011.045898f,-10010.794922f,-10010.027344f,-10009.328125f,-10008.442383f,-10007.776367f,-10007.651367f,-10006.872070f,-10005.933594f,-10004.368164f,-10003.134766f,
-9993.654297f,-10003.366211f,-10006.621094f,-10004.941406f,-10005.087891f,-10003.544922f,-10001.733398f,-10001.375000f,-10000.160156f,-9998.644531f,-9999.604492f,-9999.301758f,-10000.125977f,-10000.753906f,-10000.846680f,-10000.634766f,-9999.952148f,-9999.221680f,-9998.639648f,-9998.449219f,
-9997.794922f,-9996.842773f,-9995.230469f,-9993.741211f,-9992.465820f,-9991.809570f,-9992.299805f,-9993.420898f,-9994.970703f,-9996.614258f,-9997.971680f,-9998.537109f,-9998.976562f,-9999.146484f,-9999.162109f,-9999.833008f,-10000.063477f,-10000.157227f,-9999.857422f,-9999.538086f,
-9998.500977f,-9998.169922f,-9997.770508f,-9998.802734f,-10000.302734f,-10002.183594f,-10004.148438f,-10005.691406f,-10006.034180f,-10006.364258f,-10006.158203f,-10006.001953f,-10005.728516f,-10005.501953f,-10005.067383f,-10004.186523f,-10002.931641f,-10001.289062f,-10000.000000f,-9999.297852f,
-9999.225586f,-10000.203125f,-10001.495117f,-10002.630859f,-10003.548828f,-10004.260742f,-10004.541992f,-10004.455078f,-10004.369141f,-10004.111328f,-10004.130859f,-10004.254883f,-10003.648438f,-10002.708984f,-10001.909180f,-10001.207031f,-10000.836914f,-10001.012695f,-10001.672852f,-10002.831055f,
-10004.127930f,-10005.404297f,-10005.887695f,-10006.293945f,-10006.131836f,-10005.833984f,-10005.764648f,-10005.943359f,-10005.213867f,-10005.208008f,-10004.240234f,-10003.063477f,-10001.775391f,-10000.520508f,-10000.360352f,-10000.752930f,-10002.148438f,-10003.561523f,-10005.503906f,-10006.907227f,
-10008.226562f,-10008.981445f,-10009.205078f,-10009.023438f,-10009.324219f,-10009.532227f,-10009.601562f,-10009.537109f,-10008.778320f,-10007.678711f,-10006.396484f,-10005.688477f,-10005.712891f,-10006.546875f,-10007.772461f,-10008.932617f,-10010.553711f,-10011.197266f,-10011.528320f,-10011.009766f,
-10010.237305f,-10009.093750f,-10008.245117f,-10008.054688f,-10007.601562f,-10007.056641f,-10005.853516f,-10004.616211f,-9993.608398f,-10002.432617f,-10006.105469f,-10004.508789f,-10005.448242f,-10004.298828f,-10002.832031f,-10002.403320f,-10000.938477f,-9998.624023f,-9999.002930f,-9998.301758f,
-9999.070312f,-10000.011719f,-10000.604492f,-10000.874023f,-10000.475586f,-9999.686523f,-9999.101562f,-9998.940430f,-9998.616211f,-9998.083984f,-9996.847656f,-9995.363281f,-9993.764648f,-9992.458008f,-9992.085938f,-9992.400391f,-9993.532227f,-9995.142578f,-9996.795898f,-9997.787109f,
-9998.585938f,-9998.882812f,-9998.797852f,-9999.456055f,-9999.819336f,-10000.197266f,-10000.221680f,-10000.280273f,-9999.257812f,-9998.684570f,-9997.651367f,-9997.939453f,-9998.836914f,-10000.344727f,-10002.491211f,-10004.473633f,-10005.484375f,-10006.273438f,-10006.264648f,-10006.095703f,
-10005.855469f,-10005.824219f,-10005.783203f,-10005.393555f,-10004.621094f,-10003.048828f,-10001.463867f,-10000.172852f,-9999.335938f,-9999.616211f,-10000.502930f,-10001.680664f,-10002.867188f,-10004.085938f,-10004.733398f,-10004.981445f,-10004.874023f,-10004.518555f,-10004.592773f,-10004.818359f,
-10004.571289f,-10003.994141f,-10003.270508f,-10002.460938f,-10001.622070f,-10001.064453f,-10000.928711f,-10001.556641f,-10002.598633f,-10004.012695f,-10004.886719f,-10005.787109f,-10006.052734f,-10005.748047f,-10005.600586f,-10005.779297f,-10005.143555f,-10005.459961f,-10004.982422f,-10004.095703f,
-10002.851562f,-10001.275391f,-10000.489258f,-10000.093750f,-10000.791992f,-10001.755859f,-10003.728516f,-10005.551758f,-10007.403320f,-10008.565430f,-10009.110352f,-10008.903320f,-10009.217773f,-10009.479492f,-10009.768555f,-10010.172852f,-10009.997070f,-10009.169922f,-10007.827148f,-10006.817383f,
-10006.190430f,-10006.255859f,-10006.937500f,-10007.791016f,-10009.539062f,-10010.654297f,-10011.711914f,-10011.782227f,-10011.350586f,-10010.255859f,-10009.333008f,-10008.983398f,-10008.617188f,-10008.324219f,-10007.547852f,-10006.599609f,-9993.647461f,-10001.721680f,-10005.376953f,-10003.797852f,
-10005.096680f,-10004.552734f,-10003.584961f,-10003.669922f,-10002.265625f,-9999.293945f,-9999.175781f,-9997.977539f,-9998.281250f,-9999.118164f,-10000.083984f,-10000.900391f,-10001.035156f,-10000.398438f,-9999.813477f,-9999.470703f,-9999.187500f,-9998.893555f,-9998.075195f,-9996.886719f,
-9995.377930f,-9993.827148f,-9992.743164f,-9992.230469f,-9992.577148f,-9993.739258f,-9995.304688f,-9996.637695f,-9997.895508f,-9998.583008f,-9998.598633f,-9999.211914f,-9999.530273f,-9999.945312f,-10000.203125f,-10000.651367f,-9999.959961f,-9999.580078f,-9998.357422f,-9998.079102f,
-9998.291016f,-9999.132812f,-10000.960938f,-10003.001953f,-10004.631836f,-10006.015625f,-10006.472656f,-10006.496094f,-10006.238281f,-10006.192383f,-10006.256836f,-10006.201172f,-10005.982422f,-10004.803711f,-10003.376953f,-10001.889648f,-10000.500000f,-9999.987305f,-10000.130859f,-10000.888672f,
-10001.986328f,-10003.574219f,-10004.621094f,-10005.375000f,-10005.437500f,-10005.031250f,-10005.017578f,-10005.132812f,-10005.024414f,-10004.792969f,-10004.360352f,-10003.823242f,-10002.905273f,-10001.905273f,-10001.042969f,-10000.936523f,-10001.351562f,-10002.440430f,-10003.398438f,-10004.745117f,
-10005.565430f,-10005.518555f,-10005.429688f,-10005.529297f,-10004.817383f,-10005.142578f,-10004.993164f,-10004.509766f,-10003.688477f,-10002.257812f,-10001.252930f,-10000.292969f,-10000.232422f,-10000.469727f,-10002.049805f,-10003.820312f,-10006.050781f,-10007.729492f,-10008.858398f,-10008.937500f,
-10009.327148f,-10009.514648f,-10009.754883f,-10010.360352f,-10010.708984f,-10010.340820f,-10009.391602f,-10008.575195f,-10007.576172f,-10007.010742f,-10006.997070f,-10007.208984f,-10008.568359f,-10009.801758f,-10011.367188f,-10012.120117f,-10012.302734f,-10011.551758f,-10010.690430f,-10010.108398f,
-10009.575195f,-10009.266602f,-10008.761719f,-10008.181641f,-9993.698242f,-10001.638672f,-10005.055664f,-10003.496094f,-10004.767578f,-10004.651367f,-10004.148438f,-10004.875000f,-10003.831055f,-10000.628906f,-10000.292969f,-9998.374023f,-9997.912109f,-9998.207031f,-9999.177734f,-10000.348633f,
-10001.081055f,-10000.793945f,-10000.363281f,-9999.811523f,-9999.405273f,-9999.113281f,-9998.634766f,-9997.838867f,-9996.745117f,-9995.374023f,-9993.979492f,-9992.784180f,-9992.290039f,-9992.671875f,-9993.775391f,-9995.156250f,-9996.771484f,-9997.944336f,-9998.302734f,-9998.968750f,
-9999.230469f,-9999.540039f,-9999.847656f,-10000.575195f,-10000.336914f,-10000.407227f,-9999.434570f,-9998.888672f,-9998.549805f,-9998.625000f,-9999.742188f,-10001.370117f,-10003.262695f,-10005.117188f,-10006.144531f,-10006.566406f,-10006.438477f,-10006.315430f,-10006.262695f,-10006.290039f,
-10006.477539f,-10005.827148f,-10004.886719f,-10003.641602f,-10002.020508f,-10000.907227f,-10000.224609f,-10000.263672f,-10000.889648f,-10002.535156f,-10003.858398f,-10005.118164f,-10005.589844f,-10005.341797f,-10005.295898f,-10005.225586f,-10005.062500f,-10005.086914f,-10004.994141f,-10004.944336f,
-10004.328125f,-10003.263672f,-10001.958008f,-10001.145508f,-10000.774414f,-10001.174805f,-10001.821289f,-10003.324219f,-10004.634766f,-10005.009766f,-10005.193359f,-10005.289062f,-10004.442383f,-10004.542969f,-10004.466797f,-10004.292969f,-10004.015625f,-10003.041992f,-10002.246094f,-10001.062500f,
-10000.416992f,-9999.905273f,-10000.731445f,-10001.992188f,-10004.216797f,-10006.255859f,-10007.999023f,-10008.634766f,-10009.269531f,-10009.432617f,-10009.468750f,-10010.022461f,-10010.631836f,-10010.722656f,-10010.390625f,-10010.091797f,-10009.140625f,-10008.216797f,-10007.590820f,-10007.055664f,
-10007.632812f,-10008.508789f,-10010.179688f,-10011.400391f,-10012.261719f,-10012.119141f,-10011.527344f,-10010.945312f,-10010.125977f,-10009.618164f,-10009.197266f,-10008.927734f,-9993.825195f,-10001.656250f,-10004.911133f,-10003.452148f,-10004.476562f,-10004.396484f,-10004.193359f,-10005.461914f,
-10004.873047f,-10001.844727f,-10001.694336f,-9999.486328f,-9998.294922f,-9997.812500f,-9998.342773f,-9999.509766f,-10000.740234f,-10000.994141f,-10001.019531f,-10000.559570f,-10000.041992f,-9999.544922f,-9999.059570f,-9998.454102f,-9997.729492f,-9996.743164f,-9995.410156f,-9993.932617f,
-9992.793945f,-9992.330078f,-9992.613281f,-9993.605469f,-9995.215820f,-9996.782227f,-9997.649414f,-9998.620117f,-9999.007812f,-9999.185547f,-9999.333008f,-9999.997070f,-9999.982422f,-10000.402344f,-9999.923828f,-9999.488281f,-9999.010742f,-9998.585938f,-9998.931641f,-9999.815430f,
-10001.495117f,-10003.492188f,-10005.059570f,-10006.148438f,-10006.458984f,-10006.474609f,-10006.306641f,-10006.198242f,-10006.460938f,-10006.151367f,-10005.663086f,-10004.928711f,-10003.513672f,-10002.265625f,-10001.047852f,-10000.315430f,-10000.256836f,-10001.550781f,-10002.838867f,-10004.476562f,
-10005.497070f,-10005.710938f,-10005.890625f,-10005.743164f,-10005.367188f,-10005.333008f,-10005.356445f,-10005.643555f,-10005.442383f,-10004.635742f,-10003.363281f,-10002.147461f,-10001.159180f,-10000.732422f,-10000.696289f,-10001.917969f,-10003.428711f,-10004.279297f,-10004.969727f,-10005.377930f,
-10004.588867f,-10004.396484f,-10004.080078f,-10003.830078f,-10003.813477f,-10003.301758f,-10002.909180f,-10001.903320f,-10001.041992f,-10000.048828f,-10000.094727f,-10000.519531f,-10002.250977f,-10004.239258f,-10006.414062f,-10007.758789f,-10008.926758f,-10009.396484f,-10009.386719f,-10009.720703f,
-10010.246094f,-10010.464844f,-10010.608398f,-10010.908203f,-10010.335938f,-10009.518555f,-10008.683594f,-10007.669922f,-10007.426758f,-10007.682617f,-10008.909180f,-10010.225586f,-10011.627930f,-10012.256836f,-10012.256836f,-10012.095703f,-10011.215820f,-10010.483398f,-10009.897461f,-10009.596680f,
-9993.837891f,-10001.718750f,-10004.855469f,-10003.612305f,-10004.391602f,-10004.235352f,-10004.031250f,-10005.629883f,-10005.404297f,-10002.847656f,-10003.129883f,-10001.016602f,-9999.296875f,-9997.995117f,-9997.737305f,-9998.458984f,-9999.830078f,-10000.522461f,-10001.107422f,-10001.000000f,
-10000.584961f,-9999.939453f,-9999.340820f,-9998.786133f,-9998.318359f,-9997.801758f,-9996.827148f,-9995.433594f,-9994.000977f,-9992.818359f,-9992.223633f,-9992.486328f,-9993.675781f,-9995.315430f,-9996.624023f,-9998.031250f,-9998.759766f,-9998.991211f,-9999.047852f,-9999.516602f,
-9999.543945f,-10000.173828f,-10000.210938f,-10000.107422f,-9999.808594f,-9999.191406f,-9998.903320f,-9998.917969f,-9999.935547f,-10001.599609f,-10003.376953f,-10005.054688f,-10005.957031f,-10006.346680f,-10006.246094f,-10005.991211f,-10006.166016f,-10005.991211f,-10005.840820f,-10005.648438f,
-10004.692383f,-10003.704102f,-10002.311523f,-10000.975586f,-10000.152344f,-10000.750000f,-10001.644531f,-10003.314453f,-10004.782227f,-10005.578125f,-10006.233398f,-10006.280273f,-10005.804688f,-10005.639648f,-10005.635742f,-10006.090820f,-10006.258789f,-10005.877930f,-10004.960938f,-10003.718750f,
-10002.410156f,-10001.227539f,-10000.346680f,-10000.912109f,-10002.198242f,-10003.303711f,-10004.478516f,-10005.416992f,-10004.957031f,-10004.685547f,-10004.098633f,-10003.596680f,-10003.583008f,-10003.404297f,-10003.495117f,-10002.911133f,-10002.198242f,-10001.072266f,-10000.492188f,-10000.035156f,
-10000.916992f,-10002.413086f,-10004.597656f,-10006.494141f,-10008.237305f,-10009.258789f,-10009.473633f,-10009.670898f,-10009.992188f,-10010.141602f,-10010.499023f,-10011.261719f,-10011.202148f,-10010.770508f,-10010.053711f,-10008.883789f,-10007.975586f,-10007.489258f,-10007.882812f,-10008.803711f,
-10010.369141f,-10011.609375f,-10012.246094f,-10012.835938f,-10012.208008f,-10011.475586f,-10010.702148f,-10010.249023f,-9993.737305f,-10001.756836f,-10004.868164f,-10003.973633f,-10004.715820f,-10004.331055f,-10003.998047f,-10005.478516f,-10005.554688f,-10003.134766f,-10004.078125f,-10002.465820f,
-10000.772461f,-9999.023438f,-9998.018555f,-9997.987305f,-9998.997070f,-9999.754883f,-10000.701172f,-10001.113281f,-10001.031250f,-10000.431641f,-9999.671875f,-9998.961914f,-9998.445312f,-9998.159180f,-9997.583984f,-9996.583008f,-9995.353516f,-9993.971680f,-9992.783203f,-9992.298828f,
-9992.720703f,-9993.968750f,-9995.333008f,-9997.093750f,-9998.279297f,-9998.812500f,-9998.932617f,-9999.222656f,-9999.089844f,-9999.618164f,-9999.914062f,-10000.188477f,-10000.363281f,-10000.041992f,-9999.632812f,-9999.142578f,-9999.387695f,-10000.331055f,-10001.826172f,-10003.702148f,
-10005.111328f,-10006.038086f,-10006.291992f,-10006.060547f,-10006.057617f,-10005.764648f,-10005.601562f,-10005.718750f,-10005.263672f,-10004.808594f,-10003.751953f,-10002.302734f,-10001.034180f,-10000.916016f,-10001.155273f,-10002.427734f,-10003.916992f,-10005.091797f,-10006.259766f,-10006.742188f,
-10006.385742f,-10006.082031f,-10005.913086f,-10006.194336f,-10006.418945f,-10006.363281f,-10005.959961f,-10005.065430f,-10003.927734f,-10002.427734f,-10000.929688f,-10000.688477f,-10001.299805f,-10002.188477f,-10003.506836f,-10004.895508f,-10004.956055f,-10004.911133f,-10004.214844f,-10003.387695f,
-10003.055664f,-10002.841797f,-10003.163086f,-10003.024414f,-10002.742188f,-10001.954102f,-10001.273438f,-10000.370117f,-10000.449219f,-10001.166992f,-10002.844727f,-10004.775391f,-10006.885742f,-10008.539062f,-10009.275391f,-10009.628906f,-10009.822266f,-10009.781250f,-10009.971680f,-10010.830078f,
-10011.157227f,-10011.245117f,-10010.984375f,-10010.148438f,-10009.132812f,-10008.219727f,-10007.790039f,-10007.983398f,-10009.170898f,-10010.489258f,-10011.469727f,-10012.855469f,-10012.802734f,-10012.405273f,-10011.648438f,-10010.965820f,-9993.841797f,-10001.227539f,-10004.328125f,-10003.742188f,
-10004.754883f,-10004.248047f,-10003.740234f,-10005.220703f,-10005.234375f,-10002.937500f,-10004.309570f,-10003.333008f,-10002.083008f,-10000.318359f,-9998.759766f,-9997.926758f,-9998.265625f,-9998.736328f,-9999.779297f,-10000.705078f,-10001.164062f,-10000.920898f,-10000.233398f,-9999.442383f,
-9998.768555f,-9998.483398f,-9998.180664f,-9997.613281f,-9996.803711f,-9995.582031f,-9994.142578f,-9992.985352f,-9992.534180f,-9992.989258f,-9993.994141f,-9995.809570f,-9997.392578f,-9998.391602f,-9998.796875f,-9999.091797f,-9998.825195f,-9999.095703f,-9999.345703f,-9999.834961f,
-10000.470703f,-10000.608398f,-10000.422852f,-9999.774414f,-9999.330078f,-9999.407227f,-10000.217773f,-10001.873047f,-10003.523438f,-10005.007812f,-10005.871094f,-10005.975586f,-10006.029297f,-10005.632812f,-10005.322266f,-10005.500977f,-10005.446289f,-10005.547852f,-10005.081055f,-10003.907227f,
-10002.523438f,-10001.794922f,-10001.334961f,-10001.897461f,-10003.008789f,-10004.270508f,-10005.822266f,-10006.895508f,-10006.951172f,-10006.691406f,-10006.423828f,-10006.403320f,-10006.427734f,-10006.487305f,-10006.493164f,-10006.088867f,-10005.392578f,-10003.926758f,-10002.138672f,-10001.148438f,
-10000.892578f,-10001.167969f,-10002.223633f,-10003.833984f,-10004.482422f,-10004.917969f,-10004.460938f,-10003.484375f,-10002.813477f,-10002.324219f,-10002.612305f,-10002.713867f,-10002.856445f,-10002.573242f,-10002.097656f,-10001.122070f,-10000.551758f,-10000.376953f,-10001.158203f,-10002.643555f,
-10004.732422f,-10006.863281f,-10008.264648f,-10009.041016f,-10009.395508f,-10009.289062f,-10009.203125f,-10009.846680f,-10010.352539f,-10010.832031f,-10011.076172f,-10010.814453f,-10010.069336f,-10009.136719f,-10008.100586f,-10007.499023f,-10007.997070f,-10008.988281f,-10009.979492f,-10011.968750f,
-10012.669922f,-10012.949219f,-10012.524414f,-10011.837891f,-9993.619141f,-10000.806641f,-10003.897461f,-10003.333008f,-10004.528320f,-10004.357422f,-10003.916992f,-10005.316406f,-10005.222656f,-10002.703125f,-10004.353516f,-10004.181641f,-10003.607422f,-10002.198242f,-10000.546875f,-9999.102539f,
-9998.568359f,-9998.264648f,-9998.806641f,-9999.773438f,-10000.622070f,-10000.918945f,-10000.605469f,-9999.938477f,-9999.139648f,-9998.668945f,-9998.444336f,-9998.157227f,-9997.796875f,-9997.054688f,-9995.804688f,-9994.404297f,-9993.247070f,-9992.782227f,-9992.954102f,-9994.350586f,
-9995.971680f,-9997.433594f,-9998.337891f,-9999.030273f,-9998.942383f,-9999.103516f,-9999.129883f,-9999.689453f,-10000.613281f,-10001.236328f,-10001.522461f,-10001.228516f,-10000.537109f,-9999.906250f,-9999.819336f,-10000.709961f,-10002.025391f,-10003.723633f,-10005.142578f,-10005.802734f,
-10006.246094f,-10005.945312f,-10005.540039f,-10005.573242f,-10005.680664f,-10006.105469f,-10006.203125f,-10005.557617f,-10004.435547f,-10003.488281f,-10002.518555f,-10002.239258f,-10002.527344f,-10003.331055f,-10004.820312f,-10006.337891f,-10006.964844f,-10007.102539f,-10007.043945f,-10006.904297f,
-10006.642578f,-10006.670898f,-10006.864258f,-10006.876953f,-10006.713867f,-10005.606445f,-10003.988281f,-10002.598633f,-10001.534180f,-10000.863281f,-10001.132812f,-10002.386719f,-10003.288086f,-10004.231445f,-10004.289062f,-10003.571289f,-10002.801758f,-10002.029297f,-10002.085938f,-10002.236328f,
-10002.631836f,-10002.858398f,-10002.899414f,-10002.321289f,-10001.634766f,-10000.886719f,-10000.675781f,-10001.251953f,-10002.760742f,-10004.892578f,-10006.760742f,-10008.163086f,-10009.052734f,-10009.267578f,-10009.039062f,-10009.428711f,-10009.881836f,-10010.543945f,-10011.123047f,-10011.387695f,
-10011.204102f,-10010.642578f,-10009.481445f,-10008.356445f,-10007.962891f,-10008.106445f,-10008.485352f,-10010.519531f,-10011.720703f,-10012.786133f,-10013.033203f,-10012.725586f,-9993.770508f,-10000.528320f,-10003.245117f,-10002.668945f,-10004.108398f,-10004.479492f,-10004.241211f,-10005.711914f,
-10005.408203f,-10002.770508f,-10004.246094f,-10004.362305f,-10004.355469f,-10003.494141f,-10002.206055f,-10000.675781f,-9999.579102f,-9998.474609f,-9998.220703f,-9998.768555f,-9999.659180f,-10000.407227f,-10000.646484f,-10000.341797f,-9999.590820f,-9998.898438f,-9998.549805f,-9998.256836f,
-9998.093750f,-9997.799805f,-9996.995117f,-9995.780273f,-9994.405273f,-9993.249023f,-9992.517578f,-9993.097656f,-9994.286133f,-9995.888672f,-9997.190430f,-9998.406250f,-9998.725586f,-9998.958008f,-9998.726562f,-9999.123047f,-9999.998047f,-10000.786133f,-10001.479492f,-10001.729492f,
-10001.285156f,-10000.454102f,-9999.692383f,-9999.673828f,-10000.275391f,-10001.708008f,-10003.377930f,-10004.578125f,-10005.583984f,-10005.665039f,-10005.353516f,-10005.222656f,-10005.294922f,-10005.729492f,-10006.137695f,-10006.007812f,-10005.368164f,-10004.674805f,-10003.690430f,-10002.846680f,
-10002.342773f,-10002.397461f,-10003.419922f,-10005.003906f,-10006.108398f,-10006.830078f,-10007.241211f,-10007.293945f,-10006.883789f,-10006.791016f,-10006.934570f,-10007.118164f,-10007.315430f,-10006.702148f,-10005.613281f,-10004.386719f,-10003.000000f,-10001.570312f,-10000.949219f,-10001.406250f,
-10002.092773f,-10003.255859f,-10003.843750f,-10003.661133f,-10003.086914f,-10002.138672f,-10001.901367f,-10001.822266f,-10002.148438f,-10002.546875f,-10003.003906f,-10002.962891f,-10002.616211f,-10001.792969f,-10000.926758f,-10000.565430f,-10001.169922f,-10002.759766f,-10004.604492f,-10006.458984f,
-10007.950195f,-10008.721680f,-10008.679688f,-10008.899414f,-10009.140625f,-10009.703125f,-10010.267578f,-10010.763672f,-10011.075195f,-10011.120117f,-10010.355469f,-10009.219727f,-10008.300781f,-10007.644531f,-10007.220703f,-10008.718750f,-10009.988281f,-10011.544922f,-10012.521484f,-10012.888672f,
-9993.710938f,-10000.675781f,-10003.036133f,-10001.815430f,-10003.140625f,-10003.820312f,-10003.916992f,-10005.916992f,-10005.857422f,-10003.018555f,-10004.419922f,-10004.523438f,-10004.829102f,-10004.432617f,-10003.726562f,-10002.525391f,-10001.291016f,-9999.649414f,-9998.538086f,-9998.254883f,
-9998.647461f,-9999.483398f,-10000.248047f,-10000.521484f,-10000.169922f,-9999.542969f,-9999.167969f,-9998.770508f,-9998.575195f,-9998.522461f,-9998.114258f,-9997.314453f,-9996.110352f,-9994.694336f,-9993.258789f,-9992.890625f,-9993.187500f,-9994.382812f,-9995.713867f,-9997.407227f,
-9998.352539f,-9999.020508f,-9998.819336f,-9999.142578f,-9999.809570f,-10000.507812f,-10001.297852f,-10002.007812f,-10002.041992f,-10001.438477f,-10000.425781f,-9999.677734f,-9999.354492f,-10000.065430f,-10001.419922f,-10002.818359f,-10004.402344f,-10005.104492f,-10005.300781f,-10005.287109f,
-10005.379883f,-10005.695312f,-10006.140625f,-10006.292969f,-10006.097656f,-10005.852539f,-10005.213867f,-10004.144531f,-10003.085938f,-10002.343750f,-10002.491211f,-10003.617188f,-10004.783203f,-10005.980469f,-10007.011719f,-10007.608398f,-10007.402344f,-10007.372070f,-10007.438477f,-10007.598633f,
-10007.885742f,-10007.607422f,-10007.072266f,-10006.285156f,-10005.006836f,-10003.236328f,-10001.897461f,-10001.301758f,-10001.263672f,-10002.094727f,-10002.938477f,-10003.382812f,-10003.353516f,-10002.615234f,-10002.336914f,-10002.092773f,-10002.216797f,-10002.505859f,-10003.117188f,-10003.393555f,
-10003.540039f,-10003.043945f,-10001.981445f,-10000.987305f,-10000.625977f,-10001.224609f,-10002.450195f,-10004.271484f,-10006.176758f,-10007.621094f,-10008.174805f,-10008.583008f,-10008.810547f,-10009.267578f,-10009.663086f,-10010.104492f,-10010.631836f,-10011.141602f,-10010.965820f,-10010.191406f,
-10009.176758f,-10008.012695f,-10006.826172f,-10007.428711f,-10008.202148f,-10009.721680f,-10011.164062f,-10012.282227f,-9993.818359f,-10001.331055f,-10003.262695f,-10001.357422f,-10002.260742f,-10002.779297f,-10003.139648f,-10005.845703f,-10005.807617f,-10003.120117f,-10004.610352f,-10004.539062f,
-10004.942383f,-10004.887695f,-10004.755859f,-10004.157227f,-10003.167969f,-10001.407227f,-9999.675781f,-9998.500000f,-9998.005859f,-9998.404297f,-9999.253906f,-9999.906250f,-10000.024414f,-9999.663086f,-9999.461914f,-9999.046875f,-9998.767578f,-9998.782227f,-9998.655273f,-9998.311523f,
-9997.572266f,-9996.309570f,-9994.569336f,-9993.403320f,-9992.704102f,-9993.063477f,-9993.957031f,-9995.704102f,-9997.100586f,-9998.279297f,-9998.418945f,-9998.905273f,-9999.482422f,-10000.053711f,-10000.811523f,-10001.824219f,-10002.395508f,-10002.316406f,-10001.491211f,-10000.414062f,
-9999.345703f,-9999.138672f,-9999.728516f,-10000.829102f,-10002.570312f,-10003.748047f,-10004.534180f,-10004.888672f,-10005.196289f,-10005.499023f,-10005.903320f,-10006.154297f,-10006.320312f,-10006.573242f,-10006.526367f,-10005.560547f,-10004.328125f,-10002.977539f,-10002.218750f,-10002.505859f,
-10003.209961f,-10004.482422f,-10005.922852f,-10007.077148f,-10007.340820f,-10007.616211f,-10007.774414f,-10007.913086f,-10008.170898f,-10008.130859f,-10008.039062f,-10007.828125f,-10006.967773f,-10005.307617f,-10003.654297f,-10002.154297f,-10001.148438f,-10001.218750f,-10001.785156f,-10002.526367f,
-10002.980469f,-10002.613281f,-10002.565430f,-10002.350586f,-10002.375000f,-10002.510742f,-10003.096680f,-10003.519531f,-10004.147461f,-10004.207031f,-10003.357422f,-10002.207031f,-10001.148438f,-10000.751953f,-10001.008789f,-10002.275391f,-10004.074219f,-10005.896484f,-10007.129883f,-10007.908203f,
-10008.385742f,-10008.965820f,-10009.260742f,-10009.583984f,-10010.151367f,-10010.938477f,-10011.313477f,-10011.091797f,-10010.329102f,-10009.046875f,-10007.420898f,-10007.090820f,-10007.062500f,-10007.988281f,-10009.349609f,-10010.853516f,-9993.941406f,-10001.997070f,-10003.739258f,-10001.186523f,
-10001.432617f,-10001.419922f,-10001.774414f,-10004.899414f,-10005.052734f,-10002.890625f,-10004.595703f,-10004.303711f,-10004.596680f,-10004.585938f,-10004.766602f,-10004.782227f,-10004.372070f,-10003.082031f,-10001.329102f,-9999.630859f,-9998.243164f,-9997.872070f,-9998.360352f,-9999.058594f,
-9999.540039f,-9999.608398f,-9999.737305f,-9999.435547f,-9999.098633f,-9998.938477f,-9998.788086f,-9998.677734f,-9998.407227f,-9997.653320f,-9996.190430f,-9994.771484f,-9993.317383f,-9992.744141f,-9992.839844f,-9994.148438f,-9995.592773f,-9997.135742f,-9997.796875f,-9998.617188f,
-9999.266602f,-9999.750000f,-10000.279297f,-10001.242188f,-10002.091797f,-10002.562500f,-10002.324219f,-10001.553711f,-10000.255859f,-9999.382812f,-9999.076172f,-9999.489258f,-10000.906250f,-10002.227539f,-10003.499023f,-10004.364258f,-10005.023438f,-10005.501953f,-10005.840820f,-10005.964844f,
-10006.196289f,-10006.729492f,-10007.201172f,-10006.626953f,-10005.721680f,-10004.286133f,-10002.928711f,-10002.340820f,-10002.232422f,-10003.073242f,-10004.480469f,-10005.940430f,-10006.727539f,-10007.424805f,-10007.832031f,-10008.002930f,-10008.125977f,-10008.063477f,-10008.047852f,-10008.242188f,
-10007.893555f,-10006.824219f,-10005.483398f,-10003.639648f,-10001.853516f,-10001.022461f,-10000.908203f,-10001.478516f,-10002.139648f,-10002.187500f,-10002.563477f,-10002.584961f,-10002.692383f,-10002.686523f,-10003.054688f,-10003.235352f,-10004.044922f,-10004.572266f,-10004.241211f,-10003.520508f,
-10002.370117f,-10001.344727f,-10000.709961f,-10001.102539f,-10002.295898f,-10004.016602f,-10005.685547f,-10006.911133f,-10007.837891f,-10008.735352f,-10009.123047f,-10009.345703f,-10009.739258f,-10010.461914f,-10011.089844f,-10011.326172f,-10011.099609f,-10010.238281f,-10008.768555f,-10007.944336f,
-10007.166016f,-10007.173828f,-10007.938477f,-10009.295898f,-9993.890625f,-10002.889648f,-10004.407227f,-10001.877930f,-10001.225586f,-10000.724609f,-10000.743164f,-10003.980469f,-10004.089844f,-10002.536133f,-10004.496094f,-10004.217773f,-10004.424805f,-10004.364258f,-10004.589844f,-10005.004883f,
-10005.122070f,-10004.490234f,-10003.112305f,-10001.367188f,-9999.390625f,-9998.252930f,-9998.040039f,-9998.368164f,-9998.894531f,-9999.297852f,-9999.788086f,-9999.758789f,-9999.546875f,-9999.209961f,-9998.919922f,-9998.825195f,-9998.801758f,-9998.538086f,-9997.543945f,-9996.290039f,
-9994.534180f,-9993.196289f,-9992.479492f,-9992.981445f,-9994.044922f,-9995.601562f,-9996.663086f,-9997.901367f,-9998.785156f,-9999.401367f,-9999.799805f,-10000.597656f,-10001.460938f,-10002.248047f,-10002.605469f,-10002.400391f,-10001.343750f,-10000.216797f,-9999.274414f,-9998.914062f,
-9999.674805f,-10000.721680f,-10002.133789f,-10003.411133f,-10004.491211f,-10005.357422f,-10005.837891f,-10005.866211f,-10006.106445f,-10006.729492f,-10007.533203f,-10007.364258f,-10006.956055f,-10005.793945f,-10004.295898f,-10003.098633f,-10002.182617f,-10002.311523f,-10003.268555f,-10004.651367f,
-10005.770508f,-10006.847656f,-10007.646484f,-10008.002930f,-10008.134766f,-10007.998047f,-10007.879883f,-10008.229492f,-10008.227539f,-10007.761719f,-10006.986328f,-10005.279297f,-10003.105469f,-10001.562500f,-10000.571289f,-10000.545898f,-10000.943359f,-10001.135742f,-10001.809570f,-10002.150391f,
-10002.528320f,-10002.541992f,-10002.766602f,-10002.637695f,-10003.369141f,-10004.092773f,-10004.207031f,-10004.094727f,-10003.227539f,-10002.100586f,-10000.919922f,-10000.500000f,-10000.839844f,-10002.038086f,-10003.715820f,-10005.171875f,-10006.554688f,-10007.883789f,-10008.623047f,-10008.963867f,
-10009.292969f,-10009.825195f,-10010.438477f,-10010.898438f,-10011.153320f,-10010.851562f,-10009.904297f,-10009.043945f,-10007.893555f,-10007.096680f,-10007.082031f,-10007.882812f,-9993.817383f,-10003.510742f,-10005.392578f,-10002.937500f,-10002.009766f,-10000.621094f,-10000.027344f,-10002.842773f,
-10002.908203f,-10001.820312f,-10004.190430f,-10004.115234f,-10004.414062f,-10004.319336f,-10004.333008f,-10004.811523f,-10005.225586f,-10005.235352f,-10004.507812f,-10003.193359f,-10001.074219f,-9999.364258f,-9998.369141f,-9998.079102f,-9998.297852f,-9998.811523f,-9999.638672f,-10000.005859f,
-10000.150391f,-9999.809570f,-9999.350586f,-9999.050781f,-9998.958984f,-9999.002930f,-9998.516602f,-9997.762695f,-9996.156250f,-9994.468750f,-9993.033203f,-9992.677734f,-9993.020508f,-9994.179688f,-9995.362305f,-9996.941406f,-9998.204102f,-9999.134766f,-9999.598633f,-10000.182617f,
-10000.734375f,-10001.494141f,-10002.224609f,-10002.635742f,-10002.144531f,-10001.212891f,-10000.001953f,-9999.004883f,-9998.950195f,-9999.385742f,-10000.528320f,-10001.951172f,-10003.389648f,-10004.764648f,-10005.608398f,-10005.709961f,-10005.923828f,-10006.404297f,-10007.211914f,-10007.289062f,
-10007.372070f,-10006.752930f,-10005.544922f,-10004.133789f,-10002.654297f,-10002.024414f,-10002.287109f,-10003.205078f,-10004.356445f,-10005.658203f,-10006.896484f,-10007.620117f,-10007.998047f,-10007.920898f,-10007.597656f,-10007.813477f,-10007.904297f,-10007.936523f,-10007.816406f,-10006.688477f,
-10004.589844f,-10002.738281f,-10001.025391f,-10000.195312f,-9999.987305f,-10000.007812f,-10000.769531f,-10001.432617f,-10002.208984f,-10002.492188f,-10002.757812f,-10002.382812f,-10002.779297f,-10003.340820f,-10003.651367f,-10004.056641f,-10003.723633f,-10002.952148f,-10001.706055f,-10000.727539f,
-10000.212891f,-10000.575195f,-10001.788086f,-10003.135742f,-10004.792969f,-10006.544922f,-10007.801758f,-10008.475586f,-10008.933594f,-10009.253906f,-10009.601562f,-10009.977539f,-10010.429688f,-10010.625977f,-10010.362305f,-10009.916992f,-10008.840820f,-10007.564453f,-10006.859375f,-10006.881836f,
-9993.932617f,-10003.475586f,-10005.604492f,-10003.696289f,-10002.834961f,-10001.248047f,-10000.056641f,-10001.867188f,-10001.464844f,-10000.616211f,-10002.972656f,-10003.177734f,-10003.804688f,-10003.935547f,-10003.854492f,-10004.249023f,-10004.685547f,-10005.040039f,-10004.914062f,-10004.322266f,
-10002.619141f,-10000.842773f,-9999.262695f,-9998.212891f,-9997.731445f,-9997.881836f,-9998.636719f,-9999.281250f,-9999.901367f,-9999.862305f,-9999.541992f,-9999.122070f,-9998.857422f,-9998.948242f,-9998.734375f,-9998.555664f,-9997.456055f,-9995.954102f,-9994.272461f,-9993.250977f,
-9992.722656f,-9993.070312f,-9993.818359f,-9995.348633f,-9996.853516f,-9998.204102f,-9999.078125f,-9999.765625f,-10000.091797f,-10000.648438f,-10001.427734f,-10002.200195f,-10002.293945f,-10001.921875f,-10000.999023f,-9999.799805f,-9999.095703f,-9998.708008f,-9999.097656f,-10000.128906f,
-10001.530273f,-10003.239258f,-10004.543945f,-10005.010742f,-10005.412109f,-10005.846680f,-10006.537109f,-10006.719727f,-10007.062500f,-10006.938477f,-10006.286133f,-10005.171875f,-10003.630859f,-10002.499023f,-10002.004883f,-10002.157227f,-10002.833984f,-10003.932617f,-10005.349609f,-10006.445312f,
-10007.288086f,-10007.554688f,-10007.291992f,-10007.438477f,-10007.482422f,-10007.690430f,-10008.014648f,-10007.620117f,-10006.049805f,-10004.482422f,-10002.494141f,-10000.978516f,-9999.935547f,-9999.333984f,-9999.718750f,-10000.337891f,-10001.382812f,-10002.030273f,-10002.638672f,-10002.427734f,
-10002.663086f,-10002.956055f,-10003.225586f,-10003.791016f,-10003.913086f,-10003.760742f,-10002.930664f,-10001.936523f,-10000.933594f,-10000.459961f,-10000.761719f,-10001.493164f,-10002.931641f,-10004.752930f,-10006.400391f,-10007.531250f,-10008.456055f,-10008.893555f,-10009.149414f,-10009.335938f,
-10009.750977f,-10010.125977f,-10010.357422f,-10010.427734f,-10009.839844f,-10008.609375f,-10007.561523f,-10006.891602f,-9993.872070f,-10003.163086f,-10005.758789f,-10004.401367f,-10004.144531f,-10002.343750f,-10000.713867f,-10001.526367f,-10000.669922f,-9999.667969f,-10001.795898f,-10002.131836f,
-10003.198242f,-10003.706055f,-10003.651367f,-10003.930664f,-10004.220703f,-10004.625977f,-10004.898438f,-10004.997070f,-10003.940430f,-10002.482422f,-10000.696289f,-9999.067383f,-9997.830078f,-9997.416016f,-9997.815430f,-9998.529297f,-9999.574219f,-9999.950195f,-9999.941406f,-9999.518555f,
-9999.031250f,-9998.953125f,-9998.716797f,-9998.935547f,-9998.362305f,-9997.315430f,-9995.711914f,-9994.375977f,-9993.078125f,-9992.589844f,-9992.580078f,-9993.744141f,-9995.263672f,-9996.959961f,-9998.358398f,-9999.295898f,-9999.520508f,-9999.821289f,-10000.410156f,-10001.228516f,
-10001.745117f,-10001.979492f,-10001.631836f,-10000.630859f,-9999.625977f,-9998.564453f,-9998.128906f,-9998.532227f,-9999.605469f,-10001.456055f,-10003.189453f,-10004.187500f,-10004.916016f,-10005.426758f,-10006.015625f,-10006.181641f,-10006.546875f,-10006.739258f,-10006.653320f,-10006.012695f,
-10004.807617f,-10003.548828f,-10002.496094f,-10001.915039f,-10001.937500f,-10002.548828f,-10003.882812f,-10005.251953f,-10006.583008f,-10007.344727f,-10007.337891f,-10007.474609f,-10007.417969f,-10007.562500f,-10008.034180f,-10008.202148f,-10007.210938f,-10006.189453f,-10004.371094f,-10002.442383f,
-10000.636719f,-9999.258789f,-9999.001953f,-9999.252930f,-10000.356445f,-10001.289062f,-10002.328125f,-10002.469727f,-10002.703125f,-10002.711914f,-10002.765625f,-10003.135742f,-10003.526367f,-10003.859375f,-10003.614258f,-10002.954102f,-10001.880859f,-10000.817383f,-10000.212891f,-10000.161133f,
-10001.064453f,-10002.643555f,-10004.484375f,-10006.046875f,-10007.547852f,-10008.320312f,-10008.682617f,-10008.714844f,-10008.952148f,-10009.282227f,-10009.757812f,-10010.277344f,-10010.296875f,-10009.491211f,-10008.513672f,-10007.447266f,-9993.968750f,-10002.378906f,-10005.348633f,-10004.229492f,
-10004.774414f,-10003.334961f,-10001.689453f,-10001.926758f,-10000.566406f,-9998.810547f,-10000.373047f,-10000.569336f,-10002.000000f,-10003.072266f,-10003.443359f,-10003.893555f,-10004.047852f,-10004.219727f,-10004.499023f,-10004.942383f,-10004.545898f,-10003.717773f,-10002.251953f,-10000.474609f,
-9998.691406f,-9997.562500f,-9997.249023f,-9997.617188f,-9998.756836f,-9999.574219f,-10000.118164f,-10000.040039f,-9999.602539f,-9999.326172f,-9998.772461f,-9998.996094f,-9998.707031f,-9998.166992f,-9996.950195f,-9995.805664f,-9994.180664f,-9993.030273f,-9992.120117f,-9992.503906f,
-9993.603516f,-9995.257812f,-9997.131836f,-9998.591797f,-9999.163086f,-9999.443359f,-9999.788086f,-10000.306641f,-10000.835938f,-10001.441406f,-10001.716797f,-10001.281250f,-10000.525391f,-9999.211914f,-9998.149414f,-9997.740234f,-9998.132812f,-9999.625977f,-10001.458008f,-10002.968750f,
-10004.210938f,-10005.135742f,-10005.870117f,-10006.078125f,-10006.239258f,-10006.380859f,-10006.560547f,-10006.359375f,-10005.753906f,-10004.845703f,-10003.681641f,-10002.669922f,-10001.985352f,-10001.833984f,-10002.624023f,-10003.911133f,-10005.519531f,-10006.845703f,-10007.418945f,-10007.906250f,
-10007.958984f,-10007.881836f,-10008.155273f,-10008.445312f,-10007.867188f,-10007.463867f,-10006.237305f,-10004.420898f,-10002.299805f,-10000.244141f,-9999.178711f,-9998.665039f,-9999.341797f,-10000.227539f,-10001.625977f,-10002.358398f,-10002.978516f,-10003.014648f,-10002.913086f,-10002.821289f,
-10003.059570f,-10003.498047f,-10003.681641f,-10003.588867f,-10002.968750f,-10001.833984f,-10000.661133f,-9999.833008f,-9999.916016f,-10000.797852f,-10002.344727f,-10003.995117f,-10006.038086f,-10007.411133f,-10008.353516f,-10008.582031f,-10008.746094f,-10008.802734f,-10009.150391f,-10009.658203f,
-10010.101562f,-10009.869141f,-10009.363281f,-10008.473633f,-9993.870117f,-10001.709961f,-10004.877930f,-10003.922852f,-10005.100586f,-10004.138672f,-10002.819336f,-10003.059570f,-10001.443359f,-9998.806641f,-9999.642578f,-9999.394531f,-10000.695312f,-10001.996094f,-10002.793945f,-10003.605469f,
-10003.966797f,-10004.056641f,-10004.347656f,-10004.884766f,-10004.954102f,-10004.733398f,-10003.831055f,-10002.292969f,-10000.380859f,-9998.807617f,-9997.757812f,-9997.456055f,-9998.216797f,-9999.057617f,-9999.889648f,-10000.230469f,-10000.088867f,-9999.916992f,-9999.233398f,-9999.397461f,
-9999.246094f,-9999.033203f,-9998.190430f,-9997.457031f,-9995.867188f,-9994.477539f,-9992.853516f,-9992.407227f,-9992.739258f,-9993.875000f,-9995.724609f,-9997.391602f,-9998.418945f,-9998.968750f,-9999.353516f,-9999.727539f,-10000.213867f,-10000.969727f,-10001.657227f,-10001.765625f,
-10001.543945f,-10000.379883f,-9999.097656f,-9998.137695f,-9997.792969f,-9998.609375f,-10000.048828f,-10001.649414f,-10003.131836f,-10004.505859f,-10005.575195f,-10006.097656f,-10006.291016f,-10006.432617f,-10006.753906f,-10006.823242f,-10006.726562f,-10006.349609f,-10005.400391f,-10004.373047f,
-10003.292969f,-10002.455078f,-10002.503906f,-10003.280273f,-10004.694336f,-10006.171875f,-10007.157227f,-10008.091797f,-10008.558594f,-10008.523438f,-10008.716797f,-10009.014648f,-10008.610352f,-10008.573242f,-10007.964844f,-10006.517578f,-10004.541016f,-10002.191406f,-10000.545898f,-9999.203125f,
-9999.138672f,-9999.522461f,-10000.795898f,-10001.753906f,-10002.745117f,-10003.057617f,-10003.149414f,-10002.859375f,-10002.998047f,-10003.337891f,-10003.650391f,-10003.977539f,-10003.916016f,-10003.031250f,-10001.765625f,-10000.551758f,-9999.907227f,-9999.907227f,-10000.814453f,-10002.046875f,
-10004.117188f,-10005.804688f,-10007.342773f,-10008.016602f,-10008.508789f,-10008.598633f,-10008.895508f,-10009.307617f,-10009.872070f,-10010.012695f,-10009.990234f,-10009.523438f,-9993.946289f,-10001.351562f,-10004.473633f,-10003.374023f,-10004.922852f,-10004.307617f,-10003.375977f,-10004.090820f,
-10002.472656f,-9999.254883f,-9999.539062f,-9998.618164f,-9999.470703f,-10000.728516f,-10001.911133f,-10003.253906f,-10004.060547f,-10004.253906f,-10004.530273f,-10004.877930f,-10005.038086f,-10005.117188f,-10004.756836f,-10003.688477f,-10002.041016f,-10000.386719f,-9998.828125f,-9997.853516f,
-9997.980469f,-9998.541992f,-9999.417969f,-10000.177734f,-10000.521484f,-10000.724609f,-10000.093750f,-10000.085938f,-9999.814453f,-9999.592773f,-9998.898438f,-9998.529297f,-9997.249023f,-9995.991211f,-9994.136719f,-9993.030273f,-9992.585938f,-9992.937500f,-9994.370117f,-9995.976562f,
-9997.473633f,-9998.495117f,-9999.182617f,-9999.573242f,-9999.905273f,-10000.498047f,-10001.190430f,-10001.586914f,-10001.871094f,-10001.132812f,-10000.066406f,-9998.940430f,-9998.063477f,-9998.157227f,-9998.930664f,-10000.272461f,-10001.761719f,-10003.539062f,-10005.065430f,-10006.075195f,
-10006.472656f,-10006.593750f,-10006.835938f,-10006.843750f,-10006.973633f,-10007.062500f,-10006.490234f,-10005.785156f,-10004.723633f,-10003.506836f,-10002.912109f,-10003.037109f,-10003.957031f,-10005.289062f,-10006.500977f,-10007.922852f,-10009.005859f,-10009.284180f,-10009.516602f,-10009.659180f,
-10009.153320f,-10009.090820f,-10008.868164f,-10007.863281f,-10006.363281f,-10004.159180f,-10002.329102f,-10000.388672f,-9999.565430f,-9999.238281f,-10000.036133f,-10000.895508f,-10002.181641f,-10002.948242f,-10003.482422f,-10003.285156f,-10003.298828f,-10003.319336f,-10003.381836f,-10003.776367f,
-10004.097656f,-10003.588867f,-10002.659180f,-10001.479492f,-10000.423828f,-9999.703125f,-9999.830078f,-10000.384766f,-10002.059570f,-10003.803711f,-10005.823242f,-10007.097656f,-10008.159180f,-10008.526367f,-10008.857422f,-10009.001953f,-10009.318359f,-10009.466797f,-10009.722656f,-10009.748047f,
-9993.916016f,-10001.282227f,-10004.243164f,-10003.014648f,-10004.682617f,-10004.390625f,-10003.833008f,-10005.211914f,-10003.904297f,-10000.395508f,-10000.283203f,-9998.722656f,-9998.785156f,-9999.561523f,-10000.801758f,-10002.518555f,-10003.908203f,-10004.425781f,-10004.850586f,-10005.016602f,
-10005.119141f,-10005.341797f,-10005.451172f,-10004.930664f,-10003.792969f,-10002.411133f,-10000.589844f,-9999.007812f,-9998.317383f,-9998.221680f,-9998.735352f,-9999.661133f,-10000.416016f,-10001.151367f,-10000.836914f,-10000.812500f,-10000.461914f,-10000.121094f,-9999.498047f,-9999.398438f,
-9998.535156f,-9997.688477f,-9995.949219f,-9994.439453f,-9993.253906f,-9992.722656f,-9993.338867f,-9994.435547f,-9996.137695f,-9997.584961f,-9998.742188f,-9999.401367f,-9999.743164f,-10000.168945f,-10000.712891f,-10001.213867f,-10001.932617f,-10001.736328f,-10001.133789f,-10000.216797f,
-9999.043945f,-9998.493164f,-9998.424805f,-9999.136719f,-10000.233398f,-10002.133789f,-10003.997070f,-10005.553711f,-10006.395508f,-10006.707031f,-10006.962891f,-10006.856445f,-10007.034180f,-10007.512695f,-10007.365234f,-10007.190430f,-10006.461914f,-10005.161133f,-10004.104492f,-10003.524414f,
-10003.719727f,-10004.498047f,-10005.549805f,-10007.222656f,-10008.888672f,-10009.678711f,-10010.193359f,-10010.321289f,-10009.716797f,-10009.434570f,-10009.415039f,-10008.803711f,-10007.904297f,-10006.153320f,-10004.508789f,-10002.252930f,-10000.759766f,-9999.617188f,-9999.590820f,-9999.901367f,
-10001.150391f,-10002.287109f,-10003.394531f,-10003.592773f,-10003.732422f,-10003.540039f,-10003.273438f,-10003.551758f,-10004.083984f,-10003.973633f,-10003.580078f,-10002.793945f,-10001.629883f,-10000.399414f,-9999.767578f,-9999.453125f,-10000.340820f,-10001.684570f,-10003.825195f,-10005.568359f,
-10007.333984f,-10008.266602f,-10008.855469f,-10008.946289f,-10008.946289f,-10008.958984f,-10009.311523f,-10009.726562f,-9994.053711f,-10001.786133f,-10004.250977f,-10003.148438f,-10004.416016f,-10004.110352f,-10003.875977f,-10005.758789f,-10004.927734f,-10001.682617f,-10001.661133f,-9999.743164f,
-9998.992188f,-9998.947266f,-9999.755859f,-10001.432617f,-10003.246094f,-10004.198242f,-10004.960938f,-10005.076172f,-10005.032227f,-10005.102539f,-10005.357422f,-10005.284180f,-10004.771484f,-10004.028320f,-10002.502930f,-10000.780273f,-9999.504883f,-9998.650391f,-9998.444336f,-9999.081055f,
-9999.912109f,-10001.050781f,-10001.230469f,-10001.351562f,-10001.004883f,-10000.464844f,-9999.688477f,-9999.578125f,-9999.054688f,-9998.671875f,-9997.491211f,-9996.073242f,-9994.619141f,-9993.417969f,-9993.104492f,-9993.276367f,-9994.635742f,-9996.145508f,-9997.689453f,-9998.799805f,
-9999.362305f,-9999.666016f,-9999.926758f,-10000.221680f,-10001.014648f,-10001.261719f,-10001.266602f,-10000.938477f,-9999.997070f,-9999.307617f,-9998.650391f,-9998.616211f,-9999.026367f,-10000.574219f,-10002.433594f,-10004.320312f,-10005.721680f,-10006.444336f,-10006.849609f,-10006.654297f,
-10006.687500f,-10007.253906f,-10007.410156f,-10007.750977f,-10007.613281f,-10006.708008f,-10005.745117f,-10004.803711f,-10004.409180f,-10004.420898f,-10004.885742f,-10006.341797f,-10008.251953f,-10009.576172f,-10010.531250f,-10010.836914f,-10010.219727f,-10009.592773f,-10009.410156f,-10008.879883f,
-10008.445312f,-10007.314453f,-10006.242188f,-10004.223633f,-10002.529297f,-10000.875000f,-9999.948242f,-9999.350586f,-10000.066406f,-10001.158203f,-10002.619141f,-10003.374023f,-10003.825195f,-10003.633789f,-10003.074219f,-10002.995117f,-10003.342773f,-10003.380859f,-10003.516602f,-10003.325195f,
-10002.537109f,-10001.340820f,-10000.358398f,-9999.416992f,-9999.357422f,-9999.910156f,-10001.559570f,-10003.309570f,-10005.561523f,-10007.179688f,-10008.246094f,-10008.613281f,-10008.390625f,-10008.147461f,-10008.321289f,-10008.872070f,-9994.182617f,-10002.145508f,-10004.198242f,-10003.448242f,
-10004.378906f,-10003.952148f,-10003.685547f,-10005.706055f,-10005.408203f,-10002.495117f,-10002.883789f,-10000.893555f,-9999.537109f,-9998.631836f,-9998.617188f,-9999.794922f,-10001.730469f,-10003.136719f,-10004.491211f,-10005.009766f,-10005.115234f,-10005.065430f,-10005.244141f,-10005.308594f,
-10005.166992f,-10005.030273f,-10004.004883f,-10002.529297f,-10001.045898f,-9999.588867f,-9998.570312f,-9998.558594f,-9999.040039f,-10000.374023f,-10001.064453f,-10001.665039f,-10001.691406f,-10001.187500f,-10000.280273f,-9999.926758f,-9999.385742f,-9999.149414f,-9998.441406f,-9997.299805f,
-9995.924805f,-9994.432617f,-9993.338867f,-9992.501953f,-9993.037109f,-9994.123047f,-9995.820312f,-9997.444336f,-9998.537109f,-9999.131836f,-9999.344727f,-9999.390625f,-9999.981445f,-10000.310547f,-10000.622070f,-10000.822266f,-10000.322266f,-9999.879883f,-9999.035156f,-9998.441406f,
-9998.094727f,-9998.946289f,-10000.424805f,-10002.415039f,-10004.375977f,-10005.795898f,-10006.756836f,-10006.840820f,-10006.816406f,-10007.311523f,-10007.479492f,-10008.023438f,-10008.336914f,-10007.920898f,-10007.382812f,-10006.449219f,-10005.812500f,-10005.157227f,-10004.857422f,-10005.687500f,
-10007.467773f,-10009.205078f,-10010.734375f,-10011.638672f,-10011.395508f,-10010.690430f,-10010.260742f,-10009.476562f,-10009.106445f,-10008.333984f,-10007.799805f,-10006.222656f,-10004.677734f,-10002.893555f,-10001.259766f,-9999.698242f,-9999.487305f,-10000.077148f,-10001.547852f,-10002.771484f,
-10003.767578f,-10004.026367f,-10003.538086f,-10003.180664f,-10003.160156f,-10002.982422f,-10003.232422f,-10003.368164f,-10003.009766f,-10002.095703f,-10001.133789f,-9999.923828f,-9999.126953f,-9998.790039f,-9999.502930f,-10000.768555f,-10003.191406f,-10005.411133f,-10007.146484f,-10008.274414f,
-10008.228516f,-10007.961914f,-10007.856445f,-10008.229492f,-9993.857422f,-10002.568359f,-10004.652344f,-10004.180664f,-10005.023438f,-10004.257812f,-10003.623047f,-10005.406250f,-10005.409180f,-10002.667969f,-10003.727539f,-10002.238281f,-10000.941406f,-9999.589844f,-9998.701172f,-9998.932617f,
-10000.329102f,-10001.752930f,-10003.510742f,-10004.656250f,-10005.193359f,-10005.243164f,-10005.226562f,-10005.168945f,-10005.007812f,-10005.262695f,-10004.863281f,-10004.063477f,-10002.993164f,-10001.531250f,-10000.013672f,-9999.227539f,-9998.905273f,-9999.819336f,-10000.615234f,-10001.647461f,
-10002.234375f,-10002.109375f,-10001.294922f,-10000.678711f,-9999.858398f,-9999.449219f,-9998.970703f,-9998.250977f,-9997.389648f,-9996.246094f,-9994.986328f,-9993.486328f,-9992.994141f,-9993.129883f,-9994.341797f,-9996.032227f,-9997.586914f,-9998.692383f,-9999.209961f,-9999.190430f,
-9999.427734f,-9999.490234f,-9999.692383f,-10000.166016f,-10000.208008f,-10000.404297f,-9999.993164f,-9999.364258f,-9998.553711f,-9998.567383f,-9999.243164f,-10000.727539f,-10002.709961f,-10004.602539f,-10006.197266f,-10006.876953f,-10007.034180f,-10007.379883f,-10007.341797f,-10007.662109f,
-10008.070312f,-10008.085938f,-10008.198242f,-10007.787109f,-10007.491211f,-10006.638672f,-10005.746094f,-10005.700195f,-10006.727539f,-10008.286133f,-10010.000000f,-10011.490234f,-10011.916992f,-10011.546875f,-10011.064453f,-10009.923828f,-10009.215820f,-10008.428711f,-10008.239258f,-10007.235352f,
-10006.303711f,-10005.016602f,-10003.304688f,-10001.265625f,-10000.127930f,-9999.802734f,-10000.632812f,-10001.754883f,-10003.077148f,-10003.950195f,-10003.965820f,-10003.678711f,-10003.349609f,-10002.771484f,-10002.645508f,-10002.721680f,-10002.715820f,-10002.334961f,-10001.854492f,-10000.983398f,
-10000.080078f,-9999.167969f,-9998.853516f,-9999.131836f,-10000.991211f,-10003.168945f,-10005.249023f,-10007.229492f,-10007.776367f,-10007.898438f,-10007.699219f,-10007.760742f,-9993.848633f,-10002.283203f,-10004.356445f,-10004.149414f,-10005.167969f,-10004.364258f,-10003.533203f,-10005.147461f,
-10004.983398f,-10002.425781f,-10003.890625f,-10003.225586f,-10002.350586f,-10000.958984f,-9999.482422f,-9998.796875f,-9999.392578f,-10000.391602f,-10002.177734f,-10003.814453f,-10004.928711f,-10005.404297f,-10005.480469f,-10005.393555f,-10005.107422f,-10005.429688f,-10005.432617f,-10005.192383f,
-10004.649414f,-10003.497070f,-10001.825195f,-10000.441406f,-9999.269531f,-9999.396484f,-9999.833984f,-10000.919922f,-10001.952148f,-10002.375977f,-10001.916016f,-10001.348633f,-10000.389648f,-9999.708984f,-9999.153320f,-9998.650391f,-9998.226562f,-9997.520508f,-9996.464844f,-9994.746094f,
-9993.440430f,-9992.614258f,-9992.982422f,-9994.299805f,-9995.978516f,-9997.560547f,-9998.634766f,-9998.909180f,-9999.124023f,-9999.010742f,-9999.015625f,-9999.488281f,-9999.881836f,-10000.624023f,-10000.790039f,-10000.481445f,-9999.554688f,-9998.935547f,-9998.840820f,-9999.561523f,
-10001.128906f,-10003.108398f,-10005.136719f,-10006.505859f,-10007.166992f,-10007.660156f,-10007.584961f,-10007.627930f,-10007.881836f,-10008.071289f,-10008.669922f,-10008.818359f,-10009.057617f,-10008.338867f,-10007.231445f,-10006.453125f,-10006.600586f,-10007.557617f,-10008.972656f,-10010.704102f,
-10011.773438f,-10011.979492f,-10011.805664f,-10010.584961f,-10009.578125f,-10008.552734f,-10008.380859f,-10007.703125f,-10007.264648f,-10006.558594f,-10005.098633f,-10003.028320f,-10001.243164f,-10000.025391f,-9999.886719f,-10000.438477f,-10001.656250f,-10002.930664f,-10003.572266f,-10003.696289f,
-10003.470703f,-10002.758789f,-10002.228516f,-10002.003906f,-10002.105469f,-10002.065430f,-10002.017578f,-10001.660156f,-10001.011719f,-9999.965820f,-9998.968750f,-9998.327148f,-9999.360352f,-10001.070312f,-10003.040039f,-10005.592773f,-10006.860352f,-10007.682617f,-10007.803711f,-10007.881836f,
-9993.784180f,-10001.794922f,-10003.760742f,-10003.676758f,-10004.956055f,-10004.537109f,-10003.723633f,-10005.203125f,-10004.999023f,-10002.135742f,-10003.723633f,-10003.699219f,-10003.453125f,-10002.449219f,-10000.898438f,-9999.585938f,-9999.248047f,-9999.372070f,-10000.578125f,-10002.208008f,
-10003.744141f,-10004.833008f,-10005.370117f,-10005.511719f,-10005.159180f,-10005.326172f,-10005.457031f,-10005.550781f,-10005.500000f,-10004.952148f,-10003.622070f,-10002.119141f,-10000.376953f,-9999.634766f,-9999.267578f,-9999.918945f,-10001.009766f,-10001.961914f,-10002.101562f,-10002.017578f,
-10001.267578f,-10000.484375f,-9999.644531f,-9999.126953f,-9998.902344f,-9998.586914f,-9997.966797f,-9996.573242f,-9994.988281f,-9993.443359f,-9992.831055f,-9993.216797f,-9994.414062f,-9996.093750f,-9997.655273f,-9998.481445f,-9999.056641f,-9999.033203f,-9998.888672f,-9999.108398f,
-9999.544922f,-10000.495117f,-10001.145508f,-10001.376953f,-10000.725586f,-9999.921875f,-9999.314453f,-9999.187500f,-9999.913086f,-10001.407227f,-10003.367188f,-10005.261719f,-10006.594727f,-10007.582031f,-10007.806641f,-10007.803711f,-10007.800781f,-10007.947266f,-10008.680664f,-10009.243164f,
-10010.030273f,-10009.767578f,-10008.946289f,-10007.870117f,-10007.292969f,-10007.358398f,-10007.961914f,-10009.333984f,-10010.699219f,-10011.496094f,-10011.997070f,-10011.167969f,-10010.186523f,-10008.928711f,-10008.575195f,-10007.922852f,-10007.754883f,-10007.545898f,-10006.603516f,-10005.009766f,
-10003.182617f,-10001.466797f,-10000.348633f,-9999.877930f,-10000.436523f,-10001.593750f,-10002.635742f,-10003.364258f,-10003.662109f,-10003.230469f,-10002.522461f,-10001.935547f,-10001.856445f,-10001.843750f,-10001.983398f,-10002.011719f,-10001.875000f,-10001.162109f,-10000.036133f,-9998.813477f,
-9998.905273f,-9999.650391f,-10000.912109f,-10003.385742f,-10005.058594f,-10006.677734f,-10007.456055f,-10007.971680f,-9993.760742f,-10001.897461f,-10003.553711f,-10003.292969f,-10004.667969f,-10004.786133f,-10004.240234f,-10005.671875f,-10005.375977f,-10002.333984f,-10003.821289f,-10003.924805f,
-10004.215820f,-10003.764648f,-10002.626953f,-10001.190430f,-10000.194336f,-9999.386719f,-9999.633789f,-10000.706055f,-10002.208008f,-10003.732422f,-10004.842773f,-10005.425781f,-10005.195312f,-10005.219727f,-10005.306641f,-10005.493164f,-10005.689453f,-10005.749023f,-10005.010742f,-10003.833008f,
-10002.026367f,-10000.666016f,-9999.411133f,-9999.228516f,-9999.854492f,-10000.967773f,-10001.567383f,-10002.099609f,-10001.844727f,-10001.214844f,-10000.147461f,-9999.486328f,-9999.180664f,-9998.996094f,-9998.765625f,-9997.922852f,-9996.568359f,-9994.829102f,-9993.445312f,-9992.777344f,
-9993.033203f,-9994.261719f,-9995.911133f,-9997.191406f,-9998.274414f,-9998.594727f,-9998.545898f,-9998.532227f,-9998.840820f,-9999.684570f,-10000.554688f,-10001.282227f,-10001.132812f,-10000.577148f,-9999.931641f,-9999.204102f,-9999.062500f,-9999.737305f,-10001.131836f,-10003.079102f,
-10004.911133f,-10006.537109f,-10007.303711f,-10007.589844f,-10007.506836f,-10007.593750f,-10008.265625f,-10009.007812f,-10010.190430f,-10010.493164f,-10010.298828f,-10009.456055f,-10008.619141f,-10007.961914f,-10007.604492f,-10008.125000f,-10009.250977f,-10010.292969f,-10011.407227f,-10011.240234f,
-10010.611328f,-10009.327148f,-10008.794922f,-10008.012695f,-10007.828125f,-10007.858398f,-10007.389648f,-10006.425781f,-10005.076172f,-10003.390625f,-10001.652344f,-10000.167969f,-9999.712891f,-10000.195312f,-10001.087891f,-10002.209961f,-10003.073242f,-10003.184570f,-10002.637695f,-10001.867188f,
-10001.518555f,-10001.332031f,-10001.346680f,-10001.503906f,-10001.805664f,-10001.615234f,-10000.879883f,-9999.596680f,-9999.033203f,-9998.816406f,-9999.099609f,-10000.845703f,-10002.412109f,-10004.462891f,-10005.914062f,-10007.114258f,-9993.833008f,-10002.053711f,-10003.203125f,-10002.507812f,
-10003.832031f,-10004.304688f,-10004.081055f,-10006.057617f,-10005.846680f,-10002.633789f,-10003.928711f,-10003.844727f,-10004.389648f,-10004.422852f,-10003.863281f,-10002.729492f,-10001.538086f,-10000.082031f,-9999.375000f,-9999.544922f,-10000.513672f,-10002.085938f,-10003.686523f,-10004.827148f,
-10004.976562f,-10005.075195f,-10005.181641f,-10005.299805f,-10005.500977f,-10005.908203f,-10005.725586f,-10005.095703f,-10003.635742f,-10002.106445f,-10000.219727f,-9999.126953f,-9998.929688f,-9999.697266f,-10000.430664f,-10001.506836f,-10001.887695f,-10001.717773f,-10000.692383f,-9999.976562f,
-9999.460938f,-9999.180664f,-9999.082031f,-9998.719727f,-9997.857422f,-9996.387695f,-9994.698242f,-9993.160156f,-9992.414062f,-9992.794922f,-9994.014648f,-9995.423828f,-9996.965820f,-9997.810547f,-9998.154297f,-9998.161133f,-9998.372070f,-9998.988281f,-9999.780273f,-10000.752930f,
-10001.028320f,-10000.921875f,-10000.609375f,-9999.635742f,-9998.905273f,-9998.714844f,-9999.256836f,-10000.767578f,-10002.669922f,-10004.803711f,-10006.228516f,-10007.099609f,-10007.258789f,-10007.443359f,-10008.018555f,-10008.744141f,-10010.025391f,-10010.761719f,-10011.209961f,-10010.912109f,
-10010.263672f,-10009.309570f,-10008.239258f,-10007.755859f,-10008.181641f,-10009.010742f,-10010.438477f,-10010.994141f,-10011.011719f,-10010.021484f,-10009.548828f,-10008.672852f,-10008.374023f,-10008.366211f,-10008.125000f,-10007.625977f,-10006.914062f,-10005.680664f,-10003.816406f,-10001.678711f,
-10000.242188f,-9999.725586f,-9999.951172f,-10001.011719f,-10002.233398f,-10002.964844f,-10002.961914f,-10002.312500f,-10001.868164f,-10001.516602f,-10001.243164f,-10001.238281f,-10001.702148f,-10001.918945f,-10001.742188f,-10000.768555f,-10000.017578f,-9999.159180f,-9998.549805f,-9999.270508f,
-10000.199219f,-10002.164062f,-10003.965820f,-10005.870117f,-9993.802734f,-10003.118164f,-10003.946289f,-10002.399414f,-10003.203125f,-10003.570312f,-10003.569336f,-10006.031250f,-10006.044922f,-10002.916016f,-10004.214844f,-10003.783203f,-10004.221680f,-10004.444336f,-10004.399414f,-10003.825195f,
-10002.869141f,-10001.290039f,-9999.947266f,-9999.169922f,-9999.218750f,-10000.315430f,-10001.970703f,-10003.523438f,-10004.213867f,-10004.663086f,-10005.019531f,-10005.166992f,-10005.260742f,-10005.755859f,-10005.901367f,-10005.805664f,-10004.978516f,-10003.786133f,-10001.780273f,-10000.042969f,
-9998.925781f,-9998.930664f,-9999.300781f,-10000.498047f,-10001.380859f,-10001.805664f,-10001.185547f,-10000.665039f,-10000.027344f,-9999.547852f,-9999.298828f,-9999.173828f,-9998.805664f,-9997.900391f,-9996.443359f,-9994.597656f,-9993.065430f,-9992.459961f,-9992.786133f,-9993.783203f,
-9995.387695f,-9996.660156f,-9997.579102f,-9997.910156f,-9998.227539f,-9998.707031f,-9999.297852f,-10000.244141f,-10000.764648f,-10001.159180f,-10001.404297f,-10000.632812f,-9999.833008f,-9999.108398f,-9998.771484f,-9999.490234f,-10000.904297f,-10003.115234f,-10004.968750f,-10006.453125f,
-10007.153320f,-10007.673828f,-10008.298828f,-10008.913086f,-10010.056641f,-10010.942383f,-10011.775391f,-10012.066406f,-10011.899414f,-10011.168945f,-10009.866211f,-10008.543945f,-10007.976562f,-10008.060547f,-10009.194336f,-10010.082031f,-10010.691406f,-10010.201172f,-10010.071289f,-10009.320312f,
-10008.934570f,-10008.750977f,-10008.458008f,-10008.079102f,-10007.912109f,-10007.367188f,-10005.874023f,-10003.676758f,-10001.654297f,-10000.133789f,-9999.346680f,-9999.808594f,-10000.858398f,-10001.936523f,-10002.571289f,-10002.330078f,-10002.104492f,-10001.801758f,-10001.325195f,-10001.031250f,
-10001.392578f,-10001.700195f,-10001.988281f,-10001.517578f,-10001.023438f,-10000.028320f,-9998.928711f,-9998.693359f,-9998.681641f,-9999.950195f,-10001.544922f,-10003.751953f,-9993.734375f,-10004.137695f,-10004.714844f,-10002.617188f,-10002.616211f,-10002.535156f,-10002.451172f,-10005.508789f,
-10005.785156f,-10003.125977f,-10004.666016f,-10003.959961f,-10004.130859f,-10004.245117f,-10004.386719f,-10004.278320f,-10003.800781f,-10002.583008f,-10001.117188f,-9999.730469f,-9998.823242f,-9999.087891f,-10000.326172f,-10001.953125f,-10003.067383f,-10004.062500f,-10004.885742f,-10005.239258f,
-10005.281250f,-10005.618164f,-10005.752930f,-10005.918945f,-10005.656250f,-10005.060547f,-10003.432617f,-10001.526367f,-9999.754883f,-9998.880859f,-9998.565430f,-9999.409180f,-10000.445312f,-10001.369141f,-10001.406250f,-10001.348633f,-10000.828125f,-10000.258789f,-9999.703125f,-9999.434570f,
-9999.250000f,-9998.874023f,-9997.942383f,-9996.325195f,-9994.483398f,-9993.097656f,-9992.426758f,-9992.613281f,-9993.796875f,-9995.177734f,-9996.574219f,-9997.428711f,-9998.085938f,-9998.658203f,-9999.055664f,-9999.723633f,-10000.177734f,-10000.737305f,-10001.397461f,-10000.975586f,
-10000.494141f,-9999.668945f,-9998.751953f,-9998.642578f,-9999.244141f,-10001.027344f,-10002.920898f,-10004.831055f,-10006.173828f,-10007.247070f,-10008.170898f,-10008.803711f,-10009.761719f,-10010.540039f,-10011.378906f,-10012.032227f,-10012.366211f,-10012.246094f,-10011.245117f,-10009.606445f,
-10008.290039f,-10007.507812f,-10007.977539f,-10008.765625f,-10009.708984f,-10009.803711f,-10010.286133f,-10009.973633f,-10009.801758f,-10009.528320f,-10009.019531f,-10008.408203f,-10008.445312f,-10008.441406f,-10007.524414f,-10005.797852f,-10003.713867f,-10001.601562f,-9999.887695f,-9999.447266f,
-9999.842773f,-10000.804688f,-10001.897461f,-10002.162109f,-10002.438477f,-10002.480469f,-10002.034180f,-10001.517578f,-10001.543945f,-10001.583008f,-10001.943359f,-10001.789062f,-10001.725586f,-10001.041992f,-9999.986328f,-9999.198242f,-9998.306641f,-9998.607422f,-9999.500977f,-10001.510742f,
-9993.867188f,-10005.184570f,-10005.851562f,-10003.524414f,-10002.804688f,-10002.055664f,-10001.605469f,-10004.588867f,-10004.986328f,-10002.963867f,-10004.932617f,-10004.233398f,-10004.208984f,-10004.114258f,-10004.122070f,-10004.214844f,-10004.120117f,-10003.456055f,-10002.294922f,-10000.814453f,
-9999.268555f,-9998.706055f,-9999.159180f,-10000.401367f,-10001.592773f,-10003.011719f,-10004.350586f,-10005.098633f,-10005.308594f,-10005.511719f,-10005.486328f,-10005.611328f,-10005.633789f,-10005.559570f,-10004.504883f,-10002.882812f,-10000.913086f,-9999.414062f,-9998.366211f,-9998.485352f,
-9999.205078f,-10000.241211f,-10000.823242f,-10001.335938f,-10001.180664f,-10000.775391f,-10000.067383f,-9999.533203f,-9999.215820f,-9999.075195f,-9998.668945f,-9997.573242f,-9995.917969f,-9994.228516f,-9992.813477f,-9992.153320f,-9992.537109f,-9993.588867f,-9995.097656f,-9996.371094f,
-9997.451172f,-9998.384766f,-9998.854492f,-9999.311523f,-9999.627930f,-10000.121094f,-10000.939453f,-10000.838867f,-10000.837891f,-10000.302734f,-9999.293945f,-9998.635742f,-9998.427734f,-9999.500000f,-10000.989258f,-10002.885742f,-10004.654297f,-10006.256836f,-10007.649414f,-10008.513672f,
-10009.462891f,-10010.106445f,-10010.727539f,-10011.426758f,-10012.026367f,-10012.506836f,-10012.073242f,-10010.638672f,-10008.992188f,-10007.559570f,-10007.159180f,-10007.375000f,-10008.138672f,-10008.517578f,-10009.489258f,-10009.723633f,-10009.978516f,-10009.852539f,-10009.234375f,-10008.324219f,
-10008.264648f,-10008.452148f,-10007.989258f,-10006.902344f,-10005.176758f,-10003.052734f,-10000.830078f,-9999.581055f,-9999.059570f,-9999.453125f,-10000.526367f,-10001.100586f,-10001.886719f,-10002.437500f,-10002.330078f,-10001.872070f,-10001.708008f,-10001.370117f,-10001.505859f,-10001.396484f,
-10001.627930f,-10001.389648f,-10000.759766f,-9999.949219f,-9998.609375f,-9998.066406f,-9998.093750f,-9999.450195f,-9993.764648f,-10006.101562f,-10007.045898f,-10004.930664f,-10003.708984f,-10002.307617f,-10001.208008f,-10003.600586f,-10003.791016f,-10002.403320f,-10004.838867f,-10004.286133f,
-10004.370117f,-10004.211914f,-10003.913086f,-10003.875977f,-10003.880859f,-10003.688477f,-10003.047852f,-10001.958984f,-10000.248047f,-9999.062500f,-9998.592773f,-9999.102539f,-9999.896484f,-10001.406250f,-10003.172852f,-10004.492188f,-10005.193359f,-10005.551758f,-10005.469727f,-10005.388672f,
-10005.384766f,-10005.590820f,-10005.072266f,-10004.047852f,-10002.346680f,-10000.622070f,-9999.005859f,-9998.300781f,-9998.303711f,-9998.958008f,-9999.712891f,-10000.716797f,-10001.112305f,-10001.203125f,-10000.681641f,-10000.005859f,-9999.301758f,-9999.058594f,-9998.900391f,-9998.307617f,
-9997.139648f,-9995.616211f,-9993.896484f,-9992.508789f,-9991.936523f,-9992.217773f,-9993.311523f,-9994.642578f,-9996.063477f,-9997.528320f,-9998.387695f,-9998.911133f,-9999.172852f,-9999.439453f,-10000.125000f,-10000.152344f,-10000.541016f,-10000.500000f,-9999.805664f,-9999.046875f,
-9998.280273f,-9998.556641f,-9999.317383f,-10000.739258f,-10002.532227f,-10004.440430f,-10006.371094f,-10007.712891f,-10009.005859f,-10009.756836f,-10010.158203f,-10010.674805f,-10011.258789f,-10012.147461f,-10012.295898f,-10011.493164f,-10009.988281f,-10008.373047f,-10007.236328f,-10006.630859f,
-10006.763672f,-10006.999023f,-10008.149414f,-10008.890625f,-10009.752930f,-10010.098633f,-10009.705078f,-10008.711914f,-10008.403320f,-10008.458984f,-10008.154297f,-10007.545898f,-10006.375000f,-10004.704102f,-10002.505859f,-10000.731445f,-9999.318359f,-9998.813477f,-9999.301758f,-9999.722656f,
-10000.791992f,-10001.844727f,-10002.304688f,-10002.237305f,-10002.245117f,-10001.638672f,-10001.381836f,-10000.997070f,-10001.188477f,-10001.241211f,-10001.124023f,-10000.676758f,-9999.382812f,-9998.364258f,-9997.589844f,-9998.039062f,-9993.773438f,-10005.912109f,-10007.320312f,-10005.635742f,
-10004.625000f,-10002.915039f,-10001.176758f,-10002.596680f,-10002.456055f,-10001.248047f,-10003.730469f,-10003.583984f,-10004.073242f,-10004.208008f,-10003.822266f,-10003.628906f,-10003.513672f,-10003.504883f,-10003.297852f,-10002.829102f,-10001.465820f,-10000.200195f,-9999.117188f,-9998.849609f,
-9998.882812f,-9999.994141f,-10001.729492f,-10003.375000f,-10004.633789f,-10005.415039f,-10005.587891f,-10005.451172f,-10005.316406f,-10005.524414f,-10005.241211f,-10004.790039f,-10003.609375f,-10002.151367f,-10000.409180f,-9999.160156f,-9998.347656f,-9998.274414f,-9998.631836f,-9999.676758f,
-10000.433594f,-10001.081055f,-10001.091797f,-10000.632812f,-9999.733398f,-9999.277344f,-9999.072266f,-9998.696289f,-9997.978516f,-9996.933594f,-9995.463867f,-9993.876953f,-9992.616211f,-9992.028320f,-9992.294922f,-9993.150391f,-9994.485352f,-9996.243164f,-9997.554688f,-9998.429688f,
-9998.879883f,-9999.062500f,-9999.518555f,-9999.505859f,-9999.971680f,-10000.271484f,-10000.034180f,-9999.556641f,-9998.738281f,-9998.485352f,-9998.501953f,-9999.157227f,-10000.455078f,-10002.171875f,-10004.262695f,-10006.023438f,-10007.794922f,-10008.934570f,-10009.432617f,-10009.877930f,
-10010.328125f,-10011.289062f,-10011.758789f,-10011.625977f,-10010.640625f,-10009.390625f,-10008.088867f,-10006.913086f,-10006.277344f,-10005.967773f,-10006.814453f,-10007.635742f,-10008.915039f,-10009.752930f,-10009.895508f,-10009.244141f,-10008.907227f,-10008.758789f,-10008.396484f,-10007.877930f,
-10007.118164f,-10006.046875f,-10004.321289f,-10002.595703f,-10000.775391f,-9999.450195f,-9999.012695f,-9998.829102f,-9999.682617f,-10000.830078f,-10001.725586f,-10002.149414f,-10002.661133f,-10002.203125f,-10001.826172f,-10001.183594f,-10001.145508f,-10001.147461f,-10001.325195f,-10001.251953f,
-10000.380859f,-9999.398438f,-9998.287109f,-9998.007812f,-9993.772461f,-10005.634766f,-10007.541016f,-10006.379883f,-10005.903320f,-10004.232422f,-10002.130859f,-10002.394531f,-10001.745117f,-10000.339844f,-10002.569336f,-10002.565430f,-10003.473633f,-10003.987305f,-10003.649414f,-10003.308594f,
-10002.952148f,-10002.853516f,-10002.905273f,-10003.040039f,-10002.290039f,-10001.321289f,-9999.984375f,-9999.066406f,-9998.291016f,-9998.734375f,-10000.058594f,-10001.756836f,-10003.442383f,-10004.717773f,-10005.329102f,-10005.261719f,-10005.016602f,-10005.078125f,-10004.795898f,-10004.798828f,
-10004.208984f,-10003.324219f,-10001.812500f,-10000.378906f,-9998.876953f,-9998.005859f,-9997.635742f,-9998.349609f,-9999.182617f,-10000.213867f,-10000.875000f,-10000.819336f,-9999.899414f,-9999.243164f,-9998.818359f,-9998.419922f,-9998.050781f,-9997.575195f,-9996.648438f,-9995.265625f,
-9993.668945f,-9992.336914f,-9991.721680f,-9991.799805f,-9992.683594f,-9994.445312f,-9996.105469f,-9997.459961f,-9998.239258f,-9998.507812f,-9998.811523f,-9998.753906f,-9999.141602f,-9999.667969f,-9999.958984f,-9999.969727f,-9999.500977f,-9999.127930f,-9998.603516f,-9998.527344f,
-9999.125977f,-10000.328125f,-10002.252930f,-10004.253906f,-10006.479492f,-10008.129883f,-10008.900391f,-10009.373047f,-10009.709961f,-10010.516602f,-10011.056641f,-10011.475586f,-10011.067383f,-10010.481445f,-10009.370117f,-10007.856445f,-10006.520508f,-10005.486328f,-10005.698242f,-10006.218750f,
-10007.595703f,-10008.791992f,-10009.511719f,-10009.412109f,-10009.264648f,-10008.939453f,-10008.428711f,-10007.725586f,-10007.242188f,-10006.728516f,-10005.654297f,-10004.361328f,-10002.536133f,-10000.674805f,-9999.309570f,-9998.340820f,-9998.613281f,-9999.488281f,-10000.528320f,-10001.360352f,
-10002.462891f,-10002.362305f,-10002.128906f,-10001.304688f,-10000.970703f,-10000.767578f,-10001.022461f,-10001.265625f,-10000.938477f,-10000.361328f,-9999.299805f,-9998.583984f,-9993.713867f,-10004.987305f,-10007.221680f,-10006.416992f,-10006.779297f,-10005.331055f,-10003.253906f,-10002.984375f,
-10001.829102f,-9999.737305f,-10001.489258f,-10001.262695f,-10002.492188f,-10003.539062f,-10003.615234f,-10003.425781f,-10002.955078f,-10002.583008f,-10002.563477f,-10002.956055f,-10002.732422f,-10002.262695f,-10001.125977f,-9999.977539f,-9998.615234f,-9998.339844f,-9998.985352f,-10000.348633f,
-10002.133789f,-10003.871094f,-10005.055664f,-10005.397461f,-10005.325195f,-10005.283203f,-10004.793945f,-10004.885742f,-10004.616211f,-10004.238281f,-10003.137695f,-10001.936523f,-10000.147461f,-9998.719727f,-9997.518555f,-9997.601562f,-9998.104492f,-9999.178711f,-10000.347656f,-10000.868164f,
-10000.317383f,-9999.750977f,-9999.116211f,-9998.420898f,-9998.019531f,-9997.847656f,-9997.415039f,-9996.472656f,-9995.017578f,-9993.344727f,-9992.034180f,-9991.299805f,-9991.451172f,-9992.786133f,-9994.432617f,-9996.200195f,-9997.398438f,-9998.049805f,-9998.442383f,-9998.400391f,
-9998.542969f,-9998.955078f,-9999.436523f,-9999.809570f,-9999.831055f,-9999.702148f,-9999.027344f,-9998.527344f,-9998.462891f,-9998.946289f,-10000.339844f,-10002.223633f,-10004.672852f,-10006.841797f,-10008.160156f,-10009.006836f,-10009.474609f,-10010.096680f,-10010.455078f,-10011.029297f,
-10010.984375f,-10011.027344f,-10010.486328f,-10009.122070f,-10007.555664f,-10005.967773f,-10005.500977f,-10005.409180f,-10006.495117f,-10007.724609f,-10008.897461f,-10009.481445f,-10009.845703f,-10009.683594f,-10009.147461f,-10008.077148f,-10007.536133f,-10007.166016f,-10006.519531f,-10005.752930f,
-10004.341797f,-10002.398438f,-10000.462891f,-9998.802734f,-9998.296875f,-9998.528320f,-9999.292969f,-10000.169922f,-10001.759766f,-10002.193359f,-10002.491211f,-10001.854492f,-10001.385742f,-10000.849609f,-10000.866211f,-10001.016602f,-10000.992188f,-10000.848633f,-10000.125000f,-9999.442383f,
-9993.760742f,-10004.397461f,-10007.044922f,-10006.200195f,-10007.304688f,-10006.220703f,-10004.392578f,-10004.057617f,-10002.750977f,-9999.753906f,-10000.880859f,-9999.996094f,-10001.085938f,-10002.407227f,-10002.973633f,-10003.212891f,-10002.951172f,-10002.485352f,-10002.386719f,-10002.764648f,
-10002.848633f,-10002.798828f,-10002.033203f,-10000.976562f,-9999.376953f,-9998.574219f,-9998.454102f,-9999.157227f,-10000.576172f,-10002.407227f,-10003.990234f,-10004.870117f,-10005.282227f,-10005.473633f,-10004.971680f,-10005.101562f,-10004.988281f,-10004.924805f,-10004.208984f,-10003.412109f,
-10001.667969f,-10000.052734f,-9998.245117f,-9997.602539f,-9997.466797f,-9998.127930f,-9999.396484f,-10000.291992f,-10000.335938f,-10000.192383f,-9999.693359f,-9998.884766f,-9998.385742f,-9998.270508f,-9998.099609f,-9997.558594f,-9996.472656f,-9994.827148f,-9993.189453f,-9991.813477f,
-9991.148438f,-9991.693359f,-9992.859375f,-9994.627930f,-9996.063477f,-9997.192383f,-9997.904297f,-9998.193359f,-9998.316406f,-9998.634766f,-9999.157227f,-9999.664062f,-10000.046875f,-10000.281250f,-9999.722656f,-9999.176758f,-9998.695312f,-9998.504883f,-9999.105469f,-10000.462891f,
-10002.667969f,-10005.000977f,-10006.745117f,-10008.123047f,-10009.060547f,-10009.759766f,-10010.047852f,-10010.646484f,-10010.728516f,-10011.120117f,-10011.131836f,-10010.136719f,-10008.735352f,-10006.958984f,-10005.994141f,-10005.214844f,-10005.666992f,-10006.496094f,-10007.713867f,-10008.721680f,
-10009.670898f,-10009.988281f,-10009.824219f,-10008.690430f,-10008.132812f,-10007.716797f,-10007.208008f,-10006.819336f,-10005.916992f,-10004.155273f,-10002.112305f,-10000.120117f,-9998.899414f,-9998.307617f,-9998.431641f,-9998.893555f,-10000.521484f,-10001.303711f,-10002.228516f,-10002.094727f,
-10001.886719f,-10001.305664f,-10001.169922f,-10001.046875f,-10000.989258f,-10001.039062f,-10000.629883f,-10000.189453f,-9993.666992f,-10004.013672f,-10006.645508f,-10005.717773f,-10007.335938f,-10006.696289f,-10005.351562f,-10005.415039f,-10003.972656f,-10000.482422f,-10000.981445f,-9999.544922f,
-10000.141602f,-10001.404297f,-10002.395508f,-10003.151367f,-10003.311523f,-10002.846680f,-10002.600586f,-10002.686523f,-10002.796875f,-10003.022461f,-10002.738281f,-10002.062500f,-10000.615234f,-9999.623047f,-9998.890625f,-9998.808594f,-9999.526367f,-10001.066406f,-10002.762695f,-10004.127930f,
-10005.145508f,-10005.745117f,-10005.349609f,-10005.388672f,-10005.232422f,-10005.275391f,-10004.883789f,-10004.597656f,-10003.255859f,-10001.886719f,-9999.875000f,-9998.655273f,-9997.792969f,-9997.750000f,-9998.723633f,-9999.707031f,-10000.358398f,-10000.817383f,-10000.644531f,-9999.830078f,
-9999.149414f,-9998.864258f,-9998.687500f,-9998.458984f,-9997.893555f,-9996.609375f,-9995.096680f,-9993.455078f,-9992.109375f,-9991.786133f,-9992.125000f,-9993.514648f,-9994.879883f,-9996.407227f,-9997.533203f,-9998.247070f,-9998.462891f,-9998.636719f,-9999.015625f,-9999.437500f,
-10000.002930f,-10000.667969f,-10000.472656f,-10000.246094f,-9999.720703f,-9999.088867f,-9998.916992f,-9999.531250f,-10001.131836f,-10003.241211f,-10005.178711f,-10007.084961f,-10008.612305f,-10009.528320f,-10009.789062f,-10010.246094f,-10010.216797f,-10010.689453f,-10011.135742f,-10010.684570f,
-10009.841797f,-10008.292969f,-10007.195312f,-10005.878906f,-10005.626953f,-10005.794922f,-10006.664062f,-10007.767578f,-10009.218750f,-10010.160156f,-10010.570312f,-10009.607422f,-10009.018555f,-10008.385742f,-10007.750000f,-10007.552734f,-10007.164062f,-10005.884766f,-10004.180664f,-10002.310547f,
-10000.657227f,-9999.334961f,-9998.653320f,-9998.390625f,-9999.622070f,-10000.458008f,-10001.921875f,-10002.409180f,-10002.691406f,-10002.264648f,-10001.990234f,-10001.486328f,-10001.123047f,-10001.140625f,-10000.989258f,-10000.966797f,-9993.684570f,-10003.795898f,-10006.319336f,-10005.385742f,
-10006.887695f,-10006.748047f,-10005.836914f,-10006.546875f,-10005.598633f,-10001.784180f,-10002.041992f,-9999.911133f,-9999.689453f,-10000.429688f,-10001.455078f,-10002.578125f,-10003.338867f,-10003.195312f,-10003.076172f,-10002.920898f,-10002.858398f,-10003.062500f,-10003.067383f,-10002.777344f,
-10001.740234f,-10000.950195f,-9999.938477f,-9999.223633f,-9999.100586f,-9999.960938f,-10001.284180f,-10002.817383f,-10004.348633f,-10005.559570f,-10005.622070f,-10005.791992f,-10005.638672f,-10005.593750f,-10005.309570f,-10005.311523f,-10004.423828f,-10003.529297f,-10001.778320f,-10000.305664f,
-9998.875977f,-9998.041016f,-9998.286133f,-9998.847656f,-9999.772461f,-10000.755859f,-10001.160156f,-10000.712891f,-10000.069336f,-9999.613281f,-9999.202148f,-9998.974609f,-9998.715820f,-9997.860352f,-9996.760742f,-9995.251953f,-9993.579102f,-9992.523438f,-9991.931641f,-9992.503906f,
-9993.379883f,-9994.946289f,-9996.299805f,-9997.545898f,-9998.155273f,-9998.463867f,-9998.795898f,-9999.023438f,-9999.513672f,-10000.380859f,-10000.508789f,-10000.769531f,-10000.551758f,-9999.867188f,-9999.210938f,-9999.090820f,-9999.879883f,-10001.322266f,-10003.000000f,-10005.144531f,
-10007.246094f,-10008.683594f,-10009.257812f,-10009.774414f,-10009.665039f,-10009.974609f,-10010.552734f,-10010.493164f,-10010.285156f,-10009.325195f,-10008.519531f,-10007.071289f,-10006.290039f,-10005.733398f,-10005.901367f,-10006.567383f,-10008.127930f,-10009.533203f,-10010.671875f,-10010.301758f,
-10009.979492f,-10009.315430f,-10008.423828f,-10008.124023f,-10007.936523f,-10007.020508f,-10005.862305f,-10004.500977f,-10002.826172f,-10001.073242f,-9999.688477f,-9998.576172f,-9998.944336f,-9999.363281f,-10000.896484f,-10001.830078f,-10002.761719f,-10002.868164f,-10002.818359f,-10002.182617f,
-10001.416016f,-10001.145508f,-10000.964844f,-10001.158203f,-9993.858398f,-10003.990234f,-10006.324219f,-10005.354492f,-10006.731445f,-10006.574219f,-10005.941406f,-10006.978516f,-10006.587891f,-10003.092773f,-10003.588867f,-10000.947266f,-9999.957031f,-9999.917969f,-10000.500977f,-10001.580078f,
-10002.772461f,-10003.107422f,-10003.344727f,-10003.179688f,-10002.936523f,-10002.887695f,-10002.858398f,-10002.759766f,-10002.113281f,-10001.764648f,-10000.884766f,-9999.943359f,-9999.228516f,-9999.299805f,-9999.922852f,-10001.147461f,-10002.789062f,-10004.464844f,-10005.116211f,-10005.648438f,
-10005.638672f,-10005.503906f,-10005.151367f,-10005.150391f,-10004.548828f,-10004.117188f,-10002.933594f,-10001.648438f,-10000.107422f,-9998.789062f,-9998.277344f,-9998.123047f,-9998.809570f,-9999.937500f,-10000.836914f,-10000.960938f,-10000.656250f,-10000.246094f,-9999.580078f,-9999.124023f,
-9998.842773f,-9998.257812f,-9997.593750f,-9996.538086f,-9994.983398f,-9993.703125f,-9992.458008f,-9992.215820f,-9992.297852f,-9993.467773f,-9994.709961f,-9996.256836f,-9997.339844f,-9998.048828f,-9998.530273f,-9998.655273f,-9998.882812f,-9999.675781f,-9999.919922f,-10000.558594f,
-10000.798828f,-10000.427734f,-9999.761719f,-9999.264648f,-9999.407227f,-10000.026367f,-10001.044922f,-10002.946289f,-10005.249023f,-10007.180664f,-10008.249023f,-10009.119141f,-10009.069336f,-10009.168945f,-10009.562500f,-10009.539062f,-10009.732422f,-10009.400391f,-10009.167969f,-10008.031250f,
-10007.161133f,-10006.216797f,-10005.674805f,-10005.611328f,-10006.790039f,-10008.242188f,-10009.860352f,-10010.219727f,-10010.430664f,-10010.037109f,-10009.080078f,-10008.526367f,-10008.219727f,-10007.418945f,-10006.723633f,-10005.985352f,-10004.684570f,-10003.036133f,-10001.385742f,-9999.720703f,
-9999.202148f,-9998.911133f,-9999.952148f,-10000.902344f,-10002.249023f,-10003.005859f,-10003.411133f,-10003.079102f,-10002.105469f,-10001.529297f,-10001.061523f,-10001.173828f,-9993.981445f,-10003.897461f,-10006.014648f,-10005.375000f,-10006.569336f,-10006.364258f,-10005.873047f,-10007.269531f,
-10007.314453f,-10003.976562f,-10004.886719f,-10002.154297f,-10000.656250f,-9999.886719f,-9999.784180f,-10000.438477f,-10001.726562f,-10002.368164f,-10003.037109f,-10003.171875f,-10003.040039f,-10002.909180f,-10002.826172f,-10002.825195f,-10002.426758f,-10002.477539f,-10001.881836f,-10001.006836f,
-9999.996094f,-9999.498047f,-9999.397461f,-10000.073242f,-10001.452148f,-10003.258789f,-10004.367188f,-10005.335938f,-10005.719727f,-10005.723633f,-10005.424805f,-10005.386719f,-10004.891602f,-10004.677734f,-10003.945312f,-10002.914062f,-10001.500977f,-9999.960938f,-9998.862305f,-9997.934570f,
-9998.039062f,-9998.900391f,-9999.944336f,-10000.513672f,-10000.650391f,-10000.518555f,-9999.872070f,-9999.308594f,-9998.939453f,-9998.447266f,-9998.022461f,-9997.333008f,-9996.054688f,-9994.824219f,-9993.279297f,-9992.383789f,-9991.755859f,-9992.248047f,-9993.062500f,-9994.613281f,
-9996.018555f,-9997.180664f,-9998.079102f,-9998.404297f,-9998.593750f,-9999.290039f,-9999.562500f,-10000.357422f,-10000.902344f,-10000.862305f,-10000.432617f,-9999.832031f,-9999.628906f,-9999.582031f,-9999.873047f,-10001.218750f,-10003.364258f,-10005.546875f,-10007.027344f,-10008.409180f,
-10008.738281f,-10008.887695f,-10009.208008f,-10009.133789f,-10009.477539f,-10009.565430f,-10009.826172f,-10009.116211f,-10008.383789f,-10007.347656f,-10006.274414f,-10005.531250f,-10006.042969f,-10007.126953f,-10008.831055f,-10009.663086f,-10010.390625f,-10010.511719f,-10009.766602f,-10009.129883f,
-10008.713867f,-10007.865234f,-10007.340820f,-10006.977539f,-10006.043945f,-10004.643555f,-10002.966797f,-10001.039062f,-9999.848633f,-9998.851562f,-9999.103516f,-9999.680664f,-10001.099609f,-10002.290039f,-10003.159180f,-10003.390625f,-10002.570312f,-10001.997070f,-10001.358398f,-10001.319336f,
-9993.845703f,-10003.815430f,-10005.777344f,-10005.443359f,-10006.522461f,-10006.258789f,-10005.626953f,-10006.833984f,-10007.388672f,-10004.297852f,-10005.843750f,-10003.752930f,-10002.199219f,-10000.936523f,-10000.028320f,-9999.863281f,-10000.765625f,-10001.459961f,-10002.466797f,-10003.081055f,
-10003.265625f,-10003.180664f,-10002.958008f,-10002.876953f,-10002.530273f,-10002.915039f,-10002.853516f,-10002.441406f,-10001.635742f,-10000.946289f,-10000.244141f,-10000.159180f,-10000.851562f,-10002.317383f,-10003.578125f,-10004.953125f,-10005.821289f,-10006.147461f,-10005.988281f,-10005.818359f,
-10005.232422f,-10005.025391f,-10004.627930f,-10004.052734f,-10003.122070f,-10001.823242f,-10000.491211f,-9998.952148f,-9998.176758f,-9998.283203f,-9999.008789f,-9999.735352f,-10000.273438f,-10000.613281f,-10000.210938f,-9999.604492f,-9999.026367f,-9998.398438f,-9997.990234f,-9997.611328f,
-9996.826172f,-9996.063477f,-9994.722656f,-9993.583984f,-9992.360352f,-9992.044922f,-9992.062500f,-9993.147461f,-9994.546875f,-9996.080078f,-9997.456055f,-9998.228516f,-9998.528320f,-9999.088867f,-9999.216797f,-9999.892578f,-10000.577148f,-10000.937500f,-10001.046875f,-10000.800781f,
-10000.709961f,-10000.277344f,-9999.914062f,-10000.458984f,-10001.921875f,-10003.959961f,-10005.604492f,-10007.516602f,-10008.429688f,-10008.890625f,-10009.186523f,-10008.918945f,-10009.125000f,-10009.377930f,-10010.057617f,-10009.949219f,-10009.749023f,-10009.085938f,-10007.874023f,-10006.650391f,
-10006.326172f,-10006.641602f,-10007.879883f,-10008.765625f,-10009.860352f,-10010.592773f,-10010.345703f,-10009.799805f,-10009.228516f,-10008.168945f,-10007.476562f,-10007.264648f,-10006.737305f,-10005.855469f,-10004.568359f,-10002.847656f,-10001.400391f,-9999.824219f,-9999.131836f,-9998.885742f,
-9999.845703f,-10001.076172f,-10002.231445f,-10003.185547f,-10002.861328f,-10002.561523f,-10001.862305f,-10001.560547f,-9993.620117f,-10003.423828f,-10005.422852f,-10005.402344f,-10006.799805f,-10006.354492f,-10005.654297f,-10006.649414f,-10007.208984f,-10004.305664f,-10006.399414f,-10004.893555f,
-10003.638672f,-10002.248047f,-10000.750000f,-9999.760742f,-10000.019531f,-10000.418945f,-10001.496094f,-10002.583984f,-10003.234375f,-10003.442383f,-10003.207031f,-10002.996094f,-10002.494141f,-10002.933594f,-10003.166016f,-10003.162109f,-10002.722656f,-10002.173828f,-10001.232422f,-10000.587891f,
-10000.550781f,-10001.420898f,-10002.512695f,-10004.119141f,-10005.510742f,-10006.367188f,-10006.583008f,-10006.480469f,-10005.804688f,-10005.463867f,-10005.114258f,-10004.824219f,-10004.352539f,-10003.462891f,-10002.306641f,-10000.552734f,-9999.081055f,-9998.395508f,-9998.490234f,-9999.029297f,
-9999.784180f,-10000.626953f,-10000.686523f,-10000.315430f,-9999.663086f,-9998.816406f,-9998.223633f,-9997.842773f,-9997.317383f,-9996.935547f,-9995.947266f,-9994.880859f,-9993.372070f,-9992.353516f,-9991.610352f,-9991.996094f,-9993.016602f,-9994.641602f,-9996.420898f,-9997.764648f,
-9998.406250f,-9998.954102f,-9998.975586f,-9999.391602f,-9999.932617f,-10000.451172f,-10000.910156f,-10001.084961f,-10001.319336f,-10000.837891f,-10000.134766f,-9999.961914f,-10000.646484f,-10002.187500f,-10003.656250f,-10005.941406f,-10007.521484f,-10008.547852f,-10009.122070f,-10008.821289f,
-10008.818359f,-10009.023438f,-10009.855469f,-10010.141602f,-10010.450195f,-10010.331055f,-10009.320312f,-10008.069336f,-10007.220703f,-10006.790039f,-10007.315430f,-10007.902344f,-10009.104492f,-10010.395508f,-10010.815430f,-10010.667969f,-10010.211914f,-10009.041992f,-10008.030273f,-10007.692383f,
-10007.337891f,-10006.825195f,-10005.907227f,-10004.585938f,-10003.237305f,-10001.452148f,-10000.076172f,-9998.993164f,-9999.221680f,-10000.078125f,-10001.195312f,-10002.695312f,-10002.960938f,-10003.230469f,-10002.697266f,-10002.269531f,-9993.676758f,-10003.018555f,-10004.764648f,-10004.794922f,
-10006.192383f,-10006.066406f,-10005.362305f,-10006.574219f,-10007.069336f,-10004.213867f,-10006.755859f,-10005.803711f,-10005.207031f,-10004.154297f,-10002.507812f,-10000.884766f,-10000.291992f,-9999.970703f,-10000.513672f,-10001.571289f,-10002.409180f,-10002.960938f,-10002.977539f,-10002.894531f,
-10002.364258f,-10002.771484f,-10003.271484f,-10003.700195f,-10003.750000f,-10003.678711f,-10002.893555f,-10001.992188f,-10001.326172f,-10001.339844f,-10001.735352f,-10002.952148f,-10004.319336f,-10005.510742f,-10006.150391f,-10006.375977f,-10005.929688f,-10005.614258f,-10005.263672f,-10005.241211f,
-10005.201172f,-10004.836914f,-10004.072266f,-10002.619141f,-10000.795898f,-9999.416016f,-9998.632812f,-9998.406250f,-9998.766602f,-9999.688477f,-10000.059570f,-10000.041992f,-9999.605469f,-9998.817383f,-9998.207031f,-9997.778320f,-9997.527344f,-9997.486328f,-9997.000977f,-9996.326172f,
-9994.942383f,-9993.575195f,-9992.211914f,-9991.685547f,-9991.833984f,-9992.962891f,-9994.562500f,-9996.208984f,-9997.230469f,-9998.009766f,-9998.203125f,-9998.553711f,-9998.964844f,-9999.622070f,-10000.394531f,-10001.048828f,-10001.791992f,-10001.620117f,-10001.025391f,-10000.442383f,
-10000.318359f,-10001.021484f,-10001.770508f,-10003.743164f,-10005.535156f,-10007.022461f,-10008.037109f,-10008.037109f,-10008.092773f,-10008.270508f,-10009.149414f,-10009.743164f,-10010.521484f,-10011.015625f,-10010.534180f,-10009.645508f,-10008.672852f,-10007.708984f,-10007.299805f,-10007.066406f,
-10007.716797f,-10008.974609f,-10009.754883f,-10010.031250f,-10009.953125f,-10009.036133f,-10007.954102f,-10007.484375f,-10007.242188f,-10007.049805f,-10006.516602f,-10005.691406f,-10004.779297f,-10003.250000f,-10001.596680f,-9999.895508f,-9999.173828f,-9999.170898f,-9999.666992f,-10001.078125f,
-10001.707031f,-10002.527344f,-10002.460938f,-10002.291016f,-9993.506836f,-10002.570312f,-10003.969727f,-10003.943359f,-10005.544922f,-10005.905273f,-10005.401367f,-10006.532227f,-10007.058594f,-10004.109375f,-10006.698242f,-10006.226562f,-10006.287109f,-10005.830078f,-10004.536133f,-10002.744141f,
-10001.542969f,-10000.420898f,-10000.185547f,-10000.862305f,-10001.723633f,-10002.654297f,-10003.093750f,-10003.249023f,-10002.666016f,-10002.829102f,-10003.265625f,-10003.752930f,-10004.086914f,-10004.528320f,-10004.214844f,-10003.481445f,-10002.596680f,-10001.930664f,-10001.542969f,-10002.102539f,
-10003.167969f,-10004.546875f,-10005.640625f,-10006.394531f,-10006.343750f,-10006.136719f,-10005.609375f,-10005.586914f,-10005.663086f,-10005.625000f,-10005.369141f,-10004.524414f,-10002.892578f,-10001.293945f,-9999.799805f,-9998.710938f,-9998.375977f,-9999.083984f,-9999.648438f,-10000.078125f,
-10000.093750f,-9999.505859f,-9998.892578f,-9998.229492f,-9997.918945f,-9997.861328f,-9997.636719f,-9997.375000f,-9996.358398f,-9995.072266f,-9993.491211f,-9992.234375f,-9991.477539f,-9991.793945f,-9992.865234f,-9994.537109f,-9995.938477f,-9997.085938f,-9997.577148f,-9997.986328f,
-9998.161133f,-9998.709961f,-9999.416016f,-10000.289062f,-10001.383789f,-10001.649414f,-10001.516602f,-10001.015625f,-10000.455078f,-10000.367188f,-10000.260742f,-10001.580078f,-10003.285156f,-10005.118164f,-10006.728516f,-10007.312500f,-10007.615234f,-10007.767578f,-10008.502930f,-10009.089844f,
-10010.012695f,-10010.877930f,-10010.963867f,-10010.743164f,-10010.142578f,-10009.091797f,-10008.073242f,-10007.001953f,-10006.866211f,-10007.736328f,-10008.624023f,-10009.370117f,-10009.871094f,-10009.434570f,-10008.442383f,-10007.840820f,-10007.455078f,-10007.266602f,-10006.816406f,-10006.291992f,
-10005.874023f,-10004.896484f,-10003.530273f,-10001.695312f,-10000.272461f,-9999.373047f,-9999.011719f,-9999.848633f,-10000.526367f,-10001.769531f,-10002.310547f,-10002.624023f,-9993.573242f,-10002.499023f,-10003.678711f,-10003.095703f,-10004.788086f,-10005.458008f,-10005.303711f,-10006.797852f,
-10007.248047f,-10004.166016f,-10006.716797f,-10006.161133f,-10006.652344f,-10006.783203f,-10006.095703f,-10004.583008f,-10003.132812f,-10001.376953f,-10000.241211f,-10000.146484f,-10000.595703f,-10001.639648f,-10002.488281f,-10003.024414f,-10002.637695f,-10002.666016f,-10002.968750f,-10003.342773f,
-10003.729492f,-10004.554688f,-10004.778320f,-10004.507812f,-10003.836914f,-10002.867188f,-10001.789062f,-10001.525391f,-10001.888672f,-10003.060547f,-10004.342773f,-10005.612305f,-10006.112305f,-10006.268555f,-10005.692383f,-10005.629883f,-10005.627930f,-10005.662109f,-10005.745117f,-10005.575195f,
-10004.500000f,-10003.134766f,-10001.329102f,-9999.462891f,-9998.238281f,-9998.335938f,-9998.679688f,-9999.355469f,-9999.852539f,-9999.677734f,-9999.342773f,-9998.584961f,-9998.154297f,-9997.893555f,-9997.702148f,-9997.758789f,-9997.204102f,-9996.350586f,-9994.989258f,-9993.403320f,
-9991.918945f,-9991.372070f,-9991.602539f,-9992.893555f,-9994.358398f,-9995.886719f,-9996.855469f,-9997.605469f,-9997.810547f,-9998.293945f,-9998.813477f,-9999.686523f,-10000.901367f,-10001.577148f,-10002.023438f,-10001.963867f,-10001.418945f,-10000.852539f,-9999.955078f,-10000.355469f,
-10001.472656f,-10003.206055f,-10005.147461f,-10006.365234f,-10007.145508f,-10007.448242f,-10008.109375f,-10008.561523f,-10009.393555f,-10010.322266f,-10010.794922f,-10011.178711f,-10011.249023f,-10010.574219f,-10009.349609f,-10007.644531f,-10006.626953f,-10006.719727f,-10007.215820f,-10008.083984f,
-10009.067383f,-10009.241211f,-10008.679688f,-10008.168945f,-10007.681641f,-10007.394531f,-10006.819336f,-10006.334961f,-10006.267578f,-10005.874023f,-10005.116211f,-10003.589844f,-10001.880859f,-10000.289062f,-9999.047852f,-9998.967773f,-9999.223633f,-10000.483398f,-10001.443359f,-10002.334961f,
-9993.714844f,-10002.909180f,-10003.623047f,-10002.430664f,-10003.713867f,-10004.390625f,-10004.545898f,-10006.699219f,-10007.318359f,-10004.305664f,-10006.828125f,-10006.056641f,-10006.708008f,-10007.259766f,-10007.236328f,-10006.319336f,-10005.040039f,-10003.053711f,-10001.178711f,-10000.148438f,
-9999.732422f,-10000.415039f,-10001.416992f,-10002.345703f,-10002.381836f,-10002.554688f,-10002.919922f,-10003.216797f,-10003.515625f,-10004.481445f,-10005.085938f,-10005.331055f,-10005.161133f,-10004.282227f,-10002.881836f,-10001.829102f,-10001.216797f,-10001.673828f,-10002.698242f,-10004.199219f,
-10005.250000f,-10005.959961f,-10005.683594f,-10005.772461f,-10005.710938f,-10005.700195f,-10005.850586f,-10006.152344f,-10005.706055f,-10004.907227f,-10003.235352f,-10000.956055f,-9998.905273f,-9998.101562f,-9997.734375f,-9998.220703f,-9998.962891f,-9999.263672f,-9999.467773f,-9998.933594f,
-9998.575195f,-9998.184570f,-9997.893555f,-9998.011719f,-9997.824219f,-9997.463867f,-9996.596680f,-9995.012695f,-9993.176758f,-9991.859375f,-9991.092773f,-9991.538086f,-9992.591797f,-9994.181641f,-9995.555664f,-9996.792969f,-9997.328125f,-9997.965820f,-9998.447266f,-9999.220703f,
-10000.341797f,-10001.246094f,-10002.140625f,-10002.666992f,-10002.444336f,-10001.855469f,-10000.515625f,-9999.954102f,-10000.107422f,-10001.222656f,-10003.043945f,-10004.726562f,-10006.081055f,-10006.815430f,-10007.663086f,-10008.155273f,-10008.910156f,-10009.743164f,-10010.344727f,-10011.052734f,
-10011.757812f,-10011.690430f,-10010.704102f,-10008.793945f,-10007.078125f,-10006.180664f,-10005.842773f,-10006.309570f,-10007.381836f,-10008.057617f,-10008.159180f,-10008.033203f,-10007.719727f,-10007.501953f,-10006.821289f,-10006.247070f,-10006.269531f,-10006.254883f,-10006.091797f,-10005.133789f,
-10003.554688f,-10001.695312f,-9999.832031f,-9998.690430f,-9998.154297f,-9998.864258f,-9999.835938f,-10001.110352f,-9993.898438f,-10003.814453f,-10004.347656f,-10002.498047f,-10003.159180f,-10003.494141f,-10003.707031f,-10006.243164f,-10007.082031f,-10004.449219f,-10007.140625f,-10005.825195f,
-10006.386719f,-10007.104492f,-10007.574219f,-10007.390625f,-10006.676758f,-10005.043945f,-10002.953125f,-10001.243164f,-9999.875977f,-9999.805664f,-10000.486328f,-10001.460938f,-10001.822266f,-10002.292969f,-10002.829102f,-10003.083008f,-10003.232422f,-10004.061523f,-10004.780273f,-10005.427734f,
-10005.858398f,-10005.501953f,-10004.286133f,-10002.891602f,-10001.473633f,-10001.011719f,-10001.346680f,-10002.535156f,-10003.776367f,-10004.922852f,-10005.123047f,-10005.508789f,-10005.448242f,-10005.349609f,-10005.354492f,-10005.845703f,-10005.922852f,-10005.822266f,-10004.753906f,-10002.667969f,
-10000.275391f,-9998.719727f,-9997.454102f,-9997.313477f,-9997.834961f,-9998.324219f,-9998.993164f,-9998.861328f,-9998.715820f,-9998.296875f,-9997.833008f,-9997.801758f,-9997.764648f,-9997.825195f,-9997.547852f,-9996.338867f,-9994.664062f,-9993.080078f,-9991.544922f,-9991.024414f,
-9991.245117f,-9992.428711f,-9993.804688f,-9995.364258f,-9996.342773f,-9997.288086f,-9997.826172f,-9998.475586f,-9999.380859f,-10000.257812f,-10001.341797f,-10002.397461f,-10002.735352f,-10002.602539f,-10001.389648f,-10000.337891f,-9999.607422f,-9999.843750f,-10001.039062f,-10002.685547f,
-10004.344727f,-10005.540039f,-10006.729492f,-10007.391602f,-10008.136719f,-10008.795898f,-10009.275391f,-10009.960938f,-10011.103516f,-10011.695312f,-10011.304688f,-10009.800781f,-10007.894531f,-10006.329102f,-10005.085938f,-10004.785156f,-10005.404297f,-10006.140625f,-10006.752930f,-10007.106445f,
-10007.182617f,-10007.181641f,-10006.524414f,-10005.795898f,-10005.708984f,-10005.776367f,-10006.066406f,-10005.733398f,-10004.660156f,-10003.070312f,-10001.183594f,-9999.328125f,-9997.952148f,-9997.729492f,-9998.201172f,-9999.375000f,-9993.832031f,-10004.380859f,-10004.833984f,-10002.709961f,
-10002.555664f,-10002.301758f,-10002.245117f,-10005.136719f,-10006.165039f,-10004.059570f,-10007.072266f,-10005.935547f,-10006.320312f,-10006.916016f,-10007.446289f,-10007.708008f,-10007.567383f,-10006.581055f,-10004.779297f,-10002.863281f,-10000.728516f,-9999.768555f,-9999.712891f,-10000.370117f,
-10000.872070f,-10001.715820f,-10002.659180f,-10003.130859f,-10003.315430f,-10003.900391f,-10004.446289f,-10005.152344f,-10005.954102f,-10006.177734f,-10005.543945f,-10004.344727f,-10002.595703f,-10001.343750f,-10000.887695f,-10001.410156f,-10002.438477f,-10003.833984f,-10004.639648f,-10005.578125f,
-10005.853516f,-10005.888672f,-10005.745117f,-10006.116211f,-10006.296875f,-10006.666016f,-10006.284180f,-10004.822266f,-10002.598633f,-10000.677734f,-9998.629883f,-9997.652344f,-9997.531250f,-9997.815430f,-9998.729492f,-9999.068359f,-9999.335938f,-9999.188477f,-9998.698242f,-9998.378906f,
-9998.229492f,-9998.341797f,-9998.342773f,-9997.527344f,-9996.314453f,-9994.873047f,-9993.039062f,-9991.751953f,-9991.036133f,-9991.401367f,-9992.374023f,-9993.929688f,-9995.309570f,-9996.674805f,-9997.526367f,-9998.253906f,-9998.961914f,-9999.612305f,-10000.483398f,-10001.690430f,
-10002.361328f,-10002.835938f,-10002.178711f,-10001.191406f,-9999.958008f,-9999.409180f,-9999.734375f,-10000.901367f,-10002.503906f,-10004.064453f,-10005.734375f,-10006.874023f,-10007.942383f,-10008.615234f,-10008.974609f,-10009.391602f,-10010.554688f,-10011.497070f,-10011.712891f,-10010.963867f,
-10009.415039f,-10007.723633f,-10005.918945f,-10004.786133f,-10004.598633f,-10004.932617f,-10005.727539f,-10006.533203f,-10007.181641f,-10007.683594f,-10007.362305f,-10006.677734f,-10006.386719f,-10006.219727f,-10006.468750f,-10006.405273f,-10005.799805f,-10004.734375f,-10003.253906f,-10001.224609f,
-9999.308594f,-9998.117188f,-9997.722656f,-9998.301758f,-9993.533203f,-10004.938477f,-10005.646484f,-10003.499023f,-10002.832031f,-10001.903320f,-10001.365234f,-10003.802734f,-10004.701172f,-10003.286133f,-10006.731445f,-10005.823242f,-10006.324219f,-10006.898438f,-10007.282227f,-10007.623047f,
-10007.778320f,-10007.413086f,-10006.165039f,-10004.586914f,-10002.202148f,-10000.616211f,-9999.713867f,-9999.693359f,-9999.922852f,-10000.857422f,-10002.133789f,-10002.972656f,-10003.494141f,-10004.060547f,-10004.467773f,-10005.007812f,-10005.800781f,-10006.342773f,-10006.276367f,-10005.616211f,
-10004.050781f,-10002.463867f,-10001.384766f,-10001.123047f,-10001.558594f,-10002.746094f,-10003.807617f,-10005.247070f,-10006.038086f,-10006.509766f,-10006.528320f,-10006.848633f,-10006.832031f,-10007.276367f,-10007.304688f,-10006.501953f,-10004.824219f,-10003.078125f,-10000.715820f,-9999.083008f,
-9998.170898f,-9997.811523f,-9998.502930f,-9999.004883f,-9999.608398f,-9999.908203f,-9999.693359f,-9999.347656f,-9999.085938f,-9999.017578f,-9998.966797f,-9998.308594f,-9997.451172f,-9996.403320f,-9994.702148f,-9993.097656f,-9991.663086f,-9991.125977f,-9991.319336f,-9992.359375f,
-9993.734375f,-9995.286133f,-9996.466797f,-9997.482422f,-9998.294922f,-9998.875000f,-9999.449219f,-10000.491211f,-10001.138672f,-10001.981445f,-10001.911133f,-10001.459961f,-10000.232422f,-9999.399414f,-9998.990234f,-9999.433594f,-10000.536133f,-10002.069336f,-10003.958984f,-10005.532227f,
-10007.089844f,-10008.126953f,-10008.619141f,-10008.935547f,-10009.905273f,-10010.796875f,-10011.293945f,-10011.168945f,-10010.237305f,-10008.954102f,-10007.160156f,-10005.527344f,-10004.561523f,-10004.184570f,-10004.600586f,-10005.461914f,-10006.499023f,-10007.538086f,-10007.794922f,-10007.502930f,
-10007.338867f,-10006.983398f,-10006.984375f,-10006.840820f,-10006.360352f,-10005.684570f,-10004.753906f,-10003.023438f,-10001.105469f,-9999.350586f,-9998.175781f,-9997.871094f,-9993.703125f,-10004.916992f,-10006.007812f,-10004.240234f,-10003.673828f,-10002.205078f,-10001.129883f,-10002.784180f,
-10003.387695f,-10002.337891f,-10005.932617f,-10005.185547f,-10005.955078f,-10006.681641f,-10006.947266f,-10007.258789f,-10007.506836f,-10007.559570f,-10006.895508f,-10005.935547f,-10003.789062f,-10001.976562f,-10000.411133f,-9999.662109f,-9999.297852f,-9999.968750f,-10001.244141f,-10002.336914f,
-10003.237305f,-10003.982422f,-10004.410156f,-10004.806641f,-10005.461914f,-10006.102539f,-10006.404297f,-10006.335938f,-10005.221680f,-10003.714844f,-10002.333008f,-10001.424805f,-10001.102539f,-10001.693359f,-10002.556641f,-10004.117188f,-10005.265625f,-10006.183594f,-10006.552734f,-10006.952148f,
-10006.753906f,-10007.098633f,-10007.299805f,-10006.957031f,-10005.886719f,-10004.616211f,-10002.408203f,-10000.498047f,-9998.911133f,-9997.816406f,-9997.904297f,-9998.161133f,-9998.810547f,-9999.434570f,-9999.580078f,-9999.416016f,-9999.222656f,-9999.033203f,-9998.821289f,-9998.241211f,
-9997.604492f,-9996.982422f,-9995.692383f,-9994.169922f,-9992.476562f,-9991.251953f,-9990.695312f,-9990.983398f,-9991.975586f,-9993.386719f,-9994.720703f,-9996.031250f,-9997.114258f,-9997.838867f,-9998.310547f,-9999.203125f,-9999.754883f,-10000.776367f,-10001.143555f,-10001.366211f,
-10000.547852f,-9999.858398f,-9999.079102f,-9998.871094f,-9999.275391f,-10000.409180f,-10002.095703f,-10003.793945f,-10005.703125f,-10007.084961f,-10007.891602f,-10008.318359f,-10009.142578f,-10009.828125f,-10010.393555f,-10010.606445f,-10010.212891f,-10009.547852f,-10008.138672f,-10006.410156f,
-10004.956055f,-10003.819336f,-10003.523438f,-10004.028320f,-10005.040039f,-10006.297852f,-10007.028320f,-10007.170898f,-10007.356445f,-10007.055664f,-10006.896484f,-10006.622070f,-10006.108398f,-10005.643555f,-10005.222656f,-10003.944336f,-10002.396484f,-10000.523438f,-9998.908203f,-9997.832031f,
-9993.688477f,-10004.603516f,-10005.940430f,-10004.735352f,-10004.420898f,-10002.781250f,-10001.220703f,-10001.834961f,-10002.099609f,-10001.044922f,-10004.523438f,-10004.055664f,-10005.315430f,-10006.402344f,-10006.668945f,-10006.828125f,-10006.915039f,-10007.116211f,-10006.888672f,-10006.645508f,
-10005.069336f,-10003.432617f,-10001.522461f,-10000.128906f,-9999.057617f,-9999.221680f,-10000.279297f,-10001.574219f,-10002.972656f,-10004.129883f,-10004.810547f,-10005.123047f,-10005.487305f,-10005.951172f,-10006.333984f,-10006.750977f,-10006.191406f,-10005.082031f,-10003.729492f,-10002.427734f,
-10001.362305f,-10001.202148f,-10001.498047f,-10002.924805f,-10004.322266f,-10005.759766f,-10006.690430f,-10007.396484f,-10007.055664f,-10007.124023f,-10007.184570f,-10007.014648f,-10006.451172f,-10005.815430f,-10004.110352f,-10002.291016f,-10000.320312f,-9998.487305f,-9997.795898f,-9997.534180f,
-9998.008789f,-9998.887695f,-9999.527344f,-9999.812500f,-9999.871094f,-9999.631836f,-9999.200195f,-9998.509766f,-9997.852539f,-9997.505859f,-9996.684570f,-9995.542969f,-9993.973633f,-9992.340820f,-9991.056641f,-9990.503906f,-9990.804688f,-9991.780273f,-9993.083008f,-9994.677734f,
-9996.187500f,-9997.276367f,-9997.819336f,-9998.547852f,-9998.829102f,-9999.697266f,-10000.237305f,-10001.046875f,-10000.779297f,-10000.566406f,-9999.813477f,-9999.123047f,-9998.774414f,-9999.251953f,-10000.440430f,-10001.977539f,-10004.106445f,-10005.919922f,-10007.223633f,-10008.014648f,
-10008.768555f,-10009.129883f,-10009.500000f,-10009.657227f,-10009.641602f,-10009.558594f,-10008.797852f,-10007.344727f,-10005.743164f,-10004.008789f,-10002.833984f,-10002.698242f,-10003.373047f,-10004.612305f,-10005.751953f,-10006.473633f,-10007.259766f,-10007.266602f,-10007.083984f,-10006.595703f,
-10005.830078f,-10005.298828f,-10005.141602f,-10004.310547f,-10003.375977f,-10001.805664f,-10000.110352f,-9998.467773f,-9993.614258f,-10003.833984f,-10005.626953f,-10004.683594f,-10005.148438f,-10003.549805f,-10001.870117f,-10001.814453f,-10001.572266f,-10000.118164f,-10003.212891f,-10002.781250f,
-10004.490234f,-10006.090820f,-10006.630859f,-10006.820312f,-10006.699219f,-10006.700195f,-10006.547852f,-10006.729492f,-10005.779297f,-10004.653320f,-10002.862305f,-10001.176758f,-9999.515625f,-9998.992188f,-9999.505859f,-10000.608398f,-10002.220703f,-10003.809570f,-10004.929688f,-10005.433594f,
-10005.689453f,-10005.878906f,-10006.023438f,-10006.582031f,-10006.422852f,-10005.839844f,-10004.833984f,-10003.646484f,-10002.203125f,-10001.440430f,-10000.968750f,-10001.905273f,-10003.181641f,-10004.823242f,-10006.328125f,-10007.560547f,-10007.433594f,-10007.413086f,-10007.202148f,-10006.855469f,
-10006.447266f,-10006.287109f,-10005.212891f,-10003.872070f,-10002.012695f,-9999.801758f,-9998.456055f,-9997.490234f,-9997.459961f,-9998.170898f,-9999.030273f,-9999.795898f,-10000.247070f,-10000.242188f,-9999.762695f,-9998.960938f,-9998.035156f,-9997.659180f,-9997.113281f,-9996.376953f,
-9995.250977f,-9993.709961f,-9992.142578f,-9990.986328f,-9990.575195f,-9990.863281f,-9991.758789f,-9993.363281f,-9995.140625f,-9996.678711f,-9997.593750f,-9998.442383f,-9998.599609f,-9999.128906f,-9999.474609f,-10000.520508f,-10000.676758f,-10001.089844f,-10000.786133f,-10000.099609f,
-9999.343750f,-9999.179688f,-9999.671875f,-10000.692383f,-10002.659180f,-10004.613281f,-10006.393555f,-10007.769531f,-10008.802734f,-10009.076172f,-10009.232422f,-10009.040039f,-10009.041992f,-10009.256836f,-10009.084961f,-10008.206055f,-10006.963867f,-10005.089844f,-10003.307617f,-10002.511719f,
-10002.575195f,-10003.408203f,-10004.538086f,-10005.543945f,-10006.919922f,-10007.471680f,-10007.655273f,-10007.218750f,-10006.222656f,-10005.382812f,-10005.118164f,-10004.413086f,-10003.994141f,-10002.940430f,-10001.594727f,-9999.896484f,-9993.455078f,-10002.990234f,-10005.210938f,-10004.382812f,
-10005.558594f,-10004.251953f,-10002.708008f,-10002.431641f,-10001.855469f,-9999.578125f,-10002.091797f,-10001.437500f,-10003.267578f,-10005.254883f,-10006.245117f,-10006.744141f,-10006.728516f,-10006.652344f,-10006.556641f,-10006.967773f,-10006.528320f,-10005.934570f,-10004.497070f,-10002.773438f,
-10000.752930f,-9999.610352f,-9999.354492f,-9999.892578f,-10001.334961f,-10003.098633f,-10004.610352f,-10005.482422f,-10005.937500f,-10006.123047f,-10006.151367f,-10006.770508f,-10006.880859f,-10006.714844f,-10006.076172f,-10005.186523f,-10003.606445f,-10002.477539f,-10001.283203f,-10001.502930f,
-10002.276367f,-10003.732422f,-10005.524414f,-10007.216797f,-10007.576172f,-10007.781250f,-10007.577148f,-10007.151367f,-10006.841797f,-10006.996094f,-10006.432617f,-10005.610352f,-10004.134766f,-10001.834961f,-10000.080078f,-9998.447266f,-9997.699219f,-9997.861328f,-9998.539062f,-9999.526367f,
-10000.312500f,-10000.710938f,-10000.448242f,-9999.849609f,-9998.831055f,-9998.401367f,-9998.006836f,-9997.520508f,-9996.805664f,-9995.515625f,-9993.879883f,-9992.384766f,-9991.355469f,-9990.864258f,-9990.995117f,-9992.207031f,-9993.890625f,-9995.665039f,-9996.943359f,-9998.123047f,
-9998.477539f,-9998.869141f,-9999.103516f,-10000.226562f,-10000.614258f,-10001.472656f,-10001.710938f,-10001.216797f,-10000.378906f,-9999.763672f,-9999.602539f,-9999.864258f,-10001.272461f,-10002.975586f,-10004.913086f,-10006.756836f,-10008.214844f,-10008.702148f,-10008.988281f,-10008.622070f,
-10008.660156f,-10008.981445f,-10009.168945f,-10008.808594f,-10008.093750f,-10006.353516f,-10004.286133f,-10003.012695f,-10002.394531f,-10002.509766f,-10003.270508f,-10004.179688f,-10005.841797f,-10006.884766f,-10007.669922f,-10007.591797f,-10006.745117f,-10005.820312f,-10005.487305f,-10004.748047f,
-10004.583008f,-10003.915039f,-10002.977539f,-10001.426758f,-9993.509766f,-10002.359375f,-10004.681641f,-10003.812500f,-10005.574219f,-10004.591797f,-10003.484375f,-10003.359375f,-10002.623047f,-9999.642578f,-10001.443359f,-10000.197266f,-10001.754883f,-10003.902344f,-10005.418945f,-10006.470703f,
-10006.803711f,-10006.737305f,-10006.552734f,-10006.844727f,-10006.618164f,-10006.397461f,-10005.433594f,-10003.963867f,-10001.968750f,-10000.428711f,-9999.446289f,-9999.237305f,-10000.121094f,-10001.775391f,-10003.523438f,-10004.910156f,-10005.865234f,-10006.331055f,-10006.375977f,-10006.933594f,
-10007.083984f,-10007.094727f,-10006.814453f,-10006.373047f,-10005.040039f,-10003.946289f,-10002.329102f,-10001.886719f,-10001.996094f,-10002.917969f,-10004.630859f,-10006.631836f,-10007.657227f,-10008.416992f,-10008.480469f,-10008.053711f,-10007.646484f,-10007.790039f,-10007.421875f,-10007.026367f,
-10006.080078f,-10004.029297f,-10002.218750f,-10000.172852f,-9998.704102f,-9998.066406f,-9998.147461f,-9998.969727f,-9999.893555f,-10000.790039f,-10000.933594f,-10000.715820f,-9999.725586f,-9999.133789f,-9998.635742f,-9998.135742f,-9997.665039f,-9996.690430f,-9995.258789f,-9993.811523f,
-9992.474609f,-9991.315430f,-9990.615234f,-9991.113281f,-9992.296875f,-9993.993164f,-9995.551758f,-9997.219727f,-9998.046875f,-9998.495117f,-9998.633789f,-9999.630859f,-9999.976562f,-10001.001953f,-10001.697266f,-10001.666016f,-10001.163086f,-10000.536133f,-9999.982422f,-9999.564453f,
-10000.236328f,-10001.368164f,-10003.152344f,-10005.261719f,-10007.246094f,-10008.264648f,-10009.013672f,-10008.729492f,-10008.748047f,-10008.919922f,-10009.142578f,-10009.105469f,-10008.959961f,-10007.634766f,-10005.722656f,-10004.379883f,-10003.206055f,-10002.571289f,-10002.679688f,-10003.084961f,
-10004.644531f,-10005.981445f,-10007.406250f,-10007.974609f,-10007.551758f,-10006.689453f,-10006.238281f,-10005.172852f,-10004.895508f,-10004.335938f,-10003.711914f,-10002.501953f,-9993.490234f,-10002.102539f,-10004.389648f,-10003.430664f,-10005.301758f,-10004.756836f,-10004.023438f,-10004.272461f,
-10003.734375f,-10000.474609f,-10001.753906f,-9999.948242f,-10000.843750f,-10002.631836f,-10004.337891f,-10005.849609f,-10006.740234f,-10006.933594f,-10006.860352f,-10006.958008f,-10006.708984f,-10006.612305f,-10006.000977f,-10004.890625f,-10003.211914f,-10001.689453f,-10000.284180f,-9999.357422f,
-9999.466797f,-10000.543945f,-10002.074219f,-10003.717773f,-10005.136719f,-10006.089844f,-10006.441406f,-10007.051758f,-10007.206055f,-10007.207031f,-10007.051758f,-10006.933594f,-10005.981445f,-10005.248047f,-10003.694336f,-10002.850586f,-10002.345703f,-10002.536133f,-10003.732422f,-10005.536133f,
-10006.995117f,-10008.312500f,-10008.894531f,-10008.791016f,-10008.469727f,-10008.523438f,-10008.122070f,-10007.889648f,-10007.374023f,-10005.755859f,-10004.268555f,-10002.258789f,-10000.380859f,-9999.039062f,-9998.323242f,-9998.586914f,-9999.235352f,-10000.368164f,-10000.875000f,-10001.200195f,
-10000.560547f,-10000.047852f,-9999.504883f,-9998.848633f,-9998.381836f,-9997.645508f,-9996.497070f,-9995.376953f,-9994.149414f,-9992.713867f,-9991.362305f,-9991.113281f,-9991.523438f,-9992.683594f,-9994.095703f,-9996.042969f,-9997.372070f,-9998.152344f,-9998.470703f,-9999.418945f,
-9999.622070f,-10000.500977f,-10001.363281f,-10001.646484f,-10001.639648f,-10001.317383f,-10000.785156f,-9999.992188f,-9999.991211f,-10000.378906f,-10001.575195f,-10003.416992f,-10005.591797f,-10007.091797f,-10008.473633f,-10008.649414f,-10008.873047f,-10008.976562f,-10009.054688f,-10009.080078f,
-10009.250000f,-10008.346680f,-10006.916992f,-10005.940430f,-10004.623047f,-10003.482422f,-10002.940430f,-10002.641602f,-10003.598633f,-10004.762695f,-10006.486328f,-10007.621094f,-10007.838867f,-10007.432617f,-10007.145508f,-10005.975586f,-10005.429688f,-10004.729492f,-10004.210938f,-10003.292969f,
-9993.424805f,-10002.213867f,-10004.244141f,-10003.280273f,-10004.843750f,-10004.579102f,-10004.245117f,-10005.088867f,-10004.767578f,-10001.548828f,-10002.611328f,-10000.293945f,-10000.388672f,-10001.541992f,-10003.170898f,-10004.958984f,-10006.421875f,-10007.045898f,-10007.200195f,-10007.142578f,
-10006.711914f,-10006.495117f,-10006.035156f,-10005.259766f,-10004.045898f,-10002.821289f,-10001.294922f,-9999.879883f,-9999.232422f,-9999.574219f,-10000.603516f,-10002.240234f,-10004.002930f,-10005.492188f,-10006.344727f,-10007.127930f,-10007.298828f,-10007.174805f,-10006.998047f,-10007.039062f,
-10006.493164f,-10006.254883f,-10005.102539f,-10004.208984f,-10003.333008f,-10002.885742f,-10003.370117f,-10004.712891f,-10006.327148f,-10008.091797f,-10009.268555f,-10009.695312f,-10009.583008f,-10009.547852f,-10008.939453f,-10008.648438f,-10008.374023f,-10007.178711f,-10006.172852f,-10004.508789f,
-10002.531250f,-10000.728516f,-9999.245117f,-9998.751953f,-9998.868164f,-9999.958984f,-10000.677734f,-10001.499023f,-10001.340820f,-10001.046875f,-10000.498047f,-9999.634766f,-9998.954102f,-9998.245117f,-9997.309570f,-9996.587891f,-9995.687500f,-9994.279297f,-9992.584961f,-9991.663086f,
-9991.284180f,-9991.660156f,-9992.607422f,-9994.582031f,-9996.294922f,-9997.497070f,-9998.176758f,-9999.194336f,-9999.233398f,-9999.814453f,-10000.576172f,-10001.014648f,-10001.507812f,-10001.706055f,-10001.515625f,-10000.690430f,-10000.253906f,-10000.015625f,-10000.502930f,-10001.765625f,
-10003.862305f,-10005.687500f,-10007.719727f,-10008.570312f,-10009.193359f,-10009.327148f,-10009.226562f,-10009.095703f,-10009.354492f,-10008.779297f,-10007.926758f,-10007.559570f,-10006.430664f,-10005.123047f,-10004.125977f,-10003.175781f,-10003.338867f,-10004.044922f,-10005.680664f,-10007.188477f,
-10008.036133f,-10008.246094f,-10008.296875f,-10007.168945f,-10006.327148f,-10005.318359f,-10004.661133f,-10003.850586f,-9993.469727f,-10002.465820f,-10004.492188f,-10003.470703f,-10004.919922f,-10004.452148f,-10004.311523f,-10005.529297f,-10005.643555f,-10002.741211f,-10004.116211f,-10001.537109f,
-10000.941406f,-10001.248047f,-10002.260742f,-10003.809570f,-10005.545898f,-10006.582031f,-10007.147461f,-10007.225586f,-10006.772461f,-10006.386719f,-10005.916992f,-10005.347656f,-10004.534180f,-10003.754883f,-10002.449219f,-10000.932617f,-9999.762695f,-9999.351562f,-9999.569336f,-10000.713867f,
-10002.330078f,-10004.083984f,-10005.450195f,-10006.606445f,-10007.054688f,-10006.958008f,-10006.757812f,-10006.816406f,-10006.530273f,-10006.695312f,-10006.115234f,-10005.456055f,-10004.623047f,-10003.799805f,-10003.583008f,-10004.147461f,-10005.370117f,-10007.096680f,-10008.629883f,-10009.639648f,
-10009.999023f,-10010.145508f,-10009.486328f,-10009.071289f,-10008.839844f,-10007.931641f,-10007.362305f,-10006.220703f,-10004.514648f,-10002.664062f,-10000.736328f,-9999.483398f,-9998.848633f,-9999.412109f,-9999.893555f,-10000.913086f,-10001.203125f,-10001.351562f,-10001.093750f,-10000.266602f,
-9999.429688f,-9998.618164f,-9997.791992f,-9997.342773f,-9996.834961f,-9995.733398f,-9994.128906f,-9992.875977f,-9991.886719f,-9991.421875f,-9991.531250f,-9993.001953f,-9994.656250f,-9996.154297f,-9997.242188f,-9998.596680f,-9998.760742f,-9999.146484f,-9999.701172f,-10000.079102f,
-10000.815430f,-10001.506836f,-10001.808594f,-10001.327148f,-10000.832031f,-10000.274414f,-10000.070312f,-10000.556641f,-10002.041992f,-10003.682617f,-10005.988281f,-10007.483398f,-10008.671875f,-10009.201172f,-10009.213867f,-10008.951172f,-10009.139648f,-10008.711914f,-10008.273438f,-10008.521484f,
-10007.835938f,-10006.761719f,-10005.664062f,-10004.342773f,-10003.707031f,-10003.707031f,-10004.682617f,-10006.053711f,-10007.194336f,-10007.992188f,-10008.535156f,-10007.925781f,-10007.100586f,-10005.977539f,-10005.153320f,-10004.265625f,-9993.408203f,-10002.669922f,-10004.752930f,-10003.883789f,
-10005.040039f,-10004.353516f,-10004.119141f,-10005.375977f,-10006.022461f,-10003.358398f,-10005.222656f,-10002.977539f,-10001.942383f,-10001.459961f,-10001.648438f,-10002.611328f,-10004.330078f,-10005.744141f,-10006.862305f,-10007.376953f,-10007.072266f,-10006.525391f,-10005.797852f,-10005.133789f,
-10004.500977f,-10004.127930f,-10003.258789f,-10001.994141f,-10000.723633f,-9999.772461f,-9999.201172f,-9999.612305f,-10000.774414f,-10002.550781f,-10004.395508f,-10006.121094f,-10007.061523f,-10007.130859f,-10006.888672f,-10006.722656f,-10006.384766f,-10006.701172f,-10006.620117f,-10006.409180f,
-10005.933594f,-10005.154297f,-10004.461914f,-10004.277344f,-10004.787109f,-10006.083984f,-10007.739258f,-10009.325195f,-10010.335938f,-10010.925781f,-10010.369141f,-10009.758789f,-10009.289062f,-10008.348633f,-10007.952148f,-10007.290039f,-10006.046875f,-10004.532227f,-10002.570312f,-10000.809570f,
-9999.451172f,-9999.297852f,-9999.257812f,-10000.237305f,-10000.920898f,-10001.657227f,-10001.914062f,-10001.350586f,-10000.372070f,-9999.284180f,-9998.284180f,-9997.800781f,-9997.559570f,-9996.846680f,-9995.622070f,-9994.405273f,-9993.171875f,-9991.981445f,-9991.223633f,-9991.910156f,
-9993.153320f,-9994.762695f,-9996.296875f,-9998.221680f,-9998.750977f,-9999.114258f,-9999.341797f,-9999.355469f,-9999.982422f,-10000.898438f,-10001.658203f,-10001.688477f,-10001.496094f,-10000.998047f,-10000.358398f,-10000.119141f,-10000.773438f,-10001.879883f,-10004.155273f,-10006.163086f,
-10008.041016f,-10009.248047f,-10009.602539f,-10009.277344f,-10009.185547f,-10008.592773f,-10008.207031f,-10008.827148f,-10008.670898f,-10008.065430f,-10007.262695f,-10005.958008f,-10004.805664f,-10004.071289f,-10004.140625f,-10004.970703f,-10006.100586f,-10007.342773f,-10008.446289f,-10008.648438f,
-10008.146484f,-10007.035156f,-10005.957031f,-10004.694336f,-9993.436523f,-10002.683594f,-10004.681641f,-10004.226562f,-10005.242188f,-10004.488281f,-10004.011719f,-10005.145508f,-10005.947266f,-10003.555664f,-10005.909180f,-10004.204102f,-10003.205078f,-10002.247070f,-10001.619141f,-10001.694336f,
-10002.879883f,-10004.224609f,-10005.642578f,-10006.701172f,-10006.851562f,-10006.445312f,-10005.592773f,-10004.818359f,-10004.185547f,-10004.056641f,-10003.677734f,-10002.913086f,-10001.920898f,-10000.861328f,-9999.757812f,-9999.379883f,-9999.726562f,-10001.000000f,-10002.869141f,-10004.958984f,
-10006.458984f,-10006.976562f,-10006.942383f,-10006.675781f,-10006.229492f,-10006.477539f,-10006.661133f,-10006.892578f,-10006.958008f,-10006.541016f,-10005.778320f,-10005.111328f,-10004.758789f,-10005.244141f,-10006.466797f,-10008.165039f,-10009.642578f,-10010.791992f,-10010.683594f,-10010.159180f,
-10009.544922f,-10008.476562f,-10008.049805f,-10007.666992f,-10006.942383f,-10005.990234f,-10004.417969f,-10002.580078f,-10000.783203f,-9999.857422f,-9999.050781f,-9999.509766f,-10000.139648f,-10001.211914f,-10002.024414f,-10002.014648f,-10001.250000f,-10000.025391f,-9998.873047f,-9998.160156f,
-9997.925781f,-9997.516602f,-9996.795898f,-9995.944336f,-9994.875977f,-9993.340820f,-9991.910156f,-9991.684570f,-9992.069336f,-9993.294922f,-9994.845703f,-9997.201172f,-9998.291016f,-9998.983398f,-9999.165039f,-9998.869141f,-9999.194336f,-10000.059570f,-10001.061523f,-10001.596680f,
-10001.903320f,-10001.816406f,-10001.137695f,-10000.511719f,-10000.325195f,-10000.583984f,-10002.238281f,-10004.190430f,-10006.413086f,-10008.281250f,-10009.278320f,-10009.212891f,-10009.047852f,-10008.304688f,-10007.708008f,-10008.403320f,-10008.638672f,-10008.575195f,-10008.285156f,-10007.367188f,
-10006.145508f,-10004.978516f,-10004.125977f,-10004.099609f,-10004.686523f,-10005.855469f,-10007.188477f,-10008.198242f,-10008.358398f,-10007.690430f,-10006.661133f,-10005.169922f,-9993.174805f,-10002.524414f,-10004.442383f,-10004.243164f,-10005.292969f,-10004.491211f,-10003.938477f,-10004.856445f,
-10005.667969f,-10003.381836f,-10006.193359f,-10005.270508f,-10004.824219f,-10003.930664f,-10002.850586f,-10002.074219f,-10002.397461f,-10003.192383f,-10004.413086f,-10005.753906f,-10006.310547f,-10006.249023f,-10005.478516f,-10004.634766f,-10003.898438f,-10003.750000f,-10003.720703f,-10003.457031f,
-10003.002930f,-10002.291016f,-10001.124023f,-10000.223633f,-9999.743164f,-10000.142578f,-10001.518555f,-10003.530273f,-10005.349609f,-10006.372070f,-10006.741211f,-10006.639648f,-10006.222656f,-10006.365234f,-10006.564453f,-10007.147461f,-10007.796875f,-10008.001953f,-10007.617188f,-10007.021484f,
-10006.090820f,-10005.734375f,-10006.125000f,-10007.382812f,-10008.878906f,-10010.400391f,-10010.815430f,-10010.636719f,-10010.103516f,-10008.940430f,-10008.384766f,-10008.043945f,-10007.695312f,-10007.283203f,-10006.340820f,-10004.844727f,-10003.035156f,-10001.561523f,-10000.031250f,-9999.640625f,
-9999.658203f,-10000.560547f,-10001.537109f,-10002.048828f,-10001.675781f,-10000.513672f,-9999.327148f,-9998.350586f,-9997.918945f,-9997.645508f,-9997.311523f,-9996.962891f,-9996.372070f,-9994.940430f,-9993.289062f,-9992.316406f,-9991.735352f,-9992.089844f,-9993.135742f,-9995.453125f,
-9996.960938f,-9998.133789f,-9998.563477f,-9998.250977f,-9998.332031f,-9998.987305f,-9999.994141f,-10000.875000f,-10001.703125f,-10002.264648f,-10002.021484f,-10001.562500f,-10000.956055f,-10000.434570f,-10001.122070f,-10002.496094f,-10004.521484f,-10006.744141f,-10008.348633f,-10008.800781f,
-10008.867188f,-10008.195312f,-10007.381836f,-10007.928711f,-10008.375977f,-10008.773438f,-10009.046875f,-10008.778320f,-10007.953125f,-10006.843750f,-10005.450195f,-10004.613281f,-10004.270508f,-10004.785156f,-10005.800781f,-10007.189453f,-10007.959961f,-10007.964844f,-10007.298828f,-10005.845703f,
-9993.018555f,-10002.156250f,-10004.245117f,-10003.851562f,-10005.366211f,-10004.857422f,-10004.297852f,-10005.313477f,-10005.906250f,-10003.303711f,-10006.183594f,-10005.943359f,-10006.121094f,-10005.620117f,-10004.575195f,-10003.316406f,-10002.789062f,-10002.724609f,-10003.292969f,-10004.513672f,
-10005.397461f,-10005.864258f,-10005.508789f,-10004.855469f,-10003.991211f,-10003.549805f,-10003.459961f,-10003.362305f,-10003.247070f,-10003.030273f,-10002.175781f,-10001.173828f,-10000.162109f,-9999.685547f,-10000.225586f,-10001.752930f,-10003.570312f,-10005.086914f,-10006.045898f,-10006.501953f,
-10006.397461f,-10006.538086f,-10006.513672f,-10007.112305f,-10008.007812f,-10008.655273f,-10008.831055f,-10008.775391f,-10007.818359f,-10006.957031f,-10006.511719f,-10006.961914f,-10008.032227f,-10009.666992f,-10010.625977f,-10011.057617f,-10011.022461f,-10010.040039f,-10009.399414f,-10008.829102f,
-10008.512695f,-10008.245117f,-10007.759766f,-10006.761719f,-10005.278320f,-10003.738281f,-10001.810547f,-10000.640625f,-9999.787109f,-10000.087891f,-10000.870117f,-10001.751953f,-10001.935547f,-10001.177734f,-10000.281250f,-9999.218750f,-9998.450195f,-9998.018555f,-9997.666016f,-9997.569336f,
-9997.358398f,-9996.260742f,-9994.807617f,-9993.548828f,-9992.205078f,-9991.541992f,-9991.663086f,-9993.472656f,-9995.071289f,-9996.699219f,-9997.684570f,-9997.713867f,-9997.795898f,-9998.168945f,-9998.888672f,-9999.724609f,-10000.696289f,-10001.680664f,-10002.010742f,-10002.118164f,
-10001.605469f,-10000.730469f,-10000.562500f,-10001.052734f,-10002.474609f,-10004.613281f,-10006.688477f,-10007.841797f,-10008.593750f,-10008.400391f,-10007.557617f,-10007.912109f,-10008.237305f,-10008.720703f,-10009.250977f,-10009.448242f,-10009.207031f,-10008.596680f,-10007.237305f,-10006.039062f,
-10004.848633f,-10004.433594f,-10004.683594f,-10005.924805f,-10007.099609f,-10007.846680f,-10007.898438f,-10006.875977f,-9993.045898f,-10002.108398f,-10004.019531f,-10003.344727f,-10005.013672f,-10005.046875f,-10004.774414f,-10006.010742f,-10006.467773f,-10003.596680f,-10006.352539f,-10006.270508f,
-10006.927734f,-10006.903320f,-10006.238281f,-10004.873047f,-10003.735352f,-10002.779297f,-10002.418945f,-10003.079102f,-10003.909180f,-10004.809570f,-10005.053711f,-10004.839844f,-10004.133789f,-10003.506836f,-10003.316406f,-10003.188477f,-10003.206055f,-10003.441406f,-10003.070312f,-10002.307617f,
-10001.166016f,-10000.049805f,-9999.701172f,-10000.414062f,-10001.781250f,-10003.469727f,-10004.918945f,-10006.057617f,-10006.531250f,-10006.914062f,-10006.725586f,-10007.222656f,-10008.078125f,-10008.882812f,-10009.431641f,-10009.970703f,-10009.327148f,-10008.324219f,-10007.203125f,-10006.685547f,
-10006.963867f,-10008.230469f,-10009.359375f,-10010.316406f,-10010.907227f,-10010.380859f,-10009.915039f,-10009.207031f,-10008.794922f,-10008.420898f,-10008.127930f,-10007.555664f,-10006.511719f,-10005.234375f,-10003.264648f,-10001.523438f,-9999.828125f,-9999.281250f,-9999.499023f,-10000.379883f,
-10001.012695f,-10000.869141f,-10000.508789f,-9999.732422f,-9998.857422f,-9998.308594f,-9997.816406f,-9997.775391f,-9997.793945f,-9997.096680f,-9996.121094f,-9995.000977f,-9993.301758f,-9991.794922f,-9990.958984f,-9991.851562f,-9993.111328f,-9994.917969f,-9996.463867f,-9997.127930f,
-9997.568359f,-9997.890625f,-9998.395508f,-9999.064453f,-9999.948242f,-10001.099609f,-10001.860352f,-10002.567383f,-10002.486328f,-10001.634766f,-10000.901367f,-10000.462891f,-10000.988281f,-10002.533203f,-10004.553711f,-10006.195312f,-10007.611328f,-10008.120117f,-10007.588867f,-10007.898438f,
-10008.043945f,-10008.415039f,-10008.875977f,-10009.242188f,-10009.422852f,-10009.371094f,-10008.395508f,-10007.202148f,-10005.441406f,-10004.118164f,-10003.418945f,-10004.011719f,-10005.132812f,-10006.336914f,-10007.125000f,-10006.798828f,-9992.906250f,-10002.680664f,-10004.094727f,-10002.802734f,
-10004.294922f,-10004.511719f,-10004.667969f,-10006.416992f,-10006.993164f,-10003.967773f,-10006.585938f,-10006.496094f,-10007.354492f,-10007.733398f,-10007.707031f,-10006.768555f,-10005.535156f,-10004.002930f,-10002.666016f,-10002.284180f,-10002.437500f,-10003.280273f,-10003.990234f,-10004.392578f,
-10004.180664f,-10003.645508f,-10003.458984f,-10003.202148f,-10003.118164f,-10003.568359f,-10003.654297f,-10003.399414f,-10002.607422f,-10001.329102f,-10000.327148f,-10000.006836f,-10000.378906f,-10001.577148f,-10003.029297f,-10004.696289f,-10005.907227f,-10006.882812f,-10006.856445f,-10007.361328f,
-10008.035156f,-10008.730469f,-10009.387695f,-10010.461914f,-10010.453125f,-10009.859375f,-10008.593750f,-10007.367188f,-10006.643555f,-10007.009766f,-10007.669922f,-10008.738281f,-10009.905273f,-10010.078125f,-10010.207031f,-10009.696289f,-10009.307617f,-10008.776367f,-10008.449219f,-10008.116211f,
-10007.559570f,-10006.849609f,-10005.381836f,-10003.525391f,-10001.323242f,-9999.894531f,-9999.109375f,-9999.391602f,-9999.959961f,-10000.266602f,-10000.601562f,-10000.461914f,-9999.894531f,-9999.447266f,-9998.829102f,-9998.658203f,-9998.654297f,-9998.246094f,-9997.798828f,-9997.184570f,
-9995.671875f,-9993.815430f,-9992.198242f,-9991.881836f,-9992.146484f,-9993.433594f,-9995.083984f,-9996.357422f,-9997.416016f,-9997.996094f,-9998.521484f,-9999.041016f,-9999.661133f,-10000.657227f,-10001.561523f,-10002.652344f,-10003.200195f,-10002.875977f,-10002.083984f,-10001.106445f,
-10000.661133f,-10001.101562f,-10002.354492f,-10003.888672f,-10005.696289f,-10006.979492f,-10007.207031f,-10007.828125f,-10008.030273f,-10008.340820f,-10008.611328f,-10008.861328f,-10009.238281f,-10009.668945f,-10009.347656f,-10008.654297f,-10006.856445f,-10005.012695f,-10003.457031f,-10003.000977f,
-10003.432617f,-10004.460938f,-10005.625000f,-10006.041016f,-9993.233398f,-10003.661133f,-10004.876953f,-10002.826172f,-10003.634766f,-10003.723633f,-10004.013672f,-10006.325195f,-10007.078125f,-10004.245117f,-10006.761719f,-10006.139648f,-10006.896484f,-10007.426758f,-10007.945312f,-10007.667969f,
-10006.778320f,-10005.239258f,-10003.333984f,-10002.022461f,-10001.209961f,-10001.453125f,-10002.093750f,-10002.777344f,-10003.051758f,-10002.791016f,-10002.771484f,-10002.488281f,-10002.244141f,-10002.687500f,-10003.031250f,-10003.271484f,-10003.107422f,-10002.179688f,-10001.077148f,-10000.125000f,
-9999.554688f,-9999.911133f,-10000.875000f,-10002.539062f,-10004.197266f,-10005.730469f,-10006.156250f,-10006.882812f,-10007.503906f,-10008.082031f,-10008.662109f,-10010.037109f,-10010.647461f,-10010.750977f,-10009.882812f,-10008.535156f,-10007.173828f,-10006.582031f,-10006.412109f,-10007.045898f,
-10008.252930f,-10008.879883f,-10009.580078f,-10009.475586f,-10009.277344f,-10008.688477f,-10008.261719f,-10007.903320f,-10007.686523f,-10007.547852f,-10006.770508f,-10005.235352f,-10003.078125f,-10001.153320f,-9999.483398f,-9998.880859f,-9998.852539f,-9999.083008f,-9999.769531f,-10000.184570f,
-10000.111328f,-9999.967773f,-9999.383789f,-9999.065430f,-9998.868164f,-9998.530273f,-9998.394531f,-9998.337891f,-9997.325195f,-9995.665039f,-9993.849609f,-9992.623047f,-9991.817383f,-9992.180664f,-9993.317383f,-9994.730469f,-9996.215820f,-9997.152344f,-9997.914062f,-9998.485352f,
-9998.970703f,-9999.710938f,-10000.566406f,-10001.752930f,-10002.855469f,-10003.249023f,-10002.909180f,-10002.033203f,-10001.083008f,-10000.612305f,-10000.856445f,-10001.787109f,-10003.384766f,-10005.012695f,-10005.946289f,-10007.043945f,-10007.541016f,-10007.975586f,-10008.180664f,-10008.260742f,
-10008.605469f,-10009.220703f,-10009.464844f,-10009.395508f,-10008.009766f,-10006.154297f,-10004.241211f,-10002.885742f,-10002.422852f,-10002.688477f,-10003.614258f,-10004.271484f,-9993.116211f,-10004.927734f,-10006.026367f,-10003.314453f,-10003.164062f,-10002.832031f,-10003.118164f,-10005.973633f,
-10006.916992f,-10004.454102f,-10007.165039f,-10006.520508f,-10006.933594f,-10007.231445f,-10007.867188f,-10008.091797f,-10007.755859f,-10006.700195f,-10004.824219f,-10002.973633f,-10001.263672f,-10000.694336f,-10000.916016f,-10001.653320f,-10002.411133f,-10002.697266f,-10003.100586f,-10002.931641f,
-10002.531250f,-10002.634766f,-10002.791016f,-10003.159180f,-10003.423828f,-10003.045898f,-10002.299805f,-10001.180664f,-9999.963867f,-9999.416016f,-9999.605469f,-10000.829102f,-10002.613281f,-10004.640625f,-10005.731445f,-10006.910156f,-10007.632812f,-10008.067383f,-10008.286133f,-10009.467773f,
-10010.284180f,-10010.908203f,-10010.690430f,-10009.755859f,-10008.313477f,-10007.071289f,-10006.025391f,-10005.929688f,-10006.819336f,-10007.650391f,-10008.954102f,-10009.516602f,-10009.766602f,-10009.379883f,-10008.837891f,-10008.219727f,-10007.951172f,-10008.039062f,-10007.781250f,-10006.728516f,
-10005.035156f,-10003.139648f,-10000.995117f,-9999.563477f,-9998.691406f,-9998.500000f,-9999.196289f,-10000.006836f,-10000.576172f,-10000.946289f,-10000.648438f,-10000.257812f,-9999.780273f,-9999.184570f,-9998.897461f,-9999.079102f,-9998.503906f,-9997.387695f,-9995.907227f,-9994.333984f,
-9992.675781f,-9992.041016f,-9992.303711f,-9993.403320f,-9995.077148f,-9996.461914f,-9997.693359f,-9998.574219f,-9999.106445f,-9999.621094f,-10000.115234f,-10000.894531f,-10002.049805f,-10002.888672f,-10003.096680f,-10002.796875f,-10001.896484f,-10000.912109f,-10000.277344f,-10000.351562f,
-10001.281250f,-10002.878906f,-10004.364258f,-10006.070312f,-10007.145508f,-10007.985352f,-10008.305664f,-10008.232422f,-10008.266602f,-10008.619141f,-10008.963867f,-10009.307617f,-10008.496094f,-10007.149414f,-10005.469727f,-10003.770508f,-10002.567383f,-10001.922852f,-10002.208984f,-10002.696289f,
-9992.901367f,-10006.097656f,-10007.458984f,-10004.578125f,-10003.589844f,-10002.473633f,-10002.318359f,-10005.234375f,-10006.123047f,-10004.320312f,-10007.427734f,-10006.866211f,-10007.099609f,-10007.094727f,-10007.521484f,-10007.915039f,-10007.982422f,-10007.557617f,-10006.108398f,-10004.254883f,
-10001.957031f,-10000.494141f,-9999.841797f,-10000.098633f,-10000.848633f,-10001.498047f,-10002.396484f,-10002.601562f,-10002.374023f,-10002.252930f,-10002.145508f,-10002.327148f,-10002.750000f,-10002.831055f,-10002.657227f,-10001.887695f,-10000.532227f,-9999.366211f,-9998.731445f,-9999.164062f,
-10000.555664f,-10002.640625f,-10004.295898f,-10006.093750f,-10007.281250f,-10008.009766f,-10008.146484f,-10009.089844f,-10009.782227f,-10010.641602f,-10011.023438f,-10010.794922f,-10009.747070f,-10008.353516f,-10006.719727f,-10005.772461f,-10005.889648f,-10006.403320f,-10007.795898f,-10008.882812f,
-10009.670898f,-10009.811523f,-10009.483398f,-10008.720703f,-10008.345703f,-10008.378906f,-10008.331055f,-10007.653320f,-10006.578125f,-10005.070312f,-10002.924805f,-10000.972656f,-9999.258789f,-9998.261719f,-9998.463867f,-9999.176758f,-10000.116211f,-10000.980469f,-10001.175781f,-10001.021484f,
-10000.528320f,-9999.708984f,-9999.060547f,-9999.146484f,-9998.714844f,-9998.118164f,-9997.254883f,-9995.879883f,-9993.831055f,-9992.422852f,-9991.671875f,-9991.947266f,-9993.229492f,-9994.687500f,-9996.302734f,-9997.663086f,-9998.597656f,-9999.247070f,-9999.627930f,-10000.039062f,
-10000.984375f,-10001.936523f,-10002.579102f,-10003.009766f,-10002.629883f,-10001.750000f,-10000.737305f,-10000.060547f,-10000.074219f,-10001.105469f,-10002.596680f,-10004.633789f,-10006.333008f,-10007.789062f,-10008.631836f,-10008.788086f,-10008.728516f,-10008.750000f,-10008.853516f,-10009.259766f,
-10008.806641f,-10008.028320f,-10006.895508f,-10005.391602f,-10003.885742f,-10002.451172f,-10001.876953f,-10001.645508f,-9992.888672f,-10006.813477f,-10008.646484f,-10006.133789f,-10005.045898f,-10003.193359f,-10002.298828f,-10004.454102f,-10005.086914f,-10003.822266f,-10007.239258f,-10007.088867f,
-10007.395508f,-10007.204102f,-10007.226562f,-10007.416992f,-10007.557617f,-10007.594727f,-10006.780273f,-10005.500000f,-10003.299805f,-10001.423828f,-9999.965820f,-9999.467773f,-9999.749023f,-10000.355469f,-10001.514648f,-10002.145508f,-10002.310547f,-10002.198242f,-10001.875000f,-10001.694336f,
-10001.866211f,-10002.073242f,-10002.335938f,-10002.172852f,-10001.226562f,-10000.041992f,-9998.932617f,-9998.577148f,-9999.219727f,-10000.791016f,-10002.482422f,-10004.618164f,-10006.292969f,-10007.468750f,-10007.810547f,-10008.570312f,-10008.876953f,-10009.544922f,-10010.158203f,-10010.525391f,
-10010.189453f,-10009.227539f,-10007.648438f,-10006.241211f,-10005.572266f,-10005.411133f,-10006.366211f,-10007.512695f,-10008.636719f,-10009.347656f,-10009.510742f,-10008.867188f,-10008.465820f,-10008.289062f,-10008.108398f,-10007.622070f,-10007.009766f,-10006.109375f,-10004.502930f,-10002.649414f,
-10000.592773f,-9998.926758f,-9998.420898f,-9998.570312f,-9999.403320f,-10000.430664f,-10001.084961f,-10001.314453f,-10001.109375f,-10000.291992f,-9999.302734f,-9999.044922f,-9998.477539f,-9998.105469f,-9997.774414f,-9997.033203f,-9995.229492f,-9993.688477f,-9992.212891f,-9991.579102f,
-9992.013672f,-9993.032227f,-9994.570312f,-9996.209961f,-9997.555664f,-9998.525391f,-9999.049805f,-9999.283203f,-9999.820312f,-10000.471680f,-10001.149414f,-10002.037109f,-10002.282227f,-10002.054688f,-10001.291992f,-10000.376953f,-9999.668945f,-9999.896484f,-10000.773438f,-10002.623047f,
-10004.544922f,-10006.450195f,-10007.897461f,-10008.517578f,-10008.682617f,-10008.506836f,-10008.224609f,-10008.375000f,-10007.975586f,-10007.567383f,-10007.074219f,-10006.230469f,-10005.075195f,-10003.434570f,-10002.333984f,-10001.299805f,-9992.869141f,-10006.936523f,-10009.107422f,-10007.151367f,
-10006.214844f,-10004.151367f,-10002.702148f,-10003.876953f,-10004.051758f,-10002.826172f,-10006.372070f,-10006.708008f,-10007.476562f,-10007.506836f,-10007.319336f,-10007.217773f,-10007.140625f,-10007.294922f,-10006.931641f,-10006.351562f,-10004.635742f,-10002.775391f,-10000.782227f,-9999.529297f,
-9999.046875f,-9999.195312f,-10000.291992f,-10001.263672f,-10001.977539f,-10002.234375f,-10002.041016f,-10001.595703f,-10001.372070f,-10001.366211f,-10001.659180f,-10001.983398f,-10001.562500f,-10000.697266f,-9999.466797f,-9998.587891f,-9998.409180f,-9999.168945f,-10000.394531f,-10002.542969f,
-10004.615234f,-10006.388672f,-10007.297852f,-10008.264648f,-10008.291016f,-10008.598633f,-10009.075195f,-10009.658203f,-10009.890625f,-10009.557617f,-10008.457031f,-10007.000977f,-10005.811523f,-10004.902344f,-10005.088867f,-10005.855469f,-10007.038086f,-10008.222656f,-10009.044922f,-10008.910156f,
-10008.772461f,-10008.509766f,-10008.109375f,-10007.555664f,-10007.039062f,-10006.555664f,-10005.520508f,-10004.076172f,-10002.116211f,-10000.054688f,-9998.875977f,-9998.266602f,-9998.581055f,-9999.413086f,-10000.317383f,-10001.012695f,-10001.380859f,-10000.946289f,-9999.962891f,-9999.453125f,
-9998.583008f,-9998.092773f,-9997.977539f,-9997.825195f,-9996.505859f,-9995.302734f,-9993.587891f,-9992.211914f,-9991.723633f,-9991.952148f,-9992.990234f,-9994.592773f,-9996.256836f,-9997.687500f,-9998.712891f,-9999.182617f,-9999.497070f,-9999.722656f,-10000.120117f,-10000.960938f,
-10001.587891f,-10002.029297f,-10001.855469f,-10001.117188f,-10000.089844f,-9999.610352f,-9999.625000f,-10000.848633f,-10002.577148f,-10004.651367f,-10006.660156f,-10007.927734f,-10008.719727f,-10008.782227f,-10008.337891f,-10008.172852f,-10007.524414f,-10007.104492f,-10006.965820f,-10006.719727f,
-10006.216797f,-10004.845703f,-10003.605469f,-10001.969727f,-9993.006836f,-10006.667969f,-10009.425781f,-10007.689453f,-10007.281250f,-10005.270508f,-10003.474609f,-10003.756836f,-10003.389648f,-10001.844727f,-10004.944336f,-10005.494141f,-10006.732422f,-10007.198242f,-10007.142578f,-10006.963867f,
-10006.592773f,-10006.581055f,-10006.353516f,-10006.256836f,-10005.127930f,-10003.634766f,-10001.575195f,-9999.851562f,-9998.686523f,-9998.201172f,-9998.876953f,-9999.860352f,-10000.973633f,-10001.709961f,-10001.949219f,-10001.569336f,-10001.120117f,-10000.818359f,-10000.840820f,-10001.339844f,
-10001.276367f,-10000.871094f,-9999.818359f,-9998.821289f,-9998.069336f,-9998.069336f,-9998.549805f,-10000.307617f,-10002.385742f,-10004.536133f,-10006.080078f,-10007.523438f,-10007.638672f,-10007.731445f,-10007.945312f,-10008.394531f,-10008.875000f,-10009.040039f,-10008.565430f,-10007.450195f,
-10006.174805f,-10004.785156f,-10004.256836f,-10004.406250f,-10005.265625f,-10006.591797f,-10007.922852f,-10008.437500f,-10008.810547f,-10008.804688f,-10008.427734f,-10007.830078f,-10007.220703f,-10006.851562f,-10006.234375f,-10005.253906f,-10003.704102f,-10001.695312f,-10000.158203f,-9998.950195f,
-9998.597656f,-9998.911133f,-9999.688477f,-10000.606445f,-10001.461914f,-10001.586914f,-10000.954102f,-10000.475586f,-9999.435547f,-9998.633789f,-9998.410156f,-9998.479492f,-9997.515625f,-9996.781250f,-9995.208008f,-9993.491211f,-9992.272461f,-9991.658203f,-9991.901367f,-9993.037109f,
-9994.618164f,-9996.291992f,-9997.839844f,-9998.795898f,-9999.256836f,-9999.281250f,-9999.364258f,-9999.842773f,-10000.516602f,-10001.271484f,-10001.634766f,-10001.366211f,-10000.462891f,-9999.672852f,-9998.937500f,-9999.416016f,-10000.599609f,-10002.420898f,-10004.623047f,-10006.369141f,
-10007.896484f,-10008.458984f,-10008.311523f,-10008.087891f,-10007.210938f,-10006.559570f,-10006.415039f,-10006.452148f,-10006.515625f,-10005.665039f,-10004.682617f,-10002.875977f,-9992.987305f,-10006.048828f,-10009.327148f,-10007.821289f,-10008.143555f,-10006.285156f,-10004.487305f,-10004.375977f,
-10003.376953f,-10001.137695f,-10003.727539f,-10004.159180f,-10005.762695f,-10006.753906f,-10007.135742f,-10007.182617f,-10006.736328f,-10006.485352f,-10006.217773f,-10006.345703f,-10005.737305f,-10004.753906f,-10002.948242f,-10001.044922f,-9999.345703f,-9998.135742f,-9998.096680f,-9998.701172f,
-9999.888672f,-10001.000977f,-10001.755859f,-10001.699219f,-10001.311523f,-10000.811523f,-10000.476562f,-10000.903320f,-10001.023438f,-10001.031250f,-10000.267578f,-9999.448242f,-9998.407227f,-9997.845703f,-9997.483398f,-9998.562500f,-10000.295898f,-10002.479492f,-10004.556641f,-10006.586914f,
-10007.151367f,-10007.337891f,-10007.395508f,-10007.610352f,-10008.112305f,-10008.584961f,-10008.705078f,-10008.108398f,-10007.108398f,-10005.539062f,-10004.459961f,-10003.914062f,-10004.180664f,-10005.224609f,-10006.721680f,-10007.773438f,-10008.708008f,-10009.188477f,-10009.065430f,-10008.568359f,
-10007.790039f,-10007.323242f,-10006.877930f,-10006.214844f,-10005.163086f,-10003.454102f,-10001.817383f,-10000.222656f,-9999.214844f,-9998.808594f,-9999.062500f,-9999.874023f,-10000.963867f,-10001.597656f,-10001.485352f,-10001.342773f,-10000.416992f,-9999.331055f,-9998.869141f,-9998.909180f,
-9998.142578f,-9997.828125f,-9996.703125f,-9994.999023f,-9993.415039f,-9992.047852f,-9991.474609f,-9991.821289f,-9992.974609f,-9994.568359f,-9996.487305f,-9998.030273f,-9998.925781f,-9999.043945f,-9999.019531f,-9999.101562f,-9999.633789f,-10000.452148f,-10001.170898f,-10001.453125f,
-10000.988281f,-10000.203125f,-9999.042969f,-9998.853516f,-9999.330078f,-10000.551758f,-10002.570312f,-10004.491211f,-10006.593750f,-10007.829102f,-10008.316406f,-10008.386719f,-10007.510742f,-10006.628906f,-10006.333984f,-10006.358398f,-10006.785156f,-10006.428711f,-10005.910156f,-10004.274414f,
-9992.895508f,-10005.416992f,-10009.032227f,-10007.454102f,-10008.521484f,-10006.937500f,-10005.370117f,-10005.332031f,-10003.945312f,-10000.803711f,-10002.652344f,-10002.766602f,-10004.305664f,-10005.611328f,-10006.537109f,-10007.084961f,-10006.897461f,-10006.592773f,-10006.299805f,-10006.458008f,
-10006.213867f,-10005.728516f,-10004.409180f,-10002.679688f,-10000.824219f,-9999.099609f,-9998.295898f,-9998.229492f,-9999.063477f,-10000.236328f,-10001.336914f,-10001.759766f,-10001.708008f,-10001.321289f,-10000.820312f,-10001.105469f,-10001.253906f,-10001.467773f,-10000.965820f,-10000.493164f,
-9999.458008f,-9998.631836f,-9997.587891f,-9997.858398f,-9998.890625f,-10000.625977f,-10002.805664f,-10005.198242f,-10006.364258f,-10006.951172f,-10007.130859f,-10007.257812f,-10007.699219f,-10008.263672f,-10008.752930f,-10008.667969f,-10008.191406f,-10006.773438f,-10005.492188f,-10004.427734f,
-10004.000000f,-10004.435547f,-10005.640625f,-10006.870117f,-10008.194336f,-10009.262695f,-10009.612305f,-10009.503906f,-10008.776367f,-10008.237305f,-10007.843750f,-10007.303711f,-10006.664062f,-10005.399414f,-10003.933594f,-10002.272461f,-10000.845703f,-9999.723633f,-9999.239258f,-9999.547852f,
-10000.460938f,-10001.316406f,-10001.661133f,-10002.029297f,-10001.548828f,-10000.474609f,-9999.896484f,-9999.813477f,-9999.057617f,-9998.958984f,-9998.289062f,-9996.866211f,-9995.332031f,-9993.581055f,-9992.340820f,-9991.824219f,-9992.196289f,-9993.283203f,-9995.161133f,-9997.008789f,
-9998.355469f,-9998.837891f,-9999.044922f,-9998.968750f,-9999.358398f,-10000.007812f,-10000.788086f,-10001.448242f,-10001.519531f,-10000.991211f,-9999.765625f,-9999.209961f,-9998.990234f,-9999.416992f,-10000.873047f,-10002.488281f,-10004.759766f,-10006.445312f,-10007.667969f,-10008.328125f,
-10007.843750f,-10007.009766f,-10006.649414f,-10006.490234f,-10006.994141f,-10006.925781f,-10006.857422f,-10005.615234f,-9992.929688f,-10004.941406f,-10008.555664f,-10006.931641f,-10008.248047f,-10007.238281f,-10006.083008f,-10006.434570f,-10004.989258f,-10001.334961f,-10002.454102f,-10002.057617f,
-10003.125000f,-10004.340820f,-10005.663086f,-10006.746094f,-10007.025391f,-10006.801758f,-10006.493164f,-10006.452148f,-10006.334961f,-10006.198242f,-10005.430664f,-10004.131836f,-10002.495117f,-10000.617188f,-9999.198242f,-9998.342773f,-9998.465820f,-9999.255859f,-10000.368164f,-10001.173828f,
-10001.564453f,-10001.505859f,-10001.011719f,-10001.130859f,-10001.187500f,-10001.440430f,-10001.151367f,-10001.085938f,-10000.382812f,-9999.740234f,-9998.436523f,-9998.046875f,-9998.281250f,-9999.297852f,-10001.107422f,-10003.506836f,-10005.258789f,-10006.373047f,-10006.908203f,-10007.104492f,
-10007.489258f,-10007.973633f,-10008.607422f,-10008.896484f,-10009.040039f,-10008.106445f,-10007.043945f,-10005.847656f,-10004.874023f,-10004.587891f,-10005.137695f,-10006.114258f,-10007.461914f,-10009.001953f,-10009.862305f,-10010.274414f,-10009.773438f,-10009.201172f,-10008.743164f,-10008.153320f,
-10007.750977f,-10006.958984f,-10005.879883f,-10004.514648f,-10003.048828f,-10001.471680f,-10000.246094f,-9999.822266f,-10000.147461f,-10000.753906f,-10001.274414f,-10002.092773f,-10002.145508f,-10001.264648f,-10000.671875f,-10000.365234f,-9999.451172f,-9999.292969f,-9998.978516f,-9997.939453f,
-9996.820312f,-9995.093750f,-9993.566406f,-9992.349609f,-9991.900391f,-9992.182617f,-9993.584961f,-9995.346680f,-9997.012695f,-9997.982422f,-9998.652344f,-9998.701172f,-9999.037109f,-9999.436523f,-10000.063477f,-10000.927734f,-10001.478516f,-10001.466797f,-10000.633789f,-10000.176758f,
-9999.566406f,-9999.262695f,-9999.994141f,-10000.968750f,-10002.896484f,-10004.702148f,-10006.504883f,-10007.823242f,-10007.961914f,-10007.426758f,-10007.090820f,-10006.744141f,-10007.078125f,-10007.084961f,-10007.381836f,-10006.646484f,-9992.795898f,-10004.958008f,-10008.401367f,-10006.686523f,
-10007.958008f,-10007.230469f,-10006.541016f,-10007.331055f,-10006.117188f,-10002.232422f,-10002.896484f,-10001.868164f,-10002.206055f,-10002.992188f,-10004.457031f,-10006.037109f,-10006.972656f,-10007.148438f,-10007.030273f,-10006.791992f,-10006.557617f,-10006.441406f,-10006.003906f,-10005.107422f,
-10003.878906f,-10002.156250f,-10000.398438f,-9998.890625f,-9998.222656f,-9998.369141f,-9999.156250f,-10000.153320f,-10001.003906f,-10001.509766f,-10001.356445f,-10001.483398f,-10001.418945f,-10001.491211f,-10001.145508f,-10001.274414f,-10000.917969f,-10000.603516f,-9999.415039f,-9998.647461f,
-9998.233398f,-9998.433594f,-9999.584961f,-10001.630859f,-10003.733398f,-10005.403320f,-10006.539062f,-10007.151367f,-10007.626953f,-10007.967773f,-10008.441406f,-10008.793945f,-10009.326172f,-10008.876953f,-10008.237305f,-10007.227539f,-10005.973633f,-10005.107422f,-10004.904297f,-10005.325195f,
-10006.394531f,-10008.172852f,-10009.508789f,-10010.568359f,-10010.584961f,-10010.210938f,-10009.728516f,-10008.956055f,-10008.494141f,-10007.963867f,-10007.186523f,-10006.278320f,-10005.067383f,-10003.355469f,-10001.654297f,-10000.520508f,-10000.137695f,-10000.223633f,-10000.584961f,-10001.698242f,
-10002.385742f,-10002.009766f,-10001.741211f,-10001.389648f,-10000.242188f,-9999.775391f,-9999.483398f,-9998.664062f,-9997.990234f,-9996.594727f,-9995.128906f,-9993.534180f,-9992.368164f,-9991.868164f,-9992.527344f,-9993.816406f,-9995.551758f,-9996.951172f,-9998.283203f,-9998.825195f,
-9999.368164f,-9999.585938f,-9999.893555f,-10000.623047f,-10001.343750f,-10001.676758f,-10001.385742f,-10001.313477f,-10000.628906f,-9999.873047f,-9999.958008f,-10000.190430f,-10001.451172f,-10003.004883f,-10005.048828f,-10006.956055f,-10007.891602f,-10007.994141f,-10007.968750f,-10007.556641f,
-10007.553711f,-10007.342773f,-10007.665039f,-10007.250977f,-9993.150391f,-10005.166016f,-10008.239258f,-10006.609375f,-10007.582031f,-10006.906250f,-10006.429688f,-10007.828125f,-10007.071289f,-10003.333008f,-10004.019531f,-10002.481445f,-10001.976562f,-10001.956055f,-10002.994141f,-10004.583984f,
-10006.019531f,-10006.744141f,-10007.084961f,-10006.900391f,-10006.599609f,-10006.359375f,-10006.089844f,-10005.601562f,-10004.954102f,-10003.791992f,-10002.211914f,-10000.482422f,-9999.180664f,-9998.510742f,-9998.550781f,-9999.244141f,-10000.150391f,-10001.104492f,-10001.477539f,-10001.824219f,
-10001.823242f,-10001.709961f,-10001.204102f,-10001.288086f,-10001.202148f,-10001.267578f,-10000.558594f,-9999.795898f,-9999.056641f,-9998.573242f,-9998.765625f,-9999.907227f,-10001.707031f,-10003.457031f,-10005.075195f,-10006.237305f,-10007.066406f,-10007.402344f,-10007.671875f,-10007.870117f,
-10008.509766f,-10008.503906f,-10008.417969f,-10007.964844f,-10006.903320f,-10005.863281f,-10005.080078f,-10004.759766f,-10005.155273f,-10006.615234f,-10007.983398f,-10009.477539f,-10010.139648f,-10010.266602f,-10010.023438f,-10009.190430f,-10008.582031f,-10008.138672f,-10007.617188f,-10007.158203f,
-10006.461914f,-10005.044922f,-10003.356445f,-10001.796875f,-10000.793945f,-10000.098633f,-9999.852539f,-10000.741211f,-10001.718750f,-10001.905273f,-10002.170898f,-10002.069336f,-10000.958008f,-10000.158203f,-9999.680664f,-9998.893555f,-9998.620117f,-9997.747070f,-9996.750000f,-9995.279297f,
-9993.812500f,-9992.722656f,-9992.455078f,-9992.790039f,-9993.955078f,-9995.300781f,-9997.035156f,-9998.196289f,-9999.130859f,-9999.456055f,-9999.516602f,-9999.919922f,-10000.488281f,-10000.950195f,-10001.171875f,-10001.643555f,-10001.267578f,-10000.489258f,-10000.207031f,-9999.812500f,
-10000.152344f,-10000.964844f,-10002.539062f,-10004.525391f,-10006.051758f,-10006.937500f,-10007.515625f,-10007.479492f,-10007.322266f,-10006.889648f,-10007.046875f,-10006.749023f,-9993.124023f,-10005.499023f,-10008.266602f,-10006.786133f,-10007.347656f,-10006.443359f,-10006.003906f,-10007.790039f,
-10007.498047f,-10004.102539f,-10005.159180f,-10003.603516f,-10002.473633f,-10001.612305f,-10001.963867f,-10003.266602f,-10005.002930f,-10006.291992f,-10007.237305f,-10007.343750f,-10007.032227f,-10006.567383f,-10006.187500f,-10005.875977f,-10005.653320f,-10005.108398f,-10003.985352f,-10002.393555f,
-10000.768555f,-9999.395508f,-9998.564453f,-9998.654297f,-9999.305664f,-10000.511719f,-10001.419922f,-10002.144531f,-10002.350586f,-10002.072266f,-10001.295898f,-10001.101562f,-10001.035156f,-10001.334961f,-10001.162109f,-10000.667969f,-9999.987305f,-9999.150391f,-9998.558594f,-9998.706055f,
-9999.869141f,-10001.399414f,-10003.338867f,-10005.138672f,-10006.487305f,-10007.010742f,-10007.154297f,-10007.065430f,-10007.558594f,-10007.746094f,-10008.094727f,-10008.279297f,-10007.708984f,-10006.916016f,-10005.914062f,-10004.981445f,-10004.673828f,-10005.572266f,-10006.723633f,-10008.439453f,
-10009.733398f,-10010.506836f,-10010.684570f,-10009.936523f,-10009.127930f,-10008.553711f,-10008.061523f,-10007.856445f,-10007.669922f,-10006.730469f,-10005.435547f,-10003.811523f,-10002.447266f,-10001.041016f,-10000.040039f,-10000.422852f,-10001.376953f,-10002.067383f,-10002.925781f,-10003.299805f,
-10002.375000f,-10001.284180f,-10000.448242f,-9999.400391f,-9999.199219f,-9998.709961f,-9998.242188f,-9997.146484f,-9995.737305f,-9994.383789f,-9993.332031f,-9992.683594f,-9992.977539f,-9993.945312f,-9995.833008f,-9997.584961f,-9999.041992f,-9999.685547f,-9999.698242f,-9999.719727f,
-9999.914062f,-10000.238281f,-10000.726562f,-10001.657227f,-10001.793945f,-10001.333008f,-10001.055664f,-10000.379883f,-9999.967773f,-9999.957031f,-10000.773438f,-10002.472656f,-10004.371094f,-10006.002930f,-10007.299805f,-10007.912109f,-10007.849609f,-10007.228516f,-10007.094727f,-10006.659180f,
-9992.974609f,-10005.640625f,-10008.380859f,-10007.167969f,-10007.660156f,-10006.365234f,-10005.705078f,-10007.259766f,-10007.323242f,-10004.304688f,-10005.884766f,-10004.963867f,-10003.698242f,-10002.208984f,-10001.643555f,-10002.154297f,-10003.623047f,-10005.158203f,-10006.688477f,-10007.467773f,
-10007.577148f,-10007.169922f,-10006.651367f,-10006.271484f,-10006.172852f,-10006.071289f,-10005.505859f,-10004.405273f,-10002.959961f,-10001.270508f,-9999.696289f,-9998.933594f,-9998.784180f,-9999.702148f,-10000.833984f,-10002.014648f,-10002.741211f,-10002.742188f,-10001.932617f,-10001.411133f,
-10001.111328f,-10001.315430f,-10001.449219f,-10001.321289f,-10001.042969f,-10000.340820f,-9999.380859f,-9998.708008f,-9998.872070f,-9999.588867f,-10001.290039f,-10003.439453f,-10005.415039f,-10006.506836f,-10006.972656f,-10006.834961f,-10007.088867f,-10007.155273f,-10007.590820f,-10008.210938f,
-10008.235352f,-10008.052734f,-10007.351562f,-10006.235352f,-10005.340820f,-10005.405273f,-10005.899414f,-10007.325195f,-10008.863281f,-10010.247070f,-10011.099609f,-10010.858398f,-10010.162109f,-10009.412109f,-10008.734375f,-10008.374023f,-10008.353516f,-10007.827148f,-10007.123047f,-10005.833008f,
-10004.583984f,-10002.735352f,-10000.971680f,-10000.446289f,-10000.773438f,-10001.420898f,-10002.625000f,-10003.610352f,-10003.279297f,-10002.363281f,-10001.347656f,-9999.881836f,-9999.371094f,-9998.909180f,-9998.781250f,-9998.149414f,-9997.151367f,-9996.055664f,-9994.644531f,-9993.237305f,
-9992.486328f,-9992.539062f,-9993.942383f,-9995.833008f,-9997.764648f,-9999.050781f,-9999.505859f,-9999.547852f,-9999.424805f,-9999.437500f,-9999.729492f,-10000.770508f,-10001.337891f,-10001.404297f,-10001.545898f,-10001.107422f,-10000.392578f,-9999.727539f,-9999.525391f,-10000.422852f,
-10002.072266f,-10003.985352f,-10005.861328f,-10007.460938f,-10007.989258f,-10007.698242f,-10007.450195f,-10006.759766f,-9992.867188f,-10005.285156f,-10007.968750f,-10007.220703f,-10007.919922f,-10006.434570f,-10005.561523f,-10006.915039f,-10006.965820f,-10004.006836f,-10006.076172f,-10005.904297f,
-10005.003906f,-10003.400391f,-10002.214844f,-10001.855469f,-10002.671875f,-10003.990234f,-10005.750000f,-10007.154297f,-10007.897461f,-10007.880859f,-10007.411133f,-10006.943359f,-10006.676758f,-10006.642578f,-10006.411133f,-10005.790039f,-10004.801758f,-10003.264648f,-10001.391602f,-9999.943359f,
-9998.876953f,-9999.071289f,-9999.946289f,-10001.303711f,-10002.533203f,-10003.057617f,-10002.502930f,-10001.917969f,-10001.386719f,-10001.268555f,-10001.323242f,-10001.392578f,-10001.523438f,-10001.239258f,-10000.408203f,-9999.437500f,-9998.784180f,-9998.555664f,-9999.575195f,-10001.574219f,
-10003.877930f,-10005.598633f,-10006.677734f,-10006.827148f,-10007.050781f,-10006.933594f,-10007.170898f,-10007.850586f,-10008.259766f,-10008.654297f,-10008.528320f,-10007.666016f,-10006.592773f,-10005.988281f,-10005.740234f,-10006.505859f,-10007.776367f,-10009.396484f,-10010.776367f,-10011.213867f,
-10010.958008f,-10010.243164f,-10009.420898f,-10008.710938f,-10008.499023f,-10008.106445f,-10007.846680f,-10007.061523f,-10006.283203f,-10004.445312f,-10002.339844f,-10001.022461f,-10000.499023f,-10000.630859f,-10001.694336f,-10003.051758f,-10003.422852f,-10003.020508f,-10002.239258f,-10000.570312f,
-9999.666992f,-9998.916016f,-9998.789062f,-9998.416992f,-9997.861328f,-9997.257812f,-9995.985352f,-9994.396484f,-9992.891602f,-9992.008789f,-9992.513672f,-9993.996094f,-9996.003906f,-9997.826172f,-9998.967773f,-9999.402344f,-9999.346680f,-9999.172852f,-9999.053711f,-9999.833984f,
-10000.556641f,-10000.979492f,-10001.606445f,-10001.684570f,-10001.185547f,-10000.358398f,-9999.455078f,-9999.479492f,-10000.455078f,-10002.119141f,-10004.106445f,-10006.469727f,-10007.796875f,-10008.183594f,-10008.219727f,-10007.458984f,-9992.720703f,-10004.837891f,-10007.412109f,-10006.784180f,
-10007.585938f,-10006.428711f,-10005.400391f,-10006.789062f,-10006.588867f,-10003.644531f,-10006.009766f,-10006.545898f,-10006.397461f,-10005.178711f,-10003.792969f,-10002.708008f,-10002.566406f,-10003.134766f,-10004.465820f,-10006.038086f,-10007.221680f,-10007.748047f,-10007.595703f,-10007.240234f,
-10006.847656f,-10006.711914f,-10006.722656f,-10006.578125f,-10006.188477f,-10005.221680f,-10003.538086f,-10001.770508f,-9999.898438f,-9999.102539f,-9999.161133f,-10000.143555f,-10001.464844f,-10002.428711f,-10002.324219f,-10002.030273f,-10001.560547f,-10001.251953f,-10001.073242f,-10001.265625f,
-10001.769531f,-10002.036133f,-10001.678711f,-10000.961914f,-9999.865234f,-9998.773438f,-9998.777344f,-9999.971680f,-10001.988281f,-10003.950195f,-10005.588867f,-10006.223633f,-10006.720703f,-10006.620117f,-10006.710938f,-10007.268555f,-10007.962891f,-10008.828125f,-10009.410156f,-10009.157227f,
-10008.339844f,-10007.421875f,-10006.577148f,-10006.415039f,-10006.879883f,-10008.122070f,-10009.534180f,-10010.499023f,-10010.821289f,-10010.451172f,-10009.783203f,-10008.905273f,-10008.429688f,-10008.084961f,-10008.166992f,-10007.961914f,-10007.846680f,-10006.454102f,-10004.522461f,-10002.734375f,
-10001.313477f,-10000.536133f,-10000.864258f,-10001.965820f,-10002.683594f,-10002.832031f,-10002.546875f,-10001.058594f,-9999.988281f,-9998.912109f,-9998.595703f,-9998.360352f,-9998.198242f,-9998.206055f,-9997.472656f,-9996.229492f,-9994.448242f,-9992.846680f,-9992.245117f,-9992.725586f,
-9994.151367f,-9995.990234f,-9997.618164f,-9998.578125f,-9998.930664f,-9998.914062f,-9998.557617f,-9998.980469f,-9999.683594f,-10000.356445f,-10001.413086f,-10002.099609f,-10002.195312f,-10001.685547f,-10000.516602f,-9999.825195f,-9999.843750f,-10000.626953f,-10002.090820f,-10004.570312f,
-10006.492188f,-10007.668945f,-10008.301758f,-10007.873047f,-9992.683594f,-10004.422852f,-10006.844727f,-10006.063477f,-10007.308594f,-10006.553711f,-10005.647461f,-10006.958984f,-10006.636719f,-10003.341797f,-10005.546875f,-10006.477539f,-10007.023438f,-10006.443359f,-10005.387695f,-10004.056641f,
-10003.138672f,-10002.756836f,-10003.288086f,-10004.556641f,-10005.987305f,-10007.135742f,-10007.613281f,-10007.608398f,-10007.208008f,-10006.826172f,-10006.747070f,-10006.722656f,-10006.711914f,-10006.409180f,-10005.319336f,-10003.765625f,-10001.595703f,-9999.999023f,-9999.078125f,-9999.278320f,
-10000.318359f,-10001.604492f,-10002.047852f,-10002.316406f,-10002.178711f,-10001.805664f,-10001.230469f,-10001.183594f,-10001.627930f,-10002.126953f,-10002.270508f,-10002.116211f,-10001.125000f,-9999.640625f,-9998.718750f,-9998.833984f,-10000.080078f,-10001.850586f,-10003.844727f,-10005.105469f,
-10006.133789f,-10006.278320f,-10006.323242f,-10006.558594f,-10007.115234f,-10008.004883f,-10008.990234f,-10009.399414f,-10009.128906f,-10008.390625f,-10007.397461f,-10006.521484f,-10006.085938f,-10006.587891f,-10007.628906f,-10008.879883f,-10009.871094f,-10010.159180f,-10009.947266f,-10009.149414f,
-10008.424805f,-10007.911133f,-10007.941406f,-10008.029297f,-10008.441406f,-10007.702148f,-10006.414062f,-10004.711914f,-10002.883789f,-10001.242188f,-10000.610352f,-10001.005859f,-10001.730469f,-10002.322266f,-10002.749023f,-10001.856445f,-10000.936523f,-9999.617188f,-9998.958008f,-9998.484375f,
-9998.330078f,-9998.648438f,-9998.452148f,-9997.857422f,-9996.398438f,-9994.593750f,-9993.125977f,-9992.512695f,-9992.926758f,-9994.269531f,-9995.959961f,-9997.504883f,-9998.475586f,-9998.921875f,-9998.536133f,-9998.571289f,-9998.921875f,-9999.416016f,-10000.446289f,-10001.415039f,
-10002.116211f,-10002.208984f,-10001.372070f,-10000.485352f,-9999.741211f,-9999.530273f,-10000.100586f,-10002.091797f,-10004.166992f,-10006.047852f,-10007.451172f,-10007.722656f,-9992.785156f,-10004.504883f,-10006.534180f,-10005.326172f,-10006.419922f,-10006.168945f,-10005.574219f,-10007.141602f,
-10006.932617f,-10003.383789f,-10005.408203f,-10006.444336f,-10007.429688f,-10007.433594f,-10006.958008f,-10005.810547f,-10004.541016f,-10003.447266f,-10003.031250f,-10003.543945f,-10004.628906f,-10006.016602f,-10007.055664f,-10007.611328f,-10007.510742f,-10007.129883f,-10007.047852f,-10007.047852f,
-10007.167969f,-10007.333984f,-10006.793945f,-10005.668945f,-10003.606445f,-10001.608398f,-9999.861328f,-9999.092773f,-9999.330078f,-10000.393555f,-10001.006836f,-10001.787109f,-10002.195312f,-10002.131836f,-10001.435547f,-10001.280273f,-10001.547852f,-10002.055664f,-10002.426758f,-10002.799805f,
-10002.211914f,-10000.767578f,-9999.369141f,-9998.535156f,-9998.776367f,-9999.870117f,-10001.657227f,-10003.204102f,-10004.767578f,-10005.424805f,-10005.791016f,-10005.999023f,-10006.520508f,-10007.330078f,-10008.465820f,-10009.318359f,-10009.570312f,-10009.276367f,-10008.507812f,-10007.248047f,
-10006.105469f,-10005.756836f,-10006.012695f,-10006.992188f,-10008.208984f,-10009.072266f,-10009.457031f,-10009.125000f,-10008.494141f,-10008.004883f,-10008.000000f,-10008.191406f,-10008.888672f,-10008.670898f,-10008.060547f,-10006.805664f,-10004.959961f,-10002.834961f,-10001.375000f,-10000.755859f,
-10000.905273f,-10001.381836f,-10002.229492f,-10001.988281f,-10001.548828f,-10000.329102f,-9999.574219f,-9998.917969f,-9998.654297f,-9999.031250f,-9999.134766f,-9999.020508f,-9998.085938f,-9996.514648f,-9994.661133f,-9993.155273f,-9992.531250f,-9992.882812f,-9994.084961f,-9995.711914f,
-9997.117188f,-9998.162109f,-9998.175781f,-9998.202148f,-9998.401367f,-9998.743164f,-9999.616211f,-10000.614258f,-10001.666016f,-10002.287109f,-10002.010742f,-10001.323242f,-10000.264648f,-9999.329102f,-9998.994141f,-10000.102539f,-10001.795898f,-10003.807617f,-10005.752930f,-10006.753906f,
-9992.856445f,-10005.205078f,-10006.670898f,-10004.766602f,-10005.502930f,-10005.228516f,-10005.000000f,-10007.074219f,-10007.011719f,-10003.449219f,-10005.267578f,-10005.845703f,-10006.920898f,-10007.374023f,-10007.621094f,-10007.122070f,-10006.006836f,-10004.617188f,-10003.422852f,-10002.950195f,
-10003.280273f,-10004.462891f,-10005.874023f,-10007.026367f,-10007.515625f,-10007.358398f,-10007.364258f,-10007.282227f,-10007.304688f,-10007.695312f,-10007.646484f,-10007.181641f,-10005.714844f,-10003.846680f,-10001.684570f,-9999.987305f,-9999.189453f,-9999.519531f,-9999.895508f,-10000.965820f,
-10001.953125f,-10002.409180f,-10001.870117f,-10001.650391f,-10001.602539f,-10001.831055f,-10002.142578f,-10002.891602f,-10002.921875f,-10002.023438f,-10000.666992f,-9999.271484f,-9998.541016f,-9998.593750f,-9999.670898f,-10001.081055f,-10003.005859f,-10004.241211f,-10005.200195f,-10005.607422f,
-10006.125000f,-10006.728516f,-10007.726562f,-10008.766602f,-10009.508789f,-10009.851562f,-10009.743164f,-10008.589844f,-10007.182617f,-10006.117188f,-10005.394531f,-10005.625000f,-10006.595703f,-10007.782227f,-10008.758789f,-10009.035156f,-10008.815430f,-10008.452148f,-10008.346680f,-10008.400391f,
-10009.037109f,-10009.183594f,-10009.208984f,-10008.662109f,-10007.301758f,-10005.231445f,-10003.281250f,-10001.603516f,-10000.757812f,-10000.600586f,-10001.426758f,-10001.755859f,-10001.969727f,-10001.115234f,-10000.469727f,-9999.662109f,-9999.123047f,-9999.244141f,-9999.338867f,-9999.485352f,
-9999.185547f,-9998.256836f,-9996.585938f,-9994.699219f,-9993.163086f,-9992.324219f,-9992.463867f,-9993.660156f,-9995.140625f,-9996.705078f,-9997.401367f,-9997.660156f,-9997.858398f,-9998.050781f,-9998.612305f,-9999.336914f,-10000.460938f,-10001.429688f,-10001.848633f,-10001.775391f,
-10000.923828f,-9999.690430f,-9998.717773f,-9998.705078f,-9999.535156f,-10001.084961f,-10003.138672f,-10004.750977f,-9993.006836f,-10006.054688f,-10007.247070f,-10004.571289f,-10004.715820f,-10004.052734f,-10003.958008f,-10006.858398f,-10006.828125f,-10003.555664f,-10005.333008f,-10005.639648f,
-10006.453125f,-10006.931641f,-10007.611328f,-10007.797852f,-10007.212891f,-10006.062500f,-10004.561523f,-10003.281250f,-10002.629883f,-10003.164062f,-10004.471680f,-10005.983398f,-10007.048828f,-10007.384766f,-10007.688477f,-10007.626953f,-10007.452148f,-10007.713867f,-10007.791992f,-10007.773438f,
-10006.964844f,-10005.655273f,-10003.632812f,-10001.445312f,-9999.732422f,-9999.115234f,-9998.835938f,-9999.748047f,-10001.063477f,-10002.059570f,-10002.018555f,-10001.998047f,-10001.778320f,-10001.694336f,-10001.658203f,-10002.371094f,-10002.823242f,-10002.572266f,-10001.733398f,-10000.357422f,
-9999.092773f,-9998.204102f,-9998.280273f,-9999.097656f,-10000.918945f,-10002.552734f,-10004.145508f,-10005.043945f,-10005.784180f,-10006.336914f,-10007.095703f,-10007.986328f,-10008.871094f,-10009.673828f,-10010.239258f,-10009.539062f,-10008.406250f,-10007.070312f,-10005.617188f,-10004.955078f,
-10005.254883f,-10006.330078f,-10007.623047f,-10008.512695f,-10008.938477f,-10008.998047f,-10009.021484f,-10008.946289f,-10009.361328f,-10009.534180f,-10009.798828f,-10009.839844f,-10009.139648f,-10007.623047f,-10005.790039f,-10003.464844f,-10001.646484f,-10000.528320f,-10000.833984f,-10001.264648f,
-10001.966797f,-10001.635742f,-10001.405273f,-10000.715820f,-10000.062500f,-9999.859375f,-9999.656250f,-9999.620117f,-9999.666016f,-9999.407227f,-9998.296875f,-9996.645508f,-9994.756836f,-9992.945312f,-9992.000977f,-9992.319336f,-9993.372070f,-9995.079102f,-9996.426758f,-9997.170898f,
-9997.693359f,-9997.981445f,-9998.334961f,-9998.722656f,-9999.561523f,-10000.458984f,-10001.264648f,-10001.772461f,-10001.485352f,-10000.523438f,-9999.439453f,-9998.656250f,-9998.527344f,-9999.201172f,-10000.810547f,-10002.578125f,-9992.910156f,-10007.187500f,-10007.996094f,-10005.118164f,
-10004.339844f,-10003.233398f,-10002.906250f,-10005.871094f,-10006.191406f,-10003.514648f,-10005.648438f,-10005.741211f,-10006.282227f,-10006.567383f,-10007.255859f,-10007.813477f,-10007.791992f,-10007.259766f,-10006.022461f,-10004.506836f,-10003.109375f,-10002.845703f,-10003.573242f,-10004.934570f,
-10006.313477f,-10007.213867f,-10008.046875f,-10008.255859f,-10008.116211f,-10008.130859f,-10008.018555f,-10008.078125f,-10007.638672f,-10006.938477f,-10005.459961f,-10003.359375f,-10001.240234f,-9999.790039f,-9998.696289f,-9998.948242f,-10000.097656f,-10001.362305f,-10001.904297f,-10002.346680f,
-10002.278320f,-10002.136719f,-10001.733398f,-10002.107422f,-10002.505859f,-10002.615234f,-10002.355469f,-10001.473633f,-10000.277344f,-9998.905273f,-9998.122070f,-9998.082031f,-9999.280273f,-10000.803711f,-10002.747070f,-10004.195312f,-10005.362305f,-10006.182617f,-10006.873047f,-10007.504883f,
-10008.236328f,-10009.067383f,-10009.962891f,-10009.671875f,-10009.065430f,-10007.960938f,-10006.248047f,-10004.899414f,-10004.358398f,-10004.851562f,-10005.899414f,-10006.964844f,-10007.960938f,-10008.569336f,-10008.985352f,-10009.026367f,-10009.327148f,-10009.331055f,-10009.459961f,-10009.687500f,
-10009.432617f,-10008.642578f,-10007.387695f,-10005.105469f,-10002.792969f,-10000.883789f,-10000.387695f,-10000.422852f,-10001.174805f,-10001.272461f,-10001.603516f,-10001.359375f,-10000.959961f,-10000.709961f,-10000.187500f,-9999.720703f,-9999.677734f,-9999.719727f,-9999.125000f,-9998.093750f,
-9996.412109f,-9994.318359f,-9992.592773f,-9991.998047f,-9992.156250f,-9993.455078f,-9995.007812f,-9996.143555f,-9997.206055f,-9997.902344f,-9998.427734f,-9998.684570f,-9999.197266f,-9999.701172f,-10000.412109f,-10001.099609f,-10001.312500f,-10000.906250f,-10000.246094f,-9999.311523f,
-9998.553711f,-9998.304688f,-9999.102539f,-10000.437500f,-9992.885742f,-10008.259766f,-10009.187500f,-10006.279297f,-10004.836914f,-10002.878906f,-10001.968750f,-10004.747070f,-10004.890625f,-10003.011719f,-10005.573242f,-10005.778320f,-10006.223633f,-10006.296875f,-10006.708008f,-10007.323242f,
-10007.611328f,-10007.730469f,-10007.041016f,-10005.759766f,-10003.970703f,-10002.920898f,-10002.756836f,-10003.602539f,-10004.871094f,-10006.155273f,-10007.573242f,-10008.349609f,-10008.545898f,-10008.518555f,-10008.201172f,-10008.085938f,-10007.731445f,-10007.491211f,-10006.639648f,-10005.013672f,
-10002.921875f,-10000.954102f,-9999.068359f,-9998.415039f,-9998.969727f,-10000.057617f,-10001.014648f,-10002.005859f,-10002.399414f,-10002.534180f,-10002.034180f,-10002.054688f,-10002.091797f,-10002.235352f,-10002.417969f,-10002.161133f,-10001.375977f,-9999.948242f,-9998.614258f,-9997.699219f,
-9997.954102f,-9998.928711f,-10000.769531f,-10002.565430f,-10004.239258f,-10005.622070f,-10006.585938f,-10007.135742f,-10007.742188f,-10008.394531f,-10009.341797f,-10009.342773f,-10009.321289f,-10008.735352f,-10007.190430f,-10005.518555f,-10004.242188f,-10003.900391f,-10004.356445f,-10005.172852f,
-10006.456055f,-10007.554688f,-10008.581055f,-10009.001953f,-10009.474609f,-10009.422852f,-10009.271484f,-10009.397461f,-10009.287109f,-10009.134766f,-10008.578125f,-10006.778320f,-10004.379883f,-10001.942383f,-10000.543945f,-9999.769531f,-10000.064453f,-10000.231445f,-10000.944336f,-10001.260742f,
-10001.381836f,-10001.393555f,-10000.815430f,-9999.972656f,-9999.596680f,-9999.613281f,-9999.332031f,-9998.923828f,-9997.742188f,-9995.840820f,-9993.817383f,-9992.407227f,-9991.551758f,-9992.002930f,-9993.235352f,-9994.456055f,-9995.990234f,-9997.247070f,-9998.270508f,-9998.750977f,
-9999.200195f,-9999.284180f,-9999.592773f,-10000.159180f,-10000.607422f,-10000.702148f,-10000.681641f,-10000.010742f,-9999.082031f,-9998.109375f,-9997.984375f,-9998.515625f,-9992.857422f,-10008.491211f,-10009.743164f,-10007.342773f,-10005.755859f,-10003.321289f,-10001.727539f,-10003.672852f,
-10003.499023f,-10002.019531f,-10004.759766f,-10005.334961f,-10005.983398f,-10006.094727f,-10006.225586f,-10006.666992f,-10007.018555f,-10007.577148f,-10007.544922f,-10006.931641f,-10005.373047f,-10004.038086f,-10003.140625f,-10003.215820f,-10003.912109f,-10005.085938f,-10006.696289f,-10007.911133f,
-10008.616211f,-10008.805664f,-10008.514648f,-10008.198242f,-10007.701172f,-10007.637695f,-10007.260742f,-10006.302734f,-10004.699219f,-10002.791016f,-10000.534180f,-9999.100586f,-9998.787109f,-9999.215820f,-9999.995117f,-10001.165039f,-10001.915039f,-10002.488281f,-10002.246094f,-10002.137695f,
-10001.763672f,-10001.684570f,-10001.984375f,-10002.186523f,-10002.010742f,-10001.018555f,-9999.721680f,-9998.358398f,-9997.764648f,-9997.880859f,-9999.079102f,-10000.679688f,-10002.464844f,-10004.283203f,-10005.666992f,-10006.418945f,-10007.045898f,-10007.479492f,-10008.245117f,-10008.364258f,
-10008.722656f,-10008.718750f,-10007.732422f,-10006.242188f,-10004.728516f,-10003.741211f,-10003.408203f,-10003.561523f,-10004.599609f,-10005.738281f,-10007.121094f,-10007.978516f,-10008.874023f,-10009.043945f,-10008.790039f,-10008.728516f,-10008.586914f,-10008.769531f,-10008.814453f,-10007.809570f,
-10005.872070f,-10003.571289f,-10001.721680f,-10000.191406f,-9999.713867f,-9999.482422f,-10000.103516f,-10000.641602f,-10001.216797f,-10001.651367f,-10001.323242f,-10000.465820f,-9999.804688f,-9999.540039f,-9999.293945f,-9999.220703f,-9998.614258f,-9997.315430f,-9995.566406f,-9993.911133f,
-9992.320312f,-9991.840820f,-9992.213867f,-9992.988281f,-9994.508789f,-9996.003906f,-9997.483398f,-9998.360352f,-9999.090820f,-9999.022461f,-9999.012695f,-9999.257812f,-9999.669922f,-10000.027344f,-10000.589844f,-10000.465820f,-9999.856445f,-9998.740234f,-9998.032227f,-9997.727539f,
-9993.012695f,-10008.135742f,-10009.927734f,-10008.023438f,-10006.899414f,-10004.300781f,-10002.125977f,-10002.916016f,-10002.449219f,-10000.852539f,-10003.517578f,-10004.305664f,-10005.449219f,-10005.937500f,-10005.966797f,-10006.119141f,-10006.151367f,-10006.723633f,-10007.105469f,-10007.261719f,
-10006.394531f,-10005.332031f,-10004.098633f,-10003.507812f,-10003.410156f,-10004.044922f,-10005.457031f,-10006.949219f,-10008.235352f,-10008.959961f,-10009.000000f,-10008.598633f,-10007.813477f,-10007.541016f,-10007.171875f,-10006.739258f,-10005.806641f,-10004.433594f,-10002.286133f,-10000.489258f,
-9999.344727f,-9998.890625f,-9998.989258f,-9999.934570f,-10000.881836f,-10001.934570f,-10002.283203f,-10002.416016f,-10001.778320f,-10001.270508f,-10001.267578f,-10001.469727f,-10001.754883f,-10001.416016f,-10000.674805f,-9999.423828f,-9998.353516f,-9997.649414f,-9997.928711f,-9998.865234f,
-10000.406250f,-10002.441406f,-10004.327148f,-10005.625000f,-10006.516602f,-10006.877930f,-10007.336914f,-10007.300781f,-10007.600586f,-10007.961914f,-10007.599609f,-10006.680664f,-10005.463867f,-10004.249023f,-10003.272461f,-10002.596680f,-10002.979492f,-10003.727539f,-10005.129883f,-10006.414062f,
-10007.879883f,-10008.577148f,-10008.526367f,-10008.332031f,-10007.946289f,-10008.000977f,-10008.201172f,-10007.900391f,-10006.697266f,-10005.051758f,-10003.339844f,-10001.333984f,-10000.077148f,-9999.104492f,-9999.182617f,-9999.551758f,-10000.407227f,-10001.317383f,-10001.528320f,-10001.090820f,
-10000.350586f,-9999.682617f,-9999.139648f,-9998.844727f,-9998.536133f,-9997.890625f,-9996.812500f,-9995.484375f,-9993.700195f,-9992.483398f,-9991.857422f,-9991.783203f,-9992.824219f,-9994.190430f,-9995.993164f,-9997.373047f,-9998.711914f,-9998.896484f,-9998.806641f,-9998.696289f,
-9998.721680f,-9998.893555f,-9999.679688f,-10000.060547f,-10000.116211f,-9999.425781f,-9998.677734f,-9997.846680f,-9992.988281f,-10007.707031f,-10009.713867f,-10008.301758f,-10007.765625f,-10005.271484f,-10002.851562f,-10002.739258f,-10001.668945f,-9999.766602f,-10001.993164f,-10002.971680f,
-10004.628906f,-10005.670898f,-10005.956055f,-10006.051758f,-10005.791016f,-10006.104492f,-10006.544922f,-10007.150391f,-10006.891602f,-10006.309570f,-10005.120117f,-10004.195312f,-10003.456055f,-10003.490234f,-10004.493164f,-10006.010742f,-10007.729492f,-10009.076172f,-10009.680664f,-10009.514648f,
-10008.625000f,-10008.087891f,-10007.454102f,-10007.133789f,-10006.534180f,-10005.643555f,-10003.747070f,-10001.921875f,-10000.234375f,-9999.028320f,-9998.279297f,-9998.731445f,-9999.558594f,-10000.868164f,-10001.816406f,-10002.394531f,-10001.828125f,-10001.101562f,-10000.721680f,-10000.608398f,
-10000.942383f,-10000.980469f,-10000.774414f,-9999.877930f,-9998.730469f,-9997.511719f,-9997.029297f,-9997.222656f,-9998.303711f,-10000.320312f,-10002.589844f,-10004.529297f,-10005.940430f,-10006.548828f,-10006.989258f,-10006.846680f,-10006.938477f,-10007.322266f,-10007.309570f,-10006.876953f,
-10006.156250f,-10005.076172f,-10003.813477f,-10002.577148f,-10002.291992f,-10002.493164f,-10003.703125f,-10005.208984f,-10007.218750f,-10008.606445f,-10009.075195f,-10009.074219f,-10008.626953f,-10008.465820f,-10008.515625f,-10008.467773f,-10007.714844f,-10006.680664f,-10005.376953f,-10003.261719f,
-10001.471680f,-9999.769531f,-9999.148438f,-9999.038086f,-9999.835938f,-10001.008789f,-10001.756836f,-10001.887695f,-10001.403320f,-10000.605469f,-9999.740234f,-9998.944336f,-9998.532227f,-9998.057617f,-9997.427734f,-9996.519531f,-9994.899414f,-9993.317383f,-9991.882812f,-9990.976562f,
-9991.320312f,-9992.273438f,-9994.083984f,-9995.801758f,-9997.791016f,-9998.479492f,-9998.710938f,-9998.518555f,-9998.250977f,-9998.078125f,-9998.750000f,-9999.239258f,-9999.765625f,-9999.576172f,-9999.092773f,-9998.167969f,-9992.943359f,-10006.986328f,-10009.415039f,-10008.117188f,
-10008.520508f,-10006.298828f,-10003.987305f,-10003.567383f,-10002.083984f,-9999.311523f,-10000.940430f,-10001.532227f,-10003.343750f,-10004.833984f,-10005.551758f,-10005.901367f,-10005.633789f,-10005.717773f,-10006.088867f,-10006.859375f,-10007.140625f,-10007.141602f,-10006.375000f,-10005.454102f,
-10004.324219f,-10003.738281f,-10004.022461f,-10005.054688f,-10006.691406f,-10008.368164f,-10009.506836f,-10009.782227f,-10009.150391f,-10008.578125f,-10007.735352f,-10007.399414f,-10007.019531f,-10006.582031f,-10005.111328f,-10003.625977f,-10001.795898f,-10000.133789f,-9998.557617f,-9998.211914f,
-9998.479492f,-9999.556641f,-10000.808594f,-10001.833984f,-10001.682617f,-10001.110352f,-10000.593750f,-10000.194336f,-10000.436523f,-10000.680664f,-10000.943359f,-10000.565430f,-9999.757812f,-9998.433594f,-9997.449219f,-9996.883789f,-9997.185547f,-9998.643555f,-10000.761719f,-10003.034180f,
-10004.850586f,-10005.907227f,-10006.581055f,-10006.590820f,-10006.558594f,-10006.864258f,-10007.022461f,-10006.936523f,-10006.786133f,-10006.144531f,-10004.925781f,-10003.462891f,-10002.606445f,-10002.079102f,-10002.631836f,-10003.868164f,-10005.959961f,-10007.781250f,-10008.801758f,-10009.244141f,
-10009.077148f,-10008.820312f,-10008.707031f,-10008.704102f,-10008.222656f,-10007.723633f,-10007.062500f,-10005.231445f,-10003.363281f,-10001.170898f,-9999.875977f,-9998.993164f,-9999.255859f,-10000.226562f,-10001.190430f,-10001.854492f,-10001.837891f,-10001.254883f,-10000.437500f,-9999.279297f,
-9998.729492f,-9998.226562f,-9997.873047f,-9997.467773f,-9996.357422f,-9994.841797f,-9993.020508f,-9991.467773f,-9991.014648f,-9991.145508f,-9992.465820f,-9994.032227f,-9996.343750f,-9997.509766f,-9998.345703f,-9998.446289f,-9998.235352f,-9997.844727f,-9998.329102f,-9998.686523f,
-9999.416992f,-9999.679688f,-9999.645508f,-9998.988281f,-9992.904297f,-10006.585938f,-10009.179688f,-10007.833008f,-10008.983398f,-10007.136719f,-10005.211914f,-10004.874023f,-10003.131836f,-9999.539062f,-10000.375977f,-10000.414062f,-10001.949219f,-10003.603516f,-10004.802734f,-10005.609375f,
-10005.607422f,-10005.590820f,-10005.871094f,-10006.580078f,-10007.179688f,-10007.710938f,-10007.498047f,-10006.882812f,-10005.739258f,-10004.772461f,-10004.327148f,-10004.647461f,-10005.779297f,-10007.435547f,-10008.878906f,-10009.692383f,-10009.541016f,-10009.205078f,-10008.309570f,-10007.859375f,
-10007.555664f,-10007.388672f,-10006.314453f,-10005.287109f,-10003.624023f,-10001.867188f,-9999.720703f,-9998.553711f,-9998.009766f,-9998.458984f,-9999.599609f,-10000.818359f,-10001.231445f,-10001.083008f,-10000.670898f,-10000.113281f,-10000.207031f,-10000.426758f,-10000.956055f,-10001.051758f,
-10000.761719f,-9999.653320f,-9998.541992f,-9997.478516f,-9996.985352f,-9997.626953f,-9999.163086f,-10001.395508f,-10003.399414f,-10004.965820f,-10006.043945f,-10006.453125f,-10006.462891f,-10006.685547f,-10006.888672f,-10006.952148f,-10007.245117f,-10007.145508f,-10006.226562f,-10004.858398f,
-10003.729492f,-10002.555664f,-10002.315430f,-10002.932617f,-10004.690430f,-10006.601562f,-10008.033203f,-10009.024414f,-10009.384766f,-10009.240234f,-10009.033203f,-10008.939453f,-10008.536133f,-10008.314453f,-10008.238281f,-10006.869141f,-10005.297852f,-10002.992188f,-10001.225586f,-9999.577148f,
-9999.051758f,-9999.443359f,-10000.221680f,-10001.125977f,-10001.606445f,-10001.476562f,-10000.995117f,-9999.734375f,-9999.056641f,-9998.343750f,-9998.005859f,-9997.954102f,-9997.378906f,-9996.187500f,-9994.398438f,-9992.593750f,-9991.427734f,-9990.681641f,-9991.189453f,-9992.202148f,
-9994.398438f,-9995.812500f,-9997.279297f,-9997.927734f,-9998.091797f,-9997.750977f,-9998.091797f,-9998.183594f,-9998.855469f,-9999.380859f,-9999.741211f,-9999.518555f,-9992.782227f,-10006.245117f,-10008.788086f,-10007.370117f,-10008.791016f,-10007.499023f,-10005.972656f,-10006.173828f,
-10004.572266f,-10000.430664f,-10000.749023f,-10000.121094f,-10001.056641f,-10002.525391f,-10004.008789f,-10005.252930f,-10005.658203f,-10005.669922f,-10005.833984f,-10006.247070f,-10006.864258f,-10007.646484f,-10007.982422f,-10007.869141f,-10007.035156f,-10006.066406f,-10005.168945f,-10004.791016f,
-10005.217773f,-10006.499023f,-10007.931641f,-10009.147461f,-10009.526367f,-10009.623047f,-10008.824219f,-10008.257812f,-10007.856445f,-10007.705078f,-10006.885742f,-10006.342773f,-10005.125000f,-10003.672852f,-10001.415039f,-9999.711914f,-9998.385742f,-9998.026367f,-9998.679688f,-9999.776367f,
-10000.624023f,-10000.959961f,-10000.861328f,-10000.288086f,-10000.166992f,-10000.177734f,-10000.633789f,-10000.970703f,-10001.213867f,-10000.569336f,-9999.724609f,-9998.535156f,-9997.505859f,-9997.349609f,-9998.083984f,-9999.885742f,-10001.753906f,-10003.636719f,-10005.108398f,-10005.958008f,
-10006.165039f,-10006.342773f,-10006.452148f,-10006.469727f,-10006.973633f,-10007.398438f,-10006.968750f,-10006.088867f,-10005.086914f,-10003.659180f,-10002.789062f,-10002.721680f,-10003.894531f,-10005.538086f,-10007.127930f,-10008.577148f,-10009.548828f,-10009.728516f,-10009.547852f,-10009.320312f,
-10008.808594f,-10008.583984f,-10008.908203f,-10008.095703f,-10007.136719f,-10005.126953f,-10003.341797f,-10001.229492f,-10000.004883f,-9999.666016f,-9999.935547f,-10000.706055f,-10001.504883f,-10001.917969f,-10001.937500f,-10000.818359f,-10000.049805f,-9999.077148f,-9998.452148f,-9998.482422f,
-9998.345703f,-9997.617188f,-9996.251953f,-9994.612305f,-9993.093750f,-9991.644531f,-9991.360352f,-9991.519531f,-9993.191406f,-9994.495117f,-9996.368164f,-9997.542969f,-9998.243164f,-9998.116211f,-9998.407227f,-9998.206055f,-9998.525391f,-9999.001953f,-9999.620117f,-9999.911133f,
-9993.008789f,-10006.246094f,-10008.467773f,-10006.964844f,-10008.312500f,-10007.358398f,-10006.341797f,-10007.185547f,-10005.921875f,-10001.570312f,-10001.520508f,-10000.199219f,-10000.315430f,-10001.242188f,-10002.750000f,-10004.333008f,-10005.308594f,-10005.639648f,-10005.931641f,-10006.087891f,
-10006.527344f,-10007.299805f,-10007.972656f,-10008.335938f,-10008.004883f,-10007.342773f,-10006.253906f,-10005.332031f,-10004.990234f,-10005.631836f,-10006.716797f,-10008.098633f,-10008.979492f,-10009.674805f,-10009.285156f,-10008.767578f,-10008.293945f,-10008.006836f,-10007.224609f,-10006.939453f,
-10006.150391f,-10005.147461f,-10003.142578f,-10001.149414f,-9999.218750f,-9998.024414f,-9997.893555f,-9998.459961f,-9999.505859f,-10000.298828f,-10000.707031f,-10000.429688f,-10000.298828f,-10000.059570f,-10000.238281f,-10000.558594f,-10001.073242f,-10000.907227f,-10000.523438f,-9999.546875f,
-9998.243164f,-9997.449219f,-9997.315430f,-9998.412109f,-9999.817383f,-10001.769531f,-10003.531250f,-10004.925781f,-10005.591797f,-10005.947266f,-10006.028320f,-10005.886719f,-10006.348633f,-10007.072266f,-10007.036133f,-10006.717773f,-10006.112305f,-10004.701172f,-10003.445312f,-10002.723633f,
-10003.168945f,-10004.238281f,-10005.625977f,-10007.325195f,-10008.902344f,-10009.650391f,-10009.807617f,-10009.580078f,-10008.943359f,-10008.489258f,-10008.931641f,-10008.471680f,-10008.147461f,-10006.702148f,-10005.215820f,-10002.939453f,-10001.174805f,-10000.115234f,-9999.616211f,-9999.872070f,
-10000.704102f,-10001.521484f,-10002.193359f,-10001.573242f,-10000.976562f,-9999.818359f,-9998.845703f,-9998.678711f,-9998.652344f,-9998.250977f,-9997.447266f,-9996.253906f,-9994.704102f,-9992.823242f,-9991.787109f,-9991.127930f,-9991.918945f,-9992.764648f,-9994.676758f,-9996.259766f,
-9997.612305f,-9998.058594f,-9998.566406f,-9998.241211f,-9998.131836f,-9998.350586f,-9998.926758f,-9999.502930f,-9993.058594f,-10006.520508f,-10008.421875f,-10006.952148f,-10007.928711f,-10007.080078f,-10006.463867f,-10007.653320f,-10007.125977f,-10002.958008f,-10003.128906f,-10001.474609f,
-10000.708984f,-10000.746094f,-10001.729492f,-10003.187500f,-10004.558594f,-10005.297852f,-10005.903320f,-10006.003906f,-10006.281250f,-10006.900391f,-10007.695312f,-10008.458984f,-10008.708008f,-10008.637695f,-10007.799805f,-10006.712891f,-10005.720703f,-10005.560547f,-10005.873047f,-10006.923828f,
-10007.887695f,-10009.036133f,-10009.202148f,-10008.947266f,-10008.569336f,-10008.165039f,-10007.357422f,-10007.162109f,-10006.773438f,-10006.293945f,-10004.879883f,-10003.043945f,-10000.906250f,-9999.082031f,-9998.010742f,-9997.678711f,-9998.365234f,-9999.229492f,-10000.060547f,-10000.281250f,
-10000.376953f,-10000.075195f,-9999.969727f,-10000.102539f,-10000.719727f,-10000.971680f,-10001.150391f,-10000.710938f,-9999.592773f,-9998.585938f,-9997.768555f,-9997.991211f,-9998.605469f,-10000.111328f,-10001.778320f,-10003.491211f,-10004.687500f,-10005.462891f,-10005.721680f,-10005.535156f,
-10005.845703f,-10006.673828f,-10006.953125f,-10007.181641f,-10007.158203f,-10006.144531f,-10004.965820f,-10003.837891f,-10003.628906f,-10003.843750f,-10004.551758f,-10005.970703f,-10007.776367f,-10009.043945f,-10009.662109f,-10009.699219f,-10009.103516f,-10008.398438f,-10008.704102f,-10008.399414f,
-10008.581055f,-10007.828125f,-10006.955078f,-10005.003906f,-10003.068359f,-10001.524414f,-10000.169922f,-9999.539062f,-9999.854492f,-10000.663086f,-10001.754883f,-10001.762695f,-10001.594727f,-10000.583984f,-9999.399414f,-9998.948242f,-9998.828125f,-9998.619141f,-9998.362305f,-9997.807617f,
-9996.625977f,-9994.814453f,-9993.408203f,-9992.048828f,-9991.861328f,-9991.851562f,-9993.176758f,-9994.723633f,-9996.514648f,-9997.598633f,-9998.559570f,-9998.528320f,-9998.172852f,-9998.141602f,-9998.524414f,-9999.190430f,-9993.077148f,-10007.046875f,-10008.835938f,-10007.529297f,
-10008.222656f,-10007.076172f,-10006.291016f,-10007.782227f,-10007.641602f,-10003.780273f,-10004.398438f,-10002.684570f,-10001.311523f,-10000.527344f,-10000.742188f,-10001.759766f,-10003.236328f,-10004.387695f,-10005.488281f,-10005.866211f,-10006.142578f,-10006.486328f,-10007.040039f,-10007.868164f,
-10008.434570f,-10008.954102f,-10008.647461f,-10007.853516f,-10006.703125f,-10006.033203f,-10005.650391f,-10006.147461f,-10006.932617f,-10008.352539f,-10009.099609f,-10009.346680f,-10009.300781f,-10008.910156f,-10007.952148f,-10007.498047f,-10007.088867f,-10006.826172f,-10005.971680f,-10004.512695f,
-10002.568359f,-10000.546875f,-9998.802734f,-9997.563477f,-9997.552734f,-9998.115234f,-9999.150391f,-9999.863281f,-10000.408203f,-10000.266602f,-9999.979492f,-9999.732422f,-10000.002930f,-10000.263672f,-10000.691406f,-10000.749023f,-10000.038086f,-9999.244141f,-9998.213867f,-9997.854492f,
-9997.743164f,-9998.622070f,-9999.924805f,-10001.758789f,-10003.472656f,-10004.844727f,-10005.559570f,-10005.552734f,-10005.687500f,-10006.304688f,-10006.547852f,-10006.968750f,-10007.414062f,-10006.928711f,-10006.168945f,-10005.055664f,-10004.584961f,-10004.171875f,-10004.135742f,-10005.030273f,
-10006.729492f,-10008.426758f,-10009.596680f,-10010.172852f,-10009.837891f,-10008.984375f,-10008.972656f,-10008.381836f,-10008.636719f,-10008.283203f,-10008.028320f,-10006.608398f,-10004.944336f,-10003.347656f,-10001.405273f,-9999.962891f,-9999.540039f,-10000.008789f,-10001.250000f,-10001.776367f,
-10002.166992f,-10001.566406f,-10000.409180f,-9999.619141f,-9999.051758f,-9998.578125f,-9998.429688f,-9998.245117f,-9997.533203f,-9996.057617f,-9994.706055f,-9993.093750f,-9992.175781f,-9991.336914f,-9991.767578f,-9992.870117f,-9994.793945f,-9996.387695f,-9997.902344f,-9998.476562f,
-9998.151367f,-9997.941406f,-9997.906250f,-9998.282227f,-9992.997070f,-10007.328125f,-10009.143555f,-10008.018555f,-10008.378906f,-10006.990234f,-10005.962891f,-10007.298828f,-10007.531250f,-10004.179688f,-10005.477539f,-10004.315430f,-10002.787109f,-10001.320312f,-10000.537109f,-10000.595703f,
-10001.645508f,-10002.913086f,-10004.459961f,-10005.387695f,-10005.954102f,-10006.240234f,-10006.511719f,-10007.196289f,-10007.880859f,-10008.896484f,-10009.312500f,-10009.137695f,-10008.270508f,-10007.376953f,-10006.316406f,-10005.952148f,-10005.996094f,-10007.109375f,-10008.156250f,-10008.943359f,
-10009.479492f,-10009.410156f,-10008.500977f,-10007.788086f,-10007.226562f,-10006.994141f,-10006.612305f,-10005.742188f,-10004.369141f,-10002.635742f,-10000.603516f,-9998.586914f,-9997.501953f,-9997.222656f,-9997.927734f,-9998.914062f,-9999.996094f,-10000.334961f,-10000.230469f,-9999.784180f,
-9999.673828f,-9999.728516f,-10000.211914f,-10000.707031f,-10000.610352f,-10000.419922f,-9999.664062f,-9999.039062f,-9998.258789f,-9998.197266f,-9998.672852f,-10000.070312f,-10001.941406f,-10003.859375f,-10005.190430f,-10005.670898f,-10005.841797f,-10006.221680f,-10006.277344f,-10006.585938f,
-10007.287109f,-10007.359375f,-10007.266602f,-10006.583984f,-10006.239258f,-10005.366211f,-10004.535156f,-10004.456055f,-10005.411133f,-10007.016602f,-10008.478516f,-10009.673828f,-10009.940430f,-10009.305664f,-10009.079102f,-10008.112305f,-10008.134766f,-10007.963867f,-10008.247070f,-10007.558594f,
-10006.535156f,-10005.358398f,-10003.214844f,-10001.150391f,-9999.746094f,-9999.371094f,-10000.117188f,-10000.803711f,-10001.710938f,-10001.796875f,-10001.125977f,-10000.323242f,-9999.377930f,-9998.591797f,-9998.229492f,-9998.240234f,-9998.075195f,-9997.228516f,-9996.359375f,-9994.994141f,
-9993.753906f,-9992.125977f,-9991.448242f,-9991.531250f,-9992.993164f,-9994.699219f,-9996.611328f,-9998.085938f,-9998.265625f,-9998.263672f,-9997.999023f,-9997.987305f,-9992.993164f,-10007.228516f,-10008.984375f,-10008.249023f,-10008.591797f,-10007.121094f,-10005.883789f,-10006.750977f,
-10007.156250f,-10003.982422f,-10005.812500f,-10005.375977f,-10004.194336f,-10002.549805f,-10001.020508f,-10000.063477f,-10000.282227f,-10001.166016f,-10002.770508f,-10004.163086f,-10005.215820f,-10005.753906f,-10005.951172f,-10006.509766f,-10007.060547f,-10008.221680f,-10009.166992f,-10009.682617f,
-10009.389648f,-10008.770508f,-10007.500977f,-10006.469727f,-10005.693359f,-10006.119141f,-10006.935547f,-10007.930664f,-10008.967773f,-10009.431641f,-10008.862305f,-10008.138672f,-10007.416992f,-10006.978516f,-10006.673828f,-10006.208008f,-10005.448242f,-10004.263672f,-10002.414062f,-10000.103516f,
-9998.131836f,-9996.859375f,-9996.807617f,-9997.505859f,-9998.758789f,-9999.543945f,-9999.875977f,-9999.578125f,-9999.270508f,-9999.068359f,-9999.308594f,-9999.877930f,-10000.222656f,-10000.652344f,-10000.469727f,-10000.062500f,-9999.056641f,-9998.248047f,-9997.878906f,-9998.515625f,
-10000.039062f,-10002.104492f,-10003.899414f,-10005.053711f,-10005.602539f,-10006.009766f,-10005.963867f,-10006.034180f,-10006.664062f,-10007.052734f,-10007.566406f,-10007.516602f,-10007.684570f,-10006.870117f,-10005.686523f,-10004.778320f,-10004.832031f,-10005.890625f,-10007.152344f,-10008.691406f,
-10009.586914f,-10009.482422f,-10009.429688f,-10008.293945f,-10007.993164f,-10007.672852f,-10008.122070f,-10007.958008f,-10007.593750f,-10007.107422f,-10005.233398f,-10003.099609f,-10001.009766f,-9999.735352f,-9999.603516f,-9999.862305f,-10000.790039f,-10001.398438f,-10001.337891f,-10000.875000f,
-9999.893555f,-9998.864258f,-9998.116211f,-9997.900391f,-9997.958008f,-9997.616211f,-9997.272461f,-9996.415039f,-9995.362305f,-9993.471680f,-9991.902344f,-9991.022461f,-9991.570312f,-9992.816406f,-9994.649414f,-9996.735352f,-9997.533203f,-9998.153320f,-9998.008789f,-9997.858398f,
-9992.788086f,-10007.053711f,-10008.656250f,-10008.163086f,-10008.680664f,-10007.433594f,-10006.054688f,-10006.766602f,-10006.874023f,-10003.723633f,-10005.931641f,-10006.108398f,-10005.667969f,-10004.432617f,-10002.789062f,-10001.136719f,-10000.387695f,-10000.436523f,-10001.487305f,-10002.946289f,
-10004.301758f,-10005.232422f,-10005.568359f,-10006.091797f,-10006.397461f,-10007.395508f,-10008.569336f,-10009.585938f,-10009.924805f,-10009.982422f,-10009.053711f,-10007.851562f,-10006.468750f,-10006.029297f,-10006.150391f,-10006.867188f,-10008.042969f,-10009.001953f,-10008.933594f,-10008.521484f,
-10007.867188f,-10007.282227f,-10006.827148f,-10006.597656f,-10006.315430f,-10005.782227f,-10004.541016f,-10002.654297f,-10000.347656f,-9998.357422f,-9997.327148f,-9997.208008f,-9998.087891f,-9999.017578f,-9999.755859f,-9999.818359f,-9999.599609f,-9999.236328f,-9999.203125f,-9999.525391f,
-10000.030273f,-10000.806641f,-10001.227539f,-10001.330078f,-10000.581055f,-9999.475586f,-9998.479492f,-9998.183594f,-9998.825195f,-10000.400391f,-10002.139648f,-10003.732422f,-10004.743164f,-10005.334961f,-10005.356445f,-10005.198242f,-10005.540039f,-10005.979492f,-10006.778320f,-10007.276367f,
-10008.084961f,-10007.731445f,-10006.750000f,-10005.476562f,-10004.679688f,-10004.829102f,-10005.359375f,-10006.662109f,-10007.876953f,-10008.362305f,-10008.766602f,-10007.844727f,-10007.417969f,-10006.871094f,-10007.248047f,-10007.359375f,-10007.522461f,-10007.762695f,-10006.613281f,-10005.045898f,
-10002.952148f,-10001.175781f,-10000.090820f,-9999.454102f,-9999.873047f,-10000.566406f,-10000.978516f,-10001.044922f,-10000.409180f,-9999.481445f,-9998.396484f,-9997.866211f,-9997.883789f,-9997.807617f,-9997.893555f,-9997.619141f,-9997.173828f,-9995.601562f,-9993.804688f,-9992.209961f,
-9991.727539f,-9991.946289f,-9993.108398f,-9995.132812f,-9996.375977f,-9997.595703f,-9997.900391f,-9997.883789f,-9992.864258f,-10006.859375f,-10008.347656f,-10007.720703f,-10008.492188f,-10007.645508f,-10006.281250f,-10006.823242f,-10006.689453f,-10003.489258f,-10005.571289f,-10006.145508f,
-10006.373047f,-10005.756836f,-10004.417969f,-10002.484375f,-10000.941406f,-10000.041992f,-10000.225586f,-10001.302734f,-10002.742188f,-10004.095703f,-10004.801758f,-10005.506836f,-10005.711914f,-10006.432617f,-10007.590820f,-10008.810547f,-10009.593750f,-10010.374023f,-10010.095703f,-10009.125977f,
-10007.559570f,-10006.409180f,-10005.699219f,-10005.781250f,-10006.738281f,-10008.005859f,-10008.461914f,-10008.591797f,-10008.264648f,-10007.706055f,-10006.980469f,-10006.755859f,-10006.631836f,-10006.518555f,-10005.913086f,-10004.712891f,-10002.562500f,-10000.280273f,-9998.403320f,-9997.271484f,
-9997.409180f,-9998.094727f,-9999.056641f,-9999.558594f,-9999.688477f,-9999.422852f,-9999.256836f,-9999.236328f,-9999.623047f,-10000.426758f,-10001.228516f,-10001.929688f,-10001.662109f,-10000.701172f,-9999.496094f,-9998.466797f,-9998.174805f,-9998.967773f,-10000.273438f,-10002.029297f,
-10003.507812f,-10004.531250f,-10004.863281f,-10004.718750f,-10004.802734f,-10005.131836f,-10005.897461f,-10006.700195f,-10007.992188f,-10008.233398f,-10007.794922f,-10006.583984f,-10005.316406f,-10004.587891f,-10004.159180f,-10004.800781f,-10005.919922f,-10006.819336f,-10007.779297f,-10007.345703f,
-10007.067383f,-10006.356445f,-10006.491211f,-10006.584961f,-10006.951172f,-10007.660156f,-10007.224609f,-10006.451172f,-10004.799805f,-10002.955078f,-10001.161133f,-9999.554688f,-9999.142578f,-9999.427734f,-9999.987305f,-10000.562500f,-10000.455078f,-9999.916992f,-9998.784180f,-9997.959961f,
-9997.748047f,-9997.643555f,-9997.828125f,-9997.938477f,-9998.103516f,-9997.110352f,-9995.572266f,-9993.747070f,-9992.408203f,-9991.577148f,-9991.749023f,-9993.215820f,-9994.511719f,-9996.250977f,-9997.123047f,-9997.541016f,-9992.875977f,-10006.784180f,-10008.087891f,-10007.048828f,
-10007.840820f,-10007.518555f,-10006.509766f,-10007.179688f,-10007.091797f,-10003.539062f,-10005.474609f,-10006.093750f,-10006.749023f,-10006.675781f,-10005.840820f,-10003.978516f,-10001.942383f,-10000.193359f,-9999.372070f,-9999.715820f,-10000.898438f,-10002.488281f,-10003.675781f,-10004.786133f,
-10005.151367f,-10005.694336f,-10006.744141f,-10007.946289f,-10008.871094f,-10010.167969f,-10010.507812f,-10009.973633f,-10008.562500f,-10006.988281f,-10005.506836f,-10004.785156f,-10005.174805f,-10006.449219f,-10007.318359f,-10008.104492f,-10008.350586f,-10008.099609f,-10007.267578f,-10006.971680f,
-10006.817383f,-10006.838867f,-10006.643555f,-10006.116211f,-10004.432617f,-10002.216797f,-9999.834961f,-9997.764648f,-9996.947266f,-9997.052734f,-9997.931641f,-9998.785156f,-9999.416016f,-9999.541992f,-9999.533203f,-9999.318359f,-9999.541992f,-10000.145508f,-10001.049805f,-10002.116211f,
-10002.344727f,-10001.764648f,-10000.665039f,-9999.199219f,-9998.123047f,-9998.013672f,-9998.595703f,-10000.200195f,-10001.968750f,-10003.509766f,-10004.401367f,-10004.621094f,-10004.670898f,-10004.931641f,-10005.551758f,-10006.400391f,-10007.902344f,-10008.623047f,-10008.784180f,-10007.933594f,
-10006.541992f,-10005.183594f,-10003.853516f,-10003.547852f,-10004.204102f,-10005.175781f,-10006.604492f,-10006.855469f,-10007.008789f,-10006.355469f,-10006.365234f,-10006.334961f,-10006.658203f,-10007.518555f,-10007.569336f,-10007.482422f,-10006.500977f,-10004.940430f,-10002.806641f,-10000.398438f,
-9999.058594f,-9998.625000f,-9998.909180f,-9999.781250f,-10000.248047f,-10000.335938f,-9999.544922f,-9998.688477f,-9998.270508f,-9998.028320f,-9998.067383f,-9998.245117f,-9998.794922f,-9998.366211f,-9997.338867f,-9995.641602f,-9993.822266f,-9992.096680f,-9991.261719f,-9991.839844f,
-9992.795898f,-9994.743164f,-9996.146484f,-9997.194336f,-9992.983398f,-10007.158203f,-10008.249023f,-10006.486328f,-10007.217773f,-10006.978516f,-10006.221680f,-10007.305664f,-10007.309570f,-10003.605469f,-10005.260742f,-10005.529297f,-10006.358398f,-10006.871094f,-10006.913086f,-10005.791016f,
-10003.942383f,-10001.824219f,-10000.126953f,-9999.450195f,-9999.827148f,-10001.176758f,-10002.607422f,-10004.087891f,-10004.749023f,-10005.257812f,-10006.158203f,-10007.132812f,-10007.967773f,-10009.573242f,-10010.594727f,-10010.869141f,-10010.195312f,-10008.835938f,-10006.961914f,-10005.411133f,
-10004.829102f,-10005.502930f,-10006.211914f,-10007.349609f,-10008.106445f,-10008.315430f,-10007.515625f,-10007.134766f,-10006.693359f,-10006.588867f,-10006.566406f,-10006.717773f,-10005.874023f,-10004.381836f,-10002.105469f,-9999.444336f,-9997.641602f,-9996.767578f,-9996.947266f,-9997.679688f,
-9998.583984f,-9999.109375f,-9999.449219f,-9999.188477f,-9999.164062f,-9999.385742f,-10000.072266f,-10001.336914f,-10002.101562f,-10002.266602f,-10001.827148f,-10000.468750f,-9999.006836f,-9998.050781f,-9997.602539f,-9998.409180f,-9999.902344f,-10001.667969f,-10003.018555f,-10003.724609f,
-10003.994141f,-10004.235352f,-10004.583984f,-10005.241211f,-10006.694336f,-10007.848633f,-10008.754883f,-10008.757812f,-10007.921875f,-10006.561523f,-10004.718750f,-10003.405273f,-10003.089844f,-10003.559570f,-10005.046875f,-10005.866211f,-10006.634766f,-10006.281250f,-10006.302734f,-10006.082031f,
-10006.179688f,-10006.872070f,-10007.164062f,-10007.647461f,-10007.612305f,-10006.908203f,-10005.104492f,-10002.379883f,-10000.219727f,-9998.683594f,-9998.072266f,-9998.643555f,-9999.280273f,-9999.870117f,-9999.613281f,-9998.923828f,-9998.374023f,-9997.890625f,-9997.612305f,-9997.553711f,
-9998.290039f,-9998.383789f,-9998.243164f,-9997.259766f,-9995.596680f,-9993.460938f,-9991.831055f,-9991.192383f,-9991.233398f,-9992.685547f,-9994.165039f,-9995.671875f,-9992.916992f,-10007.944336f,-10008.760742f,-10006.376953f,-10006.590820f,-10006.125000f,-10005.593750f,-10007.465820f,
-10007.591797f,-10004.088867f,-10005.752930f,-10005.371094f,-10005.854492f,-10006.354492f,-10006.833984f,-10006.375977f,-10005.045898f,-10003.108398f,-10001.061523f,-9999.569336f,-9999.000977f,-9999.720703f,-10001.064453f,-10002.866211f,-10004.031250f,-10004.908203f,-10005.985352f,-10006.853516f,
-10007.377930f,-10008.826172f,-10009.962891f,-10010.723633f,-10010.809570f,-10010.034180f,-10008.379883f,-10006.469727f,-10005.112305f,-10004.978516f,-10005.234375f,-10006.373047f,-10007.549805f,-10008.372070f,-10008.064453f,-10007.872070f,-10007.268555f,-10006.827148f,-10006.500000f,-10006.700195f,
-10006.358398f,-10005.606445f,-10003.852539f,-10001.257812f,-9998.925781f,-9997.165039f,-9996.401367f,-9996.590820f,-9997.453125f,-9998.336914f,-9999.236328f,-9999.330078f,-9999.365234f,-9999.336914f,-9999.596680f,-10000.550781f,-10001.347656f,-10001.930664f,-10002.078125f,-10001.146484f,
-9999.877930f,-9998.620117f,-9997.449219f,-9997.416992f,-9998.272461f,-9999.947266f,-10001.596680f,-10002.897461f,-10003.752930f,-10004.302734f,-10004.630859f,-10005.039062f,-10006.123047f,-10007.208008f,-10008.299805f,-10008.913086f,-10008.733398f,-10007.903320f,-10006.151367f,-10004.296875f,
-10003.084961f,-10002.772461f,-10003.847656f,-10004.912109f,-10006.266602f,-10006.523438f,-10006.987305f,-10006.906250f,-10006.882812f,-10007.236328f,-10007.293945f,-10007.712891f,-10008.164062f,-10008.234375f,-10007.117188f,-10004.710938f,-10002.305664f,-9999.986328f,-9998.410156f,-9998.267578f,
-9998.624023f,-9999.435547f,-9999.838867f,-9999.670898f,-9999.383789f,-9998.919922f,-9998.347656f,-9997.845703f,-9998.235352f,-9998.247070f,-9998.494141f,-9998.104492f,-9996.957031f,-9995.026367f,-9993.210938f,-9991.737305f,-9990.845703f,-9991.391602f,-9992.455078f,-9994.085938f,
-9992.864258f,-10008.883789f,-10009.606445f,-10006.753906f,-10006.204102f,-10005.167969f,-10004.556641f,-10006.617188f,-10007.019531f,-10003.986328f,-10005.803711f,-10005.129883f,-10005.265625f,-10005.624023f,-10006.299805f,-10006.506836f,-10005.946289f,-10004.698242f,-10002.870117f,-10000.981445f,
-9999.520508f,-9999.315430f,-10000.003906f,-10001.581055f,-10002.977539f,-10004.246094f,-10005.613281f,-10006.524414f,-10006.909180f,-10008.048828f,-10009.115234f,-10010.150391f,-10010.897461f,-10010.950195f,-10009.960938f,-10008.188477f,-10006.373047f,-10005.349609f,-10004.779297f,-10005.323242f,
-10006.387695f,-10007.539062f,-10007.803711f,-10008.013672f,-10007.506836f,-10006.959961f,-10006.364258f,-10006.442383f,-10006.388672f,-10006.351562f,-10005.455078f,-10003.483398f,-10001.208984f,-9998.877930f,-9997.156250f,-9996.457031f,-9996.730469f,-9997.488281f,-9998.705078f,-9999.232422f,
-9999.491211f,-9999.503906f,-9999.494141f,-10000.086914f,-10000.789062f,-10001.598633f,-10002.272461f,-10001.962891f,-10001.274414f,-10000.163086f,-9998.582031f,-9997.660156f,-9997.536133f,-9998.508789f,-9999.883789f,-10001.361328f,-10002.708008f,-10003.649414f,-10004.125000f,-10004.436523f,
-10005.202148f,-10006.065430f,-10007.065430f,-10008.064453f,-10008.550781f,-10008.528320f,-10007.389648f,-10005.524414f,-10003.650391f,-10002.450195f,-10002.656250f,-10003.361328f,-10004.823242f,-10005.552734f,-10006.495117f,-10006.759766f,-10006.844727f,-10007.026367f,-10006.834961f,-10006.981445f,
-10007.657227f,-10008.338867f,-10008.076172f,-10006.521484f,-10004.425781f,-10001.785156f,-9999.418945f,-9998.334961f,-9997.883789f,-9998.370117f,-9999.060547f,-9999.372070f,-9999.551758f,-9999.384766f,-9998.879883f,-9998.168945f,-9998.225586f,-9997.958984f,-9998.360352f,-9998.471680f,
-9998.000977f,-9996.698242f,-9995.248047f,-9993.421875f,-9991.713867f,-9991.184570f,-9991.328125f,-9992.470703f,-9992.768555f,-10009.554688f,-10010.322266f,-10007.517578f,-10006.165039f,-10004.643555f,-10003.537109f,-10005.438477f,-10005.873047f,-10003.582031f,-10005.647461f,-10004.870117f,
-10004.751953f,-10004.854492f,-10005.354492f,-10005.731445f,-10005.623047f,-10005.062500f,-10003.763672f,-10002.022461f,-10000.122070f,-9999.138672f,-9998.998047f,-10000.089844f,-10001.435547f,-10003.026367f,-10004.833984f,-10006.093750f,-10006.636719f,-10007.608398f,-10008.462891f,-10009.431641f,
-10010.429688f,-10011.077148f,-10010.832031f,-10009.590820f,-10007.840820f,-10006.309570f,-10005.063477f,-10004.854492f,-10005.457031f,-10006.594727f,-10007.298828f,-10008.016602f,-10007.869141f,-10007.509766f,-10006.771484f,-10006.553711f,-10006.323242f,-10006.531250f,-10006.220703f,-10004.960938f,
-10003.107422f,-10000.701172f,-9998.399414f,-9996.865234f,-9996.312500f,-9996.590820f,-9997.749023f,-9998.587891f,-9999.188477f,-9999.532227f,-9999.564453f,-9999.889648f,-10000.370117f,-10001.066406f,-10001.839844f,-10001.923828f,-10001.843750f,-10001.197266f,-9999.737305f,-9998.434570f,
-9997.527344f,-9997.694336f,-9998.520508f,-9999.806641f,-10001.440430f,-10002.766602f,-10003.628906f,-10004.108398f,-10004.812500f,-10005.488281f,-10006.189453f,-10007.178711f,-10007.926758f,-10008.587891f,-10008.174805f,-10006.799805f,-10004.795898f,-10003.110352f,-10002.478516f,-10002.548828f,
-10003.711914f,-10004.632812f,-10005.998047f,-10006.750000f,-10007.261719f,-10007.551758f,-10007.256836f,-10007.082031f,-10007.586914f,-10008.449219f,-10008.709961f,-10007.934570f,-10006.438477f,-10004.026367f,-10001.364258f,-9999.541016f,-9998.192383f,-9997.999023f,-9998.503906f,-9999.027344f,
-9999.635742f,-9999.916016f,-9999.759766f,-9999.098633f,-9998.959961f,-9998.274414f,-9998.394531f,-9998.498047f,-9998.396484f,-9997.650391f,-9996.787109f,-9995.144531f,-9993.145508f,-9991.805664f,-9991.028320f,-9991.379883f,-9992.769531f,-10009.968750f,-10011.138672f,-10008.604492f,
-10006.947266f,-10004.955078f,-10003.216797f,-10004.437500f,-10004.718750f,-10002.947266f,-10005.308594f,-10004.562500f,-10004.531250f,-10004.532227f,-10004.739258f,-10005.018555f,-10005.074219f,-10005.049805f,-10004.398438f,-10003.201172f,-10001.333008f,-9999.885742f,-9998.907227f,-9999.212891f,
-10000.054688f,-10001.583008f,-10003.594727f,-10005.207031f,-10006.108398f,-10007.109375f,-10007.855469f,-10008.597656f,-10009.515625f,-10010.435547f,-10010.776367f,-10010.267578f,-10008.969727f,-10007.416992f,-10005.749023f,-10004.795898f,-10004.674805f,-10005.375977f,-10006.151367f,-10007.219727f,
-10007.559570f,-10007.654297f,-10007.118164f,-10006.806641f,-10006.277344f,-10006.423828f,-10006.416992f,-10005.790039f,-10004.626953f,-10002.605469f,-10000.238281f,-9998.173828f,-9996.770508f,-9996.274414f,-9996.937500f,-9997.720703f,-9998.507812f,-9999.236328f,-9999.556641f,-9999.867188f,
-10000.216797f,-10000.657227f,-10001.252930f,-10001.491211f,-10001.808594f,-10001.692383f,-10000.652344f,-9999.345703f,-9997.978516f,-9997.334961f,-9997.341797f,-9998.025391f,-9999.480469f,-10000.901367f,-10002.063477f,-10002.859375f,-10003.750000f,-10004.388672f,-10004.777344f,-10005.526367f,
-10006.218750f,-10007.203125f,-10007.431641f,-10006.797852f,-10005.102539f,-10003.407227f,-10002.179688f,-10001.469727f,-10001.973633f,-10002.675781f,-10004.097656f,-10005.217773f,-10006.208984f,-10006.883789f,-10006.795898f,-10006.557617f,-10006.827148f,-10007.533203f,-10008.016602f,-10007.827148f,
-10007.042969f,-10005.299805f,-10002.913086f,-10000.812500f,-9998.751953f,-9997.726562f,-9997.619141f,-9997.972656f,-9998.779297f,-9999.444336f,-9999.807617f,-9999.529297f,-9999.534180f,-9998.645508f,-9998.412109f,-9998.298828f,-9998.223633f,-9997.827148f,-9997.563477f,-9996.417969f,
-9994.632812f,-9992.957031f,-9991.492188f,-9990.956055f
};



void test_gemm_accuracy(ov::element::Type_t d_type = data_types::f32) {
    auto& engine = get_test_engine();

    auto in1_layout1 = layout{ ov::PartialShape{ 1, 1, 128, 64 }, d_type, format::bfyx };
    auto in1_layout2 = layout{ ov::PartialShape{ 1, 1, 64, 128 }, d_type, format::bfyx };
    auto in2_layout2 = layout{ ov::PartialShape{ 1, 1,  1,  128 }, d_type, format::bfyx };
    auto input1_1 = engine.allocate_memory(in1_layout1);
    auto input1_2 = engine.allocate_memory(in1_layout2);
    auto input2_2 = engine.allocate_memory(in2_layout2);

    if (d_type == data_types::f32) {
        set_values(input1_1, input1_data1);
        set_values(input1_2, input1_data2);
        set_values(input2_2, input2_data2);
    } else {
        std::vector<ov::float16> input1_data1_fp16(input1_data1.begin(), input1_data1.end());
        std::vector<ov::float16> input1_data2_fp16(input1_data2.begin(), input1_data2.end());
        std::vector<ov::float16> input2_data2_fp16(input2_data2.begin(), input2_data2.end());
        set_values(input1_1, input1_data1_fp16);
        set_values(input1_2, input1_data2_fp16);
        set_values(input2_2, input2_data2_fp16);
    }

    std::string input1_1_id = "transpose:Transpose_68";
    std::string input1_2_id = "multiply:Multiply_9436_autogenerated";
    std::string input2_2_id = "add:Mul_6";
    std::string gemm_id = "matmul:Div_72";
    std::string add_id = "add:Add_73";
    std::string output_id = "output";

    topology topology;
    topology.add(input_layout(input1_1_id, in1_layout1),
                input_layout(input1_2_id, in1_layout2),
                input_layout(input2_2_id, in2_layout2),
                gemm(gemm_id, { input_info(input1_1_id), input_info(input1_2_id) }, d_type, false, false, 1.0f, 0.0f, 4, 4),
                eltwise(add_id, { input_info(gemm_id), input_info(input2_2_id)}, eltwise_mode::sum, {}, d_type),
                reorder(output_id, add_id, format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc gemm_impl = { format::bfyx, "gemm_ref" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {gemm_id, gemm_impl} }));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    {
        network->set_input_data(input1_1_id, input1_1);
        network->set_input_data(input1_2_id, input1_2);
        network->set_input_data(input2_2_id, input2_2);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, output_id);

        auto output_prim_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        ASSERT_EQ(output_ptr.size(), exp_output_data.size());
        for (uint32_t i = 0; i < 10; ++i) {
            std::cout << "output[" << i << "] : " << output_ptr[i] << std::endl;
        }
        for (uint32_t i = 0; i < 10; ++i) {
            std::cout << "output_exp[" << i << "] : " << exp_output_data[i] << std::endl;
        }
        for (uint32_t i = 0; i < exp_output_data.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], exp_output_data[i]);
        }
    }
}

TEST(gemm_gpu_test1, gemm_acc_test_f32) {
    test_gemm_accuracy(data_types::f32);
}

TEST(gemm_gpu_test2, gemm_acc_test_f16) {
    test_gemm_accuracy(data_types::f16);
}

} // namespace
