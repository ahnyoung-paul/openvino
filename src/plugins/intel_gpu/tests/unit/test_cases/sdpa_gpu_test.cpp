// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "openvino/util/file_util.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace  {
// #ifdef ENABLE_ONEDNN_FOR_GPU
// Disable onednn test because onednn does not support format_tag::cbda, format_tag::badc.


struct spda_gpu_test {

    const std::string ref_data_path = "/home/ahnyoung/cldnn/cvs_164660/dumps.0512/outs.simple.binary/sdpa.ref/";
    const std::string opt_data_path = "/home/ahnyoung/cldnn/cvs_164660/dumps.0512/outs.simple.binary/sdpa.micro/";

    virtual std::string get_target_node_name() {
        return "sdpa:__module.transformer_blocks.0.attn2/aten::scaled_dot_product_attention/ScaledDotProductAttention";
    }

    virtual std::vector<std::string> get_bin_names() {
        std::vector<std::string> bin_names = {
            "program1_network1_0__001480_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src0__f16__2_990_32_64__bfyx.bin",
            "program1_network1_0__001480_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src1__f16__2_128_32_64__bfyx.bin",
            "program1_network1_0__001480_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src2__f16__2_128_32_64__bfyx.bin",
            "program1_network1_0__001480_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src3__f16__2_32_1_128__bfyx.bin",
            "program1_network1_0__001480_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_dst0__f16__2_32_990_64__bfyx.bin"
        };
        return bin_names;
    }

    virtual void load_input(cldnn::memory::ptr mem, size_t idx, const bool use_ref_inputs = false) {
        auto bin_names = get_bin_names();
        ASSERT_TRUE(idx < bin_names.size());
        std::string input_file_name = (use_ref_inputs ? ref_data_path : opt_data_path) + bin_names[idx];
        load_data_from_bin(mem, input_file_name);
    }

    void load_data_from_bin(cldnn::memory::ptr mem, const std::string filepath) {
        GPU_DEBUG_COUT << "Load data from " << filepath << std::endl;
        std::vector<uint8_t> bin = ov::util::load_binary(filepath);
        mem->copy_from(get_test_stream(), static_cast<void *>(&bin[0]), true);
    }

    virtual cldnn::memory::ptr run_network(bool is_caching_test, bool use_micro_sdpa = false, bool use_ref_inputs = false) {
        auto& engine = get_test_engine();
        cldnn::layout input0_dyn_layout({-1, -1, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input1_dyn_layout({-1, -1, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input2_dyn_layout({-1, -1, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input3_dyn_layout({-1, 32, -1, -1}, data_types::f16, format::bfyx);

        cldnn::layout input0_static_layout({2, 990, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input1_static_layout({2, 128, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input2_static_layout({2, 128, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input3_static_layout({2, 32,  1, 128}, data_types::f16, format::bfyx);

        auto input0 = engine.allocate_memory(input0_static_layout);
        auto input1 = engine.allocate_memory(input1_static_layout);
        auto input2 = engine.allocate_memory(input2_static_layout);
        auto input3 = engine.allocate_memory(input3_static_layout);

        auto data0 = engine.allocate_memory(input0_static_layout, true);
        auto data1 = engine.allocate_memory(input1_static_layout, true);
        auto data2 = engine.allocate_memory(input2_static_layout, true);
        auto data3 = engine.allocate_memory(input3_static_layout, true);

        load_input(input0, 0, use_ref_inputs);
        load_input(input1, 1, use_ref_inputs);
        load_input(input2, 2, use_ref_inputs);
        load_input(input3, 3, use_ref_inputs);

        GPU_DEBUG_COUT << "Topology: SDPA kernel test " << std::endl;
        GPU_DEBUG_COUT << "* use micro_sdpa           : " << (use_micro_sdpa ? "Yes" : "No") << std::endl;
        GPU_DEBUG_COUT << "* input0 : " << input0_static_layout.to_short_string() << ", " << input0_static_layout.count() << std::endl;
        GPU_DEBUG_COUT << "* input1 : " << input1_static_layout.to_short_string() << ", " << input1_static_layout.count() << std::endl;
        GPU_DEBUG_COUT << "* input2 : " << input2_static_layout.to_short_string() << ", " << input2_static_layout.count() << std::endl;
        GPU_DEBUG_COUT << "* input3 : " << input3_static_layout.to_short_string() << ", " << input3_static_layout.count() << std::endl;

        topology topo;
        topo.add(input_layout("input0", input0_dyn_layout));
        topo.add(input_layout("input1", input1_dyn_layout));
        topo.add(input_layout("input2", input2_dyn_layout));
        topo.add(input_layout("input3", input3_dyn_layout));
        topo.add(input_layout("data0", input0_dyn_layout));
        topo.add(input_layout("data1", input1_dyn_layout));
        topo.add(input_layout("data2", input2_dyn_layout));
        topo.add(input_layout("data3", input3_dyn_layout));
        topo.add(eltwise("sum0", {input_info("input0"), input_info("data0") }, eltwise_mode::sum));
        topo.add(eltwise("sum1", {input_info("input1"), input_info("data1") }, eltwise_mode::sum));
        topo.add(eltwise("sum2", {input_info("input2"), input_info("data2") }, eltwise_mode::sum));
        topo.add(eltwise("sum3", {input_info("input3"), input_info("data3") }, eltwise_mode::sum));
        topo.add(scaled_dot_product_attention("sdpa", {input_info("sum0"), input_info("sum1"), input_info("sum2"), input_info("sum3")},
            false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false));
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        if (use_micro_sdpa) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"sdpa", {format::type::bfyx, "sdpa_micro"}} }));
            // config.set_property(ov::intel_gpu::max_kernels_per_batch(1));
            // config.set_property(ov::intel_gpu::dump_sources_path("/home/ahnyoung/cldnn/cvs_164660/dumps.0512/kernels/sdpa.micro.unit/"));
            // config.set_property(ov::intel_gpu::dump_tensors("all"));
            // config.set_property(ov::intel_gpu::dump_tensors_format("text"));
            // config.set_property(ov::intel_gpu::dump_tensors_path("/home/ahnyoung/cldnn/cvs_164660/dumps.0512/outs/units/sdpa.micro.unit/"));
        } else {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"sdpa", {format::type::bfyx, "sdpa_ref"}} }));
            // config.set_property(ov::intel_gpu::max_kernels_per_batch(1));
            // config.set_property(ov::intel_gpu::dump_sources_path("/home/ahnyoung/cldnn/cvs_164660/dumps.0512/kernels/sdpa.ref.unit/"));
            // config.set_property(ov::intel_gpu::dump_tensors("all"));
            // config.set_property(ov::intel_gpu::dump_tensors_format("text"));
            // config.set_property(ov::intel_gpu::dump_tensors_path("/home/ahnyoung/cldnn/cvs_164660/dumps.0512/outs/units/sdpa.micro.unit/"));
        }

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);
        net->set_input_data("input2", input2);
        net->set_input_data("input3", input3);
        net->set_input_data("data0", data0);
        net->set_input_data("data1", data1);
        net->set_input_data("data2", data2);
        net->set_input_data("data3", data3);

        // {
        //     cldnn::mem_lock<ov::float16, mem_lock_type::read> data(input0, get_test_stream());
        //     for (size_t i = 0; i < 10; i++) {
        //         GPU_DEBUG_COUT << i << "] input : " << data[i] << std::endl;
        //     }
        // }

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return output;
    }

    void execute(bool is_caching_test = false, bool use_ref_inputs = false) {
        GPU_DEBUG_COUT << "********************************************************************************" << std::endl;
        GPU_DEBUG_COUT << "********************************************************************************" << std::endl;
        auto mem_ref_ptr = run_network(is_caching_test, false, use_ref_inputs);
        GPU_DEBUG_COUT << "********************************************************************************" << std::endl;
        GPU_DEBUG_COUT << "********************************************************************************" << std::endl;
        auto mem_opt_ptr = run_network(is_caching_test, true, use_ref_inputs);
        GPU_DEBUG_COUT << "********************************************************************************" << std::endl;
        GPU_DEBUG_COUT << "********************************************************************************" << std::endl;
        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        // if (ret < 0.9f) {
        {
            std::vector<std::pair<size_t, ov::float16>> differences;
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                if (std::isnan(opt_data[idx])) {
                    GPU_DEBUG_COUT << "opt_data has nan (" << opt_data[idx] << ")" << std::endl;
                    ASSERT_TRUE(false);
                }
                if (std::isnan(ref_data[idx])) {
                    GPU_DEBUG_COUT << "ref_data has nan (" << ref_data[idx] << ")" << std::endl;
                    ASSERT_TRUE(false);
                }
                ASSERT_FALSE(std::isnan(opt_data[idx]));
                float diff = std::abs(ref_data[idx] - opt_data[idx]);
                differences.push_back({idx, diff});
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            GPU_DEBUG_COUT << "Cosine Similarity : " << ret << std::endl;
            std::sort(differences.begin(), differences.end(), [](std::pair<size_t, ov::float16> a, std::pair<size_t, ov::float16> b){
                return a.second > b.second;
            });
            GPU_DEBUG_COUT << "Compare data] ref_data : act_data" << std::endl;
            for (size_t i = 0; i < 10; i++) {
                GPU_DEBUG_COUT << std::setw(8) << std::fixed << i << "] " << std::setw(12) << ref_data[i] << " : "
                    << std::setw(12) << opt_data[i] << std::endl;
            }

            size_t ii = 31812;
            GPU_DEBUG_COUT << std::setw(8) << std::fixed << ii << "] " << std::setw(12) << ref_data[ii] << " : " << std::setw(12) << opt_data[ii] << std::endl;
            ii = 31813;
            GPU_DEBUG_COUT << std::setw(8) << std::fixed << ii << "] " << std::setw(12) << ref_data[ii] << " : " << std::setw(12) << opt_data[ii] << std::endl;
            ii = 31814;
            GPU_DEBUG_COUT << std::setw(8) << std::fixed << ii << "] " << std::setw(12) << ref_data[ii] << " : " << std::setw(12) << opt_data[ii] << std::endl;
            ii = 31815;
            GPU_DEBUG_COUT << std::setw(8) << std::fixed << ii << "] " << std::setw(12) << ref_data[ii] << " : " << std::setw(12) << opt_data[ii] << std::endl;

            for (size_t i = 0; i < 10 && i < differences.size(); i++) {
                size_t idx = differences[i].first;
                GPU_DEBUG_COUT << std::setw(8) << std::fixed << idx << "] " << std::setw(12) << ref_data[idx] << " : "
                    << std::setw(12) << opt_data[idx]
                    << " (Difference: " << differences[i].second << ")" << std::endl;
            }
            ASSERT_GE(ret, 0.9f);
        }
    }

    float cosineSimilarity(cldnn::mem_lock<ov::float16, mem_lock_type::read>& vec1, cldnn::mem_lock<ov::float16, mem_lock_type::read>& memLockVec2) {
        if (vec1.size() != memLockVec2.size()) {
            std::cerr << "Vectors must be of the same size." << std::endl;
            return -1.0f;
        }

        float dotProduct = std::inner_product(vec1.begin(), vec1.end(), memLockVec2.begin(), 0.0f);

        float magnitude1 = std::sqrt(std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0f));
        float magnitude2 = std::sqrt(std::inner_product(memLockVec2.begin(), memLockVec2.end(), memLockVec2.begin(), 0.0f));

        if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
            std::cerr << "One of the vectors is zero vector." << std::endl;
            return -1.0f;
        }

        return dotProduct / (magnitude1 * magnitude2);
    }
};

struct sdpa_gpu_test_02 : public spda_gpu_test {
    virtual std::string get_target_node_name() override {
        return "sdpa:__module.transformer_blocks.0.attn1/aten::scaled_dot_product_attention/ScaledDotProductAttention";
    }

    std::vector<std::string> get_bin_names() override {
        std::vector<std::string> bin_names = {
            "program1_network1_0__001472_sdpa___module.transformer_blocks.0.attn1_aten__scaled_dot_product_attention_ScaledDotProductAttention_src0__f16__2_990_32_64__bfyx.bin",
            "program1_network1_0__001472_sdpa___module.transformer_blocks.0.attn1_aten__scaled_dot_product_attention_ScaledDotProductAttention_src1__f16__2_990_32_64__bfyx.bin",
            "program1_network1_0__001472_sdpa___module.transformer_blocks.0.attn1_aten__scaled_dot_product_attention_ScaledDotProductAttention_src2__f16__2_990_32_64__bfyx.bin",
            "program1_network1_0__001472_sdpa___module.transformer_blocks.0.attn1_aten__scaled_dot_product_attention_ScaledDotProductAttention_dst0__f16__2_32_990_64__bfyx.bin"
        };
        return bin_names;
    }

    virtual cldnn::memory::ptr run_network(bool is_caching_test, bool use_micro_sdpa = false, bool use_ref_inputs = false) override {
        auto& engine = get_test_engine();
        cldnn::layout input0_dyn_layout({-1, -1, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input1_dyn_layout({-1, -1, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input2_dyn_layout({-1, -1, 32, 64}, data_types::f16, format::bfyx);

        cldnn::layout input0_static_layout({2, 990, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input1_static_layout({2, 128, 32, 64}, data_types::f16, format::bfyx);
        cldnn::layout input2_static_layout({2, 128, 32, 64}, data_types::f16, format::bfyx);

        auto input0 = engine.allocate_memory(input0_static_layout);
        auto input1 = engine.allocate_memory(input1_static_layout);
        auto input2 = engine.allocate_memory(input2_static_layout);

        load_input(input0, 0, use_ref_inputs);
        load_input(input1, 1, use_ref_inputs);
        load_input(input2, 2, use_ref_inputs);

        GPU_DEBUG_COUT << "Topology: SDPA kernel test " << std::endl;
        GPU_DEBUG_COUT << "* use micro_sdpa           : " << (use_micro_sdpa ? "Yes" : "No") << std::endl;
        GPU_DEBUG_COUT << "* input0 : " << input0_static_layout.to_short_string() << ", " << input0_static_layout.count() << std::endl;
        GPU_DEBUG_COUT << "* input1 : " << input1_static_layout.to_short_string() << ", " << input1_static_layout.count() << std::endl;
        GPU_DEBUG_COUT << "* input2 : " << input2_static_layout.to_short_string() << ", " << input2_static_layout.count() << std::endl;

        topology topo;
        topo.add(input_layout("input0", input0_dyn_layout));
        topo.add(input_layout("input1", input1_dyn_layout));
        topo.add(input_layout("input2", input2_dyn_layout));
        topo.add(scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2")},
            false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false));
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        if (use_micro_sdpa) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"sdpa", {format::type::bfyx, "sdpa_micro"}} }));
        } else {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"sdpa", {format::type::bfyx, "sdpa_ref"}} }));
        }

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);
        net->set_input_data("input2", input2);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return output;
    }
};

struct sdpa_gpu_test_origin : public spda_gpu_test {
    const std::string ref_data_path = "/home/ahnyoung/cldnn/cvs_164660/dumps/outs/gpu.fp16.sdpa.ref.raw/";
    const std::string opt_data_path = "/home/ahnyoung/cldnn/cvs_164660/dumps/outs/gpu.fp16.sdpa.micro.raw/";

    virtual std::vector<std::string> get_bin_names() override {
        std::vector<std::string> bin_names = {
            "program1_network1_0_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_dst0__f16__2_32_990_64__bfyx.bin",
            "program1_network1_0_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src0__f16__2_990_32_64__bfyx.bin",
            "program1_network1_0_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src1__f16__2_128_32_64__bfyx.bin",
            "program1_network1_0_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src2__f16__2_128_32_64__bfyx.bin",
            "program1_network1_0_sdpa___module.transformer_blocks.0.attn2_aten__scaled_dot_product_attention_ScaledDotProductAttention_src3__f16__2_32_1_128__bfyx.bin"
        };
        return bin_names;
    }

    virtual void load_input(cldnn::memory::ptr mem, size_t idx, const bool use_ref_inputs = false) override {
        auto bin_names = get_bin_names();
        ASSERT_TRUE(idx < bin_names.size());
        std::string input_file_name = (use_ref_inputs ? ref_data_path : opt_data_path) + bin_names[idx];
        load_data_from_bin(mem, input_file_name);
    }
};

TEST(sdpa_gpu_test, basic) {
    sdpa_gpu_test_origin test;
    GPU_DEBUG_COUT << "Use origin (opt kernel) kernel inputs" << std::endl;
    test.execute(false, false);
}

TEST(sdpa_gpu_test, basic_01) {
    spda_gpu_test test;
    GPU_DEBUG_COUT << "Use ref kernel inputs" << std::endl;
    test.execute(false, true);
}
TEST(sdpa_gpu_test, basic_02) {
    spda_gpu_test test;
    GPU_DEBUG_COUT << "Use micro kernel inputs" << std::endl;
    test.execute(false, false);
}

TEST(sdpa_gpu_test, basic_03) {
    sdpa_gpu_test_02 test;
    test.execute();
}

// #endif
} // namespace
