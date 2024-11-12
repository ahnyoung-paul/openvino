// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/primitives/scatter_nd_update.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "permute_inst.h"
#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace skip_scatter_update_tests {

enum scatter_update_type {
	ScatterUpdate           = 0,
	ScatterNDUpdate         = 1,
	ScatterElementsUpdate   = 2
};

struct skip_scatter_update_params2 {
    scatter_update_type scatter_type;
    bool scatter_update_01_skipped;
    bool scatter_update_02_skipped;
};

struct iter_params {
    ov::PartialShape input_shape;
    ov::PartialShape idx_nonzero_shape;
    ov::PartialShape idx_zero_shape;
    ov::PartialShape update_nonzero_shape;
    ov::PartialShape update_zero_shape;
    bool skipped_01;
    bool skipped_03;
};

struct skip_scatter_update_params {
    scatter_update_type scatter_type;
    iter_params iter0;
    iter_params iter1;
};

class skip_scatter_update_at_runtime_test : public testing::TestWithParam<skip_scatter_update_params> {};

TEST_P(skip_scatter_update_at_runtime_test, runtime_skip) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    auto p_iter0 = p.iter0;
    auto p_iter1 = p.iter1;

    auto input_rank = p.iter0.input_shape.size();
    auto input_layout_dynamic = layout {ov::PartialShape::dynamic(input_rank), data_types::f16, format::get_default_format(input_rank)};
    auto idx_rank = p.iter0.idx_nonzero_shape.size();
    auto idx_layout_dynamic   = layout {ov::PartialShape::dynamic(idx_rank), data_types::f16, format::get_default_format(idx_rank)};

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = nullptr;

    if (p.scatter_type == scatter_update_type::ScatterElementsUpdate) {
        topology topology(input_layout("input", input_layout_dynamic),
                            input_layout("idx1", input_layout_dynamic),
                            input_layout("idx2", input_layout_dynamic),
                            input_layout("idx3", input_layout_dynamic),
                            input_layout("update1", input_layout_dynamic),
                            input_layout("update2", input_layout_dynamic),
                            input_layout("update3", input_layout_dynamic),
                            scatter_elements_update("scatter1", input_info("input"),    input_info("idx1"), input_info("update1"), 0),
                            scatter_elements_update("scatter2", input_info("scatter1"), input_info("idx2"), input_info("update2"), 0),
                            scatter_elements_update("scatter3", input_info("scatter2"), input_info("idx3"), input_info("update3"), 0),
                            reorder("reorder", input_info("scatter3"), format::get_default_format(input_rank), data_types::f32));

        network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    } else if (p.scatter_type == scatter_update_type::ScatterUpdate) {
        topology topology(input_layout("input", input_layout_dynamic),
                            input_layout("idx1", input_layout_dynamic),
                            input_layout("idx2", input_layout_dynamic),
                            input_layout("idx3", input_layout_dynamic),
                            input_layout("update1", input_layout_dynamic),
                            input_layout("update2", input_layout_dynamic),
                            input_layout("update3", input_layout_dynamic),
                            scatter_update("scatter1", input_info("input"),    input_info("idx1"), input_info("update1"), 0),
                            scatter_update("scatter2", input_info("scatter1"), input_info("idx2"), input_info("update2"), 0),
                            scatter_update("scatter3", input_info("scatter2"), input_info("idx3"), input_info("update3"), 0),
                            reorder("reorder", input_info("scatter3"), format::get_default_format(input_rank), data_types::f32));
        network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    } else if (p.scatter_type == scatter_update_type::ScatterNDUpdate) {
        topology topology(input_layout("input", input_layout_dynamic),
                            input_layout("idx1", input_layout_dynamic),
                            input_layout("idx2", input_layout_dynamic),
                            input_layout("idx3", input_layout_dynamic),
                            input_layout("update1", input_layout_dynamic),
                            input_layout("update2", input_layout_dynamic),
                            input_layout("update3", input_layout_dynamic),
                            scatter_nd_update("scatter1", input_info("input"),    input_info("idx1"), input_info("update1"), 0),
                            scatter_nd_update("scatter2", input_info("scatter1"), input_info("idx2"), input_info("update2"), 0),
                            scatter_nd_update("scatter3", input_info("scatter2"), input_info("idx3"), input_info("update3"), 0),
                            reorder("reorder", input_info("scatter3"), format::get_default_format(input_rank), data_types::f32));
        network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    }

    // auto input_mem              = engine.allocate_memory(input_layout_static);

    // auto idx1_layout_static     = p.scatter_update_01_skipped? idx1_zero_layout : idx1_nonzero_layout;
    // auto update1_layout_static  = p.scatter_update_01_skipped? update1_zero_layout : update1_nonzero_layout;

    // auto idx1_mem               = engine.allocate_memory(idx1_nonzero_layout);
    // auto update1_mem            = engine.allocate_memory(update1_nonzero_layout);

    // if (p.scatter_update_01_skipped) {
    //     idx1_mem    = engine.reinterpret_buffer(*idx1_mem, idx1_zero_layout);
    //     update1_mem = engine.reinterpret_buffer(*update1_mem, update1_zero_layout);
    // }

    // auto idx2_layout_static     = p.scatter_update_02_skipped? idx2_zero_layout : idx2_nonzero_layout;
    // auto update2_layout_static  = p.scatter_update_02_skipped? update2_zero_layout : update2_nonzero_layout;

    // auto idx2_mem               = engine.allocate_memory(idx2_nonzero_layout);
    // auto update2_mem            = engine.allocate_memory(update2_nonzero_layout);
    // if (p.scatter_update_02_skipped) {
    //     idx2_mem    = engine.reinterpret_buffer(*idx2_mem, idx2_zero_layout);
    //     update2_mem = engine.reinterpret_buffer(*update2_mem, update2_zero_layout);
    // }
    // network->set_input_data("input", input_mem);
    // network->set_input_data("idx1", idx1_mem);
    // network->set_input_data("idx2", idx2_mem);
    // network->set_input_data("update1", update1_mem);
    // network->set_input_data("update2", update2_mem);
    // auto outputs = network->execute();
    // outputs.begin()->second.get_memory();

    // auto input_inst = network->get_primitive("input");
    // auto scatter1_inst = network->get_primitive("scatter1");
    // auto scatter2_inst = network->get_primitive("scatter2");

    // ASSERT_EQ(scatter1_inst->can_be_optimized(), p.scatter_update_01_skipped);
    // ASSERT_EQ(scatter2_inst->can_be_optimized(), p.scatter_update_02_skipped);

    // if (scatter1_inst->can_be_optimized()) {
    //     ASSERT_TRUE(engine.is_the_same_buffer(scatter1_inst->dep_memory(0),     scatter1_inst->output_memory(0)));
    // } else {
    //     ASSERT_FALSE(engine.is_the_same_buffer(scatter1_inst->dep_memory(0),    scatter1_inst->output_memory(0)));
    // }

    // if (scatter2_inst->can_be_optimized()) {
    //     ASSERT_TRUE(engine.is_the_same_buffer(scatter2_inst->dep_memory(0),     scatter2_inst->output_memory(0)));
    // } else {
    //     ASSERT_FALSE(engine.is_the_same_buffer(scatter2_inst->dep_memory(0),    scatter2_inst->output_memory(0)));
    // }
}

INSTANTIATE_TEST_SUITE_P(smoke, skip_scatter_update_at_runtime_test,
    testing::ValuesIn(std::vector<skip_scatter_update_params> {
        { scatter_update_type::ScatterUpdate,           {{8,16}, {8,16}, {0,16}, {8,16}, {0,16}, true,  true},  {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  true}},
        { scatter_update_type::ScatterUpdate,           {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, true,  true},  {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  true}},
        { scatter_update_type::ScatterUpdate,           {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, true,  true},  {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  false}},
        { scatter_update_type::ScatterUpdate,           {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  true}, {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, true,  false}},

        { scatter_update_type::ScatterNDUpdate,         {{8,16}, {8,16}, {0,16}, {8,16}, {0,16}, true,  true},  {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  true}},
        { scatter_update_type::ScatterNDUpdate,         {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, true,  true},  {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  true}},
        { scatter_update_type::ScatterNDUpdate,         {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, true,  true},  {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  false}},
        { scatter_update_type::ScatterNDUpdate,         {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, false,  true}, {{1,16}, {1,16}, {0,16}, {1,16}, {0,16}, true,  false}},

        { scatter_update_type::ScatterElementsUpdate,   {{12}, {12,1}, {0,1}, {12}, {0}, true,  true}, {{16}, {16,1}, {0,1}, {16}, {0}, false,  true}},
        { scatter_update_type::ScatterElementsUpdate,   {{16}, {16,1}, {0,1}, {16}, {0}, true,  true},   {{16}, {16,1}, {0,1}, {16}, {0}, false,  true}},
        { scatter_update_type::ScatterElementsUpdate,   {{16}, {16,1}, {0,1}, {16}, {0}, true,  true},  {{16}, {16,1}, {0,1}, {16}, {0}, false,  false}},
        { scatter_update_type::ScatterElementsUpdate,   {{16}, {16,1}, {0,1}, {16}, {0}, false,  true}, {{16}, {16,1}, {0,1}, {16}, {0}, true,  false}},
    }));
}  // skip permute tests
