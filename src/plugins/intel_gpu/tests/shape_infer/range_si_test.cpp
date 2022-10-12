// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "range_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct range_si_test_params {
    data_types out_data_type;
    std::vector<double> vals;   // {start, stop, step}
    ov::Dimension::value_type expected_dim;
};

std::ostream& operator<<(std::ostream& ost, const range_si_test_params& params) {
    ost << data_type_traits::name(params.out_data_type) << ",";
    ost << "{start:" << params.vals[0] << ",stop:" << params.vals[1] << ",step:" << params.vals[2] << "},";
    ost << params.expected_dim;
    return ost;
}

class range_si_test : public testing::TestWithParam<range_si_test_params> { };

TEST_P(range_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<std::string> input_prim_ids;

    for (size_t idx = 0; idx < p.vals.size(); idx++) {
        auto prim_id = "const::data_" + std::to_string(idx);
        auto prim_mem = engine.allocate_memory(layout{ov::PartialShape{}, p.out_data_type, format::bfyx});

        switch (p.out_data_type) {
            case data_types::f16:
                set_values(prim_mem, {float_to_half(p.vals[idx])});
                break;
            case data_types::f32:
                set_values(prim_mem, {static_cast<data_type_to_type<data_types::f32>::type>(p.vals[idx])});
                break;
            case data_types::i32:
                set_values(prim_mem, {static_cast<data_type_to_type<data_types::i32>::type>(p.vals[idx])});
                break;
            case data_types::i64:
                set_values(prim_mem, {static_cast<data_type_to_type<data_types::i64>::type>(p.vals[idx])});
                break;
            case data_types::i8:
                set_values(prim_mem, {static_cast<data_type_to_type<data_types::i8>::type>(p.vals[idx])});
                break;
            case data_types::u8:
                set_values(prim_mem, {static_cast<data_type_to_type<data_types::u8>::type>(p.vals[idx])});
                break;
        }

        auto const_data_prim = std::make_shared<data>(prim_id, prim_mem);
        input_prims.push_back(const_data_prim);
        input_prim_ids.push_back(prim_id);
    }

    layout out_layout {ov::PartialShape{p.expected_dim}, p.out_data_type, format::bfyx};

    auto range_prim = std::make_shared<range>("range", input_prim_ids, out_layout);
    auto& range_node = prog.get_or_create(range_prim);

    for (auto& iprim : input_prims) {
        auto& input_node = prog.get_or_create(iprim);
        program_wrapper::add_connection(prog, input_node, range_node);
    }

    auto params = range_node.get_kernel_impl_params();
    auto res = range_inst::calc_output_layouts<ov::PartialShape>(range_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], out_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, range_si_test,
    testing::ValuesIn(std::vector<range_si_test_params>{
        {data_types::i32, {2, 23, 3}, 7},
        {data_types::i8, {2, 23, 3}, 7},
        {data_types::u8, {2, 23, 3}, 7},
        {data_types::i64, {23, 2, -3}, 7},
        {data_types::i32, {23, 2, -3}, 7},
        {data_types::f32, {1.0f, 2.5f, 0.5f}, 3},
        {data_types::f16, {1.0f, 2.5f, 0.5f}, 3}
    }));

};  // shape_infer_tests