// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/range.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<range> : public typed_program_node_base<range> {
private:
    using parent = typed_program_node_base<range>;

public:
    using parent::parent;
    program_node& input(std::size_t i = 0) const { return get_dependency(i); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {0, 1, 2}; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const layout& out_layout) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layout);
        params->output_layout = get_primitive()->output_layout;
        return params;
    }
};
using range_node = typed_program_node<range>;

template <>
class typed_primitive_inst<range> : public typed_primitive_inst_base<range> {
public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(range_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(range_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(range_node const& node);

    typed_primitive_inst(network& network, range_node const& desc);
};

using range_inst = typed_primitive_inst<range>;

}  // namespace cldnn
