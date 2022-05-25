// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <vector>
#include <string>

namespace cldnn {
primitive_type_id reduce::type_id() {
    static primitive_type_base<reduce> instance;
    return &instance;
}

std::string debug_reduce_mode(reduce_mode mode) {
    switch (mode) {
        case reduce_mode::max:
            return "max";
        case reduce_mode::min:
            return "min";
        case reduce_mode::mean:
            return "mean";
        case reduce_mode::prod:
            return "prod";
        case reduce_mode::sum:
            return "sum";
        case reduce_mode::logical_and:
            return "logical_and";
        case reduce_mode::logical_or:
            return "logical_or";
        case reduce_mode::sum_square:
            return "sum_square";
        case reduce_mode::l1:
            return "l1";
        case reduce_mode::l2:
            return "l2";
        case reduce_mode::log_sum:
            return "log_sum";
        case reduce_mode::log_sum_exp:
            return "log_sum_exp";
        default:
            return "none";
    }
}

#define USE_PARSIAL_SHAPE
layout reduce_inst::calc_output_layout(reduce_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;
    auto format_dim = input_format.dimension();
    auto output_type = input_layout.data_type;
    auto mode = desc->mode;
    auto reduce_axes = desc->axes;


#ifdef USE_PARSIAL_SHAPE
    auto in_dims_0 = input_layout.get_tensor().sizes();
    auto in_dims = input_layout.get_dims();
    std::cout << "Test with FIXED codes" << std::endl;
    std::cout << "in_dims(get_dims)     : " << in_dims << std::endl;
    std::cout << "in_dims(tensor.sizes) : " << in_dims_0 << std::endl;
    std::cout << "reduce_axes           : " << reduce_axes << std::endl;
    for (size_t a = 0; a < reduce_axes.size(); a++) {
        reduce_axes[a] = (reduce_axes[a] > 1)? (in_dims.size() - reduce_axes[a] + 1)
            : reduce_axes[a];
    }
    std::cout << "reduce_axes   : " << reduce_axes << std::endl;
#else
    std::cout << "Test with ORIGINAL codes" << std::endl;
    auto in_dims = input_layout.get_tensor().sizes();
    std::cout << "in_dims(get_dims)     : " << std::endl;
    std::cout << "in_dims(tensor.sizes) : " << in_dims << std::endl;
    std::cout << "reduce_axes   : " << reduce_axes << std::endl;
    std::cout << "reduce_axes   : " << std::endl;
#endif
    std::cout << "mode         : " << debug_reduce_mode(mode) << std::endl;
    std::cout << "input_format : " << input_format.to_string() << std::endl;
    for (size_t a = 0; a < reduce_axes.size(); a++) {
        in_dims[reduce_axes[a]] = 1;
    }

    std::vector<int32_t> updated_dims;
    if (!desc->keep_dims) {
        // Get unreduced from b-f and x-w range
        for (size_t b_f_index = 0; b_f_index < 2; b_f_index++) {
            bool index_to_remove = std::find(reduce_axes.begin(), reduce_axes.end(), b_f_index) != reduce_axes.end();
            if (!index_to_remove)
                updated_dims.push_back(in_dims[b_f_index]);
        }
        std::cout << "updated_dims[1]   : " << updated_dims << std::endl;
#ifndef USE_PARSIAL_SHAPE
        for (size_t x_w_index = format_dim - 1; x_w_index >= 2; x_w_index--) {
            bool index_to_remove = std::find(reduce_axes.begin(), reduce_axes.end(), x_w_index) != reduce_axes.end();
            if (!index_to_remove)
                updated_dims.push_back(in_dims[x_w_index]);
        }
#else
        for (size_t x_w_index = 2; x_w_index < format_dim; x_w_index++) {
            bool index_to_remove = std::find(reduce_axes.begin(), reduce_axes.end(), x_w_index) != reduce_axes.end();
            if (!index_to_remove)
                updated_dims.push_back(in_dims[x_w_index]);
        }
#endif
        std::cout << "updated_dims[2]   : " << updated_dims << std::endl;
        if (input_format.dimension() == 4 && reduce_axes.size() == 1)
            updated_dims.push_back(1);
        std::cout << "updated_dims[3]   : " << updated_dims << std::endl;
#ifndef USE_PARSIAL_SHAPE
        if (updated_dims.size() > 2)
            std::reverse(updated_dims.begin() + 2, updated_dims.end());
#endif
        // Fill updated dims to format_dim size
        while (updated_dims.size() < format_dim) {
            if (updated_dims.size() > 2) {
                updated_dims.insert(std::next(updated_dims.begin(), 2), 1);
            } else {
                updated_dims.push_back(1);
            }
        }


        std::cout << "updated_dims[4]   : " << updated_dims << std::endl;
        in_dims = std::move(updated_dims);
    }

    std::vector<reduce_mode> reduce_bool_modes = {reduce_mode::logical_and, reduce_mode::logical_or};
    if (std::find(reduce_bool_modes.begin(), reduce_bool_modes.end(), mode) != reduce_bool_modes.end())
        output_type = data_types::i8;
    else if (output_type == data_types::i8 || output_type == data_types::u8)
        output_type = data_types::f32;

    if (desc->output_data_type)
        output_type = *desc->output_data_type;

    if (node.has_fused_primitives())
        output_type = node.get_fused_output_layout().data_type;

    std::cout << "out_dims   : " << in_dims << std::endl;
#ifdef USE_PARSIAL_SHAPE
    ov::Shape shape;
    if (format_dim == 6) {
        shape = ov::Shape{in_dims[0], in_dims[1], in_dims[2], in_dims[3], in_dims[4], in_dims[5]};
    } else if (format_dim == 5) {
        shape = ov::Shape{in_dims[0], in_dims[1], in_dims[2], in_dims[3], in_dims[4]};
    } else {
        shape = ov::Shape{in_dims[0], in_dims[1], in_dims[2], in_dims[3]};
    }
    auto l = layout{output_type, input_format, ov::PartialShape(shape)};
    std::cout << "output layout : " << l.get_dims() << std::endl;
    std::cout << "output layout[get_tensor.sizes] : " << l.get_tensor().sizes(l.format) << std::endl;
    return l;
#else
    cldnn::layout l = layout(output_type, input_format, tensor{});
    if (format_dim == 6)
        l = layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4], in_dims[5]))};
    else if (format_dim == 5)
        l = layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4]))};
    else
        l = layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3]))};
    std::cout << "output layout : " << l.get_dims() << std::endl;
    std::cout << "output layout[get_tensor.sizes] : " << l.get_tensor().sizes(l.format) << std::endl;
    return l;
#endif
}

std::string reduce_inst::to_string(reduce_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite reduce_info;
    reduce_info.add("input id", node.input(0).id());
    reduce_info.add("axes", desc->axes);
    reduce_info.add("keep_dims", desc->keep_dims);
    reduce_info.add("mode", static_cast<uint16_t>(desc->mode));

    node_info->add("reduce info", reduce_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reduce_inst::typed_primitive_inst(network& network, reduce_node const& node) : parent(network, node) {}

}  // namespace cldnn
