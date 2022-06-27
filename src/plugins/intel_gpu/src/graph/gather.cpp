// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

#include "gather_shape_inference.hpp"
#include "openvino/op/gather.hpp"

namespace cldnn {
primitive_type_id gather::type_id() {
    static primitive_type_base<gather> instance;
    return &instance;
}

layout gather_inst::calc_output_layout(gather_node const& node) {
    auto desc = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto output_type = input_layout.data_type;
    auto output_format = desc->output_format;

    auto convert_format = [](cldnn::format input_format, size_t dim_size)->cldnn::format {
        auto out_format = input_format;
        if (dim_size == 5) {
            switch (input_format) {
            case format::bfyx:
                out_format = format::get_default_format(dim_size);
                break;
            case format::b_fs_yx_fsv16:
                out_format = format::b_fs_zyx_fsv16;
                break;
            case format::b_fs_yx_fsv32:
                out_format = format::b_fs_zyx_fsv32;
                break;
            case format::bs_fs_yx_bsv16_fsv16:
                out_format = format::bs_fs_zyx_bsv16_fsv16;
                break;
            default:
                break;
            }
        } else if (dim_size == 6) {
            switch (input_format) {
            case format::bfyx:
            case format::bfzyx:
                out_format = format::get_default_format(dim_size);
                break;
            default:
                break;
            }
        }
        return out_format;
    };

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    {
        int64_t axis = desc->axis;
//         auto eliminate_one_dim = [&](ov::PartialShape in)->ov::PartialShape {
//             if (in.is_dynamic()) {
//                 return in;
//             }

//             auto in_shape = in.to_shape();
//             std::vector<tensor::value_type> dims(in_shape.begin(), in_shape.end());

//             // while (!dims.empty() && dims.back() == 1) {
//             //     if (dims.size() <= (axis + 1)) {
//             //         break;
//             //     }
//             //     dims.pop_back();
//             // }

//             for (int idx = (dims.size() - 1); idx >= 0; idx--) {
//                 if (dims[idx] == 1)
//                     dims.erase((dims.begin() + idx));
//             }

//             // for (int idx = dims.size() )

//             ov::Shape out_shape(dims.begin(), dims.end());
//             return std::move(ov::PartialShape(out_shape));
//         };
// #if 0
//         auto shape_input1 = eliminate_one_dim(node.get_dependency(0).get_output_layout().size);
//         auto shape_input2 = eliminate_one_dim(node.get_dependency(1).get_output_layout().size);
// #else
//         auto shape_input1 = node.get_dependency(0).get_output_layout().size;
//         auto shape_input2 = node.get_dependency(1).get_output_layout().size;
// #endif
        ov::op::v8::Gather op;
        op.set_batch_dims(desc->batch_dim);
        std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
        std::vector<ov::PartialShape> input_shapes = {
            node.get_dependency(0).get_output_layout().size,
            node.get_dependency(1).get_output_layout().size,
            ov::PartialShape{1} // axis input is removed on gather primitive creation, so we can't use get_dependency(2)
        };

        auto axis_tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {{2, axis_tensor}};
        ov::op::util::shape_infer(&op, input_shapes, output_shapes, const_data);
        // std::cout << "*****************************************************" << std::endl;
        // std::cout << "[Gather][input_shape0] : " << shape_input1 << std::endl;
        // std::cout << "[Gather][input_shape1] : " << shape_input2 << std::endl;
        // std::cout << "[Gather][axis]         : " << axis << std::endl;
        // std::cout << "[Gather][batch_dims]   : " << desc->batch_dim << std::endl;
        // std::cout << "[Gather][output_shape] : " << output_shapes[0] << std::endl;
        // std::cout << "*****************************************************" << std::endl;

        if (output_shapes[0].is_static()) {
            auto shape = output_shapes[0].to_shape();
            std::vector<tensor::value_type> dims_converted(shape.begin(), shape.end());
            output_format = convert_format(input_layout.format, dims_converted.size());
        }
        return layout{output_type, output_format, output_shapes[0]};
    }

    auto shape = desc->output_shape.to_shape();
    std::vector<tensor::value_type> dims_converted(shape.begin(), shape.end());
    // extend shape to 4d
    for (size_t i = dims_converted.size(); i < 4; i++)
        dims_converted.push_back(1);

    output_format = convert_format(input_layout.format, dims_converted.size());
    return layout{output_type,
                  output_format,
                  tensor(format::get_default_format(dims_converted.size()), dims_converted)};
}

std::string gather_inst::to_string(gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_info;
    gather_info.add("input id", input.id());
    gather_info.add("axis", desc->axis);
    gather_info.add("batch_dim", desc->batch_dim);
    gather_info.add("output shape", cldnn::to_string(desc->output_shape));

    node_info->add("gather info", gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_inst::typed_primitive_inst(network& network, gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
