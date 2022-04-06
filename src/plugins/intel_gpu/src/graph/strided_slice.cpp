// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

#include "strided_slice_shape_inference.hpp"

namespace cldnn {
primitive_type_id strided_slice::type_id() {
    static primitive_type_base<strided_slice> instance;
    return &instance;
}

layout strided_slice_inst::calc_output_layout(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto output_format = format::get_default_format(desc->out_size.size());
    if (node.const_mem.empty()) {
        return layout{input_layout.data_type, output_format, ov::PartialShape::dynamic(input_layout.size.rank())};
    }

    {
        ov::op::v1::StridedSlice op;
        std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
        std::vector<ov::PartialShape> input_shapes = {
            node.get_dependency(0).get_output_layout().size,
            node.get_dependency(1).get_output_layout().size,
            node.get_dependency(2).get_output_layout().size,
            node.get_dependency(3).get_output_layout().size
        };

        auto begin_mask = desc->begin_mask;
        auto end_mask = desc->end_mask;
        auto new_axis_mask = desc->new_axis_mask;
        auto shrink_axis_mask = desc->shrink_axis_mask;

        op.set_begin_mask(desc->begin_mask);
        op.set_end_mask(desc->end_mask);
        op.set_new_axis_mask(desc->new_axis_mask);
        op.set_shrink_axis_mask(desc->shrink_axis_mask);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(node.const_mem[0], node.get_program().get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(node.const_mem[1], node.get_program().get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(node.const_mem[2], node.get_program().get_stream());

        auto ptr1 = lock1.data();
        auto ptr2 = lock2.data();
        auto ptr3 = lock3.data();

        auto make_tensor = [](layout l, void* memory_pointer) {
            ov::element::Type et;

            switch (l.data_type) {
                case data_types::i64: et = ov::element::i64; break;
                case data_types::i32: et = ov::element::i32; break;
                default: IE_THROW() << "unsupported element type in strided slice primitive";
            }

            return std::make_shared<ngraph::runtime::HostTensor>(et, l.size.to_shape(), memory_pointer);
        };


        auto tensor1 = make_tensor(node.const_mem[0]->get_layout(), ptr1);
        auto tensor2 = make_tensor(node.const_mem[1]->get_layout(), ptr2);
        auto tensor3 = make_tensor(node.const_mem[2]->get_layout(), ptr3);

        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, tensor1},
            {2, tensor2},
            {3, tensor3},
        };
        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
        return layout{input_layout.data_type, output_format, output_shapes[0]};
    }
    return layout{input_layout.data_type, output_format, desc->out_size};
}

void strided_slice_inst::update_shape() {
    if (!_network.shape_changed())
        return;

    auto& node = const_cast<strided_slice_node&>(dynamic_cast<const strided_slice_node&>(_node));
    auto in_mem1 = _network.get_output_memory(_node.get_dependency(1).id());
    auto in_mem2 = _network.get_output_memory(_node.get_dependency(2).id());
    auto in_mem3 = _network.get_output_memory(_node.get_dependency(3).id());

    node.const_mem = {in_mem1, in_mem2, in_mem3};

    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto new_layout = _node.type()->calc_output_layout(_node);
    auto out_layout = _node.is_valid_output_layout() ? _node.get_output_layout() : layout(data_types::f32, format::any, tensor{});
    auto out_layout_str = _node.is_valid_output_layout() ? out_layout.to_string() : "invalid";
    GPU_DEBUG_IF(debug_config->verbose >= 4) {
        GPU_DEBUG_COUT << id() << " update shape: was: " << out_layout_str << " now: " << new_layout.to_string() << std::endl;
    }
    if (!_node.is_valid_output_layout() || _node.get_output_layout() != new_layout)
        set_shape_change();
    // TODO: Get rid of this const_cast
    node.set_output_layout(new_layout);
}

std::string strided_slice_inst::to_string(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite strided_slice_info;
    strided_slice_info.add("input id", input.id());
    strided_slice_info.add("begin_param id", node.get_dependency(1).id());
    strided_slice_info.add("end_param id", node.get_dependency(2).id());
    strided_slice_info.add("stride_param id", node.get_dependency(3).id());
    strided_slice_info.add("begin mask", node.get_primitive()->begin_mask);
    strided_slice_info.add("end mask", node.get_primitive()->end_mask);
    strided_slice_info.add("new axis mask", node.get_primitive()->new_axis_mask);
    strided_slice_info.add("shrink axis mask", node.get_primitive()->shrink_axis_mask);
    strided_slice_info.add("begin_param shape", node.get_dependency(1).get_output_layout().to_string());
    strided_slice_info.add("end_param shape", node.get_dependency(2).get_output_layout().to_string());
    strided_slice_info.add("stride_param shape", node.get_dependency(3).get_output_layout().to_string());

    node_info->add("strided_slice info", strided_slice_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

strided_slice_inst::typed_primitive_inst(network& network, strided_slice_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
