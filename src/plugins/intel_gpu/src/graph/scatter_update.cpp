// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id scatter_update::type_id() {
    static primitive_type_base<scatter_update> instance;
    return &instance;
}

static size_t GetNonEmptyDimsNumber(const layout& layout) {
    if (layout.count() != 1) {
        // Count the number of "one size" dimensions starting with X to Batch
        size_t one_size_dims = 0;
        std::vector<int32_t> dims = layout.get_dims();
        for (size_t i = 0; i < dims.size(); i++) {
            if (dims[dims.size() - 1 - i] == 1)
                one_size_dims++;
            else
                break;
        }
        return dims.size() - one_size_dims;
    } else {
        return 1;
    }
}

layout scatter_update_inst::calc_output_layout(scatter_update_node const& node) {
    auto desc = node.get_primitive();

    const int32_t axis = desc->axis;
    const size_t indices_size = node.input(1).get_output_layout().count();
    const size_t input_number_of_dims = node.input(0).get_output_layout().get_rank();
    const size_t updates_number_of_dims = node.input(2).get_output_layout().get_rank();

    // convert axis for ie format
    auto ie_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        const size_t default_dims = 4;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max(input_number_of_dims, default_dims) - 2;
        ie_axis = spatial_size - spatial_axis - 1 + 2;
    }

    auto input_layout = node.input(0).get_output_layout();

    auto output_shape = input_layout.size;
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    if (static_cast<size_t>(ie_axis) < 0 || static_cast<size_t>(ie_axis) >= input_number_of_dims)
        CLDNN_ERROR_MESSAGE(node.id(), "Incorrect axis value for ScatterUpdate: Axis must be positive and less than the input tensor dimension.");

    if (indices_size > static_cast<size_t>(input_layout.get_dims()[ie_axis])) {
        CLDNN_ERROR_MESSAGE(node.id(),
            "Undefined behavior ScatterUpdate: indices size must not be larger than the output size along the Axis.");
    }

    if (static_cast<size_t>(ie_axis) > updates_number_of_dims) {
        CLDNN_ERROR_MESSAGE(node.id(),
            "Undefined behavior ScatterUpdate: indices dimention must not be larger than the updates[:Axis] dimentional size.");
    }

    return layout{output_type, input_format, output_shape};
}

std::string scatter_update_inst::to_string(scatter_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_update_info;
    scatter_update_info.add("input id", input.id());
    scatter_update_info.add("axis", desc->axis);
    scatter_update_info.add("output shape", node.input(0).get_output_layout().to_string());

    node_info->add("scatter_update info", scatter_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_update_inst::typed_primitive_inst(network& network, scatter_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
