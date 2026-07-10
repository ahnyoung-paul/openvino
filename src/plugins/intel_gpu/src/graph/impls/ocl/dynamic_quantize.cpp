// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "primitive_base.hpp"
#include "dynamic_quantize/dynamic_quantize_kernel_ref.h"
#include "dynamic_quantize/dynamic_quantize_kernel_selector.h"
#include "dynamic_quantize_inst.h"
#include "fully_connected_inst.h"

namespace cldnn {
namespace ocl {

struct dynamic_quantize_impl : typed_primitive_impl_ocl<dynamic_quantize> {
    using parent = typed_primitive_impl_ocl<dynamic_quantize>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::dynamic_quantize_kernel_selector;
    using kernel_params_t = kernel_selector::dynamic_quantize_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::dynamic_quantize_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<dynamic_quantize_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::dynamic_quantize_params>(impl_param, is_shape_agnostic);
        const auto& primitive = impl_param.typed_desc<dynamic_quantize>();
        params.outputs.push_back(convert_data_tensor(impl_param.get_output_layout(1)));
        params.name = primitive->id;
        // In Some model, the feature size could be dynamic in input0.
        // It refers to IFM value of weight of fully connected.
        auto user_node = impl_param.prog->get_node(impl_param.desc->id).get_users().front();
        if (user_node != nullptr && user_node->is_type<fully_connected>()) {
            auto& fc_node = user_node->as<fully_connected>();
            params.fc_ifm_size = fc_node.weights().get_output_layout().feature();
            // if (fc_node.id() == "fullyconnectedcompressed:__module.model.language_model.layers.0.self_attn.o_proj/ov_ext::linear/MatMul") {
            // if (params.name == "dynamicquantize:DynamicQuantize_323093") {
            {
                auto wl = fc_node.weights().get_output_layout();
                auto& in0 = params.inputs[0];
                auto out0_layout = params.outputs[0].GetLayout();
                size_t computed_input_f = in0.Feature().v;
                size_t computed_input_batch = in0.Batch().v;
                if (out0_layout == kernel_selector::DataLayout::bfyx) {
                    computed_input_f = in0.Y().v * in0.X().v;
                    computed_input_batch = in0.Batch().v * in0.Feature().v;
                }
                GPU_DEBUG_COUT << "DQ [" << primitive->id << "] fc_ifm_size debug: fc_id=" << fc_node.id()
                    << " is_shape_agnostic=" << is_shape_agnostic
                    << " weight_layout=" << wl.to_short_string()
                    << " weight_batch=" << wl.batch()
                    << " weight_feature=" << wl.feature()
                    << " fc_ifm_size=" << params.fc_ifm_size
                    << " input0_dynamic=" << in0.is_dynamic()
                    << " input0_layout=" << static_cast<int>(in0.GetLayout())
                    << " input0_batch=" << in0.Batch().v
                    << " input0_feature=" << in0.Feature().v
                    << " input0_Y=" << in0.Y().v
                    << " input0_X=" << in0.X().v
                    << " output0_layout=" << static_cast<int>(out0_layout)
                    << " computed_input_f=" << computed_input_f
                    << " computed_input_batch=" << computed_input_batch
                    << std::endl;
            }
        }

        if (impl_param.output_layouts.size() > 2)
            params.outputs.push_back(convert_data_tensor(impl_param.get_output_layout(2)));

        // Keep 2d data as bf layout
        if (primitive->input_size == 2)
            params.outputs[0] = params.outputs[0].FlattenFeatureAndSpatials();

        const auto& desc = impl_param.typed_desc<dynamic_quantize>();
        params.group_sizes = desc->attrs.group_sizes;
        params.scales_output_order = desc->attrs.scales_zp_output_order;
        params.use_asymmetric_quantization = desc->attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        params.combine_scales_and_zp = desc->attrs.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        params.generate_precomputed_reduction = desc->attrs.precomputed_reduction;

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        if (impl_param.can_be_optimized()) {
            return;
        }

        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }
};

namespace detail {

attach_dynamic_quantize_impl::attach_dynamic_quantize_impl() {
    auto types = {
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::f8e4m3,
        data_types::f8e5m2,
        data_types::f8e8m0,
    };

    auto formats = {
        format::bfyx,
    };

    implementation_map<dynamic_quantize>::add(impl_types::ocl,
                                    shape_types::any,
                                    typed_primitive_impl_ocl<dynamic_quantize>::create<dynamic_quantize_impl>,
                                    types,
                                    formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::dynamic_quantize_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::dynamic_quantize)
