// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ed_do/detection_output_kernel_ref.h"
#include "ed_do/detection_output_kernel_selector.h"
#include "experimental_detectron_detection_output_inst.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {
struct experimental_detectron_detection_output_impl
    : public typed_primitive_impl_ocl<experimental_detectron_detection_output> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_detection_output>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_detection_output_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::experimental_detectron_detection_output_params,
                                      kernel_selector::experimental_detectron_detection_output_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_detection_output_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<experimental_detectron_detection_output>& instance,
                                        int32_t unused) const override {
        kernel_arguments_data args = parent::get_arguments(instance, unused);
        args.inputs.push_back(instance.output_classes_memory());
        args.inputs.push_back(instance.output_scores_memory());

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_detection_output>();
        auto params = get_default_params<kernel_selector::experimental_detectron_detection_output_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::experimental_detectron_detection_output_optional_params>(impl_param.get_program());

        params.score_threshold = primitive->score_threshold;
        params.nms_threshold = primitive->nms_threshold;
        params.max_delta_log_wh = primitive->max_delta_log_wh;
        params.num_classes = primitive->num_classes;
        params.post_nms_count = primitive->post_nms_count;
        params.max_detections_per_image = primitive->max_detections_per_image;
        params.class_agnostic_box_regression = primitive->class_agnostic_box_regression;
        params.deltas_weights = primitive->deltas_weights;

        for (size_t i = 1; i < impl_param.input_layouts.size(); i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        return {params, optional_params};
    }

    // static size_t update_hash(size_t seed, const kernel_selector::experimental_detectron_detection_output_params& params) {
    //     seed = hash_combine(seed, params.score_threshold);
    //     seed = hash_combine(seed, params.nms_threshold);
    //     seed = hash_combine(seed, params.max_delta_log_wh);
    //     seed = hash_combine(seed, params.num_classes);
    //     seed = hash_combine(seed, params.post_nms_count);
    //     seed = hash_combine(seed, params.max_detections_per_image);
    //     seed = hash_combine(seed, params.class_agnostic_box_regression);
    //     seed = hash_combine_vec(seed, params.deltas_weights);
    //     return seed;
    // }
};

namespace detail {
attach_experimental_detectron_detection_output_impl::attach_experimental_detectron_detection_output_impl() {
    const std::vector<data_types> types {data_types::f16, data_types::f32};
    const std::vector<format::type> formats = {format::bfyx,
                        format::b_fs_yx_fsv16,
                        format::b_fs_yx_fsv32,
                        format::bs_fs_yx_bsv16_fsv16,
                        format::bs_fs_yx_bsv32_fsv32,
                        format::bs_fs_yx_bsv32_fsv16};

    implementation_map<experimental_detectron_detection_output>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_detection_output>::create<experimental_detectron_detection_output_impl>,
        types, formats);

    impl_hash_key<experimental_detectron_detection_output>::add(
        typed_primitive_impl_ocl<experimental_detectron_detection_output>
        ::get_impl_key<experimental_detectron_detection_output_impl>);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_detection_output_impl)
