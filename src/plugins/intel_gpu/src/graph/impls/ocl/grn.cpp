// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "grn/grn_kernel_selector.h"
#include "grn/grn_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct grn_impl : typed_primitive_impl_ocl<grn> {
    using parent = typed_primitive_impl_ocl<grn>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::grn_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::grn_params, kernel_selector::grn_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<grn_impl>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<grn>();
        auto params = get_default_params<kernel_selector::grn_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::grn_optional_params>(impl_param.get_program());

        params.bias = primitive->bias;

        return {params, optional_params};
    }

    static size_t update_hash(size_t seed, const kernel_selector::grn_params& params) {
        seed = hash_combine(seed, params.bias);
        return seed;
    }
};

namespace detail {

attach_grn_impl::attach_grn_impl() {
    implementation_map<grn>::add(impl_types::ocl, typed_primitive_impl_ocl<grn>::create<grn_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });

    impl_hash<grn>::add(typed_primitive_impl_ocl<grn>::get_hash_key<grn_impl>);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::grn_impl)
