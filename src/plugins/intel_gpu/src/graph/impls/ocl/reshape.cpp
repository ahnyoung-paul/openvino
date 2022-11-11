// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reshape/reshape_kernel_ref.h"
#include "reshape/reshape_kernel_selector.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct reshape_impl : public typed_primitive_impl_ocl<reshape> {
    using parent = typed_primitive_impl_ocl<reshape>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reshape_impl>(*this);
    }

public:
    static primitive_impl* create(reshape_node const& arg, const kernel_impl_params& impl_param) {
        if (arg.can_be_optimized()) {
            if (arg.id() == "reshape:812") {
                std::cout << "[" << arg.id() << "] create empty impl " << std::endl;
            }
            return new reshape_impl(arg, {});
        }
        auto reorder_params = get_default_params<kernel_selector::reshape_params>(impl_param);
        auto reorder_optional_params =
            get_default_optional_params<kernel_selector::reshape_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::reshape_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reorder_params, reorder_optional_params);

        if (arg.id() == "reshape:812") {
            std::cout << "[" << arg.id() << "], get kernel params " << reorder_params.to_cache_string_v2();
            std::cout << ", kernel_name " << best_kernels[0].kernelName;
            std::cout << ", can_be_optimized(" << (arg.can_be_optimized()? "True)" : "False)") << std::endl;
        }

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reshape = new reshape_impl(arg, best_kernels[0]);

        return reshape;
    }
};

namespace detail {

attach_reshape_impl::attach_reshape_impl() {
    implementation_map<reshape>::add(impl_types::ocl, reshape_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
