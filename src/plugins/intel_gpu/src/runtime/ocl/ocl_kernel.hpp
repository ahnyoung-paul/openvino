// Copyright (C) 2016-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace ocl {

class ocl_kernel : public kernel {
    ocl_kernel_type _compiled_kernel;
    std::string _kernel_id;

public:
    ocl_kernel(ocl_kernel_type compiled_kernel, const std::string& kernel_id)
        : _compiled_kernel(compiled_kernel)
        , _kernel_id(kernel_id) { }

    const ocl_kernel_type& get_handle() const { return _compiled_kernel; }
    ocl_kernel_type& get_handle() { return _compiled_kernel; }
    std::shared_ptr<kernel> clone() const override { return std::make_shared<ocl_kernel>(get_handle().clone(), _kernel_id); }

    size_t getProgramBinSize() const {
        cl_int ret;
        auto program = _compiled_kernel.getInfo<CL_KERNEL_PROGRAM>(&ret);
        if (ret != 0) {
            std::cout << "Fail to get info" << std::endl;
            return 0;
        }
        std::vector<size_t> binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
        if (binary_sizes.size() != 1)
            throw std::runtime_error("Invalid binaries count because cl_prog can't get program binary size");

        size_t binary_size = binary_sizes.front();
        return binary_size;
    }
};

}  // namespace ocl
}  // namespace cldnn
