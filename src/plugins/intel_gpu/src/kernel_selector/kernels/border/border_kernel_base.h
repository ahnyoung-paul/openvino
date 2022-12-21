// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// border_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct border_params : public base_params {
    DimTensor<> lt_sizes;
    DimTensor<> rb_sizes;
    BorderType b_type;
    float border_value;

    border_params() : base_params(KernelType::BORDER), b_type(BorderType::CONSTANT), border_value(0.0f) {}

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        // k.EnableBorderType(b_type);
        return k;
    }

    size_t hash() const override {
        auto seed = base_params::hash();
        auto hash_combine_dimtensor = [&](size_t s, kernel_selector::DimTensor<> tensor) -> size_t {
            s = hash_combine(s, tensor.b);
            s = hash_combine(s, tensor.f);
            s = hash_combine(s, tensor.w);
            s = hash_combine(s, tensor.x);
            s = hash_combine(s, tensor.y);
            s = hash_combine(s, tensor.x);
            return s;
        };

        seed = hash_combine_dimtensor(seed, params.lt_sizes);
        seed = hash_combine_dimtensor(seed, params.rb_sizes);
        seed = hash_combine(seed, params.border_value);
        seed = hash_combine(seed, params.b_type);
        return seed;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// border_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct border_optional_params : optional_params {
    border_optional_params() : optional_params(KernelType::BORDER) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BorderKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BorderKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const border_params& params) const;
    DispatchData SetDefault(const border_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
