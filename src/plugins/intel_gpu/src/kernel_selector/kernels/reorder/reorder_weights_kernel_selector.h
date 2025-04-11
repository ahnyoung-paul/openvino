// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class ReorderWeightsKernelSelector : public kernel_selector_base {
public:
    static ReorderWeightsKernelSelector& Instance() {
        static ReorderWeightsKernelSelector instance_;
        return instance_;
    }

    ReorderWeightsKernelSelector();

    virtual ~ReorderWeightsKernelSelector() {}

    KernelsData GetNaiveBestKernel2(const KernelList& all_impls, const Params& params) const;
    KernelsData GetNaiveBestKernel2(const Params& params, KernelType kType) const;
    KernelList GetAllImplementations2(const Params& params, KernelType kType) const;
    
    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector
