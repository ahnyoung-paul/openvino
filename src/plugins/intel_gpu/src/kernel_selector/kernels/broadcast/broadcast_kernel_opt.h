// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "broadcast_kernel_base.h"

namespace kernel_selector {
class BroadcastKernelOpt : public BroadcastKernelBase {
public:
    BroadcastKernelOpt() : BroadcastKernelBase("broadcast_gpu_opt") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

    DispatchData SetDefault(const broadcast_params& arg) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const broadcast_params& params) const override;
};
}  // namespace kernel_selector
