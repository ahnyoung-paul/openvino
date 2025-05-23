﻿// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_ONEDNN_FOR_GPU

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"

#ifdef UNUSED
#   undef UNUSED
#endif

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include "gpu/intel/microkernels/package.hpp"
#include "gpu/intel/jit/gemm/include/gemmstone/microkernel_provider.hpp"
#include "gpu/intel/microkernels/shim.hpp"
#include "common/utils.hpp"

namespace micro {

using Package = dnnl::impl::gpu::intel::micro::Package;
using HWInformation = gemmstone::HWInformation;
using GEMMProblem = gemmstone::GEMMProblem;
using ABOffset = gemmstone::ABOffset;
using GEMMStrategy = gemmstone::GEMMStrategy;
using GEMMProtocol = dnnl::impl::gpu::intel::micro::GEMMProtocol;
using MatrixLayout = gemmstone::MatrixLayout;
using Type = gemmstone::Type;
using SizeParams = gemmstone::SizeParams;
using StrategyRequirement = gemmstone::StrategyRequirement;
using ShimOptions = dnnl::impl::gpu::intel::micro::ShimOptions;
using HostLanguage = dnnl::impl::gpu::intel::micro::HostLanguage;
using Setting = dnnl::impl::gpu::intel::micro::Setting;

using dnnl::impl::utils::rnd_up_pow2;

// Wrapper for Package which is used in clKernelData with forward declaration
// to avoid including this header in many places in plugin
// which may cause symbols conflicts with oneDNN
struct MicroKernelPackage {
    MicroKernelPackage() = default;
    explicit MicroKernelPackage(Package _p) : p(_p) {}
    Package p;

    // WARNING: We serialize only microkernels settings, so after deserialization
    // other struct fields are not initializer properly and can't be used
    void save(cldnn::BinaryOutputBuffer& ob) const {
        ob << p.settings.size();
        for (auto& s : p.settings) {
            ob << s.name;
            ob << s.value;
        }
    }

    void load(cldnn::BinaryInputBuffer& ib) {
        size_t n_settings;
        ib >> n_settings;
        p.settings.clear();
        for (size_t i = 0; i < n_settings; i++) {
            Setting s;
            ib >> s.name;
            ib >> s.value;
            p.settings.push_back(s);
        }
    }
};

inline Package select_gemm_microkernel(GEMMProtocol protocol, HWInformation hw_info, SizeParams sizes, const GEMMProblem &problem,
                                        const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                        void (*strategyAdjuster)(GEMMStrategy &strategy) = nullptr) {
    return gemmstone::selectGEMMMicrokernel(protocol, hw_info, sizes, problem, reqs, strategyAdjuster);
}

static inline int alignment_for_ld(int ld) {
    return  gemmstone::alignmentForLD(ld);
}

}  // namespace micro

#undef UNUSED

#endif  // ENABLE_ONEDNN_FOR_GPU
