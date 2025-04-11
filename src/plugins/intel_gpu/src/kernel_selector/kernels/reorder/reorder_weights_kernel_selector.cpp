// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_kernel_selector.h"
#include "reorder_weights_kernel.h"
#include "reorder_weights_winograd_2x3_kernel.h"
#include "reorder_weights_winograd_6x3_kernel.h"
#include "reorder_weights_image_fyx_b_kernel.h"
#include "reorder_weights_image_winograd_6x3_kernel.h"
#include "reorder_weights_opt.h"
#include "reorder_weights_int4.h"

namespace kernel_selector {

ReorderWeightsKernelSelector::ReorderWeightsKernelSelector() {
    Attach<ReorderWeightsKernel>();
    Attach<ReorderWeightsWinograd2x3Kernel>();
    Attach<ReorderWeightsWinograd6x3Kernel>();
    Attach<ReorderWeightsImage_fyx_b_Kernel>();
    Attach<ReorderWeightsImageWinograd6x3Kernel>();
    Attach<ReorderWeightsOpt>();
    Attach<ReorderWeightsKernelInt4>();
}

KernelsData ReorderWeightsKernelSelector::GetBestKernels(const Params& params) const {
    GPU_DEBUG_COUT << "ReorderWeightsKernelSelector::GetBestKernels is called ...." << std::endl;
    return GetNaiveBestKernel2(params, KernelType::REORDER);
}

KernelsData ReorderWeightsKernelSelector::GetNaiveBestKernel2(const KernelList& all_impls, const Params& params) const {
    KernelsData kernelsData;
    std::string kernelName;

    GPU_DEBUG_COUT << "Number of all_impls : " << all_impls.size() << std::endl;
    for (const auto& implementation : all_impls) {
        // TODO: Unify this check with the Validate virtual method. Make
        // sure that the method is called here only, not in all the
        // GetKernelsData implementations.
        try {
            GPU_DEBUG_COUT << "Try to get kernel data for " << implementation->GetName() << std::endl;
            KernelsData kds = implementation->GetKernelsData(params);

            if (kds.size() && kds[0].kernels.size()) {
                kernelsData = kds;
                kernelName = implementation->GetName();
                break;
            }
        } catch (std::runtime_error& ex) {
            // we have to handle it in order to avoid exception in KernelSelector as much we can
            kernelName = (implementation != nullptr)? implementation->GetName() : "[impl is null]";
            GPU_DEBUG_COUT << "layerID: " << params.layerID << " kernel: " << kernelName << " - " << ex.what() << std::endl;
        }
    }

    // TODO: find a better place to located this assignment
    if (kernelsData.size()) {
        kernelsData[0].kernelName = kernelName;
        kernelsData[0].kernels[0].params.layerID = params.layerID;
    }

    return kernelsData;
}
KernelsData ReorderWeightsKernelSelector::GetNaiveBestKernel2(const Params& params, KernelType kType) const {
    return GetNaiveBestKernel2(GetAllImplementations2(params, kType), params);
}

KernelList ReorderWeightsKernelSelector::GetAllImplementations2(const Params& params, KernelType kType) const {
    using PriorityPair = std::pair<KernelsPriority, std::shared_ptr<KernelBase>>;
    auto comparePriority = [](const PriorityPair& firstImpl, const PriorityPair& secondImpl) {
        return firstImpl.first < secondImpl.first;
    };

    std::multiset<PriorityPair, decltype(comparePriority)> sortedImpls(comparePriority);
    KernelList result;

    auto device_features_key = params.engineInfo.get_supported_device_features_key();

    if (params.GetType() == kType) {
        GPU_DEBUG_COUT << "Found type " << ", num of kernel list " << implementations.size() << std::endl;
        ParamsKey requireKey = params.GetParamsKey();
        bool forceImplementation = !params.forceImplementation.empty();
        for (auto& impl : implementations) {
            GPU_DEBUG_COUT << "Try to match " << impl->GetName() << std::endl;
            const ParamsKey implKey = impl->GetSupportedKey();
            if (!implKey.Support2(requireKey)) {
                continue;
            }
            GPU_DEBUG_COUT << "Pass 1 step ..." << std::endl;
            auto required_device_features_key = impl->get_required_device_features_key(params);
            if (!device_features_key.supports(required_device_features_key))
                continue;
            GPU_DEBUG_COUT << "Pass 2 step ..." << std::endl;
            if (forceImplementation && params.forceImplementation != impl->GetName())
                continue;
            GPU_DEBUG_COUT << "Pass 3 step ..." << std::endl;
            sortedImpls.emplace(impl->GetKernelsPriority(params), impl);
        }

        std::transform(
            sortedImpls.begin(),
            sortedImpls.end(),
            std::back_inserter(result),
            [](const PriorityPair& impl) {
                return std::move(impl.second);
            });
    } else {
        GPU_DEBUG_COUT << "No implementation for " << params.layerID << " because of kernel type mismatch" << std::endl;
    }

    return result;
}

}  // namespace kernel_selector
