// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base.h"
#include "kernel_selector_common.h"
#include "kernel_selector.h"
#include "kernel_selector_params.h"
#include <type_traits>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <iostream>

// #define ENABLE_ENV
// #define ENABLE_ENV_PRINT

#ifdef ENABLE_ENV_PRINT
#define ENV_PRINTF(...) printf(__VA_ARGS__)
#else
#define ENV_PRINTF(...)
#endif  // ENABLE_ENV_PRINT

#define ENABLE_OFFLINE_TUNING_CACHE 1

namespace kernel_selector {

AutoTuner kernel_selector_base::autoTuner;

#ifdef ENABLE_ENV
std::string strip(const std::string str) {
    size_t start = str.find_first_not_of(' ');
    size_t end = str.find_last_not_of(' ');
    if (start == std::string::npos || end == std::string::npos) {
        return "";
    }

    return str.substr(start, end - start + 1);
}

static void AddToForceMap(ForceList& force_list, bool force_or_deny, const char* env_str) {
    std::stringstream ss;
    ss.str(GetStringEnv(env_str));

    ENV_PRINTF("ENV: %s = %s\n", env_str, ss.str().c_str());

    std::string val;
    while (std::getline(ss, val, ',')) {
        std::string kernel_name = strip(val);
        if (!kernel_name.empty()) {
            force_list[kernel_name] = force_or_deny;
        }
    }
}
#endif

kernel_selector_base::kernel_selector_base() {
#ifdef ENABLE_ENV
    AddToForceMap(forceKernels, true, "CL_DNN_FORCE_KERNELS");
    AddToForceMap(forceKernels, false, "CL_DNN_DENY_KERNELS");
#endif
}

KernelsData kernel_selector_base::GetNaiveBestKernel(const Params& params,
                                                     const optional_params& options,
                                                     KernelType kType) const {
    KernelsData kernelsData;
    std::string kernelName;
    const std::string specified_name = "convolution:0-convolutional";

    auto allImplementations = GetAllImplementations(params, options, kType);
    if (params.layerID == specified_name) {
        std::cout << "[GetNaiveBestKernel][GOLD] is called for " << params.layerID << std::endl;
    }

    for (const auto& implementation : allImplementations) {
        // TODO: Unify this check with the Validate virtual method. Make
        // sure that the method is called here only, not in all the
        // GetKernelsData implementations.
        try {
            KernelsData kds = implementation->GetKernelsData(params, options);

            if (kds.size() && kds[0].kernels.size()) {
#ifdef ENABLE_ENV
                const auto& it = forceKernels.find(implementation->GetName());
                if (it != forceKernels.end()) {
                    if (it->second == true) {
                        if (params.layerID == specified_name) {
                            std::cout << "[GetNaiveBestKernel][GOLD] " <<  kernelName << " is forced selection from " << params.layerID << std::endl;
                        }
                        ENV_PRINTF("Force: %s\n", it->first.c_str());
                        return kds;
                    } else {
                        if (params.layerID == specified_name) {
                            std::cout << "[GetNaiveBestKernel][GOLD] " <<  kernelName << " is denied from " << params.layerID << std::endl;
                        }
                        ENV_PRINTF("Deny: %s\n", it->first.c_str());
                    }
                } else {
#endif
                    kernelsData = kds;
                    kernelName = implementation->GetName();
                    if (params.layerID == specified_name) {
                        std::cout << "[GetNaiveBestKernel][GOLD] " <<  kernelName << " is selected from " << params.layerID << std::endl;
                    }
                    break;
#ifdef ENABLE_ENV
                }
#endif
            } else {
                if (params.layerID == specified_name) {
                    kernelName = implementation->GetName();
                    std::cout << "[GetNaiveBestKernel][GOLD][NOT] " << kernelName << " is not selected from " << params.layerID << std::endl;
                }
            }
        } catch (std::runtime_error& ex) {
            // we have to handle it in order to avoid exception in KernelSelector as much we can
            if (params.layerID == specified_name) {
                kernelName = (implementation != nullptr)? implementation->GetName() : "[impl is null]";
                std::cout << "[GOLD][Exception] layerID: " << params.layerID << " kenrel: "
                    << kernelName << " - " << ex.what() << std::endl;
            }
        }
    }

    // TODO: find a better place to located this assignment
    if (kernelsData.size()) {
        kernelsData[0].kernelName = kernelName;
        kernelsData[0].kernels[0].layerID = params.layerID;
    }

    return kernelsData;
}

KernelsData kernel_selector_base::GetAutoTuneBestKernel(const Params& params,
                                                        const optional_params& options,
                                                        KernelType kType) const {
    KernelsData kernelsData;
    std::string kernelName;
    const std::string specified_name = "convolution:0-convolutional";
    auto allImplementations = GetAllImplementations(params, options, kType);
    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] is called for " << params.layerID << std::endl;
    }
    auto kernel_params = static_cast<const base_params&>(params);
    bool int8_kernel = kernel_params.inputs[0].GetDType() == Datatype::INT8 || kernel_params.inputs[0].GetDType() == Datatype::UINT8;
    std::tuple<std::string, int> cachedKernelConfig;
    if (options.tuningParams.mode == TuningMode::TUNING_DISABLED && !int8_kernel) {  // Try to load kernel/config from offline cache
#if ENABLE_OFFLINE_TUNING_CACHE
        cachedKernelConfig = autoTuner.LoadKernelOffline(params.engineInfo.deviceCache, params);
        if (params.layerID == specified_name) {
            std::cout << "[GetAutoTuneBestKernel][GOLD] Call LoadKernelOffline " << params.layerID << std::endl;
            std::cout << "- std::get<0>(cachedKernelConfig): " << std::get<0>(cachedKernelConfig) << std::endl;
            std::cout << "- Params" << params.to_cache_string_v2() << std::endl;
        }
#else

        if (params.layerID == specified_name) {
            std::cout << "[GetAutoTuneBestKernel][GOLD] is calling GetNaiveBestKernel for " << params.layerID << std::endl;
        }
        return GetNaiveBestKernel(params, options, kType);
#endif
    } else if (UseCached(options.tuningParams.mode)) {  // Try to load kernel/config from on-line cache
        cachedKernelConfig = autoTuner.LoadKernelOnline(options.tuningParams.mode,
                                                        options.tuningParams.cacheFilePath,
                                                        params);
    }
    bool hashFoundInCache = !std::get<0>(cachedKernelConfig).empty();
    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Checking " << params.layerID << std::endl;
        switch (options.tuningParams.mode) {
        case TuningMode::TUNING_DISABLED:
            std::cout << "- Tuning mode: TUNING_DISABLED" << std::endl;
            break;
        case TuningMode::TUNING_USE_CACHE:
            std::cout << "- Tuning mode: TUNING_USE_CACHE" << std::endl;
            break;
        case TuningMode::TUNING_TUNE_AND_CACHE:
            std::cout << "- Tuning mode: TUNING_TUNE_AND_CACHE" << std::endl;
            break;
        case TuningMode::TUNING_USE_AND_UPDATE:
            std::cout << "- Tuning mode: TUNING_USE_AND_UPDATE" << std::endl;
            break;
        case TuningMode::TUNING_RETUNE_AND_CACHE:
            std::cout << "- Tuning mode: TUNING_RETUNE_AND_CACHE" << std::endl;
            break;
        default:
            break;
        }
        std::cout << "- hasFoundInCache: " << (hashFoundInCache? "True" : "False") << std::endl;
    }
    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Step 02 " << params.layerID << std::endl;
    }
    if (hashFoundInCache) {
        std::string cachedkernelName = std::get<0>(cachedKernelConfig);
        int autoTuneIndex = std::get<1>(cachedKernelConfig);

        for (const auto& implementation : allImplementations) {
            // TODO: make sure kernel names are unique.
            if (implementation->GetName().compare(cachedkernelName) == 0) {
                KernelsData kds = implementation->GetTunedKernelsDataByIndex(params, options, autoTuneIndex);
                if (kds.size() && kds[0].kernels.size()) {
                    kernelsData = kds;
                    kernelsData[0].kernelName = cachedkernelName;
                    kernelsData[0].kernels[0].layerID = params.layerID;
                }
                break;
            }
        }

        if (!kernelsData.empty()) {
            if (params.layerID == specified_name) {
                std::cout << "[GetAutoTuneBestKernel][GOLD] Step 02 " << cachedkernelName;
                std::cout << " - cached kernel is selected for " << params.layerID << std::endl;
            }
            return kernelsData;
        }
    }

    // Cache is not valid, remove it if performing update tasks.
    if (hashFoundInCache && PerformUpdates(options.tuningParams.mode)) {
        autoTuner.RemoveKernel(options.tuningParams.cacheFilePath, params);
    }

    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Step 03 " << params.layerID << std::endl;
    }

    if (hashFoundInCache ||  // Cache is not valid - hash exists in cache but kernelsData was empty or kernel
                             // doesn't support the required key.
        !PerformTuning(options.tuningParams.mode) ||  // On-line tuning is not allowed.
        !options.tuningParams.runner) {  // Runner is invalid - can't run on-line tuning
        // Fall back to the default path.
        return GetNaiveBestKernel(params, options, kType);
    }

    // Start on-line tuning
    assert(options.tuningParams.runner);

    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Step 05 " << params.layerID << std::endl;
    }
    for (const auto& implementation : allImplementations) {
        const ParamsKey implKey = implementation->GetSupportedKey();
        if (implKey.TuningSupport()) {
            try {
                KernelsData kds = implementation->GetKernelsDataForAutoTune(params, options);
                auto runTimes = options.tuningParams.runner->run_kernels(kds);

                for (size_t i = 0; i < kds.size(); i++) {
                    kds[i].runTime = runTimes[i].count();
                    if (kernelsData.size() == 0 || kds[i].runTime < kernelsData[0].runTime) {
                        kernelsData = {kds[i]};
                        kernelName = implementation->GetName();
                        if (params.layerID == specified_name) {
                            std::cout << "[GetAutoTuneBestKernel-01][GOLD] " <<  kernelName << " is selected from " << params.layerID << std::endl;
                        }
                    }
                }
            } catch (std::runtime_error& ex) {
                // we have to handle it in order to avoid exception in KernelSelector as much we can
                if (params.layerID == specified_name) {
                    kernelName = (implementation != nullptr)? implementation->GetName() : "[impl is null]";
                    std::cout << "[GOLD][Exception] layerID: " << params.layerID << " kenrel: "
                        << kernelName << " - " << ex.what() << std::endl;
                }
            }
        }
    }
    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Step 06 " << params.layerID << std::endl;
    }
    // try to fallback to reference kernels if no optimized were found during tuning
    if (!kernelsData.size()) {
        for (const auto& implementation : allImplementations) {
            const ParamsKey implKey = implementation->GetSupportedKey();
            // this time, check only implementations that have disabled tuning
            if (!implKey.TuningSupport()) {
                try {
                    KernelsData kds = implementation->GetKernelsDataForAutoTune(params, options);
                    auto runTimes = options.tuningParams.runner->run_kernels(kds);

                    for (size_t i = 0; i < kds.size(); i++) {
                        kds[i].runTime = runTimes[i].count();
                        if (kernelsData.size() == 0 || kds[i].runTime < kernelsData[0].runTime) {
                            kernelsData = {kds[i]};
                            kernelName = implementation->GetName();
                            if (params.layerID == specified_name) {
                                std::cout << "[GetAutoTuneBestKernel-01][GOLD] " <<  kernelName << " is selected from " << params.layerID << std::endl;
                            }
                        }
                    }
                } catch (std::runtime_error& ex) {
                    // we have to handle it in order to avoid exception in KernelSelector as much we can
                    if (params.layerID == specified_name) {
                        kernelName = (implementation != nullptr)? implementation->GetName() : "[impl is null]";
                        std::cout << "[GOLD][Exception] layerID: " << params.layerID << " kenrel: "
                            << kernelName << " - " << ex.what() << std::endl;
                    }
                }
            }
        }
    }
    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Step 07 " << params.layerID << std::endl;
    }
    if (kernelsData.size()) {
        if (params.layerID == specified_name) {
            std::cout << "[GetAutoTuneBestKernel-01][GOLD] " <<  kernelName << " is stored.... from " << params.layerID << std::endl;
        }
        kernelsData[0].kernelName = kernelName;
        kernelsData[0].kernels[0].layerID = params.layerID;
        autoTuner.StoreKernel(options.tuningParams.cacheFilePath,
                                params,
                                kernelName,
                                kernelsData[0].autoTuneIndex);
    } else {
        // Tuning failed, fall back to naive path
        return GetNaiveBestKernel(params, options, kType);
    }

    if (params.layerID == specified_name) {
        std::cout << "[GetAutoTuneBestKernel][GOLD] Step 08 " << params.layerID << std::endl;
    }
    return kernelsData;
}

KernelList kernel_selector_base::GetAllImplementations(const Params& params, const optional_params& options, KernelType kType) const {
    using PriorityPair = std::pair<KernelsPriority, std::shared_ptr<KernelBase>>;
    auto comparePriority = [](const PriorityPair& firstImpl, const PriorityPair& secondImpl) {
        return firstImpl.first < secondImpl.first;
    };

    std::multiset<PriorityPair, decltype(comparePriority)> sortedImpls(comparePriority);
    KernelList result;
    // const std::string specified_name = "convolution:Conv_8/WithoutBiases";
    const std::string specified_name = "convolution:0-convolutional";

    if (params.GetType() == kType && options.GetType() == kType) {
        // if (params.layerID == specified_name) {
        // // if (params.layerID.find(specified_name) != std::string::npos) {
        //     std::cout << "**************************************************" << std::endl;
        //     std::cout << "Before params kernel params : " << params.layerID << " - " << typeid(params).name() << std::endl;
        //     {
        //         const kernel_selector::base_params& bp = static_cast<const kernel_selector::base_params&>(params);
        //         std::cout << bp.GetParamsKey().to_string() << std::endl;
        //     }
        //     // std::cout << params.GetParamsKey().to_string() << std::endl;
        //     std::cout << "**************************************************" << std::endl;

        //     std::cout << "**************************************************" << std::endl;
        //     std::cout << "Before options.GetSupportedKey() kernel params : " << params.layerID << std::endl;
        //     std::cout << options.GetSupportedKey().to_string() << std::endl;
        //     std::cout << "**************************************************" << std::endl;
        // }
        ParamsKey requireKey = params.GetParamsKey().Merge(options.GetSupportedKey());
        // if (params.layerID == specified_name) {
        // // if (params.layerID.find(specified_name) != std::string::npos) {
        //     std::cout << "**************************************************" << std::endl;
        //     std::cout << "After requireKey kernel params : " << params.layerID << std::endl;
        //     std::cout << requireKey.to_string() << std::endl;
        //     std::cout << "**************************************************" << std::endl;
        // }

        bool forceImplementation = !params.forceImplementation.empty();
        for (auto& impl : implementations) {
            const ParamsKey implKey = impl->GetSupportedKey();
            // if (params.layerID == specified_name) {
            // // if (params.layerID.find(specified_name) != std::string::npos) {
            //     std::cout << "**************************************************" << std::endl;
            //     std::cout << " Checking Key kernel params : " << impl->GetName() << std::endl;
            //     std::cout << implKey.to_string() << std::endl;
            //     std::cout << "**************************************************" << std::endl;
            //     if (!implKey.Support_v2(requireKey)) {
            //         std::cout << "Fail to check !!" << std::endl;
            //         continue;
            //     }
            // } else {
                if (!implKey.Support(requireKey))
                    continue;
            // }
            if (forceImplementation && params.forceImplementation != impl->GetName())
                continue;
            sortedImpls.emplace(impl->GetKernelsPriority(params, options), impl);
        }

        std::transform(
            sortedImpls.begin(),
            sortedImpls.end(),
            std::back_inserter(result),
            [](const PriorityPair& impl) {
                return std::move(impl.second);
            });
    }

    if (params.layerID == specified_name) {
    // if (params.layerID.find(specified_name) != std::string::npos) {
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        std::cout << "[GOLD] kernel candidates for " << params.layerID << " : " << std::endl;
        for (auto& r : result) {
            std::cout << " * " << r->GetName() << std::endl;
        }
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    }

    return result;
}

}  // namespace kernel_selector
