// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "count_nonzero_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey CountNonzeroKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}
#define USE_ORIGINAL
CommonDispatchData CountNonzeroKernelRef::SetDefault(const count_nonzero_params& params) const {
    CommonDispatchData dispatchData;
    const auto& input = params.inputs[0];
#ifdef USE_ORIGINAL
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
#endif
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    int rank = input.Dimentions();
    if (rank == 4) {
        dispatchData.gws = {input.X().v, input.Y().v, input.Feature().v * input.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else if (rank == 5) {
        dispatchData.gws = {input.X().v, input.Y().v * input.Z().v, input.Feature().v * input.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    } else {
        dispatchData.gws = {input.X().v * input.Y().v, input.Z().v * input.W().v, input.Feature().v * input.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
    }

#ifdef USE_ORIGINAL
    dispatchData.lws =
        GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
#else
    dispatchData.lws = dispatchData.gws;
#endif

    // size_t sum = 1;
    // for (auto& v : dispatchData.gws) {
    //     sum = sum * v;
    // }

    // if (sum == 56) {
    //     std::cout << "dispatchData.gws={";
    //     for (auto& v : dispatchData.gws) {
    //         std::cout << v << ",";
    //     }
    //     std::cout << "}" << std::endl;
    //     std::cout << "dispatchData.lws={";
    //     for (auto& v : dispatchData.lws) {
    //         std::cout << v << ",";
    //     }
    //     std::cout << "}" << std::endl;
    // }

    return dispatchData;
}

DeviceFeaturesKey CountNonzeroKernelRef::get_required_device_features_key(const Params& params, const optional_params& options) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();

    return k;
}

KernelsData CountNonzeroKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::COUNT_NONZERO);

    KernelData kd = KernelData::Default<count_nonzero_params>(params);
    count_nonzero_params& newParams = *static_cast<count_nonzero_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = MakeBaseParamsJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    bool is_dynamic = newParams.inputs[0].is_dynamic();

    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const count_nonzero_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        auto& updated_kernel = kd.kernels[0];
        updated_kernel.params.workGroups.global = dispatchData.gws;
        updated_kernel.params.workGroups.local = dispatchData.lws;

        auto& gws = updated_kernel.params.workGroups.global;
        auto& lws = updated_kernel.params.workGroups.local;
        size_t gws_mul = std::accumulate(gws.begin(), gws.end(), 1, std::multiplies<size_t>());
        size_t lws_mul = std::accumulate(lws.begin(), lws.end(), 1, std::multiplies<size_t>());

        size_t buffer_size = static_cast<size_t>(std::ceil(static_cast<double>(gws_mul) / lws_mul)) * sizeof(size_t) * 2;
        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(buffer_size);
        kd.internalBufferDataType = kernel_selector::Datatype::UINT32;
    };

    // In case of count-nonzero, the output shape is static unconditionally,
    // so it should be checked as dynamic of the input shape
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     is_dynamic);
    {
        auto& args = kernel.params.arguments;
        args.clear();
        if (is_dynamic) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        auto& gws = kernel.params.workGroups.global;
        auto& lws = kernel.params.workGroups.local;
        size_t gws_mul = std::accumulate(gws.begin(), gws.end(), 1, std::multiplies<size_t>());
        size_t lws_mul = std::accumulate(lws.begin(), lws.end(), 1, std::multiplies<size_t>());

        size_t buffer_size = static_cast<size_t>(std::ceil(static_cast<double>(gws_mul) / lws_mul)) * sizeof(size_t) * 2;
        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(buffer_size);
        kd.internalBufferDataType = kernel_selector::Datatype::UINT32;
    }
    return {kd};
}

KernelsPriority CountNonzeroKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool CountNonzeroKernelRef::Validate(const Params& p, const optional_params& op) const {
    if (!KernelBaseOpenCL::Validate(p, op))
        return false;

    const auto& rp = static_cast<const count_nonzero_params&>(p);

    return Tensor::SimpleLayout(rp.inputs[0].GetLayout());
}
}  // namespace kernel_selector
