// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include <sstream>
#include <string>
#include <iostream>

#include <activation/activation_kernel_base.h>
#include "jitter.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ParamsKey
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ParamsKey::EnableInputDataType(Datatype dt) {
    switch (dt) {
        case Datatype::INT8:
            key.inputType.val.int8 = 1;
            break;
        case Datatype::UINT8:
            key.inputType.val.uint8 = 1;
            break;
        case Datatype::INT16:
            key.inputType.val.int16 = 1;
            break;
        case Datatype::UINT16:
            key.inputType.val.uint16 = 1;
            break;
        case Datatype::INT32:
            key.inputType.val.int32 = 1;
            break;
        case Datatype::UINT32:
            key.inputType.val.uint32 = 1;
            break;
        case Datatype::INT64:
            key.inputType.val.int64 = 1;
            break;
        case Datatype::F16:
            key.inputType.val.F16 = 1;
            break;
        case Datatype::F32:
            key.inputType.val.F32 = 1;
            break;
        case Datatype::BINARY:
            key.inputType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllInputDataType() { key.inputType.raw = 0xffffffff; }

void ParamsKey::EnableOutputDataType(Datatype dt) {
    switch (dt) {
        case Datatype::INT8:
            key.outputType.val.int8 = 1;
            break;
        case Datatype::UINT8:
            key.outputType.val.uint8 = 1;
            break;
        case Datatype::INT16:
            key.outputType.val.int16 = 1;
            break;
        case Datatype::UINT16:
            key.outputType.val.uint16 = 1;
            break;
        case Datatype::INT32:
            key.outputType.val.int32 = 1;
            break;
        case Datatype::UINT32:
            key.outputType.val.uint32 = 1;
            break;
        case Datatype::INT64:
            key.outputType.val.int64 = 1;
            break;
        case Datatype::F16:
            key.outputType.val.F16 = 1;
            break;
        case Datatype::F32:
            key.outputType.val.F32 = 1;
            break;
        case Datatype::BINARY:
            key.outputType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllOutputDataType() { key.outputType.raw = 0xffffffff; }

void ParamsKey::EnableInputWeightsType(WeightsType wt) {
    switch (wt) {
        case WeightsType::F16:
            key.inputWeightsType.val.F16 = 1;
            break;
        case WeightsType::F32:
            key.inputWeightsType.val.F32 = 1;
            break;
        case WeightsType::INT8:
            key.inputWeightsType.val.int8 = 1;
            break;
        case WeightsType::BINARY:
            key.inputWeightsType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllInputWeightsType() { key.inputWeightsType.raw = 0xffffffff; }

void ParamsKey::EnableOutputWeightsType(WeightsType wt) {
    switch (wt) {
        case WeightsType::F16:
            key.outputWeightsType.val.F16 = 1;
            break;
        case WeightsType::F32:
            key.outputWeightsType.val.F32 = 1;
            break;
        case WeightsType::INT8:
            key.outputWeightsType.val.int8 = 1;
            break;
        case WeightsType::BINARY:
            key.outputWeightsType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllOutputWeightsType() { key.outputWeightsType.raw = 0xffffffff; }

void ParamsKey::EnableLRNMode(LRNMode m) {
    switch (m) {
        case LRNMode::ACROSS_CHANNEL:
            key.restrict.val.dedicated.norm.across = 1;
            break;
        case LRNMode::WITHIN_CHANNEL:
            key.restrict.val.dedicated.norm.within = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableLookUpTableAxis(LookUpTableAxis m) {
    switch (m) {
        case kernel_selector::LookUpTableAxis::BATCH:
            key.restrict.val.dedicated.lookt.axisBatch = 1;
            break;
        case kernel_selector::LookUpTableAxis::FEATURE:
            key.restrict.val.dedicated.lookt.axisFeature = 1;
            break;
        case kernel_selector::LookUpTableAxis::X:
            key.restrict.val.dedicated.lookt.axisX = 1;
            break;
        case kernel_selector::LookUpTableAxis::Y:
            key.restrict.val.dedicated.lookt.axisY = 1;
            break;
        case kernel_selector::LookUpTableAxis::XYF:
            key.restrict.val.dedicated.lookt.axisXYF = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableNormalizeMode(NormalizeMode m) {
    switch (m) {
        case NormalizeMode::ACROSS_SPATIAL:
            key.restrict.val.dedicated.norm.across = 1;
            break;
        case NormalizeMode::WITHIN_SPATIAL:
            key.restrict.val.dedicated.norm.within = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableMVNMode(MVNMode m) {
    switch (m) {
        case MVNMode::ACROSS_CHANNELS:
            key.restrict.val.dedicated.mvn.across = 1;
            break;
        case MVNMode::WITHIN_CHANNELS:
            key.restrict.val.dedicated.mvn.within = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableMVNNormalizeVariance() { key.restrict.val.dedicated.mvn.normalize_variance = 1; }

void ParamsKey::EnableLRNKernelDividerMode(KernelDividerMode m) {
    switch (m) {
        case KernelDividerMode::FIXED:
            key.restrict.val.dedicated.norm.fixedKenrelDivider = 1;
            break;
        case KernelDividerMode::DYNAMIC:
            key.restrict.val.dedicated.norm.dynamicKenrelDivider = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnablePoolKernelDividerMode(KernelDividerMode m) {
    switch (m) {
        case KernelDividerMode::FIXED:
            key.restrict.val.dedicated.pooling.fixedKenrelDivider = 1;
            break;
        case KernelDividerMode::DYNAMIC:
            key.restrict.val.dedicated.pooling.dynamicKenrelDivider = 1;
            break;
        case KernelDividerMode::DYNAMIC_WITH_PADDING:
            key.restrict.val.dedicated.pooling.dynamicKenrelDividerWithPadding = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnablePoolType(PoolType t) {
    switch (t) {
        case PoolType::MAX:
            key.restrict.val.dedicated.pooling.max = 1;
            break;
        case PoolType::AVG:
            key.restrict.val.dedicated.pooling.avg = 1;
            break;
        case PoolType::MAX_WITH_ARGMAX:
            key.restrict.val.dedicated.pooling.max_with_argmax = 1;
            break;
        case PoolType::BILINEAR:
            key.restrict.val.dedicated.pooling.bilinear = 1;
            break;
        case PoolType::DEFORMABLE_BILINEAR:
            key.restrict.val.dedicated.pooling.deformable_bilinear = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnablePoolRemainder(PoolRemainder r) {
    switch (r) {
        case PoolRemainder::FLOOR:
            key.restrict.val.dedicated.pooling.floor = 1;
            break;
        case PoolRemainder::CEIL:
            key.restrict.val.dedicated.pooling.ceil = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableSoftmaxDim(SoftmaxDim d) {
    switch (d) {
        case SoftmaxDim::X:
            key.restrict.val.dedicated.softmax.dimX = 1;
            break;
        case SoftmaxDim::Y:
            key.restrict.val.dedicated.softmax.dimY = 1;
            break;
        case SoftmaxDim::FEATURE:
            key.restrict.val.dedicated.softmax.dimFeature = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableConcatAxis(ConcatAxis a) {
    switch (a) {
        case ConcatAxis::X:
            key.restrict.val.dedicated.concat.axisX = 1;
            break;
        case ConcatAxis::Y:
            key.restrict.val.dedicated.concat.axisY = 1;
            break;
        case ConcatAxis::Z:
            key.restrict.val.dedicated.concat.axisZ = 1;
            break;
        case ConcatAxis::W:
            key.restrict.val.dedicated.concat.axisW = 1;
            break;
        case ConcatAxis::FEATURE:
            key.restrict.val.dedicated.concat.axisFeature = 1;
            break;
        case ConcatAxis::BATCH:
            key.restrict.val.dedicated.concat.axisBatch = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableReampleType(ResampleType a) {
    switch (a) {
        case ResampleType::NEAREST_NEIGHBOR:
            key.restrict.val.dedicated.resample.nearest_neighbor = 1;
            break;
        case ResampleType::CAFFE_BILINEAR_INTERP:
            key.restrict.val.dedicated.resample.caffe_bilinear_interp = 1;
            break;
        case ResampleType::BILINEAR_INTERP:
            key.restrict.val.dedicated.resample.bilinear_interp = 1;
            break;
        case ResampleType::CUBIC:
            key.restrict.val.dedicated.resample.cubic = 1;
            break;
        case ResampleType::LINEAR_ONNX:
            key.restrict.val.dedicated.resample.linear_onnx = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableFusedConvEltwEltwiseStride() { key.restrict.val.dedicated.fused_conv_eltw.stride = 1; }

void ParamsKey::EnableEltwiseStride() { key.restrict.val.dedicated.eltwise.stride = 1; }

void ParamsKey::EnableArgMaxMinAxis(ArgMaxMinAxis a) {
    switch (a) {
        case ArgMaxMinAxis::X:
            key.restrict.val.dedicated.argm.axisX = 1;
            break;
        case ArgMaxMinAxis::Y:
            key.restrict.val.dedicated.argm.axisY = 1;
            break;
        case ArgMaxMinAxis::Z:
            key.restrict.val.dedicated.argm.axisZ = 1;
            break;
        case ArgMaxMinAxis::FEATURE:
            key.restrict.val.dedicated.argm.axisFeature = 1;
            break;
        case ArgMaxMinAxis::BATCH:
            key.restrict.val.dedicated.argm.axisBatch = 1;
            break;
        case ArgMaxMinAxis::XYF:
            key.restrict.val.dedicated.argm.axisXYF = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableIndexSelectAxis(IndexSelectAxis a) {
    switch (a) {
        case IndexSelectAxis::X:
            key.restrict.val.dedicated.idxsel.axisX = 1;
            break;
        case IndexSelectAxis::Y:
            key.restrict.val.dedicated.idxsel.axisY = 1;
            break;
        case IndexSelectAxis::FEATURE:
            key.restrict.val.dedicated.idxsel.axisFeature = 1;
            break;
        case IndexSelectAxis::BATCH:
            key.restrict.val.dedicated.idxsel.axisBatch = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableLookUpTableIndicesFormat(Datatype a) {
    if (a == Datatype::F32)
        key.restrict.val.dedicated.lookt.indicesF32 = 1;
    else
        key.restrict.val.dedicated.lookt.indicesOther = 1;
}

void ParamsKey::EnableFusedConvEltwiseRWOutOpt() { key.restrict.val.dedicated.fused_conv_eltw.rw_out_opt = 1; }
void ParamsKey::EnableFusedConvEltwDepthToSpaceFusing() { key.restrict.val.dedicated.fused_conv_eltw.depth_to_space_fused = 1; }


void ParamsKey::EnableQuantization(QuantizationType q) {
    switch (q) {
        case QuantizationType::NONE:
            break;
        case QuantizationType::SYMMETRIC:
            key.restrict.val.sym_quantization = 1;
            break;
        case QuantizationType::ASYMMETRIC_DATA:
            key.restrict.val.asym_d_quantization = 1;
            break;
        case QuantizationType::ASYMMETRIC_WEIGHTS:
            key.restrict.val.asym_w_quantization = 1;
            break;
        case QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS:
            key.restrict.val.asym_d_quantization = 1;
            key.restrict.val.asym_w_quantization = 1;
            break;
        default:
            break;
    }
}

bool ParamsKey::Support(const ParamsKey& k) const {
    if (!((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw))  // check if this kernel supports this params
        return false;
    if (!((key.machineInfo.raw & k.key.machineInfo.raw) ==
          key.machineInfo.raw))  // check if machine supports this kernel
        return false;
    if (!((key.inputType.raw & k.key.inputType.raw) == k.key.inputType.raw))
        return false;
    if (!((key.outputType.raw & k.key.outputType.raw) == k.key.outputType.raw))
        return false;
    if (!((key.inputWeightsType.raw & k.key.inputWeightsType.raw) == k.key.inputWeightsType.raw))
        return false;
    if (!((key.outputWeightsType.raw & k.key.outputWeightsType.raw) == k.key.outputWeightsType.raw))
        return false;
    if (!((key.inputLayout & k.key.inputLayout) != 0 || key.inputLayout == k.key.inputLayout))
        return false;
    if (!((key.outputLayout & k.key.outputLayout) != 0 || key.outputLayout == k.key.outputLayout))
        return false;
    if (!((key.weightsInputLayout & k.key.weightsInputLayout) != 0 ||
          key.weightsInputLayout == k.key.weightsInputLayout))
        return false;
    if (!((key.weightsOutputLayout & k.key.weightsOutputLayout) != 0 ||
          key.weightsOutputLayout == k.key.weightsOutputLayout))
        return false;

    return true;
}

ParamsKey ParamsKey::Merge(const ParamsKey& k) const {
    ParamsKey ret;
    ret.key.restrict.raw = key.restrict.raw | k.key.restrict.raw;
    ret.key.machineInfo.raw = key.machineInfo.raw | k.key.machineInfo.raw;
    ret.key.inputType.raw = key.inputType.raw | k.key.inputType.raw;
    ret.key.outputType.raw = key.outputType.raw | k.key.outputType.raw;
    ret.key.inputWeightsType.raw = key.inputWeightsType.raw | k.key.inputWeightsType.raw;
    ret.key.outputWeightsType.raw = key.outputWeightsType.raw | k.key.outputWeightsType.raw;
    ret.key.inputLayout = key.inputLayout | k.key.inputLayout;
    ret.key.outputLayout = key.outputLayout | k.key.outputLayout;
    ret.key.weightsInputLayout = key.weightsInputLayout | k.key.weightsInputLayout;
    ret.key.weightsOutputLayout = key.weightsOutputLayout | k.key.weightsOutputLayout;
    return ret;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ParamsKey Params::GetParamsKey() const {
    ParamsKey k;

    if (engineInfo.bSubGroupSupport) {
        k.EnableSubGroup();
    }

    if (engineInfo.bSubGroupShortSupport) {
        k.EnableSubGroupShort();
    }

    if (engineInfo.bSubGroupCharSupport) {
        k.EnableSubGroupChar();
    }

    return k;
}

std::string Params::to_string() const {
    std::stringstream s;
    s << toString(kType);
    return s.str();
}

std::string Params::to_cache_string_v2() const {
    return "";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ParamsKey optional_params::GetSupportedKey() const {
    ParamsKey k;

    for (auto l : inputLayouts) {
        k.EnableInputLayout(l);
    }

    for (auto l : outputLayouts) {
        k.EnableOutputLayout(l);
    }

    return k;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// base_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ParamsKey base_params::GetParamsKey() const {
    ParamsKey k = Params::GetParamsKey();

    bool bBatching = false;
    bool bPitches = false;
    bool bOffests = false;
    bool bDifferentTypes = false;
    bool bFP16Used = (output.GetDType() == Datatype::F16);

    for (const auto& i : inputs) {
        k.EnableInputDataType(i.GetDType());
        k.EnableInputLayout(i.GetLayout());

        bBatching |= (i.Batch().v > 1);
        bPitches |= (i.PitchesDifferFromLogicalDims());
        bOffests |= (i.GetFirstElementOffset() != 0);
        bDifferentTypes |= (i.GetDType() != output.GetDType());
        bFP16Used |= (i.GetDType() == Datatype::F16);
    }

    k.EnableOutputDataType(output.GetDType());
    k.EnableOutputLayout(output.GetLayout());

    if (bBatching) {
        k.EnableBatching();
    }

    if (bPitches || output.PitchesDifferFromLogicalDims()) {
        k.EnableTensorPitches();
    }

    if (bDifferentTypes) {
        k.EnableDifferentTypes();
    }

    if (bOffests || output.GetFirstElementOffset() != 0) {
        k.EnableTensorOffset();
    }

    if (!engineInfo.bFP16Support && bFP16Used) {
        // I'm not sure it's the best idea, but we can live with it right now
        k.EnableFP16Emulation();
    }

    return k;
}

std::string base_activation_params::to_string() const {
    std::stringstream s;
    s << "m" << m << "_n" << n << "_" << toString(function);
    return s.str();
}

std::string base_params::to_string() const {
    std::stringstream s;

    // WA to reuse old tuning cache. Code below must be replace with the following line once new cache file is merged.
    // s << Params::to_string() << "_";
    auto type_string = toString(kType);
    if (kType == KernelType::FUSED_CONV_ELTWISE) {
        type_string = "";
    }
    s << type_string << "_";

    // TODO: Remove activation from the string and recollect cache file
    bool found_fused_activation = false;
    if (!activations.empty()) {
        s << activations[0].to_string() << "_";
        found_fused_activation = true;
    }

    if (activations.empty() && !fused_ops.empty()) {
        if (fused_ops[0].GetType() == KernelType::ACTIVATION) {
            auto activation_params = fused_ops[0].GetOpParams<activation_fuse_params>()->param;
            s << activation_params.to_string() << "_";
            found_fused_activation = true;
        }
    }

    if (!found_fused_activation) {
        s << "m" << 0.f << "_n" << 0.f << "_" << toString(ActivationFunction::NONE) << "_";
    }

    for (auto input : inputs) {
        s << toString(input) << "_";
    }
    s << toString(output);

    return s.str();
}

std::string base_params::to_cache_string_v2() const {
    std::stringstream s;

    for (auto input : inputs) {
        s << toString_v2(input) << ";";
    }
    s << toString_v2(output);

    return s.str();
}



//// My debug tests
std::string to_string_input_type(const ParamsKey::Key::DataTypesKey& v) {
    std::stringstream ss;
    ss << "[BIN, F16, F32, I16, I32, I64, I8, UI16, UI32, UI8], ";
    ss << "[" << int(v.val.binary) << ",";
    ss << int(v.val.F16) << ",";
    ss << int(v.val.F32) << ",";
    ss << int(v.val.int16) << ",";
    ss << int(v.val.int32) << ",";
    ss << int(v.val.int64) << ",";
    ss << int(v.val.int8) << ",";
    ss << int(v.val.uint16) << ",";
    ss << int(v.val.uint32) << ",";
    ss << int(v.val.uint8) << "]";
    return ss.str();
}

std::string to_string_restrict(const kernel_selector::ParamsKey::Key::restrict_t& v, const std::string title) {
    std::stringstream ss;
    ss << "{ " << title << "\n";
    ss << "* activationAdditionalParamsAsInput          : " << v.val.activationAdditionalParamsAsInput << ", \n";
    ss << "* asym_d_quantization                        : " << v.val.asym_d_quantization << ", \n";
    ss << "* asym_w_quantization                        : " << v.val.asym_w_quantization << ", \n";
    ss << "* batching                                   : " << v.val.batching << ", \n";
    ss << "* biasPerFeatureMap                          : " << v.val.biasPerFeatureMap << ", \n";
    ss << "* biasPerOutput                              : " << v.val.biasPerOutput << ", \n";
    ss << "* different_input_weights_types              : " << v.val.different_input_weights_types << ", \n";
    ss << "* different_types                            : " << v.val.different_types << ", \n";
    ss << "* FP16Emulation                              : " << v.val.FP16Emulation << ", \n";
    ss << "* momentum                                   : " << v.val.momentum << ", \n";
    ss << "* nonBias                                    : " << v.val.nonBias << ", \n";
    ss << "* offset                                     : " << v.val.offset << ", \n";
    ss << "* pitches                                    : " << v.val.pitches << ", \n";
    ss << "* quantization                               : " << v.val.quantization << ", \n";
    ss << "* sym_quantization                           : " << v.val.sym_quantization << ", \n";
    ss << "* dedicated.conv.deformable                  : " << v.val.dedicated.conv.deformable << ", \n";
    ss << "* dedicated.conv.depthwise_separable_opt     : " << v.val.dedicated.conv.depthwise_separable_opt << ", \n";
    ss << "* dedicated.conv.dilation                    : " << v.val.dedicated.conv.dilation << ", \n";
    ss << "* dedicated.conv.grouped                     : " << v.val.dedicated.conv.grouped << ", \n";
    ss << "* dedicated.conv.split                       : " << v.val.dedicated.conv.split << ", \n";

    ss <<"}\n";

    return ss.str();
}

bool ParamsKey::Support_v2(const ParamsKey& k) const {
    if (!((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw)) { // check if this kernel supports this params
        std::cout << "Checking restrict : false" << std::endl;
        auto ret = to_string_restrict(key.restrict, "key.restrict");
        std::cout << ret << std::endl;
        ret = to_string_restrict(k.key.restrict, "k.key.restrict");
        std::cout << ret << std::endl;
        return false;
    } else {
        std::cout << "Checking restrict : true" << std::endl;
    }
    if (!((key.machineInfo.raw & k.key.machineInfo.raw) ==
          key.machineInfo.raw)) { // check if machine supports this kernel
            std::cout << "Checking machineInfo : false" << std::endl;
        return false;
          } else {
              std::cout << "Checking machineInfo : true" << std::endl;
          }
    if (!((key.inputType.raw & k.key.inputType.raw) == k.key.inputType.raw)) {
        std::cout << "FALSE: key.input.raw         : (" <<  key.inputType.raw  << "), " << to_string_input_type(key.inputType) << std::endl;
        std::cout << "FALSE: k.key.inputType.raw   : (" <<  k.key.inputType.raw  << "), " <<  to_string_input_type(k.key.inputType) << std::endl;
        std::cout << "FALSE: (key.inputType.raw & k.key.inputType.raw) : " << std::hex << "0x" << (key.inputType.raw & k.key.inputType.raw) << std::endl;
        return false;
    } else {
        std::cout << "TRUE: key.input.raw         : (" <<  key.inputType.raw  << "), " << to_string_input_type(key.inputType) << std::endl;
        std::cout << "TRUE: k.key.inputType.raw   : (" <<  k.key.inputType.raw  << "), " <<  to_string_input_type(k.key.inputType) << std::endl;
        std::cout << "TRUE: (key.inputType.raw & k.key.inputType.raw) : " << std::hex << "0x" << (key.inputType.raw & k.key.inputType.raw) << std::endl;
    }
    if (!((key.outputType.raw & k.key.outputType.raw) == k.key.outputType.raw)) {
        std::cout << "Checking outputType : false" << std::endl;
        return false;
    } else {
        std::cout << "Checking outputType : true" << std::endl;
    }
    if (!((key.inputWeightsType.raw & k.key.inputWeightsType.raw) == k.key.inputWeightsType.raw)) {
        std::cout << "FALSE: key.inputWeightsType.raw         : (" <<  key.inputWeightsType.raw;
        std::cout << "), " << to_string_input_type(key.inputWeightsType) << std::endl;
        std::cout << "FALSE: k.key.inputWeightsType.raw   : (" <<  k.key.inputWeightsType.raw;
        std::cout << "), " <<  to_string_input_type(k.key.inputWeightsType) << std::endl;
        std::cout << "FALSE: (key.inputWeightsType.raw & k.key.inputWeightsType.raw) : ";
        std::cout << std::hex << "0x" << (key.inputWeightsType.raw & k.key.inputWeightsType.raw) << std::endl;
        return false;
    } else {
        std::cout << "TRUE: key.inputWeightsType.raw         : (" <<  key.inputWeightsType.raw;
        std::cout << "), " << to_string_input_type(key.inputWeightsType) << std::endl;
        std::cout << "TRUE: k.key.inputWeightsType.raw   : (" <<  k.key.inputWeightsType.raw;
        std::cout << "), " <<  to_string_input_type(k.key.inputWeightsType) << std::endl;
        std::cout << "TRUE: (key.inputWeightsType.raw & k.key.inputType.raw) : ";
        std::cout << std::hex << "0x" << (key.inputWeightsType.raw & k.key.inputWeightsType.raw) << std::endl;
    }
    if (!((key.outputWeightsType.raw & k.key.outputWeightsType.raw) == k.key.outputWeightsType.raw)) {
        std::cout << "Checking outputWeightsType : false" << std::endl;
        return false;
    } else {
        std::cout << "Checking outputWeightsType : true" << std::endl;
    }
    if (!((key.inputLayout & k.key.inputLayout) != 0 || key.inputLayout == k.key.inputLayout)) {
        std::cout << "Checking inputLayout : false" << std::endl;
        return false;
    } else {
        std::cout << "Checking inputLayout : true" << std::endl;
    }
    if (!((key.outputLayout & k.key.outputLayout) != 0 || key.outputLayout == k.key.outputLayout)) {
        std::cout << "Checking outputLayout : false" << std::endl;
        return false;
    } else {
        std::cout << "Checking outputLayout : true" << std::endl;
    }
    if (!((key.weightsInputLayout & k.key.weightsInputLayout) != 0 ||
          key.weightsInputLayout == k.key.weightsInputLayout)) {
              std::cout << "Checking weightsInputLayout : false" << std::endl;
        return false;
          } else {
              std::cout << "Checking weightsInputLayout : true" << std::endl;
          }
    if (!((key.weightsOutputLayout & k.key.weightsOutputLayout) != 0 ||
          key.weightsOutputLayout == k.key.weightsOutputLayout)) {
            std::cout << "Checking weightsOutputLayout : false" << std::endl;
        return false;
          } else {
              std::cout << "Checking weightsOutputLayout : true" << std::endl;
          }

    return true;
}

std::string ParamsKey::to_string() const {
    std::stringstream ss;
    ss << "* restrict             : " << key.restrict.raw << ", \n";
    ss << "* machineInfo          : " << key.machineInfo.raw << ", \n";
    ss << "* inputType            : " << key.inputType.raw << " " << to_string_input_type(key.inputType) << ", \n";
    ss << "* outputType           : " << key.outputType.raw << " " << to_string_input_type(key.outputType) << ", \n";
    ss << "* inputWeightsType     : " << key.inputWeightsType.raw << " " << to_string_input_type(key.inputWeightsType) << ", \n";
    ss << "* outputWeightsType    : " << key.outputWeightsType.raw << " " << to_string_input_type(key.outputWeightsType) << ", \n";
    ss << "* inputLayout          : " << key.inputLayout << ", \n";
    ss << "* outputLayout         : " << key.outputLayout << ", \n";
    ss << "* weightsInputLayout   : " << key.weightsInputLayout << ", \n";
    ss << "* weightsOutputLayout  : " << key.weightsOutputLayout << ", \n";
    return ss.str();
}

ParamsKey base_params::GetParamsKey_v2() const {
    ParamsKey k = Params::GetParamsKey();

    bool bBatching = false;
    bool bPitches = false;
    bool bOffests = false;
    bool bDifferentTypes = false;
    bool bFP16Used = (output.GetDType() == Datatype::F16);

    std::cout << "# of inputs: " << inputs.size() << std::endl;
    for (const auto& i : inputs) {
        {
            switch (i.GetDType()) {
                case Datatype::INT8:
                    std::cout << "INT8 type" << std::endl;
                    break;
                case Datatype::UINT8:
                    std::cout << "UINT8 type" << std::endl;
                    break;
                case Datatype::INT16:
                    std::cout << "INT16 type" << std::endl;
                    break;
                case Datatype::UINT16:
                    std::cout << "UINT16 type" << std::endl;
                    break;
                case Datatype::INT32:
                    std::cout << "INT32 type" << std::endl;
                    break;
                case Datatype::UINT32:
                    std::cout << "UINT32 type" << std::endl;
                    break;
                case Datatype::INT64:
                    std::cout << "INT64 type" << std::endl;
                    break;
                case Datatype::F16:
                    std::cout << "F16 type" << std::endl;
                    break;
                case Datatype::F32:
                    std::cout << "F32 type" << std::endl;
                    break;
                case Datatype::BINARY:
                    std::cout << "BINARY type" << std::endl;
                    break;
                default:
                    break;
            }
        }
        k.EnableInputDataType(i.GetDType());
        k.EnableInputLayout(i.GetLayout());

        bBatching |= (i.Batch().v > 1);
        bPitches |= (i.PitchesDifferFromLogicalDims());
        bOffests |= (i.GetFirstElementOffset() != 0);
        bDifferentTypes |= (i.GetDType() != output.GetDType());
        bFP16Used |= (i.GetDType() == Datatype::F16);
    }

    k.EnableOutputDataType(output.GetDType());
    k.EnableOutputLayout(output.GetLayout());

    if (bBatching) {
        k.EnableBatching();
    }

    if (bPitches || output.PitchesDifferFromLogicalDims()) {
        k.EnableTensorPitches();
    }

    if (bDifferentTypes) {
        k.EnableDifferentTypes();
    }

    if (bOffests || output.GetFirstElementOffset() != 0) {
        k.EnableTensorOffset();
    }

    if (!engineInfo.bFP16Support && bFP16Used) {
        // I'm not sure it's the best idea, but we can live with it right now
        k.EnableFP16Emulation();
    }

    return k;
}


}  // namespace kernel_selector
