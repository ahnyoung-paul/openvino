// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/state_conversion_executor.hpp"

#include "impls/ocl/kernels_cache.hpp"
#include "openvino/core/except.hpp"

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace {

std::shared_ptr<kernel_string> make_source(state_conversion_key key) {
    auto source = std::make_shared<kernel_string>();
    source->entry_point = "state_convert_v2_" + std::to_string(static_cast<int>(key.first)) + "_" +
                          std::to_string(static_cast<int>(key.second));

    const char* input_type = nullptr;
    const char* output_type = nullptr;
    std::string value;
    switch (key.first) {
    case data_types::bf16:
        input_type = "ushort";
        output_type = "half";
        value = "convert_half_rte(as_float(((uint)input[src_index]) << 16))";
        break;
    case data_types::f32:
        input_type = "float";
        if (key.second == data_types::f16) {
            output_type = "half";
            value = "convert_half_rte(input[src_index])";
        } else {
            output_type = "double";
            value = "(double)input[src_index]";
        }
        break;
    case data_types::f64:
        input_type = "double";
        output_type = "float";
        value = "convert_float_rte(input[src_index])";
        break;
    case data_types::i32:
        input_type = "int";
        output_type = key.second == data_types::i64 ? "long" :
                      key.second == data_types::u64 ? "ulong" : "uint";
        value = "(" + std::string(output_type) + ")input[src_index]";
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported state conversion type");
    }

    if (key.first == data_types::f64 || key.second == data_types::f64)
        source->str += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    if (key.second == data_types::f16)
        source->str += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    source->str += "__kernel void " + source->entry_point + "(__global const " + input_type + "* input, "
                  "__global " + output_type + "* output, ulong count, ulong src_offset, ulong padded, ulong transpose,\n"
                  "    ulong d0, ulong d1, ulong d2, ulong d3, ulong d4, ulong d5,\n"
                  "    ulong s0, ulong s1, ulong s2, ulong s3, ulong s4, ulong s5) {\n"
                  "    size_t index = get_global_id(0);\n"
                  "    if (index >= count) return;\n"
                  "    size_t src_index = src_offset + index;\n"
                  "    if (padded) {\n"
                  "        const ulong dims[6] = {d0, d1, d2, d3, d4, d5};\n"
                  "        const ulong strides[6] = {s0, s1, s2, s3, s4, s5};\n"
                  "        size_t remaining = index;\n"
                  "        src_index = src_offset;\n"
                  "        for (int axis = 5; axis >= 0; --axis) {\n"
                  "            src_index += (remaining % dims[axis]) * strides[axis];\n"
                  "            remaining /= dims[axis];\n"
                  "        }\n"
                  "    }\n"
                  "    size_t dst_index = index;\n"
                  "    if (transpose) {\n"
                  "        size_t plane = d4 * d5;\n"
                  "        dst_index = (index / plane) * plane + (index % d5) * d4 + (index / d5) % d4;\n"
                  "    }\n"
                  "    output[dst_index] = " + value + ";\n"
                  "}\n";
    return source;
}

}  // namespace

void program::prepare_state_conversions(const std::vector<state_conversion_key>& requested_keys) {
    std::lock_guard<std::mutex> lock(_state_conversion_mutex);

    std::set<state_conversion_key> unique_keys;
    if (_engine.runtime_type() == runtime_types::ocl) {
        const auto& device_info = _engine.get_device_info();
        for (const auto& key : requested_keys) {
            const bool needs_fp64 = key.first == data_types::f64 || key.second == data_types::f64;
            const bool needs_fp16 = key.first == data_types::f16 || key.second == data_types::f16;
            if (key.first != key.second && state_conversion_executor::supports(key) &&
                (!needs_fp64 || device_info.supports_fp64) && (!needs_fp16 || device_info.supports_fp16))
                unique_keys.insert(key);
        }
    }
    std::vector<state_conversion_key> keys(unique_keys.begin(), unique_keys.end());

    if (_state_conversions_prepared) {
        const auto existing_keys = _state_conversion_executor ? _state_conversion_executor->get_keys()
                                                              : std::vector<state_conversion_key>{};
        OPENVINO_ASSERT(keys == existing_keys, "[GPU] State conversion kernels do not match the network");
        return;
    }

    if (!keys.empty()) {
        std::vector<std::shared_ptr<kernel_string>> sources;
        sources.reserve(keys.size());
        for (const auto& key : keys)
            sources.push_back(make_source(key));

        kernel_impl_params params;
        auto compiled = _kernels_cache->compile(params, sources);
        OPENVINO_ASSERT(compiled.size() == 1, "[GPU] State conversion compilation failed");
        std::vector<kernel::ptr> kernels(keys.size());
        for (const auto& entry : compiled.begin()->second) {
            OPENVINO_ASSERT(entry.second < kernels.size(), "[GPU] Invalid state conversion kernel index");
            kernels[entry.second] = entry.first;
        }
        auto executor = std::make_shared<state_conversion_executor>();
        executor->set_kernels(keys, kernels);
        _state_conversion_executor = std::move(executor);
    }
    _state_conversions_prepared = true;
}

}  // namespace cldnn
