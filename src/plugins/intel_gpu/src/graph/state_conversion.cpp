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

kernel_impl_params cache_params() {
    kernel_impl_params params;
    params.input_layouts.emplace_back(ov::PartialShape{1, 1, 1, 1}, data_types::u8, format::bfyx);
    params.output_layouts.emplace_back(ov::PartialShape{1, 1, 1, 1}, data_types::u8, format::bfyx);
    return params;
}

std::shared_ptr<kernel_string> make_source(state_conversion_key key) {
    auto source = std::make_shared<kernel_string>();
    source->entry_point = "state_convert_" + std::to_string(static_cast<int>(key.first)) + "_" +
                          std::to_string(static_cast<int>(key.second));

    const char* input_type = nullptr;
    const char* output_type = nullptr;
    std::string value;
    switch (key.first) {
    case data_types::bf16:
        input_type = "ushort";
        output_type = "half";
        value = "convert_half_rte(as_float(((uint)input[index]) << 16))";
        break;
    case data_types::f32:
        input_type = "float";
        if (key.second == data_types::f16) {
            output_type = "half";
            value = "convert_half_rte(input[index])";
        } else {
            output_type = "double";
            value = "(double)input[index]";
        }
        break;
    case data_types::f64:
        input_type = "double";
        output_type = "float";
        value = "convert_float_rte(input[index])";
        break;
    case data_types::i32:
        input_type = "int";
        output_type = key.second == data_types::i64 ? "long" :
                      key.second == data_types::u64 ? "ulong" : "uint";
        value = "(" + std::string(output_type) + ")input[index]";
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported state conversion type");
    }

    if (key.first == data_types::f64 || key.second == data_types::f64)
        source->str += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    if (key.second == data_types::f16)
        source->str += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    source->str += "__kernel void " + source->entry_point + "(__global const " + input_type + "* input, "
                  "__global " + output_type + "* output, ulong count) {\n"
                  "    size_t index = get_global_id(0);\n"
                  "    if (index < count) output[index] = " + value + ";\n"
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

        auto compiled = _kernels_cache->compile(cache_params(), sources);
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
