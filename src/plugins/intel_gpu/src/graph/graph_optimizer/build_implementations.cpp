// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "read_value_inst.h"

#include "intel_gpu/graph/state_conversion_executor.hpp"
#include "intel_gpu/runtime/itt.hpp"

#include <set>
#include <utility>

using namespace cldnn;

namespace {

kernel_impl_params state_conversion_cache_params() {
    kernel_impl_params params;
    params.input_layouts.emplace_back(ov::PartialShape{1, 1, 1, 1}, data_types::u8, format::bfyx);
    params.output_layouts.emplace_back(ov::PartialShape{1, 1, 1, 1}, data_types::u8, format::bfyx);
    return params;
}

std::shared_ptr<kernel_string> make_state_conversion_source(state_conversion_key key) {
    auto source = std::make_shared<kernel_string>();
    source->entry_point = "state_convert_" + std::to_string(static_cast<int>(key.first)) + "_" +
                          std::to_string(static_cast<int>(key.second));
    const auto input_type = key.first == data_types::bf16 ? "ushort" : "float";
    const auto value = key.first == data_types::bf16 ? "as_float(((uint)input[index]) << 16)" : "input[index]";
    source->str = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                  "__kernel void " + source->entry_point + "(__global const " + input_type + "* input, "
                  "__global half* output, ulong count) {\n"
                  "    size_t index = get_global_id(0);\n"
                  "    if (index < count) output[index] = convert_half_rte(" + value + ");\n"
                  "}\n";
    return source;
}

}  // namespace

void build_implementations::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::build_implementations");
    if (p.get_config().get_partial_build_program()) {
        return;
    }

    auto& cache = p.get_kernels_cache();
    for (const auto& n : p.get_processing_order()) {
        if (auto* impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            cache.add_kernels_source(*params, impl->get_kernels_source());
        }
    }

    std::vector<state_conversion_key> conversion_keys;
    auto conversion_params = state_conversion_cache_params();
    if (p.get_engine().runtime_type() == runtime_types::ocl) {
        std::set<state_conversion_key> unique_keys;
        for (const auto& n : p.get_processing_order()) {
            if (!n->is_type<read_value>())
                continue;
            const auto& primitive = n->as<read_value>().get_primitive();
            auto src_type = primitive->user_specified_type == ov::element::dynamic
                                ? n->get_output_layout(0).data_type
                                : primitive->user_specified_type.get_type_enum();
            state_conversion_key key{src_type, n->get_output_layout(0).data_type};
            if (key.first != key.second && state_conversion_executor::supports(key))
                unique_keys.insert(key);
        }
        conversion_keys.assign(unique_keys.begin(), unique_keys.end());
        if (!conversion_keys.empty()) {
            std::vector<std::shared_ptr<kernel_string>> sources;
            for (const auto& key : conversion_keys)
                sources.push_back(make_state_conversion_source(key));
            cache.add_kernels_source(conversion_params, sources);
        }
    }

    cache.build_all();
    for (const auto& n : p.get_processing_order()) {
        if (auto* impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            impl->init_kernels(cache, *params);
            impl->reset_kernels_source();
        }
    }
    if (!conversion_keys.empty()) {
        auto executor = std::make_shared<state_conversion_executor>();
        executor->set_kernels(conversion_keys, cache.get_kernels(conversion_params));
        p.set_state_conversion_executor(std::move(executor));
    }
    cache.reset();
}
