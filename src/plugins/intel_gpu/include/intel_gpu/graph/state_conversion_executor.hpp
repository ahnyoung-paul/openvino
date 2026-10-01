// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "openvino/core/except.hpp"

#include <cstdint>
#include <chrono>
#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace cldnn {

using state_conversion_key = std::pair<data_types, data_types>;

struct state_conversion_timing {
    double prepare_us = 0;
    double lock_us = 0;
    double bind_us = 0;
    double log_us = 0;
    double enqueue_us = 0;
};

// Holds compiled kernels shared by all states of one program.
// execute() serializes argument binding and submission across InferRequests.
class state_conversion_executor {
public:
    static bool supports(state_conversion_key key) {
        switch (key.first) {
        case data_types::bf16:
            return key.second == data_types::f16;
        case data_types::f32:
            return key.second == data_types::f16 || key.second == data_types::f64;
        case data_types::f64:
            return key.second == data_types::f32;
        case data_types::i32:
            return key.second == data_types::i64 || key.second == data_types::u64 || key.second == data_types::u32;
        default:
            return false;
        }
    }

    void set_kernels(const std::vector<state_conversion_key>& keys, const std::vector<kernel::ptr>& kernels) {
        OPENVINO_ASSERT(keys.size() == kernels.size(), "[GPU] State conversion kernel count mismatch");
        _kernels.clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            OPENVINO_ASSERT(supports(keys[i]) && kernels[i], "[GPU] Invalid state conversion kernel");
            OPENVINO_ASSERT(_kernels.emplace(keys[i], kernels[i]).second, "[GPU] Duplicate state conversion kernel");
        }
    }

    bool has_kernel(state_conversion_key key) const {
        return _kernels.find(key) != _kernels.end();
    }

    std::vector<state_conversion_key> get_keys() const {
        std::vector<state_conversion_key> keys;
        for (const auto& entry : _kernels)
            keys.push_back(entry.first);
        return keys;
    }

    std::vector<kernel::ptr> get_kernels() const {
        std::vector<kernel::ptr> kernels;
        for (const auto& entry : _kernels)
            kernels.push_back(entry.second);
        return kernels;
    }

    event::ptr execute(state_conversion_key key, memory::cptr src, memory::cptr dst, stream& stream, size_t count,
                       const std::vector<event::ptr>& dependencies = {}, state_conversion_timing* timing = nullptr) {
        using clock = std::chrono::steady_clock;
        const auto stamp = [timing] { return timing ? clock::now() : clock::time_point{}; };
        const auto elapsed = [](clock::time_point begin, clock::time_point end) {
            return std::chrono::duration<double, std::micro>(end - begin).count();
        };
        const auto start = stamp();
        auto it = _kernels.find(key);
        OPENVINO_ASSERT(it != _kernels.end(), "[GPU] State conversion kernel was not prepared");
        if (count == 0)
            return nullptr;

        kernel_arguments_desc desc;
        const auto local_size = std::min(count,
                                         static_cast<size_t>(src->get_engine()->get_device_info().max_work_group_size));
        OPENVINO_ASSERT(local_size > 0, "[GPU] Invalid state conversion work-group limit");
        OPENVINO_ASSERT(count <= std::numeric_limits<size_t>::max() - (local_size - 1),
                        "[GPU] State conversion work size is too large");
        desc.workGroups.global = {((count - 1) / local_size + 1) * local_size, 1, 1};
        desc.workGroups.local = {local_size, 1, 1};
        desc.arguments = {{argument_desc::Types::INPUT, 0},
                          {argument_desc::Types::OUTPUT, 0},
                          {argument_desc::Types::SCALAR, 0}};
        scalar_desc scalar{};
        scalar.t = scalar_desc::Types::UINT64;
        scalar.v.u64 = static_cast<uint64_t>(count);
        desc.scalars.push_back(scalar);
        desc.layerID = "state_conversion";

        kernel_arguments_data args;
        args.inputs.push_back(std::move(src));
        args.outputs.push_back(std::move(dst));
        args.scalars = &desc.scalars;

        const auto before_lock = stamp();
        std::lock_guard<std::mutex> lock(_mutex);
        const auto after_lock = stamp();
        stream.set_arguments(*it->second, desc, args);
        const auto after_bind = stamp();
        GPU_DEBUG_INFO << "[state_conversion] OpenCL enqueue kernel=" << it->second->get_id()
                       << " dtype=" << ov::element::Type(key.first) << "->" << ov::element::Type(key.second)
                       << " count=" << count << " global=" << desc.workGroups.global[0]
                       << " local=" << desc.workGroups.local[0] << std::endl;
        const auto before_enqueue = stamp();
        auto completion = stream.enqueue_kernel(*it->second, desc, args, dependencies, true);
        const auto after_enqueue = stamp();
        if (timing) {
            timing->prepare_us = elapsed(start, before_lock);
            timing->lock_us = elapsed(before_lock, after_lock);
            timing->bind_us = elapsed(after_lock, after_bind);
            timing->log_us = elapsed(after_bind, before_enqueue);
            timing->enqueue_us = elapsed(before_enqueue, after_enqueue);
        }
        return completion;
    }

private:
    std::map<state_conversion_key, kernel::ptr> _kernels;
    std::mutex _mutex;
};

}  // namespace cldnn
