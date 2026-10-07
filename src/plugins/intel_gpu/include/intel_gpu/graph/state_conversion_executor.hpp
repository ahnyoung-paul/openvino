// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/stream.hpp"

#include "openvino/core/except.hpp"

#include <cstdint>
#include <array>
#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace cldnn {

using state_conversion_key = std::pair<data_types, data_types>;

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
                       const std::vector<event::ptr>& dependencies = {},
                       const layout* source_layout = nullptr, bool transpose = false) {
        auto it = _kernels.find(key);
        OPENVINO_ASSERT(it != _kernels.end(), "[GPU] State conversion kernel was not prepared");
        if (count == 0)
            return nullptr;

        const auto& input_layout = source_layout ? *source_layout : src->get_layout();
        const auto shape = input_layout.get_shape();
        OPENVINO_ASSERT(!shape.empty() && shape.size() <= 6 && format::is_default_format(input_layout.format),
                        "[GPU] Unsupported state conversion source layout");
        OPENVINO_ASSERT(count == ov::shape_size(shape), "[GPU] State conversion element count mismatch");
        OPENVINO_ASSERT(!transpose || shape.size() >= 2, "[GPU] State transpose requires at least two axes");
        const auto physical_shape = input_layout.get_padded_dims();
        std::array<uint64_t, 6> dimensions{1, 1, 1, 1, 1, 1};
        std::array<uint64_t, 6> strides{0, 0, 0, 0, 0, 0};
        size_t pitch = 1;
        size_t source_offset = 0;
        size_t last_element = 0;
        for (size_t i = shape.size(); i-- > 0;) {
            const size_t axis = 6 - shape.size() + i;
            dimensions[axis] = shape[i];
            strides[axis] = pitch;
            OPENVINO_ASSERT(input_layout.data_padding._lower_size[i] >= 0 &&
                            input_layout.data_padding._upper_size[i] >= 0,
                            "[GPU] Negative state source padding is unsupported");
            const auto lower = static_cast<size_t>(input_layout.data_padding._lower_size[i]);
            OPENVINO_ASSERT(lower <= (std::numeric_limits<size_t>::max() - source_offset) / pitch,
                            "[GPU] State conversion source offset overflow");
            source_offset += lower * pitch;
            OPENVINO_ASSERT(shape[i] > 0 && shape[i] - 1 <= (std::numeric_limits<size_t>::max() - last_element) / pitch,
                            "[GPU] State conversion source span overflow");
            last_element += (shape[i] - 1) * pitch;
            OPENVINO_ASSERT(physical_shape[i] > 0 &&
                            static_cast<size_t>(physical_shape[i]) <= std::numeric_limits<size_t>::max() / pitch,
                            "[GPU] State conversion source pitch overflow");
            pitch *= static_cast<size_t>(physical_shape[i]);
        }
        const auto element_size = data_type_traits::size_of(key.first);
        OPENVINO_ASSERT(src->size() >= element_size && source_offset < src->size() / element_size &&
                        last_element < src->size() / element_size - source_offset,
                        "[GPU] State conversion source memory is too small");
        OPENVINO_ASSERT(count <= dst->size() / data_type_traits::size_of(key.second),
                        "[GPU] State conversion destination memory is too small");

        kernel_arguments_desc desc;
        const auto local_size = std::min(count,
                                         static_cast<size_t>(src->get_engine()->get_device_info().max_work_group_size));
        OPENVINO_ASSERT(local_size > 0, "[GPU] Invalid state conversion work-group limit");
        OPENVINO_ASSERT(count <= std::numeric_limits<size_t>::max() - (local_size - 1),
                        "[GPU] State conversion work size is too large");
        desc.workGroups.global = {((count - 1) / local_size + 1) * local_size, 1, 1};
        desc.workGroups.local = {local_size, 1, 1};
        desc.arguments = {{argument_desc::Types::INPUT, 0},
                          {argument_desc::Types::OUTPUT, 0}};
        const auto add_scalar = [&](uint64_t value) {
            desc.arguments.push_back({argument_desc::Types::SCALAR, static_cast<uint32_t>(desc.scalars.size())});
            scalar_desc scalar{};
            scalar.t = scalar_desc::Types::UINT64;
            scalar.v.u64 = value;
            desc.scalars.push_back(scalar);
        };
        add_scalar(count);
        add_scalar(source_offset);
        add_scalar(static_cast<bool>(input_layout.data_padding));
        add_scalar(transpose);
        for (auto dimension : dimensions)
            add_scalar(dimension);
        for (auto stride : strides)
            add_scalar(stride);
        desc.layerID = "state_conversion";

        kernel_arguments_data args;
        args.inputs.push_back(std::move(src));
        args.outputs.push_back(std::move(dst));
        args.scalars = &desc.scalars;

        std::lock_guard<std::mutex> lock(_mutex);
        stream.set_arguments(*it->second, desc, args);
        return stream.enqueue_kernel(*it->second, desc, args, dependencies, true);
    }

private:
    std::map<state_conversion_key, kernel::ptr> _kernels;
    std::mutex _mutex;
};

}  // namespace cldnn
