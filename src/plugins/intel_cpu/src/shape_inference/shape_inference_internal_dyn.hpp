// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "shape_inference/shape_inference_status.hpp"
#include "shape_inference_cpu.hpp"

namespace ov::intel_cpu {

/**
 * Shape inference class for operations with internal dynamism. To reflect the fact that the output shapes may only be
 * calculated after the operation has been performed, the data dependency mask is fully set.
 *
 */
class InternalDynShapeInfer final : public ShapeInferEmptyPads {
public:
    InternalDynShapeInfer() = default;
    Result infer([[maybe_unused]] const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        return {{}, ShapeInferStatus::skip};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return FULL_PORT_MASK;
    }
};

class InternalDynShapeInferFactory final : public ShapeInferFactory {
public:
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<InternalDynShapeInfer>();
    }
};

}  // namespace ov::intel_cpu
