// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class IncreasePrecisionForVisionPooler : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("IncreasePrecisionForVisionPooler");
    IncreasePrecisionForVisionPooler();
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
