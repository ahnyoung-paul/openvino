// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class IncreasePrecisionForVisionPooler : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePrecisionForVisionPooler");
    IncreasePrecisionForVisionPooler();
};

}  // namespace ov::intel_gpu
