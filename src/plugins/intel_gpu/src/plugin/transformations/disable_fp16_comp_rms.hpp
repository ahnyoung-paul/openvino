// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class DisableFP16CompForRMS: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForRMS");
    DisableFP16CompForRMS();
};

class DebugRMSFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DebugRMSFusion");
    DebugRMSFusion(const ov::intel_gpu::ExecutionConfig& config,
        std::string tag = "no_name", bool dump_graph = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
    std::string m_tag;
    bool m_dump_graph;
    const ov::intel_gpu::ExecutionConfig& m_config;
};

}   // namespace ov::intel_gpu
