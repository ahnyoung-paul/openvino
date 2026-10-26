// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class CollectSinCosConstants : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CollectSinCosConstants");
    using ConstantSet = std::unordered_set<const ov::Node*>;

    explicit CollectSinCosConstants(const std::shared_ptr<ConstantSet>& constants);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ConstantSet> m_constants;
};

bool skip_constant_precision_preservation(const std::shared_ptr<const ov::Node>& node,
                                          const CollectSinCosConstants::ConstantSet& sincos_constants);

}  // namespace ov::intel_gpu
