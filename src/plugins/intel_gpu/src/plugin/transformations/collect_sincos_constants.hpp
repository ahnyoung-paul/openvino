// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// Temporary debugging only: remove this pass and its registration after investigation.
class DumpNetwork : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DumpNetwork");

    DumpNetwork(const std::string& path, const std::string& tag);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::string m_path;
    std::string m_tag;
};

class CollectSinCosConstants : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CollectSinCosConstants");
    using ConstantSet = std::unordered_set<const ov::Node*>;

    explicit CollectSinCosConstants(const std::shared_ptr<ConstantSet>& constants);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ConstantSet> m_constants;
};

}  // namespace ov::intel_gpu
