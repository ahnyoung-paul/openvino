// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

enum class ConstantSubgraphType {
    SINGLE_COMPONENT,
    SEVERAL_COMPONENT
};

std::ostream& operator<<(std::ostream &os, ConstantSubgraphType type);

typedef std::tuple <
    ConstantSubgraphType,
    InferenceEngine::SizeVector, // input shape
    InferenceEngine::Precision,  // input precision
    std::string                  // Device name
> constResultParams;

class ConstantResultSubgraphTest : public testing::WithParamInterface<constResultParams>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<constResultParams>& obj);
    void createGraph(const ConstantSubgraphType& type, const InferenceEngine::SizeVector &inputShape, const InferenceEngine::Precision &inputPrecision);
protected:
    void SetUp() override;
};


using KernelParams = std::tuple<std::size_t, std::size_t>;

class CLKernelTest : public testing::WithParamInterface<KernelParams>,
                                virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<KernelParams> obj);
    void SetUp() override;
    void TearDown() override;
    void Run() override;
};

}  // namespace SubgraphTestsDefinitions
