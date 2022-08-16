// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <cstdlib>
namespace {

#define test_param 128

using namespace ngraph;

class ConvWinograd : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;

        auto type = element::f16;

        Shape weightShape{test_param, test_param, 3, 3};
        Shape constShape{1, test_param, 1, 1};
        Shape convInputShape{1, test_param, 32, 18};

        std::vector<float> weightVec;
        const size_t totalWeightSize = test_param * test_param * 3 * 3;
        for (int i = 0; i < totalWeightSize; i++) {
            weightVec.push_back((static_cast<float>(std::rand()) / RAND_MAX) - 0.5);
        }
        std::vector<float> constVec;
        const size_t totalVecSize = test_param;
        for (int i = 0; i < totalVecSize; i++) {
            constVec.push_back((static_cast<float>(std::rand()) * 10.0f / RAND_MAX) - 5.0f);
        }
        auto weight = opset8::Constant::create(type, weightShape, weightVec);
        auto constant = opset8::Constant::create(type, constShape, constVec);
        auto input = std::make_shared<opset8::Parameter>(type, convInputShape);
        auto conv = std::make_shared<opset8::Convolution>(input, constant, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

        function = std::make_shared<ngraph::Function>(NodeVector{conv}, ParameterVector{input});
    }
};

TEST_F(ConvWinograd, smoke_ConvWinograd) {
    Run();
}

}   // namespace