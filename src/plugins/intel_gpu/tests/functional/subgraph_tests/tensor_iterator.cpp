// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <transformations/control_flow/unroll_tensor_iterator.hpp>

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using DynamicTensorIteratorParams = typename std::tuple<
        InputShape,                             // input shapes
        ngraph::op::RecurrentSequenceDirection, // sequence direction
        std::string,                            // device name
        InferenceEngine::Precision,             // precision
        ov::AnyMap                              // configuration
        >;

/**
 * Test case with Dynamic SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class DynamicTensorIteratorTest : public testing::WithParamInterface<DynamicTensorIteratorParams>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicTensorIteratorParams> &obj) {
        InputShape data_shapes;
        ngraph::op::RecurrentSequenceDirection seq_direction;
        std::string target_device;
        InferenceEngine::Precision data_precision;
        ov::Any configuration;
        std::tie(data_shapes,
                    seq_direction,
                    target_device,
                    data_precision,
                    configuration) = obj.param;
        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({data_shapes.first}) << "_";
        result << ov::test::utils::vec2str(data_shapes.second) << "_";
        result << ")_";
        result << "direction=" << seq_direction << "_";
        result << "netPRC=" << data_precision << "_";
        result << "targetDevice=" << target_device << "_";
        return result.str();
    }

private:
    InputShape data_shapes;
    ngraph::op::RecurrentSequenceDirection seq_direction;
    InferenceEngine::Precision data_prc;

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(data_shapes,
                    seq_direction,
                    targetDevice,
                    data_prc,
                    configuration) = GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_prc);

        size_t sequence_axis = 1;
        const auto conat_init_shape = data_shapes.first;
        init_input_shapes({data_shapes});

        auto X = std::make_shared<ngraph::opset5::Parameter>(ngPrc, conat_init_shape);
        auto Y = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ov::Shape{1, 1, 128});
        auto Z = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ov::Shape{1, 1, 128});
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ov::Shape{1}, {1});
        auto squeeze_y = std::make_shared<ngraph::opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<ngraph::opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ov::Shape{1, 1, 16});
        auto Yi = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ov::Shape{1, 128});
        auto Zi = std::make_shared<ngraph::opset5::Parameter>(ngPrc, ov::Shape{1, 128});
        auto seq_body_param = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ov::PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<ngraph::opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngPrc, ov::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngPrc, ov::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngPrc, ov::Shape{512}, b_val);

        auto rnn_cell = std::make_shared<ngraph::opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ov::Shape{1}, {1});
        auto Ho = std::make_shared<ngraph::opset5::Result>(rnn_cell->output(0));

        auto Co = std::make_shared<ngraph::opset5::Result>(rnn_cell->output(1));

        auto unsqueeze_y = std::make_shared<ngraph::opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<ngraph::opset5::Result>(unsqueeze_y);

        auto body = std::make_shared<ov::Model>(ov::OutputVector{Y_out, Ho, Co}, ov::ParameterVector{Xi, Yi, Zi, seq_body_param});

        auto tensor_iterator = std::make_shared<ngraph::opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        // 2. Set PortMap
        if (seq_direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
            tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, sequence_axis);
            tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, sequence_axis);
        } else if (seq_direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
            tensor_iterator->set_sliced_input(Xi, X, -1, -1, 1, 0, sequence_axis);
            tensor_iterator->get_concatenated_slices(Y_out, -1, -1, 1, 0, sequence_axis);
        } else {
            OPENVINO_THROW("Bidirectional case is not supported.");
        }

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto shape_of = std::make_shared<ngraph::opset5::ShapeOf>(X);
        auto indices = ngraph::opset5::Constant::create(ngraph::element::i32, {1}, {1});
        auto axis = ngraph::opset5::Constant::create(ngraph::element::i32, {}, {0});
        auto seq_lengths = std::make_shared<ngraph::opset5::Gather>(shape_of, indices, axis);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<ngraph::opset5::Result>(
            std::make_shared<ngraph::opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<ngraph::opset5::Result>(
            std::make_shared<ngraph::opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<ngraph::opset5::Result>(
            std::make_shared<ngraph::opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");
        function = std::make_shared<ov::Model>(ov::NodeVector{res_ti_Y, res_ti_H, res_ti_C}, ov::ParameterVector{X, Y, Z});
    }
};


TEST_P(DynamicTensorIteratorTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

std::vector<InputShape> input_shapes = {
    InputShape(ov::PartialShape({1, -1, 16}), {{1, 30, 16}, {1, 10, 16}, {1, 5, 16}})
};

ov::AnyMap net_configuration = {
    {GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, PluginConfigParams::NO}
};

std::vector<InferenceEngine::Precision> net_precision = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32
};

std::vector<ngraph::op::RecurrentSequenceDirection> reccurent_sequence_direction = {
    ngraph::op::RecurrentSequenceDirection::FORWARD,
    ngraph::op::RecurrentSequenceDirection::REVERSE,
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicTensorIterator, DynamicTensorIteratorTest,
                        testing::Combine(
                        /* data_shape */ testing::ValuesIn(input_shapes),
                        /* direction */ testing::ValuesIn(reccurent_sequence_direction),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* data_prc */ testing::ValuesIn(net_precision),
                        /* configuration */ testing::Values<ov::AnyMap>(net_configuration)),
                        DynamicTensorIteratorTest::getTestCaseName);

} // namespace GPULayerTestsDefinitions