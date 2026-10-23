// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "intel_gpu/op/gemm.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/increase_position_ids_precision.hpp"

namespace {

void run_discovery(const std::shared_ptr<ov::Model>& model, bool disabled = false) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>();
    if (disabled) {
        manager.get_pass_config()->set_callback<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>(
            [](const std::shared_ptr<const ov::Node>&) { return true; });
    }
    manager.run_passes(model);
}

void expect_fp32_pair(const std::shared_ptr<ov::Model>& model) {
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ov::OutputVector angles;
    for (size_t i = 0; i < 2; ++i) {
        auto boundary = model->get_results()[i]->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(boundary));
        EXPECT_EQ(boundary->get_output_element_type(0), ov::element::f16);
        const auto trig = boundary->input_value(0).get_node_shared_ptr();
        if (i == 0) {
            ASSERT_TRUE(ov::is_type<ov::op::v0::Sin>(trig));
        } else {
            ASSERT_TRUE(ov::is_type<ov::op::v0::Cos>(trig));
        }
        EXPECT_EQ(trig->get_output_element_type(0), ov::element::f32);
        angles.push_back(trig->input_value(0));
    }
    ASSERT_EQ(angles[0], angles[1]);
    const auto integer_inputs = [](const ov::Output<ov::Node>& root) {
        std::map<ov::Output<ov::Node>, size_t> inputs;
        std::unordered_set<const ov::Node*> visited;
        ov::OutputVector pending{root};
        while (!pending.empty()) {
            const auto node = pending.back().get_node_shared_ptr();
            pending.pop_back();
            if (!visited.insert(node.get()).second) {
                continue;
            }
            for (const auto& input : node->input_values()) {
                if (input.get_element_type().is_real()) {
                    pending.push_back(input);
                } else {
                    ++inputs[input];
                }
            }
        }
        return inputs;
    };
    EXPECT_EQ(integer_inputs(angles[0]), integer_inputs(model->get_results()[2]->input_value(0)));
    std::unordered_set<const ov::Node*> visited;
    while (!angles.empty()) {
        const auto output = angles.back();
        angles.pop_back();
        EXPECT_EQ(output.get_element_type(), ov::element::f32) << output;
        if (!visited.insert(output.get_node()).second) {
            continue;
        }
        for (const auto& input : output.get_node()->input_values()) {
            if (input.get_element_type().is_real()) {
                if (input.get_element_type() == ov::element::f16 &&
                    ov::is_type<ov::op::v0::Parameter>(input.get_node())) {
                    EXPECT_TRUE(ov::is_type<ov::op::v0::Convert>(output.get_node()));
                    continue;
                }
                angles.push_back(input);
            }
        }
    }
    const auto nodes = model->get_ordered_ops();
    run_discovery(model);
    EXPECT_EQ(nodes, model->get_ordered_ops());
}

std::shared_ptr<ov::Model> make_pair(const ov::Output<ov::Node>& angle,
                                    const ov::ParameterVector& parameters) {
    auto sin = std::make_shared<ov::op::v0::Sin>(angle);
    auto cos = std::make_shared<ov::op::v0::Cos>(angle);
    return std::make_shared<ov::Model>(ov::OutputVector{sin, cos, angle}, parameters);
}

}  // namespace

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, TimeEmbeddingAndLayoutVariantsWithoutRoPE) {
    for (const bool layout : {false, true}) {
        SCOPED_TRACE(layout);
        auto coordinates = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        auto positions = std::make_shared<ov::op::v0::Convert>(coordinates, ov::element::f16);
        auto coefficient = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {31415.927734f});
        auto compressed = std::make_shared<ov::op::v0::Convert>(coefficient, ov::element::f16);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(positions, compressed);
        ov::Output<ov::Node> angle = multiply;
        if (layout) {
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
            angle = std::make_shared<ov::op::v1::Reshape>(angle, shape, false);
        }
        auto model = make_pair(angle, {coordinates});
        auto table = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{model->get_results()[0]->input_value(0), model->get_results()[1]->input_value(0)},
            -1);
        model->add_results({std::make_shared<ov::op::v0::Result>(table)});
        run_discovery(model);
        expect_fp32_pair(model);
        EXPECT_EQ(model->get_results()[2]->input_value(0), angle);
        EXPECT_EQ(angle.get_element_type(), ov::element::f16);
        EXPECT_EQ(table->get_output_element_type(0), ov::element::f16);
    }
}

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, IneligiblePairsAndFP32PathsStayUnchanged) {
    for (const std::string mode : {"different", "different_port", "missing", "ambiguous", "ambiguous_sin",
                                   "no_arithmetic", "fp32", "fp32_convert", "disabled", "cos_disabled", "bf16",
                                   "fixed_gemm"}) {
        SCOPED_TRACE(mode);
        auto type = mode == "fp32" || mode == "fp32_convert" ? ov::element::f32 : ov::element::f16;
        auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2});
        auto constant = ov::op::v0::Constant::create(type, ov::Shape{2}, {2});
        ov::Output<ov::Node> angle = std::make_shared<ov::op::v1::Add>(input, constant);
        if (mode == "no_arithmetic") {
            angle = input;
        } else if (mode == "fp32_convert") {
            angle = std::make_shared<ov::op::v0::Convert>(angle, ov::element::f32);
        } else if (mode == "bf16") {
            angle = std::make_shared<ov::op::v0::Convert>(angle, ov::element::bf16);
            angle = std::make_shared<ov::op::v0::Convert>(angle, ov::element::f16);
        } else if (mode == "fixed_gemm") {
            angle = std::make_shared<ov::intel_gpu::op::Gemm>(
                input, constant, std::vector<int64_t>{0}, std::vector<int64_t>{0}, std::vector<int64_t>{},
                ov::element::f16);
        }
        auto model = make_pair(angle, {input});
        if (mode == "different") {
            model->get_results()[1]->input_value(0).get_node()->input(0).replace_source_output(input);
        } else if (mode == "ambiguous") {
            model->add_results({std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Cos>(angle))});
        } else if (mode == "ambiguous_sin") {
            model->add_results({std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Sin>(angle))});
        } else if (mode == "missing") {
            model = std::make_shared<ov::Model>(
                ov::OutputVector{std::make_shared<ov::op::v0::Sin>(angle)}, ov::ParameterVector{input});
        } else if (mode == "different_port") {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto split = std::make_shared<ov::op::v1::Split>(angle, axis, 2);
            model->get_results()[0]->input_value(0).get_node()->input(0).replace_source_output(split->output(0));
            model->get_results()[1]->input_value(0).get_node()->input(0).replace_source_output(split->output(1));
        }
        const auto nodes = model->get_ordered_ops();
        if (mode == "cos_disabled") {
            ov::pass::Manager manager;
            manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>();
            manager.get_pass_config()->set_callback<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>(
                [](const std::shared_ptr<const ov::Node>& node) { return ov::is_type<ov::op::v0::Cos>(node); });
            manager.run_passes(model);
        } else {
            run_discovery(model, mode == "disabled");
        }
        EXPECT_EQ(nodes, model->get_ordered_ops());
        EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    }
}

// Synthetic compatibility probes, not certification of the corresponding real models.
TEST(IncreasePositionIdsLTXVideoCrossModelPatternTest, ReconstructsPreviouslyAnalyzedAngleStructures) {
    for (const std::string pattern : {"generic_rope", "qwen25", "qwen3", "gpt_oss", "direct", "gemm"}) {
        SCOPED_TRACE(pattern);
        auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2, 2});
        auto converted = std::make_shared<ov::op::v0::Convert>(positions, ov::element::f16);
        auto frequency = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto compressed = std::make_shared<ov::op::v0::Convert>(frequency, ov::element::f16);
        ov::Output<ov::Node> angle;
        if (pattern == "gemm") {
            angle = std::make_shared<ov::intel_gpu::op::Gemm>(converted, compressed,
                                                           std::vector<int64_t>{0, 1},
                                                           std::vector<int64_t>{0, 1},
                                                           std::vector<int64_t>{0, 1});
        } else {
            angle = std::make_shared<ov::op::v0::MatMul>(converted, compressed);
        }
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        if (pattern == "qwen25" || pattern == "gpt_oss") {
            angle = std::make_shared<ov::op::v1::Transpose>(angle, order);
        }
        if (pattern == "qwen3") {
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
            angle = std::make_shared<ov::op::v1::Reshape>(angle, shape, false);
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
            auto gather = std::make_shared<ov::op::v8::Gather>(angle, indices, axis);
            auto scatter_indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2, 1}, {0, 1});
            angle = std::make_shared<ov::op::v3::ScatterNDUpdate>(angle, scatter_indices, gather);
        }
        if (pattern == "generic_rope" || pattern == "qwen25" || pattern == "qwen3") {
            angle = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{angle, angle}, -1);
        }
        auto model = make_pair(angle, {positions});
        if (pattern == "qwen25" || pattern == "qwen3") {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
            auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
            for (size_t i = 0; i < 2; ++i) {
                auto table = std::make_shared<ov::op::v8::Gather>(
                    model->get_results()[i]->input_value(0), indices, axis);
                model->add_results({std::make_shared<ov::op::v0::Result>(table)});
            }
        }
        if (pattern == "gpt_oss") {
            auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.5f});
            for (size_t i = 0; i < 2; ++i) {
                auto result = model->get_results()[i];
                auto scaled = std::make_shared<ov::op::v1::Multiply>(result->input_value(0), scale);
                model->add_results({std::make_shared<ov::op::v0::Result>(scaled)});
            }
        }
        run_discovery(model);
        expect_fp32_pair(model);
        EXPECT_EQ(model->get_results()[2]->input_value(0), angle);
        if (pattern == "gpt_oss") {
            EXPECT_EQ(model->get_results()[3]->get_input_element_type(0), ov::element::f16);
            EXPECT_EQ(model->get_results()[4]->get_input_element_type(0), ov::element::f16);
        }
    }
}

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, EnclosingPassPreservesTimeEmbeddingAndIsIdempotent) {
    for (const auto type : {ov::element::f16, ov::element::f32}) {
        auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        auto converted = std::make_shared<ov::op::v0::Convert>(positions, type);
        auto frequency = ov::op::v0::Constant::create(type, ov::Shape{2}, {2});
        auto angle = std::make_shared<ov::op::v1::Multiply>(converted, frequency);
        auto model = make_pair(angle, {positions});
        const auto original = model->get_ordered_ops();
        ov::pass::Manager manager;
        manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecision>();
        manager.run_passes(model);
        if (type == ov::element::f16) {
            expect_fp32_pair(model);
        } else {
            EXPECT_EQ(original, model->get_ordered_ops());
        }
        const auto nodes = model->get_ordered_ops();
        manager.run_passes(model);
        EXPECT_EQ(nodes, model->get_ordered_ops());
    }
}

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, ReusesFP16ParameterBoundaryOnRepeatedApplication) {
    for (const bool enclosing : {false, true}) {
        SCOPED_TRACE(enclosing);
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2});
        auto coefficient = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2}, {2});
        auto angle = std::make_shared<ov::op::v1::Add>(input, coefficient);
        auto model = make_pair(angle, {input});
        ov::pass::Manager manager;
        if (enclosing) {
            manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecision>();
        } else {
            manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>();
        }
        manager.run_passes(model);
        expect_fp32_pair(model);
        const auto nodes = model->get_ordered_ops();
        manager.run_passes(model);
        EXPECT_EQ(nodes, model->get_ordered_ops());
        EXPECT_EQ(input->get_output_element_type(0), ov::element::f16);
        EXPECT_EQ(model->get_results()[2]->input_value(0), angle->output(0));
    }
}
