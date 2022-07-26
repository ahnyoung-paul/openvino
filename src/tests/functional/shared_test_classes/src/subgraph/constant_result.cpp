// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace SubgraphTestsDefinitions {

std::ostream& operator<<(std::ostream &os, ConstantSubgraphType type) {
    switch (type) {
        case ConstantSubgraphType::SINGLE_COMPONENT:
            os << "SINGLE_COMPONENT";
            break;
        case ConstantSubgraphType::SEVERAL_COMPONENT:
            os << "SEVERAL_COMPONENT";
            break;
        default:
             os << "UNSUPPORTED_CONST_SUBGRAPH_TYPE";
    }
    return os;
}

std::string ConstantResultSubgraphTest::getTestCaseName(const testing::TestParamInfo<constResultParams>& obj) {
    ConstantSubgraphType type;
    SizeVector IS;
    Precision inputPrecision;
    std::string targetDevice;

    std::tie(type, IS, inputPrecision, targetDevice) = obj.param;
    std::ostringstream result;
    result << "SubgraphType=" << type << "_";
    result << "IS=" << CommonTestUtils::vec2str(IS) << "_";
    result << "inPrc=" << inputPrecision << "_";
    result << "Device=" << targetDevice;
    return result.str();
}

void ConstantResultSubgraphTest::createGraph(const ConstantSubgraphType& type, const SizeVector &inputShape, const Precision &inputPrecision) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);

    ParameterVector params;
    ResultVector results;
    switch (type) {
        case ConstantSubgraphType::SINGLE_COMPONENT: {
            auto input = builder::makeConstant<float>(ngPrc, inputShape, {}, true);
            results.push_back(std::make_shared<opset3::Result>(input));
            break;
        }
        case ConstantSubgraphType::SEVERAL_COMPONENT: {
            auto input1 = builder::makeConstant<float>(ngPrc, inputShape, {}, true);
            results.push_back(std::make_shared<opset3::Result>(input1));
            auto input2 = builder::makeConstant<float>(ngPrc, inputShape, {}, true);
            results.push_back(std::make_shared<opset3::Result>(input2));
            break;
        }
        default: {
            throw std::runtime_error("Unsupported constant graph type");
        }
    }
    function = std::make_shared<Function>(results, params, "ConstResult");
}

void ConstantResultSubgraphTest::SetUp() {
    ConstantSubgraphType type;
    SizeVector IS;
    Precision inputPrecision;
    std::tie(type, IS, inputPrecision, targetDevice) = this->GetParam();

    createGraph(type, IS, inputPrecision);
}

#define GTEST_COUT std::cout << "[          ] [ INFO ] "

std::string CLKernelTest::getTestCaseName(testing::TestParamInfo<KernelParams> obj) {
    auto param  = obj.param;
    auto param01 = std::get<0>(param);
    auto param02 = std::get<1>(param);
    return "CLKernelTest_" + std::to_string(param01) + "_" + std::to_string(param02);
}

void CLKernelTest::SetUp() {
    targetDevice = "GPU";
    GTEST_COUT << "CLKernelTest::SetUp" << std::endl;
    // nGraphFunctionWithName funcPair;
    // std::tie(funcPair, m_precision, m_batchSize, targetDevice) = GetParam();
    // auto fGen = std::get<0>(funcPair);
    // m_functionName = std::get<1>(funcPair);
    // try {
    //     function = fGen(m_precision, m_batchSize);
    // } catch (...) {
    //     GTEST_SKIP();
    // }

    // std::stringstream ss;
    // auto hash = std::hash<std::string>()(GetTestName());
    // ss << "testCache_" << std::to_string(hash) << "_" << std::this_thread::get_id() << "_" << GetTimestamp();
    // for (auto& iter : configuration) {
    //     ss << "_" << iter.first << "_" << iter.second << "_";
    // }
    // m_cacheFolderName = ss.str();
    // core->SetConfig({{CONFIG_KEY(CACHE_DIR), {}}});
}

void CLKernelTest::TearDown() {
    GTEST_COUT << "CLKernelTest::TearDown" << std::endl;
    // CommonTestUtils::removeFilesWithExt(m_cacheFolderName, "blob");
    // std::remove(m_cacheFolderName.c_str());
    // core->SetConfig({{CONFIG_KEY(CACHE_DIR), {}}});
}

void CLKernelTest::Run() {
    GTEST_COUT << "CLKernelTest::Run" << std::endl;
    std::cout << "[Step01] Initialize test" << std::endl;
    core = PluginCache::get().ie("GPU");
    try {
#if defined(_WIN32) || defined(__WIN32__)
        std::string xml_path = "C:\\Users\\avagen12-dg1\\paul\\models\\mstpn_slb\\mstpn_slb_i8.xml";
        std::string bin_path = "C:\\Users\\avagen12-dg1\\paul\\models\\mstpn_slb\\mstpn_slb_i8.bin";
#else
        std::string xml_path = "/home/ahnyoung/cldnn/model/mstpn_slb/mstpn_slb_i8.xml";
        std::string bin_path = "/home/ahnyoung/cldnn/model/mstpn_slb/mstpn_slb_i8.bin";
#endif
        std::cout << "[Step02] Read network " << std::endl;
        auto net = core->ReadNetwork(xml_path, bin_path);
        std::cout << "[Step03] Load network " << std::endl;
        auto exec_network = core->LoadNetwork(net, targetDevice, configuration);
        std::cout << "[Step04] Complete to load " << std::endl;
    } catch (const Exception &ex) {
        GTEST_COUT << "Can't loadNetwork without cache for cl kernel test " << std::endl;
        GTEST_COUT << "Exception [" << ex.what() << "]" << std::endl;
        GTEST_FAIL();
    } catch (...) {
        GTEST_COUT << "Can't loadNetwork without cache for cl kernel test " << std::endl;
        GTEST_FAIL();
    }
    // SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // auto compareOutputs = [&](const std::vector<InferenceEngine::Blob::Ptr>& expected,
    //                           const std::vector<InferenceEngine::Blob::Ptr>& actual) {
    //     ASSERT_EQ(expected.size(), actual.size());
    //     for (size_t i = 0; i < expected.size(); i++) {
    //         const auto& expPtr = expected[i];
    //         const auto& actPtr = actual[i];
    //         ASSERT_NO_THROW(Compare(expPtr, actPtr));
    //     }
    // };
    // if (!function) {
    //     GTEST_COUT << "Can't create function " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    //     GTEST_SKIP();
    // }
    // if ((targetDevice.find("AUTO") == std::string::npos) && !importExportSupported(*core)) {
    //     GTEST_COUT << "Plugin doesn't support import and export - skipping test" << std::endl;
    //     GTEST_SKIP();
    // }
    // cnnNetwork = CNNNetwork{function};
    // ConfigureNetwork();
    // try {
    //     executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    //     GenerateInputs();
    //     Infer();
    // } catch (const Exception &ex) {
    //     GTEST_COUT << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    //     GTEST_COUT << "Exception [" << ex.what() << "]" << std::endl;
    //     GTEST_SKIP();
    // } catch (...) {
    //     GTEST_COUT << "Can't loadNetwork without cache for " << m_functionName << " with precision " << m_precision.get_type_name() << std::endl;
    //     GTEST_SKIP(); // skip caching test if such network is not supported by device at all
    // }
    // auto originalOutputs = GetOutputs();

    // for (int i = 0; i < 2; i++) {
    //     // Step 2: Load with cache. Export or import shall not throw
    //     executableNetwork = {}; // Destroy network object
    //     inferRequest = {};
    //     {
    //         core->SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheFolderName}});
    //         ASSERT_NO_THROW(executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration));
    //         GenerateInputs();
    //         ASSERT_NO_THROW(Infer());
    //     }
    //     // cache is created and reused
    //     ASSERT_EQ(CommonTestUtils::listFilesWithExt(m_cacheFolderName, "blob").size(), 1);
    //     compareOutputs(originalOutputs, GetOutputs());
    // }
}


TEST_P(CLKernelTest, checkCLKernelBuild) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
