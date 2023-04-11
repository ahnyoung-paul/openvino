// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 3) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <device_name>" << slog::endl;
            return EXIT_FAILURE;
        }

        const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = TSTRING2STRING(argv[1]);
        const std::string device_name = TSTRING2STRING(argv[2]);

        for (size_t i = 0; i < 40; i++) {
            std::cout << i << "-th tests ....." << std::endl;
            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            ov::Core core;

            // -------- Step 2. Read a model --------
            slog::info << "Loading model files: " << model_path << slog::endl;
            std::shared_ptr<ov::Model> model = core.read_model(model_path);
            //printInputAndOutputsInfo(*model);

            OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");

            // -------- Step 3. Set up input
            auto input = model->get_parameters().at(0);
            auto output = model->get_results().at(0);
            ov::element::Type input_type = input->get_element_type();
            ov::Shape input_shape = input->get_shape();

            // just wrap image data by ov::Tensor without allocating of new memory
            ov::Tensor input_tensor = ov::Tensor{input_type, input_shape};
            float *in_ptr = static_cast<float*>(input_tensor.data());
            if (true) {
            std::cout << "Input data: " << std::endl;
            for (size_t i = 0; i < input_tensor.get_size(); i++) {
                in_ptr[i] = 1.0;
            }
            }
            ov::Tensor input_tensor_1 = ov::Tensor{input_type, input_shape};
            float *in_ptr_1 = static_cast<float*>(input_tensor_1.data());
            if (true) {
            std::cout << "Input data: " << std::endl;
            for (size_t i = 0; i < input_tensor_1.get_size(); i++) {
                in_ptr_1[i] = 1.0;
            }
            }
            // -------- Step 5. Loading a model to the device --------
            ov::CompiledModel compiled_model = core.compile_model(model, device_name, ov::hint::performance_mode("THROUGHPUT"));
            //ov::CompiledModel compiled_model_1 = core.compile_model(model, device_name, ov::hint::performance_mode("THROUGHPUT"));
            //auto execGraphInfo = compiled_model.get_runtime_model();
            //ov::serialize(execGraphInfo, "/home/bell/gpu_exe_model.xml", "/home/bell/model.bin");
            // -------- Step 6. Create an infer request --------
            ov::InferRequest infer_request = compiled_model.create_infer_request();
            ov::InferRequest infer_request_1 = compiled_model.create_infer_request();
            // -----------------------------------------------------------------------------------------------------

            // -------- Step 7. Prepare input --------
            infer_request.set_input_tensor(input_tensor);
            infer_request_1.set_input_tensor(input_tensor_1);
            // -------- Step 8. Do inference synchronously --------
            infer_request_1.infer();
            infer_request.infer();
            // -------- Step 9. Process output

            const ov::Tensor& output_tensor = infer_request.get_output_tensor(0);
            std::cout << "out type  is " << output_tensor.get_element_type() << std::endl;
            // Print classification results
            const float *out_ptr = static_cast<float*>(output_tensor.data());
            std::cout << "Output data: " << std::endl;
            std::cout << out_ptr[0] << " ";
            std::cout << std::endl;
            if (out_ptr[0] == 0.f) {
                std::cerr << "output data should not be 0" << std::endl;
                return EXIT_FAILURE;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}