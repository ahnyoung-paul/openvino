// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "program_node.h"
#include "cldnn/runtime/engine.hpp"
#include "runtime/cldnn_itt.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <threading/ie_cpu_streams_executor.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <mutex>
#endif

std::mutex m_test_lock;
using namespace cldnn;
using namespace InferenceEngine;

void compile_graph::run(program& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::CompileGraph");
    size_t order_idx = 0;
    for (auto& node : p.get_processing_order()) {
        node->set_unique_id(std::to_string(order_idx++));
        if (!node->is_type<data>()) {
            node->get_output_layout();
        }
    }

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
    auto compile_graph_config = p.get_engine().configuration().stream_exec_config;
    compile_graph_config._name = "CLDNNPlugin executor for compile graph on load network";
    compile_graph_config._threadsPerStream = p.get_engine().get_device_info().supports_immad ? 1: compile_graph_config._threadsPerStream;
    compile_graph_config._streams = 16;
    compile_graph_config._threadsPerStream = 1;
    // compile_graph_config._threadsPerStream = (compile_graph_config._threadsPerStream) / compile_graph_config._streams;
    auto all_num_threads = compile_graph_config._streams * compile_graph_config._threadsPerStream;

    auto task_executor = std::unique_ptr<CPUStreamsExecutor>(new CPUStreamsExecutor(compile_graph_config));
    auto& proc_order = p.get_processing_order();
    std::vector<Task> tasks;
    tasks.resize(all_num_threads);
    int num_elements = proc_order.size() / all_num_threads;
    for (int j = 0; j < all_num_threads; j++) {
        tasks[j] = [this, &j, &proc_order, &all_num_threads, &num_elements] {
            int end_num = ((j+1)*num_elements < proc_order.size())? ((j+1)*num_elements) : proc_order.size();
            for (int i = j * num_elements; (i < end_num); i++) {
                auto& node = *(std::next(proc_order.begin(), i));
                if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                    node->selected_impl = node->type()->choose_impl(*node);
                }
            }
        };
    }

    auto start = std::chrono::high_resolution_clock::now();
    task_executor->runAndWait(tasks);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    task_executor.reset();

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Run without parallel_for" << std::endl;
    std::cout << "Total num: " << order_idx << std::endl;
    std::cout << "Number of Streams: " << compile_graph_config._streams << std::endl;
    std::cout << "Number of ThreadsPerStream: " << compile_graph_config._threadsPerStream << std::endl;
    std::cout << "compile_graph::run duration: " << (static_cast<double>(duration) / 1000);
    std::cout << "ms" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

#else
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
            node->selected_impl = node->type()->choose_impl(*node);
        }
    }
#endif
}
