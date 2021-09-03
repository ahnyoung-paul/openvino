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
#endif

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
    const auto n_threads = p.get_engine().configuration().n_threads;
    const auto binding_type = p.get_engine().configuration().cpu_binding_type;
    const auto core_type = p.get_engine().configuration().cpu_core_type;

    std::cout << "Binding type: " << binding_type << std::endl;
    std::cout << "Core type: " << core_type << std::endl;
    auto task_executor = std::unique_ptr<CPUStreamsExecutor>(new CPUStreamsExecutor(
            IStreamsExecutor::Config{
                "CLDNNPlugin executor for compile graph on load network",   // name
                1,                                                          // Number of streams to set number of stream, load_network has 1 stream
                n_threads,                                                  // threadsPerStream used to set max_concurrency for task_arena
                binding_type,                                               // threadBindingType
                1,                                                          // threadBindingStep used in ThreadBindingType::Cores
                0,                                                          // threadBindingOffset used in ThreadBindingType::Cores
                1,                                                          // Number of threads
                core_type}));                                               // Core type

    std::exception_ptr exception;
    auto compile_graph_func = [&] {
        try {
        auto& proc_order = p.get_processing_order();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, proc_order.size()), [&proc_order, &p](const tbb::blocked_range<size_t>& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                auto& node = *(std::next(proc_order.begin(), i));
                if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                    node->selected_impl = node->type()->choose_impl(*node);
                }
            }
        });
        } catch(...) {
            exception = std::current_exception();
        }
    };
    std::vector<Task> tasks;
    tasks.push_back(compile_graph_func);
    task_executor->runAndWait(tasks);
    task_executor.reset();
#else
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
            node->selected_impl = node->type()->choose_impl(*node);
        }
    }
#endif
}
