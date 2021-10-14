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
    auto compile_graph_confilg = p.get_engine().configuration().stream_exec_config;
    compile_graph_confilg._name = "CLDNNPlugin executor for compile graph on load network";
    compile_graph_confilg._threadsPerStream = p.get_engine().get_device_info().supports_immad ? 1: compile_graph_confilg._threadsPerStream;

    auto task_executor = std::unique_ptr<CPUStreamsExecutor>(new CPUStreamsExecutor(compile_graph_confilg));

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
