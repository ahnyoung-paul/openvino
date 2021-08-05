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
#include "cldnn_custom_arena.h"

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

using namespace cldnn;

void compile_graph::run(program_impl& p) {
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
#if 1
    const auto concurrency = n_threads;
    const auto core_type = custom::info::core_types().back(); /// running on Big cores only
    // const auto core_type = custom::info::core_types().back(); /// running on Little cores only
    auto arena = std::unique_ptr<custom::task_arena>(new custom::task_arena{
                           custom::task_arena::constraints{}.set_core_type(core_type).set_max_concurrency(concurrency)});
#else
    auto arena = std::unique_ptr<tbb::task_arena>(new tbb::task_arena());
    arena->initialize(n_threads);
#endif
    arena->execute([this, &p] {
        auto& proc_order = p.get_processing_order();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, proc_order.size()), [&proc_order, &p](const tbb::blocked_range<size_t>& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                auto& node = *(std::next(proc_order.begin(), i));
                node->set_unique_id(std::to_string(i));
                if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                    node->selected_impl = node->type()->choose_impl(*node);
                }
            }
        });
    });

    arena.reset();
#else
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
            node->selected_impl = node->type()->choose_impl(*node);
        }
    }
#endif
}
