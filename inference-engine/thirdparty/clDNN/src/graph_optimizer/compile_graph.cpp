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
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

using namespace cldnn;

// struct Observer: public tbb::task_scheduler_observer {
//     CpuSet              _mask;
//     int                 _ncpus                  = 0;
//     int                 _threadBindingStep      = 0;
//     int                 _offset                 = 0;
//     tbb::task_arena&    _my_arena;
//     Observer(tbb::task_arena&    arena,
//                 CpuSet              mask,
//                 int                 ncpus,
//                 const int           streamId,
//                 const int           threadsPerStream,
//                 const int           threadBindingStep,
//                 const int           threadBindingOffset) :
//         tbb::task_scheduler_observer(arena),
//         _my_arena(arena),
//         _mask{std::move(mask)},
//         _ncpus(ncpus),
//         _threadBindingStep(threadBindingStep),
//         _offset{streamId * threadsPerStream  + threadBindingOffset} {
//     }
//     void on_scheduler_entry(bool) override {
//         PinThreadToVacantCore(_offset + tbb::this_task_arena::current_thread_index(), _threadBindingStep, _ncpus, _mask);
//     }
//     void on_scheduler_exit(bool) override {
//         PinCurrentThreadByMask(_ncpus, _mask);
//     }
//     void observe(bool state = true) {
//         if (state) {
//             _my_arena.initialize();
//         }
//         tbb::task_scheduler_observer::observe(state);
//     }
//     ~Observer() override = default;
// };

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
    auto arena = std::unique_ptr<tbb::task_arena>(new tbb::task_arena());
    arena->initialize(n_threads);
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
