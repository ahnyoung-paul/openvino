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

inline std::string convert_fixed_length_num(const size_t num, const size_t fixed_length) {
    std::ostringstream out_strstr;
    out_strstr << std::setfill('0') << std::setw(fixed_length) << num;
    return out_strstr.str();
}

void compile_graph::run(program_impl& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::CompileGraph");
    const auto max_count_digit = size_t(log10(p.get_processing_order().size()) + 1);
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<data>()) {
            node->get_output_layout();
        }
    }

    const auto n_threads = p.get_engine().configuration().n_threads;
    auto arena = std::unique_ptr<tbb::task_arena>(new tbb::task_arena());
    arena->initialize(n_threads);
    arena->execute([this, &p, &max_count_digit] {
        auto& proc_order = p.get_processing_order();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, proc_order.size()), [&proc_order, &p, &max_count_digit](const tbb::blocked_range<size_t>& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                auto& node = *(std::next(proc_order.begin(), i));
                node->set_unique_id(convert_fixed_length_num(i, max_count_digit));
                if (!node->is_type<data>()) {
                    if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                        node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
                    }
                }
            }
        });
    });
    arena.reset();
#else
    size_t order_idx = 0;
    for (auto& node : p.get_processing_order()) {
        node->set_unique_id(convert_fixed_length_num(order_idx++, max_count_digit));
        if (!node->is_type<data>()) {
            node->get_output_layout();
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
            }
        }
    }
#endif
}
