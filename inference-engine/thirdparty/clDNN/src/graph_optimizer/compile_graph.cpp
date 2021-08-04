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
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_arena.h"
#include <iostream>
#include <cmath>
#include <iomanip>

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

using namespace cldnn;

extern "C" {
void __TBB_internal_initialize_system_topology(
    std::size_t groups_num,
    int& numa_nodes_count, int*& numa_indexes_list,
    int& core_types_count, int*& core_types_indexes_list
);
class binding_handler;

binding_handler* __TBB_internal_allocate_binding_handler(int number_of_slots, int numa_id, int core_type_id, int max_threads_per_core);
void __TBB_internal_deallocate_binding_handler(binding_handler* handler_ptr);
void __TBB_internal_apply_affinity(binding_handler* handler_ptr, int slot_num);
void __TBB_internal_restore_affinity(binding_handler* handler_ptr, int slot_num);
int __TBB_internal_get_default_concurrency(int numa_id, int core_type_id, int max_threads_per_core);
}

int get_processors_group_num() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);

    DWORD_PTR pam, sam, m = 1;
    GetProcessAffinityMask(GetCurrentProcess(), &pam, &sam);
    int nproc = 0;
    for (std::size_t i = 0; i < sizeof(DWORD_PTR) * CHAR_BIT; ++i, m <<= 1) {
        if ( pam & m )
            ++nproc;
    }
    if (nproc == static_cast<int>(si.dwNumberOfProcessors)) {
        return GetActiveProcessorGroupCount();
    }
#endif
    return 1;
}

static int  numa_nodes_count = 0;
static int* numa_nodes_indexes = nullptr;

static int  core_types_count = 0;
static int* core_types_indexes = nullptr;

bool is_binding_environment_valid() {
#if defined(_WIN32) && !defined(_WIN64)
    static bool result = [] {
        // For 32-bit Windows applications, process affinity masks can only support up to 32 logical CPUs.
        SYSTEM_INFO si;
        GetNativeSystemInfo(&si);
        if (si.dwNumberOfProcessors > 32) return false;
        return true;
    }();
    return result;
#else
    return true;
#endif /* _WIN32 && !_WIN64 */
}

void initialize_system_topology() {
    static std::once_flag is_topology_initialized;

    std::call_once(is_topology_initialized, [&]{
        if (is_binding_environment_valid()) {
            __TBB_internal_initialize_system_topology(
                get_processors_group_num(),
                numa_nodes_count, numa_nodes_indexes,
                core_types_count, core_types_indexes);
        } else {
            static int dummy_index = tbb::task_arena::automatic;

            numa_nodes_count = 1;
            numa_nodes_indexes = &dummy_index;

            core_types_count = 1;
            core_types_indexes = &dummy_index;
        }
        std::cout << "numa_nodes_count["<< std::to_string(numa_nodes_count) << "] = {";
        for (int i = 0; i < numa_nodes_count; i++) {
            std::cout << std::to_string(numa_nodes_indexes[i]) << ", ";
        }
        std::cout << "}" << std::endl;
        std::cout << "core_types_count["<< std::to_string(core_types_count) << "] = {";
        for (int i = 0; i < core_types_count; i++) {
            std::cout << std::to_string(core_types_indexes[i]) << ", ";
        }
        std::cout << "}" << std::endl;
    });
}

//! Returns the index, aka slot number, of the calling thread in its current arena
inline int current_thread_index() {
    int idx = tbb::task_arena::current_thread_index();
    std::cout << "current_thread_index(" << idx << ")" << std::endl;
    return idx == -1 ? tbb::task_arena::not_initialized : idx;
}

struct Observer: public tbb::task_scheduler_observer {
    tbb::task_arena&    _my_arena;
    binding_handler* my_binding_handler;
    Observer(tbb::task_arena&    arena) :
        tbb::task_scheduler_observer(static_cast<tbb::task_arena&>(arena)),
        _my_arena(arena) {
            initialize_system_topology();
            int num_slots = 12;
            int numa_id = -1;
            int core_type = 0;
            int max_threads_per_core = -1;
            std::cout << "num_slots: " << num_slots << std::endl;
            std::cout << "c.numa_id: " << numa_id << std::endl;
            std::cout << "c.core_type: " << core_type << std::endl;
            std::cout << "c.max_threads_per_core: " << max_threads_per_core << std::endl;
            my_binding_handler = __TBB_internal_allocate_binding_handler(num_slots, numa_id, core_type/*BIG*/, max_threads_per_core);
            if (my_binding_handler == nullptr) {
                std::cout << "Fail to create binder " << std::endl;
            }
    }
    void on_scheduler_entry(bool) override {
        if (my_binding_handler != nullptr)
            __TBB_internal_apply_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
    }
    void on_scheduler_exit(bool) override {
        if (my_binding_handler != nullptr)
            __TBB_internal_restore_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
    }
    void observe(bool state = true) {
        if (state) {
            _my_arena.initialize();
        }
        tbb::task_scheduler_observer::observe(state);
    }
    ~Observer() {
        if (my_binding_handler != nullptr)
            __TBB_internal_deallocate_binding_handler(my_binding_handler);
    }
};

void compile_graph::run(program_impl& p) {
    std::cout << "START compile_graph::run .... " << std::endl;
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
    // auto observer = std::unique_ptr<Observer>(new Observer(*arena));
    // observer->observe(true);
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
    std::cout << "END compile_graph::run .... " << std::endl;
}
