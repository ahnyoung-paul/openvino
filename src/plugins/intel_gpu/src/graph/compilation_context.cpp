// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compilation_context.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "kernel_selector/kernel_base.h"

namespace cldnn {
class CompilationTaskQueue {
    using CompilationTaskData = std::pair<size_t, ICompilationContext::Task>;

public:
    void push_task(size_t task_key, ICompilationContext::Task&& task) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue_keymap.find(task_key) == _queue_keymap.end()) {
            auto insert_it = _queue.insert(_queue.end(), {task_key, task});
            _queue_keymap.insert({task_key, insert_it});
        }
    }

    std::vector<CompilationTaskData> pop_tasks() {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<CompilationTaskData> tasks;
        size_t idx = 0;
        size_t max_num_compiled = 30;
        while (!_queue.empty() && idx < max_num_compiled) {
            tasks.push_back(_queue.front());
            _queue.pop_front();
            idx++;
        }
        return tasks;
    }

    void erase_task_key(std::vector<size_t>& removed_keys) {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto key : removed_keys) {
            _queue_keymap.erase(key);
        }
    }

private:
    std::deque<CompilationTaskData> _queue;
    std::unordered_map<size_t, std::deque<CompilationTaskData>::iterator> _queue_keymap;
    std::mutex _mutex;
};

class CompilationContext : public ICompilationContext {
public:
    CompilationContext(cldnn::network& network) {
        cldnn::program::ptr program = network.get_program();
        cldnn::engine& engine = program->get_engine();
        const ExecutionConfig& config = program->get_config();
        size_t program_id = program->get_id();

        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, config, program_id, nullptr, kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this, &network](){
            auto& cache = network.get_implementations_cache();
            while (!_stop_compilation) {
                CompilationContext::Task task;
                auto task_data_list = _queue.pop_tasks();
                if (!task_data_list.empty()) {
                    std::vector<size_t> impl_key_list;
                    std::vector<std::pair<size_t, std::unique_ptr<cldnn::primitive_impl>>> impl_list;
                    for (auto& task_data : task_data_list) {
                        auto impl_key = task_data.first;
                        auto& task = task_data.second;
                        auto impl = task();
                        if (impl != nullptr) {
                            auto kernel_ids = _kernels_cache->add_kernels_source(impl->get_kernels_source());
                            impl->set_kernel_ids(kernel_ids);
                            impl_list.push_back(std::make_pair(impl_key, std::move(impl)));
                        }
                        impl_key_list.push_back(impl_key);
                    }

                    _kernels_cache->build_all();

                    for (auto& impl_data : impl_list) {
                        auto impl_key = impl_data.first;
                        auto impl = std::move(impl_data.second);
                        impl->init_kernels(*_kernels_cache);
                        cache.add(impl_key, impl->clone());
                    }

                    _kernels_cache->reset();
                    _queue.erase_task_key(impl_key_list);

                } else {
                    std::chrono::milliseconds ms{1};
                    std::this_thread::sleep_for(ms);
                }
            }
        });
    }

    void push_task(size_t key, ICompilationContext::Task&& task) override {
        _queue.push_task(key, std::move(task));
    }

    void cancel() noexcept override {
        _stop_compilation = true;
        if (_worker.joinable())
            _worker.join();
    }

    ~CompilationContext() noexcept { cancel(); }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    std::thread _worker;
    std::atomic_bool _stop_compilation{false};

    CompilationTaskQueue _queue;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::network& network) {
    return cldnn::make_unique<CompilationContext>(network);
}

}  // namespace cldnn
