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

    std::vector<CompilationTaskData> pop_front_tasks(size_t max_num_popped_tasks = 1) {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<CompilationTaskData> tasks;
        for (size_t idx = 0; (idx < max_num_popped_tasks) && (!_queue.empty()); idx++) {
            auto& front_task = _queue.front();
            _queue.pop_front();
            tasks.push_back(std::move(front_task));
        }
        return tasks;
    }

    void erase_task_keys(std::vector<size_t> removed_keys) {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto rm_key : removed_keys) {
            _queue_keymap.erase(rm_key);
        }
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

private:
    std::deque<CompilationTaskData> _queue;
    std::unordered_map<size_t, std::deque<CompilationTaskData>::iterator> _queue_keymap;
    std::mutex _mutex;
};

class CompilationContext : public ICompilationContext {
public:
    CompilationContext(cldnn::engine& engine, const ExecutionConfig& config,
                            size_t program_id, InferenceEngine::CPUStreamsExecutor::Ptr task_executor) {
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, config, program_id, task_executor,
                                                kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            const size_t max_num_compiled_tasks = 8;
            while (!_stop_compilation) {
                if (!_queue.empty()) {
                    auto working_task_key_pairs = _queue.pop_front_tasks(max_num_compiled_tasks);
                    if (working_task_key_pairs.size() > 0) {
                        std::vector<size_t> compiled_keys;
                        std::vector<ImplKeyPairType> compiled_impl_key_sets;
                        // Add kernels sources
                        for (auto& key_task_pair : working_task_key_pairs) {
                            auto key = key_task_pair.first;
                            compiled_keys.push_back(key_task_pair.first);
                            auto& task = key_task_pair.second;
                            if (auto impl = task()) {
                                auto kernel_ids = _kernels_cache->add_kernels_source(impl->get_kernels_source());
                                impl->set_kernel_ids(kernel_ids);
                                compiled_impl_key_sets.push_back({key, std::move(impl)});
                            }
                        }

                        if (compiled_impl_key_sets.size() > 0) {
                            // Build all
                            _kernels_cache->set_single_kernel_per_batch(true);
                            _kernels_cache->build_all();

                            // Init kernels
                            for (auto& key_impl_pair : compiled_impl_key_sets) {
                                auto& impl = *key_impl_pair.second;
                                impl.init_kernels(*_kernels_cache);
                            }
                            _store_func(compiled_impl_key_sets);
                        }

                        // reset and remove tasks from queue
                        _kernels_cache->reset();
                        _queue.erase_task_keys(compiled_keys);
                    }
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

    void SetStoreFunc(Store&& store) override {
        _store_func = store;
    }

    ~CompilationContext() noexcept { cancel(); }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    std::thread _worker;
    std::atomic_bool _stop_compilation{false};

    ICompilationContext::Store _store_func;
    CompilationTaskQueue _queue;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, const ExecutionConfig& config,
                                                size_t program_id, InferenceEngine::CPUStreamsExecutor::Ptr task_executor) {
    return cldnn::make_unique<CompilationContext>(engine, config, program_id, task_executor);
}

}  // namespace cldnn
