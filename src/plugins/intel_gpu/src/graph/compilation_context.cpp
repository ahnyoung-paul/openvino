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

    bool pop_front_task(size_t& task_key, ICompilationContext::Task& task) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_queue.empty()) {
            auto front = _queue.front();
            task = front.second;
            task_key = front.first;
            _queue.pop_front();
            return true;
        }
        return false;
    }

    void erase_task_keys(std::vector<size_t> removed_keys) {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto rm_key : removed_keys) {
            _queue_keymap.erase(rm_key);
            // if (_queue_keymap.find(removed_key) != _queue_keymap.end()) {
            //     _queue_keymap.erase(removed_key);
            // }
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
    CompilationContext(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, config, program_id, nullptr, kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            const size_t max_num_compiled_tasks = 50;
            while (!_stop_compilation) {
                if (!_queue.empty()) {
                    std::unordered_map<size_t, std::unique_ptr<cldnn::primitive_impl>> impl_key_map;
                    for (size_t idx = 0; idx < max_num_compiled_tasks; idx++) {
                        CompilationContext::Task task;
                        size_t task_key;
                        if (_queue.pop_front_task(task_key, task)) {
                            auto new_impl = task();
                            if (new_impl != nullptr) {
                                impl_key_map.insert({task_key, std::move(new_impl)});
                            }
                        }
                        if (_queue.empty())
                            break;
                    }
                    if (impl_key_map.size() > 0) {
                        std::vector<size_t> working_task_keys;
                        for (auto& v : impl_key_map) {
                            working_task_keys.push_back(v.first);
                        }

                        for (auto working_key : working_task_keys) {
                            auto& working_impl = *impl_key_map[working_key];
                            auto kernel_ids = _kernels_cache->add_kernels_source(working_impl.get_kernels_source());
                            working_impl.set_kernel_ids(kernel_ids);
                        }

                        _kernels_cache->set_single_kernel_per_batch(true);
                        _kernels_cache->build_all();

                        for (auto working_key : working_task_keys) {
                            auto& working_impl = *impl_key_map[working_key];
                            working_impl.init_kernels(*_kernels_cache);
                            _store_func(working_key, working_impl);
                        }

                        _kernels_cache->reset();
                        _queue.erase_task_keys(working_task_keys);
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

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, config, program_id);
}

}  // namespace cldnn
