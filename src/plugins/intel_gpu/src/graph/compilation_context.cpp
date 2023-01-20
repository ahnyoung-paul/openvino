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
            _max_queue_size = std::max(_max_queue_size, _queue.size());
            _total_quque_inputs++;
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

    void erase_task_key(size_t removed_key) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue_keymap.find(removed_key) != _queue_keymap.end()) {
            _queue_keymap.erase(removed_key);
        }
    }

    std::string summary() {
        std::lock_guard<std::mutex> lock(_mutex);
        {
            std::stringstream ss;
            ss << "async_compilation_queue_size: " << _max_queue_size << std::endl;
            ss << "async_compilation_number_inputs: " << _total_quque_inputs << std::endl;
            ss << "async_compilation_remained: " << _queue.size() << std::endl;
            return ss.str();
        }
    }

    size_t max_size() { return _max_queue_size; }
    size_t total_queue_inputs() { return _total_quque_inputs; }

private:
    std::deque<CompilationTaskData> _queue;
    std::unordered_map<size_t, std::deque<CompilationTaskData>::iterator> _queue_keymap;
    std::mutex _mutex;
    size_t _max_queue_size = 0;
    size_t _total_quque_inputs = 0;
};

class CompilationContext : public ICompilationContext {
public:
    CompilationContext(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, config, program_id, nullptr, kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            while (!_stop_compilation) {
                CompilationContext::Task task;
                size_t task_key;
                bool success = _queue.pop_front_task(task_key, task);
                if (success) {
                    task(*_kernels_cache);
                    _queue.erase_task_key(task_key);
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

    std::string summary() override {
        return _queue.summary();
    }

    ~CompilationContext() noexcept { cancel(); }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    std::thread _worker;
    std::atomic_bool _stop_compilation{false};

    CompilationTaskQueue _queue;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, config, program_id);
}

}  // namespace cldnn
