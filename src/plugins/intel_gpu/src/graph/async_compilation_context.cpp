// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "async_compilation_context.hpp"
#include "kernels_cache.hpp"
#include "kernel_base.h"
#include "kernel_selector_helper.h"
#include "program_node.h"
#include "primitive_type_base.h"

namespace cldnn {
class AsyncCompilationContext : public IAsyncCompilationContext {
public:
    AsyncCompilationContext(InferenceEngine::CPUStreamsExecutor::Ptr task_executor) : _task_executor(task_executor) { }

    void push_task(size_t key, Task&& task) override {
        std::lock_guard<std::mutex> lock(_async_mutex);
        if (_task_keys.find(key) == _task_keys.end()) {
            _task_keys.insert(key);
            _task_executor->run(task);
        }
    }

    void remove_keys(std::vector<size_t>&& keys) override {
        std::lock_guard<std::mutex> lock(_async_mutex);
        for (auto key : keys) {
            _task_keys.erase(key);
        }
    }

    ~AsyncCompilationContext() noexcept {
        std::cout << "START AsyncCompilationContext is destroyed ..." << std::endl;
        cancel();
        _task_executor = nullptr;
        _task_keys.clear();
        std::cout << "END.. AsyncCompilationContext is destroyed ..." << std::endl;
    }

    bool is_stopped() override {
        // std::lock_guard<std::mutex> lock(_async_mutex);
        return _is_stopped;
    }

    void cancel() override {
        if (_is_stopped)
            return;

        {
            // std::lock_guard<std::mutex> lock(_async_mutex);
            _is_stopped = true;
        }
        _task_executor->Execute({[this](){ std::cout << "it is called " << std::endl; }});
    }

private:
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    std::mutex _async_mutex;
    std::unordered_set<size_t> _task_keys;
    bool _is_stopped = false;
};

std::unique_ptr<IAsyncCompilationContext> IAsyncCompilationContext::create(InferenceEngine::CPUStreamsExecutor::Ptr task_executor) {
    return cldnn::make_unique<AsyncCompilationContext>(task_executor);
}

}  // namespace cldnn