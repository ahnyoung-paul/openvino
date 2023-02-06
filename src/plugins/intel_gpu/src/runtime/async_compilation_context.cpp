// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/async_compilation_context.hpp"

namespace cldnn {
class AsyncCompilationContext : public IAsyncCompilationContext {
public:
    AsyncCompilationContext(InferenceEngine::CPUStreamsExecutor::Config config) {
        _task_executor = cldnn::make_unique<InferenceEngine::CPUStreamsExecutor>(config);
    }

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
        _task_executor.release();
        _task_keys.clear();
    }

private:
    std::unique_ptr<InferenceEngine::CPUStreamsExecutor> _task_executor;
    std::mutex _async_mutex;
    std::unordered_set<size_t> _task_keys;
};

std::unique_ptr<IAsyncCompilationContext> IAsyncCompilationContext::create(InferenceEngine::CPUStreamsExecutor::Config config) {
    return cldnn::make_unique<AsyncCompilationContext>(config);
}

}  // namespace cldnn