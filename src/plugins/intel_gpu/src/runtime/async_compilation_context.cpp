// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/async_compilation_context.hpp"
#include "kernels_cache.hpp"

namespace cldnn {
class AsyncCompilationContext : public IAsyncCompilationContext {
public:
    AsyncCompilationContext(InferenceEngine::CPUStreamsExecutor::Ptr task_executor,
                                engine& engine, const ExecutionConfig& config,
                                uint32_t prog_id) : _task_executor(task_executor) {
        _cache = cldnn::make_unique<kernels_cache>(engine, config, prog_id, nullptr);
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
        _task_keys.clear();
    }

private:
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    std::unique_ptr<kernels_cache> _cache;
    std::mutex _async_mutex;
    std::unordered_set<size_t> _task_keys;
};

std::unique_ptr<IAsyncCompilationContext> IAsyncCompilationContext::create(InferenceEngine::CPUStreamsExecutor::Ptr task_executor,
                                                            engine& engine, const ExecutionConfig& config, uint32_t prog_id) {
    return cldnn::make_unique<AsyncCompilationContext>(task_executor, engine, config, prog_id);
}

}  // namespace cldnn