// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <threading/ie_cpu_streams_executor.hpp>

namespace cldnn {

class program;

class AsyncCompilationContext : public InferenceEngine::CPUStreamsExecutor {
public:
    explicit AsyncCompilationContext(const InferenceEngine::CPUStreamsExecutor::Config& config = {}) :
                                                             InferenceEngine::CPUStreamsExecutor(config) {}

    using Task = std::function<void()>;
    void push_task(size_t key, Task&& task);
    void remove_keys(std::vector<size_t> keys);
    ~AsyncCompilationContext() = default;

    static std::unique_ptr<AsyncCompilationContext> create(cldnn::program& program);

private:
    std::mutex _mutex;
    std::unordered_set<size_t> _task_keys;
};

}  // namespace cldnn
