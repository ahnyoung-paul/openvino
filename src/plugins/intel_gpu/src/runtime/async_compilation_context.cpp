// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/async_compilation_context.hpp"
namespace cldnn {

void AsyncCompilationContext::push_task(size_t key, Task&& task) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_task_keys.find(key) == _task_keys.end()) {
        _task_keys.insert(key);
        run(task);
    }
}

void AsyncCompilationContext::remove_keys(std::vector<size_t> keys) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto key : keys) {
        _task_keys.erase(key);
    }
}

std::unique_ptr<AsyncCompilationContext> AsyncCompilationContext::create(cldnn::program& program) {
    auto task_executor_config = program.get_task_executor_config(program.get_config(), "Task executor for async compilation");
    return cldnn::make_unique<AsyncCompilationContext>(task_executor_config);
}

}  // namespace cldnn