// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "kernel_selector_helper.h"


namespace cldnn {

class network;
class program_node;

class IAsyncCompilationContext {
public:
    using Task = std::function<void()>;
    virtual void push_task(size_t key, Task&& task) = 0;
    virtual void push_task(cldnn::network& network, const cldnn::program_node *node,
                            kernel_impl_params params, size_t impl_key) = 0;
    virtual void remove_keys(std::vector<size_t>&& keys) = 0;
    virtual ~IAsyncCompilationContext() = default;

    static std::unique_ptr<IAsyncCompilationContext> create(InferenceEngine::CPUStreamsExecutor::Ptr task_executor,
                                                            engine& engine, const ExecutionConfig config, uint32_t prog_id);
};

}  // namespace cldnn
