// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernels_cache.hpp"
#include "primitive_inst.h"
#include <functional>
#include <memory>

namespace cldnn {

class ICompilationContext {
public:
    using ImplKeyPairType = std::pair<size_t, std::unique_ptr<cldnn::primitive_impl>>;
    using Store = std::function<void(std::vector<ImplKeyPairType>&)>;
    using Task = std::function<std::unique_ptr<cldnn::primitive_impl>()>;
    virtual void push_task(size_t key, Task&& task) = 0;
    virtual void cancel() noexcept = 0;
    virtual ~ICompilationContext() = default;
    virtual void SetStoreFunc(Store&& store) = 0;

    static std::unique_ptr<ICompilationContext> create(cldnn::engine& engine, const ExecutionConfig& config,
                                                    size_t program_id, InferenceEngine::CPUStreamsExecutor::Ptr task_executor);
};

}  // namespace cldnn
