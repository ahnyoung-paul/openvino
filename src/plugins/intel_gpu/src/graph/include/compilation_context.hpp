// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernels_cache.hpp"
#include <functional>
#include <memory>

namespace cldnn {

class ICompilationContext {
public:
    /**
     * return True if kernel is compiled
    */
    using Task = std::function<bool(kernels_cache&)>;
    virtual bool push_task(size_t key, Task&& task) = 0;
    virtual bool try_pop_task(ICompilationContext::Task& task) = 0;
    virtual void cancel() noexcept = 0;
    virtual ~ICompilationContext() = default;

    static std::unique_ptr<ICompilationContext> create(cldnn::engine& engine, size_t program_id);
};

}  // namespace cldnn
