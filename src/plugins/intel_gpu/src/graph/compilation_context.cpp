// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compilation_context.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "kernel_selector/kernel_base.h"

namespace cldnn {

class CompilationContext : public ICompilationContext {
public:
    using compilation_queue_t = InferenceEngine::ThreadSafeQueue<ICompilationContext::Task>;

    CompilationContext(CompilationContext&& other) = default;

    CompilationContext(cldnn::engine& engine, size_t program_id) {
        const size_t max_num_threads = 8;
        for (size_t i = 0; i < max_num_threads; i++) {
            auto thread = std::thread([this, &program_id, &engine](){
                auto m_kernels_cache = cldnn::make_unique<kernels_cache>(engine, program_id, kernel_selector::KernelBase::get_db().get_batch_header_str());
                while (!_stop_compilation) {
                    CompilationContext::Task task;
                    bool success = _queue.try_pop(task);
                    if (success) {
                        task(*m_kernels_cache);
                    } else {
                        std::chrono::milliseconds ms{1};
                        std::this_thread::sleep_for(ms);
                    }
                }
            });
            _workers.push_back(std::move(thread));
        }
    }

    void push_task(ICompilationContext::Task&& task) override {
        _queue.push(task);
    }

    void cancel() noexcept override {
        _stop_compilation = true;
        for (size_t i = 0; i < _workers.size(); i++) {
            if (_workers[i].joinable())
                _workers.at(i).join();
        }
    }

    ~CompilationContext() noexcept { cancel(); }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    compilation_queue_t _queue;
    std::vector<std::thread> _workers;
    std::atomic_bool _stop_compilation{false};
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, program_id);
}

}  // namespace cldnn
