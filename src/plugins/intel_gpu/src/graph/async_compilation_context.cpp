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
    AsyncCompilationContext(InferenceEngine::CPUStreamsExecutor::Ptr task_executor,
                                engine& engine, const ExecutionConfig config,
                                uint32_t prog_id) : _task_executor(task_executor) {
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, config, prog_id,
                        nullptr, kernel_selector::KernelBase::get_db().get_batch_header_str());
    }

    void push_task(size_t key, Task&& task) override {
        std::lock_guard<std::mutex> lock(_async_mutex);
        if (_task_keys.find(key) == _task_keys.end()) {
            _task_keys.insert(key);
            _task_executor->run(task);
        }
    }

    // impl cache => program
    // async compilation => program
    // kernel cache compile
    // key only added => key is removed when impl cache is poped
    void push_task(cldnn::network& network, const cldnn::program_node *node,
                            kernel_impl_params params, size_t impl_key) override {
        std::lock_guard<std::mutex> lock(_async_mutex);
        if (_task_keys.find(impl_key) == _task_keys.end()) {
            _task_keys.insert(impl_key);
            _task_executor->run([this, &network, node, params, impl_key](){
                if (_stop_compilation)
                    return;

                auto& cache = network.get_implementations_cache();
                {
                    bool found_key = false;
                    {
                        std::lock_guard<std::mutex> lock(network.get_impl_cache_mutex());
                        // Check existense in the cache one more time as several iterations of model execution could happens and multiple compilation
                        // tasks created for same shapes
                        found_key = cache.has(impl_key);
                    }
                    if (found_key) {
                        remove_keys({impl_key});
                        return;
                    }
                }

                if (_stop_compilation)
                    return;
                _kernels_cache->reset();
                try {
                    if (node == nullptr) {
                        std::cout << "the node is nullptr .... " << std::endl;
                    }
                    auto impl = node->type()->choose_impl(*node, params);
                    auto kernels = _kernels_cache->compile_threadsafe(impl->get_kernels_source());
                    impl->set_kernels(kernels);

                    {
                        std::lock_guard<std::mutex> lock(network.get_impl_cache_mutex());
                        cache.add(impl_key, impl->clone());
                    }
                    remove_keys({impl_key});
                } catch (std::exception& ex) {
                    std::cout << "Exception for building impl : " << ex.what() << std::endl;
                    throw ex;
                }
            });
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
        _stop_compilation = true;
    }

private:
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    std::unique_ptr<kernels_cache> _kernels_cache;
    std::mutex _async_mutex;
    std::unordered_set<size_t> _task_keys;
    std::atomic_bool _stop_compilation{false};
};

std::unique_ptr<IAsyncCompilationContext> IAsyncCompilationContext::create(InferenceEngine::CPUStreamsExecutor::Ptr task_executor,
                                                            engine& engine, const ExecutionConfig config, uint32_t prog_id) {
    return cldnn::make_unique<AsyncCompilationContext>(task_executor, engine, config, prog_id);
}

}  // namespace cldnn