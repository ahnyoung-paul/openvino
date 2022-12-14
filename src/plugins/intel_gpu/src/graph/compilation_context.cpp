// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compilation_context.hpp"
#include "kernel_selector/kernel_base.h"

namespace cldnn {

class CompilationContext : public ICompilationContext {
public:
    using data_type = std::pair<size_t, ICompilationContext::Task>;
    using data_list_type = std::list<data_type>;
    using data_list_iter = typename data_list_type::iterator;

    CompilationContext(cldnn::engine& engine, size_t program_id) {
        max_size = 0;
        cache_hit = 0;
        cache_miss = 0;
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, program_id, kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            while (!_stop_compilation) {
                CompilationContext::Task task;
                bool success = try_pop_task(task);
                if (success) {
                    if (task(*_kernels_cache)) {
                        cache_hit++;
                    } else {
                        cache_miss++;
                    }
                } else {
                    std::chrono::milliseconds ms{1};
                    std::this_thread::sleep_for(ms);
                }
            }
        });
    }

    bool push_task(size_t key, ICompilationContext::Task&& task) override {
        std::lock_guard<std::mutex> lock(_mutex);
        auto iter = _key_map.find(key);
        if (iter == _key_map.end()) {   // empty
            auto insert_it = _data_list.insert(_data_list.end(), {key, task});
            _key_map.insert({key, insert_it});
            max_size = std::max(max_size, _key_map.size());
            return true;
        } else {
            _data_list.splice(_data_list.begin(), _data_list, iter->second);
        }
        return false;
    }

    bool try_pop_task(ICompilationContext::Task& task) override {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_data_list.empty()) {
            auto front = _data_list.front();
            task = front.second;
            _key_map.erase(front.first);
            _data_list.pop_front();
            return true;
        }
        return false;
    }

    void cancel() noexcept override {
        _stop_compilation = true;
        if (_worker.joinable())
            _worker.join();
    }

    ~CompilationContext() noexcept {
        cancel();
        std::cout << "CompilationContext max_size  : " << max_size << std::endl;
        std::cout << "CompilationContext cache hit : " << cache_hit << " / " << (cache_hit + cache_miss) << std::endl;
    }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    std::thread _worker;
    std::atomic_bool _stop_compilation{false};

    data_list_type _data_list;
    std::unordered_map<size_t, data_list_iter> _key_map;
    std::mutex _mutex;
    size_t pid;
    size_t max_size;
    size_t cache_hit;
    size_t cache_miss;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, program_id);
}

}  // namespace cldnn
