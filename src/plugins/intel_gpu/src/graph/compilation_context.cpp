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

    CompilationContext(CompilationContext&& other) = default;

    CompilationContext(cldnn::engine& engine, size_t program_id) {
        const size_t max_num_threads = 4;
        for (size_t i = 0; i < max_num_threads; i++) {
            auto thread = std::thread([this, &program_id, &engine](){
                auto m_kernels_cache = cldnn::make_unique<kernels_cache>(engine, program_id, kernel_selector::KernelBase::get_db().get_batch_header_str());
                while (!_stop_compilation) {
                    CompilationContext::Task task;
                    bool success = try_pop_task(task);
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

    void cancel() noexcept override {
        _stop_compilation = true;
        for (size_t i = 0; i < _workers.size(); i++) {
            if (_workers[i].joinable())
                _workers.at(i).join();
        }
    }

    bool push_task(size_t key, ICompilationContext::Task&& task) override {
        std::lock_guard<std::mutex> lock(_mutex);
        auto iter = _key_map.find(key);
        if (iter == _key_map.end()) {   // not found
            auto insert_it = _data_list.insert(_data_list.end(), {key, task});
            _key_map.insert({key, insert_it});
            return true;
        } else {    // found
            _data_list.splice(_data_list.begin(), _data_list, iter->second);
        }
        return false;
    }

    bool try_pop_task(ICompilationContext::Task& task) {
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

    ~CompilationContext() noexcept { cancel(); }

private:
    std::vector<std::thread> _workers;
    std::atomic_bool _stop_compilation{false};

    data_list_type _data_list;
    std::unordered_map<size_t, data_list_iter> _key_map;
    std::mutex _mutex;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, program_id);
}

}  // namespace cldnn
