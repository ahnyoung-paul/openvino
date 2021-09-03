// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/stat.h>

#include <cldnn/cldnn_config.hpp>
#include <gpu/gpu_config.hpp>
#include "cldnn_config.h"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_api.h"
#include "file_utils.h"
#include "cldnn_itt.h"
#include "ie_parallel.hpp"
#include <ie_system_conf.h>
#include <thread>
#include <bitset>

#ifdef _WIN32
# include <direct.h>
#ifdef ENABLE_UNICODE_PATH_SUPPORT
# define mkdir(dir, mode) _wmkdir(dir)
#else
# define mkdir(dir, mode) _mkdir(dir)
#endif  // ENABLE_UNICODE_PATH_SUPPORT
#endif  // _WIN32

using namespace InferenceEngine;

namespace CLDNNPlugin {

static void createDirectory(std::string _path) {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widepath = FileUtils::multiByteCharToWString(_path.c_str());
    const wchar_t* path = widepath.c_str();
#else
    const char* path = _path.c_str();
#endif

    auto err = mkdir(path, 0755);
    if (err != 0 && errno != EEXIST) {
        IE_THROW() << "Couldn't create directory! (err=" << err << "; errno=" << errno << ")";
    }
}

static int getNumberOfCores(const IStreamsExecutor::Config::PreferredCoreType core_type) {
        const auto total_num_phy_cores = getNumberOfCPUCores();
        const auto total_num_big_phy_cores = getNumberOfCPUCores(true);
        const auto total_num_little_phy_cores = total_num_phy_cores - total_num_big_phy_cores;
        const auto total_num_max_threads = static_cast<int>(std::thread::hardware_concurrency());
        int num_cores = total_num_max_threads;

        if (core_type == IStreamsExecutor::Config::BIG) {
            num_cores = total_num_max_threads - total_num_little_phy_cores;
        } else if (core_type == IStreamsExecutor::Config::LITTLE) {
            num_cores = total_num_little_phy_cores;
        }

        // std::cout << "total_num_phy_cores       : " << total_num_phy_cores << std::endl;
        // std::cout << "total_num_big_phy_cores   : " << total_num_big_phy_cores << std::endl;
        // std::cout << "total_num_little_phy_cores: " << total_num_little_phy_cores << std::endl;
        // std::cout << "total_num_max_threads     : " << total_num_max_threads << std::endl;
        // std::cout << "num_cores                 : " << num_cores << std::endl;
    return num_cores;
}

static IStreamsExecutor::Config MakeTBBConfigForLoadNetwork(const IStreamsExecutor::ThreadBindingType binding_type,
                                                    const IStreamsExecutor::Config::PreferredCoreType enforeced_core_type,
                                                    const int n_threads) {
    IStreamsExecutor::Config config;
    config._threadBindingType = binding_type;
    config._threadPreferredCoreType = enforeced_core_type;
    config._streams = 1;
    config._threadsPerStream = std::min(n_threads, static_cast<int>(std::thread::hardware_concurrency()));

    if (binding_type == IStreamsExecutor::HYBRID_AWARE) {
        if (enforeced_core_type == IStreamsExecutor::Config::BIG
            || enforeced_core_type == IStreamsExecutor::Config::LITTLE) {
                if (getAvailableCoresTypes().size() > 1) { /*Hybrid CPU*/
                    int num_cores = getNumberOfCores(enforeced_core_type);
                    config._threadsPerStream = std::min(n_threads, num_cores);
                } else {
                    config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
                }
            }
    }
    return config;
}

IE_SUPPRESS_DEPRECATED_START
void Config::UpdateFromMap(const std::map<std::string, std::string>& configMap) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Config::UpdateFromMap");
    bool update_EnforcedCPUCoreType = false;
    auto prev_cpu_binding_type = cpu_thread_binding_type;
    auto prev_cpu_core_type = cpu_core_type;
    auto prev_n_threads = n_threads;
    for (auto& kvp : configMap) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        // std::cout << "[[ " << key << " ]] : " << val << std::endl;

        if (key.compare(PluginConfigParams::KEY_PERF_COUNT) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                useProfiling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                useProfiling = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableDynamicBatch = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableDynamicBatch = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DUMP_KERNELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                dumpCustomKernels = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                dumpCustomKernels = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY) == 0 ||
                   key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
            const int fixed_shift = 4;
            const uint32_t quque_priority = uVal & ((1 << fixed_shift) - 1);
            const uint32_t tbb_affinity = uVal >> fixed_shift;
            switch (quque_priority) {
                case 0:
                    queuePriority = cldnn::priority_mode_types::disabled;
                    break;
                case 1:
                    queuePriority = cldnn::priority_mode_types::low;
                    break;
                case 2:
                    queuePriority = cldnn::priority_mode_types::med;
                    break;
                case 3:
                    queuePriority = cldnn::priority_mode_types::high;
                    break;
                default:
                    IE_THROW(ParameterMismatch) << "Unsupported queue priority value: " << quque_priority;
            }
            switch (tbb_affinity) {
                case 0: // default
                    cpu_core_type = IStreamsExecutor::Config::ANY;
                    update_EnforcedCPUCoreType = false;
                    break;
                case 1: // Any (16)
                    cpu_core_type = IStreamsExecutor::Config::ANY;
                    update_EnforcedCPUCoreType = true;
                    break;
                case 2: // Little (32)
                    cpu_core_type = IStreamsExecutor::Config::LITTLE;
                    update_EnforcedCPUCoreType = true;
                    break;
                case 3: // Big (48)
                    cpu_core_type = IStreamsExecutor::Config::BIG;
                    update_EnforcedCPUCoreType = true;
                    break;
                case 4: // Round robin (64)
                    cpu_core_type = IStreamsExecutor::Config::ROUND_ROBIN;
                    update_EnforcedCPUCoreType = true;
                    break;
                default:
                    IE_THROW(ParameterMismatch) << "Unsupported tbb affinity value: " << tbb_affinity;
            }

        } else if (key.compare(GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) == 0 ||
                   key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
                case 0:
                    queueThrottle = cldnn::throttle_mode_types::disabled;
                    break;
                case 1:
                    queueThrottle = cldnn::throttle_mode_types::low;
                    break;
                case 2:
                    queueThrottle = cldnn::throttle_mode_types::med;
                    break;
                case 3:
                    queueThrottle = cldnn::throttle_mode_types::high;
                    break;
                default:
                    IE_THROW(ParameterMismatch) << "Unsupported queue throttle value: " << uVal;
            }
        } else if (key.compare(PluginConfigParams::KEY_CONFIG_FILE) == 0) {
            std::stringstream ss(val);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> configFiles(begin, end);
            for (auto& file : configFiles) {
                CLDNNCustomLayer::LoadFromFile(file, customLayers);
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_MODE) == 0) {
            if (val.compare(PluginConfigParams::TUNING_DISABLED) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_disabled;
            } else if (val.compare(PluginConfigParams::TUNING_CREATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_tune_and_cache;
            } else if (val.compare(PluginConfigParams::TUNING_USE_EXISTING) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_cache;
            } else if (val.compare(PluginConfigParams::TUNING_UPDATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_and_update;
            } else if (val.compare(PluginConfigParams::TUNING_RETUNE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_retune_and_cache;
            } else {
                IE_THROW(NotFound) << "Unsupported tuning mode value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_FILE) == 0) {
            tuningConfig.cache_file_path = val;
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_MEM_POOL) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                memory_pool_on = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                memory_pool_on = false;
            } else {
                IE_THROW(NotFound) << "Unsupported memory pool flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                graph_dumps_dir = val;
                createDirectory(graph_dumps_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_CACHE_DIR) == 0) {
            if (!val.empty()) {
                kernels_cache_dir = val;
                createDirectory(kernels_cache_dir);
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                sources_dumps_dir = val;
                createDirectory(sources_dumps_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                exclusiveAsyncRequests = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                exclusiveAsyncRequests = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) == 0) {
            if (val.compare(PluginConfigParams::GPU_THROUGHPUT_AUTO) == 0) {
                throughput_streams = 2;
            } else {
                int val_i;
                try {
                    val_i = std::stoi(val);
                } catch (const std::exception&) {
                    IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::GPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    throughput_streams = static_cast<uint16_t>(val_i);
            }
        } else if (key.compare(PluginConfigParams::KEY_DEVICE_ID) == 0) {
            // Validate if passed value is postivie number.
            try {
                int val_i = std::stoi(val);
                (void)val_i;
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_DEVICE_ID
                    << ". DeviceIDs are only represented by positive numbers";
            }
            // Set this value.
            device_id = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableInt8 = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableInt8 = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS) == 0 ||
                   key.compare(CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                nv12_two_inputs = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                nv12_two_inputs = false;
            } else {
                IE_THROW(NotFound) << "Unsupported NV12 flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enable_fp16_for_quantized_models = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enable_fp16_for_quantized_models = false;
            } else {
                IE_THROW(NotFound) << "Unsupported KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS flag value: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_MAX_NUM_THREADS) == 0) {
            int max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
            try {
                int val_i = std::stoi(val);
                if (val_i <= 0 || val_i > max_threads) {
                    n_threads = max_threads;
                } else {
                    n_threads = val_i;
                }
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << GPUConfigParams::KEY_GPU_MAX_NUM_THREADS << ": " << val
                                   << "\nSpecify the number of threads use for build as an integer."
                                   << "\nOut of range value will be set as a default value, maximum concurrent threads.";
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enable_loop_unrolling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enable_loop_unrolling = false;
            } else {
                IE_THROW(ParameterMismatch) << "Unsupported KEY_GPU_ENABLE_LOOP_UNROLLING flag value: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_CPU_BIND_THREAD) == 0) {
            if (val == CONFIG_VALUE(YES) || val == CONFIG_VALUE(NUMA)) {
                #if (defined(__APPLE__) || defined(_WIN32))
                cpu_thread_binding_type = IStreamsExecutor::ThreadBindingType::NUMA;
                #else
                cpu_thread_binding_type = (val == CONFIG_VALUE(YES))
                        ? IStreamsExecutor::ThreadBindingType::CORES : IStreamsExecutor::ThreadBindingType::NUMA;
                #endif
            } else if (val == CONFIG_VALUE(HYBRID_AWARE)) {
                cpu_thread_binding_type = IStreamsExecutor::ThreadBindingType::HYBRID_AWARE;
            } else if (val == CONFIG_VALUE(NO)) {
                cpu_thread_binding_type = IStreamsExecutor::ThreadBindingType::NONE;
            } else {
                IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_BIND_THREAD)
                                   << ". Expected only YES(binds to cores) / NO(no binding) / NUMA(binds to NUMA nodes) / "
                                                        "HYBRID_AWARE (let the runtime recognize and use the hybrid cores)";
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property key by plugin: " << key;
        }
    }
    #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    if (prev_cpu_binding_type != cpu_thread_binding_type
        || prev_cpu_core_type != cpu_core_type
        || prev_n_threads != n_threads) {
            auto set_tbb_config = [&] (const IStreamsExecutor::ThreadBindingType in_binding_type,
                bool is_set_enforced_core_type,
                const IStreamsExecutor::Config::PreferredCoreType in_enforeced_core_type,
                const int in_num_threads) {
                    if (is_set_enforced_core_type) {
                        auto streamExeConfig = MakeTBBConfigForLoadNetwork(
                                                                        IStreamsExecutor::HYBRID_AWARE,
                                                                        in_enforeced_core_type,
                                                                        in_num_threads);
                        cpu_thread_binding_type = streamExeConfig._threadBindingType;
                        cpu_core_type = streamExeConfig._threadPreferredCoreType;
                        n_threads = streamExeConfig._threadsPerStream;
                    } else if (cpu_thread_binding_type == IStreamsExecutor::HYBRID_AWARE) {
                        auto streamExeConfig = MakeTBBConfigForLoadNetwork(
                                                                        in_binding_type,
                                                                        IStreamsExecutor::Config::BIG,
                                                                        in_num_threads);
                        cpu_thread_binding_type = streamExeConfig._threadBindingType;
                        cpu_core_type = streamExeConfig._threadPreferredCoreType;
                        n_threads = streamExeConfig._threadsPerStream;
                    }
                    // std::cout << "TBB Config { ";
                    // std::cout << in_binding_type << ", ";
                    // if (is_set_enforced_core_type) {
                    //     std::cout << in_enforeced_core_type << ", ";
                    // } else {
                    //     std::cout << "Not set, ";
                    // }
                    // std::cout << in_num_threads << ", ";
                    // std::cout << cpu_thread_binding_type << ", ";
                    // std::cout << cpu_core_type << ", ";
                    // std::cout << n_threads  << " }" << std::endl;
                };
            set_tbb_config(cpu_thread_binding_type, update_EnforcedCPUCoreType, cpu_core_type, n_threads);
        }
    #endif
    // static bool is_test = true;
    // if (is_test) {
    //     is_test = false;
    //     auto get_bit_str = [&] (uint16_t val) -> std::string {
    //         std::bitset<16> y_val(val);
    //         std::stringstream ss;
    //         ss << y_val;
    //         return ss.str();
    //     };
    //     const int fixed_shift = 4;
    //     for (uint32_t j = 0; j < 4; j++) {
    //         for (uint32_t i = 0; i < 5; i++) {
    //             auto val = (i << fixed_shift) | j;
    //             std::cout << val << " : " << get_bit_str(val) << std::endl;
    //         }
    //     }
    // }

    adjustKeyMapValues();
}

void Config::adjustKeyMapValues() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Config::AdjustKeyMapValues");
    if (useProfiling)
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::NO;

    if (dumpCustomKernels)
        key_config_map[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::NO;

    if (exclusiveAsyncRequests)
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;

    if (memory_pool_on)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_MEM_POOL] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_MEM_POOL] = PluginConfigParams::NO;

    if (enableDynamicBatch)
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::NO;

    if (nv12_two_inputs)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::NO;

    if (enable_fp16_for_quantized_models)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS] = PluginConfigParams::NO;

    {
        uint32_t qp = 0;
        switch (queuePriority) {
        case cldnn::priority_mode_types::low: qp = 1; break;
        case cldnn::priority_mode_types::med: qp = 2; break;
        case cldnn::priority_mode_types::high: qp = 3; break;
        default: break;
        }
        uint32_t ct = 0;
        switch (cpu_core_type) {
        case IStreamsExecutor::Config::LITTLE:      ct = 1; break;
        case IStreamsExecutor::Config::BIG:         ct = 2; break;
        case IStreamsExecutor::Config::ROUND_ROBIN: ct = 3; break;
        case IStreamsExecutor::Config::ANY:
        default:break;
        }

        uint32_t plugin_priority = qp | (ct << 4);
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY] = std::to_string(plugin_priority);
        key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY] = std::to_string(plugin_priority);
    }
    {
        std::string qt = "0";
        switch (queueThrottle) {
        case cldnn::throttle_mode_types::low: qt = "1"; break;
        case cldnn::throttle_mode_types::med: qt = "2"; break;
        case cldnn::throttle_mode_types::high: qt = "3"; break;
        default: break;
        }
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE] = qt;
        key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE] = qt;
    }
    {
        std::string tm = PluginConfigParams::TUNING_DISABLED;
        switch (tuningConfig.mode) {
        case cldnn::tuning_mode::tuning_tune_and_cache: tm = PluginConfigParams::TUNING_CREATE; break;
        case cldnn::tuning_mode::tuning_use_cache: tm = PluginConfigParams::TUNING_USE_EXISTING; break;
        case cldnn::tuning_mode::tuning_use_and_update: tm = PluginConfigParams::TUNING_UPDATE; break;
        case cldnn::tuning_mode::tuning_retune_and_cache: tm = PluginConfigParams::TUNING_RETUNE; break;
        default: break;
        }
        key_config_map[PluginConfigParams::KEY_TUNING_MODE] = tm;
        key_config_map[PluginConfigParams::KEY_TUNING_FILE] = tuningConfig.cache_file_path;
    }

    key_config_map[CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR] = graph_dumps_dir;
    key_config_map[CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR] = sources_dumps_dir;
    key_config_map[PluginConfigParams::KEY_CACHE_DIR] = kernels_cache_dir;

    key_config_map[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(throughput_streams);
    key_config_map[PluginConfigParams::KEY_DEVICE_ID] = device_id;
    key_config_map[PluginConfigParams::KEY_CONFIG_FILE] = "";
    key_config_map[GPUConfigParams::KEY_GPU_MAX_NUM_THREADS] = std::to_string(n_threads);

    if (enable_loop_unrolling)
        key_config_map[GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING] = PluginConfigParams::YES;
    else
        key_config_map[GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING] = PluginConfigParams::NO;

    {
        std::string bt = PluginConfigParams::NO;
        switch (cpu_thread_binding_type) {
        case IStreamsExecutor::NONE: bt = PluginConfigParams::NO; break;
        case IStreamsExecutor::CORES : bt = PluginConfigParams::YES; break;
        case IStreamsExecutor::NUMA: bt = PluginConfigParams::NUMA; break;
        case IStreamsExecutor::HYBRID_AWARE: bt = PluginConfigParams::HYBRID_AWARE; break;
        default: break;
        }
        key_config_map[PluginConfigParams::KEY_CPU_BIND_THREAD] = bt;
    }
    // {
    //     std::string ct = PluginConfigParams::ANY;
    //     switch (cpu_core_type) {
    //     case IStreamsExecutor::Config::ANY: ct = PluginConfigParams::ANY; break;
    //     case IStreamsExecutor::Config::BIG: ct = PluginConfigParams::BIG; break;
    //     case IStreamsExecutor::Config::LITTLE: ct = PluginConfigParams::LITTLE; break;
    //     case IStreamsExecutor::Config::ROUND_ROBIN: ct = PluginConfigParams::ROUND_ROBIN; break;
    //     default: break;
    //     }
    //     key_config_map[PluginConfigParams::KEY_ENFORCE_CPU_CORE_TYPE] = ct;
    // }
}
IE_SUPPRESS_DEPRECATED_END

}  // namespace CLDNNPlugin
