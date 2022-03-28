// Copyright (C) 2017-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include <stdint.h>
#include <stddef.h>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <functional>

#include <iostream>

namespace cldnn {

struct primitive;

namespace meta {

// helper struct to tell wheter type T is any of given types U...
// termination case when U... is empty -> return std::false_type
template <class T, class... U>
struct is_any_of : public std::false_type {};

// helper struct to tell whether type is any of given types (U, Rest...)
// recurrence case when at least one type U is present -> returns std::true_type if std::same<T, U>::value is true,
// otherwise call is_any_of<T, Rest...> recurrently
template <class T, class U, class... Rest>
struct is_any_of<T, U, Rest...>
    : public std::conditional<std::is_same<T, U>::value, std::true_type, is_any_of<T, Rest...>>::type {};

template <class T>
struct always_false : public std::false_type {};

template <typename Ty, Ty Val>
struct always_false_ty_val : public std::false_type {};

template <typename Ty, Ty... Vals>
struct val_tuple {};

template <bool... Values>
struct all : public std::true_type {};

template <bool Val, bool... Values>
struct all<Val, Values...> : public std::integral_constant<bool, Val && all<Values...>::value> {};

}  // namespace meta

/// @cond CPP_HELPERS

/// @defgroup cpp_helpers Helpers
/// @{

#define CLDNN_API_CLASS(the_class) static_assert(std::is_standard_layout<the_class>::value, #the_class " has to be 'standard layout' class");

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type align_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? size : size - size % align + align);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type pad_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? 0 : align - size % align);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type is_aligned_to(T size, size_t align) {
    return !(size % align);
}

/// Computes ceil(@p val / @p divider) on unsigned integral numbers.
///
/// Computes division of unsigned integral numbers and rounds result up to full number (ceiling).
/// The function works for unsigned integrals only. Signed integrals are converted to corresponding
/// unsigned ones.
///
/// @tparam T1   Type of @p val. Type must be integral (SFINAE).
/// @tparam T2   Type of @p divider. Type must be integral (SFINAE).
///
/// @param val       Divided value. If value is signed, it will be converted to corresponding unsigned type.
/// @param divider   Divider value. If value is signed, it will be converted to corresponding unsigned type.
///
/// @return   Result of ceil(@p val / @p divider). The type of result is determined as if in normal integral
///           division, except each operand is converted to unsigned type if necessary.
template <typename T1, typename T2>
constexpr auto ceil_div(T1 val, T2 divider)
-> typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
    decltype(std::declval<typename std::make_unsigned<T1>::type>() / std::declval<typename std::make_unsigned<T2>::type>())>::type {
    typedef typename std::make_unsigned<T1>::type UT1;
    typedef typename std::make_unsigned<T2>::type UT2;
    typedef decltype(std::declval<UT1>() / std::declval<UT2>()) RetT;

    return static_cast<RetT>((static_cast<UT1>(val) + static_cast<UT2>(divider) - 1U) / static_cast<UT2>(divider));
}

/// Rounds @p val to nearest multiply of @p rounding that is greater or equal to @p val.
///
/// The function works for unsigned integrals only. Signed integrals are converted to corresponding
/// unsigned ones.
///
/// @tparam T1       Type of @p val. Type must be integral (SFINAE).
/// @tparam T2       Type of @p rounding. Type must be integral (SFINAE).
///
/// @param val        Value to round up. If value is signed, it will be converted to corresponding unsigned type.
/// @param rounding   Rounding value. If value is signed, it will be converted to corresponding unsigned type.
///
/// @return   @p val rounded up to nearest multiply of @p rounding. The type of result is determined as if in normal integral
///           division, except each operand is converted to unsigned type if necessary.
template <typename T1, typename T2>
constexpr auto round_up_to(T1 val, T2 rounding)
-> typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
    decltype(std::declval<typename std::make_unsigned<T1>::type>() / std::declval<typename std::make_unsigned<T2>::type>())>::type {
    typedef typename std::make_unsigned<T1>::type UT1;
    typedef typename std::make_unsigned<T2>::type UT2;
    typedef decltype(std::declval<UT1>() / std::declval<UT2>()) RetT;

    return static_cast<RetT>(ceil_div(val, rounding) * static_cast<UT2>(rounding));
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&& ... args) {
    return std::unique_ptr<T>( new T(std::forward<Args>(args)...) );
}

template <typename derived_type, typename base_type, typename std::enable_if<std::is_base_of<base_type, derived_type>::value, int>::type = 0>
inline derived_type* downcast(base_type* base) {
    if (auto casted = dynamic_cast<derived_type*>(base))
        return casted;

    throw std::runtime_error("Unable to cast pointer from base to derived type");
}

template <typename derived_type, typename base_type, typename std::enable_if<std::is_base_of<base_type, derived_type>::value, int>::type = 0>
inline derived_type& downcast(base_type& base) {
    try {
        return dynamic_cast<derived_type&>(base);
    } catch (std::bad_cast& /* ex */) {
        throw std::runtime_error("Unable to cast reference from base to derived type");
    }
    throw std::runtime_error("downcast failed with unhadnled exception");
}

template <typename T>
inline bool all_ones(const std::vector<T> vec) {
    return std::all_of(vec.begin(), vec.end(), [](const T& val) { return val == 1; });
}

template <typename T>
inline bool all_zeroes(const std::vector<T> vec) {
    return std::all_of(vec.begin(), vec.end(), [](const T& val) { return val == 0; });
}

template <typename T>
inline bool any_one(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) { return val == 1; });
}

template <typename T>
inline bool any_zero(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) { return val == 0; });
}

template <typename T>
inline bool any_not_one(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) { return val != 1; });
}

template <typename T>
inline bool any_not_zero(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) { return val != 0; });
}

// Helpers to get string for types that have operator<< defined
template <typename T>
inline std::string to_string(const T& v) {
    std::stringstream s;
    s << v;
    return s.str();
}

/// @}
/// @endcond
/// @}

/// @brief LRU cache which remove the least recently used data when cache is full.
template<typename TypeK, typename TypeD>
class LRUCache {
public:
    struct CacheEntry {
        TypeD   data;
        size_t  size;   // the size of data. it is necessary to check total data size is limited to capacity
    };

private:
    std::list<TypeK> order;
    std::unordered_map<TypeK, std::pair<typename std::list<TypeK>::iterator, LRUCache::CacheEntry>> cache_entry_map;
    size_t curr_data_size;
    const size_t capacity;

    std::mutex mtx;

public:
    explicit LRUCache(size_t caps) : curr_data_size(0), capacity(caps) {}

    ~LRUCache() {
        clear();
    }

    std::tuple<TypeD, bool> get(TypeK key, std::function<LRUCache::CacheEntry(void)> create_new_data) {
        std::lock_guard<std::mutex> lock(mtx);
        TypeD data;
        bool is_found = true;
        if (cache_entry_map.find(key) == cache_entry_map.end()) {
            if (create_new_data) {
                auto new_entry = create_new_data();
                add_new_entry(key, new_entry);
                data = cache_entry_map[key].second.data;
            }
            is_found = false;
        } else {
            // Move current data to front of deque
            order.erase(cache_entry_map[key].first);
            order.push_front(key);
            cache_entry_map[key].first = order.begin();
            data = cache_entry_map[key].second.data;
        }
        return std::make_pair(data, is_found);
    }

    std::tuple<TypeD, bool> find(TypeK key) {
        std::lock_guard<std::mutex> lock(mtx);
        bool is_found = false;
        TypeD data;

        if (cache_entry_map.find(key) == cache_entry_map.end()) {
            is_found = false;
        } else {
            order.erase(cache_entry_map[key].first);
            order.push_front(key);
            cache_entry_map[key].first = order.begin();
            is_found = true;
            data = cache_entry_map[key].second.data;
        }

        return std::make_pair(data, is_found);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        order.clear();
        cache_entry_map.clear();
        curr_data_size = 0;
    }

    size_t count() const {
       return cache_entry_map.size();
    }

    size_t get_current_data_size() {
        return curr_data_size;
    }

    std::list<TypeK> get_all_keys() const {
        return order;
    }

private:
    void add_new_entry(TypeK key, LRUCache::CacheEntry entry) {
        if (cache_entry_map.find(key) == cache_entry_map.end()) {
            if (curr_data_size > 0 && capacity < (curr_data_size + entry.size)) {
                //  Remove cache at the end of order
                curr_data_size -= cache_entry_map[order.back()].second.size;
                cache_entry_map.erase(order.back());
                order.pop_back();
            }
        } else {
            order.erase(cache_entry_map[key].first);
        }
        order.push_front(key);
        cache_entry_map[key] = std::make_pair(order.begin(), entry);
        curr_data_size += entry.size;
        std::cout << "cache_entry_map size=" << cache_entry_map.size() << std::endl;
    }
};
}  // namespace cldnn
