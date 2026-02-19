// core/constexpr_sort.h - Constexpr-compatible sorting algorithms
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// std::sort and std::ranges::sort are not constexpr in C++20 (constexpr
// std::sort arrives in C++26). graph_builder::finalise() needs sorting
// at compile time, so we provide our own.
//
// ALGORITHM CHOICES:
// - constexpr_sort: Hybrid insertion sort / heap sort.
//   Insertion sort for small ranges (≤ 32 elements), heap sort for larger.
//   Heap sort is O(n log n) worst-case with no recursion (avoids
//   constexpr stack depth limits) and is in-place.
//
// - constexpr_stable_sort: Bottom-up merge sort using auxiliary buffer.
//   O(n log n) worst-case, stable. Requires constexpr_vector as auxiliary
//   storage. Used by graph_builder::finalise() for deterministic edge ordering.
//
// - constexpr_is_sorted: Predicate check.
//
// INTERFACE:
// All functions operate on iterator pairs, matching STL conventions.
// Default comparator is std::less<>{} (ascending order).
//
// CONSTEXPR SAFETY:
// No dynamic allocation, no UB, all loops bounded, all operations
// on value types (no reinterpret_cast, no placement new).

#ifndef CTDP_CORE_CONSTEXPR_SORT_H
#define CTDP_CORE_CONSTEXPR_SORT_H

#include "constexpr_vector.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <utility>

namespace ctdp {

namespace detail {

// =========================================================================
// Insertion sort — O(n²), but fast for small n and stable
// =========================================================================

template<typename RandomIt, typename Compare>
constexpr void insertion_sort(RandomIt first, RandomIt last, Compare comp) {
    if (first == last) return;
    for (auto i = first + 1; i != last; ++i) {
        auto key = std::move(*i);
        auto j = i;
        while (j != first && comp(key, *(j - 1))) {
            *j = std::move(*(j - 1));
            --j;
        }
        *j = std::move(key);
    }
}

// =========================================================================
// Heap sort — O(n log n) worst-case, in-place, no recursion
// =========================================================================

template<typename RandomIt, typename Compare>
constexpr void sift_down(RandomIt first, std::size_t len, std::size_t i,
                         Compare comp) {
    while (true) {
        std::size_t largest = i;
        std::size_t left = 2 * i + 1;
        std::size_t right = 2 * i + 2;

        if (left < len && comp(*(first + largest), *(first + left))) {
            largest = left;
        }
        if (right < len && comp(*(first + largest), *(first + right))) {
            largest = right;
        }
        if (largest == i) return;

        auto tmp = std::move(*(first + i));
        *(first + i) = std::move(*(first + largest));
        *(first + largest) = std::move(tmp);
        i = largest;
    }
}

template<typename RandomIt, typename Compare>
constexpr void heap_sort(RandomIt first, RandomIt last, Compare comp) {
    auto len = static_cast<std::size_t>(last - first);
    if (len <= 1) return;

    // Build max-heap
    for (std::size_t i = len / 2; i > 0; --i) {
        sift_down(first, len, i - 1, comp);
    }

    // Extract elements
    for (std::size_t i = len - 1; i > 0; --i) {
        auto tmp = std::move(*first);
        *first = std::move(*(first + i));
        *(first + i) = std::move(tmp);
        sift_down(first, i, 0, comp);
    }
}

// =========================================================================
// Bottom-up merge sort — O(n log n), stable, uses auxiliary storage
// =========================================================================

template<typename RandomIt, typename Compare>
constexpr void merge_inplace(RandomIt first, RandomIt mid, RandomIt last,
                             RandomIt aux, Compare comp) {
    auto n = static_cast<std::size_t>(last - first);
    // Copy to auxiliary
    for (std::size_t i = 0; i < n; ++i) {
        *(aux + i) = std::move(*(first + i));
    }

    auto left_end = static_cast<std::size_t>(mid - first);
    std::size_t i = 0;          // left cursor in aux
    std::size_t j = left_end;   // right cursor in aux
    auto out = first;

    while (i < left_end && j < n) {
        // Use !comp(right, left) for stability: equal elements
        // taken from left side first.
        if (!comp(*(aux + j), *(aux + i))) {
            *out = std::move(*(aux + i));
            ++i;
        } else {
            *out = std::move(*(aux + j));
            ++j;
        }
        ++out;
    }
    while (i < left_end) {
        *out = std::move(*(aux + i));
        ++i;
        ++out;
    }
    while (j < n) {
        *out = std::move(*(aux + j));
        ++j;
        ++out;
    }
}

} // namespace detail

// =========================================================================
// Public API
// =========================================================================

/// Threshold below which insertion sort is used instead of heap sort.
/// 32 is a common practical threshold; for constexpr evaluation the
/// crossover point is less critical than correctness.
inline constexpr std::size_t sort_insertion_threshold = 32;

/// Sort elements in [first, last) using comparator.
/// Not stable. O(n log n) worst-case.
///
/// Uses insertion sort for small ranges, heap sort for larger.
///
/// Example:
/// ```cpp
/// constexpr_vector<int, 10> v{5, 3, 1, 4, 2};
/// ctdp::constexpr_sort(v.begin(), v.end());
/// // v == {1, 2, 3, 4, 5}
///
/// // Custom comparator (descending):
/// ctdp::constexpr_sort(v.begin(), v.end(), std::greater<>{});
/// ```
template<typename RandomIt, typename Compare = std::less<>>
constexpr void constexpr_sort(RandomIt first, RandomIt last,
                              Compare comp = Compare{}) {
    auto n = static_cast<std::size_t>(last - first);
    if (n <= 1) return;

    if (n <= sort_insertion_threshold) {
        detail::insertion_sort(first, last, comp);
    } else {
        detail::heap_sort(first, last, comp);
    }
}

/// Stable sort elements in [first, last) using comparator.
/// Requires auxiliary storage of at least (last - first) elements.
///
/// O(n log n) worst-case. Bottom-up merge sort.
///
/// AuxIt must point to writable storage of the same value type,
/// with at least (last - first) elements available.
///
/// Example:
/// ```cpp
/// constexpr_vector<std::pair<int,int>, 10> edges{...};
/// constexpr_vector<std::pair<int,int>, 10> aux{};
/// aux.resize(edges.size());
/// ctdp::constexpr_stable_sort(edges.begin(), edges.end(),
///                             aux.begin(), my_comparator);
/// ```
template<typename RandomIt, typename AuxIt, typename Compare = std::less<>>
constexpr void constexpr_stable_sort(RandomIt first, RandomIt last,
                                     AuxIt aux,
                                     Compare comp = Compare{}) {
    auto n = static_cast<std::size_t>(last - first);
    if (n <= 1) return;

    // Small ranges: insertion sort is stable
    if (n <= sort_insertion_threshold) {
        detail::insertion_sort(first, last, comp);
        return;
    }

    // Bottom-up merge sort
    for (std::size_t width = 1; width < n; width *= 2) {
        for (std::size_t i = 0; i + width < n; i += 2 * width) {
            auto mid_offset = i + width;
            auto end_offset = i + 2 * width;
            if (end_offset > n) end_offset = n;

            detail::merge_inplace(
                first + i,
                first + mid_offset,
                first + end_offset,
                aux,
                comp
            );
        }
    }
}

/// Convenience: stable sort a constexpr_vector in-place.
/// Allocates auxiliary storage internally (another constexpr_vector).
///
/// Example:
/// ```cpp
/// constexpr_vector<int, 100> v{5, 3, 1, 4, 2};
/// ctdp::constexpr_stable_sort_vec(v);
/// ```
template<typename T, std::size_t MaxN, typename Compare = std::less<>>
constexpr void constexpr_stable_sort_vec(constexpr_vector<T, MaxN>& v,
                                         Compare comp = Compare{}) {
    if (v.size() <= 1) return;

    constexpr_vector<T, MaxN> aux{};
    aux.resize(v.size());
    constexpr_stable_sort(v.begin(), v.end(), aux.begin(), comp);
}

/// Check if [first, last) is sorted according to comparator.
///
/// Example:
/// ```cpp
/// static_assert(ctdp::constexpr_is_sorted(v.begin(), v.end()));
/// ```
template<typename RandomIt, typename Compare = std::less<>>
[[nodiscard]] constexpr bool constexpr_is_sorted(RandomIt first, RandomIt last,
                                                 Compare comp = Compare{}) {
    if (first == last) return true;
    auto prev = first;
    for (auto it = first + 1; it != last; ++it) {
        if (comp(*it, *prev)) return false;
        prev = it;
    }
    return true;
}

/// Remove consecutive duplicates from a sorted range.
/// Returns iterator to new logical end.
///
/// Precondition: [first, last) is sorted according to comp (or
/// equivalently, equal elements are adjacent).
///
/// Example:
/// ```cpp
/// constexpr_vector<int, 10> v{1, 1, 2, 3, 3};
/// auto new_end = ctdp::constexpr_unique(v.begin(), v.end());
/// v.resize(new_end - v.begin());
/// // v == {1, 2, 3}
/// ```
template<typename ForwardIt, typename BinaryPred>
constexpr ForwardIt constexpr_unique(ForwardIt first, ForwardIt last,
                                     BinaryPred pred) {
    if (first == last) return last;

    auto result = first;
    auto prev = first;
    ++first;

    while (first != last) {
        if (!pred(*prev, *first)) {
            ++result;
            if (result != first) {
                *result = std::move(*first);
            }
            prev = result;
        }
        ++first;
    }
    return ++result;
}

/// Remove consecutive duplicates using operator==.
template<typename ForwardIt>
constexpr ForwardIt constexpr_unique(ForwardIt first, ForwardIt last) {
    return constexpr_unique(first, last,
        [](auto const& a, auto const& b) { return a == b; });
}

} // namespace ctdp

#endif // CTDP_CORE_CONSTEXPR_SORT_H
