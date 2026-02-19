// core/index_utils.h - Pack expansion helpers and index_sequence utilities
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Several framework components need common index_sequence patterns:
// - Iterating over compile-time index ranges
// - Applying a callable to each index in a pack
// - Mapping index sequences to transformed sequences
// - Extracting sub-sequences
//
// These utilities avoid repeating the same lambda + index_sequence
// boilerplate across plan_traversal, algebra, and solver code.
//
// All functions are constexpr and work in both compile-time and
// runtime contexts without modification.

#ifndef CTDP_CORE_INDEX_UTILS_H
#define CTDP_CORE_INDEX_UTILS_H

#include <cstddef>
#include <type_traits>
#include <utility>

namespace ctdp {

// =============================================================================
// for_each_index<N>(f) — call f(integral_constant<size_t, I>) for I in [0, N)
// =============================================================================

/// Invoke a callable once per index in [0, N), passing each index
/// as a std::integral_constant for use as a template argument.
///
/// Example:
/// ```cpp
/// for_each_index<4>([](auto I) {
///     constexpr size_t i = decltype(I)::value;
///     // use i as a compile-time constant
/// });
/// ```
template<std::size_t N, typename F>
constexpr void for_each_index(F&& fn) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (fn(std::integral_constant<std::size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});
}

// =============================================================================
// apply_with_indices(f, tuple) — call f(I, element) for each tuple element
// =============================================================================

/// Invoke a callable with (index, element) for each element of a tuple.
///
/// Example:
/// ```cpp
/// auto t = std::make_tuple(3.14, 42, true);
/// apply_with_indices([](auto I, auto const& elem) {
///     // I is integral_constant, elem is the tuple element
/// }, t);
/// ```
template<typename F, typename Tuple>
constexpr void apply_with_indices(F&& fn, Tuple&& tup) {
    constexpr auto N = std::tuple_size_v<std::remove_cvref_t<Tuple>>;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (fn(std::integral_constant<std::size_t, Is>{},
            std::get<Is>(std::forward<Tuple>(tup))), ...);
    }(std::make_index_sequence<N>{});
}

// =============================================================================
// static_for<Begin, End>(f) — call f(integral_constant<size_t, I>) for [Begin, End)
// =============================================================================

/// Compile-time for-loop over a half-open index range [Begin, End).
///
/// Example:
/// ```cpp
/// static_for<2, 5>([](auto I) {
///     // called with I = 2, 3, 4
/// });
/// ```
template<std::size_t Begin, std::size_t End, typename F>
constexpr void static_for(F&& fn) {
    if constexpr (Begin < End) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (fn(std::integral_constant<std::size_t, Begin + Is>{}), ...);
        }(std::make_index_sequence<End - Begin>{});
    }
}

// =============================================================================
// pack_size_v<Ts...> — number of types in a parameter pack
// =============================================================================

/// Compile-time constant for the number of types in a parameter pack.
template<typename... Ts>
inline constexpr std::size_t pack_size_v = sizeof...(Ts);

// =============================================================================
// index_of_v<T, Ts...> — find position of T in a type pack
// =============================================================================

namespace detail {

template<typename T, typename... Ts>
struct index_of_impl;

template<typename T, typename First, typename... Rest>
struct index_of_impl<T, First, Rest...> {
    static constexpr std::size_t value =
        std::is_same_v<T, First> ? 0 : 1 + index_of_impl<T, Rest...>::value;
};

template<typename T>
struct index_of_impl<T> {
    static_assert(sizeof(T) == 0, "Type not found in parameter pack");
    static constexpr std::size_t value = 0;
};

} // namespace detail

/// Compile-time index of type T within a parameter pack.
/// Static assertion failure if T is not present.
///
/// Example:
/// ```cpp
/// static_assert(index_of_v<int, double, int, char> == 1);
/// ```
template<typename T, typename... Ts>
inline constexpr std::size_t index_of_v = detail::index_of_impl<T, Ts...>::value;

// =============================================================================
// contains_v<T, Ts...> — check if T is in a type pack
// =============================================================================

/// True if T is one of Ts...
template<typename T, typename... Ts>
inline constexpr bool contains_v = (std::is_same_v<T, Ts> || ...);

} // namespace ctdp

#endif // CTDP_CORE_INDEX_UTILS_H
