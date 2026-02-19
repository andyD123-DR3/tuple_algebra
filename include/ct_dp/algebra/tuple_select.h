// ct_dp/algebra/tuple_select.h
//
// Lane projection: select specific lanes from a tuple.
//
// tuple_select<Is...>(tuple) extracts the lanes at indices Is...,
// producing a smaller tuple. This is used for:
//   - Extracting specific statistics from a variadic reduction result
//   - Projecting a full metric vector onto the dimensions of interest
//   - Feeding subsets of lanes into downstream computations
//
// Example:
//   auto full = make_tuple(count, sum, sum_sq, min, max);
//   auto extrema = tuple_select<3, 4>(full);  // (min, max)
//   auto moments = tuple_select<0, 1, 2>(full);  // (count, sum, sum_sq)
//
// Part of P3666R2 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_TUPLE_SELECT_H
#define CT_DP_ALGEBRA_TUPLE_SELECT_H

#include <cstddef>
#include <tuple>

namespace ct_dp::algebra {

/// Select specific lanes from a tuple by index.
///
/// @tparam Is  Indices of lanes to extract (zero-based).
/// @param  tup The source tuple.
/// @return A tuple containing only the selected lanes, in the order specified.
template<std::size_t... Is, typename Tuple>
constexpr auto tuple_select(const Tuple& tup) noexcept {
    return std::make_tuple(std::get<Is>(tup)...);
}

/// Select a single lane from a tuple (convenience alias for std::get).
/// Returns by value, not a reference to a tuple element.
template<std::size_t I, typename Tuple>
constexpr auto tuple_lane(const Tuple& tup) noexcept {
    return std::get<I>(tup);
}

/// The number of lanes in a tuple.
template<typename Tuple>
inline constexpr std::size_t tuple_lane_count = std::tuple_size_v<Tuple>;

}  // namespace ct_dp::algebra

#endif  // CT_DP_ALGEBRA_TUPLE_SELECT_H
