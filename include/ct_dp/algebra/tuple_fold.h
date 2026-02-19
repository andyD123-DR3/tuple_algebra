// ct_dp/algebra/tuple_fold.h
//
// Horizontal reduction: fold a tuple down to a single value.
//
// tuple_fold<Op>(tuple) applies a binary operation across all elements
// of a tuple, left-to-right. This is the horizontal counterpart to
// elementwise operations (which are vertical/lane-parallel).
//
// Example:
//   tuple_fold(plus_fn{}, make_tuple(1, 2, 3, 4)) == 10
//   tuple_fold(min_fn{},  make_tuple(5, 2, 8, 1)) == 1
//
// With an initial value:
//   tuple_fold(plus_fn{}, 100, make_tuple(1, 2, 3)) == 106
//
// Part of P3666R2 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_TUPLE_FOLD_H
#define CT_DP_ALGEBRA_TUPLE_FOLD_H

#include <cstddef>
#include <tuple>
#include <utility>

namespace ct_dp::algebra {

namespace detail {

// Fold with initial value: init op t[0] op t[1] op ... op t[N-1]
template<typename Op, typename Acc, typename Tuple, std::size_t... Is>
constexpr auto fold_impl(Op op, Acc acc, const Tuple& tup,
                         std::index_sequence<Is...>) noexcept {
    ((acc = op(acc, std::get<Is>(tup))), ...);
    return acc;
}

// Fold without initial value: t[0] op t[1] op ... op t[N-1]
template<typename Op, typename Tuple, std::size_t... Is>
constexpr auto fold_no_init_impl(Op op, const Tuple& tup,
                                 std::index_sequence<Is...>) noexcept {
    auto acc = std::get<0>(tup);
    ((acc = op(acc, std::get<Is + 1>(tup))), ...);
    return acc;
}

}  // namespace detail

/// Fold a tuple with a binary operation and an initial value.
///
/// Computes: init op t[0] op t[1] op ... op t[N-1]
/// Left-associative (left fold).
///
/// @param op   Binary operation: (Acc, Element) -> Acc
/// @param init Initial accumulator value.
/// @param tup  Tuple to fold over.
/// @return     The final accumulated value.
template<typename Op, typename Init, typename Tuple>
constexpr auto tuple_fold(Op op, Init init, const Tuple& tup) noexcept {
    constexpr auto N = std::tuple_size_v<Tuple>;
    if constexpr (N == 0) {
        return init;
    } else {
        return detail::fold_impl(op, init, tup, std::make_index_sequence<N>{});
    }
}

/// Fold a tuple with a binary operation, no initial value.
///
/// Computes: t[0] op t[1] op ... op t[N-1]
/// Requires tuple to have at least one element.
///
/// @param op  Binary operation.
/// @param tup Tuple to fold over (must be non-empty).
/// @return    The final value.
template<typename Op, typename Tuple>
constexpr auto tuple_fold(Op op, const Tuple& tup) noexcept {
    constexpr auto N = std::tuple_size_v<Tuple>;
    static_assert(N > 0, "Cannot fold an empty tuple without an initial value");

    if constexpr (N == 1) {
        return std::get<0>(tup);
    } else {
        return detail::fold_no_init_impl(
            op, tup, std::make_index_sequence<N - 1>{});
    }
}

}  // namespace ct_dp::algebra

#endif  // CT_DP_ALGEBRA_TUPLE_FOLD_H
