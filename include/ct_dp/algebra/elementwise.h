// ct_dp/algebra/elementwise.h
//
// Elementwise operations on tuples.
//
// elementwise_binary_op<Ops...>  :  Tuple x Tuple -> Tuple
//   Applies the i-th operation to the i-th elements of two tuples.
//   Primary use: accumulator combine in transform_reduce.
//   Example: elementwise_binary_op<plus_fn, plus_fn, min_fn>
//            combines (a0,a1,a2) with (b0,b1,b2) to give
//            (a0+b0, a1+b1, min(a2,b2)).
//
// elementwise_unary_op<Ops...>  :  Tuple -> Tuple
//   Applies the i-th operation to the i-th element of a tuple.
//   Primary use: per-lane post-processing after reduction.
//   Example: elementwise_unary_op<identity_t, identity_t, sqrt_t>
//            transforms (count, sum, sum_sq) to (count, sum, sqrt(sum_sq)).
//
// Part of P3666R2 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_ELEMENTWISE_H
#define CT_DP_ALGEBRA_ELEMENTWISE_H

#include <cstddef>
#include <tuple>
#include <utility>

namespace ct_dp::algebra {

// ---------------------------------------------------------------------------
// elementwise_binary_op: Tuple x Tuple -> Tuple
// ---------------------------------------------------------------------------

/// Applies binary operations element-wise to two tuples.
///
/// Two modes:
///   Heterogeneous (N ops for N-tuple):
///     Given ops = (op0, ..., opN-1) and tuples a, b:
///     result = (op0(a[0], b[0]), ..., opN-1(a[N-1], b[N-1]))
///
///   Broadcast (1 op for N-tuple):
///     Given ops = (op0) and N-tuples a, b:
///     result = (op0(a[0], b[0]), ..., op0(a[N-1], b[N-1]))
///
/// This is the accumulator combine function for variadic reductions.
/// When used with transform_reduce, it defines how two partial results
/// are merged lane-by-lane.
///
/// @tparam Ops  Binary operation types: either 1 (broadcast) or N (per-lane).
template<typename... Ops>
struct elementwise_binary_op {
    std::tuple<Ops...> ops;

    static constexpr std::size_t op_count = sizeof...(Ops);

    /// Number of output lanes.
    /// In broadcast mode (op_count == 1), lane_count is determined by the
    /// tuple arguments at call time. In heterogeneous mode, lane_count == op_count.
    static constexpr std::size_t lane_count = sizeof...(Ops);

    /// Default construct (requires all Ops are default-constructible).
    constexpr elementwise_binary_op() noexcept
        requires (std::is_default_constructible_v<Ops> && ...) = default;

    /// Construct from individual operations.
    constexpr explicit elementwise_binary_op(Ops... ops_in) noexcept
        : ops{std::move(ops_in)...} {}

    /// Apply element-wise: combine two tuples using per-lane operations.
    ///
    /// Mandates: sizeof...(Ops) is 1 or sizeof...(Ops) == tuple_size_v<TupleL>.
    ///           tuple_size_v<TupleL> == tuple_size_v<TupleR>.
    template<typename TupleL, typename TupleR>
    constexpr auto operator()(const TupleL& lhs, const TupleR& rhs) const noexcept {
        constexpr auto N = std::tuple_size_v<std::remove_cvref_t<TupleL>>;
        static_assert(std::tuple_size_v<std::remove_cvref_t<TupleR>> == N,
                      "LHS and RHS tuple sizes must match");
        static_assert(op_count == 1 || op_count == N,
                      "Number of operations must be 1 (broadcast) or equal to tuple size");

        if constexpr (op_count == 1 && N != 1) {
            // Broadcast mode: single op applied to all lanes
            return broadcast_impl(lhs, rhs, std::make_index_sequence<N>{});
        } else {
            // Heterogeneous mode: per-lane ops
            return apply_impl(lhs, rhs, std::index_sequence_for<Ops...>{});
        }
    }

private:
    template<typename TupleL, typename TupleR, std::size_t... Is>
    constexpr auto apply_impl(const TupleL& lhs, const TupleR& rhs,
                              std::index_sequence<Is...>) const noexcept {
        return std::make_tuple(
            std::get<Is>(ops)(std::get<Is>(lhs), std::get<Is>(rhs))...
        );
    }

    template<typename TupleL, typename TupleR, std::size_t... Is>
    constexpr auto broadcast_impl(const TupleL& lhs, const TupleR& rhs,
                                  std::index_sequence<Is...>) const noexcept {
        return std::make_tuple(
            std::get<0>(ops)(std::get<Is>(lhs), std::get<Is>(rhs))...
        );
    }
};

/// Deduction guide: deduce Ops from constructor arguments.
template<typename... Ops>
elementwise_binary_op(Ops...) -> elementwise_binary_op<Ops...>;


// ---------------------------------------------------------------------------
// elementwise_unary_op: Tuple -> Tuple
// ---------------------------------------------------------------------------

/// Applies N unary operations element-wise to an N-tuple.
///
/// Given ops = (op0, op1, ..., opN-1) and tuple t:
///   result = (op0(t[0]), op1(t[1]), ..., opN-1(t[N-1]))
///
/// Primary use: post-processing lanes after a reduction completes.
/// For example, dividing sum by count to get mean, or taking sqrt
/// of variance.
///
/// @tparam Ops  Unary operation types, one per lane.
template<typename... Ops>
struct elementwise_unary_op {
    std::tuple<Ops...> ops;

    static constexpr std::size_t lane_count = sizeof...(Ops);

    constexpr elementwise_unary_op() noexcept
        requires (std::is_default_constructible_v<Ops> && ...) = default;

    constexpr explicit elementwise_unary_op(Ops... ops_in) noexcept
        : ops{std::move(ops_in)...} {}

    /// Apply element-wise: transform each lane independently.
    template<typename Tuple>
    constexpr auto operator()(const Tuple& tup) const noexcept {
        static_assert(std::tuple_size_v<std::remove_cvref_t<Tuple>> == lane_count,
                      "Tuple size must match number of operations");

        return apply_impl(tup, std::index_sequence_for<Ops...>{});
    }

private:
    template<typename Tuple, std::size_t... Is>
    constexpr auto apply_impl(const Tuple& tup,
                              std::index_sequence<Is...>) const noexcept {
        return std::make_tuple(
            std::get<Is>(ops)(std::get<Is>(tup))...
        );
    }
};

/// Deduction guide.
template<typename... Ops>
elementwise_unary_op(Ops...) -> elementwise_unary_op<Ops...>;

}  // namespace ct_dp::algebra

#endif  // CT_DP_ALGEBRA_ELEMENTWISE_H
