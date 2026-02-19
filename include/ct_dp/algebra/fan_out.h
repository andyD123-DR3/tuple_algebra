// ct_dp/algebra/fan_out.h
//
// Fan-out unary operation: Scalar -> Tuple.
//
// fan_out_unary_op<Ops...> applies N unary operations to a single scalar,
// producing an N-tuple. This is the "moments" pattern:
//
//   fan_out<constant_t<1>, identity_t, power_t<2>, power_t<3>, power_t<4>>(x)
//     = (1, x, x^2, x^3, x^4)
//
// In a transform_reduce pipeline, fan_out is the transform step:
//   each input element x is expanded into a tuple of derived values,
//   which are then reduced lane-by-lane via elementwise_binary_op.
//
// Part of P3666R2 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_FAN_OUT_H
#define CT_DP_ALGEBRA_FAN_OUT_H

#include <cstddef>
#include <tuple>
#include <utility>

namespace ct_dp::algebra {

/// Applies N unary operations to a single scalar, producing an N-tuple.
///
/// Given ops = (op0, op1, ..., opN-1) and scalar x:
///   result = (op0(x), op1(x), ..., opN-1(x))
///
/// This is the entry point of a variadic reduction: each input value
/// is fanned out into a tuple of per-lane contributions. The canonical
/// example is moments computation:
///
///   fan_out_unary_op<constant_t<1>, identity_t, power_t<2>>(x)
///     = (1, x, x^2)  -- count, sum, sum-of-squares
///
/// The operations share the input value x but produce independent
/// outputs. An optimising compiler can identify shared subexpressions
/// (e.g., x^2 used for both x^2 and x^4 = (x^2)^2) when the
/// addition chain optimisation is applied at a higher level.
///
/// @tparam Ops  Unary operation types, one per output lane.
template<typename... Ops>
struct fan_out_unary_op {
    std::tuple<Ops...> ops;

    static constexpr std::size_t lane_count = sizeof...(Ops);

    constexpr fan_out_unary_op() noexcept
        requires (std::is_default_constructible_v<Ops> && ...) = default;

    constexpr explicit fan_out_unary_op(Ops... ops_in) noexcept
        : ops{std::move(ops_in)...} {}

    /// Fan out: apply all operations to a single input value.
    template<typename T>
    constexpr auto operator()(const T& x) const noexcept {
        return apply_impl(x, std::index_sequence_for<Ops...>{});
    }

private:
    template<typename T, std::size_t... Is>
    constexpr auto apply_impl(const T& x, std::index_sequence<Is...>) const noexcept {
        return std::make_tuple(std::get<Is>(ops)(x)...);
    }
};

/// Deduction guide.
template<typename... Ops>
fan_out_unary_op(Ops...) -> fan_out_unary_op<Ops...>;

}  // namespace ct_dp::algebra

#endif  // CT_DP_ALGEBRA_FAN_OUT_H
