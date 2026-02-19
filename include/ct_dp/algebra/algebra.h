// ct_dp/algebra/algebra.h
//
// Convenience header: includes the complete tuple algebra library.
//
// Components:
//   operations.h      — Typed operations: identity, power, constant, min, max, plus, etc.
//   elementwise.h     — elementwise_binary_op, elementwise_unary_op
//   fan_out.h         — fan_out_unary_op (scalar -> tuple)
//   tuple_select.h    — Lane projection (tuple -> smaller tuple)
//   tuple_fold.h      — Horizontal fold (tuple -> scalar)
//   make_reduction.h  — reduction_lane, tuple_reduction, make_tuple_reduction
//
// Usage pattern (variadic statistics in one pass):
//
//   #include <ct_dp/algebra/algebra.h>
//   using namespace ct_dp::algebra;
//
//   constexpr auto stats = make_tuple_reduction(
//       reduction_lane{constant_t<1>{}, plus_fn{}, 0},
//       reduction_lane{identity_t{},    plus_fn{}, 0.0},
//       reduction_lane{power_t<2>{},    plus_fn{}, 0.0},
//       reduction_lane{identity_t{},    min_fn{},  +inf},
//       reduction_lane{identity_t{},    max_fn{},  -inf}
//   );
//
//   auto [count, sum, sum_sq, min, max] = stats.reduce(data);
//
// Part of P3666R0 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_ALGEBRA_H
#define CT_DP_ALGEBRA_ALGEBRA_H

#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/elementwise.h"
#include "ct_dp/algebra/fan_out.h"
#include "ct_dp/algebra/tuple_select.h"
#include "ct_dp/algebra/tuple_fold.h"
#include "ct_dp/algebra/make_reduction.h"

#endif  // CT_DP_ALGEBRA_ALGEBRA_H
