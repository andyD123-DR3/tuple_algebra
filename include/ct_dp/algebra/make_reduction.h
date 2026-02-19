// ct_dp/algebra/make_reduction.h
//
// Synthesis of complete tuple reductions from lane descriptors.
//
// A reduction_lane describes one lane of a variadic reduction:
//   { transform, reduce, init }
//
// make_reduction(lanes...) synthesises the three components needed
// for a transform_reduce call:
//   - init():      tuple of per-lane identity values
//   - transform(x): fan_out applying per-lane transforms to input x
//   - combine(a,b): elementwise_binary_op applying per-lane reduces
//
// This is the main entry point for constructing variadic reductions.
// The user describes what they want per lane; the library assembles
// the machinery.
//
// Example:
//   constexpr auto stats = make_reduction(
//       reduction_lane{constant_t<1>{}, plus_fn{}, 0},       // count
//       reduction_lane{identity_t{},    plus_fn{}, 0.0},     // sum
//       reduction_lane{power_t<2>{},    plus_fn{}, 0.0},     // sum_sq
//       reduction_lane{identity_t{},    min_fn{},  +inf},    // min
//       reduction_lane{identity_t{},    max_fn{},  -inf}     // max
//   );
//
//   auto result = stats.reduce(begin, end);
//   // result is tuple<int, double, double, double, double>
//   //            = (count, sum, sum_sq, min, max)
//
// Part of P3666R2 tuple algebra targeting C++26.
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#ifndef CT_DP_ALGEBRA_MAKE_REDUCTION_H
#define CT_DP_ALGEBRA_MAKE_REDUCTION_H

#include <cstddef>
#include <tuple>
#include <utility>

#include "ct_dp/algebra/elementwise.h"
#include "ct_dp/algebra/fan_out.h"

namespace ct_dp::algebra {

// ---------------------------------------------------------------------------
// reduction_lane: describes one lane of a variadic reduction
// ---------------------------------------------------------------------------

/// A single lane of a variadic reduction.
///
/// @tparam Transform  Unary op applied to each input element for this lane.
/// @tparam Reduce     Binary op used to combine partial results for this lane.
/// @tparam Init       Type of the identity/initial value for this lane.
///
/// Example lanes:
///   reduction_lane{constant_t<1>{}, plus_fn{}, 0}        // count
///   reduction_lane{identity_t{},    plus_fn{}, 0.0}      // sum
///   reduction_lane{power_t<2>{},    plus_fn{}, 0.0}      // sum of squares
///   reduction_lane{identity_t{},    min_fn{},  +HUGE}    // minimum
///   reduction_lane{identity_t{},    max_fn{},  -HUGE}    // maximum
template<typename Transform, typename Reduce, typename Init>
struct reduction_lane {
    Transform transform;
    Reduce    reduce;
    Init      init;
};

/// Deduction guide.
template<typename T, typename R, typename I>
reduction_lane(T, R, I) -> reduction_lane<T, R, I>;


// ---------------------------------------------------------------------------
// tuple_reduction: the assembled reduction machinery
// ---------------------------------------------------------------------------

/// Complete variadic reduction assembled from lane descriptors.
///
/// Holds the three components needed for transform_reduce:
///   - init_value:  tuple of per-lane identity values
///   - transform_op: fan_out applying per-lane transforms to an input element
///   - combine_op:   elementwise binary op for per-lane accumulation
///
/// Also provides a convenience `reduce(first, last)` method that executes
/// the full transform_reduce loop.
///
/// @tparam Lanes  Pack of reduction_lane types, one per output lane.
template<typename... Lanes>
struct tuple_reduction {
    std::tuple<Lanes...> lanes;

    static constexpr std::size_t lane_count = sizeof...(Lanes);

    /// Not default-constructible: lanes must be provided.
    constexpr tuple_reduction() noexcept = delete;

    constexpr explicit tuple_reduction(Lanes... lanes_in) noexcept
        : lanes{std::move(lanes_in)...} {}

    // -- init --

    /// The identity/initial value as a tuple, one element per lane.
    constexpr auto init_value() const noexcept {
        return init_impl(std::index_sequence_for<Lanes...>{});
    }

    // -- transform (fan-out) --

    /// The fan-out transform: maps a single input value to an N-tuple.
    constexpr auto transform_fn() const noexcept {
        return make_transform(std::index_sequence_for<Lanes...>{});
    }

    /// Convenience: apply the transform to a value directly.
    template<typename T>
    constexpr auto transform(T x) const noexcept {
        return transform_fn()(x);
    }

    // -- combine (elementwise binary) --

    /// The elementwise combine operation for accumulator merging.
    constexpr auto combine_fn() const noexcept {
        return make_combine(std::index_sequence_for<Lanes...>{});
    }

    /// Convenience: combine two accumulator tuples directly.
    template<typename TupleL, typename TupleR>
    constexpr auto combine(const TupleL& lhs, const TupleR& rhs) const noexcept {
        return combine_fn()(lhs, rhs);
    }

    // -- full reduction --

    /// Execute the complete transform_reduce over an iterator range.
    ///
    /// Equivalent to:
    ///   auto acc = init_value();
    ///   for (auto it = first; it != last; ++it)
    ///       acc = combine(acc, transform(*it));
    ///   return acc;
    ///
    /// This is a sequential left fold. For parallel execution, extract
    /// init_value(), combine_fn(), and transform_fn() and pass to
    /// std::transform_reduce with an execution policy.
    template<typename InputIt>
    constexpr auto reduce(InputIt first, InputIt last) const noexcept {
        auto acc = init_value();
        auto xform = transform_fn();
        auto comb = combine_fn();
        for (; first != last; ++first) {
            acc = comb(acc, xform(*first));
        }
        return acc;
    }

    /// Execute over a container (convenience).
    template<typename Container>
    constexpr auto reduce(const Container& c) const noexcept {
        using std::begin;
        using std::end;
        return reduce(begin(c), end(c));
    }

private:
    template<std::size_t... Is>
    constexpr auto init_impl(std::index_sequence<Is...>) const noexcept {
        return std::make_tuple(std::get<Is>(lanes).init...);
    }

    template<std::size_t... Is>
    constexpr auto make_transform(std::index_sequence<Is...>) const noexcept {
        return fan_out_unary_op{std::get<Is>(lanes).transform...};
    }

    template<std::size_t... Is>
    constexpr auto make_combine(std::index_sequence<Is...>) const noexcept {
        return elementwise_binary_op{std::get<Is>(lanes).reduce...};
    }
};

/// Deduction guide.
template<typename... Lanes>
tuple_reduction(Lanes...) -> tuple_reduction<Lanes...>;


// ---------------------------------------------------------------------------
// make_reduction: the primary factory
// ---------------------------------------------------------------------------

/// Construct a tuple_reduction from individual lane descriptors.
///
/// This is the main user-facing entry point for defining variadic reductions.
///
/// @param lanes  One reduction_lane per output statistic.
/// @return       A tuple_reduction<decay_t<Lanes>...> ready for use with
///               reduce() or for extracting init/transform/combine components.
template<typename... Lanes>
constexpr auto make_reduction(Lanes... lanes) noexcept {
    return tuple_reduction{std::move(lanes)...};
}

/// Backward-compatible alias (R1 name).
template<typename... Lanes>
constexpr auto make_tuple_reduction(Lanes... lanes) noexcept {
    return make_reduction(std::move(lanes)...);
}

}  // namespace ct_dp::algebra

#endif  // CT_DP_ALGEBRA_MAKE_REDUCTION_H
