// ctdp/space/reduction_properties.h — Algebraic property summariser
//
// Given a tuple_reduction<Lanes...>, extracts aggregate properties across
// all lanes. These predicates drive the conditional dimensions in the
// reduction-optimisation search space:
//
//   all_associative  → enables tree_shape dimension (non-flat reduction trees)
//   all_commutative  → enables permutation of lane processing order
//   all_have_identity → enables vector-width dimension (partial-lane init)
//   any_idempotent   → enables early-termination in search
//   lane_count       → drives per-lane sub-space sizing
//
// Usage:
//   constexpr auto stats = make_reduction(
//       reduction_lane{identity_t{}, plus_fn{}, 0},
//       reduction_lane{identity_t{}, min_fn{},  INT_MAX}
//   );
//   constexpr auto props = reduction_properties(stats);
//   static_assert(props.all_associative);
//   static_assert(props.all_have_identity);
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_REDUCTION_PROPERTIES_H
#define CTDP_SPACE_REDUCTION_PROPERTIES_H

#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// reduction_properties_t — aggregate property summary
//
// All fields are constexpr-friendly bools computed at construction.
// The intent is: one summariser object, many conditional dimensions
// that read it.
// ═══════════════════════════════════════════════════════════════════════

struct reduction_properties_t {
    std::size_t lane_count = 0;

    // Aggregate properties across all lanes
    bool all_associative    = false;  // every lane's reduce op is declared associative
    bool all_commutative    = false;  // every lane's reduce op is declared commutative
    bool all_have_identity  = false;  // every lane's reduce op has identity<T>()
    bool all_idempotent     = false;  // every lane's reduce op is declared idempotent

    // "Any" variants for search heuristics
    bool any_non_associative = false;  // at least one lane is NOT associative
    bool any_non_commutative = false;  // at least one lane is NOT commutative
    bool any_idempotent      = false;  // at least one lane IS idempotent

    // Per-lane transform classification
    bool all_identity_transforms = false;  // every lane's transform is identity_t
    bool any_power_transforms    = false;  // at least one lane has a power transform
};

// ═══════════════════════════════════════════════════════════════════════
// Property extraction from a tuple_reduction
// ═══════════════════════════════════════════════════════════════════════

namespace detail {

// Extract properties from a single reduction_lane
template <typename Lane>
struct lane_properties {
    using transform_type = std::remove_cvref_t<decltype(std::declval<Lane>().transform)>;
    using reduce_type    = std::remove_cvref_t<decltype(std::declval<Lane>().reduce)>;
    using init_type      = std::remove_cvref_t<decltype(std::declval<Lane>().init)>;

    static constexpr bool associative =
        ct_dp::algebra::detail::has_declared_assoc<reduce_type>;

    static constexpr bool commutative =
        ct_dp::algebra::detail::has_declared_commut<reduce_type>;

    static constexpr bool idempotent =
        ct_dp::algebra::detail::has_declared_idemp<reduce_type>;

    static constexpr bool has_identity =
        ct_dp::algebra::has_identity<reduce_type, init_type>;

    static constexpr bool identity_transform =
        ct_dp::algebra::is_identity_transform_v<transform_type>;

    static constexpr bool power_transform =
        ct_dp::algebra::is_power_transform_v<transform_type>;
};

// Fold properties across all lanes of a tuple_reduction
template <typename... Lanes, std::size_t... Is>
constexpr reduction_properties_t
summarise_lanes(const std::tuple<Lanes...>& /*lanes*/,
                std::index_sequence<Is...>) {
    reduction_properties_t props;
    props.lane_count = sizeof...(Lanes);

    // All/any aggregation using fold expressions
    props.all_associative =
        (lane_properties<Lanes>::associative && ...);
    props.all_commutative =
        (lane_properties<Lanes>::commutative && ...);
    props.all_have_identity =
        (lane_properties<Lanes>::has_identity && ...);
    props.all_idempotent =
        (lane_properties<Lanes>::idempotent && ...);

    props.any_non_associative =
        (!lane_properties<Lanes>::associative || ...);
    props.any_non_commutative =
        (!lane_properties<Lanes>::commutative || ...);
    props.any_idempotent =
        (lane_properties<Lanes>::idempotent || ...);

    props.all_identity_transforms =
        (lane_properties<Lanes>::identity_transform && ...);
    props.any_power_transforms =
        (lane_properties<Lanes>::power_transform || ...);

    return props;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════

/// Extract aggregate algebraic properties from a tuple_reduction.
///
/// The returned object is constexpr-friendly and can be used directly
/// as the predicate for conditional_dim construction:
///
///   auto props = reduction_properties(my_reduction);
///   auto space = descriptor_space("red_opt",
///       tile_dim,
///       conditional_dim(props.all_associative, tree_shape_dim),
///       conditional_dim(props.all_have_identity, vec_width_dim)
///   );
template <typename... Lanes>
constexpr reduction_properties_t
reduction_properties(const ct_dp::algebra::tuple_reduction<Lanes...>& red) {
    return detail::summarise_lanes(red.lanes,
        std::index_sequence_for<Lanes...>{});
}

} // namespace ctdp::space

#endif // CTDP_SPACE_REDUCTION_PROPERTIES_H
