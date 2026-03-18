// ctdp/space/reduction_space.h — Reduction-optimisation search space generator
//
// The bridge between the algebra layer (make_reduction, reduction_lane) and
// the space layer (descriptor_space, conditional_dim, feature_bridge).
//
// Given a tuple_reduction, automatically generates the appropriate search space
// with conditionally-present dimensions gated by algebraic properties:
//
//   tile_size        — always present (power_2, controls work granularity)
//   tree_shape       — present only when all_associative (reduction tree topology)
//   vec_width        — present only when all_have_identity (SIMD lane width)
//   lane_order_perm  — present only when all_commutative (future: permutation)
//
// Usage:
//   auto stats = make_reduction(
//       reduction_lane{identity_t{}, plus_fn{}, 0},
//       reduction_lane{identity_t{}, min_fn{},  INT_MAX}
//   );
//   auto [space, bridge] = make_reduction_opt_space(stats);
//   // space has 3 dims: tile_size + tree_shape (active) + vec_width (active)
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_REDUCTION_SPACE_H
#define CTDP_SPACE_REDUCTION_SPACE_H

#include "ctdp/space/conditional.h"
#include "ctdp/space/descriptor.h"
#include "ctdp/space/reduction_properties.h"

#include <utility>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// Tree shape enum — topologies for parallel reduction trees
// ═══════════════════════════════════════════════════════════════════════

enum class tree_shape : int {
    flat       = 0,  // Sequential left fold (always legal)
    binary     = 1,  // Balanced binary tree (requires associativity)
    pairwise   = 2,  // Pairwise summation (requires associativity)
};

// ═══════════════════════════════════════════════════════════════════════
// Default dimension descriptors for reduction optimisation
// ═══════════════════════════════════════════════════════════════════════

/// Tile size: power-of-two work granularity for the reduction.
/// Range [64, 4096] gives 7 levels: 64, 128, 256, 512, 1024, 2048, 4096.
inline auto default_tile_dim() {
    return power_2("tile_size", 64, 4096);
}

/// Tree shape: topology of the parallel reduction tree.
/// Only meaningful when all lanes are associative.
inline auto default_tree_shape_dim() {
    return make_enum_vals("tree_shape",
        {tree_shape::flat, tree_shape::binary, tree_shape::pairwise});
}

/// SIMD vector width: number of elements processed per SIMD operation.
/// Only meaningful when all lanes have identity elements (for partial-lane init).
inline auto default_vec_width_dim() {
    return make_int_set("vec_width", {1, 4, 8, 16});
}

// ═══════════════════════════════════════════════════════════════════════
// make_reduction_opt_space — the generator
// ═══════════════════════════════════════════════════════════════════════

/// Result type: space + bridge bundled together.
template <typename Space, typename Bridge>
struct reduction_opt_space_result {
    Space  space;
    Bridge bridge;
};

/// Generate a reduction-optimisation search space from a tuple_reduction.
///
/// The space has three dimensions:
///   1. tile_size    — always present (power_2, 64..4096)
///   2. tree_shape   — conditional on all_associative
///   3. vec_width    — conditional on all_have_identity
///
/// Returns a struct with .space and .bridge members.
///
/// Example:
///   auto red = make_reduction(
///       reduction_lane{identity_t{}, plus_fn{}, 0},
///       reduction_lane{identity_t{}, min_fn{},  INT_MAX}
///   );
///   auto result = make_reduction_opt_space(red);
///   result.space.enumerate([&](auto const& pt) { ... });
///   auto features = result.bridge.encode(pt);
template <typename... Lanes>
auto make_reduction_opt_space(
    const ct_dp::algebra::tuple_reduction<Lanes...>& red)
{
    auto props = reduction_properties(red);

    auto space = descriptor_space("reduction_opt",
        default_tile_dim(),
        conditional_dim(props.all_associative,   default_tree_shape_dim()),
        conditional_dim(props.all_have_identity, default_vec_width_dim())
    );

    auto bridge = default_bridge(space);

    return reduction_opt_space_result{std::move(space), std::move(bridge)};
}

/// Overload accepting custom dimension descriptors.
///
/// Allows callers to override the default tile/tree/vec descriptors
/// while still using the property-driven conditionality logic.
template <typename... Lanes, typename TileDim, typename TreeDim, typename VecDim>
auto make_reduction_opt_space(
    const ct_dp::algebra::tuple_reduction<Lanes...>& red,
    TileDim tile_dim,
    TreeDim tree_dim,
    VecDim  vec_dim)
{
    auto props = reduction_properties(red);

    auto space = descriptor_space("reduction_opt",
        std::move(tile_dim),
        conditional_dim(props.all_associative,   std::move(tree_dim)),
        conditional_dim(props.all_have_identity, std::move(vec_dim))
    );

    auto bridge = default_bridge(space);

    return reduction_opt_space_result{std::move(space), std::move(bridge)};
}

/// Overload accepting pre-computed properties directly.
///
/// Used by tree_space child factories that have group-level properties
/// (from make_group_properties) but no tuple_reduction for the subset.
inline auto make_reduction_opt_space(const reduction_properties_t& props)
{
    auto space = descriptor_space("reduction_opt",
        default_tile_dim(),
        conditional_dim(props.all_associative,   default_tree_shape_dim()),
        conditional_dim(props.all_have_identity, default_vec_width_dim())
    );

    auto bridge = default_bridge(space);

    return reduction_opt_space_result{std::move(space), std::move(bridge)};
}

} // namespace ctdp::space

#endif // CTDP_SPACE_REDUCTION_SPACE_H
