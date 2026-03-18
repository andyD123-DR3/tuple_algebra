// ctdp/space/reduction_tree_space.h — Reduction-specific tree_space helpers
//
// Bridges the algebra layer (reduction properties, fusibility) to the
// partition-rooted tree_space from tree_space.h.
//
// Contains:
//   make_group_properties   — extract + reindex properties for a lane subset
//   make_fusibility_filter  — legality predicate from fingerprints
//   make_reduction_child_factory — builds per-group opt space
//   make_reduction_tree_space    — convenience assembler
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_REDUCTION_TREE_SPACE_H
#define CTDP_SPACE_REDUCTION_TREE_SPACE_H

#include "ctdp/space/tree_space.h"
#include "ctdp/space/reduction_space.h"
#include "ctdp/space/reduction_properties.h"

#include <span>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// make_group_properties — subset + reindex reduction properties
//
// Copies every per-lane property array for the specified lane indices
// into a new reduction_properties_t with lane_count = subset size.
// Recomputes all aggregate booleans from the reindexed arrays.
//
// The lane_indices span is ephemeral — this function copies what it
// needs and does not retain the span.
// ═══════════════════════════════════════════════════════════════════════

inline reduction_properties_t make_group_properties(
    const reduction_properties_t& full,
    std::span<const std::size_t> lane_indices)
{
    reduction_properties_t gp{};
    gp.lane_count = lane_indices.size();

    // Copy per-lane arrays, reindexing from full[original] to gp[0..n)
    for (std::size_t i = 0; i < gp.lane_count; ++i) {
        auto src = lane_indices[i];
        gp.lane_associative[i]        = full.lane_associative[src];
        gp.lane_commutative[i]        = full.lane_commutative[src];
        gp.lane_has_identity[i]       = full.lane_has_identity[src];
        gp.lane_idempotent[i]         = full.lane_idempotent[src];
        gp.lane_identity_transform[i] = full.lane_identity_transform[src];
        gp.lane_exact_associative[i]  = full.lane_exact_associative[src];
        gp.lane_exact_commutative[i]  = full.lane_exact_commutative[src];
        gp.lane_transform_cost[i]     = full.lane_transform_cost[src];
        gp.lane_fingerprint[i]        = full.lane_fingerprint[src];
    }

    // Recompute aggregates from the subset
    gp.all_associative    = true;
    gp.all_commutative    = true;
    gp.all_have_identity  = true;
    gp.all_idempotent     = true;
    gp.all_identity_transforms = true;
    gp.any_non_associative  = false;
    gp.any_non_commutative  = false;
    gp.any_idempotent       = false;
    gp.any_power_transforms = false;
    gp.all_exact_associative  = true;
    gp.all_exact_commutative  = true;

    for (std::size_t i = 0; i < gp.lane_count; ++i) {
        if (!gp.lane_associative[i])        gp.all_associative = false;
        if (!gp.lane_commutative[i])        gp.all_commutative = false;
        if (!gp.lane_has_identity[i])       gp.all_have_identity = false;
        if (!gp.lane_idempotent[i])         gp.all_idempotent = false;
        if (!gp.lane_identity_transform[i]) gp.all_identity_transforms = false;
        if (!gp.lane_exact_associative[i])  gp.all_exact_associative = false;
        if (!gp.lane_exact_commutative[i])  gp.all_exact_commutative = false;

        if (!gp.lane_associative[i])  gp.any_non_associative = true;
        if (!gp.lane_commutative[i])  gp.any_non_commutative = true;
        if (gp.lane_idempotent[i])    gp.any_idempotent = true;
        if (gp.lane_transform_cost[i] != transform_cost::free)
            gp.any_power_transforms = true;
    }

    return gp;
}

// ═══════════════════════════════════════════════════════════════════════
// make_fusibility_filter — legality predicate from fingerprints
//
// A partition is legal iff every co-grouped pair is fusible.
// When all lanes are fusible, ALL Bell(N) partitions pass.
// When no lanes are fusible, only singletons pass.
// Pure and deterministic: captures props by value.
// ═══════════════════════════════════════════════════════════════════════

template <std::size_t N>
auto make_fusibility_filter(const reduction_properties_t& props) {
    return [props](const partition_value<N>& pv) -> bool {
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j)
                if (pv.same_group(i, j) && !props.fusible(i, j))
                    return false;
        return true;
    };
}

// ═══════════════════════════════════════════════════════════════════════
// make_reduction_child_factory — per-group opt space from properties
//
// Returns a factory matching the tree_space contract:
//   factory(root, group_label, lane_indices) → child space
//
// Internally: extracts group properties, calls
// make_reduction_opt_space(group_props), returns .space.
//
// Pure and deterministic. Captures full_props by value.
// The lane_indices span is ephemeral and not retained.
// ═══════════════════════════════════════════════════════════════════════

inline auto make_reduction_child_factory(
    const reduction_properties_t& full_props)
{
    return [full_props](
        const auto& /*root*/,
        std::size_t /*group_label*/,
        std::span<const std::size_t> lane_indices)
    {
        auto gp = make_group_properties(full_props, lane_indices);
        return make_reduction_opt_space(gp).space;
    };
}

// ═══════════════════════════════════════════════════════════════════════
// make_reduction_tree_space — convenience assembler
//
// Given a tuple_reduction, builds:
//   partition root + fusibility filter + reduction child factory
// Returns a tree_space ready for enumeration/solving.
// ═══════════════════════════════════════════════════════════════════════

template <typename... Lanes>
auto make_reduction_tree_space(
    const ct_dp::algebra::tuple_reduction<Lanes...>& red)
{
    auto props = reduction_properties(red);
    constexpr auto NL = sizeof...(Lanes);

    return make_tree_space<NL>(
        "reduction_tree",
        make_partition<NL>("grouping"),
        make_reduction_child_factory(props),
        make_fusibility_filter<NL>(props));
}

/// Build a tree_bridge for a reduction tree_space.
/// Uses default_bridge as the per-group bridge factory.
template <typename... Lanes>
auto make_reduction_tree_bridge(
    const ct_dp::algebra::tuple_reduction<Lanes...>& red)
{
    auto props = reduction_properties(red);
    constexpr auto NL = sizeof...(Lanes);

    auto child_factory = make_reduction_child_factory(props);

    // Determine child feature width from any single-group space
    // (width is stable across all groups due to conditional_dim).
    auto sample_space = make_reduction_opt_space(props).space;
    auto sample_bridge = default_bridge(sample_space);
    auto child_width = sample_bridge.num_features();

    auto bridge_factory = [](const auto& space) {
        return default_bridge(space);
    };

    return make_tree_bridge<NL>(
        make_partition<NL>("grouping"),
        std::move(child_factory),
        bridge_factory,
        child_width);
}

} // namespace ctdp::space

#endif // CTDP_SPACE_REDUCTION_TREE_SPACE_H
