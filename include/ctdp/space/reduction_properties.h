// ctdp/space/reduction_properties.h — Algebraic property summariser
//
// Given a tuple_reduction<Lanes...>, extracts aggregate and per-lane
// properties. These predicates drive conditional dimensions, partition
// legality, and cost-model heuristics in the reduction-optimisation
// search space.
//
// Three levels of property:
//
//   Aggregate (all/any):
//     all_associative  → enables tree_shape dimension
//     all_commutative  → enables lane-order permutation
//     all_have_identity → enables vector-width dimension
//     any_idempotent   → enables early-termination in search
//
//   Type-safe aggregate:
//     all_exact_associative  → safe to reorder for init_type
//     all_exact_commutative  → safe to permute for init_type
//     (conservative gate: requires integral/enum init types)
//
//   Per-lane:
//     lane_associative[i]  → drives partition legality
//     lane_fingerprint[i]  → lanes with same fingerprint are fusible
//     lane_transform_cost[i] → gates transform+reduce fusion
//
// Usage:
//   auto props = reduction_properties(my_reduction);
//
//   // Aggregate gates for conditional dimensions:
//   conditional_dim(props.all_associative, tree_shape_dim)
//
//   // Per-lane for partition legality:
//   bool can_fuse = props.fusible(0, 1);
//
// Part of the CT-DP space constructor algebra.
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_SPACE_REDUCTION_PROPERTIES_H
#define CTDP_SPACE_REDUCTION_PROPERTIES_H

#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace ctdp::space {

// ═══════════════════════════════════════════════════════════════════════
// Transform cost classification
// ═══════════════════════════════════════════════════════════════════════

/// Cost class for a lane's transform operation.
/// Drives whether transform+reduce fusion is worthwhile.
enum class transform_cost : std::uint8_t {
    free,       // identity, constant — zero ALU ops, always fuse
    cheap,      // power (non-identity) — 1-2 ALU ops, usually fuse
    expensive,  // user-defined, negate, abs — unknown cost, cost-model decides
};

// Value-initialised arrays of transform_cost must default to free.
static_assert(static_cast<int>(transform_cost::free) == 0,
    "transform_cost::free must be 0 for value-initialised arrays");

// ═══════════════════════════════════════════════════════════════════════
// reduction_properties_t — aggregate + per-lane property summary
//
// All fields are constexpr-friendly. The struct is value-type with
// fixed-size arrays sized to max_lanes (16). Lane counts beyond this
// trigger a static_assert at extraction time.
// ═══════════════════════════════════════════════════════════════════════

struct reduction_properties_t {
    static constexpr std::size_t max_lanes = 16;

    std::size_t lane_count = 0;

    // ── Aggregate properties (abstract declaration) ──────────────
    bool all_associative        = false;
    bool all_commutative        = false;
    bool all_have_identity      = false;
    bool all_idempotent         = false;

    bool any_non_associative    = false;
    bool any_non_commutative    = false;
    bool any_idempotent         = false;

    bool all_identity_transforms = false;
    bool any_power_transforms    = false;

    // ── Type-safe aggregate properties ───────────────────────────
    // Abstract declaration AND init_type is exact (integral/enum).
    // This is the conservative gate for parallel execution safety.
    // float plus_fn declares associative, but
    // declares_associative_v<plus_fn, double> is false.
    bool all_exact_associative  = false;
    bool all_exact_commutative  = false;

    // ── Per-lane property arrays ─────────────────────────────────
    // Indexed by lane [0..lane_count). Entries beyond lane_count are zero.
    // These drive the partition(N) legality predicate: two lanes can
    // share a fusion group only if their per-lane properties are compatible.
    std::array<bool, max_lanes> lane_associative{};
    std::array<bool, max_lanes> lane_commutative{};
    std::array<bool, max_lanes> lane_has_identity{};
    std::array<bool, max_lanes> lane_idempotent{};
    std::array<bool, max_lanes> lane_identity_transform{};

    // Type-safe per-lane (declaration AND exact type)
    std::array<bool, max_lanes> lane_exact_associative{};
    std::array<bool, max_lanes> lane_exact_commutative{};

    // ── Transform cost per lane ──────────────────────────────────
    std::array<transform_cost, max_lanes> lane_transform_cost{};

    // ── Fusibility fingerprint per lane ──────────────────────────
    // Encodes the algebraic signature of each lane as a small integer.
    // Two lanes i,j can be fused iff fingerprint[i] == fingerprint[j].
    // Encoding: 4-bit field from {associative, commutative, has_identity, idempotent}.
    std::array<unsigned, max_lanes> lane_fingerprint{};

    // ── Derived convenience queries ──────────────────────────────

    /// Number of distinct fingerprints (= number of fusibility classes).
    std::size_t num_fingerprint_classes() const {
        if (lane_count == 0) return 0;
        unsigned seen = 0;  // bitmask, works for fingerprints 0..15
        for (std::size_t i = 0; i < lane_count; ++i)
            seen |= (1u << lane_fingerprint[i]);
        std::size_t count = 0;
        while (seen) { count += (seen & 1u); seen >>= 1; }
        return count;
    }

    /// Are all lanes fusible with each other? (single fingerprint class)
    bool all_fusible() const {
        return num_fingerprint_classes() <= 1;
    }

    /// Are two specific lanes fusible?
    /// Returns false if either index is out of range.
    bool fusible(std::size_t i, std::size_t j) const {
        if (i >= lane_count || j >= lane_count) return false;
        return lane_fingerprint[i] == lane_fingerprint[j];
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Property extraction from a tuple_reduction
// ═══════════════════════════════════════════════════════════════════════

namespace detail {

// ── Transform cost detection ─────────────────────────────────────────

// Detect constant_t via its `value` static member (but not identity_t).
// NOTE: Any user-defined transform with a static `value` member will be
// misclassified as free. This is acceptable for built-in types. A future
// fix would use a dedicated is_constant_t tag or concept.
template <typename T, typename = void>
inline constexpr bool is_constant_transform_v = false;
template <typename T>
inline constexpr bool is_constant_transform_v<T,
    std::void_t<decltype(T::value)>> = !ct_dp::algebra::is_identity_transform_v<T>;

/// Classify a transform type's runtime cost.
template <typename Transform>
constexpr transform_cost classify_transform() {
    if constexpr (ct_dp::algebra::is_identity_transform_v<Transform>) {
        return transform_cost::free;
    } else if constexpr (is_constant_transform_v<Transform>) {
        return transform_cost::free;
    } else if constexpr (ct_dp::algebra::is_power_transform_v<Transform>) {
        return transform_cost::cheap;
    } else {
        return transform_cost::expensive;
    }
}

// ── Per-lane property extraction ─────────────────────────────────────

template <typename Lane>
struct lane_properties {
    using transform_type = std::remove_cvref_t<decltype(std::declval<Lane>().transform)>;
    using reduce_type    = std::remove_cvref_t<decltype(std::declval<Lane>().reduce)>;
    using init_type      = std::remove_cvref_t<decltype(std::declval<Lane>().init)>;

    // Abstract declaration (type-independent)
    static constexpr bool associative =
        ct_dp::algebra::detail::has_declared_assoc<reduce_type>;
    static constexpr bool commutative =
        ct_dp::algebra::detail::has_declared_commut<reduce_type>;
    static constexpr bool idempotent =
        ct_dp::algebra::detail::has_declared_idemp<reduce_type>;
    static constexpr bool has_identity =
        ct_dp::algebra::has_identity<reduce_type, init_type>;

    // Type-safe (declaration AND init_type is exact)
    static constexpr bool exact_associative =
        ct_dp::algebra::declares_associative_v<reduce_type, init_type>;
    static constexpr bool exact_commutative =
        ct_dp::algebra::declares_commutative_v<reduce_type, init_type>;

    // Transform classification
    static constexpr bool identity_transform =
        ct_dp::algebra::is_identity_transform_v<transform_type>;
    static constexpr bool power_transform =
        ct_dp::algebra::is_power_transform_v<transform_type>
        && !ct_dp::algebra::is_identity_transform_v<transform_type>;

    static constexpr transform_cost cost = classify_transform<transform_type>();

    // Fusibility fingerprint: 4-bit encoding of algebraic signature.
    // Encoding v1: bit 0 = associative, bit 1 = commutative,
    //              bit 2 = has_identity, bit 3 = idempotent.
    // This is a versioned contract: adding a 5th property changes all
    // fingerprint values and invalidates any persisted partition data.
    static constexpr unsigned fingerprint =
        (associative  ? 1u : 0u) |
        (commutative  ? 2u : 0u) |
        (has_identity ? 4u : 0u) |
        (idempotent   ? 8u : 0u);
};

// ── Fold across all lanes ────────────────────────────────────────────

template <typename... Lanes, std::size_t... Is>
constexpr reduction_properties_t
summarise_lanes(const std::tuple<Lanes...>& /*lanes*/,
                std::index_sequence<Is...>) {
    static_assert(sizeof...(Lanes) <= reduction_properties_t::max_lanes,
        "reduction_properties: lane count exceeds max_lanes (16)");

    reduction_properties_t props;
    props.lane_count = sizeof...(Lanes);

    // ── Aggregate (abstract declaration) ──────────────────────
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

    // ── Type-safe aggregate ──────────────────────────────────
    props.all_exact_associative =
        (lane_properties<Lanes>::exact_associative && ...);
    props.all_exact_commutative =
        (lane_properties<Lanes>::exact_commutative && ...);

    // ── Per-lane arrays ──────────────────────────────────────
    ((props.lane_associative[Is]        = lane_properties<Lanes>::associative), ...);
    ((props.lane_commutative[Is]        = lane_properties<Lanes>::commutative), ...);
    ((props.lane_has_identity[Is]       = lane_properties<Lanes>::has_identity), ...);
    ((props.lane_idempotent[Is]         = lane_properties<Lanes>::idempotent), ...);
    ((props.lane_identity_transform[Is] = lane_properties<Lanes>::identity_transform), ...);
    ((props.lane_exact_associative[Is]  = lane_properties<Lanes>::exact_associative), ...);
    ((props.lane_exact_commutative[Is]  = lane_properties<Lanes>::exact_commutative), ...);
    ((props.lane_transform_cost[Is]     = lane_properties<Lanes>::cost), ...);
    ((props.lane_fingerprint[Is]        = lane_properties<Lanes>::fingerprint), ...);

    return props;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════

/// Extract aggregate and per-lane algebraic properties from a tuple_reduction.
template <typename... Lanes>
constexpr reduction_properties_t
reduction_properties(const ct_dp::algebra::tuple_reduction<Lanes...>& red) {
    return detail::summarise_lanes(red.lanes,
        std::index_sequence_for<Lanes...>{});
}

} // namespace ctdp::space

#endif // CTDP_SPACE_REDUCTION_PROPERTIES_H
