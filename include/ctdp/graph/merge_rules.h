// graph/transforms/merge_rules.h — Named merge policies for property combination
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// When nodes are fused (via coarsen/contract), their properties must be
// combined.  kernel_info::merged_with() hardcodes one policy (sum flops,
// sum bytes, AND fusability).  But different properties need different
// merge strategies:
//
//   - Latency:      max (critical path dominates)
//   - Bandwidth:    sum (traffic is additive)
//   - Precision:    stricter (must satisfy all fused nodes)
//   - Feature sets: union (fused node has all features)
//   - Validity:     and (all nodes must be valid)
//
// This header provides a vocabulary of 10 named merge policies, each a
// constexpr callable: (T, T) → T.  They compose with property_map and
// fuse_group_result to merge any external annotation during coarsening.
//
// The merge_property() free function applies a policy across a group,
// reducing all values in the group to a single merged result.
//
// POLICIES:
//   merge::max_of      — std::max(a, b)
//   merge::min_of      — std::min(a, b)
//   merge::sum         — a + b
//   merge::union_of    — a | b     (bitwise OR for flag sets)
//   merge::intersect   — a & b     (bitwise AND for flag sets)
//   merge::logical_and — a && b
//   merge::logical_or  — a || b
//   merge::stricter    — min for numeric limits (same as min_of)
//   merge::first       — keep a    (first encountered wins)
//   merge::second      — keep b    (last encountered wins)
//
// Plus a sentinel:
//   merge::fail        — always throws (property must not be merged)

#ifndef CTDP_GRAPH_MERGE_RULES_H
#define CTDP_GRAPH_MERGE_RULES_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "property_map.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ctdp::graph {

// =============================================================================
// Named merge policies
// =============================================================================

namespace merge {

/// Keep the maximum of two values (e.g., latency, critical path).
struct max_of {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& a, T const& b) const noexcept {
        return (a >= b) ? a : b;
    }
};

/// Keep the minimum of two values (e.g., deadlines, cache line size).
struct min_of {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& a, T const& b) const noexcept {
        return (a <= b) ? a : b;
    }
};

/// Sum two values (e.g., flops, bytes, counts).
struct sum {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& a, T const& b) const noexcept {
        return a + b;
    }
};

/// Bitwise OR (e.g., capability flags, feature bitmasks).
struct union_of {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& a, T const& b) const noexcept {
        return a | b;
    }
};

/// Bitwise AND (e.g., required feature intersection).
struct intersect {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& a, T const& b) const noexcept {
        return a & b;
    }
};

/// Logical AND (e.g., is_fusable, is_valid).
struct logical_and {
    [[nodiscard]] constexpr bool operator()(bool a, bool b) const noexcept {
        return a && b;
    }
};

/// Logical OR (e.g., needs_barrier, has_side_effect).
struct logical_or {
    [[nodiscard]] constexpr bool operator()(bool a, bool b) const noexcept {
        return a || b;
    }
};

/// Stricter of two limits — alias for min_of.
/// Semantic intent: when fusing nodes with different constraints,
/// the merged constraint is the tightest (smallest limit).
using stricter = min_of;

/// Keep the first value encountered (left operand).
struct first {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& a, T const& /*b*/) const noexcept {
        return a;
    }
};

/// Keep the last value encountered (right operand).
struct second {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& /*a*/, T const& b) const noexcept {
        return b;
    }
};

/// Sentinel: property must not be merged.  Throws at merge time.
/// Use for properties where merging is a logic error (e.g., unique IDs).
struct fail {
    template<typename T>
    [[nodiscard]] constexpr T operator()(T const& /*a*/, T const& /*b*/) const {
        throw std::logic_error(
            "merge::fail: property must not be merged during fusion");
    }
};

} // namespace merge

// =============================================================================
// merge_property: apply a policy across a fusion group
// =============================================================================

/// Merge property values for all nodes in a group using a given policy.
///
/// Template parameters:
/// - Value: property value type
/// - MaxV: capacity of the property_map
/// - Policy: merge policy callable (Value, Value) → Value
///
/// Parameters:
/// - pmap: property_map containing per-node values
/// - group_of: maps each node to its group id
/// - group_count: number of groups
/// - policy: the merge rule to apply
///
/// Returns: property_map<Value, MaxV> with group_count entries, where
/// each entry is the merged value for that group.
///
/// Example:
/// ```cpp
/// // Merge latencies using max_of across fusion groups
/// constexpr auto merged_lat = merge_property(
///     latency_map, fg.group_of, fg.group_count, merge::max_of{});
/// // merged_lat[0] = max latency in group 0
/// ```
template<typename Value, std::size_t MaxV, typename Policy>
[[nodiscard]] constexpr property_map<Value, MaxV>
merge_property(property_map<Value, MaxV> const& pmap,
               property_map<std::uint16_t, MaxV> const& group_of,
               std::size_t group_count,
               Policy policy) {
    property_map<Value, MaxV> result;
    result.resize(group_count);

    // Track which groups have been initialised.
    std::array<bool, MaxV> initialised{};
    for (std::size_t i = 0; i < MaxV; ++i) {
        initialised[i] = false;
    }

    for (std::size_t i = 0; i < pmap.size(); ++i) {
        auto const grp = static_cast<std::size_t>(group_of[i]);
        if (!initialised[grp]) {
            result[grp] = pmap[i];
            initialised[grp] = true;
        } else {
            result[grp] = policy(result[grp], pmap[i]);
        }
    }

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_MERGE_RULES_H
