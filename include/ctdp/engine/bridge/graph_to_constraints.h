// engine/bridge/graph_to_constraints.h - Constraint extraction from graphs
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// The DP engine needs to know which schedules are legal. Constraints
// come from two sources:
//
// 1. DEPENDENCY CONSTRAINTS (from graph edges):
//    "Node u must complete before node v starts."
//    Encoded as (predecessor, successor) pairs.
//
// 2. RESOURCE CONSTRAINTS (from kernel_info):
//    "Total bytes in a time-slice must not exceed cache capacity."
//    "Total register pressure in a fused group must not exceed limit."
//
// This header extracts both kinds from a schedule_space and packages
// them for the DP solver's legality checker.
//
// USAGE:
// constexpr auto space = build_schedule_space<8, 16>(g, km);
// constexpr auto deps  = extract_dependencies<8, 16>(g, space);
// constexpr auto res   = resource_constraint{.max_bytes_per_group = 65536};
// static_assert(check_resource(space, res, /*group_id=*/0));

#ifndef CTDP_GRAPH_TO_CONSTRAINTS_H
#define CTDP_GRAPH_TO_CONSTRAINTS_H

#include "../../graph/graph_concepts.h"
#include "../../graph/constexpr_graph.h"
#include "graph_to_space.h"
#include "../../graph/kernel_info.h"
#include "../../graph/property_map.h"
#include "../../core/constexpr_vector.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// Dependency constraint: u must execute before v
// =============================================================================

/// A single dependency: predecessor must finish before successor starts.
struct dependency {
    std::uint16_t pred_rank;  // topo_rank of predecessor
    std::uint16_t succ_rank;  // topo_rank of successor

    friend constexpr bool operator==(dependency, dependency) = default;
};

/// All dependency constraints extracted from a graph.
template<std::size_t MaxE>
struct dependency_set {
    ctdp::constexpr_vector<dependency, MaxE> deps{};

    /// Number of dependencies.
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return deps.size();
    }

    /// Check if a proposed schedule respects all dependencies.
    ///
    /// schedule[i] = time slot assigned to node at topo_rank i.
    /// Legal if: for every dependency, schedule[pred] < schedule[succ].
    template<std::size_t MaxV>
    [[nodiscard]] constexpr bool
    is_legal(std::array<std::uint16_t, MaxV> const& schedule,
             std::size_t node_count) const noexcept {
        for (std::size_t i = 0; i < deps.size(); ++i) {
            auto const& d = deps[i];
            if (d.pred_rank < node_count && d.succ_rank < node_count) {
                if (schedule[d.pred_rank] >= schedule[d.succ_rank]) {
                    return false;
                }
            }
        }
        return true;
    }
};

/// Extract all dependency constraints from a graph + schedule_space.
///
/// Each directed edge u→v becomes a dependency:
///   (topo_rank(u), topo_rank(v))
///
/// Since the schedule_space is already in topo order, pred_rank < succ_rank
/// is guaranteed for DAGs.
template<std::size_t MaxV, std::size_t MaxE, graph_queryable G>
[[nodiscard]] constexpr dependency_set<MaxE>
extract_dependencies(G const& g, schedule_space<MaxV> const& space) {
    dependency_set<MaxE> result;
    auto const V = g.node_count();

    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        auto const u_rank = space.topo_index[u];
        for (auto v : g.out_neighbors(uid)) {
            auto const v_rank = space.topo_index[v.value];
            result.deps.push_back(dependency{u_rank, v_rank});
        }
    }

    return result;
}

// =============================================================================
// Resource constraints
// =============================================================================

/// Hardware resource limits for legality checking.
///
/// These model cache/register/bandwidth constraints that determine
/// whether a fusion group or time-slice is feasible.
struct resource_constraint {
    /// Maximum bytes of memory traffic per fusion group.
    /// 0 = unconstrained.
    std::size_t max_bytes_per_group = 0;

    /// Maximum flops per fusion group (throughput limit).
    /// 0 = unconstrained.
    std::size_t max_flops_per_group = 0;

    /// Maximum number of nodes per fusion group.
    /// 0 = unconstrained.
    std::size_t max_nodes_per_group = 0;

    /// Is this constraint satisfied by a group with given metrics?
    [[nodiscard]] constexpr bool
    check(std::size_t group_bytes,
          std::size_t group_flops,
          std::size_t group_nodes) const noexcept {
        if (max_bytes_per_group > 0 && group_bytes > max_bytes_per_group)
            return false;
        if (max_flops_per_group > 0 && group_flops > max_flops_per_group)
            return false;
        if (max_nodes_per_group > 0 && group_nodes > max_nodes_per_group)
            return false;
        return true;
    }

    friend constexpr bool
    operator==(resource_constraint const&, resource_constraint const&) = default;
};

/// Unconstrained: no resource limits.
inline constexpr resource_constraint unconstrained_resources{};

/// Check resource constraint for a specific fusion group in the space.
///
/// Aggregates flops/bytes/node_count for all nodes in the group and
/// checks against the resource constraint.
template<std::size_t MaxV>
[[nodiscard]] constexpr bool
check_resource(schedule_space<MaxV> const& space,
               resource_constraint const& rc,
               std::uint16_t group_id) noexcept {
    std::size_t bytes = 0;
    std::size_t flops = 0;
    std::size_t nodes = 0;

    for (std::size_t i = 0; i < space.size(); ++i) {
        if (space[i].group_id == group_id) {
            bytes += space[i].info.total_bytes();
            flops += space[i].info.flops;
            nodes++;
        }
    }

    return rc.check(bytes, flops, nodes);
}

/// Check resource constraints for ALL groups in the space.
///
/// Returns true only if every group satisfies the constraint.
template<std::size_t MaxV>
[[nodiscard]] constexpr bool
check_all_resources(schedule_space<MaxV> const& space,
                    resource_constraint const& rc) noexcept {
    for (std::size_t gid = 0; gid < space.group_count; ++gid) {
        if (!check_resource(space, rc,
                static_cast<std::uint16_t>(gid))) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Combined constraint check
// =============================================================================

/// Full constraint summary for a schedule_space.
template<std::size_t MaxV, std::size_t MaxE>
struct constraint_summary {
    dependency_set<MaxE> dependencies{};
    resource_constraint resources{};
    bool all_resources_ok = true;
    std::size_t dependency_count = 0;
    std::size_t critical_path_length = 0;  // longest chain in topo order
};

/// Build a complete constraint summary from graph + space.
///
/// Computes dependencies, checks resource constraints, and measures
/// the critical path (longest dependency chain).
template<std::size_t MaxV, std::size_t MaxE, graph_queryable G>
[[nodiscard]] constexpr constraint_summary<MaxV, MaxE>
build_constraints(G const& g,
                  schedule_space<MaxV> const& space,
                  resource_constraint const& rc = unconstrained_resources) {
    constraint_summary<MaxV, MaxE> result;

    // Dependencies.
    result.dependencies = extract_dependencies<MaxV, MaxE>(g, space);
    result.dependency_count = result.dependencies.size();
    result.resources = rc;

    // Resource check.
    result.all_resources_ok = check_all_resources(space, rc);

    // Critical path: longest chain measured in topo ranks.
    // For each node, dist[rank] = longest path ending at rank.
    std::array<std::uint16_t, MaxV> dist{};
    std::uint16_t max_dist = 0;

    for (std::size_t i = 0; i < result.dependencies.size(); ++i) {
        auto const& d = result.dependencies.deps[i];
        auto const candidate =
            static_cast<std::uint16_t>(dist[d.pred_rank] + 1);
        if (candidate > dist[d.succ_rank]) {
            dist[d.succ_rank] = candidate;
        }
        if (dist[d.succ_rank] > max_dist) {
            max_dist = dist[d.succ_rank];
        }
    }
    result.critical_path_length = max_dist;

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_TO_CONSTRAINTS_H
