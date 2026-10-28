// engine/bridge/segmentation_dp.h — Composite DP over graph partitions
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// After graph analysis (topo sort, fusion groups, coarsening), the
// optimization problem decomposes into independent per-segment sub-problems.
// Each segment (fusion group) can be optimized independently, then the
// segment-level solutions are combined respecting inter-segment constraints.
//
// This is the Phase 4 → Phase 5 bridge:
// - Phase 4 provides: graph, fusion groups, coarsened DAG, topo order
// - Phase 5 uses:     per-segment DP, merge rules, global assembly
//
// The segmentation_dp framework:
// 1. Partitions the schedule_space into per-group segments
// 2. Runs a user-supplied local solver on each segment
// 3. Assembles segment results into a global solution
// 4. Validates inter-segment constraints (dependency ordering)
//
// The local solver is a callable:
//   (segment_view) → segment_result<MaxV>
//
// This keeps segmentation_dp agnostic to the specific DP algorithm —
// the same framework works for per-element argmin, interval DP,
// beam search, or any other solver.
//
// COMPLEXITY: O(V + E + G * local_solver_cost)
//   where G = number of segments, V = total nodes, E = total edges.

#ifndef CTDP_GRAPH_SEGMENTATION_DP_H
#define CTDP_GRAPH_SEGMENTATION_DP_H

#include "../../graph/capacity_guard.h"
#include "../../graph/graph_concepts.h"
#include "../../graph/graph_traits.h"
#include "../../graph/property_map.h"
#include "../../graph/topological_sort.h"
#include "graph_to_space.h"
#include "graph_to_constraints.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ctdp::graph {

// =============================================================================
// Segment view: read-only slice of a schedule_space for one group
// =============================================================================

/// A view into a schedule_space for a single fusion group.
///
/// Contains the descriptors belonging to one group, in topo order.
/// Passed to the local solver so it sees only its own segment.
template<std::size_t MaxV>
struct segment_view {
    std::uint16_t group_id = 0;
    ctdp::constexpr_vector<node_descriptor, MaxV> nodes{};

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return nodes.size();
    }

    [[nodiscard]] constexpr node_descriptor const&
    operator[](std::size_t i) const noexcept {
        return nodes[i];
    }

    /// Total flops in this segment.
    [[nodiscard]] constexpr std::size_t total_flops() const noexcept {
        std::size_t s = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i)
            s += nodes[i].info.flops;
        return s;
    }

    /// Total bytes in this segment.
    [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
        std::size_t s = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i)
            s += nodes[i].info.total_bytes();
        return s;
    }
};

// =============================================================================
// Segment result: output of a local solver for one group
// =============================================================================

/// Result of solving a single segment.
///
/// The local solver returns:
/// - cost: the optimized cost for this segment
/// - choice: per-node strategy index (meaning is solver-specific)
/// - feasible: whether a valid solution was found
template<std::size_t MaxV>
struct segment_result {
    std::uint16_t group_id = 0;
    double cost = std::numeric_limits<double>::infinity();
    std::array<std::uint16_t, MaxV> choice{};
    bool feasible = true;
};

// =============================================================================
// Global solution: assembled from segment results
// =============================================================================

/// The assembled global solution from all segments.
///
/// Contains per-node choices, per-segment costs, and the total cost.
template<std::size_t MaxV>
struct segmented_solution {
    /// Per-node strategy choice (indexed by topo_rank).
    std::array<std::uint16_t, MaxV> choice{};

    /// Per-segment cost (indexed by group_id).
    std::array<double, MaxV> segment_cost{};

    /// Total cost (sum of segment costs).
    double total_cost = 0.0;

    /// Number of segments.
    std::size_t segment_count = 0;

    /// True if all segments found feasible solutions.
    bool all_feasible = true;

    /// Number of segments that were infeasible.
    std::size_t infeasible_count = 0;
};

// Finite metric guard usable in C++20 constant evaluation.
[[nodiscard]] constexpr bool finite_segment_cost(double x) noexcept {
    constexpr auto inf = std::numeric_limits<double>::infinity();
    return x == x && x != inf && x != -inf;
}

// =============================================================================
// Segmentation engine: partition → solve → assemble
// =============================================================================

/// Partition a schedule_space into per-group segment views.
///
/// Returns an array of segment_views, one per group, each containing
/// the descriptors belonging to that group in topo order.
template<std::size_t MaxV>
[[nodiscard]] constexpr
std::array<segment_view<MaxV>, MaxV>
partition_segments(schedule_space<MaxV> const& space) {
    std::array<segment_view<MaxV>, MaxV> segments{};

    for (std::size_t g = 0; g < space.group_count; ++g) {
        segments[g].group_id = static_cast<std::uint16_t>(g);
    }

    // Descriptors are already in topo order, so segment ordering is preserved.
    for (std::size_t i = 0; i < space.size(); ++i) {
        auto const gid = space[i].group_id;
        segments[gid].nodes.push_back(space[i]);
    }

    return segments;
}

/// Run segmented DP: partition the space, solve each segment, assemble.
///
/// Template parameters:
/// - MaxV: vertex capacity
/// - Solver: callable (segment_view<MaxV> const&) → segment_result<MaxV>
///
/// Parameters:
/// - space: the schedule_space (from build_schedule_space or _fused)
/// - solver: local per-segment solver
///
/// Returns: segmented_solution with per-node choices and total cost.
///
/// Example:
/// ```cpp
/// // A trivial solver that picks strategy 0 for every node
/// // and costs each node by its flops.
/// constexpr auto trivial_solver = [](auto const& seg) {
///     segment_result<8> r;
///     r.group_id = seg.group_id;
///     r.cost = 0.0;
///     for (std::size_t i = 0; i < seg.size(); ++i) {
///         r.choice[i] = 0;
///         r.cost += static_cast<double>(seg[i].info.flops);
///     }
///     r.feasible = true;
///     return r;
/// };
///
/// constexpr auto sol = solve_segmented(space, trivial_solver);
/// static_assert(sol.all_feasible);
/// ```
template<std::size_t MaxV, typename Solver>
[[nodiscard]] constexpr segmented_solution<MaxV>
solve_segmented(schedule_space<MaxV> const& space,
                Solver solver) {
    segmented_solution<MaxV> sol{};
    sol.segment_count = space.group_count;

    auto const segments = partition_segments(space);

    for (std::size_t g = 0; g < space.group_count; ++g) {
        auto const& seg = segments[g];
        auto const sr = solver(seg);

        sol.segment_cost[g] = sr.cost;
        sol.total_cost += sr.cost;

        if (!sr.feasible || !finite_segment_cost(sr.cost)) {
            sol.all_feasible = false;
            sol.infeasible_count++;
        }

        // Map local choices back to global topo-rank indices.
        for (std::size_t i = 0; i < seg.size(); ++i) {
            auto const rank = seg[i].topo_rank;
            sol.choice[rank] = sr.choice[i];
        }
    }

    return sol;
}

/// Basic validation of an assembled segmented result.
/// This rejects infeasible segments, nonfinite costs, inconsistent cost sums
/// and impossible dependency ranks. It does NOT prove semantic compatibility
/// of user-defined strategies: see the overload accepting the schedule and
/// resource constraints for additional (still necessary, not sufficient) checks.
template<std::size_t MaxV, std::size_t MaxE>
[[nodiscard]] constexpr bool
validate_segmented(segmented_solution<MaxV> const& sol,
                   dependency_set<MaxE> const& deps) noexcept {
    if (!sol.all_feasible || sol.infeasible_count != 0 ||
        sol.segment_count > MaxV || !finite_segment_cost(sol.total_cost))
        return false;
    double sum = 0.0;
    for (std::size_t i = 0; i < sol.segment_count; ++i) {
        if (!finite_segment_cost(sol.segment_cost[i]))
            return false;
        sum += sol.segment_cost[i];
    }
    if (sum != sol.total_cost) return false;
    for (std::size_t i = 0; i < deps.size(); ++i) {
        auto const& d = deps.deps[i];
        if (d.pred_rank >= MaxV || d.succ_rank >= MaxV ||
            d.pred_rank >= d.succ_rank)
            return false;
    }
    return true;
}

/// Additional global structural/resource validation. User-defined strategy
/// combinations and observation contracts must still be checked by the
/// domain-specific executor before considering a plan admitted.
template<std::size_t MaxV, std::size_t MaxE>
[[nodiscard]] constexpr bool
validate_segmented(segmented_solution<MaxV> const& sol,
                   dependency_set<MaxE> const& deps,
                   schedule_space<MaxV> const& space,
                   resource_constraint const& resources = unconstrained_resources) noexcept {
    if (!validate_segmented(sol, deps) || !space.is_dag ||
        space.group_count != sol.segment_count)
        return false;
    for (std::size_t i = 0; i < deps.size(); ++i) {
        if (deps.deps[i].pred_rank >= space.size() ||
            deps.deps[i].succ_rank >= space.size())
            return false;
    }
    return check_all_resources(space, resources);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SEGMENTATION_DP_H
