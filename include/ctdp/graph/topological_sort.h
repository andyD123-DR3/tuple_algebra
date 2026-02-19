// graph/algorithms/topological_sort.h - Constexpr topological ordering
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Kahn's algorithm (BFS-based).
// Complexity: O(V + E)
// Determinism: when multiple nodes have in-degree 0, the smallest node_id
// is chosen first. This gives a unique, reproducible topological order.
//
// DESIGN RATIONALE:
// Kahn's (not DFS-based) because:
// - Naturally produces the order in forward sequence
// - Deterministic tie-breaking is trivial (scan for smallest ready node)
// - Detects cycles (if output size < V, graph has a cycle)
// - No recursion — constexpr-friendly

#ifndef CTDP_GRAPH_TOPOLOGICAL_SORT_H
#define CTDP_GRAPH_TOPOLOGICAL_SORT_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include <ctdp/core/constexpr_vector.h>
#include <ctdp/core/ct_limits.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Result of topological sort.
///
/// - order: nodes in topological order (dependencies before dependents)
/// - is_dag: true if graph is a DAG; false if cycle detected
///   When is_dag is false, order contains the partial ordering of
///   non-cyclic nodes (may be empty or incomplete).
template<std::size_t MaxV>
struct topo_result {
    ctdp::constexpr_vector<node_id, MaxV> order{};
    bool is_dag = true;
};

/// Topological sort via Kahn's algorithm.
///
/// Deterministic: when multiple nodes have in-degree 0, the smallest
/// node_id is chosen first.
///
/// Template parameters:
/// - G: graph type satisfying graph_queryable
/// - MaxV: maximum vertex count (default: ct_limits::topo_sort_max)
///
/// Example:
/// ```cpp
/// constexpr auto g = make_diamond();  // 0→1, 0→2, 1→3, 2→3
/// constexpr auto result = topological_sort(g);
/// static_assert(result.is_dag);
/// // result.order: {0, 1, 2, 3} (deterministic)
/// ```
template<graph_queryable G,
         std::size_t MaxV = ctdp::ct_limits::topo_sort_max>
[[nodiscard]] constexpr topo_result<MaxV>
topological_sort(G const& g) {
    guard_algorithm<MaxV>(g.node_count(), "topological_sort: V exceeds MaxV");
    topo_result<MaxV> result;
    auto const V = g.node_count();

    if (V == 0) {
        return result;
    }

    // Step 1: Compute in-degrees.
    std::array<std::uint16_t, MaxV> in_degree{};
    for (std::size_t u = 0; u < V; ++u) {
        for (auto v : g.out_neighbors(node_id{static_cast<std::uint16_t>(u)})) {
            in_degree[v.value]++;
        }
    }

    // Step 2: Track which nodes are ready (in-degree == 0) and not yet emitted.
    // We use a boolean array + linear scan for deterministic smallest-first.
    std::array<bool, MaxV> ready{};
    for (std::size_t u = 0; u < V; ++u) {
        if (in_degree[u] == 0) {
            ready[u] = true;
        }
    }

    // Step 3: Kahn's iteration.
    for (std::size_t iteration = 0; iteration < V; ++iteration) {
        // Find smallest ready node (deterministic tie-break).
        std::size_t chosen = V;  // sentinel: no node found
        for (std::size_t u = 0; u < V; ++u) {
            if (ready[u]) {
                chosen = u;
                break;
            }
        }

        if (chosen == V) {
            // No ready node but not all nodes emitted → cycle.
            result.is_dag = false;
            return result;
        }

        // Emit chosen node.
        ready[chosen] = false;
        result.order.push_back(node_id{static_cast<std::uint16_t>(chosen)});

        // Decrement in-degrees of successors.
        for (auto v : g.out_neighbors(node_id{static_cast<std::uint16_t>(chosen)})) {
            in_degree[v.value]--;
            if (in_degree[v.value] == 0) {
                ready[v.value] = true;
            }
        }
    }

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_TOPOLOGICAL_SORT_H
