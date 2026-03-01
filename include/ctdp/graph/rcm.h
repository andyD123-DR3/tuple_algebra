// graph/algorithms/rcm.h - Constexpr Reverse Cuthill-McKee bandwidth reduction
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Reverse Cuthill-McKee (RCM) reordering.
//
// Given a symmetric (undirected) graph, RCM produces a node permutation
// that reduces the bandwidth of the adjacency matrix.  Bandwidth is
// defined as:  max |perm[u] - perm[v]|  over all edges (u, v).
//
// The algorithm:
//   1. Find a pseudo-peripheral starting node via two BFS rounds.
//   2. BFS from that node, visiting neighbours in order of increasing
//      degree (Cuthill-McKee ordering).
//   3. Reverse the resulting order (the "R" in RCM).
//
// Disconnected graphs are handled: each connected component is
// discovered and BFS'd separately, so every node receives a position.
//
// Complexity: O(V + E) for the BFS traversals, O(V * max_degree) for
//             the per-level degree sorting (selection sort within each
//             BFS level, bounded by the level width × degree).
//
// GUARANTEES:
// - Deterministic: ties in degree broken by ascending node_id
// - Correct: verify_rcm() proves the reported bandwidth matches the
//            permutation applied to the actual edge set
// - Handles disconnected graphs (multiple components)
//
// DESIGN RATIONALE:
// Bandwidth reduction is the undirected analogue of topological sort —
// it finds a node ordering that localises adjacency.  In optimisation,
// this directly improves cache performance for sparse matrix operations
// (SpMV, triangular solve) and reduces fill-in for direct solvers.
//
// Requires symmetric_graph_queryable because bandwidth is an undirected
// property: the adjacency matrix must be symmetric for |i-j| to be
// meaningful.
//
// TRAITS RETROFIT:
// MaxV derived from graph_traits<G>::max_nodes.  node_index_t<G> used
// for permutation arrays and working storage.

#ifndef CTDP_GRAPH_RCM_H
#define CTDP_GRAPH_RCM_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include "symmetric_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =========================================================================
// Result type
// =========================================================================

/// Result of Reverse Cuthill-McKee reordering.
///
/// - permutation[old] = new:  maps original node id → RCM position
/// - inverse[new] = old:      maps RCM position → original node id
/// - bandwidth_before:        bandwidth under the identity permutation
/// - bandwidth_after:         bandwidth under the RCM permutation
/// - node_count:              number of nodes in the graph
/// - verified:                true if verify_rcm() confirmed correctness
template<std::size_t MaxV>
struct rcm_result {
    std::array<std::uint16_t, MaxV> permutation{};
    std::array<std::uint16_t, MaxV> inverse{};
    std::size_t bandwidth_before = 0;
    std::size_t bandwidth_after  = 0;
    std::size_t node_count       = 0;
    bool verified                = false;
};

/// Factory: construct a default rcm_result sized for graph g.
template<typename G>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr auto make_rcm_result(G const& /*g*/) {
    return rcm_result<graph_traits<G>::max_nodes>{};
}

// =========================================================================
// Bandwidth computation
// =========================================================================

namespace detail {

/// Compute the bandwidth of a symmetric graph under a given permutation.
///
/// bandwidth = max |perm[u] - perm[v]|  over all edges (u, v)
///
/// For the identity permutation, pass perm[i] = i.
template<std::size_t MaxV, symmetric_graph_queryable G>
[[nodiscard]] constexpr std::size_t
compute_bandwidth(G const& g,
                  std::array<std::uint16_t, MaxV> const& perm) {
    std::size_t bw = 0;
    auto const V = g.node_count();
    for (std::size_t u = 0; u < V; ++u) {
        auto uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.neighbors(uid)) {
            auto pu = static_cast<std::size_t>(perm[u]);
            auto pv = static_cast<std::size_t>(perm[v.value]);
            auto diff = pu > pv ? pu - pv : pv - pu;
            if (diff > bw) bw = diff;
        }
    }
    return bw;
}

/// Find the node with minimum degree in the graph.
/// Ties broken by smallest node_id.
template<symmetric_graph_queryable G>
[[nodiscard]] constexpr std::uint16_t
find_min_degree_node(G const& g) {
    auto const V = g.node_count();
    std::uint16_t best = 0;
    std::size_t best_deg = g.degree(node_id{0});
    for (std::size_t i = 1; i < V; ++i) {
        auto d = g.degree(node_id{static_cast<std::uint16_t>(i)});
        if (d < best_deg) {
            best_deg = d;
            best = static_cast<std::uint16_t>(i);
        }
    }
    return best;
}

/// BFS from a source node, returning the last node visited (most distant).
/// Used to find a pseudo-peripheral node.
template<std::size_t MaxV, symmetric_graph_queryable G>
[[nodiscard]] constexpr std::uint16_t
bfs_farthest(G const& g, std::uint16_t source) {
    auto const V = g.node_count();
    constexpr std::uint16_t NIL = 0xFFFF;

    std::array<std::uint16_t, MaxV> queue{};
    std::array<bool, MaxV> visited{};
    for (std::size_t i = 0; i < V; ++i) visited[i] = false;

    std::size_t head = 0, tail = 0;
    queue[tail++] = source;
    visited[source] = true;
    std::uint16_t last = source;

    while (head < tail) {
        auto u = queue[head++];
        last = u;
        for (auto nb : g.neighbors(node_id{u})) {
            if (!visited[nb.value]) {
                visited[nb.value] = true;
                queue[tail++] = nb.value;
            }
        }
    }
    return last;
}

/// Find a pseudo-peripheral starting node for RCM.
///
/// Strategy: start from the minimum-degree node, BFS to find the
/// farthest node, then BFS again from there.  The farthest node
/// from the second BFS is a good pseudo-peripheral candidate.
/// We use the starting node of the second BFS (i.e. the farthest
/// from the min-degree node) as the RCM root.
template<std::size_t MaxV, symmetric_graph_queryable G>
[[nodiscard]] constexpr std::uint16_t
find_pseudo_peripheral(G const& g) {
    auto start = find_min_degree_node(g);
    auto far1 = bfs_farthest<MaxV>(g, start);
    // far1 is a reasonable pseudo-peripheral node.
    // A second round refines it.
    auto far2 = bfs_farthest<MaxV>(g, far1);
    // Use far1 as the BFS root: it is far from far2, hence peripheral.
    return far1;
}

/// Constexpr selection sort of a sub-range by degree (ascending),
/// ties broken by ascending node_id.  Operates on queue[begin..end).
template<symmetric_graph_queryable G, std::size_t MaxV>
constexpr void
sort_by_degree(G const& g,
               std::array<std::uint16_t, MaxV>& arr,
               std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
        std::size_t min_idx = i;
        auto min_deg = g.degree(node_id{arr[i]});
        for (std::size_t j = i + 1; j < end; ++j) {
            auto dj = g.degree(node_id{arr[j]});
            if (dj < min_deg || (dj == min_deg && arr[j] < arr[min_idx])) {
                min_idx = j;
                min_deg = dj;
            }
        }
        if (min_idx != i) {
            auto tmp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = tmp;
        }
    }
}

} // namespace detail

// =========================================================================
// Post-condition: verify RCM result
// =========================================================================

/// O(E) verification: recompute bandwidth under the RCM permutation
/// and check it matches the reported value.  Also checks that
/// permutation and inverse are consistent bijections.
template<std::size_t MaxV, symmetric_graph_queryable G>
[[nodiscard]] constexpr bool
verify_rcm(G const& g, rcm_result<MaxV>& result) {
    auto const V = g.node_count();

    // Check bijection: perm and inverse must be consistent.
    for (std::size_t i = 0; i < V; ++i) {
        if (result.inverse[result.permutation[i]] != i) {
            result.verified = false;
            return false;
        }
        if (result.permutation[result.inverse[i]] != i) {
            result.verified = false;
            return false;
        }
    }

    // Check all positions are in range [0, V).
    for (std::size_t i = 0; i < V; ++i) {
        if (result.permutation[i] >= V || result.inverse[i] >= V) {
            result.verified = false;
            return false;
        }
    }

    // Recompute bandwidth and compare.
    auto bw = detail::compute_bandwidth<MaxV>(g, result.permutation);
    if (bw != result.bandwidth_after) {
        result.verified = false;
        return false;
    }

    result.verified = true;
    return true;
}

// =========================================================================
// Reverse Cuthill-McKee algorithm
// =========================================================================

/// Reverse Cuthill-McKee reordering for bandwidth reduction.
///
/// MaxV is derived from graph_traits<G>::max_nodes.
/// Requires symmetric_graph_queryable AND sized_graph.
///
/// The algorithm:
///   1. Find a pseudo-peripheral starting node (two BFS rounds).
///   2. BFS from that node, visiting neighbours in ascending-degree order.
///   3. Reverse the BFS ordering to get the RCM permutation.
///   4. Compute bandwidth before and after.
///   5. Verify the result.
///
/// Disconnected graphs: each connected component is handled separately.
/// Components are processed in order of their pseudo-peripheral node.
///
/// Example:
/// ```cpp
/// constexpr auto sg = build_banded_graph();
/// constexpr auto r = rcm(sg);
/// static_assert(r.bandwidth_after <= r.bandwidth_before);
/// static_assert(r.verified);
/// ```
template<typename G>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr auto
rcm(G const& g) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;

    guard_algorithm<MaxV>(g.node_count(), "rcm: V exceeds MaxV");
    rcm_result<MaxV> result;
    auto const V = g.node_count();
    result.node_count = V;

    if (V == 0) {
        result.verified = true;
        return result;
    }

    // Single-node graph.
    if (V == 1) {
        result.permutation[0] = 0;
        result.inverse[0] = 0;
        result.bandwidth_before = 0;
        result.bandwidth_after = 0;
        result.verified = true;
        return result;
    }

    // Identity permutation for "before" bandwidth.
    std::array<std::uint16_t, MaxV> identity{};
    for (std::size_t i = 0; i < V; ++i)
        identity[i] = static_cast<std::uint16_t>(i);
    result.bandwidth_before = detail::compute_bandwidth<MaxV>(g, identity);

    // --- Cuthill-McKee BFS ---
    // cm_order[k] = original node id in position k (CM ordering, pre-reversal).
    std::array<std::uint16_t, MaxV> cm_order{};
    std::array<bool, MaxV> visited{};
    for (std::size_t i = 0; i < V; ++i) visited[i] = false;

    std::size_t cm_pos = 0;   // next write position in cm_order

    // Process each connected component.
    for (std::size_t seed = 0; seed < V; ++seed) {
        if (visited[seed]) continue;

        // Find pseudo-peripheral node within this component.
        // First, BFS from seed to find farthest node (component-aware).
        auto far1 = detail::bfs_farthest<MaxV>(g, static_cast<std::uint16_t>(seed));
        auto far2 = detail::bfs_farthest<MaxV>(g, far1);
        // Use far1 as starting node for CM BFS.
        auto start = far1;

        // BFS with degree-ordered expansion.
        // We use cm_order itself as the BFS queue (tail is cm_pos + pending).
        std::size_t head = cm_pos;
        cm_order[cm_pos] = start;
        visited[start] = true;
        std::size_t tail = cm_pos + 1;

        while (head < tail) {
            auto u = cm_order[head++];

            // Collect unvisited neighbours into a temporary buffer.
            std::size_t nbr_start = tail;
            for (auto nb : g.neighbors(node_id{u})) {
                if (!visited[nb.value]) {
                    visited[nb.value] = true;
                    cm_order[tail++] = nb.value;
                }
            }
            // Sort the newly added neighbours by ascending degree.
            detail::sort_by_degree<G, MaxV>(g, cm_order, nbr_start, tail);
        }

        cm_pos = tail;
    }

    // --- Reverse to get RCM ordering ---
    // RCM: inverse[k] = cm_order[V - 1 - k]
    for (std::size_t k = 0; k < V; ++k) {
        auto old_node = cm_order[V - 1 - k];
        result.inverse[k] = old_node;
        result.permutation[old_node] = static_cast<std::uint16_t>(k);
    }

    // Compute bandwidth under the RCM permutation.
    result.bandwidth_after =
        detail::compute_bandwidth<MaxV>(g, result.permutation);

    // Verify.
    (void)verify_rcm<MaxV>(g, result);

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_RCM_H
