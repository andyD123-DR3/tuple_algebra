// graph/transforms/transpose.h — Reverse all edge directions
// Part of the compile-time DP library (C++20)
//
// ALGORITHM:
// Given a directed graph G, produce Gᵀ where every edge (u→v) in G
// becomes (v→u) in Gᵀ.  Node count, edge count, and capacity are
// preserved.  Node IDs are identity-mapped.
//
// COMPLEXITY: O(V + E)
//
// DESIGN RATIONALE:
// Transpose is a fundamental primitive needed for:
// - Reverse reachability (which nodes can reach a target?)
// - In-degree computation
// - Bi-directional Dijkstra preparation
// - SCC verification (Kosaraju's second pass)
//
// The implementation uses graph_builder::finalise() which canonicalises
// the result (sorted, deduplicated, no self-edges).  Since the input
// graph is already canonical, transpose preserves edge count exactly.

#ifndef CTDP_GRAPH_TRANSPOSE_H
#define CTDP_GRAPH_TRANSPOSE_H

#include "capacity_guard.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_concepts.h"
#include "graph_traits.h"

#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Transpose a directed graph: reverse every edge direction.
///
/// Returns a new constexpr_graph<Cap> with the same node count and
/// edge count, but all edges reversed.  Node IDs are identity-mapped
/// (node i in the input corresponds to node i in the output).
///
/// Template parameters:
/// - Cap: capacity policy for the output graph (defaults to input's Cap)
///
/// Preconditions:
/// - G satisfies graph_queryable
/// - G::node_count() <= Cap::max_v
///
/// Example:
/// ```cpp
/// // Diamond: 0→1, 0→2, 1→3, 2→3
/// constexpr auto g = make_diamond();
/// constexpr auto gt = transpose(g);
/// // gt has edges: 1→0, 2→0, 3→1, 3→2
/// static_assert(gt.node_count() == 4);
/// static_assert(gt.edge_count() == 4);
/// ```
template<typename Cap, graph_queryable G>
[[nodiscard]] constexpr constexpr_graph<Cap>
transpose(G const& g) {
    constexpr std::size_t MaxV = Cap::max_v;
    guard_algorithm<MaxV>(g.node_count(), "transpose: V exceeds Cap::max_v");

    auto const V = g.node_count();

    graph_builder<Cap> builder;
    for (std::size_t i = 0; i < V; ++i) {
        (void)builder.add_node();
    }

    // Reverse every edge: (u→v) becomes (v→u).
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            builder.add_edge(v, uid);  // reversed
        }
    }

    return builder.finalise();
}

/// Convenience overload: deduce Cap from the graph's traits.
///
/// This overload uses the same capacity as the input graph, which is
/// always safe since transpose preserves node and edge counts.
template<graph_queryable G>
[[nodiscard]] constexpr auto transpose(G const& g) {
    using traits = graph_traits<G>;
    using out_cap = cap_from<traits::max_nodes, traits::max_edges>;
    return transpose<out_cap>(g);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_TRANSPOSE_H
