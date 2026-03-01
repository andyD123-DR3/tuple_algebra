// graph/transforms/subgraph.h — Induced subgraph extraction
// Part of the compile-time DP library (C++20)
//
// ALGORITHM:
// Given a directed graph G and a predicate P over node_ids, produce the
// induced subgraph G' containing:
// - All nodes u where P(u) is true
// - All edges (u→v) from G where both P(u) and P(v) are true
//
// Nodes in G' are renumbered contiguously from 0.  The result includes
// a forward mapping (old_node → new_node) and an inverse mapping
// (new_node → old_node) so callers can translate between the two
// descriptor spaces.
//
// COMPLEXITY: O(V + E)
//
// DESIGN RATIONALE:
// Subgraph extraction is the complement of coarsening:
// - coarsen() *merges* groups of nodes into super-nodes
// - subgraph() *extracts* a subset without merging
//
// Use cases:
// - Focus analysis on a connected component
// - Strip I/O nodes before scheduling compute kernels
// - Extract a pipeline stage for separate optimisation
//
// The predicate is a callable taking node_id, returning bool.
// This matches the functional style used throughout CT-DP
// (everything is a callable — cost functions, constraints, predicates).

#ifndef CTDP_GRAPH_SUBGRAPH_H
#define CTDP_GRAPH_SUBGRAPH_H

#include "capacity_guard.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_concepts.h"
#include "graph_traits.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Result of induced subgraph extraction.
///
/// Contains the subgraph, the number of retained nodes, and bidirectional
/// node mappings between the original and subgraph descriptor spaces.
///
/// - forward_map[old_id] = new_id   (or invalid_node if not retained)
/// - inverse_map[new_id] = old_id
template<std::size_t MaxV, std::size_t MaxE>
struct subgraph_result {
    constexpr_graph<cap_from<MaxV, MaxE>> graph{};
    std::size_t retained_count = 0;

    /// old → new.  forward_map[i] == invalid_node if node i was excluded.
    std::array<node_id, MaxV> forward_map{};

    /// new → old.  Only the first retained_count entries are valid.
    std::array<node_id, MaxV> inverse_map{};
};

/// Deduction alias: subgraph_result sized from graph traits.
template<typename G>
using subgraph_result_for = subgraph_result<
    graph_traits<G>::max_nodes,
    graph_traits<G>::max_edges>;

/// Extract the induced subgraph containing nodes that satisfy a predicate.
///
/// Template parameters:
/// - Cap: capacity policy for the output graph
/// - G: source graph type (must satisfy graph_queryable)
/// - Pred: callable (node_id) → bool
///
/// Parameters:
/// - g: the source graph
/// - pred: predicate selecting which nodes to retain
///
/// Returns: subgraph_result containing the induced subgraph and
/// bidirectional node mappings.
///
/// Example:
/// ```cpp
/// // Diamond: 0→1, 0→2, 1→3, 2→3
/// constexpr auto g = make_diamond();
/// // Keep only the left branch: nodes 0, 1, 3
/// constexpr auto sr = induced_subgraph(g,
///     [](node_id n) { return n.value != 2; });
/// static_assert(sr.retained_count == 3);
/// static_assert(sr.graph.node_count() == 3);
/// static_assert(sr.graph.edge_count() == 2); // 0→1, 1→3 (renumbered)
/// ```
template<typename Cap, graph_queryable G, typename Pred>
[[nodiscard]] constexpr subgraph_result<Cap::max_v, Cap::max_e>
induced_subgraph(G const& g, Pred pred) {
    constexpr std::size_t MaxV = Cap::max_v;
    constexpr std::size_t MaxE = Cap::max_e;
    guard_algorithm<MaxV>(g.node_count(), "induced_subgraph: V exceeds Cap::max_v");

    auto const V = g.node_count();
    subgraph_result<MaxV, MaxE> result{};

    // Initialise forward_map to invalid_node.
    for (std::size_t i = 0; i < MaxV; ++i) {
        result.forward_map[i] = invalid_node;
    }

    // Pass 1: identify retained nodes and build mappings.
    std::size_t new_count = 0;
    for (std::size_t i = 0; i < V; ++i) {
        auto const nid = node_id{static_cast<std::uint16_t>(i)};
        if (pred(nid)) {
            auto const new_id = node_id{static_cast<std::uint16_t>(new_count)};
            result.forward_map[i] = new_id;
            result.inverse_map[new_count] = nid;
            ++new_count;
        }
    }
    result.retained_count = new_count;

    // Pass 2: build subgraph using the builder.
    graph_builder<cap_from<MaxV, MaxE>> builder;
    for (std::size_t i = 0; i < new_count; ++i) {
        (void)builder.add_node();
    }

    for (std::size_t u = 0; u < V; ++u) {
        if (result.forward_map[u] == invalid_node) continue;

        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        auto const new_u = result.forward_map[u];

        for (auto v : g.out_neighbors(uid)) {
            auto const new_v = result.forward_map[to_index(v)];
            if (new_v != invalid_node) {
                builder.add_edge(new_u, new_v);
            }
        }
    }

    result.graph = builder.finalise();
    return result;
}

/// Convenience overload: deduce Cap from the graph's traits.
template<graph_queryable G, typename Pred>
[[nodiscard]] constexpr auto induced_subgraph(G const& g, Pred pred) {
    using traits = graph_traits<G>;
    using out_cap = cap_from<traits::max_nodes, traits::max_edges>;
    return induced_subgraph<out_cap>(g, pred);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SUBGRAPH_H
