// graph/algorithms/connected_components.h - Weakly connected components
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Union-Find with path compression and union by rank.
// Complexity: O(V + E · alpha(V)) ~ O(V + E) (amortised)
//
// SEMANTICS: Weakly connected components -- edge direction is ignored.
// For a directed graph, nodes u and v are in the same component if
// there exists an undirected path between them (ignoring arrow direction).
//
// DESIGN RATIONALE:
// Union-Find (not BFS/DFS) because:
// - Avoids building transpose graph or scanning for in-neighbors
// - Near-linear time
// - Simple constexpr implementation
// - No recursion or stack management needed
//
// TRAITS RETROFIT (Phase 7):
// MaxV derived from graph_traits<G>::max_nodes.  Working arrays use
// the compile-time max_nodes capacity.  node_index_t<G> used for
// component ids and internal union-find arrays.

#ifndef CTDP_GRAPH_CONNECTED_COMPONENTS_H
#define CTDP_GRAPH_CONNECTED_COMPONENTS_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "graph_traits.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Result of connected components analysis.
///
/// - component_of[n]: component id for node n (0-based, dense)
/// - component_count: total number of connected components
///
/// Component ids are renumbered to [0, component_count) in ascending
/// order of the smallest node_id in each component.
template<std::size_t MaxV>
struct components_result {
    std::array<std::uint16_t, MaxV> component_of{};
    std::size_t component_count = 0;
};

/// Factory: construct a default components_result sized for graph g.
template<sized_graph G>
[[nodiscard]] constexpr auto make_components_result(G const& /*g*/) {
    return components_result<graph_traits<G>::max_nodes>{};
}

/// Weakly connected components via Union-Find.
///
/// Treats edges as undirected: nodes connected by a directed edge in
/// either direction belong to the same component.
///
/// MaxV is derived from graph_traits<G>::max_nodes.
/// Requires sized_graph (capacity queries for array sizing).
///
/// Example:
/// ```cpp
/// constexpr auto g = make_disconnected();  // 0->1, 2->3
/// constexpr auto result = connected_components(g);
/// static_assert(result.component_count == 2);
/// ```
template<sized_graph G>
[[nodiscard]] constexpr auto
connected_components(G const& g) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;
    using index_t = node_index_t<G>;

    guard_algorithm<MaxV>(g.node_count(), "connected_components: V exceeds MaxV");
    components_result<MaxV> result;
    auto const V = g.node_count();

    if (V == 0) {
        return result;
    }

    // Union-Find data structures.
    std::array<index_t, MaxV> parent{};
    std::array<index_t, MaxV> rank{};

    // Initialise: each node is its own component.
    for (std::size_t i = 0; i < V; ++i) {
        parent[i] = static_cast<index_t>(i);
        rank[i] = 0;
    }

    // Find with path compression (iterative).
    auto find = [&](index_t x) -> index_t {
        auto root = x;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[x] != root) {
            auto next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };

    // Union by rank.
    auto unite = [&](index_t a, index_t b) {
        auto ra = find(a);
        auto rb = find(b);
        if (ra == rb) return;
        if (rank[ra] < rank[rb]) {
            parent[ra] = rb;
        } else if (rank[ra] > rank[rb]) {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra]++;
        }
    };

    // Process all edges (direction ignored: union both endpoints).
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = static_cast<index_t>(u);
        for (auto v : g.out_neighbors(node_id{uid})) {
            unite(uid, static_cast<index_t>(v.value));
        }
    }

    // Renumber components to dense [0, component_count).
    constexpr index_t UNASSIGNED = node_nil_v<G>;
    std::array<index_t, MaxV> root_to_comp{};
    for (std::size_t i = 0; i < V; ++i) {
        root_to_comp[i] = UNASSIGNED;
    }

    std::uint16_t next_comp = 0;
    for (std::size_t i = 0; i < V; ++i) {
        auto const root = find(static_cast<index_t>(i));
        if (root_to_comp[root] == UNASSIGNED) {
            root_to_comp[root] = next_comp;
            next_comp++;
        }
        result.component_of[i] = root_to_comp[root];
    }

    result.component_count = static_cast<std::size_t>(next_comp);
    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CONNECTED_COMPONENTS_H
