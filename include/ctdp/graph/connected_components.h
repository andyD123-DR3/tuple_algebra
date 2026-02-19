// graph/algorithms/connected_components.h - Weakly connected components
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Union-Find with path compression and union by rank.
// Complexity: O(V + E · α(V)) ≈ O(V + E) (amortised)
//
// SEMANTICS: Weakly connected components — edge direction is ignored.
// For a directed graph, nodes u and v are in the same component if
// there exists an undirected path between them (ignoring arrow direction).
//
// DESIGN RATIONALE:
// Union-Find (not BFS/DFS) because:
// - Avoids building transpose graph or scanning for in-neighbors
// - Near-linear time
// - Simple constexpr implementation
// - No recursion or stack management needed

#ifndef CTDP_GRAPH_CONNECTED_COMPONENTS_H
#define CTDP_GRAPH_CONNECTED_COMPONENTS_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include <ctdp/core/ct_limits.h>

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

/// Weakly connected components via Union-Find.
///
/// Treats edges as undirected: nodes connected by a directed edge in
/// either direction belong to the same component.
///
/// Template parameters:
/// - G: graph type satisfying graph_queryable
/// - MaxV: maximum vertex count (default: ct_limits::connected_components_max)
///
/// Example:
/// ```cpp
/// // Two components: {0,1} and {2,3}
/// constexpr auto g = make_disconnected();  // 0→1, 2→3
/// constexpr auto result = connected_components(g);
/// static_assert(result.component_count == 2);
/// static_assert(result.component_of[0] == result.component_of[1]);
/// static_assert(result.component_of[2] == result.component_of[3]);
/// ```
template<graph_queryable G,
         std::size_t MaxV = ctdp::ct_limits::connected_components_max>
[[nodiscard]] constexpr components_result<MaxV>
connected_components(G const& g) {
    guard_algorithm<MaxV>(g.node_count(), "connected_components: V exceeds MaxV");
    components_result<MaxV> result;
    auto const V = g.node_count();

    if (V == 0) {
        return result;
    }

    // Union-Find data structures.
    std::array<std::uint16_t, MaxV> parent{};
    std::array<std::uint16_t, MaxV> rank{};

    // Initialise: each node is its own component.
    for (std::size_t i = 0; i < V; ++i) {
        parent[i] = static_cast<std::uint16_t>(i);
        rank[i] = 0;
    }

    // Find with path compression (iterative).
    auto find = [&](std::uint16_t x) -> std::uint16_t {
        // Find root.
        auto root = x;
        while (parent[root] != root) {
            root = parent[root];
        }
        // Path compression.
        while (parent[x] != root) {
            auto next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };

    // Union by rank.
    auto unite = [&](std::uint16_t a, std::uint16_t b) {
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
        auto const uid = static_cast<std::uint16_t>(u);
        for (auto v : g.out_neighbors(node_id{uid})) {
            unite(uid, v.value);
        }
    }

    // Renumber components to dense [0, component_count).
    // Scan nodes in order — the first node in each component defines its id.
    constexpr std::uint16_t UNASSIGNED = 0xFFFF;
    std::array<std::uint16_t, MaxV> root_to_comp{};
    for (std::size_t i = 0; i < V; ++i) {
        root_to_comp[i] = UNASSIGNED;
    }

    std::uint16_t next_comp = 0;
    for (std::size_t i = 0; i < V; ++i) {
        auto const root = find(static_cast<std::uint16_t>(i));
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
