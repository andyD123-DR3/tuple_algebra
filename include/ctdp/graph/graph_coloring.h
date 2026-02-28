// graph/algorithms/graph_coloring.h - Constexpr greedy graph coloring
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Greedy Welsh-Powell (largest-degree-first ordering).
//
// Complexity: O(V*log(V) + V*E/V) = O(V*log(V) + E) for the sort +
//             greedy assignment.
//
// GUARANTEES:
// - Uses at most max_degree + 1 colours (greedy upper bound)
// - Deterministic: ties in degree broken by node_id (ascending)
// - Correct: verify_coloring() proves no two adjacent nodes share a colour
//
// DESIGN RATIONALE:
// Graph coloring is inherently an undirected problem.  We require
// symmetric_graph_queryable to make this constraint machine-checkable.
//
// TRAITS RETROFIT (Phase 7):
// MaxV derived from graph_traits<G>::max_nodes.  node_index_t<G> used
// for colour assignments, node ordering, and result arrays.

#ifndef CTDP_GRAPH_COLORING_H
#define CTDP_GRAPH_COLORING_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include "symmetric_graph.h"
#include <ctdp/core/constexpr_sort.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =========================================================================
// Result type
// =========================================================================

/// Result of graph coloring.
///
/// - color_of[n]: colour id assigned to node n (0-based)
/// - color_count: number of colours used (upper bound on chromatic number)
/// - max_degree_plus_one: the greedy upper bound (Delta+1)
/// - verified: true if verify_coloring() has confirmed correctness
template<std::size_t MaxV>
struct coloring_result {
    std::array<std::uint16_t, MaxV> color_of{};
    std::size_t node_count = 0;
    std::size_t color_count = 0;
    std::size_t max_degree_plus_one = 0;
    bool verified = false;
};

/// Factory: construct a default coloring_result sized for graph g.
template<typename G>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr auto make_coloring_result(G const& /*g*/) {
    return coloring_result<graph_traits<G>::max_nodes>{};
}

// =========================================================================
// Post-condition: verify colouring is legal
// =========================================================================

/// O(E) verification: no two adjacent nodes share a colour.
///
/// Returns true iff the colouring is valid.  Sets result.verified = true
/// on success.
template<std::size_t MaxV, symmetric_graph_queryable G>
[[nodiscard]] constexpr bool
verify_coloring(G const& g, coloring_result<MaxV>& result) {
    auto const V = g.node_count();
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        auto const u_color = result.color_of[u];
        for (auto v : g.neighbors(uid)) {
            if (result.color_of[v.value] == u_color) {
                result.verified = false;
                return false;
            }
        }
    }
    result.verified = true;
    return true;
}

// =========================================================================
// Default ordering: Welsh-Powell (largest degree first)
// =========================================================================

/// Welsh-Powell ordering: nodes sorted by descending degree.
/// Ties broken by ascending node_id for determinism.
struct welsh_powell_order {
    template<typename G>
        requires (symmetric_graph_queryable<G> && sized_graph<G>)
    constexpr void operator()(
        G const& g,
        std::array<node_index_t<G>, graph_traits<G>::max_nodes>& order,
        std::size_t V) const
    {
        using index_t = node_index_t<G>;

        // Populate order array.
        for (std::size_t i = 0; i < V; ++i) {
            order[i] = static_cast<index_t>(i);
        }

        // Sort by descending degree, then ascending node_id.
        for (std::size_t i = 0; i < V; ++i) {
            for (std::size_t j = i + 1; j < V; ++j) {
                auto di = g.degree(node_id{order[i]});
                auto dj = g.degree(node_id{order[j]});
                bool do_swap = (dj > di) ||
                            (dj == di && order[j] < order[i]);
                if (do_swap) {
                    auto tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
            }
        }
    }
};

// =========================================================================
// Greedy graph coloring algorithm
// =========================================================================

/// Greedy graph coloring with configurable node ordering.
///
/// MaxV is derived from graph_traits<G>::max_nodes.
/// Requires symmetric_graph_queryable AND sized_graph.
///
/// The algorithm:
/// 1. Sort nodes according to Order (default: descending degree)
/// 2. For each node in order, assign the smallest colour not used
///    by any neighbour.
/// 3. Verify the result (O(E) check).
///
/// Example:
/// ```cpp
/// constexpr auto sg = build_petersen_graph();
/// constexpr auto cr = graph_coloring(sg);
/// static_assert(cr.color_count <= 4);
/// static_assert(cr.verified);
/// ```
template<typename G, typename Order = welsh_powell_order>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr auto
graph_coloring(G const& g, Order order_fn = {}) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;
    using index_t = node_index_t<G>;

    guard_algorithm<MaxV>(g.node_count(), "graph_coloring: V exceeds MaxV");
    coloring_result<MaxV> result;
    auto const V = g.node_count();
    result.node_count = V;

    if (V == 0) {
        result.verified = true;
        return result;
    }

    // Compute max degree for quality metric.
    std::size_t max_deg = 0;
    for (std::size_t i = 0; i < V; ++i) {
        auto d = g.degree(node_id{static_cast<index_t>(i)});
        if (d > max_deg) max_deg = d;
    }
    result.max_degree_plus_one = max_deg + 1;

    // Step 1: Compute node ordering.
    std::array<index_t, MaxV> node_order{};
    order_fn(g, node_order, V);

    // Step 2: Greedy colouring.
    constexpr index_t UNCOLORED = node_nil_v<G>;
    for (std::size_t i = 0; i < V; ++i) {
        result.color_of[i] = UNCOLORED;
    }

    // Temporary: which colours are used by neighbours of current node.
    std::array<bool, MaxV> used{};

    std::size_t colors_used = 0;

    for (std::size_t idx = 0; idx < V; ++idx) {
        auto const u = node_order[idx];

        // Clear the used array (only up to colors_used + 1).
        for (std::size_t c = 0; c <= colors_used && c < MaxV; ++c) {
            used[c] = false;
        }

        // Mark colours used by neighbours.
        for (auto nbr : g.neighbors(node_id{u})) {
            auto const nc = result.color_of[nbr.value];
            if (nc != UNCOLORED && nc < MaxV) {
                used[nc] = true;
            }
        }

        // Find smallest unused colour.
        std::uint16_t chosen = 0;
        while (chosen < MaxV && used[chosen]) {
            ++chosen;
        }

        result.color_of[u] = chosen;
        if (chosen >= colors_used) {
            colors_used = chosen + 1;
        }
    }

    result.color_count = colors_used;

    // Step 3: Verify correctness.
    (void)verify_coloring<MaxV>(g, result);

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_COLORING_H
