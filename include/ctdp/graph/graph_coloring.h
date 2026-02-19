// graph/algorithms/graph_coloring.h - Constexpr greedy graph coloring
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Greedy Welsh-Powell (largest-degree-first ordering).
//
// Complexity: O(V·log(V) + V·E/V) = O(V·log(V) + E) for the sort +
//             greedy assignment.  Each node checks its neighbours'
//             colours → total work proportional to sum of degrees = 2E.
//
// GUARANTEES:
// - Uses at most max_degree + 1 colours (greedy upper bound)
// - Deterministic: ties in degree broken by node_id (ascending)
// - Correct: verify_coloring() proves no two adjacent nodes share a colour
//
// DESIGN RATIONALE:
// Graph coloring is inherently an undirected problem.  We require
// symmetric_graph_queryable to make this constraint machine-checkable.
// Attempting to colour a directed graph is a concept mismatch caught
// at compile time.
//
// The ordering function is a template parameter so users can substitute
// DSatur, LDO, or custom heuristics without touching the colouring loop.

#ifndef CTDP_GRAPH_COLORING_H
#define CTDP_GRAPH_COLORING_H

#include "capacity_guard.h"
#include "graph_concepts.h"
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
/// - max_degree_plus_one: the greedy upper bound (Δ+1)
/// - verified: true if verify_coloring() has confirmed correctness
template<std::size_t MaxV>
struct coloring_result {
    std::array<std::uint16_t, MaxV> color_of{};
    std::size_t node_count = 0;
    std::size_t color_count = 0;
    std::size_t max_degree_plus_one = 0;
    bool verified = false;
};

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
    template<std::size_t MaxV, symmetric_graph_queryable G>
    constexpr void operator()(
        G const& g,
        std::array<std::uint16_t, MaxV>& order,
        std::size_t V) const
    {
        // Populate order array.
        for (std::size_t i = 0; i < V; ++i) {
            order[i] = static_cast<std::uint16_t>(i);
        }

        // Sort by descending degree, then ascending node_id.
        // Use a simple O(n²) sort since V is typically small in
        // constexpr context and we can't allocate.
        for (std::size_t i = 0; i < V; ++i) {
            for (std::size_t j = i + 1; j < V; ++j) {
                auto di = g.degree(node_id{order[i]});
                auto dj = g.degree(node_id{order[j]});
                bool swap = (dj > di) ||
                            (dj == di && order[j] < order[i]);
                if (swap) {
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
/// Template parameters:
/// - MaxV: maximum vertex capacity
/// - G: graph type satisfying symmetric_graph_queryable
/// - Order: ordering functor (default: welsh_powell_order)
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
/// constexpr auto cr = graph_coloring<10>(sg);
/// static_assert(cr.color_count <= 4);  // Petersen is 3-chromatic
/// static_assert(cr.verified);
/// ```
template<std::size_t MaxV,
         symmetric_graph_queryable G,
         typename Order = welsh_powell_order>
[[nodiscard]] constexpr coloring_result<MaxV>
graph_coloring(G const& g, Order order_fn = {}) {
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
        auto d = g.degree(node_id{static_cast<std::uint16_t>(i)});
        if (d > max_deg) max_deg = d;
    }
    result.max_degree_plus_one = max_deg + 1;

    // Step 1: Compute node ordering.
    std::array<std::uint16_t, MaxV> node_order{};
    order_fn(g, node_order, V);

    // Step 2: Greedy colouring.
    // UNCOLORED sentinel: 0xFFFF (since max colours ≤ V ≤ MaxV ≤ 65535).
    constexpr std::uint16_t UNCOLORED = 0xFFFF;
    for (std::size_t i = 0; i < V; ++i) {
        result.color_of[i] = UNCOLORED;
    }

    // Temporary: which colours are used by neighbours of current node.
    // Size max_deg+1 would suffice, but we use MaxV for constexpr safety.
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
