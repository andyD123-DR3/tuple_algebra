// graph/algorithms/bipartite_matching.h - Constexpr Hopcroft-Karp matching
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Hopcroft-Karp maximum cardinality matching.
//
// Complexity: O(E · √V) where V = L + R, E = edge count.
//             At ct_limits::bipartite_matching_max = 64, this is
//             at most 64 · 8 = 512 per phase, with √128 ≈ 12 phases,
//             well within constexpr evaluation budgets.
//
// GUARANTEES:
// - Returns a maximum cardinality matching (provably optimal)
// - Deterministic: same graph → same matching (BFS/DFS order is CSR order)
// - Correct: verify_matching() proves all matched pairs are valid edges
//   and no vertex appears twice
//
// DESIGN RATIONALE:
// Hopcroft-Karp is the textbook algorithm for bipartite matching.
// It finds augmenting paths in phases using BFS (shortest augmenting
// path layers) then DFS (vertex-disjoint augmenting paths along layers).
// Each phase increases matching size by at least one, and the number
// of phases is O(√V), giving the O(E√V) bound.
//
// The algorithm constrains on bipartite_graph_queryable, ensuring that
// the left/right partition is structurally enforced, not assumed.

#ifndef CTDP_GRAPH_BIPARTITE_MATCHING_H
#define CTDP_GRAPH_BIPARTITE_MATCHING_H

#include "bipartite_graph.h"
#include "capacity_guard.h"
#include "graph_concepts.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =========================================================================
// Result type
// =========================================================================

/// Result of bipartite matching.
///
/// - match_left[i]:  right index matched to left i, or NIL if unmatched
/// - match_right[j]: left index matched to right j, or NIL if unmatched
/// - match_count:    number of matched pairs (= |matching|)
/// - left_count:     number of left vertices in the graph
/// - right_count:    number of right vertices in the graph
/// - verified:       true if verify_matching() has confirmed correctness
template<std::size_t MaxL, std::size_t MaxR>
struct matching_result {
    static constexpr std::size_t NIL = ~std::size_t{0};

    std::array<std::size_t, MaxL> match_left{};   // left i → right j
    std::array<std::size_t, MaxR> match_right{};   // right j → left i
    std::size_t match_count = 0;
    std::size_t left_count = 0;
    std::size_t right_count = 0;
    bool verified = false;

    /// Is left node i matched?
    [[nodiscard]] constexpr bool left_matched(std::size_t i) const {
        return match_left[i] != NIL;
    }

    /// Is right node j matched?
    [[nodiscard]] constexpr bool right_matched(std::size_t j) const {
        return match_right[j] != NIL;
    }

    /// Is this a perfect matching? (every left node matched)
    [[nodiscard]] constexpr bool is_perfect_left() const noexcept {
        return match_count == left_count;
    }

    /// Is this a perfect matching? (every right node matched)
    [[nodiscard]] constexpr bool is_perfect_right() const noexcept {
        return match_count == right_count;
    }

    /// Is this a complete perfect matching? (both sides fully matched)
    [[nodiscard]] constexpr bool is_perfect() const noexcept {
        return match_count == left_count && match_count == right_count;
    }
};

// =========================================================================
// Verification
// =========================================================================

/// O(E) verification that a matching is valid:
/// 1. Every matched pair (i, j) corresponds to an actual edge
/// 2. No left vertex appears in two pairs
/// 3. No right vertex appears in two pairs
/// 4. match_left and match_right are consistent
///
/// Sets result.verified = true on success.
template<std::size_t MaxL, std::size_t MaxR, bipartite_graph_queryable G>
[[nodiscard]] constexpr bool
verify_matching(G const& g, matching_result<MaxL, MaxR>& result) {
    constexpr auto NIL = matching_result<MaxL, MaxR>::NIL;
    auto const L = g.left_count();
    auto const R = g.right_count();

    // Check consistency: match_left[i] = j ⟺ match_right[j] = i
    for (std::size_t i = 0; i < L; ++i) {
        auto const j = result.match_left[i];
        if (j == NIL) continue;
        if (j >= R) { result.verified = false; return false; }
        if (result.match_right[j] != i) {
            result.verified = false;
            return false;
        }
    }
    for (std::size_t j = 0; j < R; ++j) {
        auto const i = result.match_right[j];
        if (i == NIL) continue;
        if (i >= L) { result.verified = false; return false; }
        if (result.match_left[i] != j) {
            result.verified = false;
            return false;
        }
    }

    // Check each matched pair is a real edge
    for (std::size_t i = 0; i < L; ++i) {
        auto const j = result.match_left[i];
        if (j == NIL) continue;

        bool found = false;
        for (auto rj : g.left_neighbors(i)) {
            if (rj == j) { found = true; break; }
        }
        if (!found) { result.verified = false; return false; }
    }

    // Check count
    std::size_t count = 0;
    for (std::size_t i = 0; i < L; ++i) {
        if (result.match_left[i] != NIL) ++count;
    }
    if (count != result.match_count) {
        result.verified = false;
        return false;
    }

    result.verified = true;
    return true;
}

// =========================================================================
// Hopcroft-Karp algorithm
// =========================================================================

namespace detail {

/// BFS phase: find shortest augmenting path layers.
///
/// Returns true if at least one augmenting path exists (i.e. BFS
/// reached a free right vertex).
///
/// After BFS, dist[u] gives the layer of left node u in the layered
/// graph.  Only left nodes reachable via alternating paths from free
/// left nodes have finite distance.
template<std::size_t MaxL, std::size_t MaxR, bipartite_graph_queryable G>
constexpr bool hopcroft_karp_bfs(
    G const& g,
    std::array<std::size_t, MaxL>& match_left,
    std::array<std::size_t, MaxR>& match_right,
    std::array<std::size_t, MaxL>& dist)
{
    constexpr std::size_t NIL = ~std::size_t{0};
    constexpr std::size_t INF = ~std::size_t{0};
    auto const L = g.left_count();

    // BFS queue: array of left indices, head/tail pointers.
    std::array<std::size_t, MaxL> queue{};
    std::size_t qhead = 0, qtail = 0;

    // Initialise: free left nodes at distance 0, matched at INF.
    for (std::size_t u = 0; u < L; ++u) {
        if (match_left[u] == NIL) {
            dist[u] = 0;
            queue[qtail++] = u;
        } else {
            dist[u] = INF;
        }
    }

    bool found = false;

    while (qhead < qtail) {
        auto const u = queue[qhead++];

        // Explore right neighbors of left node u.
        for (auto v : g.left_neighbors(u)) {
            auto const w = match_right[v];  // left node matched to right v
            if (w == NIL) {
                // Right node v is free — augmenting path endpoint found.
                found = true;
                // Don't break: continue BFS to find ALL shortest paths.
            } else if (dist[w] == INF) {
                // Follow the matching edge back to the left side.
                dist[w] = dist[u] + 1;
                queue[qtail++] = w;
            }
        }
    }

    return found;
}

/// DFS phase: find an augmenting path from left node u along BFS layers.
///
/// If found, augments the matching (flips edges along the path).
/// Returns true if an augmenting path was found.
template<std::size_t MaxL, std::size_t MaxR, bipartite_graph_queryable G>
constexpr bool hopcroft_karp_dfs(
    G const& g,
    std::size_t u,
    std::array<std::size_t, MaxL>& match_left,
    std::array<std::size_t, MaxR>& match_right,
    std::array<std::size_t, MaxL>& dist)
{
    constexpr std::size_t NIL = ~std::size_t{0};
    constexpr std::size_t INF = ~std::size_t{0};

    for (auto v : g.left_neighbors(u)) {
        auto const w = match_right[v];
        if (w == NIL ||
            (dist[w] == dist[u] + 1 &&
             hopcroft_karp_dfs<MaxL, MaxR>(g, w, match_left, match_right, dist)))
        {
            // Augment: match u ↔ v
            match_left[u] = v;
            match_right[v] = u;
            return true;
        }
    }

    // No augmenting path from u — remove from layered graph.
    dist[u] = INF;
    return false;
}

} // namespace detail

/// Hopcroft-Karp maximum cardinality matching on a bipartite graph.
///
/// Template parameters:
/// - MaxL: maximum left partition size (capacity)
/// - MaxR: maximum right partition size (capacity)
/// - G: graph type satisfying bipartite_graph_queryable
///
/// Returns matching_result with verified = true.
///
/// Example:
/// ```cpp
/// constexpr auto bg = /* build bipartite_graph */;
/// constexpr auto m = hopcroft_karp<4, 4>(bg);
/// static_assert(m.match_count == 3);
/// static_assert(m.verified);
/// ```
template<std::size_t MaxL, std::size_t MaxR,
         bipartite_graph_queryable G>
[[nodiscard]] constexpr matching_result<MaxL, MaxR>
hopcroft_karp(G const& g) {
    guard_algorithm<MaxL>(g.left_count(),
        "hopcroft_karp: left_count exceeds MaxL");
    guard_algorithm<MaxR>(g.right_count(),
        "hopcroft_karp: right_count exceeds MaxR");

    constexpr std::size_t NIL = matching_result<MaxL, MaxR>::NIL;

    matching_result<MaxL, MaxR> result;
    result.left_count = g.left_count();
    result.right_count = g.right_count();

    auto const L = g.left_count();

    // Initialise: all vertices unmatched.
    for (std::size_t i = 0; i < MaxL; ++i) result.match_left[i] = NIL;
    for (std::size_t j = 0; j < MaxR; ++j) result.match_right[j] = NIL;

    // BFS distance array for left nodes.
    std::array<std::size_t, MaxL> dist{};

    // Main loop: BFS to find augmenting path layers, DFS to augment.
    while (detail::hopcroft_karp_bfs<MaxL, MaxR>(
               g, result.match_left, result.match_right, dist))
    {
        for (std::size_t u = 0; u < L; ++u) {
            if (result.match_left[u] == NIL) {
                detail::hopcroft_karp_dfs<MaxL, MaxR>(
                    g, u, result.match_left, result.match_right, dist);
            }
        }
    }

    // Count matched pairs.
    std::size_t count = 0;
    for (std::size_t i = 0; i < L; ++i) {
        if (result.match_left[i] != NIL) ++count;
    }
    result.match_count = count;

    // Verify correctness.
    (void)verify_matching<MaxL, MaxR>(g, result);

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_BIPARTITE_MATCHING_H
