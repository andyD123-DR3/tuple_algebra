// graph/algorithms/min_cut.h - Constexpr Stoer-Wagner minimum cut
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Stoer-Wagner (dense matrix contraction).
// Complexity: O(V^3)
//
// DESIGN RATIONALE:
// At V <= 64 the dense V x V weight matrix fits in 32 KiB with perfect
// cache locality.  Node contraction merges rows/columns in-place.
// The algorithm is trivially constexpr: no heap, no recursion, just
// nested loops over a flat matrix.
//
// Union-find on CSR is documented as a fallback for memory-constrained
// large runtime graphs, but is NOT implemented here (dense primary).
//
// CANONICAL SUMMATION FOR VERIFICATION:
// Iterate u = 0..V-1, for each outgoing edge u->v in CSR order,
// include iff u < v.  Same order in algorithm and verifier gives
// bitwise identical IEEE 754 results.
//
// TRAITS:
// MaxV from graph_traits<G>::max_nodes.  Weight matrix is a flat
// std::array<double, MaxV * MaxV>.  Partition tracked via merged_into
// array of node_index_t<G>.

#ifndef CTDP_GRAPH_MIN_CUT_H
#define CTDP_GRAPH_MIN_CUT_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include "symmetric_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ctdp::graph {

// =========================================================================
// Result type
// =========================================================================

/// Result of Stoer-Wagner minimum cut.
///
/// - cut_weight: weight of the minimum cut
/// - partition[n]: 0 or 1 indicating which side of the cut node n belongs to
/// - node_count: number of nodes in the graph
/// - verified: true if verify_min_cut has confirmed correctness
template<std::size_t MaxV>
struct min_cut_result {
    double cut_weight = std::numeric_limits<double>::infinity();
    std::array<std::uint8_t, MaxV> partition{};
    std::size_t node_count = 0;
    bool verified = false;
};

/// Factory: construct a default min_cut_result sized for graph g.
template<typename G>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr auto make_min_cut_result(G const& /*g*/) {
    return min_cut_result<graph_traits<G>::max_nodes>{};
}

// =========================================================================
// Verification
// =========================================================================

/// O(E) verification of minimum cut: sum edges crossing the partition.
///
/// Uses canonical summation order: for each edge (u, v) with u < v
/// in CSR order, add weight if partition[u] != partition[v].
/// This matches the algorithm's contraction semantics for bitwise
/// identical IEEE 754 results.
template<std::size_t MaxV, typename G, typename WeightFn>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr bool
verify_min_cut(G const& g, WeightFn weight,
               min_cut_result<MaxV>& result)
{
    auto const V = g.node_count();
    double cut_sum = 0.0;

    // Canonical order: u < v, CSR iteration.
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            if (u < static_cast<std::size_t>(v.value)) {
                if (result.partition[u] != result.partition[v.value]) {
                    cut_sum += weight(uid, v);
                }
            }
        }
    }

    // Check bitwise equality (canonical summation guarantees this).
    auto const diff = result.cut_weight - cut_sum;
    if (diff < -1e-12 || diff > 1e-12) {
        result.verified = false;
        return false;
    }

    result.verified = true;
    return true;
}

// =========================================================================
// Stoer-Wagner algorithm
// =========================================================================

/// Stoer-Wagner minimum cut for undirected weighted graphs.
///
/// Template parameters:
/// - G: graph type satisfying symmetric_graph_queryable AND sized_graph
/// - WeightFn: callable (node_id, node_id) -> double
///
/// Preconditions:
/// - All edge weights are non-negative
/// - Graph is connected (otherwise the minimum cut is 0)
///
/// Example:
/// ```cpp
/// auto weight = [](node_id u, node_id v) -> double { return 1.0; };
/// constexpr auto result = stoer_wagner(g, weight);
/// ```
template<typename G, typename WeightFn>
    requires (symmetric_graph_queryable<G> && sized_graph<G>)
[[nodiscard]] constexpr auto
stoer_wagner(G const& g, WeightFn weight) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;
    using index_t = node_index_t<G>;

    guard_algorithm<MaxV>(g.node_count(), "stoer_wagner: V exceeds MaxV");

    min_cut_result<MaxV> best;
    auto const V = g.node_count();
    best.node_count = V;

    if (V <= 1) {
        best.cut_weight = 0.0;
        best.verified = true;
        return best;
    }

    // =====================================================================
    // Dense weight matrix: w[u * MaxV + v] = weight of edge (u, v).
    // =====================================================================
    std::array<double, MaxV * MaxV> w{};

    // Populate from graph.
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<index_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            auto const vi = static_cast<std::size_t>(v.value);
            if (u != vi) {
                w[u * MaxV + vi] = weight(uid, v);
            }
        }
    }

    // Track which super-node each original node belongs to.
    // merged_into[u] = the super-node u has been merged into.
    std::array<index_t, MaxV> merged_into{};
    for (std::size_t i = 0; i < V; ++i) {
        merged_into[i] = static_cast<index_t>(i);
    }

    // active[i] = true if super-node i is still in the contracted graph.
    std::array<bool, MaxV> active{};
    for (std::size_t i = 0; i < V; ++i) {
        active[i] = true;
    }

    // For each phase's cut, track which side of the partition each node is on.
    // best_partition[n] = which side of the best cut so far.
    std::array<std::uint8_t, MaxV> best_part{};

    // Phase partitions: for each cut-of-the-phase, s/t defines the partition.
    // t's side is the set of original nodes merged into t.
    // We track this via merged_into chains.

    double best_cut = std::numeric_limits<double>::infinity();
    std::size_t active_count = V;

    // V-1 phases of Stoer-Wagner.
    for (std::size_t phase = 0; phase + 1 < V; ++phase) {
        // ================================================================
        // Minimum-cut phase: find the most tightly connected vertex ordering.
        // ================================================================

        std::array<double, MaxV> key{};  // accumulated weight to "added" set
        std::array<bool, MaxV> in_A{};   // already added to set A

        // Find first active node.
        index_t start = 0;
        for (std::size_t i = 0; i < V; ++i) {
            if (active[i]) { start = static_cast<index_t>(i); break; }
        }

        in_A[start] = true;
        // Initialise keys from start.
        for (std::size_t i = 0; i < V; ++i) {
            if (active[i] && i != start) {
                key[i] = w[start * MaxV + i];
            }
        }

        index_t prev = start;  // second-to-last added
        index_t last = start;  // last added

        // Add active_count - 1 more vertices.
        for (std::size_t step = 1; step < active_count; ++step) {
            // Find the active, non-added vertex with maximum key.
            index_t next = node_nil_v<G>;
            double max_key = -1.0;
            for (std::size_t i = 0; i < V; ++i) {
                if (active[i] && !in_A[i] && key[i] > max_key) {
                    max_key = key[i];
                    next = static_cast<index_t>(i);
                }
            }

            in_A[next] = true;

            // Update keys.
            for (std::size_t i = 0; i < V; ++i) {
                if (active[i] && !in_A[i]) {
                    key[i] += w[next * MaxV + i];
                }
            }

            prev = last;
            last = next;
        }

        // The cut-of-the-phase is the weight of the last-added vertex.
        double phase_cut = key[last];

        if (phase_cut < best_cut) {
            best_cut = phase_cut;

            // Partition: nodes merged into 'last' are on one side (1),
            // everything else on the other side (0).
            for (std::size_t i = 0; i < V; ++i) {
                best_part[i] = 0;
            }
            // Find all original nodes that have been merged into 'last'.
            // Chase merged_into to find root, check if root == last.
            for (std::size_t i = 0; i < V; ++i) {
                auto n = static_cast<index_t>(i);
                while (merged_into[n] != n) {
                    n = merged_into[n];
                }
                if (n == last) {
                    best_part[i] = 1;
                }
            }
        }

        // ================================================================
        // Contract: merge 'last' into 'prev'.
        // ================================================================

        // Redirect last's merged_into to prev.
        merged_into[last] = prev;

        // Merge edge weights: for every active node i != prev, last,
        // w[prev][i] += w[last][i]  (and symmetric).
        for (std::size_t i = 0; i < V; ++i) {
            if (!active[i] || i == static_cast<std::size_t>(prev) ||
                i == static_cast<std::size_t>(last)) continue;
            w[prev * MaxV + i] += w[last * MaxV + i];
            w[i * MaxV + prev] += w[i * MaxV + last];
        }

        // Deactivate 'last'.
        active[last] = false;
        active_count--;
    }

    best.cut_weight = best_cut;
    for (std::size_t i = 0; i < V; ++i) {
        best.partition[i] = best_part[i];
    }

    // Verify.
    (void)verify_min_cut<MaxV>(g, weight, best);

    return best;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_MIN_CUT_H
