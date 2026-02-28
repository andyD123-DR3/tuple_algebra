// graph/algorithms/shortest_path.h - Constexpr Dijkstra's shortest path
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Dijkstra with index-based constexpr binary min-heap.
// Complexity: O((V + E) log V)
//
// DESIGN RATIONALE:
// A single binary-heap implementation serves both constexpr and runtime
// graphs.  The O(V^2) linear-scan alternative was dropped -- with runtime
// graphs on the table, maintaining two code paths is worse than the
// microsecond difference at small V.
//
// The binary heap is index-based: a position array tracks where each
// node sits in the heap, enabling O(log V) decrease-key via sift-up.
//
// Weight input is a callable WeightFn(node_id from, node_id to) -> double.
// This decouples the algorithm from any specific weight storage (property
// maps, weighted_view, lambdas, etc.).
//
// TRAITS:
// MaxV from graph_traits<G>::max_nodes.  All working arrays (dist, pred,
// heap, position) use compile-time capacity.  node_index_t<G> for
// predecessor and node references.

#ifndef CTDP_GRAPH_SHORTEST_PATH_H
#define CTDP_GRAPH_SHORTEST_PATH_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "graph_traits.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ctdp::graph {

// =========================================================================
// Result type
// =========================================================================

/// Result of Dijkstra's shortest path computation.
///
/// - dist[n]: shortest distance from source to node n (INFINITY if unreachable)
/// - pred[n]: predecessor of n on shortest path (node_nil_v if source or unreachable)
/// - source: the source node
/// - node_count: number of nodes in the graph
/// - verified: true if verify_shortest_path has confirmed correctness
template<std::size_t MaxV>
struct shortest_path_result {
    std::array<double, MaxV> dist{};
    std::array<std::uint16_t, MaxV> pred{};
    std::uint16_t source = 0xFFFF;
    std::size_t node_count = 0;
    bool verified = false;
};

/// Factory: construct a shortest_path_result initialised for graph g.
template<sized_graph G>
[[nodiscard]] constexpr auto make_shortest_path_result(G const& g) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;

    shortest_path_result<MaxV> r{};
    for (std::size_t i = 0; i < MaxV; ++i) {
        r.dist[i] = std::numeric_limits<double>::infinity();
        r.pred[i] = node_nil_v<G>;
    }
    r.node_count = g.node_count();
    return r;
}

// =========================================================================
// Verification
// =========================================================================

/// O(E) verification of shortest-path optimality.
///
/// Checks the triangle inequality: for every edge u->v with weight w,
///   dist[v] <= dist[u] + w
/// Also checks that dist[source] == 0 and predecessor consistency.
template<std::size_t MaxV, sized_graph G, typename WeightFn>
[[nodiscard]] constexpr bool
verify_shortest_path(G const& g, WeightFn weight,
                     shortest_path_result<MaxV>& result)
{
    auto const V = g.node_count();
    auto const src = result.source;

    // Source distance must be 0.
    if (result.dist[src] != 0.0) {
        result.verified = false;
        return false;
    }

    // Triangle inequality on every edge.
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        auto const du = result.dist[u];
        if (du == std::numeric_limits<double>::infinity()) continue;

        for (auto v : g.out_neighbors(uid)) {
            auto const w = weight(uid, v);
            auto const relaxed = du + w;
            if (relaxed < result.dist[v.value] - 1e-12) {
                result.verified = false;
                return false;
            }
        }
    }

    // Predecessor consistency: pred[v] == u implies dist[v] == dist[u] + w(u,v).
    for (std::size_t v = 0; v < V; ++v) {
        if (v == src) continue;
        auto const p = result.pred[v];
        if (p == 0xFFFF) continue;  // unreachable
        auto const expected = result.dist[p] +
            weight(node_id{p}, node_id{static_cast<std::uint16_t>(v)});
        auto const diff = result.dist[v] - expected;
        if (diff < -1e-12 || diff > 1e-12) {
            result.verified = false;
            return false;
        }
    }

    result.verified = true;
    return true;
}

// =========================================================================
// Dijkstra's algorithm
// =========================================================================

/// Dijkstra's shortest path from a single source.
///
/// Template parameters:
/// - G: graph type satisfying sized_graph
/// - WeightFn: callable (node_id, node_id) -> double
///
/// Preconditions:
/// - All edge weights are non-negative
/// - source is a valid node in g
///
/// Example:
/// ```cpp
/// auto weight = [](node_id u, node_id v) -> double {
///     // ... return weight of edge u->v
/// };
/// constexpr auto result = dijkstra(g, node_id{0}, weight);
/// ```
template<sized_graph G, typename WeightFn>
[[nodiscard]] constexpr auto
dijkstra(G const& g, node_id source, WeightFn weight) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;
    using index_t = node_index_t<G>;

    guard_algorithm<MaxV>(g.node_count(), "dijkstra: V exceeds MaxV");

    auto result = make_shortest_path_result(g);
    auto const V = g.node_count();
    result.source = source.value;
    result.dist[source.value] = 0.0;

    if (V == 0) {
        result.verified = true;
        return result;
    }

    // =====================================================================
    // Index-based binary min-heap
    // =====================================================================
    //
    // heap[0..heap_size): node indices ordered by dist[].
    // pos[node]: index into heap[] where this node sits (MaxV if not in heap).
    //
    // Operations:
    //   sift_up(i):   restore heap property upward from position i
    //   sift_down(i): restore heap property downward from position i
    //   extract_min(): remove and return the minimum-distance node
    //   decrease_key(node): called after dist[node] decreased; sift up

    std::array<index_t, MaxV> heap{};
    std::array<index_t, MaxV> pos{};
    std::size_t heap_size = 0;

    constexpr index_t NOT_IN_HEAP = node_nil_v<G>;

    // Initialise: all nodes start in the heap.
    for (std::size_t i = 0; i < V; ++i) {
        heap[i] = static_cast<index_t>(i);
        pos[i] = static_cast<index_t>(i);
    }
    heap_size = V;

    // Heap swap helper.
    auto heap_swap = [&](std::size_t a, std::size_t b) {
        auto na = heap[a];
        auto nb = heap[b];
        heap[a] = nb;
        heap[b] = na;
        pos[na] = static_cast<index_t>(b);
        pos[nb] = static_cast<index_t>(a);
    };

    // Sift up: move element at position i toward root.
    auto sift_up = [&](std::size_t i) {
        while (i > 0) {
            std::size_t parent = (i - 1) / 2;
            if (result.dist[heap[i]] < result.dist[heap[parent]]) {
                heap_swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    };

    // Sift down: move element at position i toward leaves.
    auto sift_down = [&](std::size_t i) {
        while (true) {
            std::size_t smallest = i;
            std::size_t left = 2 * i + 1;
            std::size_t right = 2 * i + 2;
            if (left < heap_size &&
                result.dist[heap[left]] < result.dist[heap[smallest]]) {
                smallest = left;
            }
            if (right < heap_size &&
                result.dist[heap[right]] < result.dist[heap[smallest]]) {
                smallest = right;
            }
            if (smallest != i) {
                heap_swap(i, smallest);
                i = smallest;
            } else {
                break;
            }
        }
    };

    // Build min-heap (source has dist 0, rest INFINITY — source bubbles to top).
    // Since only source has finite dist, a single sift_up suffices.
    sift_up(pos[source.value]);

    // Main Dijkstra loop.
    while (heap_size > 0) {
        // Extract min.
        auto const u = heap[0];
        heap_swap(0, heap_size - 1);
        pos[u] = NOT_IN_HEAP;
        heap_size--;
        if (heap_size > 0) {
            sift_down(0);
        }

        auto const du = result.dist[u];
        if (du == std::numeric_limits<double>::infinity()) {
            break;  // remaining nodes unreachable
        }

        // Relax outgoing edges.
        auto const uid = node_id{u};
        for (auto v : g.out_neighbors(uid)) {
            auto const w = weight(uid, v);
            auto const new_dist = du + w;
            if (new_dist < result.dist[v.value]) {
                result.dist[v.value] = new_dist;
                result.pred[v.value] = u;
                // Decrease key: sift up in heap.
                if (pos[v.value] != NOT_IN_HEAP) {
                    sift_up(pos[v.value]);
                }
            }
        }
    }

    // Verify result.
    (void)verify_shortest_path<MaxV>(g, weight, result);

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SHORTEST_PATH_H
