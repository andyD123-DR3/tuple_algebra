// graph/algorithms/scc.h - Constexpr strongly connected components (Tarjan)
// Part of the compile-time DP library (C++20)
//
// ALGORITHM: Iterative Tarjan's algorithm.
// Complexity: O(V + E)
// Determinism: nodes visited in node_id order. Component numbering
// follows reverse topological order (SCCs numbered as they are completed).
//
// DESIGN RATIONALE:
// Iterative (not recursive) because constexpr evaluation has limited
// recursion depth and we need explicit control over stack usage.
// Uses an explicit call stack with frames tracking DFS state.
//
// TRAITS RETROFIT (Phase 7):
// MaxV derived from graph_traits<G>::max_nodes.  node_index_t<G> used
// for discovery indices, lowlinks, and stack contents.

#ifndef CTDP_GRAPH_SCC_H
#define CTDP_GRAPH_SCC_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ctdp::graph {

/// Result of strongly connected components analysis.
///
/// - component_of[n]: component id for node n (0-based)
/// - component_count: total number of SCCs
///
/// Component ids are assigned in reverse topological order of the
/// condensation DAG (i.e., source SCCs get higher ids).
template<std::size_t MaxV>
struct scc_result {
    std::array<std::uint16_t, MaxV> component_of{};
    std::size_t component_count = 0;
};

/// Factory: construct a default scc_result sized for graph g.
template<sized_graph G>
[[nodiscard]] constexpr auto make_scc_result(G const& /*g*/) {
    return scc_result<graph_traits<G>::max_nodes>{};
}

/// Strongly connected components via iterative Tarjan's algorithm.
///
/// MaxV is derived from graph_traits<G>::max_nodes.
/// Requires sized_graph (capacity queries for array sizing).
///
/// Example:
/// ```cpp
/// constexpr auto g = make_chain();  // 0->1->2->3
/// constexpr auto result = scc(g);
/// static_assert(result.component_count == 4);  // DAG: each node is own SCC
/// ```
template<sized_graph G>
[[nodiscard]] constexpr auto
scc(G const& g) {
    constexpr std::size_t MaxV = graph_traits<G>::max_nodes;
    using index_t = node_index_t<G>;

    guard_algorithm<MaxV>(g.node_count(), "scc: V exceeds MaxV");
    scc_result<MaxV> result;
    auto const V = g.node_count();

    if (V == 0) {
        return result;
    }

    // Tarjan's state.
    constexpr index_t UNVISITED = node_nil_v<G>;

    std::array<index_t, MaxV> index_of{};    // discovery index
    std::array<index_t, MaxV> lowlink{};      // lowest reachable index
    std::array<bool, MaxV> on_stack{};         // currently on Tarjan stack
    std::array<bool, MaxV> assigned{};         // already assigned to an SCC

    for (std::size_t i = 0; i < V; ++i) {
        index_of[i] = UNVISITED;
    }

    // Tarjan's stack (nodes awaiting SCC assignment).
    ctdp::constexpr_vector<index_t, MaxV> tarjan_stack;

    // DFS call stack frame.
    struct frame {
        index_t node;
        index_t neighbor_idx;   // which neighbor we're processing next
        index_t neighbor_count;
    };
    ctdp::constexpr_vector<frame, MaxV> call_stack;

    index_t next_index = 0;

    // Process each unvisited node (deterministic: ascending node_id order).
    for (std::size_t start = 0; start < V; ++start) {
        if (index_of[start] != UNVISITED) {
            continue;
        }

        // Push initial frame.
        auto const s = static_cast<index_t>(start);
        index_of[s] = next_index;
        lowlink[s] = next_index;
        next_index++;
        on_stack[s] = true;
        tarjan_stack.push_back(s);

        // Count neighbors for start node.
        index_t start_nc = 0;
        {
            auto const range = g.out_neighbors(node_id{s});
            if constexpr (requires { range.size(); }) {
                start_nc = static_cast<index_t>(range.size());
            } else {
                for ([[maybe_unused]] auto _ : range) start_nc++;
            }
        }
        call_stack.push_back(frame{s, 0, start_nc});

        // Iterative DFS loop.
        while (!call_stack.empty()) {
            auto& top = call_stack.back();

            if (top.neighbor_idx < top.neighbor_count) {
                // Get the next neighbor.
                index_t w = 0;
                auto const range = g.out_neighbors(node_id{top.node});
                if constexpr (std::is_pointer_v<decltype(range.begin())>) {
                    w = range.begin()[top.neighbor_idx].value;
                } else {
                    index_t idx = 0;
                    for (auto nbr : range) {
                        if (idx == top.neighbor_idx) {
                            w = nbr.value;
                            break;
                        }
                        idx++;
                    }
                }
                top.neighbor_idx++;

                if (index_of[w] == UNVISITED) {
                    // Tree edge: "recurse" into w.
                    index_of[w] = next_index;
                    lowlink[w] = next_index;
                    next_index++;
                    on_stack[w] = true;
                    tarjan_stack.push_back(w);

                    // Count w's neighbors.
                    index_t w_nc = 0;
                    {
                        auto const w_range = g.out_neighbors(node_id{w});
                        if constexpr (requires { w_range.size(); }) {
                            w_nc = static_cast<index_t>(w_range.size());
                        } else {
                            for ([[maybe_unused]] auto _ : w_range) w_nc++;
                        }
                    }
                    call_stack.push_back(frame{w, 0, w_nc});
                } else if (on_stack[w]) {
                    // Back edge: update lowlink.
                    if (index_of[w] < lowlink[top.node]) {
                        lowlink[top.node] = index_of[w];
                    }
                }
            } else {
                // All neighbors processed. Check if this is an SCC root.
                auto const u = top.node;
                call_stack.pop_back();

                if (lowlink[u] == index_of[u]) {
                    // u is the root of an SCC. Pop everything up to u.
                    auto const comp_id = static_cast<index_t>(result.component_count);
                    while (true) {
                        auto const w = tarjan_stack.back();
                        tarjan_stack.pop_back();
                        on_stack[w] = false;
                        assigned[w] = true;
                        result.component_of[w] = comp_id;
                        if (w == u) break;
                    }
                    result.component_count++;
                }

                // Update parent's lowlink.
                if (!call_stack.empty()) {
                    auto const parent = call_stack.back().node;
                    if (lowlink[u] < lowlink[parent]) {
                        lowlink[parent] = lowlink[u];
                    }
                }
            }
        }
    }

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SCC_H
