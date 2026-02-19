// graph/representation/implicit_graph.h - Procedurally-generated graphs
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Implicit graphs store no edges. They satisfy graph_queryable but not
// graph_mutable, enforcing the "materialise before transforming" rule.
// Generators must produce neighbors in a deterministic order for a
// given node_id (documented precondition).
//
// The generator's return type determines the adjacency representation.
// For constexpr-safe generators, this should be constexpr_vector<node_id, K>.

#ifndef CTDP_GRAPH_IMPLICIT_GRAPH_H
#define CTDP_GRAPH_IMPLICIT_GRAPH_H

#include "graph_concepts.h"

#include <cstddef>

namespace ctdp::graph {

/// Graph defined by a procedural generator rather than stored edges.
///
/// The generator is called with a node_id and must return a range-like
/// collection of node_id (typically constexpr_vector<node_id, K>).
///
/// Satisfies graph_queryable but not sized_graph (no edge_count —
/// computing it would require visiting every node).
///
/// Determinism precondition: the generator must produce the same
/// neighbor list for the same node_id across all calls.
///
/// Template parameters:
/// - Generator: callable (node_id) → range-of-node_id
///
/// Example:
/// ```cpp
/// // Linear chain: each node points to next
/// auto gen = [n = 5](node_id u) {
///     constexpr_vector<node_id, 1> result;
///     if (u.value + 1 < n) result.push_back(node_id{u.value + 1});
///     return result;
/// };
/// implicit_graph g(5, gen);
/// ```
template<typename Generator>
class implicit_graph {
public:
    constexpr implicit_graph() = default;

    constexpr explicit implicit_graph(std::size_t V, Generator gen)
        : V_(V), gen_(gen) {}

    // =========================================================================
    // graph_queryable interface
    // =========================================================================

    /// Number of vertices in the graph.
    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return V_;
    }

    /// Returns whatever the generator returns for node u.
    /// For constexpr-safe generators, this is typically
    /// constexpr_vector<node_id, K>.
    [[nodiscard]] constexpr auto out_neighbors(node_id u) const {
        return gen_(u);
    }

    // =========================================================================
    // Additional queries
    // =========================================================================

    /// True if the graph has no vertices.
    [[nodiscard]] constexpr bool empty() const noexcept {
        return V_ == 0;
    }

    /// Check if a node_id is valid for this graph.
    [[nodiscard]] constexpr bool has_node(node_id u) const noexcept {
        return static_cast<std::size_t>(u.value) < V_;
    }

private:
    std::size_t V_{};
    Generator gen_{};
};

/// Deduction guide: implicit_graph(size_t, gen) → implicit_graph<Gen>
template<typename G>
implicit_graph(std::size_t, G) -> implicit_graph<G>;

} // namespace ctdp::graph

#endif // CTDP_GRAPH_IMPLICIT_GRAPH_H
