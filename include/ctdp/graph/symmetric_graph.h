// graph/symmetric_graph.h — Undirected graph representation
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// constexpr_graph is directed: edges u→v and v→u are distinct.  Many
// optimisation problems operate on inherently undirected structures:
// conflict graphs (coloring), interference graphs, compatibility graphs.
//
// symmetric_graph<Cap> wraps constexpr_graph<Cap::max_v, 2*Cap::max_e>.
// The builder enforces symmetry: add_edge(u, v) inserts BOTH directions.
// Cap::max_e is the number of UNDIRECTED edges (logical edges).  The internal
// CSR stores 2*Cap::max_e directed edges.
//
// The type satisfies graph_queryable, so all existing algorithms work.
// out_neighbors(u) returns all adjacent nodes (both directions), which
// is the correct semantics for an undirected graph.  neighbors() is
// provided as a semantic alias.
//
// WHEN TO USE WHICH TYPE:
//   constexpr_graph  — DAGs, dependency ordering, dataflow (directed)
//   symmetric_graph  — conflicts, coloring, partitioning (undirected)

#ifndef CTDP_GRAPH_SYMMETRIC_GRAPH_H
#define CTDP_GRAPH_SYMMETRIC_GRAPH_H

#include "constexpr_graph.h"
#include "capacity_types.h"
#include "graph_builder.h"
#include "graph_concepts.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace ctdp::graph {

// =============================================================================
// symmetric_graph: undirected graph via symmetric CSR
// =============================================================================

/// Fixed-capacity, immutable, undirected graph.
///
/// Template parameters:
/// - Cap::max_v: Maximum number of vertices
/// - Cap::max_e: Maximum number of UNDIRECTED edges (logical edges)
///
/// Internally stores 2×Cap::max_e directed edges in a constexpr_graph.
/// out_neighbors(u) returns ALL adjacent nodes — there is no distinction
/// between in-neighbours and out-neighbours.
///
/// Constructed exclusively via symmetric_graph_builder::finalise().
///
/// Example:
/// ```cpp
/// constexpr auto g = []() {
///     symmetric_graph_builder<cap_from<4, 8>> b;
///     auto n0 = b.add_node();
///     auto n1 = b.add_node();
///     auto n2 = b.add_node();
///     b.add_edge(n0, n1);   // adds both n0→n1 and n1→n0
///     b.add_edge(n1, n2);
///     return b.finalise();
/// }();
/// static_assert(g.node_count() == 3);
/// static_assert(g.undirected_edge_count() == 2);
/// ```
template<typename Cap = cap::medium>
class symmetric_graph {
    static_assert(Cap::max_v <= 65535,
        "symmetric_graph: Cap::max_v exceeds uint16_t range (65535)");
public:
    static constexpr std::size_t max_vertices = Cap::max_v;
    static constexpr std::size_t max_undirected_edges = Cap::max_e;
    static constexpr std::size_t max_directed_edges = 2 * Cap::max_e;

    using inner_graph_type = constexpr_graph<cap_from<Cap::max_v, 2 * Cap::max_e>>;

    constexpr symmetric_graph() = default;

    // =========================================================================
    // Size queries
    // =========================================================================

    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return inner_.node_count();
    }

    /// Number of directed edges stored (= 2 × undirected edges).
    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return inner_.edge_count();
    }

    /// Number of undirected (logical) edges.
    [[nodiscard]] constexpr std::size_t undirected_edge_count() const noexcept {
        return inner_.edge_count() / 2;
    }

    /// Maximum number of vertices this graph can hold.
    [[nodiscard]] constexpr std::size_t node_capacity() const noexcept {
        return Cap::max_v;
    }

    /// Maximum number of directed edges this graph can hold (= 2 × max undirected).
    [[nodiscard]] constexpr std::size_t edge_capacity() const noexcept {
        return 2 * Cap::max_e;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return inner_.empty();
    }

    // =========================================================================
    // Adjacency access
    // =========================================================================

    /// All adjacent nodes (undirected: same as in + out for directed).
    /// This is the primary query for undirected algorithms.
    [[nodiscard]] constexpr auto
    neighbors(node_id u) const noexcept {
        return inner_.out_neighbors(u);
    }

    /// Alias for graph_queryable satisfaction.
    /// For a symmetric graph, out_neighbors == in_neighbors == neighbors.
    [[nodiscard]] constexpr auto
    out_neighbors(node_id u) const noexcept {
        return inner_.out_neighbors(u);
    }

    /// Degree (number of adjacent nodes).
    [[nodiscard]] constexpr std::size_t
    degree(node_id u) const noexcept {
        return inner_.out_degree(u);
    }

    /// Alias for graph_queryable compatibility.
    [[nodiscard]] constexpr std::size_t
    out_degree(node_id u) const noexcept {
        return inner_.out_degree(u);
    }

    /// Bounded adjacency extraction (constexpr-safe by-value).
    template<std::size_t MaxDegree>
    [[nodiscard]] constexpr auto
    neighbors_bounded(node_id u) const {
        return inner_.template out_neighbors_bounded<MaxDegree>(u);
    }

    // =========================================================================
    // Degree analysis
    // =========================================================================

    [[nodiscard]] constexpr std::size_t max_degree() const noexcept {
        return inner_.max_out_degree();
    }

    // =========================================================================
    // Node query
    // =========================================================================

    [[nodiscard]] constexpr bool has_node(node_id u) const noexcept {
        return inner_.has_node(u);
    }

    // =========================================================================
    // Edge position access (for weighted_view)
    // =========================================================================

    /// CSR offset where node u's edges begin (forwarded to inner directed graph).
    [[nodiscard]] constexpr std::size_t
    edge_begin_offset(node_id u) const noexcept {
        return inner_.edge_begin_offset(u);
    }

    /// Edge ID range for node u (forwarded to inner directed graph).
    [[nodiscard]] constexpr std::pair<edge_id, edge_id>
    edge_range(node_id u) const noexcept {
        return inner_.edge_range(u);
    }

    /// Target node of a specific edge (forwarded to inner directed graph).
    [[nodiscard]] constexpr node_id
    edge_target(edge_id e) const noexcept {
        return inner_.edge_target(e);
    }

    // =========================================================================
    // Topology token
    // =========================================================================

    /// Topology fingerprint (forwarded to inner directed graph).
    [[nodiscard]] constexpr topology_token token() const noexcept {
        return inner_.token();
    }

    // =========================================================================
    // Edge query
    // =========================================================================

    /// Check if u and v are adjacent (O(degree(u)) scan).
    [[nodiscard]] constexpr bool
    adjacent(node_id u, node_id v) const noexcept {
        for (auto const& w : inner_.out_neighbors(u)) {
            if (w == v) return true;
        }
        return false;
    }

    /// Access the inner directed graph (for algorithms that need CSR).
    [[nodiscard]] constexpr inner_graph_type const&
    directed() const noexcept { return inner_; }

private:
    inner_graph_type inner_{};

    template<typename>
    friend class symmetric_graph_builder;
};

// Verify concept satisfaction.
static_assert(graph_queryable<symmetric_graph<cap_from<8, 16>>>);
static_assert(sized_graph<symmetric_graph<cap_from<8, 16>>>);

// =============================================================================
// symmetric_graph_builder: auto-symmetrising construction
// =============================================================================

/// Builder for symmetric_graph.
///
/// add_edge(u, v) automatically inserts both u→v and v→u into the inner
/// graph_builder.  Self-edges are silently ignored (consistent with
/// graph_builder::finalise() dedup).
///
/// Cap::max_e is the number of UNDIRECTED edges.  Internally the builder
/// allocates capacity for 2×Cap::max_e directed edges.
///
/// Example:
/// ```cpp
/// symmetric_graph_builder<cap_from<4, 8>> b;
/// auto a = b.add_node();
/// auto c = b.add_node();
/// b.add_edge(a, c);          // adds a→c and c→a
/// auto g = b.finalise();     // symmetric_graph<cap_from<4, 8>>
/// ```
template<typename Cap = cap::medium>
class symmetric_graph_builder {
public:
    [[nodiscard]] constexpr node_id add_node() {
        return inner_.add_node();
    }

    [[nodiscard]] constexpr node_id add_nodes(std::size_t count) {
        return inner_.add_nodes(count);
    }

    /// Add an undirected edge {u, v}.
    /// Inserts both u→v and v→u.  Self-edges (u == v) are silently
    /// skipped (finalise would remove them anyway).
    /// Precondition: u and v must refer to nodes previously added via add_node().
    constexpr void add_edge(node_id u, node_id v) {
        if (u == v) return;  // no self-edges in undirected graphs
        // Validation delegated to inner builder's add_edge.
        inner_.add_edge(u, v);
        inner_.add_edge(v, u);
    }

    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return inner_.node_count();
    }

    /// Number of directed edges queued (= 2 × undirected edges added).
    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return inner_.edge_count();
    }

    /// Build the immutable symmetric_graph.
    [[nodiscard]] constexpr symmetric_graph<Cap> finalise() const {
        symmetric_graph<Cap> g;
        g.inner_ = inner_.finalise();
        return g;
    }

private:
    graph_builder<cap_from<Cap::max_v, 2 * Cap::max_e>> inner_{};
};

// =============================================================================
// Concept: symmetric_graph_queryable
// =============================================================================

/// A graph that guarantees symmetric adjacency.
///
/// Algorithms that require undirected input (coloring, min-cut, etc.)
/// should constrain on this concept rather than graph_queryable.
///
/// Satisfied by symmetric_graph.  NOT satisfied by constexpr_graph
/// (which is directed and may have asymmetric adjacency).
template<typename G>
concept symmetric_graph_queryable =
    graph_queryable<G> &&
    requires(G const& g, node_id u) {
        { g.neighbors(u) };
        { g.undirected_edge_count() } -> std::convertible_to<std::size_t>;
    };

static_assert(symmetric_graph_queryable<symmetric_graph<cap_from<8, 16>>>);
static_assert(!symmetric_graph_queryable<constexpr_graph<cap_from<8, 16>>>);

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SYMMETRIC_GRAPH_H
