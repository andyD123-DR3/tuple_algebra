// graph/symmetric_graph.h — Undirected graph representation
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// constexpr_graph is directed: edges u→v and v→u are distinct.  Many
// optimisation problems operate on inherently undirected structures:
// conflict graphs (coloring), interference graphs, compatibility graphs.
//
// symmetric_graph<MaxV, MaxE> wraps constexpr_graph<MaxV, 2*MaxE>.
// The builder enforces symmetry: add_edge(u, v) inserts BOTH directions.
// MaxE is the number of UNDIRECTED edges (logical edges).  The internal
// CSR stores 2*MaxE directed edges.
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
#include "graph_builder.h"
#include "graph_concepts.h"

#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// symmetric_graph: undirected graph via symmetric CSR
// =============================================================================

/// Fixed-capacity, immutable, undirected graph.
///
/// Template parameters:
/// - MaxV: Maximum number of vertices
/// - MaxE: Maximum number of UNDIRECTED edges (logical edges)
///
/// Internally stores 2×MaxE directed edges in a constexpr_graph.
/// out_neighbors(u) returns ALL adjacent nodes — there is no distinction
/// between in-neighbours and out-neighbours.
///
/// Constructed exclusively via symmetric_graph_builder::finalise().
///
/// Example:
/// ```cpp
/// constexpr auto g = []() {
///     symmetric_graph_builder<4, 8> b;
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
template<std::size_t MaxV, std::size_t MaxE>
class symmetric_graph {
    static_assert(MaxV <= 65535,
        "symmetric_graph: MaxV exceeds uint16_t range (65535)");
public:
    static constexpr std::size_t max_vertices = MaxV;
    static constexpr std::size_t max_undirected_edges = MaxE;
    static constexpr std::size_t max_directed_edges = 2 * MaxE;

    using inner_graph_type = constexpr_graph<MaxV, 2 * MaxE>;

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

    template<std::size_t, std::size_t>
    friend class symmetric_graph_builder;
};

// Verify concept satisfaction.
static_assert(graph_queryable<symmetric_graph<8, 16>>);
static_assert(sized_graph<symmetric_graph<8, 16>>);

// =============================================================================
// symmetric_graph_builder: auto-symmetrising construction
// =============================================================================

/// Builder for symmetric_graph.
///
/// add_edge(u, v) automatically inserts both u→v and v→u into the inner
/// graph_builder.  Self-edges are silently ignored (consistent with
/// graph_builder::finalise() dedup).
///
/// MaxE is the number of UNDIRECTED edges.  Internally the builder
/// allocates capacity for 2×MaxE directed edges.
///
/// Example:
/// ```cpp
/// symmetric_graph_builder<4, 8> b;
/// auto a = b.add_node();
/// auto c = b.add_node();
/// b.add_edge(a, c);          // adds a→c and c→a
/// auto g = b.finalise();     // symmetric_graph<4, 8>
/// ```
template<std::size_t MaxV, std::size_t MaxE>
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
    [[nodiscard]] constexpr symmetric_graph<MaxV, MaxE> finalise() const {
        symmetric_graph<MaxV, MaxE> g;
        g.inner_ = inner_.finalise();
        return g;
    }

private:
    graph_builder<MaxV, 2 * MaxE> inner_{};
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

static_assert(symmetric_graph_queryable<symmetric_graph<8, 16>>);
static_assert(!symmetric_graph_queryable<constexpr_graph<8, 16>>);

} // namespace ctdp::graph

#endif // CTDP_GRAPH_SYMMETRIC_GRAPH_H
