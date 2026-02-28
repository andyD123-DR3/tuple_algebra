// graph/representation/constexpr_graph.h - CSR graph representation
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// The primary explicit representation is a value-level, fixed-capacity graph.
// Internal format is CSR-like (offsets + adjacency) for minimal overhead and
// deterministic iteration.
//
// MaxDegree is NOT a graph template parameter. The graph stores arbitrary-degree
// nodes within the MaxE edge budget. Bounded adjacency extraction is an
// algorithm-level choice via out_neighbors_bounded<K>().
//
// KEY INVARIANT: constexpr_graph is immutable after construction. Transforms
// return new graphs. This avoids descriptor invalidation and keeps constexpr
// reasoning tractable.
//
// ADJACENCY ACCESS CONTRACT:
// - out_neighbors(): lightweight range over internal CSR storage. Safe within
//   a single constexpr expression.
// - out_neighbors_bounded<K>(): copies into constexpr_vector. For constexpr
//   contexts where adjacency must persist across statements.

#ifndef CTDP_GRAPH_CONSTEXPR_GRAPH_H
#define CTDP_GRAPH_CONSTEXPR_GRAPH_H

#include "graph_concepts.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ctdp::graph {

// Forward declaration for friend access.
template<std::size_t, std::size_t>
class graph_builder;

/// Fixed-capacity, immutable, CSR-format directed graph.
///
/// Template parameters:
/// - MaxV: Maximum number of vertices (capacity, not actual count)
/// - MaxE: Maximum number of directed edges (capacity, not actual count)
///
/// The graph is constructed via graph_builder::finalise() which produces
/// canonicalised adjacency (sorted by (u, v), deduplicated, no self-edges).
///
/// Example:
/// ```cpp
/// constexpr auto g = []() {
///     graph_builder<8, 16> b;
///     auto n0 = b.add_node();
///     auto n1 = b.add_node();
///     auto n2 = b.add_node();
///     b.add_edge(n0, n1);
///     b.add_edge(n0, n2);
///     b.add_edge(n1, n2);
///     return b.finalise();
/// }();
/// static_assert(g.node_count() == 3);
/// static_assert(g.edge_count() == 3);
/// ```
template<std::size_t MaxV, std::size_t MaxE>
class constexpr_graph {
    static_assert(MaxV <= 65535,
        "constexpr_graph: MaxV exceeds uint16_t range (65535)");
public:
    using size_type = std::uint16_t;

    static constexpr std::size_t max_vertices = MaxV;
    static constexpr std::size_t max_edges = MaxE;

    constexpr constexpr_graph() = default;

    // =========================================================================
    // Size queries
    // =========================================================================

    /// Number of vertices in the graph.
    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return V_;
    }

    /// Number of directed edges in the graph.
    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return E_;
    }

    /// Maximum number of vertices this graph can hold (template parameter MaxV).
    [[nodiscard]] constexpr std::size_t node_capacity() const noexcept {
        return MaxV;
    }

    /// Maximum number of directed edges this graph can hold (template parameter MaxE).
    [[nodiscard]] constexpr std::size_t edge_capacity() const noexcept {
        return MaxE;
    }

    /// True if the graph has no vertices.
    [[nodiscard]] constexpr bool empty() const noexcept {
        return V_ == 0;
    }

    // =========================================================================
    // Adjacency access — unbounded (lightweight range)
    // =========================================================================

    /// Lightweight range over internal CSR neighbor storage.
    ///
    /// Safe within a single constexpr expression. Algorithms that store
    /// adjacency across statements in constexpr contexts must use
    /// out_neighbors_bounded<K>() instead.
    struct adjacency_range {
        node_id const* begin_;
        node_id const* end_;

        [[nodiscard]] constexpr node_id const* begin() const noexcept {
            return begin_;
        }
        [[nodiscard]] constexpr node_id const* end() const noexcept {
            return end_;
        }
        [[nodiscard]] constexpr std::size_t size() const noexcept {
            return static_cast<std::size_t>(end_ - begin_);
        }
        [[nodiscard]] constexpr bool empty() const noexcept {
            return begin_ == end_;
        }
    };

    /// Returns a lightweight range over u's out-neighbors.
    ///
    /// Precondition: to_index(u) < node_count()
    [[nodiscard]] constexpr adjacency_range
    out_neighbors(node_id u) const noexcept {
        auto const idx = static_cast<std::size_t>(u.value);
        auto const b = static_cast<std::size_t>(offsets_[idx]);
        auto const e = static_cast<std::size_t>(offsets_[idx + 1]);
        return {neighbors_.data() + b, neighbors_.data() + e};
    }

    /// Out-degree of a specific node.
    [[nodiscard]] constexpr std::size_t
    out_degree(node_id u) const noexcept {
        auto const idx = static_cast<std::size_t>(u.value);
        return static_cast<std::size_t>(offsets_[idx + 1]) -
               static_cast<std::size_t>(offsets_[idx]);
    }

    // =========================================================================
    // Adjacency access — bounded (constexpr-safe by-value)
    // =========================================================================

    /// Returns adjacency as constexpr_vector by value.
    ///
    /// Constexpr-safe: no pointer lifetime issues across evaluation boundaries.
    /// The K ceiling is an algorithm choice — if a node's degree exceeds K,
    /// push_back hits the vector's capacity assertion.
    ///
    /// Precondition: to_index(u) < node_count()
    /// Precondition: out_degree(u) <= MaxDegree
    template<std::size_t MaxDegree>
    [[nodiscard]] constexpr ctdp::constexpr_vector<node_id, MaxDegree>
    out_neighbors_bounded(node_id u) const {
        ctdp::constexpr_vector<node_id, MaxDegree> result{};
        auto const idx = static_cast<std::size_t>(u.value);
        auto const b = static_cast<std::size_t>(offsets_[idx]);
        auto const e = static_cast<std::size_t>(offsets_[idx + 1]);
        for (auto i = b; i < e; ++i) {
            result.push_back(neighbors_[i]);
        }
        return result;
    }

    // =========================================================================
    // Edge position access (for external edge property maps)
    // =========================================================================

    /// CSR offset where node u's outgoing edges begin.
    ///
    /// The edges from node u occupy CSR positions
    /// [edge_begin_offset(u), edge_begin_offset(u) + out_degree(u)).
    /// Edge IDs are valid only for THIS graph instance — they are NOT
    /// stable across graph construction or transformation.
    ///
    /// Precondition: to_index(u) < node_count()
    [[nodiscard]] constexpr std::size_t
    edge_begin_offset(node_id u) const noexcept {
        return static_cast<std::size_t>(
            offsets_[static_cast<std::size_t>(u.value)]);
    }

    /// Edge ID range for node u: [begin, end) as a pair of edge_id.
    ///
    /// This is the "one true primitive" for edge identity — all weighted
    /// iteration can be built from edge_range + edge target lookup.
    [[nodiscard]] constexpr std::pair<edge_id, edge_id>
    edge_range(node_id u) const noexcept {
        auto const idx = static_cast<std::size_t>(u.value);
        return {edge_id{static_cast<std::size_t>(offsets_[idx])},
                edge_id{static_cast<std::size_t>(offsets_[idx + 1])}};
    }

    /// Target node of a specific edge (by CSR position).
    [[nodiscard]] constexpr node_id
    edge_target(edge_id e) const noexcept {
        return neighbors_[e.value];
    }

    // =========================================================================
    // Topology token — fingerprint for graph↔map binding
    // =========================================================================

    /// Deterministic fingerprint of this graph's topology.
    ///
    /// Derived from (V, E, CSR adjacency) via FNV-1a.  Two graphs with
    /// identical topology produce identical tokens.  Any structural
    /// transform (coarsen, rebuild, filter) produces a different token.
    [[nodiscard]] constexpr topology_token token() const noexcept {
        // FNV-1a 64-bit
        std::uint64_t h = 14695981039346656037ULL;
        auto mix = [&](std::uint64_t byte) constexpr {
            h ^= byte;
            h *= 1099511628211ULL;
        };
        mix(V_);
        mix(E_);
        for (std::size_t i = 0; i <= V_; ++i) {
            mix(static_cast<std::uint64_t>(offsets_[i]));
        }
        for (std::size_t i = 0; i < E_; ++i) {
            mix(static_cast<std::uint64_t>(neighbors_[i].value));
        }
        return topology_token{h};
    }

    // =========================================================================
    // Degree analysis
    // =========================================================================

    /// Maximum out-degree across all nodes.
    [[nodiscard]] constexpr std::size_t max_out_degree() const noexcept {
        std::size_t max_deg = 0;
        for (std::size_t v = 0; v < V_; ++v) {
            auto const deg = static_cast<std::size_t>(offsets_[v + 1]) -
                             static_cast<std::size_t>(offsets_[v]);
            if (deg > max_deg) {
                max_deg = deg;
            }
        }
        return max_deg;
    }

    // =========================================================================
    // Node iteration
    // =========================================================================

    /// Check if a node_id is valid for this graph.
    [[nodiscard]] constexpr bool
    has_node(node_id u) const noexcept {
        return static_cast<std::size_t>(u.value) < V_;
    }

private:
    std::size_t V_ = 0;
    std::size_t E_ = 0;

    // CSR format: offsets_[v] .. offsets_[v+1] gives the range of
    // neighbors for node v in the neighbors_ array.
    // offsets_ has V_+1 valid entries; offsets_[V_] == E_.
    std::array<size_type, MaxV + 1> offsets_{};
    std::array<node_id, MaxE> neighbors_{};

    // graph_builder needs private access to populate CSR arrays.
    template<std::size_t, std::size_t>
    friend class graph_builder;

    // graph_equal needs access to compare internal arrays.
    template<std::size_t MV, std::size_t ME>
    friend constexpr bool graph_equal(
        constexpr_graph<MV, ME> const& a,
        constexpr_graph<MV, ME> const& b) noexcept;
};

// Verify concept satisfaction.
static_assert(graph_queryable<constexpr_graph<8, 16>>);
static_assert(sized_graph<constexpr_graph<8, 16>>);

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CONSTEXPR_GRAPH_H
