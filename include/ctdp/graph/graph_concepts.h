// graph/representation/graph_concepts.h - Descriptor types and graph concepts
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Descriptors communicate intent and enforce separation between topology
// and semantics. node_id is an opaque handle — all semantics live in
// external property maps (BGL-style).
//
// graph_queryable is the single concept for Phase 4. All transforms are
// builder/rebuild-based — no graph_mutable concept until a concrete mutable
// type (adjacency_list) is specified.
//
// DESIGN LIMIT: uint16_t constrains all graphs to ≤65,535 nodes.
// ct_limits ceilings (64–512) are far below this.

#ifndef CTDP_GRAPH_CONCEPTS_H
#define CTDP_GRAPH_CONCEPTS_H

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// Descriptor Type
// =============================================================================

/// Opaque node identifier.
///
/// Valid only for the specific graph instance that produced it.
/// Structural transforms return new graphs with new descriptor spaces,
/// plus remappings to relate old nodes to new nodes.
struct node_id {
    std::uint16_t value{};

    friend constexpr bool operator==(node_id, node_id) = default;
    friend constexpr auto operator<=>(node_id, node_id) = default;
};

/// Convert node_id to index for array access.
[[nodiscard]] constexpr std::size_t to_index(node_id n) noexcept {
    return static_cast<std::size_t>(n.value);
}

/// Sentinel value for invalid/unassigned node references.
inline constexpr node_id invalid_node{std::uint16_t{0xFFFF}};

/// Opaque edge identifier (CSR position in the targets array).
///
/// Valid only for the specific graph instance that produced it.
/// Edge IDs are NOT stable across graph construction or transformation.
/// Persisting edge IDs beyond the lifetime of the graph instance is
/// undefined behaviour.  Use topology_token to detect stale IDs.
struct edge_id {
    std::size_t value{};

    friend constexpr bool operator==(edge_id, edge_id) = default;
    friend constexpr auto operator<=>(edge_id, edge_id) = default;
};

/// Convert edge_id to index for array access.
[[nodiscard]] constexpr std::size_t to_index(edge_id e) noexcept {
    return e.value;
}

/// Sentinel value for invalid/unassigned edge references.
inline constexpr edge_id invalid_edge{~std::size_t{0}};

/// Topology token — lightweight fingerprint of graph structure.
///
/// Deterministically derived from the CSR content (node count, edge count,
/// and a hash of the adjacency).  Two graphs with identical topology
/// produce identical tokens.  A graph rebuild, coarsening, or any
/// structural transform produces a different token.
///
/// Used by edge_property_map and weighted_view to detect stale or
/// mismatched edge annotations at constexpr evaluation time.
struct topology_token {
    std::uint64_t value{};

    friend constexpr bool operator==(topology_token, topology_token) = default;
};

// =============================================================================
// Graph Concept
// =============================================================================

/// A graph_queryable provides immutable adjacency queries.
///
/// Requirements:
/// - node_count(): number of nodes in the graph
/// - out_neighbors(u): returns range-like of node_id
///
/// Satisfied by:
/// - constexpr_graph<MaxV, MaxE>
/// - implicit_graph<Generator>
///
/// No graph_mutable concept in Phase 4. All transforms are
/// builder/rebuild-based via graph_builder::finalise().
template<typename G>
concept graph_queryable =
    requires(G const& g, node_id u) {
        { g.node_count() } -> std::convertible_to<std::size_t>;
        { g.out_neighbors(u) };
    };

/// A sized_graph additionally provides edge_count() and capacity queries.
/// constexpr_graph satisfies this; implicit_graph typically does not.
///
/// Capacity queries (node_capacity, edge_capacity) report the maximum
/// number of nodes/edges the graph's fixed-size storage can hold.
/// For constexpr graphs these are the MaxV/MaxE template parameters.
/// For runtime graphs these are the capacity set at construction.
///
/// Algorithms use capacity queries to size working arrays via
/// graph_traits<G>::make_node_array/make_edge_array.
template<typename G>
concept sized_graph =
    graph_queryable<G> &&
    requires(G const& g) {
        { g.edge_count() } -> std::convertible_to<std::size_t>;
        { g.node_capacity() } -> std::convertible_to<std::size_t>;
        { g.edge_capacity() } -> std::convertible_to<std::size_t>;
    };

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CONCEPTS_H
