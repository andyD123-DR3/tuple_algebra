// graph/array_helpers.h — Free helper functions for traits-based array construction
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Algorithms call make_node_array<T>(g) rather than directly invoking
// graph_traits<G>::template make_node_array<T>(g.node_capacity()).
// These one-line delegators provide a cleaner call-site syntax and
// automatically extract capacity from the graph.

#ifndef CTDP_GRAPH_ARRAY_HELPERS_H
#define CTDP_GRAPH_ARRAY_HELPERS_H

#include "graph_traits.h"

#include <cstddef>

namespace ctdp::graph {

// =============================================================================
// make_node_array — construct a node-indexed array from a graph
// =============================================================================

/// Construct a value-initialised node array sized to g.node_capacity().
template<typename T, typename G>
constexpr auto make_node_array(G const& g) {
    return graph_traits<G>::template make_node_array<T>(g.node_capacity());
}

/// Construct a filled node array sized to g.node_capacity().
template<typename T, typename G>
constexpr auto make_node_array(G const& g, T const& fill) {
    return graph_traits<G>::template make_node_array<T>(g.node_capacity(), fill);
}

// =============================================================================
// make_edge_array — construct an edge-indexed array from a graph
// =============================================================================

/// Construct a value-initialised edge array sized to g.edge_capacity().
template<typename T, typename G>
constexpr auto make_edge_array(G const& g) {
    return graph_traits<G>::template make_edge_array<T>(g.edge_capacity());
}

/// Construct a filled edge array sized to g.edge_capacity().
template<typename T, typename G>
constexpr auto make_edge_array(G const& g, T const& fill) {
    return graph_traits<G>::template make_edge_array<T>(g.edge_capacity(), fill);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_ARRAY_HELPERS_H
