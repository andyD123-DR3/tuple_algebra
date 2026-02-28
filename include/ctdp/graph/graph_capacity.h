// graph/graph_capacity.h — Convenience aliases for graph types with capacity policies
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Graph templates take a single Cap struct that bundles max_v and max_e.
// Named capacity policies (cap::tiny through cap::xlarge) provide reusable,
// self-documenting tiers.  This header provides convenience aliases that
// bind cap policies to concrete graph/builder types:
//
//   auto b = directed_builder<cap::small>{};
//   auto n0 = b.add_node();
//   ...
//   auto g = b.finalise();
//
// instead of:
//
//   graph_builder<cap_from<16, 64>> b;
//   ...
//
// Namespace scoping pattern:
//
//   namespace my_project {
//       using Cap = cap::large;
//       constexpr_graph<Cap> g;
//       graph_builder<Cap>   b;
//   }

#ifndef CTDP_GRAPH_CAPACITY_H
#define CTDP_GRAPH_CAPACITY_H

#include "capacity_types.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "symmetric_graph.h"

namespace ctdp::graph {

// =============================================================================
// Convenience aliases — named-policy → concrete type
// =============================================================================

/// Directed graph with capacity from a named policy.
template<capacity_policy Cap = cap::medium>
using directed = constexpr_graph<Cap>;

/// Builder for directed graphs with capacity from a named policy.
template<capacity_policy Cap = cap::medium>
using directed_builder = graph_builder<Cap>;

/// Undirected (symmetric) graph with capacity from a named policy.
template<capacity_policy Cap = cap::medium>
using undirected = symmetric_graph<Cap>;

/// Builder for undirected graphs with capacity from a named policy.
template<capacity_policy Cap = cap::medium>
using undirected_builder = symmetric_graph_builder<Cap>;

// =============================================================================
// Runtime-intent aliases — same underlying types, named for clarity
// =============================================================================
//
// These aliases are identical to the directed/undirected versions but signal
// that the graph will be constructed from runtime data (files, user input).

/// Runtime graph — same type as directed<Cap>, named for intent.
template<capacity_policy Cap = cap::medium>
using rt_graph = constexpr_graph<Cap>;

/// Runtime builder — same type as directed_builder<Cap>, named for intent.
template<capacity_policy Cap = cap::medium>
using rt_builder = graph_builder<Cap>;

/// Runtime symmetric builder — same type as undirected_builder<Cap>.
template<capacity_policy Cap = cap::medium>
using symmetric_rt_builder = symmetric_graph_builder<Cap>;

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CAPACITY_H
