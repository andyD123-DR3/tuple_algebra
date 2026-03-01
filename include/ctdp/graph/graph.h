// graph/graph.h — Umbrella header for the CT-DP graph library
// Part of the compile-time DP library (C++20)
//
// Single-include convenience header.  Pulls in all graph library
// components: representation, algorithms, transforms, annotations,
// construction helpers, I/O, and traits.
//
// Usage:
//   #include <ctdp/graph/graph.h>
//
// This is intentionally heavyweight.  For compilation-time-sensitive
// translation units, prefer including individual headers.

#ifndef CTDP_GRAPH_GRAPH_H
#define CTDP_GRAPH_GRAPH_H

// --- Core types & concepts ---
#include "graph_concepts.h"
#include "capacity_types.h"
#include "capacity_guard.h"
#include "graph_traits.h"

// --- Representation ---
#include "constexpr_graph.h"
#include "symmetric_graph.h"
#include "implicit_graph.h"
#include "runtime_graph.h"
#include "graph_equal.h"

// --- Construction ---
#include "graph_builder.h"
#include "weighted_graph_builder.h"
#include "from_pipeline.h"
#include "from_stencil.h"

// --- Annotations ---
#include "property_map.h"
#include "kernel_info.h"
#include "edge_property_map.h"
#include "weighted_view.h"
#include "graph_capacity.h"

// --- Algorithms ---
#include "topological_sort.h"
#include "connected_components.h"
#include "scc.h"
#include "shortest_path.h"
#include "min_cut.h"
#include "graph_coloring.h"
#include "rcm.h"
#include "bipartite_graph.h"
#include "bipartite_matching.h"

// --- Transforms ---
#include "transpose.h"
#include "subgraph.h"
#include "contract.h"
#include "coarsen.h"
#include "fusion_legal.h"
#include "fuse_group.h"
#include "merge_rules.h"

// --- I/O ---
// graph_io.h is intentionally excluded from the umbrella header because
// its runtime I/O functions pull in <iostream> and <string>, which are
// expensive to compile.  Include graph_io.h directly when needed:
//   #include <ctdp/graph/graph_io.h>

// --- Helpers ---
#include "array_helpers.h"

#endif // CTDP_GRAPH_GRAPH_H
