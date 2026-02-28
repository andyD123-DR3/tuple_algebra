// engine/bridge/graph_types.h - Type aliases for engine use
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// The engine uses fixed-capacity graph types with standard bounds.
// Rather than scatter MaxV/MaxE literals throughout engine code, this
// header provides named aliases for common configurations.
//
// Engine code should use these aliases exclusively:
//   engine_graph      — directed DAG for scheduling
//   engine_sym_graph  — undirected conflict/interference graph
//   engine_rt_graph   — runtime-constructed equivalent
//
// This also provides result type aliases so engine code doesn't
// need to spell out MaxV in result templates.

#ifndef CTDP_ENGINE_GRAPH_TYPES_H
#define CTDP_ENGINE_GRAPH_TYPES_H

#include "../../graph/constexpr_graph.h"
#include "../../graph/symmetric_graph.h"
#include "../../graph/runtime_graph.h"
#include "../../graph/graph_traits.h"
#include "../../graph/graph_coloring.h"
#include "../../graph/connected_components.h"
#include "../../graph/scc.h"
#include "../../graph/topological_sort.h"
#include "../../graph/shortest_path.h"
#include "../../graph/min_cut.h"
#include "../../graph/kernel_info.h"

namespace ctdp::engine {

// =============================================================================
// Standard engine capacity bounds
// =============================================================================

/// Standard vertex capacity for engine graphs.
/// Covers typical kernel DAGs (up to 64 compute kernels).
inline constexpr std::size_t engine_max_v = 64;

/// Standard edge capacity for engine graphs.
/// Supports dense-ish DAGs: up to 256 dependency edges.
inline constexpr std::size_t engine_max_e = 256;

// =============================================================================
// Graph type aliases
// =============================================================================

/// Directed DAG for scheduling / dependency analysis.
using engine_graph = ctdp::graph::constexpr_graph<engine_max_v, engine_max_e>;

/// Undirected graph for conflict analysis (coloring, min-cut).
using engine_sym_graph = ctdp::graph::symmetric_graph<engine_max_v, engine_max_e>;

/// Runtime-constructed directed graph.
using engine_rt_graph = ctdp::graph::runtime_graph<engine_max_v, engine_max_e>;

// =============================================================================
// Graph builder aliases
// =============================================================================

using engine_graph_builder = ctdp::graph::graph_builder<engine_max_v, engine_max_e>;
using engine_sym_builder = ctdp::graph::symmetric_graph_builder<engine_max_v, engine_max_e>;
using engine_rt_builder = ctdp::graph::runtime_graph_builder<engine_max_v, engine_max_e>;

// =============================================================================
// Result type aliases
// =============================================================================

using engine_topo_result = ctdp::graph::topo_result<engine_max_v>;
using engine_cc_result = ctdp::graph::components_result<engine_max_v>;
using engine_scc_result = ctdp::graph::scc_result<engine_max_v>;
using engine_coloring_result = ctdp::graph::coloring_result<engine_max_v>;
using engine_sp_result = ctdp::graph::shortest_path_result<engine_max_v>;
using engine_min_cut_result = ctdp::graph::min_cut_result<engine_max_v>;

// =============================================================================
// Kernel map alias
// =============================================================================

using engine_kernel_map = ctdp::graph::kernel_map<engine_max_v>;

} // namespace ctdp::engine

#endif // CTDP_ENGINE_GRAPH_TYPES_H
