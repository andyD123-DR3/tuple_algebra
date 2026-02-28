// graph/construction/from_pipeline.h - Linear pipeline graph factory
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Pipelines (linear chains of processing stages) are a primary domain for
// kernel fusion. from_pipeline(N) creates an explicit constexpr_graph
// with edges 0→1→2→...→N-1.

#ifndef CTDP_GRAPH_FROM_PIPELINE_H
#define CTDP_GRAPH_FROM_PIPELINE_H

#include "constexpr_graph.h"
#include "graph_builder.h"

#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Create a linear pipeline graph: 0→1→2→...→N-1
///
/// Template parameters:
/// - MaxV: Maximum vertex capacity (must be >= N)
/// - MaxE: Maximum edge capacity (must be >= N-1; defaults to MaxV)
///
/// Parameters:
/// - N: Number of pipeline stages (nodes)
///
/// Returns: constexpr_graph<cap_from<MaxV, MaxE>> with N nodes and N-1 edges
///
/// Example:
/// ```cpp
/// constexpr auto g = from_pipeline<8>(5);
/// // Graph: 0→1→2→3→4
/// static_assert(g.node_count() == 5);
/// static_assert(g.edge_count() == 4);
/// ```
template<std::size_t MaxV, std::size_t MaxE = MaxV>
[[nodiscard]] constexpr constexpr_graph<cap_from<MaxV, MaxE>>
from_pipeline(std::size_t N) {
    graph_builder<cap_from<MaxV, MaxE>> b;

    if (N == 0) {
        return b.finalise();
    }

    auto prev = b.add_node();
    for (std::size_t i = 1; i < N; ++i) {
        auto cur = b.add_node();
        b.add_edge(prev, cur);
        prev = cur;
    }

    return b.finalise();
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_FROM_PIPELINE_H
