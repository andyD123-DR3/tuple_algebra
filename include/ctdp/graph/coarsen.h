// graph/transforms/coarsen.h - Graph coarsening via node grouping
// Part of the compile-time DP library (C++20)
//
// ALGORITHM:
// Given a graph G and a group assignment (node → group_id), produce a
// coarsened graph G' where:
// - Each group becomes a single super-node
// - Edges between groups are preserved (deduplicated)
// - Self-edges (within a group) are removed
// - kernel_info is merged via merged_with() for grouped nodes
//
// COMPLEXITY: O(V + E)
//
// DESIGN RATIONALE:
// Coarsening is the workhorse of hierarchical graph optimization.
// After fusion_legal identifies what *can* fuse, and fuse_group decides
// what *should* fuse, coarsen builds the contracted graph for the next
// level of analysis (e.g., scheduling the fused super-nodes).
//
// The group assignment is a property_map<uint16_t, MaxV> where
// group_of[node] ∈ [0, group_count). This matches the output format
// of connected_components and scc.

#ifndef CTDP_GRAPH_COARSEN_H
#define CTDP_GRAPH_COARSEN_H

#include "capacity_guard.h"
#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "kernel_info.h"
#include "property_map.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Result of graph coarsening.
///
/// Contains the coarsened graph, the merged kernel_map, and the mapping
/// from original nodes to super-nodes (same as input group_of).
template<std::size_t MaxV, std::size_t MaxE>
struct coarsen_result {
    constexpr_graph<MaxV, MaxE> graph{};
    kernel_map<MaxV> kernels{};
    std::size_t group_count = 0;
};

/// Coarsen a graph by collapsing groups of nodes into super-nodes.
///
/// Template parameters:
/// - MaxV: vertex capacity for coarsened graph
/// - MaxE: edge capacity for coarsened graph
/// - SrcMaxV: vertex capacity of source kernel_map
/// - G: source graph type
///
/// Parameters:
/// - g: the source graph
/// - kmap: kernel_info for each node in g
/// - group_of: maps each node to its group id ∈ [0, group_count)
/// - group_count: number of groups (= nodes in coarsened graph)
///
/// The coarsened graph has group_count nodes, with edges between
/// groups (no self-edges, deduplicated by graph_builder::finalise).
/// Kernel info is merged: all nodes in a group contribute to the
/// super-node's aggregate flops/bytes/fusability.
///
/// Example:
/// ```cpp
/// // Chain 0→1→2→3, group {0,1} and {2,3}
/// constexpr auto g = make_chain();
/// constexpr auto kmap = make_uniform_kernel_map<8>(g,
///     kernel_info{.flops = 10, .bytes_read = 40});
/// constexpr property_map<std::uint16_t, 8> groups(4, 0);
/// // groups[0]=0, groups[1]=0, groups[2]=1, groups[3]=1
/// constexpr auto cr = coarsen<8, 16>(g, kmap, groups, 2);
/// static_assert(cr.graph.node_count() == 2);
/// static_assert(cr.kernels[node_id{0}].flops == 20);  // merged
/// ```
template<std::size_t MaxV, std::size_t MaxE,
         std::size_t SrcMaxV,
         graph_queryable G>
[[nodiscard]] constexpr coarsen_result<MaxV, MaxE>
coarsen(G const& g,
        kernel_map<SrcMaxV> const& kmap,
        property_map<std::uint16_t, SrcMaxV> const& group_of,
        std::size_t group_count) {
    // --- Capacity guards ---
    require_node_id_range<MaxV>();
    require_capacity(group_count, MaxV,
        "coarsen: group_count exceeds MaxV capacity of output graph");

    // --- Input capacity guard ---
    auto const V = g.node_count();
    guard_algorithm<SrcMaxV>(V, "coarsen: V exceeds MaxV");
    if (V > 0 && group_count > 0) {
        for (std::size_t i = 0; i < V; ++i) {
            // operator[] on group_of validates i < group_of.size()
            auto const grp = static_cast<std::size_t>(group_of[i]);
            require_capacity(grp, group_count - 1,
                "coarsen: group_of[i] >= group_count (invalid group id)");
        }
    }

    coarsen_result<MaxV, MaxE> result;
    result.group_count = group_count;

    // Step 1: Build coarsened kernel_map by merging within groups.
    result.kernels.resize(group_count);
    std::array<bool, MaxV> first_in_group{};
    for (std::size_t i = 0; i < group_count; ++i) {
        first_in_group[i] = true;
    }

    for (std::size_t i = 0; i < V; ++i) {
        auto const grp = group_of[i];
        if (first_in_group[grp]) {
            result.kernels[grp] = kmap[i];
            first_in_group[grp] = false;
        } else {
            result.kernels[grp] =
                result.kernels[grp].merged_with(kmap[i]);
        }
    }

    // Step 2: Build coarsened graph edges.
    graph_builder<MaxV, MaxE> builder;
    for (std::size_t i = 0; i < group_count; ++i) {
        [[maybe_unused]] auto n = builder.add_node();
    }

    for (std::size_t u = 0; u < V; ++u) {
        auto const gu = group_of[u];
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            auto const gv = group_of[v];
            if (gu != gv) {
                // Cross-group edge → edge in coarsened graph.
                // Duplicates and self-edges handled by finalise().
                builder.add_edge(
                    node_id{gu},
                    node_id{gv});
            }
        }
    }

    result.graph = builder.finalise();
    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_COARSEN_H
