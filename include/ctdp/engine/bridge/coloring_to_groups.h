// engine/bridge/coloring_to_groups.h - Map graph colouring to execution groups
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Graph colouring assigns colours so no two adjacent nodes share a colour.
// Same-colour nodes are independent and can execute in parallel.
//
// This bridge maps colour ids to group ids, producing a fuse_group_result
// that feeds into build_schedule_space_fused().
//
// SEMANTIC NOTE:
// Fusion groups (from find_fusion_groups) mean "nodes fused into one kernel".
// Colour groups (from here) mean "nodes that can execute in parallel".
// These are orthogonal concepts reusing the same group_id mechanism.
// The schedule_space consumer doesn't distinguish — it just uses group_id
// for co-scheduling.  If both fusion and parallelism are needed, the
// coloring should be applied AFTER fusion (colour the coarsened graph).

#ifndef CTDP_ENGINE_COLORING_TO_GROUPS_H
#define CTDP_ENGINE_COLORING_TO_GROUPS_H

#include "../../graph/graph_coloring.h"
#include "../../graph/fuse_group.h"

#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Convert a coloring_result into a fuse_group_result.
///
/// Each colour id becomes a group id.  Nodes with the same colour
/// (= independent set) are assigned to the same group.
///
/// The result is always a valid DAG grouping because independent
/// nodes have no edges between them.
///
/// Example:
/// ```cpp
/// constexpr auto sg = build_row_conflict_graph(pattern);
/// constexpr auto cr = graph_coloring<MaxV>(sg);
/// constexpr auto fg = coloring_to_groups(cr);
/// constexpr auto space = build_schedule_space_fused<MaxV, MaxE>(g, kmap, fg);
/// ```
template<std::size_t MaxV>
[[nodiscard]] constexpr fuse_group_result<MaxV>
coloring_to_groups(coloring_result<MaxV> const& cr) {
    fuse_group_result<MaxV> result;
    auto const V = cr.node_count;

    result.group_of = property_map<std::uint16_t, MaxV>(V, 0);
    for (std::size_t i = 0; i < V; ++i) {
        result.group_of[i] = cr.color_of[i];
    }
    result.group_count = cr.color_count;
    result.is_valid_dag = true;   // independent sets → no intra-group edges
    result.fused_edge_count = 0;  // colouring doesn't fuse edges

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_ENGINE_COLORING_TO_GROUPS_H
