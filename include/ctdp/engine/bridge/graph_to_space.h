// engine/bridge/graph_to_space.h - Graph-to-DP-space bridge
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// The DP engine works on descriptors (what to optimize) and candidates
// (choices per descriptor). The graph library works on nodes, edges,
// and kernel_info. This header bridges the two:
//
//   Graph world:  nodes, edges, kernel_info, fusion groups, topo order
//       ↓
//   DP world:     node_descriptors (sorted by topo), strategy choices,
//                 cost model, dependency constraints
//
// A node_descriptor bundles everything the DP solver needs about one
// node: its identity, kernel_info, position in topo order, group_id,
// and predecessor/successor lists for constraint checking.
//
// The schedule_space is the top-level object: an array of
// node_descriptors in topological order, ready for the DP engine to
// assign strategies and compute costs.
//
// USAGE:
// constexpr auto g = make_chain();
// constexpr auto km = make_uniform_kernel_map<8>(g, ...);
// constexpr auto space = build_schedule_space<8, 16>(g, km);
// // space.descriptors[0..3] in topo order, with kernel_info attached

#ifndef CTDP_GRAPH_TO_SPACE_H
#define CTDP_GRAPH_TO_SPACE_H

#include "../../graph/graph_concepts.h"
#include "../../graph/constexpr_graph.h"
#include "../../graph/kernel_info.h"
#include "../../graph/property_map.h"
#include "../../graph/topological_sort.h"
#include "../../graph/fuse_group.h"
#include "../../core/constexpr_vector.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// Node descriptor: per-node DP input
// =============================================================================

/// Everything the DP solver needs to know about one computation node.
///
/// Produced by build_schedule_space(). Consumed by the cost model and
/// constraint checker.
///
/// Fields:
/// - id:           original node_id in the source graph
/// - topo_rank:    position in topological order (0 = first to execute)
/// - group_id:     fusion group (nodes in same group execute together)
/// - info:         kernel_info (flops, bytes, tag, fusability)
/// - pred_count:   number of predecessor nodes (in-degree)
/// - succ_count:   number of successor nodes (out-degree)
struct node_descriptor {
    node_id id{};
    std::uint16_t topo_rank = 0;
    std::uint16_t group_id = 0;
    kernel_info info{};
    std::uint16_t pred_count = 0;
    std::uint16_t succ_count = 0;

    friend constexpr bool
    operator==(node_descriptor const&, node_descriptor const&) = default;
};

// =============================================================================
// Schedule space: the DP search space derived from a graph
// =============================================================================

/// A DP-ready representation of a computation graph.
///
/// Contains node_descriptors in topological order, plus summary
/// statistics. This is the primary input to graph-based DP solvers.
///
/// Template parameter:
/// - MaxV: maximum vertex count
template<std::size_t MaxV>
struct schedule_space {
    /// Node descriptors in topological order.
    /// descriptors[0] has no unsatisfied dependencies.
    ctdp::constexpr_vector<node_descriptor, MaxV> descriptors{};

    /// Number of fusion groups.
    std::size_t group_count = 0;

    /// True if the graph is a DAG (should always be true for valid inputs).
    bool is_dag = true;

    /// Lookup: original node_id → index in descriptors array.
    /// topo_index[n.value] = position in descriptors[] for node n.
    std::array<std::uint16_t, MaxV> topo_index{};

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    /// Number of nodes.
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return descriptors.size();
    }

    /// Access descriptor by topo rank.
    [[nodiscard]] constexpr node_descriptor const&
    operator[](std::size_t rank) const noexcept {
        return descriptors[rank];
    }

    /// Access descriptor by original node_id.
    [[nodiscard]] constexpr node_descriptor const&
    by_node(node_id n) const noexcept {
        return descriptors[topo_index[n.value]];
    }

    /// Get all descriptors in the same fusion group.
    [[nodiscard]] constexpr ctdp::constexpr_vector<node_descriptor, MaxV>
    group_members(std::uint16_t gid) const {
        ctdp::constexpr_vector<node_descriptor, MaxV> result;
        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            if (descriptors[i].group_id == gid) {
                result.push_back(descriptors[i]);
            }
        }
        return result;
    }

    /// Total flops across all nodes.
    [[nodiscard]] constexpr std::size_t total_flops() const noexcept {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            sum += descriptors[i].info.flops;
        }
        return sum;
    }

    /// Total bytes across all nodes.
    [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            sum += descriptors[i].info.total_bytes();
        }
        return sum;
    }
};

// =============================================================================
// Builder: graph + kernel_map → schedule_space
// =============================================================================

/// Build a schedule_space from a graph and its kernel annotations.
///
/// Steps:
/// 1. Topological sort → execution order
/// 2. Compute in-degree / out-degree per node
/// 3. Assign topo_rank to each node
/// 4. Each node gets its own fusion group (singleton groups)
///
/// For fusion-aware scheduling, use build_schedule_space_fused()
/// which accepts pre-computed fusion groups.
///
/// Template parameters:
/// - MaxV: vertex capacity
/// - MaxE: edge capacity (for internal topo sort)
///
/// Example:
/// ```cpp
/// constexpr auto g = make_chain();
/// constexpr auto km = make_uniform_kernel_map<8>(g, ...);
/// constexpr auto space = build_schedule_space<8, 16>(g, km);
/// static_assert(space.size() == 4);
/// static_assert(space[0].topo_rank == 0);
/// ```
template<std::size_t MaxV, std::size_t MaxE, graph_queryable G>
[[nodiscard]] constexpr schedule_space<MaxV>
build_schedule_space(G const& g, kernel_map<MaxV> const& kmap) {
    schedule_space<MaxV> space;
    auto const V = g.node_count();

    if (V == 0) return space;

    // Step 1: Topological sort.
    auto const topo = topological_sort(g);
    space.is_dag = topo.is_dag;

    if (!topo.is_dag) {
        // Can't schedule a cyclic graph. Return empty.
        return space;
    }

    // Step 2: Compute in-degree and out-degree.
    std::array<std::uint16_t, MaxV> in_deg{};
    std::array<std::uint16_t, MaxV> out_deg{};
    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            out_deg[u]++;
            in_deg[v.value]++;
        }
    }

    // Step 3: Build descriptors in topo order.
    for (std::size_t rank = 0; rank < topo.order.size(); ++rank) {
        auto const nid = topo.order[rank];
        node_descriptor desc;
        desc.id = nid;
        desc.topo_rank = static_cast<std::uint16_t>(rank);
        desc.group_id = nid.value;  // singleton group
        desc.info = kmap[nid];
        desc.pred_count = in_deg[nid.value];
        desc.succ_count = out_deg[nid.value];
        space.descriptors.push_back(desc);
        space.topo_index[nid.value] = static_cast<std::uint16_t>(rank);
    }

    space.group_count = V;  // singleton groups
    return space;
}

/// Build a schedule_space with pre-computed fusion groups.
///
/// Nodes in the same fusion group share a group_id. The DP solver
/// can use this to co-schedule fused nodes.
///
/// Example:
/// ```cpp
/// constexpr auto fg = find_fusion_groups<8, 16>(g, kmap);
/// constexpr auto space = build_schedule_space_fused<8, 16>(g, kmap, fg);
/// // Nodes in same fusion group have same group_id
/// ```
template<std::size_t MaxV, std::size_t MaxE, graph_queryable G>
[[nodiscard]] constexpr schedule_space<MaxV>
build_schedule_space_fused(G const& g,
                           kernel_map<MaxV> const& kmap,
                           fuse_group_result<MaxV> const& fg) {
    auto space = build_schedule_space<MaxV, MaxE>(g, kmap);

    // Overwrite singleton group_ids with fusion group assignments.
    for (std::size_t i = 0; i < space.descriptors.size(); ++i) {
        auto const nid = space.descriptors[i].id;
        space.descriptors[i].group_id = fg.group_of[nid];
    }
    space.group_count = fg.group_count;

    return space;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_TO_SPACE_H
