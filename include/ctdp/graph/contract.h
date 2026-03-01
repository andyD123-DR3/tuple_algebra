// graph/transforms/contract.h — Graph contraction via node group merging
// Part of the compile-time DP library (C++20)
//
// ALGORITHM:
// Given a directed graph G and a group assignment (node → group_id),
// produce a contracted graph G' where:
// - Each group becomes a single super-node
// - Edges between groups are preserved (deduplicated)
// - Self-edges (intra-group) are removed
//
// COMPLEXITY: O(V + E)
//
// DESIGN RATIONALE:
// contract() is the pure topology version of coarsen().  The difference:
// - coarsen() merges both topology AND kernel_info properties
// - contract() operates on topology alone — no kernel_info dependency
//
// This separation follows the CT-DP principle that topology and semantics
// are orthogonal.  Property merging belongs to merge_rules (Phase 5).
//
// Use cases:
// - Simplify a graph after component analysis
// - Build multi-level hierarchies for iterative refinement
// - Pre-process before scheduling (collapse known-fusible regions)
//
// The group assignment uses property_map<uint16_t, MaxV> — the same
// format produced by connected_components, scc, and fuse_group.

#ifndef CTDP_GRAPH_CONTRACT_H
#define CTDP_GRAPH_CONTRACT_H

#include "capacity_guard.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include "property_map.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Result of graph contraction.
///
/// Contains the contracted graph, group count, and the group assignment
/// echoed back for convenience (callers often need it alongside the graph).
template<std::size_t MaxV, std::size_t MaxE>
struct contract_result {
    constexpr_graph<cap_from<MaxV, MaxE>> graph{};
    std::size_t group_count = 0;
};

/// Deduction alias: contract_result sized from graph traits.
template<typename G>
using contract_result_for = contract_result<
    graph_traits<G>::max_nodes,
    graph_traits<G>::max_edges>;

/// Contract a graph by collapsing groups of nodes into super-nodes.
///
/// Template parameters:
/// - OutCap: capacity policy for the contracted graph
/// - SrcMaxV: vertex capacity of the source property_map
/// - G: source graph type (must satisfy graph_queryable)
///
/// Parameters:
/// - g: the source graph
/// - group_of: maps each node to its group id ∈ [0, group_count)
/// - group_count: number of groups (= nodes in contracted graph)
///
/// The contracted graph has group_count nodes.  An edge exists from
/// super-node A to super-node B iff any node in group A has an edge
/// to any node in group B in the original graph.  Intra-group edges
/// (self-edges on super-nodes) are removed by finalise().
///
/// Example:
/// ```cpp
/// // Chain: 0→1→2→3
/// // Groups: {0,1} → group 0, {2,3} → group 1
/// constexpr auto g = make_chain4();
/// constexpr property_map<std::uint16_t, 8> groups(4, 0);
/// // groups[0]=0, groups[1]=0, groups[2]=1, groups[3]=1
/// constexpr auto cr = contract(g, groups, 2);
/// static_assert(cr.graph.node_count() == 2);
/// static_assert(cr.graph.edge_count() == 1); // group0 → group1
/// ```
template<typename OutCap, std::size_t SrcMaxV, graph_queryable G>
[[nodiscard]] constexpr contract_result<OutCap::max_v, OutCap::max_e>
contract(G const& g,
         property_map<std::uint16_t, SrcMaxV> const& group_of,
         std::size_t group_count) {
    constexpr std::size_t MaxV = OutCap::max_v;
    require_node_id_range<MaxV>();
    require_capacity(group_count, MaxV,
        "contract: group_count exceeds OutCap::max_v capacity");

    auto const V = g.node_count();
    guard_algorithm<SrcMaxV>(V, "contract: V exceeds SrcMaxV");

    // Validate group assignments.
    if (V > 0 && group_count > 0) {
        for (std::size_t i = 0; i < V; ++i) {
            auto const grp = static_cast<std::size_t>(group_of[i]);
            require_capacity(grp, group_count - 1,
                "contract: group_of[i] >= group_count (invalid group id)");
        }
    }

    contract_result<MaxV, OutCap::max_e> result{};
    result.group_count = group_count;

    // Build contracted graph: one node per group, cross-group edges only.
    graph_builder<OutCap> builder;
    for (std::size_t i = 0; i < group_count; ++i) {
        (void)builder.add_node();
    }

    for (std::size_t u = 0; u < V; ++u) {
        auto const gu = group_of[u];
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            auto const gv = group_of[v];
            if (gu != gv) {
                builder.add_edge(node_id{gu}, node_id{gv});
            }
        }
    }

    result.graph = builder.finalise();
    return result;
}

/// Convenience overload: deduce OutCap from the graph's traits.
template<std::size_t SrcMaxV, graph_queryable G>
[[nodiscard]] constexpr auto
contract(G const& g,
         property_map<std::uint16_t, SrcMaxV> const& group_of,
         std::size_t group_count) {
    using traits = graph_traits<G>;
    using out_cap = cap_from<traits::max_nodes, traits::max_edges>;
    return contract<out_cap>(g, group_of, group_count);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CONTRACT_H
