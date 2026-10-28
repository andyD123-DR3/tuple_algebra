// graph/transforms/fuse_group.h - Fusion group identification
// Part of the compile-time DP library (C++20)
//
// ALGORITHM:
// 1. Find all legal fusion pairs (via fusion_legal)
// 2. Build undirected "fusability graph" from those pairs
// 3. Find connected components → candidate fusion groups
// 4. Validate: coarsened graph must remain a DAG
// 5. If cyclic, split offending groups (greedy fallback: each node alone)
//
// COMPLEXITY: O(V + E) for the common case
//
// DESIGN RATIONALE:
// Fusion group discovery is the bridge between legality (what *can* fuse)
// and coarsening (the contracted graph). The key constraint is that
// fusing nodes must not introduce cycles — otherwise the computation
// has no valid execution order.
//
// For DAG inputs (the normal case for computation graphs), the
// connected-components approach produces maximal fusion groups that
// respect acyclicity. For graphs with pre-existing cycles, SCC
// analysis handles the cyclic substructure.

#ifndef CTDP_GRAPH_FUSE_GROUP_H
#define CTDP_GRAPH_FUSE_GROUP_H

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "kernel_info.h"
#include "property_map.h"
#include "fusion_legal.h"
#include "topological_sort.h"
#include "connected_components.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

/// Result of fusion group analysis.
///
/// - group_of[n]: group id for node n (0-based, dense)
/// - group_count: number of groups (= number of super-nodes after coarsening)
/// - is_valid_dag: true if the coarsened graph is a DAG
/// - fused_edge_count: number of edges eliminated by fusion
template<std::size_t MaxV>
struct fuse_group_result {
    property_map<std::uint16_t, MaxV> group_of{};
    std::size_t group_count = 0;
    bool is_valid_dag = true;
    std::size_t fused_edge_count = 0;
};

/// Identify fusion groups: maximal sets of adjacent fusable nodes.
///
/// Strategy:
/// 1. Build an undirected graph of fusion-eligible edges
/// 2. Connected components → candidate groups
/// 3. Coarsen and check DAG property
/// 4. If coarsened graph is cyclic, fall back to singleton groups
///
/// Template parameters:
/// - MaxV: vertex capacity
/// - MaxE: edge capacity
/// - Policy: fusion policy (default: same_tag_policy)
///
/// Example:
/// ```cpp
/// // Chain: 0→1→2→3, all same tag + fusable
/// constexpr auto g = make_chain();
/// constexpr auto kmap = make_uniform_kernel_map<8>(g,
///     kernel_info{.tag = kernel_tag{1}, .is_fusable = true});
/// constexpr auto fg = find_fusion_groups<8, 16>(g, kmap);
/// // All 4 nodes fuse into 1 group (chain is a DAG, no cycle risk)
/// static_assert(fg.group_count == 1);
/// ```
template<std::size_t MaxV, std::size_t MaxE,
         graph_queryable G,
         typename Policy = same_tag_policy>
[[nodiscard]] constexpr fuse_group_result<MaxV>
find_fusion_groups(G const& g,
                   kernel_map<MaxV> const& kmap,
                   Policy policy = {}) {
    fuse_group_result<MaxV> result;
    auto const V = g.node_count();
    guard_algorithm<MaxV>(V, "find_fusion_groups: V exceeds MaxV");

    if (V == 0) {
        return result;
    }

    // Step 1: Find legal fusion pairs.
    auto const legal = fusion_pairs<MaxV, MaxE>(g, kmap, policy);

    if (legal.pairs.size() == 0) {
        // No fusable edges → each node is its own group.
        result.group_of.resize(V);
        for (std::size_t i = 0; i < V; ++i) {
            result.group_of[i] = static_cast<std::uint16_t>(i);
        }
        result.group_count = V;
        result.is_valid_dag = true;
        result.fused_edge_count = 0;
        return result;
    }

    // Step 2: Connected components of eligible edges, using disjoint sets.
    // The earlier undirected graph inserted two directed edges for every
    // eligible input edge but allocated only MaxE slots. Even a 3-node,
    // 2-edge chain overflowed a perfectly valid MaxE=2 input capacity.
    // Union-find avoids the duplicate-edge allocation entirely.
    std::array<std::size_t, MaxV> parent{};
    for (std::size_t i = 0; i < V; ++i) parent[i] = i;
    auto find_root = [&](std::size_t i) constexpr -> std::size_t {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (std::size_t i = 0; i < legal.pairs.size(); ++i) {
        const auto a = find_root(legal.pairs[i].u.value);
        const auto b = find_root(legal.pairs[i].v.value);
        if (a != b) parent[b] = a;
    }
    std::array<std::size_t, MaxV> root_group{};
    root_group.fill(MaxV);
    std::array<std::size_t, MaxV> component_of{};
    std::size_t component_count = 0;
    for (std::size_t i = 0; i < V; ++i) {
        const auto root = find_root(i);
        if (root_group[root] == MaxV)
            root_group[root] = component_count++;
        component_of[i] = root_group[root];
    }

    // Step 4: Validate acyclicity of the coarsened graph.
    // Build the coarsened graph directly (lightweight check).
    graph_builder<cap_from<MaxV, MaxE>> cb;
    for (std::size_t i = 0; i < component_count; ++i) {
        [[maybe_unused]] auto n = cb.add_node();
    }

    std::size_t cross_edges = 0;
    std::size_t intra_edges = 0;
    for (std::size_t u = 0; u < V; ++u) {
        auto const gu = component_of[u];
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            auto const gv = component_of[v.value];
            if (gu != gv) {
                cb.add_edge(node_id{gu}, node_id{gv});
                cross_edges++;
            } else {
                intra_edges++;
            }
        }
    }

    auto const cgraph = cb.finalise();
    auto const topo = topological_sort(cgraph);

    if (topo.is_dag) {
        // Groups are valid — coarsened graph is acyclic.
        result.group_of.resize(V);
        for (std::size_t i = 0; i < V; ++i) {
            result.group_of[i] = static_cast<std::uint16_t>(component_of[i]);
        }
        result.group_count = component_count;
        result.is_valid_dag = true;
        result.fused_edge_count = intra_edges;
    } else {
        // Fallback: coarsened graph has cycles.
        // Conservative: revert to singleton groups.
        result.group_of.resize(V);
        for (std::size_t i = 0; i < V; ++i) {
            result.group_of[i] = static_cast<std::uint16_t>(i);
        }
        result.group_count = V;
        result.is_valid_dag = false;
        result.fused_edge_count = 0;
    }

    return result;
}

/// Count how many nodes are in each fusion group.
///
/// Returns property_map<std::size_t, MaxV> where entry[g] is the
/// size of group g.
template<std::size_t MaxV>
[[nodiscard]] constexpr property_map<std::size_t, MaxV>
group_sizes(fuse_group_result<MaxV> const& fg) {
    property_map<std::size_t, MaxV> sizes(fg.group_count, 0);
    for (std::size_t i = 0; i < fg.group_of.size(); ++i) {
        sizes[fg.group_of[i]]++;
    }
    return sizes;
}

/// Maximum group size (largest fusion group).
template<std::size_t MaxV>
[[nodiscard]] constexpr std::size_t
max_group_size(fuse_group_result<MaxV> const& fg) {
    auto const sizes = group_sizes(fg);
    std::size_t max_sz = 0;
    for (std::size_t i = 0; i < fg.group_count; ++i) {
        if (sizes[i] > max_sz) max_sz = sizes[i];
    }
    return max_sz;
}

/// Number of groups containing more than one node (actual fusions).
template<std::size_t MaxV>
[[nodiscard]] constexpr std::size_t
fused_group_count(fuse_group_result<MaxV> const& fg) {
    auto const sizes = group_sizes(fg);
    std::size_t count = 0;
    for (std::size_t i = 0; i < fg.group_count; ++i) {
        if (sizes[i] > 1) count++;
    }
    return count;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_FUSE_GROUP_H
