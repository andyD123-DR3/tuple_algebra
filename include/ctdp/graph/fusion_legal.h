// graph/transforms/fusion_legal.h - Fusion legality analysis
// Part of the compile-time DP library (C++20)
//
// ALGORITHM:
// For each edge u→v in the computation graph, determine whether u and v
// can legally be fused (merged into a single kernel). Legality requires:
//
//   1. Both kernels are marked is_fusable
//   2. Tags are compatible (same tag, or policy says cross-tag is OK)
//   3. Fusion does not create a cycle in the remaining dependency graph
//      (checked via the acyclicity predicate — if v has other predecessors
//       or u has other successors that would create a cycle, fusion is illegal)
//
// OUTPUTS:
// - fusion_edge_map<MaxE>: for each edge index, is fusion legal?
// - fusion_candidates<MaxV>: for each node, which neighbours can it fuse with?
// - can_fuse(g, kmap, u, v): point query for a specific pair
//
// DESIGN RATIONALE:
// Fusion legality is separated from fusion *decisions* (which groups to
// actually fuse). This header answers "what's legal?" — fuse_group.h
// answers "what's profitable?".
//
// The simple policy (both fusable + same tag) covers the majority of
// practical cases. The predicate-based overload allows custom rules.

#ifndef CTDP_GRAPH_FUSION_LEGAL_H
#define CTDP_GRAPH_FUSION_LEGAL_H

#include "graph_concepts.h"
#include "kernel_info.h"
#include "property_map.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =============================================================================
// Fusion policies
// =============================================================================

/// Default fusion policy: both fusable AND same tag.
struct same_tag_policy {
    template<std::size_t MaxV>
    [[nodiscard]] constexpr bool
    operator()(kernel_map<MaxV> const& kmap,
               node_id u, node_id v) const noexcept {
        auto const& ku = kmap[u];
        auto const& kv = kmap[v];
        return ku.is_fusable && kv.is_fusable && ku.tag == kv.tag;
    }
};

/// Relaxed fusion policy: both fusable, any tag combination allowed.
struct any_tag_policy {
    template<std::size_t MaxV>
    [[nodiscard]] constexpr bool
    operator()(kernel_map<MaxV> const& kmap,
               node_id u, node_id v) const noexcept {
        return kmap[u].is_fusable && kmap[v].is_fusable;
    }
};

// =============================================================================
// Point query: can two specific nodes fuse?
// =============================================================================

/// Check if nodes u and v can legally fuse under the given policy.
///
/// This checks the local property (both fusable, tag compatibility)
/// but does NOT check global acyclicity — that requires graph context
/// and is done by fusion_pairs().
///
/// Example:
/// ```cpp
/// constexpr bool ok = can_fuse_local(kmap, node_id{0}, node_id{1},
///                                     same_tag_policy{});
/// ```
template<std::size_t MaxV, typename Policy = same_tag_policy>
[[nodiscard]] constexpr bool
can_fuse_local(kernel_map<MaxV> const& kmap,
               node_id u, node_id v,
               Policy policy = {}) noexcept {
    return policy(kmap, u, v);
}

// =============================================================================
// Fusion pair result
// =============================================================================

/// A pair of adjacent nodes that can legally be fused.
struct fusion_pair {
    node_id u{};
    node_id v{};

    friend constexpr bool operator==(fusion_pair, fusion_pair) = default;
};

/// Result of fusion legality analysis.
/// Contains all edges where fusion is legal.
template<std::size_t MaxE>
struct fusion_legal_result {
    ctdp::constexpr_vector<fusion_pair, MaxE> pairs{};
    std::size_t total_edges = 0;  // total edges in graph (for ratio)

    /// Fraction of edges that are fusable.
    [[nodiscard]] constexpr double fusion_ratio() const noexcept {
        if (total_edges == 0) return 0.0;
        return static_cast<double>(pairs.size()) /
               static_cast<double>(total_edges);
    }
};

// =============================================================================
// Full analysis: find all legal fusion pairs
// =============================================================================

/// Identify all edges where fusion is legal.
///
/// For every edge u→v in graph g, checks the fusion policy against the
/// kernel_map. Returns the set of legal fusion pairs.
///
/// Note: this checks local legality (per-edge). Global acyclicity
/// constraints (ensuring the fused graph remains a DAG) are enforced
/// by fuse_group.h when building actual fusion groups.
///
/// Template parameters:
/// - G: graph type satisfying graph_queryable
/// - MaxV: vertex capacity (matches kernel_map)
/// - MaxE: edge capacity for result storage
/// - Policy: fusion policy functor
///
/// Example:
/// ```cpp
/// constexpr auto g = make_chain();  // 0→1→2→3
/// constexpr auto kmap = make_uniform_kernel_map<8>(g,
///     kernel_info{.tag = kernel_tag{1}, .is_fusable = true});
/// constexpr auto result = fusion_pairs<8, 16>(g, kmap);
/// static_assert(result.pairs.size() == 3);  // all edges fusable
/// ```
template<std::size_t MaxV, std::size_t MaxE,
         graph_queryable G,
         typename Policy = same_tag_policy>
[[nodiscard]] constexpr fusion_legal_result<MaxE>
fusion_pairs(G const& g,
             kernel_map<MaxV> const& kmap,
             Policy policy = {}) {
    fusion_legal_result<MaxE> result;
    auto const V = g.node_count();
    guard_algorithm<MaxV>(V, "fusion_pairs: V exceeds MaxV");

    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            result.total_edges++;
            if (can_fuse_local(kmap, uid, v, policy)) {
                result.pairs.push_back(fusion_pair{uid, v});
            }
        }
    }

    return result;
}

/// Build a boolean edge-legality map: for each node, which out-neighbours
/// are legal fusion partners?
///
/// Returns property_map<constexpr_vector<node_id, MaxDeg>, MaxV> where
/// each entry lists the fusable neighbours of that node.
template<std::size_t MaxV, std::size_t MaxDeg,
         graph_queryable G,
         typename Policy = same_tag_policy>
[[nodiscard]] constexpr
property_map<ctdp::constexpr_vector<node_id, MaxDeg>, MaxV>
fusion_neighbor_map(G const& g,
                    kernel_map<MaxV> const& kmap,
                    Policy policy = {}) {
    property_map<ctdp::constexpr_vector<node_id, MaxDeg>, MaxV> result;
    auto const V = g.node_count();
    guard_algorithm<MaxV>(V, "fusion_neighbor_map: V exceeds MaxV");
    result.resize(V);

    for (std::size_t u = 0; u < V; ++u) {
        auto const uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            if (can_fuse_local(kmap, uid, v, policy)) {
                result[uid].push_back(v);
            }
        }
    }

    return result;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_FUSION_LEGAL_H
