// graph/construction/weighted_graph_builder.h - Factories for weighted graph views
// Part of the compile-time DP library (C++20)
//
// Factory functions create edge_property_map instances bound to a graph's
// topology_token.  This binding is checked by weighted_view's constructor,
// preventing stale or mismatched weight maps.
//
// DESIGN NOTE:
// We build the graph first, then populate weights by walking the finalised
// CSR structure.  This avoids weight/edge synchronisation bugs because
// edge CSR positions are only determined after finalise() sorts and deduplicates.
//
// For symmetric graphs, make_symmetric_weight_map guarantees that both
// directed edges (u→v and v→u) receive the same weight, which is the
// invariant required by symmetric_weighted_queryable.

#ifndef CTDP_GRAPH_WEIGHTED_GRAPH_BUILDER_H
#define CTDP_GRAPH_WEIGHTED_GRAPH_BUILDER_H

#include "edge_property_map.h"
#include "graph_concepts.h"
#include "symmetric_graph.h"
#include "weighted_view.h"

#include <cstddef>
#include <cstdint>


namespace ctdp::graph {

// =========================================================================
// Weight map factories — all bind the topology token automatically
// =========================================================================

/// Build an edge weight map by calling fn(src, dst) for every edge in CSR order.
///
/// The function receives (node_id src, node_id dst) and returns the weight.
/// The resulting map is bound to the graph's topology_token.
template<typename Weight, std::size_t MaxE, typename Graph, typename Fn>
    requires graph_queryable<Graph> &&
             requires(Graph const& g) {
                 { g.token() } -> std::same_as<topology_token>;
             }
[[nodiscard]] constexpr edge_property_map<Weight, MaxE>
make_weight_map(Graph const& g, Fn&& fn) {
    std::size_t total_edges = 0;
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        total_edges += g.out_degree(node_id{static_cast<std::uint16_t>(i)});
    }

    edge_property_map<Weight, MaxE> emap(total_edges, Weight{}, g.token());

    std::size_t edge_idx = 0;
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        auto const u = node_id{static_cast<std::uint16_t>(i)};
        for (auto const& v : g.out_neighbors(u)) {
            emap[edge_idx] = fn(u, v);
            ++edge_idx;
        }
    }
    return emap;
}

/// Build a symmetric edge weight map.
///
/// Calls fn(lo, hi) where lo = min(u,v), hi = max(u,v) for canonical
/// ordering.  Assigns the same weight to both directed edges u→v and v→u.
///
/// INVARIANT: after construction, weight(u→v) == weight(v→u) for all edges.
/// This is guaranteed by construction, not just by precondition.
///
/// The map is bound to the symmetric graph's topology_token.
template<typename Weight, std::size_t MaxE, std::size_t MaxV>
    requires (MaxE > 0)
[[nodiscard]] constexpr edge_property_map<Weight, 2 * MaxE>
make_symmetric_weight_map(
    symmetric_graph<cap_from<MaxV, MaxE>> const& g,
    auto&& fn)
{
    auto const& d = g.directed();
    edge_property_map<Weight, 2 * MaxE> emap(
        d.edge_count(), Weight{}, g.token());

    std::size_t edge_idx = 0;
    for (std::size_t i = 0; i < d.node_count(); ++i) {
        auto const u = node_id{static_cast<std::uint16_t>(i)};
        for (auto const& v : d.out_neighbors(u)) {
            auto const lo = (u.value <= v.value) ? u : v;
            auto const hi = (u.value <= v.value) ? v : u;
            emap[edge_idx] = fn(lo, hi);
            ++edge_idx;
        }
    }
    return emap;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_WEIGHTED_GRAPH_BUILDER_H
