// graph/annotation/weighted_view.h - Weighted graph view (Option C wrapper)
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE — Why Option C (weighted_view wrapper)?
//
// Three options were evaluated for adding weighted edge support:
//
//   Option A — BGL-style external only:
//     Pro:  Zero changes to existing types.
//     Con:  No concept enforces agreement.  Noisy algorithm signatures.
//
//   Option B — Baked-in weights in CSR:
//     Pro:  Locality during constexpr evaluation.
//     Con:  Dead storage in unweighted algorithms.  Template creep.
//           Multiple weight interpretations require graph duplication.
//
//   Option C — weighted_view wrapper (chosen):
//     Pro:  constexpr_graph unchanged.  Unweighted algorithms unaffected.
//           Multiple weight interpretations on same topology.
//           Concept-enforced availability of weight access.
//           Constructor enforces size and topology_token agreement.
//     Con:  One accessor (edge_begin_offset) added to constexpr_graph.
//           Acceptable at ≤256 node ceiling.
//
// BINDING MODEL:
// weighted_view is a non-owning view.  The graph and edge_property_map
// must outlive the view.  The constructor checks:
//   1. weights.size() == graph.edge_count()     (size agreement)
//   2. weights.token() == graph.token()          (topology agreement)
// A mismatch is a constexpr evaluation failure (immediate, not silent).
//
// SYMMETRIC WEIGHT INVARIANT:
// For symmetric_weighted_queryable, symmetry is:
//   - Guaranteed by construction when using make_symmetric_weight_map
//   - Validated by check when constructing via verify_symmetric_weights
//   - Required by precondition otherwise
// Algorithms constraining on symmetric_weighted_queryable can rely on
// weight(u→v) == weight(v→u).

#ifndef CTDP_GRAPH_WEIGHTED_VIEW_H
#define CTDP_GRAPH_WEIGHTED_VIEW_H

#include "constexpr_graph.h"
#include "edge_property_map.h"
#include "graph_concepts.h"
#include "symmetric_graph.h"
#include <ctdp/core/constexpr_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ctdp::graph {

// =========================================================================
// weighted_edge — a single (target, weight, eid) triple
// =========================================================================

template<typename Weight>
struct weighted_edge {
    node_id target;
    Weight  weight;
    edge_id eid;  // CSR position — for property updates

    constexpr bool operator==(weighted_edge const&) const = default;
};

// =========================================================================
// weighted_adjacency_range — lightweight range over weighted edges
// =========================================================================

template<typename Weight, std::size_t MaxE>
class weighted_adjacency_range {
public:
    constexpr weighted_adjacency_range(
        node_id const* nbr_begin,
        node_id const* nbr_end,
        edge_property_map<Weight, MaxE> const* weights,
        std::size_t edge_offset) noexcept
        : nbr_begin_(nbr_begin)
        , nbr_end_(nbr_end)
        , weights_(weights)
        , edge_offset_(edge_offset)
    {}

    struct iterator {
        node_id const* ptr;
        edge_property_map<Weight, MaxE> const* weights;
        std::size_t edge_idx;

        constexpr weighted_edge<Weight> operator*() const {
            return {*ptr, (*weights)[edge_idx], edge_id{edge_idx}};
        }

        constexpr iterator& operator++() {
            ++ptr;
            ++edge_idx;
            return *this;
        }

        constexpr bool operator!=(iterator const& other) const {
            return ptr != other.ptr;
        }
        constexpr bool operator==(iterator const& other) const {
            return ptr == other.ptr;
        }
    };

    [[nodiscard]] constexpr iterator begin() const noexcept {
        return {nbr_begin_, weights_, edge_offset_};
    }
    [[nodiscard]] constexpr iterator end() const noexcept {
        auto count = static_cast<std::size_t>(nbr_end_ - nbr_begin_);
        return {nbr_end_, weights_, edge_offset_ + count};
    }
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return static_cast<std::size_t>(nbr_end_ - nbr_begin_);
    }
    [[nodiscard]] constexpr bool empty() const noexcept {
        return nbr_begin_ == nbr_end_;
    }

private:
    node_id const* nbr_begin_;
    node_id const* nbr_end_;
    edge_property_map<Weight, MaxE> const* weights_;
    std::size_t edge_offset_;
};

// =========================================================================
// weighted_view — bundles graph + edge weight map
// =========================================================================

/// Immutable non-owning view combining a graph with edge weights.
///
/// Models weighted_graph_queryable.  The graph and edge_property_map
/// must outlive this view.
///
/// Constructor enforces:
///   1. weights.size() == graph.edge_count()  (size agreement)
///   2. weights.token() == graph.token()       (topology agreement, if bound)
///
/// No default constructor — prevents dangling empty views.
template<typename Graph, typename Weight, std::size_t MaxE>
class weighted_view {
    Graph const* graph_;
    edge_property_map<Weight, MaxE> const* weights_;

public:
    using weight_type = Weight;
    static constexpr std::size_t max_edges = MaxE;

    /// No default constructor — view must always be bound.
    weighted_view() = delete;

    /// Construct with binding checks.
    ///
    /// Throws (or fails consteval) if:
    ///   - weights.size() != graph.edge_count()
    ///   - weights have a non-zero token that doesn't match graph.token()
    constexpr weighted_view(Graph const& graph,
                             edge_property_map<Weight, MaxE> const& weights)
        : graph_(&graph), weights_(&weights)
    {
        if (weights.size() != graph.edge_count())
            throw std::logic_error(
                "weighted_view: weight map size != graph edge count");
        // Token check: if the map was bound (non-zero token), verify match.
        // Unbound maps (token == 0, from manual construction) skip this.
        if (weights.token().value != 0 &&
            !(weights.token() == graph.token()))
            throw std::logic_error(
                "weighted_view: topology token mismatch — "
                "weight map was built for a different graph");
    }

    // -----------------------------------------------------------------
    // graph_queryable forwarding
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return graph_->node_count();
    }

    [[nodiscard]] constexpr auto out_neighbors(node_id u) const noexcept {
        return graph_->out_neighbors(u);
    }

    [[nodiscard]] constexpr std::size_t out_degree(node_id u) const noexcept {
        return graph_->out_degree(u);
    }

    [[nodiscard]] constexpr bool has_node(node_id u) const noexcept {
        return graph_->has_node(u);
    }

    // -----------------------------------------------------------------
    // sized_graph forwarding
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return graph_->edge_count();
    }

    // -----------------------------------------------------------------
    // Edge position forwarding (the "one true primitive")
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr std::size_t
    edge_begin_offset(node_id u) const noexcept {
        return graph_->edge_begin_offset(u);
    }

    [[nodiscard]] constexpr std::pair<edge_id, edge_id>
    edge_range(node_id u) const noexcept {
        return graph_->edge_range(u);
    }

    [[nodiscard]] constexpr node_id
    edge_target(edge_id e) const noexcept {
        return graph_->edge_target(e);
    }

    [[nodiscard]] constexpr topology_token token() const noexcept {
        return graph_->token();
    }

    // -----------------------------------------------------------------
    // Weighted queries — the new capability
    // -----------------------------------------------------------------

    /// Weight of a specific edge by edge_id.
    [[nodiscard]] constexpr Weight
    edge_weight(edge_id e) const {
        return (*weights_)[e];
    }

    /// Weight of a specific edge by raw index (backward compat).
    [[nodiscard]] constexpr Weight
    edge_weight(std::size_t idx) const {
        return (*weights_)[idx];
    }

    /// Weighted adjacency range for node u.
    [[nodiscard]] constexpr auto
    weighted_out_neighbors(node_id u) const noexcept {
        auto nbr = graph_->out_neighbors(u);
        auto offset = graph_->edge_begin_offset(u);
        return weighted_adjacency_range<Weight, MaxE>(
            nbr.begin(), nbr.end(), weights_, offset);
    }

    /// Bounded weighted adjacency (by-value, constexpr-safe).
    template<std::size_t MaxDegree>
    [[nodiscard]] constexpr
    ctdp::constexpr_vector<weighted_edge<Weight>, MaxDegree>
    weighted_out_neighbors_bounded(node_id u) const {
        ctdp::constexpr_vector<weighted_edge<Weight>, MaxDegree> result{};
        auto offset = graph_->edge_begin_offset(u);
        auto nbr = graph_->out_neighbors(u);
        std::size_t i = 0;
        for (auto const& v : nbr) {
            result.push_back({v, (*weights_)[offset + i],
                              edge_id{offset + i}});
            ++i;
        }
        return result;
    }

    // -----------------------------------------------------------------
    // Access to underlying components
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr Graph const& graph() const noexcept {
        return *graph_;
    }

    [[nodiscard]] constexpr edge_property_map<Weight, MaxE> const&
    weights() const noexcept {
        return *weights_;
    }
};

// =========================================================================
// Factory
// =========================================================================

template<typename Graph, typename Weight, std::size_t MaxE>
[[nodiscard]] constexpr auto
make_weighted_view(Graph const& graph,
                   edge_property_map<Weight, MaxE> const& weights) {
    return weighted_view<Graph, Weight, MaxE>(graph, weights);
}

// =========================================================================
// Symmetric weight validation
// =========================================================================

/// Verify that edge weights are symmetric: for every edge (u,v) with
/// weight w, the reverse edge (v,u) also has weight w.
///
/// Intended for constexpr validation when constructing symmetric
/// weighted views.  Cost: O(E × max_degree) — fine at CT-DP graph
/// sizes (≤512 nodes).
///
/// Returns true if all reverse edges exist and have matching weights.
template<std::size_t MaxV, std::size_t MaxE, typename Weight>
[[nodiscard]] constexpr bool
verify_symmetric_weights(
    symmetric_graph<MaxV, MaxE> const& g,
    edge_property_map<Weight, 2 * MaxE> const& weights)
{
    auto const& d = g.directed();
    for (std::size_t i = 0; i < d.node_count(); ++i) {
        auto const u = node_id{static_cast<std::uint16_t>(i)};
        auto u_offset = d.edge_begin_offset(u);
        std::size_t k = 0;
        for (auto const& v : d.out_neighbors(u)) {
            Weight w_uv = weights[u_offset + k];
            // Find reverse edge v→u
            auto v_offset = d.edge_begin_offset(v);
            std::size_t m = 0;
            bool found = false;
            for (auto const& w : d.out_neighbors(v)) {
                if (w == u) {
                    if (!(weights[v_offset + m] == w_uv))
                        return false;  // weight asymmetry
                    found = true;
                    break;
                }
                ++m;
            }
            if (!found) return false;  // missing reverse edge
            ++k;
        }
    }
    return true;
}

// =========================================================================
// Concepts
// =========================================================================

/// A weighted graph provides edge weight access alongside adjacency.
///
/// Refines graph_queryable.  Algorithms that need edge weights constrain
/// on this.  Algorithms that don't constrain on graph_queryable — unchanged.
///
/// Concept-enforced availability of weight access; constructor enforces
/// size/topology agreement.
template<typename G>
concept weighted_graph_queryable =
    graph_queryable<G> &&
    requires(G const& g, node_id u) {
        typename G::weight_type;
        { g.edge_weight(edge_id{0}) } -> std::convertible_to<typename G::weight_type>;
        { g.weighted_out_neighbors(u) };
    };

/// A symmetric weighted graph: undirected + weighted.
///
/// Guarantees that for every edge (u,v) with weight w, the reverse
/// edge (v,u) also has weight w.  Required by Stoer-Wagner min-cut.
///
/// Symmetry is:
///   - Guaranteed by construction (make_symmetric_weight_map), OR
///   - Validated by check (verify_symmetric_weights), OR
///   - Required by precondition
/// The concept checks structural capability; symmetry of values is
/// an invariant enforced by the construction/validation path chosen.
template<typename G>
concept symmetric_weighted_queryable =
    weighted_graph_queryable<G> &&
    requires(G const& g, node_id u) {
        { g.graph().neighbors(u) };
        { g.graph().undirected_edge_count() } -> std::convertible_to<std::size_t>;
    };

// =========================================================================
// Concept verification
// =========================================================================

namespace detail_verify {
template<std::size_t V, std::size_t E>
using test_directed_weighted =
    weighted_view<constexpr_graph<V, E>, double, E>;
} // namespace detail_verify

static_assert(graph_queryable<
    detail_verify::test_directed_weighted<8, 16>>);
static_assert(weighted_graph_queryable<
    detail_verify::test_directed_weighted<8, 16>>);
static_assert(!symmetric_weighted_queryable<
    detail_verify::test_directed_weighted<8, 16>>);

} // namespace ctdp::graph

#endif // CTDP_GRAPH_WEIGHTED_VIEW_H
