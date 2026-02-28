// graph/annotation/edge_property_map.h - External property map for graph edges
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE (Option C — weighted_view wrapper):
//
// Edge properties live external to graph topology.  weighted_view
// bundles graph + map into a single object modelling weighted_graph_queryable.
//
// BINDING INVARIANT:
// Each edge_property_map records the topology_token and edge_count of the
// graph it was built for.  weighted_view checks these at construction.
// This turns "weights silently wrong after rebuild" into an immediate
// constexpr failure.
//
// Edge IDs are NOT stable across graph construction or transformation.
// Persisting edge IDs beyond the lifetime of the graph instance is
// undefined behaviour.

#ifndef CTDP_GRAPH_EDGE_PROPERTY_MAP_H
#define CTDP_GRAPH_EDGE_PROPERTY_MAP_H

#include "capacity_guard.h"
#include "graph_concepts.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ctdp::graph {

/// External property map: edge CSR position → Value.
///
/// Indexed by edge_id (or raw std::size_t for backward compat).
/// Stores one Value per edge up to MaxE.
///
/// Records the topology_token of the graph it was created for.
/// weighted_view checks this token at construction to prevent
/// stale or mismatched weight maps.
///
/// Template parameters:
/// - Value: the property type (e.g. double for weights, int for capacities)
/// - MaxE:  maximum edge capacity (matches constexpr_graph<MaxV, MaxE>::max_edges)
template<typename Value, std::size_t MaxE>
class edge_property_map {
    std::array<Value, MaxE> data_{};
    std::size_t size_ = 0;
    topology_token token_{};

public:
    constexpr edge_property_map() = default;

    /// Construct with known size and default value (no token binding).
    /// Prefer the factory functions which bind the token automatically.
    constexpr edge_property_map(std::size_t n, Value default_val)
        : size_(n)
    {
        require_capacity(n, MaxE,
            "edge_property_map: initial size exceeds MaxE");
        for (std::size_t i = 0; i < n; ++i) {
            data_[i] = default_val;
        }
    }

    /// Construct with token binding.
    constexpr edge_property_map(std::size_t n, Value default_val,
                                 topology_token tok)
        : size_(n), token_(tok)
    {
        require_capacity(n, MaxE,
            "edge_property_map: initial size exceeds MaxE");
        for (std::size_t i = 0; i < n; ++i) {
            data_[i] = default_val;
        }
    }

    // -----------------------------------------------------------------
    // Size and binding
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] constexpr topology_token token() const noexcept {
        return token_;
    }

    constexpr void resize(std::size_t n) {
        require_capacity(n, MaxE,
            "edge_property_map::resize: size exceeds MaxE");
        size_ = n;
    }

    constexpr void resize(std::size_t n, Value const& val) {
        require_capacity(n, MaxE,
            "edge_property_map::resize: size exceeds MaxE");
        for (std::size_t i = 0; i < n; ++i) {
            data_[i] = val;
        }
        size_ = n;
    }

    /// Bind to a graph's topology token after manual construction.
    constexpr void bind(topology_token tok) noexcept {
        token_ = tok;
    }

    // -----------------------------------------------------------------
    // Element access — by edge_id (preferred) or raw index
    // -----------------------------------------------------------------

    constexpr Value& operator[](edge_id e) {
        if (e.value >= size_)
            throw std::out_of_range("edge_property_map: edge_id out of bounds");
        return data_[e.value];
    }

    constexpr Value const& operator[](edge_id e) const {
        if (e.value >= size_)
            throw std::out_of_range("edge_property_map: edge_id out of bounds");
        return data_[e.value];
    }

    constexpr Value& operator[](std::size_t idx) {
        if (idx >= size_)
            throw std::out_of_range("edge_property_map: index out of bounds");
        return data_[idx];
    }

    constexpr Value const& operator[](std::size_t idx) const {
        if (idx >= size_)
            throw std::out_of_range("edge_property_map: index out of bounds");
        return data_[idx];
    }

    // -----------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr bool
    operator==(edge_property_map const& other) const noexcept {
        if (size_ != other.size_) return false;
        if (!(token_ == other.token_)) return false;
        for (std::size_t i = 0; i < size_; ++i) {
            if (!(data_[i] == other.data_[i])) return false;
        }
        return true;
    }
};

// BGL-style free functions

template<typename V, std::size_t M>
[[nodiscard]] constexpr V const&
get(edge_property_map<V, M> const& emap, edge_id e) {
    return emap[e];
}

template<typename V, std::size_t M>
constexpr void
put(edge_property_map<V, M>& emap, edge_id e, V const& val) {
    emap[e] = val;
}

// Backward-compatible overloads (raw index)
template<typename V, std::size_t M>
[[nodiscard]] constexpr V const&
get(edge_property_map<V, M> const& emap, std::size_t idx) {
    return emap[idx];
}

template<typename V, std::size_t M>
constexpr void
put(edge_property_map<V, M>& emap, std::size_t idx, V const& val) {
    emap[idx] = val;
}

// =========================================================================
// Factories — these bind the topology token automatically
// =========================================================================

/// Create an edge_property_map with uniform initial value, bound to a graph.
template<typename Value, std::size_t MaxE, sized_graph G>
    requires requires(G const& g) { { g.token() } -> std::same_as<topology_token>; }
[[nodiscard]] constexpr edge_property_map<Value, MaxE>
make_uniform_edge_map(G const& g, Value const& val) {
    return edge_property_map<Value, MaxE>(g.edge_count(), val, g.token());
}

/// Create an edge_property_map by applying a function to each edge index.
template<typename Value, std::size_t MaxE, sized_graph G, typename Fn>
    requires requires(G const& g) { { g.token() } -> std::same_as<topology_token>; }
[[nodiscard]] constexpr edge_property_map<Value, MaxE>
make_edge_map(G const& g, Fn&& fn) {
    require_capacity(g.edge_count(), MaxE,
        "make_edge_map: graph edge_count exceeds MaxE");
    edge_property_map<Value, MaxE> emap(g.edge_count(),
                                         Value{}, g.token());
    for (std::size_t i = 0; i < g.edge_count(); ++i) {
        emap[i] = fn(i);
    }
    return emap;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_EDGE_PROPERTY_MAP_H
