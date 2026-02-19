// graph/annotation/property_map.h - External property map for graph nodes
// Part of the compile-time DP library (C++20)
//
// BGL-style external property maps — graph topology is separate from
// semantic annotations. Array-based storage: O(1) access by node_id.

#ifndef CTDP_GRAPH_PROPERTY_MAP_H
#define CTDP_GRAPH_PROPERTY_MAP_H

#include "capacity_guard.h"
#include "graph_concepts.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ctdp::graph {

/// External property map: node_id → Value.
/// Array-backed, O(1) access. Stores one Value per node up to MaxV.
template<typename Value, std::size_t MaxV>
class property_map {
    std::array<Value, MaxV> data_{};
    std::size_t size_ = 0;

public:
    constexpr property_map() = default;

    constexpr property_map(std::size_t n, Value default_val)
        : size_(n)
    {
        require_capacity(n, MaxV,
            "property_map: initial size exceeds MaxV");
        for (std::size_t i = 0; i < n; ++i) {
            data_[i] = default_val;
        }
    }

    // -----------------------------------------------------------------
    // Size management
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return size_;
    }

    /// Set active size (new elements keep their default-constructed value).
    constexpr void resize(std::size_t n) {
        require_capacity(n, MaxV,
            "property_map::resize: size exceeds MaxV");
        size_ = n;
    }

    /// Set active size and fill [0, n) with val.
    constexpr void resize(std::size_t n, Value const& val) {
        require_capacity(n, MaxV,
            "property_map::resize: size exceeds MaxV");
        for (std::size_t i = 0; i < n; ++i) {
            data_[i] = val;
        }
        size_ = n;
    }

    // -----------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------

    constexpr Value& operator[](node_id n) {
        if (static_cast<std::size_t>(n.value) >= size_)
            throw std::out_of_range("property_map: node_id out of bounds");
        return data_[n.value];
    }

    constexpr Value const& operator[](node_id n) const {
        if (static_cast<std::size_t>(n.value) >= size_)
            throw std::out_of_range("property_map: node_id out of bounds");
        return data_[n.value];
    }

    constexpr Value& operator[](std::size_t i) {
        if (i >= size_)
            throw std::out_of_range("property_map: index out of bounds");
        return data_[i];
    }

    constexpr Value const& operator[](std::size_t i) const {
        if (i >= size_)
            throw std::out_of_range("property_map: index out of bounds");
        return data_[i];
    }

    // -----------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr bool
    operator==(property_map const& other) const noexcept {
        if (size_ != other.size_) return false;
        for (std::size_t i = 0; i < size_; ++i) {
            if (!(data_[i] == other.data_[i])) return false;
        }
        return true;
    }
};

// BGL-style free functions

template<typename V, std::size_t M>
[[nodiscard]] constexpr V const&
get(property_map<V, M> const& pmap, node_id n) {
    return pmap[n];
}

template<typename V, std::size_t M>
constexpr void
put(property_map<V, M>& pmap, node_id n, V const& val) {
    pmap[n] = val;
}

// Factories

template<typename Value, std::size_t MaxV,
         graph_queryable G, typename Fn>
[[nodiscard]] constexpr property_map<Value, MaxV>
make_property_map(G const& g, Fn&& fn) {
    require_capacity(g.node_count(), MaxV,
        "make_property_map: graph node_count exceeds MaxV");
    property_map<Value, MaxV> pmap;
    pmap.resize(g.node_count());
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        auto const nid = node_id{static_cast<std::uint16_t>(i)};
        pmap[i] = fn(nid);
    }
    return pmap;
}

template<typename Value, std::size_t MaxV, graph_queryable G>
[[nodiscard]] constexpr property_map<Value, MaxV>
make_uniform_property_map(G const& g, Value const& val) {
    return property_map<Value, MaxV>(g.node_count(), val);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_PROPERTY_MAP_H
