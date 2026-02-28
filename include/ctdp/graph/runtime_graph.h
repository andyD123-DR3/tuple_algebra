// graph/representation/runtime_graph.h - Runtime-constructed CSR graph
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// constexpr_graph<Cap> requires all construction inside a consteval
// context.  runtime_graph<Cap> shares the same compile-time capacity
// bounds (so result types and working arrays use the same Cap::max_v)
// but is constructed at runtime from dynamic data sources: files,
// network, user input, runtime-computed topologies.
//
// Internal storage uses std::vector for CSR offsets and neighbors.
// The graph satisfies graph_queryable + sized_graph, so ALL existing
// algorithms work unchanged.
//
// Cap::max_v / Cap::max_e are compile-time upper bounds.  Actual
// node/edge counts are set at construction and can be smaller.
// This means the same algorithm instantiation handles both constexpr
// and runtime graphs of the same capacity class.
//
// CONSTRUCTION:
//   runtime_graph_builder<Cap> builds the graph incrementally,
//   then finalise() produces an immutable runtime_graph.
//   Canonicalisation rules match graph_builder: sort, dedup, no self-edges.

#ifndef CTDP_GRAPH_RUNTIME_GRAPH_H
#define CTDP_GRAPH_RUNTIME_GRAPH_H

#include "capacity_types.h"
#include "graph_concepts.h"
#include "graph_traits.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ctdp::graph {

// Forward declaration for friend access.
template<typename>
class runtime_graph_builder;

// =============================================================================
// runtime_graph<Cap>
// =============================================================================

/// Runtime-constructed, immutable, CSR-format directed graph.
///
/// Template parameter:
/// - Cap: a capacity_policy providing max_v and max_e as compile-time bounds.
///
/// These bounds allow algorithm result types (which use std::array<T, MaxV>)
/// to compile.  The actual graph can be smaller.
///
/// Constructed via runtime_graph_builder<Cap>::finalise().
///
/// Example:
/// ```cpp
/// runtime_graph_builder<cap::medium> b;
/// b.add_node(); b.add_node(); b.add_node();
/// b.add_edge(node_id{0}, node_id{1});
/// b.add_edge(node_id{1}, node_id{2});
/// auto g = b.finalise();
/// auto topo = topological_sort(g);
/// ```
template<typename Cap = cap::medium>
class runtime_graph {
    static constexpr std::size_t MaxV = Cap::max_v;
    static constexpr std::size_t MaxE = Cap::max_e;

    static_assert(MaxV <= 65535,
        "runtime_graph: Cap::max_v exceeds uint16_t range (65535)");
public:
    using size_type = std::uint16_t;

    static constexpr std::size_t max_vertices = MaxV;
    static constexpr std::size_t max_edges = MaxE;

    runtime_graph() = default;

    // =========================================================================
    // Size queries
    // =========================================================================

    [[nodiscard]] std::size_t node_count() const noexcept { return V_; }
    [[nodiscard]] std::size_t edge_count() const noexcept { return E_; }
    [[nodiscard]] std::size_t node_capacity() const noexcept { return MaxV; }
    [[nodiscard]] std::size_t edge_capacity() const noexcept { return MaxE; }
    [[nodiscard]] bool empty() const noexcept { return V_ == 0; }

    // =========================================================================
    // Adjacency access
    // =========================================================================

    struct adjacency_range {
        node_id const* begin_;
        node_id const* end_;

        [[nodiscard]] node_id const* begin() const noexcept { return begin_; }
        [[nodiscard]] node_id const* end() const noexcept { return end_; }
        [[nodiscard]] std::size_t size() const noexcept {
            return static_cast<std::size_t>(end_ - begin_);
        }
        [[nodiscard]] bool empty() const noexcept { return begin_ == end_; }
    };

    [[nodiscard]] adjacency_range
    out_neighbors(node_id u) const noexcept {
        auto const idx = static_cast<std::size_t>(u.value);
        auto const b = static_cast<std::size_t>(offsets_[idx]);
        auto const e = static_cast<std::size_t>(offsets_[idx + 1]);
        return {neighbors_.data() + b, neighbors_.data() + e};
    }

    [[nodiscard]] std::size_t
    out_degree(node_id u) const noexcept {
        auto const idx = static_cast<std::size_t>(u.value);
        return static_cast<std::size_t>(offsets_[idx + 1]) -
               static_cast<std::size_t>(offsets_[idx]);
    }

    [[nodiscard]] std::size_t max_out_degree() const noexcept {
        std::size_t max_deg = 0;
        for (std::size_t v = 0; v < V_; ++v) {
            auto const deg = static_cast<std::size_t>(offsets_[v + 1]) -
                             static_cast<std::size_t>(offsets_[v]);
            if (deg > max_deg) max_deg = deg;
        }
        return max_deg;
    }

    [[nodiscard]] bool has_node(node_id u) const noexcept {
        return static_cast<std::size_t>(u.value) < V_;
    }

private:
    std::size_t V_ = 0;
    std::size_t E_ = 0;
    std::vector<size_type> offsets_;
    std::vector<node_id> neighbors_;

    template<typename>
    friend class runtime_graph_builder;
};

// Verify concept satisfaction.
static_assert(graph_queryable<runtime_graph<cap_from<8, 16>>>);
static_assert(sized_graph<runtime_graph<cap_from<8, 16>>>);

// =============================================================================
// graph_traits specialisation for runtime_graph
// =============================================================================

template<typename Cap>
struct graph_traits<runtime_graph<Cap>> {
    static constexpr bool is_constexpr_storage = false;
    static constexpr std::size_t max_nodes = Cap::max_v;
    static constexpr std::size_t max_edges = Cap::max_e;

    using node_index_type = std::uint16_t;

    template<typename T>
    using node_array = std::array<T, Cap::max_v>;

    template<typename T>
    using edge_array = std::array<T, Cap::max_e>;

    template<typename T>
    static node_array<T> make_node_array(std::size_t /*cap*/) {
        return node_array<T>{};
    }

    template<typename T>
    static node_array<T> make_node_array(std::size_t /*cap*/, T const& fill) {
        node_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }

    template<typename T>
    static edge_array<T> make_edge_array(std::size_t /*cap*/) {
        return edge_array<T>{};
    }

    template<typename T>
    static edge_array<T> make_edge_array(std::size_t /*cap*/, T const& fill) {
        edge_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }
};

/// Const-qualified: strips const and delegates.
template<typename Cap>
struct graph_traits<runtime_graph<Cap> const>
    : graph_traits<runtime_graph<Cap>> {};

// =============================================================================
// runtime_graph_builder
// =============================================================================

/// Builder for runtime_graph.
///
/// Same canonicalisation rules as graph_builder:
/// 1. Edges sorted by (src, dst)
/// 2. Duplicates removed
/// 3. Self-edges removed
///
/// Uses std::vector internally — not constexpr.
template<typename Cap = cap::medium>
class runtime_graph_builder {
    static constexpr std::size_t MaxV = Cap::max_v;
    static constexpr std::size_t MaxE = Cap::max_e;
public:
    struct edge_pair {
        std::uint16_t src;
        std::uint16_t dst;

        bool operator==(edge_pair const&) const = default;
        bool operator<(edge_pair const& o) const {
            if (src != o.src) return src < o.src;
            return dst < o.dst;
        }
    };

    [[nodiscard]] node_id add_node() {
        if (V_ >= MaxV)
            throw std::length_error("runtime_graph_builder: node count exceeds MaxV");
        auto const id = static_cast<std::uint16_t>(V_);
        ++V_;
        return node_id{id};
    }

    [[nodiscard]] node_id add_nodes(std::size_t count) {
        if (V_ + count > MaxV)
            throw std::length_error("runtime_graph_builder: would exceed MaxV");
        auto const first = static_cast<std::uint16_t>(V_);
        V_ += count;
        return node_id{first};
    }

    void add_edge(node_id u, node_id v) {
        if (V_ == 0)
            throw std::logic_error("runtime_graph_builder: no nodes");
        if (static_cast<std::size_t>(u.value) >= V_)
            throw std::out_of_range("runtime_graph_builder: source not in graph");
        if (static_cast<std::size_t>(v.value) >= V_)
            throw std::out_of_range("runtime_graph_builder: target not in graph");
        edges_.push_back(edge_pair{u.value, v.value});
    }

    [[nodiscard]] std::size_t node_count() const noexcept { return V_; }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edges_.size(); }

    /// Build the immutable runtime_graph.
    [[nodiscard]] runtime_graph<Cap> finalise() const {
        runtime_graph<Cap> g;
        g.V_ = V_;

        if (edges_.empty()) {
            g.offsets_.assign(V_ + 1, 0);
            g.E_ = 0;
            return g;
        }

        // Sort edges.
        auto sorted = edges_;
        std::sort(sorted.begin(), sorted.end());

        // Filter: remove self-edges and duplicates.
        std::vector<edge_pair> clean;
        clean.reserve(sorted.size());
        for (auto const& e : sorted) {
            if (e.src == e.dst) continue;
            if (!clean.empty() && clean.back() == e) continue;
            clean.push_back(e);
        }

        if (clean.size() > MaxE)
            throw std::length_error("runtime_graph_builder: edges exceed MaxE after dedup");

        g.E_ = clean.size();

        // Build CSR.
        g.offsets_.assign(V_ + 1, 0);
        for (auto const& e : clean) {
            g.offsets_[e.src + 1]++;
        }
        for (std::size_t i = 1; i <= V_; ++i) {
            g.offsets_[i] = static_cast<typename runtime_graph<Cap>::size_type>(
                g.offsets_[i] + g.offsets_[i - 1]);
        }

        g.neighbors_.resize(clean.size());
        for (std::size_t i = 0; i < clean.size(); ++i) {
            g.neighbors_[i] = node_id{clean[i].dst};
        }

        return g;
    }

private:
    std::size_t V_ = 0;
    std::vector<edge_pair> edges_;
};

// =============================================================================
// symmetric_runtime_graph_builder — convenience for undirected runtime graphs
// =============================================================================

/// Builder for undirected runtime_graph.
/// add_edge(u, v) inserts both directions.  The resulting graph satisfies
/// graph_queryable and sized_graph (but NOT symmetric_graph_queryable,
/// since it returns a runtime_graph, not a symmetric_graph).
///
/// For algorithms requiring symmetric_graph_queryable, use
/// symmetric_graph_builder for constexpr graphs.  This builder is for
/// runtime usage where the symmetry guarantee is documentation-level.
template<typename Cap = cap::medium>
class symmetric_runtime_graph_builder {
public:
    [[nodiscard]] node_id add_node() { return inner_.add_node(); }
    [[nodiscard]] node_id add_nodes(std::size_t n) { return inner_.add_nodes(n); }

    void add_edge(node_id u, node_id v) {
        if (u == v) return;
        inner_.add_edge(u, v);
        inner_.add_edge(v, u);
    }

    [[nodiscard]] auto finalise() const { return inner_.finalise(); }

private:
    runtime_graph_builder<cap_from<Cap::max_v, 2 * Cap::max_e>> inner_;
};

} // namespace ctdp::graph

#endif // CTDP_GRAPH_RUNTIME_GRAPH_H
