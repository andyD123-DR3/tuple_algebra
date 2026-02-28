// graph/construction/graph_builder.h - Incremental graph construction
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// graph_builder accumulates edges incrementally, then finalise() produces
// an immutable constexpr_graph in CSR format. This separates the mutable
// construction phase from the immutable analysis phase.
//
// finalise() CANONICALISATION RULES:
// 1. Edges sorted by (u, v) using stable sort.
// 2. Duplicate (u, v) pairs removed (first occurrence kept).
// 3. Self-edges (u, u) removed.
// 4. CSR offsets built from sorted, deduplicated edge list.
//
// Depends on core/constexpr_sort (Phase 1a) for constexpr sorting.

#ifndef CTDP_GRAPH_BUILDER_H
#define CTDP_GRAPH_BUILDER_H

#include "capacity_guard.h"
#include "capacity_types.h"
#include "constexpr_graph.h"
#include <ctdp/core/constexpr_sort.h>
#include <ctdp/core/constexpr_vector.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ctdp::graph {

/// Incremental builder for constexpr_graph.
///
/// Usage:
/// ```cpp
/// constexpr auto g = []() {
///     graph_builder<cap_from<4, 8>> b;
///     auto a = b.add_node();
///     auto c = b.add_node();
///     b.add_edge(a, c);
///     return b.finalise();
/// }();
/// ```
///
/// Template parameters:
/// - Cap::max_v: Maximum vertex capacity
/// - Cap::max_e: Maximum edge capacity (before dedup; final graph may have fewer)
template<typename Cap = cap::medium>
class graph_builder {
    static_assert(Cap::max_v <= 65535,
        "graph_builder: Cap::max_v exceeds uint16_t range (65535)");
    static_assert(Cap::max_v > 0, "graph_builder: Cap::max_v must be positive");
public:
    /// Edge as a pair of raw uint16_t values (source, target).
    struct edge_pair {
        std::uint16_t src;
        std::uint16_t dst;

        constexpr bool operator==(edge_pair const&) const = default;

        /// Lexicographic comparison for sorting.
        constexpr bool operator<(edge_pair const& other) const {
            if (src != other.src) return src < other.src;
            return dst < other.dst;
        }
    };

    // =========================================================================
    // Construction API
    // =========================================================================

    /// Add a new node. Returns its node_id.
    /// Nodes are numbered sequentially from 0.
    ///
    /// Precondition: V_ < Cap::max_v (capacity not exhausted).
    /// Precondition: V_ < 65536 (uint16_t range for node_id).
    [[nodiscard]] constexpr node_id add_node() {
        require_capacity(V_, Cap::max_v - 1,
            "graph_builder::add_node: node count would exceed Cap::max_v");
        require_capacity(V_, std::size_t{65535},
            "graph_builder::add_node: node count would exceed uint16_t range");
        auto const id = static_cast<std::uint16_t>(V_);
        ++V_;
        return node_id{id};
    }

    /// Add a directed edge from u to v.
    /// Self-edges and duplicates are accepted here; finalise() removes them.
    /// Precondition: u and v must refer to nodes previously added via add_node().
    constexpr void add_edge(node_id u, node_id v) {
        if (V_ == 0)
            throw std::logic_error("graph_builder::add_edge: no nodes in graph");
        require_capacity(static_cast<std::size_t>(u.value), V_ - 1,
            "graph_builder::add_edge: source node_id not in graph");
        require_capacity(static_cast<std::size_t>(v.value), V_ - 1,
            "graph_builder::add_edge: target node_id not in graph");
        edges_.push_back(edge_pair{u.value, v.value});
    }

    /// Add multiple nodes at once. Returns the first node_id.
    ///
    /// Precondition: V_ + count <= Cap::max_v.
    /// Precondition: V_ + count <= 65536.
    [[nodiscard]] constexpr node_id add_nodes(std::size_t count) {
        require_capacity(V_ + count - 1, Cap::max_v - 1,
            "graph_builder::add_nodes: would exceed Cap::max_v");
        require_capacity(V_ + count, std::size_t{65536},
            "graph_builder::add_nodes: would exceed uint16_t range");
        auto const first = static_cast<std::uint16_t>(V_);
        V_ += count;
        return node_id{first};
    }

    // =========================================================================
    // Query
    // =========================================================================

    /// Number of nodes added so far.
    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return V_;
    }

    /// Number of edges added so far (including potential duplicates/self-edges).
    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return edges_.size();
    }

    // =========================================================================
    // Finalisation
    // =========================================================================

    /// Build an immutable constexpr_graph from accumulated edges.
    ///
    /// Canonicalisation:
    /// 1. Stable sort edges by (src, dst)
    /// 2. Remove self-edges (src == dst)
    /// 3. Remove duplicate (src, dst) pairs
    /// 4. Build CSR offsets and neighbor array
    ///
    /// Returns: constexpr_graph<Cap> with actual V_ and E_ set.
    [[nodiscard]] constexpr constexpr_graph<Cap> finalise() const {
        constexpr_graph<Cap> g;
        g.V_ = V_;

        if (edges_.empty()) {
            // No edges: all offsets are 0.
            for (std::size_t i = 0; i <= V_; ++i) {
                g.offsets_[i] = 0;
            }
            g.E_ = 0;
            return g;
        }

        // Step 1: Copy and sort edges by (src, dst).
        auto sorted = edges_;
        ctdp::constexpr_stable_sort_vec(sorted,
            [](edge_pair const& a, edge_pair const& b) {
                return a < b;
            });

        // Step 2+3: Filter out self-edges and duplicates into a clean list.
        ctdp::constexpr_vector<edge_pair, Cap::max_e> clean;
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            auto const& e = sorted[i];

            // Skip self-edges.
            if (e.src == e.dst) {
                continue;
            }

            // Skip duplicates (sorted, so check previous in clean list).
            if (!clean.empty() && clean.back() == e) {
                continue;
            }

            clean.push_back(e);
        }

        // Step 4: Build CSR offsets and neighbors.
        g.E_ = clean.size();

        // Initialise all offsets to 0.
        for (std::size_t i = 0; i <= V_; ++i) {
            g.offsets_[i] = 0;
        }

        // Count edges per source node.
        for (std::size_t i = 0; i < clean.size(); ++i) {
            auto const src_idx = static_cast<std::size_t>(clean[i].src);
            // Increment the bucket after src (will prefix-sum later).
            g.offsets_[src_idx + 1]++;
        }

        // Prefix sum to get offsets.
        for (std::size_t i = 1; i <= V_; ++i) {
            g.offsets_[i] = static_cast<typename constexpr_graph<Cap>::size_type>(
                static_cast<std::size_t>(g.offsets_[i]) +
                static_cast<std::size_t>(g.offsets_[i - 1]));
        }

        // Fill neighbor array (edges are already sorted, so just copy in order).
        for (std::size_t i = 0; i < clean.size(); ++i) {
            g.neighbors_[i] = node_id{clean[i].dst};
        }

        return g;
    }

private:
    std::size_t V_ = 0;
    ctdp::constexpr_vector<edge_pair, Cap::max_e> edges_{};
};

} // namespace ctdp::graph

#endif // CTDP_GRAPH_BUILDER_H
