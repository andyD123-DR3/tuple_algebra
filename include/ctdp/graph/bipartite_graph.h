// graph/representation/bipartite_graph.h - Bipartite graph type and builder
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Bipartite matching (Hopcroft-Karp) requires structural knowledge of the
// left/right vertex partition.  Rather than accepting an arbitrary graph
// + partition predicate (which risks misuse), we provide a dedicated type
// that enforces bipartiteness by construction — mirroring how
// symmetric_graph enforces undirectedness by construction.
//
// INTERNAL REPRESENTATION:
// bipartite_graph wraps constexpr_graph<cap_from<MaxL + MaxR, MaxE>>.
// Nodes [0, L) are the LEFT partition.
// Nodes [L, L+R) are the RIGHT partition.
// All edges go from left to right (directed).  The builder rejects
// edges within a partition or from right to left.
//
// TERMINOLOGY:
// "Left index" = [0, L)  — used in algorithm APIs and results.
// "Right index" = [0, R) — used in algorithm APIs and results.
// "Internal node_id" = [0, L+R) — used inside the CSR.
// Translation: right_index i → internal node_id (L + i).
//
// WHEN TO USE:
// - Task-to-machine assignment (tasks = left, machines = right)
// - Variable-to-constraint matching in CSP
// - Register allocation (variables = left, registers = right)
// - Any maximum cardinality matching problem on a bipartite structure


#ifndef CTDP_GRAPH_BIPARTITE_GRAPH_H
#define CTDP_GRAPH_BIPARTITE_GRAPH_H

#include "capacity_guard.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_concepts.h"

#include <cstddef>
#include <cstdint>

namespace ctdp::graph {

// =========================================================================
// bipartite_graph — immutable bipartite graph with enforced partition
// =========================================================================

/// Fixed-capacity, immutable bipartite graph.
///
/// Template parameters:
/// - MaxL: maximum number of LEFT vertices
/// - MaxR: maximum number of RIGHT vertices
/// - MaxE: maximum number of edges (left → right only)
///
/// Constructed exclusively via bipartite_graph_builder::finalise().
///
/// Example:
/// ```cpp
/// constexpr auto bg = []() {
///     bipartite_graph_builder<4, 4, 8> b;
///     b.set_partition(3, 3);      // 3 left, 3 right
///     b.add_edge(0, 0);           // left 0 → right 0
///     b.add_edge(0, 1);           // left 0 → right 1
///     b.add_edge(1, 2);           // left 1 → right 2
///     b.add_edge(2, 1);           // left 2 → right 1
///     return b.finalise();
/// }();
/// static_assert(bg.left_count() == 3);
/// static_assert(bg.right_count() == 3);
/// static_assert(bg.edge_count() == 4);
/// ```
template<std::size_t MaxL, std::size_t MaxR, std::size_t MaxE>
class bipartite_graph {
    static_assert(MaxL + MaxR <= 65535,
        "bipartite_graph: MaxL + MaxR exceeds uint16_t range");
    static_assert(MaxL > 0 && MaxR > 0,
        "bipartite_graph: both partitions must be non-empty");

public:
    using inner_graph_type = constexpr_graph<cap_from<MaxL + MaxR, MaxE>>;

    static constexpr std::size_t max_left = MaxL;
    static constexpr std::size_t max_right = MaxR;
    static constexpr std::size_t max_edges = MaxE;

    constexpr bipartite_graph() = default;

    // -----------------------------------------------------------------
    // Partition queries
    // -----------------------------------------------------------------

    /// Number of left vertices.
    [[nodiscard]] constexpr std::size_t left_count() const noexcept {
        return L_;
    }

    /// Number of right vertices.
    [[nodiscard]] constexpr std::size_t right_count() const noexcept {
        return R_;
    }

    /// Total number of vertices (left + right).
    [[nodiscard]] constexpr std::size_t node_count() const noexcept {
        return inner_.node_count();
    }

    /// Number of edges (all left → right).
    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return inner_.edge_count();
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return inner_.empty();
    }

    // -----------------------------------------------------------------
    // Left → Right adjacency
    // -----------------------------------------------------------------

    /// Neighbors of left node u (returned as RIGHT indices [0, R)).
    ///
    /// Internally, the CSR stores right nodes at [L, L+R).
    /// This method translates back to [0, R) for algorithm convenience.
    ///
    /// Precondition: left_idx < left_count()
    struct right_neighbor_range {
        node_id const* begin_;
        node_id const* end_;
        std::uint16_t left_count_;

        struct iterator {
            node_id const* ptr;
            std::uint16_t left_count;

            /// Yields right index [0, R) by subtracting left_count.
            [[nodiscard]] constexpr std::size_t operator*() const {
                return static_cast<std::size_t>(ptr->value) -
                       static_cast<std::size_t>(left_count);
            }
            constexpr iterator& operator++() { ++ptr; return *this; }
            constexpr bool operator!=(iterator const& o) const {
                return ptr != o.ptr;
            }
            constexpr bool operator==(iterator const& o) const {
                return ptr == o.ptr;
            }
        };

        [[nodiscard]] constexpr iterator begin() const noexcept {
            return {begin_, left_count_};
        }
        [[nodiscard]] constexpr iterator end() const noexcept {
            return {end_, left_count_};
        }
        [[nodiscard]] constexpr std::size_t size() const noexcept {
            return static_cast<std::size_t>(end_ - begin_);
        }
        [[nodiscard]] constexpr bool empty() const noexcept {
            return begin_ == end_;
        }
    };

    /// Neighbors of left node u as right indices [0, R).
    [[nodiscard]] constexpr right_neighbor_range
    left_neighbors(std::size_t left_idx) const noexcept {
        auto const u = node_id{static_cast<std::uint16_t>(left_idx)};
        auto nbr = inner_.out_neighbors(u);
        return {nbr.begin(), nbr.end(),
                static_cast<std::uint16_t>(L_)};
    }

    /// Out-degree of a left node.
    [[nodiscard]] constexpr std::size_t
    left_degree(std::size_t left_idx) const noexcept {
        return inner_.out_degree(
            node_id{static_cast<std::uint16_t>(left_idx)});
    }

    // -----------------------------------------------------------------
    // graph_queryable forwarding (for general graph algorithms)
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr auto
    out_neighbors(node_id u) const noexcept {
        return inner_.out_neighbors(u);
    }

    [[nodiscard]] constexpr std::size_t
    out_degree(node_id u) const noexcept {
        return inner_.out_degree(u);
    }

    [[nodiscard]] constexpr bool
    has_node(node_id u) const noexcept {
        return inner_.has_node(u);
    }

    /// Access the inner directed graph (for algorithms needing raw CSR).
    [[nodiscard]] constexpr inner_graph_type const&
    directed() const noexcept { return inner_; }

private:
    inner_graph_type inner_{};
    std::size_t L_ = 0;
    std::size_t R_ = 0;

    template<std::size_t, std::size_t, std::size_t>
    friend class bipartite_graph_builder;
};

// =========================================================================
// bipartite_graph_builder
// =========================================================================

/// Incremental builder for bipartite_graph.
///
/// Usage:
/// 1. set_partition(L, R) — declare partition sizes
/// 2. add_edge(left_idx, right_idx) — add edges across partition
/// 3. finalise() — produce immutable bipartite_graph
///
/// Edges within a partition or from right to left are rejected.
/// This enforces bipartiteness by construction.
template<std::size_t MaxL, std::size_t MaxR, std::size_t MaxE>
class bipartite_graph_builder {
    static_assert(MaxL + MaxR <= 65535,
        "bipartite_graph_builder: MaxL + MaxR exceeds uint16_t range");

public:
    /// Declare the partition sizes.
    ///
    /// Must be called exactly once before add_edge().
    /// Precondition: L <= MaxL, R <= MaxR.
    constexpr void set_partition(std::size_t L, std::size_t R) {
        require_capacity(L, MaxL,
            "bipartite_graph_builder: L exceeds MaxL");
        require_capacity(R, MaxR,
            "bipartite_graph_builder: R exceeds MaxR");
        if (partition_set_)
            throw std::logic_error(
                "bipartite_graph_builder: partition already set");
        L_ = L;
        R_ = R;
        partition_set_ = true;

        // Add all nodes to inner builder: [0, L) left, [L, L+R) right.
        (void)inner_.add_nodes(L + R);
    }

    /// Add an edge from left_idx to right_idx.
    ///
    /// Precondition: set_partition() has been called.
    /// Precondition: left_idx < L, right_idx < R.
    constexpr void add_edge(std::size_t left_idx, std::size_t right_idx) {
        if (!partition_set_)
            throw std::logic_error(
                "bipartite_graph_builder: call set_partition() first");
        if (left_idx >= L_)
            throw std::logic_error(
                "bipartite_graph_builder: left_idx out of range");
        if (right_idx >= R_)
            throw std::logic_error(
                "bipartite_graph_builder: right_idx out of range");

        auto const u = node_id{static_cast<std::uint16_t>(left_idx)};
        auto const v = node_id{
            static_cast<std::uint16_t>(L_ + right_idx)};
        inner_.add_edge(u, v);
    }

    /// Number of edges added so far.
    [[nodiscard]] constexpr std::size_t edge_count() const noexcept {
        return inner_.edge_count();
    }

    /// Build the immutable bipartite_graph.
    [[nodiscard]] constexpr bipartite_graph<MaxL, MaxR, MaxE>
    finalise() const {
        if (!partition_set_)
            throw std::logic_error(
                "bipartite_graph_builder: call set_partition() before "
                "finalise()");
        bipartite_graph<MaxL, MaxR, MaxE> bg;
        bg.inner_ = inner_.finalise();
        bg.L_ = L_;
        bg.R_ = R_;
        return bg;
    }

private:
    graph_builder<cap_from<MaxL + MaxR, MaxE>> inner_{};
    std::size_t L_ = 0;
    std::size_t R_ = 0;
    bool partition_set_ = false;
};

// =========================================================================
// Concept: bipartite_graph_queryable
// =========================================================================

/// A bipartite graph provides partitioned adjacency queries.
///
/// Algorithms that require bipartite structure (Hopcroft-Karp, etc.)
/// constrain on this concept.  The left/right partition is enforced
/// by construction — not discovered or assumed.
///
/// Satisfied by bipartite_graph.  NOT satisfied by constexpr_graph
/// or symmetric_graph (which have no partition structure).
template<typename G>
concept bipartite_graph_queryable =
    requires(G const& g) {
        { g.left_count() } -> std::convertible_to<std::size_t>;
        { g.right_count() } -> std::convertible_to<std::size_t>;
        { g.left_neighbors(std::size_t{0}) };
        { g.left_degree(std::size_t{0}) } -> std::convertible_to<std::size_t>;
        { g.edge_count() } -> std::convertible_to<std::size_t>;
    };

// =========================================================================
// Concept verification
// =========================================================================

static_assert(graph_queryable<bipartite_graph<4, 4, 8>>);
static_assert(bipartite_graph_queryable<bipartite_graph<4, 4, 8>>);
static_assert(!bipartite_graph_queryable<constexpr_graph<cap_from<8, 16>>>);
// symmetric_graph negative check lives in test_bipartite_matching.cpp
// to avoid pulling symmetric_graph.h into this header.

} // namespace ctdp::graph

#endif // CTDP_GRAPH_BIPARTITE_GRAPH_H
