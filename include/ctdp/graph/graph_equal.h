// graph/representation/graph_equal.h - Structural equality for graphs
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Structural equality is essential for testing transforms, contraction,
// and fusion. Two graphs are equal if they have the same node count,
// same edge count, and identical canonicalised adjacency for each node.
//
// Since constexpr_graph stores CSR in canonicalised order (sorted adjacency),
// equality reduces to comparing the offsets and neighbor arrays.

#ifndef CTDP_GRAPH_EQUAL_H
#define CTDP_GRAPH_EQUAL_H

#include "constexpr_graph.h"

#include <cstddef>

namespace ctdp::graph {

/// Structural equality: same node count, same edge count,
/// same canonicalised adjacency for each node.
///
/// Because constexpr_graph stores edges in canonicalised CSR order,
/// this is a direct comparison of the internal arrays.
///
/// Example:
/// ```cpp
/// constexpr auto g1 = make_triangle();
/// constexpr auto g2 = make_triangle();
/// static_assert(graph_equal(g1, g2));
/// ```
template<std::size_t MaxV, std::size_t MaxE>
[[nodiscard]] constexpr bool
graph_equal(constexpr_graph<MaxV, MaxE> const& a,
            constexpr_graph<MaxV, MaxE> const& b) noexcept {
    // Different node or edge count → not equal.
    if (a.V_ != b.V_ || a.E_ != b.E_) {
        return false;
    }

    // Compare CSR offsets [0..V_].
    for (std::size_t i = 0; i <= a.V_; ++i) {
        if (a.offsets_[i] != b.offsets_[i]) {
            return false;
        }
    }

    // Compare neighbor arrays [0..E_).
    for (std::size_t i = 0; i < a.E_; ++i) {
        if (a.neighbors_[i] != b.neighbors_[i]) {
            return false;
        }
    }

    return true;
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_EQUAL_H
