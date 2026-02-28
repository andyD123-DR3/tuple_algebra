// graph/capacity_types.h — Capacity policy types for graph templates
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE (§3.4–3.6):
// Graph templates take a SINGLE capacity struct instead of two numeric
// template parameters.  This struct travels as one token through template
// arguments, namespace aliases, and deduction guides:
//
//   constexpr_graph<cap::small>  g;              // named tier
//   constexpr_graph<>            g2;             // defaults to cap::medium
//   constexpr_graph<cap_from<8>> g3;             // inline, MaxE = 4*8 = 32
//
//   namespace my_project {
//       using Cap = cap::large;                  // scope override
//       constexpr_graph<Cap> g;
//   }
//
// Five named tiers cover the common cases.  cap_from<V, E> bridges raw
// numbers for backward compatibility and ad-hoc sizing.

#ifndef CTDP_GRAPH_CAPACITY_TYPES_H
#define CTDP_GRAPH_CAPACITY_TYPES_H

#include <concepts>
#include <cstddef>

namespace ctdp::graph {

// =============================================================================
// capacity_policy concept
// =============================================================================

/// A type satisfies capacity_policy if it provides positive max_v and max_e.
template<typename C>
concept capacity_policy = requires {
    { C::max_v } -> std::convertible_to<std::size_t>;
    { C::max_e } -> std::convertible_to<std::size_t>;
} && (C::max_v > 0) && (C::max_e > 0);

// =============================================================================
// Named capacity tiers
// =============================================================================

namespace cap {

/// 8 vertices, 24 edges — unit tests, tiny examples.
struct tiny {
    static constexpr std::size_t max_v = 8;
    static constexpr std::size_t max_e = 24;
};

/// 16 vertices, 64 edges — small demos, stencils.
struct small {
    static constexpr std::size_t max_v = 16;
    static constexpr std::size_t max_e = 64;
};

/// 64 vertices, 256 edges — moderate graphs, typical fusion.
struct medium {
    static constexpr std::size_t max_v = 64;
    static constexpr std::size_t max_e = 256;
};

/// 256 vertices, 1024 edges — large compile-time graphs.
struct large {
    static constexpr std::size_t max_v = 256;
    static constexpr std::size_t max_e = 1024;
};

/// 1024 vertices, 4096 edges — upper bound of constexpr feasibility.
struct xlarge {
    static constexpr std::size_t max_v = 1024;
    static constexpr std::size_t max_e = 4096;
};

} // namespace cap

// =============================================================================
// cap_from<V, E> — inline capacity from raw numbers
// =============================================================================
//
// Bridges existing code and enables ad-hoc sizing:
//   constexpr_graph<cap_from<8, 16>>   g;      // explicit V and E
//   constexpr_graph<cap_from<8>>        g;      // E defaults to 4*V = 32

template<std::size_t MaxV, std::size_t MaxE = 4 * MaxV>
struct cap_from {
    static constexpr std::size_t max_v = MaxV;
    static constexpr std::size_t max_e = MaxE;
};

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CAPACITY_TYPES_H
