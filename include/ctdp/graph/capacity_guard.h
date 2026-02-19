// graph/capacity_guard.h — Uniform capacity checks for graph algorithms
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// Every graph algorithm allocates fixed-capacity arrays sized by MaxV or
// MaxE.  If the actual graph exceeds these capacities the result is
// silent out-of-bounds writes — undefined behaviour at constexpr time
// manifests as an opaque "not a constant expression" error instead of
// a clear diagnostic.
//
// This header provides a single enforcement point:
//
//   require_capacity(actual, limit, "topological_sort: V exceeds MaxV");
//
// At constexpr time: throw makes the call non-constant, so the compiler
// emits a diagnostic that includes the message string.
// At runtime: throws std::length_error with the message.  (Callers that need
// runtime traps can also assert() independently.)
//
// Additionally codifies the uint16_t design limit:
//
//   require_node_id_range<MaxV>()   — static_assert(MaxV <= 65535)
//
// Use at the top of every graph algorithm template.

#ifndef CTDP_GRAPH_CAPACITY_GUARD_H
#define CTDP_GRAPH_CAPACITY_GUARD_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ctdp::graph {

// =========================================================================
// Runtime/constexpr capacity check
// =========================================================================

/// Check that an actual count does not exceed the fixed capacity.
///
/// At constexpr time: the throw makes the call non-constant, producing
/// a compiler diagnostic that includes `msg`.
/// At runtime: throws std::length_error(msg).
///
/// Usage:
///   require_capacity(g.node_count(), MaxV, "topological_sort: V > MaxV");
///   require_capacity(g.edge_count(), MaxE, "topological_sort: E > MaxE");
constexpr void require_capacity(std::size_t actual,
                                std::size_t limit,
                                char const* msg)
{
    if (actual > limit) {
        throw std::length_error(msg);  // constexpr: makes call non-constant
    }
}

// =========================================================================
// Static node_id range codification
// =========================================================================

/// Codify the uint16_t design limit: MaxV must fit in a uint16_t.
/// Call as: require_node_id_range<MaxV>();
/// The static_assert fires at template instantiation time.
template<std::size_t MaxV>
constexpr void require_node_id_range() {
    static_assert(MaxV <= std::size_t{65535},
        "MaxV exceeds uint16_t range (65535). "
        "node_id uses uint16_t — graphs larger than 65535 nodes "
        "are not supported.");
}

// =========================================================================
// Convenience: combined V + E check for algorithm entry
// =========================================================================

/// Standard preamble for graph algorithms that allocate arrays of size
/// MaxV.  Checks both the uint16_t design limit and runtime V <= MaxV.
///
/// Usage at the top of every algorithm:
///   guard_algorithm<MaxV>(g.node_count(), "topological_sort");
template<std::size_t MaxV>
constexpr void guard_algorithm(std::size_t actual_V,
                               char const* algo_name)
{
    require_node_id_range<MaxV>();
    require_capacity(actual_V, MaxV, algo_name);
}

} // namespace ctdp::graph

#endif // CTDP_GRAPH_CAPACITY_GUARD_H
