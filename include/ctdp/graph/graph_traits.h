// graph/graph_traits.h — Type traits decoupling storage from algorithms
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// graph_traits<G> is the single customisation point that tells algorithms
// how to allocate working arrays for a given graph type.  This decouples
// storage decisions (std::array vs rt_array) from algorithm logic.
//
// Each specialisation provides:
//   - is_constexpr_storage: bool — whether arrays are fixed-size (constexpr-safe)
//   - node_index_type: the integer type used for node indices in results
//   - node_array<T>: alias for the appropriate array type (std::array or rt_array)
//   - edge_array<T>: alias for the appropriate array type
//   - make_node_array<T>(cap): construct a node-indexed array
//   - make_node_array<T>(cap, fill): construct with fill value
//   - make_edge_array<T>(cap): construct an edge-indexed array
//   - make_edge_array<T>(cap, fill): construct with fill value
//
// The primary template is intentionally undefined — a compilation error
// for unsupported graph types gives a clear diagnostic.
//
// SPECIALISATION GUIDE:
// To add traits for a new graph type:
//   1. Specialise graph_traits<YourType>
//   2. Provide all the members listed above
//   3. Ensure your graph type provides node_capacity() and edge_capacity()
//   4. Add a static_assert verifying concept satisfaction

#ifndef CTDP_GRAPH_TRAITS_H
#define CTDP_GRAPH_TRAITS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

// Forward declarations — avoid circular includes.
namespace ctdp::graph {
template<std::size_t MaxV, std::size_t MaxE> class constexpr_graph;
template<std::size_t MaxV, std::size_t MaxE> class symmetric_graph;
} // namespace ctdp::graph

namespace ctdp::graph {

// =============================================================================
// Primary template (intentionally incomplete — must be specialised)
// =============================================================================

/// Primary graph_traits template.
///
/// This is intentionally not defined.  Using an unsupported graph type
/// in a traits context produces a clear "incomplete type" error.
/// To support a new graph type, provide a full specialisation.
template<typename G>
struct graph_traits;  // primary — must be specialised

// =============================================================================
// node_index_t<G> — convenience alias
// =============================================================================

/// The node index type for graph G, drawn from its traits.
template<typename G>
using node_index_t = typename graph_traits<G>::node_index_type;

// =============================================================================
// node_nil_v<G> — sentinel value adapting to index width
// =============================================================================

/// Sentinel value for "no node" / "unvisited" / "no predecessor".
/// Equal to the maximum value of node_index_t<G>.
template<typename G>
inline constexpr node_index_t<G> node_nil_v =
    std::numeric_limits<node_index_t<G>>::max();

// =============================================================================
// Specialisation: constexpr_graph<MaxV, MaxE>
// =============================================================================

template<std::size_t MaxV, std::size_t MaxE>
struct graph_traits<constexpr_graph<MaxV, MaxE>> {
    static constexpr bool is_constexpr_storage = true;
    static constexpr std::size_t max_nodes = MaxV;
    static constexpr std::size_t max_edges = MaxE;

    using node_index_type = std::uint16_t;

    template<typename T>
    using node_array = std::array<T, MaxV>;

    template<typename T>
    using edge_array = std::array<T, MaxE>;

    // --- make_node_array ---

    template<typename T>
    static constexpr node_array<T> make_node_array(std::size_t /*cap*/) {
        return node_array<T>{};
    }

    template<typename T>
    static constexpr node_array<T> make_node_array(std::size_t /*cap*/, T const& fill) {
        node_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }

    // --- make_edge_array ---

    template<typename T>
    static constexpr edge_array<T> make_edge_array(std::size_t /*cap*/) {
        return edge_array<T>{};
    }

    template<typename T>
    static constexpr edge_array<T> make_edge_array(std::size_t /*cap*/, T const& fill) {
        edge_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }
};

// =============================================================================
// Specialisation: symmetric_graph<MaxV, MaxE>
// =============================================================================
//
// symmetric_graph stores 2*MaxE directed edges internally, but its
// logical edge capacity is MaxE undirected edges.  Algorithms operating
// on the symmetric graph should size edge arrays to 2*MaxE (the directed
// edge count), which matches the internal CSR capacity.

template<std::size_t MaxV, std::size_t MaxE>
struct graph_traits<symmetric_graph<MaxV, MaxE>> {
    static constexpr bool is_constexpr_storage = true;
    static constexpr std::size_t max_nodes = MaxV;
    static constexpr std::size_t max_edges = 2 * MaxE;

    using node_index_type = std::uint16_t;

    template<typename T>
    using node_array = std::array<T, MaxV>;

    // Edge arrays sized to directed edge capacity (2*MaxE).
    template<typename T>
    using edge_array = std::array<T, 2 * MaxE>;

    // --- make_node_array ---

    template<typename T>
    static constexpr node_array<T> make_node_array(std::size_t /*cap*/) {
        return node_array<T>{};
    }

    template<typename T>
    static constexpr node_array<T> make_node_array(std::size_t /*cap*/, T const& fill) {
        node_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }

    // --- make_edge_array ---

    template<typename T>
    static constexpr edge_array<T> make_edge_array(std::size_t /*cap*/) {
        return edge_array<T>{};
    }

    template<typename T>
    static constexpr edge_array<T> make_edge_array(std::size_t /*cap*/, T const& fill) {
        edge_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }
};

} // namespace ctdp::graph

#endif // CTDP_GRAPH_TRAITS_H
