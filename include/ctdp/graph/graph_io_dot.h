// graph/graph_io_dot.h — Graphviz DOT export
// Part of the compile-time DP library (C++20)
//
// Writes graphs in DOT format for visualisation with Graphviz.
// Uses `digraph` for directed graphs, `graph` for symmetric.
// No parsing, no graph_io_detail dependency, no <iostream>.

#ifndef CTDP_GRAPH_IO_DOT_H
#define CTDP_GRAPH_IO_DOT_H

#include "graph_concepts.h"
#include "graph_traits.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace ctdp::graph::io {

/// Write a graph in Graphviz DOT format.
///
/// Uses `digraph` for directed graphs, `graph` for symmetric.
///
/// Example output:
/// ```dot
/// digraph G {
///   0 -> 1;
///   0 -> 2;
/// }
/// ```
template<typename G>
void write_dot(std::ostream& os, G const& g,
               std::string_view graph_name = "G")
{
    constexpr bool sym = is_symmetric_graph_v<G>;
    constexpr const char* edge_op = sym ? " -- " : " -> ";
    os << (sym ? "graph " : "digraph ") << graph_name << " {\n";

    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        if (nbrs.begin() == nbrs.end()) {
            // Isolated node — emit standalone so it appears in the DOT output.
            os << "  " << u << ";\n";
            continue;
        }
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            if constexpr (sym) {
                if (u > to_index(v)) continue;
            }
            os << "  " << u << edge_op << to_index(v) << ";\n";
        }
    }

    os << "}\n";
}

/// Write a weighted graph in DOT format with edge labels.
template<typename G, typename WeightFn>
    requires std::invocable<WeightFn, G const&, std::size_t>
void write_dot(std::ostream& os, G const& g, WeightFn weight_fn,
               std::string_view graph_name = "G")
{
    constexpr bool sym = is_symmetric_graph_v<G>;
    constexpr const char* edge_op = sym ? " -- " : " -> ";
    os << (sym ? "graph " : "digraph ") << graph_name << " {\n";

    std::size_t eidx = 0;
    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        if (nbrs.begin() == nbrs.end()) {
            os << "  " << u << ";\n";
            continue;
        }
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            if constexpr (sym) {
                if (u > to_index(v)) {
                    ++eidx;
                    continue;
                }
            }
            os << "  " << u << edge_op << to_index(v)
               << " [label=\"" << weight_fn(g, eidx) << "\"];\n";
            ++eidx;
        }
    }

    os << "}\n";
}

} // namespace ctdp::graph::io

#endif // CTDP_GRAPH_IO_DOT_H
