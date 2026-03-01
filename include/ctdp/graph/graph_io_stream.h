// graph/graph_io_stream.h — Runtime stream-based graph I/O
// Part of the compile-time DP library (C++20)
//
// Stream-based reading and writing of graphs in the text format.
// Uses <istream> and <ostream> — NOT <iostream> — to avoid pulling
// in static initialisation of std::cin/cout/cerr/clog.
//
// For constexpr parsing from string literals, use graph_io_parse.h
// instead (no stream dependency at all).

#ifndef CTDP_GRAPH_IO_STREAM_H
#define CTDP_GRAPH_IO_STREAM_H

#include "graph_io_detail.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include "symmetric_graph.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ctdp::graph::io {

// =============================================================================
// Runtime I/O: stream-based writing
// =============================================================================

/// Write a graph in the text format (matches parse format for round-trip).
///
/// Output for directed graphs:
///   nodes N
///   edge 0 1
///   edge 0 2
///   ...
///
/// Output for symmetric graphs adds a `symmetric` flag line.
template<typename G>
void write(std::ostream& os, G const& g) {
    os << "nodes " << g.node_count() << '\n';

    if constexpr (is_symmetric_graph_v<G>) {
        os << "symmetric\n";
    }

    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            // For symmetric graphs, only emit each undirected edge once (u <= v).
            if constexpr (is_symmetric_graph_v<G>) {
                if (u > to_index(v)) continue;
            }
            os << "edge " << u << ' ' << to_index(v) << '\n';
        }
    }
}

/// Write a weighted graph.  WeightFn(graph, edge_index) -> numeric weight.
template<typename G, typename WeightFn>
    requires std::invocable<WeightFn, G const&, std::size_t>
void write(std::ostream& os, G const& g, WeightFn weight_fn) {
    os << "nodes " << g.node_count() << '\n';

    if constexpr (is_symmetric_graph_v<G>) {
        os << "symmetric\n";
    }

    std::size_t eidx = 0;
    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            if constexpr (is_symmetric_graph_v<G>) {
                if (u > to_index(v)) {
                    ++eidx;
                    continue;
                }
            }
            os << "edge " << u << ' ' << to_index(v)
               << ' ' << weight_fn(g, eidx) << '\n';
            ++eidx;
        }
    }
}

// =============================================================================
// Runtime I/O: stream-based reading
// =============================================================================

/// Read a directed graph from a stream.  Same format as parse_directed.
template<capacity_policy Cap = cap::medium>
[[nodiscard]] auto read_directed(std::istream& is)
    -> constexpr_graph<Cap>
{
    graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::string line;

    while (std::getline(is, line)) {
        std::string_view sv(line);
        std::size_t pos = detail::skip_hspace(sv, 0);

        if (detail::at_eol(sv, pos)) continue;

        if (detail::starts_with(sv, pos, "nodes")) {
            pos += 5;
            pos = detail::skip_hspace(sv, pos);
            auto [n, next] = detail::parse_uint(sv, pos);
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            continue;
        }

        if (detail::starts_with(sv, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error("read_directed: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(sv, pos);
            auto [src, p1] = detail::parse_uint(sv, pos);
            pos = detail::skip_hspace(sv, p1);
            auto [dst, p2] = detail::parse_uint(sv, pos);

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error("read_directed: node id out of range");
            }
            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});
            continue;
        }
    }

    if (!have_nodes) {
        throw std::runtime_error("read_directed: missing 'nodes' line");
    }

    return b.finalise();
}

/// Read a symmetric (undirected) graph from a stream.
template<capacity_policy Cap = cap::medium>
[[nodiscard]] auto read_symmetric(std::istream& is)
    -> symmetric_graph<Cap>
{
    symmetric_graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::string line;

    while (std::getline(is, line)) {
        std::string_view sv(line);
        std::size_t pos = detail::skip_hspace(sv, 0);

        if (detail::at_eol(sv, pos)) continue;

        if (detail::starts_with(sv, pos, "nodes")) {
            pos += 5;
            pos = detail::skip_hspace(sv, pos);
            auto [n, next] = detail::parse_uint(sv, pos);
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            continue;
        }

        if (detail::starts_with(sv, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error("read_symmetric: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(sv, pos);
            auto [src, p1] = detail::parse_uint(sv, pos);
            pos = detail::skip_hspace(sv, p1);
            auto [dst, p2] = detail::parse_uint(sv, pos);

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error("read_symmetric: node id out of range");
            }
            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});
            continue;
        }
    }

    if (!have_nodes) {
        throw std::runtime_error("read_symmetric: missing 'nodes' line");
    }

    return b.finalise();
}

} // namespace ctdp::graph::io

#endif // CTDP_GRAPH_IO_STREAM_H
