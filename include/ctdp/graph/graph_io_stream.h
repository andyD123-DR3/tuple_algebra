// graph/graph_io_stream.h — Runtime stream-based graph I/O
// Part of the compile-time DP library (C++20)
//
// Stream-based reading and writing of graphs in the text format.
// Uses <istream> and <ostream> — NOT <iostream> — to avoid pulling
// in static initialisation of std::cin/cout/cerr/clog.
//
// For constexpr parsing from string literals, use graph_io_parse.h
// instead (no stream dependency at all).
//
// ROUND-TRIP INVARIANT:
//   write(stream, g)      → read_directed(stream)            = g
//   write(stream, g, wfn) → read_weighted_directed(stream)   = {g, weights}
//   (and similarly for symmetric variants)
//
// DEFAULT CAPACITY:
//   Runtime readers default to cap::large (256/1024) rather than
//   cap::medium, because runtime readers often handle graphs of unknown
//   size and should avoid surprising capacity failures.
//
// TRAILING WEIGHT TOKENS:
//   Unweighted readers (read_directed, read_symmetric) silently skip
//   any trailing tokens after SRC DST on edge lines.  This allows the
//   same file to be read by both weighted and unweighted readers.

#ifndef CTDP_GRAPH_IO_STREAM_H
#define CTDP_GRAPH_IO_STREAM_H

#include "graph_io_detail.h"
#include "constexpr_graph.h"
#include "edge_property_map.h"
#include "graph_builder.h"
#include "graph_concepts.h"
#include "graph_traits.h"
#include "symmetric_graph.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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
template<graph_queryable G>
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
template<graph_queryable G, typename WeightFn>
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
///
/// Default capacity is cap::large (256 nodes, 1024 edges) — generous
/// to avoid surprising failures when reading graphs of unknown size.
/// Pass an explicit policy for tighter control: read_directed<cap::small>(is).
template<capacity_policy Cap = cap::large>
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
///
/// Default capacity is cap::large — see read_directed for rationale.
template<capacity_policy Cap = cap::large>
[[nodiscard]] auto read_symmetric(std::istream& is)
    -> symmetric_graph<Cap>
{
    symmetric_graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::size_t edge_count_undirected = 0;
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

            // Each undirected edge produces two directed edges in storage.
            // Check BEFORE add_edge to give a clear diagnostic rather than
            // a cryptic builder overflow (std::length_error).
            if (2 * (edge_count_undirected + 1) > Cap::max_e) {
                throw std::runtime_error(
                    "read_symmetric: doubled edge count exceeds capacity "
                    "(each undirected edge produces two directed edges)");
            }

            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});
            ++edge_count_undirected;
            continue;
        }
    }

    if (!have_nodes) {
        throw std::runtime_error("read_symmetric: missing 'nodes' line");
    }

    return b.finalise();
}

} // namespace ctdp::graph::io

// =============================================================================
// Runtime I/O: stream-based reading (weighted)
// =============================================================================

namespace ctdp::graph::io {

/// Read a weighted directed graph from a stream.
///
/// Returns {graph, edge_property_map<double>}.  Edge lines may optionally
/// include a weight after DST; edges without a weight default to 0.0.
///
/// The weight map is indexed by the graph's canonical CSR edge order.
/// Because the builder sorts edges, input edge order may differ from
/// storage order — weight matching uses (src, dst) lookup.
template<capacity_policy Cap = cap::large>
[[nodiscard]] auto read_weighted_directed(std::istream& is)
    -> std::pair<constexpr_graph<Cap>,
                 edge_property_map<double, Cap::max_e>>
{
    graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::string line;

    struct edge_record {
        std::uint16_t src{};
        std::uint16_t dst{};
        double weight{};
    };
    std::array<edge_record, Cap::max_e> edges{};
    std::size_t edge_count = 0;

    while (std::getline(is, line)) {
        std::string_view sv(line);
        std::size_t pos = detail::skip_hspace(sv, 0);

        if (detail::at_eol(sv, pos)) continue;

        if (detail::starts_with(sv, pos, "nodes")) {
            if (have_nodes) {
                throw std::runtime_error(
                    "read_weighted_directed: duplicate 'nodes' line");
            }
            pos += 5;
            pos = detail::skip_hspace(sv, pos);
            auto [n, next] = detail::parse_uint(sv, pos);
            if (n > Cap::max_v) {
                throw std::runtime_error(
                    "read_weighted_directed: node count exceeds capacity");
            }
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            continue;
        }

        if (detail::starts_with(sv, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error(
                    "read_weighted_directed: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(sv, pos);
            auto [src, p1] = detail::parse_uint(sv, pos);
            pos = detail::skip_hspace(sv, p1);
            auto [dst, p2] = detail::parse_uint(sv, pos);
            pos = p2;

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error(
                    "read_weighted_directed: node id out of range");
            }

            // Optional weight token after dst.
            double w = 0.0;
            auto after_dst = detail::skip_hspace(sv, pos);
            if (!detail::at_eol(sv, after_dst)) {
                auto [wval, p3] = detail::parse_double(sv, after_dst);
                w = wval;
            }

            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});

            if (edge_count >= Cap::max_e) {
                throw std::runtime_error(
                    "read_weighted_directed: edge count exceeds capacity");
            }
            edges[edge_count] = edge_record{
                static_cast<std::uint16_t>(src),
                static_cast<std::uint16_t>(dst), w};
            ++edge_count;
            continue;
        }
    }

    if (!have_nodes) {
        throw std::runtime_error(
            "read_weighted_directed: missing 'nodes' line");
    }

    auto g = b.finalise();

    // Build weight map.  The graph's edge order is canonicalised (sorted by
    // (src, dst)), so we match edges by (src, dst) pair.
    edge_property_map<double, Cap::max_e> weights(g.edge_count(), 0.0);

    std::size_t eidx = 0;
    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            for (std::size_t k = 0; k < edge_count; ++k) {
                if (edges[k].src == static_cast<std::uint16_t>(u) &&
                    edges[k].dst == static_cast<std::uint16_t>(to_index(v))) {
                    weights[eidx] = edges[k].weight;
                    break;
                }
            }
            ++eidx;
        }
    }

    return {g, weights};
}

/// Read a weighted symmetric graph from a stream.
///
/// Returns {symmetric_graph, edge_property_map<double>}.
/// Each `edge SRC DST WEIGHT` line produces two directed edges in storage
/// (both with the same weight).  The doubled edge count is validated
/// against capacity.
template<capacity_policy Cap = cap::large>
[[nodiscard]] auto read_weighted_symmetric(std::istream& is)
    -> std::pair<symmetric_graph<Cap>,
                 edge_property_map<double, Cap::max_e>>
{
    symmetric_graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::string line;

    struct edge_record {
        std::uint16_t src{};
        std::uint16_t dst{};
        double weight{};
    };
    std::array<edge_record, Cap::max_e> edges{};
    std::size_t edge_count = 0;  // undirected count (before doubling)

    while (std::getline(is, line)) {
        std::string_view sv(line);
        std::size_t pos = detail::skip_hspace(sv, 0);

        if (detail::at_eol(sv, pos)) continue;

        if (detail::starts_with(sv, pos, "nodes")) {
            if (have_nodes) {
                throw std::runtime_error(
                    "read_weighted_symmetric: duplicate 'nodes' line");
            }
            pos += 5;
            pos = detail::skip_hspace(sv, pos);
            auto [n, next] = detail::parse_uint(sv, pos);
            if (n > Cap::max_v) {
                throw std::runtime_error(
                    "read_weighted_symmetric: node count exceeds capacity");
            }
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            continue;
        }

        if (detail::starts_with(sv, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error(
                    "read_weighted_symmetric: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(sv, pos);
            auto [src, p1] = detail::parse_uint(sv, pos);
            pos = detail::skip_hspace(sv, p1);
            auto [dst, p2] = detail::parse_uint(sv, pos);
            pos = p2;

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error(
                    "read_weighted_symmetric: node id out of range");
            }

            // Optional weight token.
            double w = 0.0;
            auto after_dst = detail::skip_hspace(sv, pos);
            if (!detail::at_eol(sv, after_dst)) {
                auto [wval, p3] = detail::parse_double(sv, after_dst);
                w = wval;
            }

            // Doubling check.
            if (2 * (edge_count + 1) > Cap::max_e) {
                throw std::runtime_error(
                    "read_weighted_symmetric: doubled edge count exceeds capacity "
                    "(each undirected edge produces two directed edges)");
            }

            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});

            edges[edge_count] = edge_record{
                static_cast<std::uint16_t>(src),
                static_cast<std::uint16_t>(dst), w};
            ++edge_count;
            continue;
        }
    }

    if (!have_nodes) {
        throw std::runtime_error(
            "read_weighted_symmetric: missing 'nodes' line");
    }

    auto g = b.finalise();

    // Build weight map.  Symmetric graphs store both (u,v) and (v,u);
    // we assign the same weight to both directions.
    edge_property_map<double, Cap::max_e> weights(g.edge_count(), 0.0);

    std::size_t eidx = 0;
    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            // Match either direction — (src,dst) or (dst,src).
            for (std::size_t k = 0; k < edge_count; ++k) {
                if ((edges[k].src == static_cast<std::uint16_t>(u) &&
                     edges[k].dst == static_cast<std::uint16_t>(to_index(v))) ||
                    (edges[k].dst == static_cast<std::uint16_t>(u) &&
                     edges[k].src == static_cast<std::uint16_t>(to_index(v)))) {
                    weights[eidx] = edges[k].weight;
                    break;
                }
            }
            ++eidx;
        }
    }

    return {g, weights};
}

} // namespace ctdp::graph::io

// =============================================================================
// Tag-object convenience overloads
// =============================================================================
//
// These allow capacity to be specified via a tag object instead of an
// explicit template argument:
//
//   auto g = io::read_directed(stream, cap::small{});
//   auto [g, w] = io::read_weighted_directed(stream, cap::small{});
//
// The tag is not used at runtime — it exists solely for type deduction.

namespace ctdp::graph::io {

/// Read a directed graph with capacity deduced from a tag object.
template<capacity_policy Cap>
[[nodiscard]] auto read_directed(std::istream& is, Cap /*tag*/)
    -> constexpr_graph<Cap>
{
    return read_directed<Cap>(is);
}

/// Read a symmetric graph with capacity deduced from a tag object.
template<capacity_policy Cap>
[[nodiscard]] auto read_symmetric(std::istream& is, Cap /*tag*/)
    -> symmetric_graph<Cap>
{
    return read_symmetric<Cap>(is);
}

/// Read a weighted directed graph with capacity deduced from a tag object.
template<capacity_policy Cap>
[[nodiscard]] auto read_weighted_directed(std::istream& is, Cap /*tag*/)
    -> std::pair<constexpr_graph<Cap>,
                 edge_property_map<double, Cap::max_e>>
{
    return read_weighted_directed<Cap>(is);
}

/// Read a weighted symmetric graph with capacity deduced from a tag object.
template<capacity_policy Cap>
[[nodiscard]] auto read_weighted_symmetric(std::istream& is, Cap /*tag*/)
    -> std::pair<symmetric_graph<Cap>,
                 edge_property_map<double, Cap::max_e>>
{
    return read_weighted_symmetric<Cap>(is);
}

} // namespace ctdp::graph::io

#endif // CTDP_GRAPH_IO_STREAM_H
