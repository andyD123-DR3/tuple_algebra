// graph/graph_io_parse.h — Constexpr graph parsers
// Part of the compile-time DP library (C++20)
//
// Constexpr parsing of directed, symmetric, and weighted directed graphs
// from string_view text.  No iostream dependency — constexpr-only
// consumers pay for nothing they don't use.
//
// TEXT FORMAT:
//   # comment
//   nodes N              (must appear before any edge lines)
//   edge SRC DST         (directed edge)
//   edge SRC DST WEIGHT  (weighted edge — only in parse_weighted variants)
//   symmetric            (flag: parse as undirected — only in parse_symmetric)
//
// Lines starting with # are comments.  Blank lines are ignored.
// Node IDs are zero-based unsigned integers.  Weights are decimal numbers
// (integer or integer.fraction, optional leading minus).

#ifndef CTDP_GRAPH_IO_PARSE_H
#define CTDP_GRAPH_IO_PARSE_H

#include "graph_io_detail.h"
#include "constexpr_graph.h"
#include "edge_property_map.h"
#include "graph_builder.h"
#include "graph_concepts.h"
#include "symmetric_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ctdp::graph::io {

// =============================================================================
// Constexpr parsing: directed graph
// =============================================================================

/// Parse a directed graph from a text description (constexpr-safe).
///
/// The text must contain a `nodes N` line before any `edge` lines.
/// Lines starting with `#` are comments.  Blank lines are ignored.
///
/// Example:
/// ```cpp
/// constexpr auto g = io::parse_directed<cap::small>(
///     "nodes 3\n"
///     "edge 0 1\n"
///     "edge 1 2\n"
/// );
/// static_assert(g.node_count() == 3);
/// static_assert(g.edge_count() == 2);
/// ```
template<capacity_policy Cap = cap::medium>
[[nodiscard]] constexpr auto parse_directed(std::string_view text)
    -> constexpr_graph<Cap>
{
    graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::size_t pos = 0;

    while (pos < text.size()) {
        pos = detail::skip_hspace(text, pos);

        // Skip \r (Windows line endings).
        if (pos < text.size() && text[pos] == '\r') {
            ++pos;
            continue;
        }

        // Blank line or comment.
        if (detail::at_eol(text, pos)) {
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        // "nodes N"
        if (detail::starts_with(text, pos, "nodes")) {
            if (have_nodes) {
                throw std::runtime_error("parse_directed: duplicate 'nodes' line");
            }
            pos += 5;
            pos = detail::skip_hspace(text, pos);
            auto [n, next] = detail::parse_uint(text, pos);
            pos = next;
            if (n > Cap::max_v) {
                throw std::runtime_error("parse_directed: node count exceeds capacity");
            }
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        // "edge SRC DST [WEIGHT]"
        if (detail::starts_with(text, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error("parse_directed: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(text, pos);
            auto [src, p1] = detail::parse_uint(text, pos);
            pos = detail::skip_hspace(text, p1);
            auto [dst, p2] = detail::parse_uint(text, pos);
            pos = p2;

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error("parse_directed: node id out of range");
            }
            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});

            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        // "symmetric" flag — ignore in directed parse.
        if (detail::starts_with(text, pos, "symmetric")) {
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        // Unknown line — error.
        throw std::runtime_error("parse_directed: unrecognised line");
    }

    if (!have_nodes) {
        throw std::runtime_error("parse_directed: missing 'nodes' line");
    }

    return b.finalise();
}

// =============================================================================
// Constexpr parsing: symmetric (undirected) graph
// =============================================================================

/// Parse a symmetric (undirected) graph from text.
///
/// Same format as parse_directed.  Each `edge SRC DST` adds edges in
/// both directions.
template<capacity_policy Cap = cap::medium>
[[nodiscard]] constexpr auto parse_symmetric(std::string_view text)
    -> symmetric_graph<Cap>
{
    symmetric_graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::size_t pos = 0;

    while (pos < text.size()) {
        pos = detail::skip_hspace(text, pos);

        if (pos < text.size() && text[pos] == '\r') {
            ++pos;
            continue;
        }

        if (detail::at_eol(text, pos)) {
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        if (detail::starts_with(text, pos, "nodes")) {
            if (have_nodes) {
                throw std::runtime_error("parse_symmetric: duplicate 'nodes' line");
            }
            pos += 5;
            pos = detail::skip_hspace(text, pos);
            auto [n, next] = detail::parse_uint(text, pos);
            pos = next;
            if (n > Cap::max_v) {
                throw std::runtime_error("parse_symmetric: node count exceeds capacity");
            }
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        if (detail::starts_with(text, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error("parse_symmetric: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(text, pos);
            auto [src, p1] = detail::parse_uint(text, pos);
            pos = detail::skip_hspace(text, p1);
            auto [dst, p2] = detail::parse_uint(text, pos);
            pos = p2;

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error("parse_symmetric: node id out of range");
            }
            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});

            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        // "symmetric" flag — acknowledged but no action needed.
        if (detail::starts_with(text, pos, "symmetric")) {
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        throw std::runtime_error("parse_symmetric: unrecognised line");
    }

    if (!have_nodes) {
        throw std::runtime_error("parse_symmetric: missing 'nodes' line");
    }

    return b.finalise();
}

// =============================================================================
// Constexpr parsing: weighted directed graph
// =============================================================================

/// Parse a weighted directed graph.
///
/// Each `edge SRC DST WEIGHT` line has an optional third numeric field.
/// Edges without a weight get 0.0.
///
/// Returns a pair of {graph, edge_property_map<double, MaxE>}.
template<capacity_policy Cap = cap::medium>
[[nodiscard]] constexpr auto parse_weighted_directed(std::string_view text)
    -> std::pair<constexpr_graph<Cap>,
                 edge_property_map<double, Cap::max_e>>
{
    // First pass: count edges so we can size the property map.
    // Actually, build the graph first, then assign weights in order.
    graph_builder<Cap> b;
    bool have_nodes = false;
    std::size_t node_count = 0;
    std::size_t pos = 0;

    // Temporary edge storage: src, dst, weight triples.
    // We need to record them because builder canonicalises edge order.
    struct edge_record {
        std::uint16_t src{};
        std::uint16_t dst{};
        double weight{};
    };
    // Use a fixed array — we know max edges.
    std::array<edge_record, Cap::max_e> edges{};
    std::size_t edge_count = 0;

    while (pos < text.size()) {
        pos = detail::skip_hspace(text, pos);

        if (pos < text.size() && text[pos] == '\r') {
            ++pos;
            continue;
        }

        if (detail::at_eol(text, pos)) {
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        if (detail::starts_with(text, pos, "nodes")) {
            if (have_nodes) {
                throw std::runtime_error("parse_weighted_directed: duplicate 'nodes' line");
            }
            pos += 5;
            pos = detail::skip_hspace(text, pos);
            auto [n, next] = detail::parse_uint(text, pos);
            pos = next;
            if (n > Cap::max_v) {
                throw std::runtime_error("parse_weighted_directed: node count exceeds capacity");
            }
            node_count = n;
            for (std::size_t i = 0; i < n; ++i) {
                (void)b.add_node();
            }
            have_nodes = true;
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        if (detail::starts_with(text, pos, "edge")) {
            if (!have_nodes) {
                throw std::runtime_error("parse_weighted_directed: 'edge' before 'nodes'");
            }
            pos += 4;
            pos = detail::skip_hspace(text, pos);
            auto [src, p1] = detail::parse_uint(text, pos);
            pos = detail::skip_hspace(text, p1);
            auto [dst, p2] = detail::parse_uint(text, pos);
            pos = p2;

            if (src >= node_count || dst >= node_count) {
                throw std::runtime_error("parse_weighted_directed: node id out of range");
            }

            double w = 0.0;
            auto after_dst = detail::skip_hspace(text, pos);
            if (!detail::at_eol(text, after_dst)) {
                auto [wval, p3] = detail::parse_double(text, after_dst);
                w = wval;
                pos = p3;
            }

            b.add_edge(node_id{static_cast<std::uint16_t>(src)},
                        node_id{static_cast<std::uint16_t>(dst)});

            edges[edge_count] = edge_record{static_cast<std::uint16_t>(src),
                                             static_cast<std::uint16_t>(dst), w};
            ++edge_count;

            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        if (detail::starts_with(text, pos, "symmetric")) {
            pos = detail::skip_to_eol(text, pos);
            continue;
        }

        throw std::runtime_error("parse_weighted_directed: unrecognised line");
    }

    if (!have_nodes) {
        throw std::runtime_error("parse_weighted_directed: missing 'nodes' line");
    }

    auto g = b.finalise();

    // Build weight map.  The graph's edge order is canonicalised (sorted by
    // (src, dst)), so we need to match edges by (src, dst) pair.
    edge_property_map<double, Cap::max_e> weights(g.edge_count(), 0.0);

    // For each edge in the graph's CSR order, find its weight.
    // The graph iterates edges in canonical order: node 0's neighbors sorted,
    // then node 1's, etc.
    std::size_t eidx = 0;
    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto nbrs = g.out_neighbors(node_id{static_cast<std::uint16_t>(u)});
        for (auto it = nbrs.begin(); it != nbrs.end(); ++it) {
            auto v = *it;
            // Find matching edge record.
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

} // namespace ctdp::graph::io

#endif // CTDP_GRAPH_IO_PARSE_H
