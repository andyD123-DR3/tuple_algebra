// tests/graph/test_io_layering.cpp — Layering invariant tests
//
// Verifies the structural properties of the graph I/O header split:
//   1. graph_io_detail.h compiles standalone (no graph types, no iostream)
//   2. graph_io_parse.h works for constexpr parsing (no iostream needed)
//
// These are compile-time checks enforced via static_assert.
// The runtime TEST bodies exist so gtest discovers and reports them.

#include "ctdp/graph/graph_io_detail.h"  // must compile before any graph header
#include "ctdp/graph/graph_io_parse.h"

#include <gtest/gtest.h>

// =============================================================================
// 1. graph_io_detail.h: standalone constexpr verification
// =============================================================================

// These static_asserts prove graph_io_detail.h functions work without
// any graph type being visible.
static_assert(ctdp::graph::io::detail::at_eol("\n", 0));
static_assert(!ctdp::graph::io::detail::at_eol("x", 0));
static_assert(ctdp::graph::io::detail::skip_hspace("  x", 0) == 2);
static_assert(ctdp::graph::io::detail::skip_to_eol("ab\ncd", 0) == 3);
static_assert(ctdp::graph::io::detail::starts_with("nodes 5", 0, "nodes"));
static_assert(!ctdp::graph::io::detail::starts_with("edge", 0, "nodes"));

TEST(IOLayering, DetailStandalone) {
    // Runtime echo of the static_asserts above — ensures gtest reports this.
    EXPECT_TRUE(ctdp::graph::io::detail::at_eol("\n", 0));
    EXPECT_EQ(ctdp::graph::io::detail::skip_hspace("\t x", 0), 2u);

    auto [val, pos] = ctdp::graph::io::detail::parse_uint("42abc", 0);
    EXPECT_EQ(val, 42u);
    EXPECT_EQ(pos, 2u);

    auto [dval, dpos] = ctdp::graph::io::detail::parse_double("-3.14x", 0);
    EXPECT_NEAR(dval, -3.14, 1e-9);
    EXPECT_EQ(dpos, 5u);
}

// =============================================================================
// 2. graph_io_parse.h: constexpr parsing without iostream
// =============================================================================

// This constexpr graph is constructed at compile time using only
// graph_io_parse.h.  If this header transitively included <iostream>,
// it would still compile — but the layering intent would be violated.
// The real firewall is that this TU does NOT include graph_io_stream.h
// or graph_io_dot.h, proving parse-only consumers need no stream headers.
constexpr auto k_layering_graph =
    ctdp::graph::io::parse_directed<ctdp::graph::cap::tiny>(
        "nodes 3\nedge 0 1\nedge 1 2\n");
static_assert(k_layering_graph.node_count() == 3);
static_assert(k_layering_graph.edge_count() == 2);

constexpr auto k_layering_sym =
    ctdp::graph::io::parse_symmetric<ctdp::graph::cap::tiny>(
        "nodes 2\nedge 0 1\n");
static_assert(k_layering_sym.node_count() == 2);
static_assert(k_layering_sym.undirected_edge_count() == 1);

TEST(IOLayering, ParseWithoutIOStream) {
    EXPECT_EQ(k_layering_graph.node_count(), 3u);
    EXPECT_EQ(k_layering_graph.edge_count(), 2u);
    EXPECT_EQ(k_layering_sym.node_count(), 2u);
    EXPECT_EQ(k_layering_sym.undirected_edge_count(), 1u);
}
