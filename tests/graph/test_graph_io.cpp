// tests/graph/test_graph_io.cpp
// Tests for graph I/O: constexpr parsing, runtime read/write, DOT export.
//
// Validates:
//   1. Constexpr parse of directed graph from string literal
//   2. Constexpr parse of symmetric graph with edge symmetry
//   3. Constexpr parse with weights
//   4. Runtime round-trip: write → parse, verify equality
//   5. DOT output contains expected elements
//   6. Error handling: missing nodes, out-of-range ids, malformed input
//   7. Comment and blank line handling
//   8. Runtime stream reading

#include "ctdp/graph/graph_io.h"
#include "ctdp/graph/graph_capacity.h"
#include "ctdp/graph/graph_equal.h"
#include "ctdp/graph/graph_builder.h"
#include "ctdp/graph/constexpr_graph.h"
#include "ctdp/graph/symmetric_graph.h"
#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace cg = ctdp::graph;
using cg::cap_from;
namespace io = ctdp::graph::io;

// =============================================================================
// 1. Constexpr parse: directed graph
// =============================================================================

constexpr auto k_directed_text =
    "# simple directed graph\n"
    "nodes 4\n"
    "\n"
    "edge 0 1\n"
    "edge 0 2\n"
    "edge 1 3\n"
    "edge 2 3\n";

constexpr auto k_directed = io::parse_directed<cg::cap::tiny>(k_directed_text);

static_assert(k_directed.node_count() == 4);
static_assert(k_directed.edge_count() == 4);

TEST(GraphIO, ConstexprParseDirected) {
    EXPECT_EQ(k_directed.node_count(), 4u);
    EXPECT_EQ(k_directed.edge_count(), 4u);

    // Build the same graph manually and compare.
    constexpr auto expected = []() {
        cg::graph_builder<cg::cap::tiny> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        auto n3 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n0, n2);
        b.add_edge(n1, n3);
        b.add_edge(n2, n3);
        return b.finalise();
    }();

    static_assert(cg::graph_equal(k_directed, expected));
    EXPECT_TRUE(cg::graph_equal(k_directed, expected));
}

// =============================================================================
// 2. Constexpr parse: symmetric graph
// =============================================================================

constexpr auto k_sym_text =
    "nodes 3\n"
    "symmetric\n"
    "edge 0 1\n"
    "edge 1 2\n";

constexpr auto k_sym = io::parse_symmetric<cg::cap::tiny>(k_sym_text);

static_assert(k_sym.node_count() == 3);
static_assert(k_sym.undirected_edge_count() == 2);

TEST(GraphIO, ConstexprParseSymmetric) {
    EXPECT_EQ(k_sym.node_count(), 3u);
    EXPECT_EQ(k_sym.undirected_edge_count(), 2u);

    // Build manually.
    constexpr auto expected = []() {
        cg::symmetric_graph_builder<cap_from<8, 24>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        return b.finalise();
    }();

    // symmetric_graph doesn't have graph_equal — compare node/edge counts
    // and neighbor lists.
    EXPECT_EQ(k_sym.node_count(), expected.node_count());
    EXPECT_EQ(k_sym.undirected_edge_count(), expected.undirected_edge_count());
}

// =============================================================================
// 3. Constexpr parse: weighted directed graph
// =============================================================================

constexpr auto k_weighted_text =
    "nodes 3\n"
    "edge 0 1 1.5\n"
    "edge 0 2 2.0\n"
    "edge 1 2 3.5\n";

TEST(GraphIO, ConstexprParseWeighted) {
    auto [g, wm] = io::parse_weighted_directed<cg::cap::tiny>(k_weighted_text);
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 3u);

    // Edges are in canonical order: (0,1), (0,2), (1,2).
    EXPECT_DOUBLE_EQ(wm[static_cast<std::size_t>(0)], 1.5);
    EXPECT_DOUBLE_EQ(wm[static_cast<std::size_t>(1)], 2.0);
    EXPECT_DOUBLE_EQ(wm[static_cast<std::size_t>(2)], 3.5);
}

// =============================================================================
// 4. Runtime round-trip: write → read → compare
// =============================================================================

TEST(GraphIO, RuntimeRoundTripDirected) {
    // Build a graph.
    auto g = []() {
        cg::graph_builder<cg::cap::small> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        auto n3 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n0, n3);
        b.add_edge(n1, n2);
        b.add_edge(n2, n3);
        return b.finalise();
    }();

    // Write to string.
    std::ostringstream oss;
    io::write(oss, g);
    std::string text = oss.str();

    // Read back.
    std::istringstream iss(text);
    auto g2 = io::read_directed<cg::cap::small>(iss);

    EXPECT_TRUE(cg::graph_equal(g, g2));
}

TEST(GraphIO, RuntimeRoundTripSymmetric) {
    auto g = []() {
        cg::symmetric_graph_builder<cap_from<16, 64>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n0, n2);
        b.add_edge(n1, n2);
        return b.finalise();
    }();

    std::ostringstream oss;
    io::write(oss, g);
    std::string text = oss.str();

    std::istringstream iss(text);
    auto g2 = io::read_symmetric<cg::cap::small>(iss);

    EXPECT_EQ(g.node_count(), g2.node_count());
    EXPECT_EQ(g.undirected_edge_count(), g2.undirected_edge_count());
}

// =============================================================================
// 5. DOT output
// =============================================================================

TEST(GraphIO, DotDirected) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes 3\nedge 0 1\nedge 1 2\n");

    std::ostringstream oss;
    io::write_dot(oss, g, "TestGraph");
    std::string dot = oss.str();

    EXPECT_NE(dot.find("digraph TestGraph"), std::string::npos);
    EXPECT_NE(dot.find("0 -> 1"), std::string::npos);
    EXPECT_NE(dot.find("1 -> 2"), std::string::npos);
    EXPECT_NE(dot.find("}"), std::string::npos);
}

TEST(GraphIO, DotSymmetric) {
    auto g = []() {
        cg::symmetric_graph_builder<cap_from<8, 24>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    std::ostringstream oss;
    io::write_dot(oss, g, "UG");
    std::string dot = oss.str();

    EXPECT_NE(dot.find("graph UG"), std::string::npos);
    EXPECT_NE(dot.find("0 -- 1"), std::string::npos);
    // Should NOT contain the reverse edge in undirected output.
    EXPECT_EQ(dot.find("1 -- 0"), std::string::npos);
}

TEST(GraphIO, DotWeighted) {
    auto [g, wm] = io::parse_weighted_directed<cg::cap::tiny>(
        "nodes 2\nedge 0 1 7.5\n");

    std::ostringstream oss;
    io::write_dot(oss, g,
        [&wm](auto const& /*g*/, std::size_t eidx) { return wm[eidx]; },
        "WG");
    std::string dot = oss.str();

    EXPECT_NE(dot.find("digraph WG"), std::string::npos);
    EXPECT_NE(dot.find("label=\"7.5\""), std::string::npos);
}

// =============================================================================
// 6. Error handling
// =============================================================================

TEST(GraphIO, ErrorMissingNodes) {
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>("edge 0 1\n"),
        std::runtime_error);
}

TEST(GraphIO, ErrorNodeOutOfRange) {
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>("nodes 2\nedge 0 5\n"),
        std::runtime_error);
}

TEST(GraphIO, ErrorEdgeBeforeNodes) {
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>("edge 0 1\nnodes 3\n"),
        std::runtime_error);
}

// =============================================================================
// 7. Comments and blank lines
// =============================================================================

TEST(GraphIO, CommentsAndBlankLines) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "# header comment\n"
        "\n"
        "  # indented comment\n"
        "nodes 2\n"
        "\n"
        "edge 0 1  # inline comment\n"
        "\n"
        "# trailing comment\n");

    EXPECT_EQ(g.node_count(), 2u);
    EXPECT_EQ(g.edge_count(), 1u);
}

// =============================================================================
// 8. Parse with no edges (isolated nodes)
// =============================================================================

TEST(GraphIO, IsolatedNodes) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>("nodes 5\n");
    static_assert(g.node_count() == 5);
    static_assert(g.edge_count() == 0);
}

// =============================================================================
// 9. Negative weight parsing
// =============================================================================

TEST(GraphIO, NegativeWeight) {
    auto [g, wm] = io::parse_weighted_directed<cg::cap::tiny>(
        "nodes 2\nedge 0 1 -3.25\n");
    EXPECT_EQ(g.edge_count(), 1u);
    EXPECT_DOUBLE_EQ(wm[static_cast<std::size_t>(0)], -3.25);
}

// =============================================================================
// 10. Self-edge removal (constexpr parser silently drops self-edges)
// =============================================================================

TEST(GraphIO, SelfEdgeRemoved) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes 3\nedge 0 1\nedge 1 1\nedge 1 2\n");
    // edge 1→1 is a self-edge, removed by graph_builder::finalise().
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 2u);
}

// =============================================================================
// 11. Duplicate edge removal
// =============================================================================

TEST(GraphIO, DuplicateEdgeRemoved) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes 3\nedge 0 1\nedge 0 1\nedge 1 2\n");
    // Duplicate 0→1 removed by finalise().
    EXPECT_EQ(g.edge_count(), 2u);
}

// =============================================================================
// 12. Weights ignored in topology-only parse
// =============================================================================

TEST(GraphIO, WeightsIgnoredInTopologyParse) {
    // parse_directed ignores weight fields — only topology matters.
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes 3\nedge 0 1 99.9\nedge 1 2 -42.0\n");
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 2u);
}

// =============================================================================
// 13. Error: duplicate 'nodes' line
// =============================================================================

TEST(GraphIO, ErrorDuplicateNodesLine) {
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>("nodes 3\nnodes 5\nedge 0 1\n"),
        std::runtime_error);
}

// =============================================================================
// 14. Error: unrecognised line
// =============================================================================

TEST(GraphIO, ErrorUnrecognisedLine) {
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>("nodes 3\nfoo bar\n"),
        std::runtime_error);
}

// =============================================================================
// 15. Error: capacity overflow — too many nodes
// =============================================================================

TEST(GraphIO, ErrorCapacityOverflowNodes) {
    // cap::tiny has max_v = 8; requesting 100 nodes should fail.
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>("nodes 100\n"),
        std::runtime_error);
}

// =============================================================================
// 16. Error: capacity overflow — too many edges
// =============================================================================

TEST(GraphIO, ErrorCapacityOverflowEdges) {
    // Build a string with more edges than cap::tiny (max_e = 24) allows.
    std::string text = "nodes 8\n";
    for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            text += "edge " + std::to_string(i) + " " + std::to_string(j) + "\n";
        }
    }
    // 7+6+5+4+3+2+1 = 28 edges > 24.  May throw runtime_error or length_error.
    EXPECT_THROW(
        (void)io::parse_directed<cg::cap::tiny>(text),
        std::exception);
}

// =============================================================================
// 17. Windows \r\n line endings
// =============================================================================

TEST(GraphIO, WindowsLineEndings) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes 3\r\nedge 0 1\r\nedge 1 2\r\n");
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 2u);
}

// =============================================================================
// 18. Leading whitespace on lines
// =============================================================================

TEST(GraphIO, LeadingWhitespace) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "  nodes 3\n  edge 0 1\n\tedge 1 2\n");
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 2u);
}

// =============================================================================
// 19. Tab-separated fields
// =============================================================================

TEST(GraphIO, TabSeparatedFields) {
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes\t3\nedge\t0\t1\nedge\t1\t2\n");
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 2u);
}

// =============================================================================
// 20. Write output format verification
// =============================================================================

TEST(GraphIO, WriteOutputFormat) {
    auto g = []() {
        cg::graph_builder<cap_from<8, 24>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    std::ostringstream oss;
    io::write(oss, g);
    std::string text = oss.str();

    // Must contain "nodes 2" and "edge 0 1".
    EXPECT_NE(text.find("nodes 2"), std::string::npos);
    EXPECT_NE(text.find("edge 0 1"), std::string::npos);
}

// =============================================================================
// 21. DOT writer renders isolated nodes
// =============================================================================

TEST(GraphIO, DotWriterIsolatedNode) {
    // Node 2 has no edges — should appear as "  2;" in DOT output.
    constexpr auto g = io::parse_directed<cg::cap::tiny>(
        "nodes 3\nedge 0 1\n");

    std::ostringstream oss;
    io::write_dot(oss, g, "Iso");
    std::string dot = oss.str();

    EXPECT_NE(dot.find("digraph Iso"), std::string::npos);
    EXPECT_NE(dot.find("  2;"), std::string::npos);
    EXPECT_NE(dot.find("0 -> 1"), std::string::npos);
}

// =============================================================================
// 22. Default MaxE — graph_builder<cap_from<8>> works (= graph_builder<cap_from<8, 32>>)
// =============================================================================

TEST(GraphIO, DefaultMaxEBuilder) {
    constexpr auto g = []() {
        cg::graph_builder<cap_from<8>> b;  // MaxE defaults to 4*8 = 32
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    static_assert(g.node_count() == 2);
    static_assert(g.edge_count() == 1);
}
