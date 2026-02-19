// tests/graph/test_graph_coloring.cc
// Tests for graph coloring algorithm and coloring_to_groups bridge.

#include "ctdp/graph/graph_coloring.h"
#include "ctdp/engine/bridge/coloring_to_groups.h"
#include "ctdp/graph/symmetric_graph.h"
#include "ctdp/graph/graph_builder.h"
#include <gtest/gtest.h>

using namespace ctdp::graph;

// =============================================================================
// Helper: build common test graphs
// =============================================================================

// Empty graph: 0 nodes
constexpr auto make_empty_sym() {
    symmetric_graph_builder<4, 4> b;
    return b.finalise();
}

// Single node, no edges
constexpr auto make_isolated_node() {
    symmetric_graph_builder<4, 4> b;
    (void)b.add_node();
    return b.finalise();
}

// Single edge: 0 — 1
constexpr auto make_single_edge() {
    symmetric_graph_builder<4, 4> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    b.add_edge(n0, n1);
    return b.finalise();
}

// Triangle: 0 — 1 — 2 — 0
constexpr auto make_triangle() {
    symmetric_graph_builder<4, 4> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n0);
    return b.finalise();
}

// Path: 0 — 1 — 2 — 3
constexpr auto make_path4() {
    symmetric_graph_builder<4, 4> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Complete graph K4: every pair connected
constexpr auto make_k4() {
    symmetric_graph_builder<4, 8> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n0, n3);
    b.add_edge(n1, n2);
    b.add_edge(n1, n3);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Star: node 0 connected to all others
constexpr auto make_star5() {
    symmetric_graph_builder<8, 8> b;
    auto center = b.add_node();
    for (std::size_t i = 0; i < 4; ++i) {
        auto leaf = b.add_node();
        b.add_edge(center, leaf);
    }
    return b.finalise();
}

// Bipartite K_{2,3}: {0,1} connected to {2,3,4}
constexpr auto make_bipartite() {
    symmetric_graph_builder<8, 8> b;
    for (std::size_t i = 0; i < 5; ++i) (void)b.add_node();
    b.add_edge(node_id{0}, node_id{2});
    b.add_edge(node_id{0}, node_id{3});
    b.add_edge(node_id{0}, node_id{4});
    b.add_edge(node_id{1}, node_id{2});
    b.add_edge(node_id{1}, node_id{3});
    b.add_edge(node_id{1}, node_id{4});
    return b.finalise();
}

// Petersen graph: 10 nodes, 15 edges, 3-regular, chromatic number = 3
constexpr auto make_petersen() {
    symmetric_graph_builder<10, 16> b;
    for (std::size_t i = 0; i < 10; ++i) (void)b.add_node();
    // Outer cycle: 0-1-2-3-4-0
    b.add_edge(node_id{0}, node_id{1});
    b.add_edge(node_id{1}, node_id{2});
    b.add_edge(node_id{2}, node_id{3});
    b.add_edge(node_id{3}, node_id{4});
    b.add_edge(node_id{4}, node_id{0});
    // Inner pentagram: 5-7-9-6-8-5
    b.add_edge(node_id{5}, node_id{7});
    b.add_edge(node_id{7}, node_id{9});
    b.add_edge(node_id{9}, node_id{6});
    b.add_edge(node_id{6}, node_id{8});
    b.add_edge(node_id{8}, node_id{5});
    // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    b.add_edge(node_id{0}, node_id{5});
    b.add_edge(node_id{1}, node_id{6});
    b.add_edge(node_id{2}, node_id{7});
    b.add_edge(node_id{3}, node_id{8});
    b.add_edge(node_id{4}, node_id{9});
    return b.finalise();
}

// Cycle C5 (odd cycle, chromatic number = 3)
constexpr auto make_cycle5() {
    symmetric_graph_builder<8, 8> b;
    for (std::size_t i = 0; i < 5; ++i) (void)b.add_node();
    for (std::size_t i = 0; i < 5; ++i) {
        b.add_edge(
            node_id{static_cast<std::uint16_t>(i)},
            node_id{static_cast<std::uint16_t>((i + 1) % 5)});
    }
    return b.finalise();
}

// =============================================================================
// Test: empty graph
// =============================================================================

TEST(GraphColoring, EmptyGraph) {
    constexpr auto g = make_empty_sym();
    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 0);
    static_assert(cr.verified);
    EXPECT_EQ(cr.color_count, 0u);
}

// =============================================================================
// Test: single isolated node → 1 colour
// =============================================================================

TEST(GraphColoring, SingleNode) {
    constexpr auto g = make_isolated_node();
    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 1);
    static_assert(cr.color_of[0] == 0);
    static_assert(cr.verified);
}

// =============================================================================
// Test: single edge → 2 colours
// =============================================================================

TEST(GraphColoring, SingleEdge) {
    constexpr auto g = make_single_edge();
    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 2);
    static_assert(cr.color_of[0] != cr.color_of[1]);
    static_assert(cr.verified);
}

// =============================================================================
// Test: triangle → 3 colours
// =============================================================================

TEST(GraphColoring, Triangle) {
    constexpr auto g = make_triangle();
    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 3);
    static_assert(cr.verified);

    // All three must be distinct.
    EXPECT_NE(cr.color_of[0], cr.color_of[1]);
    EXPECT_NE(cr.color_of[1], cr.color_of[2]);
    EXPECT_NE(cr.color_of[0], cr.color_of[2]);
}

// =============================================================================
// Test: path graph → 2 colours (bipartite)
// =============================================================================

TEST(GraphColoring, Path4) {
    constexpr auto g = make_path4();
    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 2);
    static_assert(cr.verified);

    // Alternating: 0 and 2 same colour, 1 and 3 same colour.
    EXPECT_EQ(cr.color_of[0], cr.color_of[2]);
    EXPECT_EQ(cr.color_of[1], cr.color_of[3]);
    EXPECT_NE(cr.color_of[0], cr.color_of[1]);
}

// =============================================================================
// Test: complete graph K4 → 4 colours
// =============================================================================

TEST(GraphColoring, CompleteK4) {
    constexpr auto g = make_k4();
    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 4);
    static_assert(cr.verified);

    // All four must be distinct.
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            EXPECT_NE(cr.color_of[i], cr.color_of[j]);
        }
    }
}

// =============================================================================
// Test: star graph → 2 colours
// =============================================================================

TEST(GraphColoring, Star5) {
    constexpr auto g = make_star5();
    constexpr auto cr = graph_coloring<8>(g);
    static_assert(cr.color_count == 2);
    static_assert(cr.verified);

    // Center gets one colour, all leaves get another.
    auto center_color = cr.color_of[0];
    for (std::size_t i = 1; i < 5; ++i) {
        EXPECT_NE(cr.color_of[i], center_color);
    }
}

// =============================================================================
// Test: bipartite graph → 2 colours
// =============================================================================

TEST(GraphColoring, Bipartite) {
    constexpr auto g = make_bipartite();
    constexpr auto cr = graph_coloring<8>(g);
    static_assert(cr.color_count == 2);
    static_assert(cr.verified);

    // {0,1} share a colour, {2,3,4} share a colour.
    EXPECT_EQ(cr.color_of[0], cr.color_of[1]);
    EXPECT_EQ(cr.color_of[2], cr.color_of[3]);
    EXPECT_EQ(cr.color_of[3], cr.color_of[4]);
    EXPECT_NE(cr.color_of[0], cr.color_of[2]);
}

// =============================================================================
// Test: odd cycle C5 → 3 colours
// =============================================================================

TEST(GraphColoring, OddCycle5) {
    constexpr auto g = make_cycle5();
    constexpr auto cr = graph_coloring<8>(g);
    static_assert(cr.color_count == 3);
    static_assert(cr.verified);
}

// =============================================================================
// Test: Petersen graph → at most 4 colours (optimal is 3)
// =============================================================================

TEST(GraphColoring, PetersenGraph) {
    constexpr auto g = make_petersen();
    constexpr auto cr = graph_coloring<10>(g);

    // Petersen is 3-chromatic. Greedy Welsh-Powell should achieve
    // at most 4 (max_degree + 1 = 4).
    static_assert(cr.color_count <= 4);
    static_assert(cr.verified);
    static_assert(cr.max_degree_plus_one == 4);

    // Verify at runtime too.
    EXPECT_LE(cr.color_count, 4u);
    EXPECT_TRUE(cr.verified);
}

// =============================================================================
// Test: quality metric — max_degree_plus_one
// =============================================================================

TEST(GraphColoring, QualityMetric) {
    constexpr auto g = make_k4();
    constexpr auto cr = graph_coloring<4>(g);

    // K4: max degree = 3, so greedy bound = 4.
    static_assert(cr.max_degree_plus_one == 4);
    // K4 is 4-chromatic, so greedy achieves optimal.
    static_assert(cr.color_count == cr.max_degree_plus_one);
}

// =============================================================================
// Test: disconnected graph
// =============================================================================

TEST(GraphColoring, DisconnectedGraph) {
    // Two isolated edges: {0-1} and {2-3}
    constexpr auto g = []() {
        symmetric_graph_builder<4, 4> b;
        for (std::size_t i = 0; i < 4; ++i) (void)b.add_node();
        b.add_edge(node_id{0}, node_id{1});
        b.add_edge(node_id{2}, node_id{3});
        return b.finalise();
    }();

    constexpr auto cr = graph_coloring<4>(g);
    static_assert(cr.color_count == 2);
    static_assert(cr.verified);

    // Each edge needs 2 colours, but colours can be reused across components.
    EXPECT_NE(cr.color_of[0], cr.color_of[1]);
    EXPECT_NE(cr.color_of[2], cr.color_of[3]);
}

// =============================================================================
// Test: coloring_to_groups bridge
// =============================================================================

TEST(ColoringToGroups, TriangleBridge) {
    constexpr auto g = make_triangle();
    constexpr auto cr = graph_coloring<4>(g);
    constexpr auto fg = coloring_to_groups(cr);

    static_assert(fg.group_count == 3);
    static_assert(fg.is_valid_dag);
    static_assert(fg.fused_edge_count == 0);

    // group_of mirrors color_of.
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(fg.group_of[i], cr.color_of[i]);
    }
}

TEST(ColoringToGroups, BipartiteBridge) {
    constexpr auto g = make_bipartite();
    constexpr auto cr = graph_coloring<8>(g);
    constexpr auto fg = coloring_to_groups(cr);

    static_assert(fg.group_count == 2);

    // Two groups: {0,1} and {2,3,4}
    EXPECT_EQ(fg.group_of[0], fg.group_of[1]);
    EXPECT_EQ(fg.group_of[2], fg.group_of[3]);
    EXPECT_EQ(fg.group_of[3], fg.group_of[4]);
    EXPECT_NE(fg.group_of[0], fg.group_of[2]);
}

TEST(ColoringToGroups, EmptyBridge) {
    constexpr auto g = make_empty_sym();
    constexpr auto cr = graph_coloring<4>(g);
    constexpr auto fg = coloring_to_groups(cr);

    static_assert(fg.group_count == 0);
}

// =============================================================================
// Test: constexpr validation — full pipeline
// =============================================================================

constexpr auto coloring_pipeline_test() {
    auto g = make_path4();
    auto cr = graph_coloring<4>(g);
    auto fg = coloring_to_groups(cr);
    return fg;
}

static_assert(coloring_pipeline_test().group_count == 2);
static_assert(coloring_pipeline_test().is_valid_dag);

// =============================================================================
// Test: concept enforcement — constexpr_graph does NOT satisfy
// =============================================================================

TEST(GraphColoring, ConceptEnforcement) {
    // symmetric_graph satisfies symmetric_graph_queryable.
    using SG = symmetric_graph<4, 4>;
    static_assert(symmetric_graph_queryable<SG>);

    // constexpr_graph does NOT satisfy symmetric_graph_queryable.
    using CG = constexpr_graph<4, 4>;
    static_assert(!symmetric_graph_queryable<CG>);

    // Therefore graph_coloring(constexpr_graph{}) would be a compile error.
    // (We can't test that here, but the concept enforcement is proven.)
}

// =============================================================================
// Test: Welsh-Powell ordering gives priority to high-degree nodes
// =============================================================================

TEST(GraphColoring, WelshPowellOrdering) {
    // Star graph: center (node 0) has degree 4, leaves have degree 1.
    // Welsh-Powell should process center first → gets colour 0.
    constexpr auto g = make_star5();
    constexpr auto cr = graph_coloring<8>(g);

    // Center should get colour 0 (processed first, no neighbours coloured yet).
    EXPECT_EQ(cr.color_of[0], 0u);
}

// =============================================================================
// Test: SpMV-like banded graph (row conflict pattern)
// =============================================================================

TEST(GraphColoring, BandedRowConflict) {
    // Simulate a tridiagonal row-conflict graph: row i conflicts with i±1.
    // This is a path graph — should be 2-colourable.
    constexpr std::size_t N = 8;
    constexpr auto g = []() {
        symmetric_graph_builder<N, N> b;
        for (std::size_t i = 0; i < N; ++i) (void)b.add_node();
        for (std::size_t i = 0; i + 1 < N; ++i) {
            b.add_edge(
                node_id{static_cast<std::uint16_t>(i)},
                node_id{static_cast<std::uint16_t>(i + 1)});
        }
        return b.finalise();
    }();

    constexpr auto cr = graph_coloring<N>(g);
    static_assert(cr.color_count == 2);
    static_assert(cr.verified);

    // Groups via bridge.
    constexpr auto fg = coloring_to_groups(cr);
    static_assert(fg.group_count == 2);

    // Even rows in one group, odd rows in another.
    for (std::size_t i = 0; i < N; i += 2) {
        EXPECT_EQ(fg.group_of[i], fg.group_of[0]);
    }
    for (std::size_t i = 1; i < N; i += 2) {
        EXPECT_EQ(fg.group_of[i], fg.group_of[1]);
    }
}
