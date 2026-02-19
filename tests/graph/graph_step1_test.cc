// graph/test/graph_step1_test.cc - Tests for constexpr_graph, graph_builder,
//                                  graph_concepts, graph_equal
// Part of the compile-time DP library (C++20)

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_equal.h"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// Helper: build common test graphs at compile time
// =============================================================================

// Empty graph: 0 nodes, 0 edges
constexpr auto make_empty() {
    graph_builder<8, 16> b;
    return b.finalise();
}

// Single node, no edges
constexpr auto make_singleton() {
    graph_builder<8, 16> b;
    [[maybe_unused]] auto n = b.add_node();
    return b.finalise();
}

// Linear chain: 0→1→2→3
constexpr auto make_chain() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Triangle: 0→1, 0→2, 1→2
constexpr auto make_triangle() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n1, n2);
    return b.finalise();
}

// Diamond: 0→1, 0→2, 1→3, 2→3
constexpr auto make_diamond() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n1, n3);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Graph with duplicates and self-edges (tests canonicalisation)
constexpr auto make_dirty() {
    graph_builder<8, 32> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    // Duplicate edges
    b.add_edge(n0, n1);
    b.add_edge(n0, n1);
    b.add_edge(n0, n1);
    // Self-edges
    b.add_edge(n0, n0);
    b.add_edge(n1, n1);
    // Normal edge
    b.add_edge(n1, n2);
    // Reverse order edge
    b.add_edge(n2, n0);
    // More duplicates
    b.add_edge(n1, n2);
    return b.finalise();
}

// Disconnected: 0→1, 2→3 (two components)
constexpr auto make_disconnected() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Star: 0→1, 0→2, 0→3, 0→4
constexpr auto make_star() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    auto n4 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n0, n3);
    b.add_edge(n0, n4);
    return b.finalise();
}

// =============================================================================
// Compile-time verification (static_assert)
// =============================================================================

// --- node_id ---
static_assert(node_id{0} == node_id{0});
static_assert(node_id{0} != node_id{1});
static_assert(node_id{0} < node_id{1});
static_assert(node_id{5} > node_id{3});
static_assert(to_index(node_id{42}) == 42);
static_assert(invalid_node.value == 0xFFFF);

// --- Empty graph ---
static_assert(make_empty().node_count() == 0);
static_assert(make_empty().edge_count() == 0);
static_assert(make_empty().empty());

// --- Singleton ---
static_assert(make_singleton().node_count() == 1);
static_assert(make_singleton().edge_count() == 0);
static_assert(!make_singleton().empty());
static_assert(make_singleton().out_degree(node_id{0}) == 0);

// --- Chain: 0→1→2→3 ---
static_assert(make_chain().node_count() == 4);
static_assert(make_chain().edge_count() == 3);
static_assert(make_chain().out_degree(node_id{0}) == 1);
static_assert(make_chain().out_degree(node_id{1}) == 1);
static_assert(make_chain().out_degree(node_id{2}) == 1);
static_assert(make_chain().out_degree(node_id{3}) == 0);
static_assert(make_chain().max_out_degree() == 1);
static_assert(make_chain().has_node(node_id{3}));
static_assert(!make_chain().has_node(node_id{4}));

// --- Triangle: 0→1, 0→2, 1→2 ---
static_assert(make_triangle().node_count() == 3);
static_assert(make_triangle().edge_count() == 3);
static_assert(make_triangle().out_degree(node_id{0}) == 2);
static_assert(make_triangle().out_degree(node_id{1}) == 1);
static_assert(make_triangle().out_degree(node_id{2}) == 0);
static_assert(make_triangle().max_out_degree() == 2);

// --- Diamond: 0→1, 0→2, 1→3, 2→3 ---
static_assert(make_diamond().node_count() == 4);
static_assert(make_diamond().edge_count() == 4);
static_assert(make_diamond().out_degree(node_id{0}) == 2);
static_assert(make_diamond().out_degree(node_id{3}) == 0);

// --- Dirty graph (tests canonicalisation) ---
// 3 duplicates of 0→1 → 1 edge; 2 self-edges removed; 1→2 deduped → 1 edge
// Clean edges: 0→1, 1→2, 2→0
static_assert(make_dirty().node_count() == 3);
static_assert(make_dirty().edge_count() == 3);

// --- Disconnected ---
static_assert(make_disconnected().node_count() == 4);
static_assert(make_disconnected().edge_count() == 2);

// --- Star ---
static_assert(make_star().node_count() == 5);
static_assert(make_star().edge_count() == 4);
static_assert(make_star().out_degree(node_id{0}) == 4);
static_assert(make_star().max_out_degree() == 4);

// --- out_neighbors_bounded ---
static_assert([]() {
    auto g = make_triangle();
    auto nbrs = g.out_neighbors_bounded<4>(node_id{0});
    return nbrs.size() == 2 &&
           nbrs[0] == node_id{1} &&
           nbrs[1] == node_id{2};
}());

static_assert([]() {
    auto g = make_chain();
    auto nbrs = g.out_neighbors_bounded<4>(node_id{1});
    return nbrs.size() == 1 && nbrs[0] == node_id{2};
}());

// --- graph_equal ---
static_assert(graph_equal(make_empty(), make_empty()));
static_assert(graph_equal(make_triangle(), make_triangle()));
static_assert(graph_equal(make_diamond(), make_diamond()));
static_assert(!graph_equal(make_triangle(), make_chain()));

// Two builders producing the same graph should be equal
static_assert([]() {
    // Build triangle in different order
    graph_builder<8, 16> b1;
    auto a = b1.add_node();
    auto c = b1.add_node();
    auto e = b1.add_node();
    b1.add_edge(a, c);
    b1.add_edge(a, e);
    b1.add_edge(c, e);

    graph_builder<8, 16> b2;
    auto x = b2.add_node();
    auto y = b2.add_node();
    auto z = b2.add_node();
    // Edges in reverse order — but finalise canonicalises
    b2.add_edge(y, z);
    b2.add_edge(x, z);
    b2.add_edge(x, y);

    return graph_equal(b1.finalise(), b2.finalise());
}());

// graph_equal detects different edge sets
static_assert([]() {
    graph_builder<8, 16> b1;
    auto a = b1.add_node();
    auto c = b1.add_node();
    b1.add_edge(a, c);

    graph_builder<8, 16> b2;
    auto x = b2.add_node();
    auto y = b2.add_node();
    b2.add_edge(y, x);  // Reversed edge

    return !graph_equal(b1.finalise(), b2.finalise());
}());

// --- Concept verification ---
static_assert(graph_queryable<constexpr_graph<4, 8>>);
static_assert(graph_queryable<constexpr_graph<64, 256>>);
static_assert(sized_graph<constexpr_graph<4, 8>>);

// --- Builder: add_nodes ---
static_assert([]() {
    graph_builder<16, 32> b;
    auto first = b.add_nodes(5);
    return first == node_id{0} && b.node_count() == 5;
}());

static_assert([]() {
    graph_builder<16, 32> b;
    [[maybe_unused]] auto n0 = b.add_node();  // node 0
    auto first = b.add_nodes(3);
    return first == node_id{1} && b.node_count() == 4;
}());

// =============================================================================
// Runtime tests (Google Test)
// =============================================================================

TEST(NodeId, BasicProperties) {
    constexpr node_id a{3};
    constexpr node_id b{5};
    constexpr node_id c{3};

    EXPECT_EQ(a, c);
    EXPECT_NE(a, b);
    EXPECT_LT(a, b);
    EXPECT_GT(b, a);
    EXPECT_EQ(to_index(a), 3u);
}

TEST(NodeId, InvalidSentinel) {
    EXPECT_EQ(invalid_node.value, 0xFFFF);
    EXPECT_NE(invalid_node, node_id{0});
}

TEST(ConstexprGraph, EmptyGraph) {
    constexpr auto g = make_empty();
    EXPECT_EQ(g.node_count(), 0u);
    EXPECT_EQ(g.edge_count(), 0u);
    EXPECT_TRUE(g.empty());
}

TEST(ConstexprGraph, Singleton) {
    constexpr auto g = make_singleton();
    EXPECT_EQ(g.node_count(), 1u);
    EXPECT_EQ(g.edge_count(), 0u);
    EXPECT_FALSE(g.empty());

    auto nbrs = g.out_neighbors(node_id{0});
    EXPECT_EQ(nbrs.size(), 0u);
}

TEST(ConstexprGraph, ChainTopology) {
    constexpr auto g = make_chain();
    EXPECT_EQ(g.node_count(), 4u);
    EXPECT_EQ(g.edge_count(), 3u);

    // Node 0 → {1}
    auto n0 = g.out_neighbors(node_id{0});
    EXPECT_EQ(n0.size(), 1u);
    EXPECT_EQ(n0.begin()->value, 1);

    // Node 1 → {2}
    auto n1 = g.out_neighbors(node_id{1});
    EXPECT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1.begin()->value, 2);

    // Node 3 → {}
    auto n3 = g.out_neighbors(node_id{3});
    EXPECT_EQ(n3.size(), 0u);
}

TEST(ConstexprGraph, TriangleTopology) {
    constexpr auto g = make_triangle();

    // Node 0 → {1, 2} (sorted)
    auto n0 = g.out_neighbors(node_id{0});
    EXPECT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0.begin()[0].value, 1);
    EXPECT_EQ(n0.begin()[1].value, 2);

    // Node 1 → {2}
    auto n1 = g.out_neighbors(node_id{1});
    EXPECT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1.begin()->value, 2);
}

TEST(ConstexprGraph, OutNeighborsBounded) {
    constexpr auto g = make_star();

    auto nbrs = g.out_neighbors_bounded<8>(node_id{0});
    EXPECT_EQ(nbrs.size(), 4u);
    EXPECT_EQ(nbrs[0], node_id{1});
    EXPECT_EQ(nbrs[1], node_id{2});
    EXPECT_EQ(nbrs[2], node_id{3});
    EXPECT_EQ(nbrs[3], node_id{4});
}

TEST(ConstexprGraph, OutDegree) {
    constexpr auto g = make_diamond();
    EXPECT_EQ(g.out_degree(node_id{0}), 2u);
    EXPECT_EQ(g.out_degree(node_id{1}), 1u);
    EXPECT_EQ(g.out_degree(node_id{2}), 1u);
    EXPECT_EQ(g.out_degree(node_id{3}), 0u);
}

TEST(ConstexprGraph, MaxOutDegree) {
    EXPECT_EQ(make_chain().max_out_degree(), 1u);
    EXPECT_EQ(make_star().max_out_degree(), 4u);
    EXPECT_EQ(make_empty().max_out_degree(), 0u);
    EXPECT_EQ(make_singleton().max_out_degree(), 0u);
}

TEST(ConstexprGraph, HasNode) {
    constexpr auto g = make_chain();
    EXPECT_TRUE(g.has_node(node_id{0}));
    EXPECT_TRUE(g.has_node(node_id{3}));
    EXPECT_FALSE(g.has_node(node_id{4}));
    EXPECT_FALSE(g.has_node(node_id{100}));
}

TEST(GraphBuilder, Canonicalisation_SelfEdgesRemoved) {
    constexpr auto g = []() {
        graph_builder<4, 8> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n0);  // self-edge
        b.add_edge(n0, n1);
        b.add_edge(n1, n1);  // self-edge
        return b.finalise();
    }();

    EXPECT_EQ(g.node_count(), 2u);
    EXPECT_EQ(g.edge_count(), 1u);  // Only 0→1 survives
}

TEST(GraphBuilder, Canonicalisation_DuplicatesRemoved) {
    constexpr auto g = []() {
        graph_builder<4, 16> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        // Five copies of same edge
        b.add_edge(n0, n1);
        b.add_edge(n0, n1);
        b.add_edge(n0, n1);
        b.add_edge(n0, n1);
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    EXPECT_EQ(g.edge_count(), 1u);
}

TEST(GraphBuilder, Canonicalisation_EdgeOrdering) {
    // Edges added in reverse order should produce same graph
    constexpr auto g = []() {
        graph_builder<4, 8> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        // Add in reverse: 2→0, 1→2, 0→1
        b.add_edge(n2, n0);
        b.add_edge(n1, n2);
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    // Should be sorted: node 0 → {1}, node 1 → {2}, node 2 → {0}
    auto nbrs0 = g.out_neighbors(node_id{0});
    ASSERT_EQ(nbrs0.size(), 1u);
    EXPECT_EQ(nbrs0.begin()[0].value, 1);

    auto nbrs1 = g.out_neighbors(node_id{1});
    ASSERT_EQ(nbrs1.size(), 1u);
    EXPECT_EQ(nbrs1.begin()[0].value, 2);

    auto nbrs2 = g.out_neighbors(node_id{2});
    ASSERT_EQ(nbrs2.size(), 1u);
    EXPECT_EQ(nbrs2.begin()[0].value, 0);
}

TEST(GraphBuilder, DirtyGraphCanonicalisation) {
    constexpr auto g = make_dirty();
    // Original: 0→1 x3, 0→0, 1→1, 1→2 x2, 2→0
    // Clean: 0→1, 1→2, 2→0  (3 edges)
    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 3u);
    EXPECT_EQ(g.out_degree(node_id{0}), 1u);
    EXPECT_EQ(g.out_degree(node_id{1}), 1u);
    EXPECT_EQ(g.out_degree(node_id{2}), 1u);
}

TEST(GraphBuilder, AddNodes) {
    constexpr auto g = []() {
        graph_builder<16, 32> b;
        auto first = b.add_nodes(5);
        b.add_edge(first, node_id{static_cast<std::uint16_t>(first.value + 1)});
        return b.finalise();
    }();

    EXPECT_EQ(g.node_count(), 5u);
    EXPECT_EQ(g.edge_count(), 1u);
}

TEST(GraphBuilder, NoEdges) {
    constexpr auto g = []() {
        graph_builder<4, 8> b;
        [[maybe_unused]] auto first = b.add_nodes(3);
        return b.finalise();
    }();

    EXPECT_EQ(g.node_count(), 3u);
    EXPECT_EQ(g.edge_count(), 0u);
    EXPECT_EQ(g.out_degree(node_id{0}), 0u);
    EXPECT_EQ(g.out_degree(node_id{1}), 0u);
    EXPECT_EQ(g.out_degree(node_id{2}), 0u);
}

TEST(GraphEqual, IdenticalGraphs) {
    EXPECT_TRUE(graph_equal(make_empty(), make_empty()));
    EXPECT_TRUE(graph_equal(make_singleton(), make_singleton()));
    EXPECT_TRUE(graph_equal(make_chain(), make_chain()));
    EXPECT_TRUE(graph_equal(make_triangle(), make_triangle()));
    EXPECT_TRUE(graph_equal(make_diamond(), make_diamond()));
    EXPECT_TRUE(graph_equal(make_star(), make_star()));
}

TEST(GraphEqual, DifferentGraphs) {
    EXPECT_FALSE(graph_equal(make_chain(), make_triangle()));
    EXPECT_FALSE(graph_equal(make_empty(), make_singleton()));
    EXPECT_FALSE(graph_equal(make_diamond(), make_star()));
}

TEST(GraphEqual, SameStructureDifferentBuild) {
    // Build same triangle two different ways
    auto g1 = []() {
        graph_builder<8, 16> b;
        auto a = b.add_node();
        auto c = b.add_node();
        auto e = b.add_node();
        b.add_edge(a, c);
        b.add_edge(a, e);
        b.add_edge(c, e);
        return b.finalise();
    }();

    auto g2 = []() {
        graph_builder<8, 16> b;
        auto x = b.add_node();
        auto y = b.add_node();
        auto z = b.add_node();
        // Different insertion order + duplicates + self-edge
        b.add_edge(y, z);
        b.add_edge(x, z);
        b.add_edge(x, y);
        b.add_edge(x, y);  // dup
        b.add_edge(y, y);  // self
        return b.finalise();
    }();

    EXPECT_TRUE(graph_equal(g1, g2));
}

TEST(GraphEqual, ReversedEdge) {
    auto g1 = []() {
        graph_builder<4, 4> b;
        auto a = b.add_node();
        auto c = b.add_node();
        b.add_edge(a, c);
        return b.finalise();
    }();

    auto g2 = []() {
        graph_builder<4, 4> b;
        auto a = b.add_node();
        auto c = b.add_node();
        b.add_edge(c, a);  // reversed
        return b.finalise();
    }();

    EXPECT_FALSE(graph_equal(g1, g2));
}

TEST(GraphEqual, SameNodesDifferentEdges) {
    auto g1 = []() {
        graph_builder<4, 4> b;
        auto a = b.add_node();
        auto c = b.add_node();
        [[maybe_unused]] auto e = b.add_node();
        b.add_edge(a, c);
        return b.finalise();
    }();

    auto g2 = []() {
        graph_builder<4, 4> b;
        auto a = b.add_node();
        [[maybe_unused]] auto mid = b.add_node();
        auto e = b.add_node();
        b.add_edge(a, e);
        return b.finalise();
    }();

    EXPECT_FALSE(graph_equal(g1, g2));
}

// --- Adjacency iteration ---
TEST(ConstexprGraph, AdjacencyRangeIteration) {
    constexpr auto g = make_diamond();

    // Node 0 → {1, 2}
    std::size_t count = 0;
    for (auto n : g.out_neighbors(node_id{0})) {
        if (count == 0) { EXPECT_EQ(n.value, 1); }
        if (count == 1) { EXPECT_EQ(n.value, 2); }
        ++count;
    }
    EXPECT_EQ(count, 2u);
}

// --- Large graph stress test ---
TEST(GraphBuilder, LargerGraph) {
    // Build a 64-node chain
    constexpr auto g = []() {
        graph_builder<128, 256> b;
        node_id prev = b.add_node();
        for (int i = 1; i < 64; ++i) {
            auto cur = b.add_node();
            b.add_edge(prev, cur);
            prev = cur;
        }
        return b.finalise();
    }();

    EXPECT_EQ(g.node_count(), 64u);
    EXPECT_EQ(g.edge_count(), 63u);
    EXPECT_EQ(g.max_out_degree(), 1u);
    EXPECT_EQ(g.out_degree(node_id{0}), 1u);
    EXPECT_EQ(g.out_degree(node_id{63}), 0u);
}

// --- Edge pipeline simulation (from Phase 1a constexpr_sort) ---
TEST(GraphBuilder, EdgeSortingPipeline) {
    // Simulate graph_builder::finalise() edge canonicalization
    // by building edges in completely random order
    constexpr auto g = []() {
        graph_builder<8, 32> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        auto n3 = b.add_node();

        // Add edges in reverse and mixed order
        b.add_edge(n3, n0);
        b.add_edge(n1, n3);
        b.add_edge(n0, n2);
        b.add_edge(n2, n3);
        b.add_edge(n0, n1);
        b.add_edge(n2, n1);

        return b.finalise();
    }();

    EXPECT_EQ(g.edge_count(), 6u);

    // Verify sorted adjacency
    // Node 0: {1, 2}
    auto n0 = g.out_neighbors_bounded<4>(node_id{0});
    ASSERT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], node_id{1});
    EXPECT_EQ(n0[1], node_id{2});

    // Node 1: {3}
    auto n1 = g.out_neighbors_bounded<4>(node_id{1});
    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], node_id{3});

    // Node 2: {1, 3}
    auto n2 = g.out_neighbors_bounded<4>(node_id{2});
    ASSERT_EQ(n2.size(), 2u);
    EXPECT_EQ(n2[0], node_id{1});
    EXPECT_EQ(n2[1], node_id{3});

    // Node 3: {0}
    auto n3 = g.out_neighbors_bounded<4>(node_id{3});
    ASSERT_EQ(n3.size(), 1u);
    EXPECT_EQ(n3[0], node_id{0});
}

// --- Compile-time construction + query ---
TEST(ConstexprGraph, CompileTimeConstruction) {
    // This verifies the full pipeline works at compile time
    constexpr auto g = make_diamond();
    constexpr auto nbrs = g.out_neighbors_bounded<4>(node_id{0});
    static_assert(nbrs.size() == 2);
    static_assert(nbrs[0] == node_id{1});
    static_assert(nbrs[1] == node_id{2});

    // Also verify edge count from dirty graph
    constexpr auto dirty = make_dirty();
    static_assert(dirty.edge_count() == 3);  // After canonicalisation
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
