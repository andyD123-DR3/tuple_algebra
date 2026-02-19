// graph/test/graph_step2_test.cc - Tests for implicit_graph, from_stencil,
//                                  from_pipeline
// Part of the compile-time DP library (C++20)

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_equal.h"
#include "implicit_graph.h"
#include "from_stencil.h"
#include "from_pipeline.h"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// Compile-time verification: implicit_graph
// =============================================================================

// Simple generator: linear chain
constexpr auto make_implicit_chain(std::size_t N) {
    auto gen = [N](node_id u) {
        ctdp::constexpr_vector<node_id, 2> result;
        if (static_cast<std::size_t>(u.value) + 1 < N) {
            result.push_back(node_id{
                static_cast<std::uint16_t>(u.value + 1)});
        }
        return result;
    };
    return implicit_graph{N, gen};
}

// Verify concept satisfaction
static_assert(graph_queryable<
    implicit_graph<decltype([](node_id) {
        return ctdp::constexpr_vector<node_id, 1>{};
    })>>);

// Basic implicit_graph properties
static_assert([]() {
    auto g = make_implicit_chain(5);
    return g.node_count() == 5 && !g.empty() && g.has_node(node_id{4})
           && !g.has_node(node_id{5});
}());

// Adjacency from implicit chain
static_assert([]() {
    auto g = make_implicit_chain(4);
    auto n0 = g.out_neighbors(node_id{0});
    auto n3 = g.out_neighbors(node_id{3});
    return n0.size() == 1 && n0[0] == node_id{1}
           && n3.size() == 0;  // Last node has no successors
}());

// Empty implicit graph
static_assert([]() {
    auto gen = [](node_id) {
        return ctdp::constexpr_vector<node_id, 1>{};
    };
    auto g = implicit_graph{std::size_t{0}, gen};
    return g.empty() && g.node_count() == 0;
}());

// =============================================================================
// Compile-time verification: from_stencil
// =============================================================================

// 1D stencil: 3-point on 5 nodes → {left, right} neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 1>{5},
        std::array<std::array<int, 1>, 2>{{{-1}, {1}}}
    );
    return g.node_count() == 5;
}());

// 1D stencil: interior node has 2 neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 1>{5},
        std::array<std::array<int, 1>, 2>{{{-1}, {1}}}
    );
    auto nbrs = g.out_neighbors(node_id{2});
    return nbrs.size() == 2
           && nbrs[0] == node_id{1}   // left
           && nbrs[1] == node_id{3};  // right
}());

// 1D stencil: left boundary has 1 neighbor
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 1>{5},
        std::array<std::array<int, 1>, 2>{{{-1}, {1}}}
    );
    auto nbrs = g.out_neighbors(node_id{0});
    return nbrs.size() == 1 && nbrs[0] == node_id{1};
}());

// 1D stencil: right boundary has 1 neighbor
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 1>{5},
        std::array<std::array<int, 1>, 2>{{{-1}, {1}}}
    );
    auto nbrs = g.out_neighbors(node_id{4});
    return nbrs.size() == 1 && nbrs[0] == node_id{3};
}());

// 2D 5-point stencil on 4×4 grid (16 nodes)
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );
    return g.node_count() == 16;
}());

// 2D 5-point: corner node (0,0) = node 0 → 2 neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );
    auto nbrs = g.out_neighbors(node_id{0});
    // (0,0): offsets (-1,0) invalid, (1,0)→(1,0)=4, (0,-1) invalid, (0,1)→(0,1)=1
    return nbrs.size() == 2;
}());

// 2D 5-point: interior node (1,1) = node 5 → 4 neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );
    // Node 5 = (1,1) in row-major 4-wide grid
    auto nbrs = g.out_neighbors(node_id{5});
    // (-1,0)→(0,1)=1, (1,0)→(2,1)=9, (0,-1)→(1,0)=4, (0,1)→(1,2)=6
    return nbrs.size() == 4;
}());

// 2D 5-point: edge node (0,1) = node 1 → 3 neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );
    auto nbrs = g.out_neighbors(node_id{1});
    // (0,1): (-1,0) invalid, (1,0)→(1,1)=5, (0,-1)→(0,0)=0, (0,1)→(0,2)=2
    return nbrs.size() == 3;
}());

// 2D: verify specific neighbor values for interior node
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );
    auto nbrs = g.out_neighbors(node_id{5}); // (1,1)
    return nbrs[0] == node_id{1}   // (0,1) = 0*4+1 = 1
        && nbrs[1] == node_id{9}   // (2,1) = 2*4+1 = 9
        && nbrs[2] == node_id{4}   // (1,0) = 1*4+0 = 4
        && nbrs[3] == node_id{6};  // (1,2) = 1*4+2 = 6
}());

// 3D stencil: 6-point on 3×3×3 cube (27 nodes)
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 3>{3, 3, 3},
        std::array<std::array<int, 3>, 6>{
            {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}}}
    );
    return g.node_count() == 27;
}());

// 3D: center node (1,1,1) = node 13 → 6 neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 3>{3, 3, 3},
        std::array<std::array<int, 3>, 6>{
            {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}}}
    );
    auto nbrs = g.out_neighbors(node_id{13}); // flat index of (1,1,1) = 1*9+1*3+1 = 13
    return nbrs.size() == 6;
}());

// 3D: corner node (0,0,0) = node 0 → 3 neighbors
static_assert([]() {
    auto g = from_stencil(
        std::array<std::size_t, 3>{3, 3, 3},
        std::array<std::array<int, 3>, 6>{
            {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}}}
    );
    auto nbrs = g.out_neighbors(node_id{0});
    return nbrs.size() == 3;
}());

// =============================================================================
// Compile-time verification: from_pipeline
// =============================================================================

static_assert([]() {
    auto g = from_pipeline<8>(5);
    return g.node_count() == 5 && g.edge_count() == 4;
}());

static_assert([]() {
    auto g = from_pipeline<8>(1);
    return g.node_count() == 1 && g.edge_count() == 0;
}());

static_assert([]() {
    auto g = from_pipeline<8>(0);
    return g.node_count() == 0 && g.edge_count() == 0;
}());

// Pipeline adjacency: each node → next, last → none
static_assert([]() {
    auto g = from_pipeline<8>(4);
    return g.out_degree(node_id{0}) == 1
        && g.out_degree(node_id{1}) == 1
        && g.out_degree(node_id{2}) == 1
        && g.out_degree(node_id{3}) == 0;
}());

// Pipeline: node 0 → {1}
static_assert([]() {
    auto g = from_pipeline<8>(4);
    auto nbrs = g.out_neighbors_bounded<2>(node_id{0});
    return nbrs.size() == 1 && nbrs[0] == node_id{1};
}());

// Pipeline: node 2 → {3}
static_assert([]() {
    auto g = from_pipeline<8>(4);
    auto nbrs = g.out_neighbors_bounded<2>(node_id{2});
    return nbrs.size() == 1 && nbrs[0] == node_id{3};
}());

// Pipeline equals hand-built chain
static_assert([]() {
    auto pipeline = from_pipeline<8>(4);
    graph_builder<8, 8> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return graph_equal(pipeline, b.finalise());
}());

// =============================================================================
// Runtime tests (Google Test)
// =============================================================================

// --- implicit_graph ---

TEST(ImplicitGraph, BasicChain) {
    auto g = make_implicit_chain(5);
    EXPECT_EQ(g.node_count(), 5u);
    EXPECT_FALSE(g.empty());
    EXPECT_TRUE(g.has_node(node_id{4}));
    EXPECT_FALSE(g.has_node(node_id{5}));
}

TEST(ImplicitGraph, ChainAdjacency) {
    auto g = make_implicit_chain(4);
    // Node 0 → {1}
    auto n0 = g.out_neighbors(node_id{0});
    EXPECT_EQ(n0.size(), 1u);
    EXPECT_EQ(n0[0], node_id{1});

    // Node 2 → {3}
    auto n2 = g.out_neighbors(node_id{2});
    EXPECT_EQ(n2.size(), 1u);
    EXPECT_EQ(n2[0], node_id{3});

    // Node 3 → {} (last)
    auto n3 = g.out_neighbors(node_id{3});
    EXPECT_EQ(n3.size(), 0u);
}

TEST(ImplicitGraph, EmptyGraph) {
    auto gen = [](node_id) {
        return ctdp::constexpr_vector<node_id, 1>{};
    };
    auto g = implicit_graph{std::size_t{0}, gen};
    EXPECT_TRUE(g.empty());
    EXPECT_EQ(g.node_count(), 0u);
}

TEST(ImplicitGraph, FullyConnectedSmall) {
    // 3 nodes, each connected to all others
    auto gen = [](node_id u) {
        ctdp::constexpr_vector<node_id, 3> result;
        for (std::uint16_t i = 0; i < 3; ++i) {
            if (i != u.value) {
                result.push_back(node_id{i});
            }
        }
        return result;
    };
    auto g = implicit_graph{std::size_t{3}, gen};

    for (std::uint16_t i = 0; i < 3; ++i) {
        auto nbrs = g.out_neighbors(node_id{i});
        EXPECT_EQ(nbrs.size(), 2u);
    }
}

// --- from_stencil ---

TEST(FromStencil, OneDimensional) {
    constexpr auto g = from_stencil(
        std::array<std::size_t, 1>{10},
        std::array<std::array<int, 1>, 2>{{{-1}, {1}}}
    );

    EXPECT_EQ(g.node_count(), 10u);

    // Left boundary: 1 neighbor
    EXPECT_EQ(g.out_neighbors(node_id{0}).size(), 1u);

    // Interior: 2 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{5}).size(), 2u);

    // Right boundary: 1 neighbor
    EXPECT_EQ(g.out_neighbors(node_id{9}).size(), 1u);
}

TEST(FromStencil, TwoDimensional5Point) {
    constexpr auto g = from_stencil(
        std::array<std::size_t, 2>{8, 8},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );

    EXPECT_EQ(g.node_count(), 64u);

    // Corner (0,0) = node 0: 2 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{0}).size(), 2u);

    // Edge (0,3) = node 3: 3 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{3}).size(), 3u);

    // Interior (3,3) = node 27: 4 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{27}).size(), 4u);

    // Corner (7,7) = node 63: 2 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{63}).size(), 2u);
}

TEST(FromStencil, TwoDimensional9Point) {
    // 9-point stencil includes diagonals
    constexpr auto g = from_stencil(
        std::array<std::size_t, 2>{5, 5},
        std::array<std::array<int, 2>, 8>{
            {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}}}
    );

    EXPECT_EQ(g.node_count(), 25u);

    // Corner (0,0): 3 neighbors (right, below, diagonal)
    EXPECT_EQ(g.out_neighbors(node_id{0}).size(), 3u);

    // Interior (2,2) = node 12: 8 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{12}).size(), 8u);

    // Edge (0,2) = node 2: 5 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{2}).size(), 5u);
}

TEST(FromStencil, ThreeDimensional6Point) {
    constexpr auto g = from_stencil(
        std::array<std::size_t, 3>{4, 4, 4},
        std::array<std::array<int, 3>, 6>{
            {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}}}
    );

    EXPECT_EQ(g.node_count(), 64u);

    // Corner (0,0,0) = node 0: 3 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{0}).size(), 3u);

    // Center (2,2,2) = node 2*16+2*4+2 = 42: 6 neighbors
    EXPECT_EQ(g.out_neighbors(node_id{42}).size(), 6u);
}

TEST(FromStencil, SpecificNeighborValues2D) {
    constexpr auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );

    // Node 5 = (1,1): neighbors at (0,1)=1, (2,1)=9, (1,0)=4, (1,2)=6
    auto nbrs = g.out_neighbors(node_id{5});
    ASSERT_EQ(nbrs.size(), 4u);
    EXPECT_EQ(nbrs[0], node_id{1});
    EXPECT_EQ(nbrs[1], node_id{9});
    EXPECT_EQ(nbrs[2], node_id{4});
    EXPECT_EQ(nbrs[3], node_id{6});
}

TEST(FromStencil, SingleCell) {
    // 1×1 grid: no valid neighbors for any stencil
    constexpr auto g = from_stencil(
        std::array<std::size_t, 1>{1},
        std::array<std::array<int, 1>, 2>{{{-1}, {1}}}
    );

    EXPECT_EQ(g.node_count(), 1u);
    EXPECT_EQ(g.out_neighbors(node_id{0}).size(), 0u);
}

TEST(FromStencil, AsymmetricStencil) {
    // Forward-only stencil: only rightward neighbors
    constexpr auto g = from_stencil(
        std::array<std::size_t, 1>{5},
        std::array<std::array<int, 1>, 2>{{{1}, {2}}}
    );

    // Node 0: neighbors at 1 and 2
    auto n0 = g.out_neighbors(node_id{0});
    EXPECT_EQ(n0.size(), 2u);

    // Node 3: neighbor at 4 only (5 would be out of bounds)
    auto n3 = g.out_neighbors(node_id{3});
    EXPECT_EQ(n3.size(), 1u);

    // Node 4: no valid neighbors
    auto n4 = g.out_neighbors(node_id{4});
    EXPECT_EQ(n4.size(), 0u);
}

// --- from_pipeline ---

TEST(FromPipeline, Empty) {
    constexpr auto g = from_pipeline<4>(0);
    EXPECT_EQ(g.node_count(), 0u);
    EXPECT_EQ(g.edge_count(), 0u);
}

TEST(FromPipeline, SingleNode) {
    constexpr auto g = from_pipeline<4>(1);
    EXPECT_EQ(g.node_count(), 1u);
    EXPECT_EQ(g.edge_count(), 0u);
}

TEST(FromPipeline, FourStages) {
    constexpr auto g = from_pipeline<8>(4);
    EXPECT_EQ(g.node_count(), 4u);
    EXPECT_EQ(g.edge_count(), 3u);

    // 0 → {1}
    auto n0 = g.out_neighbors_bounded<2>(node_id{0});
    ASSERT_EQ(n0.size(), 1u);
    EXPECT_EQ(n0[0], node_id{1});

    // 1 → {2}
    auto n1 = g.out_neighbors_bounded<2>(node_id{1});
    ASSERT_EQ(n1.size(), 1u);
    EXPECT_EQ(n1[0], node_id{2});

    // 3 → {} (sink)
    EXPECT_EQ(g.out_degree(node_id{3}), 0u);
}

TEST(FromPipeline, LargePipeline) {
    constexpr auto g = from_pipeline<128>(64);
    EXPECT_EQ(g.node_count(), 64u);
    EXPECT_EQ(g.edge_count(), 63u);
    EXPECT_EQ(g.max_out_degree(), 1u);
}

TEST(FromPipeline, EqualsHandBuilt) {
    constexpr auto pipeline = from_pipeline<8>(3);

    constexpr auto manual = []() {
        graph_builder<8, 8> b;
        auto a = b.add_node();
        auto c = b.add_node();
        auto e = b.add_node();
        b.add_edge(a, c);
        b.add_edge(c, e);
        return b.finalise();
    }();

    EXPECT_TRUE(graph_equal(pipeline, manual));
}

// --- Stencil concept verification ---
TEST(FromStencil, SatisfiesGraphQueryable) {
    constexpr auto g = from_stencil(
        std::array<std::size_t, 2>{4, 4},
        std::array<std::array<int, 2>, 4>{{{-1,0},{1,0},{0,-1},{0,1}}}
    );

    // Verify we can use graph_queryable operations
    EXPECT_EQ(g.node_count(), 16u);
    auto nbrs = g.out_neighbors(node_id{0});
    EXPECT_GE(nbrs.size(), 1u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
