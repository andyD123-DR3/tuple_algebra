// tests/graph/test_symmetric_graph.cc
// Tests for graph/symmetric_graph.h — undirected graph type.

#include "ctdp/graph/symmetric_graph.h"
#include "ctdp/graph/connected_components.h"
#include <gtest/gtest.h>

using namespace ctdp::graph;

// =============================================================================
// Construction
// =============================================================================

TEST(SymmetricGraph, EmptyGraph) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        return b.finalise();
    }();

    static_assert(g.node_count() == 0);
    static_assert(g.edge_count() == 0);
    static_assert(g.undirected_edge_count() == 0);
    static_assert(g.empty());
}

TEST(SymmetricGraph, SingleEdge) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    static_assert(g.node_count() == 2);
    static_assert(g.undirected_edge_count() == 1);
    static_assert(g.edge_count() == 2);  // two directed edges stored
}

TEST(SymmetricGraph, Triangle) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        b.add_edge(n0, n2);
        return b.finalise();
    }();

    static_assert(g.node_count() == 3);
    static_assert(g.undirected_edge_count() == 3);
    static_assert(g.edge_count() == 6);
}

TEST(SymmetricGraph, SelfEdgeIgnored) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        [[maybe_unused]] auto n1 = b.add_node();
        b.add_edge(n0, n0);  // self-edge — should be ignored
        return b.finalise();
    }();

    static_assert(g.edge_count() == 0);
    static_assert(g.undirected_edge_count() == 0);
}

TEST(SymmetricGraph, DuplicateEdgeDeduped) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n0, n1);  // duplicate
        b.add_edge(n1, n0);  // reverse of existing — also duplicate
        return b.finalise();
    }();

    // Only one logical undirected edge.
    static_assert(g.undirected_edge_count() == 1);
    static_assert(g.edge_count() == 2);
}

// =============================================================================
// Adjacency symmetry
// =============================================================================

TEST(SymmetricGraph, NeighborsSymmetric) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<8, 16>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        auto n3 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n0, n2);
        b.add_edge(n2, n3);
        return b.finalise();
    }();

    // n0 has neighbors {1, 2}
    EXPECT_EQ(g.degree(node_id{0}), 2u);

    // n1 has neighbor {0}
    EXPECT_EQ(g.degree(node_id{1}), 1u);

    // n2 has neighbors {0, 3}
    EXPECT_EQ(g.degree(node_id{2}), 2u);

    // n3 has neighbor {2}
    EXPECT_EQ(g.degree(node_id{3}), 1u);
}

TEST(SymmetricGraph, NeighborsAlias) {
    // neighbors() and out_neighbors() return the same thing.
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    auto nbrs = g.neighbors(node_id{0});
    auto out_nbrs = g.out_neighbors(node_id{0});

    EXPECT_EQ(nbrs.size(), out_nbrs.size());
    EXPECT_EQ(nbrs.size(), 1u);
}

TEST(SymmetricGraph, AdjacentQuery) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        [[maybe_unused]] auto n2 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    static_assert(g.adjacent(node_id{0}, node_id{1}));
    static_assert(g.adjacent(node_id{1}, node_id{0}));
    static_assert(!g.adjacent(node_id{0}, node_id{2}));
    static_assert(!g.adjacent(node_id{1}, node_id{2}));
}

// =============================================================================
// Degree analysis
// =============================================================================

TEST(SymmetricGraph, MaxDegree) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<8, 16>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        auto n3 = b.add_node();
        // n0 is the hub: degree 3
        b.add_edge(n0, n1);
        b.add_edge(n0, n2);
        b.add_edge(n0, n3);
        return b.finalise();
    }();

    static_assert(g.max_degree() == 3);
    static_assert(g.degree(node_id{0}) == 3);
    static_assert(g.degree(node_id{1}) == 1);
}

// =============================================================================
// Concept satisfaction
// =============================================================================

TEST(SymmetricGraph, Concepts) {
    using SG = symmetric_graph<cap_from<8, 16>>;
    using DG = constexpr_graph<cap_from<8, 16>>;

    static_assert(graph_queryable<SG>);
    static_assert(sized_graph<SG>);
    static_assert(symmetric_graph_queryable<SG>);

    // Directed graph does NOT satisfy symmetric_graph_queryable.
    static_assert(graph_queryable<DG>);
    static_assert(!symmetric_graph_queryable<DG>);
}

// =============================================================================
// Algorithm compatibility: connected_components works on symmetric_graph
// =============================================================================

TEST(SymmetricGraph, ConnectedComponents) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<8, 16>> b;
        // Component 1: {0, 1, 2}
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);

        // Component 2: {3, 4}
        auto n3 = b.add_node();
        auto n4 = b.add_node();
        b.add_edge(n3, n4);

        return b.finalise();
    }();

    // connected_components accepts graph_queryable — symmetric_graph satisfies it.
    constexpr auto cc = connected_components(g);
    static_assert(cc.component_count == 2);
    static_assert(cc.component_of[0] == cc.component_of[1]);
    static_assert(cc.component_of[1] == cc.component_of[2]);
    static_assert(cc.component_of[3] == cc.component_of[4]);
    static_assert(cc.component_of[0] != cc.component_of[3]);
}

// =============================================================================
// Larger graph: Petersen graph (10 nodes, 15 edges)
// =============================================================================

TEST(SymmetricGraph, PetersenGraph) {
    constexpr auto g = []() {
        symmetric_graph_builder<cap_from<16, 32>> b;
        for (int i = 0; i < 10; ++i) (void)b.add_node();

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
    }();

    static_assert(g.node_count() == 10);
    static_assert(g.undirected_edge_count() == 15);

    // Petersen graph is 3-regular: every node has degree 3.
    static_assert(g.degree(node_id{0}) == 3);
    static_assert(g.degree(node_id{5}) == 3);
    static_assert(g.max_degree() == 3);

    // Connected.
    constexpr auto cc = connected_components(g);
    static_assert(cc.component_count == 1);
}

// =============================================================================
// Constexpr validation
// =============================================================================

constexpr auto make_path_4() {
    symmetric_graph_builder<cap_from<8, 8>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

static_assert(make_path_4().node_count() == 4);
static_assert(make_path_4().undirected_edge_count() == 3);
static_assert(make_path_4().degree(node_id{0}) == 1);
static_assert(make_path_4().degree(node_id{1}) == 2);
static_assert(make_path_4().adjacent(node_id{0}, node_id{1}));
static_assert(!make_path_4().adjacent(node_id{0}, node_id{2}));
