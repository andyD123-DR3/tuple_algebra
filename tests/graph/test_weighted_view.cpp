// tests/graph/test_weighted_view.cpp
// Tests for Option C weighted graph infrastructure:
//   edge_property_map, weighted_view, weighted_graph_queryable concept
//
// Validates:
//   1. edge_property_map basic operations + topology token binding
//   2. constexpr_graph edge_begin_offset, edge_range, edge_target, token
//   3. weighted_view models weighted_graph_queryable
//   4. weighted_view constructor enforces size + token agreement
//   5. weighted_out_neighbors yields correct (target, weight, eid) triples
//   6. weighted_out_neighbors_bounded (by-value, constexpr-safe)
//   7. symmetric_graph + make_symmetric_weight_map + verify_symmetric_weights
//   8. Full constexpr pipeline with concept-constrained algorithm
//   9. Edge ordering determinism (same input → same ids)
//  10. Stale map detection (rebuild → token mismatch)
//  11. Concept satisfaction and negative checks

#include "ctdp/graph/edge_property_map.h"
#include "ctdp/graph/weighted_view.h"
#include "ctdp/graph/weighted_graph_builder.h"
#include "ctdp/graph/graph_builder.h"
#include "ctdp/graph/symmetric_graph.h"
#include "ctdp/graph/constexpr_graph.h"
#include <gtest/gtest.h>

using namespace ctdp::graph;

// =========================================================================
// Test graph fixture: diamond graph  0 → 1, 0 → 2, 1 → 3, 2 → 3
//
//       0
//      / .
//     1   2      (directed: edges go downward)
//      . /
//       3
//
// 4 nodes, 4 edges.  CSR after finalise (sorted by src, then dst):
//   node 0: edges at positions [0, 1] → targets 1, 2
//   node 1: edge  at position  [2]    → target 3
//   node 2: edge  at position  [3]    → target 3
//   node 3: no outgoing edges
// =========================================================================

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

constexpr auto diamond = make_diamond();

// =========================================================================
// 1. edge_property_map basics
// =========================================================================

TEST(EdgePropertyMap, ConstructDefault) {
    constexpr edge_property_map<double, 16> m;
    static_assert(m.size() == 0);
    EXPECT_EQ(m.size(), 0u);
}

TEST(EdgePropertyMap, ConstructWithSize) {
    constexpr edge_property_map<double, 16> m(4, 1.5);
    static_assert(m.size() == 4);
    static_assert(m[std::size_t{0}] == 1.5);
    static_assert(m[std::size_t{3}] == 1.5);
    EXPECT_DOUBLE_EQ(m[std::size_t{2}], 1.5);
}

TEST(EdgePropertyMap, ConstructWithToken) {
    constexpr auto tok = diamond.token();
    constexpr edge_property_map<double, 16> m(4, 0.0, tok);
    static_assert(m.token() == tok);
    EXPECT_EQ(m.token(), tok);
}

TEST(EdgePropertyMap, MutateAndCompare) {
    edge_property_map<int, 8> a(4, 0);
    edge_property_map<int, 8> b(4, 0);
    EXPECT_EQ(a, b);

    a[std::size_t{2}] = 42;
    EXPECT_NE(a, b);

    b[std::size_t{2}] = 42;
    EXPECT_EQ(a, b);
}

TEST(EdgePropertyMap, EdgeIdAccess) {
    edge_property_map<double, 8> m(4, 0.0);
    m[edge_id{1}] = 3.14;
    EXPECT_DOUBLE_EQ(m[edge_id{1}], 3.14);
    EXPECT_DOUBLE_EQ(get(m, edge_id{1}), 3.14);
}

TEST(EdgePropertyMap, OutOfBoundsThrows) {
    edge_property_map<int, 4> m(2, 0);
    EXPECT_THROW(m[std::size_t{2}], std::out_of_range);
    EXPECT_THROW(m[edge_id{99}], std::out_of_range);
}

TEST(EdgePropertyMap, MakeUniformEdgeMapBindsToken) {
    constexpr auto m = make_uniform_edge_map<double, 16>(diamond, 1.0);
    static_assert(m.size() == 4);
    static_assert(m[std::size_t{0}] == 1.0);
    static_assert(m.token() == diamond.token());
    EXPECT_EQ(m.token(), diamond.token());
}

TEST(EdgePropertyMap, MakeEdgeMapFromFunction) {
    constexpr auto m = make_edge_map<int, 16>(diamond,
        [](std::size_t i) constexpr { return static_cast<int>(i * 10); });
    static_assert(m[std::size_t{0}] == 0);
    static_assert(m[std::size_t{1}] == 10);
    static_assert(m[std::size_t{2}] == 20);
    static_assert(m[std::size_t{3}] == 30);
    static_assert(m.token() == diamond.token());
}

// =========================================================================
// 2. constexpr_graph: edge_begin_offset, edge_range, edge_target, token
// =========================================================================

TEST(ConstexprGraph, EdgeBeginOffset) {
    static_assert(diamond.edge_begin_offset(node_id{0}) == 0);
    static_assert(diamond.edge_begin_offset(node_id{1}) == 2);
    static_assert(diamond.edge_begin_offset(node_id{2}) == 3);
    static_assert(diamond.edge_begin_offset(node_id{3}) == 4);
}

TEST(ConstexprGraph, EdgeRange) {
    constexpr auto r0 = diamond.edge_range(node_id{0});
    static_assert(r0.first == edge_id{0});
    static_assert(r0.second == edge_id{2});

    constexpr auto r3 = diamond.edge_range(node_id{3});
    static_assert(r3.first == r3.second);  // no outgoing edges
}

TEST(ConstexprGraph, EdgeTarget) {
    static_assert(diamond.edge_target(edge_id{0}) == node_id{1});
    static_assert(diamond.edge_target(edge_id{1}) == node_id{2});
    static_assert(diamond.edge_target(edge_id{2}) == node_id{3});
    static_assert(diamond.edge_target(edge_id{3}) == node_id{3});
}

TEST(ConstexprGraph, TopologyToken) {
    constexpr auto tok = diamond.token();
    static_assert(tok.value != 0);  // non-trivial hash

    // Same construction → same token
    constexpr auto diamond2 = make_diamond();
    static_assert(diamond.token() == diamond2.token());

    EXPECT_EQ(diamond.token(), diamond2.token());
}

TEST(ConstexprGraph, DifferentGraphDifferentToken) {
    // Build a different graph — triangle
    constexpr auto triangle = []() {
        graph_builder<8, 16> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        b.add_edge(n2, n0);
        return b.finalise();
    }();

    static_assert(!(diamond.token() == triangle.token()));
    EXPECT_NE(diamond.token(), triangle.token());
}

// =========================================================================
// 3. weighted_view construction and concept satisfaction
// =========================================================================

TEST(WeightedView, ConceptSatisfaction) {
    static_assert(graph_queryable<
        weighted_view<constexpr_graph<8, 16>, double, 16>>);
    static_assert(weighted_graph_queryable<
        weighted_view<constexpr_graph<8, 16>, double, 16>>);
    static_assert(!symmetric_weighted_queryable<
        weighted_view<constexpr_graph<8, 16>, double, 16>>);

    static_assert(weighted_graph_queryable<
        weighted_view<symmetric_graph<8, 16>, double, 32>>);
    static_assert(symmetric_weighted_queryable<
        weighted_view<symmetric_graph<8, 16>, double, 32>>);
}

TEST(WeightedView, ForwardsGraphQueries) {
    constexpr auto weights = make_uniform_edge_map<double, 16>(diamond, 1.0);
    auto wv = make_weighted_view(diamond, weights);

    EXPECT_EQ(wv.node_count(), 4u);
    EXPECT_EQ(wv.edge_count(), 4u);
    EXPECT_EQ(wv.out_degree(node_id{0}), 2u);
    EXPECT_EQ(wv.out_degree(node_id{3}), 0u);
    EXPECT_TRUE(wv.has_node(node_id{2}));
    EXPECT_EQ(wv.token(), diamond.token());
}

// =========================================================================
// 4. Constructor enforcement — size and token mismatch detection
// =========================================================================

TEST(WeightedView, SizeMismatchThrows) {
    // Map with 3 edges, graph has 4 — must fail
    edge_property_map<double, 16> bad_map(3, 0.0);
    EXPECT_THROW((void)make_weighted_view(diamond, bad_map), std::logic_error);
}

TEST(WeightedView, TokenMismatchThrows) {
    // Build a different graph, get its token
    auto triangle = []() {
        graph_builder<8, 16> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        b.add_edge(n2, n0);
        b.add_edge(n0, n2);
        return b.finalise();
    }();

    // Both have 4 edges, but different topology
    ASSERT_EQ(triangle.edge_count(), diamond.edge_count());
    ASSERT_NE(triangle.token(), diamond.token());

    // Map bound to triangle, applied to diamond — must fail
    auto wrong_map = make_uniform_edge_map<double, 16>(triangle, 1.0);
    EXPECT_THROW((void)make_weighted_view(diamond, wrong_map), std::logic_error);
}

TEST(WeightedView, UnboundMapAccepted) {
    // Manual construction without token — should be accepted (size check only)
    edge_property_map<double, 16> unbound(4, 1.0);
    EXPECT_EQ(unbound.token().value, 0u);
    EXPECT_NO_THROW((void)make_weighted_view(diamond, unbound));
}

// =========================================================================
// 5. weighted_out_neighbors — correct triples
// =========================================================================

TEST(WeightedView, WeightedOutNeighbors) {
    constexpr auto weights = make_edge_map<double, 16>(diamond,
        [](std::size_t i) constexpr { return static_cast<double>(i + 1); });
    auto wv = make_weighted_view(diamond, weights);

    // Node 0 → targets {1, 2} with weights {1.0, 2.0}
    std::vector<weighted_edge<double>> edges0;
    for (auto const& e : wv.weighted_out_neighbors(node_id{0})) {
        edges0.push_back(e);
    }
    ASSERT_EQ(edges0.size(), 2u);
    EXPECT_EQ(edges0[0].target, node_id{1});
    EXPECT_DOUBLE_EQ(edges0[0].weight, 1.0);
    EXPECT_EQ(edges0[0].eid, edge_id{0});
    EXPECT_EQ(edges0[1].target, node_id{2});
    EXPECT_DOUBLE_EQ(edges0[1].weight, 2.0);
    EXPECT_EQ(edges0[1].eid, edge_id{1});

    // Node 3 → no outgoing edges
    auto range3 = wv.weighted_out_neighbors(node_id{3});
    EXPECT_TRUE(range3.empty());
}

// =========================================================================
// 6. weighted_out_neighbors_bounded (by-value constexpr-safe)
// =========================================================================

TEST(WeightedView, BoundedWeightedNeighbors) {
    constexpr auto weights = make_edge_map<double, 16>(diamond,
        [](std::size_t i) constexpr { return static_cast<double>(i * 100); });
    auto wv = make_weighted_view(diamond, weights);

    auto nbr0 = wv.weighted_out_neighbors_bounded<4>(node_id{0});
    EXPECT_EQ(nbr0.size(), 2u);
    EXPECT_EQ(nbr0[0].target, node_id{1});
    EXPECT_DOUBLE_EQ(nbr0[0].weight, 0.0);
    EXPECT_EQ(nbr0[0].eid, edge_id{0});
    EXPECT_EQ(nbr0[1].target, node_id{2});
    EXPECT_DOUBLE_EQ(nbr0[1].weight, 100.0);
    EXPECT_EQ(nbr0[1].eid, edge_id{1});
}

// =========================================================================
// 7. Symmetric weighted graph + validation
// =========================================================================

constexpr auto make_sym_triangle() {
    symmetric_graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n0, n2);
    return b.finalise();
}

constexpr auto sym_tri = make_sym_triangle();

TEST(SymmetricWeighted, MakeSymmetricWeightMapBindsToken) {
    constexpr auto weights = make_symmetric_weight_map<double, 16>(sym_tri,
        [](node_id, node_id) constexpr { return 1.0; });
    static_assert(weights.token() == sym_tri.token());
}

TEST(SymmetricWeighted, VerifySymmetricWeightsPass) {
    constexpr auto weights = make_symmetric_weight_map<double, 16>(sym_tri,
        [](node_id lo, node_id hi) constexpr {
            return static_cast<double>(lo.value * 10 + hi.value);
        });

    constexpr bool ok = verify_symmetric_weights(sym_tri, weights);
    static_assert(ok);
    EXPECT_TRUE(ok);
}

TEST(SymmetricWeighted, VerifySymmetricWeightsFail) {
    // Manually create an asymmetric weight map
    auto weights = make_symmetric_weight_map<double, 16>(sym_tri,
        [](node_id, node_id) { return 1.0; });

    // Corrupt one edge's weight to break symmetry
    weights[std::size_t{0}] = 999.0;

    EXPECT_FALSE(verify_symmetric_weights(sym_tri, weights));
}

TEST(SymmetricWeighted, ConceptSatisfied) {
    constexpr auto weights = make_symmetric_weight_map<double, 16>(sym_tri,
        [](node_id, node_id) constexpr { return 1.0; });
    auto wv = make_weighted_view(sym_tri, weights);

    EXPECT_EQ(wv.node_count(), 3u);
    EXPECT_EQ(wv.edge_count(), 6u);  // 3 undirected = 6 directed

    std::vector<weighted_edge<double>> edges;
    for (auto const& e : wv.weighted_out_neighbors(node_id{0})) {
        edges.push_back(e);
    }
    EXPECT_EQ(edges.size(), 2u);
}

// =========================================================================
// 8. Full constexpr pipeline with concept-constrained algorithm
// =========================================================================

template<weighted_graph_queryable G>
constexpr auto min_weight_edge(G const& g, node_id u) {
    using W = typename G::weight_type;
    weighted_edge<W> best{invalid_node, W{}, invalid_edge};
    bool first = true;
    for (auto const& e : g.weighted_out_neighbors(u)) {
        if (first || e.weight < best.weight) {
            best = e;
            first = false;
        }
    }
    return best;
}

TEST(WeightedView, ConstrainedAlgorithm) {
    constexpr auto weights = []() constexpr {
        edge_property_map<double, 16> m(4, 0.0, diamond.token());
        m[std::size_t{0}] = 10.0;  // 0→1
        m[std::size_t{1}] = 5.0;   // 0→2
        m[std::size_t{2}] = 3.0;   // 1→3
        m[std::size_t{3}] = 7.0;   // 2→3
        return m;
    }();

    auto wv = make_weighted_view(diamond, weights);

    auto best0 = min_weight_edge(wv, node_id{0});
    EXPECT_EQ(best0.target, node_id{2});
    EXPECT_DOUBLE_EQ(best0.weight, 5.0);
    EXPECT_EQ(best0.eid, edge_id{1});

    auto best1 = min_weight_edge(wv, node_id{1});
    EXPECT_EQ(best1.target, node_id{3});
    EXPECT_DOUBLE_EQ(best1.weight, 3.0);
}

// =========================================================================
// 9. Edge ordering determinism
// =========================================================================

TEST(WeightedView, EdgeOrderingDeterminism) {
    // Build same graph twice with edges in DIFFERENT insertion order
    constexpr auto g1 = []() {
        graph_builder<8, 16> b;
        auto n0 = b.add_node(); auto n1 = b.add_node();
        auto n2 = b.add_node(); auto n3 = b.add_node();
        b.add_edge(n0, n1); b.add_edge(n0, n2);
        b.add_edge(n1, n3); b.add_edge(n2, n3);
        return b.finalise();
    }();

    constexpr auto g2 = []() {
        graph_builder<8, 16> b;
        auto n0 = b.add_node(); auto n1 = b.add_node();
        auto n2 = b.add_node(); auto n3 = b.add_node();
        // Reversed insertion order
        b.add_edge(n2, n3); b.add_edge(n1, n3);
        b.add_edge(n0, n2); b.add_edge(n0, n1);
        return b.finalise();
    }();

    // Same topology → same token → same edge IDs
    static_assert(g1.token() == g2.token());
    static_assert(g1.edge_target(edge_id{0}) == g2.edge_target(edge_id{0}));
    static_assert(g1.edge_target(edge_id{1}) == g2.edge_target(edge_id{1}));
    static_assert(g1.edge_target(edge_id{2}) == g2.edge_target(edge_id{2}));
    static_assert(g1.edge_target(edge_id{3}) == g2.edge_target(edge_id{3}));

    // Weights built for g1 can be used with g2
    constexpr auto w1 = make_uniform_edge_map<double, 16>(g1, 1.0);
    auto wv2 = make_weighted_view(g2, w1);  // no throw — tokens match
    EXPECT_EQ(wv2.edge_count(), 4u);
}

// =========================================================================
// 10. Stale map after rebuild (token mismatch)
// =========================================================================

TEST(WeightedView, StaleMapAfterRebuild) {
    // Build graph, create weights
    auto g1 = []() {
        graph_builder<8, 16> b;
        (void)b.add_node(); (void)b.add_node(); (void)b.add_node();
        b.add_edge(node_id{0}, node_id{1});
        b.add_edge(node_id{1}, node_id{2});
        return b.finalise();
    }();

    auto old_weights = make_uniform_edge_map<double, 16>(g1, 1.0);

    // "Rebuild" — add an extra edge → different topology
    auto g2 = []() {
        graph_builder<8, 16> b;
        (void)b.add_node(); (void)b.add_node(); (void)b.add_node();
        b.add_edge(node_id{0}, node_id{1});
        b.add_edge(node_id{1}, node_id{2});
        b.add_edge(node_id{0}, node_id{2});  // extra edge
        return b.finalise();
    }();

    // old_weights has wrong size AND wrong token for g2
    EXPECT_THROW((void)make_weighted_view(g2, old_weights), std::logic_error);
}

// =========================================================================
// 11. Negative concept checks
// =========================================================================

static_assert(!weighted_graph_queryable<constexpr_graph<8, 16>>);
static_assert(!weighted_graph_queryable<symmetric_graph<8, 16>>);
static_assert(!symmetric_weighted_queryable<
    weighted_view<constexpr_graph<8, 16>, double, 16>>);
