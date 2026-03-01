// tests/graph/test_transforms.cpp — Tests for graph transform operations
//
// Tests: transpose, induced_subgraph, contract
// Coverage: constexpr proofs (static_assert), runtime verification,
// edge cases (empty, single node, self-edges, disconnected),
// round-trip properties, mapping correctness.

#include <ctdp/graph/transpose.h>
#include <ctdp/graph/subgraph.h>
#include <ctdp/graph/contract.h>
#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/graph_traits.h>
#include <ctdp/graph/property_map.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =========================================================================
// Test graph builders
// =========================================================================

// Empty graph: 0 nodes, 0 edges.
constexpr auto make_empty() {
    graph_builder<cap_from<8, 16>> b;
    return b.finalise();
}

// Single node, no edges.
constexpr auto make_single() {
    graph_builder<cap_from<8, 16>> b;
    (void)b.add_node();
    return b.finalise();
}

// Two nodes, one edge: 0→1.
constexpr auto make_edge() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    b.add_edge(n0, n1);
    return b.finalise();
}

// Diamond: 0→1, 0→2, 1→3, 2→3.
constexpr auto make_diamond() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n1, n3);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Chain: 0→1→2→3.
constexpr auto make_chain4() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

// Fan-out: 0→1, 0→2, 0→3, 0→4.
constexpr auto make_fan_out() {
    graph_builder<cap_from<8, 16>> b;
    for (int i = 0; i < 5; ++i) (void)b.add_node();
    for (int i = 1; i < 5; ++i)
        b.add_edge(node_id{0}, node_id{static_cast<std::uint16_t>(i)});
    return b.finalise();
}

// Fan-in: 1→0, 2→0, 3→0, 4→0.
constexpr auto make_fan_in() {
    graph_builder<cap_from<8, 16>> b;
    for (int i = 0; i < 5; ++i) (void)b.add_node();
    for (int i = 1; i < 5; ++i)
        b.add_edge(node_id{static_cast<std::uint16_t>(i)}, node_id{0});
    return b.finalise();
}

// Cycle: 0→1→2→0.
constexpr auto make_cycle3() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node(); auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n0);
    return b.finalise();
}

// Disconnected: {0→1}, {2→3} — two separate components.
constexpr auto make_disconnected() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n2, n3);
    return b.finalise();
}

// =========================================================================
// Helper: check edge exists in a graph
// =========================================================================
constexpr bool has_edge(auto const& g, std::uint16_t u, std::uint16_t v) {
    auto const uid = node_id{u};
    for (auto n : g.out_neighbors(uid)) {
        if (n.value == v) return true;
    }
    return false;
}

// =========================================================================
// TRANSPOSE — constexpr proofs
// =========================================================================

namespace {

constexpr auto empty_g = make_empty();
constexpr auto empty_t = transpose(empty_g);
static_assert(empty_t.node_count() == 0);
static_assert(empty_t.edge_count() == 0);

constexpr auto single_g = make_single();
constexpr auto single_t = transpose(single_g);
static_assert(single_t.node_count() == 1);
static_assert(single_t.edge_count() == 0);

constexpr auto edge_g = make_edge();
constexpr auto edge_t = transpose(edge_g);
static_assert(edge_t.node_count() == 2);
static_assert(edge_t.edge_count() == 1);
// Original: 0→1.  Transposed: 1→0.
static_assert(has_edge(edge_t, 1, 0));
static_assert(!has_edge(edge_t, 0, 1));

constexpr auto diamond_g = make_diamond();
constexpr auto diamond_t = transpose(diamond_g);
static_assert(diamond_t.node_count() == 4);
static_assert(diamond_t.edge_count() == 4);
// Original: 0→1, 0→2, 1→3, 2→3.
// Transposed: 1→0, 2→0, 3→1, 3→2.
static_assert(has_edge(diamond_t, 1, 0));
static_assert(has_edge(diamond_t, 2, 0));
static_assert(has_edge(diamond_t, 3, 1));
static_assert(has_edge(diamond_t, 3, 2));
static_assert(!has_edge(diamond_t, 0, 1));
static_assert(!has_edge(diamond_t, 0, 2));

// Double transpose = original topology.
constexpr auto diamond_tt = transpose(diamond_t);
static_assert(diamond_tt.node_count() == diamond_g.node_count());
static_assert(diamond_tt.edge_count() == diamond_g.edge_count());
static_assert(has_edge(diamond_tt, 0, 1));
static_assert(has_edge(diamond_tt, 0, 2));
static_assert(has_edge(diamond_tt, 1, 3));
static_assert(has_edge(diamond_tt, 2, 3));

// Cycle: transpose preserves edge count, reverses direction.
constexpr auto cycle_g = make_cycle3();
constexpr auto cycle_t = transpose(cycle_g);
static_assert(cycle_t.node_count() == 3);
static_assert(cycle_t.edge_count() == 3);
// Original: 0→1, 1→2, 2→0.  Transposed: 1→0, 2→1, 0→2.
static_assert(has_edge(cycle_t, 1, 0));
static_assert(has_edge(cycle_t, 2, 1));
static_assert(has_edge(cycle_t, 0, 2));

// Fan-out ↔ fan-in duality.
constexpr auto fan_out_g = make_fan_out();
constexpr auto fan_out_t = transpose(fan_out_g);
static_assert(fan_out_t.node_count() == 5);
static_assert(fan_out_t.edge_count() == 4);
// Transposed fan-out should look like fan-in.
static_assert(has_edge(fan_out_t, 1, 0));
static_assert(has_edge(fan_out_t, 2, 0));
static_assert(has_edge(fan_out_t, 3, 0));
static_assert(has_edge(fan_out_t, 4, 0));

} // anonymous namespace

// =========================================================================
// TRANSPOSE — runtime tests
// =========================================================================

TEST(TransposeTest, EmptyGraph) {
    constexpr auto g = make_empty();
    auto gt = transpose(g);
    EXPECT_EQ(gt.node_count(), 0u);
    EXPECT_EQ(gt.edge_count(), 0u);
}

TEST(TransposeTest, SingleNode) {
    constexpr auto g = make_single();
    auto gt = transpose(g);
    EXPECT_EQ(gt.node_count(), 1u);
    EXPECT_EQ(gt.edge_count(), 0u);
}

TEST(TransposeTest, DiamondReversal) {
    constexpr auto g = make_diamond();
    auto gt = transpose(g);
    EXPECT_EQ(gt.node_count(), 4u);
    EXPECT_EQ(gt.edge_count(), 4u);

    // Verify reversed edges.
    EXPECT_TRUE(has_edge(gt, 1, 0));
    EXPECT_TRUE(has_edge(gt, 2, 0));
    EXPECT_TRUE(has_edge(gt, 3, 1));
    EXPECT_TRUE(has_edge(gt, 3, 2));

    // Verify original edges are gone.
    EXPECT_FALSE(has_edge(gt, 0, 1));
    EXPECT_FALSE(has_edge(gt, 0, 2));
}

TEST(TransposeTest, DoubleTransposeIsIdentity) {
    constexpr auto g = make_diamond();
    auto gtt = transpose(transpose(g));
    EXPECT_EQ(gtt.node_count(), g.node_count());
    EXPECT_EQ(gtt.edge_count(), g.edge_count());

    // Check every edge in original exists in double-transposed.
    for (std::size_t u = 0; u < g.node_count(); ++u) {
        auto uid = node_id{static_cast<std::uint16_t>(u)};
        for (auto v : g.out_neighbors(uid)) {
            EXPECT_TRUE(has_edge(gtt, uid.value, v.value))
                << "Missing edge " << u << "→" << v.value;
        }
    }
}

TEST(TransposeTest, ChainReversal) {
    constexpr auto g = make_chain4();
    auto gt = transpose(g);
    // Original: 0→1→2→3.  Transposed: 1→0, 2→1, 3→2.
    EXPECT_TRUE(has_edge(gt, 1, 0));
    EXPECT_TRUE(has_edge(gt, 2, 1));
    EXPECT_TRUE(has_edge(gt, 3, 2));
    EXPECT_FALSE(has_edge(gt, 0, 1));
}

TEST(TransposeTest, FanOutBecomesFanIn) {
    constexpr auto g = make_fan_out();
    auto gt = transpose(g);
    // All leaf nodes should now point to node 0.
    for (std::uint16_t i = 1; i < 5; ++i) {
        EXPECT_TRUE(has_edge(gt, i, 0));
        EXPECT_FALSE(has_edge(gt, 0, i));
    }
}

TEST(TransposeTest, DisconnectedComponents) {
    constexpr auto g = make_disconnected();
    auto gt = transpose(g);
    EXPECT_EQ(gt.node_count(), 4u);
    EXPECT_EQ(gt.edge_count(), 2u);
    EXPECT_TRUE(has_edge(gt, 1, 0));
    EXPECT_TRUE(has_edge(gt, 3, 2));
}

TEST(TransposeTest, PreservesNodeAndEdgeCount) {
    constexpr auto g = make_chain4();
    auto gt = transpose(g);
    EXPECT_EQ(gt.node_count(), g.node_count());
    EXPECT_EQ(gt.edge_count(), g.edge_count());
}

// =========================================================================
// SUBGRAPH — constexpr proofs
// =========================================================================

namespace {

// Keep all nodes → same graph.
constexpr auto sub_all = induced_subgraph(diamond_g,
    [](node_id) { return true; });
static_assert(sub_all.retained_count == 4);
static_assert(sub_all.graph.node_count() == 4);
static_assert(sub_all.graph.edge_count() == 4);

// Keep no nodes → empty graph.
constexpr auto sub_none = induced_subgraph(diamond_g,
    [](node_id) { return false; });
static_assert(sub_none.retained_count == 0);
static_assert(sub_none.graph.node_count() == 0);
static_assert(sub_none.graph.edge_count() == 0);

// Diamond: keep nodes 0, 1, 3 (exclude node 2).
// Retained edges: 0→1, 1→3 (0→2 excluded, 2→3 excluded).
constexpr auto sub_no2 = induced_subgraph(diamond_g,
    [](node_id n) { return n.value != 2; });
static_assert(sub_no2.retained_count == 3);
static_assert(sub_no2.graph.node_count() == 3);
static_assert(sub_no2.graph.edge_count() == 2);

// Forward map check: node 0→0, node 1→1, node 2→invalid, node 3→2.
static_assert(sub_no2.forward_map[0].value == 0);
static_assert(sub_no2.forward_map[1].value == 1);
static_assert(sub_no2.forward_map[2] == invalid_node);
static_assert(sub_no2.forward_map[3].value == 2);

// Inverse map check: new 0→old 0, new 1→old 1, new 2→old 3.
static_assert(sub_no2.inverse_map[0].value == 0);
static_assert(sub_no2.inverse_map[1].value == 1);
static_assert(sub_no2.inverse_map[2].value == 3);

// Edges in subgraph (renumbered): 0→1, 1→2.
static_assert(has_edge(sub_no2.graph, 0, 1));
static_assert(has_edge(sub_no2.graph, 1, 2));
static_assert(!has_edge(sub_no2.graph, 0, 2)); // no direct 0→3 in original

// Empty input.
constexpr auto sub_empty = induced_subgraph(empty_g,
    [](node_id) { return true; });
static_assert(sub_empty.retained_count == 0);

// Single node, kept.
constexpr auto sub_single = induced_subgraph(single_g,
    [](node_id) { return true; });
static_assert(sub_single.retained_count == 1);
static_assert(sub_single.graph.node_count() == 1);
static_assert(sub_single.graph.edge_count() == 0);

} // anonymous namespace

// =========================================================================
// SUBGRAPH — runtime tests
// =========================================================================

TEST(SubgraphTest, KeepAll) {
    constexpr auto g = make_diamond();
    auto sr = induced_subgraph(g, [](node_id) { return true; });
    EXPECT_EQ(sr.retained_count, 4u);
    EXPECT_EQ(sr.graph.node_count(), 4u);
    EXPECT_EQ(sr.graph.edge_count(), 4u);
}

TEST(SubgraphTest, KeepNone) {
    constexpr auto g = make_diamond();
    auto sr = induced_subgraph(g, [](node_id) { return false; });
    EXPECT_EQ(sr.retained_count, 0u);
    EXPECT_EQ(sr.graph.node_count(), 0u);
    EXPECT_EQ(sr.graph.edge_count(), 0u);
}

TEST(SubgraphTest, DiamondExcludeMiddle) {
    constexpr auto g = make_diamond();
    // Exclude node 2.
    auto sr = induced_subgraph(g, [](node_id n) { return n.value != 2; });
    EXPECT_EQ(sr.retained_count, 3u);
    EXPECT_EQ(sr.graph.edge_count(), 2u);

    // Edges: 0→1, 1→2(was 3).
    EXPECT_TRUE(has_edge(sr.graph, 0, 1));
    EXPECT_TRUE(has_edge(sr.graph, 1, 2));
}

TEST(SubgraphTest, ChainKeepFirstTwo) {
    constexpr auto g = make_chain4();
    // Keep nodes 0, 1 only.
    auto sr = induced_subgraph(g, [](node_id n) { return n.value < 2; });
    EXPECT_EQ(sr.retained_count, 2u);
    EXPECT_EQ(sr.graph.node_count(), 2u);
    EXPECT_EQ(sr.graph.edge_count(), 1u); // 0→1
    EXPECT_TRUE(has_edge(sr.graph, 0, 1));
}

TEST(SubgraphTest, ChainKeepOddNodes) {
    constexpr auto g = make_chain4();
    // Keep nodes 1, 3.  Edge 1→2→3 — but node 2 excluded, so no edges.
    auto sr = induced_subgraph(g, [](node_id n) { return n.value % 2 == 1; });
    EXPECT_EQ(sr.retained_count, 2u);
    EXPECT_EQ(sr.graph.edge_count(), 0u);
}

TEST(SubgraphTest, DisconnectedKeepOneComponent) {
    constexpr auto g = make_disconnected(); // {0→1}, {2→3}
    // Keep component 2 only: nodes 2, 3.
    auto sr = induced_subgraph(g, [](node_id n) { return n.value >= 2; });
    EXPECT_EQ(sr.retained_count, 2u);
    EXPECT_EQ(sr.graph.edge_count(), 1u);
    // Renumbered: old 2→new 0, old 3→new 1.
    EXPECT_TRUE(has_edge(sr.graph, 0, 1));
}

TEST(SubgraphTest, ForwardAndInverseMapsConsistent) {
    constexpr auto g = make_fan_out(); // 0→1, 0→2, 0→3, 0→4
    // Keep 0, 2, 4 (every other node).
    auto sr = induced_subgraph(g,
        [](node_id n) { return n.value == 0 || n.value == 2 || n.value == 4; });

    EXPECT_EQ(sr.retained_count, 3u);

    // Check forward map.
    EXPECT_EQ(sr.forward_map[0].value, 0);
    EXPECT_EQ(sr.forward_map[1], invalid_node);
    EXPECT_EQ(sr.forward_map[2].value, 1);
    EXPECT_EQ(sr.forward_map[3], invalid_node);
    EXPECT_EQ(sr.forward_map[4].value, 2);

    // Check inverse map.
    EXPECT_EQ(sr.inverse_map[0].value, 0);
    EXPECT_EQ(sr.inverse_map[1].value, 2);
    EXPECT_EQ(sr.inverse_map[2].value, 4);

    // Round-trip consistency: inverse(forward(i)) == i for retained nodes.
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        if (sr.forward_map[i] != invalid_node) {
            auto new_id = sr.forward_map[i];
            EXPECT_EQ(sr.inverse_map[to_index(new_id)].value,
                      static_cast<std::uint16_t>(i));
        }
    }
}

TEST(SubgraphTest, EmptyInput) {
    constexpr auto g = make_empty();
    auto sr = induced_subgraph(g, [](node_id) { return true; });
    EXPECT_EQ(sr.retained_count, 0u);
}

// =========================================================================
// CONTRACT — constexpr proofs
// =========================================================================

namespace {

// Chain 0→1→2→3, groups: {0,1}=0, {2,3}=1.
constexpr auto chain_contract() {
    constexpr auto g = make_chain4();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0;
    groups[1] = 0;
    groups[2] = 1;
    groups[3] = 1;
    return contract(g, groups, std::size_t{2});
}
constexpr auto cc_chain = chain_contract();
static_assert(cc_chain.graph.node_count() == 2);
static_assert(cc_chain.graph.edge_count() == 1); // group0→group1
static_assert(cc_chain.group_count == 2);
static_assert(has_edge(cc_chain.graph, 0, 1));
static_assert(!has_edge(cc_chain.graph, 1, 0));

// Diamond 0→1, 0→2, 1→3, 2→3.
// Groups: {0}=0, {1,2}=1, {3}=2.
constexpr auto diamond_contract() {
    constexpr auto g = make_diamond();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0;
    groups[1] = 1;
    groups[2] = 1;
    groups[3] = 2;
    return contract(g, groups, std::size_t{3});
}
constexpr auto cc_diamond = diamond_contract();
static_assert(cc_diamond.graph.node_count() == 3);
static_assert(cc_diamond.graph.edge_count() == 2); // 0→1, 1→2
static_assert(has_edge(cc_diamond.graph, 0, 1));
static_assert(has_edge(cc_diamond.graph, 1, 2));

// All nodes in one group → zero edges.
constexpr auto all_one_group() {
    constexpr auto g = make_diamond();
    property_map<std::uint16_t, 8> groups(4, 0);
    // All in group 0.
    return contract(g, groups, std::size_t{1});
}
constexpr auto cc_one = all_one_group();
static_assert(cc_one.graph.node_count() == 1);
static_assert(cc_one.graph.edge_count() == 0);

// Each node its own group → same as original.
constexpr auto identity_contract() {
    constexpr auto g = make_diamond();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0;
    groups[1] = 1;
    groups[2] = 2;
    groups[3] = 3;
    return contract(g, groups, std::size_t{4});
}
constexpr auto cc_id = identity_contract();
static_assert(cc_id.graph.node_count() == 4);
static_assert(cc_id.graph.edge_count() == 4);

} // anonymous namespace

// =========================================================================
// CONTRACT — runtime tests
// =========================================================================

TEST(ContractTest, ChainTwoGroups) {
    constexpr auto g = make_chain4();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 0;
    groups[2] = 1; groups[3] = 1;
    auto cr = contract(g, groups, std::size_t{2});
    EXPECT_EQ(cr.graph.node_count(), 2u);
    EXPECT_EQ(cr.graph.edge_count(), 1u);
    EXPECT_TRUE(has_edge(cr.graph, 0, 1));
}

TEST(ContractTest, DiamondThreeGroups) {
    constexpr auto g = make_diamond();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 1;
    groups[2] = 1; groups[3] = 2;
    auto cr = contract(g, groups, std::size_t{3});
    EXPECT_EQ(cr.graph.node_count(), 3u);
    EXPECT_EQ(cr.graph.edge_count(), 2u);
    EXPECT_TRUE(has_edge(cr.graph, 0, 1));
    EXPECT_TRUE(has_edge(cr.graph, 1, 2));
}

TEST(ContractTest, AllOneGroup) {
    constexpr auto g = make_diamond();
    property_map<std::uint16_t, 8> groups(4, 0);
    auto cr = contract(g, groups, std::size_t{1});
    EXPECT_EQ(cr.graph.node_count(), 1u);
    EXPECT_EQ(cr.graph.edge_count(), 0u);
}

TEST(ContractTest, IdentityContraction) {
    constexpr auto g = make_diamond();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 1;
    groups[2] = 2; groups[3] = 3;
    auto cr = contract(g, groups, std::size_t{4});
    EXPECT_EQ(cr.graph.node_count(), g.node_count());
    EXPECT_EQ(cr.graph.edge_count(), g.edge_count());
}

TEST(ContractTest, DisconnectedMerge) {
    constexpr auto g = make_disconnected(); // {0→1}, {2→3}
    // Merge across components: {0,2}=0, {1,3}=1.
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 1;
    groups[2] = 0; groups[3] = 1;
    auto cr = contract(g, groups, std::size_t{2});
    EXPECT_EQ(cr.graph.node_count(), 2u);
    // Both original edges (0→1 and 2→3) become group0→group1 (deduplicated).
    EXPECT_EQ(cr.graph.edge_count(), 1u);
    EXPECT_TRUE(has_edge(cr.graph, 0, 1));
}

TEST(ContractTest, CycleMerge) {
    constexpr auto g = make_cycle3(); // 0→1→2→0
    // Merge {0,1}=0, {2}=1.
    property_map<std::uint16_t, 8> groups(3, 0);
    groups[0] = 0; groups[1] = 0; groups[2] = 1;
    auto cr = contract(g, groups, std::size_t{2});
    EXPECT_EQ(cr.graph.node_count(), 2u);
    // Edges: 0→1 (from 1→2 cross-group), 1→0 (from 2→0 cross-group).
    EXPECT_EQ(cr.graph.edge_count(), 2u);
    EXPECT_TRUE(has_edge(cr.graph, 0, 1));
    EXPECT_TRUE(has_edge(cr.graph, 1, 0));
}

TEST(ContractTest, EmptyGraph) {
    constexpr auto g = make_empty();
    property_map<std::uint16_t, 8> groups(0, 0);
    auto cr = contract(g, groups, std::size_t{0});
    EXPECT_EQ(cr.graph.node_count(), 0u);
    EXPECT_EQ(cr.graph.edge_count(), 0u);
}

TEST(ContractTest, SingleNodeOneGroup) {
    constexpr auto g = make_single();
    property_map<std::uint16_t, 8> groups(1, 0);
    auto cr = contract(g, groups, std::size_t{1});
    EXPECT_EQ(cr.graph.node_count(), 1u);
    EXPECT_EQ(cr.graph.edge_count(), 0u);
}

// =========================================================================
// CROSS-TRANSFORM — composition tests
// =========================================================================

TEST(CrossTransformTest, TransposeOfSubgraph) {
    constexpr auto g = make_diamond();
    // Subgraph: keep {0, 1, 3}.  Edges: 0→1, 1→2(was 3).
    auto sr = induced_subgraph(g, [](node_id n) { return n.value != 2; });
    auto gt = transpose(sr.graph);
    EXPECT_EQ(gt.node_count(), 3u);
    EXPECT_EQ(gt.edge_count(), 2u);
    EXPECT_TRUE(has_edge(gt, 1, 0));
    EXPECT_TRUE(has_edge(gt, 2, 1));
}

TEST(CrossTransformTest, SubgraphOfTranspose) {
    constexpr auto g = make_diamond();
    auto gt = transpose(g);
    // Transpose has: 1→0, 2→0, 3→1, 3→2.
    // Subgraph: keep {0, 1, 3}.  Edges: 1→0, 3→1.
    auto sr = induced_subgraph(gt, [](node_id n) { return n.value != 2; });
    EXPECT_EQ(sr.retained_count, 3u);
    EXPECT_EQ(sr.graph.edge_count(), 2u);
}

TEST(CrossTransformTest, ContractThenTranspose) {
    constexpr auto g = make_chain4();
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 0;
    groups[2] = 1; groups[3] = 1;
    auto cr = contract(g, groups, std::size_t{2});
    auto gt = transpose(cr.graph);
    EXPECT_EQ(gt.node_count(), 2u);
    EXPECT_EQ(gt.edge_count(), 1u);
    EXPECT_TRUE(has_edge(gt, 1, 0)); // reversed
}

TEST(CrossTransformTest, TransposeThenContract) {
    constexpr auto g = make_chain4(); // 0→1→2→3
    auto gt = transpose(g);           // 1→0, 2→1, 3→2
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 0;
    groups[2] = 1; groups[3] = 1;
    auto cr = contract(gt, groups, std::size_t{2});
    EXPECT_EQ(cr.graph.node_count(), 2u);
    // In transposed chain, 2→1 crosses groups: group1→group0.
    EXPECT_EQ(cr.graph.edge_count(), 1u);
    EXPECT_TRUE(has_edge(cr.graph, 1, 0));
}

// =========================================================================
// EDGE CASES — capacity and large predicate
// =========================================================================

TEST(TransformEdgeCases, ExplicitCapOverride) {
    constexpr auto g = make_edge(); // 2 nodes, 1 edge
    // Use smaller output cap explicitly.
    auto gt = transpose<cap_from<4, 4>>(g);
    EXPECT_EQ(gt.node_count(), 2u);
    EXPECT_EQ(gt.edge_count(), 1u);
}

TEST(TransformEdgeCases, SubgraphSingleRetained) {
    constexpr auto g = make_fan_out(); // 0→1, 0→2, 0→3, 0→4
    // Keep only node 0: no edges remain.
    auto sr = induced_subgraph(g, [](node_id n) { return n.value == 0; });
    EXPECT_EQ(sr.retained_count, 1u);
    EXPECT_EQ(sr.graph.node_count(), 1u);
    EXPECT_EQ(sr.graph.edge_count(), 0u);
}

TEST(TransformEdgeCases, ContractDedupCrossGroupEdges) {
    // Fan-out: 0→1, 0→2, 0→3, 0→4.
    // Groups: {0}=0, {1,2,3,4}=1.
    // All 4 edges become group0→group1 but should deduplicate to 1.
    constexpr auto g = make_fan_out();
    property_map<std::uint16_t, 8> groups(5, 0);
    groups[0] = 0;
    for (std::size_t i = 1; i < 5; ++i) groups[i] = 1;
    auto cr = contract(g, groups, std::size_t{2});
    EXPECT_EQ(cr.graph.edge_count(), 1u);
}
