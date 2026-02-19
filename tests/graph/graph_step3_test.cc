// graph/test/graph_step3_test.cc - Tests for topological_sort, scc,
//                                  connected_components
// Part of the compile-time DP library (C++20)

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "topological_sort.h"
#include "scc.h"
#include "connected_components.h"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// Test graph factories
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

// Triangle DAG: 0→1, 0→2, 1→2
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

// Disconnected DAG: 0→1, 2→3 (two components)
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

// 3-cycle: 0→1→2→0
constexpr auto make_cycle3() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n0);
    return b.finalise();
}

// Cycle with tail: 0→1→2→1 (tail 0, cycle {1,2})
constexpr auto make_cycle_with_tail() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n1);
    return b.finalise();
}

// Two separate cycles: {0→1→0}, {2→3→4→2}
constexpr auto make_two_cycles() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    auto n4 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n0);
    b.add_edge(n2, n3);
    b.add_edge(n3, n4);
    b.add_edge(n4, n2);
    return b.finalise();
}

// Self-loop: node 0 has edge 0→0, node 1 is isolated
// (builder removes self-edges, so use cycle instead)
// Actually: use a 1-node self-loop via make_dirty pattern
// Builder strips self-edges, so the smallest cycle is 2 nodes.
// Let's make: 0→1→0 (2-cycle)
constexpr auto make_two_cycle() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n0);
    return b.finalise();
}

// Complex DAG: wide then narrow
//   0→1, 0→2, 0→3, 1→4, 2→4, 3→4, 4→5
constexpr auto make_wide_dag() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    auto n4 = b.add_node();
    auto n5 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n0, n2);
    b.add_edge(n0, n3);
    b.add_edge(n1, n4);
    b.add_edge(n2, n4);
    b.add_edge(n3, n4);
    b.add_edge(n4, n5);
    return b.finalise();
}

// Mixed graph: has both an SCC and acyclic parts
// Structure: 0→1→2→0 (cycle), 3→0 (feeds into cycle), 2→4 (exits cycle)
constexpr auto make_mixed_scc() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    auto n4 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n0);  // cycle: 0→1→2→0
    b.add_edge(n3, n0);  // 3 feeds into cycle
    b.add_edge(n2, n4);  // cycle feeds into 4
    return b.finalise();
}

// Four isolated nodes: 0, 1, 2, 3 — no edges
constexpr auto make_isolated4() {
    graph_builder<8, 16> b;
    [[maybe_unused]] auto _ = b.add_nodes(4);
    return b.finalise();
}

// Disconnected: cycle + DAG component
// Component A: 0→1→0 (cycle)
// Component B: 2→3→4 (chain)
constexpr auto make_disconnected_mixed() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    auto n4 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n0);
    b.add_edge(n2, n3);
    b.add_edge(n3, n4);
    return b.finalise();
}

// Two independent sources converging: 0→2, 1→2, 2→3
constexpr auto make_converge() {
    graph_builder<8, 16> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    auto n2 = b.add_node();
    auto n3 = b.add_node();
    b.add_edge(n0, n2);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

// =============================================================================
// Helper: validate topological order (for every edge u→v, u before v)
// =============================================================================
template<std::size_t MaxV, typename G>
constexpr bool is_valid_topo_order(topo_result<MaxV> const& r, G const& g) {
    if (!r.is_dag) return false;
    auto const V = g.node_count();
    if (r.order.size() != V) return false;

    // Build position array.
    std::array<std::size_t, MaxV> pos{};
    for (std::size_t i = 0; i < V; ++i) {
        pos[r.order[i].value] = i;
    }

    // Check every edge.
    for (std::size_t u = 0; u < V; ++u) {
        for (auto v : g.out_neighbors(node_id{static_cast<std::uint16_t>(u)})) {
            if (pos[u] >= pos[v.value]) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Compile-time verification: topological_sort
// =============================================================================

// --- Empty graph ---
constexpr auto topo_empty = topological_sort(make_empty());
static_assert(topo_empty.is_dag);
static_assert(topo_empty.order.size() == 0);

// --- Singleton ---
constexpr auto topo_single = topological_sort(make_singleton());
static_assert(topo_single.is_dag);
static_assert(topo_single.order.size() == 1);
static_assert(topo_single.order[0] == node_id{0});

// --- Chain: 0→1→2→3 ---
constexpr auto topo_chain = topological_sort(make_chain());
static_assert(topo_chain.is_dag);
static_assert(topo_chain.order.size() == 4);
static_assert(topo_chain.order[0] == node_id{0});
static_assert(topo_chain.order[1] == node_id{1});
static_assert(topo_chain.order[2] == node_id{2});
static_assert(topo_chain.order[3] == node_id{3});
static_assert(is_valid_topo_order(topo_chain, make_chain()));

// --- Diamond: deterministic tie-break {0,1,2,3} ---
constexpr auto topo_diamond = topological_sort(make_diamond());
static_assert(topo_diamond.is_dag);
static_assert(topo_diamond.order.size() == 4);
static_assert(topo_diamond.order[0] == node_id{0});
static_assert(topo_diamond.order[1] == node_id{1});
static_assert(topo_diamond.order[2] == node_id{2});
static_assert(topo_diamond.order[3] == node_id{3});
static_assert(is_valid_topo_order(topo_diamond, make_diamond()));

// --- Triangle: 0→1, 0→2, 1→2 → order {0,1,2} ---
constexpr auto topo_triangle = topological_sort(make_triangle());
static_assert(topo_triangle.is_dag);
static_assert(topo_triangle.order.size() == 3);
static_assert(topo_triangle.order[0] == node_id{0});
static_assert(topo_triangle.order[1] == node_id{1});
static_assert(topo_triangle.order[2] == node_id{2});
static_assert(is_valid_topo_order(topo_triangle, make_triangle()));

// --- Star: 0→{1,2,3,4} → order {0,1,2,3,4} (smallest-first tie-break) ---
constexpr auto topo_star = topological_sort(make_star());
static_assert(topo_star.is_dag);
static_assert(topo_star.order.size() == 5);
static_assert(topo_star.order[0] == node_id{0});
static_assert(topo_star.order[1] == node_id{1});
static_assert(topo_star.order[2] == node_id{2});
static_assert(topo_star.order[3] == node_id{3});
static_assert(topo_star.order[4] == node_id{4});
static_assert(is_valid_topo_order(topo_star, make_star()));

// --- Disconnected DAG: {0,1,2,3} smallest-first ---
constexpr auto topo_disc = topological_sort(make_disconnected());
static_assert(topo_disc.is_dag);
static_assert(topo_disc.order.size() == 4);
static_assert(topo_disc.order[0] == node_id{0});
static_assert(topo_disc.order[1] == node_id{1});
static_assert(topo_disc.order[2] == node_id{2});
static_assert(topo_disc.order[3] == node_id{3});
static_assert(is_valid_topo_order(topo_disc, make_disconnected()));

// --- Cycle detection: 3-cycle ---
constexpr auto topo_cycle3 = topological_sort(make_cycle3());
static_assert(!topo_cycle3.is_dag);

// --- Cycle detection: 2-cycle ---
constexpr auto topo_two_cycle = topological_sort(make_two_cycle());
static_assert(!topo_two_cycle.is_dag);

// --- Cycle with tail: not a DAG ---
constexpr auto topo_tail = topological_sort(make_cycle_with_tail());
static_assert(!topo_tail.is_dag);

// --- Wide DAG: 0→{1,2,3}→4→5 ---
constexpr auto topo_wide = topological_sort(make_wide_dag());
static_assert(topo_wide.is_dag);
static_assert(topo_wide.order.size() == 6);
static_assert(topo_wide.order[0] == node_id{0});
// After 0 removed: 1,2,3 all have in-degree 0 → smallest first
static_assert(topo_wide.order[1] == node_id{1});
static_assert(topo_wide.order[2] == node_id{2});
static_assert(topo_wide.order[3] == node_id{3});
static_assert(topo_wide.order[4] == node_id{4});
static_assert(topo_wide.order[5] == node_id{5});
static_assert(is_valid_topo_order(topo_wide, make_wide_dag()));

// --- Converge: 0→2, 1→2, 2→3 → order {0,1,2,3} ---
constexpr auto topo_conv = topological_sort(make_converge());
static_assert(topo_conv.is_dag);
static_assert(topo_conv.order.size() == 4);
static_assert(topo_conv.order[0] == node_id{0});
static_assert(topo_conv.order[1] == node_id{1});
static_assert(topo_conv.order[2] == node_id{2});
static_assert(topo_conv.order[3] == node_id{3});
static_assert(is_valid_topo_order(topo_conv, make_converge()));

// --- Isolated nodes: all in-degree 0, smallest-first ---
constexpr auto topo_iso = topological_sort(make_isolated4());
static_assert(topo_iso.is_dag);
static_assert(topo_iso.order.size() == 4);
static_assert(topo_iso.order[0] == node_id{0});
static_assert(topo_iso.order[1] == node_id{1});
static_assert(topo_iso.order[2] == node_id{2});
static_assert(topo_iso.order[3] == node_id{3});

// =============================================================================
// Compile-time verification: SCC
// =============================================================================

// --- Empty graph ---
constexpr auto scc_empty = scc(make_empty());
static_assert(scc_empty.component_count == 0);

// --- Singleton: 1 SCC ---
constexpr auto scc_single = scc(make_singleton());
static_assert(scc_single.component_count == 1);
static_assert(scc_single.component_of[0] == 0);

// --- Chain (DAG): each node is its own SCC ---
constexpr auto scc_chain = scc(make_chain());
static_assert(scc_chain.component_count == 4);
// Each node should be in a different component.
static_assert(scc_chain.component_of[0] != scc_chain.component_of[1]);
static_assert(scc_chain.component_of[1] != scc_chain.component_of[2]);
static_assert(scc_chain.component_of[2] != scc_chain.component_of[3]);

// --- 3-cycle: all in one SCC ---
constexpr auto scc_cycle3 = scc(make_cycle3());
static_assert(scc_cycle3.component_count == 1);
static_assert(scc_cycle3.component_of[0] == scc_cycle3.component_of[1]);
static_assert(scc_cycle3.component_of[1] == scc_cycle3.component_of[2]);

// --- 2-cycle: all in one SCC ---
constexpr auto scc_two_cycle = scc(make_two_cycle());
static_assert(scc_two_cycle.component_count == 1);
static_assert(scc_two_cycle.component_of[0] == scc_two_cycle.component_of[1]);

// --- Diamond (DAG): 4 SCCs ---
constexpr auto scc_diamond = scc(make_diamond());
static_assert(scc_diamond.component_count == 4);
static_assert(scc_diamond.component_of[0] != scc_diamond.component_of[1]);
static_assert(scc_diamond.component_of[0] != scc_diamond.component_of[2]);
static_assert(scc_diamond.component_of[0] != scc_diamond.component_of[3]);

// --- Two cycles: 2 SCCs ---
constexpr auto scc_two = scc(make_two_cycles());
static_assert(scc_two.component_count == 2);
static_assert(scc_two.component_of[0] == scc_two.component_of[1]);
static_assert(scc_two.component_of[2] == scc_two.component_of[3]);
static_assert(scc_two.component_of[3] == scc_two.component_of[4]);
static_assert(scc_two.component_of[0] != scc_two.component_of[2]);

// --- Cycle with tail: {0} is its own SCC, {1,2} is another ---
constexpr auto scc_tail = scc(make_cycle_with_tail());
static_assert(scc_tail.component_count == 2);
static_assert(scc_tail.component_of[1] == scc_tail.component_of[2]);
static_assert(scc_tail.component_of[0] != scc_tail.component_of[1]);

// --- Mixed SCC: {0,1,2} cycle, {3} and {4} separate ---
constexpr auto scc_mixed = scc(make_mixed_scc());
static_assert(scc_mixed.component_count == 3);
static_assert(scc_mixed.component_of[0] == scc_mixed.component_of[1]);
static_assert(scc_mixed.component_of[1] == scc_mixed.component_of[2]);
static_assert(scc_mixed.component_of[0] != scc_mixed.component_of[3]);
static_assert(scc_mixed.component_of[0] != scc_mixed.component_of[4]);
static_assert(scc_mixed.component_of[3] != scc_mixed.component_of[4]);

// --- Isolated nodes: each is its own SCC ---
constexpr auto scc_iso = scc(make_isolated4());
static_assert(scc_iso.component_count == 4);
static_assert(scc_iso.component_of[0] != scc_iso.component_of[1]);
static_assert(scc_iso.component_of[1] != scc_iso.component_of[2]);
static_assert(scc_iso.component_of[2] != scc_iso.component_of[3]);

// --- Disconnected mixed: {0,1} cycle SCC, {2},{3},{4} separate SCCs ---
constexpr auto scc_disc_mix = scc(make_disconnected_mixed());
static_assert(scc_disc_mix.component_count == 4);
static_assert(scc_disc_mix.component_of[0] == scc_disc_mix.component_of[1]);
static_assert(scc_disc_mix.component_of[2] != scc_disc_mix.component_of[3]);
static_assert(scc_disc_mix.component_of[3] != scc_disc_mix.component_of[4]);

// =============================================================================
// Compile-time verification: connected_components
// =============================================================================

// --- Empty graph ---
constexpr auto cc_empty = connected_components(make_empty());
static_assert(cc_empty.component_count == 0);

// --- Singleton: 1 component ---
constexpr auto cc_single = connected_components(make_singleton());
static_assert(cc_single.component_count == 1);
static_assert(cc_single.component_of[0] == 0);

// --- Chain: 1 component (all connected) ---
constexpr auto cc_chain = connected_components(make_chain());
static_assert(cc_chain.component_count == 1);
static_assert(cc_chain.component_of[0] == cc_chain.component_of[1]);
static_assert(cc_chain.component_of[1] == cc_chain.component_of[2]);
static_assert(cc_chain.component_of[2] == cc_chain.component_of[3]);

// --- Disconnected: 2 components ---
constexpr auto cc_disc = connected_components(make_disconnected());
static_assert(cc_disc.component_count == 2);
static_assert(cc_disc.component_of[0] == cc_disc.component_of[1]);
static_assert(cc_disc.component_of[2] == cc_disc.component_of[3]);
static_assert(cc_disc.component_of[0] != cc_disc.component_of[2]);

// --- Dense renumbering: first component is 0, second is 1 ---
static_assert(cc_disc.component_of[0] == 0);
static_assert(cc_disc.component_of[2] == 1);

// --- Star: 1 component ---
constexpr auto cc_star = connected_components(make_star());
static_assert(cc_star.component_count == 1);

// --- Isolated nodes: each is its own component ---
constexpr auto cc_iso = connected_components(make_isolated4());
static_assert(cc_iso.component_count == 4);
static_assert(cc_iso.component_of[0] == 0);
static_assert(cc_iso.component_of[1] == 1);
static_assert(cc_iso.component_of[2] == 2);
static_assert(cc_iso.component_of[3] == 3);

// --- Cycle: direction ignored → 1 component ---
constexpr auto cc_cycle = connected_components(make_cycle3());
static_assert(cc_cycle.component_count == 1);
static_assert(cc_cycle.component_of[0] == cc_cycle.component_of[1]);
static_assert(cc_cycle.component_of[1] == cc_cycle.component_of[2]);

// --- Two cycles: 2 components ---
constexpr auto cc_two = connected_components(make_two_cycles());
static_assert(cc_two.component_count == 2);
static_assert(cc_two.component_of[0] == cc_two.component_of[1]);
static_assert(cc_two.component_of[2] == cc_two.component_of[3]);
static_assert(cc_two.component_of[3] == cc_two.component_of[4]);
static_assert(cc_two.component_of[0] != cc_two.component_of[2]);

// --- Mixed SCC: all 5 nodes weakly connected → 1 component ---
constexpr auto cc_mixed = connected_components(make_mixed_scc());
static_assert(cc_mixed.component_count == 1);

// --- Disconnected mixed: 2 weakly connected components ---
constexpr auto cc_disc_mix = connected_components(make_disconnected_mixed());
static_assert(cc_disc_mix.component_count == 2);
static_assert(cc_disc_mix.component_of[0] == cc_disc_mix.component_of[1]);
static_assert(cc_disc_mix.component_of[2] == cc_disc_mix.component_of[3]);
static_assert(cc_disc_mix.component_of[3] == cc_disc_mix.component_of[4]);
static_assert(cc_disc_mix.component_of[0] != cc_disc_mix.component_of[2]);
// Dense renumbering
static_assert(cc_disc_mix.component_of[0] == 0);
static_assert(cc_disc_mix.component_of[2] == 1);

// --- Converge DAG: 1 weakly connected component ---
constexpr auto cc_conv = connected_components(make_converge());
static_assert(cc_conv.component_count == 1);

// --- Diamond: 1 weakly connected component ---
constexpr auto cc_diamond = connected_components(make_diamond());
static_assert(cc_diamond.component_count == 1);

// =============================================================================
// Runtime tests: topological_sort
// =============================================================================

class TopoSort : public ::testing::Test {};

TEST_F(TopoSort, EmptyGraph) {
    constexpr auto g = make_empty();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    EXPECT_EQ(r.order.size(), 0u);
}

TEST_F(TopoSort, Singleton) {
    constexpr auto g = make_singleton();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 1u);
    EXPECT_EQ(r.order[0], node_id{0});
}

TEST_F(TopoSort, ChainOrdering) {
    constexpr auto g = make_chain();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 4u);
    EXPECT_EQ(r.order[0], node_id{0});
    EXPECT_EQ(r.order[1], node_id{1});
    EXPECT_EQ(r.order[2], node_id{2});
    EXPECT_EQ(r.order[3], node_id{3});
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, DiamondDeterministic) {
    constexpr auto g = make_diamond();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 4u);
    // Deterministic: after removing 0, both 1 and 2 ready → pick 1 first
    EXPECT_EQ(r.order[0], node_id{0});
    EXPECT_EQ(r.order[1], node_id{1});
    EXPECT_EQ(r.order[2], node_id{2});
    EXPECT_EQ(r.order[3], node_id{3});
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, TriangleOrdering) {
    constexpr auto g = make_triangle();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 3u);
    EXPECT_EQ(r.order[0], node_id{0});
    EXPECT_EQ(r.order[1], node_id{1});
    EXPECT_EQ(r.order[2], node_id{2});
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, StarSmallestFirst) {
    constexpr auto g = make_star();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 5u);
    EXPECT_EQ(r.order[0], node_id{0});
    // After 0, nodes 1-4 all ready → smallest first
    EXPECT_EQ(r.order[1], node_id{1});
    EXPECT_EQ(r.order[2], node_id{2});
    EXPECT_EQ(r.order[3], node_id{3});
    EXPECT_EQ(r.order[4], node_id{4});
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, CycleDetection3) {
    constexpr auto g = make_cycle3();
    constexpr auto r = topological_sort(g);
    EXPECT_FALSE(r.is_dag);
}

TEST_F(TopoSort, CycleDetection2) {
    constexpr auto g = make_two_cycle();
    constexpr auto r = topological_sort(g);
    EXPECT_FALSE(r.is_dag);
}

TEST_F(TopoSort, CycleWithTail) {
    constexpr auto g = make_cycle_with_tail();
    constexpr auto r = topological_sort(g);
    EXPECT_FALSE(r.is_dag);
}

TEST_F(TopoSort, DisconnectedDAG) {
    constexpr auto g = make_disconnected();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 4u);
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, WideDag) {
    constexpr auto g = make_wide_dag();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 6u);
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, ConvergeDAG) {
    constexpr auto g = make_converge();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 4u);
    EXPECT_TRUE(is_valid_topo_order(r, g));
}

TEST_F(TopoSort, IsolatedNodes) {
    constexpr auto g = make_isolated4();
    constexpr auto r = topological_sort(g);
    EXPECT_TRUE(r.is_dag);
    ASSERT_EQ(r.order.size(), 4u);
    // All have in-degree 0 → smallest first
    EXPECT_EQ(r.order[0], node_id{0});
    EXPECT_EQ(r.order[1], node_id{1});
    EXPECT_EQ(r.order[2], node_id{2});
    EXPECT_EQ(r.order[3], node_id{3});
}

// =============================================================================
// Runtime tests: SCC
// =============================================================================

class SCCTest : public ::testing::Test {};

TEST_F(SCCTest, EmptyGraph) {
    constexpr auto g = make_empty();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 0u);
}

TEST_F(SCCTest, Singleton) {
    constexpr auto g = make_singleton();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 1u);
    EXPECT_EQ(r.component_of[0], 0u);
}

TEST_F(SCCTest, ChainAllSeparate) {
    constexpr auto g = make_chain();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 4u);
    // All different
    EXPECT_NE(r.component_of[0], r.component_of[1]);
    EXPECT_NE(r.component_of[1], r.component_of[2]);
    EXPECT_NE(r.component_of[2], r.component_of[3]);
}

TEST_F(SCCTest, Cycle3OneSCC) {
    constexpr auto g = make_cycle3();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 1u);
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    EXPECT_EQ(r.component_of[1], r.component_of[2]);
}

TEST_F(SCCTest, TwoCycleSCC) {
    constexpr auto g = make_two_cycle();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 1u);
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
}

TEST_F(SCCTest, DiamondFourSCCs) {
    constexpr auto g = make_diamond();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 4u);
}

TEST_F(SCCTest, TwoCyclesTwoSCCs) {
    constexpr auto g = make_two_cycles();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 2u);
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    EXPECT_EQ(r.component_of[2], r.component_of[3]);
    EXPECT_EQ(r.component_of[3], r.component_of[4]);
    EXPECT_NE(r.component_of[0], r.component_of[2]);
}

TEST_F(SCCTest, CycleWithTail) {
    constexpr auto g = make_cycle_with_tail();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 2u);
    EXPECT_EQ(r.component_of[1], r.component_of[2]);
    EXPECT_NE(r.component_of[0], r.component_of[1]);
}

TEST_F(SCCTest, MixedSCC) {
    constexpr auto g = make_mixed_scc();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 3u);
    // {0,1,2} in same SCC
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    EXPECT_EQ(r.component_of[1], r.component_of[2]);
    // 3 and 4 separate
    EXPECT_NE(r.component_of[0], r.component_of[3]);
    EXPECT_NE(r.component_of[0], r.component_of[4]);
    EXPECT_NE(r.component_of[3], r.component_of[4]);
}

TEST_F(SCCTest, IsolatedNodes) {
    constexpr auto g = make_isolated4();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 4u);
    EXPECT_NE(r.component_of[0], r.component_of[1]);
    EXPECT_NE(r.component_of[1], r.component_of[2]);
    EXPECT_NE(r.component_of[2], r.component_of[3]);
}

TEST_F(SCCTest, DisconnectedMixed) {
    constexpr auto g = make_disconnected_mixed();
    constexpr auto r = scc(g);
    EXPECT_EQ(r.component_count, 4u);
    // {0,1} form a cycle SCC
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    // {2}, {3}, {4} separate
    EXPECT_NE(r.component_of[2], r.component_of[3]);
    EXPECT_NE(r.component_of[3], r.component_of[4]);
}

TEST_F(SCCTest, ComponentIdsAreBounded) {
    // Verify component_of values are in [0, component_count)
    constexpr auto g = make_mixed_scc();
    constexpr auto r = scc(g);
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        EXPECT_LT(r.component_of[i], r.component_count);
    }
}

// =============================================================================
// Runtime tests: connected_components
// =============================================================================

class ConnectedComponents : public ::testing::Test {};

TEST_F(ConnectedComponents, EmptyGraph) {
    constexpr auto g = make_empty();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 0u);
}

TEST_F(ConnectedComponents, Singleton) {
    constexpr auto g = make_singleton();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 1u);
    EXPECT_EQ(r.component_of[0], 0u);
}

TEST_F(ConnectedComponents, ChainOneComponent) {
    constexpr auto g = make_chain();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 1u);
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        EXPECT_EQ(r.component_of[i], 0u);
    }
}

TEST_F(ConnectedComponents, DisconnectedTwoComponents) {
    constexpr auto g = make_disconnected();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 2u);
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    EXPECT_EQ(r.component_of[2], r.component_of[3]);
    EXPECT_NE(r.component_of[0], r.component_of[2]);
}

TEST_F(ConnectedComponents, DenseRenumbering) {
    constexpr auto g = make_disconnected();
    constexpr auto r = connected_components(g);
    // First component (containing node 0) gets id 0
    EXPECT_EQ(r.component_of[0], 0u);
    // Second component (containing node 2) gets id 1
    EXPECT_EQ(r.component_of[2], 1u);
}

TEST_F(ConnectedComponents, StarOneComponent) {
    constexpr auto g = make_star();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 1u);
}

TEST_F(ConnectedComponents, IsolatedNodes) {
    constexpr auto g = make_isolated4();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 4u);
    // Dense renumbering: each node gets its own id in order
    EXPECT_EQ(r.component_of[0], 0u);
    EXPECT_EQ(r.component_of[1], 1u);
    EXPECT_EQ(r.component_of[2], 2u);
    EXPECT_EQ(r.component_of[3], 3u);
}

TEST_F(ConnectedComponents, DirectionIgnored) {
    // In make_chain: 0→1→2→3, direction ignored → all connected
    constexpr auto g = make_chain();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 1u);
}

TEST_F(ConnectedComponents, CycleOneComponent) {
    constexpr auto g = make_cycle3();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 1u);
}

TEST_F(ConnectedComponents, TwoCyclesTwoComponents) {
    constexpr auto g = make_two_cycles();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 2u);
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    EXPECT_EQ(r.component_of[2], r.component_of[3]);
    EXPECT_EQ(r.component_of[3], r.component_of[4]);
    EXPECT_NE(r.component_of[0], r.component_of[2]);
}

TEST_F(ConnectedComponents, MixedSCCOneWeakComponent) {
    constexpr auto g = make_mixed_scc();
    constexpr auto r = connected_components(g);
    // All nodes reachable ignoring direction → 1 component
    EXPECT_EQ(r.component_count, 1u);
}

TEST_F(ConnectedComponents, DisconnectedMixedTwoComponents) {
    constexpr auto g = make_disconnected_mixed();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 2u);
    EXPECT_EQ(r.component_of[0], r.component_of[1]);
    EXPECT_EQ(r.component_of[2], r.component_of[3]);
    EXPECT_EQ(r.component_of[3], r.component_of[4]);
    EXPECT_NE(r.component_of[0], r.component_of[2]);
}

TEST_F(ConnectedComponents, ComponentIdsAreDense) {
    // All component_of values must be in [0, component_count)
    constexpr auto g = make_two_cycles();
    constexpr auto r = connected_components(g);
    for (std::size_t i = 0; i < g.node_count(); ++i) {
        EXPECT_LT(r.component_of[i], r.component_count);
    }
}

TEST_F(ConnectedComponents, DiamondOneComponent) {
    constexpr auto g = make_diamond();
    constexpr auto r = connected_components(g);
    EXPECT_EQ(r.component_count, 1u);
}
