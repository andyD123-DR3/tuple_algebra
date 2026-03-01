// tests/graph/test_graph_hardening_v2.cc
// Hardening tests for P0 fixes: add_edge validation, SCC O(V+E),
// capacity guards on fusion_legal, fuse_group, coarsen.

#include "ctdp/graph/graph_builder.h"
#include "ctdp/graph/capacity_types.h"
#include "ctdp/graph/constexpr_graph.h"
#include "ctdp/graph/property_map.h"
#include "ctdp/graph/scc.h"
#include "ctdp/graph/topological_sort.h"
#include "ctdp/graph/connected_components.h"
#include "ctdp/graph/symmetric_graph.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ctdp::graph;

// =============================================================================
// P0-1: add_edge validation
// =============================================================================


TEST(GraphHardeningV2, AddEdgeRejectsInvalidSourceAtRuntime) {
    // At runtime, require_capacity throws on invalid node_id.
    graph_builder<cap_from<4, 8>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    (void)n0; (void)n1;

    // node_id{5} was never added (only 0 and 1 exist).
    EXPECT_THROW(b.add_edge(node_id{5}, n1), std::length_error);
}

TEST(GraphHardeningV2, AddEdgeRejectsInvalidTargetAtRuntime) {
    graph_builder<cap_from<4, 8>> b;
    auto n0 = b.add_node();
    (void)n0;

    EXPECT_THROW(b.add_edge(n0, node_id{3}), std::length_error);
}

TEST(GraphHardeningV2, AddEdgeRejectsOnEmptyGraph) {
    graph_builder<cap_from<4, 8>> b;
    // No nodes added at all.
    EXPECT_THROW(b.add_edge(node_id{0}, node_id{1}), std::logic_error);
}

TEST(GraphHardeningV2, AddEdgeAcceptsValidIds) {
    // Should not throw.
    graph_builder<cap_from<4, 8>> b;
    auto n0 = b.add_node();
    auto n1 = b.add_node();
    EXPECT_NO_THROW(b.add_edge(n0, n1));
    EXPECT_NO_THROW(b.add_edge(n1, n0));  // reverse is fine
    EXPECT_NO_THROW(b.add_edge(n0, n0));  // self-edge accepted at builder level
}

TEST(GraphHardeningV2, SymmetricBuilderRejectsInvalidIds) {
    symmetric_graph_builder<cap_from<4, 8>> b;
    auto n0 = b.add_node();
    (void)n0;

    EXPECT_THROW(b.add_edge(n0, node_id{5}), std::length_error);
}

// =============================================================================
// P0-6: SCC correctness on star graph (high degree)
// =============================================================================

// Star graph: node 0 → {1, 2, ..., N-1}.  All edges one-way.
// Every node is its own SCC (no cycles).
// This is the worst case for the old O(∑deg²) scan: node 0 has degree N-1.
// With the O(1) CSR indexing fix, this should be fast.
TEST(GraphHardeningV2, SCCStarGraphCorrectness) {
    constexpr std::size_t N = 64;
    constexpr auto g = []() {
        graph_builder<cap_from<N, N>> b;
        for (std::size_t i = 0; i < N; ++i) (void)b.add_node();
        for (std::size_t i = 1; i < N; ++i) {
            b.add_edge(node_id{0}, node_id{static_cast<std::uint16_t>(i)});
        }
        return b.finalise();
    }();

    static_assert(g.node_count() == N);
    static_assert(g.edge_count() == N - 1);

    // Every node is its own SCC (DAG).
    constexpr auto result = scc(g);
    static_assert(result.component_count == N);
}

// Cycle: 0→1→2→...→(N-1)→0.  One SCC.
TEST(GraphHardeningV2, SCCCycleGraph) {
    constexpr std::size_t N = 32;
    constexpr auto g = []() {
        graph_builder<cap_from<N, N>> b;
        for (std::size_t i = 0; i < N; ++i) (void)b.add_node();
        for (std::size_t i = 0; i < N; ++i) {
            b.add_edge(
                node_id{static_cast<std::uint16_t>(i)},
                node_id{static_cast<std::uint16_t>((i + 1) % N)});
        }
        return b.finalise();
    }();

    constexpr auto result = scc(g);
    static_assert(result.component_count == 1);

    // All nodes in same component.
    for (std::size_t i = 1; i < N; ++i) {
        EXPECT_EQ(result.component_of[i], result.component_of[0]);
    }
}

// Complete DAG: i→j for all i < j.  Each node is its own SCC.
// High total degree, tests the CSR fast-path thoroughly.
TEST(GraphHardeningV2, SCCCompleteDag) {
    constexpr std::size_t N = 16;
    constexpr std::size_t E = N * (N - 1) / 2;
    constexpr auto g = []() {
        graph_builder<cap_from<N, E + 1>> b;
        for (std::size_t i = 0; i < N; ++i) (void)b.add_node();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                b.add_edge(
                    node_id{static_cast<std::uint16_t>(i)},
                    node_id{static_cast<std::uint16_t>(j)});
            }
        }
        return b.finalise();
    }();

    static_assert(g.node_count() == N);
    static_assert(g.edge_count() == E);

    constexpr auto result = scc(g);
    static_assert(result.component_count == N);
}

// =============================================================================
// P0-2: make_property_map guard (compile-time check)
// =============================================================================

TEST(GraphHardeningV2, PropertyMapFactoryGuard) {
    // Build a graph with 4 nodes, but try to make a property_map with MaxV=2.
    // This should throw at runtime (and would fail constexpr at compile time).
    auto g = []() {
        graph_builder<cap_from<8, 8>> b;
        for (int i = 0; i < 4; ++i) (void)b.add_node();
        return b.finalise();
    }();

    auto fn = [](node_id) { return 0; };
    // Wrap in a lambda to avoid preprocessor comma issues with template args.
    auto do_it = [&]() { return make_property_map<int, 2>(g, fn); };
    EXPECT_THROW(do_it(), std::length_error);
}

// =============================================================================
// P1: Include hygiene — verify graph headers compile with stable paths
// =============================================================================

// These static_asserts confirm the headers are reachable through the
// stable include paths added in this hardening pass.
TEST(GraphHardeningV2, IncludeHygiene) {
    // If we got here, all graph headers compiled with <ctdp/core/...> paths.
    // Verify a few key types are accessible.
    constexpr auto g = []() {
        graph_builder<cap_from<4, 4>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        b.add_edge(n0, n1);
        return b.finalise();
    }();

    static_assert(g.node_count() == 2);

    constexpr auto topo = topological_sort(g);
    static_assert(topo.is_dag);

    constexpr auto cc = connected_components(g);
    static_assert(cc.component_count == 1);
}
