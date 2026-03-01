// tests/graph/test_graph_umbrella.cpp — Umbrella header compilation test
//
// Verifies that #include <ctdp/graph/graph.h> compiles without errors
// and all major types are accessible through it.

#include <ctdp/graph/graph.h>

#include <gtest/gtest.h>

using namespace ctdp::graph;

// Smoke test: build a graph, run an algorithm, use a transform,
// all via the umbrella header.
TEST(GraphUmbrellaTest, FullPipelineThroughUmbrella) {
    constexpr auto g = []() {
        graph_builder<cap_from<8, 16>> b;
        auto n0 = b.add_node(); auto n1 = b.add_node();
        auto n2 = b.add_node(); auto n3 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n0, n2);
        b.add_edge(n1, n3);
        b.add_edge(n2, n3);
        return b.finalise();
    }();

    // Representation.
    static_assert(g.node_count() == 4);
    static_assert(g.edge_count() == 4);

    // Algorithm.
    constexpr auto topo = topological_sort(g);
    static_assert(topo.is_dag);

    // Transform.
    constexpr auto gt = transpose(g);
    static_assert(gt.node_count() == 4);
    static_assert(gt.edge_count() == 4);

    // Annotation.
    constexpr auto km = make_uniform_kernel_map<8>(g,
        kernel_info{.flops = 10, .bytes_read = 4});
    static_assert(km.size() == 4);

    // Merge rule.
    property_map<std::size_t, 8> latencies(4, 0);
    latencies[0] = 5; latencies[1] = 10;
    latencies[2] = 3; latencies[3] = 8;
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 0;
    groups[2] = 1; groups[3] = 1;
    auto merged = merge_property(latencies, groups,
        std::size_t{2}, merge::max_of{});
    EXPECT_EQ(merged[0], 10u);  // max(5, 10)
    EXPECT_EQ(merged[1], 8u);   // max(3, 8)

    // Traits.
    using traits = graph_traits<decltype(g)>;
    static_assert(traits::max_nodes == 8);
    static_assert(traits::max_edges == 16);

    SUCCEED();
}

TEST(GraphUmbrellaTest, TypesAccessible) {
    // Verify key types are reachable through the umbrella.
    [[maybe_unused]] node_id n{0};
    [[maybe_unused]] edge_id e{0};
    [[maybe_unused]] topology_token t{};
    [[maybe_unused]] kernel_info ki{};
    [[maybe_unused]] kernel_tag kt{};
    [[maybe_unused]] merge::sum s{};
    [[maybe_unused]] merge::max_of m{};
    [[maybe_unused]] merge::fail f{};
    SUCCEED();
}
