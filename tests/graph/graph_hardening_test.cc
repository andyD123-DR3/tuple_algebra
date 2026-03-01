// tests/graph/graph_hardening_test.cc
// Tests for v0.5.4 graph hardening: capacity guards, builder overflow
// detection, coarsen group validation, property_map bounds checking.

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "graph_equal.h"
#include "property_map.h"
#include "kernel_info.h"
#include "topological_sort.h"
#include "scc.h"
#include "connected_components.h"
#include "coarsen.h"
#include "capacity_guard.h"
#include "from_pipeline.h"
#include "graph_to_space.h"
#include "graph_to_constraints.h"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// 1. constexpr_graph static_assert: uint16_t codification (item 4B)
// =============================================================================

// This should compile fine: MaxV within uint16_t range.
static_assert(constexpr_graph<cap_from<512, 1024>>::max_vertices == 512);

// The following would fail to compile (uncomment to verify):
// constexpr_graph<cap_from<70000, 1024>> too_big;  // static_assert: MaxV > 65535

TEST(GraphHardening, Uint16StaticAssert) {
    // Verify the static_assert doesn't fire for valid sizes.
    constexpr_graph<cap_from<64, 128>> small;
    EXPECT_EQ(small.node_count(), 0u);
}

// =============================================================================
// 2. graph_builder overflow detection (item 2B)
// =============================================================================

TEST(GraphHardening, BuilderAddNodeWithinCapacity) {
    // Normal usage should work fine.
    constexpr auto g = []() {
        graph_builder<cap_from<4, 8>> b;
        auto n0 = b.add_node();
        auto n1 = b.add_node();
        auto n2 = b.add_node();
        auto n3 = b.add_node();
        b.add_edge(n0, n1);
        b.add_edge(n1, n2);
        b.add_edge(n2, n3);
        return b.finalise();
    }();
    static_assert(g.node_count() == 4);
    static_assert(g.edge_count() == 3);
    EXPECT_EQ(g.node_count(), 4u);
}

TEST(GraphHardening, BuilderAddNodesWithinCapacity) {
    constexpr auto g = []() {
        graph_builder<cap_from<8, 16>> b;
        auto first = b.add_nodes(5);
        // first should be node 0
        b.add_edge(first, node_id{1});
        return b.finalise();
    }();
    static_assert(g.node_count() == 5);
    EXPECT_EQ(g.node_count(), 5u);
}

// The following would fail at compile time due to capacity guard:
// constexpr auto overflow = []() {
//     graph_builder<cap_from<2, 4>> b;
//     b.add_node(); b.add_node();
//     b.add_node();  // BOOM: V_ >= MaxV
//     return b.finalise();
// }();

// =============================================================================
// 3. Capacity guards on algorithms (item 2A)
// =============================================================================

TEST(GraphHardening, TopoSortCapacityGuardPasses) {
    constexpr auto g = from_pipeline<8>(4);
    constexpr auto result = topological_sort(g);
    static_assert(result.is_dag);
    static_assert(result.order.size() == 4);
    EXPECT_TRUE(result.is_dag);
}

TEST(GraphHardening, SCCCapacityGuardPasses) {
    constexpr auto g = from_pipeline<8>(4);
    constexpr auto result = scc(g);
    // Chain: each node is its own SCC
    static_assert(result.component_count == 4);
    EXPECT_EQ(result.component_count, 4u);
}

TEST(GraphHardening, ConnectedComponentsCapacityGuardPasses) {
    constexpr auto g = from_pipeline<8>(4);
    constexpr auto result = connected_components(g);
    // Chain: all connected
    static_assert(result.component_count == 1);
    EXPECT_EQ(result.component_count, 1u);
}

// =============================================================================
// 4. Coarsen group validation (item 2D)
// =============================================================================

TEST(GraphHardening, CoarsenValidGroupAssignment) {
    // Chain 0→1→2→3, group {0,1}=0 and {2,3}=1
    constexpr auto g = from_pipeline<8>(4);
    [[maybe_unused]] constexpr auto kmap = make_uniform_kernel_map<8>(g,
        kernel_info{.flops = 10, .bytes_read = 40});

    constexpr auto result = []() {
        auto g_inner = from_pipeline<8>(4);
        auto km = make_uniform_kernel_map<8>(g_inner,
            kernel_info{.flops = 10, .bytes_read = 40});

        property_map<std::uint16_t, 8> groups(4, std::uint16_t{0});
        groups[std::size_t{2}] = 1;
        groups[std::size_t{3}] = 1;

        return coarsen<8, 16>(g_inner, km, groups, 2);
    }();

    static_assert(result.graph.node_count() == 2);
    static_assert(result.graph.edge_count() == 1);  // one cross-group edge
    EXPECT_EQ(result.graph.node_count(), 2u);
}

// The following would fail at compile time: group_of[i] >= group_count
// constexpr auto bad_coarsen = []() {
//     auto g = from_pipeline<8>(4);
//     auto km = make_uniform_kernel_map<8>(g, kernel_info{.flops = 10});
//     property_map<std::uint16_t, 8> groups(4, std::uint16_t{0});
//     groups[std::size_t{2}] = 5;  // group 5 but group_count is 2!
//     return coarsen<8, 16>(g, km, groups, 2);
// }();

// =============================================================================
// 5. Property map bounds checking (item 2C)
// =============================================================================

TEST(GraphHardening, PropertyMapValidAccess) {
    property_map<int, 8> pmap(4, 42);
    EXPECT_EQ(pmap[std::size_t{0}], 42);
    EXPECT_EQ(pmap[std::size_t{3}], 42);
    EXPECT_EQ(pmap[node_id{0}], 42);
    EXPECT_EQ(pmap[node_id{3}], 42);
}

TEST(GraphHardening, PropertyMapBoundsCheckCompileTime) {
    // Valid access at constexpr
    constexpr auto val = []() {
        property_map<int, 8> pmap(4, 99);
        return pmap[std::size_t{3}];
    }();
    static_assert(val == 99);
}

// The following would fail at compile time: index >= size_
// constexpr auto oob = []() {
//     property_map<int, 8> pmap(4, 99);
//     return pmap[std::size_t{4}];  // BOOM: 4 >= size_ (4)
// }();

// =============================================================================
// 6. Engine/bridge forwarding shims work (item 3)
// =============================================================================

TEST(GraphHardening, EngineBridgeForwardingWorks) {
    // graph_to_space.h and graph_to_constraints.h are now forwarding
    // shims to engine/bridge/.  Test that the includes still work.
    constexpr auto g = from_pipeline<8>(4);
    constexpr auto kmap = make_uniform_kernel_map<8>(g,
        kernel_info{.flops = 100, .bytes_read = 400});

    constexpr auto space = build_schedule_space<8, 16>(g, kmap);
    static_assert(space.size() == 4);
    static_assert(space.is_dag);
    static_assert(space[0].topo_rank == 0);

    constexpr auto deps = extract_dependencies<8, 16>(g, space);
    static_assert(deps.size() == 3);  // 3 edges in chain of 4

    EXPECT_EQ(space.size(), 4u);
    EXPECT_EQ(deps.size(), 3u);
}

// =============================================================================
// 7. Guard helper: require_capacity at constexpr time
// =============================================================================

TEST(GraphHardening, RequireCapacityPassesWhenOK) {
    // This should not throw at runtime.
    require_capacity(10, 20, "test");
    require_capacity(0, 0, "test");  // edge case: 0 <= 0
}

TEST(GraphHardening, RequireCapacityCompileTime) {
    // Verify it works at compile time.
    constexpr auto result = []() {
        require_capacity(5, 10, "test");
        return true;
    }();
    static_assert(result);
}

// The following would fail at compile time:
// constexpr auto fail = []() {
//     require_capacity(11, 10, "test: 11 > 10");
//     return true;
// }();

// =============================================================================
// 8. Merged kernel info after valid coarsen
// =============================================================================

TEST(GraphHardening, CoarsenMergesKernelInfo) {
    constexpr auto result = []() {
        auto g = from_pipeline<8>(4);
        auto km = make_uniform_kernel_map<8>(g,
            kernel_info{.flops = 10, .bytes_read = 40});

        property_map<std::uint16_t, 8> groups(4, std::uint16_t{0});
        groups[std::size_t{2}] = 1;
        groups[std::size_t{3}] = 1;

        return coarsen<8, 16>(g, km, groups, 2);
    }();

    // Group 0 has nodes {0,1}: merged flops = 20
    static_assert(result.kernels[std::size_t{0}].flops == 20);
    // Group 1 has nodes {2,3}: merged flops = 20
    static_assert(result.kernels[std::size_t{1}].flops == 20);

    EXPECT_EQ(result.kernels[std::size_t{0}].flops, 20u);
    EXPECT_EQ(result.kernels[std::size_t{1}].flops, 20u);
}
