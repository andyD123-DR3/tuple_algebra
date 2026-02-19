// graph/test/graph_step6_test.cc - Tests for graph_to_space, graph_to_constraints
// Part of the compile-time DP library (C++20)

#include "graph_concepts.h"
#include "constexpr_graph.h"
#include "graph_builder.h"
#include "property_map.h"
#include "kernel_info.h"
#include "fusion_legal.h"
#include "fuse_group.h"
#include "topological_sort.h"
#include "graph_to_space.h"
#include "graph_to_constraints.h"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =============================================================================
// Test graph factories
// =============================================================================

constexpr auto make_empty() {
    graph_builder<8, 16> b;
    return b.finalise();
}

constexpr auto make_singleton() {
    graph_builder<8, 16> b;
    [[maybe_unused]] auto n = b.add_node();
    return b.finalise();
}

// 0→1→2→3
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

// 0→1, 0→2, 1→3, 2→3
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

// 0→1, 2→3 (disconnected)
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

// 0→1, 0→2, 0→3, 0→4
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

// Two sources converge: 0→2, 1→2, 2→3
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

inline constexpr kernel_tag tag_A{1};
inline constexpr kernel_tag tag_B{2};

// =============================================================================
// Compile-time verification: build_schedule_space
// =============================================================================

// --- Empty graph ---
constexpr auto space_empty = build_schedule_space<8, 16>(
    make_empty(), kernel_map<8>{});
static_assert(space_empty.size() == 0);
static_assert(space_empty.is_dag);
static_assert(space_empty.group_count == 0);
static_assert(space_empty.total_flops() == 0);
static_assert(space_empty.total_bytes() == 0);

// --- Singleton ---
constexpr auto g_single = make_singleton();
constexpr auto km_single = [] {
    kernel_map<8> km(1, default_kernel_info);
    km[node_id{0}] = kernel_info{.tag = tag_A, .flops = 50,
                                  .bytes_read = 200, .bytes_written = 8};
    return km;
}();
constexpr auto space_single = build_schedule_space<8, 16>(g_single, km_single);
static_assert(space_single.size() == 1);
static_assert(space_single.is_dag);
static_assert(space_single[0].id == node_id{0});
static_assert(space_single[0].topo_rank == 0);
static_assert(space_single[0].info.flops == 50);
static_assert(space_single[0].pred_count == 0);
static_assert(space_single[0].succ_count == 0);
static_assert(space_single.total_flops() == 50);
static_assert(space_single.total_bytes() == 208);

// --- Chain 0→1→2→3 ---
constexpr auto g_chain = make_chain();
constexpr auto km_chain = [] {
    kernel_map<8> km(4, default_kernel_info);
    km[node_id{0}] = kernel_info{.tag = tag_A, .flops = 100,
                                  .bytes_read = 400, .bytes_written = 8};
    km[node_id{1}] = kernel_info{.tag = tag_A, .flops = 200,
                                  .bytes_read = 800, .bytes_written = 16};
    km[node_id{2}] = kernel_info{.tag = tag_B, .flops = 150,
                                  .bytes_read = 600, .bytes_written = 8};
    km[node_id{3}] = kernel_info{.tag = tag_B, .flops = 50,
                                  .bytes_read = 200, .bytes_written = 4};
    return km;
}();
constexpr auto space_chain = build_schedule_space<8, 16>(g_chain, km_chain);
static_assert(space_chain.size() == 4);
static_assert(space_chain.is_dag);
// Topo order for chain is deterministic: 0, 1, 2, 3
static_assert(space_chain[0].id == node_id{0});
static_assert(space_chain[1].id == node_id{1});
static_assert(space_chain[2].id == node_id{2});
static_assert(space_chain[3].id == node_id{3});
static_assert(space_chain[0].topo_rank == 0);
static_assert(space_chain[3].topo_rank == 3);
// Kernel info preserved
static_assert(space_chain[0].info.flops == 100);
static_assert(space_chain[1].info.flops == 200);
static_assert(space_chain[2].info.tag == tag_B);
// Degree counts
static_assert(space_chain[0].pred_count == 0);
static_assert(space_chain[0].succ_count == 1);
static_assert(space_chain[1].pred_count == 1);
static_assert(space_chain[1].succ_count == 1);
static_assert(space_chain[3].pred_count == 1);
static_assert(space_chain[3].succ_count == 0);
// Singleton groups by default
static_assert(space_chain.group_count == 4);
static_assert(space_chain[0].group_id == 0);
static_assert(space_chain[1].group_id == 1);
// Total metrics
static_assert(space_chain.total_flops() == 500);

// --- by_node lookup ---
static_assert(space_chain.by_node(node_id{0}).topo_rank == 0);
static_assert(space_chain.by_node(node_id{3}).topo_rank == 3);
static_assert(space_chain.by_node(node_id{2}).info.flops == 150);

// --- Diamond 0→1, 0→2, 1→3, 2→3 ---
constexpr auto g_diamond = make_diamond();
constexpr auto km_diamond = [] {
    kernel_map<8> km(4, default_kernel_info);
    for (std::size_t i = 0; i < 4; ++i) {
        km[i] = kernel_info{.tag = tag_A,
            .flops = static_cast<std::size_t>(i + 1) * 10,
            .bytes_read = 100, .bytes_written = 8};
    }
    return km;
}();
constexpr auto space_diamond = build_schedule_space<8, 16>(g_diamond, km_diamond);
static_assert(space_diamond.size() == 4);
static_assert(space_diamond.is_dag);
// Topo: {0, 1, 2, 3} (deterministic smallest-first)
static_assert(space_diamond[0].id == node_id{0});
static_assert(space_diamond[3].id == node_id{3});
// Node 0: out-degree 2, in-degree 0
static_assert(space_diamond[0].pred_count == 0);
static_assert(space_diamond[0].succ_count == 2);
// Node 3: in-degree 2, out-degree 0
static_assert(space_diamond[3].pred_count == 2);
static_assert(space_diamond[3].succ_count == 0);

// =============================================================================
// Compile-time verification: build_schedule_space_fused
// =============================================================================

constexpr auto km_chain_fuse = [] {
    kernel_map<8> km(4, default_kernel_info);
    km[node_id{0}] = kernel_info{.tag = tag_A, .flops = 100,
                                  .bytes_read = 400, .bytes_written = 8,
                                  .is_fusable = true};
    km[node_id{1}] = kernel_info{.tag = tag_A, .flops = 200,
                                  .bytes_read = 800, .bytes_written = 16,
                                  .is_fusable = true};
    km[node_id{2}] = kernel_info{.tag = tag_B, .flops = 150,
                                  .bytes_read = 600, .bytes_written = 8,
                                  .is_fusable = true};
    km[node_id{3}] = kernel_info{.tag = tag_B, .flops = 50,
                                  .bytes_read = 200, .bytes_written = 4,
                                  .is_fusable = true};
    return km;
}();

constexpr auto fg_chain = find_fusion_groups<8, 16>(g_chain, km_chain_fuse);
constexpr auto space_fused = build_schedule_space_fused<8, 16>(
    g_chain, km_chain_fuse, fg_chain);
// Mixed tags A,A,B,B → 2 fusion groups
static_assert(space_fused.size() == 4);
static_assert(space_fused.group_count == 2);
// Nodes 0,1 share a group; nodes 2,3 share a group
static_assert(space_fused[0].group_id == space_fused[1].group_id);
static_assert(space_fused[2].group_id == space_fused[3].group_id);
static_assert(space_fused[0].group_id != space_fused[2].group_id);

// --- group_members query ---
constexpr auto grp0_members = space_fused.group_members(
    space_fused[0].group_id);
static_assert(grp0_members.size() == 2);
static_assert(grp0_members[0].info.flops == 100);
static_assert(grp0_members[1].info.flops == 200);

// =============================================================================
// Compile-time verification: extract_dependencies
// =============================================================================

// --- Chain: 3 dependencies ---
constexpr auto deps_chain = extract_dependencies<8, 16>(g_chain, space_chain);
static_assert(deps_chain.size() == 3);
// Edge 0→1: rank(0)=0, rank(1)=1
static_assert(deps_chain.deps[0].pred_rank == 0);
static_assert(deps_chain.deps[0].succ_rank == 1);
// Edge 1→2: rank(1)=1, rank(2)=2
static_assert(deps_chain.deps[1].pred_rank == 1);
static_assert(deps_chain.deps[1].succ_rank == 2);
// All pred_rank < succ_rank (DAG property)
static_assert(deps_chain.deps[0].pred_rank < deps_chain.deps[0].succ_rank);
static_assert(deps_chain.deps[1].pred_rank < deps_chain.deps[1].succ_rank);
static_assert(deps_chain.deps[2].pred_rank < deps_chain.deps[2].succ_rank);

// --- Diamond: 4 dependencies ---
constexpr auto deps_diamond = extract_dependencies<8, 16>(g_diamond, space_diamond);
static_assert(deps_diamond.size() == 4);

// --- Empty: 0 dependencies ---
constexpr auto deps_empty = extract_dependencies<8, 16>(make_empty(), space_empty);
static_assert(deps_empty.size() == 0);

// --- Legality check: trivial schedule (rank = time slot) is legal ---
constexpr auto trivial_schedule = std::array<std::uint16_t, 8>{0, 1, 2, 3};
static_assert(deps_chain.is_legal(trivial_schedule, 4));

// --- Illegal schedule: reversed ---
constexpr auto reversed_schedule = std::array<std::uint16_t, 8>{3, 2, 1, 0};
static_assert(!deps_chain.is_legal(reversed_schedule, 4));

// =============================================================================
// Compile-time verification: resource_constraint
// =============================================================================

// --- Unconstrained: always passes ---
static_assert(unconstrained_resources.check(999999, 999999, 999999));
static_assert(check_all_resources(space_chain, unconstrained_resources));

// --- Bytes constraint ---
constexpr resource_constraint rc_bytes{.max_bytes_per_group = 500};
// Chain singleton groups: node 0 = 408 bytes (ok), node 1 = 816 (too much)
static_assert(check_resource(space_chain, rc_bytes, 0));   // 408 ≤ 500
static_assert(!check_resource(space_chain, rc_bytes, 1));  // 816 > 500
static_assert(!check_all_resources(space_chain, rc_bytes));

// --- Flops constraint ---
constexpr resource_constraint rc_flops{.max_flops_per_group = 150};
static_assert(check_resource(space_chain, rc_flops, 0));   // 100 ≤ 150
static_assert(!check_resource(space_chain, rc_flops, 1));  // 200 > 150
static_assert(check_resource(space_chain, rc_flops, 2));   // 150 ≤ 150
static_assert(check_resource(space_chain, rc_flops, 3));   // 50 ≤ 150

// --- Nodes-per-group constraint ---
constexpr resource_constraint rc_nodes{.max_nodes_per_group = 1};
static_assert(check_all_resources(space_chain, rc_nodes));  // all singletons

// --- Fused space: group of 2 exceeds max_nodes=1 ---
static_assert(!check_all_resources(space_fused, rc_nodes));

// --- Generous constraint for fused space ---
constexpr resource_constraint rc_generous{
    .max_bytes_per_group = 2000,
    .max_flops_per_group = 500,
    .max_nodes_per_group = 4
};
static_assert(check_all_resources(space_fused, rc_generous));

// =============================================================================
// Compile-time verification: build_constraints (full summary)
// =============================================================================

constexpr auto cs_chain = build_constraints<8, 16>(
    g_chain, space_chain, unconstrained_resources);
static_assert(cs_chain.dependency_count == 3);
static_assert(cs_chain.all_resources_ok);
// Critical path: 0→1→2→3, length 3
static_assert(cs_chain.critical_path_length == 3);

constexpr auto cs_diamond = build_constraints<8, 16>(
    g_diamond, space_diamond, unconstrained_resources);
static_assert(cs_diamond.dependency_count == 4);
static_assert(cs_diamond.all_resources_ok);
// Critical path: 0→1→3 or 0→2→3, length 2
static_assert(cs_diamond.critical_path_length == 2);

constexpr auto cs_star = build_constraints<8, 16>(
    make_star(),
    build_schedule_space<8, 16>(make_star(), [] {
        kernel_map<8> km(5, default_kernel_info);
        for (std::size_t i = 0; i < 5; ++i) {
            km[i] = kernel_info{.flops = 10, .bytes_read = 40};
        }
        return km;
    }()),
    unconstrained_resources);
static_assert(cs_star.dependency_count == 4);
// Critical path: 0→{1,2,3,4}, length 1 (star is shallow)
static_assert(cs_star.critical_path_length == 1);

constexpr auto cs_empty = build_constraints<8, 16>(
    make_empty(), space_empty, unconstrained_resources);
static_assert(cs_empty.dependency_count == 0);
static_assert(cs_empty.critical_path_length == 0);
static_assert(cs_empty.all_resources_ok);

// --- Converge graph: 0→2, 1→2, 2→3 ---
constexpr auto g_conv = make_converge();
constexpr auto km_conv = [] {
    kernel_map<8> km(4, default_kernel_info);
    for (std::size_t i = 0; i < 4; ++i) {
        km[i] = kernel_info{.flops = 10, .bytes_read = 40};
    }
    return km;
}();
constexpr auto space_conv = build_schedule_space<8, 16>(g_conv, km_conv);
constexpr auto cs_conv = build_constraints<8, 16>(
    g_conv, space_conv, unconstrained_resources);
static_assert(cs_conv.dependency_count == 3);
// Critical path: 0→2→3 (or 1→2→3), length 2
static_assert(cs_conv.critical_path_length == 2);

// =============================================================================
// Runtime tests: schedule_space
// =============================================================================

class ScheduleSpaceTest : public ::testing::Test {};

TEST_F(ScheduleSpaceTest, EmptyGraph) {
    auto space = build_schedule_space<8, 16>(make_empty(), kernel_map<8>{});
    EXPECT_EQ(space.size(), 0u);
    EXPECT_TRUE(space.is_dag);
    EXPECT_EQ(space.total_flops(), 0u);
}

TEST_F(ScheduleSpaceTest, Singleton) {
    constexpr auto g = make_singleton();
    kernel_map<8> km(1, default_kernel_info);
    km[node_id{0}] = kernel_info{.flops = 42, .bytes_read = 100};
    auto space = build_schedule_space<8, 16>(g, km);
    EXPECT_EQ(space.size(), 1u);
    EXPECT_EQ(space[0].id, node_id{0});
    EXPECT_EQ(space[0].topo_rank, 0);
    EXPECT_EQ(space[0].info.flops, 42u);
    EXPECT_EQ(space[0].pred_count, 0);
    EXPECT_EQ(space[0].succ_count, 0);
    EXPECT_EQ(space.total_flops(), 42u);
}

TEST_F(ScheduleSpaceTest, ChainTopoOrder) {
    auto space = build_schedule_space<8, 16>(g_chain, km_chain);
    ASSERT_EQ(space.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(space[i].id, (node_id{static_cast<std::uint16_t>(i)}));
        EXPECT_EQ(space[i].topo_rank, static_cast<std::uint16_t>(i));
    }
}

TEST_F(ScheduleSpaceTest, ChainDegreeCounts) {
    auto space = build_schedule_space<8, 16>(g_chain, km_chain);
    EXPECT_EQ(space[0].pred_count, 0);
    EXPECT_EQ(space[0].succ_count, 1);
    EXPECT_EQ(space[1].pred_count, 1);
    EXPECT_EQ(space[1].succ_count, 1);
    EXPECT_EQ(space[2].pred_count, 1);
    EXPECT_EQ(space[2].succ_count, 1);
    EXPECT_EQ(space[3].pred_count, 1);
    EXPECT_EQ(space[3].succ_count, 0);
}

TEST_F(ScheduleSpaceTest, DiamondDegreeCounts) {
    auto space = build_schedule_space<8, 16>(g_diamond, km_diamond);
    EXPECT_EQ(space[0].succ_count, 2);
    EXPECT_EQ(space[3].pred_count, 2);
}

TEST_F(ScheduleSpaceTest, ByNodeLookup) {
    auto space = build_schedule_space<8, 16>(g_chain, km_chain);
    EXPECT_EQ(space.by_node(node_id{0}).topo_rank, 0);
    EXPECT_EQ(space.by_node(node_id{2}).info.flops, 150u);
    EXPECT_EQ(space.by_node(node_id{3}).topo_rank, 3);
}

TEST_F(ScheduleSpaceTest, TotalMetrics) {
    auto space = build_schedule_space<8, 16>(g_chain, km_chain);
    EXPECT_EQ(space.total_flops(), 500u);
    // bytes: (400+8) + (800+16) + (600+8) + (200+4) = 2036
    EXPECT_EQ(space.total_bytes(), 2036u);
}

TEST_F(ScheduleSpaceTest, SingletonGroupsByDefault) {
    auto space = build_schedule_space<8, 16>(g_chain, km_chain);
    EXPECT_EQ(space.group_count, 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(space[i].group_id, space[i].id.value);
    }
}

TEST_F(ScheduleSpaceTest, FusedGroups) {
    auto space = build_schedule_space_fused<8, 16>(
        g_chain, km_chain_fuse, fg_chain);
    EXPECT_EQ(space.group_count, 2u);
    EXPECT_EQ(space[0].group_id, space[1].group_id);
    EXPECT_EQ(space[2].group_id, space[3].group_id);
    EXPECT_NE(space[0].group_id, space[2].group_id);
}

TEST_F(ScheduleSpaceTest, GroupMembers) {
    auto space = build_schedule_space_fused<8, 16>(
        g_chain, km_chain_fuse, fg_chain);
    auto members = space.group_members(space[0].group_id);
    EXPECT_EQ(members.size(), 2u);
    EXPECT_EQ(members[0].info.flops, 100u);
    EXPECT_EQ(members[1].info.flops, 200u);
}

TEST_F(ScheduleSpaceTest, DisconnectedGraph) {
    constexpr auto g = make_disconnected();
    kernel_map<8> km(4, default_kernel_info);
    for (std::size_t i = 0; i < 4; ++i) {
        km[i] = kernel_info{.flops = 10};
    }
    auto space = build_schedule_space<8, 16>(g, km);
    EXPECT_EQ(space.size(), 4u);
    EXPECT_TRUE(space.is_dag);
    // Topo order for disconnected: 0,1,2,3 or 0,2,1,3 (smallest first)
    // Since 0→1 and 2→3, nodes {0,2} are initial, smallest=0 first
    EXPECT_EQ(space[0].id, node_id{0});
}

// =============================================================================
// Runtime tests: dependencies
// =============================================================================

class DependencyTest : public ::testing::Test {};

TEST_F(DependencyTest, ChainDependencies) {
    auto deps = extract_dependencies<8, 16>(g_chain, space_chain);
    EXPECT_EQ(deps.size(), 3u);
    // All pred < succ (DAG)
    for (std::size_t i = 0; i < deps.size(); ++i) {
        EXPECT_LT(deps.deps[i].pred_rank, deps.deps[i].succ_rank);
    }
}

TEST_F(DependencyTest, DiamondDependencies) {
    auto deps = extract_dependencies<8, 16>(g_diamond, space_diamond);
    EXPECT_EQ(deps.size(), 4u);
}

TEST_F(DependencyTest, EmptyDependencies) {
    auto deps = extract_dependencies<8, 16>(make_empty(), space_empty);
    EXPECT_EQ(deps.size(), 0u);
}

TEST_F(DependencyTest, LegalTrivialSchedule) {
    auto deps = extract_dependencies<8, 16>(g_chain, space_chain);
    std::array<std::uint16_t, 8> schedule{0, 1, 2, 3};
    EXPECT_TRUE(deps.is_legal(schedule, 4));
}

TEST_F(DependencyTest, IllegalReversedSchedule) {
    auto deps = extract_dependencies<8, 16>(g_chain, space_chain);
    std::array<std::uint16_t, 8> schedule{3, 2, 1, 0};
    EXPECT_FALSE(deps.is_legal(schedule, 4));
}

TEST_F(DependencyTest, IllegalPartialViolation) {
    auto deps = extract_dependencies<8, 16>(g_chain, space_chain);
    // Swap nodes 1 and 2: 0, 2, 1, 3 — violates 1→2 (rank1 must precede rank2)
    std::array<std::uint16_t, 8> schedule{0, 2, 1, 3};
    EXPECT_FALSE(deps.is_legal(schedule, 4));
}

TEST_F(DependencyTest, DiamondLegalParallel) {
    auto deps = extract_dependencies<8, 16>(g_diamond, space_diamond);
    // Nodes 1 and 2 can execute at same time slot (parallel)
    // schedule: 0→slot0, 1→slot1, 2→slot1, 3→slot2
    std::array<std::uint16_t, 8> schedule{0, 1, 1, 2};
    EXPECT_TRUE(deps.is_legal(schedule, 4));
}

// =============================================================================
// Runtime tests: resource_constraint
// =============================================================================

class ResourceConstraintTest : public ::testing::Test {};

TEST_F(ResourceConstraintTest, Unconstrained) {
    EXPECT_TRUE(unconstrained_resources.check(999999, 999999, 999999));
    EXPECT_TRUE(check_all_resources(space_chain, unconstrained_resources));
}

TEST_F(ResourceConstraintTest, BytesConstraint) {
    resource_constraint rc{.max_bytes_per_group = 500};
    EXPECT_TRUE(check_resource(space_chain, rc, 0));   // 408
    EXPECT_FALSE(check_resource(space_chain, rc, 1));  // 816
}

TEST_F(ResourceConstraintTest, FlopsConstraint) {
    resource_constraint rc{.max_flops_per_group = 150};
    EXPECT_TRUE(check_resource(space_chain, rc, 0));   // 100
    EXPECT_FALSE(check_resource(space_chain, rc, 1));  // 200
    EXPECT_TRUE(check_resource(space_chain, rc, 2));   // 150 (equal)
    EXPECT_TRUE(check_resource(space_chain, rc, 3));   // 50
}

TEST_F(ResourceConstraintTest, NodesConstraint) {
    resource_constraint rc{.max_nodes_per_group = 1};
    EXPECT_TRUE(check_all_resources(space_chain, rc));   // all singletons
    EXPECT_FALSE(check_all_resources(space_fused, rc));  // fused groups > 1
}

TEST_F(ResourceConstraintTest, CombinedConstraint) {
    resource_constraint rc{
        .max_bytes_per_group = 2000,
        .max_flops_per_group = 500,
        .max_nodes_per_group = 4
    };
    EXPECT_TRUE(check_all_resources(space_fused, rc));
}

TEST_F(ResourceConstraintTest, FusedGroupResourceCheck) {
    // Group 0 of fused space: nodes 0,1 → 100+200 flops, (408+816) bytes
    auto space = build_schedule_space_fused<8, 16>(
        g_chain, km_chain_fuse, fg_chain);
    auto g0 = space[0].group_id;
    auto g1 = space[2].group_id;

    resource_constraint rc{.max_flops_per_group = 250};
    EXPECT_FALSE(check_resource(space, rc, g0));  // 300 > 250
    EXPECT_TRUE(check_resource(space, rc, g1));   // 200 ≤ 250
}

// =============================================================================
// Runtime tests: constraint_summary
// =============================================================================

class ConstraintSummaryTest : public ::testing::Test {};

TEST_F(ConstraintSummaryTest, ChainSummary) {
    auto cs = build_constraints<8, 16>(g_chain, space_chain);
    EXPECT_EQ(cs.dependency_count, 3u);
    EXPECT_TRUE(cs.all_resources_ok);
    EXPECT_EQ(cs.critical_path_length, 3u);
}

TEST_F(ConstraintSummaryTest, DiamondSummary) {
    auto cs = build_constraints<8, 16>(g_diamond, space_diamond);
    EXPECT_EQ(cs.dependency_count, 4u);
    EXPECT_EQ(cs.critical_path_length, 2u);  // max depth = 2
}

TEST_F(ConstraintSummaryTest, StarSummary) {
    constexpr auto g = make_star();
    kernel_map<8> km(5, default_kernel_info);
    for (std::size_t i = 0; i < 5; ++i) {
        km[i] = kernel_info{.flops = 10, .bytes_read = 40};
    }
    auto space = build_schedule_space<8, 16>(g, km);
    auto cs = build_constraints<8, 16>(g, space);
    EXPECT_EQ(cs.dependency_count, 4u);
    EXPECT_EQ(cs.critical_path_length, 1u);
}

TEST_F(ConstraintSummaryTest, ConvergeSummary) {
    auto cs = build_constraints<8, 16>(g_conv, space_conv);
    EXPECT_EQ(cs.dependency_count, 3u);
    EXPECT_EQ(cs.critical_path_length, 2u);
}

TEST_F(ConstraintSummaryTest, EmptySummary) {
    auto cs = build_constraints<8, 16>(make_empty(), space_empty);
    EXPECT_EQ(cs.dependency_count, 0u);
    EXPECT_EQ(cs.critical_path_length, 0u);
    EXPECT_TRUE(cs.all_resources_ok);
}

TEST_F(ConstraintSummaryTest, WithResourceViolation) {
    resource_constraint rc{.max_flops_per_group = 50};
    auto cs = build_constraints<8, 16>(g_chain, space_chain, rc);
    EXPECT_FALSE(cs.all_resources_ok);  // node 1 has 200 flops > 50
}

TEST_F(ConstraintSummaryTest, DisconnectedSummary) {
    constexpr auto g = make_disconnected();
    kernel_map<8> km(4, default_kernel_info);
    for (std::size_t i = 0; i < 4; ++i) {
        km[i] = kernel_info{.flops = 10, .bytes_read = 40};
    }
    auto space = build_schedule_space<8, 16>(g, km);
    auto cs = build_constraints<8, 16>(g, space);
    EXPECT_EQ(cs.dependency_count, 2u);
    // Critical path: max(0→1, 2→3) = 1
    EXPECT_EQ(cs.critical_path_length, 1u);
}
