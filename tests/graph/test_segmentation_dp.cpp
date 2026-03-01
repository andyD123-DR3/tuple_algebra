// tests/graph/test_segmentation_dp.cpp — Tests for composite DP over graph partitions
//
// Tests: segment partitioning, trivial/custom solvers,
// solution assembly, dependency validation, constexpr proofs.

#include <ctdp/engine/bridge/segmentation_dp.h>
#include <ctdp/engine/bridge/graph_to_space.h>
#include <ctdp/engine/bridge/graph_to_constraints.h>
#include <ctdp/graph/constexpr_graph.h>
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/kernel_info.h>
#include <ctdp/graph/fuse_group.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cmath>

using namespace ctdp::graph;

// =========================================================================
// Test graphs + kernel maps
// =========================================================================

// Chain: 0→1→2→3, uniform kernel_info.
constexpr auto make_chain4() {
    graph_builder<cap_from<8, 16>> b;
    auto n0 = b.add_node(); auto n1 = b.add_node();
    auto n2 = b.add_node(); auto n3 = b.add_node();
    b.add_edge(n0, n1);
    b.add_edge(n1, n2);
    b.add_edge(n2, n3);
    return b.finalise();
}

constexpr auto chain_g = make_chain4();
constexpr auto chain_km = make_uniform_kernel_map<8>(chain_g,
    kernel_info{.tag = kernel_tag{1}, .flops = 100, .bytes_read = 40,
                .bytes_written = 10, .is_fusable = true});

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

constexpr auto diamond_g = make_diamond();
constexpr auto diamond_km = []() {
    return make_kernel_map<8>(diamond_g,
        [](node_id n) {
            return kernel_info{
                .tag = kernel_tag{1},
                .flops = static_cast<std::size_t>((n.value + 1) * 100),
                .bytes_read = 40,
                .bytes_written = 10,
                .is_fusable = true
            };
        });
}();

// =========================================================================
// Trivial solver: strategy 0, cost = total flops
// =========================================================================

struct trivial_solver {
    template<std::size_t MaxV>
    constexpr segment_result<MaxV>
    operator()(segment_view<MaxV> const& seg) const {
        segment_result<MaxV> r;
        r.group_id = seg.group_id;
        r.cost = 0.0;
        for (std::size_t i = 0; i < seg.size(); ++i) {
            r.choice[i] = 0;
            r.cost += static_cast<double>(seg[i].info.flops);
        }
        r.feasible = true;
        return r;
    }
};

// =========================================================================
// Partition tests
// =========================================================================

TEST(SegmentationDPTest, PartitionSingletonGroups) {
    // Singleton groups (no fusion): each node is its own segment.
    auto space = build_schedule_space<8, 16>(chain_g, chain_km);
    auto segments = partition_segments(space);

    EXPECT_EQ(space.group_count, 4u);
    for (std::size_t g = 0; g < 4; ++g) {
        EXPECT_EQ(segments[g].size(), 1u)
            << "Group " << g << " should have 1 node";
    }
}

TEST(SegmentationDPTest, PartitionFusedGroups) {
    // Fuse all 4 chain nodes into 1 group.
    auto fg = find_fusion_groups<8, 16>(chain_g, chain_km);
    auto space = build_schedule_space_fused<8, 16>(chain_g, chain_km, fg);

    EXPECT_EQ(fg.group_count, 1u);
    auto segments = partition_segments(space);
    EXPECT_EQ(segments[0].size(), 4u);
}

TEST(SegmentationDPTest, PartitionPreservesTopoOrder) {
    auto fg = find_fusion_groups<8, 16>(chain_g, chain_km);
    auto fused_space = build_schedule_space_fused<8, 16>(chain_g, chain_km, fg);
    auto segments = partition_segments(fused_space);

    // All nodes in segment 0, in topo order.
    for (std::size_t i = 1; i < segments[0].size(); ++i) {
        EXPECT_LT(segments[0][i - 1].topo_rank, segments[0][i].topo_rank)
            << "Topo order violated in segment";
    }
}

TEST(SegmentationDPTest, SegmentViewMetrics) {
    auto fg = find_fusion_groups<8, 16>(chain_g, chain_km);
    auto space = build_schedule_space_fused<8, 16>(chain_g, chain_km, fg);
    auto segments = partition_segments(space);

    // 4 nodes × 100 flops = 400
    EXPECT_EQ(segments[0].total_flops(), 400u);
    // 4 nodes × (40+10) bytes = 200
    EXPECT_EQ(segments[0].total_bytes(), 200u);
}

// =========================================================================
// Solver tests
// =========================================================================

TEST(SegmentationDPTest, TrivialSolverSingletonGroups) {
    auto space = build_schedule_space<8, 16>(chain_g, chain_km);
    auto sol = solve_segmented(space, trivial_solver{});

    EXPECT_TRUE(sol.all_feasible);
    EXPECT_EQ(sol.segment_count, 4u);
    EXPECT_DOUBLE_EQ(sol.total_cost, 400.0);  // 4 × 100

    // Each segment costs 100 flops.
    for (std::size_t g = 0; g < 4; ++g) {
        EXPECT_DOUBLE_EQ(sol.segment_cost[g], 100.0);
    }
}

TEST(SegmentationDPTest, TrivialSolverFusedGroups) {
    auto fg = find_fusion_groups<8, 16>(chain_g, chain_km);
    auto space = build_schedule_space_fused<8, 16>(chain_g, chain_km, fg);
    auto sol = solve_segmented(space, trivial_solver{});

    EXPECT_TRUE(sol.all_feasible);
    EXPECT_EQ(sol.segment_count, 1u);
    EXPECT_DOUBLE_EQ(sol.total_cost, 400.0);
    EXPECT_DOUBLE_EQ(sol.segment_cost[0], 400.0);
}

TEST(SegmentationDPTest, DiamondNonUniformCosts) {
    // Diamond with non-uniform flops: 100, 200, 300, 400
    auto space = build_schedule_space<8, 16>(diamond_g, diamond_km);
    auto sol = solve_segmented(space, trivial_solver{});

    EXPECT_TRUE(sol.all_feasible);
    EXPECT_DOUBLE_EQ(sol.total_cost, 1000.0);  // 100+200+300+400
}

TEST(SegmentationDPTest, InfeasibleSolver) {
    // A solver that always reports infeasible.
    auto infeasible = [](auto const& seg) {
        segment_result<8> r;
        r.group_id = seg.group_id;
        r.cost = 0.0;
        r.feasible = false;
        return r;
    };

    auto space = build_schedule_space<8, 16>(chain_g, chain_km);
    auto sol = solve_segmented(space, infeasible);

    EXPECT_FALSE(sol.all_feasible);
    EXPECT_EQ(sol.infeasible_count, 4u);
}

TEST(SegmentationDPTest, CustomStrategyChoices) {
    // Solver that picks strategy = topo_rank for each node.
    auto rank_solver = [](auto const& seg) {
        segment_result<8> r;
        r.group_id = seg.group_id;
        r.cost = 1.0;
        for (std::size_t i = 0; i < seg.size(); ++i) {
            r.choice[i] = seg[i].topo_rank;
        }
        r.feasible = true;
        return r;
    };

    auto space = build_schedule_space<8, 16>(chain_g, chain_km);
    auto sol = solve_segmented(space, rank_solver);

    // Check that global choices reflect topo_rank.
    for (std::size_t rank = 0; rank < 4; ++rank) {
        EXPECT_EQ(sol.choice[rank], static_cast<std::uint16_t>(rank));
    }
}

// =========================================================================
// Validation tests
// =========================================================================

TEST(SegmentationDPTest, ValidatePasses) {
    auto space = build_schedule_space<8, 16>(chain_g, chain_km);
    auto deps = extract_dependencies<8, 16>(chain_g, space);
    auto sol = solve_segmented(space, trivial_solver{});

    EXPECT_TRUE(validate_segmented(sol, deps));
}

TEST(SegmentationDPTest, ValidateDiamondPasses) {
    auto space = build_schedule_space<8, 16>(diamond_g, diamond_km);
    auto deps = extract_dependencies<8, 16>(diamond_g, space);
    auto sol = solve_segmented(space, trivial_solver{});

    EXPECT_TRUE(validate_segmented(sol, deps));
}

// =========================================================================
// Constexpr proofs
// =========================================================================

namespace {

constexpr auto ce_space = build_schedule_space<8, 16>(chain_g, chain_km);
constexpr auto ce_segments = partition_segments(ce_space);

// 4 singleton groups.
static_assert(ce_space.group_count == 4);
static_assert(ce_segments[0].size() == 1);
static_assert(ce_segments[1].size() == 1);
static_assert(ce_segments[2].size() == 1);
static_assert(ce_segments[3].size() == 1);

// Solve at constexpr time.
constexpr auto ce_sol = solve_segmented(ce_space, trivial_solver{});
static_assert(ce_sol.all_feasible);
static_assert(ce_sol.segment_count == 4);
static_assert(ce_sol.total_cost == 400.0);

// Dependencies.
constexpr auto ce_deps = extract_dependencies<8, 16>(chain_g, ce_space);
static_assert(validate_segmented(ce_sol, ce_deps));

} // anonymous namespace
