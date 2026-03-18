// tests/test_reduction_tree_space.cpp
// Tests for ctdp/space/reduction_tree_space.h — reduction specialisation
//
// Coverage:
//   Section 9: make_group_properties (R4.2)
//   Section 10: make_fusibility_filter (R4.9, R4.10)
//   Section 11: make_reduction_opt_space(properties) overload
//   Section 12: Nested solve protocol (R4.11)
//   Section 13: End-to-end integration (R4.12)

#include "ctdp/space/reduction_tree_space.h"
#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <vector>

namespace {

using namespace ctdp::space;
using namespace ct_dp::algebra;

// ── Non-associative op at namespace scope (Clang requirement) ────────
struct noassoc_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a + b; }
};

// ════════════════════════════════════════════════════════════════════════
// Section 9: make_group_properties (R4.2)
// ════════════════════════════════════════════════════════════════════════

class GroupPropsTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto red = make_reduction(
            reduction_lane{identity_t{}, plus_fn{}, 0},           // lane 0
            reduction_lane{identity_t{}, plus_fn{}, 0},           // lane 1
            reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()},  // lane 2
            reduction_lane{identity_t{}, max_fn{},  std::numeric_limits<int>::min()},  // lane 3
            reduction_lane{identity_t{}, noassoc_fn{}, 0}         // lane 4
        );
        full_props = reduction_properties(red);
    }
    reduction_properties_t full_props{};
};

TEST_F(GroupPropsTest, PlusGroup) {
    std::size_t lanes[] = {0, 1};
    auto gp = make_group_properties(full_props, lanes);

    EXPECT_EQ(gp.lane_count, 2u);
    EXPECT_TRUE(gp.all_associative);
    EXPECT_TRUE(gp.all_commutative);
    EXPECT_TRUE(gp.all_have_identity);
    EXPECT_FALSE(gp.all_idempotent);
}

TEST_F(GroupPropsTest, MinMaxGroup) {
    std::size_t lanes[] = {2, 3};
    auto gp = make_group_properties(full_props, lanes);

    EXPECT_EQ(gp.lane_count, 2u);
    EXPECT_TRUE(gp.all_associative);
    EXPECT_TRUE(gp.all_idempotent);
    EXPECT_TRUE(gp.all_have_identity);
}

TEST_F(GroupPropsTest, SingleLane) {
    std::size_t lanes[] = {4};
    auto gp = make_group_properties(full_props, lanes);

    EXPECT_EQ(gp.lane_count, 1u);
    // noassoc_fn is not declared associative
    EXPECT_FALSE(gp.all_associative);
}

TEST_F(GroupPropsTest, LaneCount) {
    std::size_t lanes[] = {0, 2, 4};
    auto gp = make_group_properties(full_props, lanes);
    EXPECT_EQ(gp.lane_count, 3u);
}

TEST_F(GroupPropsTest, FullReindex) {
    // Verify per-lane arrays are correctly reindexed
    std::size_t lanes[] = {2, 4};  // min lane, noassoc lane
    auto gp = make_group_properties(full_props, lanes);

    // gp[0] should be full[2], gp[1] should be full[4]
    EXPECT_EQ(gp.lane_fingerprint[0], full_props.lane_fingerprint[2]);
    EXPECT_EQ(gp.lane_fingerprint[1], full_props.lane_fingerprint[4]);
    EXPECT_EQ(gp.lane_associative[0], full_props.lane_associative[2]);
    EXPECT_EQ(gp.lane_associative[1], full_props.lane_associative[4]);
    EXPECT_EQ(gp.lane_idempotent[0], full_props.lane_idempotent[2]);
    EXPECT_EQ(gp.lane_idempotent[1], full_props.lane_idempotent[4]);
}

// ════════════════════════════════════════════════════════════════════════
// Section 10: make_fusibility_filter (R4.9, R4.10)
// ════════════════════════════════════════════════════════════════════════

TEST(Fusibility, AllFusible_AllLegal) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);
    auto filter = make_fusibility_filter<3>(props);

    auto part = make_partition<3>("g");
    std::size_t legal = 0;
    for (std::size_t i = 0; i < part.cardinality(); ++i)
        if (filter(part.value_at(i))) ++legal;

    // All Bell(3)=5 should pass
    EXPECT_EQ(legal, 5u);
}

TEST(Fusibility, Mixed) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);
    auto filter = make_fusibility_filter<3>(props);

    auto part = make_partition<3>("g");
    std::size_t legal = 0;
    for (std::size_t i = 0; i < part.cardinality(); ++i)
        if (filter(part.value_at(i))) ++legal;

    // {0,1}+{2} and {0}+{1}+{2}
    EXPECT_EQ(legal, 2u);
}

TEST(Fusibility, AllDifferent) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()},
        reduction_lane{identity_t{}, noassoc_fn{}, 0}
    );
    auto props = reduction_properties(red);
    auto filter = make_fusibility_filter<3>(props);

    auto part = make_partition<3>("g");
    std::size_t legal = 0;
    for (std::size_t i = 0; i < part.cardinality(); ++i)
        if (filter(part.value_at(i))) ++legal;

    // Only singletons
    EXPECT_EQ(legal, 1u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 11: make_reduction_opt_space(properties) overload
// ════════════════════════════════════════════════════════════════════════

TEST(PropsOverload, MatchesTupleVersion) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);

    auto from_red = make_reduction_opt_space(red);
    auto from_props = make_reduction_opt_space(props);

    EXPECT_EQ(from_red.space.cardinality(), from_props.space.cardinality());
    EXPECT_EQ(from_red.bridge.num_features(), from_props.bridge.num_features());

    // Encode same point through both → same features
    auto pt = std::tuple{128, tree_shape::binary, 4};
    auto f1 = from_red.bridge.encode(pt);
    auto f2 = from_props.bridge.encode(pt);
    EXPECT_EQ(f1, f2);
}

TEST(PropsOverload, InactiveConditionals) {
    // Non-associative, no identity → tree_shape and vec_width inactive
    reduction_properties_t props{};
    props.lane_count = 1;
    props.all_associative = false;
    props.all_have_identity = false;

    auto result = make_reduction_opt_space(props);
    // Only tile_size active: 7 levels (64..4096)
    EXPECT_EQ(result.space.cardinality(), 7u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 12: Nested solve protocol (R4.11)
// ════════════════════════════════════════════════════════════════════════

TEST(NestedSolve, Separable) {
    // 3 lanes: 2 plus + 1 min → 2 legal partitions
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto ts = make_reduction_tree_space(red);
    auto props = reduction_properties(red);

    // Separable cost: root_cost = num_groups, child_cost = first dim value (tile)
    auto cost = [](const auto& pt) -> double {
        double c = static_cast<double>(pt.num_groups());
        for (const auto& plan : pt.group_plans)
            c += static_cast<double>(std::get<0>(plan));
        return c;
    };

    // Exhaustive: find minimum over all composite points
    double best_exhaustive = 1e18;
    ts.enumerate([&](const auto& pt) {
        double c = cost(pt);
        if (c < best_exhaustive) best_exhaustive = c;
    });

    // Nested: for each legal partition, independently minimise each group
    double best_nested = 1e18;
    auto root = make_partition<3>("g");
    auto filter = make_fusibility_filter<3>(props);
    auto child_factory = make_reduction_child_factory(props);

    for (std::size_t i = 0; i < root.cardinality(); ++i) {
        auto rv = root.value_at(i);
        if (!filter(rv)) continue;

        double composite_cost = static_cast<double>(rv.num_groups());
        auto K = rv.num_groups();

        for (std::size_t g = 0; g < K; ++g) {
            std::array<std::size_t, 3> lane_buf{};
            auto count = group_lanes(rv, g, lane_buf);
            auto lanes = std::span<const std::size_t>{lane_buf.data(), count};
            auto child = child_factory(rv, g, lanes);

            double best_child = 1e18;
            child.enumerate([&](const auto& child_pt) {
                double cc = static_cast<double>(std::get<0>(child_pt));
                if (cc < best_child) best_child = cc;
            });
            composite_cost += best_child;
        }

        if (composite_cost < best_nested) best_nested = composite_cost;
    }

    // Must agree (separable cost)
    EXPECT_DOUBLE_EQ(best_exhaustive, best_nested);
}

TEST(NestedSolve, FewerEvaluations) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto ts = make_reduction_tree_space(red);
    auto props = reduction_properties(red);

    // Count exhaustive evaluations
    std::size_t exhaustive_evals = ts.cardinality();

    // Count nested evaluations: for each legal partition, sum of child cardinalities
    std::size_t nested_evals = 0;
    auto root = make_partition<3>("g");
    auto filter = make_fusibility_filter<3>(props);
    auto child_factory = make_reduction_child_factory(props);

    for (std::size_t i = 0; i < root.cardinality(); ++i) {
        auto rv = root.value_at(i);
        if (!filter(rv)) continue;
        auto K = rv.num_groups();
        for (std::size_t g = 0; g < K; ++g) {
            std::array<std::size_t, 3> lane_buf{};
            auto count = group_lanes(rv, g, lane_buf);
            auto lanes = std::span<const std::size_t>{lane_buf.data(), count};
            auto child = child_factory(rv, g, lanes);
            nested_evals += child.cardinality();
        }
    }

    EXPECT_LT(nested_evals, exhaustive_evals);
}

// ════════════════════════════════════════════════════════════════════════
// Section 13: End-to-end integration (R4.12)
// ════════════════════════════════════════════════════════════════════════

TEST(E2E, FiveLaneStats) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()},
        reduction_lane{identity_t{}, max_fn{},  std::numeric_limits<int>::min()}
    );
    auto props = reduction_properties(red);

    // Precondition: min and max are fusible
    ASSERT_TRUE(props.fusible(3, 4));

    auto ts = make_reduction_tree_space(red);
    auto tb = make_reduction_tree_bridge(red);

    EXPECT_GT(ts.cardinality(), 0u);

    // Feature width: 5*4/2=10 pairwise + 5*5=25 child → 35
    EXPECT_EQ(tb.num_features(), 35u);

    // Verify {0,1,2}+{3,4} partition exists in enumeration
    bool found_target = false;
    std::size_t count = 0;
    std::size_t feature_width = tb.num_features();

    // Only check structure for first few thousand points (full enum is billions)
    auto filter = make_fusibility_filter<5>(props);
    auto root_desc = make_partition<5>("g");
    for (std::size_t i = 0; i < root_desc.cardinality(); ++i) {
        auto rv = root_desc.value_at(i);
        if (!filter(rv)) continue;
        ++count;

        if (rv.labels[0] == 0 && rv.labels[1] == 0 &&
            rv.labels[2] == 0 && rv.labels[3] == 1 &&
            rv.labels[4] == 1) {
            found_target = true;
        }
    }

    EXPECT_TRUE(found_target);
    EXPECT_EQ(count, 10u);  // 10 legal partitions of 5 lanes

    // Encode one point to verify features
    auto rv = root_desc.value_at(0);  // [0,0,0,0,0] all one group
    ASSERT_TRUE(filter(rv));

    auto child_factory = make_reduction_child_factory(props);
    std::array<std::size_t, 5> lane_buf{};
    auto lcount = group_lanes(rv, 0, lane_buf);
    auto lanes = std::span<const std::size_t>{lane_buf.data(), lcount};
    auto child = child_factory(rv, 0, lanes);
    auto child_pt = child.point_at(0);

    tree_point<5, decltype(child_pt)> tp{rv, {child_pt}};
    auto features = tb.encode(tp);
    EXPECT_EQ(features.size(), feature_width);
}

TEST(E2E, BackwardsCompat) {
    // make_reduction_opt_space still works
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto result = make_reduction_opt_space(red);
    EXPECT_GT(result.space.cardinality(), 0u);
    EXPECT_GT(result.bridge.num_features(), 0u);
}

} // namespace
