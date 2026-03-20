// tests/test_integration_reduction.cpp
// Sprint 6 — Example 4: Tiling + Fusion + Reduction pipeline
//
// Proves correct composition of the full framework stack: partition,
// tree_space, tree_bridge, nested solve, flat projection equivalence,
// and ordinal dimension extension.
//
// Evidence type: Composition (framework-on-framework).

#include "ctdp/space/reduction_tree_space.h"
#include "ctdp/space/counted_view.h"
#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace {

using namespace ctdp::space;
using namespace ct_dp::algebra;

// ── 5-lane stats fixture ────────────────────────────────────────────

auto make_5lane_reduction() {
    return make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},           // count
        reduction_lane{identity_t{}, plus_fn{}, 0},           // sum
        reduction_lane{identity_t{}, plus_fn{}, 0},           // sum_sq
        reduction_lane{identity_t{}, min_fn{},
                       std::numeric_limits<int>::max()},      // min
        reduction_lane{identity_t{}, max_fn{},
                       std::numeric_limits<int>::min()});     // max
}

// ════════════════════════════════════════════════════════════════════════
// R6.15: Legal partition count with fusibility preconditions
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationReduction, FusibilityPreconditions) {
    auto red = make_5lane_reduction();
    auto props = reduction_properties(red);

    // Preconditions
    ASSERT_TRUE(props.fusible(3, 4));    // min + max share fingerprint
    ASSERT_FALSE(props.fusible(0, 3));   // plus + min differ
    ASSERT_EQ(props.num_fingerprint_classes(), 2u);

    // Count legal partitions
    auto filter = make_fusibility_filter<5>(props);
    auto root = make_partition<5>("g");
    std::size_t legal = 0;
    for (std::size_t i = 0; i < root.cardinality(); ++i)
        if (filter(root.value_at(i))) ++legal;

    EXPECT_EQ(legal, 10u);
}

// ════════════════════════════════════════════════════════════════════════
// R6.16: Feature width (derived from child bridge, then pinned)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationReduction, FeatureWidth) {
    auto red = make_5lane_reduction();
    auto props = reduction_properties(red);
    auto tb = make_reduction_tree_bridge(red);

    // Derive expected width from child bridge contract
    auto child_space = make_reduction_opt_space(props).space;
    auto child_bridge = default_bridge(child_space);
    std::size_t expected = 5 * (5 - 1) / 2          // root pairwise = 10
        + 5 * child_bridge.num_features();           // N * child features

    EXPECT_EQ(tb.num_features(), expected);
    EXPECT_EQ(expected, 35u);  // pin current known-good value
}

// ════════════════════════════════════════════════════════════════════════
// R6.17: Feature width stable across ALL legal partitions
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationReduction, FeatureWidthStable) {
    auto red = make_5lane_reduction();
    auto props = reduction_properties(red);
    auto tb = make_reduction_tree_bridge(red);
    auto child_factory = make_reduction_child_factory(props);
    auto filter = make_fusibility_filter<5>(props);
    auto root = make_partition<5>("g");

    std::size_t expected_width = tb.num_features();
    std::size_t tested = 0;

    for (std::size_t i = 0; i < root.cardinality(); ++i) {
        auto pv = root.value_at(i);
        if (!filter(pv)) continue;

        // Build one child point per group via enumeration (not point_at)
        auto K = pv.num_groups();
        using child_pt_t = typename decltype(
            child_factory(pv, 0, std::span<const std::size_t>{}))::point_type;
        std::vector<child_pt_t> child_pts;

        for (std::size_t g = 0; g < K; ++g) {
            std::array<std::size_t, 5> lane_buf{};
            auto count = group_lanes(pv, g, lane_buf);
            auto lanes = std::span<const std::size_t>{lane_buf.data(), count};
            auto child = child_factory(pv, g, lanes);

            child_pt_t first{};
            bool got = false;
            child.enumerate([&](const auto& p) {
                if (!got) { first = p; got = true; }
            });
            ASSERT_TRUE(got) << "Empty child space for partition " << i
                             << " group " << g;
            child_pts.push_back(first);
        }

        tree_point<5, child_pt_t> tp{pv, child_pts};
        auto features = tb.encode(tp);
        EXPECT_EQ(features.size(), expected_width)
            << "Width mismatch for partition " << i
            << " (" << K << " groups)";
        ++tested;
    }

    EXPECT_EQ(tested, 10u);  // all 10 legal partitions
}

// ════════════════════════════════════════════════════════════════════════
// R6.18: Nested solve (separable additive cost, small tile range)
// ════════════════════════════════════════════════════════════════════════

class NestedSolveTest : public ::testing::Test {
protected:
    void SetUp() override {
        red_ = make_reduction(
            reduction_lane{identity_t{}, plus_fn{}, 0},
            reduction_lane{identity_t{}, plus_fn{}, 0},
            reduction_lane{identity_t{}, min_fn{},
                           std::numeric_limits<int>::max()});
        props_ = reduction_properties(red_);

        // Custom child factory with small tile range
        child_factory_ = [this](
            const auto& /*root*/, std::size_t /*group*/,
            std::span<const std::size_t> lane_indices) {
                auto gp = make_group_properties(props_, lane_indices);
                return descriptor_space("small",
                    power_2("tile_size", 64, 256),
                    conditional_dim(gp.all_associative,
                                    default_tree_shape_dim()),
                    conditional_dim(gp.all_have_identity,
                                    default_vec_width_dim()));
        };

        filter_ = decltype(filter_)(make_fusibility_filter<3>(props_));
    }

    decltype(make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},
                       std::numeric_limits<int>::max()})) red_{
        make_reduction(
            reduction_lane{identity_t{}, plus_fn{}, 0},
            reduction_lane{identity_t{}, plus_fn{}, 0},
            reduction_lane{identity_t{}, min_fn{},
                           std::numeric_limits<int>::max()})};
    reduction_properties_t props_{};

    // Store factory as std::function to avoid complex type deduction
    std::function<decltype(descriptor_space("",
        power_2("", 64, 256),
        conditional_dim(true, default_tree_shape_dim()),
        conditional_dim(true, default_vec_width_dim())))(
            const partition_value<3>&, std::size_t,
            std::span<const std::size_t>)> child_factory_;

    decltype(make_fusibility_filter<3>(
        std::declval<reduction_properties_t>())) filter_{
            make_fusibility_filter<3>(props_)};
};

TEST_F(NestedSolveTest, SeparableCostAgreement) {
    auto ts = make_tree_space<3>("test", make_partition<3>("g"),
        child_factory_, filter_);

    // Separable additive cost: root_cost = num_groups, child = tile_size
    auto cost = [](const auto& pt) -> double {
        double c = static_cast<double>(pt.num_groups());
        for (const auto& plan : pt.group_plans)
            c += static_cast<double>(std::get<0>(plan));
        return c;
    };

    // Exhaustive
    double best_exhaustive = 1e18;
    ts.enumerate([&](const auto& pt) {
        double c = cost(pt);
        if (c < best_exhaustive) best_exhaustive = c;
    });

    // Nested
    double best_nested = 1e18;
    auto root = make_partition<3>("g");
    for (std::size_t i = 0; i < root.cardinality(); ++i) {
        auto rv = root.value_at(i);
        if (!filter_(rv)) continue;

        double composite = static_cast<double>(rv.num_groups());
        for (std::size_t g = 0; g < rv.num_groups(); ++g) {
            std::array<std::size_t, 3> buf{};
            auto cnt = group_lanes(rv, g, buf);
            auto lanes = std::span<const std::size_t>{buf.data(), cnt};
            auto child = child_factory_(rv, g, lanes);

            double best_child = 1e18;
            child.enumerate([&](const auto& cpt) {
                double cc = static_cast<double>(std::get<0>(cpt));
                if (cc < best_child) best_child = cc;
            });
            composite += best_child;
        }
        if (composite < best_nested) best_nested = composite;
    }

    EXPECT_DOUBLE_EQ(best_exhaustive, best_nested);
}

TEST_F(NestedSolveTest, FewerEvaluations) {
    auto ts = make_tree_space<3>("test", make_partition<3>("g"),
        child_factory_, filter_);

    std::size_t exhaustive_count = ts.cardinality();

    std::size_t nested_evals = 0;
    auto root = make_partition<3>("g");
    for (std::size_t i = 0; i < root.cardinality(); ++i) {
        auto rv = root.value_at(i);
        if (!filter_(rv)) continue;
        for (std::size_t g = 0; g < rv.num_groups(); ++g) {
            std::array<std::size_t, 3> buf{};
            auto cnt = group_lanes(rv, g, buf);
            auto lanes = std::span<const std::size_t>{buf.data(), cnt};
            auto child = child_factory_(rv, g, lanes);
            nested_evals += child.cardinality();
        }
    }

    EXPECT_LT(nested_evals, exhaustive_count);
    EXPECT_EQ(exhaustive_count, 47952u);
}

// ════════════════════════════════════════════════════════════════════════
// R6.19: Flat projection equivalence (all-plus 3-lane)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationReduction, FlatProjectionEquivalence) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0});

    auto ts = make_reduction_tree_space(red);
    auto flat = make_flat_tree_space("flat",
        make_reduction_opt_space(red).space);

    // From tree_space: all-one-group child points
    using child_pt = typename decltype(ts)::child_point_type;
    std::vector<child_pt> tree_pts;
    ts.enumerate([&](const auto& pt) {
        if (pt.root.num_groups() == 1)
            tree_pts.push_back(pt.group_plans[0]);
    });

    // From flat: projected child points
    std::vector<child_pt> flat_pts;
    flat.enumerate([&](const auto& pt) {
        flat_pts.push_back(pt.group_plans[0]);
    });

    std::sort(tree_pts.begin(), tree_pts.end());
    std::sort(flat_pts.begin(), flat_pts.end());

    EXPECT_FALSE(tree_pts.empty());
    EXPECT_EQ(tree_pts, flat_pts);
}

// ════════════════════════════════════════════════════════════════════════
// R6.20: Ordinal integration (isolated, custom bridge)
// ════════════════════════════════════════════════════════════════════════

enum class repro : int {
    nondet = 0, deterministic = 1, reproducible = 2, bitwise = 3
};

TEST(IntegrationReduction, OrdinalIntegration) {
    // Properties for an all-plus group (all active)
    reduction_properties_t props{};
    props.lane_count = 2;
    props.all_associative = true;
    props.all_commutative = true;
    props.all_have_identity = true;

    // Extended 4-dim space: base 3 reduction dims + ordinal
    auto space = descriptor_space("extended",
        power_2("tile_size", 64, 4096),
        conditional_dim(props.all_associative, default_tree_shape_dim()),
        conditional_dim(props.all_have_identity, default_vec_width_dim()),
        make_ordinal("repro", {repro::nondet, repro::deterministic,
                               repro::reproducible, repro::bitwise}));

    // Cardinality = base (84) × 4 ordinal values
    auto base_space = make_reduction_opt_space(props).space;
    EXPECT_EQ(space.cardinality(), base_space.cardinality() * 4);

    // Custom bridge for 4-dim space (not make_reduction_tree_bridge)
    auto bridge = default_bridge(space);
    // Feature width = base 5 + 1 normalised rank = 6
    EXPECT_EQ(bridge.num_features(), 6u);

    // Encode a point with repro::reproducible
    auto pt = std::tuple{128, tree_shape::binary, 4, repro::reproducible};
    auto f = bridge.encode(pt);
    ASSERT_EQ(f.size(), 6u);

    // Last feature is ordinal normalised rank: 2/3 ≈ 0.6667
    EXPECT_NEAR(f[5], 2.0 / 3.0, 1e-9);
}

// ════════════════════════════════════════════════════════════════════════
// R6.21: Bridge protocol (tree_bridge)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationReduction, BridgeProtocol) {
    auto red = make_5lane_reduction();
    auto ts = make_reduction_tree_space(red);
    auto tb = make_reduction_tree_bridge(red);

    // Get first tree_point via enumeration
    using pt_t = typename decltype(ts)::point_type;
    pt_t first_pt{};
    bool got = false;
    ts.enumerate([&](const auto& pt) {
        if (!got) { first_pt = pt; got = true; }
    });
    ASSERT_TRUE(got);

    auto features = tb.encode(first_pt);
    EXPECT_EQ(features.size(), tb.num_features());
}

// ════════════════════════════════════════════════════════════════════════
// R6.15, R6.16: Pipeline summary (diagnostic)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationReduction, PipelineSummary) {
    auto red = make_5lane_reduction();
    auto props = reduction_properties(red);
    auto tb = make_reduction_tree_bridge(red);

    auto child_bridge = default_bridge(make_reduction_opt_space(props).space);
    std::size_t expected_width = 5 * (5 - 1) / 2
        + 5 * child_bridge.num_features();

    EXPECT_EQ(props.lane_count, 5u);
    EXPECT_EQ(tb.num_features(), expected_width);

    auto filter = make_fusibility_filter<5>(props);
    auto root = make_partition<5>("g");
    std::size_t legal = 0;
    for (std::size_t i = 0; i < root.cardinality(); ++i)
        if (filter(root.value_at(i))) ++legal;
    EXPECT_EQ(legal, 10u);
}

} // namespace
