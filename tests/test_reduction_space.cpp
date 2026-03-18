// tests/test_reduction_space.cpp
// Tests for ctdp/space/reduction_properties.h and ctdp/space/reduction_space.h
//
// Sprint 2, Tasks 2.4 + 2.6 + 2.7:
//   2.4  reduction_properties extraction from known reductions
//   2.6  make_reduction_opt_space generates correct dimensions
//   2.7  End-to-end: algebra → properties → space → enumerate → encode
//
// Coverage:
//   Section 1: reduction_properties — all-associative+identity reduction
//   Section 2: reduction_properties — mixed reduction (assoc but no identity)
//   Section 3: reduction_properties — non-associative reduction
//   Section 4: reduction_properties — single lane edge cases
//   Section 5: reduction_properties — transform classification
//   Section 6: make_reduction_opt_space — all-assoc+identity (both dims active)
//   Section 7: make_reduction_opt_space — assoc only (no identity → vec inactive)
//   Section 8: make_reduction_opt_space — neither (both dims inactive)
//   Section 9: make_reduction_opt_space — custom descriptors
//   Section 10: End-to-end integration (algebra → space → enumerate → encode)

#include "ctdp/space/reduction_space.h"
#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

using namespace ct_dp::algebra;
using namespace ctdp::space;

// ════════════════════════════════════════════════════════════════════════
// Reusable reduction definitions
// ════════════════════════════════════════════════════════════════════════

// All lanes: associative, commutative, have identity
auto make_stats_reduction() {
    return make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},           // count
        reduction_lane{identity_t{},    plus_fn{}, 0.0},         // sum
        reduction_lane{power_t<2>{},    plus_fn{}, 0.0},         // sum_sq
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()},  // min
        reduction_lane{identity_t{},    max_fn{},  std::numeric_limits<double>::lowest()} // max
    );
}

// Non-associative: custom reduce op without declared_associative
struct concat_fn {
    // No declared_associative, declared_commutative, or declared_idempotent
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a + b; }
};

// ════════════════════════════════════════════════════════════════════════
// Section 1: reduction_properties — all-assoc+identity (stats reduction)
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, StatsReduction_LaneCount) {
    auto props = reduction_properties(make_stats_reduction());
    EXPECT_EQ(props.lane_count, 5u);
}

TEST(ReductionProperties, StatsReduction_AllAssociative) {
    auto props = reduction_properties(make_stats_reduction());
    EXPECT_TRUE(props.all_associative);
    EXPECT_FALSE(props.any_non_associative);
}

TEST(ReductionProperties, StatsReduction_AllCommutative) {
    auto props = reduction_properties(make_stats_reduction());
    EXPECT_TRUE(props.all_commutative);
    EXPECT_FALSE(props.any_non_commutative);
}

TEST(ReductionProperties, StatsReduction_AllHaveIdentity) {
    auto props = reduction_properties(make_stats_reduction());
    EXPECT_TRUE(props.all_have_identity);
}

TEST(ReductionProperties, StatsReduction_Idempotent) {
    auto props = reduction_properties(make_stats_reduction());
    // min and max are idempotent; plus is not → not all_idempotent, but any_idempotent
    EXPECT_FALSE(props.all_idempotent);
    EXPECT_TRUE(props.any_idempotent);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: reduction_properties — mixed (all assoc, but no identity check)
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, AllAssocAllCommut_PlusAndMin) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    EXPECT_EQ(props.lane_count, 2u);
    EXPECT_TRUE(props.all_associative);
    EXPECT_TRUE(props.all_commutative);
    EXPECT_TRUE(props.all_have_identity);
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: reduction_properties — non-associative operation
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, NonAssociativeOp_BreaksAll) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{},   0},       // associative
        reduction_lane{identity_t{}, concat_fn{}, 0}        // NOT associative
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.all_associative);
    EXPECT_TRUE(props.any_non_associative);
    EXPECT_FALSE(props.all_commutative);
    EXPECT_TRUE(props.any_non_commutative);
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: reduction_properties — single lane edge cases
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, SingleLane_Plus) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);

    EXPECT_EQ(props.lane_count, 1u);
    EXPECT_TRUE(props.all_associative);
    EXPECT_TRUE(props.all_commutative);
    EXPECT_TRUE(props.all_have_identity);
    EXPECT_FALSE(props.all_idempotent);
}

TEST(ReductionProperties, SingleLane_Min) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, min_fn{}, std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.all_associative);
    EXPECT_TRUE(props.all_commutative);
    EXPECT_TRUE(props.all_idempotent);
    EXPECT_TRUE(props.any_idempotent);
}

TEST(ReductionProperties, SingleLane_NonAssociative) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, concat_fn{}, 0}
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.all_associative);
    EXPECT_TRUE(props.any_non_associative);
}

// ════════════════════════════════════════════════════════════════════════
// Section 5: reduction_properties — transform classification
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, AllIdentityTransforms) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.all_identity_transforms);
    EXPECT_FALSE(props.any_power_transforms);
}

TEST(ReductionProperties, MixedTransforms) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0.0},
        reduction_lane{power_t<2>{}, plus_fn{}, 0.0}
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.all_identity_transforms);
    EXPECT_TRUE(props.any_power_transforms);
}

TEST(ReductionProperties, ConstantTransform) {
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0}  // count
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.all_identity_transforms);  // constant is not identity
    EXPECT_FALSE(props.any_power_transforms);
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: make_reduction_opt_space — both dims active
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionOptSpace, StatsReduction_BothDimsActive) {
    auto result = make_reduction_opt_space(make_stats_reduction());

    // tile_size: 7 values (64..4096)
    // tree_shape: 3 values (active — all associative)
    // vec_width: 4 values (active — all have identity)
    // Total: 7 × 3 × 4 = 84
    EXPECT_EQ(result.space.cardinality(), 7u * 3u * 4u);
}

TEST(ReductionOptSpace, StatsReduction_EnumerationCountMatchesCardinality) {
    auto result = make_reduction_opt_space(make_stats_reduction());

    std::size_t count = 0;
    result.space.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, result.space.cardinality());
}

TEST(ReductionOptSpace, StatsReduction_FeatureWidth) {
    auto result = make_reduction_opt_space(make_stats_reduction());

    // tile_size (log2 → 1) + tree_shape (one-hot → 3) + vec_width (raw → 1) = 5
    EXPECT_EQ(result.bridge.num_features(), 5u);
}

TEST(ReductionOptSpace, StatsReduction_AllTreeShapesAppear) {
    auto result = make_reduction_opt_space(make_stats_reduction());

    std::array<int, 3> tree_counts{};
    result.space.enumerate([&](auto const& pt) {
        auto shape = std::get<1>(pt);
        tree_counts[static_cast<std::size_t>(shape)]++;
    });

    // Each tree shape appears: 7 tiles × 4 vec widths = 28 times
    for (int c : tree_counts) {
        EXPECT_EQ(c, 28);
    }
}

TEST(ReductionOptSpace, StatsReduction_AllVecWidthsAppear) {
    auto result = make_reduction_opt_space(make_stats_reduction());

    std::vector<int> vec_values;
    result.space.enumerate([&](auto const& pt) {
        vec_values.push_back(std::get<2>(pt));
    });

    // Check all 4 vec widths are present
    for (int expected : {1, 4, 8, 16}) {
        EXPECT_NE(std::find(vec_values.begin(), vec_values.end(), expected),
                  vec_values.end()) << "Missing vec_width=" << expected;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 7: make_reduction_opt_space — assoc only, no identity
// ════════════════════════════════════════════════════════════════════════

// Associative op without identity<T>() — must be at namespace scope
// because Clang rejects templates and static constexpr in local classes.
struct assoc_no_ident_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a + b; }
    static constexpr bool declared_associative = true;
    static constexpr bool declared_commutative = true;
    static constexpr bool declared_idempotent  = false;
    // No identity<T>() method
};

TEST(ReductionOptSpace, AssocNoIdentity_VecWidthInactive) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, assoc_no_ident_fn{}, 0}
    );
    auto props = reduction_properties(red);
    EXPECT_TRUE(props.all_associative);
    EXPECT_FALSE(props.all_have_identity);

    auto result = make_reduction_opt_space(red);

    // tile: 7 × tree_shape: 3 (active) × vec_width: 1 (inactive) = 21
    EXPECT_EQ(result.space.cardinality(), 7u * 3u * 1u);

    // vec_width always = 1 (default)
    result.space.enumerate([&](auto const& pt) {
        EXPECT_EQ(std::get<2>(pt), 1);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Section 8: make_reduction_opt_space — neither assoc nor identity
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionOptSpace, NeitherAssocNorIdentity_BothDimsInactive) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, concat_fn{}, 0}
    );
    auto props = reduction_properties(red);
    EXPECT_FALSE(props.all_associative);
    EXPECT_FALSE(props.all_have_identity);

    auto result = make_reduction_opt_space(red);

    // tile: 7 × tree_shape: 1 (inactive) × vec_width: 1 (inactive) = 7
    EXPECT_EQ(result.space.cardinality(), 7u);

    // Only flat tree shape and default vec width
    result.space.enumerate([&](auto const& pt) {
        EXPECT_EQ(std::get<1>(pt), tree_shape::flat);
        EXPECT_EQ(std::get<2>(pt), 1);
    });
}

TEST(ReductionOptSpace, NeitherAssocNorIdentity_FeatureWidthSameAsFullActive) {
    auto red_full = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto red_none = make_reduction(
        reduction_lane{identity_t{}, concat_fn{}, 0}
    );

    auto result_full = make_reduction_opt_space(red_full);
    auto result_none = make_reduction_opt_space(red_none);

    // Feature width MUST be the same regardless of which dims are active
    EXPECT_EQ(result_full.bridge.num_features(),
              result_none.bridge.num_features());
}

// ════════════════════════════════════════════════════════════════════════
// Section 9: make_reduction_opt_space — custom descriptors
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionOptSpace, CustomDescriptors_OverrideTileRange) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );

    auto result = make_reduction_opt_space(red,
        power_2("tile_size", 128, 1024),   // custom: 4 levels
        default_tree_shape_dim(),           // keep default
        default_vec_width_dim()             // keep default
    );

    // 4 tiles × 3 tree shapes (active) × 4 vec widths (active) = 48
    EXPECT_EQ(result.space.cardinality(), 4u * 3u * 4u);
}

TEST(ReductionOptSpace, CustomDescriptors_CustomVecWidths) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );

    auto result = make_reduction_opt_space(red,
        default_tile_dim(),
        default_tree_shape_dim(),
        make_int_set("vec_width", {1, 2, 4, 8, 16, 32})  // 6 values
    );

    // 7 tiles × 3 tree shapes × 6 vec widths = 126
    EXPECT_EQ(result.space.cardinality(), 7u * 3u * 6u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 10: End-to-end integration
// algebra → properties → space → enumerate → encode
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionOptSpaceE2E, FullPipeline_AllActive) {
    // Step 1: Define a reduction (algebra layer)
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},            // count
        reduction_lane{identity_t{},    plus_fn{}, 0.0},          // sum
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()} // min
    );

    // Step 2: Extract properties
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_count, 3u);
    EXPECT_TRUE(props.all_associative);
    EXPECT_TRUE(props.all_have_identity);

    // Step 3: Generate search space
    auto result = make_reduction_opt_space(red);
    auto& space = result.space;
    auto& bridge = result.bridge;

    // Step 4: Verify space structure
    // 7 tiles × 3 tree shapes (active) × 4 vec widths (active) = 84
    EXPECT_EQ(space.cardinality(), 84u);

    // Step 5: Enumerate and encode every point
    std::size_t count = 0;
    space.enumerate([&](auto const& pt) {
        auto features = bridge.encode(pt);

        // Feature vector has the right width
        EXPECT_EQ(features.size(), bridge.num_features());

        // Tile dimension (log2 encoding) is always reasonable
        EXPECT_GE(features[0], 6.0);   // log2(64) = 6
        EXPECT_LE(features[0], 12.0);  // log2(4096) = 12

        // Tree shape (one-hot, slots 1-3) has exactly one 1.0
        double tree_sum = features[1] + features[2] + features[3];
        EXPECT_DOUBLE_EQ(tree_sum, 1.0);

        // Vec width (raw encoding, slot 4) is one of {1, 4, 8, 16}
        int vec = static_cast<int>(features[4]);
        EXPECT_TRUE(vec == 1 || vec == 4 || vec == 8 || vec == 16)
            << "Unexpected vec_width feature: " << vec;

        ++count;
    });
    EXPECT_EQ(count, 84u);
}

TEST(ReductionOptSpaceE2E, FullPipeline_TreeInactive) {
    // Non-associative reduction: tree shape should be inactive
    auto red = make_reduction(
        reduction_lane{identity_t{}, concat_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{},   0}
    );

    auto props = reduction_properties(red);
    EXPECT_FALSE(props.all_associative);  // concat_fn breaks it

    auto result = make_reduction_opt_space(red);
    auto& space = result.space;
    auto& bridge = result.bridge;

    // 7 tiles × 1 (inactive tree) × 1 (inactive vec — no has_identity for concat_fn) = 7
    EXPECT_EQ(space.cardinality(), 7u);

    // Feature width still 5 (stable encoding)
    EXPECT_EQ(bridge.num_features(), 5u);

    // Enumerate and verify encoding
    space.enumerate([&](auto const& pt) {
        auto features = bridge.encode(pt);

        // Tree shape (one-hot, slots 1-3) should be ALL ZEROS when inactive
        EXPECT_DOUBLE_EQ(features[1], 0.0) << "Inactive tree_shape should encode as 0";
        EXPECT_DOUBLE_EQ(features[2], 0.0) << "Inactive tree_shape should encode as 0";
        EXPECT_DOUBLE_EQ(features[3], 0.0) << "Inactive tree_shape should encode as 0";

        // Vec width slot should have the default value
        // (raw encoding of default = 1)
        EXPECT_DOUBLE_EQ(features[4], 1.0);
    });
}

TEST(ReductionOptSpaceE2E, FullPipeline_ReductionStillComputes) {
    // Verify the reduction itself still works (algebra layer not broken)
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()}
    );

    std::vector<double> data = {3.0, 1.0, 4.0, 1.0, 5.0};
    auto result = red.reduce(data);

    EXPECT_EQ(std::get<0>(result), 5);      // count
    EXPECT_DOUBLE_EQ(std::get<1>(result), 14.0);  // sum
    EXPECT_DOUBLE_EQ(std::get<2>(result), 1.0);   // min

    // Now generate the optimisation space for this same reduction
    auto opt = make_reduction_opt_space(red);
    EXPECT_EQ(opt.space.cardinality(), 84u);  // all dims active
}

} // namespace
