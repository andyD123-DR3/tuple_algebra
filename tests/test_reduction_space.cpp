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

// ════════════════════════════════════════════════════════════════════════
// Sections 11-16: Extended properties (per-lane, type-safe, cost, fingerprint)
//
// These are in a separate anonymous namespace because they need
// additional helper types at namespace scope.
// ════════════════════════════════════════════════════════════════════════

namespace {

using namespace ct_dp::algebra;
using namespace ctdp::space;

// ════════════════════════════════════════════════════════════════════════
// Section 11: Type-safe aggregate properties
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, TypeSafe_IntLanes_ExactAssociative) {
    // Integer lanes: declares_associative_v<plus_fn, int> is true
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.all_associative);
    EXPECT_TRUE(props.all_exact_associative);
    EXPECT_TRUE(props.all_exact_commutative);
}

TEST(ReductionProperties, TypeSafe_DoubleLanes_NotExactAssociative) {
    // Double lanes: declares_associative_v<plus_fn, double> is false
    // (double is not an exact type)
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0.0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<double>::max()}
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.all_associative);        // abstract: yes
    EXPECT_FALSE(props.all_exact_associative);  // type-safe: no (double)
    EXPECT_FALSE(props.all_exact_commutative);
}

TEST(ReductionProperties, TypeSafe_MixedIntDouble_NotExact) {
    // Mix of int and double lanes — exact fails on double lanes
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},       // int: exact
        reduction_lane{identity_t{},    plus_fn{}, 0.0}      // double: not exact
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.all_associative);
    EXPECT_FALSE(props.all_exact_associative);  // broken by double lane
}

TEST(ReductionProperties, TypeSafe_PerLane_MixedTypes) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},       // int: exact
        reduction_lane{identity_t{}, plus_fn{}, 0.0}      // double: not exact
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.lane_exact_associative[0]);   // int lane
    EXPECT_FALSE(props.lane_exact_associative[1]);  // double lane
    EXPECT_TRUE(props.lane_exact_commutative[0]);
    EXPECT_FALSE(props.lane_exact_commutative[1]);
}

// ════════════════════════════════════════════════════════════════════════
// Section 12: Per-lane property arrays
// ════════════════════════════════════════════════════════════════════════

// Non-associative op at namespace scope (Clang requirement)
struct custom_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a + b; }
    // No declared properties at all
};

TEST(ReductionProperties, PerLane_MixedAssociativity) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{},   0},   // assoc
        reduction_lane{identity_t{}, custom_fn{}, 0},   // NOT assoc
        reduction_lane{identity_t{}, min_fn{},    std::numeric_limits<int>::max()} // assoc
    );
    auto props = reduction_properties(red);

    EXPECT_EQ(props.lane_count, 3u);
    EXPECT_TRUE(props.lane_associative[0]);
    EXPECT_FALSE(props.lane_associative[1]);
    EXPECT_TRUE(props.lane_associative[2]);

    EXPECT_FALSE(props.all_associative);
    EXPECT_TRUE(props.any_non_associative);
}

TEST(ReductionProperties, PerLane_IdentityPresence) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},          // has identity
        reduction_lane{identity_t{}, custom_fn{}, 0}         // no identity
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.lane_has_identity[0]);
    EXPECT_FALSE(props.lane_has_identity[1]);
}

TEST(ReductionProperties, PerLane_Idempotent) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},          // not idempotent
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}, // idempotent
        reduction_lane{identity_t{}, max_fn{},  std::numeric_limits<int>::lowest()} // idempotent
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.lane_idempotent[0]);
    EXPECT_TRUE(props.lane_idempotent[1]);
    EXPECT_TRUE(props.lane_idempotent[2]);
}

TEST(ReductionProperties, PerLane_IdentityTransform) {
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},     // not identity transform
        reduction_lane{identity_t{},    plus_fn{}, 0.0},   // identity transform
        reduction_lane{power_t<2>{},    plus_fn{}, 0.0}    // not identity transform
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.lane_identity_transform[0]);
    EXPECT_TRUE(props.lane_identity_transform[1]);
    EXPECT_FALSE(props.lane_identity_transform[2]);
}

TEST(ReductionProperties, PerLane_EntriesBeyondLaneCountAreZero) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);

    EXPECT_EQ(props.lane_count, 1u);
    // Beyond lane_count: should be default-initialised (false / 0)
    EXPECT_FALSE(props.lane_associative[1]);
    EXPECT_FALSE(props.lane_commutative[1]);
    EXPECT_EQ(props.lane_fingerprint[1], 0u);
    EXPECT_EQ(props.lane_transform_cost[1], transform_cost::free);
}

// ════════════════════════════════════════════════════════════════════════
// Section 13: Transform cost classification
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, TransformCost_Identity_IsFree) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::free);
}

TEST(ReductionProperties, TransformCost_Constant_IsFree) {
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::free);
}

TEST(ReductionProperties, TransformCost_Power2_IsCheap) {
    auto red = make_reduction(
        reduction_lane{power_t<2>{}, plus_fn{}, 0.0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::cheap);
}

TEST(ReductionProperties, TransformCost_Power3_IsCheap) {
    auto red = make_reduction(
        reduction_lane{power_t<3>{}, plus_fn{}, 0.0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::cheap);
}

TEST(ReductionProperties, TransformCost_Negate_IsExpensive) {
    auto red = make_reduction(
        reduction_lane{negate_t{}, plus_fn{}, 0.0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::expensive);
}

TEST(ReductionProperties, TransformCost_Abs_IsExpensive) {
    auto red = make_reduction(
        reduction_lane{abs_t{}, plus_fn{}, 0.0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::expensive);
}

TEST(ReductionProperties, TransformCost_StatsReduction_PerLane) {
    // count=constant(free), sum=identity(free), sumsq=power(cheap),
    // min=identity(free), max=identity(free)
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{power_t<2>{},    plus_fn{}, 0.0},
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()},
        reduction_lane{identity_t{},    max_fn{},  std::numeric_limits<double>::lowest()}
    );
    auto props = reduction_properties(red);

    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::free);     // constant
    EXPECT_EQ(props.lane_transform_cost[1], transform_cost::free);     // identity
    EXPECT_EQ(props.lane_transform_cost[2], transform_cost::cheap);    // power_t<2>
    EXPECT_EQ(props.lane_transform_cost[3], transform_cost::free);     // identity
    EXPECT_EQ(props.lane_transform_cost[4], transform_cost::free);     // identity
}

// ════════════════════════════════════════════════════════════════════════
// Section 14: Fusibility fingerprint
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, Fingerprint_SameOpsSameFingerprint) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{power_t<2>{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);

    // Both lanes use plus_fn (assoc+commut+identity, not idempotent)
    // Fingerprint only depends on reduce op properties, not transform
    EXPECT_EQ(props.lane_fingerprint[0], props.lane_fingerprint[1]);
    EXPECT_TRUE(props.fusible(0, 1));
}

TEST(ReductionProperties, Fingerprint_DifferentOps_DifferentFingerprint) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},          // assoc, commut, identity, NOT idemp
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()} // assoc, commut, identity, idemp
    );
    auto props = reduction_properties(red);

    // plus and min differ on idempotent → different fingerprint
    EXPECT_NE(props.lane_fingerprint[0], props.lane_fingerprint[1]);
    EXPECT_FALSE(props.fusible(0, 1));
}

TEST(ReductionProperties, Fingerprint_MinMaxSameFingerprint) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, min_fn{}, std::numeric_limits<int>::max()},
        reduction_lane{identity_t{}, max_fn{}, std::numeric_limits<int>::lowest()}
    );
    auto props = reduction_properties(red);

    // min and max: both assoc, commut, identity, idempotent → same fingerprint
    EXPECT_EQ(props.lane_fingerprint[0], props.lane_fingerprint[1]);
    EXPECT_TRUE(props.fusible(0, 1));
}

TEST(ReductionProperties, Fingerprint_NonAssocVsAssoc) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{},   0},
        reduction_lane{identity_t{}, custom_fn{}, 0}
    );
    auto props = reduction_properties(red);

    EXPECT_NE(props.lane_fingerprint[0], props.lane_fingerprint[1]);
    EXPECT_FALSE(props.fusible(0, 1));
}

TEST(ReductionProperties, Fingerprint_Encoding) {
    // plus_fn: assoc=1, commut=1, identity=1, idempotent=0 → 0b0111 = 7
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);
    EXPECT_EQ(props.lane_fingerprint[0], 7u);

    // min_fn: assoc=1, commut=1, identity=1, idempotent=1 → 0b1111 = 15
    auto red2 = make_reduction(
        reduction_lane{identity_t{}, min_fn{}, std::numeric_limits<int>::max()}
    );
    auto props2 = reduction_properties(red2);
    EXPECT_EQ(props2.lane_fingerprint[0], 15u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 15: Fusibility convenience queries
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, AllFusible_HomogeneousOps) {
    auto red = make_reduction(
        reduction_lane{identity_t{},    plus_fn{}, 0},
        reduction_lane{power_t<2>{},    plus_fn{}, 0},
        reduction_lane{constant_t<1>{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);

    EXPECT_TRUE(props.all_fusible());
    EXPECT_EQ(props.num_fingerprint_classes(), 1u);
}

TEST(ReductionProperties, AllFusible_HeterogeneousOps_False) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    EXPECT_FALSE(props.all_fusible());
    EXPECT_EQ(props.num_fingerprint_classes(), 2u);
}

TEST(ReductionProperties, NumFingerprintClasses_ThreeDistinct) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{},   0},          // assoc+commut+ident
        reduction_lane{identity_t{}, min_fn{},    std::numeric_limits<int>::max()}, // +idempotent
        reduction_lane{identity_t{}, custom_fn{}, 0}           // none
    );
    auto props = reduction_properties(red);

    EXPECT_EQ(props.num_fingerprint_classes(), 3u);
}

TEST(ReductionProperties, Fusible_OutOfBounds_ReturnsFalse) {
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);

    // In-range: same op → fusible
    EXPECT_TRUE(props.fusible(0, 1));
    // Out of range: beyond lane_count → false, not UB
    EXPECT_FALSE(props.fusible(0, 2));
    EXPECT_FALSE(props.fusible(2, 0));
    EXPECT_FALSE(props.fusible(15, 15));
}

// ════════════════════════════════════════════════════════════════════════
// Section 16: Stats reduction — full property check
// ════════════════════════════════════════════════════════════════════════

TEST(ReductionProperties, StatsReduction_FullPropertyInventory) {
    // count=constant/plus, sum=identity/plus, sumsq=power2/plus,
    // min=identity/min, max=identity/max
    auto red = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{power_t<2>{},    plus_fn{}, 0.0},
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()},
        reduction_lane{identity_t{},    max_fn{},  std::numeric_limits<double>::lowest()}
    );
    auto props = reduction_properties(red);

    // Aggregate
    EXPECT_TRUE(props.all_associative);
    EXPECT_TRUE(props.all_commutative);
    EXPECT_TRUE(props.all_have_identity);
    EXPECT_FALSE(props.all_idempotent);       // plus is not idempotent

    // Type-safe: lanes 0 is int (exact), lanes 1-4 are double (not exact)
    EXPECT_FALSE(props.all_exact_associative);
    EXPECT_TRUE(props.lane_exact_associative[0]);   // int plus
    EXPECT_FALSE(props.lane_exact_associative[1]);  // double plus
    EXPECT_FALSE(props.lane_exact_associative[2]);  // double plus

    // Per-lane associativity (all true — abstract)
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(props.lane_associative[i]) << "lane " << i;
        EXPECT_TRUE(props.lane_commutative[i]) << "lane " << i;
        EXPECT_TRUE(props.lane_has_identity[i]) << "lane " << i;
    }

    // Idempotent: lanes 3,4 (min, max) are idempotent
    EXPECT_FALSE(props.lane_idempotent[0]);
    EXPECT_FALSE(props.lane_idempotent[1]);
    EXPECT_FALSE(props.lane_idempotent[2]);
    EXPECT_TRUE(props.lane_idempotent[3]);
    EXPECT_TRUE(props.lane_idempotent[4]);

    // Fingerprint: plus lanes share one, min/max lanes share another
    EXPECT_EQ(props.lane_fingerprint[0], props.lane_fingerprint[1]);  // both plus
    EXPECT_EQ(props.lane_fingerprint[1], props.lane_fingerprint[2]);  // both plus
    EXPECT_EQ(props.lane_fingerprint[3], props.lane_fingerprint[4]);  // both min/max
    EXPECT_NE(props.lane_fingerprint[0], props.lane_fingerprint[3]);  // plus ≠ min

    // Fusibility
    EXPECT_TRUE(props.fusible(0, 1));   // plus + plus
    EXPECT_TRUE(props.fusible(3, 4));   // min + max
    EXPECT_FALSE(props.fusible(0, 3));  // plus ≠ min
    EXPECT_EQ(props.num_fingerprint_classes(), 2u);

    // Transform cost
    EXPECT_EQ(props.lane_transform_cost[0], transform_cost::free);   // constant
    EXPECT_EQ(props.lane_transform_cost[1], transform_cost::free);   // identity
    EXPECT_EQ(props.lane_transform_cost[2], transform_cost::cheap);  // power_t<2>
    EXPECT_EQ(props.lane_transform_cost[3], transform_cost::free);   // identity
    EXPECT_EQ(props.lane_transform_cost[4], transform_cost::free);   // identity
}

} // namespace