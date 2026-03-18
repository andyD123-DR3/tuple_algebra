// tests/test_conditional_dim.cpp
// Tests for ctdp/space/conditional.h
//
// Sprint 2, Task 2.2: Tests for conditional_dim
//
// Coverage:
//   1. Active conditional behaves identically to wrapped descriptor
//   2. Inactive conditional has cardinality 1
//   3. Feature encoding width is stable (same in active and inactive)
//   4. Feature values are correct when active
//   5. Feature values are all zeros when inactive
//   6. Works with each ordinate type (enum_vals, bool_flag, int_set, power_2)
//   7. Integration with descriptor_space (cardinality, enumeration, encoding)
//   8. Concept satisfaction
//   9. Factory helpers

#include "ctdp/space/conditional.h"
#include "ctdp/space/descriptor.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

namespace {

using namespace ctdp::space;

// ── Test enums ──────────────────────────────────────────────────────────────

enum class TreeShape : int { Flat, Binary, Pairwise, Canonical };
enum class Strategy : int { Generic, SWAR, Loop, Unrolled };

// ════════════════════════════════════════════════════════════════════════
// Section 1: Concept satisfaction
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, ConceptSatisfied_BoolFlag) {
    auto cd = conditional_dim(true, bool_flag("flag"));
    static_assert(dimension_descriptor<decltype(cd)>);
}

TEST(ConditionalDim, ConceptSatisfied_EnumVals) {
    auto cd = conditional_dim(true,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));
    static_assert(dimension_descriptor<decltype(cd)>);
}

TEST(ConditionalDim, ConceptSatisfied_IntSet) {
    auto cd = conditional_dim(true, make_int_set("vec", {1, 4, 8, 16}));
    static_assert(dimension_descriptor<decltype(cd)>);
}

TEST(ConditionalDim, ConceptSatisfied_Power2) {
    auto cd = conditional_dim(true, power_2("tile", 8, 64));
    static_assert(dimension_descriptor<decltype(cd)>);
}

TEST(ConditionalDim, ConceptSatisfied_PositiveInt) {
    auto cd = conditional_dim(true, positive_int("threads", 1, 8));
    static_assert(dimension_descriptor<decltype(cd)>);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: Active conditional_dim — identical to wrapped descriptor
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, Active_EnumVals_CardinalityMatchesWrapped) {
    auto base = make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                        TreeShape::Pairwise, TreeShape::Canonical});
    auto cd = conditional_dim(true, base);

    EXPECT_TRUE(cd.is_active());
    EXPECT_EQ(cd.cardinality(), base.cardinality());
    EXPECT_EQ(cd.cardinality(), 4u);
}

TEST(ConditionalDim, Active_EnumVals_EnumerationMatchesWrapped) {
    auto base = make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                        TreeShape::Pairwise, TreeShape::Canonical});
    auto cd = conditional_dim(true, base);

    for (std::size_t i = 0; i < cd.cardinality(); ++i) {
        EXPECT_EQ(cd.value_at(i), base.value_at(i));
    }
}

TEST(ConditionalDim, Active_EnumVals_ContainsMembership) {
    auto cd = conditional_dim(true,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    EXPECT_TRUE(cd.contains(TreeShape::Flat));
    EXPECT_TRUE(cd.contains(TreeShape::Binary));
    EXPECT_TRUE(cd.contains(TreeShape::Pairwise));
    EXPECT_TRUE(cd.contains(TreeShape::Canonical));
}

TEST(ConditionalDim, Active_EnumVals_IndexOfRoundTrips) {
    auto cd = conditional_dim(true,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    for (std::size_t i = 0; i < cd.cardinality(); ++i) {
        auto val = cd.value_at(i);
        EXPECT_EQ(cd.index_of(val), i);
    }
}

TEST(ConditionalDim, Active_IntSet_FullBehaviour) {
    auto base = make_int_set("vec", {1, 4, 8, 16});
    auto cd = conditional_dim(true, base);

    EXPECT_EQ(cd.cardinality(), 4u);
    EXPECT_EQ(cd.value_at(0), 1);
    EXPECT_EQ(cd.value_at(1), 4);
    EXPECT_EQ(cd.value_at(2), 8);
    EXPECT_EQ(cd.value_at(3), 16);
    EXPECT_TRUE(cd.contains(4));
    EXPECT_FALSE(cd.contains(2));
    EXPECT_EQ(cd.index_of(8), 2u);
}

TEST(ConditionalDim, Active_Power2_FullBehaviour) {
    auto cd = conditional_dim(true, power_2("tile", 8, 64));

    EXPECT_EQ(cd.cardinality(), 4u);  // 8, 16, 32, 64
    EXPECT_EQ(cd.value_at(0), 8);
    EXPECT_EQ(cd.value_at(3), 64);
    EXPECT_TRUE(cd.contains(32));
    EXPECT_FALSE(cd.contains(48));
}

TEST(ConditionalDim, Active_BoolFlag_FullBehaviour) {
    auto cd = conditional_dim(true, bool_flag("use_fma"));

    EXPECT_EQ(cd.cardinality(), 2u);
    EXPECT_EQ(cd.value_at(0), false);
    EXPECT_EQ(cd.value_at(1), true);
    EXPECT_TRUE(cd.contains(true));
    EXPECT_TRUE(cd.contains(false));
}

TEST(ConditionalDim, Active_PositiveInt_FullBehaviour) {
    auto cd = conditional_dim(true, positive_int("threads", 1, 4));

    EXPECT_EQ(cd.cardinality(), 4u);
    EXPECT_EQ(cd.value_at(0), 1);
    EXPECT_EQ(cd.value_at(3), 4);
    EXPECT_TRUE(cd.contains(3));
    EXPECT_FALSE(cd.contains(0));
    EXPECT_FALSE(cd.contains(5));
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: Inactive conditional_dim — cardinality 1, default value
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, Inactive_EnumVals_CardinalityIsOne) {
    auto cd = conditional_dim(false,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    EXPECT_FALSE(cd.is_active());
    EXPECT_EQ(cd.cardinality(), 1u);
}

TEST(ConditionalDim, Inactive_EnumVals_SingleDefaultValue) {
    auto cd = conditional_dim(false,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    // Inactive dimension has exactly one legal value: the first value of the wrapped descriptor
    EXPECT_EQ(cd.value_at(0), TreeShape::Flat);
    EXPECT_EQ(cd.default_value(), TreeShape::Flat);
}

TEST(ConditionalDim, Inactive_EnumVals_ContainsOnlyDefault) {
    auto cd = conditional_dim(false,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    EXPECT_TRUE(cd.contains(TreeShape::Flat));     // the default
    EXPECT_FALSE(cd.contains(TreeShape::Binary));   // not legal when inactive
    EXPECT_FALSE(cd.contains(TreeShape::Pairwise));
    EXPECT_FALSE(cd.contains(TreeShape::Canonical));
}

TEST(ConditionalDim, Inactive_EnumVals_IndexOfDefault) {
    auto cd = conditional_dim(false,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary}));

    EXPECT_EQ(cd.index_of(TreeShape::Flat), 0u);
    // Non-default value: index should be out-of-range (>= cardinality)
    EXPECT_GE(cd.index_of(TreeShape::Binary), cd.cardinality());
}

TEST(ConditionalDim, Inactive_IntSet_CardinalityIsOne) {
    auto cd = conditional_dim(false, make_int_set("vec", {1, 4, 8, 16}));

    EXPECT_EQ(cd.cardinality(), 1u);
    EXPECT_EQ(cd.value_at(0), 1);   // first value = default
    EXPECT_TRUE(cd.contains(1));
    EXPECT_FALSE(cd.contains(4));
}

TEST(ConditionalDim, Inactive_Power2_CardinalityIsOne) {
    auto cd = conditional_dim(false, power_2("tile", 8, 64));

    EXPECT_EQ(cd.cardinality(), 1u);
    EXPECT_EQ(cd.value_at(0), 8);
    EXPECT_TRUE(cd.contains(8));
    EXPECT_FALSE(cd.contains(16));
}

TEST(ConditionalDim, Inactive_BoolFlag_CardinalityIsOne) {
    auto cd = conditional_dim(false, bool_flag("use_fma"));

    EXPECT_EQ(cd.cardinality(), 1u);
    EXPECT_EQ(cd.value_at(0), false);
    EXPECT_TRUE(cd.contains(false));
    EXPECT_FALSE(cd.contains(true));
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: Feature encoding — width stability
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, FeatureWidth_SameWhenActiveAndInactive_EnumVals) {
    auto base = make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                        TreeShape::Pairwise, TreeShape::Canonical});
    auto active   = conditional_dim(true, base);
    auto inactive = conditional_dim(false, base);

    // one-hot encoding: width = number of categories = 4
    EXPECT_EQ(active.feature_width(), 4u);
    EXPECT_EQ(inactive.feature_width(), 4u);
    EXPECT_EQ(active.feature_width(), inactive.feature_width());
}

TEST(ConditionalDim, FeatureWidth_SameWhenActiveAndInactive_IntSet) {
    auto base = make_int_set("vec", {1, 4, 8, 16});
    auto active   = conditional_dim(true, base);
    auto inactive = conditional_dim(false, base);

    // raw encoding: width = 1
    EXPECT_EQ(active.feature_width(), 1u);
    EXPECT_EQ(inactive.feature_width(), 1u);
}

TEST(ConditionalDim, FeatureWidth_SameWhenActiveAndInactive_Power2) {
    auto base = power_2("tile", 8, 64);
    auto active   = conditional_dim(true, base);
    auto inactive = conditional_dim(false, base);

    // log2 encoding: width = 1
    EXPECT_EQ(active.feature_width(), 1u);
    EXPECT_EQ(inactive.feature_width(), 1u);
}

TEST(ConditionalDim, FeatureWidth_SameWhenActiveAndInactive_BoolFlag) {
    auto base = bool_flag("fma");
    auto active   = conditional_dim(true, base);
    auto inactive = conditional_dim(false, base);

    // binary encoding: width = 1
    EXPECT_EQ(active.feature_width(), 1u);
    EXPECT_EQ(inactive.feature_width(), 1u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 5: Feature encoding — values when active
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, FeatureValues_Active_EnumVals_OneHot) {
    auto cd = conditional_dim(true,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    std::vector<double> features(cd.feature_width(), -1.0);

    // Encode Flat → [1, 0, 0, 0]
    cd.write_features(TreeShape::Flat, features.data());
    EXPECT_DOUBLE_EQ(features[0], 1.0);
    EXPECT_DOUBLE_EQ(features[1], 0.0);
    EXPECT_DOUBLE_EQ(features[2], 0.0);
    EXPECT_DOUBLE_EQ(features[3], 0.0);

    // Encode Pairwise → [0, 0, 1, 0]
    cd.write_features(TreeShape::Pairwise, features.data());
    EXPECT_DOUBLE_EQ(features[0], 0.0);
    EXPECT_DOUBLE_EQ(features[1], 0.0);
    EXPECT_DOUBLE_EQ(features[2], 1.0);
    EXPECT_DOUBLE_EQ(features[3], 0.0);
}

TEST(ConditionalDim, FeatureValues_Active_IntSet_Raw) {
    auto cd = conditional_dim(true, make_int_set("vec", {1, 4, 8, 16}));

    double feature = -1.0;
    cd.write_features(8, &feature);
    EXPECT_DOUBLE_EQ(feature, 8.0);
}

TEST(ConditionalDim, FeatureValues_Active_Power2_Log2) {
    auto cd = conditional_dim(true, power_2("tile", 8, 64));

    double feature = -1.0;
    cd.write_features(32, &feature);
    EXPECT_DOUBLE_EQ(feature, 5.0);  // log2(32) = 5
}

TEST(ConditionalDim, FeatureValues_Active_BoolFlag_Binary) {
    auto cd = conditional_dim(true, bool_flag("fma"));

    double feature = -1.0;
    cd.write_features(true, &feature);
    EXPECT_DOUBLE_EQ(feature, 1.0);

    cd.write_features(false, &feature);
    EXPECT_DOUBLE_EQ(feature, 0.0);
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: Feature encoding — all zeros when inactive
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, FeatureValues_Inactive_EnumVals_AllZeros) {
    auto cd = conditional_dim(false,
        make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                TreeShape::Pairwise, TreeShape::Canonical}));

    std::vector<double> features(cd.feature_width(), -1.0);

    // Even when passed the default value, features should be all zeros
    cd.write_features(TreeShape::Flat, features.data());
    for (std::size_t i = 0; i < cd.feature_width(); ++i) {
        EXPECT_DOUBLE_EQ(features[i], 0.0) << "Feature " << i << " should be 0.0";
    }
}

TEST(ConditionalDim, FeatureValues_Inactive_IntSet_Zero) {
    auto cd = conditional_dim(false, make_int_set("vec", {1, 4, 8, 16}));

    double feature = -1.0;
    cd.write_features(1, &feature);  // default value
    EXPECT_DOUBLE_EQ(feature, 0.0);
}

TEST(ConditionalDim, FeatureValues_Inactive_Power2_Zero) {
    auto cd = conditional_dim(false, power_2("tile", 8, 64));

    double feature = -1.0;
    cd.write_features(8, &feature);  // default value
    EXPECT_DOUBLE_EQ(feature, 0.0);
}

TEST(ConditionalDim, FeatureValues_Inactive_BoolFlag_Zero) {
    auto cd = conditional_dim(false, bool_flag("fma"));

    double feature = -1.0;
    cd.write_features(false, &feature);  // default value
    EXPECT_DOUBLE_EQ(feature, 0.0);
}

// ════════════════════════════════════════════════════════════════════════
// Section 7: Integration with descriptor_space
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, InSpace_ActiveDimAffectsCardinality) {
    // Space with: power_2 (4 values) × conditional enum (4 values when active)
    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        conditional_dim(true,
            make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                    TreeShape::Pairwise, TreeShape::Canonical}))
    );

    // 4 tiles × 4 tree shapes = 16
    EXPECT_EQ(space.cardinality(), 16u);
}

TEST(ConditionalDim, InSpace_InactiveDimReducesCardinality) {
    // Same space, but tree shape is inactive
    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        conditional_dim(false,
            make_enum_vals("tree", {TreeShape::Flat, TreeShape::Binary,
                                    TreeShape::Pairwise, TreeShape::Canonical}))
    );

    // 4 tiles × 1 (inactive) = 4
    EXPECT_EQ(space.cardinality(), 4u);
}

TEST(ConditionalDim, InSpace_EnumerationCountMatchesCardinality) {
    auto space_active = descriptor_space("test_active",
        make_int_set("vec", {1, 4, 8}),
        conditional_dim(true, bool_flag("fma"))
    );

    auto space_inactive = descriptor_space("test_inactive",
        make_int_set("vec", {1, 4, 8}),
        conditional_dim(false, bool_flag("fma"))
    );

    // Active: 3 × 2 = 6
    std::size_t count_active = 0;
    space_active.enumerate([&](auto const&) { ++count_active; });
    EXPECT_EQ(count_active, 6u);
    EXPECT_EQ(count_active, space_active.cardinality());

    // Inactive: 3 × 1 = 3
    std::size_t count_inactive = 0;
    space_inactive.enumerate([&](auto const&) { ++count_inactive; });
    EXPECT_EQ(count_inactive, 3u);
    EXPECT_EQ(count_inactive, space_inactive.cardinality());
}

TEST(ConditionalDim, InSpace_InactiveEnumeratesOnlyDefaultValue) {
    auto space = descriptor_space("test",
        make_int_set("vec", {1, 4, 8}),
        conditional_dim(false, bool_flag("fma"))
    );

    // All enumerated points should have fma = false (the default)
    space.enumerate([&](auto const& pt) {
        EXPECT_EQ(std::get<1>(pt), false);
    });
}

TEST(ConditionalDim, InSpace_ActiveEnumeratesAllValues) {
    auto space = descriptor_space("test",
        make_int_set("vec", {1, 4}),
        conditional_dim(true, bool_flag("fma"))
    );

    std::vector<bool> fma_values;
    space.enumerate([&](auto const& pt) {
        fma_values.push_back(std::get<1>(pt));
    });

    // 2 vec × 2 fma = 4 points
    ASSERT_EQ(fma_values.size(), 4u);
    // Should contain both true and false
    EXPECT_NE(std::find(fma_values.begin(), fma_values.end(), true),
              fma_values.end());
    EXPECT_NE(std::find(fma_values.begin(), fma_values.end(), false),
              fma_values.end());
}

// ════════════════════════════════════════════════════════════════════════
// Section 8: Feature encoding stability in descriptor_space context
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, InSpace_FeatureBridgeWidth_StableAcrossActiveInactive) {
    auto base_tree = make_enum_vals("tree",
        {TreeShape::Flat, TreeShape::Binary, TreeShape::Pairwise, TreeShape::Canonical});

    auto space_active = descriptor_space("active",
        power_2("tile", 8, 64),
        conditional_dim(true, base_tree)
    );

    auto space_inactive = descriptor_space("inactive",
        power_2("tile", 8, 64),
        conditional_dim(false, base_tree)
    );

    auto bridge_active   = default_bridge(space_active);
    auto bridge_inactive = default_bridge(space_inactive);

    // Feature width must be identical regardless of active/inactive
    EXPECT_EQ(bridge_active.num_features(), bridge_inactive.num_features());

    // tile (log2 → 1 feature) + tree (one-hot → 4 features) = 5
    EXPECT_EQ(bridge_active.num_features(), 5u);
}

TEST(ConditionalDim, InSpace_FeatureBridgeEncoding_InactiveProducesZeros) {
    auto base_tree = make_enum_vals("tree",
        {TreeShape::Flat, TreeShape::Binary, TreeShape::Pairwise, TreeShape::Canonical});

    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        conditional_dim(false, base_tree)
    );

    auto bridge = default_bridge(space);
    auto pt = space.point_at(0);  // first point: tile=8, tree=Flat (default)

    auto features = bridge.encode(pt);

    // tile (log2(8)=3): should be 3.0
    EXPECT_DOUBLE_EQ(features[0], 3.0);

    // tree (inactive, one-hot): ALL ZEROS
    // The bridge uses encoding_card_of (= full wrapped cardinality = 4) for
    // feature width, and conditional_dim::index_of returns out-of-range when
    // inactive, so the one-hot loop writes all zeros.
    EXPECT_DOUBLE_EQ(features[1], 0.0);
    EXPECT_DOUBLE_EQ(features[2], 0.0);
    EXPECT_DOUBLE_EQ(features[3], 0.0);
    EXPECT_DOUBLE_EQ(features[4], 0.0);

    // Total width: tile(1) + tree(4) = 5
    EXPECT_EQ(features.size(), 5u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 9: Factory helpers
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, Factory_MakeConditional) {
    auto cd = make_conditional(true, bool_flag("flag"));
    EXPECT_TRUE(cd.is_active());
    EXPECT_EQ(cd.cardinality(), 2u);
}

TEST(ConditionalDim, Factory_AlwaysActive) {
    auto cd = always_active(make_int_set("vec", {1, 4, 8}));
    EXPECT_TRUE(cd.is_active());
    EXPECT_EQ(cd.cardinality(), 3u);
}

TEST(ConditionalDim, Factory_AlwaysInactive) {
    auto cd = always_inactive(make_int_set("vec", {1, 4, 8}));
    EXPECT_FALSE(cd.is_active());
    EXPECT_EQ(cd.cardinality(), 1u);
    EXPECT_EQ(cd.value_at(0), 1);
}

// ════════════════════════════════════════════════════════════════════════
// Section 10: Name and metadata preservation
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, NamePreserved_Active) {
    auto cd = conditional_dim(true, bool_flag("use_fma"));
    EXPECT_EQ(cd.get_name(), "use_fma");
}

TEST(ConditionalDim, NamePreserved_Inactive) {
    auto cd = conditional_dim(false, bool_flag("use_fma"));
    EXPECT_EQ(cd.get_name(), "use_fma");
}

TEST(ConditionalDim, EncodingPreserved) {
    auto cd_log = conditional_dim(true, power_2("tile", 8, 64));
    EXPECT_EQ(cd_log.default_encoding(), encoding_hint::log2);

    auto cd_hot = conditional_dim(false,
        make_enum_vals("strat", {Strategy::Generic, Strategy::SWAR}));
    EXPECT_EQ(cd_hot.default_encoding(), encoding_hint::one_hot);
}

TEST(ConditionalDim, WrappedDescriptorAccessible) {
    auto base = power_2("tile", 8, 64);
    auto cd = conditional_dim(true, base);

    // Can retrieve the wrapped descriptor
    auto const& inner = cd.descriptor();
    EXPECT_EQ(inner.lo, 8);
    EXPECT_EQ(inner.hi, 64);
    EXPECT_EQ(inner.cardinality(), 4u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 11: Realistic scenario — reduction properties gate dimensions
// ════════════════════════════════════════════════════════════════════════

TEST(ConditionalDim, ReductionScenario_AssocEnablesTreeShape) {
    // Simulates: if all lanes are associative, tree_shape dimension exists
    constexpr bool all_associative = true;

    auto space = descriptor_space("reduction_opt",
        power_2("tile", 64, 4096),
        conditional_dim(all_associative,
            make_enum_vals("tree_shape",
                {TreeShape::Flat, TreeShape::Binary,
                 TreeShape::Pairwise, TreeShape::Canonical}))
    );

    // 7 tile sizes (64,128,256,512,1024,2048,4096) × 4 tree shapes = 28
    EXPECT_EQ(space.cardinality(), 7u * 4u);

    // All tree shapes should appear in enumeration
    std::array<int, 4> tree_counts{};
    space.enumerate([&](auto const& pt) {
        auto shape = std::get<1>(pt);
        tree_counts[static_cast<std::size_t>(shape)]++;
    });
    for (int c : tree_counts) {
        EXPECT_EQ(c, 7);  // each shape appears once per tile size
    }
}

TEST(ConditionalDim, ReductionScenario_NonAssocDisablesTreeShape) {
    // Simulates: if NOT all lanes are associative, tree_shape is inactive
    constexpr bool all_associative = false;

    auto space = descriptor_space("reduction_opt",
        power_2("tile", 64, 4096),
        conditional_dim(all_associative,
            make_enum_vals("tree_shape",
                {TreeShape::Flat, TreeShape::Binary,
                 TreeShape::Pairwise, TreeShape::Canonical}))
    );

    // 7 tile sizes × 1 (inactive tree) = 7
    EXPECT_EQ(space.cardinality(), 7u);

    // Only the default tree shape (Flat) should appear
    space.enumerate([&](auto const& pt) {
        EXPECT_EQ(std::get<1>(pt), TreeShape::Flat);
    });
}

TEST(ConditionalDim, ReductionScenario_IdentityEnablesVecWidth) {
    // Simulates: if all lanes have identity, vector-width axis exists
    constexpr bool all_have_identity = true;

    auto space = descriptor_space("reduction_opt",
        power_2("tile", 64, 256),
        conditional_dim(all_have_identity,
            make_int_set("vec", {1, 4, 8, 16}))
    );

    // 3 tiles (64,128,256) × 4 vec widths = 12
    EXPECT_EQ(space.cardinality(), 3u * 4u);
}

TEST(ConditionalDim, ReductionScenario_NoIdentityDisablesVecWidth) {
    constexpr bool all_have_identity = false;

    auto space = descriptor_space("reduction_opt",
        power_2("tile", 64, 256),
        conditional_dim(all_have_identity,
            make_int_set("vec", {1, 4, 8, 16}))
    );

    // 3 tiles × 1 (inactive vec) = 3
    EXPECT_EQ(space.cardinality(), 3u);

    // Only default vec width (1) should appear
    space.enumerate([&](auto const& pt) {
        EXPECT_EQ(std::get<1>(pt), 1);
    });
}

TEST(ConditionalDim, ReductionScenario_MultipleConditionals) {
    // Both tree shape and vec width are conditional
    constexpr bool all_associative = true;
    constexpr bool all_have_identity = false;

    auto space = descriptor_space("reduction_opt",
        power_2("tile", 64, 256),
        conditional_dim(all_associative,
            make_enum_vals("tree_shape",
                {TreeShape::Flat, TreeShape::Binary, TreeShape::Canonical})),
        conditional_dim(all_have_identity,
            make_int_set("vec", {1, 4, 8}))
    );

    // 3 tiles × 3 tree shapes (active) × 1 vec (inactive) = 9
    EXPECT_EQ(space.cardinality(), 3u * 3u * 1u);
}

} // namespace
