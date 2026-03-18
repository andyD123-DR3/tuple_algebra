// tests/test_partition.cpp
// Tests for ctdp/space/partition.h — partition ordinate type
//
// Sprint 3: partition(N) ordinate type with canonical enumeration,
//           pairwise co-membership encoding, and legality filtering.
//
// Coverage:
//   Section 1: Bell numbers and cardinality
//   Section 2: partition_value — structure and queries
//   Section 3: Canonical form (restricted growth string)
//   Section 4: Ranking and unranking round-trips
//   Section 5: First/last values and boundary cases
//   Section 6: Feature encoding — pairwise co-membership
//   Section 7: Integration with descriptor_space and feature_bridge
//   Section 8: Integration with conditional_dim
//   Section 9: Legality filtering with reduction_properties fingerprint
//   Section 10: constexpr verification
//   Section 11: Dimension descriptor concept satisfaction
//   Section 12: Degenerate cases (N=1, N=2)

#include "ctdp/space/partition.h"
#include "ctdp/space/descriptor.h"
#include "ctdp/space/conditional.h"
#include "ctdp/space/reduction_properties.h"
#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <set>
#include <vector>

namespace {

using namespace ctdp::space;

// ════════════════════════════════════════════════════════════════════════
// Section 1: Bell numbers and cardinality
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, BellNumbers_SmallValues) {
    EXPECT_EQ(bell_number<1>, 1u);
    EXPECT_EQ(bell_number<2>, 2u);
    EXPECT_EQ(bell_number<3>, 5u);
    EXPECT_EQ(bell_number<4>, 15u);
    EXPECT_EQ(bell_number<5>, 52u);
    EXPECT_EQ(bell_number<6>, 203u);
    EXPECT_EQ(bell_number<7>, 877u);
    EXPECT_EQ(bell_number<8>, 4140u);
}

TEST(Partition, Cardinality_MatchesBell) {
    EXPECT_EQ(make_partition<1>("g").cardinality(), 1u);
    EXPECT_EQ(make_partition<2>("g").cardinality(), 2u);
    EXPECT_EQ(make_partition<3>("g").cardinality(), 5u);
    EXPECT_EQ(make_partition<5>("g").cardinality(), 52u);
    EXPECT_EQ(make_partition<8>("g").cardinality(), 4140u);
}

TEST(Partition, Bell12_PracticalCeiling) {
    EXPECT_EQ(make_partition<12>("g").cardinality(), 4213597u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: partition_value — structure and queries
// ════════════════════════════════════════════════════════════════════════

TEST(PartitionValue, NumGroups) {
    partition_value<5> all_one{};
    all_one.labels = {0, 0, 0, 0, 0};
    EXPECT_EQ(all_one.num_groups(), 1u);

    partition_value<5> singletons{};
    singletons.labels = {0, 1, 2, 3, 4};
    EXPECT_EQ(singletons.num_groups(), 5u);

    partition_value<5> two_groups{};
    two_groups.labels = {0, 0, 0, 1, 1};
    EXPECT_EQ(two_groups.num_groups(), 2u);

    partition_value<5> three_groups{};
    three_groups.labels = {0, 0, 1, 1, 2};
    EXPECT_EQ(three_groups.num_groups(), 3u);
}

TEST(PartitionValue, SameGroup) {
    partition_value<5> pv{};
    pv.labels = {0, 0, 1, 1, 2};

    EXPECT_TRUE(pv.same_group(0, 1));
    EXPECT_TRUE(pv.same_group(2, 3));
    EXPECT_FALSE(pv.same_group(0, 2));
    EXPECT_FALSE(pv.same_group(0, 4));
    EXPECT_FALSE(pv.same_group(2, 4));
    EXPECT_TRUE(pv.same_group(0, 0));  // self
}

TEST(PartitionValue, Equality) {
    partition_value<3> a{};
    a.labels = {0, 0, 1};
    partition_value<3> b{};
    b.labels = {0, 0, 1};
    partition_value<3> c{};
    c.labels = {0, 1, 0};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(PartitionValue, Ordering) {
    partition_value<3> a{};
    a.labels = {0, 0, 0};
    partition_value<3> b{};
    b.labels = {0, 0, 1};

    EXPECT_LT(a, b);
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: Canonical form (restricted growth string)
// ════════════════════════════════════════════════════════════════════════

TEST(PartitionValue, IsCanonical_Valid) {
    partition_value<4> a{};
    a.labels = {0, 0, 1, 2};
    EXPECT_TRUE(a.is_canonical());

    partition_value<4> b{};
    b.labels = {0, 1, 0, 2};
    EXPECT_TRUE(b.is_canonical());

    partition_value<4> c{};
    c.labels = {0, 0, 0, 0};
    EXPECT_TRUE(c.is_canonical());
}

TEST(PartitionValue, IsCanonical_Invalid) {
    partition_value<4> a{};
    a.labels = {0, 2, 1, 0};  // 2 appears before 1
    EXPECT_FALSE(a.is_canonical());

    partition_value<4> b{};
    b.labels = {1, 0, 0, 0};  // first element must be 0
    EXPECT_FALSE(b.is_canonical());
}

TEST(Partition, AllEnumerated_AreCanonical) {
    auto p = make_partition<6>("g");
    for (std::size_t i = 0; i < p.cardinality(); ++i) {
        auto val = p.value_at(i);
        EXPECT_TRUE(val.is_canonical()) << "Partition " << i << " is not canonical";
    }
}

TEST(Partition, AllEnumerated_AreDistinct) {
    auto p = make_partition<5>("g");
    std::set<std::vector<std::uint8_t>> seen;
    for (std::size_t i = 0; i < p.cardinality(); ++i) {
        auto val = p.value_at(i);
        std::vector<std::uint8_t> v(val.labels.begin(), val.labels.end());
        EXPECT_TRUE(seen.insert(v).second)
            << "Duplicate partition at index " << i;
    }
    EXPECT_EQ(seen.size(), 52u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: Ranking and unranking round-trips
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, RoundTrip_N3) {
    auto p = make_partition<3>("g");
    for (std::size_t i = 0; i < p.cardinality(); ++i) {
        auto val = p.value_at(i);
        EXPECT_EQ(p.index_of(val), i) << "Round-trip failed at index " << i;
    }
}

TEST(Partition, RoundTrip_N5) {
    auto p = make_partition<5>("g");
    for (std::size_t i = 0; i < p.cardinality(); ++i) {
        auto val = p.value_at(i);
        EXPECT_EQ(p.index_of(val), i) << "Round-trip failed at index " << i;
    }
}

TEST(Partition, RoundTrip_N8) {
    auto p = make_partition<8>("g");
    for (std::size_t i = 0; i < p.cardinality(); ++i) {
        auto val = p.value_at(i);
        EXPECT_EQ(p.index_of(val), i) << "Round-trip failed at index " << i;
    }
}

TEST(Partition, IndexOf_NonCanonical_ReturnsOutOfRange) {
    auto p = make_partition<4>("g");
    partition_value<4> bad{};
    bad.labels = {0, 2, 1, 0};  // not canonical
    EXPECT_GE(p.index_of(bad), p.cardinality());
}

TEST(Partition, Contains_Canonical) {
    auto p = make_partition<3>("g");
    partition_value<3> good{};
    good.labels = {0, 0, 1};
    EXPECT_TRUE(p.contains(good));
}

TEST(Partition, Contains_NonCanonical) {
    auto p = make_partition<3>("g");
    partition_value<3> bad{};
    bad.labels = {1, 0, 0};
    EXPECT_FALSE(p.contains(bad));
}

// ════════════════════════════════════════════════════════════════════════
// Section 5: First/last values and boundary cases
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, FirstValue_AllOneGroup) {
    auto p = make_partition<5>("g");
    auto first = p.value_at(0);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(first.labels[i], 0u);
    }
    EXPECT_EQ(first.num_groups(), 1u);
}

TEST(Partition, LastValue_AllSingletons) {
    auto p = make_partition<5>("g");
    auto last = p.value_at(p.cardinality() - 1);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(last.labels[i], static_cast<std::uint8_t>(i));
    }
    EXPECT_EQ(last.num_groups(), 5u);
}

TEST(Partition, N3_AllFivePartitions) {
    auto p = make_partition<3>("g");
    // Bell(3) = 5:
    // [0,0,0] [0,0,1] [0,1,0] [0,1,1] [0,1,2]
    auto v0 = p.value_at(0);
    EXPECT_EQ(v0.labels[0], 0u); EXPECT_EQ(v0.labels[1], 0u); EXPECT_EQ(v0.labels[2], 0u);

    auto v1 = p.value_at(1);
    EXPECT_EQ(v1.labels[0], 0u); EXPECT_EQ(v1.labels[1], 0u); EXPECT_EQ(v1.labels[2], 1u);

    auto v2 = p.value_at(2);
    EXPECT_EQ(v2.labels[0], 0u); EXPECT_EQ(v2.labels[1], 1u); EXPECT_EQ(v2.labels[2], 0u);

    auto v3 = p.value_at(3);
    EXPECT_EQ(v3.labels[0], 0u); EXPECT_EQ(v3.labels[1], 1u); EXPECT_EQ(v3.labels[2], 1u);

    auto v4 = p.value_at(4);
    EXPECT_EQ(v4.labels[0], 0u); EXPECT_EQ(v4.labels[1], 1u); EXPECT_EQ(v4.labels[2], 2u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: Feature encoding — pairwise co-membership
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, FeatureWidth) {
    EXPECT_EQ(make_partition<1>("g").feature_width(), 0u);   // 0 pairs
    EXPECT_EQ(make_partition<2>("g").feature_width(), 1u);   // 1 pair
    EXPECT_EQ(make_partition<3>("g").feature_width(), 3u);   // 3 pairs
    EXPECT_EQ(make_partition<5>("g").feature_width(), 10u);  // 10 pairs
    EXPECT_EQ(make_partition<12>("g").feature_width(), 66u); // 66 pairs
}

TEST(Partition, WriteFeatures_AllOneGroup) {
    auto p = make_partition<4>("g");
    auto val = p.value_at(0);  // [0,0,0,0]

    std::vector<double> f(p.feature_width(), -1.0);
    p.write_features(val, f.data());

    // All pairs co-member → all 1.0
    for (std::size_t i = 0; i < f.size(); ++i)
        EXPECT_DOUBLE_EQ(f[i], 1.0) << "Feature " << i;
}

TEST(Partition, WriteFeatures_AllSingletons) {
    auto p = make_partition<4>("g");
    auto val = p.value_at(p.cardinality() - 1);  // [0,1,2,3]

    std::vector<double> f(p.feature_width(), -1.0);
    p.write_features(val, f.data());

    // No pairs co-member → all 0.0
    for (std::size_t i = 0; i < f.size(); ++i)
        EXPECT_DOUBLE_EQ(f[i], 0.0) << "Feature " << i;
}

TEST(Partition, WriteFeatures_SpecificPartition) {
    auto p = make_partition<5>("g");

    // [0,0,1,1,2]: pairs (0,1) and (2,3) co-member
    partition_value<5> pv{};
    pv.labels = {0, 0, 1, 1, 2};

    std::vector<double> f(10, -1.0);
    p.write_features(pv, f.data());

    // Pair order: (0,1) (0,2) (0,3) (0,4) (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
    EXPECT_DOUBLE_EQ(f[0], 1.0);  // (0,1) same group
    EXPECT_DOUBLE_EQ(f[1], 0.0);  // (0,2)
    EXPECT_DOUBLE_EQ(f[2], 0.0);  // (0,3)
    EXPECT_DOUBLE_EQ(f[3], 0.0);  // (0,4)
    EXPECT_DOUBLE_EQ(f[4], 0.0);  // (1,2)
    EXPECT_DOUBLE_EQ(f[5], 0.0);  // (1,3)
    EXPECT_DOUBLE_EQ(f[6], 0.0);  // (1,4)
    EXPECT_DOUBLE_EQ(f[7], 1.0);  // (2,3) same group
    EXPECT_DOUBLE_EQ(f[8], 0.0);  // (2,4)
    EXPECT_DOUBLE_EQ(f[9], 0.0);  // (3,4)
}

// ════════════════════════════════════════════════════════════════════════
// Section 7: Integration with descriptor_space and feature_bridge
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, InSpace_Cardinality) {
    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        make_partition<3>("grouping")
    );
    // 4 tiles × Bell(3)=5 = 20
    EXPECT_EQ(space.cardinality(), 20u);
}

TEST(Partition, InSpace_EnumerationCount) {
    auto space = descriptor_space("test",
        make_partition<4>("grouping")
    );
    std::size_t count = 0;
    space.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, 15u);
    EXPECT_EQ(count, space.cardinality());
}

TEST(Partition, InSpace_BridgeFeatureWidth) {
    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        make_partition<3>("grouping")
    );
    auto bridge = default_bridge(space);
    // tile(log2:1) + partition(pairwise:3) = 4
    EXPECT_EQ(bridge.num_features(), 4u);
}

TEST(Partition, InSpace_BridgeEncoding) {
    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        make_partition<3>("grouping")
    );
    auto bridge = default_bridge(space);

    partition_value<3> pv{};
    pv.labels = {0, 0, 1};  // items 0,1 together; item 2 alone

    auto features = bridge.encode(std::tuple{32, pv});
    EXPECT_EQ(features.size(), 4u);
    EXPECT_DOUBLE_EQ(features[0], 5.0);  // log2(32)
    EXPECT_DOUBLE_EQ(features[1], 1.0);  // pair (0,1) co-member
    EXPECT_DOUBLE_EQ(features[2], 0.0);  // pair (0,2)
    EXPECT_DOUBLE_EQ(features[3], 0.0);  // pair (1,2)
}

TEST(Partition, InSpace_PointAtRoundTrip) {
    auto space = descriptor_space("test",
        make_partition<3>("grouping"),
        bool_flag("turbo")
    );
    // Bell(3)=5 × 2 = 10
    EXPECT_EQ(space.cardinality(), 10u);

    for (std::size_t i = 0; i < space.cardinality(); ++i) {
        auto pt = space.point_at(i);
        (void)pt;  // just verify no crash
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 8: Integration with conditional_dim
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, ConditionalDim_Active) {
    auto space = descriptor_space("test",
        conditional_dim(true, make_partition<3>("grouping")),
        power_2("tile", 8, 32)
    );
    // Bell(3)=5 × 3 tiles = 15
    EXPECT_EQ(space.cardinality(), 15u);
}

TEST(Partition, ConditionalDim_Inactive) {
    auto space = descriptor_space("test",
        conditional_dim(false, make_partition<3>("grouping")),
        power_2("tile", 8, 32)
    );
    // 1 (inactive) × 3 tiles = 3
    EXPECT_EQ(space.cardinality(), 3u);
}

TEST(Partition, ConditionalDim_FeatureWidthStable) {
    auto part_dim = make_partition<4>("grouping");

    auto space_on = descriptor_space("on",
        conditional_dim(true, part_dim));
    auto space_off = descriptor_space("off",
        conditional_dim(false, part_dim));

    auto b_on  = default_bridge(space_on);
    auto b_off = default_bridge(space_off);

    // Both must have same width: partition(4) → 6 pairwise features
    EXPECT_EQ(b_on.num_features(), b_off.num_features());
    EXPECT_EQ(b_on.num_features(), 6u);
}

TEST(Partition, ConditionalDim_InactiveEncodesZeros) {
    auto space = descriptor_space("test",
        conditional_dim(false, make_partition<3>("grouping"))
    );
    auto bridge = default_bridge(space);

    auto pt = space.point_at(0);
    auto features = bridge.encode(pt);
    // All pairwise features should be 0.0 (inactive)
    EXPECT_EQ(features.size(), 3u);
    for (std::size_t i = 0; i < features.size(); ++i)
        EXPECT_DOUBLE_EQ(features[i], 0.0) << "Feature " << i;
}

// ════════════════════════════════════════════════════════════════════════
// Section 9: Legality filtering with reduction_properties fingerprint
// ════════════════════════════════════════════════════════════════════════

// Non-associative op at namespace scope (Clang requirement)
struct noassoc_fn {
    template<typename T>
    constexpr T operator()(T a, T b) const noexcept { return a + b; }
};

TEST(Partition, LegalityFilter_PlusAndMin) {
    using namespace ct_dp::algebra;

    // 3 lanes: plus, plus, min
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    auto part = make_partition<3>("grouping");

    std::size_t legal = 0;
    for (std::size_t i = 0; i < part.cardinality(); ++i) {
        auto pv = part.value_at(i);
        bool ok = true;
        for (std::size_t a = 0; a < 3 && ok; ++a)
            for (std::size_t b = a + 1; b < 3 && ok; ++b)
                if (pv.same_group(a, b) && !props.fusible(a, b))
                    ok = false;
        if (ok) ++legal;
    }

    // Legal: {0,1}+{2} and {0}+{1}+{2}
    EXPECT_EQ(legal, 2u);
}

TEST(Partition, LegalityFilter_AllSameFingerprint) {
    using namespace ct_dp::algebra;

    // 3 lanes all plus — all fusible
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    auto props = reduction_properties(red);
    EXPECT_TRUE(props.all_fusible());

    auto part = make_partition<3>("grouping");

    // All 5 partitions should be legal
    std::size_t legal = 0;
    for (std::size_t i = 0; i < part.cardinality(); ++i) {
        auto pv = part.value_at(i);
        bool ok = true;
        for (std::size_t a = 0; a < 3 && ok; ++a)
            for (std::size_t b = a + 1; b < 3 && ok; ++b)
                if (pv.same_group(a, b) && !props.fusible(a, b))
                    ok = false;
        if (ok) ++legal;
    }
    EXPECT_EQ(legal, 5u);
}

TEST(Partition, LegalityFilter_AllDifferentFingerprint) {
    using namespace ct_dp::algebra;

    // 3 lanes: plus, min, noassoc — all different fingerprints
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{},    0},
        reduction_lane{identity_t{}, min_fn{},     std::numeric_limits<int>::max()},
        reduction_lane{identity_t{}, noassoc_fn{}, 0}
    );
    auto props = reduction_properties(red);

    auto part = make_partition<3>("grouping");

    // Only singletons legal: {0}+{1}+{2}
    std::size_t legal = 0;
    for (std::size_t i = 0; i < part.cardinality(); ++i) {
        auto pv = part.value_at(i);
        bool ok = true;
        for (std::size_t a = 0; a < 3 && ok; ++a)
            for (std::size_t b = a + 1; b < 3 && ok; ++b)
                if (pv.same_group(a, b) && !props.fusible(a, b))
                    ok = false;
        if (ok) ++legal;
    }
    EXPECT_EQ(legal, 1u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 10: constexpr verification
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, Constexpr_BellNumbers) {
    static_assert(bell_number<1> == 1);
    static_assert(bell_number<5> == 52);
    static_assert(bell_number<8> == 4140);
}

TEST(Partition, Constexpr_ValueAtAndIndexOf) {
    constexpr auto p = make_partition<4>("g");
    static_assert(p.cardinality() == 15);
    constexpr auto first = p.value_at(0);
    static_assert(first.labels[0] == 0);
    static_assert(first.num_groups() == 1);
    constexpr auto idx = p.index_of(first);
    static_assert(idx == 0);
}

TEST(Partition, Constexpr_FeatureWidth) {
    constexpr auto p = make_partition<5>("g");
    static_assert(p.feature_width() == 10);
}

// ════════════════════════════════════════════════════════════════════════
// Section 11: Dimension descriptor concept
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, ConceptSatisfied) {
    static_assert(dimension_descriptor<partition_desc<3>>);
    static_assert(dimension_descriptor<partition_desc<5>>);
    static_assert(dimension_descriptor<partition_desc<12>>);
}

// ════════════════════════════════════════════════════════════════════════
// Section 12: Degenerate cases
// ════════════════════════════════════════════════════════════════════════

TEST(Partition, N1_SinglePartition) {
    auto p = make_partition<1>("g");
    EXPECT_EQ(p.cardinality(), 1u);
    EXPECT_EQ(p.feature_width(), 0u);  // no pairs

    auto val = p.value_at(0);
    EXPECT_EQ(val.labels[0], 0u);
    EXPECT_EQ(val.num_groups(), 1u);
    EXPECT_EQ(p.index_of(val), 0u);
}

TEST(Partition, N2_TwoPartitions) {
    auto p = make_partition<2>("g");
    EXPECT_EQ(p.cardinality(), 2u);
    EXPECT_EQ(p.feature_width(), 1u);  // one pair

    // [0,0] — together
    auto v0 = p.value_at(0);
    EXPECT_EQ(v0.labels[0], 0u);
    EXPECT_EQ(v0.labels[1], 0u);
    EXPECT_TRUE(v0.same_group(0, 1));

    // [0,1] — apart
    auto v1 = p.value_at(1);
    EXPECT_EQ(v1.labels[0], 0u);
    EXPECT_EQ(v1.labels[1], 1u);
    EXPECT_FALSE(v1.same_group(0, 1));

    // Features
    std::vector<double> f(1);
    p.write_features(v0, f.data());
    EXPECT_DOUBLE_EQ(f[0], 1.0);  // together

    p.write_features(v1, f.data());
    EXPECT_DOUBLE_EQ(f[0], 0.0);  // apart
}

} // namespace
