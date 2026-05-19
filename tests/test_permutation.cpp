// tests/test_permutation.cpp
// Canonical permutation descriptor integration tests

#include "ctdp/space/conditional.h"
#include "ctdp/space/descriptor.h"
#include "ctdp/space/permutation.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <set>
#include <vector>

namespace {

using namespace ctdp::space;

TEST(PermutationValue, IdentityAndValidity) {
    permutation_value<4> v;
    EXPECT_TRUE(v.is_valid());
    EXPECT_TRUE(v.is_identity());
    EXPECT_EQ(v[0], 0u);
    EXPECT_EQ(v[3], 3u);
}

TEST(PermutationValue, InvalidCases) {
    permutation_value<4> dup(std::array<std::size_t, 4>{0, 0, 2, 3});
    permutation_value<4> oor(std::array<std::size_t, 4>{0, 1, 2, 5});
    EXPECT_FALSE(dup.is_valid());
    EXPECT_FALSE(oor.is_valid());
}

TEST(PermutationDesc, Cardinality) {
    EXPECT_EQ(make_permutation<1>("p").cardinality(), 1u);
    EXPECT_EQ(make_permutation<2>("p").cardinality(), 2u);
    EXPECT_EQ(make_permutation<3>("p").cardinality(), 6u);
    EXPECT_EQ(make_permutation<4>("p").cardinality(), 24u);
    EXPECT_EQ(make_permutation<5>("p").cardinality(), 120u);
}

TEST(PermutationDesc, RankUnrankRoundTrip) {
    auto desc = make_permutation<4>("order");
    for (std::size_t i = 0; i < desc.cardinality(); ++i) {
        auto v = desc.value_at(i);
        EXPECT_TRUE(v.is_valid()) << "invalid at index " << i;
        EXPECT_EQ(desc.index_of(v), i) << "round-trip failed at " << i;
    }
}

TEST(PermutationDesc, IdentityAndReverseEnumeration) {
    auto desc = make_permutation<4>("order");
    auto first = desc.value_at(0);
    auto last = desc.value_at(desc.cardinality() - 1);

    EXPECT_TRUE(first.is_identity());
    EXPECT_EQ(last[0], 3u);
    EXPECT_EQ(last[1], 2u);
    EXPECT_EQ(last[2], 1u);
    EXPECT_EQ(last[3], 0u);
}

TEST(PermutationDesc, FeatureWidth) {
    EXPECT_EQ(make_permutation<1>("p").feature_width(), 0u);
    EXPECT_EQ(make_permutation<4>("p").feature_width(), 6u);
    EXPECT_EQ(make_permutation<5>("p").feature_width(), 10u);
}

TEST(PermutationDesc, PrecedenceEncodingIdentityAndReverse) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> identity;
    permutation_value<4> reverse(std::array<std::size_t, 4>{3, 2, 1, 0});

    double fi[6]{};
    double fr[6]{};
    desc.write_features(identity, fi);
    desc.write_features(reverse, fr);

    for (double x : fi) EXPECT_DOUBLE_EQ(x, 1.0);
    for (double x : fr) EXPECT_DOUBLE_EQ(x, 0.0);
}

TEST(PermutationDesc, PrecedenceEncodingKnownPermutation) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> v(std::array<std::size_t, 4>{2, 0, 3, 1});
    double f[6]{};
    desc.write_features(v, f);

    EXPECT_DOUBLE_EQ(f[0], 1.0); // (0,1)
    EXPECT_DOUBLE_EQ(f[1], 0.0); // (0,2)
    EXPECT_DOUBLE_EQ(f[2], 1.0); // (0,3)
    EXPECT_DOUBLE_EQ(f[3], 0.0); // (1,2)
    EXPECT_DOUBLE_EQ(f[4], 0.0); // (1,3)
    EXPECT_DOUBLE_EQ(f[5], 1.0); // (2,3)
}

TEST(PermutationDesc, NeighboursAreAllSwapPairs) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> id;
    std::set<permutation_value<4>> seen;

    desc.neighbours(id, [&](auto const& nb) { seen.insert(nb); });
    EXPECT_EQ(seen.size(), 6u); // 4 choose 2
    EXPECT_EQ(seen.count(id), 0u);
}

TEST(PermutationSpaceIntegration, DescriptorSpaceEnumeratesAllPermutations) {
    auto desc = make_permutation<4>("order");
    auto space = descriptor_space("perm_test", desc);

    using point_t = typename decltype(space)::point_type;
    std::set<point_t> seen;
    std::size_t idx = 0;
    space.enumerate([&](auto const& pt) {
        seen.insert(pt);
        EXPECT_EQ(pt, std::tuple{desc.value_at(idx)});
        ++idx;
    });

    EXPECT_EQ(idx, desc.cardinality());
    EXPECT_EQ(seen.size(), desc.cardinality());
    EXPECT_EQ(space.feature_width(), desc.feature_width());
}

TEST(PermutationSpaceIntegration, PointAtMatchesValueAt) {
    auto desc = make_permutation<4>("order");
    auto space = descriptor_space("perm_test", desc);
    for (std::size_t i = 0; i < desc.cardinality(); ++i) {
        EXPECT_EQ(space.point_at(i), std::tuple{desc.value_at(i)});
    }
}

TEST(PermutationSpaceIntegration, DefaultBridgeEncodesPermutationAndScalarDims) {
    auto space = descriptor_space("mixed",
        make_permutation<4>("order"),
        bool_flag("vectorize"));
    auto bridge = default_bridge(space);

    EXPECT_EQ(bridge.num_features(), 7u); // 6 precedence + 1 binary

    std::vector<double> features(bridge.num_features(), -1.0);
    bridge.write_features(std::tuple{permutation_value<4>{}, true}, features);

    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(features[i], 1.0);
    }
    EXPECT_DOUBLE_EQ(features[6], 1.0);
}

TEST(PermutationSpaceIntegration, GetDimAsIntGuardsStructuredDims) {
    auto space = descriptor_space("mixed",
        make_permutation<3>("order"),
        positive_int("tile", 1, 4));

    auto pt = std::tuple{make_permutation<3>("order").value_at(2), 3};
    EXPECT_EQ(space.get_dim_as_int(pt, "order"), -1);
    EXPECT_EQ(space.get_dim_as_int(pt, "tile"), 3);
}

TEST(PermutationSpaceIntegration, ConditionalPermutationPreservesFeatureWidth) {
    auto active_space = descriptor_space("active",
        conditional_dim(true, make_permutation<3>("order")));
    auto inactive_space = descriptor_space("inactive",
        conditional_dim(false, make_permutation<3>("order")));

    auto active_bridge = default_bridge(active_space);
    auto inactive_bridge = default_bridge(inactive_space);

    EXPECT_EQ(active_space.feature_width(), 3u);
    EXPECT_EQ(inactive_space.feature_width(), 3u);
    EXPECT_EQ(active_bridge.num_features(), 3u);
    EXPECT_EQ(inactive_bridge.num_features(), 3u);

    std::vector<double> inactive_features(3, -1.0);
    inactive_bridge.write_features(std::tuple{permutation_value<3>{}}, inactive_features);
    for (double x : inactive_features) EXPECT_DOUBLE_EQ(x, 0.0);
}

static_assert(dimension_descriptor<permutation_desc<4>>);
static_assert(search_space<decltype(descriptor_space("perm", make_permutation<4>("p")))>);
static_assert(countable_space<decltype(descriptor_space("perm", make_permutation<4>("p")))>);
static_assert(indexable_space<decltype(descriptor_space("perm", make_permutation<4>("p")))>);
static_assert(factored_space<decltype(descriptor_space("perm", make_permutation<4>("p")))>);

} // namespace

