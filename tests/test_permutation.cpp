// tests/test_permutation.cpp
// Sprint 7 — Permutation ordinate type in the space algebra
//
// 27 tests across 18 sections:
//   1: permutation_value basics
//   2: cardinality
//   3: N=1 edge case
//   4: overflow ceiling
//   5: enumeration
//   6: rank/unrank
//   7: feature encoding
//   8: Kendall tau
//   9: concept satisfaction
//  10: encoded_as
//  11: composition in descriptor_space
//  12: get_dim_as_int guard
//  13: counted_view
//  14: section/fix
//  15: conditional permutation
//  16: tree_space child
//  17: neighbours
//  18: bridge protocol + equivalence

#include "ctdp/space/permutation.h"
#include "ctdp/space/descriptor.h"
#include "ctdp/space/counted_view.h"
#include "ctdp/space/conditional.h"
#include "ctdp/space/reduction_tree_space.h"
#include "ctdp/solver/spaces/permutation_space.h"
#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace {

using namespace ctdp::space;

// ════════════════════════════════════════════════════════════════════════
// Section 1: permutation_value<N> basics (R7.24)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationValue, DefaultIsIdentity) {
    permutation_value<4> v;
    EXPECT_TRUE(v.is_valid());
    EXPECT_TRUE(v.is_identity());
    EXPECT_EQ(v[0], 0u);
    EXPECT_EQ(v[1], 1u);
    EXPECT_EQ(v[2], 2u);
    EXPECT_EQ(v[3], 3u);
}

TEST(PermutationValue, ExplicitConstruction) {
    permutation_value<4> v(std::array<std::size_t, 4>{2, 0, 3, 1});
    EXPECT_TRUE(v.is_valid());
    EXPECT_FALSE(v.is_identity());
    EXPECT_EQ(v[0], 2u);
    EXPECT_EQ(v[1], 0u);
}

TEST(PermutationValue, RejectsDuplicates) {
    permutation_value<4> v(std::array<std::size_t, 4>{0, 0, 1, 2});
    EXPECT_FALSE(v.is_valid());
}

TEST(PermutationValue, RejectsOutOfRange) {
    permutation_value<4> v(std::array<std::size_t, 4>{0, 1, 2, 5});
    EXPECT_FALSE(v.is_valid());
}

TEST(PermutationValue, Ordering) {
    permutation_value<3> a(std::array<std::size_t, 3>{0, 1, 2});
    permutation_value<3> b(std::array<std::size_t, 3>{0, 2, 1});
    EXPECT_LT(a, b);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: Cardinality (R7.2)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, Cardinality) {
    EXPECT_EQ(make_permutation<1>("a").cardinality(), 1u);
    EXPECT_EQ(make_permutation<2>("a").cardinality(), 2u);
    EXPECT_EQ(make_permutation<3>("a").cardinality(), 6u);
    EXPECT_EQ(make_permutation<4>("a").cardinality(), 24u);
    EXPECT_EQ(make_permutation<5>("a").cardinality(), 120u);
    EXPECT_EQ(make_permutation<6>("a").cardinality(), 720u);
    EXPECT_EQ(make_permutation<7>("a").cardinality(), 5040u);
    EXPECT_EQ(make_permutation<8>("a").cardinality(), 40320u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: N=1 edge case (R7.19)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, N1EdgeCase) {
    auto desc = make_permutation<1>("trivial");
    EXPECT_EQ(desc.cardinality(), 1u);
    EXPECT_EQ(desc.feature_width(), 0u);

    auto v = desc.value_at(0);
    EXPECT_TRUE(v.is_valid());
    EXPECT_TRUE(v.is_identity());
    EXPECT_EQ(v[0], 0u);

    // write_features writes nothing (0 features)
    double buf[1] = {999.0};
    desc.write_features(v, buf);
    EXPECT_DOUBLE_EQ(buf[0], 999.0);  // untouched
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: Overflow ceiling (R7.27)
// ════════════════════════════════════════════════════════════════════════

// static_assert(N <= 20) is in permutation_desc. N=20 compiles:
static_assert(dimension_descriptor<permutation_desc<20>>);
// N=21 would fail: static_assert fires. Cannot test in code.

// ════════════════════════════════════════════════════════════════════════
// Section 5: Enumeration (R7.6, R7.7)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, Enumeration) {
    auto desc = make_permutation<4>("order");
    auto space = descriptor_space("test", desc);

    using pt_t = typename decltype(space)::point_type;
    std::set<pt_t> seen;
    std::size_t idx = 0;
    space.enumerate([&](const auto& pt) {
        seen.insert(pt);
        // R7.7: enumeration order matches value_at
        auto expected = std::tuple{desc.value_at(idx)};
        EXPECT_EQ(pt, expected) << "Mismatch at index " << idx;
        // All valid
        EXPECT_TRUE(std::get<0>(pt).is_valid());
        ++idx;
    });

    EXPECT_EQ(seen.size(), 24u);  // no duplicates
    EXPECT_EQ(idx, 24u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: Rank/unrank round-trip (R7.4, R7.5, R7.25)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, RankUnrankRoundTrip) {
    auto desc = make_permutation<4>("order");
    for (std::size_t i = 0; i < desc.cardinality(); ++i) {
        auto v = desc.value_at(i);
        EXPECT_TRUE(v.is_valid()) << "Invalid at index " << i;
        EXPECT_EQ(desc.index_of(v), i) << "Round-trip failed at " << i;
    }
}

TEST(PermutationDesc, UnrankRankRoundTrip) {
    auto desc = make_permutation<4>("order");
    auto space = descriptor_space("test", desc);
    space.enumerate([&](const auto& pt) {
        auto v = std::get<0>(pt);
        auto idx = desc.index_of(v);
        EXPECT_EQ(desc.value_at(idx), v);
    });
}

TEST(PermutationDesc, IdentityAndReverse) {
    auto desc = make_permutation<4>("order");

    // value_at(0) == identity (factoradic ordering)
    auto first = desc.value_at(0);
    EXPECT_TRUE(first.is_identity());

    // value_at(N!-1) == reverse
    auto last = desc.value_at(23);
    EXPECT_EQ(last[0], 3u);
    EXPECT_EQ(last[1], 2u);
    EXPECT_EQ(last[2], 1u);
    EXPECT_EQ(last[3], 0u);
}

TEST(PermutationDesc, ExhaustiveN6) {
    auto desc = make_permutation<6>("order");
    for (std::size_t i = 0; i < 720; ++i) {
        auto v = desc.value_at(i);
        EXPECT_TRUE(v.is_valid());
        EXPECT_EQ(desc.index_of(v), i);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 7: Feature encoding (R7.8–R7.10, R7.12)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, FeatureWidth) {
    EXPECT_EQ(make_permutation<4>("a").feature_width(), 6u);
    EXPECT_EQ(make_permutation<5>("a").feature_width(), 10u);
    EXPECT_EQ(make_permutation<8>("a").feature_width(), 28u);
}

TEST(PermutationDesc, IdentityAllOnes) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> identity;  // default = identity
    double f[6];
    desc.write_features(identity, f);
    for (int i = 0; i < 6; ++i)
        EXPECT_DOUBLE_EQ(f[i], 1.0) << "Feature " << i;
}

TEST(PermutationDesc, ReverseAllZeros) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> rev(std::array<std::size_t, 4>{3, 2, 1, 0});
    double f[6];
    desc.write_features(rev, f);
    for (int i = 0; i < 6; ++i)
        EXPECT_DOUBLE_EQ(f[i], 0.0) << "Feature " << i;
}

TEST(PermutationDesc, KnownPermutation) {
    auto desc = make_permutation<4>("order");
    // [2,0,3,1]: pos[0]=1, pos[1]=3, pos[2]=0, pos[3]=2
    permutation_value<4> v(std::array<std::size_t, 4>{2, 0, 3, 1});
    double f[6];
    desc.write_features(v, f);
    // (0,1): 1<3 → 1, (0,2): 1<0 → 0, (0,3): 1<2 → 1
    // (1,2): 3<0 → 0, (1,3): 3<2 → 0, (2,3): 0<2 → 1
    EXPECT_DOUBLE_EQ(f[0], 1.0);
    EXPECT_DOUBLE_EQ(f[1], 0.0);
    EXPECT_DOUBLE_EQ(f[2], 1.0);
    EXPECT_DOUBLE_EQ(f[3], 0.0);
    EXPECT_DOUBLE_EQ(f[4], 0.0);
    EXPECT_DOUBLE_EQ(f[5], 1.0);
}

// ════════════════════════════════════════════════════════════════════════
// Section 8: Kendall tau (R7.11)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, KendallTau) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> identity;
    permutation_value<4> rev(std::array<std::size_t, 4>{3, 2, 1, 0});
    permutation_value<4> swap01(std::array<std::size_t, 4>{1, 0, 2, 3});

    double fi[6], fr[6], fs[6];
    desc.write_features(identity, fi);
    desc.write_features(rev, fr);
    desc.write_features(swap01, fs);

    // Hamming distance
    auto hamming = [](double* a, double* b, std::size_t n) {
        std::size_t d = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (a[i] != b[i]) ++d;
        return d;
    };

    EXPECT_EQ(hamming(fi, fi, 6), 0u);   // identity vs identity
    EXPECT_EQ(hamming(fi, fr, 6), 6u);   // identity vs reverse
    EXPECT_EQ(hamming(fi, fs, 6), 1u);   // identity vs single swap
}

// ════════════════════════════════════════════════════════════════════════
// Section 9: Concept satisfaction (R7.1)
// ════════════════════════════════════════════════════════════════════════

static_assert(dimension_descriptor<permutation_desc<4>>);

TEST(PermutationDesc, ConceptProperties) {
    auto desc = make_permutation<4>("order");
    EXPECT_EQ(permutation_desc<4>::kind, dim_kind::permutation);
    EXPECT_EQ(desc.default_encoding(), encoding_hint::precedence);
}

// ════════════════════════════════════════════════════════════════════════
// Section 10: encoded_as (R7.21)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, EncodedAs) {
    auto desc = make_permutation<4>("order");
    // precedence is accepted
    auto same = desc.encoded_as(encoding_hint::precedence);
    EXPECT_EQ(same.name, desc.name);

    // one_hot is rejected
    EXPECT_THROW(desc.encoded_as(encoding_hint::one_hot),
                 std::invalid_argument);
    EXPECT_THROW(desc.encoded_as(encoding_hint::raw),
                 std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════
// Section 11: Composition in descriptor_space (R7.13, R7.14)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, DescriptorSpaceComposition) {
    auto space = descriptor_space("test",
        power_2("tile", 64, 256),
        make_permutation<4>("order"));
    // 3 tiles × 24 permutations = 72
    EXPECT_EQ(space.cardinality(), 72u);

    auto bridge = default_bridge(space);
    // 1 (log2) + 6 (precedence) = 7
    EXPECT_EQ(bridge.num_features(), 7u);

    // Encode: tile=128, order={2,0,3,1}
    auto pt = std::tuple{128,
        permutation_value<4>(std::array<std::size_t, 4>{2, 0, 3, 1})};
    auto f = bridge.encode(pt);
    ASSERT_EQ(f.size(), 7u);

    EXPECT_DOUBLE_EQ(f[0], 7.0);  // log2(128)
    // precedence: [1,0,1,0,0,1]
    EXPECT_DOUBLE_EQ(f[1], 1.0);
    EXPECT_DOUBLE_EQ(f[2], 0.0);
    EXPECT_DOUBLE_EQ(f[3], 1.0);
    EXPECT_DOUBLE_EQ(f[4], 0.0);
    EXPECT_DOUBLE_EQ(f[5], 0.0);
    EXPECT_DOUBLE_EQ(f[6], 1.0);
}

// ════════════════════════════════════════════════════════════════════════
// Section 12: get_dim_as_int guard (R7.23)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, GetDimAsInt) {
    auto space = descriptor_space("test",
        power_2("tile", 64, 256),
        make_permutation<4>("order"));

    auto pt = std::tuple{128,
        permutation_value<4>(std::array<std::size_t, 4>{2, 0, 3, 1})};

    // Tile dimension returns int value
    EXPECT_EQ(space.get_dim_as_int(pt, "tile"), 128);
    // Permutation dimension returns -1 (not int-convertible)
    EXPECT_EQ(space.get_dim_as_int(pt, "order"), -1);
}

// ════════════════════════════════════════════════════════════════════════
// Section 13: counted_view (R7.15)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, CountedView) {
    auto desc = make_permutation<4>("order");
    auto space = descriptor_space("test", desc);

    // Only permutations where element 0 is first: (N-1)! = 6
    auto view = make_counted_view(space, [](const auto& pt) {
        return std::get<0>(pt)[0] == 0;
    });

    EXPECT_EQ(view.cardinality(), 6u);

    view.enumerate([](const auto& pt) {
        EXPECT_EQ(std::get<0>(pt)[0], 0u);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Section 14: section/fix (R7.16)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, SectionFix) {
    auto space = descriptor_space("test",
        power_2("tile", 64, 256),
        make_permutation<4>("order"));

    // Fix tile, permutation survives
    auto fixed = ctdp::space::section<0>(space, 128);
    EXPECT_EQ(decltype(fixed)::rank, 1u);

    std::size_t count = 0;
    fixed.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, 24u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 15: Conditional permutation (R7.20)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, ConditionalActive) {
    auto dim = conditional_dim(true, make_permutation<4>("order"));
    EXPECT_EQ(dim.cardinality(), 24u);

    double features[6];
    permutation_value<4> identity;
    dim.write_features(identity, features);
    // Active: all-ones for identity
    for (int i = 0; i < 6; ++i)
        EXPECT_DOUBLE_EQ(features[i], 1.0);
}

TEST(PermutationIntegration, ConditionalInactive) {
    auto dim = conditional_dim(false, make_permutation<4>("order"));
    EXPECT_EQ(dim.cardinality(), 1u);

    double features[6];
    auto v = dim.value_at(0);
    dim.write_features(v, features);
    // Inactive: sentinels (0.0)
    for (int i = 0; i < 6; ++i)
        EXPECT_DOUBLE_EQ(features[i], 0.0) << "Sentinel " << i;
}

// ════════════════════════════════════════════════════════════════════════
// Section 16: tree_space child (R7.26)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, TreeSpaceChild) {
    using namespace ct_dp::algebra;
    // 3-lane all-plus reduction
    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0});
    auto props = reduction_properties(red);

    // Child factory that includes a permutation dimension
    auto child_factory = [props](
        const auto&, std::size_t,
        std::span<const std::size_t> lane_indices) {
            auto gp = make_group_properties(props, lane_indices);
            return descriptor_space("child",
                power_2("tile", 64, 256),
                conditional_dim(gp.all_associative,
                                default_tree_shape_dim()),
                make_permutation<3>("lane_order"));
    };

    auto filter = make_fusibility_filter<3>(props);
    auto ts = make_tree_space<3>("test", make_partition<3>("g"),
        child_factory, filter);

    // Verify it enumerates without crashing and has reasonable cardinality
    std::size_t count = 0;
    ts.enumerate([&](const auto&) { ++count; });
    EXPECT_GT(count, 0u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 17: Neighbours (R7.17)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationDesc, Neighbours) {
    auto desc = make_permutation<4>("order");
    permutation_value<4> identity;

    std::vector<permutation_value<4>> nbs;
    desc.neighbours(identity, [&](const auto& nb) {
        nbs.push_back(nb);
    });

    // 4*3/2 = 6 neighbours
    EXPECT_EQ(nbs.size(), 6u);

    for (const auto& nb : nbs) {
        EXPECT_TRUE(nb.is_valid());
        EXPECT_NE(nb, identity);
        // Differs by exactly one transposition: exactly 2 positions differ
        std::size_t diffs = 0;
        for (std::size_t i = 0; i < 4; ++i)
            if (nb[i] != identity[i]) ++diffs;
        EXPECT_EQ(diffs, 2u);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 18: Bridge protocol + equivalence (R7.18, R7.22)
// ════════════════════════════════════════════════════════════════════════

TEST(PermutationIntegration, BridgeProtocol) {
    auto space = descriptor_space("test", make_permutation<4>("order"));
    auto bridge = default_bridge(space);

    auto pt = std::tuple{permutation_value<4>()};  // identity
    auto features = bridge.encode(pt);
    EXPECT_EQ(features.size(), bridge.num_features());
    EXPECT_EQ(features.size(), 6u);
}

TEST(PermutationIntegration, EquivalenceWithSolver) {
    // Solver's permutation_space<4>
    ctdp::permutation_space<4> solver_space;
    std::set<std::array<std::size_t, 4>> solver_set;
    solver_space.enumerate([&](const auto& c) { solver_set.insert(c); });

    // Space-layer permutation_desc<4>
    auto desc = make_permutation<4>("order");
    std::set<std::array<std::size_t, 4>> space_set;
    for (std::size_t i = 0; i < desc.cardinality(); ++i) {
        auto v = desc.value_at(i);
        space_set.insert(v.order);  // extract raw array
    }

    EXPECT_EQ(solver_set.size(), 24u);
    EXPECT_EQ(solver_set, space_set);
}

} // namespace
