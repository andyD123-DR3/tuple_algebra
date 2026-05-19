#include "ctdp/solver/solver.h"
#include "ct_dp/space/binary_cut_desc.hpp"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

TEST(IntervalSubstrateCanonical, CanonicalAliasesMatchLegacyTypes) {
    static_assert(std::is_same_v<ctdp::solver::interval_context,
                                 ct_dp::solver::interval_context>);
    static_assert(std::is_same_v<ctdp::solver::policies::all_binary_splits,
                                 ct_dp::solver::all_binary_splits>);
    static_assert(std::is_same_v<ctdp::solver::plans::interval_partition_plan,
                                 ct_dp::plan::interval_partition_plan>);
    static_assert(std::is_same_v<ctdp::solver::memo::triangular_memo<int>,
                                 ct_dp::solver::triangular_memo<int>>);
}

TEST(IntervalSubstrateCanonical, IntervalContextIsAvailableFromUmbrellaHeader) {
    ctdp::solver::interval_context ctx{3, 9};

    EXPECT_EQ(ctx.start(), 3u);
    EXPECT_EQ(ctx.end(), 9u);
    EXPECT_EQ(ctx.size(), 6u);

    auto left = ctx.left(5);
    auto right = ctx.right(5);
    EXPECT_EQ(left.start(), 3u);
    EXPECT_EQ(left.end(), 5u);
    EXPECT_EQ(right.start(), 5u);
    EXPECT_EQ(right.end(), 9u);
}

TEST(IntervalSubstrateCanonical, AllBinarySplitsEnumeratesInteriorCuts) {
    ctdp::solver::policies::all_binary_splits policy;
    ctdp::solver::interval_context ctx{2, 7};
    std::vector<size_t> cuts;

    policy.for_each(ctx, [&](size_t k) { cuts.push_back(k); });

    ASSERT_EQ(cuts.size(), 4u);
    EXPECT_EQ(cuts[0], 3u);
    EXPECT_EQ(cuts[1], 4u);
    EXPECT_EQ(cuts[2], 5u);
    EXPECT_EQ(cuts[3], 6u);
}

TEST(IntervalSubstrateCanonical, IntervalPartitionPlanSupportsRuntimeAndDescriptorConstruction) {
    ctdp::solver::interval_context ctx{10, 20};

    auto from_split = ctdp::solver::plans::interval_partition_plan::from_split(ctx, 15);
    EXPECT_TRUE(from_split.is_legal());
    EXPECT_TRUE(from_split.preserves_size());
    EXPECT_EQ(from_split.left_ctx.end(), 15u);
    EXPECT_EQ(from_split.right_ctx.start(), 15u);

    ct_dp::space::binary_cut_desc<10> desc;
    auto from_desc = ctdp::solver::plans::interval_partition_plan::make(ctx, desc, 4);
    EXPECT_EQ(from_desc.split, 15u);
    EXPECT_TRUE(from_desc.is_legal());
    EXPECT_TRUE(from_desc.preserves_size());
}

TEST(IntervalSubstrateCanonical, TriangularMemoStoresLooksUpAndClears) {
    ctdp::solver::memo::triangular_memo<std::pair<int, int>> memo{6};

    EXPECT_EQ(memo.capacity(), 6u);
    EXPECT_EQ(memo.size(), 0u);

    memo.store(ctdp::solver::interval_context{0, 3}, {1, 2});
    memo.store(ctdp::solver::interval_context{2, 6}, {3, 4});

    auto first = memo.lookup(ctdp::solver::interval_context{0, 3});
    auto second = memo.lookup(ctdp::solver::interval_context{2, 6});
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, std::make_pair(1, 2));
    EXPECT_EQ(*second, std::make_pair(3, 4));
    EXPECT_EQ(memo.size(), 2u);

    memo.clear();
    EXPECT_EQ(memo.size(), 0u);
    EXPECT_FALSE(memo.lookup(ctdp::solver::interval_context{0, 3}).has_value());
    EXPECT_FALSE(memo.lookup(ctdp::solver::interval_context{2, 6}).has_value());
}

