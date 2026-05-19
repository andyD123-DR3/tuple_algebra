#include "ct_dp/solver/all_binary_splits.hpp"
#include "ct_dp/solver/interval_context.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

using namespace ct_dp::solver;

TEST(AllBinarySplits, EmitsCorrectCount) {
    all_binary_splits policy;

    {
        interval_context ctx{0, 3};
        std::vector<size_t> splits;
        policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });
        EXPECT_EQ(splits.size(), 2u);
    }

    {
        interval_context ctx{0, 5};
        std::vector<size_t> splits;
        policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });
        EXPECT_EQ(splits.size(), 4u);
    }
}

TEST(AllBinarySplits, EmitsInCorrectRangeAndOrder) {
    all_binary_splits policy;
    interval_context ctx{2, 7};

    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });

    ASSERT_EQ(splits.size(), 4u);
    EXPECT_EQ(splits[0], 3u);
    EXPECT_EQ(splits[1], 4u);
    EXPECT_EQ(splits[2], 5u);
    EXPECT_EQ(splits[3], 6u);

    for (size_t k : splits) {
        EXPECT_GT(k, ctx.i);
        EXPECT_LT(k, ctx.j);
    }
}

TEST(AllBinarySplits, NoDuplicates) {
    all_binary_splits policy;
    interval_context ctx{0, 10};

    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });

    for (size_t i = 0; i < splits.size(); ++i) {
        for (size_t j = i + 1; j < splits.size(); ++j) {
            EXPECT_NE(splits[i], splits[j]);
        }
    }
}

TEST(AllBinarySplits, MinimalValidIntervals) {
    all_binary_splits policy;

    {
        interval_context ctx{0, 2};
        std::vector<size_t> splits;
        policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });
        ASSERT_EQ(splits.size(), 1u);
        EXPECT_EQ(splits[0], 1u);
    }

    {
        interval_context ctx{5, 7};
        std::vector<size_t> splits;
        policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });
        ASSERT_EQ(splits.size(), 1u);
        EXPECT_EQ(splits[0], 6u);
    }
}

TEST(AllBinarySplits, ConstCorrectAndOffsetIndependent) {
    const all_binary_splits policy;
    std::vector<std::pair<size_t, size_t>> intervals{{0, 5}, {10, 15}, {100, 105}};

    for (auto [i, j] : intervals) {
        interval_context ctx{i, j};
        std::vector<size_t> splits;
        policy.for_each(ctx, [&](size_t k) { splits.push_back(k); });

        ASSERT_EQ(splits.size(), 4u);
        EXPECT_EQ(splits.front(), i + 1);
        EXPECT_EQ(splits.back(), j - 1);
    }
}

