#include "ct_dp/solver/interval_context.hpp"
#include "ct_dp/solver/triangular_memo.hpp"

#include <gtest/gtest.h>

#include <utility>

using namespace ct_dp::solver;

TEST(TriangularMemo, EmptyLookupReturnsNullopt) {
    triangular_memo<int> memo{10};
    EXPECT_FALSE(memo.lookup(interval_context{0, 5}).has_value());
}

TEST(TriangularMemo, StoreAndLookup) {
    triangular_memo<int> memo{10};
    interval_context ctx{0, 5};

    memo.store(ctx, 42);
    auto result = memo.lookup(ctx);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST(TriangularMemo, DistinctIntervalsDoNotCollide) {
    triangular_memo<int> memo{10};

    memo.store(interval_context{0, 3}, 100);
    memo.store(interval_context{1, 4}, 200);
    memo.store(interval_context{2, 5}, 300);

    EXPECT_EQ(*memo.lookup(interval_context{0, 3}), 100);
    EXPECT_EQ(*memo.lookup(interval_context{1, 4}), 200);
    EXPECT_EQ(*memo.lookup(interval_context{2, 5}), 300);
}

TEST(TriangularMemo, DenseIndexingCoversAllValidIntervals) {
    triangular_memo<int> memo{5};

    int value = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j <= 5; ++j) {
            memo.store(interval_context{i, j}, value++);
        }
    }

    value = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j <= 5; ++j) {
            EXPECT_EQ(*memo.lookup(interval_context{i, j}), value++);
        }
    }
}

TEST(TriangularMemo, ClearAndSize) {
    triangular_memo<int> memo{10};

    EXPECT_EQ(memo.size(), 0u);

    memo.store(interval_context{0, 5}, 42);
    memo.store(interval_context{1, 6}, 43);
    EXPECT_EQ(memo.size(), 2u);

    memo.store(interval_context{0, 5}, 100);
    EXPECT_EQ(memo.size(), 2u);

    memo.clear();
    EXPECT_EQ(memo.size(), 0u);
    EXPECT_FALSE(memo.lookup(interval_context{0, 5}).has_value());
    EXPECT_FALSE(memo.lookup(interval_context{1, 6}).has_value());
}

TEST(TriangularMemo, GenericValueTypes) {
    {
        triangular_memo<double> memo{10};
        memo.store(interval_context{0, 5}, 3.14159);
        ASSERT_TRUE(memo.lookup(interval_context{0, 5}).has_value());
        EXPECT_DOUBLE_EQ(*memo.lookup(interval_context{0, 5}), 3.14159);
    }

    {
        using CostPair = std::pair<int, int>;
        triangular_memo<CostPair> memo{10};
        memo.store(interval_context{0, 5}, {100, 200});
        auto result = memo.lookup(interval_context{0, 5});
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->first, 100);
        EXPECT_EQ(result->second, 200);
    }
}

TEST(TriangularMemo, CapacityAndBoundaryIntervals) {
    triangular_memo<int> memo{5};
    EXPECT_EQ(memo.capacity(), 5u);

    memo.store(interval_context{4, 5}, 999);
    EXPECT_EQ(*memo.lookup(interval_context{4, 5}), 999);

    memo.store(interval_context{0, 5}, 100);
    memo.store(interval_context{1, 5}, 101);
    memo.store(interval_context{2, 5}, 102);
    memo.store(interval_context{3, 5}, 103);

    EXPECT_EQ(*memo.lookup(interval_context{0, 5}), 100);
    EXPECT_EQ(*memo.lookup(interval_context{1, 5}), 101);
    EXPECT_EQ(*memo.lookup(interval_context{2, 5}), 102);
    EXPECT_EQ(*memo.lookup(interval_context{3, 5}), 103);
}

TEST(TriangularMemo, StorageSizeFormula) {
    triangular_memo<int> memo{4};

    int val = 0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j <= 4; ++j) {
            memo.store(interval_context{i, j}, val++);
        }
    }

    EXPECT_EQ(memo.size(), 10u); // 4 * 5 / 2
    EXPECT_EQ(memo.capacity(), 4u);
}

