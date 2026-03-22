#include "ct_dp/solver/triangular_memo.hpp"
#include "ct_dp/solver/interval_context.hpp"
#include <gtest/gtest.h>
#include <utility>

using namespace ct_dp::solver;

// Test empty lookup returns nullopt
TEST(TriangularMemo, EmptyLookupReturnsNullopt) {
    triangular_memo<int> memo{10};
    
    auto result = memo.lookup(interval_context{0, 5});
    
    EXPECT_FALSE(result.has_value());
}

// Test store and lookup
TEST(TriangularMemo, StoreAndLookup) {
    triangular_memo<int> memo{10};
    interval_context ctx{0, 5};
    
    memo.store(ctx, 42);
    auto result = memo.lookup(ctx);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

// Test distinct intervals don't collide
TEST(TriangularMemo, DistinctIntervalsNoCollide) {
    triangular_memo<int> memo{10};
    
    memo.store(interval_context{0, 3}, 100);
    memo.store(interval_context{1, 4}, 200);
    memo.store(interval_context{2, 5}, 300);
    
    EXPECT_EQ(*memo.lookup(interval_context{0, 3}), 100);
    EXPECT_EQ(*memo.lookup(interval_context{1, 4}), 200);
    EXPECT_EQ(*memo.lookup(interval_context{2, 5}), 300);
}

// Test dense indexing is correct
TEST(TriangularMemo, DenseIndexingCorrect) {
    triangular_memo<int> memo{5};
    
    // Store values in all valid positions
    int value = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j <= 5; ++j) {
            memo.store(interval_context{i, j}, value++);
        }
    }
    
    // Verify all can be retrieved
    value = 0;
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j <= 5; ++j) {
            EXPECT_EQ(*memo.lookup(interval_context{i, j}), value++);
        }
    }
}

// Test clear
TEST(TriangularMemo, Clear) {
    triangular_memo<int> memo{10};
    
    memo.store(interval_context{0, 5}, 42);
    memo.store(interval_context{1, 6}, 43);
    
    memo.clear();
    
    EXPECT_FALSE(memo.lookup(interval_context{0, 5}).has_value());
    EXPECT_FALSE(memo.lookup(interval_context{1, 6}).has_value());
}

// Test size
TEST(TriangularMemo, Size) {
    triangular_memo<int> memo{10};
    
    EXPECT_EQ(memo.size(), 0);
    
    memo.store(interval_context{0, 5}, 42);
    EXPECT_EQ(memo.size(), 1);
    
    memo.store(interval_context{1, 6}, 43);
    EXPECT_EQ(memo.size(), 2);
    
    memo.store(interval_context{0, 5}, 100);  // Overwrite
    EXPECT_EQ(memo.size(), 2);  // Still 2
    
    memo.clear();
    EXPECT_EQ(memo.size(), 0);
}

// Test generic value type: int
TEST(TriangularMemo, GenericValueInt) {
    triangular_memo<int> memo{10};
    
    memo.store(interval_context{0, 5}, 42);
    EXPECT_EQ(*memo.lookup(interval_context{0, 5}), 42);
}

// Test generic value type: double
TEST(TriangularMemo, GenericValueDouble) {
    triangular_memo<double> memo{10};
    
    memo.store(interval_context{0, 5}, 3.14159);
    EXPECT_DOUBLE_EQ(*memo.lookup(interval_context{0, 5}), 3.14159);
}

// Test generic value type: pair (multi-objective)
TEST(TriangularMemo, GenericValuePair) {
    using CostPair = std::pair<int, int>;
    triangular_memo<CostPair> memo{10};
    
    memo.store(interval_context{0, 5}, {100, 200});
    auto result = memo.lookup(interval_context{0, 5});
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->first, 100);
    EXPECT_EQ(result->second, 200);
}

// Test capacity
TEST(TriangularMemo, Capacity) {
    triangular_memo<int> memo{10};
    EXPECT_EQ(memo.capacity(), 10);
    
    triangular_memo<int> memo2{50};
    EXPECT_EQ(memo2.capacity(), 50);
}

// Test maximum valid index (boundary case)
TEST(TriangularMemo, MaximumValidIndex) {
    triangular_memo<int> memo{5};
    
    // Store at maximum valid index: [4, 5)
    memo.store(interval_context{4, 5}, 999);
    
    auto result = memo.lookup(interval_context{4, 5});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 999);
    
    // Verify all maximum indices for different starts
    memo.store(interval_context{0, 5}, 100);
    memo.store(interval_context{1, 5}, 101);
    memo.store(interval_context{2, 5}, 102);
    memo.store(interval_context{3, 5}, 103);
    
    EXPECT_EQ(*memo.lookup(interval_context{0, 5}), 100);
    EXPECT_EQ(*memo.lookup(interval_context{1, 5}), 101);
    EXPECT_EQ(*memo.lookup(interval_context{2, 5}), 102);
    EXPECT_EQ(*memo.lookup(interval_context{3, 5}), 103);
}

// Test storage size calculation is correct
TEST(TriangularMemo, StorageSizeFormula) {
    // For n endpoints, storage = n(n+1)/2
    // This stores all valid intervals [i,j) where i < j <= n
    
    {
        triangular_memo<int> memo{4};
        // Endpoints 0,1,2,3,4: Valid intervals with i < j <= 4:
        // [0,1),[0,2),[0,3),[0,4) = 4
        // [1,2),[1,3),[1,4) = 3
        // [2,3),[2,4) = 2
        // [3,4) = 1
        // Total: 4+3+2+1 = 10 = 4*5/2
        
        // Fill all valid intervals
        int val = 0;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = i + 1; j <= 4; ++j) {
                memo.store(interval_context{i, j}, val++);
            }
        }
        
        EXPECT_EQ(memo.size(), 10);  // 4*5/2
        EXPECT_EQ(memo.capacity(), 4);
    }
    
    {
        triangular_memo<int> memo{10};
        // For n=10: 10*11/2 = 55 possible intervals
        EXPECT_EQ(memo.capacity(), 10);
        EXPECT_EQ(memo.size(), 0);  // None stored yet
    }
}
