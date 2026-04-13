#include "ct_dp/solver/all_binary_splits.hpp"
#include "ct_dp/solver/interval_context.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstddef>

using namespace ct_dp::solver;

// Test emits correct number of splits
TEST(AllBinarySplits, EmitsCorrectCount) {
    all_binary_splits policy;
    
    // For interval [i, j), should emit j - i - 1 splits
    {
        interval_context ctx{0, 3};
        std::vector<size_t> splits;
        
        policy.for_each(ctx, [&](size_t k) {
            splits.push_back(k);
        });
        
        EXPECT_EQ(splits.size(), 2);  // 3 - 0 - 1 = 2
    }
    
    {
        interval_context ctx{0, 5};
        std::vector<size_t> splits;
        
        policy.for_each(ctx, [&](size_t k) {
            splits.push_back(k);
        });
        
        EXPECT_EQ(splits.size(), 4);  // 5 - 0 - 1 = 4
    }
}

// Test emits splits in range (i, j)
TEST(AllBinarySplits, EmitsInCorrectRange) {
    all_binary_splits policy;
    interval_context ctx{2, 7};  // [2, 7)
    
    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) {
        splits.push_back(k);
    });
    
    // Should emit: 3, 4, 5, 6 (between 2 and 7)
    ASSERT_EQ(splits.size(), 4);
    EXPECT_EQ(splits[0], 3);
    EXPECT_EQ(splits[1], 4);
    EXPECT_EQ(splits[2], 5);
    EXPECT_EQ(splits[3], 6);
    
    // Verify all in range: ctx.i < k < ctx.j
    for (size_t k : splits) {
        EXPECT_GT(k, ctx.i);
        EXPECT_LT(k, ctx.j);
    }
}

// Test emits splits in order
TEST(AllBinarySplits, EmitsInOrder) {
    all_binary_splits policy;
    interval_context ctx{0, 10};
    
    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) {
        splits.push_back(k);
    });
    
    // Should emit 1, 2, 3, ..., 9 in order
    ASSERT_EQ(splits.size(), 9);
    for (size_t i = 0; i < splits.size(); ++i) {
        EXPECT_EQ(splits[i], i + 1);
    }
}

// Test no duplicates
TEST(AllBinarySplits, NoDuplicates) {
    all_binary_splits policy;
    interval_context ctx{0, 10};
    
    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) {
        splits.push_back(k);
    });
    
    // Check for duplicates
    for (size_t i = 0; i < splits.size(); ++i) {
        for (size_t j = i + 1; j < splits.size(); ++j) {
            EXPECT_NE(splits[i], splits[j]) << "Duplicate at " << i << " and " << j;
        }
    }
}

// Test single split case
TEST(AllBinarySplits, SingleSplitInterval) {
    all_binary_splits policy;
    interval_context ctx{0, 2};  // [0, 2)
    
    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) {
        splits.push_back(k);
    });
    
    // Only one split point: k=1
    ASSERT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0], 1);
}

// Test emits no splits for invalid interval size
// Note: interval_context enforces i < j, so we can't test [0,1) here
// But we can test that minimal valid interval has one split
TEST(AllBinarySplits, MinimalValidInterval) {
    all_binary_splits policy;
    interval_context ctx{5, 7};  // Size 2, minimal for one split
    
    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) {
        splits.push_back(k);
    });
    
    // Exactly one split: k=6
    ASSERT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0], 6);
}

// Test callback is const-correct
TEST(AllBinarySplits, ConstCorrect) {
    const all_binary_splits policy;  // const policy
    interval_context ctx{0, 5};
    
    std::vector<size_t> splits;
    policy.for_each(ctx, [&](size_t k) {
        splits.push_back(k);
    });
    
    EXPECT_EQ(splits.size(), 4);
}

// Test with different interval positions
TEST(AllBinarySplits, DifferentIntervalPositions) {
    all_binary_splits policy;
    
    // Test intervals at different positions
    std::vector<std::pair<size_t, size_t>> intervals = {
        {0, 5},   // Start at 0
        {10, 15}, // Mid-range
        {100, 105} // Large offset
    };
    
    for (auto [i, j] : intervals) {
        interval_context ctx{i, j};
        std::vector<size_t> splits;
        
        policy.for_each(ctx, [&](size_t k) {
            splits.push_back(k);
        });
        
        // All should emit 4 splits
        EXPECT_EQ(splits.size(), 4) << "Failed for [" << i << ", " << j << ")";
        
        // First split should be i+1, last should be j-1
        EXPECT_EQ(splits.front(), i + 1);
        EXPECT_EQ(splits.back(), j - 1);
    }
}
