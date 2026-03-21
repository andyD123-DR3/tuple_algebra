// test_interval_context.cpp - Unit tests for interval_context
// Sprint 8: Context law tests
// Author: Andrew Drakeford

#include "ct_dp/solver/interval_context.hpp"
#include <gtest/gtest.h>

using namespace ct_dp::solver;

// Test: Basic Queries
TEST(IntervalContext, BasicQueries) {
    interval_context ctx{5, 12};
    
    EXPECT_EQ(ctx.size(), 7);
    EXPECT_EQ(ctx.start(), 5);
    EXPECT_EQ(ctx.end(), 12);
    EXPECT_EQ(ctx.i, 5);  // Direct member access
    EXPECT_EQ(ctx.j, 12);
}

// Test: Subinterval Laws
TEST(IntervalContext, SubintervalLaws) {
    interval_context ctx{10, 20};
    
    for (size_t k = 11; k < 20; ++k) {  // Valid k ∈ (10, 20)
        auto left = ctx.left(k);
        auto right = ctx.right(k);
        
        // Law: left interval is [i, k)
        EXPECT_EQ(left.start(), ctx.start()) << "Failed at k=" << k;
        EXPECT_EQ(left.end(), k) << "Failed at k=" << k;
        
        // Law: right interval is [k, j)
        EXPECT_EQ(right.start(), k) << "Failed at k=" << k;
        EXPECT_EQ(right.end(), ctx.end()) << "Failed at k=" << k;
        
        // Law: left.end() == right.start() == k
        EXPECT_EQ(left.end(), right.start()) << "Failed at k=" << k;
        EXPECT_EQ(left.end(), k) << "Failed at k=" << k;
        
        // Law: size preservation
        EXPECT_EQ(left.size() + right.size(), ctx.size()) 
            << "Failed at k=" << k;
    }
}

// Test: Boundary Conditions
TEST(IntervalContext, BoundaryConditions) {
    interval_context ctx{0, 10};
    
    // First valid cut (k=1)
    {
        auto left = ctx.left(1);
        auto right = ctx.right(1);
        
        EXPECT_EQ(left.size(), 1);
        EXPECT_EQ(right.size(), 9);
        EXPECT_EQ(left.start(), 0);
        EXPECT_EQ(left.end(), 1);
        EXPECT_EQ(right.start(), 1);
        EXPECT_EQ(right.end(), 10);
    }
    
    // Last valid cut (k=9)
    {
        auto left = ctx.left(9);
        auto right = ctx.right(9);
        
        EXPECT_EQ(left.size(), 9);
        EXPECT_EQ(right.size(), 1);
        EXPECT_EQ(left.start(), 0);
        EXPECT_EQ(left.end(), 9);
        EXPECT_EQ(right.start(), 9);
        EXPECT_EQ(right.end(), 10);
    }
}

// Test: Size Preservation Law
TEST(IntervalContext, SizePreservationLaw) {
    // Test multiple interval sizes
    for (size_t len = 2; len <= 20; ++len) {
        interval_context ctx{0, len};
        
        // For all valid cuts
        for (size_t k = 1; k < len; ++k) {
            auto left = ctx.left(k);
            auto right = ctx.right(k);
            
            EXPECT_EQ(left.size() + right.size(), ctx.size())
                << "Failed for len=" << len << ", k=" << k;
        }
    }
}

// Test: Partition Coverage Law
TEST(IntervalContext, PartitionCoverageLaw) {
    interval_context ctx{5, 15};  // [5, 15), size=10
    
    for (size_t k = 6; k < 15; ++k) {
        auto left = ctx.left(k);
        auto right = ctx.right(k);
        
        // Law: Left covers [i, k)
        EXPECT_EQ(left.start(), ctx.start());
        EXPECT_EQ(left.end(), k);
        
        // Law: Right covers [k, j)
        EXPECT_EQ(right.start(), k);
        EXPECT_EQ(right.end(), ctx.end());
        
        // Law: No gap between left and right
        EXPECT_EQ(left.end(), right.start());
        
        // Law: Together they cover the whole interval
        EXPECT_EQ(left.start(), ctx.start());
        EXPECT_EQ(right.end(), ctx.end());
    }
}

// Test: Compile-Time Construction
TEST(IntervalContext, CompileTimeConstruction) {
    constexpr interval_context ctx{10, 20};
    static_assert(ctx.size() == 10, "Size should be j-i");
    static_assert(ctx.start() == 10, "Start should be i");
    static_assert(ctx.end() == 20, "End should be j");
}

// Test: Compile-Time Subintervals
TEST(IntervalContext, CompileTimeSubintervals) {
    constexpr interval_context ctx{10, 20};
    constexpr auto left = ctx.left(15);
    constexpr auto right = ctx.right(15);
    
    static_assert(left.start() == 10, "Left starts at i");
    static_assert(left.end() == 15, "Left ends at k");
    static_assert(right.start() == 15, "Right starts at k");
    static_assert(right.end() == 20, "Right ends at j");
    static_assert(left.size() + right.size() == 10, "Size preserved");
}

// Test: Non-Zero Start
TEST(IntervalContext, NonZeroStart) {
    interval_context ctx{50, 60};
    
    EXPECT_EQ(ctx.size(), 10);
    EXPECT_EQ(ctx.start(), 50);
    EXPECT_EQ(ctx.end(), 60);
    
    auto left = ctx.left(55);
    auto right = ctx.right(55);
    
    EXPECT_EQ(left.start(), 50);
    EXPECT_EQ(left.end(), 55);
    EXPECT_EQ(right.start(), 55);
    EXPECT_EQ(right.end(), 60);
}

// Test: Large Intervals
TEST(IntervalContext, LargeIntervals) {
    interval_context ctx{0, 1000};
    
    EXPECT_EQ(ctx.size(), 1000);
    
    auto left = ctx.left(500);
    auto right = ctx.right(500);
    
    EXPECT_EQ(left.size(), 500);
    EXPECT_EQ(right.size(), 500);
}

// Test: Nested Subintervals
TEST(IntervalContext, NestedSubintervals) {
    interval_context ctx{0, 20};
    
    // Create left half [0, 10)
    auto left_half = ctx.left(10);
    EXPECT_EQ(left_half.size(), 10);
    
    // Split left half further [0, 5) and [5, 10)
    auto left_quarter = left_half.left(5);
    auto mid_quarter = left_half.right(5);
    
    EXPECT_EQ(left_quarter.start(), 0);
    EXPECT_EQ(left_quarter.end(), 5);
    EXPECT_EQ(mid_quarter.start(), 5);
    EXPECT_EQ(mid_quarter.end(), 10);
    EXPECT_EQ(left_quarter.size() + mid_quarter.size(), left_half.size());
}

// Test: Minimum Size Interval
TEST(IntervalContext, MinimumSizeInterval) {
    interval_context ctx{0, 2};  // Size 2 (minimum for binary cut)
    
    EXPECT_EQ(ctx.size(), 2);
    
    // Only one valid cut point (k=1)
    auto left = ctx.left(1);
    auto right = ctx.right(1);
    
    EXPECT_EQ(left.size(), 1);
    EXPECT_EQ(right.size(), 1);
}
