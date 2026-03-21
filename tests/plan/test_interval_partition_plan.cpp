// test_interval_partition_plan.cpp - Unit tests for interval_partition_plan
// Sprint 8: Plan term tests
// Author: Andrew Drakeford

#include "ct_dp/plan/interval_partition_plan.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace ct_dp::plan;
using namespace ct_dp::solver;
using namespace ct_dp::space;

// Test: Factory Construction
TEST(IntervalPartitionPlan, FactoryConstruction) {
    interval_context ctx{10, 20};  // Length 10
    binary_cut_desc<10> desc;
    
    auto plan = interval_partition_plan::make(ctx, desc, 4);  // ord=4 → rel=5 → k=15
    
    EXPECT_EQ(plan.whole.start(), 10);
    EXPECT_EQ(plan.whole.end(), 20);
    EXPECT_EQ(plan.split, 15);
    EXPECT_EQ(plan.left_ctx.start(), 10);
    EXPECT_EQ(plan.left_ctx.end(), 15);
    EXPECT_EQ(plan.right_ctx.start(), 15);
    EXPECT_EQ(plan.right_ctx.end(), 20);
}

// Test: Legality Invariant
TEST(IntervalPartitionPlan, LegalityInvariant) {
    interval_context ctx{0, 10};
    binary_cut_desc<10> desc;
    
    // Law: All valid ordinals produce legal plans
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        EXPECT_TRUE(plan.is_legal()) << "Failed at ordinal " << ord;
    }
}

// Test: Size Preservation Law
TEST(IntervalPartitionPlan, SizePreservationLaw) {
    interval_context ctx{5, 25};  // Length 20
    binary_cut_desc<20> desc;
    
    // Law: Partition preserves total size
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        EXPECT_TRUE(plan.preserves_size()) << "Failed at ordinal " << ord;
        EXPECT_EQ(plan.left_size() + plan.right_size(), 20) 
            << "Failed at ordinal " << ord;
    }
}

// Test: Absolute Cut Laws
TEST(IntervalPartitionPlan, AbsoluteCutLaws) {
    interval_context ctx{0, 10};
    binary_cut_desc<10> desc;
    
    // Law: First ordinal maps to start+1
    {
        auto first = interval_partition_plan::make(ctx, desc, 0);
        EXPECT_EQ(first.split, ctx.start() + 1);
        EXPECT_EQ(first.split, 1);
    }
    
    // Law: Last ordinal maps to end-1
    {
        auto last = interval_partition_plan::make(ctx, desc, desc.size()-1);
        EXPECT_EQ(last.split, ctx.end() - 1);
        EXPECT_EQ(last.split, 9);
    }
    
    // Law: Split always strictly interior
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        EXPECT_GT(plan.split, ctx.start()) << "Failed at ordinal " << ord;
        EXPECT_LT(plan.split, ctx.end()) << "Failed at ordinal " << ord;
    }
}

// Test: Boundary Conditions
TEST(IntervalPartitionPlan, BoundaryConditions) {
    interval_context ctx{0, 10};
    binary_cut_desc<10> desc;
    
    // First cut: ordinal 0 → rel 1 → k=1
    {
        auto first = interval_partition_plan::make(ctx, desc, 0);
        EXPECT_EQ(first.split, 1);
        EXPECT_EQ(first.left_size(), 1);
        EXPECT_EQ(first.right_size(), 9);
        EXPECT_TRUE(first.is_legal());
        EXPECT_TRUE(first.preserves_size());
    }
    
    // Last cut: ordinal 8 → rel 9 → k=9
    {
        auto last = interval_partition_plan::make(ctx, desc, 8);
        EXPECT_EQ(last.split, 9);
        EXPECT_EQ(last.left_size(), 9);
        EXPECT_EQ(last.right_size(), 1);
        EXPECT_TRUE(last.is_legal());
        EXPECT_TRUE(last.preserves_size());
    }
}

// Test: Non-Zero Start Interval
TEST(IntervalPartitionPlan, NonZeroStartInterval) {
    interval_context ctx{50, 60};  // [50, 60), length 10
    binary_cut_desc<10> desc;
    
    // Middle cut: ordinal 4 → rel 5 → k=55
    auto plan = interval_partition_plan::make(ctx, desc, 4);
    
    EXPECT_EQ(plan.whole.start(), 50);
    EXPECT_EQ(plan.whole.end(), 60);
    EXPECT_EQ(plan.split, 55);
    EXPECT_EQ(plan.left_ctx.start(), 50);
    EXPECT_EQ(plan.left_ctx.end(), 55);
    EXPECT_EQ(plan.right_ctx.start(), 55);
    EXPECT_EQ(plan.right_ctx.end(), 60);
    EXPECT_TRUE(plan.is_legal());
    EXPECT_TRUE(plan.preserves_size());
}

// Test: Minimal Descriptor (Len=2)
TEST(IntervalPartitionPlan, MinimalDescriptor) {
    interval_context ctx{0, 2};
    binary_cut_desc<2> desc;
    
    // Only one valid cut: ordinal 0 → rel 1 → k=1
    auto plan = interval_partition_plan::make(ctx, desc, 0);
    
    EXPECT_EQ(plan.split, 1);
    EXPECT_EQ(plan.left_size(), 1);
    EXPECT_EQ(plan.right_size(), 1);
    EXPECT_TRUE(plan.is_legal());
    EXPECT_TRUE(plan.preserves_size());
}

// Test: Subinterval Disjoint Coverage
TEST(IntervalPartitionPlan, SubintervalDisjointCoverage) {
    interval_context ctx{0, 20};
    binary_cut_desc<20> desc;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        
        // Law: Left and right meet at split point
        EXPECT_EQ(plan.left_ctx.end(), plan.split);
        EXPECT_EQ(plan.right_ctx.start(), plan.split);
        
        // Law: Left and right are disjoint
        EXPECT_EQ(plan.left_ctx.end(), plan.right_ctx.start());
        
        // Law: Together they cover whole
        EXPECT_EQ(plan.left_ctx.start(), plan.whole.start());
        EXPECT_EQ(plan.right_ctx.end(), plan.whole.end());
    }
}

// Test: Compile-Time Construction
TEST(IntervalPartitionPlan, CompileTimeConstruction) {
    constexpr interval_context ctx{0, 10};
    constexpr binary_cut_desc<10> desc;
    constexpr auto plan = interval_partition_plan::make(ctx, desc, 4);
    
    static_assert(plan.split == 5, "Split should be at 5");
    static_assert(plan.left_size() == 5, "Left size should be 5");
    static_assert(plan.right_size() == 5, "Right size should be 5");
    static_assert(plan.is_legal(), "Plan should be legal");
    static_assert(plan.preserves_size(), "Size should be preserved");
}

// Test: Ordinal to Split Mapping
TEST(IntervalPartitionPlan, OrdinalToSplitMapping) {
    interval_context ctx{10, 20};  // [10, 20), length 10
    binary_cut_desc<10> desc;
    
    // Verify: ordinal i → split at 10 + (i+1)
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        size_t expected_split = 10 + (ord + 1);
        EXPECT_EQ(plan.split, expected_split) << "Failed at ordinal " << ord;
    }
}

// Test: Left-Right Size Symmetry
TEST(IntervalPartitionPlan, LeftRightSizeSymmetry) {
    interval_context ctx{0, 20};
    binary_cut_desc<20> desc;
    
    // Middle cut should produce equal-sized subintervals
    auto middle = interval_partition_plan::make(ctx, desc, 9);  // ord=9 → rel=10
    EXPECT_EQ(middle.split, 10);
    EXPECT_EQ(middle.left_size(), 10);
    EXPECT_EQ(middle.right_size(), 10);
}

// Test: Multiple Plans from Same Context
TEST(IntervalPartitionPlan, MultiplePlansFromSameContext) {
    interval_context ctx{0, 10};
    binary_cut_desc<10> desc;
    
    std::vector<interval_partition_plan> plans;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        plans.push_back(interval_partition_plan::make(ctx, desc, ord));
    }
    
    // Verify: 9 distinct plans
    ASSERT_EQ(plans.size(), 9);
    
    // Verify: All splits are distinct
    for (size_t i = 0; i < plans.size(); ++i) {
        for (size_t j = i + 1; j < plans.size(); ++j) {
            EXPECT_NE(plans[i].split, plans[j].split);
        }
    }
}

// Test: Plan Accessors
TEST(IntervalPartitionPlan, PlanAccessors) {
    interval_context ctx{5, 15};
    binary_cut_desc<10> desc;
    auto plan = interval_partition_plan::make(ctx, desc, 3);  // split at 8
    
    EXPECT_EQ(plan.left_size(), 4);
    EXPECT_EQ(plan.right_size(), 6);
    EXPECT_EQ(plan.split, 9);
    EXPECT_EQ(plan.whole.size(), 10);
}

// Test: Legality Check Rejects Invalid Manual Construction
// Note: interval_partition_plan is a plain value term; callers may construct
// illegal values directly. Use make(...) for invariant-preserving construction.
TEST(IntervalPartitionPlan, LegalityCheckRejectsInvalidManualConstruction) {
    interval_partition_plan bad{
        interval_context{0, 4},
        0,  // Split at boundary (illegal)
        interval_context{0, 1},
        interval_context{1, 4}
    };
    EXPECT_FALSE(bad.is_legal());
}
