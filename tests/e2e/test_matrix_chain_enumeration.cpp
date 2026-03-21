// test_matrix_chain_enumeration.cpp - Integration test
// Sprint 8: Matrix chain top-level enumeration
// Author: Andrew Drakeford

#include "ct_dp/space/binary_cut_desc.hpp"
#include "ct_dp/solver/interval_context.hpp"
#include "ct_dp/plan/interval_partition_plan.hpp"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <algorithm>

using namespace ct_dp::space;
using namespace ct_dp::solver;
using namespace ct_dp::plan;

// Test: Matrix Chain Top-Level Enumeration
TEST(Integration, MatrixChainTopLevelEnumeration) {
    // Canonical example: 4 matrices
    // A₀(10×20), A₁(20×30), A₂(30×40), A₃(40×50)
    
    constexpr size_t N = 4;  // Number of matrices (logical length)
    std::array<size_t, N+1> dims{10, 20, 30, 40, 50};  // N+1 dimension points
    
    binary_cut_desc<N> desc;  // Value-oriented descriptor
    interval_context ctx{0, N};  // Solve interval [0, 4)
    
    std::vector<size_t> top_costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {  // ord ∈ {0,1,2}
        // Create partition plan
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        
        size_t k = plan.split;  // k ∈ {1,2,3}
        
        // Cost of top-level split: multiply (A₀...A_{k-1}) × (A_k...A_{N-1})
        // Left result: dims[0] × dims[k]
        // Right result: dims[k] × dims[N]
        // Final multiplication: dims[0] × dims[k] × dims[N]
        size_t cost = dims[0] * dims[k] * dims[N];
        
        top_costs.push_back(cost);
    }
    
    // Verification: Three cuts tried
    ASSERT_EQ(top_costs.size(), 3);
    EXPECT_EQ(top_costs[0], 10 * 20 * 50);  // k=1: 10,000
    EXPECT_EQ(top_costs[1], 10 * 30 * 50);  // k=2: 15,000
    EXPECT_EQ(top_costs[2], 10 * 40 * 50);  // k=3: 20,000
    
    // Best top-level split is k=1
    auto min_cost = *std::min_element(top_costs.begin(), top_costs.end());
    EXPECT_EQ(min_cost, 10000);
}

// Test: Partition Plan Generation
TEST(Integration, PartitionPlanGeneration) {
    constexpr size_t N = 4;
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    std::vector<interval_partition_plan> plans;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        
        EXPECT_TRUE(plan.is_legal());
        EXPECT_TRUE(plan.preserves_size());
        
        plans.push_back(plan);
    }
    
    ASSERT_EQ(plans.size(), 3);
    
    // Verify subinterval sizes for each plan
    EXPECT_EQ(plans[0].left_ctx.size(), 1);   // k=1: [0,1) and [1,4)
    EXPECT_EQ(plans[0].right_ctx.size(), 3);
    
    EXPECT_EQ(plans[1].left_ctx.size(), 2);   // k=2: [0,2) and [2,4)
    EXPECT_EQ(plans[1].right_ctx.size(), 2);
    
    EXPECT_EQ(plans[2].left_ctx.size(), 3);   // k=3: [0,3) and [3,4)
    EXPECT_EQ(plans[2].right_ctx.size(), 1);
}

// Test: Small Matrix Chain (3 matrices)
TEST(Integration, SmallMatrixChain) {
    // 3 matrices: A₀(5×10), A₁(10×15), A₂(15×20)
    
    constexpr size_t N = 3;
    std::array<size_t, N+1> dims{5, 10, 15, 20};
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    std::vector<size_t> costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        size_t k = plan.split;
        size_t cost = dims[0] * dims[k] * dims[N];
        costs.push_back(cost);
    }
    
    ASSERT_EQ(costs.size(), 2);
    EXPECT_EQ(costs[0], 5 * 10 * 20);   // k=1: 1,000
    EXPECT_EQ(costs[1], 5 * 15 * 20);   // k=2: 1,500
    
    auto min_cost = *std::min_element(costs.begin(), costs.end());
    EXPECT_EQ(min_cost, 1000);
}

// Test: Larger Matrix Chain (5 matrices)
TEST(Integration, LargerMatrixChain) {
    // 5 matrices
    constexpr size_t N = 5;
    std::array<size_t, N+1> dims{10, 20, 30, 40, 50, 60};
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    std::vector<size_t> costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        size_t k = plan.split;
        size_t cost = dims[0] * dims[k] * dims[N];
        costs.push_back(cost);
    }
    
    EXPECT_EQ(costs.size(), 4);  // 4 possible top-level cuts
    
    // Verify all plans are legal
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        EXPECT_TRUE(plan.is_legal());
        EXPECT_TRUE(plan.preserves_size());
    }
}

// Test: Uniform Dimensions
TEST(Integration, UniformDimensions) {
    // All matrices have same dimensions (10×10)
    constexpr size_t N = 4;
    std::array<size_t, N+1> dims{10, 10, 10, 10, 10};
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    std::vector<size_t> costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        size_t k = plan.split;
        size_t cost = dims[0] * dims[k] * dims[N];
        costs.push_back(cost);
    }
    
    // All cuts have same cost (10 * 10 * 10 = 1000)
    for (auto cost : costs) {
        EXPECT_EQ(cost, 1000);
    }
}

// Test: Asymmetric Dimensions
TEST(Integration, AsymmetricDimensions) {
    // Asymmetric matrix dimensions
    constexpr size_t N = 4;
    std::array<size_t, N+1> dims{1, 100, 1, 100, 1};
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    std::vector<size_t> costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        size_t k = plan.split;
        size_t cost = dims[0] * dims[k] * dims[N];
        costs.push_back(cost);
    }
    
    // Verify different costs for different cuts
    EXPECT_NE(costs[0], costs[1]);
    EXPECT_NE(costs[1], costs[2]);
}

// Test: Descriptor + Context + Plan Integration
TEST(Integration, DescriptorContextPlanIntegration) {
    constexpr size_t N = 4;
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    // Verify integration chain: descriptor → context → plan
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        // Step 1: Descriptor provides relative cut
        size_t rel = desc.ordinal_to_relative_cut(ord);
        
        // Step 2: Plan integrates descriptor with context
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        
        // Step 3: Verify absolute cut matches integration
        size_t expected_split = ctx.start() + rel;
        EXPECT_EQ(plan.split, expected_split);
        
        // Step 4: Verify subintervals match
        EXPECT_EQ(plan.left_ctx.start(), ctx.start());
        EXPECT_EQ(plan.left_ctx.end(), plan.split);
        EXPECT_EQ(plan.right_ctx.start(), plan.split);
        EXPECT_EQ(plan.right_ctx.end(), ctx.end());
    }
}

// Test: Minimal Matrix Chain (2 matrices)
TEST(Integration, MinimalMatrixChain) {
    // 2 matrices: A₀(10×20), A₁(20×30)
    constexpr size_t N = 2;
    std::array<size_t, N+1> dims{10, 20, 30};
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    EXPECT_EQ(desc.size(), 1);  // Only one cut
    
    auto plan = interval_partition_plan::make(ctx, desc, 0);
    EXPECT_EQ(plan.split, 1);
    EXPECT_EQ(plan.left_size(), 1);
    EXPECT_EQ(plan.right_size(), 1);
    
    size_t cost = dims[0] * dims[1] * dims[N];
    EXPECT_EQ(cost, 10 * 20 * 30);
}

// Test: Cost Function with Plan
TEST(Integration, CostFunctionWithPlan) {
    constexpr size_t N = 4;
    std::array<size_t, N+1> dims{10, 20, 30, 40, 50};
    
    auto cost_fn = [&dims, N](const interval_partition_plan& plan) {
        return dims[0] * dims[plan.split] * dims[N];
    };
    
    binary_cut_desc<N> desc;
    interval_context ctx{0, N};
    
    std::vector<size_t> costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        costs.push_back(cost_fn(plan));
    }
    
    EXPECT_EQ(costs[0], 10000);
    EXPECT_EQ(costs[1], 15000);
    EXPECT_EQ(costs[2], 20000);
}

// Test: Non-Zero Start Interval
TEST(Integration, NonZeroStartInterval) {
    // Simulating a subproblem: matrices [2..5] out of larger chain
    constexpr size_t N = 3;  // 3 matrices in subproblem
    std::array<size_t, N+1> dims{30, 40, 50, 60};  // Dims for subproblem
    
    binary_cut_desc<N> desc;
    interval_context ctx{2, 5};  // Subproblem interval [2, 5)
    
    std::vector<size_t> costs;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        auto plan = interval_partition_plan::make(ctx, desc, ord);
        
        // Verify split is in correct range
        EXPECT_GE(plan.split, 3);
        EXPECT_LE(plan.split, 4);
        
        // Cost calculation uses absolute indices
        size_t k_local = plan.split - ctx.start();
        size_t cost = dims[0] * dims[k_local] * dims[N];
        costs.push_back(cost);
    }
    
    EXPECT_EQ(costs.size(), 2);
}
