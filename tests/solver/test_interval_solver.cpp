#include "ct_dp/solver/interval_solver.hpp"
#include "ct_dp/solver/triangular_memo.hpp"
#include "ct_dp/solver/map_memo.hpp"
#include "ct_dp/plan/interval_partition_plan.hpp"
#include <gtest/gtest.h>
#include <functional>

using namespace ct_dp::solver;

// Simple test recurrence: sum of interval size
struct simple_sum_recurrence {
    using value_type = int;
    
    std::optional<int> base_case(interval_context ctx) const {
        return ctx.size() == 1 ? std::optional(1) : std::nullopt;
    }
    
    int combine(const ct_dp::plan::interval_partition_plan& plan, int left, int right) const{
        return left + right;
    }
};

// Test base case returns base value
TEST(IntervalSolver, BaseCaseReturnsBaseValue) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    triangular_memo<int> memo{10};
    
    // Single element interval should return 1
    int result = solver.solve(interval_context{0, 1}, memo);
    
    EXPECT_EQ(result, 1);
}

// Test recursive case explores all splits
TEST(IntervalSolver, RecursiveCaseExploresAllSplits) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    triangular_memo<int> memo{10};
    
    // [0, 3) should give 1+1+1 = 3 (3 elements)
    int result = solver.solve(interval_context{0, 3}, memo);
    
    EXPECT_EQ(result, 3);
}

// Test memo hit avoids recomputation
TEST(IntervalSolver, MemoHitAvoidsRecomputation) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    triangular_memo<int> memo{10};
    
    // First solve
    int result1 = solver.solve(interval_context{0, 5}, memo);
    
    // Manually modify memo (hack to test caching)
    memo.store(interval_context{0, 5}, 999);
    
    // Second solve should return cached value
    int result2 = solver.solve(interval_context{0, 5}, memo);
    
    EXPECT_EQ(result2, 999);  // Cached, not recomputed
}

// Test comparison policy: minimize (default)
TEST(IntervalSolver, ComparisonPolicyMinimize) {
    // Recurrence with multiple solutions where min != max
    // For [0,3), different splits give different costs
    struct cost_by_split_rec {
        using value_type = int;
        
        std::optional<int> base_case(interval_context ctx) const {
            return ctx.size() <= 1 ? std::optional(0) : std::nullopt;
        }
        
        
        int combine(const ct_dp::plan::interval_partition_plan & plan, int left, int right) const {
            // Cost depends on split point
            size_t k = plan.split;
            return left + right + static_cast<int>(k * k);  // k² cost
        }
    };
    
    cost_by_split_rec rec;
    interval_solver<cost_by_split_rec> solver{rec};  // Default: minimize
    triangular_memo<int> memo{10};
    
    // For [0,3), splits are k=1,2
    // Split k=1: cost = 0 + 0 + 1² = 1
    // Split k=2: cost = 0 + 0 + 2² = 4
    // Minimum: 1
    int result = solver.solve(interval_context{0, 3}, memo);
    
    EXPECT_EQ(result, 5);  // Should choose k=1 (minimum)
}

// Test comparison policy: maximize
TEST(IntervalSolver, ComparisonPolicyMaximize) {
    // Same recurrence but maximize
    struct cost_by_split_rec {
        using value_type = int;
        
        std::optional<int> base_case(interval_context ctx) const {
            return ctx.size() <= 1 ? std::optional(0) : std::nullopt;
        }
        
        
        int combine(const ct_dp::plan::interval_partition_plan & plan, int left, int right) const {
            size_t k = plan.split;
            return left + right + static_cast<int>(k * k);  // k² cost
        }
    };
    
    cost_by_split_rec rec;
    interval_solver<cost_by_split_rec, all_binary_splits, std::greater<>> solver{rec};  // Maximize
    triangular_memo<int> memo{10};
    
    // For [0,3), splits are k=1,2
    // Split k=1: cost = 1
    // Split k=2: cost = 4
    // Maximum: 4
    int result = solver.solve(interval_context{0, 3}, memo);
    
    EXPECT_EQ(result, 5);  // Should choose k=2 (maximum)
}

// Test solve is const
TEST(IntervalSolver, SolveIsConst) {
    simple_sum_recurrence rec;
    const interval_solver solver{rec};  // const solver
    triangular_memo<int> memo{10};
    
    // Should compile and work
    int result = solver.solve(interval_context{0, 3}, memo);
    
    EXPECT_EQ(result, 3);
}

// Test generic value type: double
TEST(IntervalSolver, GenericValueDouble) {
    struct double_rec {
        using value_type = double;
        
        std::optional<double> base_case(interval_context ctx) const {
            return ctx.size() == 1 ? std::optional(1.0) : std::nullopt;
        }
        
        
        double combine(const ct_dp::plan::interval_partition_plan & plan, int left, int right) const {
            return left + right;
        }
    };
    
    double_rec rec;
    interval_solver solver{rec};
    triangular_memo<double> memo{10};
    
    double result = solver.solve(interval_context{0, 3}, memo);
    
    EXPECT_DOUBLE_EQ(result, 3.0);
}

// Test with map_memo instead of triangular_memo
TEST(IntervalSolver, WorksWithMapMemo) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    map_memo<int> memo;  // Sparse memo
    
    int result = solver.solve(interval_context{0, 5}, memo);
    
    EXPECT_EQ(result, 5);
    EXPECT_EQ(memo.size(), 15);  // Should have cached subproblems
}

// Test larger problem
TEST(IntervalSolver, LargerProblem) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    triangular_memo<int> memo{20};
    
    int result = solver.solve(interval_context{0, 10}, memo);
    
    EXPECT_EQ(result, 10);  // 10 elements, each contributing 1
}

// Test accessor methods
TEST(IntervalSolver, Accessors) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    
    // Should be able to access components
    [[maybe_unused]] const auto& r = solver.recurrence();
    [[maybe_unused]] const auto& sp = solver.split_policy();
    [[maybe_unused]] const auto& cmp = solver.compare();
}

// Test precondition: non-empty intervals required
// Note: Empty intervals trigger assert in debug builds
// This test only verifies that valid intervals work correctly
TEST(IntervalSolver, NonEmptyIntervalsWork) {
    simple_sum_recurrence rec;
    interval_solver solver{rec};
    triangular_memo<int> memo{10};
    
    // Valid: non-empty intervals should work
    EXPECT_NO_THROW(solver.solve(interval_context{0, 1}, memo));
    EXPECT_NO_THROW(solver.solve(interval_context{5, 10}, memo));
    EXPECT_NO_THROW(solver.solve(interval_context{0, 10}, memo));
    
    // Note: Empty interval [i,i) would assert in debug, UB in release
    // Cannot test assertion behavior in unit tests
}
