#include "ct_dp/solver/interval_solver.hpp"
#include "ct_dp/solver/triangular_memo.hpp"
#include "ct_dp/solver/map_memo.hpp"
#include "ct_dp/plan/interval_partition_plan.hpp"
#include <gtest/gtest.h>
#include <array>
#include <span>
#include <cstdint>

using namespace ct_dp::solver;

/// Matrix chain multiplication recurrence
///
/// Problem: Given matrices A₀, A₁, ..., Aₙ₋₁ with dimensions,
/// find the optimal parenthesization that minimizes scalar multiplications.
///
/// dims[i] gives the left dimension of matrix Aᵢ
/// dims[i+1] gives the right dimension of matrix Aᵢ
/// So dims has size n+1 for n matrices
struct matrix_chain_recurrence {
    using value_type = std::uint64_t;
    
    std::span<const size_t> dims;  // size = num_matrices + 1
    
    std::optional<value_type> base_case(interval_context ctx) const {
        // Single matrix [i, i+1) has zero multiplication cost
        return ctx.size() == 1 
            ? std::optional<value_type>{0} 
            : std::nullopt;
    }
    
    value_type combine(const interval_partition_plan& plan,
                       value_type left,
                       value_type right) const {
        // Extract interval boundaries from plan
        const size_t i = plan.whole.i;
        const size_t k = plan.split;
        const size_t j = plan.whole.j;
        
        // Cost to multiply result matrices:
        // Left result: dims[i] × dims[k]
        // Right result: dims[k] × dims[j]
        // Multiplication: dims[i] × dims[k] × dims[j] scalar ops
        return left + right
             + static_cast<value_type>(dims[i]) * dims[k] * dims[j];
    }
};

// Test single matrix (base case)
TEST(MatrixChainSolver, SingleMatrix) {
    std::array<size_t, 2> dims{10, 20};  // One 10×20 matrix
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{1};
    
    auto result = solver.solve(interval_context{0, 1}, memo);
    
    // Single matrix has zero multiplication cost
    EXPECT_EQ(result, 0u);
}

// Test two matrices
TEST(MatrixChainSolver, TwoMatrices) {
    std::array<size_t, 3> dims{10, 20, 30};  // A₀: 10×20, A₁: 20×30
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{2};
    
    auto result = solver.solve(interval_context{0, 2}, memo);
    
    // Only one way to multiply: A₀ * A₁
    // Cost: 10 × 20 × 30 = 6000
    EXPECT_EQ(result, 6000u);
}

// Test three matrices
TEST(MatrixChainSolver, ThreeMatrices) {
    std::array<size_t, 4> dims{10, 20, 30, 40};
    // A₀: 10×20, A₁: 20×30, A₂: 30×40
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{3};
    
    auto result = solver.solve(interval_context{0, 3}, memo);
    
    // Two parenthesizations:
    // (A₀A₁)A₂: (10×20×30) + (10×30×40) = 6000 + 12000 = 18000
    // A₀(A₁A₂): (20×30×40) + (10×20×40) = 24000 + 8000 = 32000
    // Optimal: 18000
    EXPECT_EQ(result, 18000u);
}

// Test classic four matrices - CORRECTED to 38000
TEST(MatrixChainSolver, ClassicFourMatrices) {
    std::array<size_t, 5> dims{10, 20, 30, 40, 50};
    // A₀: 10×20, A₁: 20×30, A₂: 30×40, A₃: 40×50
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{4};
    
    auto result = solver.solve(interval_context{0, 4}, memo);
    
    // CORRECT optimal parenthesization: ((A₀A₁)A₂)A₃
    // Step 1: A₀A₁ = 10 × 20 × 30 = 6,000
    // Step 2: (A₀A₁)A₂ = 10 × 30 × 40 = 12,000
    // Step 3: ((A₀A₁)A₂)A₃ = 10 × 40 × 50 = 20,000
    // Total: 6,000 + 12,000 + 20,000 = 38,000
    EXPECT_EQ(result, 38000u);  // NOT 30000!
}

// Test CLRS alternative example
TEST(MatrixChainSolver, CLRSAlternativeExample) {
    std::array<size_t, 5> dims{5, 10, 3, 12, 5};
    // A₀: 5×10, A₁: 10×3, A₂: 3×12, A₃: 12×5
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{4};
    
    auto result = solver.solve(interval_context{0, 4}, memo);
    
    // Optimal cost from CLRS: 405
    EXPECT_EQ(result, 405u);
}

// Test asymmetric dimensions
TEST(MatrixChainSolver, AsymmetricDimensions) {
    std::array<size_t, 4> dims{1, 100, 1, 100};
    // A₀: 1×100, A₁: 100×1, A₂: 1×100
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{3};
    
    auto result = solver.solve(interval_context{0, 3}, memo);
    
    // (A₀A₁)A₂: (1×100×1) + (1×1×100) = 100 + 100 = 200
    // A₀(A₁A₂): (100×1×100) + (1×100×100) = 10000 + 10000 = 20000
    // Optimal: 200
    EXPECT_EQ(result, 200u);
}

// Test memo size is correct
TEST(MatrixChainSolver, MemoSizeCorrect) {
    std::array<size_t, 5> dims{10, 20, 30, 40, 50};
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{4};
    
    solver.solve(interval_context{0, 4}, memo);
    
    // For n=4 matrices, we have subproblems [i,j) where i < j <= 4
    // [0,1), [0,2), [0,3), [0,4)
    // [1,2), [1,3), [1,4)
    // [2,3), [2,4)
    // [3,4)
    // Total: 10 subproblems = 4*5/2
    EXPECT_EQ(memo.size(), 10);
}

// Test larger problem
TEST(MatrixChainSolver, LargerProblem) {
    std::array<size_t, 11> dims{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110};
    // 10 matrices
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{10};
    
    auto result = solver.solve(interval_context{0, 10}, memo);
    
    // Should complete and return some cost
    EXPECT_GT(result, 0u);
    
    // Verify all subproblems were computed
    // For n=10, we have n(n+1)/2 = 55 subproblems
    EXPECT_EQ(memo.size(), 55);
}

// Test with map_memo (unbounded)
TEST(MatrixChainSolver, WorksWithMapMemo) {
    std::array<size_t, 5> dims{10, 20, 30, 40, 50};
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    map_memo<std::uint64_t> memo;  // Sparse memo
    
    auto result = solver.solve(interval_context{0, 4}, memo);
    
    EXPECT_EQ(result, 38000u);
    EXPECT_EQ(memo.size(), 10);  // Same number of subproblems
}

// Test verifying combine uses full plan
TEST(MatrixChainSolver, CombineUsesFullPlan) {
    std::array<size_t, 4> dims{2, 3, 4, 5};
    // A₀: 2×3, A₁: 3×4, A₂: 4×5
    
    matrix_chain_recurrence rec{dims};
    interval_solver solver{rec};
    triangular_memo<std::uint64_t> memo{3};
    
    auto result = solver.solve(interval_context{0, 3}, memo);
    
    // (A₀A₁)A₂: (2×3×4) + (2×4×5) = 24 + 40 = 64
    // A₀(A₁A₂): (3×4×5) + (2×3×5) = 60 + 30 = 90
    // Optimal: 64
    EXPECT_EQ(result, 64u);
}
