// demos/matrix_chain_demo.cpp
// End-to-end proof: matrix chain multiplication via interval_dp.
// Compile-time + runtime verification of the Cormen et al. test case.
//
// Build:
//   g++ -std=c++20 -Wall -Wextra -Wpedantic -I.. matrix_chain_demo.cpp -o matrix_chain_demo
//
// Expected output:
//   Matrix chain multiplication — Cormen et al. (CLRS)
//   Dimensions: [30, 35, 15, 5, 10, 20, 25]
//   Optimal cost: 15125 scalar multiplications
//   Subproblems solved: 15
//   Split-point evaluations: 35
//   Compile-time verified: YES

#include "ctdp/solver/solver.h"
#include <cstdio>

using namespace ctdp;

// Full compile-time computation
constexpr auto solve_cormen() {
    constexpr std::array<std::size_t, 7> dims{30, 35, 15, 5, 10, 20, 25};
    auto space = interval_split_space<7>{.n = 6};
    auto cost  = make_chain_cost(dims);
    return interval_dp(space, cost);
}

// Compile-time verification — these are the acceptance criteria
static_assert(solve_cormen().predicted_cost == 15125.0,
    "Cormen matrix chain: expected optimal cost 15125");
static_assert(solve_cormen().is_feasible(),
    "Cormen matrix chain: result must be feasible");
static_assert(solve_cormen().stats.candidates_pruned == 0,
    "interval_dp: no constraint pruning");
static_assert(solve_cormen().stats.subproblems_evaluated == 15,
    "interval_dp: 15 subproblems for 6 matrices");
static_assert(solve_cormen().stats.candidates_evaluated == 35,
    "interval_dp: 35 split-point evaluations for 6 matrices");

// Also verify via select_and_run dispatch
constexpr auto solve_via_dispatch() {
    constexpr std::array<std::size_t, 7> dims{30, 35, 15, 5, 10, 20, 25};
    auto space = interval_split_space<7>{.n = 6};
    auto cost  = make_chain_cost(dims);
    return select_and_run(space, cost);
}

static_assert(solve_via_dispatch().predicted_cost == 15125.0,
    "select_and_run must dispatch to interval_dp and find same result");

int main() {
    constexpr auto result = solve_cormen();

    std::printf("Matrix chain multiplication — Cormen et al. (CLRS)\n");
    std::printf("Dimensions: [30, 35, 15, 5, 10, 20, 25]\n");
    std::printf("Optimal cost: %.0f scalar multiplications\n", result.predicted_cost);
    std::printf("Subproblems solved: %zu\n", result.stats.subproblems_evaluated);
    std::printf("Split-point evaluations: %zu\n", result.stats.candidates_evaluated);
    std::printf("Compile-time verified: YES\n");

    return 0;
}
