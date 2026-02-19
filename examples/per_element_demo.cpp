// demos/per_element_demo.cpp
// End-to-end proof: factored optimisation via per_element_argmin.
// Demonstrates element-level cost and constraints with typed descriptors.
//
// Build:
//   g++ -std=c++20 -Wall -Wextra -Wpedantic -I.. per_element_demo.cpp -o per_element_demo

#include "ctdp/solver/solver.h"
#include <cstdio>

using namespace ctdp;

enum class Strategy { Baseline, Optimised, Aggressive };
enum class Field    { Tag, Length, Value, Checksum };

// Cost table: 4 fields × 3 strategies
constexpr double cost_table[4][3] = {
    { 10.0,  5.0,  2.0 },   // Tag:      Aggressive cheapest
    { 20.0, 12.0, 30.0 },   // Length:   Optimised cheapest
    {  8.0,  3.0,  1.0 },   // Value:    Aggressive cheapest
    { 15.0,  7.0, 25.0 },   // Checksum: Optimised cheapest
};

constexpr auto make_space() {
    return per_element_space<Strategy, 4, 3, Field>{
        .descriptors = { Field::Tag, Field::Length, Field::Value, Field::Checksum },
        .strategies  = { Strategy::Baseline, Strategy::Optimised, Strategy::Aggressive },
    };
}

// Element cost: (Field, choice_index) → double
constexpr auto field_cost = [](Field f, std::size_t choice) -> double {
    return cost_table[static_cast<int>(f)][choice];
};

// Unconstrained: optimal = 2 + 12 + 1 + 7 = 22
constexpr auto solve_unconstrained() {
    auto space = make_space();
    return per_element_argmin(space, field_cost);
}

static_assert(solve_unconstrained().predicted_cost == 22.0);
static_assert(solve_unconstrained().params[0] == Strategy::Aggressive);
static_assert(solve_unconstrained().params[1] == Strategy::Optimised);

// Constrained: no Aggressive (choice 2) at any position
// This is an element predicate — per-position, correct by construction.
constexpr auto no_aggressive = [](Field /*f*/, std::size_t choice) -> bool {
    return choice != 2;  // choice 2 = Aggressive
};

constexpr auto solve_constrained() {
    auto space = make_space();
    return per_element_argmin(space, field_cost, no_aggressive);
}

// Optimal without Aggressive: 5 + 12 + 3 + 7 = 27
static_assert(solve_constrained().predicted_cost == 27.0);
static_assert(solve_constrained().stats.candidates_pruned > 0);

// Via select_and_run — dispatches to per_element_argmin
constexpr auto solve_via_dispatch() {
    auto space = make_space();
    auto cost = additive_cost{
        [](std::size_t pos, Strategy s) constexpr -> double {
            return cost_table[pos][static_cast<std::size_t>(s)];
        }
    };
    return select_and_run(space, cost);
}

static_assert(solve_via_dispatch().predicted_cost == 22.0);

int main() {
    constexpr auto unconstrained = solve_unconstrained();
    constexpr auto constrained   = solve_constrained();

    std::printf("Per-element optimisation demo\n");
    std::printf("4 fields × 3 strategies\n\n");

    std::printf("Unconstrained: cost = %.0f\n", unconstrained.predicted_cost);
    std::printf("  Evaluated: %zu  Pruned: %zu\n",
        unconstrained.stats.candidates_evaluated, unconstrained.stats.candidates_pruned);

    std::printf("\nConstrained (no Aggressive): cost = %.0f\n",
        constrained.predicted_cost);
    std::printf("  Evaluated: %zu  Pruned: %zu\n",
        constrained.stats.candidates_evaluated, constrained.stats.candidates_pruned);

    std::printf("\nCompile-time verified: YES\n");
    return 0;
}
