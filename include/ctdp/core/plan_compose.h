// core/plan_compose.h - Composing plans from independent subproblems
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// When a problem decomposes into independent subproblems, each solved
// separately, plan_compose combines the sub-plans into a single
// composite_plan with aggregated cost and statistics.
//
// The key design choice is HOW costs combine:
// - Additive (default): total cost = sum of sub-costs. This is correct
//   when subproblems execute sequentially and costs are latencies, or
//   when costs represent resource usage that accumulates.
// - Maximum: total cost = max of sub-costs. Correct when subproblems
//   execute in parallel and the bottleneck determines total cost.
// - Custom: caller provides a combine function.
//
// composite_plan (defined in plan.h) stores the sub-plans and combined cost.
// plan_compose.h provides the functions to construct composite_plans.

#ifndef CTDP_CORE_PLAN_COMPOSE_H
#define CTDP_CORE_PLAN_COMPOSE_H

#include "plan.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace ctdp {

// =========================================================================
// compose_additive: total cost = sum of sub-costs (default)
// =========================================================================

/// Compose plans with additive cost combination.
///
/// Use when subproblems are independent and costs accumulate
/// (sequential execution, resource consumption).
///
/// Example:
/// ```cpp
/// auto plan_a = solver_a(space_a, cost_fn_a);  // plan<CandA>
/// auto plan_b = solver_b(space_b, cost_fn_b);  // plan<CandB>
/// auto combined = compose_additive(plan_a, plan_b);
/// // combined.combined_cost == plan_a.predicted_cost + plan_b.predicted_cost
/// ```
template<typename... Candidates>
[[nodiscard]] constexpr composite_plan<Candidates...>
compose_additive(plan<Candidates> const&... plans) {
    double total = 0.0;
    bool any_infeasible = false;

    // Sum costs; propagate infeasibility
    ((any_infeasible = any_infeasible || plans.is_infeasible()), ...);
    if (any_infeasible) {
        return composite_plan<Candidates...>{};  // Infeasible
    }

    ((total += plans.predicted_cost), ...);

    return composite_plan<Candidates...>{
        std::tuple<plan<Candidates>...>{plans...},
        total
    };
}

/// Move overload for compose_additive.
template<typename... Candidates>
[[nodiscard]] constexpr composite_plan<Candidates...>
compose_additive(plan<Candidates>&&... plans) {
    double total = 0.0;
    bool any_infeasible = false;

    ((any_infeasible = any_infeasible || plans.is_infeasible()), ...);
    if (any_infeasible) {
        return composite_plan<Candidates...>{};
    }

    ((total += plans.predicted_cost), ...);

    return composite_plan<Candidates...>{
        std::tuple<plan<Candidates>...>{std::move(plans)...},
        total
    };
}

// =========================================================================
// compose_max: total cost = max of sub-costs
// =========================================================================

/// Compose plans with max-cost combination.
///
/// Use when subproblems execute in parallel and the bottleneck
/// determines total cost.
///
/// Example:
/// ```cpp
/// auto combined = compose_max(plan_gpu, plan_cpu);
/// // combined.combined_cost == max(plan_gpu.cost, plan_cpu.cost)
/// ```
template<typename... Candidates>
[[nodiscard]] constexpr composite_plan<Candidates...>
compose_max(plan<Candidates> const&... plans) {
    double worst = 0.0;
    bool any_infeasible = false;

    ((any_infeasible = any_infeasible || plans.is_infeasible()), ...);
    if (any_infeasible) {
        return composite_plan<Candidates...>{};
    }

    ((worst = (plans.predicted_cost > worst ? plans.predicted_cost : worst)), ...);

    return composite_plan<Candidates...>{
        std::tuple<plan<Candidates>...>{plans...},
        worst
    };
}

// =========================================================================
// compose_custom: caller-provided cost combiner
// =========================================================================

/// Compose plans with a custom cost combination function.
///
/// The combiner receives all sub-plan costs and returns the combined cost.
///
/// Example:
/// ```cpp
/// auto combined = compose_custom(
///     [](double a, double b) { return 0.7 * a + 0.3 * b; },
///     plan_latency, plan_power
/// );
/// ```
template<typename CostCombiner, typename... Candidates>
[[nodiscard]] constexpr composite_plan<Candidates...>
compose_custom(CostCombiner combiner, plan<Candidates> const&... plans) {
    bool any_infeasible = false;
    ((any_infeasible = any_infeasible || plans.is_infeasible()), ...);
    if (any_infeasible) {
        return composite_plan<Candidates...>{};
    }

    double combined_cost = combiner(plans.predicted_cost...);

    return composite_plan<Candidates...>{
        std::tuple<plan<Candidates>...>{plans...},
        combined_cost
    };
}

// =========================================================================
// Pair convenience: compose exactly two plans
// =========================================================================

/// Compose two plans with additive cost (convenience shorthand).
template<typename CandA, typename CandB>
[[nodiscard]] constexpr composite_plan<CandA, CandB>
compose(plan<CandA> const& a, plan<CandB> const& b) {
    return compose_additive(a, b);
}

} // namespace ctdp

#endif // CTDP_CORE_PLAN_COMPOSE_H
