// core/plan.h - Core output type for dynamic programming framework
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// plan<Candidate> is the central vocabulary type of the framework.
// Every solver returns a plan. Every deployment consumes a plan.
//
// The plan is parameterized ONLY on Candidate type (from search space).
// This is the key genericity mechanism - equivalent to BGL's graph concepts.
//
// Cost is fixed as double for simplicity:
// - Most DP problems use floating-point costs
// - Integer costs convert to double without loss (up to 2^53)
// - Multi-objective handled by plan_set, not individual plans
// - Can template on Cost later if truly needed
//
// plan<C> lives in core/ (not solver/) to break dependency cycles:
// - Solver produces plan<C>
// - Instantiation consumes plan<C>
// - If plan<C> were in solver/, instantiation would depend on solver
// - By putting it in core/, both tiers depend downward

#ifndef CTDP_CORE_PLAN_H
#define CTDP_CORE_PLAN_H

#include "solve_stats.h"
#include <concepts>
#include <limits>
#include <tuple>

namespace ctdp {

/// Result of a dynamic programming solve.
///
/// Contains the optimal candidate, predicted cost, and solve statistics.
/// This is the central output type of the framework - all solvers return plan<C>.
///
/// Template parameter:
/// - Candidate: Type from search space (per_element_candidate, interval_candidate, etc.)
///
/// Example:
/// ```cpp
/// auto result = sequence_dp(space, cost_fn);
/// // result is plan<per_element_candidate<Strategy, 10>>
///
/// std::cout << "Cost: " << result.predicted_cost << "\n";
/// std::cout << "Evaluated: " << result.stats.subproblems_evaluated << "\n";
///
/// // Execute the plan
/// execute_plan(result, descriptors);
/// ```
template<typename Candidate>
struct plan {
    /// The optimal candidate found by the solver.
    /// Type is determined by the search space.
    Candidate params;
    
    /// Predicted cost of this plan.
    /// - For feasible solutions: non-negative cost
    /// - For infeasible solutions: infinity
    /// - Multi-objective costs handled by plan_set (Pareto frontier)
    double predicted_cost;
    
    /// Statistics from the solve process.
    /// Tracks subproblems evaluated, memoization hits, etc.
    solve_stats stats;
    
    // =============================================================================
    // Constructors
    // =============================================================================
    
    /// Default constructor - creates infeasible plan.
    constexpr plan() 
        : params{}, 
          predicted_cost{std::numeric_limits<double>::infinity()},
          stats{} 
    {}
    
    /// Construct plan with candidate and cost (default stats).
    constexpr plan(Candidate p, double cost)
        : params{std::move(p)},
          predicted_cost{cost},
          stats{}
    {}
    
    /// Construct plan with all fields.
    constexpr plan(Candidate p, double cost, solve_stats s)
        : params{std::move(p)},
          predicted_cost{cost},
          stats{std::move(s)}
    {}
    
    // =============================================================================
    // Query
    // =============================================================================
    
    /// Check if plan is feasible (cost is finite).
    [[nodiscard]] constexpr bool is_feasible() const {
        return predicted_cost < std::numeric_limits<double>::infinity()
            && predicted_cost == predicted_cost;  // NaN check
    }
    
    /// Check if plan is infeasible (cost is infinity).
    [[nodiscard]] constexpr bool is_infeasible() const {
        return !is_feasible();
    }
    
    // =============================================================================
    // Comparison
    // =============================================================================
    
    /// Plans are ordered by cost (lower is better).
    /// Infeasible plans (cost = infinity) compare greater than any feasible plan.
    constexpr bool operator<(plan const& other) const {
        return predicted_cost < other.predicted_cost;
    }
    
    constexpr bool operator>(plan const& other) const {
        return predicted_cost > other.predicted_cost;
    }
    
    constexpr bool operator<=(plan const& other) const {
        return predicted_cost <= other.predicted_cost;
    }
    
    constexpr bool operator>=(plan const& other) const {
        return predicted_cost >= other.predicted_cost;
    }
    
    /// Equality compares candidate and cost (not stats - stats are metadata).
    /// Two plans are equal if they represent the same solution at the same cost.
    constexpr bool operator==(plan const& other) const
        requires std::equality_comparable<Candidate>
    {
        return params == other.params && predicted_cost == other.predicted_cost;
    }
    
    constexpr bool operator!=(plan const& other) const
        requires std::equality_comparable<Candidate>
    {
        return !(*this == other);
    }
    
    /// Compare by cost only (ignoring candidate and stats).
    /// Use when you only care about optimality, not which solution achieves it.
    [[nodiscard]] constexpr bool cost_equal(plan const& other) const {
        return predicted_cost == other.predicted_cost;
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Create an infeasible plan.
/// Used when no valid solution exists.
template<typename Candidate>
constexpr plan<Candidate> make_infeasible_plan() {
    return plan<Candidate>{};  // Default constructor creates infeasible
}

/// Check if a plan is better than another (strict improvement).
template<typename Candidate>
constexpr bool is_better(plan<Candidate> const& a, plan<Candidate> const& b) {
    return a.predicted_cost < b.predicted_cost;
}

/// Check if a plan is at least as good as another.
template<typename Candidate>
constexpr bool is_at_least_as_good(plan<Candidate> const& a, plan<Candidate> const& b) {
    return a.predicted_cost <= b.predicted_cost;
}

/// Get the better of two plans (lower cost).
template<typename Candidate>
constexpr plan<Candidate> const& min_plan(plan<Candidate> const& a, plan<Candidate> const& b) {
    return (a < b) ? a : b;
}

/// Get the worse of two plans (higher cost).
template<typename Candidate>
constexpr plan<Candidate> const& max_plan(plan<Candidate> const& a, plan<Candidate> const& b) {
    return (a > b) ? a : b;
}

// =============================================================================
// Composite Plans (for plan composition)
// =============================================================================

/// Composite plan combining multiple sub-plans.
///
/// Used by plan_compose.h to represent composed optimization.
/// Example: Jointly optimize parsing strategy + layout strategy.
///
/// Template parameters:
/// - Candidates...: Candidate types from each sub-problem
template<typename... Candidates>
struct composite_plan {
    /// Sub-plans for each component problem.
    std::tuple<plan<Candidates>...> sub_plans;
    
    /// Combined cost (how sub-costs are combined depends on composition operator).
    double combined_cost;
    
    /// Aggregate statistics from all sub-solves.
    solve_stats combined_stats;
    
    /// Default constructor - all infeasible.
    constexpr composite_plan()
        : sub_plans{},
          combined_cost{std::numeric_limits<double>::infinity()},
          combined_stats{}
    {}
    
    /// Construct from sub-plans and combined cost.
    constexpr composite_plan(
        std::tuple<plan<Candidates>...> plans,
        double cost
    ) : sub_plans{std::move(plans)},
        combined_cost{cost},
        combined_stats{}
    {
        // Aggregate stats from sub-plans
        std::apply([this](auto const&... ps) {
            ((combined_stats += ps.stats), ...);
        }, sub_plans);
    }
    
    /// Check if composite plan is feasible.
    [[nodiscard]] constexpr bool is_feasible() const {
        return combined_cost < std::numeric_limits<double>::infinity()
            && combined_cost == combined_cost;  // NaN check
    }
    
    /// Access i-th sub-plan.
    template<size_t I>
    [[nodiscard]] constexpr auto const& get() const {
        return std::get<I>(sub_plans);
    }
    
    template<size_t I>
    [[nodiscard]] constexpr auto& get() {
        return std::get<I>(sub_plans);
    }
};

} // namespace ctdp

#endif // CTDP_CORE_PLAN_H
