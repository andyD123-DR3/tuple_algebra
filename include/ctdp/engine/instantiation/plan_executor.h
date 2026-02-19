// engine/instantiation/plan_executor.h — Execute a plan via dispatch table
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// This is the final link in the framework pipeline:
//
//   Search space  →  Solver  →  plan<Candidate>
//                                     │
//   strategy_map + dispatch_table ─────┘
//                                     │
//   plan_executor  →  runtime actions (function calls, codegen, etc.)
//
// execute_plan iterates through the plan's strategy assignments and
// invokes the corresponding implementation via the dispatch table.
//
// Three overloads:
//
// 1. execute_plan(plan, dispatch, visitor)
//    Calls visitor(position, impl) for each position.
//    Works with per_element_candidate directly.
//
// 2. execute_plan(plan, descriptors, dispatch, visitor)
//    Calls visitor(position, descriptor, impl) for each position.
//    Uses candidate_traits for generic traversal.
//
// 3. execute_plan_with_context(plan, dispatch, context, action)
//    Calls action(position, impl, context) — for in-place mutation.
//
// All three are constexpr.
//
// USAGE:
// ```cpp
// // Solve
// auto result = beam_search(space, cost_fn, budget);
//
// // Build dispatch
// auto dt = make_uniform_dispatch<Strategy, FnPtr>(
//     std::pair{Strategy::Fast, &fast_impl},
//     std::pair{Strategy::Safe, &safe_impl}
// );
//
// // Execute
// execute_plan(result, dt, [](std::size_t pos, auto fn) {
//     fn(data[pos]);
// });
// ```

#ifndef CTDP_ENGINE_PLAN_EXECUTOR_H
#define CTDP_ENGINE_PLAN_EXECUTOR_H

#include "dispatch_table.h"
#include "../../core/plan.h"
#include "../../core/candidate_traits.h"
#include "../../core/per_element_candidate.h"

#include <cstddef>
#include <type_traits>

namespace ctdp {

// =========================================================================
// Concept: per_element_plan — direct array-indexed plan
// =========================================================================

namespace detail {

template<typename C>
struct is_per_element_candidate : std::false_type {};

template<typename S, std::size_t N>
struct is_per_element_candidate<per_element_candidate<S, N>> : std::true_type {};

} // namespace detail

template<typename C>
concept per_element_plan = detail::is_per_element_candidate<C>::value;

// =========================================================================
// execute_plan: plan + dispatch → visitor(position, impl)
// =========================================================================

/// Execute a per-element plan through a dispatch table.
///
/// For each position i in [0, N), looks up the strategy chosen by the
/// plan, resolves it to an implementation via the dispatch table, and
/// calls visitor(i, impl).
///
/// Example:
/// ```cpp
/// execute_plan(result, dt, [&](std::size_t pos, auto const& fn) {
///     fn(input[pos], output[pos]);
/// });
/// ```
template<typename Strategy, std::size_t N,
         typename Dispatch, typename Visitor>
    requires dispatchable<Dispatch, Strategy>
constexpr void
execute_plan(plan<per_element_candidate<Strategy, N>> const& p,
             Dispatch const& dt,
             Visitor visitor) {
    for (std::size_t i = 0; i < N; ++i) {
        auto const& impl = dt.dispatch(i, p.params[i]);
        visitor(i, impl);
    }
}

// =========================================================================
// execute_plan: plan + descriptors + dispatch → visitor(pos, desc, impl)
// =========================================================================

/// Execute a plan with descriptors.
///
/// Uses candidate_traits to iterate (descriptor, strategy) pairs,
/// then resolves each strategy via the dispatch table.
///
/// Calls visitor(position, descriptor, impl) for each assignment.
///
/// This is the most generic overload — works with any candidate type
/// that has candidate_traits.
///
/// Example:
/// ```cpp
/// execute_plan(result, field_descriptors, dt,
///     [](std::size_t pos, auto const& desc, auto const& fn) {
///         fn(desc.data, desc.length);
///     });
/// ```
template<typename Candidate, typename Descriptors,
         typename Dispatch, typename Visitor>
    requires has_candidate_traits<Candidate>
constexpr void
execute_plan(plan<Candidate> const& p,
             Descriptors const& descriptors,
             Dispatch const& dt,
             Visitor visitor) {
    std::size_t pos = 0;
    candidate_traits<Candidate>::for_each_assignment(
        p.params, descriptors,
        [&](auto const& desc, auto const& strategy) {
            auto const& impl = dt.dispatch(pos, strategy);
            visitor(pos, desc, impl);
            ++pos;
        }
    );
}

// =========================================================================
// execute_plan_with_context: in-place mutation pattern
// =========================================================================

/// Execute a plan with a mutable context object.
///
/// Calls action(position, impl, context) for each position.
/// The context is passed by reference for accumulation patterns.
///
/// Example:
/// ```cpp
/// struct RunContext { double total_time = 0.0; };
/// RunContext ctx;
///
/// execute_plan_with_context(result, dt, ctx,
///     [](std::size_t pos, auto const& fn, RunContext& c) {
///         c.total_time += fn(pos);
///     });
/// ```
template<typename Strategy, std::size_t N,
         typename Dispatch, typename Context, typename Action>
    requires dispatchable<Dispatch, Strategy>
constexpr void
execute_plan_with_context(plan<per_element_candidate<Strategy, N>> const& p,
                          Dispatch const& dt,
                          Context& ctx,
                          Action action) {
    for (std::size_t i = 0; i < N; ++i) {
        auto const& impl = dt.dispatch(i, p.params[i]);
        action(i, impl, ctx);
    }
}

// =========================================================================
// collect_implementations: extract impl values from plan
// =========================================================================

/// Extract the implementation for each position into an array.
///
/// Returns std::array<Impl, N> where result[i] = dispatch(i, plan[i]).
///
/// Useful for pre-computing a flat dispatch table from a solved plan,
/// avoiding repeated lookups during hot loops.
///
/// Example:
/// ```cpp
/// auto impls = collect_implementations(result, dt);
/// // impls[i] is the function pointer for position i
/// for (std::size_t i = 0; i < N; ++i) impls[i](data[i]);
/// ```
template<typename Impl, typename Strategy, std::size_t N,
         typename Dispatch>
    requires dispatchable<Dispatch, Strategy>
[[nodiscard]] constexpr std::array<Impl, N>
collect_implementations(plan<per_element_candidate<Strategy, N>> const& p,
                        Dispatch const& dt) {
    std::array<Impl, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = dt.dispatch(i, p.params[i]);
    }
    return result;
}

// =========================================================================
// Execution statistics
// =========================================================================

/// Statistics from plan execution.
struct execution_stats {
    std::size_t positions_executed = 0;
    std::size_t positions_skipped = 0;   // if visitor returns false
};

/// Execute a plan with early-exit support.
///
/// Calls visitor(position, impl) which returns bool.
/// Stops on first false return (useful for error handling).
///
/// Returns execution_stats showing how many positions were executed.
template<typename Strategy, std::size_t N,
         typename Dispatch, typename Visitor>
    requires dispatchable<Dispatch, Strategy>
[[nodiscard]] constexpr execution_stats
execute_plan_checked(plan<per_element_candidate<Strategy, N>> const& p,
                     Dispatch const& dt,
                     Visitor visitor) {
    execution_stats stats;
    for (std::size_t i = 0; i < N; ++i) {
        auto const& impl = dt.dispatch(i, p.params[i]);
        if (visitor(i, impl)) {
            stats.positions_executed++;
        } else {
            stats.positions_skipped = N - i;
            return stats;
        }
    }
    return stats;
}

} // namespace ctdp

#endif // CTDP_ENGINE_PLAN_EXECUTOR_H
