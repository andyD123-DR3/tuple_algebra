#ifndef CT_DP_SOLVER_INTERVAL_SOLVER_HPP
#define CT_DP_SOLVER_INTERVAL_SOLVER_HPP

#include "ct_dp/solver/interval_context.hpp"
#include "ct_dp/plan/interval_partition_plan.hpp"
#include "ct_dp/solver/all_binary_splits.hpp"
#include <optional>
#include <functional>
#include <cassert>
#include <concepts>
#include <utility>
#include <cstddef>

namespace ct_dp {
namespace solver {

/// Interval Recurrence Concept
///
/// A type R satisfies IntervalRecurrence if it provides:
/// 1. value_type - the result type (e.g., int, double, custom)
/// 2. base_case(ctx) - returns value if base, nullopt if should recurse
/// 3. combine(plan, left, right) - combines child results
///
/// IMPORTANT: combine must ALWAYS succeed for emitted splits.
/// If a problem has infeasible combinations, either:
/// - Encode in value_type (e.g., optional<T>)
/// - Filter in split policy
/// - Use future partial-recurrence extension (Sprint 11+)
template<class R>
concept IntervalRecurrence =
requires(const R& r,
         interval_context ctx,
         const interval_partition_plan& plan,
         const typename R::value_type& left,
         const typename R::value_type& right) {
    typename R::value_type;
    
    { r.base_case(ctx) }
        -> std::same_as<std::optional<typename R::value_type>>;
    
    { r.combine(plan, left, right) }
        -> std::same_as<typename R::value_type>;  // Never fails
};

/// Memo Concept
///
/// A type M satisfies Memo<M, Value> if it provides:
/// 1. lookup(ctx) const -> optional<Value>  (const!)
/// 2. store(ctx, value) -> void
///
/// Memo implementations may be:
/// - Dense (triangular_memo) for small bounded problems
/// - Sparse (map_memo) for large/unbounded problems
/// - Custom (user-defined)
template<class M, class Value>
concept Memo =
requires(M& m, const M& cm, interval_context ctx, Value v) {
    { cm.lookup(ctx) } -> std::same_as<std::optional<Value>>;
    { m.store(ctx, v) } -> std::same_as<void>;
};

/// Optimal-cost solver for binary interval partition recurrences
///
/// This solver computes the optimal value for non-empty binary interval
/// partition recurrences. It assumes all emitted splits combine successfully.
///
/// Template parameters:
/// - Recurrence: Problem-specific cost function
/// - SplitPolicy: Enumerates legal splits (default: all k in (i,j))
/// - Compare: Optimization direction (default: std::less<> = minimize)
///
/// Scope:
/// - Non-empty intervals (ctx.i < ctx.j)
/// - Binary partitions (one split point)
/// - Total recurrences (combine always succeeds)
/// - Optimal cost only (no reconstruction in Sprint 9)
///
/// Example (minimize):
///   interval_solver<matrix_chain_recurrence> solver{rec};
///   triangular_memo<uint64_t> memo{N};
///   auto cost = solver.solve(interval_context{0, N}, memo);
///
/// Example (maximize):
///   interval_solver<profit_rec, all_binary_splits, std::greater<>> solver{rec};
///   auto profit = solver.solve(ctx, memo);
///
/// Sprint 9 Limitation: Total Recurrences Only
///
/// This solver assumes all emitted splits combine successfully.
/// If a problem needs branch-level infeasibility after child evaluation:
/// - Option A: Encode in value_type (e.g., optional<Cost>)
/// - Option B: Pre-filter in split policy
/// - Option C: Extend in future sprint (partial recurrences, Sprint 11+)
template<class Recurrence,
         class SplitPolicy = all_binary_splits,
         class Compare = std::less<>>
requires IntervalRecurrence<Recurrence>
class interval_solver {
public:
    using value_type = typename Recurrence::value_type;
    
    /// Construct solver with recurrence, split policy, and comparison
    ///
    /// @param recurrence Problem-specific cost function
    /// @param split_policy Enumerates legal splits (default: all binary)
    /// @param compare Optimization direction (default: minimize)
    explicit interval_solver(Recurrence recurrence,
                             SplitPolicy split_policy = {},
                             Compare compare = {})
        : recurrence_(std::move(recurrence)),
          split_policy_(std::move(split_policy)),
          compare_(std::move(compare)) {}
    
    /// Solve optimal cost for interval [ctx.i, ctx.j)
    ///
    /// Precondition: ctx.i < ctx.j (non-empty interval)
    /// Precondition: memo can store all subproblems in [ctx.i, ctx.j)
    /// Postcondition: Returns optimal value over all emitted splits
    ///
    /// This method is const: solver does not mutate during solving.
    /// Memo is passed by non-const reference as it caches results.
    ///
    /// Algorithm:
    /// 1. Check memo for cached result
    /// 2. Check if base case via recurrence.base_case(ctx)
    /// 3. Enumerate splits via split_policy.for_each(ctx, ...)
    /// 4. For each split k:
    ///    - Build plan via interval_partition_plan::from_split(ctx, k)
    ///    - Recursively solve left and right subproblems
    ///    - Combine via recurrence.combine(plan, left, right)
    ///    - Update best if better (according to compare)
    /// 5. Store best in memo and return
    ///
    /// Complexity: O(n³) time for n = ctx.size(), assuming O(1) memo and combine
    template<class M>
    requires Memo<M, value_type>
    value_type solve(interval_context ctx, M& memo) const {
        assert(ctx.i < ctx.j && "Empty interval not allowed");
        
        // 1. Check memo
        if (auto cached = memo.lookup(ctx)) {
            return *cached;
        }
        
        // 2. Base case
        if (auto base = recurrence_.base_case(ctx)) {
            memo.store(ctx, *base);
            return *base;
        }
        
        // 3. Enumerate and solve recursively
        std::optional<value_type> best;
        
        split_policy_.for_each(ctx, [&](size_t k) {
            // Build plan from runtime split
            auto plan = interval_partition_plan::from_split(ctx, k);
            
            // Recurse on children
            value_type left = solve(plan.left_ctx, memo);
            value_type right = solve(plan.right_ctx, memo);
            
            // Combine (always succeeds - policy filters infeasible)
            value_type candidate = recurrence_.combine(plan, left, right);
            
            // Update best (first valid or better)
            if (!best || compare_(candidate, *best)) {
                best = std::move(candidate);
            }
        });
        
        // 4. Store and return
        assert(best.has_value() && "No valid splits found - check split policy");
        memo.store(ctx, *best);
        return *best;
    }
    
    /// Access underlying recurrence (for inspection/testing)
    const Recurrence& recurrence() const { return recurrence_; }
    
    /// Access split policy (for inspection/testing)
    const SplitPolicy& split_policy() const { return split_policy_; }
    
    /// Access comparison function (for inspection/testing)
    const Compare& compare() const { return compare_; }
    
private:
    Recurrence recurrence_;
    SplitPolicy split_policy_;
    Compare compare_;
};

} // namespace solver
} // namespace ct_dp

#endif // CT_DP_SOLVER_INTERVAL_SOLVER_HPP
