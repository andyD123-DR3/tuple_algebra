// core/plan_set.h - Bounded collection of non-dominated plans
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// plan_set<C, MaxN> maintains a bounded Pareto frontier of plans.
//
// Single-objective mode (default):
//   Sorted by predicted_cost ascending. Insert keeps top-K by cost.
//   Dominated = strictly worse cost. Pareto frontier = all plans with
//   distinct costs (ties kept — they may differ in candidate).
//
// Multi-objective mode:
//   Caller provides a dominance predicate: dominates(a, b) → bool.
//   Insert removes all plans dominated by the new plan, and rejects
//   the new plan if it's dominated by any existing plan.
//
// INVARIANTS:
// - Size never exceeds MaxN.
// - In single-objective mode, plans are sorted by cost (ascending).
// - No plan in the set dominates another (Pareto optimality).
// - Insertion of a plan that would exceed capacity evicts the worst plan
//   (highest cost in single-objective mode).
//
// USAGE:
// ```cpp
// plan_set<Candidate, 10> ps;
// ps.insert(plan<Candidate>{cand1, 3.0});
// ps.insert(plan<Candidate>{cand2, 1.0});
// ps.insert(plan<Candidate>{cand3, 2.0});
// // ps.best() → plan with cost 1.0
// // ps.size() == 3
// ```

#ifndef CTDP_CORE_PLAN_SET_H
#define CTDP_CORE_PLAN_SET_H

#include "plan.h"
#include "constexpr_vector.h"

#include <cstddef>
#include <limits>

namespace ctdp {

/// Default dominance: plan a dominates plan b if a has strictly lower cost.
struct cost_dominance {
    template<typename Candidate>
    [[nodiscard]] constexpr bool operator()(
        plan<Candidate> const& a,
        plan<Candidate> const& b) const noexcept {
        return a.predicted_cost < b.predicted_cost;
    }
};

/// Bounded collection of non-dominated plans (Pareto frontier).
///
/// Template parameters:
/// - Candidate: Plan candidate type
/// - MaxN: Maximum number of plans in the set
/// - Dominance: Predicate where dominates(a, b) means a is strictly
///   better than b. Default is cost_dominance (lower cost wins).
///
/// Complexity:
/// - insert: O(n) scan + O(n) compaction
/// - best: O(1) (first element in sorted order)
/// - contains: O(n) linear scan
///
/// Example — single-objective:
/// ```cpp
/// plan_set<Candidate, 10> ps;
/// ps.insert(plan<Candidate>{c1, 5.0});
/// ps.insert(plan<Candidate>{c2, 2.0});
/// ps.insert(plan<Candidate>{c3, 8.0});
/// assert(ps.best().predicted_cost == 2.0);
/// ```
///
/// Example — multi-objective with custom dominance:
/// ```cpp
/// struct pareto_dom {
///     bool operator()(plan<C> const& a, plan<C> const& b) const {
///         return a.cost1 <= b.cost1 && a.cost2 <= b.cost2
///             && (a.cost1 < b.cost1 || a.cost2 < b.cost2);
///     }
/// };
/// plan_set<C, 50, pareto_dom> frontier;
/// ```
template<typename Candidate, std::size_t MaxN,
         typename Dominance = cost_dominance>
class plan_set {
public:
    using plan_type = plan<Candidate>;
    using size_type = std::size_t;

    constexpr plan_set() = default;

    // =========================================================================
    // Capacity
    // =========================================================================

    [[nodiscard]] constexpr bool empty() const noexcept {
        return plans_.empty();
    }

    [[nodiscard]] constexpr size_type size() const noexcept {
        return plans_.size();
    }

    [[nodiscard]] constexpr size_type capacity() const noexcept {
        return MaxN;
    }

    // =========================================================================
    // Access
    // =========================================================================

    /// Best plan (lowest cost in single-objective mode).
    /// Precondition: !empty().
    [[nodiscard]] constexpr plan_type const& best() const {
        return plans_[0];
    }

    /// Worst plan in the set (highest cost in single-objective mode).
    /// Precondition: !empty().
    [[nodiscard]] constexpr plan_type const& worst() const {
        return plans_[plans_.size() - 1];
    }

    /// Access plan at index (sorted order).
    [[nodiscard]] constexpr plan_type const& operator[](size_type i) const {
        return plans_[i];
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    [[nodiscard]] constexpr auto begin() const noexcept { return plans_.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return plans_.end(); }

    // =========================================================================
    // Modifiers
    // =========================================================================

    /// Insert a plan into the set.
    ///
    /// Returns true if the plan was inserted (not dominated).
    /// Returns false if the plan was rejected (dominated by existing plan).
    ///
    /// Behaviour:
    /// 1. If new plan is dominated by any existing plan → reject (return false).
    /// 2. Remove all existing plans dominated by the new plan.
    /// 3. Insert the new plan in sorted position (by cost).
    /// 4. If size exceeds MaxN after insert → evict worst plan.
    ///
    /// Infeasible plans (infinite cost) are never inserted.
    [[nodiscard]] constexpr bool insert(plan_type const& p) {
        if (!p.is_feasible()) return false;

        Dominance dom{};

        // Check if new plan is dominated by any existing plan
        for (size_type i = 0; i < plans_.size(); ++i) {
            if (dom(plans_[i], p)) {
                return false;  // Existing plan dominates new one
            }
        }

        // Remove all existing plans dominated by new plan
        size_type write = 0;
        for (size_type i = 0; i < plans_.size(); ++i) {
            if (!dom(p, plans_[i])) {
                if (write != i) {
                    plans_[write] = plans_[i];
                }
                ++write;
            }
        }
        plans_.resize(write);

        // Find sorted insertion position (by cost, ascending)
        size_type pos = 0;
        while (pos < plans_.size() &&
               plans_[pos].predicted_cost <= p.predicted_cost) {
            ++pos;
        }

        // If at capacity, check if new plan is worse than all existing
        if (plans_.size() >= MaxN) {
            if (pos >= MaxN) {
                return false;  // Would be evicted immediately
            }
            // Make room by removing worst
            plans_.pop_back();
        }

        // Insert at sorted position
        plans_.insert(plans_.begin() + pos, p);
        return true;
    }

    /// Insert with move semantics.
    [[nodiscard]] constexpr bool insert(plan_type&& p) {
        if (!p.is_feasible()) return false;

        Dominance dom{};

        for (size_type i = 0; i < plans_.size(); ++i) {
            if (dom(plans_[i], p)) {
                return false;
            }
        }

        size_type write = 0;
        for (size_type i = 0; i < plans_.size(); ++i) {
            if (!dom(p, plans_[i])) {
                if (write != i) {
                    plans_[write] = std::move(plans_[i]);
                }
                ++write;
            }
        }
        plans_.resize(write);

        size_type pos = 0;
        while (pos < plans_.size() &&
               plans_[pos].predicted_cost <= p.predicted_cost) {
            ++pos;
        }

        if (plans_.size() >= MaxN) {
            if (pos >= MaxN) {
                return false;
            }
            plans_.pop_back();
        }

        plans_.insert(plans_.begin() + pos, std::move(p));
        return true;
    }

    /// Clear all plans.
    constexpr void clear() noexcept {
        plans_.clear();
    }

    // =========================================================================
    // Queries
    // =========================================================================

    /// Check if any feasible plan exists in the set.
    [[nodiscard]] constexpr bool has_feasible() const noexcept {
        return !empty();  // Infeasible plans are never inserted
    }

    /// Merge another plan_set into this one.
    /// Each plan from other is individually inserted (dominance-checked).
    /// Returns number of plans actually inserted.
    constexpr size_type merge(plan_set const& other) {
        size_type inserted = 0;
        for (size_type i = 0; i < other.size(); ++i) {
            if (insert(other[i])) {
                ++inserted;
            }
        }
        return inserted;
    }

    // =========================================================================
    // Comparison
    // =========================================================================

    constexpr bool operator==(plan_set const& other) const
        requires std::equality_comparable<Candidate>
    {
        if (plans_.size() != other.plans_.size()) return false;
        for (size_type i = 0; i < plans_.size(); ++i) {
            if (!(plans_[i] == other.plans_[i])) return false;
        }
        return true;
    }

private:
    constexpr_vector<plan_type, MaxN> plans_{};
};

} // namespace ctdp

#endif // CTDP_CORE_PLAN_SET_H
