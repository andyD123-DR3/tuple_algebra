// ctdp/solver/cost_models/additive.h
// Compile-time dynamic programming framework — Analytics: Solver
// Additive cost: sum of per-element costs.
// Natural partner for per_element_space.  Satisfies cost_function_for.
//
// Descriptor capture pattern: the domain constructs additive_cost by closing
// the per-element cost over descriptors at construction time.

#ifndef CTDP_SOLVER_COST_MODELS_ADDITIVE_H
#define CTDP_SOLVER_COST_MODELS_ADDITIVE_H

#include <concepts>
#include <cstddef>

namespace ctdp {

// ---------------------------------------------------------------------------
// additive_cost: wraps a per-element cost function, sums over positions.
// element_cost signature: (position_index, strategy) → double
//
// Requires Candidate to be sized and indexable (e.g. std::array).
// Not compatible with tuple-based candidates — use weighted_cost for those.
// ---------------------------------------------------------------------------
template<typename PerElementCost>
struct additive_cost {
    PerElementCost element_cost;

    template<typename Candidate>
        requires requires(Candidate const& c, std::size_t i) {
            { c.size() } -> std::convertible_to<std::size_t>;
            c[i];
        }
    [[nodiscard]] constexpr auto operator()(Candidate const& c) const -> double {
        double total = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i)
            total += element_cost(i, c[i]);
        return total;
    }
};

// Deduction guide
template<typename PerElementCost>
additive_cost(PerElementCost) -> additive_cost<PerElementCost>;

} // namespace ctdp

#endif // CTDP_SOLVER_COST_MODELS_ADDITIVE_H
