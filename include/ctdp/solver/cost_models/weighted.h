// ctdp/solver/cost_models/weighted.h
// Compile-time dynamic programming framework — Analytics: Solver
// Weighted cost: linear combination of N cost functions.
// Satisfies cost_function_for.

#ifndef CTDP_SOLVER_COST_MODELS_WEIGHTED_H
#define CTDP_SOLVER_COST_MODELS_WEIGHTED_H

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace ctdp {

template<typename... Costs>
struct weighted_cost {
    std::tuple<Costs...> costs;
    std::array<double, sizeof...(Costs)> weights;

    template<typename Candidate>
    [[nodiscard]] constexpr auto operator()(Candidate const& c) const -> double {
        double total = 0.0;
        apply_costs(c, total, std::index_sequence_for<Costs...>{});
        return total;
    }

private:
    template<typename Candidate, std::size_t... Is>
    constexpr void apply_costs(Candidate const& c, double& total,
                                std::index_sequence<Is...>) const {
        ((total += weights[Is] * std::get<Is>(costs)(c)), ...);
    }
};

// Factory
template<typename... Costs>
[[nodiscard]] constexpr auto make_weighted_cost(
    std::array<double, sizeof...(Costs)> const& w,
    Costs&&... costs)
{
    return weighted_cost<std::decay_t<Costs>...>{
        std::tuple{std::forward<Costs>(costs)...}, w};
}

} // namespace ctdp

#endif // CTDP_SOLVER_COST_MODELS_WEIGHTED_H
