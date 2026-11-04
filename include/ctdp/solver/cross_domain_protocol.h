#pragma once
// CT-DP cross-domain exploration protocol (experimental, C++20, header-only).
//
// A domain owns its problem instance, search-space representation and
// numerical/physical contract. An optimizer owns its constraints, objective,
// solver family and search budget. CT-DP does not flatten either into a
// Cartesian candidate vector or substitute a universal solver.
//
// The result retains the native solver report unmodified: constexpr plans,
// interval split tables, Pareto frontiers, truncation and search statistics.
// This is EXPLORATION, never numerical qualification or physical admission.
// See docs/design/ctdp_cross_domain_consolidation_requirements_20261104.md.
#include <concepts>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ctdp::solver::integration {

// Explicit model-search status. No status here asserts that generated
// numerical kernels have been executed or independently qualified.
enum class model_status {
    unverified,
    complete_for_declared_space,
    bounded_approximate
};

struct search_description {
    std::string_view family{};
    model_status status{model_status::unverified};
};

// Unlike a general-purpose IR, these are references to domain-owned objects;
// the space can be a descriptor, partition tree, interval space, recursive
// root key, or any other type. There are no allocation or type-erasure costs.
template<class Instance, class Space, class Contract>
struct problem_view {
    Instance const& instance;
    Space const& space;
    Contract const& contract;
};

template<class Instance, class Space, class Contract>
[[nodiscard]] constexpr auto problem(Instance const& instance,
                                     Space const& space,
                                     Contract const& contract) noexcept {
    return problem_view<Instance,Space,Contract>{instance,space,contract};
}

template<class Native>
struct exploration {
    Native native;
    search_description search;
    // There is deliberately NO admitted(), execute() or qualified winner:
    // those are domain-specific responsibilities after independent assessment.
};

// Solver is a typed strategy accepting the problem view. It may call any
// existing CT-DP algorithm, including a specialised interval or Pareto solver.
// It must report its own completeness explicitly; unknown is the default.
// Unlike a post-hoc proposal selector, this boundary wraps the actual solve.
template<class Problem, class Solver>
    requires std::invocable<Solver, Problem const&> &&
        (!std::is_void_v<std::invoke_result_t<Solver,Problem const&>>)
[[nodiscard]] constexpr auto explore(Problem const& p, Solver&& solver,
                                     search_description description) {
    if(description.family.empty())
        throw std::invalid_argument("CT-DP integration: missing solver identity");
    using native_type=std::remove_cvref_t<
        std::invoke_result_t<Solver,Problem const&>>;
    native_type native=std::invoke(std::forward<Solver>(solver),p);
    if constexpr (requires { native.approximate; }) {
        if(native.approximate &&
           description.status==model_status::complete_for_declared_space)
            throw std::invalid_argument(
                "CT-DP integration: bounded frontier cannot claim complete search");
    }
    return exploration<native_type>{std::move(native),description};
}

// A small adapter for existing space-aware solver functions. It preserves
// the native type and gives the solver the original space and domain instance,
// not an enumerated or erased surrogate. Constraints and costs remain owned
// by the supplied strategy.
template<class Optimizer>
struct space_strategy {
    Optimizer optimizer;
    template<class Problem>
        requires std::invocable<Optimizer const&,
                                decltype(std::declval<Problem const&>().space),
                                decltype(std::declval<Problem const&>().instance),
                                decltype(std::declval<Problem const&>().contract)>
    [[nodiscard]] constexpr decltype(auto)
    operator()(Problem const& p) const {
        return std::invoke(optimizer,p.space,p.instance,p.contract);
    }
};

template<class Optimizer>
[[nodiscard]] constexpr auto adapt_space_solver(Optimizer optimizer) {
    return space_strategy<Optimizer>{std::move(optimizer)};
}

} // namespace ctdp::solver::integration
