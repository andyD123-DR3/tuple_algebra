#ifndef CTDP_CALIBRATOR_PLAN_BUILDER_H
#define CTDP_CALIBRATOR_PLAN_BUILDER_H

// ctdp::calibrator::plan_builder — One-call pipeline: profile → plan
//
// Convenience functions that compose the cost model + solver + plan
// construction in a single call.  These are the primary user-facing
// entry points for the solver integration.
//
// Usage:
//   // From a lookup profile + scenario:
//   auto plan = build_plan(profile, scenario);
//
//   // With constraints:
//   auto plan = build_plan(profile, scenario,
//       [](auto const& pt) { return pt.digits <= 8; });
//
//   // Multi-objective:
//   auto plan = build_pareto_plan(scenario.points(),
//       latency_model, memory_model, "latency", "memory");
//
//   // Beam search for large spaces:
//   auto plan = build_beam_plan(profile, points, beam_config{.beam_width = 20});

#include "calibration_profile.h"
#include "cost_model.h"
#include "plan.h"
#include "scenario.h"
#include "solver.h"

#include <string>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

// ─── build_plan: lookup profile + scenario ───────────────────────

/// Build a plan from a lookup profile and a scenario.
/// Uses exhaustive_search — evaluates every point.
///
template <typename Space, typename Callable, Scenario S>
    requires std::same_as<typename S::callable_type, Callable>
[[nodiscard]] auto build_plan(
    calibration_profile<Space, Callable> const& profile,
    S const& scenario)
    -> plan<Space, Callable>
{
    profile_cost_model<Space, Callable> model{profile};
    return exhaustive_search<Space, Callable>(
        model, scenario, profile.provenance);
}

/// Build a plan from a lookup profile and an explicit point set.
template <typename Space, typename Callable>
[[nodiscard]] auto build_plan(
    calibration_profile<Space, Callable> const& profile,
    std::vector<typename Space::point_type> const& points)
    -> plan<Space, Callable>
{
    profile_cost_model<Space, Callable> model{profile};
    return exhaustive_search<Space, Callable>(
        model, points, profile.provenance);
}

// ─── build_plan: with constraint ─────────────────────────────────

/// Build a plan from a lookup profile with a feasibility constraint.
template <typename Space, typename Callable, Scenario S, typename Pred>
    requires std::same_as<typename S::callable_type, Callable>
[[nodiscard]] auto build_plan(
    calibration_profile<Space, Callable> const& profile,
    S const& scenario,
    Pred&& constraint)
    -> plan<Space, Callable>
{
    profile_cost_model<Space, Callable> model{profile};
    auto const& pts = scenario.points();
    std::vector<typename Space::point_type> vec(pts.begin(), pts.end());
    return filtered_search<Space, Callable>(
        model, vec, std::forward<Pred>(constraint), profile.provenance);
}

/// Build a plan from a lookup profile + point set with a constraint.
template <typename Space, typename Callable, typename Pred>
[[nodiscard]] auto build_plan(
    calibration_profile<Space, Callable> const& profile,
    std::vector<typename Space::point_type> const& points,
    Pred&& constraint)
    -> plan<Space, Callable>
{
    profile_cost_model<Space, Callable> model{profile};
    return filtered_search<Space, Callable>(
        model, points, std::forward<Pred>(constraint), profile.provenance);
}

// ─── build_plan: linear profile + encoder ────────────────────────

/// Build a plan from a linear profile using a FeatureEncoder.
template <typename Space, typename Callable, typename Encoder>
    requires FeatureEncoder<Encoder, typename Space::point_type>
[[nodiscard]] auto build_linear_plan(
    calibration_profile<Space, Callable> const& profile,
    std::vector<typename Space::point_type> const& points,
    Encoder encoder)
    -> plan<Space, Callable>
{
    linear_cost_model<Space, Callable, Encoder> model{profile, std::move(encoder)};
    return exhaustive_search<Space, Callable>(
        model, points, profile.provenance);
}

// ─── build_pareto_plan: multi-objective ──────────────────────────

/// Build a multi-objective Pareto plan from a composite cost model.
template <typename Space, typename Callable>
[[nodiscard]] auto build_pareto_plan(
    composite_cost_model<typename Space::point_type> const& model,
    std::vector<typename Space::point_type> const& points,
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    return pareto_search<Space, Callable>(model, points, std::move(prov));
}

/// Convenience: build a 2-objective Pareto plan from two CostModels.
template <typename Space, typename Callable, typename M1, typename M2>
    requires CostModel<M1, typename Space::point_type>
          && CostModel<M2, typename Space::point_type>
[[nodiscard]] auto build_pareto_plan(
    std::vector<typename Space::point_type> const& points,
    M1 const& model1, M2 const& model2,
    std::string name1 = "objective_0",
    std::string name2 = "objective_1",
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;
    composite_cost_model<point_type> composite;
    composite.add_objective(std::move(name1), model1);
    composite.add_objective(std::move(name2), model2);
    return pareto_search<Space, Callable>(
        composite, points, std::move(prov));
}

// ─── build_beam_plan: beam search ────────────────────────────────

/// Build a plan using beam search (bounded top-K).
template <typename Space, typename Callable>
[[nodiscard]] auto build_beam_plan(
    calibration_profile<Space, Callable> const& profile,
    std::vector<typename Space::point_type> const& points,
    beam_config const& config = {})
    -> plan<Space, Callable>
{
    profile_cost_model<Space, Callable> model{profile};
    return beam_search<Space, Callable>(
        model, points, config, profile.provenance);
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_PLAN_BUILDER_H
