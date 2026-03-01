#ifndef CTDP_CALIBRATOR_SOLVER_H
#define CTDP_CALIBRATOR_SOLVER_H

// ctdp::calibrator::solver — Optimisation algorithms
//
// Design v2.2 §8.6:
//   Wire cost_model as the cost_function for exhaustive_search /
//   beam_search → produce plan<Space, Callable>.
//
// Algorithms:
//
//   exhaustive_search    Evaluates every point in the scenario's space.
//                        Optimal for small spaces (< ~10k points).
//
//   filtered_search      Like exhaustive but skips infeasible points.
//                        Accepts a ConstrainedCostModel or explicit pred.
//
//   pareto_search        Multi-objective: evaluates all points, marks
//                        Pareto-optimal entries in the plan.
//
//   beam_search          Heuristic: explores the most promising beam_width
//                        candidates per stage.  For large spaces where
//                        exhaustive is too expensive.
//
// All solvers share the same contract:
//   Input:  a CostModel + a Scenario (or vector<point_type>)
//   Output: plan<Space, Callable>

#include "cost_model.h"
#include "plan.h"
#include "scenario.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

// ─── exhaustive_search ───────────────────────────────────────────

/// Evaluate every point in the space, return a plan sorted by cost.
///
/// @tparam Space     Search space type
/// @tparam Callable  Kernel identity
/// @tparam Model     CostModel<point_type>
///
/// @param model     Cost function (typically profile_cost_model)
/// @param points    All points to evaluate
/// @param prov      Provenance from the source profile
///
template <typename Space, typename Callable, typename Model>
    requires CostModel<Model, typename Space::point_type>
[[nodiscard]] auto exhaustive_search(
    Model const& model,
    std::vector<typename Space::point_type> const& points,
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;
    plan<Space, Callable> result;
    result.provenance   = std::move(prov);
    result.solver_name  = "exhaustive_search";

    result.entries.reserve(points.size());
    for (auto const& pt : points) {
        double c = model.cost(pt);
        if (c >= 0.0) {  // skip unknown points (cost == -1)
            result.entries.push_back(plan_entry<point_type>{
                .point          = pt,
                .cost_ns        = c,
                .objectives     = {c},
                .pareto_optimal = false
            });
        }
    }

    result.evaluated_points = result.entries.size();
    result.sort_by_cost();

    // In single-objective, the cheapest is Pareto-optimal
    if (!result.entries.empty()) {
        result.entries.front().pareto_optimal = true;
    }

    return result;
}

/// Convenience overload: extract points from a Scenario.
template <typename Space, typename Callable, typename Model, Scenario S>
    requires CostModel<Model, typename Space::point_type>
          && std::same_as<typename S::callable_type, Callable>
[[nodiscard]] auto exhaustive_search(
    Model const& model,
    S const& scenario,
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    auto const& pts = scenario.points();
    std::vector<typename Space::point_type> vec(pts.begin(), pts.end());
    return exhaustive_search<Space, Callable>(model, vec, std::move(prov));
}

// ─── filtered_search ─────────────────────────────────────────────

/// Like exhaustive_search, but skips infeasible points.
///
/// If the model is a ConstrainedCostModel, uses model.feasible().
/// Otherwise, an explicit predicate can be provided.
///
template <typename Space, typename Callable, typename Model, typename Pred>
    requires CostModel<Model, typename Space::point_type>
          && std::invocable<Pred, typename Space::point_type const&>
[[nodiscard]] auto filtered_search(
    Model const& model,
    std::vector<typename Space::point_type> const& points,
    Pred&& predicate,
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;
    plan<Space, Callable> result;
    result.provenance   = std::move(prov);
    result.solver_name  = "filtered_search";

    result.entries.reserve(points.size());
    std::size_t evaluated = 0;
    for (auto const& pt : points) {
        if (!predicate(pt)) continue;
        ++evaluated;
        double c = model.cost(pt);
        if (c >= 0.0) {
            result.entries.push_back(plan_entry<point_type>{
                .point          = pt,
                .cost_ns        = c,
                .objectives     = {c},
                .pareto_optimal = false
            });
        }
    }

    result.evaluated_points = evaluated;
    result.sort_by_cost();

    if (!result.entries.empty()) {
        result.entries.front().pareto_optimal = true;
    }

    return result;
}

/// Overload using ConstrainedCostModel's own feasibility check.
template <typename Space, typename Callable, typename Model>
    requires ConstrainedCostModel<Model, typename Space::point_type>
[[nodiscard]] auto filtered_search(
    Model const& model,
    std::vector<typename Space::point_type> const& points,
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    return filtered_search<Space, Callable>(
        model, points,
        [&model](auto const& pt) { return model.feasible(pt); },
        std::move(prov));
}

// ─── pareto_search (multi-objective) ─────────────────────────────

/// Multi-objective search: evaluates all points across multiple
/// objectives and identifies the Pareto frontier.
///
/// A point is Pareto-optimal (non-dominated) if no other point is
/// better in all objectives simultaneously.
///
/// @param model     composite_cost_model with N objectives
/// @param points    All points to evaluate
///
template <typename Space, typename Callable>
[[nodiscard]] auto pareto_search(
    composite_cost_model<typename Space::point_type> const& model,
    std::vector<typename Space::point_type> const& points,
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;
    auto const dim = model.dimension();
    assert(dim > 0);

    plan<Space, Callable> result;
    result.provenance   = std::move(prov);
    result.solver_name  = "pareto_search";

    // Evaluate all points
    result.entries.reserve(points.size());
    for (auto const& pt : points) {
        auto costs = model.evaluate(pt);
        result.entries.push_back(plan_entry<point_type>{
            .point          = pt,
            .cost_ns        = costs.empty() ? 0.0 : costs[0],
            .objectives     = std::move(costs),
            .pareto_optimal = false
        });
    }
    result.evaluated_points = result.entries.size();

    // Identify Pareto frontier: O(N² × D) — fine for typical spaces
    auto const N = result.entries.size();
    for (std::size_t i = 0; i < N; ++i) {
        bool dominated = false;
        for (std::size_t j = 0; j < N && !dominated; ++j) {
            if (i == j) continue;
            // Check if j dominates i:
            // j must be <= i in all objectives and < in at least one
            bool all_leq = true;
            bool any_lt  = false;
            for (std::size_t d = 0; d < dim; ++d) {
                double oi = result.entries[i].objectives[d];
                double oj = result.entries[j].objectives[d];
                if (oj > oi) { all_leq = false; break; }
                if (oj < oi) any_lt = true;
            }
            if (all_leq && any_lt) dominated = true;
        }
        result.entries[i].pareto_optimal = !dominated;
    }

    // Sort by primary objective (cost_ns)
    result.sort_by_cost();

    return result;
}

// ─── beam_search (heuristic for large spaces) ────────────────────

/// Configuration for beam search.
struct beam_config {
    std::size_t beam_width = 10;    ///< Number of candidates per stage
    std::size_t max_stages = 100;   ///< Maximum expansion stages
};

/// Beam search: maintains a fixed-width beam of the best candidates.
///
/// For structured spaces where points can be enumerated in stages
/// (e.g., choosing one dimension at a time), beam search avoids
/// evaluating the full cross-product.
///
/// This simplified version works as a top-K selection from the full
/// point set — effectively a bounded exhaustive search that retains
/// only the best beam_width results.  For true staged beam search,
/// the user provides a stage_expander that generates candidates from
/// partial solutions.
///
template <typename Space, typename Callable, typename Model>
    requires CostModel<Model, typename Space::point_type>
[[nodiscard]] auto beam_search(
    Model const& model,
    std::vector<typename Space::point_type> const& points,
    beam_config const& config = {},
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;
    plan<Space, Callable> result;
    result.provenance   = std::move(prov);
    result.solver_name  = "beam_search";

    // Evaluate all points (beam search's value is in staged expansion;
    // this flat version simply limits output size)
    struct candidate {
        point_type point;
        double     cost;
    };
    std::vector<candidate> beam;
    beam.reserve(points.size());

    for (auto const& pt : points) {
        double c = model.cost(pt);
        if (c < 0.0) continue;
        beam.push_back({pt, c});
    }

    result.evaluated_points = beam.size();

    // Keep top beam_width
    auto effective_width = std::min(config.beam_width, beam.size());
    std::partial_sort(beam.begin(),
        beam.begin() + static_cast<std::ptrdiff_t>(effective_width),
        beam.end(),
        [](candidate const& a, candidate const& b) {
            return a.cost < b.cost;
        });
    beam.resize(effective_width);

    result.entries.reserve(effective_width);
    for (auto const& cand : beam) {
        result.entries.push_back(plan_entry<point_type>{
            .point          = cand.point,
            .cost_ns        = cand.cost,
            .objectives     = {cand.cost},
            .pareto_optimal = false
        });
    }

    if (!result.entries.empty()) {
        result.entries.front().pareto_optimal = true;
    }

    return result;
}

// ─── Staged beam search ──────────────────────────────────────────

/// A StageExpander generates candidate points from partial solutions.
///
/// Required:
///   expander.expand(beam) → vector<point_type>
///   expander.complete(pt) → bool  (is this a complete solution?)
///
template <typename E, typename PointType>
concept StageExpander = requires(E& e,
    std::vector<PointType> const& beam, PointType const& pt)
{
    { e.expand(beam) } -> std::convertible_to<std::vector<PointType>>;
    { e.complete(pt) } -> std::convertible_to<bool>;
};

/// Staged beam search with a user-provided StageExpander.
///
/// The expander generates new candidates from the current beam at
/// each stage.  The search terminates when all candidates are complete
/// or max_stages is reached.
///
template <typename Space, typename Callable, typename Model, typename Expander>
    requires CostModel<Model, typename Space::point_type>
          && StageExpander<Expander, typename Space::point_type>
[[nodiscard]] auto staged_beam_search(
    Model const& model,
    Expander& expander,
    std::vector<typename Space::point_type> const& initial_beam,
    beam_config const& config = {},
    dataset_provenance prov = {})
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;

    struct scored {
        point_type pt;
        double cost;
    };

    // Initialise beam with scored initial points
    std::vector<point_type> current_beam = initial_beam;
    std::vector<scored> complete_solutions;
    std::size_t total_evaluated = 0;

    for (std::size_t stage = 0; stage < config.max_stages; ++stage) {
        if (current_beam.empty()) break;

        // Expand
        auto candidates = expander.expand(current_beam);
        if (candidates.empty()) break;

        // Score
        std::vector<scored> scored_cands;
        scored_cands.reserve(candidates.size());
        for (auto const& pt : candidates) {
            double c = model.cost(pt);
            ++total_evaluated;
            if (c >= 0.0) {
                scored_cands.push_back({pt, c});
            }
        }

        // Separate complete from partial
        std::vector<scored> partial;
        for (auto& sc : scored_cands) {
            if (expander.complete(sc.pt)) {
                complete_solutions.push_back(std::move(sc));
            } else {
                partial.push_back(std::move(sc));
            }
        }

        // Prune beam
        auto width = std::min(config.beam_width, partial.size());
        std::partial_sort(partial.begin(),
            partial.begin() + static_cast<std::ptrdiff_t>(width),
            partial.end(),
            [](scored const& a, scored const& b) {
                return a.cost < b.cost;
            });
        partial.resize(width);

        current_beam.clear();
        current_beam.reserve(width);
        for (auto const& sc : partial) {
            current_beam.push_back(sc.pt);
        }
    }

    // Build plan from complete solutions
    plan<Space, Callable> result;
    result.provenance       = std::move(prov);
    result.solver_name      = "staged_beam_search";
    result.evaluated_points = total_evaluated;

    result.entries.reserve(complete_solutions.size());
    for (auto const& sc : complete_solutions) {
        result.entries.push_back(plan_entry<point_type>{
            .point          = sc.pt,
            .cost_ns        = sc.cost,
            .objectives     = {sc.cost},
            .pareto_optimal = false
        });
    }

    result.sort_by_cost();
    if (!result.entries.empty()) {
        result.entries.front().pareto_optimal = true;
    }

    return result;
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_SOLVER_H
