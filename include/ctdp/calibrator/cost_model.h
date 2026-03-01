#ifndef CTDP_CALIBRATOR_COST_MODEL_H
#define CTDP_CALIBRATOR_COST_MODEL_H

// ctdp::calibrator::cost_model — Cost function adapter
//
// Design v2.2 §5.6, §8.6:
//   Wraps a calibration_profile into a callable cost function that
//   the solver algorithms consume.
//
//   CostModel concept:
//     - model.cost(point) → double   (predicted wall time in ns)
//     - model.feasible(point) → bool (optional constraint check)
//
//   The profile_cost_model adapter wraps calibration_profile and
//   dispatches to lookup or linear prediction based on the profile's
//   active_model.
//
//   For multi-objective optimisation, CompositeCostModel evaluates
//   multiple objective functions and returns a vector of costs.

#include "calibration_profile.h"
#include "feature_encoder.h"

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <type_traits>
#include <vector>

namespace ctdp::calibrator {

// ─── CostModel concept ──────────────────────────────────────────

/// A CostModel maps a space point to a scalar cost.
///
/// Required:
///   model.cost(pt) → double
///
template <typename M, typename PointType>
concept CostModel = requires(M const& m, PointType const& pt) {
    { m.cost(pt) } -> std::convertible_to<double>;
};

/// Optional: a constrained CostModel also checks feasibility.
template <typename M, typename PointType>
concept ConstrainedCostModel = CostModel<M, PointType> &&
    requires(M const& m, PointType const& pt) {
        { m.feasible(pt) } -> std::convertible_to<bool>;
    };

// ─── profile_cost_model ─────────────────────────────────────────

/// Wraps a calibration_profile<Space, Callable> as a CostModel.
///
/// Dispatches to lookup or linear prediction based on the profile's
/// active_model.  For linear models, requires an encoder at
/// construction time.
///
template <typename Space, typename Callable, typename Encoder = void>
struct profile_cost_model {
    using point_type    = typename Space::point_type;
    using callable_type = Callable;
    using profile_type  = calibration_profile<Space, Callable>;

    profile_type const* profile_ = nullptr;

    explicit profile_cost_model(profile_type const& prof)
        : profile_{&prof} {}

    /// Predict cost from the lookup model.
    [[nodiscard]] double cost(point_type const& pt) const {
        assert(profile_ != nullptr);
        assert(profile_->active_model == profile_type::model_type::lookup);
        return profile_->predict_lookup(pt);
    }

    /// All points in a lookup profile are feasible.
    [[nodiscard]] bool feasible(point_type const&) const noexcept {
        return true;
    }
};

/// Specialisation for linear models: requires a FeatureEncoder.
template <typename Space, typename Callable, typename Encoder>
    requires (!std::is_void_v<Encoder>)
          && FeatureEncoder<Encoder, typename Space::point_type>
struct linear_cost_model {
    using point_type    = typename Space::point_type;
    using callable_type = Callable;
    using profile_type  = calibration_profile<Space, Callable>;

    profile_type const* profile_ = nullptr;
    Encoder             encoder_;

    linear_cost_model(profile_type const& prof, Encoder enc)
        : profile_{&prof}, encoder_{std::move(enc)} {}

    [[nodiscard]] double cost(point_type const& pt) const {
        assert(profile_ != nullptr);
        auto features = encoder_.encode(pt);
        std::span<const float> sp(features.data(), features.size());
        return profile_->predict_linear(sp);
    }

    [[nodiscard]] bool feasible(point_type const&) const noexcept {
        return true;
    }
};

// ─── constrained_cost_model ─────────────────────────────────────

/// Wraps any CostModel with an additional feasibility predicate.
///
/// Usage:
///   auto model = make_constrained(base_model,
///       [](auto const& pt) { return pt.size <= 1024; });
///
template <typename BaseCostModel, typename PointType, typename Pred>
struct constrained_cost_model {
    BaseCostModel base_;
    Pred          predicate_;

    constrained_cost_model(BaseCostModel base, Pred pred)
        : base_{std::move(base)}, predicate_{std::move(pred)} {}

    [[nodiscard]] double cost(PointType const& pt) const {
        return base_.cost(pt);
    }

    [[nodiscard]] bool feasible(PointType const& pt) const {
        return predicate_(pt);
    }
};

/// Factory for constrained cost models.
template <typename PointType, typename M, typename Pred>
    requires CostModel<M, PointType>
[[nodiscard]] auto make_constrained(M model, Pred pred) {
    return constrained_cost_model<M, PointType, Pred>{
        std::move(model), std::move(pred)};
}

// ─── composite_cost_model (multi-objective) ─────────────────────

/// Evaluates multiple objective functions at a point.
/// Returns a vector of costs where index 0 is the primary objective.
///
/// Usage:
///   composite_cost_model model;
///   model.add_objective("latency", latency_model);
///   model.add_objective("memory",  memory_model);
///   auto costs = model.evaluate(pt);  // {latency_ns, memory_bytes}
///
template <typename PointType>
struct composite_cost_model {
    using cost_fn = std::function<double(PointType const&)>;

    struct objective {
        std::string name;
        cost_fn     fn;
    };

    std::vector<objective> objectives;

    /// Add an objective function.
    void add_objective(std::string name, cost_fn fn) {
        objectives.push_back({std::move(name), std::move(fn)});
    }

    /// Add an objective from any CostModel.
    template <typename M>
        requires CostModel<M, PointType>
    void add_objective(std::string name, M const& model) {
        objectives.push_back({std::move(name),
            [&model](PointType const& pt) { return model.cost(pt); }});
    }

    /// Evaluate all objectives at a point.
    [[nodiscard]] std::vector<double> evaluate(PointType const& pt) const {
        std::vector<double> result;
        result.reserve(objectives.size());
        for (auto const& obj : objectives) {
            result.push_back(obj.fn(pt));
        }
        return result;
    }

    /// Primary (index-0) cost for single-objective compatibility.
    [[nodiscard]] double cost(PointType const& pt) const {
        assert(!objectives.empty());
        return objectives[0].fn(pt);
    }

    /// Number of objectives.
    [[nodiscard]] std::size_t dimension() const noexcept {
        return objectives.size();
    }

    /// Objective names.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> result;
        result.reserve(objectives.size());
        for (auto const& obj : objectives) {
            result.push_back(obj.name);
        }
        return result;
    }
};

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_COST_MODEL_H
