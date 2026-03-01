#ifndef CTDP_CALIBRATOR_CALIBRATION_PROFILE_H
#define CTDP_CALIBRATOR_CALIBRATION_PROFILE_H

// ctdp::calibrator::calibration_profile — Fitted cost model parameters
//
// Design v2.2 §5.6:
//   Fitted model parameters consumed by solver cost models.
//   Templated on <Space, Callable>.  A profile fitted from
//   fix_swar_parser data cannot be accidentally applied to a
//   fix_lookup_parser plan — the type mismatch is a compile error.
//
// The profile is the bridge between calibration (runtime measurement)
// and optimisation (compile-time DP).  It stores the fitted parameters
// that a cost_function uses to evaluate candidates.
//
// Model types:
//   lookup_model    — direct table: point → cost (exact, no interpolation)
//   linear_model    — β₀ + β₁x₁ + ... + βₙxₙ (requires FeatureEncoder)
//   custom_model    — user-provided functor (for domain-specific models)

#include "data_point.h"
#include "feature_encoder.h"
#include "provenance.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

// ─── Model parameters ────────────────────────────────────────────

/// Lookup model: stores median_ns for each measured point.
/// Works with any hashable/comparable point_type.
/// Exact — no interpolation, no generalisation.
template <typename PointType>
struct lookup_model {
    /// Stored (point → median_ns) pairs
    std::vector<std::pair<PointType, double>> entries;

    /// Predict cost for a point.  Returns -1.0 if not found.
    [[nodiscard]] double predict(PointType const& pt) const {
        for (auto const& [p, cost] : entries) {
            if (p == pt) return cost;
        }
        return -1.0;
    }

    /// Number of entries
    [[nodiscard]] std::size_t size() const noexcept {
        return entries.size();
    }
};

/// Linear model: cost = intercept + Σ(coefficients[i] * features[i])
/// Requires a FeatureEncoder to convert point_type → features.
template <std::size_t NumFeatures>
struct linear_model {
    double intercept = 0.0;
    std::array<double, NumFeatures> coefficients{};

    /// R² goodness-of-fit (set during fitting)
    double r_squared = 0.0;

    /// Predict cost from a pre-encoded feature vector
    [[nodiscard]] double predict(
        std::array<float, NumFeatures> const& features) const noexcept
    {
        double y = intercept;
        for (std::size_t i = 0; i < NumFeatures; ++i) {
            y += coefficients[i] * static_cast<double>(features[i]);
        }
        return y;
    }
};

// ─── calibration_profile ─────────────────────────────────────────

/// A calibration profile: fitted cost model keyed by <Space, Callable>.
///
/// @tparam Space     The search space type
/// @tparam Callable  The kernel identity (compile-time type key)
///
/// The profile carries:
///   - The fitted model (lookup or linear)
///   - Provenance from the source dataset
///   - Fit quality metrics
///
/// The type parameters enforce that a profile fitted from one kernel's
/// measurements cannot be applied to a different kernel's plan.
///
template <typename Space, typename Callable>
struct calibration_profile {
    using space_type    = Space;
    using callable_type = Callable;
    using point_type    = typename Space::point_type;

    // ─── Model storage (variant-style, one is active) ────────────

    /// Lookup model (direct table)
    lookup_model<point_type> lookup;

    /// Linear model coefficients (populated if linear fit was used)
    /// Width is determined at fit time; stored as vector for flexibility.
    double linear_intercept = 0.0;
    std::vector<double> linear_coefficients;
    double linear_r_squared = 0.0;

    // ─── Fit metadata ────────────────────────────────────────────

    enum class model_type : int {
        none    = 0,
        lookup  = 1,
        linear  = 2
    };

    model_type active_model = model_type::none;

    /// Provenance of the source dataset
    dataset_provenance provenance;

    /// Number of data points used for fitting
    std::size_t training_points = 0;

    // ─── Prediction ──────────────────────────────────────────────

    /// Predict cost (ns) for a space point using the lookup model.
    /// Only valid when active_model == model_type::lookup.
    [[nodiscard]] double predict_lookup(point_type const& pt) const {
        return lookup.predict(pt);
    }

    /// Predict cost (ns) from an encoded feature vector using linear model.
    /// Only valid when active_model == model_type::linear.
    [[nodiscard]] double predict_linear(
        std::span<const float> features) const noexcept
    {
        double y = linear_intercept;
        auto n = std::min(features.size(), linear_coefficients.size());
        for (std::size_t i = 0; i < n; ++i) {
            y += linear_coefficients[i] * static_cast<double>(features[i]);
        }
        return y;
    }
};

// ─── Profile fitting ─────────────────────────────────────────────

/// Fit a lookup profile from a dataset.
/// Every measured point gets its median_ns stored directly.
///
template <typename Space, typename Callable, typename MetricSnapshot>
[[nodiscard]] auto fit_lookup_profile(
    calibration_dataset<Space, Callable, MetricSnapshot> const& dataset)
    -> calibration_profile<Space, Callable>
{
    calibration_profile<Space, Callable> profile;
    profile.active_model   = calibration_profile<Space, Callable>::model_type::lookup;
    profile.provenance     = dataset.provenance;
    profile.training_points = dataset.size();

    profile.lookup.entries.reserve(dataset.size());
    for (auto const& dp : dataset.points) {
        profile.lookup.entries.emplace_back(dp.space_point, dp.median_ns);
    }

    return profile;
}

/// Fit a linear profile from a dataset using ordinary least squares.
/// Requires a FeatureEncoder to convert point_type → numeric features.
///
/// Uses the normal equations: β = (X'X)⁻¹X'y
/// This is a simple implementation for moderate-dimensional problems.
/// For production use with many features, consider regularisation.
///
template <typename Space, typename Callable, typename MetricSnapshot,
          typename Encoder>
    requires FeatureEncoder<Encoder, typename Space::point_type>
[[nodiscard]] auto fit_linear_profile(
    calibration_dataset<Space, Callable, MetricSnapshot> const& dataset,
    Encoder const& encoder)
    -> calibration_profile<Space, Callable>
{
    constexpr auto W = Encoder::width;
    auto const N = dataset.size();

    calibration_profile<Space, Callable> profile;
    profile.active_model    = calibration_profile<Space, Callable>::model_type::linear;
    profile.provenance      = dataset.provenance;
    profile.training_points = N;

    if (N == 0) return profile;

    // Encode all points and collect labels
    std::vector<std::array<float, W>> X(N);
    std::vector<double> y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X[i] = encoder.encode(dataset.points[i].space_point);
        y[i] = dataset.points[i].median_ns;
    }

    // Compute means for centering (improves numerical stability)
    std::array<double, W> x_mean{};
    double y_mean = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            x_mean[j] += static_cast<double>(X[i][j]);
        }
        y_mean += y[i];
    }
    for (std::size_t j = 0; j < W; ++j) x_mean[j] /= static_cast<double>(N);
    y_mean /= static_cast<double>(N);

    // X'X (W×W) and X'y (W×1) with centered data
    std::array<std::array<double, W>, W> XtX{};
    std::array<double, W> Xty{};

    for (std::size_t i = 0; i < N; ++i) {
        double yi = y[i] - y_mean;
        for (std::size_t j = 0; j < W; ++j) {
            double xij = static_cast<double>(X[i][j]) - x_mean[j];
            Xty[j] += xij * yi;
            for (std::size_t k = 0; k < W; ++k) {
                double xik = static_cast<double>(X[i][k]) - x_mean[k];
                XtX[j][k] += xij * xik;
            }
        }
    }

    // Solve via Gauss-Jordan elimination with partial pivoting
    // Augmented matrix: [XtX | Xty]
    std::array<std::array<double, W + 1>, W> aug{};
    for (std::size_t i = 0; i < W; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            aug[i][j] = XtX[i][j];
        }
        aug[i][W] = Xty[i];
    }

    for (std::size_t col = 0; col < W; ++col) {
        // Partial pivot
        std::size_t max_row = col;
        double max_val = std::abs(aug[col][col]);
        for (std::size_t row = col + 1; row < W; ++row) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                max_row = row;
            }
        }
        if (max_row != col) std::swap(aug[col], aug[max_row]);

        // Singular check (skip degenerate features)
        if (std::abs(aug[col][col]) < 1e-12) continue;

        double diag = aug[col][col];
        for (std::size_t j = col; j <= W; ++j) {
            aug[col][j] /= diag;
        }

        for (std::size_t row = 0; row < W; ++row) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (std::size_t j = col; j <= W; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract coefficients (centered)
    std::array<double, W> beta{};
    for (std::size_t j = 0; j < W; ++j) {
        beta[j] = aug[j][W];
    }

    // Recover intercept: β₀ = ȳ - Σ(βⱼ * x̄ⱼ)
    double intercept = y_mean;
    for (std::size_t j = 0; j < W; ++j) {
        intercept -= beta[j] * x_mean[j];
    }

    // Compute R²
    double ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double predicted = intercept;
        for (std::size_t j = 0; j < W; ++j) {
            predicted += beta[j] * static_cast<double>(X[i][j]);
        }
        ss_res += (y[i] - predicted) * (y[i] - predicted);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    double r_sq = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 0.0;

    // Store
    profile.linear_intercept = intercept;
    profile.linear_coefficients.assign(beta.begin(), beta.end());
    profile.linear_r_squared = r_sq;

    return profile;
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_CALIBRATION_PROFILE_H
