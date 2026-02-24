#ifndef CTDP_SOLVER_COST_MODELS_DATA_PREPROCESS_H
#define CTDP_SOLVER_COST_MODELS_DATA_PREPROCESS_H

// ctdp v0.7.0 — Data preprocessing pipeline
//
// Sits in front of the model trainers. Two stages:
//
//   1. (Optional) Spike outlier removal: discard observations whose target
//      deviates by more than k standard deviations from the robust centre.
//      Uses median + MAD for the initial pass (resistant to the very outliers
//      we're trying to remove), then recomputes mean/std on the clean set.
//
//   2. Normalise & centre targets: z-score transform (subtract mean, divide
//      by std) so that downstream models see well-conditioned data regardless
//      of the raw cost scale.
//
// The preprocessor records the fitted transform so that model predictions
// can be mapped back to the original scale via inverse().
//
// Usage:
//   data_preprocessor<Point> pp;
//   pp.set_outlier_sigma(3.0);          // optional — default is no filtering
//   auto result = pp.fit_transform(raw_observations);
//   auto model  = factory.build(result.observations);  // train on clean data
//   // wrap for prediction in original units:
//   auto wrapped = make_transformed_model(std::move(model), result.transform);
//   double pred  = wrapped.predict(some_point);  // in original cost scale

#include "performance_model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace ctdp::cost_models {

// ─── Target transform ───────────────────────────────────────────────────
// Records the affine mapping:  normalised = (raw - mean) / std
// and its inverse:             raw = normalised * std + mean

struct target_transform {
    double mean = 0.0;
    double std  = 1.0;

    double forward(double raw) const {
        return (raw - mean) / std;
    }

    double inverse(double normalised) const {
        return normalised * std + mean;
    }
};

// ─── Outlier info ───────────────────────────────────────────────────────

template <typename Point>
struct outlier_report {
    std::vector<std::size_t>          removed_indices;
    std::vector<observation<Point>>   removed_observations;
    double                            threshold_low  = 0.0;
    double                            threshold_high = 0.0;
    double                            centre         = 0.0;  // robust centre used
    double                            spread         = 0.0;  // robust spread used
};

// ─── Preprocessing result ───────────────────────────────────────────────

template <typename Point>
struct preprocessed_result {
    std::vector<observation<Point>>   observations;    // clean, targets normalised
    target_transform                  transform;       // for inverse mapping
    std::optional<outlier_report<Point>> outliers;     // populated if filtering was on
    std::size_t                       n_original = 0;  // count before filtering
};

// ─── data_preprocessor ──────────────────────────────────────────────────

template <typename Point>
class data_preprocessor {
    std::optional<double> outlier_sigma_;  // if set, remove > k*σ spikes

public:
    data_preprocessor() = default;

    // Enable spike outlier removal.  Observations whose target deviates by
    // more than k standard deviations from the robust centre are discarded.
    // Typical values: 2.5 – 4.0.  Pass std::nullopt to disable.
    void set_outlier_sigma(std::optional<double> k) { outlier_sigma_ = k; }
    std::optional<double> outlier_sigma() const { return outlier_sigma_; }

    // ── fit_transform ───────────────────────────────────────────────────
    // 1. Optionally remove spike outliers (robust detection via median+MAD)
    // 2. Compute target mean/std on the clean set
    // 3. Z-score normalise the targets

    preprocessed_result<Point> fit_transform(
            const std::vector<observation<Point>>& raw) const {

        preprocessed_result<Point> result;
        result.n_original = raw.size();

        if (raw.empty()) return result;

        // ── Stage 1: outlier removal (optional) ─────────────────────────
        std::vector<observation<Point>> clean;

        if (outlier_sigma_.has_value() && raw.size() >= 4) {
            double k = outlier_sigma_.value();

            // Robust centre: median of targets
            std::vector<double> sorted_targets;
            sorted_targets.reserve(raw.size());
            for (const auto& o : raw) sorted_targets.push_back(o.cost);
            std::sort(sorted_targets.begin(), sorted_targets.end());

            double median = (sorted_targets.size() % 2 == 0)
                ? 0.5 * (sorted_targets[sorted_targets.size()/2 - 1]
                        + sorted_targets[sorted_targets.size()/2])
                : sorted_targets[sorted_targets.size()/2];

            // Robust spread: MAD (median absolute deviation) * 1.4826
            // The 1.4826 factor makes MAD a consistent estimator of σ
            // for normally distributed data.
            std::vector<double> abs_devs(sorted_targets.size());
            for (std::size_t i = 0; i < sorted_targets.size(); ++i)
                abs_devs[i] = std::abs(sorted_targets[i] - median);
            std::sort(abs_devs.begin(), abs_devs.end());

            double mad = (abs_devs.size() % 2 == 0)
                ? 0.5 * (abs_devs[abs_devs.size()/2 - 1]
                        + abs_devs[abs_devs.size()/2])
                : abs_devs[abs_devs.size()/2];

            double robust_sigma = mad * 1.4826;

            // Guard: if robust_sigma is tiny, all data is near-identical
            // — don't filter anything
            if (robust_sigma < 1e-15) {
                clean = raw;
            } else {
                double lo = median - k * robust_sigma;
                double hi = median + k * robust_sigma;

                outlier_report<Point> report;
                report.centre         = median;
                report.spread         = robust_sigma;
                report.threshold_low  = lo;
                report.threshold_high = hi;

                for (std::size_t i = 0; i < raw.size(); ++i) {
                    if (raw[i].cost < lo || raw[i].cost > hi) {
                        report.removed_indices.push_back(i);
                        report.removed_observations.push_back(raw[i]);
                    } else {
                        clean.push_back(raw[i]);
                    }
                }

                result.outliers = std::move(report);
            }
        } else {
            clean = raw;
        }

        if (clean.empty()) {
            // Everything was an outlier — fall back to raw data
            clean = raw;
            if (result.outliers.has_value()) {
                result.outliers->removed_indices.clear();
                result.outliers->removed_observations.clear();
            }
        }

        // ── Stage 2: compute target statistics on clean data ────────────
        const auto n = clean.size();
        double sum = 0.0;
        for (const auto& o : clean) sum += o.cost;
        double mean = sum / static_cast<double>(n);

        double var = 0.0;
        for (const auto& o : clean) {
            double d = o.cost - mean;
            var += d * d;
        }
        double std_dev = std::sqrt(var / static_cast<double>(n));
        if (std_dev < 1e-15) std_dev = 1.0;  // constant targets

        result.transform.mean = mean;
        result.transform.std  = std_dev;

        // ── Stage 3: normalise targets ──────────────────────────────────
        result.observations.reserve(n);
        for (const auto& o : clean) {
            result.observations.push_back(
                observation<Point>{o.point, result.transform.forward(o.cost)});
        }

        return result;
    }
};

// ─── transformed_model ──────────────────────────────────────────────────
// Wraps any model trained on normalised data, applying the inverse
// transform so that predict() returns values in the original cost scale.

template <typename Point>
class transformed_model {
    any_model<Point> inner_;
    target_transform transform_;

public:
    transformed_model() = default;
    transformed_model(any_model<Point> m, target_transform t)
        : inner_(std::move(m)), transform_(std::move(t)) {}

    double predict(const Point& p) const {
        return transform_.inverse(inner_.predict(p));
    }

    model_quality quality() const { return inner_.quality(); }
    std::string name() const { return inner_.name(); }

    const target_transform& transform() const { return transform_; }
    explicit operator bool() const { return static_cast<bool>(inner_); }
};

// Convenience factory function
template <typename Point>
transformed_model<Point> make_transformed_model(
        any_model<Point> m, target_transform t) {
    return transformed_model<Point>(std::move(m), std::move(t));
}

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_DATA_PREPROCESS_H
