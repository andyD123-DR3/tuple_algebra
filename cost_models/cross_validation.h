#ifndef CTDP_SOLVER_COST_MODELS_CROSS_VALIDATION_H
#define CTDP_SOLVER_COST_MODELS_CROSS_VALIDATION_H

// ctdp v0.7.0 — Generic K-fold cross-validation infrastructure
// Stratified by target value rank for balanced folds.

#include "performance_model.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

namespace ctdp::cost_models {

// ─── Fold assignment ────────────────────────────────────────────────────
// Stratified: sort observations by target rank, deal round-robin into folds.
// This ensures each fold has a representative spread of target values.

inline std::vector<std::size_t> stratified_fold_assignment(
        std::span<const double> targets, std::size_t k) {
    const auto n = targets.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](auto a, auto b){ return targets[a] < targets[b]; });

    std::vector<std::size_t> folds(n);
    for (std::size_t i = 0; i < n; ++i)
        folds[idx[i]] = i % k;
    return folds;
}

// ─── CV result ──────────────────────────────────────────────────────────

struct cv_result {
    double oos_r2          = 0.0;
    double oos_spearman_rho= 0.0;
    double oos_rmse        = 0.0;
    std::vector<double> oos_predictions;   // aligned with original order
    std::vector<double> oos_actuals;       // aligned with original order
};

// ─── k_fold_cv ──────────────────────────────────────────────────────────
// Generic K-fold cross-validation.
//
// TrainerFn: (train_obs) -> model_like
//   where model_like has .predict(point) -> double
//
// Usage:
//   auto result = k_fold_cv(observations, 5,
//       [&](const auto& train) { return linear_trainer{ext}.build(train); });

template <typename Point, typename TrainerFn>
cv_result k_fold_cv(
        const std::vector<observation<Point>>& obs,
        std::size_t k,
        TrainerFn trainer) {
    const auto n = obs.size();
    if (n < k || k < 2) return {};

    // Extract targets for stratification
    std::vector<double> targets(n);
    for (std::size_t i = 0; i < n; ++i)
        targets[i] = obs[i].cost;

    auto folds = stratified_fold_assignment(targets, k);

    // Collect out-of-sample predictions in original order
    std::vector<double> oos_pred(n, 0.0);
    std::vector<double> oos_actual(n, 0.0);

    for (std::size_t fold = 0; fold < k; ++fold) {
        // Split
        std::vector<observation<Point>> train_obs;
        std::vector<std::size_t> test_idx;

        for (std::size_t i = 0; i < n; ++i) {
            if (folds[i] == fold) {
                test_idx.push_back(i);
            } else {
                train_obs.push_back(obs[i]);
            }
        }

        if (train_obs.empty() || test_idx.empty()) continue;

        // Train on K-1 folds
        auto model = trainer(train_obs);

        // Predict on held-out fold
        for (auto ti : test_idx) {
            oos_pred[ti]   = model.predict(obs[ti].point);
            oos_actual[ti] = obs[ti].cost;
        }
    }

    // Compute OOS metrics from the collected predictions
    cv_result result;
    result.oos_predictions = oos_pred;
    result.oos_actuals     = oos_actual;

    std::span<const double> act_span(oos_actual);
    std::span<const double> pred_span(oos_pred);

    result.oos_r2           = compute_r2(act_span, pred_span);
    result.oos_spearman_rho = compute_spearman(act_span, pred_span);
    result.oos_rmse         = compute_rmse(act_span, pred_span);

    return result;
}

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_CROSS_VALIDATION_H
