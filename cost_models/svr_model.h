#ifndef CTDP_SOLVER_COST_MODELS_SVR_MODEL_H
#define CTDP_SOLVER_COST_MODELS_SVR_MODEL_H

// ctdp v0.7.0 — Kernel ridge regression performance model
// RBF kernel: K(x,y) = exp(-γ ||x-y||²)
// Two-stage hyperparameter tuning:
//   Stage 1: Coarse grid search λ×γ via 5-fold CV Spearman ρ
//   Stage 2: Fine-tune ±0.5 log-steps around winner
// Final model: retrain on full data with tuned (λ, γ)

#include "performance_model.h"
#include "feature_extract.h"
#include "cross_validation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ctdp::cost_models {

// ─── svr_model (kernel ridge regression) ────────────────────────────────

template <typename Point, typename Extractor = log2_interactions_extractor>
class svr_model {
    std::vector<double> alpha_;           // dual coefficients
    std::vector<std::vector<double>> sv_; // support vectors (all training features)
    double y_mean_ = 0.0;
    double gamma_ = 1.0;
    double lambda_ = 0.01;
    zscore_params zscore_;
    Extractor extractor_;
    model_quality quality_;

    static double rbf_kernel(const std::vector<double>& a,
                             const std::vector<double>& b,
                             double gamma) {
        double sq = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            sq += d * d;
        }
        return std::exp(-gamma * sq);
    }

public:
    svr_model() = default;
    svr_model(std::vector<double> alpha, std::vector<std::vector<double>> sv,
              double ym, double g, double lam, zscore_params zs,
              Extractor ext, model_quality q)
        : alpha_(std::move(alpha)), sv_(std::move(sv)),
          y_mean_(ym), gamma_(g), lambda_(lam),
          zscore_(std::move(zs)), extractor_(std::move(ext)),
          quality_(std::move(q)) {}

    double predict(const Point& p) const {
        auto raw = extractor_(p);
        auto feat = apply_zscore(raw, zscore_);
        double sum = y_mean_;
        for (std::size_t i = 0; i < sv_.size(); ++i)
            sum += alpha_[i] * rbf_kernel(feat, sv_[i], gamma_);
        return sum;
    }

    model_quality quality() const { return quality_; }
    std::string name() const { return "svr_rbf"; }

    double gamma() const { return gamma_; }
    double lambda() const { return lambda_; }
    const std::vector<double>& alpha() const { return alpha_; }
    const std::vector<std::vector<double>>& support_vectors() const { return sv_; }
    const zscore_params& zscore() const { return zscore_; }
    double y_mean() const { return y_mean_; }
};

// ─── Cholesky solve for kernel systems ──────────────────────────────────

namespace krr_detail {

inline std::vector<double> solve_spd(std::vector<std::vector<double>>& A,
                                     std::vector<double>& b) {
    const auto n = A.size();
    // In-place Cholesky L
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < j; ++k)
            s += A[j][k] * A[j][k];
        A[j][j] = std::sqrt(A[j][j] - s);
        for (std::size_t i = j + 1; i < n; ++i) {
            s = 0.0;
            for (std::size_t k = 0; k < j; ++k)
                s += A[i][k] * A[j][k];
            A[i][j] = (A[i][j] - s) / A[j][j];
        }
    }
    // Forward: L z = b
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < i; ++k)
            s += A[i][k] * b[k];
        b[i] = (b[i] - s) / A[i][i];
    }
    // Back: L^T x = z
    for (std::size_t i = n; i-- > 0;) {
        double s = 0.0;
        for (std::size_t k = i + 1; k < n; ++k)
            s += A[k][i] * b[k];
        b[i] = (b[i] - s) / A[i][i];
    }
    return b;
}

} // namespace krr_detail

// ─── svr_trainer ────────────────────────────────────────────────────────

template <typename Point, typename Extractor = log2_interactions_extractor>
class svr_trainer {
    Extractor extractor_;

    static double rbf_kernel(const std::vector<double>& a,
                             const std::vector<double>& b,
                             double gamma) {
        double sq = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            sq += d * d;
        }
        return std::exp(-gamma * sq);
    }

public:
    explicit svr_trainer(Extractor ext = {}) : extractor_(std::move(ext)) {}

    // Build KRR with fixed (lambda, gamma), no CV
    svr_model<Point, Extractor> build_fixed(
            const std::vector<observation<Point>>& obs,
            double lambda, double gamma) const {
        auto X_raw = extract_features(extractor_, obs);
        auto y = extract_targets(obs);
        const auto n = obs.size();

        auto zs = compute_zscore(X_raw);
        std::vector<std::vector<double>> X(n);
        for (std::size_t i = 0; i < n; ++i)
            X[i] = apply_zscore(X_raw[i], zs);

        // Center targets
        double y_mean = 0.0;
        for (auto v : y) y_mean += v;
        y_mean /= static_cast<double>(n);
        std::vector<double> yc(n);
        for (std::size_t i = 0; i < n; ++i)
            yc[i] = y[i] - y_mean;

        // Build kernel matrix K + λI
        std::vector<std::vector<double>> K(n, std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i) {
            K[i][i] = 1.0 + lambda;
            for (std::size_t j = i + 1; j < n; ++j) {
                double k = rbf_kernel(X[i], X[j], gamma);
                K[i][j] = k;
                K[j][i] = k;
            }
        }

        // Solve (K + λI) α = y_centered
        auto alpha = krr_detail::solve_spd(K, yc);

        // In-sample predictions
        std::vector<double> yhat(n);
        for (std::size_t i = 0; i < n; ++i) {
            double s = y_mean;
            for (std::size_t j = 0; j < n; ++j)
                s += alpha[j] * rbf_kernel(X[i], X[j], gamma);
            yhat[i] = s;
        }

        model_quality q;
        q.r2 = compute_r2(y, yhat);
        q.spearman_rho = compute_spearman(
            std::span<const double>(y), std::span<const double>(yhat));
        q.n_samples = n;
        q.n_params = n;  // effective params ≈ trace(K(K+λI)^{-1})
        q.model_name = "svr_rbf";

        return svr_model<Point, Extractor>(
            std::move(alpha), std::move(X), y_mean, gamma, lambda,
            std::move(zs), extractor_, std::move(q));
    }

    // Evaluate (lambda, gamma) pair using 5-fold CV Spearman ρ
    double cv_spearman(
            const std::vector<observation<Point>>& obs,
            double lambda, double gamma) const {
        if (obs.size() < 5) return -1.0;

        auto cv = k_fold_cv<Point>(obs, 5,
            [this, lambda, gamma](const std::vector<observation<Point>>& train) {
                return this->build_fixed(train, lambda, gamma);
            });
        return cv.oos_spearman_rho;
    }

    // Build with two-stage hyperparameter tuning via 5-fold CV
    svr_model<Point, Extractor> build(
            const std::vector<observation<Point>>& obs) const {
        // ── Stage 1: Coarse grid search ──
        std::vector<double> log_lambdas = {-3.0, -2.0, -1.0, 0.0, 1.0};
        std::vector<double> log_gammas  = {-3.0, -2.0, -1.0, 0.0, 1.0};

        double best_rho = -2.0;
        double best_log_lam = 0.0, best_log_gam = 0.0;

        for (double ll : log_lambdas) {
            for (double lg : log_gammas) {
                double lam = std::pow(10.0, ll);
                double gam = std::pow(10.0, lg);
                double rho = cv_spearman(obs, lam, gam);
                if (rho > best_rho) {
                    best_rho = rho;
                    best_log_lam = ll;
                    best_log_gam = lg;
                }
            }
        }

        // ── Stage 2: Fine-tune ±0.5 log-steps ──
        std::vector<double> fine_offsets = {-0.5, -0.25, 0.0, 0.25, 0.5};
        for (double dl : fine_offsets) {
            for (double dg : fine_offsets) {
                double ll = best_log_lam + dl;
                double lg = best_log_gam + dg;
                double lam = std::pow(10.0, ll);
                double gam = std::pow(10.0, lg);
                double rho = cv_spearman(obs, lam, gam);
                if (rho > best_rho) {
                    best_rho = rho;
                    best_log_lam = ll;
                    best_log_gam = lg;
                }
            }
        }

        double final_lambda = std::pow(10.0, best_log_lam);
        double final_gamma  = std::pow(10.0, best_log_gam);

        // ── Final model: retrain on full data ──
        auto model = build_fixed(obs, final_lambda, final_gamma);

        // ── OOS metrics from CV with best hyperparams ──
        if (obs.size() >= 5) {
            auto cv = k_fold_cv<Point>(obs, 5,
                [this, final_lambda, final_gamma](
                    const std::vector<observation<Point>>& train) {
                    return this->build_fixed(train, final_lambda, final_gamma);
                });

            auto q = model.quality();
            q.oos_r2 = cv.oos_r2;
            q.oos_spearman_rho = cv.oos_spearman_rho;
            q.oos_rmse = cv.oos_rmse;

            return svr_model<Point, Extractor>(
                model.alpha(), model.support_vectors(), model.y_mean(),
                model.gamma(), model.lambda(), model.zscore(),
                extractor_, std::move(q));
        }
        return model;
    }
};

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_SVR_MODEL_H
