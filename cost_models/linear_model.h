#ifndef CTDP_SOLVER_COST_MODELS_LINEAR_MODEL_H
#define CTDP_SOLVER_COST_MODELS_LINEAR_MODEL_H

// ctdp v0.7.0 — Linear regression performance model
// OLS via normal equations with z-score normalisation.
// Quality: in-sample + closed-form LOO + 5-fold CV OOS metrics.

#include "performance_model.h"
#include "feature_extract.h"
#include "cross_validation.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace ctdp::cost_models {

// ─── Dense matrix helpers (column-major, minimal) ───────────────────────

namespace linalg {

struct matrix {
    std::size_t rows = 0, cols = 0;
    std::vector<double> data;  // column-major

    matrix() = default;
    matrix(std::size_t r, std::size_t c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    double& operator()(std::size_t i, std::size_t j) { return data[j * rows + i]; }
    double  operator()(std::size_t i, std::size_t j) const { return data[j * rows + i]; }
};

// X^T X
inline matrix xtx(const matrix& X) {
    matrix R(X.cols, X.cols);
    for (std::size_t j = 0; j < X.cols; ++j)
        for (std::size_t k = j; k < X.cols; ++k) {
            double s = 0.0;
            for (std::size_t i = 0; i < X.rows; ++i)
                s += X(i, j) * X(i, k);
            R(j, k) = R(k, j) = s;
        }
    return R;
}

// X^T y
inline std::vector<double> xty(const matrix& X, std::span<const double> y) {
    std::vector<double> r(X.cols, 0.0);
    for (std::size_t j = 0; j < X.cols; ++j)
        for (std::size_t i = 0; i < X.rows; ++i)
            r[j] += X(i, j) * y[i];
    return r;
}

// Solve A x = b via Cholesky (A must be symmetric positive definite)
inline std::vector<double> cholesky_solve(matrix A, std::vector<double> b) {
    const auto n = A.rows;
    // Cholesky: A = L L^T
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < j; ++k)
            s += A(j, k) * A(j, k);
        A(j, j) = std::sqrt(A(j, j) - s);
        for (std::size_t i = j + 1; i < n; ++i) {
            s = 0.0;
            for (std::size_t k = 0; k < j; ++k)
                s += A(i, k) * A(j, k);
            A(i, j) = (A(i, j) - s) / A(j, j);
        }
    }
    // Forward: L z = b
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < i; ++k)
            s += A(i, k) * b[k];
        b[i] = (b[i] - s) / A(i, i);
    }
    // Back: L^T x = z
    for (std::size_t i = n; i-- > 0;) {
        double s = 0.0;
        for (std::size_t k = i + 1; k < n; ++k)
            s += A(k, i) * b[k];
        b[i] = (b[i] - s) / A(i, i);
    }
    return b;
}

// Hat matrix diagonal: h_ii = X_i^T (X^T X)^{-1} X_i
// Uses Cholesky factor L of X^T X. Computes (X^T X)^{-1} fully.
inline std::vector<double> hat_diagonal(const matrix& X) {
    const auto n = X.rows;
    const auto p = X.cols;
    auto A = xtx(X);

    // Invert via solving A * col = e_j for each j
    matrix Ainv(p, p);
    for (std::size_t j = 0; j < p; ++j) {
        std::vector<double> ej(p, 0.0);
        ej[j] = 1.0;
        auto col = cholesky_solve(A, ej);
        for (std::size_t i = 0; i < p; ++i)
            Ainv(i, j) = col[i];
    }

    // h_ii = sum_j sum_k X(i,j) * Ainv(j,k) * X(i,k)
    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < p; ++k)
                sum += X(i, j) * Ainv(j, k) * X(i, k);
        h[i] = sum;
    }
    return h;
}

} // namespace linalg

// ─── Z-score normalisation ──────────────────────────────────────────────

struct zscore_params {
    std::vector<double> mean;
    std::vector<double> stddev;
};

inline zscore_params compute_zscore(const std::vector<std::vector<double>>& X) {
    if (X.empty()) return {};
    const auto n = X.size();
    const auto p = X[0].size();
    zscore_params zs;
    zs.mean.resize(p, 0.0);
    zs.stddev.resize(p, 0.0);

    for (const auto& row : X)
        for (std::size_t j = 0; j < p; ++j)
            zs.mean[j] += row[j];
    for (std::size_t j = 0; j < p; ++j)
        zs.mean[j] /= static_cast<double>(n);

    for (const auto& row : X)
        for (std::size_t j = 0; j < p; ++j) {
            double d = row[j] - zs.mean[j];
            zs.stddev[j] += d * d;
        }
    for (std::size_t j = 0; j < p; ++j) {
        zs.stddev[j] = std::sqrt(zs.stddev[j] / static_cast<double>(n));
        if (zs.stddev[j] < 1e-15) zs.stddev[j] = 1.0;  // constant feature
    }
    return zs;
}

inline std::vector<double> apply_zscore(const std::vector<double>& x,
                                        const zscore_params& zs) {
    std::vector<double> r(x.size());
    for (std::size_t j = 0; j < x.size(); ++j)
        r[j] = (x[j] - zs.mean[j]) / zs.stddev[j];
    return r;
}

// ─── linear_model ───────────────────────────────────────────────────────

template <typename Point, typename Extractor = log2_interactions_extractor>
class linear_model {
    std::vector<double> beta_;   // coefficients (intercept is beta_[0])
    zscore_params zscore_;
    Extractor extractor_;
    model_quality quality_;

public:
    linear_model() = default;
    linear_model(std::vector<double> beta, zscore_params zs,
                 Extractor ext, model_quality q)
        : beta_(std::move(beta)), zscore_(std::move(zs)),
          extractor_(std::move(ext)), quality_(std::move(q)) {}

    double predict(const Point& p) const {
        auto raw = extractor_(p);
        auto feat = apply_zscore(raw, zscore_);
        double yhat = beta_[0]; // intercept
        for (std::size_t j = 0; j < feat.size(); ++j)
            yhat += beta_[j + 1] * feat[j];
        return yhat;
    }

    model_quality quality() const { return quality_; }
    std::string name() const { return std::string("linear_") + Extractor::feature_name(); }
    const std::vector<double>& coefficients() const { return beta_; }
    const zscore_params& zscore() const { return zscore_; }
};

// ─── linear_trainer ─────────────────────────────────────────────────────

template <typename Point, typename Extractor = log2_interactions_extractor>
class linear_trainer {
    Extractor extractor_;

public:
    explicit linear_trainer(Extractor ext = {}) : extractor_(std::move(ext)) {}

    // Build a linear model from observations (no CV, just fit)
    linear_model<Point, Extractor> build_raw(
            const std::vector<observation<Point>>& obs) const {
        auto X_raw = extract_features(extractor_, obs);
        auto y = extract_targets(obs);
        auto zs = compute_zscore(X_raw);

        const auto n = obs.size();
        const auto p = X_raw[0].size();

        // Build design matrix with intercept column
        linalg::matrix X(n, p + 1);
        for (std::size_t i = 0; i < n; ++i) {
            X(i, 0) = 1.0;  // intercept
            auto feat = apply_zscore(X_raw[i], zs);
            for (std::size_t j = 0; j < p; ++j)
                X(i, j + 1) = feat[j];
        }

        // OLS: beta = (X^T X)^{-1} X^T y
        auto A = linalg::xtx(X);
        // Tikhonov regularisation (tiny) for numerical stability
        for (std::size_t j = 0; j < A.rows; ++j)
            A(j, j) += 1e-10;
        auto rhs = linalg::xty(X, y);
        auto beta = linalg::cholesky_solve(std::move(A), std::move(rhs));

        // In-sample predictions
        std::vector<double> yhat(n);
        for (std::size_t i = 0; i < n; ++i) {
            double v = beta[0];
            auto feat = apply_zscore(X_raw[i], zs);
            for (std::size_t j = 0; j < p; ++j)
                v += beta[j + 1] * feat[j];
            yhat[i] = v;
        }

        model_quality q;
        q.r2 = compute_r2(y, yhat);
        q.spearman_rho = compute_spearman(
            std::span<const double>(y), std::span<const double>(yhat));
        q.n_samples = n;
        q.n_params = p + 1;
        q.model_name = std::string("linear_") + Extractor::feature_name();

        // Closed-form LOO R² via hat matrix
        // LOO residual_i = (y_i - yhat_i) / (1 - h_ii)
        auto h = linalg::hat_diagonal(X);
        double mean_y = 0.0;
        for (auto v : y) mean_y += v;
        mean_y /= static_cast<double>(n);

        double ss_tot = 0.0, ss_loo = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double d = y[i] - mean_y;
            ss_tot += d * d;
            double loo_res = (y[i] - yhat[i]) / (1.0 - h[i]);
            ss_loo += loo_res * loo_res;
        }
        q.loo_r2 = (ss_tot > 1e-30) ? 1.0 - ss_loo / ss_tot
                                     : std::numeric_limits<double>::quiet_NaN();

        return linear_model<Point, Extractor>(
            std::move(beta), std::move(zs), extractor_, std::move(q));
    }

    // Build with full quality metrics (in-sample + 5-fold CV)
    linear_model<Point, Extractor> build(
            const std::vector<observation<Point>>& obs) const {
        // First fit on full data
        auto model = build_raw(obs);

        // Then run 5-fold CV for OOS metrics
        if (obs.size() >= 5) {
            auto cv = k_fold_cv<Point>(obs, 5,
                [this](const std::vector<observation<Point>>& train) {
                    return this->build_raw(train);
                });

            auto q = model.quality();
            q.oos_r2 = cv.oos_r2;
            q.oos_spearman_rho = cv.oos_spearman_rho;
            q.oos_rmse = cv.oos_rmse;

            return linear_model<Point, Extractor>(
                model.coefficients(), model.zscore(), extractor_, std::move(q));
        }
        return model;
    }
};

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_LINEAR_MODEL_H
