#ifndef CTDP_SOLVER_COST_MODELS_PERFORMANCE_MODEL_H
#define CTDP_SOLVER_COST_MODELS_PERFORMANCE_MODEL_H

// ctdp v0.7.0 — Performance model concepts and types
// Cross-validation metrics added for genuine out-of-sample evaluation

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vector>
#include <algorithm>
#include <any>
#include <functional>
#include <memory>
#include <type_traits>

namespace ctdp::cost_models {

// ─── observation ────────────────────────────────────────────────────────
// A single (point, measured_cost) pair from benchmarking a tile kernel.

template <typename Point>
struct observation {
    Point point;       // e.g. tile_shape{TM, TN, TK}
    double cost;       // measured wall-clock time or throughput inverse
};

// ─── model_quality ──────────────────────────────────────────────────────
// Quality metrics for a fitted model. Both in-sample and out-of-sample.

struct model_quality {
    // In-sample metrics (computed on training data)
    double r2              = std::numeric_limits<double>::quiet_NaN();
    double spearman_rho    = std::numeric_limits<double>::quiet_NaN();

    // Out-of-sample metrics (5-fold cross-validation)
    double oos_r2          = std::numeric_limits<double>::quiet_NaN();
    double oos_spearman_rho= std::numeric_limits<double>::quiet_NaN();
    double oos_rmse        = std::numeric_limits<double>::quiet_NaN();

    // Leave-one-out R² (closed-form, linear only)
    double loo_r2          = std::numeric_limits<double>::quiet_NaN();

    // Model complexity
    std::size_t n_samples  = 0;
    std::size_t n_params   = 0;

    // Model identifier
    std::string model_name;
};

// ─── metric helpers ─────────────────────────────────────────────────────

inline double compute_r2(std::span<const double> actual,
                         std::span<const double> predicted) {
    const auto n = actual.size();
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();

    double mean_y = 0.0;
    for (auto v : actual) mean_y += v;
    mean_y /= static_cast<double>(n);

    double ss_tot = 0.0, ss_res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d_tot = actual[i] - mean_y;
        double d_res = actual[i] - predicted[i];
        ss_tot += d_tot * d_tot;
        ss_res += d_res * d_res;
    }
    if (ss_tot < 1e-30) return std::numeric_limits<double>::quiet_NaN();
    return 1.0 - ss_res / ss_tot;
}

inline double compute_rmse(std::span<const double> actual,
                           std::span<const double> predicted) {
    const auto n = actual.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = actual[i] - predicted[i];
        ss += d * d;
    }
    return std::sqrt(ss / static_cast<double>(n));
}

namespace detail {
    inline std::vector<double> rank_values(std::span<const double> v) {
        const auto n = v.size();
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](auto a, auto b){ return v[a] < v[b]; });

        std::vector<double> ranks(n);
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i;
            while (j < n && v[idx[j]] == v[idx[i]]) ++j;
            double avg_rank = 0.5 * (static_cast<double>(i + j) - 1.0);
            for (std::size_t k = i; k < j; ++k)
                ranks[idx[k]] = avg_rank;
            i = j;
        }
        return ranks;
    }
} // namespace detail

inline double compute_spearman(std::span<const double> x,
                               std::span<const double> y) {
    const auto n = x.size();
    if (n < 3) return std::numeric_limits<double>::quiet_NaN();

    auto rx = detail::rank_values(x);
    auto ry = detail::rank_values(y);

    double mx = 0.0, my = 0.0;
    for (std::size_t i = 0; i < n; ++i) { mx += rx[i]; my += ry[i]; }
    mx /= static_cast<double>(n);
    my /= static_cast<double>(n);

    double num = 0.0, dx2 = 0.0, dy2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double a = rx[i] - mx, b = ry[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    double denom = std::sqrt(dx2 * dy2);
    if (denom < 1e-30) return std::numeric_limits<double>::quiet_NaN();
    return num / denom;
}

// ─── performance_model concept ──────────────────────────────────────────

template <typename M, typename Point>
concept performance_model = requires(const M& m, const Point& p) {
    { m.predict(p) } -> std::convertible_to<double>;
    { m.quality() } -> std::convertible_to<model_quality>;
    { m.name() } -> std::convertible_to<std::string>;
};

// ─── any_model type-erased wrapper ──────────────────────────────────────

template <typename Point>
class any_model {
    struct concept_t {
        virtual ~concept_t() = default;
        virtual double predict(const Point& p) const = 0;
        virtual model_quality quality() const = 0;
        virtual std::string name() const = 0;
    };

    template <typename M>
    struct model_t final : concept_t {
        M model_;
        explicit model_t(M m) : model_(std::move(m)) {}
        double predict(const Point& p) const override { return model_.predict(p); }
        model_quality quality() const override { return model_.quality(); }
        std::string name() const override { return model_.name(); }
    };

    std::unique_ptr<concept_t> impl_;

public:
    any_model() = default;

    template <typename M>
        requires performance_model<M, Point>
    explicit any_model(M m)
        : impl_(std::make_unique<model_t<M>>(std::move(m))) {}

    double predict(const Point& p) const { return impl_->predict(p); }
    model_quality quality() const { return impl_->quality(); }
    std::string name() const { return impl_->name(); }
    explicit operator bool() const { return impl_ != nullptr; }
};

} // namespace ctdp::cost_models

#endif // CTDP_SOLVER_COST_MODELS_PERFORMANCE_MODEL_H
