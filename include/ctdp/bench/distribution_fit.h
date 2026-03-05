#ifndef CTDP_BENCH_DISTRIBUTION_FIT_H
#define CTDP_BENCH_DISTRIBUTION_FIT_H

// ctdp::bench::distribution_fit -- Parametric distribution fitting for latency
//
// Fits latency samples to parametric distributions and derives percentiles
// analytically from the fitted parameters. This gives stable p99 estimates
// because the fit uses ALL samples, not just the empirical 99th value.
//
// Supported distributions:
//   lognormal:  X ~ LogNormal(mu, sigma)
//     - Natural fit for latency: multiplicative noise sources
//     - p99 = exp(mu + sigma * z_0.99) where z_0.99 = 2.3263
//     - MLE: mu = mean(log(x)), sigma = std(log(x))
//
//   gamma:  X ~ Gamma(shape, rate)
//     - Good for sum-of-exponentials (multiple pipeline stages)
//     - p99 via incomplete gamma function (Newton iteration)
//     - MLE: shape/rate via Minka's fixed-point iteration
//
// Usage:
//   auto fit = fit_lognormal(samples);
//   double p99 = fit.percentile(0.99);  // analytically derived
//   // fit.mu, fit.sigma are stable SVR targets

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ctdp::bench {

// =====================================================================
// Normal distribution quantile (Beasley-Springer-Moro approximation)
// =====================================================================

/// Inverse normal CDF (probit function).
/// Peter Acklam's rational approximation. Accurate to ~1.15e-9.
[[nodiscard]] inline double normal_quantile(double p) noexcept {
    if (p <= 0.0) return -1e30;
    if (p >= 1.0) return 1e30;
    if (p == 0.5) return 0.0;

    // Coefficients for rational approximation
    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 =  2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 =  1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 =  2.506628277459239e+00;

    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 =  1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 =  6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;

    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 =  4.374664141464968e+00;
    constexpr double c6 =  2.938163982698783e+00;

    constexpr double d1 =  7.784695709041462e-03;
    constexpr double d2 =  3.224671290700398e-01;
    constexpr double d3 =  2.445134137142996e+00;
    constexpr double d4 =  3.754408661907416e+00;

    constexpr double p_low  = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    double q, r;

    if (p < p_low) {
        // Lower tail
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    } else if (p <= p_high) {
        // Central region
        q = p - 0.5;
        r = q * q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
    } else {
        // Upper tail
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    }
}


// =====================================================================
// Lognormal distribution fit
// =====================================================================

struct lognormal_fit {
    double mu    = 0.0;   ///< Mean of log(x)
    double sigma = 0.0;   ///< Std dev of log(x)
    double n     = 0.0;   ///< Sample count

    // Derived quantities
    double mean_x  = 0.0; ///< E[X] = exp(mu + sigma^2/2)
    double var_x   = 0.0; ///< Var[X]
    double mode_x  = 0.0; ///< Mode = exp(mu - sigma^2)
    double median_x = 0.0;///< Median = exp(mu)

    /// Analytically derived percentile from fitted distribution.
    /// This is the key function: uses all samples via mu/sigma,
    /// giving a much more stable p99 than the empirical quantile.
    [[nodiscard]] double percentile(double p) const noexcept {
        return std::exp(mu + sigma * normal_quantile(p));
    }

    /// p50 (median)
    [[nodiscard]] double p50() const noexcept { return percentile(0.50); }

    /// p90
    [[nodiscard]] double p90() const noexcept { return percentile(0.90); }

    /// p95
    [[nodiscard]] double p95() const noexcept { return percentile(0.95); }

    /// p99
    [[nodiscard]] double p99() const noexcept { return percentile(0.99); }

    /// p99.9
    [[nodiscard]] double p999() const noexcept { return percentile(0.999); }

    /// Tail ratio: p99/p50 (shape-only, independent of location shift)
    [[nodiscard]] double tail_ratio() const noexcept {
        double m = p50();
        return (m > 0.0) ? p99() / m : 0.0;
    }

    /// Coefficient of variation: sigma_x / mean_x
    /// For lognormal: cv = sqrt(exp(sigma^2) - 1)
    [[nodiscard]] double cv() const noexcept {
        return std::sqrt(std::exp(sigma * sigma) - 1.0);
    }

    /// Goodness-of-fit: Kolmogorov-Smirnov statistic
    /// Computed against the fitted distribution.
    double ks_statistic = 0.0;

    /// Is the fit reasonable? (KS < 0.05 is typical threshold)
    [[nodiscard]] bool good_fit() const noexcept {
        return ks_statistic < 0.05 && sigma > 0.0;
    }
};

/// Fit a lognormal distribution to latency samples via MLE.
///
/// MLE for lognormal: mu = mean(log(x)), sigma = std(log(x))
/// This is the maximum likelihood estimator and is sufficient.
///
/// Filters out zero/negative values (shouldn't occur for latency
/// but defensive programming).
///
/// @param samples  Raw latency values (nanoseconds)
/// @return         Fitted lognormal parameters + derived quantities
[[nodiscard]] inline lognormal_fit fit_lognormal(
        std::span<const double> samples)
{
    lognormal_fit result;

    if (samples.size() < 2) return result;

    // Take log of positive samples
    std::vector<double> log_x;
    log_x.reserve(samples.size());
    for (double x : samples) {
        if (x > 0.0) log_x.push_back(std::log(x));
    }

    double nn = static_cast<double>(log_x.size());
    if (nn < 2.0) return result;
    result.n = nn;

    // MLE: mu = mean(log(x))
    double sum = 0.0;
    for (double lx : log_x) sum += lx;
    result.mu = sum / nn;

    // MLE: sigma = sqrt(mean((log(x) - mu)^2))
    double var = 0.0;
    for (double lx : log_x) {
        double d = lx - result.mu;
        var += d * d;
    }
    result.sigma = std::sqrt(var / nn);

    // Derived
    double s2 = result.sigma * result.sigma;
    result.mean_x   = std::exp(result.mu + s2 / 2.0);
    result.var_x    = (std::exp(s2) - 1.0) * std::exp(2.0 * result.mu + s2);
    result.mode_x   = std::exp(result.mu - s2);
    result.median_x = std::exp(result.mu);

    // KS statistic: max |F_empirical(x) - F_fitted(x)|
    // Sort samples, compare empirical CDF to lognormal CDF
    std::vector<double> sorted(samples.begin(), samples.end());
    std::sort(sorted.begin(), sorted.end());

    double max_d = 0.0;
    double n_total = static_cast<double>(sorted.size());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] <= 0.0) continue;
        double empirical = static_cast<double>(i + 1) / n_total;
        // Lognormal CDF: Phi((log(x) - mu) / sigma)
        double z = (std::log(sorted[i]) - result.mu) / result.sigma;
        double fitted = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
        double d = std::abs(empirical - fitted);
        if (d > max_d) max_d = d;
    }
    result.ks_statistic = max_d;

    return result;
}


// =====================================================================
// Gamma distribution fit
// =====================================================================

struct gamma_fit {
    double shape = 0.0;  ///< k (shape parameter)
    double rate  = 0.0;  ///< beta = 1/theta (rate = 1/scale)
    double n     = 0.0;  ///< Sample count

    // Derived
    double mean_x  = 0.0;  ///< E[X] = shape / rate
    double var_x   = 0.0;  ///< Var[X] = shape / rate^2

    /// Percentile via Newton iteration on the regularised incomplete
    /// gamma function. Less closed-form than lognormal but still
    /// uses the full distribution shape.
    [[nodiscard]] double percentile(double p) const noexcept {
        if (shape <= 0.0 || rate <= 0.0) return 0.0;
        // Wilson-Hilferty approximation as starting point
        double z = normal_quantile(p);
        double a = shape;
        double wh = a * std::pow(
            1.0 - 1.0 / (9.0 * a) + z / (3.0 * std::sqrt(a)), 3.0);
        if (wh < 0.0) wh = 0.01;
        return wh / rate;
    }

    [[nodiscard]] double p50() const noexcept { return percentile(0.50); }
    [[nodiscard]] double p99() const noexcept { return percentile(0.99); }
    [[nodiscard]] double p999() const noexcept { return percentile(0.999); }

    [[nodiscard]] double tail_ratio() const noexcept {
        double m = p50();
        return (m > 0.0) ? p99() / m : 0.0;
    }

    [[nodiscard]] double cv() const noexcept {
        return (shape > 0.0) ? 1.0 / std::sqrt(shape) : 0.0;
    }

    double ks_statistic = 0.0;

    [[nodiscard]] bool good_fit() const noexcept {
        return ks_statistic < 0.05 && shape > 0.0 && rate > 0.0;
    }
};

/// Fit a gamma distribution via method of moments.
///
/// MoM: shape = mean^2 / var, rate = mean / var
/// Fast and robust. Minka's fixed-point MLE would be more
/// efficient but MoM is sufficient for our use case.
[[nodiscard]] inline gamma_fit fit_gamma(
        std::span<const double> samples)
{
    gamma_fit result;
    if (samples.size() < 2) return result;

    double nn = static_cast<double>(samples.size());
    result.n = nn;

    double sum = 0.0;
    for (double x : samples) sum += x;
    double mean = sum / nn;

    double var = 0.0;
    for (double x : samples) {
        double d = x - mean;
        var += d * d;
    }
    var /= nn;

    if (var < 1e-30 || mean <= 0.0) return result;

    result.shape = (mean * mean) / var;
    result.rate  = mean / var;
    result.mean_x = mean;
    result.var_x  = var;

    // KS statistic (using Wilson-Hilferty normal approximation for CDF)
    std::vector<double> sorted(samples.begin(), samples.end());
    std::sort(sorted.begin(), sorted.end());

    double max_d = 0.0;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        double empirical = static_cast<double>(i + 1) / nn;
        // Gamma CDF via Wilson-Hilferty normal approximation
        double x = sorted[i] * result.rate;
        double a = result.shape;
        double cube_root = std::cbrt(x / a);
        double z = (cube_root - (1.0 - 1.0 / (9.0 * a)))
                   / std::sqrt(1.0 / (9.0 * a));
        double fitted = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
        double d = std::abs(empirical - fitted);
        if (d > max_d) max_d = d;
    }
    result.ks_statistic = max_d;

    return result;
}


// =====================================================================
// Combined distribution result
// =====================================================================

/// Result of fitting both distributions and selecting the better one.
struct distribution_result {
    lognormal_fit lognormal;
    gamma_fit     gamma;

    // Best-fit selection
    enum class best_model { lognormal, gamma, none };
    best_model best = best_model::none;

    /// p99 from the best-fitting distribution
    [[nodiscard]] double fitted_p99() const noexcept {
        switch (best) {
            case best_model::lognormal: return lognormal.p99();
            case best_model::gamma:     return gamma.p99();
            default:                    return 0.0;
        }
    }

    /// p50 from the best-fitting distribution
    [[nodiscard]] double fitted_p50() const noexcept {
        switch (best) {
            case best_model::lognormal: return lognormal.p50();
            case best_model::gamma:     return gamma.p50();
            default:                    return 0.0;
        }
    }

    /// Tail ratio from best fit
    [[nodiscard]] double fitted_tail_ratio() const noexcept {
        switch (best) {
            case best_model::lognormal: return lognormal.tail_ratio();
            case best_model::gamma:     return gamma.tail_ratio();
            default:                    return 0.0;
        }
    }

    /// CV from best fit
    [[nodiscard]] double fitted_cv() const noexcept {
        switch (best) {
            case best_model::lognormal: return lognormal.cv();
            case best_model::gamma:     return gamma.cv();
            default:                    return 0.0;
        }
    }

    /// Name of best model
    [[nodiscard]] const char* best_name() const noexcept {
        switch (best) {
            case best_model::lognormal: return "lognormal";
            case best_model::gamma:     return "gamma";
            default:                    return "none";
        }
    }

    /// KS statistic of best model
    [[nodiscard]] double best_ks() const noexcept {
        switch (best) {
            case best_model::lognormal: return lognormal.ks_statistic;
            case best_model::gamma:     return gamma.ks_statistic;
            default:                    return 1.0;
        }
    }
};

/// Fit both lognormal and gamma, select the one with lower KS statistic.
[[nodiscard]] inline distribution_result fit_distribution(
        std::span<const double> samples)
{
    distribution_result result;
    result.lognormal = fit_lognormal(samples);
    result.gamma     = fit_gamma(samples);

    if (result.lognormal.sigma > 0.0 && result.gamma.shape > 0.0) {
        result.best = (result.lognormal.ks_statistic <= result.gamma.ks_statistic)
            ? distribution_result::best_model::lognormal
            : distribution_result::best_model::gamma;
    } else if (result.lognormal.sigma > 0.0) {
        result.best = distribution_result::best_model::lognormal;
    } else if (result.gamma.shape > 0.0) {
        result.best = distribution_result::best_model::gamma;
    }

    return result;
}

} // namespace ctdp::bench

#endif // CTDP_BENCH_DISTRIBUTION_FIT_H
