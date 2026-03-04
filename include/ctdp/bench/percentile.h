#ifndef CTDP_BENCH_PERCENTILE_H
#define CTDP_BENCH_PERCENTILE_H

// ctdp::bench::percentile -- Tail-latency statistics for p99 calibration
//
// Extends statistics.h with percentile computation over raw timing
// vectors.  Designed for HFT-grade measurement where p99 and p99.9
// are the decision-relevant metrics, not median.
//
// Usage:
//   auto pctl = compute_percentiles(raw_ns);
//   double p99  = pctl.p99;
//   double tail = pctl.p999;
//
// Design notes:
//   - Full sort is acceptable: 100K doubles sorts in <5ms
//   - Interpolation between adjacent values at fractional indices
//   - Does NOT modify the input span (copies internally)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ctdp::bench {

// --- Percentile result ----------------------------------------------

/// Complete latency distribution summary.
struct percentile_result {
    double mean  = 0.0;
    double p50   = 0.0;
    double p90   = 0.0;
    double p95   = 0.0;
    double p99   = 0.0;
    double p999  = 0.0;
    double max   = 0.0;
    std::size_t samples = 0;

    /// Tail-to-median ratio: how much worse is p99 vs p50?
    [[nodiscard]] double tail_ratio() const noexcept {
        return (p50 > 0.0) ? p99 / p50 : 0.0;
    }

    /// Jitter: p99 - p50 (absolute tail penalty in ns)
    [[nodiscard]] double jitter_ns() const noexcept {
        return p99 - p50;
    }
};

// --- Core percentile function ---------------------------------------

/// Compute a single percentile from sorted data.
/// Uses linear interpolation between adjacent values.
///
/// @param sorted   Data in ascending order
/// @param p        Percentile in [0, 100]
/// @return         Interpolated value at the given percentile
[[nodiscard]] inline double percentile_sorted(
    std::span<const double> sorted, double p) noexcept
{
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];

    double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    auto lo = static_cast<std::size_t>(idx);
    auto hi = std::min(lo + 1, sorted.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/// Compute a single percentile from unsorted data.
/// Copies and sorts internally.
[[nodiscard]] inline double percentile(
    std::span<const double> data, double p)
{
    if (data.empty()) return 0.0;
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    return percentile_sorted(sorted, p);
}

// --- Full distribution computation ----------------------------------

/// Compute the complete latency distribution from raw timing data.
///
/// Single sort, then extract all percentiles in one pass.
/// Prefer this over calling percentile() multiple times.
[[nodiscard]] inline percentile_result compute_percentiles(
    std::span<const double> data)
{
    percentile_result result;
    result.samples = data.size();

    if (data.empty()) return result;

    // Compute mean
    double sum = 0.0;
    for (double v : data) sum += v;
    result.mean = sum / static_cast<double>(data.size());

    // Sort once, extract all percentiles
    std::vector<double> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());

    auto s = std::span<const double>{sorted};
    result.p50  = percentile_sorted(s, 50.0);
    result.p90  = percentile_sorted(s, 90.0);
    result.p95  = percentile_sorted(s, 95.0);
    result.p99  = percentile_sorted(s, 99.0);
    result.p999 = percentile_sorted(s, 99.9);
    result.max  = sorted.back();

    return result;
}

} // namespace ctdp::bench

#endif // CTDP_BENCH_PERCENTILE_H
