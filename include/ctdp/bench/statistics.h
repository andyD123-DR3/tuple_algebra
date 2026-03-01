#ifndef CTDP_BENCH_STATISTICS_H
#define CTDP_BENCH_STATISTICS_H

// ctdp::bench::statistics — Robust statistics for benchmark data
//
// median() — O(N log N) via partial sort
// mad()    — Median Absolute Deviation

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ctdp::bench {

/// Compute the median of a span of doubles.
/// Copies internally — does not modify the input.
[[nodiscard]] inline double median(std::span<const double> data) {
    if (data.empty()) return 0.0;
    if (data.size() == 1) return data[0];

    std::vector<double> sorted(data.begin(), data.end());
    auto mid = sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2);
    std::nth_element(sorted.begin(), mid, sorted.end());

    if (sorted.size() % 2 == 0) {
        double upper = *mid;
        auto lower_it = std::max_element(sorted.begin(), mid);
        return (*lower_it + upper) / 2.0;
    }
    return *mid;
}

/// Compute the Median Absolute Deviation (MAD).
/// MAD = median(|x_i - median(x)|)
[[nodiscard]] inline double mad(std::span<const double> data) {
    if (data.size() < 2) return 0.0;

    double med = median(data);

    std::vector<double> deviations(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        deviations[i] = std::abs(data[i] - med);
    }

    return median(std::span<const double>{deviations});
}

} // namespace ctdp::bench

#endif // CTDP_BENCH_STATISTICS_H
