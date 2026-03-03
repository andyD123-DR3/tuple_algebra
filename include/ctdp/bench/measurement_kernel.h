#ifndef CTDP_BENCH_MEASUREMENT_KERNEL_H
#define CTDP_BENCH_MEASUREMENT_KERNEL_H

// ctdp::bench::measurement_kernel — Core measurement functions
//
// measure_once()     — single measurement: warmup → setup → measure → snapshot
// measure_repeated() — multiple reps: returns vectors of timings + snapshots
//
// Measurement sequence per rep:
//   1. Warmup: N × (fn + DoNotOptimize + ClobberMemory)
//   2. Setup hook (cache thrash slot)
//   3. metric.start()
//   4. steady_clock::now()
//   5. M × (fn + DoNotOptimize + ClobberMemory)
//   6. steady_clock::now()
//   7. metric.stop()
//   8. Compute wall_ns
//   9. metric.snapshot()
//  10. compute_derived(wall_ns) via if constexpr

#include "compiler_barrier.h"
#include "metric.h"
#include "statistics.h"

#include <chrono>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace ctdp::bench {

// ─── Result types ───────────────────────────────────────────────────

/// Result of a single measurement
template <typename SnapshotType>
struct measurement_result {
    double       wall_ns  = 0.0;   ///< Wall-clock nanoseconds for the interval
    SnapshotType snapshot{};        ///< Metric snapshot for the interval
};

/// Result of repeated measurements
template <typename SnapshotType>
struct repeated_result {
    double                    median_ns  = 0.0; ///< Median wall_ns across reps
    double                    mad_ns     = 0.0; ///< MAD of wall_ns across reps
    std::vector<double>       all_ns;           ///< Per-rep wall_ns
    std::vector<SnapshotType> all_snapshots;    ///< Per-rep metric snapshots

    /// Invariant: all_ns.size() == all_snapshots.size()
    [[nodiscard]] std::size_t reps() const noexcept { return all_ns.size(); }
};

// ─── SetupCallable concept ──────────────────────────────────────────

template <typename S>
concept SetupCallable = std::invocable<S>;

// ─── measure_once ───────────────────────────────────────────────────

/// Single measurement with warmup, setup hook, and metric snapshot.
///
/// @tparam Callable      Callable returning result_token
/// @tparam Setup         Callable invoked between warmup and measurement
/// @tparam M             Metric type
/// @param fn             The function to measure
/// @param setup          Setup hook (e.g., cache thrash)
/// @param metric         Metric instance
/// @param warmup_iters   Number of warmup iterations
/// @param measure_iters  Number of measurement iterations (wall_ns is total)
template <typename Callable, typename Setup, Metric M>
    requires std::invocable<Callable> && SetupCallable<Setup>
[[nodiscard]] auto measure_once(
    Callable&& fn,
    Setup&& setup,
    M& metric,
    std::size_t warmup_iters,
    std::size_t measure_iters)
    -> measurement_result<typename M::snapshot_type>
{
    using snapshot_t = typename M::snapshot_type;

    // 1. Warmup
    for (std::size_t i = 0; i < warmup_iters; ++i) {
        auto tok = fn();
        DoNotOptimize(tok.value);
        ClobberMemory();
    }

    // 2. Setup hook
    setup();

    // 3–6. Measure
    metric.start();
    auto t0 = std::chrono::steady_clock::now();

    result_token accumulated{0};
    for (std::size_t i = 0; i < measure_iters; ++i) {
        auto tok = fn();
        DoNotOptimize(tok.value);  // force materialisation per iteration
        accumulated = mix_token(accumulated, tok);
        ClobberMemory();
    }

    auto t1 = std::chrono::steady_clock::now();
    metric.stop();

    // Consume accumulated token
    accumulated.consume();

    // 7–8. Compute wall_ns
    double wall_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();

    // 9. Snapshot
    snapshot_t snap = metric.snapshot();

    // 10. Compute derived ratios if snapshot supports it
    if constexpr (requires(snapshot_t& s, double ns) { s.compute_derived(ns); }) {
        snap.compute_derived(wall_ns);
    }

    return measurement_result<snapshot_t>{wall_ns, snap};
}

// ─── measure_repeated ───────────────────────────────────────────────

/// Repeated measurements: runs measure_once `reps` times and collects
/// all timings and snapshots. Computes median and MAD of wall_ns.
///
/// @param fn             The function to measure
/// @param setup          Setup hook (called before each rep's measurement)
/// @param metric         Metric instance (reused across reps)
/// @param reps           Number of repetitions
/// @param warmup_iters   Warmup iterations per rep
/// @param measure_iters  Measurement iterations per rep
template <typename Callable, typename Setup, Metric M>
    requires std::invocable<Callable> && SetupCallable<Setup>
[[nodiscard]] auto measure_repeated(
    Callable&& fn,
    Setup&& setup,
    M& metric,
    std::size_t reps,
    std::size_t warmup_iters,
    std::size_t measure_iters)
    -> repeated_result<typename M::snapshot_type>
{
    using snapshot_t = typename M::snapshot_type;
    repeated_result<snapshot_t> result;

    result.all_ns.reserve(reps);
    result.all_snapshots.reserve(reps);

    for (std::size_t r = 0; r < reps; ++r) {
        auto single = measure_once(fn, setup, metric,
                                   warmup_iters, measure_iters);
        result.all_ns.push_back(single.wall_ns);
        result.all_snapshots.push_back(std::move(single.snapshot));
    }

    // Compute robust statistics
    result.median_ns = median(std::span<const double>{result.all_ns});
    result.mad_ns    = mad(std::span<const double>{result.all_ns});

    return result;
}

} // namespace ctdp::bench

#endif // CTDP_BENCH_MEASUREMENT_KERNEL_H
