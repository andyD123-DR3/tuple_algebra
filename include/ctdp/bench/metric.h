#ifndef CTDP_BENCH_METRIC_H
#define CTDP_BENCH_METRIC_H

// ctdp::bench::metric — Metric concept and implementations
//
// Metric concept: start/stop/snapshot — no CSV, no derived computation.
// counter_metric: wraps perf_counter_group, snapshot returns counter_snapshot.
// null_metric:    zero-cost stub for timing-only benchmarks.

#include "perf_counter.h"

#include <concepts>

namespace ctdp::bench {

// ─── Metric concept ─────────────────────────────────────────────────

/// A Metric is anything that can bracket a measurement interval
/// and produce a snapshot of what happened.
///
/// Design note: compute_derived() is NOT part of the concept.
/// measurement_kernel calls it via `if constexpr` after harvesting,
/// which keeps the concept clean and the Metric implementations simple.
template <typename M>
concept Metric = requires(M& m) {
    typename M::snapshot_type;
    { m.start()    } -> std::same_as<void>;
    { m.stop()     } -> std::same_as<void>;
    { m.snapshot() } -> std::same_as<typename M::snapshot_type>;
};

// ─── counter_metric ─────────────────────────────────────────────────

/// Full hardware counter metric.
/// Wraps perf_counter_group; snapshot_type is counter_snapshot.
class counter_metric {
public:
    using snapshot_type = counter_snapshot;

    counter_metric() = default;

    void start() noexcept { group_.start(); }
    void stop()  noexcept { group_.stop();  }

    [[nodiscard]] snapshot_type snapshot() const noexcept {
        return group_.snapshot();
    }

    /// Whether Tier 1 (perf_event) counters are available
    [[nodiscard]] bool tier1_available() const noexcept {
        return group_.tier1_available();
    }

private:
    perf_counter_group group_;
};

// ─── null_metric ────────────────────────────────────────────────────

/// Zero-cost metric for timing-only benchmarks.
/// Returns an empty struct as its snapshot.
class null_metric {
public:
    struct null_snapshot {};
    using snapshot_type = null_snapshot;

    void start() noexcept {}
    void stop()  noexcept {}

    [[nodiscard]] snapshot_type snapshot() const noexcept { return {}; }
};

// Verify concepts
static_assert(Metric<counter_metric>);
static_assert(Metric<null_metric>);

} // namespace ctdp::bench

#endif // CTDP_BENCH_METRIC_H
