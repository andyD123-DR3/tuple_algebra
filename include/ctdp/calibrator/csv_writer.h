#ifndef CTDP_CALIBRATOR_CSV_WRITER_H
#define CTDP_CALIBRATOR_CSV_WRITER_H

// ctdp::calibrator::csv_writer — Serialise data_points to CSV
//
// Separated from the Metric concept (design v2.2 §5).
// Uses pluggable formatters:
//   counter_snapshot_formatter — formats bench::counter_snapshot columns
//   null_snapshot_formatter    — no-op for null_metric
//
// write_csv() is parameterised on SpacePoint + Formatter.
// The SpacePoint is serialised via a PointFormatter concept
// (user-provided, specific to each scenario).

#include "data_point.h"

#include <ctdp/bench/perf_counter.h>
#include <ctdp/bench/metric.h>

#include <concepts>
#include <iomanip>
#include <ostream>
#include <span>
#include <sstream>
#include <string>

namespace ctdp::calibrator {

// ─── Formatter concepts ─────────────────────────────────────────────

/// A SnapshotFormatter provides CSV header and row formatting for
/// a metric's snapshot type.
template <typename F, typename Snapshot>
concept SnapshotFormatter = requires(F const& f, Snapshot const& snap) {
    { f.csv_header() } -> std::convertible_to<std::string>;
    { f.to_csv(snap) } -> std::convertible_to<std::string>;
};

/// A PointFormatter provides CSV header and row formatting for
/// a scenario's point type.
template <typename F, typename Point>
concept PointFormatter = requires(F const& f, Point const& pt) {
    { f.csv_header() } -> std::convertible_to<std::string>;
    { f.to_csv(pt)   } -> std::convertible_to<std::string>;
};

// ─── counter_snapshot_formatter ─────────────────────────────────────

/// Formats bench::counter_snapshot as CSV columns.
struct counter_snapshot_formatter {
    [[nodiscard]] static auto csv_header() -> std::string {
        return "tsc_cycles,thread_cpu_ns,instructions,cache_refs,cache_misses,"
               "branches,branch_misses,ipc,cache_miss_rate,branch_miss_rate,"
               "cycles_per_ns,insns_per_ns,cache_refs_per_ns,branches_per_ns";
    }

    [[nodiscard]] static auto to_csv(bench::counter_snapshot const& snap)
        -> std::string
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(4);
        os << snap.tsc_cycles << ","
           << snap.thread_cpu_ns << ","
           << snap.instructions << ","
           << snap.cache_references << ","
           << snap.cache_misses << ","
           << snap.branches << ","
           << snap.branch_misses << ","
           << snap.ipc << ","
           << snap.cache_miss_rate << ","
           << snap.branch_miss_rate << ","
           << snap.cycles_per_ns << ","
           << snap.instructions_per_ns << ","
           << snap.cache_refs_per_ns << ","
           << snap.branches_per_ns;
        return os.str();
    }
};

/// No-op formatter for null_metric::null_snapshot
struct null_snapshot_formatter {
    [[nodiscard]] static auto csv_header() -> std::string { return ""; }
    [[nodiscard]] static auto to_csv(bench::null_metric::null_snapshot const&)
        -> std::string { return ""; }
};

// Verify formatter concepts
static_assert(SnapshotFormatter<counter_snapshot_formatter, bench::counter_snapshot>);
static_assert(SnapshotFormatter<null_snapshot_formatter, bench::null_metric::null_snapshot>);

// ─── write_csv ──────────────────────────────────────────────────────

/// Write a vector of data_points as CSV.
///
/// Column layout:
///   [point columns] , median_ns , mad_ns , rep , wall_ns , [snapshot columns]
///
/// Each data_point expands to `reps` rows (one per rep), plus the
/// point-level median/mad columns repeated for convenience.
///
/// @tparam SpacePoint       The scenario's point type
/// @tparam MetricSnapshot   The metric's snapshot type
/// @tparam PtFmt            PointFormatter for SpacePoint
/// @tparam SnapFmt          SnapshotFormatter for MetricSnapshot
template <typename SpacePoint,
          typename MetricSnapshot,
          PointFormatter<SpacePoint> PtFmt,
          SnapshotFormatter<MetricSnapshot> SnapFmt>
void write_csv(
    std::ostream& os,
    std::span<const data_point<SpacePoint, MetricSnapshot>> points,
    PtFmt const& pt_fmt,
    SnapFmt const& snap_fmt)
{
    // Header
    os << pt_fmt.csv_header()
       << ",median_ns,mad_ns,rep,wall_ns";
    auto snap_header = snap_fmt.csv_header();
    if (!snap_header.empty()) {
        os << "," << snap_header;
    }
    os << "\n";

    // Rows: one per (point, rep) pair
    for (auto const& dp : points) {
        auto pt_csv = pt_fmt.to_csv(dp.space_point);

        for (std::size_t r = 0; r < dp.reps(); ++r) {
            os << pt_csv
               << "," << std::fixed << std::setprecision(2)
               << dp.median_ns
               << "," << dp.mad_ns
               << "," << r
               << "," << dp.raw_timings[r];

            auto snap_csv = snap_fmt.to_csv(dp.raw_snapshots[r]);
            if (!snap_csv.empty()) {
                os << "," << snap_csv;
            }
            os << "\n";
        }
    }
}

/// Convenience overload: writes summary rows (one per point, median only).
/// No per-rep expansion.
template <typename SpacePoint,
          typename MetricSnapshot,
          PointFormatter<SpacePoint> PtFmt,
          SnapshotFormatter<MetricSnapshot> SnapFmt>
void write_csv_summary(
    std::ostream& os,
    std::span<const data_point<SpacePoint, MetricSnapshot>> points,
    PtFmt const& pt_fmt,
    SnapFmt const& snap_fmt)
{
    // Header
    os << pt_fmt.csv_header()
       << ",median_ns,mad_ns,reps,relative_mad";
    auto snap_header = snap_fmt.csv_header();
    if (!snap_header.empty()) {
        // Use first rep's snapshot as representative
        os << "," << snap_header;
    }
    os << "\n";

    // One row per point
    for (auto const& dp : points) {
        os << pt_fmt.to_csv(dp.space_point)
           << "," << std::fixed << std::setprecision(2)
           << dp.median_ns
           << "," << dp.mad_ns
           << "," << dp.reps()
           << "," << std::setprecision(4) << dp.relative_mad();

        if (!snap_header.empty() && !dp.raw_snapshots.empty()) {
            os << "," << snap_fmt.to_csv(dp.raw_snapshots[0]);
        }
        os << "\n";
    }
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_CSV_WRITER_H
