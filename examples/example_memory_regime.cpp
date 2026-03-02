// ═══════════════════════════════════════════════════════════════════
// ctdp-calibrator Example 2: Memory Regime Characterisation
// ═══════════════════════════════════════════════════════════════════
//
// Measures array traversal latency across working set sizes that
// span the cache hierarchy: L1 → L2 → L3 → DRAM.
//
// This is the foundation of platform-aware optimisation: knowing
// where the cache boundaries actually fall on *this* hardware lets
// the compile-time DP select tile sizes, blocking factors, and data
// layouts that stay cache-resident.
//
// The output shows the characteristic "staircase" pattern where
// latency jumps at each cache boundary.
//
// Build:
//   g++ -std=c++20 -O2 -I include -I examples examples/example_memory_regime.cpp -o memory_regime
//
// Run:
//   ./memory_regime

#include "calibration_scenarios.h"

#include <ctdp/bench/environment.h>
#include <ctdp/bench/measurement_kernel.h>
#include <ctdp/bench/metric.h>
#include <ctdp/bench/statistics.h>

#include <ctdp/calibrator/calibration_dataset.h>
#include <ctdp/calibrator/calibration_profile.h>
#include <ctdp/calibrator/cost_model.h>
#include <ctdp/calibrator/dispatch_table.h>
#include <ctdp/calibrator/plan.h>
#include <ctdp/calibrator/plan_builder.h>
#include <ctdp/calibrator/provenance.h>
#include <ctdp/calibrator/solver.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ex  = ctdp::calibrator::examples;
namespace cal = ctdp::calibrator;
namespace bench = ctdp::bench;

using null_snap = bench::null_metric::null_snapshot;

struct memory_space {
    using point_type = ex::memory_point;
};

// ─── Measure a scenario ──────────────────────────────────────────

template <cal::Scenario S>
auto measure_scenario(S& scenario, std::size_t reps = 15)
    -> std::vector<cal::data_point<typename S::point_type, null_snap>>
{
    bench::null_metric nm;

    using dp_t = cal::data_point<typename S::point_type, null_snap>;
    std::vector<dp_t> results;
    results.reserve(scenario.points().size());

    for (auto const& pt : scenario.points()) {
        scenario.prepare(pt);
        auto fn = [&]() -> bench::result_token {
            return scenario.execute(pt);
        };
        auto meas = bench::measure_repeated(fn, []{}, nm, reps, 30, 3);
        results.push_back(dp_t{
            .space_point    = pt,
            .median_ns      = meas.median_ns,
            .mad_ns         = meas.mad_ns,
            .raw_timings    = std::move(meas.all_ns),
            .raw_snapshots  = std::move(meas.all_snapshots),
            .env            = bench::capture_environment()
        });
    }
    return results;
}

// ─── Format bytes for display ────────────────────────────────────

std::string format_bytes(std::size_t bytes) {
    if (bytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f MiB",
            static_cast<double>(bytes) / (1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f KiB",
            static_cast<double>(bytes) / 1024.0);
        return buf;
    }
    return std::to_string(bytes) + " B";
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  ctdp-calibrator: Memory Regime Characterisation    ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ────────────────────────────────────────────────────────────
    // Step 1: Define the scenario
    // ────────────────────────────────────────────────────────────
    //
    // Working sets from 4 KiB to 64 MiB, logarithmically spaced.
    // This spans: L1 (~48 KiB), L2 (~1 MiB), L3 (~36 MiB), DRAM.

    std::printf("Step 1: Defining memory regime scenario\n");
    ex::memory_regime_scenario scenario(4, 65536, 20);

    std::printf("  Working sets: %zu sizes from %s to %s\n",
        scenario.points().size(),
        format_bytes(scenario.points().front().bytes).c_str(),
        format_bytes(scenario.points().back().bytes).c_str());

    auto env = bench::capture_environment();
    // L1d size not in environment_context; use typical value
    constexpr std::size_t l1d_estimate = 48 * 1024;  // 48 KiB typical
    std::printf("  Platform: L1d≈%s (estimated), LLC=%s\n\n",
        format_bytes(l1d_estimate).c_str(),
        format_bytes(env.llc_bytes).c_str());

    // ────────────────────────────────────────────────────────────
    // Step 2: Measure
    // ────────────────────────────────────────────────────────────

    std::printf("Step 2: Measuring traversal latency (15 reps)...\n");
    auto data = measure_scenario(scenario, 15);
    std::printf("  Done: %zu measurements\n\n", data.size());

    // ────────────────────────────────────────────────────────────
    // Step 3: Dataset + profile + plan
    // ────────────────────────────────────────────────────────────

    auto prov = cal::capture_provenance(scenario, env, 15);
    auto dataset = cal::make_dataset<
        memory_space, ex::memory_traverse_tag, null_snap>(
            std::move(data), prov);
    auto profile = cal::fit_lookup_profile(dataset);

    // Extract points for plan building
    std::vector<ex::memory_point> all_pts;
    all_pts.reserve(dataset.points.size());
    for (auto const& dp : dataset.points) {
        all_pts.push_back(dp.space_point);
    }

    auto plan = cal::build_plan<memory_space, ex::memory_traverse_tag>(
        profile, all_pts);

    std::printf("Step 3: Profile → Plan built (%zu entries)\n\n",
        plan.size());

    // ────────────────────────────────────────────────────────────
    // Step 4: Display the latency staircase
    // ────────────────────────────────────────────────────────────

    std::printf("Step 4: Latency staircase\n\n");
    std::printf("  %-10s  %8s  %6s  %s\n",
        "Size", "Lat (ns)", "MAD", "Regime");
    std::printf("  %-10s  %8s  %6s  %s\n",
        "──────────", "────────", "──────", "──────────────────────");

    // Compute per-element latency (ns per cache line traversed)
    for (auto const& e : plan.entries) {
        auto const& pt = e.point;
        auto lines = pt.bytes / static_cast<std::size_t>(pt.stride);
        if (lines == 0) lines = 1;
        double per_line = e.cost_ns / static_cast<double>(lines);

        // Classify regime
        std::string regime;
        if (pt.bytes <= l1d_estimate) {
            regime = "L1d";
        } else if (pt.bytes <= l1d_estimate * 20) {
            // Rough L2 estimate (actual L2 size varies)
            regime = "L2";
        } else if (pt.bytes <= env.llc_bytes) {
            regime = "L3/LLC";
        } else {
            regime = "DRAM";
        }

        // ASCII bar chart
        int bar_len = static_cast<int>(per_line * 3.0);
        if (bar_len > 50) bar_len = 50;
        if (bar_len < 1)  bar_len = 1;
        std::string bar(static_cast<std::size_t>(bar_len), '#');

        std::printf("  %-10s  %8.1f  %6.1f  %-8s %s\n",
            format_bytes(pt.bytes).c_str(),
            e.cost_ns,
            0.0,  // MAD not stored in plan entries
            regime.c_str(),
            bar.c_str());
    }
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 5: Optimal working set (lowest total latency)
    // ────────────────────────────────────────────────────────────

    auto table = cal::make_dispatch_table(plan);

    std::printf("Step 5: Dispatch table\n");
    std::printf("  %s\n", table.summary().c_str());

    table.dispatch([](ex::memory_point const& pt, double cost) {
        std::printf("  → Fastest working set: %s at %.1f ns\n",
            format_bytes(pt.bytes).c_str(), cost);
    });
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 6: Regime boundary detection
    // ────────────────────────────────────────────────────────────
    //
    // Find the largest working set that stays within ~2x of the
    // L1 latency.  This is the effective "tile size limit" for
    // compile-time blocking decisions.

    std::printf("Step 6: Regime boundary detection\n");

    double baseline_ns = plan.entries.front().cost_ns;
    auto baseline_lines = plan.entries.front().point.bytes
        / static_cast<std::size_t>(plan.entries.front().point.stride);
    if (baseline_lines == 0) baseline_lines = 1;
    double baseline_per_line = baseline_ns / static_cast<double>(baseline_lines);

    std::size_t l1_limit = 0, l2_limit = 0;
    for (auto const& e : plan.entries) {
        auto lines = e.point.bytes / static_cast<std::size_t>(e.point.stride);
        if (lines == 0) lines = 1;
        double per_line = e.cost_ns / static_cast<double>(lines);

        if (per_line < baseline_per_line * 2.0) {
            l1_limit = e.point.bytes;
        }
        if (per_line < baseline_per_line * 5.0) {
            l2_limit = e.point.bytes;
        }
    }

    std::printf("  Baseline per-line: %.2f ns/line\n", baseline_per_line);
    std::printf("  Effective L1 range: up to %s (<2x baseline)\n",
        format_bytes(l1_limit).c_str());
    std::printf("  Effective L2 range: up to %s (<5x baseline)\n",
        format_bytes(l2_limit).c_str());
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 7: Constrained plan — "L1-resident only"
    // ────────────────────────────────────────────────────────────

    auto l1_plan = cal::build_plan<memory_space, ex::memory_traverse_tag>(
        profile, all_pts,
        [&](ex::memory_point const& pt) {
            return pt.bytes <= l1d_estimate;
        });

    std::printf("Step 7: L1-resident configurations\n");
    std::printf("  Entries: %zu (out of %zu total)\n",
        l1_plan.size(), plan.size());
    if (!l1_plan.empty()) {
        std::printf("  Largest L1-resident: %s\n",
            format_bytes(l1_plan.entries.back().point.bytes).c_str());
    }
    std::printf("\n");

    // ────────────────────────────────────────────────────────────

    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  The latency staircase shows where cache boundaries ║\n");
    std::printf("║  fall on THIS hardware.  Tile sizes chosen by       ║\n");
    std::printf("║  compile-time DP should target the L1 or L2 range   ║\n");
    std::printf("║  to stay cache-resident.                            ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    return 0;
}
