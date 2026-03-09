// ═══════════════════════════════════════════════════════════════════
// ctdp-calibrator Example 1: Integer Parser Strategy Selection
// ═══════════════════════════════════════════════════════════════════
//
// Demonstrates the full calibration pipeline:
//
//   1. Define a Scenario (parser strategies × digit counts)
//   2. Measure each configuration via the calibration harness
//   3. Build a typed dataset with provenance
//   4. Fit a calibration profile (lookup model)
//   5. Build an optimisation plan (exhaustive search)
//   6. Create a dispatch table for O(1) optimal lookup
//   7. Export the plan as a wisdom file (FFTW-style)
//   8. Emit a constexpr C++ header for compile-time use
//   9. Validate the plan against fresh measurements
//
// Build:
//   g++ -std=c++20 -O2 -I include -I examples examples/example_parser_pipeline.cpp -o parser_pipeline
//
// Run:
//   ./parser_pipeline

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
#include <ctdp/calibrator/plan_emit.h>
#include <ctdp/calibrator/plan_validate.h>
#include <ctdp/calibrator/provenance.h>
#include <ctdp/calibrator/solver.h>
#include <ctdp/calibrator/wisdom.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ex  = ctdp::calibrator::examples;
namespace cal = ctdp::calibrator;
namespace bench = ctdp::bench;

using null_snap = bench::null_metric::null_snapshot;

// ─── Space wrapper (provides point_type) ─────────────────────────

struct parser_space {
    using point_type = ex::parser_point;
};

// ─── Point serialiser for wisdom/emit ────────────────────────────

struct parser_serialiser {
    std::string operator()(ex::parser_point const& pt) const {
        return "{" + std::to_string(pt.digits) + ", "
               + std::string(ex::strategy_name(pt.strategy)) + "}";
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Helper: measure a scenario → data_points
// ═══════════════════════════════════════════════════════════════════

template <cal::Scenario S>
auto measure_scenario(S& scenario, std::size_t reps = 10)
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
        auto meas = bench::measure_repeated(fn, []{}, nm, reps, 50, 5);
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

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  ctdp-calibrator: Integer Parser Strategy Example   ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ────────────────────────────────────────────────────────────
    // Step 1: Define the scenario
    // ────────────────────────────────────────────────────────────
    //
    // A parser_strategy_scenario enumerates every combination of
    // digit count (1..8) × strategy (generic, loop, swar, unrolled).
    // This is the "space" that the calibrator will explore.

    std::printf("Step 1: Creating scenario (digits 1-8 × 4 strategies)\n");
    ex::parser_strategy_scenario scenario(1, 8);

    std::printf("  Points in space: %zu\n\n",
                scenario.points().size());

    // ────────────────────────────────────────────────────────────
    // Step 2: Measure every configuration
    // ────────────────────────────────────────────────────────────
    //
    // The harness runs each (digits, strategy) pair multiple times,
    // collecting wall-clock timings with anti-elision barriers.

    std::printf("Step 2: Measuring all configurations (10 reps each)...\n");
    auto data = measure_scenario(scenario, 10);
    std::printf("  Collected %zu data points\n\n", data.size());

    // ────────────────────────────────────────────────────────────
    // Step 3: Build a typed dataset
    // ────────────────────────────────────────────────────────────
    //
    // The dataset carries type parameters <Space, Callable> that
    // prevent data from one kernel being fed to another's cost model.

    auto env  = bench::capture_environment();
    auto prov = cal::capture_provenance(scenario, env, 10);

    auto dataset = cal::make_dataset<
        parser_space,            // The search space
        ex::integer_parser_tag,  // The kernel identity (callable_type)
        null_snap                // Metric snapshot type
    >(std::move(data), prov);

    std::printf("Step 3: Dataset built\n");
    std::printf("  Space:    parser_space (digits × strategy)\n");
    std::printf("  Callable: integer_parser_tag\n");
    std::printf("  Points:   %zu\n\n", dataset.size());

    // ────────────────────────────────────────────────────────────
    // Step 4: Fit a calibration profile
    // ────────────────────────────────────────────────────────────
    //
    // A lookup profile stores the measured median_ns for each point.
    // It's the empirical cost surface of this kernel on this hardware.

    auto profile = cal::fit_lookup_profile(dataset);

    std::printf("Step 4: Profile fitted (lookup model)\n");
    std::printf("  Training points: %zu\n\n", profile.training_points);

    // ────────────────────────────────────────────────────────────
    // Step 5: Build an optimisation plan
    // ────────────────────────────────────────────────────────────
    //
    // The solver evaluates every point through the cost model and
    // produces a plan sorted by cost.  build_plan() composes the
    // cost model + exhaustive search in one call.

    auto plan = cal::build_plan<parser_space, ex::integer_parser_tag>(
        profile, scenario);

    std::printf("Step 5: Plan built (exhaustive search)\n");
    std::printf("  Entries: %zu\n", plan.size());
    std::printf("  Optimal: %.2f ns — %d digit(s), %s\n\n",
        plan.optimal_cost(),
        plan.optimal_point().digits,
        std::string(ex::strategy_name(plan.optimal_point().strategy)).c_str());

    // ────────────────────────────────────────────────────────────
    // Step 6: Top-10 configurations
    // ────────────────────────────────────────────────────────────

    std::printf("Step 6: Top-10 configurations\n");
    std::printf("  %-4s  %-10s  %10s\n", "Rank", "Config", "Cost (ns)");
    std::printf("  %-4s  %-10s  %10s\n", "────", "──────────", "──────────");

    auto top10 = plan.top_n(10);
    for (std::size_t i = 0; i < top10.size(); ++i) {
        auto const& e = top10[i];
        char config[32];
        std::snprintf(config, sizeof(config), "%d/%s",
            e.point.digits,
            std::string(ex::strategy_name(e.point.strategy)).c_str());
        std::printf("  #%-3zu  %-10s  %10.2f\n", i, config, e.cost_ns);
    }
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 7: Dispatch table
    // ────────────────────────────────────────────────────────────
    //
    // The dispatch table is the "manufacturing die" — the frozen
    // optimisation result ready for O(1) lookup at runtime.

    auto table = cal::make_dispatch_table(plan);

    std::printf("Step 7: Dispatch table built\n");
    std::printf("  %s\n\n", table.summary().c_str());

    // Demonstrate dispatch: call a visitor with the optimal point
    table.dispatch([](ex::parser_point const& pt, double cost) {
        std::printf("  → Dispatching to optimal: %d digits, %s (%.2f ns)\n",
            pt.digits,
            std::string(ex::strategy_name(pt.strategy)).c_str(),
            cost);
    });
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 8: Strategy comparison per digit count
    // ────────────────────────────────────────────────────────────

    std::printf("Step 8: Strategy comparison by digit count\n");
    std::printf("  %-6s  %8s  %8s  %8s  %8s  | %-10s\n",
        "Digits", "generic", "loop", "swar", "unrolled", "Winner");
    std::printf("  %-6s  %8s  %8s  %8s  %8s  | %-10s\n",
        "──────", "────────", "────────", "────────", "────────", "──────────");

    for (int d = 1; d <= 8; ++d) {
        double costs[4];
        ex::parse_strategy strategies[] = {
            ex::parse_strategy::generic, ex::parse_strategy::loop,
            ex::parse_strategy::swar, ex::parse_strategy::unrolled
        };

        int best_idx = 0;
        for (int s = 0; s < 4; ++s) {
            costs[s] = table.predict({d, strategies[s]});
            if (costs[s] < costs[best_idx]) best_idx = s;
        }

        std::printf("  %-6d  %8.2f  %8.2f  %8.2f  %8.2f  | %-10s",
            d, costs[0], costs[1], costs[2], costs[3],
            std::string(ex::strategy_name(strategies[best_idx])).c_str());

        // Mark the winner
        if (d <= 3) std::printf("  ← small digits");
        if (d >= 7) std::printf("  ← large digits");
        std::printf("\n");
    }
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 9: Export wisdom file
    // ────────────────────────────────────────────────────────────
    //
    // The wisdom file is a human-readable serialisation of the plan.
    // It can be saved, version-controlled, and reloaded on the next
    // build — avoiding the need to re-calibrate every time.

    std::printf("Step 9: Exporting wisdom file\n");

    std::ostringstream wisdom_out;
    cal::write_wisdom(wisdom_out, plan, parser_serialiser{});
    auto wisdom_text = wisdom_out.str();

    // Show first few lines
    std::istringstream preview(wisdom_text);
    std::string line;
    int line_count = 0;
    while (std::getline(preview, line) && line_count < 12) {
        std::printf("  %s\n", line.c_str());
        ++line_count;
    }
    if (plan.size() > 5) {
        std::printf("  ... (%zu more entries)\n", plan.size() - 5);
    }
    std::printf("\n");

    // ────────────────────────────────────────────────────────────
    // Step 10: Emit constexpr header
    // ────────────────────────────────────────────────────────────

    std::printf("Step 10: Emitting constexpr header (first 15 lines)\n");

    std::ostringstream header_out;
    cal::emit_plan_header(header_out, plan,
        "parser_plan", "parser_space", "integer_parser_tag",
        parser_serialiser{});

    std::istringstream header_preview(header_out.str());
    line_count = 0;
    while (std::getline(header_preview, line) && line_count < 15) {
        std::printf("  %s\n", line.c_str());
        ++line_count;
    }
    std::printf("  ...\n\n");

    // ────────────────────────────────────────────────────────────
    // Step 11: Validate plan against fresh measurements
    // ────────────────────────────────────────────────────────────

    std::printf("Step 11: Validating plan (re-measuring)...\n");

    cal::validation_config vcfg;
    vcfg.reps      = 5;
    vcfg.tolerance = 0.50;  // 50% tolerance (containers are noisy)
    vcfg.flush_cache = false;

    auto vr = cal::validate_profile<parser_space, ex::integer_parser_tag>(
        profile, scenario, vcfg);

    std::printf("  Points validated: %zu\n", vr.total_points);
    std::printf("  Pass rate:        %.0f%%\n", vr.pass_rate() * 100.0);
    std::printf("  Max rel. error:   %.1f%%\n", vr.max_relative_error * 100.0);
    std::printf("  Mean rel. error:  %.1f%%\n\n", vr.mean_relative_error * 100.0);

    // ────────────────────────────────────────────────────────────
    // Step 12: Constrained plan — "best swar-only for each digit"
    // ────────────────────────────────────────────────────────────

    std::printf("Step 12: Constrained plan (SWAR-only)\n");

    auto swar_plan = cal::build_plan<parser_space, ex::integer_parser_tag>(
        profile, scenario,
        [](ex::parser_point const& pt) {
            return pt.strategy == ex::parse_strategy::swar;
        });

    auto swar_table = cal::make_dispatch_table(swar_plan);

    std::printf("  SWAR entries: %zu\n", swar_plan.size());
    std::printf("  Best SWAR: %d digit(s) at %.2f ns\n",
        swar_table.optimal_point().digits,
        swar_table.optimal_cost());
    std::printf("  vs overall best: %.2f ns (%.1f%% overhead)\n\n",
        table.optimal_cost(),
        (swar_table.optimal_cost() / table.optimal_cost() - 1.0) * 100.0);

    // ────────────────────────────────────────────────────────────

    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  Pipeline complete.                                 ║\n");
    std::printf("║                                                     ║\n");
    std::printf("║  The dispatch table is the frozen result of         ║\n");
    std::printf("║  calibration — the \"manufacturing die\" that         ║\n");
    std::printf("║  delivers optimal code at zero runtime cost.        ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    return 0;
}
