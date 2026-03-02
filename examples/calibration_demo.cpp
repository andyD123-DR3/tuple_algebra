// ctdp-calibrator end-to-end demo
//
// Demonstrates the complete calibration pipeline:
//
//   1. Define a Scenario (parser strategy space)
//   2. Measure all points via calibration_harness
//   3. Build a typed Dataset with provenance
//   4. Fit a lookup Profile (cost model)
//   5. Run exhaustive solver → Plan
//   6. Build a DispatchTable (runtime dispatch)
//   7. Emit constexpr header (compile-time wisdom)
//   8. Write FFTW-style wisdom file
//   9. Round-trip: read wisdom → reconstruct plan → verify
//
// Build:
//   g++ -std=c++20 -O2 -I include -I examples calibration_demo.cpp -o demo
//
// Run:
//   ./demo

#include "calibration_scenarios.h"

#include <ctdp/bench.h>
#include <ctdp/calibrator.h>
#include <ctdp/calibrator/dispatch_table.h>
#include <ctdp/calibrator/plan_emit.h>
#include <ctdp/calibrator/wisdom.h>

#include <cstdio>
#include <optional>
#include <sstream>
#include <string>

namespace ex  = ctdp::calibrator::examples;
namespace cal = ctdp::calibrator;
namespace bench = ctdp::bench;

using null_snap = bench::null_metric::null_snapshot;

// ─── Space definition ────────────────────────────────────────────

struct parser_space {
    using point_type = ex::parser_point;
};

// ─── Point serialiser ────────────────────────────────────────────

struct parser_serialiser {
    static const char* strategy_name(ex::parse_strategy s) {
        switch (s) {
            case ex::parse_strategy::generic:  return "generic";
            case ex::parse_strategy::loop:     return "loop";
            case ex::parse_strategy::swar:     return "swar";
            case ex::parse_strategy::unrolled: return "unrolled";
        }
        return "?";
    }

    std::string operator()(ex::parser_point const& pt) const {
        return "{" + std::to_string(pt.digits) + ","
               + std::to_string(static_cast<int>(pt.strategy)) + "}";
    }

    static std::string label(ex::parser_point const& pt) {
        return std::to_string(pt.digits) + "d_"
               + strategy_name(pt.strategy);
    }
};

// ─── Point parser (for wisdom round-trip) ────────────────────────

auto parse_point(std::string const& s)
    -> std::optional<ex::parser_point>
{
    if (s.size() < 3 || s.front() != '{') return std::nullopt;
    auto inner = s.substr(1, s.find('}') - 1);
    auto comma = inner.find(',');
    if (comma == std::string::npos) return std::nullopt;
    return ex::parser_point{
        std::stoi(inner.substr(0, comma)),
        static_cast<ex::parse_strategy>(std::stoi(inner.substr(comma + 1)))
    };
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("╔═══════════════════════════════════════════════════════╗\n");
    std::printf("║  ctdp-calibrator end-to-end demo                     ║\n");
    std::printf("╚═══════════════════════════════════════════════════════╝\n\n");

    // ── 1. Scenario ──────────────────────────────────────────────
    std::printf("1. Creating scenario: parser_strategy (1..4 digits × 4 strategies)\n");
    ex::parser_strategy_scenario scenario(1, 4);
    std::printf("   → %zu space points\n\n",
                scenario.points().size());

    // ── 2. Measure ───────────────────────────────────────────────
    std::printf("2. Measuring all points (5 reps each)...\n");
    bench::null_metric nm;
    using dp_t = cal::data_point<ex::parser_point, null_snap>;
    std::vector<dp_t> data;
    data.reserve(scenario.points().size());

    for (auto const& pt : scenario.points()) {
        scenario.prepare(pt);
        auto fn = [&]() -> bench::result_token {
            return scenario.execute(pt);
        };
        auto meas = bench::measure_repeated(fn, []{}, nm, 5, 20, 1);
        data.push_back(dp_t{
            .space_point   = pt,
            .median_ns     = meas.median_ns,
            .mad_ns        = meas.mad_ns,
            .raw_timings   = std::move(meas.all_ns),
            .raw_snapshots = std::move(meas.all_snapshots),
            .env           = bench::capture_environment()
        });
    }
    std::printf("   → %zu data points collected\n\n", data.size());

    // ── 3. Dataset ───────────────────────────────────────────────
    std::printf("3. Building typed dataset with provenance\n");
    auto env = bench::capture_environment();
    auto prov = cal::capture_provenance(scenario, env, 5);
    auto ds = cal::make_dataset<parser_space, ex::integer_parser_tag, null_snap>(
        std::move(data), prov);
    std::printf("   → dataset<%s, %s>: %zu points\n\n",
                "parser_space", "integer_parser_tag", ds.size());

    // ── 4. Profile ───────────────────────────────────────────────
    std::printf("4. Fitting lookup profile\n");
    auto profile = cal::fit_lookup_profile(ds);
    std::printf("   → %zu training points, model=lookup\n\n",
                profile.training_points);

    // ── 5. Solver → Plan ─────────────────────────────────────────
    std::printf("5. Running exhaustive solver\n");
    auto plan = cal::build_plan<parser_space, ex::integer_parser_tag>(
        profile, scenario);
    std::printf("   → plan: %zu entries, optimal=%.2f ns\n",
                plan.size(), plan.optimal_cost());
    auto const& best = plan.optimal_point();
    std::printf("   → optimal: digits=%d, strategy=%s\n\n",
                best.digits,
                parser_serialiser::strategy_name(best.strategy));

    // ── 6. Dispatch table ────────────────────────────────────────
    std::printf("6. Building dispatch table\n");
    auto table = cal::make_dispatch_table(plan);
    std::printf("   → %s\n\n", table.summary().c_str());

    // Top 5
    std::printf("   Top 5 configurations:\n");
    auto top5 = table.top_n(5);
    for (auto const& e : top5) {
        std::printf("     rank %zu: %s → %.2f ns%s\n",
                    e.rank,
                    parser_serialiser::label(e.point).c_str(),
                    e.cost_ns,
                    e.pareto_optimal ? " [Pareto]" : "");
    }
    std::printf("\n");

    // Dispatch optimal
    std::printf("   Dispatching optimal configuration:\n");
    table.dispatch([](ex::parser_point const& pt, double cost) {
        std::printf("     → execute(%dd, %s) at predicted %.2f ns\n",
                    pt.digits,
                    parser_serialiser::strategy_name(pt.strategy),
                    cost);
    });
    std::printf("\n");

    // ── 7. Emit constexpr header ─────────────────────────────────
    std::printf("7. Emitting constexpr header\n");
    std::ostringstream header_out;
    cal::emit_plan_header(header_out, plan,
        "parser_wisdom", "parser_space", "integer_parser_tag",
        parser_serialiser{});
    auto header_code = header_out.str();
    std::printf("   → %zu bytes of constexpr C++\n",
                header_code.size());

    // Show first few lines
    std::istringstream preview(header_code);
    std::string line;
    int shown = 0;
    while (std::getline(preview, line) && shown < 8) {
        std::printf("   │ %s\n", line.c_str());
        ++shown;
    }
    std::printf("   │ ...\n\n");

    // ── 8. Write wisdom ──────────────────────────────────────────
    std::printf("8. Writing wisdom file\n");
    std::ostringstream wisdom_out;
    cal::write_wisdom(wisdom_out, plan, parser_serialiser{});
    auto wisdom_text = wisdom_out.str();
    std::printf("   → %zu bytes of wisdom data\n",
                wisdom_text.size());

    // Show header
    std::istringstream wpreview(wisdom_text);
    shown = 0;
    while (std::getline(wpreview, line) && shown < 12) {
        std::printf("   │ %s\n", line.c_str());
        ++shown;
    }
    std::printf("   │ ...\n\n");

    // ── 9. Round-trip ────────────────────────────────────────────
    std::printf("9. Wisdom round-trip: read → reconstruct → verify\n");
    auto wf = cal::read_wisdom_string(wisdom_text);
    std::printf("   → parsed: version=%d, solver=%s, %zu entries\n",
                wf.metadata.version,
                wf.metadata.solver.c_str(),
                wf.size());

    auto restored = cal::reconstruct_plan<parser_space, ex::integer_parser_tag>(
        wf, [](std::string const& s) { return parse_point(s); });
    std::printf("   → reconstructed plan: %zu entries\n",
                restored.entries.size());

    // Verify costs match
    bool match = true;
    for (std::size_t i = 0; i < plan.entries.size(); ++i) {
        double diff = std::abs(plan.entries[i].cost_ns
                               - restored.entries[i].cost_ns);
        if (diff > 0.01) { match = false; break; }
    }
    std::printf("   → costs match: %s\n\n", match ? "YES ✓" : "NO ✗");

    // ── 10. Validate ─────────────────────────────────────────────
    std::printf("10. Validating plan predictions against fresh measurements\n");
    cal::validation_config vcfg;
    vcfg.reps = 3;
    vcfg.tolerance = 0.50;
    vcfg.flush_cache = false;
    auto vr = cal::validate_profile<parser_space, ex::integer_parser_tag>(
        profile, scenario, vcfg);
    std::printf("    → %zu/%zu passed (%.0f%%) at %.0f%% tolerance\n\n",
                vr.points_within_tol, vr.total_points,
                vr.pass_rate() * 100.0, vcfg.tolerance * 100.0);

    // ── Summary ──────────────────────────────────────────────────
    std::printf("╔═══════════════════════════════════════════════════════╗\n");
    std::printf("║  Pipeline complete                                    ║\n");
    std::printf("║                                                       ║\n");
    std::printf("║  Scenario → Harness → Dataset → Profile → Solver →   ║\n");
    std::printf("║  Plan → DispatchTable → Emit → Wisdom → Round-trip   ║\n");
    std::printf("║                                                       ║\n");
    std::printf("║  %2zu space points measured                             ║\n",
                ds.size());
    std::printf("║  %2zu plan entries (sorted by cost)                    ║\n",
                plan.size());
    std::printf("║  Optimal: %dd %s at %.2f ns         ║\n",
                best.digits,
                parser_serialiser::strategy_name(best.strategy),
                plan.optimal_cost());
    std::printf("║  Wisdom round-trip: %s                            ║\n",
                match ? "verified ✓" : "FAILED ✗  ");
    std::printf("╚═══════════════════════════════════════════════════════╝\n");

    return 0;
}
