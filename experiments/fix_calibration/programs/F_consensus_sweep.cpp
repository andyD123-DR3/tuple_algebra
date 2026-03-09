// experiments/fix_calibration/programs/F_consensus_sweep.cpp
//
// Program F: exhaustive consensus-subspace sweep.
//
// Question: "What is the exact optimum within the consensus subspace?"
// Method:   Direct measurement of all 1,024 configs + 21-point audit.
//
// The consensus subspace is derived from Programs A–E:
//   7 field positions are fixed (unanimous or near-unanimous agreement)
//   5 field positions are uncertain (1, 6, 7, 8, 9)
//   4 strategies × 5 positions = 4^5 = 1,024 configs
//
// This is a reduced-space exact sweep, NOT a search of the full 4^12
// space.  The fixed-position audit (7 positions × 3 alternatives = 21
// one-flip variants) checks whether the consensus locking was safe.
// If any audit config beats the sweep winner, the consensus at that
// position is suspect.
//
// No SVR, no beam search, no Phase 1/Phase 2 split.
//
// Build target:
//   fix_experiment_F       Mock or real RDTSC (single executable)

#include "sweep_configs.h"
#include "baselines.h"
#include "report_types.h"
#include "output_table.h"
#include "json_report.h"

#ifdef CTDP_FIX_EXPERIMENT_MOCK
#include "mock_measurer.h"
#else
#include "rdtsc_adapter.h"
#include "compiled_measurer.h"
#include "experiment_config.h"
#endif

#include <ctdp/calibrator/fix_et_parser.h>

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

// ─────────────────────────────────────────────────────────────────────
//  Sweep result entry
// ─────────────────────────────────────────────────────────────────────

struct sweep_entry {
    std::size_t             index;
    fix::fix_config         config;
    fxe::measurement_result result;
};

// ─────────────────────────────────────────────────────────────────────
//  Audit result: one-flip variant of a consensus-fixed position
// ─────────────────────────────────────────────────────────────────────

struct audit_entry {
    int                     fixed_position;  // which consensus position was flipped
    fxe::Strategy           original;        // consensus strategy
    fxe::Strategy           alternative;     // what we flipped to
    fix::fix_config         config;
    fxe::measurement_result result;
};

// ─────────────────────────────────────────────────────────────────────
//  Build ExperimentReport from sweep results (non-mutating)
// ─────────────────────────────────────────────────────────────────────

static fxe::ExperimentReport build_sweep_report(
    const std::vector<sweep_entry>& entries,
    const std::vector<fxe::BaselineResult>& bl_results,
    std::size_t top_n)
{
    // Sort a copy by measured p50
    auto sorted = entries;
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) {
            return a.result.p50_ns < b.result.p50_ns;
        });

    fxe::ExperimentReport report;
    report.program_id         = "F";
    report.program_name       = "F_consensus_sweep";
    report.extractor_name     = "exhaustive";
    report.target_description = "direct measurement (reduced-space)";
    report.target_units       = "ns";
    report.beam_width         = top_n;
    report.beam_depth         = static_cast<std::size_t>(fxe::N_UNCERTAIN);
    report.exhaustive         = true;

    // Model metrics: not applicable
    report.model.cv_rmse    = 0.0;
    report.model.cv_r2      = std::numeric_limits<double>::quiet_NaN();
    report.model.n_training = fxe::sweep_configs.size();
    report.model.n_features = 0;
    report.model.n_folds    = 0;

    // Top-N by measured p50
    std::size_t n = std::min(sorted.size(), top_n);
    report.candidates.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        fxe::CandidateResult cr;
        cr.config_index     = sorted[i].index;
        cr.config_label     = fix::config_to_string(sorted[i].config);
        cr.predicted_target = sorted[i].result.p50_ns;
        cr.measured_p50_ns  = sorted[i].result.p50_ns;
        cr.measured_p99_ns  = sorted[i].result.p99_ns;
        cr.rank             = i + 1;
        report.candidates.push_back(std::move(cr));
    }

    report.best_measured_index = 0;
    report.baselines = bl_results;

    return report;
}

// ─────────────────────────────────────────────────────────────────────
//  Print audit results
// ─────────────────────────────────────────────────────────────────────

static void print_audit(
    const std::vector<audit_entry>& audits,
    double sweep_best_p50)
{
    std::printf("\n  Fixed-position audit (21 one-flip variants):\n");
    std::printf("  %-5s %-10s %-10s %10s %10s  %s\n",
        "Pos", "Consensus", "Flipped", "p50 (ns)", "Delta", "Verdict");
    std::printf("  %s\n", std::string(65, '-').c_str());

    int suspect_count = 0;
    for (const auto& a : audits) {
        double delta = a.result.p50_ns - sweep_best_p50;
        const char* verdict = (delta < 0.0) ? "SUSPECT" : "ok";
        if (delta < 0.0) ++suspect_count;

        std::printf("  %-5d %-10s %-10s %10.2f %+10.2f  %s\n",
            a.fixed_position,
            std::string(fix::strategy_char(a.original)).c_str(),
            std::string(fix::strategy_char(a.alternative)).c_str(),
            a.result.p50_ns,
            delta,
            verdict);
    }

    if (suspect_count == 0) {
        std::printf("\n  Audit: PASS — no one-flip variant beats sweep winner\n");
    } else {
        std::printf("\n  Audit: %d SUSPECT positions — consensus locking may lose "
            "degrees of freedom\n", suspect_count);
    }
}

// ─────────────────────────────────────────────────────────────────────

template<typename Measurer>
static int run_sweep(Measurer& measurer) {
    // ── Sweep all 1,024 configs ──
    std::vector<sweep_entry> entries;
    entries.reserve(fxe::sweep_configs.size());
    for (std::size_t i = 0; i < fxe::sweep_configs.size(); ++i) {
        auto mr = measurer.measure(fxe::sweep_configs[i]);
        entries.push_back({i, fxe::sweep_configs[i], mr});
        if ((i + 1) % 256 == 0)
            std::printf("    %zu / %zu ...\n", i + 1, fxe::sweep_configs.size());
    }
    std::printf("  Measured %zu sweep configs\n", entries.size());

    // Find sweep winner (for audit baseline)
    auto best_it = std::min_element(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) {
            return a.result.p50_ns < b.result.p50_ns;
        });
    double sweep_best_p50 = best_it->result.p50_ns;
    auto sweep_winner = best_it->config;

    // ── Measure 21 audit configs ──
    // One-flip variants of each consensus-fixed position, using the
    // compile-time audit_configs (based on sweep_configs[0]).
    // Tests whether flipping a fixed position improves the result.
    std::printf("  Measuring %zu audit configs...\n", fxe::N_AUDIT);

    std::vector<audit_entry> audits;
    audits.reserve(fxe::N_AUDIT);
    std::size_t audit_idx = 0;
    for (int f = 0; f < fxe::N_FIXED; ++f) {
        int pos = fxe::CONSENSUS_FIXED[f].position;
        auto fixed_strat = fxe::CONSENSUS_FIXED[f].strategy;
        for (int s = 0; s < fix::NUM_STRATEGIES; ++s) {
            auto alt = static_cast<fxe::Strategy>(s);
            if (alt == fixed_strat) continue;

            const auto& cfg = fxe::audit_configs[audit_idx];
            auto mr = measurer.measure(cfg);
            audits.push_back({pos, fixed_strat, alt, cfg, mr});
            ++audit_idx;
        }
    }

    // ── Measure baselines ──
    std::vector<fxe::BaselineResult> bl_results;
    bl_results.reserve(fxe::num_baselines);
    for (const auto& bl : fxe::baselines) {
        auto mr = measurer.measure(bl.config);
        bl_results.push_back({
            std::string(bl.name),
            fix::config_to_string(bl.config),
            mr.p50_ns, mr.p99_ns
        });
    }

    // ── Report ──
    auto report = build_sweep_report(entries, bl_results, fxe::N_BEAM);
    fxe::print_report(report);

    // Sweep range
    auto worst_it = std::max_element(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) {
            return a.result.p50_ns < b.result.p50_ns;
        });
    std::printf("  Sweep range: %.2f – %.2f ns (p50, %zu configs)\n",
        sweep_best_p50, worst_it->result.p50_ns, entries.size());

    // Audit
    print_audit(audits, sweep_best_p50);

    // JSON
    try {
        fxe::write_json_report_to("results", report);
        std::printf("\n  -> results/%s.json written\n", report.program_id.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Warning: %s\n", e.what());
    }

    return 0;
}

// ─────────────────────────────────────────────────────────────────────

#ifdef CTDP_FIX_EXPERIMENT_MOCK

int main() {
    std::printf("Program F — consensus sweep (MOCK MODE)\n\n");
    std::printf("  Sweep space: %zu configs (%d uncertain positions)\n",
        fxe::sweep_configs.size(), fxe::N_UNCERTAIN);
    std::printf("  Audit: %zu one-flip variants (%d fixed positions)\n",
        fxe::N_AUDIT, fxe::N_FIXED);

    fxe::mock_measurer measurer{
        .seed = fxe::MSG_POOL_SEED, .noise_sigma = fxe::MOCK_NOISE_SIGMA};

    return run_sweep(measurer);
}

#else

int main() {
    std::printf("Program F — consensus sweep (RDTSC)\n\n");
    std::printf("  Sweep space: %zu configs (%d uncertain positions)\n",
        fxe::sweep_configs.size(), fxe::N_UNCERTAIN);
    std::printf("  Audit: %zu one-flip variants (%d fixed positions)\n",
        fxe::N_AUDIT, fxe::N_FIXED);

    auto messages = fix::generate_message_pool(fxe::POOL_SIZE, fxe::EVAL_POOL_SEED);
    auto mconfig  = fxe::default_measurement_config();
    fxe::rdtsc_adapter adapter{messages, mconfig};

    // Dispatch table: 1,024 sweep + 26 ancillary (5 baselines + 21 audit)
    fxe::compiled_measurer_dual<
        fxe::rdtsc_adapter, fxe::sweep_configs, fxe::sweep_ancillary>
        measurer{adapter};

    std::printf("  Dispatch table: %zu + %zu entries\n",
        fxe::sweep_configs.size(), fxe::sweep_ancillary.size());

    return run_sweep(measurer);
}

#endif
