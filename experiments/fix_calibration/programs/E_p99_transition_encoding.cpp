// experiments/fix_calibration/programs/E_p99_transition_encoding.cpp
//
// Program E: log-transformed p99 experiment.
//
// Question: "Do adjacency features help?"
// Target:   log(p99) (log-nanoseconds)
// Encoding: transition_extractor (56 features = 36 one-hot + 4 counts + 16 transitions)
//
// Build targets (per-program, no global phase switch):
//
//   fix_experiment_C          Mock or Phase 1 (discovery)
//   fix_experiment_C_verify   Phase 2 (verification)
//
// Three compile modes selected by preprocessor defines:
//
//   CTDP_FIX_EXPERIMENT_MOCK:
//     Uses mock_measurer -- runs both phases in-process, no RDTSC.
//     For CI and development testing.
//
//   Default (no defines):
//     Phase 1: measures training_pool via compiled_measurer + rdtsc_adapter.
//     Trains SVR, runs beam search, writes generated/E_candidates.h,
//     generated/E_predicted_scores.h, and generated/E_reference_panel.h
//     to the source tree.
//
//   CTDP_PHASE2:
//     Phase 2: re-derives discovery on the training pool as a stability
//     diagnostic, then measures compiled Phase 1 candidates + baselines
//     on a separate evaluation pool (EVAL_POOL_SEED).  Reports SVR
//     forecast accuracy (in log-space against log(p99)) and cross-run
//     reference panel drift.

#include "experiment_runner.h"
#include "candidate_writer.h"
#include "output_table.h"
#include "json_report.h"

#ifdef CTDP_FIX_EXPERIMENT_MOCK
#include "mock_measurer.h"
#else
#include "rdtsc_adapter.h"
#include "compiled_measurer.h"
#endif

#ifdef CTDP_PHASE2
#include "E_candidates.h"
#include "E_predicted_scores.h"
#include "E_reference_panel.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

static fxe::experiment_params make_params() {
    return fxe::experiment_params{
        .program_id         = "E",
        .program_name       = "E_p99_transition_encoding",
        .target_description = "log(p99)",
        .target_units       = "log(ns)",
    };
}

static auto target_fn() {
    return [](const fxe::measurement_result& r) { return std::log(r.p99_ns); };
}

static fxe::candidate_provenance make_provenance(const char* mode) {
    return fxe::candidate_provenance{
        .program_id         = "E",
        .mode               = mode,
        .target_description = "log(p99)",
        .extractor_name     = "transition_extractor",
        .beam_width         = fxe::BEAM_WIDTH,
        .train_seed         = fxe::TRAIN_SEED,
        .n_train            = fxe::training_pool.size(),
    };
}

// -----------------------------------------------------------------
//  Helpers: generated-header writers, forecast accuracy, ref panel
// -----------------------------------------------------------------

/// Fixed seed for the reference panel workload -- independent of both
/// MSG_POOL_SEED and EVAL_POOL_SEED so it acts as a stable ruler.
static constexpr std::uint32_t REF_POOL_SEED = 777;

/// Write a generated header containing a constexpr array of doubles.
static void write_predicted_scores_header(
    const std::filesystem::path& path,
    const std::vector<double>& scores,
    const char* array_name)
{
    std::ofstream os(path);
    if (!os) {
        throw std::runtime_error(
            "write_predicted_scores_header: unable to open "
            + path.string());
    }
    os << "#ifndef CTDP_FIX_EXPERIMENT_E_PREDICTED_SCORES_H\n";
    os << "#define CTDP_FIX_EXPERIMENT_E_PREDICTED_SCORES_H\n\n";
    os << "#include <array>\n\n";
    os << "namespace ctdp::fix_experiment::generated {\n";
    os << "inline constexpr std::array<double, "
       << scores.size() << "> " << array_name << " = {";
    os << std::setprecision(17);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (i != 0) os << ", ";
        os << scores[i];
    }
    os << "};\n";
    os << "} // namespace ctdp::fix_experiment::generated\n\n";
    os << "#endif\n";
}

/// Write a generated header containing baseline p50 reference-panel values
/// from Phase 1.  This panel is used as a machine-state ruler, not as the
/// log(p99) experiment target.
static void write_reference_panel_header(
    const std::filesystem::path& path,
    const std::vector<double>& p50s,
    const char* array_name)
{
    std::ofstream os(path);
    if (!os) {
        throw std::runtime_error(
            "write_reference_panel_header: unable to open "
            + path.string());
    }
    os << "#ifndef CTDP_FIX_EXPERIMENT_E_REFERENCE_PANEL_H\n";
    os << "#define CTDP_FIX_EXPERIMENT_E_REFERENCE_PANEL_H\n\n";
    os << "#include <array>\n\n";
    os << "namespace ctdp::fix_experiment::generated {\n";
    os << "inline constexpr std::array<double, "
       << p50s.size() << "> " << array_name << " = {";
    os << std::setprecision(17);
    for (std::size_t i = 0; i < p50s.size(); ++i) {
        if (i != 0) os << ", ";
        os << p50s[i];
    }
    os << "};\n";
    os << "} // namespace ctdp::fix_experiment::generated\n\n";
    os << "#endif\n";
}

// -- Forecast accuracy (log(p99)-targeted) ------------------------
//
// C's training target is log(p99), so forecast error is computed in
// log-space: predicted_target - log(measured_p99_ns).
// Relative error is back-transformed via |exp(err) - 1|.
// The shared run_verification() picks best_measured_index by p50,
// which is wrong for C's target.  We compute the p99-based
// best-measured index locally.

struct forecast_summary {
    double      mae_log     = std::numeric_limits<double>::quiet_NaN();
    double      rmse_log    = std::numeric_limits<double>::quiet_NaN();
    double      mean_rel_pct  = std::numeric_limits<double>::quiet_NaN();
    double      median_rel_pct = std::numeric_limits<double>::quiet_NaN();
    bool        top1_correct = false;
    std::size_t predicted_top1_measured_rank = 0;  // 1-based
};

static forecast_summary compute_forecast_summary(
    const fxe::ExperimentReport& report)
{
    forecast_summary out{};
    std::vector<double> rel_pcts;
    rel_pcts.reserve(report.candidates.size());

    double abs_sum = 0.0;
    double sq_sum  = 0.0;
    std::size_t n  = 0;

    for (const auto& c : report.candidates) {
        if (!std::isfinite(c.predicted_target)) continue;
        if (c.measured_p99_ns <= 0.0) continue;
        // C targets log(p99): compare prediction in log-space.
        const double actual_log_p99 = std::log(c.measured_p99_ns);
        const double err     = c.predicted_target - actual_log_p99;
        const double abs_err = std::abs(err);
        abs_sum += abs_err;
        sq_sum  += err * err;
        ++n;
        // Back-transform to relative percentage: |exp(err) - 1| * 100.
        rel_pcts.push_back(100.0 * std::abs(std::exp(err) - 1.0));
    }

    if (n > 0) {
        out.mae_log  = abs_sum / static_cast<double>(n);
        out.rmse_log = std::sqrt(sq_sum / static_cast<double>(n));
    }
    if (!rel_pcts.empty()) {
        out.mean_rel_pct = std::accumulate(rel_pcts.begin(), rel_pcts.end(), 0.0)
                         / static_cast<double>(rel_pcts.size());
        std::sort(rel_pcts.begin(), rel_pcts.end());
        const std::size_t m = rel_pcts.size();
        out.median_rel_pct = (m % 2 == 1)
            ? rel_pcts[m / 2]
            : 0.5 * (rel_pcts[m / 2 - 1] + rel_pcts[m / 2]);
    }

    // Best measured by p99 (not p50) -- computed locally because the
    // shared runner's best_measured_index is always p50-based.
    std::size_t best_p99_idx = 0;
    if (!report.candidates.empty()) {
        for (std::size_t i = 1; i < report.candidates.size(); ++i) {
            if (report.candidates[i].measured_p99_ns <
                report.candidates[best_p99_idx].measured_p99_ns) {
                best_p99_idx = i;
            }
        }
    }
    out.top1_correct = (best_p99_idx == 0);
    out.predicted_top1_measured_rank = best_p99_idx + 1;
    return out;
}

static void print_forecast_summary(const forecast_summary& s) {
    std::printf(
        "\nForecast accuracy on verified shortlist (target: log(p99)):\n"
        "  MAE  in log-space            : %.6f log(ns)\n"
        "  RMSE in log-space            : %.6f log(ns)\n"
        "  Mean relative error          : %.3f %%\n"
        "  Median relative error        : %.3f %%\n"
        "  Top-1 exact (by p99)         : %s\n"
        "  Predicted top-1 measured rank: %zu\n",
        s.mae_log, s.rmse_log, s.mean_rel_pct, s.median_rel_pct,
        s.top1_correct ? "yes" : "no",
        s.predicted_top1_measured_rank);
}

// -- Fixed reference panel ----------------------------------------

struct reference_point {
    std::string name;
    std::string config_label;
    double      p50_ns;
};

struct reference_summary {
    std::vector<reference_point> points;
    double mean_p50_ns   = std::numeric_limits<double>::quiet_NaN();
    double median_p50_ns = std::numeric_limits<double>::quiet_NaN();
};

template<class Measurer>
static reference_summary measure_reference_panel(Measurer& measurer) {
    reference_summary out{};
    out.points.reserve(fxe::num_baselines);
    std::vector<double> vals;
    vals.reserve(fxe::num_baselines);

    for (const auto& bl : fxe::baselines) {
        auto mr = measurer.measure(bl.config);
        out.points.push_back(reference_point{
            std::string(bl.name),
            fix::config_to_string(bl.config),
            mr.p50_ns});
        vals.push_back(mr.p50_ns);
    }

    if (!vals.empty()) {
        out.mean_p50_ns = std::accumulate(vals.begin(), vals.end(), 0.0)
                        / static_cast<double>(vals.size());
        std::sort(vals.begin(), vals.end());
        const std::size_t n = vals.size();
        out.median_p50_ns = (n % 2 == 1)
            ? vals[n / 2]
            : 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    }
    return out;
}

static void print_reference_summary(
    const char* title, const reference_summary& s)
{
    std::printf("\n%s\n", title);
    for (const auto& p : s.points)
        std::printf("  %-14s %-12s p50 = %.3f ns\n",
            p.name.c_str(), p.config_label.c_str(), p.p50_ns);
    std::printf("  Mean p50   : %.3f ns\n", s.mean_p50_ns);
    std::printf("  Median p50 : %.3f ns\n", s.median_p50_ns);
}

static void print_reference_drift(
    const char* label,
    const reference_summary& baseline,
    const reference_summary& current)
{
    if (baseline.points.size() != current.points.size()) {
        std::printf("\nReference drift (%s): unavailable (size mismatch)\n",
            label);
        return;
    }
    double max_abs_shift = 0.0;
    for (std::size_t i = 0; i < baseline.points.size(); ++i) {
        const double d = current.points[i].p50_ns
                       - baseline.points[i].p50_ns;
        max_abs_shift = std::max(max_abs_shift, std::abs(d));
    }
    std::printf(
        "\nReference panel drift (%s):\n"
        "  Mean shift  : %.3f ns\n"
        "  Median shift: %.3f ns\n"
        "  Max |shift| : %.3f ns\n",
        label,
        current.mean_p50_ns   - baseline.mean_p50_ns,
        current.median_p50_ns - baseline.median_p50_ns,
        max_abs_shift);
}

static reference_summary load_phase1_reference(
    const double* p50s, std::size_t n)
{
    reference_summary out{};
    out.points.reserve(n);
    std::vector<double> vals;
    vals.reserve(n);

    for (std::size_t i = 0; i < n && i < fxe::baselines.size(); ++i) {
        out.points.push_back(reference_point{
            std::string(fxe::baselines[i].name),
            fix::config_to_string(fxe::baselines[i].config),
            p50s[i]});
        vals.push_back(p50s[i]);
    }

    if (!vals.empty()) {
        out.mean_p50_ns = std::accumulate(vals.begin(), vals.end(), 0.0)
                        / static_cast<double>(vals.size());
        std::sort(vals.begin(), vals.end());
        const std::size_t m = vals.size();
        out.median_p50_ns = (m % 2 == 1)
            ? vals[m / 2]
            : 0.5 * (vals[m / 2 - 1] + vals[m / 2]);
    }
    return out;
}

// -----------------------------------------------------------------

#ifdef CTDP_FIX_EXPERIMENT_MOCK

// Mock mode: both phases in one run, no real measurement.
int main() {
    std::printf("Program E - log(p99) transitions (MOCK MODE)\n\n");

    fxe::mock_measurer measurer{
        .seed = fxe::MSG_POOL_SEED, .noise_sigma = fxe::MOCK_NOISE_SIGMA};

    auto report = fxe::run_experiment<fxe::transition_extractor>(
        make_params(), measurer, target_fn());

    fxe::print_report(report);

    try {
        fxe::write_json_report_to("results", report);
        std::printf("  -> results/%s.json written\n",
            report.program_id.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Warning: %s\n", e.what());
    }

    return 0;
}

#elif !defined(CTDP_PHASE2)

// Phase 1: discovery -- measure training pool, beam search, write candidates.
int main() {
    std::printf("Program E - log(p99) transitions (Phase 1: discovery)\n\n");

    auto mconfig = fxe::default_measurement_config();

    // Reference panel FIRST -- before heavy work, so the ruler is not
    // polluted by prior cache/thermal state.
    auto ref_messages = fix::generate_message_pool(fxe::POOL_SIZE, REF_POOL_SEED);
    fxe::rdtsc_adapter ref_adapter{ref_messages, mconfig};
    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::baseline_configs>
        ref_measurer{ref_adapter};
    auto ref_panel = measure_reference_panel(ref_measurer);
    print_reference_summary(
        "Reference panel (Phase 1, measured before discovery):", ref_panel);

    // Persist Phase 1 reference panel for cross-run drift comparison.
    {
        std::vector<double> ref_p50s;
        ref_p50s.reserve(ref_panel.points.size());
        for (const auto& p : ref_panel.points)
            ref_p50s.push_back(p.p50_ns);
        auto ref_path = std::filesystem::path(CTDP_EXPERIMENT_GENERATED_DIR)
            / "E_reference_panel.h";
        write_reference_panel_header(
            ref_path, ref_p50s, "E_reference_panel");
        std::printf("  -> %s written (%zu values)\n",
            ref_path.string().c_str(), ref_p50s.size());
    }

    // Discovery.
    auto messages = fix::generate_message_pool(fxe::POOL_SIZE, fxe::MSG_POOL_SEED);
    fxe::rdtsc_adapter adapter{messages, mconfig};
    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::training_pool>
        measurer{adapter};

    std::printf("\n  Training pool: %zu configs\n", fxe::training_pool.size());
    std::printf("  Measuring training set...\n");

    auto discovery = fxe::run_discovery<fxe::transition_extractor>(
        make_params(), measurer, target_fn());

    std::printf("  Discovery complete: %zu candidates\n",
        discovery.candidates.size());
    std::printf("  CV R2: %.4f  |  CV RMSE: %.4f\n",
        discovery.model.cv_r2, discovery.model.cv_rmse);

    // Write generated headers to canonical source-tree location.
    auto header_path = std::filesystem::path(CTDP_EXPERIMENT_GENERATED_DIR)
        / "E_candidates.h";
    fxe::write_candidate_header(header_path, discovery.candidates,
        "E_candidates", make_provenance("rdtsc"));
    std::printf("\n  -> %s written (%zu candidates)\n",
        header_path.string().c_str(), discovery.candidates.size());

    auto pred_path = std::filesystem::path(CTDP_EXPERIMENT_GENERATED_DIR)
        / "E_predicted_scores.h";
    write_predicted_scores_header(
        pred_path, discovery.predicted_scores, "E_predicted_scores");
    std::printf("  -> %s written (%zu scores)\n",
        pred_path.string().c_str(), discovery.predicted_scores.size());

    std::printf("\n  Rebuild fix_experiment_C_verify for Phase 2.\n");

    return 0;
}

#else // CTDP_PHASE2

// Phase 2: verification -- diagnostic re-derivation, forecast accuracy,
// cross-run reference drift, then measure compiled Phase 1 candidates
// + baselines on a separate evaluation pool.
int main() {
    std::printf("Program E - log(p99) transitions (Phase 2: verification)\n\n");

    auto mconfig = fxe::default_measurement_config();
    using namespace ctdp::fix_experiment::generated;

    // Compile-time check: generated artifacts must be aligned.
    static_assert(E_predicted_scores.size() == E_candidates.size(),
        "E_predicted_scores must match E_candidates size");
    static_assert(E_reference_panel.size() == fxe::num_baselines,
        "E_reference_panel must match baseline count");

    // -- Reference panel FIRST -- before any heavy work.
    auto ref_messages = fix::generate_message_pool(fxe::POOL_SIZE, REF_POOL_SEED);
    fxe::rdtsc_adapter ref_adapter{ref_messages, mconfig};
    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::baseline_configs>
        ref_measurer{ref_adapter};
    auto phase2_ref = measure_reference_panel(ref_measurer);
    print_reference_summary(
        "Reference panel (Phase 2, measured before rediscovery):",
        phase2_ref);

    // Load the persisted Phase 1 reference panel and compare.
    auto phase1_ref = load_phase1_reference(
        E_reference_panel.data(), E_reference_panel.size());
    print_reference_summary(
        "Reference panel (Phase 1, persisted):", phase1_ref);
    print_reference_drift(
        "Phase 2 vs Phase 1", phase1_ref, phase2_ref);

    // -- Stage 1: re-derive discovery on the TRAINING pool as a diagnostic.
    // Advisory only -- real-mode RDTSC noise can legitimately cause drift.
    auto train_msgs = fix::generate_message_pool(
        fxe::POOL_SIZE, fxe::MSG_POOL_SEED);
    fxe::rdtsc_adapter train_adapter{train_msgs, mconfig};
    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::training_pool>
        train_measurer{train_adapter};

    auto rediscovery = fxe::run_discovery<fxe::transition_extractor>(
        make_params(), train_measurer, target_fn());

    bool exact_match = true;
    if (rediscovery.candidates.size() != E_candidates.size()) {
        exact_match = false;
    } else {
        for (std::size_t i = 0; i < E_candidates.size(); ++i) {
            if (rediscovery.candidates[i] != E_candidates[i]) {
                exact_match = false;
                std::printf("  Warning: rediscovery mismatch at index %zu: "
                    "rediscovery=%s compiled=%s\n",
                    i,
                    fix::config_to_string(rediscovery.candidates[i]).c_str(),
                    fix::config_to_string(E_candidates[i]).c_str());
                break;
            }
        }
    }

    if (exact_match) {
        std::printf("  Consistency check: PASS (%zu candidates match)\n",
            rediscovery.candidates.size());
    } else {
        std::printf("  Consistency check: NON-IDENTICAL in real mode; "
            "continuing with compiled Phase 1 candidates\n");
    }

    // -- Stage 2: measure compiled candidates + baselines on a SEPARATE
    // evaluation pool.  Dispatch table covers E_candidates (Pool1) and
    // baseline_configs (Pool2) -- exactly the configs run_verification
    // will query.
    //
    // Note: model metrics come from Phase 2 rediscovery, not Phase 1.
    // The candidates and predicted scores are the persisted Phase 1
    // artifacts.
    fxe::discovery_result compiled_discovery;
    compiled_discovery.candidates.assign(
        E_candidates.begin(), E_candidates.end());
    compiled_discovery.predicted_scores.assign(
        E_predicted_scores.begin(), E_predicted_scores.end());
    compiled_discovery.model          = rediscovery.model;
    compiled_discovery.extractor_name = rediscovery.extractor_name;

    auto eval_msgs = fix::generate_message_pool(
        fxe::POOL_SIZE, fxe::EVAL_POOL_SEED);
    fxe::rdtsc_adapter eval_adapter{eval_msgs, mconfig};
    fxe::compiled_measurer_dual<
        fxe::rdtsc_adapter, E_candidates, fxe::baseline_configs>
        eval_measurer{eval_adapter};

    std::printf("\n  Measuring %zu compiled candidates + %zu baselines...\n",
        compiled_discovery.candidates.size(), fxe::num_baselines);

    auto report = fxe::run_verification(
        make_params(), eval_measurer, compiled_discovery);

    std::printf(
        "\n  Note: model CV metrics below are from Phase 2 rediscovery diagnostics;\n"
        "  candidates and predicted scores are persisted Phase 1 artifacts.\n");
    std::printf(
        "  Note: the shared report's best-measured / mis-rank summary is p50-based;\n"
        "  the E-specific forecast summary after the table uses the correct log(p99) target.\n");

    fxe::print_report(report);

    auto fs = compute_forecast_summary(report);
    print_forecast_summary(fs);

    try {
        fxe::write_json_report_to("results", report);
        std::printf("  -> results/%s.json written\n",
            report.program_id.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Warning: %s\n", e.what());
    }

    return 0;
}

#endif