// experiments/fix_calibration/programs/A_p50_baseline.cpp
//
// Program A: p50 baseline experiment.
//
// Question: "Does the pipeline work on the easy target?"
// Target:   raw p50 (nanoseconds)
// Encoding: onehot_extractor (36 features)
//
// Build targets (per-program, no global phase switch):
//
//   fix_experiment_A          Mock or Phase 1 (discovery)
//   fix_experiment_A_verify   Phase 2 (verification)
//
// Three compile modes selected by preprocessor defines:
//
//   CTDP_FIX_EXPERIMENT_MOCK:
//     Uses mock_measurer — runs both phases in-process, no RDTSC.
//     For CI and development testing.
//
//   Default (no defines):
//     Phase 1: measures training_pool via compiled_measurer + rdtsc_adapter.
//     Trains SVR, runs beam search, writes generated/A_candidates.h to
//     the source tree via CTDP_EXPERIMENT_GENERATED_DIR (set by CMake).
//
//   CTDP_PHASE2:
//     Phase 2: measures training_pool + A_candidates via dual dispatch.
//     Re-derives discovery and verifies consistency against compiled array.
//     Uses a separate evaluation message pool (EVAL_POOL_SEED) to test
//     generalisation beyond the training workload.

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
#include "A_candidates.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

static fxe::experiment_params make_params() {
    return fxe::experiment_params{
        .program_id         = "A",
        .program_name       = "A_p50_baseline",
        .target_description = "raw p50",
        .target_units       = "ns",
    };
}

static auto target_fn() {
    return [](const fxe::measurement_result& r) { return r.p50_ns; };
}

static fxe::candidate_provenance make_provenance(const char* mode) {
    return fxe::candidate_provenance{
        .program_id         = "A",
        .mode               = mode,
        .target_description = "raw p50",
        .extractor_name     = "onehot_extractor",
        .beam_width         = fxe::BEAM_WIDTH,
        .train_seed         = fxe::TRAIN_SEED,
        .n_train            = fxe::training_pool.size(),
    };
}

// ─────────────────────────────────────────────────────────────────────

#ifdef CTDP_FIX_EXPERIMENT_MOCK

// Mock mode: both phases in one run, no real measurement.
int main() {
    std::printf("Program A — p50 baseline (MOCK MODE)\n\n");

    fxe::mock_measurer measurer{
        .seed = fxe::MSG_POOL_SEED, .noise_sigma = fxe::MOCK_NOISE_SIGMA};

    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), measurer, target_fn());

    fxe::print_report(report);

    // Write JSON if results/ directory is writable
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

// Phase 1: discovery — measure training pool, beam search, write candidates.
int main() {
    std::printf("Program A — p50 baseline (Phase 1: discovery)\n\n");

    auto messages = fix::generate_message_pool(fxe::POOL_SIZE, fxe::MSG_POOL_SEED);
    auto mconfig  = fxe::default_measurement_config();
    fxe::rdtsc_adapter adapter{messages, mconfig};

    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::training_pool>
        measurer{adapter};

    std::printf("  Training pool: %zu configs\n", fxe::training_pool.size());
    std::printf("  Samples per config: %zu\n", fxe::SAMPLES);
    std::printf("  Measuring training set...\n");

    auto discovery = fxe::run_discovery<fxe::onehot_extractor>(
        make_params(), measurer, target_fn());

    std::printf("  Discovery complete: %zu candidates\n",
        discovery.candidates.size());
    std::printf("  CV R²: %.4f  |  CV RMSE: %.4f\n",
        discovery.model.cv_r2, discovery.model.cv_rmse);

    // Write generated header to canonical source-tree location.
    auto header_path = std::filesystem::path(CTDP_EXPERIMENT_GENERATED_DIR)
        / "A_candidates.h";
    fxe::write_candidate_header(header_path, discovery.candidates,
        "A_candidates", make_provenance("rdtsc"));
    std::printf("\n  -> %s written (%zu candidates)\n",
        header_path.string().c_str(), discovery.candidates.size());
    std::printf("  Rebuild fix_experiment_A_verify for Phase 2.\n");

    return 0;
}

#else // CTDP_PHASE2

// Phase 2: verification — measure candidates + baselines, produce report.
int main() {
    std::printf("Program A — p50 baseline (Phase 2: verification)\n\n");

    // Use a SEPARATE evaluation message pool for honest verification.
    auto messages = fix::generate_message_pool(fxe::POOL_SIZE, fxe::EVAL_POOL_SEED);
    auto mconfig  = fxe::default_measurement_config();
    fxe::rdtsc_adapter adapter{messages, mconfig};

    using namespace ctdp::fix_experiment::generated;
    fxe::compiled_measurer_dual<fxe::rdtsc_adapter, fxe::training_pool, A_candidates>
        measurer{adapter};

    // Re-derive discovery (deterministic) and verify consistency.
    auto discovery = fxe::run_discovery<fxe::onehot_extractor>(
        make_params(), measurer, target_fn());

    fxe::verify_candidates_match(discovery, A_candidates);
    std::printf("  Consistency check: PASS (%zu candidates match)\n",
        discovery.candidates.size());

    std::printf("  Measuring %zu candidates + %zu baselines...\n",
        discovery.candidates.size(), fxe::num_baselines);

    auto report = fxe::run_verification(make_params(), measurer, discovery);
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

#endif
