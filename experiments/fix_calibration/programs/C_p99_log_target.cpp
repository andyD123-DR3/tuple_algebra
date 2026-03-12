// experiments/fix_calibration/programs/C_p99_log_target.cpp
//
// Program C: log-transformed p99 experiment.
//
// Question: "Does log transform close the gap?"
// Target:   log(p99) (log-nanoseconds)
// Encoding: onehot_extractor (36 features)

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
#include "C_candidates.h"
#endif

#include <cmath>
#include <cstdio>
#include <filesystem>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

static fxe::experiment_params make_params() {
    return fxe::experiment_params{
        .program_id         = "C",
        .program_name       = "C_p99_log_target",
        .target_description = "log(p99)",
        .target_units       = "log(ns)",
    };
}

static auto target_fn() {
    return [](const fxe::measurement_result& r) { return std::log(r.p99_ns); };
}

static fxe::candidate_provenance make_provenance(const char* mode) {
    return fxe::candidate_provenance{
        .program_id = "C", .mode = mode,
        .target_description = "log(p99)", .extractor_name = "onehot_extractor",
        .beam_width = fxe::BEAM_WIDTH, .train_seed = fxe::TRAIN_SEED,
        .n_train = fxe::training_pool.size(),
    };
}

#ifdef CTDP_FIX_EXPERIMENT_MOCK

int main() {
    std::printf("Program C — log(p99) one-hot (MOCK MODE)\n\n");
    fxe::mock_measurer measurer{
        .seed = fxe::MSG_POOL_SEED, .noise_sigma = fxe::MOCK_NOISE_SIGMA};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), measurer, target_fn());
    fxe::print_report(report);
    try {
        fxe::write_json_report_to("results", report);
        std::printf("  -> results/%s.json written\n", report.program_id.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Warning: %s\n", e.what());
    }
    return 0;
}

#elif !defined(CTDP_PHASE2)

int main() {
    std::printf("Program C — log(p99) one-hot (Phase 1: discovery)\n\n");
    auto messages = fix::generate_message_pool(fxe::POOL_SIZE, fxe::MSG_POOL_SEED);
    auto mconfig  = fxe::default_measurement_config();
    fxe::rdtsc_adapter adapter{messages, mconfig};
    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::training_pool>
        measurer{adapter};
    std::printf("  Training pool: %zu configs\n", fxe::training_pool.size());
    std::printf("  Measuring training set...\n");
    auto discovery = fxe::run_discovery<fxe::onehot_extractor>(
        make_params(), measurer, target_fn());
    std::printf("  Discovery complete: %zu candidates\n",
        discovery.candidates.size());
    std::printf("  CV R²: %.4f  |  CV RMSE: %.4f\n",
        discovery.model.cv_r2, discovery.model.cv_rmse);
    auto header_path = std::filesystem::path(CTDP_EXPERIMENT_GENERATED_DIR)
        / "C_candidates.h";
    fxe::write_candidate_header(header_path, discovery.candidates,
        "C_candidates", make_provenance("rdtsc"));
    std::printf("\n  -> %s written (%zu candidates)\n",
        header_path.string().c_str(), discovery.candidates.size());
    std::printf("  Rebuild fix_experiment_C_verify for Phase 2.\n");
    return 0;
}

#else // CTDP_PHASE2

int main() {
    std::printf("Program C — log(p99) one-hot (Phase 2: verification)\n\n");

    auto mconfig = fxe::default_measurement_config();
    using namespace ctdp::fix_experiment::generated;

    // Stage 1: re-derive discovery on TRAINING pool for consistency check.
    auto train_msgs = fix::generate_message_pool(fxe::POOL_SIZE, fxe::MSG_POOL_SEED);
    fxe::rdtsc_adapter train_adapter{train_msgs, mconfig};
    fxe::compiled_measurer_single<fxe::rdtsc_adapter, fxe::training_pool>
        train_measurer{train_adapter};

    auto discovery = fxe::run_discovery<fxe::onehot_extractor>(
        make_params(), train_measurer, target_fn());

    fxe::verify_candidates_match(discovery, C_candidates);
    std::printf("  Consistency check: PASS (%zu candidates match)\n",
        discovery.candidates.size());

    // Stage 2: measure candidates + baselines on SEPARATE evaluation pool.
    auto eval_msgs = fix::generate_message_pool(fxe::POOL_SIZE, fxe::EVAL_POOL_SEED);
    fxe::rdtsc_adapter eval_adapter{eval_msgs, mconfig};
    fxe::compiled_measurer_dual<fxe::rdtsc_adapter, fxe::training_pool, C_candidates>
        eval_measurer{eval_adapter};

    auto report = fxe::run_verification(make_params(), eval_measurer, discovery);
    fxe::print_report(report);
    try {
        fxe::write_json_report_to("results", report);
        std::printf("  -> results/%s.json written\n", report.program_id.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Warning: %s\n", e.what());
    }
    return 0;
}

#endif