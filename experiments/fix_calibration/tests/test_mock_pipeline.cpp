// experiments/fix_calibration/tests/test_mock_pipeline.cpp
//
// End-to-end tests for the experiment pipeline using mock_measurer.
// Calls run_discovery() and run_verification() directly — never parses
// stdout, never writes files.
//
// Tests:
//   MockPipeline/PipelineCompletes        — full run returns valid report
//   MockPipeline/ReportInvariants         — sorted ranks, valid indices
//   MockPipeline/ModelQuality             — CV R² > 0.5 (mock is learnable)
//   MockPipeline/BeamBeatsWorstBaseline   — best candidate ≤ worst baseline
//   MockPipeline/BaselinesPopulated       — 5 baselines with valid measurements
//   MockPipeline/Determinism              — identical runs → identical reports
//   MockPipeline/DiscoveryVerifySplit     — Phase 1 + Phase 2 match combined
//   MockPipeline/CountExtractor           — pipeline works with 40-dim features
//   MockPipeline/TransitionExtractor      — pipeline works with 56-dim features
//   MockPipeline/LogP99Target             — log-transform target variant
//   MockPipeline/CandidateWriter          — generated header is valid C++

#include "experiment_runner.h"
#include "candidate_writer.h"
#include "mock_measurer.h"
#include "output_table.h"
#include "json_report.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

// ── Shared helpers ────────────────────────────────────────────────────

static fxe::experiment_params make_params(
    const std::string& id = "T",
    const std::string& name = "test_pipeline")
{
    return fxe::experiment_params{
        .program_id         = id,
        .program_name       = name,
        .target_description = "raw p50",
        .target_units       = "ns",
        .beam_width         = 10,
    };
}

static auto p50_target() {
    return [](const fxe::measurement_result& r) { return r.p50_ns; };
}

static auto log_p99_target() {
    return [](const fxe::measurement_result& r) { return std::log(r.p99_ns); };
}

// =================================================================
// Full pipeline with onehot_extractor
// =================================================================

TEST(MockPipeline, PipelineCompletes) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto params = make_params();

    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        params, m, p50_target());

    EXPECT_EQ(report.program_id, "T");
    EXPECT_EQ(report.program_name, "test_pipeline");
    EXPECT_EQ(report.extractor_name, "onehot_extractor");
    EXPECT_FALSE(report.candidates.empty());
    EXPECT_LE(report.candidates.size(), params.beam_width);
}

TEST(MockPipeline, ReportInvariants) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), m, p50_target());

    // Ranks are 1-based, sequential, unique
    std::set<std::size_t> ranks;
    for (const auto& c : report.candidates) {
        EXPECT_GE(c.rank, 1u);
        EXPECT_LE(c.rank, report.candidates.size());
        EXPECT_TRUE(ranks.insert(c.rank).second)
            << "duplicate rank " << c.rank;
    }

    // Sorted by rank (ascending)
    for (std::size_t i = 1; i < report.candidates.size(); ++i) {
        EXPECT_LT(report.candidates[i-1].rank, report.candidates[i].rank);
    }

    // best_measured_index in bounds
    if (!report.candidates.empty()) {
        EXPECT_LT(report.best_measured_index, report.candidates.size());
    }

    // All measurements positive
    for (const auto& c : report.candidates) {
        EXPECT_GT(c.measured_p50_ns, 0.0) << "config " << c.config_label;
        EXPECT_GT(c.measured_p99_ns, 0.0) << "config " << c.config_label;
        EXPECT_GE(c.measured_p99_ns, c.measured_p50_ns)
            << "p99 < p50 for " << c.config_label;
    }

    // validate_report doesn't throw
    EXPECT_NO_THROW(fxe::validate_report(report));
}

TEST(MockPipeline, ModelQuality) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), m, p50_target());

    // Mock cost model is highly learnable — R² should be high
    EXPECT_GT(report.model.cv_r2, 0.5)
        << "CV R² = " << report.model.cv_r2
        << " — mock cost model should be easily learnable";
    EXPECT_EQ(report.model.n_training, fxe::N_TRAIN);
    EXPECT_EQ(report.model.n_features, fxe::onehot_extractor::DIM);
    EXPECT_EQ(report.model.n_folds, 5u);
}

TEST(MockPipeline, BeamBeatsWorstBaseline) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), m, p50_target());

    ASSERT_FALSE(report.candidates.empty());
    ASSERT_FALSE(report.baselines.empty());

    double best_candidate_p50 =
        report.candidates[report.best_measured_index].measured_p50_ns;

    // Worst baseline by p50
    double worst_baseline_p50 = 0.0;
    for (const auto& bl : report.baselines) {
        worst_baseline_p50 = std::max(worst_baseline_p50, bl.measured_p50_ns);
    }

    EXPECT_LT(best_candidate_p50, worst_baseline_p50)
        << "Best candidate p50 (" << best_candidate_p50
        << ") should beat worst baseline p50 (" << worst_baseline_p50 << ")";
}

TEST(MockPipeline, BaselinesPopulated) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), m, p50_target());

    EXPECT_EQ(report.baselines.size(), fxe::num_baselines);
    for (const auto& bl : report.baselines) {
        EXPECT_FALSE(bl.name.empty());
        EXPECT_FALSE(bl.config_label.empty());
        EXPECT_GT(bl.measured_p50_ns, 0.0) << bl.name;
        EXPECT_GT(bl.measured_p99_ns, 0.0) << bl.name;
        EXPECT_GE(bl.measured_p99_ns, bl.measured_p50_ns) << bl.name;
    }

    // Check known baseline names
    std::set<std::string> names;
    for (const auto& bl : report.baselines)
        names.insert(bl.name);
    EXPECT_TRUE(names.count("all_unrolled"));
    EXPECT_TRUE(names.count("all_swar"));
    EXPECT_TRUE(names.count("all_generic"));
    EXPECT_TRUE(names.count("phase10g_opt"));
}

TEST(MockPipeline, Determinism) {
    auto params = make_params();

    fxe::mock_measurer m1{.seed = 42, .noise_sigma = 0.3};
    auto r1 = fxe::run_experiment<fxe::onehot_extractor>(
        params, m1, p50_target());

    fxe::mock_measurer m2{.seed = 42, .noise_sigma = 0.3};
    auto r2 = fxe::run_experiment<fxe::onehot_extractor>(
        params, m2, p50_target());

    ASSERT_EQ(r1.candidates.size(), r2.candidates.size());
    for (std::size_t i = 0; i < r1.candidates.size(); ++i) {
        EXPECT_EQ(r1.candidates[i].config_label,
                  r2.candidates[i].config_label) << "index " << i;
        EXPECT_DOUBLE_EQ(r1.candidates[i].measured_p50_ns,
                         r2.candidates[i].measured_p50_ns) << "index " << i;
    }
    EXPECT_EQ(r1.best_measured_index, r2.best_measured_index);

    ASSERT_EQ(r1.baselines.size(), r2.baselines.size());
    for (std::size_t i = 0; i < r1.baselines.size(); ++i) {
        EXPECT_DOUBLE_EQ(r1.baselines[i].measured_p50_ns,
                         r2.baselines[i].measured_p50_ns);
    }
}

// =================================================================
// Phase 1 + Phase 2 split
// =================================================================

TEST(MockPipeline, DiscoveryVerifySplit) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto params = make_params();

    // Phase 1
    auto disc = fxe::run_discovery<fxe::onehot_extractor>(
        params, m, p50_target());

    EXPECT_FALSE(disc.candidates.empty());
    EXPECT_EQ(disc.candidates.size(), disc.predicted_scores.size());
    EXPECT_GT(disc.model.cv_r2, 0.5);
    EXPECT_EQ(disc.extractor_name, "onehot_extractor");

    // Phase 2
    auto report = fxe::run_verification(params, m, disc);

    EXPECT_EQ(report.candidates.size(), disc.candidates.size());
    EXPECT_EQ(report.baselines.size(), fxe::num_baselines);
    EXPECT_NO_THROW(fxe::validate_report(report));

    // Combined should produce the same result
    fxe::mock_measurer m2{.seed = 42, .noise_sigma = 0.3};
    auto combined = fxe::run_experiment<fxe::onehot_extractor>(
        params, m2, p50_target());

    ASSERT_EQ(report.candidates.size(), combined.candidates.size());
    for (std::size_t i = 0; i < report.candidates.size(); ++i) {
        EXPECT_EQ(report.candidates[i].config_label,
                  combined.candidates[i].config_label);
    }
}

// =================================================================
// Extractor variants
// =================================================================

TEST(MockPipeline, CountExtractor) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::count_extractor>(
        make_params(), m, p50_target());

    EXPECT_FALSE(report.candidates.empty());
    EXPECT_EQ(report.model.n_features, fxe::count_extractor::DIM);
    EXPECT_EQ(report.extractor_name, "count_extractor");
    EXPECT_GT(report.model.cv_r2, 0.5);
}

TEST(MockPipeline, TransitionExtractor) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::transition_extractor>(
        make_params(), m, p50_target());

    EXPECT_FALSE(report.candidates.empty());
    EXPECT_EQ(report.model.n_features, fxe::transition_extractor::DIM);
    EXPECT_EQ(report.extractor_name, "transition_extractor");
    EXPECT_GT(report.model.cv_r2, 0.5);
}

// =================================================================
// Target function variant
// =================================================================

TEST(MockPipeline, LogP99Target) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto params = make_params("C", "C_p99_log_target");
    params.target_description = "log(p99)";
    params.target_units = "log(ns)";

    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        params, m, log_p99_target());

    EXPECT_EQ(report.program_id, "C");
    EXPECT_EQ(report.target_description, "log(p99)");
    EXPECT_FALSE(report.candidates.empty());
    // Predictions should be in log space
    for (const auto& c : report.candidates) {
        EXPECT_GT(c.predicted_target, 0.0)
            << "log(p99) prediction should be positive";
    }
    // But measurements are always in ns
    for (const auto& c : report.candidates) {
        EXPECT_GT(c.measured_p50_ns, 1.0);  // > 1 ns (not log)
        EXPECT_GT(c.measured_p99_ns, 1.0);
    }
}

// =================================================================
// candidate_writer
// =================================================================

TEST(MockPipeline, CandidateWriterProducesExpectedMarkers) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto disc = fxe::run_discovery<fxe::onehot_extractor>(
        make_params("A", "A_p50_baseline"), m, p50_target());

    // Write to a temp file
    auto tmp_dir = std::filesystem::temp_directory_path() / "ctdp_test";
    auto header_path = tmp_dir / "A_candidates.h";
    ASSERT_NO_THROW(
        fxe::write_candidate_header(header_path, disc.candidates, "A_candidates"));

    // File exists and is non-empty
    EXPECT_TRUE(std::filesystem::exists(header_path));
    std::ifstream ifs(header_path);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());

    // Contains key markers
    EXPECT_NE(content.find("#pragma once"), std::string::npos);
    EXPECT_NE(content.find("A_candidates"), std::string::npos);
    EXPECT_NE(content.find("fix_config"), std::string::npos);
    EXPECT_NE(content.find("S::Unrolled"), std::string::npos)
        << "Expected at least one Unrolled strategy in candidates";

    // Array size matches
    std::string size_str = std::to_string(disc.candidates.size());
    EXPECT_NE(content.find("fix_config, " + size_str), std::string::npos)
        << "Expected array size " << size_str;

    // Cleanup
    std::filesystem::remove_all(tmp_dir);
}

// =================================================================
// Output integration (report flows through print and JSON)
// =================================================================

TEST(MockPipeline, OutputTableDoesNotCrash) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), m, p50_target());

    std::ostringstream oss;
    EXPECT_NO_THROW(fxe::print_report_to(oss, report));

    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("test_pipeline"), std::string::npos);
    EXPECT_NE(output.find("Baselines"), std::string::npos);
}

TEST(MockPipeline, JsonSerialisation) {
    fxe::mock_measurer m{.seed = 42, .noise_sigma = 0.3};
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), m, p50_target());

    std::string json;
    EXPECT_NO_THROW(json = fxe::to_json(report));
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("\"program_id\""), std::string::npos);
    EXPECT_NE(json.find("\"baselines\""), std::string::npos);
    EXPECT_NE(json.find("\"candidates\""), std::string::npos);
    EXPECT_NE(json.find("\"best_measured_index\""), std::string::npos);
}

// =================================================================
// Zero-noise measurer (exact ground truth)
// =================================================================

TEST(MockPipeline, ZeroNoiseBaselines) {
    fxe::zero_noise_measurer zm;
    auto report = fxe::run_experiment<fxe::onehot_extractor>(
        make_params(), zm, p50_target());

    // With zero noise, baselines should match exact ground truth
    for (const auto& bl : report.baselines) {
        if (bl.name == "all_generic") {
            // all_generic has the highest p50 in the mock cost model
            EXPECT_GT(bl.measured_p50_ns, 40.0);
        }
    }

    // With zero noise, model should be near-perfect
    EXPECT_GT(report.model.cv_r2, 0.9)
        << "Zero-noise mock should give very high R²";
}
