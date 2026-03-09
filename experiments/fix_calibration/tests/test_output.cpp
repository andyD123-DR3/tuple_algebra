// test_output.cpp — Unit tests for PR3 output layer
//
// Covers: JSON escaping, empty candidates, invalid best_measured_index,
//         JSON file I/O, console table formatting, stream state restoration,
//         callback signature compatibility.

#include "report_types.h"
#include "output_table.h"
#include "json_report.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ctdp::fix_experiment {
namespace {

// ── Helper: build a minimal valid report ─────────────────────────────

ExperimentReport make_report() {
    ExperimentReport r;
    r.program_id         = "T";
    r.program_name       = "T_test";
    r.extractor_name     = "test_extractor";
    r.target_description = "raw p50";
    r.target_units       = "ns";
    r.beam_width         = 4;
    r.beam_depth         = 3;
    r.model = {/*cv_rmse=*/0.5, /*cv_r2=*/0.95,
               /*n_training=*/100, /*n_features=*/36, /*n_folds=*/5};
    r.candidates.push_back({42, "CS|SF|CS|SK|CS|SF", 25.3, 27.8, 35.1, 1});
    r.candidates.push_back({17, "CS|CS|CS|CS|CS|CS", 26.1, 28.2, 36.5, 2});
    r.candidates.push_back({91, "SF|SF|CS|SK|CS|SF", 27.0, 26.9, 33.8, 3});
    r.best_measured_index = 2;  // rank 3 is fastest by measurement
    return r;
}

ExperimentReport make_empty_report() {
    ExperimentReport r;
    r.program_id         = "E";
    r.program_name       = "E_empty";
    r.extractor_name     = "none";
    r.target_description = "none";
    r.target_units       = "";
    r.beam_width         = 0;
    r.beam_depth         = 0;
    r.model = {0.0, 0.0, 0, 0, 0};
    // candidates left empty
    r.best_measured_index = 0;
    return r;
}

// ── JSON escaping ────────────────────────────────────────────────────

TEST(JsonEscape, QuotesAndBackslash) {
    auto r = make_report();
    r.candidates[0].config_label = "has\"quote\\backslash";
    std::string json = to_json(r);
    EXPECT_NE(json.find("has\\\"quote\\\\backslash"), std::string::npos);
}

TEST(JsonEscape, NewlineAndTab) {
    auto r = make_report();
    r.extractor_name = "line\none\ttab";
    std::string json = to_json(r);
    EXPECT_NE(json.find("line\\none\\ttab"), std::string::npos);
}

TEST(JsonEscape, ControlChar) {
    auto r = make_report();
    r.program_name = std::string("ctrl") + '\x01' + "x";
    std::string json = to_json(r);
    EXPECT_NE(json.find("ctrl\\u0001x"), std::string::npos);
}

// ── Empty candidates ─────────────────────────────────────────────────

TEST(JsonReport, EmptyCandidates) {
    auto r = make_empty_report();
    std::string json = to_json(r);
    // Should have an empty candidates array
    EXPECT_NE(json.find("\"candidates\": ["), std::string::npos);
    // Should NOT have best_p50_ns (no candidates to reference)
    EXPECT_EQ(json.find("best_p50_ns"), std::string::npos);
}

TEST(OutputTable, EmptyCandidates) {
    auto r = make_empty_report();
    std::ostringstream oss;
    print_report_to(oss, r);
    EXPECT_NE(oss.str().find("(no candidates)"), std::string::npos);
}

// ── Invalid best_measured_index ──────────────────────────────────────

TEST(JsonReport, InvalidBestIndexThrows) {
    auto r = make_report();
    r.best_measured_index = 999;
    EXPECT_THROW(to_json(r), std::out_of_range);
}

TEST(OutputTable, InvalidBestIndexThrows) {
    auto r = make_report();
    r.best_measured_index = 999;
    std::ostringstream oss;
    EXPECT_THROW(print_report_to(oss, r), std::out_of_range);
}

// ── JSON file I/O ────────────────────────────────────────────────────

TEST(JsonReport, WriteToTempDir) {
    auto r = make_report();
    auto tmp = std::filesystem::temp_directory_path() / "ctdp_test_output";
    // Clean up from any prior failed run (ignore errors on Windows)
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);

    write_json_report_to(tmp, r);

    // File should exist with program_id as filename
    auto path = tmp / (r.program_id + ".json");
    EXPECT_TRUE(std::filesystem::exists(path));

    // Read and check it contains key fields — scoped so ifstream
    // is closed before cleanup (Windows holds file locks)
    {
        std::ifstream ifs(path);
        std::string content((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
        EXPECT_NE(content.find("\"program_id\": \"T\""), std::string::npos);
        EXPECT_NE(content.find("\"target_units\": \"ns\""), std::string::npos);
        EXPECT_NE(content.find("\"predicted_target\""), std::string::npos);
    }

    // Clean up (ignore errors — antivirus/indexer may hold locks briefly)
    std::filesystem::remove_all(tmp, ec);
}

// ── Console output content checks ────────────────────────────────────

TEST(OutputTable, ContainsKeyFields) {
    auto r = make_report();
    std::ostringstream oss;
    print_report_to(oss, r);
    std::string out = oss.str();

    EXPECT_NE(out.find("Program T"), std::string::npos);
    EXPECT_NE(out.find("T_test"), std::string::npos);
    EXPECT_NE(out.find("test_extractor"), std::string::npos);
    EXPECT_NE(out.find("raw p50"), std::string::npos);
    EXPECT_NE(out.find("(ns)"), std::string::npos);
}

TEST(OutputTable, MisRankDetected) {
    auto r = make_report();
    r.best_measured_index = 2;
    std::ostringstream oss;
    print_report_to(oss, r);
    EXPECT_NE(oss.str().find("model mis-rank"), std::string::npos);
}

TEST(OutputTable, NoMisRankWhenBestMatchesPrediction) {
    auto r = make_report();
    r.best_measured_index = 0;
    std::ostringstream oss;
    print_report_to(oss, r);
    EXPECT_EQ(oss.str().find("model mis-rank"), std::string::npos);
}

// ── Stream state restoration ─────────────────────────────────────────

TEST(OutputTable, StreamStateRestored) {
    auto r = make_report();
    std::ostringstream oss;

    // Set distinctive formatting state
    oss << std::scientific << std::setprecision(10);
    auto flags_before = oss.flags();
    auto prec_before  = oss.precision();

    print_report_to(oss, r);

    // Flags and precision must be restored
    EXPECT_EQ(oss.flags(), flags_before);
    EXPECT_EQ(oss.precision(), prec_before);
}

// ── JSON schema completeness ─────────────────────────────────────────

TEST(JsonReport, SchemaHasAllFields) {
    auto r = make_report();
    std::string json = to_json(r);

    // Top-level fields
    EXPECT_NE(json.find("\"program_id\""), std::string::npos);
    EXPECT_NE(json.find("\"program_name\""), std::string::npos);
    EXPECT_NE(json.find("\"extractor_name\""), std::string::npos);
    EXPECT_NE(json.find("\"target_description\""), std::string::npos);
    EXPECT_NE(json.find("\"target_units\""), std::string::npos);
    EXPECT_NE(json.find("\"beam_width\""), std::string::npos);
    EXPECT_NE(json.find("\"beam_depth\""), std::string::npos);
    EXPECT_NE(json.find("\"best_measured_index\""), std::string::npos);
    EXPECT_NE(json.find("\"best_p50_ns\""), std::string::npos);
    EXPECT_NE(json.find("\"best_p99_ns\""), std::string::npos);

    // Model fields
    EXPECT_NE(json.find("\"cv_rmse\""), std::string::npos);
    EXPECT_NE(json.find("\"cv_r2\""), std::string::npos);
    EXPECT_NE(json.find("\"n_training\""), std::string::npos);
    EXPECT_NE(json.find("\"n_features\""), std::string::npos);
    EXPECT_NE(json.find("\"n_folds\""), std::string::npos);

    // Candidate fields
    EXPECT_NE(json.find("\"predicted_target\""), std::string::npos);
    EXPECT_NE(json.find("\"measured_p50_ns\""), std::string::npos);
    EXPECT_NE(json.find("\"measured_p99_ns\""), std::string::npos);
}

// ── Callback signature compatibility ─────────────────────────────────
// Static check that the one-arg forms are assignable to
// std::function<void(const ExperimentReport&)>.

TEST(CallbackSignature, PrintReportIsOneArg) {
    std::function<void(const ExperimentReport&)> fn = print_report;
    (void)fn;  // compiles = passes
}

TEST(CallbackSignature, WriteJsonReportIsOneArg) {
    std::function<void(const ExperimentReport&)> fn = write_json_report;
    (void)fn;  // compiles = passes
}

} // anonymous namespace
} // namespace ctdp::fix_experiment
