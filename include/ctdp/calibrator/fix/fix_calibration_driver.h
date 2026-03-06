#ifndef CTDP_CALIBRATOR_FIX_CALIBRATION_DRIVER_H
#define CTDP_CALIBRATOR_FIX_CALIBRATION_DRIVER_H

// ============================================================
//  fix_calibration_driver.h  —  CT-DP FIX Parser Optimiser  (Phase 3)
//
//  Calibration driver for the Trivial tier (N=4, 64 plans).
//
//  Wires together all Phase 1 + Phase 2 infrastructure:
//    fix_field_descriptor.h → make_data_points_from_schema
//    fix_bench.h            → bench_corpus, make_message_pool
//    fix_strategy_ids.h     → CostTable<4,double>, trivial_index
//    fix_dp_search.h        → exhaustive_search, beam_dp,
//                              verify_beam_matches_exhaustive
//
//  Public API:
//
//  CalibrationResult run_trivial_calibration(cal_cfg, bench_cfg)
//    1. Builds corpus skeleton (64 DataPoint<4>).
//    2. Generates message pool.
//    3. Calls bench_corpus() — measures all 64 plans via A1/A2/B.
//    4. Populates CostTable<4,double> from Ok/Suspicious DataPoints.
//    5. Runs exhaustive_search<4>() — optimal plan + ranked corpus.
//    6. Runs beam_dp<4>()           — verification vs exhaustive.
//    7. Returns CalibrationResult.
//
//  corpus_to_cost_table(corpus) — exposed for testing.
//
//  mock_calibration_result(p99_fn, ...) — CI-safe test helper;
//    fills CalibrationResult from a lambda without touching hardware.
//
//  Hardware note:
//    run_trivial_calibration calls bench_corpus (RDTSC).
//    Compiled EXCLUDE_FROM_ALL; runs on physical machine only.
//    mock_calibration_result is hardware-free and CI-safe.
//
//  Dependencies: fix_bench.h, fix_dp_search.h
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/fix_bench.h>
#include <ctdp/calibrator/fix/fix_dp_search.h>

#include <cassert>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  CalibrationConfig — run-time tunables
// ─────────────────────────────────────────────────────────────────────────────

struct CalibrationConfig {
    std::size_t message_pool_size = 1024;
    int         beam_width        = 4;
    bool        verbose           = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  CalibrationResult
// ─────────────────────────────────────────────────────────────────────────────

struct CalibrationResult {
    // CostTable has no default ctor (needs a SchemaIndex reference).
    // Provide an explicit default that binds to trivial_index.
    CalibrationResult()
        : cost_table(trivial_index, 0.0)
    {}

    std::vector<DataPoint<4>> corpus;
    CostTable<4, double>      cost_table;
    DpResult<4>               exhaustive{};
    DpResult<4>               beam{};

    int n_ok           = 0;
    int n_suspicious   = 0;
    int n_hard_deleted = 0;
    int beam_width_used = 4;

    [[nodiscard]] bool beam_agrees() const noexcept {
        return exhaustive.found() && beam.found() &&
               exhaustive.optimal_id == beam.optimal_id;
    }

    void print_top_n(int n = 10) const {
        const int sz = static_cast<int>(exhaustive.ranked_corpus.size());
        if (n > sz) n = sz;
        std::printf("\n=== Top %d plans (Trivial schema N=4) ===\n", n);
        std::printf("  %-5s  %-6s  %-10s\n", "Rank", "Plan", "p99 (ns)");
        for (int i = 0; i < n; ++i) {
            const auto& rp = exhaustive.ranked_corpus[static_cast<std::size_t>(i)];
            std::printf("  %-5d  %-6s  %.2f\n",
                        i + 1,
                        plan_string<4>(rp.plan).c_str(),
                        rp.cost);
        }
        std::printf("\n");
    }

    void print_summary() const {
        std::printf("\n=== Trivial Calibration Summary (N=4, 64 plans) ===\n");
        std::printf("  Corpus: %d Ok  %d Suspicious  %d HardDeleted\n",
                    n_ok, n_suspicious, n_hard_deleted);
        std::printf("  Measured in cost table: %d\n", exhaustive.n_measured);
        if (exhaustive.found())
            std::printf("  Optimal (exhaustive):  %-6s  %.2f ns\n",
                        exhaustive.plan_str().c_str(), exhaustive.optimal_cost);
        else
            std::printf("  Optimal: NOT FOUND\n");
        if (beam.found())
            std::printf("  Optimal (beam w=%d):   %-6s  %.2f ns  %s\n",
                        beam_width_used,
                        beam.plan_str().c_str(),
                        beam.optimal_cost,
                        beam_agrees() ? "[AGREES]" : "[DISAGREES!]");
        std::printf("\n");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  corpus_to_cost_table
//
//  Extracts mean_ab_p99_ns() from each Ok/Suspicious DataPoint.
//  B run is excluded (cold-cache sensitivity, not steady-state cost).
//  HardDeleted → entry stays 0.0 (= unmeasured sentinel).
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline CostTable<4, double>
corpus_to_cost_table(const std::vector<DataPoint<4>>& corpus)
{
    CostTable<4, double> table(trivial_index, 0.0);
    for (const auto& dp : corpus) {
        if (dp.is_hard_deleted()) continue;
        const int32_t id = trivial_index.encode(dp.plan);
        if (id < 0) continue;
        const double cost = dp.latency_estimate();  // mean_ab_p99_ns()
        if (cost > 0.0) table[id] = cost;
    }
    return table;
}

// ─────────────────────────────────────────────────────────────────────────────
//  mock_calibration_result — CI-safe test helper
//
//  p99_fn(dense_id) → simulated warm-run p99 in ns.
//  a2_jitter: A2 = A1 + a2_jitter  (controls DataPointStatus)
//  b_offset:  B  = A1 + b_offset   (cold-cache — excluded from cost)
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline CalibrationResult
mock_calibration_result(
    std::function<double(int32_t)> p99_fn,
    double a2_jitter  = 0.5,
    double b_offset   = 5.0,
    int    beam_width = 4)
{
    CalibrationResult result;
    result.beam_width_used = beam_width;

    result.corpus = make_data_points_from_schema(trivial_schema);

    for (auto& dp : result.corpus) {
        const int32_t id = trivial_index.encode(dp.plan);
        const double  a1 = p99_fn(id);
        PerfSample s1{}, s2{}, sb{};
        s1.p99_ns = a1;
        s2.p99_ns = a1 + a2_jitter;
        sb.p99_ns = a1 + b_offset;
        dp.runs   = RunTriple{ s1, s2, sb };
        dp.update_status();
    }

    for (const auto& dp : result.corpus) {
        if (dp.is_ok())               ++result.n_ok;
        else if (dp.is_suspicious())  ++result.n_suspicious;
        else                          ++result.n_hard_deleted;
    }

    result.cost_table = corpus_to_cost_table(result.corpus);
    result.exhaustive = exhaustive_search<4>(result.cost_table);
    result.beam       = beam_dp<4>(result.cost_table, beam_width);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  run_trivial_calibration — Phase 3 hardware entry point
//
//  Calls bench_corpus() (RDTSC). Do NOT call from CI tests.
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline CalibrationResult
run_trivial_calibration(const CalibrationConfig& cal_cfg   = {},
                        const BenchConfig&        bench_cfg = {})
{
    CalibrationResult result;
    result.beam_width_used = cal_cfg.beam_width;

    if (cal_cfg.verbose)
        std::printf("[Phase 3] Building corpus skeleton (64 plans)...\n");
    result.corpus = make_data_points_from_schema(trivial_schema);
    assert(static_cast<int>(result.corpus.size()) == trivial_index.size());

    if (cal_cfg.verbose)
        std::printf("[Phase 3] Generating message pool (%zu messages)...\n",
                    cal_cfg.message_pool_size);
    const auto messages = make_message_pool(cal_cfg.message_pool_size);

    if (cal_cfg.verbose)
        std::printf("[Phase 3] Measuring 64 plans (A1/A2/B protocol)...\n");

    BenchConfig eff = bench_cfg;
    if (cal_cfg.verbose && !eff.progress_cb) {
        eff.progress_cb = [](int done, int total) {
            std::printf("  [%d/%d]\r", done, total);
            std::fflush(stdout);
        };
    }
    bench_corpus(result.corpus, messages, eff);
    if (cal_cfg.verbose) std::printf("\n");

    for (const auto& dp : result.corpus) {
        if (dp.is_ok())               ++result.n_ok;
        else if (dp.is_suspicious())  ++result.n_suspicious;
        else                          ++result.n_hard_deleted;
    }
    if (cal_cfg.verbose)
        std::printf("[Phase 3] Status: %d Ok  %d Suspicious  %d HardDeleted\n",
                    result.n_ok, result.n_suspicious, result.n_hard_deleted);

    result.cost_table = corpus_to_cost_table(result.corpus);

    if (cal_cfg.verbose)
        std::printf("[Phase 3] Running exhaustive_search<4>()...\n");
    result.exhaustive = exhaustive_search<4>(result.cost_table);

    if (cal_cfg.verbose)
        std::printf("[Phase 3] Running beam_dp<4>(w=%d)...\n", cal_cfg.beam_width);
    result.beam = beam_dp<4>(result.cost_table, cal_cfg.beam_width);

    if (cal_cfg.verbose) {
        const bool agrees = verify_beam_matches_exhaustive<4>(
            result.cost_table, cal_cfg.beam_width);
        std::printf("[Phase 3] beam_dp %s exhaustive.\n",
                    agrees ? "AGREES with" : "DISAGREES with");
    }
    return result;
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_CALIBRATION_DRIVER_H
