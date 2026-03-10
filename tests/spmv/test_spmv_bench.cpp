// test_spmv_bench.cpp — Tests for the SpMV bench adapter and search (PR3)
//
// Tests cover:
//   - measure_plan: returns finite results for CSR and DIA
//   - measure_plan: ns_per_row is positive and reasonable
//   - measure_plan: iteration count meets minimum
//   - search_candidates: evaluates all candidates
//   - search_candidates: selects a valid winner
//   - search_candidates: all_results populated
//   - End-to-end: analysis → candidates → search → winner
//
// Note: these are integration tests that do real timing.
// They use short measurement windows (low min_iterations, short
// target_seconds) to keep test runtime reasonable.
//
// Build: requires GTest, spmv_graph.h, spmv_schema.h, spmv_app_types.h
//
// Part of the CT-DP framework — tests/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#include "spmv_bench.h"
#include "spmv_correctness.h"

#include "ctdp/domain/spmv/spmv_schema.h"
#include "spmv_app_types.h"

#include "spmv/spmv_graph.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

using namespace ctdp::graph;
using namespace ctdp::domain::spmv;
using namespace ctdp::demo::spmv;

// ============================================================================
// Executor specialisations (same as PR2 tests — needed for typed dispatch)
// ============================================================================

namespace ctdp::domain::spmv {

template<>
struct spmv_executor<spmv_format::csr> {
    static void run(demo_csr_matrix const& m,
                    double const* x, double* y) {
        spmv_exec_csr(m, x, y);
    }
};

template<>
struct spmv_executor<spmv_format::dia> {
    static void run(demo_dia_matrix const& m,
                    double const* x, double* y) {
        spmv_exec_dia(m, x, y);
    }
};

} // namespace ctdp::domain::spmv

// ============================================================================
// Test config: short measurement for fast tests
// ============================================================================

namespace {

constexpr std::size_t MaxR   = 512;
constexpr std::size_t MaxNNZ = 1600;

// Fast bench config for tests — just enough to get stable-ish results
spmv_bench_config const test_cfg{
    /* min_iterations    */ 100,
    /* warmup_iterations */ 10,
    /* target_seconds    */ 0.05
};

struct bench_test_instance {
    demo_csr_matrix csr;
    demo_dia_matrix dia;
    std::vector<double> x;
    std::size_t n;
    std::size_t nnz;
    spmv_space space;
};

bench_test_instance make_bench_instance(std::size_t n) {
    auto pat     = make_tridiagonal<MaxR, MaxNNZ>(n);
    auto metrics = compute_metrics(pat);
    auto rec     = recommend_format(metrics);
    auto cs      = construct_candidates(metrics, rec);

    tridiag_values v{};
    bench_test_instance ti;
    ti.n     = n;
    ti.nnz   = pat.nnz;
    ti.csr   = build_csr(pat, v);
    ti.dia   = build_dia(pat, v);
    ti.x     = make_input_vector(n);
    ti.space = spmv_space{cs};
    return ti;
}

} // anonymous namespace

// ============================================================================
// measure_plan tests
// ============================================================================

TEST(MeasurePlan, CSRReturnsFinite) {
    auto ti = make_bench_instance(256);
    SpmvPlan plan{spmv_format::csr};

    auto m = measure_plan(plan, ti.csr, ti.dia, ti.x.data(),
                          ti.n, ti.nnz, test_cfg);

    EXPECT_EQ(m.format, spmv_format::csr);
    EXPECT_GT(m.ns_per_row, 0.0);
    EXPECT_LT(m.ns_per_row, 1e6);  // sanity: less than 1ms per row
    EXPECT_GT(m.total_ns, 0.0);
    EXPECT_GT(m.gflops, 0.0);
}

TEST(MeasurePlan, DIAReturnsFinite) {
    auto ti = make_bench_instance(256);
    SpmvPlan plan{spmv_format::dia};

    auto m = measure_plan(plan, ti.csr, ti.dia, ti.x.data(),
                          ti.n, ti.nnz, test_cfg);

    EXPECT_EQ(m.format, spmv_format::dia);
    EXPECT_GT(m.ns_per_row, 0.0);
    EXPECT_LT(m.ns_per_row, 1e6);
    EXPECT_GT(m.total_ns, 0.0);
    EXPECT_GT(m.gflops, 0.0);
}

TEST(MeasurePlan, MeetsMinIterations) {
    auto ti = make_bench_instance(64);
    SpmvPlan plan{spmv_format::csr};

    spmv_bench_config cfg{200, 10, 0.01};
    auto m = measure_plan(plan, ti.csr, ti.dia, ti.x.data(),
                          ti.n, ti.nnz, cfg);

    EXPECT_GE(m.iterations, 200u);
}

TEST(MeasurePlan, NsPerRowScalesWithSize) {
    // ns_per_row should be roughly stable across sizes
    // (not growing linearly with n)
    auto ti_small = make_bench_instance(64);
    auto ti_large = make_bench_instance(256);
    SpmvPlan plan{spmv_format::csr};

    auto m_small = measure_plan(plan, ti_small.csr, ti_small.dia,
                                ti_small.x.data(), ti_small.n,
                                ti_small.nnz, test_cfg);
    auto m_large = measure_plan(plan, ti_large.csr, ti_large.dia,
                                ti_large.x.data(), ti_large.n,
                                ti_large.nnz, test_cfg);

    // ns_per_row should be within 10× of each other
    // (very loose — just checking the metric isn't broken)
    EXPECT_GT(m_small.ns_per_row, 0.0);
    EXPECT_GT(m_large.ns_per_row, 0.0);
    double ratio = m_large.ns_per_row / m_small.ns_per_row;
    EXPECT_GT(ratio, 0.1);
    EXPECT_LT(ratio, 10.0);
}

// ============================================================================
// search_candidates tests
// ============================================================================

TEST(SearchCandidates, EvaluatesAllCandidates) {
    auto ti = make_bench_instance(256);

    auto sr = search_candidates(ti.space, ti.csr, ti.dia,
                                ti.x.data(), ti.n, ti.nnz, test_cfg);

    EXPECT_EQ(sr.candidates_evaluated, 2u);
    EXPECT_EQ(sr.all_results.size(), 2u);
}

TEST(SearchCandidates, SelectsValidWinner) {
    auto ti = make_bench_instance(256);

    auto sr = search_candidates(ti.space, ti.csr, ti.dia,
                                ti.x.data(), ti.n, ti.nnz, test_cfg);

    // Winner should have a valid format
    EXPECT_TRUE(sr.best_plan.format == spmv_format::csr
             || sr.best_plan.format == spmv_format::dia);

    // Winner's ns_per_row should be finite and positive
    EXPECT_GT(sr.best_result.ns_per_row, 0.0);
    EXPECT_LT(sr.best_result.ns_per_row,
              std::numeric_limits<double>::infinity());
}

TEST(SearchCandidates, WinnerIsActuallyBest) {
    auto ti = make_bench_instance(256);

    auto sr = search_candidates(ti.space, ti.csr, ti.dia,
                                ti.x.data(), ti.n, ti.nnz, test_cfg);

    // The winner's ns_per_row should be <= all other results
    for (auto const& r : sr.all_results) {
        EXPECT_LE(sr.best_result.ns_per_row, r.ns_per_row + 1e-6);
    }
}

TEST(SearchCandidates, AllResultsHaveCorrectFormats) {
    auto ti = make_bench_instance(256);

    auto sr = search_candidates(ti.space, ti.csr, ti.dia,
                                ti.x.data(), ti.n, ti.nnz, test_cfg);

    ASSERT_EQ(sr.all_results.size(), 2u);
    EXPECT_EQ(sr.all_results[0].format, spmv_format::csr);
    EXPECT_EQ(sr.all_results[1].format, spmv_format::dia);
}

// ============================================================================
// End-to-end: full pipeline from analysis to winner
// ============================================================================

TEST(EndToEnd, AnalysisToWinner) {
    constexpr std::size_t N = 256;
    auto pat     = make_tridiagonal<MaxR, MaxNNZ>(N);
    auto metrics = compute_metrics(pat);
    auto rec     = recommend_format(metrics);
    auto cs      = construct_candidates(metrics, rec);
    spmv_space space{cs};

    tridiag_values v{};
    auto csr = build_csr(pat, v);
    auto dia = build_dia(pat, v);
    auto x   = make_input_vector(N);

    // Correctness gate
    std::vector<double> y_csr(N), y_dia(N);
    spmv_exec_csr(csr, x.data(), y_csr.data());
    spmv_exec_dia(dia, x.data(), y_dia.data());
    auto agree = compare_outputs(y_csr.data(), y_dia.data(), N);
    ASSERT_TRUE(agree.ok);
    auto check = analytic_check(y_csr.data(), N);
    ASSERT_TRUE(check.ok);

    // Search
    auto sr = search_candidates(space, csr, dia, x.data(), N,
                                pat.nnz, test_cfg);

    EXPECT_EQ(sr.candidates_evaluated, 2u);
    EXPECT_GT(sr.best_result.ns_per_row, 0.0);

    // Execute winner through typed boundary
    std::vector<double> y_final(N, -1.0);
    visit_plan(sr.best_plan, [&](auto fmt_tag) {
        constexpr auto F = decltype(fmt_tag)::value;
        if constexpr (F == spmv_format::csr)
            spmv_executor<F>::run(csr, x.data(), y_final.data());
        else if constexpr (F == spmv_format::dia)
            spmv_executor<F>::run(dia, x.data(), y_final.data());
    });

    // Winner should also pass analytic check
    auto final_check = analytic_check(y_final.data(), N);
    EXPECT_TRUE(final_check.ok);
}
