// test_spmv_app_types.cpp — Tests for the SpMV demo application layer (PR1)
//
// Tests cover:
//   - build_csr: structural correctness, value placement
//   - build_dia: diagonal discovery, offset sorting, value placement
//   - spmv_exec_csr: analytic check on {-1,2,-1} tridiagonal
//   - spmv_exec_dia: analytic check on {-1,2,-1} tridiagonal
//   - Cross-format agreement between CSR and DIA
//   - Correctness utilities: compare_outputs, analytic_check
//   - Constexpr factory and analysis verification (static_assert)
//
// Build: requires GTest and the framework's spmv_graph.h
//
// Part of the CT-DP framework — tests/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#include "spmv_app_types.h"
#include "spmv_correctness.h"

#include "spmv/spmv_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

using namespace ctdp::graph;
using namespace ctdp::demo::spmv;

// ============================================================================
// Helper: create pattern and build both matrix formats for a given n
// ============================================================================

namespace {

constexpr std::size_t MaxR   = 512;
constexpr std::size_t MaxNNZ = 1600;

struct test_instance {
    demo_csr_matrix csr;
    demo_dia_matrix dia;
    std::vector<double> x;
    std::size_t n;
};

test_instance make_test_instance(std::size_t n) {
    auto pat = make_tridiagonal<MaxR, MaxNNZ>(n);
    tridiag_values v{};
    test_instance ti;
    ti.n   = n;
    ti.csr = build_csr(pat, v);
    ti.dia = build_dia(pat, v);
    ti.x   = make_input_vector(n);
    return ti;
}

} // anonymous namespace

// ============================================================================
// Factory + pattern tests
// ============================================================================

TEST(Factory, TridiagonalPatternSmall) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);

    EXPECT_EQ(pat.rows, N);
    EXPECT_EQ(pat.cols, N);

    // Row 0: 2 entries (main + super)
    EXPECT_EQ(pat.row_ptr[1] - pat.row_ptr[0], 2u);

    // Interior rows: 3 entries each
    for (std::size_t r = 1; r < N - 1; ++r) {
        EXPECT_EQ(pat.row_ptr[r + 1] - pat.row_ptr[r], 3u)
            << "row " << r << " should have 3 entries";
    }

    // Last row: 2 entries (sub + main)
    EXPECT_EQ(pat.row_ptr[N] - pat.row_ptr[N - 1], 2u);

    // Total NNZ: 2 + (N-2)*3 + 2 = 3N - 2
    EXPECT_EQ(pat.nnz, 3 * N - 2);
}

TEST(Factory, TridiagonalColumnIndices) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);

    // Row 0: columns should be {0, 1}
    EXPECT_EQ(pat.col_idx[pat.row_ptr[0]], 0u);
    EXPECT_EQ(pat.col_idx[pat.row_ptr[0] + 1], 1u);

    // Row 3 (interior): columns should be {2, 3, 4}
    EXPECT_EQ(pat.col_idx[pat.row_ptr[3]], 2u);
    EXPECT_EQ(pat.col_idx[pat.row_ptr[3] + 1], 3u);
    EXPECT_EQ(pat.col_idx[pat.row_ptr[3] + 2], 4u);

    // Row 7 (last): columns should be {6, 7}
    EXPECT_EQ(pat.col_idx[pat.row_ptr[7]], 6u);
    EXPECT_EQ(pat.col_idx[pat.row_ptr[7] + 1], 7u);
}

// ============================================================================
// build_csr tests
// ============================================================================

TEST(BuildCSR, StructureMatchesPattern) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{};
    auto csr = build_csr(pat, v);

    EXPECT_EQ(csr.rows, N);
    EXPECT_EQ(csr.cols, N);
    EXPECT_EQ(csr.nnz, pat.nnz);

    // row_ptr should match pattern exactly
    for (std::size_t i = 0; i <= N; ++i) {
        EXPECT_EQ(csr.row_ptr[i], pat.row_ptr[i]);
    }

    // col_idx should match pattern exactly
    for (std::size_t k = 0; k < pat.nnz; ++k) {
        EXPECT_EQ(csr.col_idx[k], pat.col_idx[k]);
    }
}

TEST(BuildCSR, ValuesCorrect) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{-1.0, 2.0, -1.0};
    auto csr = build_csr(pat, v);

    // Row 0: [2, -1]
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[0]],     2.0);
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[0] + 1], -1.0);

    // Row 3 (interior): [-1, 2, -1]
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[3]],     -1.0);
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[3] + 1],  2.0);
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[3] + 2], -1.0);

    // Row 7 (last): [-1, 2]
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[7]],     -1.0);
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[7] + 1],  2.0);
}

TEST(BuildCSR, CustomCoefficients) {
    constexpr std::size_t N = 4;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{-0.5, 3.0, -0.5};
    auto csr = build_csr(pat, v);

    // Row 1 (interior): [-0.5, 3.0, -0.5]
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[1]],     -0.5);
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[1] + 1],  3.0);
    EXPECT_DOUBLE_EQ(csr.val[csr.row_ptr[1] + 2], -0.5);
}

// ============================================================================
// build_dia tests
// ============================================================================

TEST(BuildDIA, DiagonalDiscovery) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{};
    auto dia = build_dia(pat, v);

    EXPECT_EQ(dia.rows, N);
    EXPECT_EQ(dia.cols, N);
    EXPECT_EQ(dia.num_diags, 3u);

    // Offsets should be sorted: {-1, 0, +1}
    ASSERT_EQ(dia.offsets.size(), 3u);
    EXPECT_EQ(dia.offsets[0], -1);
    EXPECT_EQ(dia.offsets[1],  0);
    EXPECT_EQ(dia.offsets[2], +1);
}

TEST(BuildDIA, StorageSize) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{};
    auto dia = build_dia(pat, v);

    // Storage should be exactly num_diags × rows
    EXPECT_EQ(dia.data.size(), 3u * N);
}

TEST(BuildDIA, ValuesCorrect) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{-1.0, 2.0, -1.0};
    auto dia = build_dia(pat, v);

    // Diagonal 0 (offset -1, sub-diagonal): cm1 = -1
    // Valid for rows 1..N-1
    for (std::size_t r = 1; r < N; ++r) {
        EXPECT_DOUBLE_EQ(dia.data[0 * N + r], -1.0)
            << "sub-diagonal at row " << r;
    }
    // Row 0 should be zero (no sub-diagonal entry)
    EXPECT_DOUBLE_EQ(dia.data[0 * N + 0], 0.0);

    // Diagonal 1 (offset 0, main diagonal): c0 = 2
    for (std::size_t r = 0; r < N; ++r) {
        EXPECT_DOUBLE_EQ(dia.data[1 * N + r], 2.0)
            << "main diagonal at row " << r;
    }

    // Diagonal 2 (offset +1, super-diagonal): cp1 = -1
    // Valid for rows 0..N-2
    for (std::size_t r = 0; r < N - 1; ++r) {
        EXPECT_DOUBLE_EQ(dia.data[2 * N + r], -1.0)
            << "super-diagonal at row " << r;
    }
    // Row N-1 should be zero (no super-diagonal entry)
    EXPECT_DOUBLE_EQ(dia.data[2 * N + (N - 1)], 0.0);
}

// ============================================================================
// CSR kernel tests
// ============================================================================

TEST(CSRKernel, IdentityMatrix) {
    // Build a 4×4 identity manually
    demo_csr_matrix m;
    m.rows = 4; m.cols = 4; m.nnz = 4;
    m.row_ptr = {0, 1, 2, 3, 4};
    m.col_idx = {0, 1, 2, 3};
    m.val     = {1.0, 1.0, 1.0, 1.0};

    std::vector<double> x = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> y(4, 0.0);

    spmv_exec_csr(m, x.data(), y.data());

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(CSRKernel, TridiagonalAnalytic) {
    auto ti = make_test_instance(8);
    std::vector<double> y(ti.n, 0.0);
    spmv_exec_csr(ti.csr, ti.x.data(), y.data());

    // Interior rows and row 0 should be zero
    for (std::size_t i = 0; i < ti.n - 1; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12)
            << "CSR: row " << i << " should be zero";
    }
    // Last row: n + 1
    EXPECT_NEAR(y[ti.n - 1], static_cast<double>(ti.n + 1), 1e-12);
}

TEST(CSRKernel, TridiagonalAnalyticLarger) {
    auto ti = make_test_instance(64);
    std::vector<double> y(ti.n, 0.0);
    spmv_exec_csr(ti.csr, ti.x.data(), y.data());

    for (std::size_t i = 0; i < ti.n - 1; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12)
            << "CSR n=64: row " << i;
    }
    EXPECT_NEAR(y[ti.n - 1], static_cast<double>(ti.n + 1), 1e-12);
}

// ============================================================================
// DIA kernel tests
// ============================================================================

TEST(DIAKernel, TridiagonalAnalytic) {
    auto ti = make_test_instance(8);
    std::vector<double> y(ti.n, 0.0);
    spmv_exec_dia(ti.dia, ti.x.data(), y.data());

    for (std::size_t i = 0; i < ti.n - 1; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12)
            << "DIA: row " << i << " should be zero";
    }
    EXPECT_NEAR(y[ti.n - 1], static_cast<double>(ti.n + 1), 1e-12);
}

TEST(DIAKernel, TridiagonalAnalyticLarger) {
    auto ti = make_test_instance(64);
    std::vector<double> y(ti.n, 0.0);
    spmv_exec_dia(ti.dia, ti.x.data(), y.data());

    for (std::size_t i = 0; i < ti.n - 1; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12)
            << "DIA n=64: row " << i;
    }
    EXPECT_NEAR(y[ti.n - 1], static_cast<double>(ti.n + 1), 1e-12);
}

// ============================================================================
// Cross-format agreement
// ============================================================================

TEST(CrossFormat, CSRvsDIA_n8) {
    auto ti = make_test_instance(8);
    std::vector<double> y_csr(ti.n, 0.0);
    std::vector<double> y_dia(ti.n, 0.0);

    spmv_exec_csr(ti.csr, ti.x.data(), y_csr.data());
    spmv_exec_dia(ti.dia, ti.x.data(), y_dia.data());

    auto r = compare_outputs(y_csr.data(), y_dia.data(), ti.n);
    EXPECT_TRUE(r.ok) << "max_abs_diff = " << r.max_abs_diff;
    EXPECT_DOUBLE_EQ(r.max_abs_diff, 0.0);
}

TEST(CrossFormat, CSRvsDIA_n256) {
    auto ti = make_test_instance(256);
    std::vector<double> y_csr(ti.n, 0.0);
    std::vector<double> y_dia(ti.n, 0.0);

    spmv_exec_csr(ti.csr, ti.x.data(), y_csr.data());
    spmv_exec_dia(ti.dia, ti.x.data(), y_dia.data());

    auto r = compare_outputs(y_csr.data(), y_dia.data(), ti.n);
    EXPECT_TRUE(r.ok) << "max_abs_diff = " << r.max_abs_diff;
}

// ============================================================================
// Correctness utility tests
// ============================================================================

TEST(Correctness, CompareOutputsIdentical) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {1.0, 2.0, 3.0};
    auto r = compare_outputs(a.data(), b.data(), 3);
    EXPECT_TRUE(r.ok);
    EXPECT_DOUBLE_EQ(r.max_abs_diff, 0.0);
}

TEST(Correctness, CompareOutputsDifferent) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {1.0, 2.5, 3.0};
    auto r = compare_outputs(a.data(), b.data(), 3);
    EXPECT_FALSE(r.ok);
    EXPECT_DOUBLE_EQ(r.max_abs_diff, 0.5);
}

TEST(Correctness, AnalyticCheckPass) {
    // Construct the expected output for n=4: {0, 0, 0, 5}
    std::vector<double> y = {0.0, 0.0, 0.0, 5.0};
    auto r = analytic_check(y.data(), 4);
    EXPECT_TRUE(r.ok) << "max_abs_diff = " << r.max_abs_diff;
}

TEST(Correctness, AnalyticCheckFail) {
    // Wrong last value
    std::vector<double> y = {0.0, 0.0, 0.0, 4.0};
    auto r = analytic_check(y.data(), 4);
    EXPECT_FALSE(r.ok);
    EXPECT_DOUBLE_EQ(r.max_abs_diff, 1.0);
}

TEST(Correctness, AnalyticCheckFullPipeline) {
    // Run the full pipeline: factory → build → kernel → analytic check
    auto ti = make_test_instance(16);
    std::vector<double> y(ti.n, 0.0);
    spmv_exec_csr(ti.csr, ti.x.data(), y.data());

    auto r = analytic_check(y.data(), ti.n);
    EXPECT_TRUE(r.ok) << "max_abs_diff = " << r.max_abs_diff;
}

TEST(Correctness, MakeInputVector) {
    auto x = make_input_vector(5);
    ASSERT_EQ(x.size(), 5u);
    EXPECT_DOUBLE_EQ(x[0], 1.0);
    EXPECT_DOUBLE_EQ(x[1], 2.0);
    EXPECT_DOUBLE_EQ(x[2], 3.0);
    EXPECT_DOUBLE_EQ(x[3], 4.0);
    EXPECT_DOUBLE_EQ(x[4], 5.0);
}

// ============================================================================
// Constexpr / static_assert tests (framework analysis layer)
// ============================================================================

TEST(ConstexprAnalysis, TridiagonalMetrics) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);

    // These should be evaluable at compile time
    static_assert(metrics.num_diagonals == 3);
    static_assert(metrics.bandwidth == 1);

    // Runtime checks for floating-point metrics
    EXPECT_NEAR(metrics.uniformity(), 1.0, 0.1);
    EXPECT_GT(metrics.dia_efficiency(), 0.9);
}

TEST(ConstexprAnalysis, TridiagonalRecommendation) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);

    static_assert(rec.format == spmv_format::dia,
        "Framework should recommend DIA for tridiagonal");

    EXPECT_EQ(rec.format, spmv_format::dia);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(EdgeCase, SmallestTridiagonal_n2) {
    // n=2: matrix is [[2, -1], [-1, 2]]
    auto pat = make_tridiagonal<16, 64>(2);
    tridiag_values v{};
    auto csr = build_csr(pat, v);
    auto dia = build_dia(pat, v);

    auto x = make_input_vector(2);  // {1, 2}
    std::vector<double> y_csr(2, 0.0);
    std::vector<double> y_dia(2, 0.0);

    spmv_exec_csr(csr, x.data(), y_csr.data());
    spmv_exec_dia(dia, x.data(), y_dia.data());

    // y_0 = 2*1 - 1*2 = 0
    // y_1 = -1*1 + 2*2 = 3 = n+1
    EXPECT_NEAR(y_csr[0], 0.0, 1e-12);
    EXPECT_NEAR(y_csr[1], 3.0, 1e-12);

    auto agree = compare_outputs(y_csr.data(), y_dia.data(), 2);
    EXPECT_TRUE(agree.ok);

    auto check = analytic_check(y_csr.data(), 2);
    EXPECT_TRUE(check.ok);
}

TEST(EdgeCase, Tridiagonal_n3) {
    auto ti = make_test_instance(3);
    std::vector<double> y_csr(3, 0.0);
    std::vector<double> y_dia(3, 0.0);

    spmv_exec_csr(ti.csr, ti.x.data(), y_csr.data());
    spmv_exec_dia(ti.dia, ti.x.data(), y_dia.data());

    // y = {0, 0, 4} for n=3
    EXPECT_NEAR(y_csr[0], 0.0, 1e-12);
    EXPECT_NEAR(y_csr[1], 0.0, 1e-12);
    EXPECT_NEAR(y_csr[2], 4.0, 1e-12);

    auto agree = compare_outputs(y_csr.data(), y_dia.data(), 3);
    EXPECT_TRUE(agree.ok);

    auto check = analytic_check(y_csr.data(), 3);
    EXPECT_TRUE(check.ok);
}
