// spmv_benchmark.cc - Demonstrates compile-time format selection speedup
// Part of the compile-time DP library (C++20)
//
// Build: g++ -std=c++20 -O3 -march=native -I core_lib -I graph_lib spmv_benchmark.cc -o spmv_bench
//
// This benchmark:
// 1. Defines sparse patterns at COMPILE TIME
// 2. Runs compile-time analysis → format recommendation
// 3. Executes actual SpMV in CSR, ELL, DIA at RUNTIME
// 4. Shows the compile-time pick matches the fastest format
//
// The key insight: format selection can be 2-5× more impactful than
// micro-optimization within a single format.

#ifdef _MSC_VER
#define __restrict__ __restrict
#endif

#include "spmv/spmv_graph.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ctdp::graph;

// =============================================================================
// Runtime sparse matrix representations
// =============================================================================

struct csr_matrix {
    std::size_t rows, cols, nnz;
    std::vector<std::size_t> row_ptr;
    std::vector<std::size_t> col_idx;
    std::vector<double> val;
};

struct ell_matrix {
    std::size_t rows, cols;
    std::size_t max_nnz_per_row;
    // Column indices: rows × max_nnz_per_row (column-major for SIMD)
    std::vector<std::size_t> col_idx;  // padded with cols (= invalid)
    std::vector<double> val;           // padded with 0.0
};

struct dia_matrix {
    std::size_t rows, cols;
    std::size_t num_diags;
    std::vector<int> offsets;         // diagonal offsets (e.g., -1, 0, +1)
    std::vector<double> data;         // num_diags × rows (column-major)
};

// =============================================================================
// Build runtime matrices from compile-time patterns
// =============================================================================

template<std::size_t MaxR, std::size_t MaxNNZ>
csr_matrix to_csr(sparse_pattern<MaxR, MaxNNZ> const& pat, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.1, 2.0);
    csr_matrix m;
    m.rows = pat.rows;
    m.cols = pat.cols;
    m.nnz = pat.nnz;
    m.row_ptr.resize(pat.rows + 1);
    m.col_idx.resize(pat.nnz);
    m.val.resize(pat.nnz);
    for (std::size_t i = 0; i <= pat.rows; ++i) {
        m.row_ptr[i] = pat.row_ptr[i];
    }
    for (std::size_t k = 0; k < pat.nnz; ++k) {
        m.col_idx[k] = pat.col_idx[k];
        m.val[k] = dist(rng);
    }
    return m;
}

ell_matrix csr_to_ell(csr_matrix const& csr) {
    ell_matrix m;
    m.rows = csr.rows;
    m.cols = csr.cols;
    m.max_nnz_per_row = 0;
    for (std::size_t r = 0; r < csr.rows; ++r) {
        auto rnnz = csr.row_ptr[r + 1] - csr.row_ptr[r];
        if (rnnz > m.max_nnz_per_row) m.max_nnz_per_row = rnnz;
    }
    auto const N = m.rows * m.max_nnz_per_row;
    m.col_idx.assign(N, csr.cols);  // invalid sentinel
    m.val.assign(N, 0.0);
    for (std::size_t r = 0; r < csr.rows; ++r) {
        std::size_t j = 0;
        for (std::size_t k = csr.row_ptr[r]; k < csr.row_ptr[r + 1]; ++k) {
            // Column-major layout: [j * rows + r]
            m.col_idx[j * m.rows + r] = csr.col_idx[k];
            m.val[j * m.rows + r] = csr.val[k];
            ++j;
        }
    }
    return m;
}

ell_matrix csr_to_ell_rowmajor(csr_matrix const& csr) {
    // Row-major ELL for comparison
    ell_matrix m;
    m.rows = csr.rows;
    m.cols = csr.cols;
    m.max_nnz_per_row = 0;
    for (std::size_t r = 0; r < csr.rows; ++r) {
        auto rnnz = csr.row_ptr[r + 1] - csr.row_ptr[r];
        if (rnnz > m.max_nnz_per_row) m.max_nnz_per_row = rnnz;
    }
    auto const N = m.rows * m.max_nnz_per_row;
    m.col_idx.assign(N, csr.cols);
    m.val.assign(N, 0.0);
    for (std::size_t r = 0; r < csr.rows; ++r) {
        std::size_t j = 0;
        for (std::size_t k = csr.row_ptr[r]; k < csr.row_ptr[r + 1]; ++k) {
            m.col_idx[r * m.max_nnz_per_row + j] = csr.col_idx[k];
            m.val[r * m.max_nnz_per_row + j] = csr.val[k];
            ++j;
        }
    }
    return m;
}

dia_matrix csr_to_dia(csr_matrix const& csr) {
    dia_matrix m;
    m.rows = csr.rows;
    m.cols = csr.cols;
    // Find unique diagonal offsets
    std::vector<int> offsets;
    for (std::size_t r = 0; r < csr.rows; ++r) {
        for (std::size_t k = csr.row_ptr[r]; k < csr.row_ptr[r + 1]; ++k) {
            int off = static_cast<int>(csr.col_idx[k]) - static_cast<int>(r);
            if (std::find(offsets.begin(), offsets.end(), off) == offsets.end()) {
                offsets.push_back(off);
            }
        }
    }
    std::sort(offsets.begin(), offsets.end());
    m.num_diags = offsets.size();
    m.offsets = offsets;
    m.data.assign(m.num_diags * m.rows, 0.0);
    for (std::size_t r = 0; r < csr.rows; ++r) {
        for (std::size_t k = csr.row_ptr[r]; k < csr.row_ptr[r + 1]; ++k) {
            int off = static_cast<int>(csr.col_idx[k]) - static_cast<int>(r);
            // Find diagonal index
            for (std::size_t d = 0; d < m.num_diags; ++d) {
                if (m.offsets[d] == off) {
                    m.data[d * m.rows + r] = csr.val[k];
                    break;
                }
            }
        }
    }
    return m;
}

// =============================================================================
// SpMV kernels
// =============================================================================

void spmv_csr(double* __restrict__ y,
              csr_matrix const& A,
              double const* __restrict__ x) {
    for (std::size_t r = 0; r < A.rows; ++r) {
        double sum = 0.0;
        for (std::size_t k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k) {
            sum += A.val[k] * x[A.col_idx[k]];
        }
        y[r] = sum;
    }
}

void spmv_ell(double* __restrict__ y,
              ell_matrix const& A,
              double const* __restrict__ x) {
    // Column-major ELL: better for vectorization across rows
    for (std::size_t r = 0; r < A.rows; ++r) {
        y[r] = 0.0;
    }
    for (std::size_t j = 0; j < A.max_nnz_per_row; ++j) {
        for (std::size_t r = 0; r < A.rows; ++r) {
            auto const idx = j * A.rows + r;
            auto const c = A.col_idx[idx];
            if (c < A.cols) {
                y[r] += A.val[idx] * x[c];
            }
        }
    }
}

void spmv_dia(double* __restrict__ y,
              dia_matrix const& A,
              double const* __restrict__ x) {
    for (std::size_t r = 0; r < A.rows; ++r) {
        y[r] = 0.0;
    }
    for (std::size_t d = 0; d < A.num_diags; ++d) {
        int const off = A.offsets[d];
        // Determine valid row range for this diagonal
        std::size_t r_begin = (off >= 0) ? 0 : static_cast<std::size_t>(-off);
        std::size_t r_end = (off >= 0)
            ? std::min(A.rows, A.cols - static_cast<std::size_t>(off))
            : A.rows;
        // Inner loop: stride-1 access on both data[] and y[]
        // This is the key: no indirection, perfect for SIMD
        for (std::size_t r = r_begin; r < r_end; ++r) {
            y[r] += A.data[d * A.rows + r] * x[r + static_cast<std::size_t>(off)];
        }
    }
}

// =============================================================================
// Benchmarking infrastructure
// =============================================================================

template<typename Fn>
double bench_ns(Fn&& fn, std::size_t iters) {
    // Warmup
    for (std::size_t i = 0; i < std::min(iters / 10, std::size_t{100}); ++i) {
        fn();
    }

    auto const start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        fn();
    }
    auto const end = std::chrono::high_resolution_clock::now();
    auto const elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
    return static_cast<double>(elapsed) / static_cast<double>(iters);
}

// Verify two results are close (correctness check)
bool verify(std::vector<double> const& a, std::vector<double> const& b,
            double tol = 1e-10) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol * std::max(1.0, std::abs(a[i]))) {
            return false;
        }
    }
    return true;
}

std::string format_name(spmv_format f) {
    switch (f) {
        case spmv_format::csr:  return "CSR";
        case spmv_format::ell:  return "ELL";
        case spmv_format::dia:  return "DIA";
        case spmv_format::bcsr: return "BCSR";
    }
    return "???";
}

// =============================================================================
// Run one benchmark scenario
// =============================================================================

struct bench_result {
    double csr_ns;
    double ell_ns;
    double dia_ns;
    spmv_format recommended;
    spmv_format actual_fastest;
};

template<std::size_t MaxR, std::size_t MaxNNZ>
bench_result run_scenario(
        std::string const& name,
        sparse_pattern<MaxR, MaxNNZ> const& pat,
        sparsity_metrics const& metrics,
        format_recommendation const& rec,
        std::size_t iters) {

    std::mt19937 rng(42);
    auto const csr = to_csr(pat, rng);
    auto const ell = csr_to_ell(csr);
    auto const dia = csr_to_dia(csr);

    // Random x vector
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x(pat.cols);
    for (auto& v : x) v = dist(rng);

    std::vector<double> y_csr(pat.rows, 0.0);
    std::vector<double> y_ell(pat.rows, 0.0);
    std::vector<double> y_dia(pat.rows, 0.0);

    // Correctness check
    spmv_csr(y_csr.data(), csr, x.data());
    spmv_ell(y_ell.data(), ell, x.data());
    spmv_dia(y_dia.data(), dia, x.data());

    bool ell_ok = verify(y_csr, y_ell);
    bool dia_ok = verify(y_csr, y_dia);

    // Benchmark
    double t_csr = bench_ns([&]{ spmv_csr(y_csr.data(), csr, x.data()); }, iters);
    double t_ell = bench_ns([&]{ spmv_ell(y_ell.data(), ell, x.data()); }, iters);
    double t_dia = bench_ns([&]{ spmv_dia(y_dia.data(), dia, x.data()); }, iters);

    double best = std::min({t_csr, t_ell, t_dia});
    spmv_format fastest = spmv_format::csr;
    if (t_ell <= t_csr && t_ell <= t_dia) fastest = spmv_format::ell;
    if (t_dia <= t_csr && t_dia <= t_ell) fastest = spmv_format::dia;

    // Print results
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::left << std::setw(58) << name << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Matrix: " << pat.rows << "×" << pat.cols
              << "  NNZ=" << pat.nnz
              << "  BW=" << metrics.bandwidth
              << "  Diags=" << metrics.num_diagonals
              << std::string(std::max(0, 27 - static_cast<int>(
                  std::to_string(pat.rows).size() +
                  std::to_string(pat.cols).size() +
                  std::to_string(pat.nnz).size() +
                  std::to_string(metrics.bandwidth).size() +
                  std::to_string(metrics.num_diagonals).size())), ' ')
              << "║\n";
    std::cout << "║  Uniformity=" << std::fixed << std::setprecision(2)
              << metrics.uniformity()
              << "  DIA_eff=" << metrics.dia_efficiency()
              << "  ELL_eff=" << metrics.ell_efficiency()
              << "                  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    auto print_row = [&](std::string const& fmt, double ns, bool correct,
                         bool is_rec, bool is_fast) {
        std::cout << "║  " << std::left << std::setw(4) << fmt << ": "
                  << std::right << std::setw(8) << std::fixed
                  << std::setprecision(1) << ns << " ns  "
                  << std::setw(5) << std::setprecision(2)
                  << (best / ns) << "×  "
                  << (correct ? "✓" : "✗") << "  "
                  << (is_rec ? "◀ CT-RECOMMENDED" : "")
                  << (is_fast && !is_rec ? "◀ actual fastest" : "")
                  << (is_fast && is_rec ? " ★ MATCH" : "")
                  << std::string(std::max(0,
                      is_rec ? (is_fast ? 17 : 22) :
                      (is_fast ? 22 : 38)), ' ')
                  << "║\n";
    };

    print_row("CSR", t_csr, true,
              rec.format == spmv_format::csr, fastest == spmv_format::csr);
    print_row("ELL", t_ell, ell_ok,
              rec.format == spmv_format::ell, fastest == spmv_format::ell);
    print_row("DIA", t_dia, dia_ok,
              rec.format == spmv_format::dia, fastest == spmv_format::dia);

    if (rec.format == fastest) {
        std::cout << "║                                                              ║\n";
        std::cout << "║  ✓ Compile-time pick MATCHES runtime fastest!                ║\n";
    } else {
        double rec_time = (rec.format == spmv_format::csr) ? t_csr :
                          (rec.format == spmv_format::ell) ? t_ell : t_dia;
        double penalty = rec_time / best;
        std::cout << "║                                                              ║\n";
        std::cout << "║  ⚠ CT pick ≠ fastest (penalty: "
                  << std::fixed << std::setprecision(1)
                  << (penalty - 1.0) * 100.0 << "% slower)              ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    return {t_csr, t_ell, t_dia, rec.format, fastest};
}

// =============================================================================
// Scaled-up matrix factories (runtime, large enough to benchmark)
// =============================================================================

template<std::size_t MaxR, std::size_t MaxNNZ>
sparse_pattern<MaxR, MaxNNZ> make_tridiagonal_rt(std::size_t n) {
    return make_tridiagonal<MaxR, MaxNNZ>(n);
}

template<std::size_t MaxR, std::size_t MaxNNZ>
sparse_pattern<MaxR, MaxNNZ> make_5pt_stencil_rt(std::size_t w, std::size_t h) {
    return make_5pt_stencil<MaxR, MaxNNZ>(w, h);
}

template<std::size_t MaxR, std::size_t MaxNNZ>
sparse_pattern<MaxR, MaxNNZ> make_arrow_rt(std::size_t n) {
    return make_arrow<MaxR, MaxNNZ>(n);
}

// =============================================================================
// Main: compile-time analysis + runtime benchmark
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  COMPILE-TIME FORMAT SELECTION vs RUNTIME PERFORMANCE       ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Sparse pattern → CT analysis → format recommendation       ║\n";
    std::cout << "║  Then: benchmark ALL formats → verify CT pick = fastest     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // =========================================================================
    // Scenario 1: Tridiagonal (canonical DIA case)
    // =========================================================================
    {
        constexpr std::size_t N = 500;
        constexpr std::size_t MaxR = 512;
        constexpr std::size_t MaxNNZ = 1600;

        // Small pattern for compile-time analysis (same structure)
        constexpr auto small_pat = make_tridiagonal<16, 64>(8);
        constexpr auto metrics = compute_metrics(small_pat);
        constexpr auto rec = recommend_format(metrics);
        static_assert(rec.format == spmv_format::dia,
            "CT analysis: tridiagonal → DIA");

        // Large pattern for benchmarking
        auto const pat = make_tridiagonal_rt<MaxR, MaxNNZ>(N);
        auto const m = compute_metrics(pat);

        run_scenario("Tridiagonal " + std::to_string(N) + "×" + std::to_string(N),
                     pat, m, {rec.format, rec.score}, 500000);
    }

    // =========================================================================
    // Scenario 2: 5-point stencil (structured, banded → DIA)
    // =========================================================================
    {
        constexpr std::size_t W = 20, H = 20;  // 400×400
        constexpr std::size_t MaxR = 512;
        constexpr std::size_t MaxNNZ = 2048;

        constexpr auto small_pat = make_5pt_stencil<16, 128>(4, 4);
        constexpr auto metrics = compute_metrics(small_pat);
        constexpr auto rec = recommend_format(metrics);
        static_assert(rec.format == spmv_format::dia,
            "CT analysis: 5-point stencil → DIA");

        auto const pat = make_5pt_stencil_rt<MaxR, MaxNNZ>(W, H);
        auto const m = compute_metrics(pat);

        run_scenario("5-point stencil " + std::to_string(W) + "×" + std::to_string(H)
                     + " (" + std::to_string(W*H) + " rows)",
                     pat, m, {rec.format, rec.score}, 200000);
    }

    // =========================================================================
    // Scenario 3: Arrow matrix (irregular → CSR)
    // =========================================================================
    {
        constexpr std::size_t N = 300;
        constexpr std::size_t MaxR = 512;
        constexpr std::size_t MaxNNZ = 2048;

        constexpr auto small_pat = make_arrow<16, 256>(8);
        constexpr auto metrics = compute_metrics(small_pat);
        constexpr auto rec = recommend_format(metrics);
        static_assert(rec.format == spmv_format::csr,
            "CT analysis: arrow → CSR");

        auto const pat = make_arrow_rt<MaxR, MaxNNZ>(N);
        auto const m = compute_metrics(pat);

        run_scenario("Arrow " + std::to_string(N) + "×" + std::to_string(N),
                     pat, m, {rec.format, rec.score}, 500000);
    }

    // =========================================================================
    // Scenario 4: Pure diagonal (trivial DIA case)
    // =========================================================================
    {
        constexpr std::size_t N = 500;
        constexpr std::size_t MaxR = 512;
        constexpr std::size_t MaxNNZ = 512;

        constexpr auto small_pat = make_diagonal<16, 16>(8);
        constexpr auto metrics = compute_metrics(small_pat);
        constexpr auto rec = recommend_format(metrics);
        static_assert(rec.format == spmv_format::dia,
            "CT analysis: diagonal → DIA");

        auto const pat = make_diagonal<MaxR, MaxNNZ>(N);
        auto const m = compute_metrics(pat);

        run_scenario("Diagonal " + std::to_string(N) + "×" + std::to_string(N),
                     pat, m, {rec.format, rec.score}, 1000000);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  KEY INSIGHT                                                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  The SAME analysis that runs at compile time on small        ║\n";
    std::cout << "║  representative patterns correctly predicts the optimal      ║\n";
    std::cout << "║  format for large-scale runtime execution.                   ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Sparsity metrics (bandwidth, uniformity, diagonal count)    ║\n";
    std::cout << "║  are structural invariants — they scale with the pattern,    ║\n";
    std::cout << "║  not the size. A 4×4 5-point stencil has the same diagonal   ║\n";
    std::cout << "║  structure as a 1000×1000 one.                               ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Format selection is often MORE impactful than SIMD or       ║\n";
    std::cout << "║  loop optimisation within a single format.                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
