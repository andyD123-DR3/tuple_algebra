// spmv_tridiag_demo.cpp — CT-DP SpMV tridiagonal demo driver
//
// End-to-end vertical slice through the CT-DP framework:
//
//   structural factory  →  framework analysis  →  candidate construction
//   →  correctness gate  →  measurement-based search  →  typed execution
//   →  structured reporting
//
// This demo sits alongside the existing spmv_benchmark.cc. Both solve
// the same problem. The existing benchmark is a flat script; this demo
// routes through the framework's architectural layers. The side-by-side
// comparison shows what the framework adds: explicit plan algebra, typed
// dispatch, space-layer integration, and structured analysis-to-measurement
// reporting.
//
// Plan algebra: choose(leaf(csr), leaf(dia))
//
// Build: part of examples/framework/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#include "ctdp/domain/spmv/spmv_schema.h"
#include "spmv_app_types.h"
#include "spmv_bench.h"
#include "spmv_correctness.h"

#include "spmv/spmv_graph.h"

#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ctdp::graph;
using namespace ctdp::domain::spmv;
using namespace ctdp::demo::spmv;

// ============================================================================
// Wire the demo's typed executors to the framework's spmv_executor<F>
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
// Reporting
// ============================================================================

namespace {

constexpr std::size_t MaxR   = 32768;
constexpr std::size_t MaxNNZ = 100000;

void print_separator() {
    std::cout << "============================================================\n";
}

/// Convert spmv_format to string using the schema's plan_format_name.
std::string format_name_str(spmv_format f) {
    return plan_format_name(SpmvPlan{f});
}

void report_case(
    std::size_t n,
    sparsity_metrics const& metrics,
    format_recommendation const& rec,
    spmv_search_result const& sr)
{
    print_separator();
    std::cout << "CT-DP SpMV Demo: Tridiagonal (n = " << n << ")\n";
    print_separator();

    // --- Structural analysis (framework) ---
    std::cout << "\n--- Structural Analysis (framework) ---\n";
    std::cout << "  Dimension:        " << n << "\n";
    std::cout << "  Bandwidth:        " << metrics.bandwidth << "\n";
    std::cout << "  Num diagonals:    " << metrics.num_diagonals << "\n";
    std::cout << "  Uniformity:       " << std::fixed << std::setprecision(2)
              << metrics.uniformity() << "\n";
    std::cout << "  DIA efficiency:   " << std::fixed << std::setprecision(2)
              << metrics.dia_efficiency() << "\n";
    std::cout << "  Recommendation:   " << format_name_str(rec.format) << "\n";

    // --- Plan space ---
    std::cout << "\n--- Plan Space ---\n";
    std::cout << "  AST: choose(";
    for (std::size_t i = 0; i < sr.all_results.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "leaf(" << plan_format_name(SpmvPlan{sr.all_results[i].format}) << ")";
    }
    std::cout << ")\n";
    std::cout << "  Candidates: " << sr.candidates_evaluated << "\n";

    // --- Measurement ---
    std::cout << "\n--- Measurement ---\n";
    for (auto const& r : sr.all_results) {
        std::cout << "  " << std::left << std::setw(4)
                  << plan_format_name(SpmvPlan{r.format}) << ": "
                  << std::right << std::setw(8) << std::fixed
                  << std::setprecision(2) << r.ns_per_row << " ns/row  ("
                  << std::setprecision(2) << r.gflops << " GFLOP/s)  ["
                  << r.iterations << " iterations]\n";
    }

    // --- Result ---
    std::cout << "\n--- Result ---\n";
    std::cout << "  Winner: "
              << plan_format_name(sr.best_plan) << "\n";

    // Speedup over the other candidate
    if (sr.all_results.size() == 2) {
        double other_ns = (sr.all_results[0].format == sr.best_plan.format)
            ? sr.all_results[1].ns_per_row
            : sr.all_results[0].ns_per_row;
        if (sr.best_result.ns_per_row > 0.0) {
            double speedup = other_ns / sr.best_result.ns_per_row;
            std::string other_name =
                (sr.all_results[0].format == sr.best_plan.format)
                    ? plan_format_name(SpmvPlan{sr.all_results[1].format})
                    : plan_format_name(SpmvPlan{sr.all_results[0].format});
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
                      << speedup << "x faster than " << other_name << "\n";
        }
    }

    // Agreement with framework recommendation
    bool agreed = (sr.best_plan.format == rec.format);
    std::cout << "  Agreed with recommendation: "
              << (agreed ? "YES" : "NO") << "\n";

    print_separator();
    std::cout << "\n";
}

} // anonymous namespace

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CT-DP SpMV Framework Demo\n";
    std::cout << "Tridiagonal operator {-1, 2, -1}\n";
    std::cout << "Plan algebra: choose(leaf(CSR), leaf(DIA))\n";
    print_separator();
    std::cout << "\n";

    // Compile-time verification on small pattern
    constexpr auto small_pat     = make_tridiagonal<16, 64>(8);
    constexpr auto small_metrics = compute_metrics(small_pat);
    constexpr auto small_rec     = recommend_format(small_metrics);
    static_assert(small_rec.format == spmv_format::dia,
        "Framework recommends DIA for tridiagonal at compile time");

    // Size sweep
    constexpr std::array<std::size_t, 5> sizes = {64, 256, 1024, 4096, 16384};

    int exit_code = 0;

    for (auto n : sizes) {
        // 1. Structural construction (framework factory)
        auto pat = make_tridiagonal<MaxR, MaxNNZ>(n);

        // 2. Analysis (framework layer)
        auto metrics = compute_metrics(pat);
        auto rec     = recommend_format(metrics);

        // 3. Candidate construction (schema layer)
        auto cs = construct_candidates(metrics, rec);
        spmv_space space{cs};

        // 4. Build application-level matrix representations
        tridiag_values coeffs{};
        auto csr = build_csr(pat, coeffs);
        auto dia = build_dia(pat, coeffs);

        // 5. Deterministic input
        auto x = make_input_vector(n);

        // 6. Correctness gate — must pass before measurement
        std::vector<double> y_csr(n), y_dia(n);
        spmv_exec_csr(csr, x.data(), y_csr.data());
        spmv_exec_dia(dia, x.data(), y_dia.data());

        auto agree = compare_outputs(y_csr.data(), y_dia.data(), n);
        auto check_csr = analytic_check(y_csr.data(), n);
        auto check_dia = analytic_check(y_dia.data(), n);

        if (!agree.ok || !check_csr.ok || !check_dia.ok) {
            std::cout << "CORRECTNESS FAILURE at n=" << n << "\n";
            std::cout << "  CSR vs DIA agreement: "
                      << (agree.ok ? "PASS" : "FAIL")
                      << " (max |diff| = " << agree.max_abs_diff << ")\n";
            std::cout << "  Analytic check (CSR): "
                      << (check_csr.ok ? "PASS" : "FAIL") << "\n";
            std::cout << "  Analytic check (DIA): "
                      << (check_dia.ok ? "PASS" : "FAIL") << "\n";
            exit_code = 1;
            continue;
        }

        // 7. Search by measurement
        auto sr = search_candidates(space, csr, dia, x.data(), n, pat.nnz);

        // 8. Report
        report_case(n, metrics, rec, sr);

        // 9. Execute winner through typed boundary (final validation)
        std::vector<double> y_final(n, -1.0);
        visit_plan(sr.best_plan, [&](auto fmt_tag) {
            constexpr auto F = decltype(fmt_tag)::value;
            if constexpr (F == spmv_format::csr)
                spmv_executor<F>::run(csr, x.data(), y_final.data());
            else if constexpr (F == spmv_format::dia)
                spmv_executor<F>::run(dia, x.data(), y_final.data());
        });

        auto final_check = analytic_check(y_final.data(), n);
        if (!final_check.ok) {
            std::cout << "  TYPED EXECUTION FAILURE at n=" << n << "\n";
            exit_code = 1;
        }
    }

    // Summary
    std::cout << "\n";
    print_separator();
    std::cout << "Summary\n";
    print_separator();
    std::cout << "\n";
    std::cout << "  The framework's structural analysis identifies the\n";
    std::cout << "  tridiagonal operator as diagonal-friendly (3 diagonals,\n";
    std::cout << "  bandwidth 1, high DIA efficiency).\n";
    std::cout << "\n";
    std::cout << "  The plan algebra expresses this as:\n";
    std::cout << "    choose(leaf(CSR), leaf(DIA))\n";
    std::cout << "\n";
    std::cout << "  Measurement-based search evaluates both candidates\n";
    std::cout << "  through the typed executor path (visit_plan ->\n";
    std::cout << "  spmv_executor<F>::run) and selects the winner.\n";
    std::cout << "\n";
    std::cout << "  Compare with spmv_benchmark.cc which solves the same\n";
    std::cout << "  problem as a flat script without framework integration.\n";
    print_separator();
    std::cout << "\n";

    return exit_code;
}
