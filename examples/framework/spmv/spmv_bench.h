// spmv_bench.h — Measurement adapter and search for the CT-DP SpMV demo.
//
// Provides:
//   - spmv_measurement: per-candidate timing result
//   - spmv_bench_config: measurement protocol parameters
//   - measure_plan(): time a single candidate through the typed executor path
//   - spmv_search_result: aggregated search outcome
//   - search_candidates(): exhaustive measurement-based search over spmv_space
//
// The measurement function invokes candidates through visit_plan →
// spmv_executor<F>::run, ensuring that what is measured is exactly
// what typed execution produces.
//
// Primary metric: nanoseconds per row.
//
// Part of the CT-DP framework — examples/framework/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_EXAMPLES_SPMV_BENCH_H
#define CTDP_EXAMPLES_SPMV_BENCH_H

#include "ctdp/domain/spmv/spmv_schema.h"
#include "spmv_app_types.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <vector>

namespace ctdp::demo::spmv {

// ============================================================================
// Measurement result
// ============================================================================

struct spmv_measurement {
    ctdp::graph::spmv_format format;
    double      ns_per_row  = 0.0;    // primary metric
    double      total_ns    = 0.0;    // per-iteration wall clock
    double      gflops      = 0.0;    // 2 * nnz / total_ns
    std::size_t iterations  = 0;
};

// ============================================================================
// Bench configuration
// ============================================================================

struct spmv_bench_config {
    std::size_t min_iterations    = 1000;
    std::size_t warmup_iterations = 100;
    double      target_seconds    = 0.5;
};

// ============================================================================
// measure_plan — time a single candidate
//
// Invokes the candidate through the typed executor path:
//   visit_plan → spmv_executor<F>::run
//
// This ensures measurement covers the real typed dispatch path,
// not a shortcut.
// ============================================================================

inline spmv_measurement measure_plan(
    ctdp::domain::spmv::SpmvPlan const& plan,
    demo_csr_matrix const& csr,
    demo_dia_matrix const& dia,
    double const* x,
    std::size_t n,
    std::size_t nnz,
    spmv_bench_config const& cfg = {})
{
    using ctdp::domain::spmv::visit_plan;
    using ctdp::domain::spmv::spmv_executor;

    std::vector<double> y(n, 0.0);

    // Warmup — through the typed path
    for (std::size_t i = 0; i < cfg.warmup_iterations; ++i) {
        visit_plan(plan, [&](auto fmt_tag) {
            constexpr auto F = decltype(fmt_tag)::value;
            if constexpr (F == ctdp::graph::spmv_format::csr)
                spmv_executor<F>::run(csr, x, y.data());
            else if constexpr (F == ctdp::graph::spmv_format::dia)
                spmv_executor<F>::run(dia, x, y.data());
        });
    }

    // Measurement loop
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t iters = 0;
    double elapsed_ns = 0.0;

    while (iters < cfg.min_iterations
           || elapsed_ns < cfg.target_seconds * 1e9) {
        visit_plan(plan, [&](auto fmt_tag) {
            constexpr auto F = decltype(fmt_tag)::value;
            if constexpr (F == ctdp::graph::spmv_format::csr)
                spmv_executor<F>::run(csr, x, y.data());
            else if constexpr (F == ctdp::graph::spmv_format::dia)
                spmv_executor<F>::run(dia, x, y.data());
        });
        ++iters;

        auto now = std::chrono::high_resolution_clock::now();
        elapsed_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - start).count());
    }

    double per_iter_ns = elapsed_ns / static_cast<double>(iters);
    double flops = 2.0 * static_cast<double>(nnz);

    return spmv_measurement{
        plan.format,
        per_iter_ns / static_cast<double>(n),   // ns_per_row
        per_iter_ns,                              // total_ns
        flops / per_iter_ns,                      // GFLOP/s
        iters
    };
}

// ============================================================================
// Search result
// ============================================================================

struct spmv_search_result {
    ctdp::domain::spmv::SpmvPlan   best_plan;
    spmv_measurement                best_result;
    std::size_t                     candidates_evaluated = 0;
    std::vector<spmv_measurement>   all_results;
};

// ============================================================================
// search_candidates — exhaustive measurement-based search
//
// Evaluates every candidate in the space, selects the best by ns_per_row.
// For a two-candidate space this is trivially correct and avoids premature
// cost-model work.
// ============================================================================

inline spmv_search_result search_candidates(
    ctdp::domain::spmv::spmv_space const& space,
    demo_csr_matrix const& csr,
    demo_dia_matrix const& dia,
    double const* x,
    std::size_t n,
    std::size_t nnz,
    spmv_bench_config const& cfg = {})
{
    spmv_search_result sr;
    sr.best_result.ns_per_row = std::numeric_limits<double>::infinity();

    space.for_each([&](ctdp::domain::spmv::SpmvPlan const& plan) {
        auto result = measure_plan(plan, csr, dia, x, n, nnz, cfg);
        sr.all_results.push_back(result);
        sr.candidates_evaluated++;

        if (result.ns_per_row < sr.best_result.ns_per_row) {
            sr.best_plan   = plan;
            sr.best_result = result;
        }
    });

    return sr;
}

} // namespace ctdp::demo::spmv

#endif // CTDP_EXAMPLES_SPMV_BENCH_H
