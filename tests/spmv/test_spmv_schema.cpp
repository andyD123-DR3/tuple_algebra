// test_spmv_schema.cpp — Tests for the SpMV schema layer (PR2)
//
// Tests cover:
//   - SpmvPlan: equality, format name
//   - visit_plan: dispatches to correct typed path for each format
//   - construct_candidates: produces correct set for tridiagonal,
//     excludes DIA for unfriendly patterns
//   - spmv_space: iteration, size, point access
//   - spmv_executor<F>: specialisation wiring (using demo app types)
//   - Constexpr candidate construction and static_assert verification
//
// Build: requires GTest, spmv_graph.h, spmv_schema.h, spmv_app_types.h
//
// Part of the CT-DP framework — tests/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#include "ctdp/domain/spmv/spmv_schema.h"
#include "spmv_app_types.h"
#include "spmv_correctness.h"

#include "ctdp/domain/spmv/spmv_graph.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

using namespace ctdp::graph;
using namespace ctdp::domain::spmv;
using namespace ctdp::demo::spmv;

// ============================================================================
// Wire the demo's typed executors to the framework's spmv_executor<F>
//
// This is what an application does: specialise spmv_executor for each
// format it supports, delegating to its own kernel implementations.
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
// SpmvPlan tests
// ============================================================================

TEST(SpmvPlan, Equality) {
    SpmvPlan a{spmv_format::csr};
    SpmvPlan b{spmv_format::csr};
    SpmvPlan c{spmv_format::dia};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(SpmvPlan, FormatName) {
    EXPECT_STREQ(plan_format_name(SpmvPlan{spmv_format::csr}),  "CSR");
    EXPECT_STREQ(plan_format_name(SpmvPlan{spmv_format::dia}),  "DIA");
    EXPECT_STREQ(plan_format_name(SpmvPlan{spmv_format::ell}),  "ELL");
    EXPECT_STREQ(plan_format_name(SpmvPlan{spmv_format::bcsr}), "BCSR");
}

// ============================================================================
// visit_plan tests
// ============================================================================

TEST(VisitPlan, DispatchCSR) {
    SpmvPlan p{spmv_format::csr};
    spmv_format dispatched = spmv_format::dia;  // wrong default

    visit_plan(p, [&](auto fmt_tag) {
        dispatched = decltype(fmt_tag)::value;
    });

    EXPECT_EQ(dispatched, spmv_format::csr);
}

TEST(VisitPlan, DispatchDIA) {
    SpmvPlan p{spmv_format::dia};
    spmv_format dispatched = spmv_format::csr;  // wrong default

    visit_plan(p, [&](auto fmt_tag) {
        dispatched = decltype(fmt_tag)::value;
    });

    EXPECT_EQ(dispatched, spmv_format::dia);
}

TEST(VisitPlan, DispatchELL) {
    SpmvPlan p{spmv_format::ell};
    spmv_format dispatched = spmv_format::csr;

    visit_plan(p, [&](auto fmt_tag) {
        dispatched = decltype(fmt_tag)::value;
    });

    EXPECT_EQ(dispatched, spmv_format::ell);
}

TEST(VisitPlan, ReturnsValue) {
    SpmvPlan p{spmv_format::dia};

    int result = visit_plan(p, [](auto fmt_tag) -> int {
        if constexpr (decltype(fmt_tag)::value == spmv_format::dia)
            return 42;
        else
            return 0;
    });

    EXPECT_EQ(result, 42);
}

// ============================================================================
// visit_plan + typed executor: end-to-end typed dispatch
// ============================================================================

TEST(VisitPlan, TypedExecutorCSR) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{};
    auto csr = build_csr(pat, v);
    auto x   = make_input_vector(N);
    std::vector<double> y(N, -1.0);

    SpmvPlan p{spmv_format::csr};
    visit_plan(p, [&](auto fmt_tag) {
        constexpr auto F = decltype(fmt_tag)::value;
        if constexpr (F == spmv_format::csr) {
            spmv_executor<F>::run(csr, x.data(), y.data());
        }
    });

    // Analytic check: interior zeros, last = n+1
    for (std::size_t i = 0; i < N - 1; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12);
    }
    EXPECT_NEAR(y[N - 1], static_cast<double>(N + 1), 1e-12);
}

TEST(VisitPlan, TypedExecutorDIA) {
    constexpr std::size_t N = 8;
    auto pat = make_tridiagonal<16, 64>(N);
    tridiag_values v{};
    auto dia = build_dia(pat, v);
    auto x   = make_input_vector(N);
    std::vector<double> y(N, -1.0);

    SpmvPlan p{spmv_format::dia};
    visit_plan(p, [&](auto fmt_tag) {
        constexpr auto F = decltype(fmt_tag)::value;
        if constexpr (F == spmv_format::dia) {
            spmv_executor<F>::run(dia, x.data(), y.data());
        }
    });

    for (std::size_t i = 0; i < N - 1; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-12);
    }
    EXPECT_NEAR(y[N - 1], static_cast<double>(N + 1), 1e-12);
}

// ============================================================================
// Candidate construction tests
// ============================================================================

TEST(CandidateConstruction, TridiagonalProducesCSRandDIA) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);

    auto cs = construct_candidates(metrics, rec);

    EXPECT_EQ(cs.count, 2u);
    EXPECT_EQ(cs.plans[0].format, spmv_format::csr);
    EXPECT_EQ(cs.plans[1].format, spmv_format::dia);
}

TEST(CandidateConstruction, DiagonalMatrixProducesCSRandDIA) {
    constexpr auto pat     = make_diagonal<16, 16>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);

    auto cs = construct_candidates(metrics, rec);

    // Diagonal: 1 diagonal, bandwidth 0, perfect DIA efficiency
    EXPECT_EQ(cs.count, 2u);
    EXPECT_EQ(cs.plans[0].format, spmv_format::csr);
    EXPECT_EQ(cs.plans[1].format, spmv_format::dia);
}

TEST(CandidateConstruction, ArrowMatrixExcludesDIA) {
    // Arrow matrix: dense first/last row, sparse elsewhere.
    // High diagonal count relative to size → DIA should be excluded.
    constexpr auto pat     = make_arrow<16, 256>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);

    auto cs = construct_candidates(metrics, rec);

    // Should only have CSR (DIA excluded by the diagonal guard)
    EXPECT_EQ(cs.count, 1u);
    EXPECT_EQ(cs.plans[0].format, spmv_format::csr);
}

TEST(CandidateConstruction, FivePointStencilProducesCSRandDIA) {
    constexpr auto pat     = make_5pt_stencil<16, 128>(4, 4);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);

    auto cs = construct_candidates(metrics, rec);

    EXPECT_EQ(cs.count, 2u);
    EXPECT_EQ(cs.plans[0].format, spmv_format::csr);
    EXPECT_EQ(cs.plans[1].format, spmv_format::dia);
}

// ============================================================================
// Constexpr candidate construction
// ============================================================================

TEST(ConstexprSchema, CandidateConstructionIsConstexpr) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);
    constexpr auto cs      = construct_candidates(metrics, rec);

    static_assert(cs.count == 2);
    static_assert(cs.plans[0].format == spmv_format::csr);
    static_assert(cs.plans[1].format == spmv_format::dia);
}

TEST(ConstexprSchema, PlanEquality) {
    constexpr SpmvPlan a{spmv_format::csr};
    constexpr SpmvPlan b{spmv_format::csr};
    constexpr SpmvPlan c{spmv_format::dia};

    static_assert(a == b);
    static_assert(a != c);
}

// ============================================================================
// spmv_space tests
// ============================================================================

TEST(SpmvSpace, SizeMatchesCandidates) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);
    auto cs = construct_candidates(metrics, rec);

    spmv_space space{cs};
    EXPECT_EQ(space.size(), 2u);
}

TEST(SpmvSpace, PointAccess) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);
    auto cs = construct_candidates(metrics, rec);

    spmv_space space{cs};
    EXPECT_EQ(space.point(0).format, spmv_format::csr);
    EXPECT_EQ(space.point(1).format, spmv_format::dia);
}

TEST(SpmvSpace, ForEachVisitsAll) {
    constexpr auto pat     = make_tridiagonal<16, 64>(8);
    constexpr auto metrics = compute_metrics(pat);
    constexpr auto rec     = recommend_format(metrics);
    auto cs = construct_candidates(metrics, rec);

    spmv_space space{cs};

    std::vector<spmv_format> visited;
    space.for_each([&](SpmvPlan const& p) {
        visited.push_back(p.format);
    });

    ASSERT_EQ(visited.size(), 2u);
    EXPECT_EQ(visited[0], spmv_format::csr);
    EXPECT_EQ(visited[1], spmv_format::dia);
}

TEST(SpmvSpace, SingleCandidateSpace) {
    // Manually construct a CSR-only space
    spmv_candidate_set cs;
    cs.add(SpmvPlan{spmv_format::csr});

    spmv_space space{cs};
    EXPECT_EQ(space.size(), 1u);
    EXPECT_EQ(space.point(0).format, spmv_format::csr);
}
