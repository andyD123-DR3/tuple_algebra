// spmv_schema.h — CT-DP plan types, typed dispatch, and candidate
// construction for the SpMV domain.
//
// This is a FRAMEWORK-LEVEL header. It lives under include/ctdp/domain/spmv/
// alongside spmv_graph.h because plan types, typed dispatch, and candidate
// construction are reusable framework concepts, not demo-specific code.
//
// The header provides:
//   - SpmvPlan: value-level plan point used during search
//   - spmv_executor<F>: typed executor declarations (specialised by applications)
//   - visit_plan(): value-to-type lowering via std::integral_constant
//   - spmv_candidate_set / construct_candidates(): explicit plan-space construction
//   - spmv_space: thin adapter presenting candidates as a searchable space
//
// In the five-constructor plan algebra, the first demo's plan space is:
//
//   choose(leaf(csr), leaf(dia))
//
// Part of the CT-DP framework — include/ctdp/domain/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_DOMAIN_SPMV_SCHEMA_H
#define CTDP_DOMAIN_SPMV_SCHEMA_H

#include "ctdp/domain/spmv/spmv_graph.h"   // spmv_format, sparsity_metrics,
                                // format_recommendation

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ctdp::domain::spmv {

// ============================================================================
// Value-level plan — used during search
//
// Uses the framework's existing spmv_format enum directly.
// No parallel enum introduced.
// ============================================================================

struct SpmvPlan {
    ctdp::graph::spmv_format format;

    constexpr bool operator==(SpmvPlan const& o) const noexcept {
        return format == o.format;
    }
    constexpr bool operator!=(SpmvPlan const& o) const noexcept {
        return format != o.format;
    }
};

/// Human-readable format name for reporting.
constexpr char const* plan_format_name(SpmvPlan const& p) noexcept {
    switch (p.format) {
        case ctdp::graph::spmv_format::csr:  return "CSR";
        case ctdp::graph::spmv_format::ell:  return "ELL";
        case ctdp::graph::spmv_format::dia:  return "DIA";
        case ctdp::graph::spmv_format::bcsr: return "BCSR";
    }
    return "???";
}

// ============================================================================
// Typed executor — specialised by application code
//
// The framework declares the template; applications provide specialisations
// that wire to their own kernel implementations.
//
// This is where "the plan IS a type" enters the SpMV domain.
// ============================================================================

/// Primary template — intentionally left undefined.
/// Applications must specialise for each format they support.
template<ctdp::graph::spmv_format F>
struct spmv_executor;

// ============================================================================
// Value-to-type lowering: visit_plan
//
// Converts a runtime SpmvPlan into a compile-time type tag via
// std::integral_constant. This is what makes the type boundary real
// rather than cosmetic.
//
// The caller receives a type-level tag and can use it to select
// the correct spmv_executor<F> specialisation.
//
// Currently supports csr and dia for the first demo.
// Extend the switch as new formats are added to the candidate set.
// ============================================================================

template<typename Fn>
decltype(auto) visit_plan(SpmvPlan p, Fn&& fn) {
    using fmt = ctdp::graph::spmv_format;
    switch (p.format) {
        case fmt::csr:
            return fn(std::integral_constant<fmt, fmt::csr>{});
        case fmt::dia:
            return fn(std::integral_constant<fmt, fmt::dia>{});
        case fmt::ell:
            return fn(std::integral_constant<fmt, fmt::ell>{});
        case fmt::bcsr:
            return fn(std::integral_constant<fmt, fmt::bcsr>{});
    }
    std::abort();  // unreachable if switch covers all enum values
}

// ============================================================================
// Candidate construction
//
// Explicit, small, policy-light. Converts analysis results into a legal
// candidate set. For the first demo this always produces {csr, dia} on
// tridiagonal input, but the existence of a construction phase is what
// turns the example into a plan-space demo rather than a hard-coded
// benchmark.
// ============================================================================

struct spmv_candidate_set {
    static constexpr std::size_t max_candidates = 4;

    std::size_t count = 0;
    std::array<SpmvPlan, max_candidates> plans{};

    constexpr void add(SpmvPlan p) {
        if (count < max_candidates) {
            plans[count++] = p;
        }
    }

    constexpr SpmvPlan const& operator[](std::size_t i) const {
        return plans[i];
    }
};

/// Construct the legal candidate set from analysis results.
///
/// CSR is always included as the generic baseline.
/// DIA is included when the analysis indicates diagonal-friendly structure:
///   - small number of occupied diagonals
///   - high DIA efficiency (low padding waste)
///
/// ELL and BCSR can be added in follow-on work by extending this function.
constexpr spmv_candidate_set construct_candidates(
    ctdp::graph::sparsity_metrics const& metrics,
    ctdp::graph::format_recommendation const& /*rec*/)
{
    spmv_candidate_set cs;

    // CSR: always legal
    cs.add(SpmvPlan{ctdp::graph::spmv_format::csr});

    // DIA: legal when structure is diagonal-friendly
    // Guard: few diagonals and high efficiency (low padding)
    if (metrics.num_diagonals > 0
        && metrics.num_diagonals <= 2 * metrics.bandwidth + 1
        && metrics.dia_efficiency() > 0.5) {
        cs.add(SpmvPlan{ctdp::graph::spmv_format::dia});
    }

    return cs;
}

// ============================================================================
// Space adapter
//
// Presents spmv_candidate_set as a searchable space compatible with
// the framework's exhaustive_search pattern. Conceptually this is a
// one-axis descriptor_space over spmv_format.
// ============================================================================

struct spmv_space {
    using point_type = SpmvPlan;

    spmv_candidate_set candidates;

    constexpr std::size_t size() const noexcept {
        return candidates.count;
    }

    constexpr SpmvPlan point(std::size_t i) const {
        return candidates[i];
    }

    /// Enumerate all candidates, invoking f with each plan point.
    template<typename F>
    void for_each(F&& f) const {
        for (std::size_t i = 0; i < candidates.count; ++i) {
            f(candidates.plans[i]);
        }
    }
};

} // namespace ctdp::domain::spmv

#endif // CTDP_DOMAIN_SPMV_SCHEMA_H
