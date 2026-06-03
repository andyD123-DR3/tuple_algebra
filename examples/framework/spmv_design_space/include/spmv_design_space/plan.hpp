#pragma once

#include "spmv_design_space/contracts.hpp"
#include "spmv_design_space/facts.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace ctdp::spmv_dsl {

struct plan_descriptor {
    std::string name;
    contract_level contract = contract_level::strict_expression;
    storage_kind storage = storage_kind::csr;
    decomposition_kind decomposition = decomposition_kind::flat_rows;
    ordering_kind ordering = ordering_kind::natural;
    colouring_kind colouring = colouring_kind::none;
    preconditioner_kind preconditioner = preconditioner_kind::fixed_diagonal_jacobi;
    threading_kind threading = threading_kind::serial;
    simd_kind simd = simd_kind::scalar;
    fusion_kind fusion = fusion_kind::none;
    reduction_kind reduction = reduction_kind::canonical_pairwise;
    executor_kind executor = executor_kind::csr_executor;
};

struct candidate_result {
    plan_descriptor plan;
    legality_result legality;
    bool conformance_passed = false;
    double best_ns = 0.0;
    double median_ns = 0.0;
    double mean_ns = 0.0;
    double rho = 0.0;
    std::string execution_path;
    std::string plan_tree;
    std::string recursive_search_tree;
    std::string recursive_search_trace;
};

inline std::string describe_plan(const plan_descriptor& p) {
    std::ostringstream os;
    os << p.name
       << " | contract=" << to_string(p.contract)
       << " | storage=" << to_string(p.storage)
       << " | decomposition=" << to_string(p.decomposition)
       << " | ordering=" << to_string(p.ordering)
       << " | colouring=" << to_string(p.colouring)
       << " | preconditioner=" << to_string(p.preconditioner)
       << " | threading=" << to_string(p.threading)
       << " | simd=" << to_string(p.simd)
       << " | fusion=" << to_string(p.fusion)
       << " | reduction=" << to_string(p.reduction)
       << " | executor=" << to_string(p.executor);
    return os.str();
}

inline legality_result analyse_legality(
    const expression_contract& expr,
    const sparse_facts& facts,
    const plan_descriptor& p) {

    if ((p.storage == storage_kind::matrix_free_stencil || p.storage == storage_kind::dia) &&
        !facts.stencil_like) {
        return {.structurally_legal = false,
                .reason = "DIA and matrix-free stencil layouts require recognised stencil facts"};
    }

    if (p.colouring == colouring_kind::red_black_stencil && !facts.stencil_like) {
        return {.structurally_legal = false,
                .reason = "red-black colouring requires stencil facts"};
    }

    if (p.preconditioner == preconditioner_kind::coloured_smoother_solver_family &&
        p.colouring == colouring_kind::none) {
        return {.structurally_legal = false,
                .reason = "coloured smoother requires a colouring"};
    }

    if (strict_requires_fixed_preconditioner(expr.level) &&
        p.preconditioner != preconditioner_kind::fixed_diagonal_jacobi) {
        return {.numerically_legal = false,
                .reason = "strict-expression contract requires the fixed preconditioner binding"};
    }

    const auto caps = capabilities_for(p.reduction);
    if (expr.fixed_observed_reduction_tree && !caps.preserves_observed_reduction) {
        return {.observation_legal = false,
                .reason = "strict observation contract requires the canonical observed reduction tree"};
    }

    if (p.executor == executor_kind::csr_executor && p.storage != storage_kind::csr) {
        return {.structurally_legal = false,
                .reason = "CSR executor requires CSR storage"};
    }

    if (p.executor == executor_kind::dia_executor && p.storage != storage_kind::dia) {
        return {.structurally_legal = false,
                .reason = "DIA executor requires DIA storage"};
    }

    if (p.executor == executor_kind::matrix_free_executor &&
        p.storage != storage_kind::matrix_free_stencil) {
        return {.structurally_legal = false,
                .reason = "matrix-free executor requires matrix-free stencil storage"};
    }

    return {};
}

inline std::string simd_suffix(simd_kind simd) {
    switch (simd) {
    case simd_kind::scalar: return "block1";
    case simd_kind::lanes4: return "block4";
    case simd_kind::lanes8: return "block8";
    }
    return "block?";
}

inline std::string fusion_suffix(fusion_kind fusion) {
    switch (fusion) {
    case fusion_kind::none: return "unfused";
    case fusion_kind::row_local_fused: return "fused";
    }
    return "fusion?";
}

inline void add_width_and_fusion_candidates(std::vector<plan_descriptor>& out,
                                            std::string prefix,
                                            storage_kind storage,
                                            executor_kind executor,
                                            decomposition_kind decomposition = decomposition_kind::flat_rows,
                                            threading_kind threading = threading_kind::serial) {
    const simd_kind widths[] = {simd_kind::scalar, simd_kind::lanes4, simd_kind::lanes8};
    const fusion_kind fusions[] = {fusion_kind::none, fusion_kind::row_local_fused};
    for (const auto width : widths) {
        for (const auto fusion : fusions) {
            out.push_back({
                .name = prefix + "/" + simd_suffix(width) + "/" + fusion_suffix(fusion) + "/jacobi/canonical",
                .storage = storage,
                .decomposition = decomposition,
                .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
                .threading = threading,
                .simd = width,
                .fusion = fusion,
                .executor = executor
            });
        }
    }
}

inline std::vector<plan_descriptor> generate_candidate_plans(const sparse_facts& facts) {
    std::vector<plan_descriptor> out;

    add_width_and_fusion_candidates(out,
                                    "strict/csr/flat",
                                    storage_kind::csr,
                                    executor_kind::csr_executor);

    if (facts.stencil_like) {
        add_width_and_fusion_candidates(out,
                                        "strict/dia/flat",
                                        storage_kind::dia,
                                        executor_kind::dia_executor);

        add_width_and_fusion_candidates(out,
                                        "strict/matrix-free/flat",
                                        storage_kind::matrix_free_stencil,
                                        executor_kind::matrix_free_executor);

        out.push_back({
            .name = "strict/matrix-free/recursive/block4/unfused/jacobi/canonical",
            .storage = storage_kind::matrix_free_stencil,
            .decomposition = decomposition_kind::recursive_grid_bisection,
            .ordering = ordering_kind::natural,
            .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
            .threading = threading_kind::recursive_tasks,
            .simd = simd_kind::lanes4,
            .fusion = fusion_kind::none,
            .executor = executor_kind::matrix_free_executor
        });

        out.push_back({
            .name = "strict/matrix-free/recursive/block8/fused/jacobi/canonical",
            .storage = storage_kind::matrix_free_stencil,
            .decomposition = decomposition_kind::recursive_grid_bisection,
            .ordering = ordering_kind::natural,
            .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
            .threading = threading_kind::recursive_tasks,
            .simd = simd_kind::lanes8,
            .fusion = fusion_kind::row_local_fused,
            .executor = executor_kind::matrix_free_executor
        });

        out.push_back({
            .name = "strict/matrix-free/red-black/block8/fused/jacobi/canonical",
            .storage = storage_kind::matrix_free_stencil,
            .decomposition = decomposition_kind::blocked_rows,
            .colouring = colouring_kind::red_black_stencil,
            .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
            .threading = threading_kind::colour_phases,
            .simd = simd_kind::lanes8,
            .fusion = fusion_kind::row_local_fused,
            .executor = executor_kind::matrix_free_executor
        });

        out.push_back({
            .name = "solver-family/matrix-free/no-preconditioner",
            .contract = contract_level::solver_family,
            .storage = storage_kind::matrix_free_stencil,
            .decomposition = decomposition_kind::recursive_grid_bisection,
            .preconditioner = preconditioner_kind::none_solver_family,
            .executor = executor_kind::matrix_free_executor
        });
    }

    out.push_back({
        .name = "illegal/coloured-smoother-without-colouring",
        .contract = contract_level::solver_family,
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::flat_rows,
        .colouring = colouring_kind::none,
        .preconditioner = preconditioner_kind::coloured_smoother_solver_family,
        .executor = executor_kind::matrix_free_executor
    });

    out.push_back({
        .name = "illegal/fused-thread-local-unordered-rho-merge",
        .storage = storage_kind::csr,
        .decomposition = decomposition_kind::blocked_rows,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes4,
        .fusion = fusion_kind::row_local_fused,
        .reduction = reduction_kind::thread_local_unordered_witness,
        .executor = executor_kind::csr_executor
    });

    return out;
}

inline expression_contract contract_for_plan(const expression_contract& base, contract_level level) {
    auto c = base;
    c.level = level;
    c.fixed_preconditioner_binding = level == contract_level::strict_expression;
    return c;
}

} // namespace ctdp::spmv_dsl
