#include "spmv_design_space/search.hpp"

#include "test_support.hpp"
#include <iostream>
#include <string>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(10, 10);
    const auto reference = execute_strict_reference(problem, 0.125);

    plan_descriptor csr{
        .name = "csr strict",
        .storage = storage_kind::csr,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .executor = executor_kind::csr_executor
    };
    const auto csr_result = execute_plan(problem, csr, 0.125);
    SPMV_REQUIRE(conforms_to_reference(csr_result, reference, csr));


    plan_descriptor dia{
        .name = "dia strict",
        .storage = storage_kind::dia,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes4,
        .executor = executor_kind::dia_executor
    };
    const auto dia_result = execute_plan(problem, dia, 0.125);
    SPMV_REQUIRE(conforms_to_reference(dia_result, reference, dia));
    SPMV_REQUIRE(dia_result.execution_path.find("dia_executor") != std::string::npos);

    plan_descriptor fused{
        .name = "matrix free fused strict",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::blocked_rows,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes8,
        .fusion = fusion_kind::rzpu,
        .executor = executor_kind::matrix_free_executor
    };
    const auto fused_result = execute_plan(problem, fused, 0.125);
    SPMV_REQUIRE(conforms_to_reference(fused_result, reference, fused));
    SPMV_REQUIRE(fused_result.execution_path.find("[RZPU]") != std::string::npos);

    const fusion_kind partitions[] = {
        fusion_kind::r_z_p_u,
        fusion_kind::rz_p_u,
        fusion_kind::r_zp_u,
        fusion_kind::r_z_pu,
        fusion_kind::rzp_u,
        fusion_kind::rz_pu,
        fusion_kind::r_zpu,
        fusion_kind::rzpu
    };
    for (const auto partition : partitions) {
        plan_descriptor partition_plan{
            .name = "matrix free fusion partition strict",
            .storage = storage_kind::matrix_free_stencil,
            .decomposition = decomposition_kind::blocked_rows,
            .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
            .simd = simd_kind::lanes4,
            .fusion = partition,
            .executor = executor_kind::matrix_free_executor
        };
        const auto partition_result = execute_plan(problem, partition_plan, 0.125);
        SPMV_REQUIRE(conforms_to_reference(partition_result, reference, partition_plan));
    }

    plan_descriptor mf{
        .name = "matrix free strict",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::recursive_grid_bisection,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .executor = executor_kind::matrix_free_executor
    };
    const auto mf_result = execute_plan(problem, mf, 0.125);
    SPMV_REQUIRE(conforms_to_reference(mf_result, reference, mf));


    persistent_row_pool pool{2};
    execution_context ctx{.pool = &pool, .task_grain = 16};

    plan_descriptor red_black{
        .name = "matrix free red-black strict",
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::blocked_rows,
        .colouring = colouring_kind::red_black_stencil,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .threading = threading_kind::colour_phases,
        .simd = simd_kind::lanes8,
        .executor = executor_kind::matrix_free_executor
    };
    const auto rb_result = execute_plan(problem, red_black, 0.125, &ctx);
    SPMV_REQUIRE(conforms_to_reference(rb_result, reference, red_black));
    SPMV_REQUIRE(!rb_result.visited_rows.empty());
    SPMV_REQUIRE(rb_result.visited_rows[0] == 0);
    SPMV_REQUIRE(rb_result.visited_rows[1] != 1); // red-black phase order is not plain row-major order.
    SPMV_REQUIRE(rb_result.execution_path.find("red_black_stencil") != std::string::npos);
    SPMV_REQUIRE(rb_result.execution_path.find("colour_phases(2T)") != std::string::npos);
    SPMV_REQUIRE(rb_result.execution_path.find("lanes8(8 lanes)") != std::string::npos);
    SPMV_REQUIRE(rb_result.worker_count == 2);
    SPMV_REQUIRE(rb_result.simd_lanes == 8);

    const expression_contract contract{};
    search_options options;
    options.iterations = 5;
    options.warmup = 1;
    options.threads = 2;
    options.task_grain = 16;
    const auto results = run_design_space_search(problem, contract, 0.125, options);
    SPMV_REQUIRE(!results.empty());
    SPMV_REQUIRE(select_best_legal(results) != nullptr);

    bool saw_illegal = false;
    bool saw_legal_conforming = false;
    bool saw_dia_candidate = false;
    bool saw_fused_candidate = false;
    for (const auto& r : results) {
        saw_illegal = saw_illegal || !r.legality.legal();
        saw_dia_candidate = saw_dia_candidate || (r.plan.storage == storage_kind::dia && r.conformance_passed);
        saw_fused_candidate = saw_fused_candidate || (r.plan.fusion == fusion_kind::rzpu && r.conformance_passed);
        saw_legal_conforming = saw_legal_conforming || r.strict_conforming;
        if (r.strict_conforming) {
            SPMV_REQUIRE(r.executed);
            SPMV_REQUIRE(r.best_ns >= 0.0);
            SPMV_REQUIRE(r.median_ns >= 0.0);
            SPMV_REQUIRE(r.mean_ns >= 0.0);
        }
    }
    SPMV_REQUIRE(saw_illegal);
    SPMV_REQUIRE(saw_legal_conforming);
    SPMV_REQUIRE(saw_dia_candidate);
    SPMV_REQUIRE(saw_fused_candidate);

    search_options relaxed_options = options;
    relaxed_options.scope = candidate_scope::strict_and_relaxed_executable;
    const auto relaxed_results = run_design_space_search(problem, contract, 0.125, relaxed_options);
    SPMV_REQUIRE(select_best_legal(relaxed_results) != nullptr);
    SPMV_REQUIRE(select_fastest_executed(relaxed_results) != nullptr);
    SPMV_REQUIRE(select_fastest_non_strict_executed(relaxed_results) != nullptr);

    bool saw_executed_unordered_witness = false;
    bool saw_unexecuted_structural_rejection = false;
    for (const auto& r : relaxed_results) {
        if (r.plan.reduction == reduction_kind::thread_local_unordered_witness) {
            saw_executed_unordered_witness = saw_executed_unordered_witness ||
                (r.executed && r.relaxed_executable && !r.strict_conforming);
            SPMV_REQUIRE(!r.legality.legal());
            SPMV_REQUIRE(!r.legality.observation_legal);
        }
        if (r.plan.preconditioner == preconditioner_kind::coloured_smoother_solver_family) {
            saw_unexecuted_structural_rejection = saw_unexecuted_structural_rejection ||
                (!r.executed && !r.relaxed_executable);
        }
        if (r.executed && !r.strict_conforming) {
            SPMV_REQUIRE(r.best_ns >= 0.0);
            SPMV_REQUIRE(r.median_ns >= 0.0);
            SPMV_REQUIRE(r.x_next_max_abs_delta >= 0.0);
        }
    }
    SPMV_REQUIRE(saw_executed_unordered_witness);
    SPMV_REQUIRE(saw_unexecuted_structural_rejection);


    const auto banded_problem = make_tridiagonal_banded_problem(64);
    const auto banded_reference = execute_strict_reference(banded_problem, 0.125);
    const auto banded_facts = analyse_problem(banded_problem);
    SPMV_REQUIRE(banded_facts.banded);
    SPMV_REQUIRE(banded_facts.lower_bandwidth == 1);
    SPMV_REQUIRE(banded_facts.upper_bandwidth == 1);

    plan_descriptor banded_dia{
        .name = "banded dia strict",
        .storage = storage_kind::dia,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes8,
        .fusion = fusion_kind::rzpu,
        .executor = executor_kind::dia_executor
    };
    const auto banded_dia_result = execute_plan(banded_problem, banded_dia, 0.125);
    SPMV_REQUIRE(conforms_to_reference(banded_dia_result, banded_reference, banded_dia));

    plan_descriptor banded_mf{
        .name = "banded matrix-free strict",
        .storage = storage_kind::matrix_free_stencil,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .simd = simd_kind::lanes8,
        .fusion = fusion_kind::rzpu,
        .executor = executor_kind::matrix_free_executor
    };
    const auto banded_mf_result = execute_plan(banded_problem, banded_mf, 0.125);
    SPMV_REQUIRE(conforms_to_reference(banded_mf_result, banded_reference, banded_mf));

    const auto banded_results = run_design_space_search(banded_problem, contract, 0.125, options);
    SPMV_REQUIRE(select_best_legal(banded_results) != nullptr);
    bool saw_banded_dia = false;
    bool saw_banded_mf = false;
    for (const auto& r : banded_results) {
        saw_banded_dia = saw_banded_dia || (r.strict_conforming && r.plan.storage == storage_kind::dia);
        saw_banded_mf = saw_banded_mf || (r.strict_conforming && r.plan.storage == storage_kind::matrix_free_stencil);
    }
    SPMV_REQUIRE(saw_banded_dia);
    SPMV_REQUIRE(saw_banded_mf);

    const auto larger_problem = make_stencil_problem(64, 64);
    search_options large_options;
    large_options.iterations = 3;
    large_options.warmup = 1;
    large_options.threads = 2;
    large_options.task_grain = 512;
    const auto larger_results = run_design_space_search(larger_problem, contract, 0.125, large_options);
    SPMV_REQUIRE(select_best_legal(larger_results) != nullptr);
    bool saw_threaded_candidate = false;
    for (const auto& r : larger_results) {
        saw_threaded_candidate = saw_threaded_candidate ||
            (r.legality.legal() && uses_parallel_threading(r.plan.threading));
    }
    SPMV_REQUIRE(saw_threaded_candidate);

    std::cout << "executor tests PASS\n";
}
