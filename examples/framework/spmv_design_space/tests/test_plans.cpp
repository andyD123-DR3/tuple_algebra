#include "spmv_design_space/search.hpp"

#include <algorithm>
#include "test_support.hpp"
#include <iostream>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(8, 8);
    const auto facts = analyse_problem(problem);
    const expression_contract strict{};
    const auto plans = generate_candidate_plans(facts);

    SPMV_REQUIRE(plans.size() >= 70);

    const auto has_dia = std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) {
        return p.storage == storage_kind::dia && p.executor == executor_kind::dia_executor;
    });
    SPMV_REQUIRE(has_dia);

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
        const auto has_partition = std::any_of(plans.begin(), plans.end(), [partition](const plan_descriptor& p) {
            return p.fusion == partition;
        });
        SPMV_REQUIRE(has_partition);
    }

    const auto has_all_widths =
        std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) { return p.simd == simd_kind::scalar; }) &&
        std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) { return p.simd == simd_kind::lanes4; }) &&
        std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) { return p.simd == simd_kind::lanes8; });
    SPMV_REQUIRE(has_all_widths);

    const auto has_recursive = std::any_of(plans.begin(), plans.end(), [](const plan_descriptor& p) {
        return p.decomposition == decomposition_kind::recursive_grid_bisection &&
               p.storage == storage_kind::matrix_free_stencil;
    });
    SPMV_REQUIRE(has_recursive);

    auto illegal_unordered = plan_descriptor{
        .name = "bad reduction",
        .storage = storage_kind::csr,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .reduction = reduction_kind::thread_local_unordered_witness,
        .executor = executor_kind::csr_executor
    };
    const auto bad_reduction = analyse_legality(strict, facts, illegal_unordered);
    SPMV_REQUIRE(!bad_reduction.legal());
    SPMV_REQUIRE(!bad_reduction.observation_legal);

    auto illegal_smoother = plan_descriptor{
        .name = "bad smoother",
        .contract = contract_level::solver_family,
        .storage = storage_kind::matrix_free_stencil,
        .decomposition = decomposition_kind::flat_rows,
        .ordering = ordering_kind::natural,
        .colouring = colouring_kind::none,
        .preconditioner = preconditioner_kind::coloured_smoother_solver_family,
        .executor = executor_kind::matrix_free_executor
    };
    auto solver_contract = contract_for_plan(strict, contract_level::solver_family);
    const auto bad_smoother = analyse_legality(solver_contract, facts, illegal_smoother);
    SPMV_REQUIRE(!bad_smoother.legal());
    SPMV_REQUIRE(!bad_smoother.structurally_legal);

    auto solver_none = plan_descriptor{
        .name = "solver family no preconditioner",
        .contract = contract_level::solver_family,
        .storage = storage_kind::matrix_free_stencil,
        .preconditioner = preconditioner_kind::none_solver_family,
        .executor = executor_kind::matrix_free_executor
    };
    const auto solver_ok = analyse_legality(solver_contract, facts, solver_none);
    SPMV_REQUIRE(solver_ok.legal());

    const auto strict_bad = analyse_legality(strict, facts, solver_none);
    SPMV_REQUIRE(!strict_bad.legal());
    SPMV_REQUIRE(!strict_bad.numerically_legal);

    const auto hybrid_problem = make_stencil_with_remainder_problem(8, 8, 7);
    const auto hybrid_facts = analyse_problem(hybrid_problem);
    const auto hybrid_plans = generate_candidate_plans(hybrid_facts);
    const auto has_hybrid = std::any_of(hybrid_plans.begin(), hybrid_plans.end(), [](const plan_descriptor& p) {
        return p.storage == storage_kind::dia_csr_remainder &&
               p.executor == executor_kind::hybrid_dia_csr_executor;
    });
    SPMV_REQUIRE(has_hybrid);

    plan_descriptor regular_dia{
        .name = "regular dia drops remainder",
        .storage = storage_kind::dia,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .executor = executor_kind::dia_executor
    };
    const auto dia_rejected = analyse_legality(strict, hybrid_facts, regular_dia);
    SPMV_REQUIRE(!dia_rejected.legal());
    SPMV_REQUIRE(!dia_rejected.structurally_legal);

    plan_descriptor hybrid_ok{
        .name = "hybrid dia csr",
        .storage = storage_kind::dia_csr_remainder,
        .preconditioner = preconditioner_kind::fixed_diagonal_jacobi,
        .executor = executor_kind::hybrid_dia_csr_executor
    };
    const auto hybrid_legal = analyse_legality(strict, hybrid_facts, hybrid_ok);
    SPMV_REQUIRE(hybrid_legal.legal());

    hardware_profile lanes4{};
    lanes4.max_simd_lanes = 4;
    plan_descriptor lanes8 = hybrid_ok;
    lanes8.simd = simd_kind::lanes8;
    const auto width_rejected = analyse_legality(strict, hybrid_facts, lanes8, lanes4);
    SPMV_REQUIRE(!width_rejected.legal());
    SPMV_REQUIRE(!width_rejected.target_legal);

    hardware_profile single_thread{};
    single_thread.max_worker_threads = 1;
    plan_descriptor threaded = hybrid_ok;
    threaded.threading = threading_kind::recursive_tasks;
    const auto threaded_rejected = analyse_legality(strict, hybrid_facts, threaded, single_thread);
    SPMV_REQUIRE(!threaded_rejected.legal());
    SPMV_REQUIRE(!threaded_rejected.target_legal);

    const auto wisdom = emit_plan_wisdom(hybrid_ok);
    SPMV_REQUIRE(wisdom.find("dia_csr_remainder") != std::string::npos);
    SPMV_REQUIRE(wisdom.find("hybrid_dia_csr_executor") != std::string::npos);

    std::cout << "plan tests PASS\n";
}
