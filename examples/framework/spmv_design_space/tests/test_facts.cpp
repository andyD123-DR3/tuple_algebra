#include "spmv_design_space/facts.hpp"

#include "test_support.hpp"
#include <iostream>

int main() {
    using namespace ctdp::spmv_dsl;

    const auto problem = make_stencil_problem(8, 6);
    const auto facts = analyse_problem(problem);

    SPMV_REQUIRE(facts.rows == 48);
    SPMV_REQUIRE(facts.cols == 48);
    SPMV_REQUIRE(facts.square);
    SPMV_REQUIRE(facts.stencil_like);
    SPMV_REQUIRE(facts.diagonal_preconditioner_available);
    SPMV_REQUIRE(facts.min_nnz_per_row >= 3);
    SPMV_REQUIRE(facts.max_nnz_per_row <= 5);
    SPMV_REQUIRE(facts.connected_components == 1);
    SPMV_REQUIRE(facts.estimated_colour_count == 2);
    SPMV_REQUIRE(facts.kind == problem_kind::stencil_2d);
    SPMV_REQUIRE(!facts.banded);
    SPMV_REQUIRE(!facts.tridiagonal);
    SPMV_REQUIRE(facts.lower_bandwidth == 8);
    SPMV_REQUIRE(facts.upper_bandwidth == 8);
    SPMV_REQUIRE(facts.num_diagonals == 5);
    SPMV_REQUIRE(verify_red_black_colouring(problem));

    const auto banded = make_tridiagonal_banded_problem(16);
    const auto banded_facts = analyse_problem(banded);
    SPMV_REQUIRE(banded.kind == problem_kind::tridiagonal_banded_1d);
    SPMV_REQUIRE(banded_facts.kind == problem_kind::tridiagonal_banded_1d);
    SPMV_REQUIRE(banded_facts.rows == 16);
    SPMV_REQUIRE(banded_facts.nnz == 46);
    SPMV_REQUIRE(banded_facts.stencil_like);
    SPMV_REQUIRE(banded_facts.banded);
    SPMV_REQUIRE(banded_facts.tridiagonal);
    SPMV_REQUIRE(banded_facts.lower_bandwidth == 1);
    SPMV_REQUIRE(banded_facts.upper_bandwidth == 1);
    SPMV_REQUIRE(banded_facts.num_diagonals == 3);
    SPMV_REQUIRE(banded_facts.max_nnz_per_row == 3);
    SPMV_REQUIRE(verify_red_black_colouring(banded));

    std::cout << "fact tests PASS\n";
}
