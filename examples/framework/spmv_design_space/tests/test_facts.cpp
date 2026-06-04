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
    SPMV_REQUIRE(facts.irregular_remainder_nnz == 0);
    SPMV_REQUIRE(!facts.has_irregular_remainder);
    SPMV_REQUIRE(verify_red_black_colouring(problem));

    const auto hybrid_problem = make_stencil_with_remainder_problem(8, 6, 5);
    const auto hybrid_facts = analyse_problem(hybrid_problem);
    SPMV_REQUIRE(hybrid_facts.stencil_like);
    SPMV_REQUIRE(hybrid_facts.has_irregular_remainder);
    SPMV_REQUIRE(hybrid_facts.irregular_remainder_nnz > 0);
    SPMV_REQUIRE(hybrid_facts.nnz == facts.nnz + hybrid_facts.irregular_remainder_nnz);

    std::cout << "fact tests PASS\n";
}
