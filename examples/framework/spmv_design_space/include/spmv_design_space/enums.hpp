#pragma once

#include <string_view>

namespace ctdp::spmv_dsl {

enum class contract_level {
    strict_expression,
    expression_family,
    solver_family,
    backend_defined,
    unchecked
};

enum class storage_kind {
    csr,
    matrix_free_stencil
};

enum class decomposition_kind {
    flat_rows,
    blocked_rows,
    recursive_grid_bisection
};

enum class ordering_kind {
    natural,
    rcm_descriptor
};

enum class colouring_kind {
    none,
    red_black_stencil,
    greedy_descriptor
};

enum class preconditioner_kind {
    fixed_diagonal_jacobi,
    none_solver_family,
    coloured_smoother_solver_family
};

enum class threading_kind {
    serial,
    static_blocks,
    recursive_tasks,
    colour_phases
};

enum class simd_kind {
    scalar,
    lanes4,
    lanes8
};

enum class reduction_kind {
    canonical_pairwise,
    thread_local_unordered_witness
};

enum class executor_kind {
    reference,
    csr_executor,
    matrix_free_executor
};

constexpr std::string_view to_string(contract_level v) noexcept {
    switch (v) {
    case contract_level::strict_expression: return "strict_expression";
    case contract_level::expression_family: return "expression_family";
    case contract_level::solver_family: return "solver_family";
    case contract_level::backend_defined: return "backend_defined";
    case contract_level::unchecked: return "unchecked";
    }
    return "unknown";
}

constexpr std::string_view to_string(storage_kind v) noexcept {
    switch (v) {
    case storage_kind::csr: return "csr";
    case storage_kind::matrix_free_stencil: return "matrix_free_stencil";
    }
    return "unknown";
}

constexpr std::string_view to_string(decomposition_kind v) noexcept {
    switch (v) {
    case decomposition_kind::flat_rows: return "flat_rows";
    case decomposition_kind::blocked_rows: return "blocked_rows";
    case decomposition_kind::recursive_grid_bisection: return "recursive_grid_bisection";
    }
    return "unknown";
}

constexpr std::string_view to_string(ordering_kind v) noexcept {
    switch (v) {
    case ordering_kind::natural: return "natural";
    case ordering_kind::rcm_descriptor: return "rcm_descriptor";
    }
    return "unknown";
}

constexpr std::string_view to_string(colouring_kind v) noexcept {
    switch (v) {
    case colouring_kind::none: return "none";
    case colouring_kind::red_black_stencil: return "red_black_stencil";
    case colouring_kind::greedy_descriptor: return "greedy_descriptor";
    }
    return "unknown";
}

constexpr std::string_view to_string(preconditioner_kind v) noexcept {
    switch (v) {
    case preconditioner_kind::fixed_diagonal_jacobi: return "fixed_diagonal_jacobi";
    case preconditioner_kind::none_solver_family: return "none_solver_family";
    case preconditioner_kind::coloured_smoother_solver_family: return "coloured_smoother_solver_family";
    }
    return "unknown";
}

constexpr std::string_view to_string(threading_kind v) noexcept {
    switch (v) {
    case threading_kind::serial: return "serial";
    case threading_kind::static_blocks: return "static_blocks";
    case threading_kind::recursive_tasks: return "recursive_tasks";
    case threading_kind::colour_phases: return "colour_phases";
    }
    return "unknown";
}

constexpr std::string_view to_string(simd_kind v) noexcept {
    switch (v) {
    case simd_kind::scalar: return "scalar";
    case simd_kind::lanes4: return "lanes4";
    case simd_kind::lanes8: return "lanes8";
    }
    return "unknown";
}

constexpr std::string_view to_string(reduction_kind v) noexcept {
    switch (v) {
    case reduction_kind::canonical_pairwise: return "canonical_pairwise";
    case reduction_kind::thread_local_unordered_witness: return "thread_local_unordered_witness";
    }
    return "unknown";
}

constexpr std::string_view to_string(executor_kind v) noexcept {
    switch (v) {
    case executor_kind::reference: return "reference";
    case executor_kind::csr_executor: return "csr_executor";
    case executor_kind::matrix_free_executor: return "matrix_free_executor";
    }
    return "unknown";
}

} // namespace ctdp::spmv_dsl
