#pragma once

#include "spmv_design_space/problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <vector>

namespace ctdp::spmv_dsl {

struct sparse_facts {
    problem_kind kind = problem_kind::stencil_2d;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz = 0;
    bool square = false;
    bool stencil_like = false;
    bool banded = false;
    bool tridiagonal = false;
    std::size_t lower_bandwidth = 0;
    std::size_t upper_bandwidth = 0;
    std::size_t num_diagonals = 0;
    bool diagonal_preconditioner_available = false;
    std::size_t min_nnz_per_row = 0;
    std::size_t max_nnz_per_row = 0;
    double mean_nnz_per_row = 0.0;
    double row_length_variance = 0.0;
    std::size_t connected_components = 0;
    std::size_t estimated_colour_count = 0;
};

inline sparse_facts analyse_problem(const stencil_problem& p) {
    sparse_facts f;
    f.kind = p.kind;
    f.rows = p.csr.rows;
    f.cols = p.csr.cols;
    f.nnz = p.csr.values.size();
    f.square = f.rows == f.cols;
    f.stencil_like = true;
    f.banded = p.kind == problem_kind::tridiagonal_banded_1d;
    f.tridiagonal = p.kind == problem_kind::tridiagonal_banded_1d;
    f.diagonal_preconditioner_available = true;

    f.min_nnz_per_row = static_cast<std::size_t>(-1);
    f.max_nnz_per_row = 0;
    double sum = 0.0;
    double sum2 = 0.0;
    for (std::size_t r = 0; r < f.rows; ++r) {
        const auto len = p.csr.row_ptr[r + 1] - p.csr.row_ptr[r];
        f.min_nnz_per_row = std::min(f.min_nnz_per_row, len);
        f.max_nnz_per_row = std::max(f.max_nnz_per_row, len);
        sum += static_cast<double>(len);
        sum2 += static_cast<double>(len * len);
    }
    f.mean_nnz_per_row = sum / static_cast<double>(f.rows);
    f.row_length_variance = sum2 / static_cast<double>(f.rows) - f.mean_nnz_per_row * f.mean_nnz_per_row;

    std::vector<std::ptrdiff_t> offsets;
    offsets.reserve(f.nnz);
    for (std::size_t r = 0; r < f.rows; ++r) {
        for (auto k = p.csr.row_ptr[r]; k < p.csr.row_ptr[r + 1]; ++k) {
            const auto c = p.csr.col_idx[k];
            if (c <= r) {
                f.lower_bandwidth = std::max(f.lower_bandwidth, r - c);
            } else {
                f.upper_bandwidth = std::max(f.upper_bandwidth, c - r);
            }
            offsets.push_back(static_cast<std::ptrdiff_t>(c) - static_cast<std::ptrdiff_t>(r));
        }
    }
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    f.num_diagonals = offsets.size();

    f.connected_components = 1;
    f.estimated_colour_count = 2;
    return f;
}

inline std::vector<std::size_t> red_black_colours(const stencil_problem& p) {
    std::vector<std::size_t> colours(p.size());
    for (std::size_t r = 0; r < p.height; ++r) {
        for (std::size_t c = 0; c < p.width; ++c) {
            colours[index_of(r, c, p.width)] = (r + c) & 1U;
        }
    }
    return colours;
}

inline bool verify_red_black_colouring(const stencil_problem& p) {
    const auto colours = red_black_colours(p);
    for (std::size_t row = 0; row < p.csr.rows; ++row) {
        for (auto k = p.csr.row_ptr[row]; k < p.csr.row_ptr[row + 1]; ++k) {
            const auto col = p.csr.col_idx[k];
            if (col != row && colours[col] == colours[row]) {
                return false;
            }
        }
    }
    return true;
}

} // namespace ctdp::spmv_dsl
