#pragma once

#include "spmv_design_space/problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <vector>

namespace ctdp::spmv_dsl {

struct sparse_facts {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz = 0;
    bool square = false;
    bool stencil_like = false;
    bool diagonal_preconditioner_available = false;
    bool has_irregular_remainder = false;
    std::size_t irregular_remainder_nnz = 0;
    std::size_t min_nnz_per_row = 0;
    std::size_t max_nnz_per_row = 0;
    double mean_nnz_per_row = 0.0;
    double row_length_variance = 0.0;
    std::size_t connected_components = 0;
    std::size_t estimated_colour_count = 0;
};

inline sparse_facts analyse_problem(const stencil_problem& p) {
    sparse_facts f;
    f.rows = p.csr.rows;
    f.cols = p.csr.cols;
    f.nnz = p.csr.values.size();
    f.square = f.rows == f.cols;
    f.stencil_like = true;
    f.diagonal_preconditioner_available = true;
    f.has_irregular_remainder = p.has_irregular_remainder();
    f.irregular_remainder_nnz = p.irregular_remainder_nnz;

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
