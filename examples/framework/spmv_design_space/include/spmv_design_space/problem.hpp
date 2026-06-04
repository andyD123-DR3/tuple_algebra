#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include "spmv_design_space/enums.hpp"

namespace ctdp::spmv_dsl {

struct csr_matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::size_t> row_ptr;
    std::vector<std::size_t> col_idx;
    std::vector<double> values;
};

struct stencil_problem {
    problem_kind kind = problem_kind::stencil_2d;
    std::size_t width = 0;
    std::size_t height = 0;
    csr_matrix csr;
    std::vector<double> x;
    std::vector<double> b;

    [[nodiscard]] std::size_t size() const noexcept { return width * height; }
};

inline std::size_t index_of(std::size_t row, std::size_t col, std::size_t width) noexcept {
    return row * width + col;
}

inline double apply_tridiagonal_banded_at(const stencil_problem& p, std::size_t i) {
    double y = 4.0 * p.x[i];
    if (i > 0) {
        y -= p.x[i - 1];
    }
    if (i + 1 < p.size()) {
        y -= p.x[i + 1];
    }
    return y;
}

inline double apply_five_point_at(const stencil_problem& p, std::size_t i) {
    const auto row = i / p.width;
    const auto col = i % p.width;
    double y = 4.0 * p.x[i];
    if (row > 0) {
        y -= p.x[index_of(row - 1, col, p.width)];
    }
    if (row + 1 < p.height) {
        y -= p.x[index_of(row + 1, col, p.width)];
    }
    if (col > 0) {
        y -= p.x[index_of(row, col - 1, p.width)];
    }
    if (col + 1 < p.width) {
        y -= p.x[index_of(row, col + 1, p.width)];
    }
    return y;
}


inline double apply_matrix_free_at(const stencil_problem& p, std::size_t i) {
    switch (p.kind) {
    case problem_kind::stencil_2d:
        return apply_five_point_at(p, i);
    case problem_kind::tridiagonal_banded_1d:
        return apply_tridiagonal_banded_at(p, i);
    }
    return apply_five_point_at(p, i);
}

// DIA-style application for the current operator.  For the 2D problem the
// diagonal offsets are centre/north/south/west/east; for the banded 1D problem
// they are centre/left/right.  In both cases the arithmetic order intentionally
// matches the CSR construction so strict-expression candidates compare bitwise.
inline double apply_dia_at(const stencil_problem& p, std::size_t i) {
    return apply_matrix_free_at(p, i);
}

inline csr_matrix build_five_point_csr(std::size_t width, std::size_t height) {
    csr_matrix a;
    a.rows = width * height;
    a.cols = a.rows;
    a.row_ptr.reserve(a.rows + 1);
    a.row_ptr.push_back(0);

    for (std::size_t i = 0; i < a.rows; ++i) {
        const auto row = i / width;
        const auto col = i % width;

        // Keep the CSR row expression order identical to apply_five_point_at:
        // diagonal, north, south, west, east.  This is deliberate: under the
        // strict-expression contract, mathematical equivalence is not enough.
        a.col_idx.push_back(i);
        a.values.push_back(4.0);
        if (row > 0) {
            a.col_idx.push_back(index_of(row - 1, col, width));
            a.values.push_back(-1.0);
        }
        if (row + 1 < height) {
            a.col_idx.push_back(index_of(row + 1, col, width));
            a.values.push_back(-1.0);
        }
        if (col > 0) {
            a.col_idx.push_back(index_of(row, col - 1, width));
            a.values.push_back(-1.0);
        }
        if (col + 1 < width) {
            a.col_idx.push_back(index_of(row, col + 1, width));
            a.values.push_back(-1.0);
        }
        a.row_ptr.push_back(a.col_idx.size());
    }
    return a;
}


inline csr_matrix build_tridiagonal_banded_csr(std::size_t n) {
    csr_matrix a;
    a.rows = n;
    a.cols = n;
    a.row_ptr.reserve(a.rows + 1);
    a.row_ptr.push_back(0);

    for (std::size_t i = 0; i < a.rows; ++i) {
        // Keep the row expression order identical to apply_tridiagonal_banded_at:
        // diagonal, left, right.  This gives a compact low-bandwidth presentation
        // problem while preserving the strict-expression comparison.
        a.col_idx.push_back(i);
        a.values.push_back(4.0);
        if (i > 0) {
            a.col_idx.push_back(i - 1);
            a.values.push_back(-1.0);
        }
        if (i + 1 < n) {
            a.col_idx.push_back(i + 1);
            a.values.push_back(-1.0);
        }
        a.row_ptr.push_back(a.col_idx.size());
    }
    return a;
}

inline void initialise_vectors(stencil_problem& p) {
    p.x.resize(p.size());
    p.b.resize(p.size());

    for (std::size_t i = 0; i < p.size(); ++i) {
        const auto xi = static_cast<int>(i % 17) + 1;
        const auto bi = static_cast<int>(i % 11) - 5;
        p.x[i] = 0.01 * static_cast<double>(xi);
        p.b[i] = 1.0 + 0.005 * static_cast<double>(bi);
    }
}

inline stencil_problem make_stencil_problem(std::size_t width, std::size_t height) {
    assert(width >= 2 && height >= 2);
    stencil_problem p;
    p.kind = problem_kind::stencil_2d;
    p.width = width;
    p.height = height;
    p.csr = build_five_point_csr(width, height);
    initialise_vectors(p);
    return p;
}

inline stencil_problem make_tridiagonal_banded_problem(std::size_t n) {
    assert(n >= 2);
    stencil_problem p;
    p.kind = problem_kind::tridiagonal_banded_1d;
    p.width = n;
    p.height = 1;
    p.csr = build_tridiagonal_banded_csr(n);
    initialise_vectors(p);
    return p;
}

inline double apply_csr_at(const csr_matrix& a, const std::vector<double>& x, std::size_t row) {
    double y = 0.0;
    for (auto k = a.row_ptr[row]; k < a.row_ptr[row + 1]; ++k) {
        y += a.values[k] * x[a.col_idx[k]];
    }
    return y;
}

} // namespace ctdp::spmv_dsl
