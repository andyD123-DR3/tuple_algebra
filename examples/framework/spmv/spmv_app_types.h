// spmv_app_types.h — Application-level matrix types, builders, and kernels
// for the CT-DP SpMV tridiagonal demo.
//
// These are the demo's OWN types, independent of the existing
// spmv_benchmark.cc types. The demo reuses the framework's structural
// analysis layer (spmv_graph.h) but owns everything at the application
// level: matrix representations, format conversion, and kernels.
//
// Part of the CT-DP framework — examples/framework/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_EXAMPLES_SPMV_APP_TYPES_H
#define CTDP_EXAMPLES_SPMV_APP_TYPES_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// Framework structural types (reused, not owned)
#include "ctdp/domain/spmv/spmv_graph.h"

namespace ctdp::demo::spmv {

// ============================================================================
// Coefficient payload — separate from structure
// ============================================================================

struct tridiag_values {
    double cm1 = -1.0;
    double c0  =  2.0;
    double cp1 = -1.0;
};

// ============================================================================
// Application-level CSR representation
// ============================================================================

struct demo_csr_matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz  = 0;
    std::vector<std::size_t> row_ptr;
    std::vector<std::size_t> col_idx;
    std::vector<double>      val;
};

// ============================================================================
// Application-level DIA representation
// Sized by actual active diagonal count, not full 2n universe.
// Storage layout: data[d * rows + r] for diagonal d, row r.
// ============================================================================

struct demo_dia_matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t num_diags = 0;
    std::vector<int>    offsets;   // active diagonal offsets, sorted
    std::vector<double> data;     // num_diags × rows, column-major
};

// ============================================================================
// Builders: sparse_pattern + tridiag_values → application matrix types
//
// These are one-directional construction functions.
// They do NOT mutate the structural pattern.
// ============================================================================

/// Build a demo_csr_matrix from a framework sparse_pattern and tridiagonal
/// coefficient values. The pattern provides structure; values provide numerics.
template<std::size_t MaxR, std::size_t MaxNNZ>
demo_csr_matrix build_csr(
    ctdp::graph::sparse_pattern<MaxR, MaxNNZ> const& pat,
    tridiag_values const& v)
{
    demo_csr_matrix m;
    m.rows = pat.rows;
    m.cols = pat.cols;
    m.nnz  = pat.nnz;
    m.row_ptr.resize(pat.rows + 1);
    m.col_idx.resize(pat.nnz);
    m.val.resize(pat.nnz);

    for (std::size_t i = 0; i <= pat.rows; ++i) {
        m.row_ptr[i] = pat.row_ptr[i];
    }

    for (std::size_t r = 0; r < pat.rows; ++r) {
        for (std::size_t k = pat.row_ptr[r]; k < pat.row_ptr[r + 1]; ++k) {
            std::size_t c = pat.col_idx[k];
            m.col_idx[k] = c;

            // Assign coefficient based on diagonal position
            if (c + 1 == r)       m.val[k] = v.cm1;   // sub-diagonal
            else if (c == r)      m.val[k] = v.c0;    // main diagonal
            else if (c == r + 1)  m.val[k] = v.cp1;   // super-diagonal
            else                  m.val[k] = 0.0;     // should not happen for tridiag
        }
    }

    return m;
}

/// Build a demo_dia_matrix from a framework sparse_pattern and tridiagonal
/// coefficient values. Discovers active diagonals from the pattern.
template<std::size_t MaxR, std::size_t MaxNNZ>
demo_dia_matrix build_dia(
    ctdp::graph::sparse_pattern<MaxR, MaxNNZ> const& pat,
    tridiag_values const& v)
{
    demo_dia_matrix m;
    m.rows = pat.rows;
    m.cols = pat.cols;

    // Discover active diagonal offsets from the pattern
    std::vector<int> offsets;
    for (std::size_t r = 0; r < pat.rows; ++r) {
        for (std::size_t k = pat.row_ptr[r]; k < pat.row_ptr[r + 1]; ++k) {
            int off = static_cast<int>(pat.col_idx[k]) - static_cast<int>(r);
            if (std::find(offsets.begin(), offsets.end(), off) == offsets.end()) {
                offsets.push_back(off);
            }
        }
    }
    std::sort(offsets.begin(), offsets.end());

    m.num_diags = offsets.size();
    m.offsets   = offsets;
    m.data.assign(m.num_diags * m.rows, 0.0);

    // Fill diagonal values from pattern + coefficients
    for (std::size_t r = 0; r < pat.rows; ++r) {
        for (std::size_t k = pat.row_ptr[r]; k < pat.row_ptr[r + 1]; ++k) {
            int off = static_cast<int>(pat.col_idx[k]) - static_cast<int>(r);
            std::size_t c = pat.col_idx[k];

            // Determine coefficient
            double coeff = 0.0;
            if (c + 1 == r)       coeff = v.cm1;
            else if (c == r)      coeff = v.c0;
            else if (c == r + 1)  coeff = v.cp1;

            // Find diagonal index and store
            for (std::size_t d = 0; d < m.num_diags; ++d) {
                if (m.offsets[d] == off) {
                    m.data[d * m.rows + r] = coeff;
                    break;
                }
            }
        }
    }

    return m;
}

// ============================================================================
// SpMV kernels — the demo's own implementations
// ============================================================================

/// CSR SpMV: y = A * x
inline void spmv_exec_csr(
    demo_csr_matrix const& A,
    double const* x,
    double* y)
{
    for (std::size_t r = 0; r < A.rows; ++r) {
        double sum = 0.0;
        for (std::size_t k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k) {
            sum += A.val[k] * x[A.col_idx[k]];
        }
        y[r] = sum;
    }
}

/// DIA SpMV: y = A * x
/// No index indirection — stride-1 access on data[] and y[].
inline void spmv_exec_dia(
    demo_dia_matrix const& A,
    double const* x,
    double* y)
{
    for (std::size_t r = 0; r < A.rows; ++r) {
        y[r] = 0.0;
    }

    for (std::size_t d = 0; d < A.num_diags; ++d) {
        int const off = A.offsets[d];
        std::size_t r_begin = (off >= 0) ? 0 : static_cast<std::size_t>(-off);
        std::size_t r_end   = (off >= 0)
            ? std::min(A.rows, A.cols - static_cast<std::size_t>(off))
            : A.rows;

        for (std::size_t r = r_begin; r < r_end; ++r) {
            y[r] += A.data[d * A.rows + r]
                  * x[r + static_cast<std::size_t>(off)];
        }
    }
}

} // namespace ctdp::demo::spmv

#endif // CTDP_EXAMPLES_SPMV_APP_TYPES_H

