// spmv_correctness.h — Correctness validation for the CT-DP SpMV demo.
//
// Three-layer validation:
//   Layer 1: Deterministic input generation
//   Layer 2: Cross-format agreement
//   Layer 3: Analytic spot-check (known result for {-1,2,-1} on linear input)
//
// Part of the CT-DP framework — examples/framework/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_EXAMPLES_SPMV_CORRECTNESS_H
#define CTDP_EXAMPLES_SPMV_CORRECTNESS_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace ctdp::demo::spmv {

// ============================================================================
// Result type
// ============================================================================

struct correctness_result {
    bool   ok           = true;
    double max_abs_diff = 0.0;
};

// ============================================================================
// Layer 1: Deterministic input generation
// ============================================================================

/// Generate input vector x_i = i + 1 (closed-form, deterministic).
/// This specific choice makes the analytic check possible.
inline std::vector<double> make_input_vector(std::size_t n) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i + 1);
    }
    return x;
}

// ============================================================================
// Layer 2: Cross-format agreement
// ============================================================================

/// Compare two output vectors element-wise.
/// Returns ok=true if max absolute difference is within tolerance.
inline correctness_result compare_outputs(
    double const* y_a,
    double const* y_b,
    std::size_t n,
    double tol = 1e-12)
{
    correctness_result r;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = std::abs(y_a[i] - y_b[i]);
        if (diff > r.max_abs_diff) {
            r.max_abs_diff = diff;
        }
        if (diff > tol) {
            r.ok = false;
        }
    }
    return r;
}

// ============================================================================
// Layer 3: Analytic spot-check
//
// For the {-1, 2, -1} tridiagonal operator applied to x_i = i + 1:
//
//   Row 0:     y_0     =  2·1  - 1·2            =  0
//   Interior:  y_i     = -(i) + 2·(i+1) - (i+2) =  0
//   Row n-1:   y_{n-1} = -(n-1) + 2·n            =  n + 1
//
// All interior values and the first row are zero.
// Only the last row is nonzero: y_{n-1} = n + 1.
//
// This assumes the standard boundary pattern where:
//   Row 0 has entries at columns {0, 1}       (no sub-diagonal)
//   Row n-1 has entries at columns {n-2, n-1} (no super-diagonal)
// ============================================================================

inline correctness_result analytic_check(
    double const* y,
    std::size_t n,
    double tol = 1e-12)
{
    correctness_result r;

    // All rows except the last should be zero
    for (std::size_t i = 0; i < n - 1; ++i) {
        double diff = std::abs(y[i]);
        if (diff > r.max_abs_diff) {
            r.max_abs_diff = diff;
        }
        if (diff > tol) {
            r.ok = false;
        }
    }

    // Last row: y_{n-1} = n + 1
    double expected_last = static_cast<double>(n + 1);
    double diff_last = std::abs(y[n - 1] - expected_last);
    if (diff_last > r.max_abs_diff) {
        r.max_abs_diff = diff_last;
    }
    if (diff_last > tol) {
        r.ok = false;
    }

    return r;
}

} // namespace ctdp::demo::spmv

#endif // CTDP_EXAMPLES_SPMV_CORRECTNESS_H
