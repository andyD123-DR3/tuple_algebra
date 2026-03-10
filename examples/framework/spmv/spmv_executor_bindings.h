// spmv_executor_bindings.h - Bind demo kernels to framework spmv_executor<F>.
//
// This header is the wiring seam between:
//   - framework-level plan/executor declarations (spmv_schema.h)
//   - demo-level matrix types and kernels (spmv_app_types.h)
//
// Part of the CT-DP framework - examples/framework/spmv/
// Copyright (c) 2026 Andrew Drakeford. All rights reserved.

#ifndef CTDP_EXAMPLES_SPMV_EXECUTOR_BINDINGS_H
#define CTDP_EXAMPLES_SPMV_EXECUTOR_BINDINGS_H

#include "ctdp/domain/spmv/spmv_schema.h"
#include "spmv_app_types.h"

namespace ctdp::domain::spmv {

template<>
struct spmv_executor<ctdp::graph::spmv_format::csr> {
    static void run(ctdp::demo::spmv::demo_csr_matrix const& m,
                    double const* x,
                    double* y) {
        ctdp::demo::spmv::spmv_exec_csr(m, x, y);
    }
};

template<>
struct spmv_executor<ctdp::graph::spmv_format::dia> {
    static void run(ctdp::demo::spmv::demo_dia_matrix const& m,
                    double const* x,
                    double* y) {
        ctdp::demo::spmv::spmv_exec_dia(m, x, y);
    }
};

} // namespace ctdp::domain::spmv

#endif // CTDP_EXAMPLES_SPMV_EXECUTOR_BINDINGS_H
