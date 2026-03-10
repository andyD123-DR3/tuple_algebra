## apply_executor_bindings_fix.ps1
## Run from the repo root: D:\tuple
## powershell -ExecutionPolicy Bypass -File apply_executor_bindings_fix.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if (-not $root) { $root = Get-Location }

Write-Host "Applying executor bindings fix from: $root"

# ============================================================================
# 1. Create spmv_executor_bindings.h
# ============================================================================
$bindingsPath = Join-Path $root "examples\framework\spmv\spmv_executor_bindings.h"
$bindingsContent = @'
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
'@
Set-Content -Path $bindingsPath -Value $bindingsContent -Encoding UTF8
Write-Host "  Created: $bindingsPath"

# ============================================================================
# 2. Fix spmv_bench.h — replace two includes with one bindings include
# ============================================================================
$benchPath = Join-Path $root "examples\framework\spmv\spmv_bench.h"
$bench = Get-Content $benchPath -Raw
$bench = $bench -replace '#include "ctdp/domain/spmv/spmv_schema.h"\r?\n#include "spmv_app_types.h"', '#include "spmv_executor_bindings.h"'
Set-Content -Path $benchPath -Value $bench -Encoding UTF8 -NoNewline
Write-Host "  Fixed:   $benchPath"

# ============================================================================
# 3. Fix test_spmv_schema.cpp — add bindings include, remove local specialisations
# ============================================================================
$schemaTestPath = Join-Path $root "tests\spmv\test_spmv_schema.cpp"
$schemaTest = Get-Content $schemaTestPath -Raw

# Add bindings include after spmv_app_types.h
$schemaTest = $schemaTest -replace '(#include "spmv_app_types.h")', "`$1`r`n#include ""spmv_executor_bindings.h"""

# Remove the local executor specialisation block
$pattern = '(?s)// =+\r?\n// Wire the demo''s typed executors.*?} // namespace ctdp::domain::spmv\r?\n'
$schemaTest = $schemaTest -replace $pattern, ''

Set-Content -Path $schemaTestPath -Value $schemaTest -Encoding UTF8 -NoNewline
Write-Host "  Fixed:   $schemaTestPath"

# ============================================================================
# 4. Fix test_spmv_bench.cpp — add bindings include, remove local specialisations
# ============================================================================
$benchTestPath = Join-Path $root "tests\spmv\test_spmv_bench.cpp"
$benchTest = Get-Content $benchTestPath -Raw

# Add bindings include after spmv_app_types.h
$benchTest = $benchTest -replace '(#include "spmv_app_types.h")', "`$1`r`n#include ""spmv_executor_bindings.h"""

# Remove the local executor specialisation block
$pattern2 = '(?s)// =+\r?\n// Executor specialisations.*?} // namespace ctdp::domain::spmv\r?\n'
$benchTest = $benchTest -replace $pattern2, ''

Set-Content -Path $benchTestPath -Value $benchTest -Encoding UTF8 -NoNewline
Write-Host "  Fixed:   $benchTestPath"

# ============================================================================
# 5. Fix spmv_tridiag_demo.cpp — add bindings include, remove local specialisations
# ============================================================================
$demoPath = Join-Path $root "examples\framework\spmv\spmv_tridiag_demo.cpp"
$demo = Get-Content $demoPath -Raw

# Add bindings include after spmv_app_types.h
$demo = $demo -replace '(#include "spmv_app_types.h")', "`$1`r`n#include ""spmv_executor_bindings.h"""

# Remove the local executor specialisation block
$pattern3 = '(?s)// =+\r?\n// Wire the demo''s typed executors.*?} // namespace ctdp::domain::spmv\r?\n'
$demo = $demo -replace $pattern3, ''

Set-Content -Path $demoPath -Value $demo -Encoding UTF8 -NoNewline
Write-Host "  Fixed:   $demoPath"

# ============================================================================
# Done
# ============================================================================
Write-Host ""
Write-Host "All changes applied. Now run:"
Write-Host "  git add examples/framework/spmv/spmv_executor_bindings.h examples/framework/spmv/spmv_bench.h examples/framework/spmv/spmv_tridiag_demo.cpp tests/spmv/test_spmv_schema.cpp tests/spmv/test_spmv_bench.cpp"
Write-Host "  git commit -m ""Fix: centralize SpMV executor bindings"""
Write-Host "  git push"
