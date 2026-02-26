# CT-DP Examples

## Original Examples (use framework headers)

| Example | Solver | Key lesson |
|---------|--------|------------|
| `matrix_chain_demo` | `interval_dp` | Classic DP, compile-time + runtime, Cormen Ch.15 |
| `per_element_demo` | `per_element_argmin` + `select_and_run` | Factored optimisation, additive cost, dynamic constraints |
| `fix_cacheline` | Consteval subset DP | Cache-line layout for FIX protocol, 1.39× on hot path |
| `spmv_benchmark` | Graph pipeline | Compile-time format selection, runtime validation |
| `variadic_reduce` | Tuple algebra | 7-lane single-pass statistics (P3666R2) |
| `gemm_tile_optimisation` | ML cost model | Calibration: preprocess → train → search |

## Standalone Examples (pure C++20, no framework imports)

These demonstrate the CT-DP **pattern** — constexpr search → NTTP dispatch → zero-cost executor — without importing framework types. They prove the architecture works and serve as the specification for what the framework must support.

| Example | Dims | Space | Speedup | Key lesson |
|---------|------|-------|---------|------------|
| `standalone/loop_nest_opt` | 5 (order, simd, isa, fma, aligned) | 192→92 feasible | 47× | Cross-dimension constraints: SIMD validity depends on loop order |
| `standalone/dep_break_opt` | 3 (strategy, unroll, accum) | 105 | 3.3× | Same space, 3 instances, 3 different optimal strategies |
| `standalone/cache_size_opt` | 1 (cache_size 1–8) | 8 | varies | Cost-as-simulation; analytical model ≠ runtime reality |

**Build requirements:** AVX2 + FMA (standalone examples use `<immintrin.h>`)

## Framework Example (uses `ctdp_space`)

| Example | Dims | Space | Speedup | Key lesson |
|---------|------|-------|---------|------------|
| `framework/cache_line_layout` | 4 policy dims | 270 | 1.42× | Framework searches cost model weights; subset DP solves the partition |

**Build note:** Needs `-fconstexpr-ops-limit=500000000` (GCC) because 270 DP evaluations at compile time exceed default limits.

## Common Pattern

All examples follow the same pipeline, whether hand-rolled or framework-based:

| Step | What | Standalone | Framework |
|------|------|-----------|-----------|
| 1. Space | Define searchable dimensions | `enum` + `struct config_point` | `descriptor_space(make_enum_vals(...), ...)` |
| 2. Validity | Filter illegal combinations | `is_valid(pt, instance)` | `filter_valid(space, predicate)` |
| 3. Search | Find optimum | `constexpr for` loops | `exhaustive_search_with_cost(space, cost)` |
| 4. Dispatch | Result → template args | `using opt = Executor<cfg>` | `using opt = dispatch<Executor, best>` |
| 5. Execute | Run specialised code | `opt::execute(...)` | `opt::execute(...)` |

## Building

All examples build with the main CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCTDP_BUILD_EXAMPLES=ON
cmake --build build
```

Run individually:

```bash
./build/loop_nest_opt        # 47× speedup on 512×512 matmul
./build/dep_break_opt        # 3 dependency instances
./build/cache_size_opt       # 3 cache instances
./build/cache_line_layout    # Policy search wrapping DP
```

## Checklist

See `CHECKLIST.md` for the 7-point gate every example must pass before commit.
