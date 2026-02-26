# Loop Nest Optimisation

**Type:** Standalone (no framework imports)
**Dimensions:** 5 (loop_order × simd_strategy × isa_level × use_fma × aligned)
**Space:** 192 total → 92 feasible after constraint filtering
**Speedup:** ~47× (AVX2+FMA IKJ vs scalar IJK on 512×512 matmul)

## What It Demonstrates

The most complete exercise of the CT-DP pipeline. Five enumerated dimensions with cross-dimension constraints: INNERMOST_SIMD requires the innermost loop dimension (determined by `order`) to be vectorisable (a property of the instance). The constraint couples `order` and `simd` — you can't evaluate SIMD validity without knowing which dimension landed innermost.

## Pipeline Steps

1. **Space** — 6 loop orders × 4 SIMD strategies × 2 ISA levels × 2 FMA × 2 aligned = 192 points
2. **Validity** — `is_valid(pt, inst)` filters: INNERMOST_SIMD needs vectorisable innermost dim; FMA requires AVX2; aligned requires non-scalar SIMD. Reduces 192 → 92 feasible.
3. **Search** — constexpr exhaustive search with hand-written cost function (stride penalty, SIMD bonus, L1 fit bonus, FMA bonus)
4. **Dispatch** — `config_point` struct as NTTP → `matmul_executor<ct_cfg>`
5. **Execute** — AVX2 FMA inner loop on j dimension with aligned loads/stores

## Instance

```
loop_instance matmul_3d = {
    .size         = {512, 512, 512},
    .stride       = {512, 1, 512},     // j has unit stride
    .vectorisable = {false, true, false},
    .alignment    = {32, 32, 32},
    .L1_size      = 32768,
    .simd_width   = 8
};
```

The instance makes j (dimension 1) the only vectorisable dimension with unit stride. This means only loop orders that place j innermost can use INNERMOST_SIMD: IKJ and KIJ.

## Bug Found During Development

The original hand-rolled demo computed order and SIMD strategy independently. It selected IKJ (j innermost) but evaluated OUTER_SIMD by checking the original dimension 2 (k, non-vectorisable) rather than the reordered innermost (j, vectorisable). The CT-DP formulation prevents this: the constraint sees the complete point (order + strategy together).

## Safety Enforcement

`static_assert(d2 == 1)` in the SIMD branch ensures that the INNERMOST_SIMD executor code path only compiles when j is actually innermost. The constraint already prevents INNERMOST_SIMD when j isn't innermost, but the executor enforces it explicitly — belt and braces.

## Checklist Compliance

1. **No dead knobs** — every dimension changes generated code: order changes loop nesting, simd changes vectorisation, isa changes instruction width, fma changes multiply-add fusion, aligned changes load/store instructions
2. **No lying metadata** — total=192, feasible counted by enumeration
3. **Single solve** — one `solve()` function, called as `static constexpr auto ct_result = solve(matmul_3d)`
4. **Search matches executor** — INNERMOST_SIMD has AVX2 branch; scalar fallback for others; `static_assert(d2 == 1)` guards the SIMD path
5. **Constraints structural** — `is_valid()` rejects incoherent (order, simd) pairs before cost evaluation
6. **Bridge from same space** — N/A (standalone, no bridge)
7. **Non-trivial correctness** — non-uniform inputs `(7*i+3)%64 * 0.01`, scalar reference comparison, max error reported

## Build

```bash
g++ -std=c++20 -O2 -mavx2 -mfma standalone/loop_nest_opt.cpp -o loop_nest_opt
./loop_nest_opt
```
