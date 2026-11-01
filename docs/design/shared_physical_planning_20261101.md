# Shared physical planning consolidation — CT-DP

**Experimental library API**, CT-DP C++20; no numerical backend or
production dispatch changes.

## Why this lives in CT-DP

The qualified DN SpMV, market-price-level replay, and finite-field
Cooley–Tukey experiments independently exposed reusable search mechanics:
physical index maps, bounded recursion, fusion legality, multiobjective
subproblem DP, and economic stopping. These mechanisms now have one
CT-DP home rather than parallel algorithmic type systems.

This consolidation **reuses** existing CT-DP building blocks:

- ctdp::space::permutation_value<N> for statically ranked index-axis
  permutations (new typed-factor-space adapter);
- existing ctdp::cost_vector<D> for finite-model Pareto dominance;
- existing actual-lowering/evidence boundaries remain in domain users.
  A predicted cost or an asserted fusion witness is **not** a proof of
  numerical equivalence or a device qualification.

## Added library headers

- include/ctdp/space/index_geometry.h: checked 2D affine views,
  row/column-major/strided/offset and a bounded, bijective mixed-radix
  Cartesian logical-to-physical mapping. A static-rank adapter directly
  consumes CT-DP permutation_value; dynamic-rank clients remain supported.
  All products, physical extent, aliased strides, out-of-range access
  and malformed axis mappings fail closed.
- include/ctdp/solver/plans/physical_recursive.h: bounded source-order
  region recursion, complete source partition checking, explicit
  physical boundary-state/observation fusion witness and a domain-provided
  join. Retains conservative measured probe-affordability and post-
  admission amortisation gates. No implementation publishes partial plans.
- include/ctdp/solver/algorithms/compositional_pareto.h: generic memoised
  AND/OR DP with a domain-supplied expansion recurrence, CT-DP
  cost_vector<D>, stable tie order and optional priority. Candidate and
  subproblem budgets fail closed; frontier pruning preserves Pareto
  objective extrema or rejects inadequate budgets. Any truncated
  frontier explicitly reports approximate and budget_pruned.

## Correctness model

A structural property/precondition is permission to *attempt* a
physical rewrite, not an actual executed-numerical proof. The domain
lowerer must independently validate arithmetic, output identity,
hardware features, numeric precision/rounding, memory bounds,
representation transitions and complete canonical observations.

Global Pareto exactness requires a monotone compositional recurrence.
The generic DP can produce an exact model frontier under that condition
when its frontier is untruncated; it never promises that a model predicts
actual runtime or that FP algorithms with differing associativity are
bitwise equivalent.

These APIs intentionally retain C++20 header-only status and no
global runtime state. Nothing here invokes or substitutes CUDA/HIP/Metal
or other production code.

## Independent qualification

- Complete source-order partitions, invalid cuts and lowerers, finite
  recursion/leaf limits, explicit interface permission to fuse and
  measured search amortisation with adverse values.
- Strided 2D transpose, offset/padded views, mixed-radix {2,4,7}
  Cartesian permutations; physical scatter-gather and inverse maps,
  overflow, non-injective storage, invalid axis ordering.
- Model-independent exhaustive oracle for radix-2/4/7 factorisation
  Pareto fronts, including non-powers of two, stable tie handling,
  adversarial beam pruning with explicit extrema preservation,
  generation budgets, cyclic graphs and NaN/negative model admission.
- New standalone driver experiments/shared_physical_planning compiles
  without GoogleTest, under GCC 13, Clang 18, MSVC and native ARM64,
  ASan/UBSan and deterministic test-result comparisons.
- All three tests also registered in default CT-DP CTest; existing
  full CT-DP CI remains a separate compatibility gate.

## DN integration follows qualification

Once this CT-DP branch passes independently, update the pinned
tuple_algebra revision used by DN. Replace DN's duplicated index
geometry and generic recursion/economics implementation with thin
forwarding compatibility headers and delegate its NTT multiobjective
factorisation DP to the generic CT-DP compositional search.
Requalify *actual* DN SpMV, stateful replay and NTT executors,
including the exact GCC/ARM/MSVC fingerprints and all sanitizer cases.

The shared core will not import DN-specific headers or the domain's
NTT/SpMV mathematical details. The dependency direction is CT-DP
core -> domain adapters, never the reverse.
