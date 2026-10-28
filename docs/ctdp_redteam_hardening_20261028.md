# CT-DP red-team hardening — 28 October 2026

This record separates **implemented P0 fixes** from broader research
limitations. A green CI run qualifies the enumerated tests only; it is not
a proof that arbitrary plans are numerically equivalent.

## Concrete corrections in PR #42

1. **Critical path:** longest-path recurrence advances in topological order,
   so relabelling a DAG's vertices cannot invalidate a single-pass relaxation.
2. **Fusion capacity:** eligibility connectivity uses disjoint-set union
   rather than adding a reverse edge to a MaxE-capacity directed graph.
   Group legality still checks that coarsening yields a DAG.
3. **Segmented results:** reject infeasible segments, NaN/infinite costs and
   inconsistent sums. Optional schedule/resource validation checks global
   structural constraints. It explicitly does not prove that domain-specific
   strategy combinations preserve observations.
4. **Cost admission:** generic argmin, legacy exhaustive search, Pareto and
   direct selectors reject invalid/nonfinite model outputs. Beam and local
   search do not treat NaN as a valid objective.
5. **Strict reduction search:** by default the space does *not* enable
   reassociation or identity-padded SIMD based only on mathematical traits.
   Broader policies require explicit opt-in; a numeric certificate for a
   particular physical lowering remains an executor responsibility.
6. **Absent conditional dimensions:** schema-v2 features append a
   presence mask. The earlier all-zero encoding silently conflated
   "inactive" with "active, selected false/zero". Old cost-model features
   and wisdom must be regenerated.
7. **Test coverage:** the default CTest suite now registers previously
   dormant tree-space, reduction-tree, Pareto, adversarial graph and engine
   tests. It includes counterexample-driven static_assert and runtime tests.
   Standalone assertion tests keep assertions active in Release.
8. **Beam-search claims:** narrowed beam pruning is explicitly heuristic;
   the absence of a feasible survivor does not imply the global feasible
   set is empty. Counterexample tests preserve that limitation.

## Deliberately unresolved research questions

- A first-class typed rewrite-certificate catalogue is not yet in place.
  Graph tags and the four-bit reduction fingerprint are structural
  *co-traversal* hints, not full numeric equivalence witnesses.
- General search over sequences of enabling transformations (sort, record
  order, process, restore order), with correct propagation of properties
  and loss of previous properties, remains research work.
- Interval/subset DP optimality requires a sufficient boundary state and
  correct separable costs; no global guarantee is implied for arbitrary
  memory, register, storage or observation dependencies.
- Learned cost selection still requires held-out shapes, measured shortlist
  decisions, and calibration/compilation amortisation. A regression in
  model-only prediction cannot be repaired by treating it as a proof.
- A compiler-proof story across arbitrary user-defined code requires
  explicit validation obligations and likely dedicated translation
  validation; none is claimed here.

## Admission policy

Only a candidate with a real lowering, explicit contract and required
numerical checks should be presented as an admitted executable algorithm.
A search-space point by itself is **not** proof that the transformation is
semantically legal. Do not merge this branch without GCC, Clang and MSVC
CI passing on its final SHA.
