# Sprint 9 Stage 2 — Interval Representation and Integration Plan

**Status:** Draft — superseded as the umbrella note by the follow-on ADR and execution/spec notes below.  
**Date:** May 20, 2026  
**Scope:** Stage 2 only (post-Stage-1 follow-on)  
**Related:** `docs/adr_partition_rooted_tree_canonical.md`, `docs/adr_interval_rooted_representation_family.md`, `docs/design/sprint9_stage1_interval_solver_slice.md`, `docs/design/interval_stage2_execution_plan.md`, `docs/design/phase_a_interval_rooted_candidate.md`, `include/ctdp/solver/algorithms/interval_solver.h`, `include/ctdp/solver/algorithms/interval_dp.h`

## Follow-on documents

This note now serves as the umbrella Stage 2 overview. The concrete follow-on documents are:

- `docs/adr_interval_rooted_representation_family.md` — ADR-style architectural decision for the interval-rooted sibling family
- `docs/design/interval_stage2_execution_plan.md` — phased implementation/ticket plan for Stage 2
- `docs/design/phase_a_interval_rooted_candidate.md` — Phase A specification for the first public interval-rooted candidate/plan shape
- `docs/design/phase_b1_interval_choice_tracking_scope.md` — narrow Phase B1 scoping note for direct rooted output from `interval_solver`
- `docs/design/phase_b1_interval_choice_tracking_cpp_api.md` — concrete API-shape note for the first rooted-output solve path on `interval_solver`

## 1. Purpose

This note defines the **Stage 2 direction** for Sprint 9 after the Stage 1 interval solver substrate has landed.

Stage 1 intentionally delivered only a narrow interval-solving substrate:

- `interval_context`
- `interval_partition_plan`
- `all_binary_splits`
- `triangular_memo<Value>`
- narrow `interval_solver`

Stage 2 is where the broader deferred Sprint 9 value belongs:

- richer public interval-rooted representations
- broader recurrence and memo support
- interval-rooted reconstruction and integration
- interval-rooted bridge / model work if still justified

The goal is still **not** to weaken or replace the canonical partition-rooted tree family.

## 2. Architectural boundary

The ADR remains in force:

- `tree_space` and `reduction_tree_space` remain the canonical **partition-rooted** tree search family
- ordered contiguous interval decompositions remain a **sibling representation family**
- Stage 2 must not retrofit interval semantics into partition-tree types just to create a false common abstraction

That means Stage 2 should add interval-rooted capabilities with their own:

- representation types
- reconstruction rules
- legality contracts
- optional bridge / feature schema
- solver integration

It should not change the meaning of:

- `tree_space`
- `reduction_tree_space`
- `tree_point`
- `tree_bridge`

## 3. Stage 1 baseline we are building on

Stage 2 starts from the fact that Stage 1 already provides a stable minimum substrate.

### 3.1 Landed vocabulary

The current Stage 1 interval vocabulary gives us:

- half-open runtime interval identity via `interval_context`
- runtime split-to-plan composition via `interval_partition_plan::from_split(...)`
- default split enumeration via `all_binary_splits`
- dense interval memoization via `triangular_memo<Value>`
- a narrow `interval_solver` with value-only solve and stats-bearing solve

### 3.2 Existing specialized path still retained

The existing specialized matrix-chain-style path also remains valid:

- `interval_dp`
- `interval_split_space`
- `interval_split_candidate`

Stage 2 must therefore avoid an "everything is interval_solver now" rewrite unless later evidence makes that clearly better.

## 4. Stage 2 design objective

Deliver a broader but still conformant capability:

> Introduce interval-rooted representations and broader solver-side support as a sibling architecture family, while preserving partition-rooted trees as canonical and avoiding premature tree-family unification.

This objective implies a **representation-first** Stage 2 rather than a purely solver-generality-first Stage 2.

## 5. Why Stage 2 should be representation-first

Stage 1 solved the substrate problem.

The main gap now is not "can we recurse over intervals?" — we can. The main gap is:

- what interval-rooted plans/candidates look like publicly
- how interval solutions are reconstructed and consumed
- where interval-specific legality and feature contracts live
- how interval-rooted outputs relate to existing solver and model APIs

If Stage 2 broadens recurrence mechanics first without settling the public representation story, it risks producing infrastructure without a stable product surface.

## 6. Stage 2 scope

Stage 2 should include four major capability areas.

### 6.1 Public interval-rooted representation family

Stage 2 should introduce a real interval-rooted public representation family with its own vocabulary types.

Likely additions:

- an interval-rooted candidate or plan value that represents the chosen split structure
- explicit reconstruction helpers from memoized split decisions or recursive solver state
- interval-rooted node semantics that are independent of partition labels or group semantics

The key point is that this representation must be **owned by the interval family**, not borrowed from the partition-tree family.

### 6.2 Reconstruction and solution materialization

Stage 1 gives value solving and split-plan composition, but only a narrow answer shape.

Stage 2 should add:

- reconstruction of interval-rooted decomposition structure
- reconstruction from split tables and/or recurrence-owned choice state
- stable interval-rooted output types suitable for downstream consumers

This is the missing bridge between "interval recurrence machinery exists" and "an interval solution is a first-class framework result."

### 6.3 Broader solver and memo protocols

Stage 2 should broaden the narrow Stage 1 solver contracts, but only in ways motivated by real representation needs.

Candidate Stage 2 expansions:

- branch-level infeasibility or partial recurrences
- recurrence protocols that return both value and reconstruction choice
- additional split policies beyond `all_binary_splits`
- additional memo backends such as sparse / map-based memoization where justified
- better formalization of interval-specific stats and reconstruction metadata

This should remain an **interval-family solver layer**, not a claim to be a universal DP abstraction.

### 6.4 Interval-rooted bridge / model integration

If interval-rooted outputs need feature extraction or learned-model support, Stage 2 is the first point where that can be designed honestly.

Any such work must define its own:

- interval-rooted feature schema
- stable encoding contract
- downstream consumer expectations

It must not overload `tree_bridge` or pretend interval-rooted features are just a small variation of partition-tree features.

## 7. Proposed Stage 2 public surface

The Stage 2 public surface should still be deliberate, not sprawling.

### 7.1 Likely stable additions

Strong Stage 2 candidates:

- interval-rooted plan / candidate representation types
- reconstruction helpers for interval solutions
- one additional memo backend if a real use case requires it
- one broader recurrence interface for reconstruction-aware solving
- additional split-policy types only where semantics are clear

### 7.2 Existing Stage 1 surface retained

Retain and keep stable:

- `interval_context`
- `interval_partition_plan`
- `all_binary_splits`
- `triangular_memo<Value>`
- narrow `interval_solver`

### 7.3 Existing specialized interval DP retained

Retain unless and until replacement is clearly justified:

- `interval_dp`
- `interval_split_space`
- `interval_split_candidate`

Stage 2 may later provide an adaptation path, but should not begin by forcing migration.

## 8. Representation design guidance

The interval-rooted family should obey its own laws.

### 8.1 Interval-rooted semantics

A Stage 2 interval representation should make these ideas explicit:

- every internal node owns a contiguous interval `[i, j)`
- every internal node owns a split `k` such that `i < k < j`
- left and right children are ordered
- reconstruction preserves coverage and contiguity

### 8.2 No partition leakage

Stage 2 interval representations should not depend on:

- group labels
- partition canonicalization
- per-group child slots
- partition-tree bridge width assumptions

If a representation needs any of those, it likely belongs to the partition family instead.

### 8.3 Fixed-tree + permutation remains separate

A fixed-tree + input-permutation model is still not the same as a general interval-rooted decomposition family.

It may eventually become another sibling ordered-tree representation, but it should not be treated as equivalent to Stage 2 interval decomposition.

## 9. Solver design guidance

Stage 2 solver work should be shaped by concrete representation needs.

### 9.1 Recommended solver expansions

Promising next-step expansions include:

- recurrence interfaces that can return reconstruction data
- partial or infeasible branch handling
- selective split policies
- solver outputs that include interval-rooted materialized structure, not only scalar value

### 9.2 Expansions to avoid initially

Avoid starting Stage 2 with:

- a grand unified recurrence concept for all future DP problems
- a universal memo interface spanning unrelated solver families
- tree-family-generic reconstruction APIs
- intrusive changes to `select_and_run` just to route interval-rooted work automatically

These are more likely consequences of mature Stage 2 designs than prerequisites for them.

## 10. Relationship to existing `interval_dp`

Stage 2 should treat `interval_dp` pragmatically.

### 10.1 Near-term position

Near term:

- `interval_dp` remains the compact specialized path for classic split-table interval DP
- Stage 2 develops interval-rooted family capabilities beside it

### 10.2 Possible later convergence

Only after Stage 2 representation and reconstruction stabilize should we evaluate whether:

- `interval_dp` should delegate to richer interval machinery internally
- `interval_split_candidate` should be reframed as one representation inside the interval family
- a compatibility layer should translate between specialized and richer interval outputs

Those are later refactor questions, not Stage 2 entry conditions.

## 11. Placement and naming

Stage 2 should continue moving productized value toward the canonical `ctdp` namespace.

Recommended direction:

- interval-rooted public types should appear under `ctdp::solver` or adjacent canonical namespaces
- legacy `ct_dp` implementation residue should not become the long-term public surface
- namespace consolidation should be driven by shipped public types, not by a mass rename of unfinished branch artifacts

## 12. Stage 2 non-goals

The following remain explicitly out of scope for Stage 2.

### 12.1 No weakening of the partition-tree ADR

Stage 2 does not:

- generalize `tree_space` into a universal tree abstraction
- change the semantics of `tree_point`
- change the semantics of `tree_bridge`
- collapse partition-rooted and interval-rooted families into one nominal API

### 12.2 No premature universal solver abstraction

Stage 2 does not promise:

- one recurrence concept for all solver families
- one memo abstraction for all optimization problems
- one reconstruction protocol across unrelated domains

### 12.3 No forced migration of existing users

Stage 2 does not require immediate migration away from:

- `interval_dp`
- `interval_split_space`
- current partition-rooted tree search

### 12.4 No fixed-tree permutation fold-in

Stage 2 does not treat:

- fixed-tree + input-permutation
- reduction-tree permutation
- ordered-tree permutation more broadly

as solved just because interval-rooted decomposition becomes richer.

## 13. Recommended phased implementation

### Phase A — Representation and reconstruction

Prioritize:

- interval-rooted public plan/candidate types
- reconstruction helpers
- invariants and tests for interval-rooted structure

This is the most important Stage 2 phase because it defines the product surface.

### Phase B — Broader recurrence and memo support

Then add:

- reconstruction-aware recurrence protocols
- branch infeasibility support if needed
- additional memo backends where justified
- targeted split-policy extensions

### Phase C — Interval-rooted integration

Only after representation stabilizes, evaluate:

- interval-rooted feature schemas
- bridge-like encoding for interval solutions
- learned-model integration for interval-rooted search

### Phase D — Cleanup and optional convergence work

After the family is stable, consider:

- namespace consolidation
- relation to `interval_dp`
- compatibility adapters
- common abstractions, but only if at least two real sibling tree families now expose a meaningful shared core

## 14. Suggested acceptance criteria for Stage 2

A useful Stage 2 deliverable should satisfy most of the following:

- interval-rooted solutions are first-class public values, not just implicit recursion state
- reconstruction is stable and testable
- broader interval solving remains clearly distinct from partition-tree solving
- any new memo or split-policy abstractions are justified by concrete use cases
- interval-rooted integration has its own contracts rather than borrowing partition-tree assumptions

## 15. Summary

The right Stage 2 is **not** "make Stage 1 more abstract."

The right Stage 2 is:

> Add a real interval-rooted representation family, reconstruction path, and broader interval-specific solver support as a sibling architecture beside the canonical partition-rooted tree family.

That keeps the ADR intact, gives the deferred Sprint 9 work a coherent home, and avoids the two main failure modes:

- pretending interval-rooted and partition-rooted trees are the same thing
- building more generic solver machinery before the interval family has a stable public representation.




