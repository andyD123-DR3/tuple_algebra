# Sprint 9 Stage 1 — Narrow Conformant Interval Solver Slice

**Status:** Proposed  
**Date:** May 19, 2026  
**Scope:** Stage 1 only (narrow harvest)  
**Related:** `docs/adr_partition_rooted_tree_canonical.md`, `include/ctdp/solver/algorithms/interval_dp.h`, `include/ctdp/solver/spaces/interval_split_space.h`

## 1. Purpose

This note defines the **narrow conformant Sprint 9 slice** that can be harvested from `feature/sprint9-interval-solver` with the smoothest fit to the existing architecture.

The goal is **not** to introduce a new generic tree architecture. The goal is to add a small, reusable **interval solver substrate** for ordered binary split problems while preserving the current architectural boundary:

- `tree_space` / `reduction_tree_space` remain the canonical **partition-rooted** tree representation
- interval-rooted decomposition remains a **sibling solver capability**, not a change to the partition-tree model

## 2. Why Sprint 9 cannot be merged as-is

The branch contains real value, but it does not integrate cleanly without redesign.

### 2.1 Parallel namespace stack

The useful Sprint 9 files live under `ct_dp::*`, while the evolving canonical public surface lives under `ctdp::*`. A direct merge would deepen architectural duplication instead of consolidating it.

### 2.2 Different representation family

Sprint 9 is about **ordered contiguous interval decomposition**:

- runtime interval identity
- interior split points
- ordered binary subproblems

This is not the same family as the current partition-rooted tree model:

- set partitions over a fixed item set
- canonical group labels
- per-group child spaces

Per the ADR, interval-rooted work should therefore be introduced as a **sibling representation family**, not folded into `tree_space`.

### 2.3 Existing interval baseline already on `main`

Current `main` already provides:

- `interval_dp`
- `interval_split_space`
- `interval_split_candidate`
- matrix-chain tests

So Sprint 9 is not introducing interval solving from zero. It is adding **generalized interval solver infrastructure** beyond the existing specialized DP path.

### 2.4 Branch contains non-product noise

The Sprint 9 branch diff includes obviously non-harvestable artifacts such as:

- stray files under `include/ct_dp/`
- backup CMake files
- test layout noise

This reinforces that Sprint 9 should be **harvested selectively**, not merged wholesale.

## 3. Design objective

Deliver a small additive capability:

> A reusable interval solver substrate for ordered binary split problems, implemented as a sibling solver path, without changing partition-rooted trees and without replacing the current specialized `interval_dp` path.

This is the Stage 1 deliverable.

## 4. Architectural fit

### 4.1 Existing architecture suggests a sibling solver path

The current architecture already tolerates specialized solver-side capabilities where they are justified by problem structure. `interval_dp` is the precedent: it is not forced into `descriptor_space` or `tree_space`; it exists as a solver-oriented interval algorithm.

Sprint 9 Stage 1 should follow the same pattern:

- additive, not invasive
- solver-side first
- interval-rooted and ordered
- clearly separate from partition-rooted tree search

### 4.2 ADR boundary

Per `docs/adr_partition_rooted_tree_canonical.md`, Stage 1 must not:

- generalize `tree_space`
- change `tree_point`
- change `tree_bridge`
- introduce interval semantics into the partition-tree path

That ADR is a design constraint, not a suggestion.

## 5. Stage 1 slice

Stage 1 includes only the pieces that clearly fit and clearly add value.

### 5.1 Interval vocabulary

Adopt a minimal interval vocabulary for ordered binary decomposition:

- `interval_context`
  - half-open interval `[i, j)`
  - `size()`, `start()`, `end()`
  - `left(k)`, `right(k)`

This vocabulary is already explored in Sprint 8/9 branch code and fits cleanly as solver-side infrastructure.

### 5.2 Runtime split-to-plan composition

Adopt an interval plan term that composes runtime interval context with a chosen split:

- `interval_partition_plan`
  - whole interval
  - absolute split point
  - left/right child intervals
  - legality / size-preservation checks
  - runtime construction from split point

The key Stage 1 addition is a direct runtime constructor such as:

- `interval_partition_plan::from_split(ctx, k)`

This is the cleanest descriptor+context composition point for runtime interval solving.

### 5.3 Default split policy

Adopt a small split-policy abstraction with one concrete Stage 1 policy:

- `all_binary_splits`

This should enumerate all legal interior split points for a non-empty interval.

The abstraction is justified because it isolates “which splits are legal” from the solver core without requiring a full generic framework.

### 5.4 Dense interval memo

Adopt the dense triangular memo backend:

- `triangular_memo<Value>`

This is the strongest Stage 1 harvest candidate because it is:

- useful independently
- conceptually clean
- not tied to tree-space questions
- reusable across ordered interval recurrences

### 5.5 Narrow generic interval solver (optional upper bound of Stage 1)

If Stage 1 includes a solver abstraction, it should be deliberately narrow:

- ordered binary interval decomposition only
- runtime intervals only
- optimal value solving only
- external memo supplied by caller
- total recurrences only for Stage 1

A narrow `interval_solver` is acceptable only if it is presented as:

- a reusable solver substrate
- not a universal DP abstraction
- not a replacement for `interval_dp`

## 6. Explicit Stage 1 non-goals

The following are intentionally **out of scope** for Stage 1.

### 6.1 No partition-tree changes

Stage 1 does not modify:

- `tree_space`
- `reduction_tree_space`
- partition-rooted tree points
- partition-rooted bridges

### 6.2 No feature bridge / ML integration

Stage 1 does not define:

- interval-tree feature schemas
- bridge integration for interval-rooted representations
- learned-model integration for interval decomposition

### 6.3 No broad public recurrence framework

Stage 1 does not promise:

- partial recurrences
- branch-level infeasibility protocols beyond the narrow first cut
- generalized reconstruction strategies
- fully abstract memo/planner ecosystems

### 6.4 No replacement of `interval_dp`

Stage 1 does not remove or rewrite:

- `interval_dp`
- `interval_split_space`
- `interval_split_candidate`

These remain the current specialized path for matrix-chain-like interval DP.

### 6.5 No tree-family unification

Stage 1 does not attempt to unify:

- partition trees
- interval trees
- ordered-tree variants
- fixed-tree + permutation models

That belongs to later stages, if ever justified.

## 7. Proposed public surface for Stage 1

The Stage 1 public surface should be intentionally small.

### Stable vocabulary / substrate candidates

- `interval_context`
- `interval_partition_plan`
- `all_binary_splits`
- `triangular_memo<Value>`

### Conditional / phase-gated API

- `interval_solver<...>` only if the implementation remains narrow and useful enough to justify productizing it now

### Existing API retained unchanged

- `interval_dp`
- `interval_split_space`
- `interval_split_candidate`

## 8. Placement and naming

The smoothest fit is to move harvested value toward the canonical `ctdp` stack rather than perpetuate a parallel `ct_dp` public architecture.

Suggested eventual placement:

- `ctdp::solver::interval_context`
- `ctdp::solver::plans::interval_partition_plan`
- `ctdp::solver::policies::all_binary_splits`
- `ctdp::solver::memo::triangular_memo`
- `ctdp::solver::algorithms::interval_solver` (if included in Stage 1)

This note does **not** require a single-step namespace migration of every branch artifact. It does require that harvested product value not remain permanently architecturally second-class.

## 9. Relationship to existing `interval_dp`

Stage 1 positions `interval_dp` as the current specialized interval algorithm and the new Sprint 9 harvest as a sibling substrate.

That means:

- `interval_dp` remains valid and supported
- Stage 1 adds reusable lower-level interval machinery
- the question “should `interval_dp` later delegate to the new substrate?” is deferred

This avoids duplication panic while also avoiding premature replacement.

## 10. Implementation plan

### Phase 1 — Vocabulary and memo

Implement:

- interval context in canonical placement
- runtime `interval_partition_plan::from_split(...)`
- `all_binary_splits`
- `triangular_memo`

Add focused tests for:

- interval context laws
- split-to-plan construction laws
- split policy enumeration
- triangular memo correctness

### Phase 2 — Narrow interval solver

If justified after Phase 1, implement:

- narrow `interval_solver`

Add tests that:

- compare simple recurrence behavior
- compare matrix-chain-like recurrence behavior with existing `interval_dp`
- verify memo/policy interaction

### Phase 3 — Deferred relationship work

Only after Stage 1 lands should we decide whether:

- `interval_dp` remains permanently specialized
- or delegates internally to the narrower reusable solver substrate

## 11. Stage 2 (explicitly deferred)

The following are Stage 2 concerns and are **not** part of the current slice:

- richer public interval-rooted candidate/tree representations
- broader recurrence generality
- multiple public memo backends
- interval-tree feature bridge/model integration
- fixed-tree + input-permutation models
- tree-family common abstractions
- broader namespace/public API consolidation

## 12. Summary

The smoothest fit with the current architecture is:

> Harvest Sprint 9 as a small sibling interval solver substrate — interval vocabulary, runtime split-to-plan composition, default split policy, dense triangular memoization, and optionally a narrow reusable interval solver — while leaving partition-rooted trees, bridges, and broader tree-family generalization untouched.

That gives Sprint 9 a conformant Stage 1 deliverable with real value and controlled risk.


