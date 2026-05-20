# Phase B1 Scope — `interval_solver` Choice Tracking for Rooted Output

**Status:** Implemented slice reference note  
**Date:** May 20, 2026  
**Scope:** Stage 2 Phase B1 scoping note  
**Related:** `docs/design/interval_stage2_execution_plan.md`, `docs/design/phase_a_interval_rooted_candidate.md`, `include/ctdp/solver/algorithms/interval_solver.h`, `include/ctdp/solver/interval_rooted_candidate.h`, `include/ctdp/solver/algorithms/interval_dp.h`

## 1. Purpose

This note scopes the smallest useful follow-on after the landed Phase A interval-rooted public surface.

The corresponding first B1 implementation slice has now landed in `include/ctdp/solver/algorithms/interval_solver.h`; this note remains useful as the scope/rationale record for that work.

The goal is **not** to redesign the interval solver into a grand universal DP abstraction.

The goal is narrower:

> define the minimal Stage 2 Phase B1 extension that can drive `interval_rooted_plan` materialization from `interval_solver` while keeping the current Stage 1 value-only recurrence path valid.

## 2. Current baseline

The current interval family now has two stable pillars.

### 2.1 Stage 1 solver substrate

`include/ctdp/solver/algorithms/interval_solver.h` currently provides:

- `interval_recurrence`
- `interval_memo`
- `interval_solver<Recurrence, SplitPolicy, Compare>`
- `solve(...) -> value_type`
- `solve_with_stats(...) -> interval_solve_result<value_type>`

The current recurrence contract is intentionally narrow:

- `base_case(ctx) -> optional<value_type>`
- `combine(plan, left, right) -> value_type`

### 2.2 Stage 2 Phase A public representation

`include/ctdp/solver/interval_rooted_candidate.h` now provides:

- `interval_rooted_candidate<MaxN>`
- `interval_rooted_plan<MaxN>`
- deterministic traversals
- canonicalization
- reconstruction helpers from split callbacks and split tables
- `reconstruct_interval_rooted_plan(plan<interval_split_candidate<MaxN>>)`

This means the missing gap is no longer public representation.

The missing gap is: how a generic interval solve path can retain the winning split choices needed to materialize that representation directly.

## 3. Problem statement

Today `interval_solver` solves only for the optimal **value**.

For each internal interval `[i, j)`, it enumerates candidate splits `k`, evaluates `combine(plan, left, right)`, and keeps only the best scalar/objective result.

Once the recursion returns, the winning split structure is gone.

That is sufficient for Stage 1 value solving, but insufficient for direct production of:

- `interval_rooted_candidate<MaxN>`
- `interval_rooted_plan<MaxN>`

## 4. Key design observation

For the current interval family, the reconstruction choice needed for Phase B1 is the winning **split point** `k` for each internal interval.

That choice is already known by the solver at the moment a better candidate is selected.

Therefore the smallest B1 extension does **not** require a broad redesign where every recurrence must invent and return its own custom choice object.

The solver already owns the relevant decomposition step:

- it has `interval_partition_plan`
- it knows the active interval `[i, j)`
- it knows the candidate split `k`
- it decides which split wins under `Compare`

This suggests that B1 should begin with **solver-owned choice tracking**, not recurrence-owned arbitrary metadata.

## 5. Recommended B1 shape

### 5.1 Preserve the Stage 1 recurrence concept

B1 should preserve the existing narrow recurrence path unchanged:

- `interval_recurrence` remains supported as-is
- current `solve(...)` and `solve_with_stats(...)` remain valid and stable
- existing Stage 1 tests should continue to pass unchanged

This is the compatibility anchor.

### 5.2 Add a sibling rooted-output solve path

B1 should add a sibling solve path whose job is:

1. solve values exactly as today
2. record the best split `k` for each internal interval as the solve proceeds
3. reconstruct an `interval_rooted_candidate<MaxN>` from those winning splits
4. return that candidate in normal `ctdp::plan<...>` vocabulary

Conceptually, the output target should be:

- `interval_rooted_plan<MaxN>`

The important point is not the exact method name yet.

The important point is that the new path is **additive**, not a replacement for value-only solve.

### 5.3 Keep choice metadata solver-owned

For B1, the choice record should be only what Phase A reconstruction actually needs:

- the winning split point `k` for each represented internal interval

That means B1 likely needs a dedicated split-choice storage object, such as:

- a dense triangular split memo/table, or
- an equivalent fixed-capacity interval-indexed structure

This is analogous in spirit to what `interval_dp` already does with a split table, but should remain owned by the broader interval family rather than folded back into the legacy specialized API.

### 5.4 Reuse Phase A reconstruction machinery

B1 should not invent a second reconstruction vocabulary.

Instead, the solve path should flow into the already-landed Phase A materialization surface:

- record winning splits during solve
- reconstruct via the same half-open `[i, j)` + absolute `k` semantics
- produce canonical `interval_rooted_candidate<MaxN>`
- wrap it in `interval_rooted_plan<MaxN>` with cost/stats preserved

That keeps B1 representation-first instead of creating a parallel hidden tree format.

## 6. Compatibility story

B1 should explicitly support coexistence among three interval paths.

### 6.1 Current Stage 1 path stays valid

Value-only users keep:

- `interval_solver::solve(...)`
- `interval_solver::solve_with_stats(...)`

No migration should be forced.

### 6.2 `interval_dp` stays specialized

`interval_dp` should remain the compact specialized split-table solver.

B1 should not assume that `interval_dp` must delegate through `interval_solver`.

The current adaptation path remains valid:

- `interval_dp(...) -> plan<interval_split_candidate<MaxN>>`
- `reconstruct_interval_rooted_plan(...) -> interval_rooted_plan<MaxN>`

### 6.3 New rooted-output path is additive

The B1 path should be a new capability for users who want:

- generic interval recurrence solving
- direct materialized interval-rooted output
- preserved `predicted_cost` and `stats`

That additive path is now present through:

- `interval_solver::solve_rooted<MaxN>(...)`
- `interval_solver::solve_rooted_with_stats<MaxN>(...)`

The current public example-facing path is demonstrated in:

- `examples/matrix_chain_demo.cpp`

## 7. What B1 should not do yet

To keep the slice narrow, B1 should defer the following.

### 7.1 No sparse memo decision yet

Do not tie B1 to a new sparse memo backend.

That belongs to B2 only if a real use case appears.

### 7.2 No general infeasible-branch protocol yet

Do not require partial/infeasible branch semantics in the first B1 slice.

That belongs to B3.

### 7.3 No recurrence-owned arbitrary metadata yet

Do not require each recurrence to define custom reconstruction payload types unless a concrete use case proves the winning split alone is insufficient.

The first question is whether solver-owned split tracking is enough.

For ordinary interval-rooted decomposition output, it should be.

### 7.4 No `select_and_run` integration yet

Do not force automatic dispatch or replacement of existing specialized interval paths as part of B1.

That would broaden the change far beyond the minimal rooted-output extension.

## 8. Candidate API direction (non-binding)

A plausible eventual shape is a sibling rooted solve API that is explicit about capacity and reconstruction target.

Examples of the sort of API to evaluate later:

```cpp
template<std::size_t MaxN, class Memo>
auto solve_rooted(interval_context ctx, Memo& value_memo) const
    -> interval_rooted_plan<MaxN>;

template<std::size_t MaxN, class Memo>
auto solve_rooted_with_stats(interval_context ctx, Memo& value_memo) const
    -> interval_rooted_plan<MaxN>;
```

Or equivalently, a separate helper/adaptor that uses the existing solver internally.

This note does **not** commit to exact API spelling.

It only narrows the intended semantics:

- same recurrence math
- same split policy semantics
- extra tracked output = winning split structure
- resulting public artifact = `interval_rooted_plan<MaxN>`

## 9. Acceptance criteria for B1

A good B1 implementation should satisfy all of the following:

- existing Stage 1 `interval_recurrence` code continues to compile unchanged
- current `interval_solver` value-only tests continue to pass unchanged
- one new rooted-output solve path can materialize canonical `interval_rooted_candidate<MaxN>`
- rooted-output solve preserves `predicted_cost` and `stats`
- rooted-output solve uses Phase A half-open interval semantics
- no B2/B3/B4 concerns are entangled into the first implementation

## 10. Summary

The smallest useful B1 is **not** “make recurrences return arbitrary trees.”

The smallest useful B1 is:

> keep the current value recurrence contract, let the solver record winning split choices, and reconstruct `interval_rooted_plan<MaxN>` through the already-landed Phase A public representation surface.

That is enough to connect Stage 1 solving to Stage 2 rooted output without prematurely broadening memo, infeasibility, or dispatch concerns.


