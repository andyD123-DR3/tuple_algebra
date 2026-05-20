# Phase A Specification — `interval_rooted_candidate`

**Status:** Draft  
**Date:** May 20, 2026  
**Scope:** Stage 2 Phase A public type specification  
**Related:** `docs/adr_interval_rooted_representation_family.md`, `docs/design/interval_stage2_execution_plan.md`, `include/ctdp/core/plan.h`, `include/ctdp/solver/algorithms/interval_solver.h`

## 1. Purpose

This note specifies the first public interval-rooted solution type for Stage 2.

The purpose of this type is to make interval-rooted solutions first-class public values without borrowing semantics from the partition-rooted tree family.

The type should represent an **ordered contiguous binary decomposition** over leaves `0..n-1` using the Stage 1 half-open interval convention.

## 2. Design choice

Phase A should introduce a canonical public candidate type named:

- `interval_rooted_candidate<MaxN>`

The associated plan type should reuse the existing framework result vocabulary:

- `ctdp::plan<interval_rooted_candidate<MaxN>>`

If an alias is desired for readability, it should be only an alias, not a new result struct:

- `using interval_rooted_plan = ctdp::plan<interval_rooted_candidate<MaxN>>;`

This keeps the framework's existing `plan<Candidate>` vocabulary intact.

## 3. Why this is a new type

The existing `interval_split_candidate<MaxN>` is useful, but it is not the right Stage 2 public interval-rooted candidate.

It records split choices for **all solved subproblems** in a classic split table. That is appropriate for specialized interval DP, but it is not the same thing as a first-class interval-rooted solution value.

A public interval-rooted candidate should represent the **materialized chosen decomposition tree**.

That means it must distinguish between:

- intervals that are part of the chosen decomposition
- intervals that were merely subproblems considered during optimization

## 4. Public semantics

`interval_rooted_candidate<MaxN>` represents a single ordered binary decomposition over the active leaf range `0..n-1`.

### 4.1 Interval convention

The family uses the Stage 1 convention:

- every interval is half-open `[i, j)`
- every internal split is an absolute split point `k`
- legality requires `i < k < j`

This is intentionally different from the legacy `interval_dp` closed-interval / last-of-left storage convention.

### 4.2 Root convention

For `n > 0`, the root interval is always:

- `[0, n)`

For `n == 0`, the candidate represents the empty decomposition.

### 4.3 Ordered children

For every internal node `[i, j)` with split `k`:

- left child is `[i, k)`
- right child is `[k, j)`

Children are ordered and that order is semantically meaningful.

## 5. Representation model

Phase A should use a value-semantic representation that supports deterministic traversal and testing without heap-owned recursive nodes.

### 5.1 Recommended storage shape

Recommended internal shape:

- `std::size_t n`
- dense table over potential intervals in `0..n`
- each table entry stores either:
  - `absent` — interval is not part of the chosen decomposition
  - `leaf` — interval is a represented leaf interval of size 1
  - `split k` — interval is a represented internal interval split at absolute point `k`

A simple implementation strategy is a dense `std::array<std::size_t, MaxN * (MaxN + 1)>`-style table with sentinels, or equivalent compact storage.

The key property is not the exact storage container. The key property is that the candidate is:

- value-semantic
- deterministic
- free of partition-family assumptions
- able to distinguish represented vs non-represented intervals

### 5.2 Why not recursive owned nodes first

Phase A should avoid a heap-owning recursive node tree as the primary candidate representation because that would complicate:

- constexpr friendliness
- equality semantics
- trivial copying / testing
- integration with existing fixed-capacity framework patterns

Recursive node *views* may still be provided over the candidate.

## 6. Required public operations

`interval_rooted_candidate<MaxN>` should expose operations centered on interval semantics.

### 6.1 Core queries

Required queries:

- `empty()`
- `size()` or `leaf_count()` returning `n`
- `root_interval()` returning `[0, n)` for non-empty candidates
- `contains(i, j)` — whether interval `[i, j)` is part of the chosen decomposition
- `is_leaf(i, j)` — whether `[i, j)` is a represented leaf
- `is_internal(i, j)` — whether `[i, j)` is a represented internal node
- `split(i, j)` — absolute split point for represented internal nodes

### 6.2 Structural access

Required structural access:

- `left_interval(i, j)` derived from `split(i, j)`
- `right_interval(i, j)` derived from `split(i, j)`
- deterministic traversal helpers or iterators

### 6.3 Validation

Required validation support:

- `is_legal()` or equivalent invariant checker
- `is_canonical()` if the implementation distinguishes legality from canonical storage normalization

## 7. Companion view type

Phase A should also define a lightweight interval-rooted node/view abstraction.

Recommended name:

- `interval_rooted_node_ref<MaxN>`

This should be a non-owning view over:

- a candidate
- a represented interval `[i, j)`

It should provide:

- `interval()`
- `is_leaf()`
- `is_internal()`
- `split()`
- `left()`
- `right()`

This gives a tree-like consumer API without forcing recursive owned-node storage.

## 8. Invariants

The candidate must satisfy the following invariants.

### 8.1 Root legality

If `n > 0`:

- the root interval `[0, n)` is represented

If `n == 0`:

- no intervals are represented

### 8.2 Leaf legality

A represented leaf interval must have size `1`.

That means:

- `[i, i+1)` may be leaf
- no larger interval may be marked leaf

### 8.3 Internal legality

A represented internal interval `[i, j)` must have exactly one split `k` such that:

- `i < k < j`

### 8.4 Child closure

If `[i, j)` is represented internal with split `k`, then:

- left child `[i, k)` is represented
- right child `[k, j)` is represented

### 8.5 No stray represented intervals

Every represented interval must be reachable from the root by repeated child expansion.

This is what distinguishes a materialized interval-rooted candidate from a generic split table over all subproblems.

### 8.6 Coverage law

The represented tree must cover exactly the active leaf range `0..n-1`.

That implies:

- leaves are disjoint
- leaves appear in order
- leaf intervals collectively cover `[0, n)`

## 9. Equality semantics

Equality should be structural over the active represented tree.

Two candidates are equal iff:

- they have the same `n`
- they represent the same reachable intervals
- corresponding internal intervals use the same absolute split points

Unreachable storage garbage must not affect equality.

This strongly suggests canonical normalization or explicit reachability-aware equality.

## 10. Reconstruction contract

Phase A reconstruction helpers should build `interval_rooted_candidate<MaxN>` from interval-family decision sources.

### 10.1 Reconstruction inputs

Supported inputs should include at least one of:

- recursive solver choice state
- split-table-like structures
- explicit split assignment callbacks

### 10.2 Reconstruction output law

Reconstruction must produce a candidate satisfying all invariants above.

In particular:

- all reachable intervals are represented
- no unreachable intervals are represented
- interval and split conventions use Stage 1 half-open semantics

### 10.3 Relationship to `interval_partition_plan`

`interval_partition_plan` remains the local split-composition term.

It should not itself become the full public candidate type.

Instead:

- `interval_partition_plan` describes one internal node decomposition step
- `interval_rooted_candidate` describes the whole chosen decomposition

## 11. Compatibility guidance

### 11.1 With `interval_solver`

Stage 2 should eventually allow `interval_solver`-based solving to materialize `interval_rooted_candidate` results.

That likely requires a reconstruction-aware recurrence extension in Phase B.

### 11.2 With `interval_dp`

Compatibility with `interval_dp` should come through reconstruction helpers or adapters.

The specialized `interval_split_candidate<MaxN>` should remain valid, but it should not block introduction of the richer public interval-rooted candidate.

### 11.3 With `plan<Candidate>`

The framework should continue using:

- `ctdp::plan<interval_rooted_candidate<MaxN>>`

rather than inventing a new parallel result vocabulary.

## 12. Non-goals for Phase A

Phase A does not require:

- learned-model integration
- feature encoding
- split-policy generalization
- sparse memoization
- universal tree abstractions
- replacement of `interval_dp`
- fixed-tree + permutation support

## 13. Suggested test surface

Phase A tests should cover:

- empty candidate
- single-leaf candidate
- balanced binary decomposition
- maximally skewed decomposition
- illegal split rejection
- reachability / no-stray-interval invariant
- equality over active represented tree only
- deterministic traversal order
- reconstruction from small known trees

## 14. Summary

The recommended Phase A public type is:

- `interval_rooted_candidate<MaxN>`

with:

- Stage 1 half-open interval semantics
- explicit represented-vs-absent interval distinction
- deterministic tree-like views over value-semantic storage
- reuse of `ctdp::plan<Candidate>` for full-plan results

That gives Stage 2 a real interval-rooted public solution type without forcing interval work back through partition-rooted tree abstractions.

