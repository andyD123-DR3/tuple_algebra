# ADR: Keep Partition-Rooted Tree as Canonical

**Decision:** Keep the current partition-rooted tree representation as the canonical tree model for now.  
**Date:** May 19, 2026  
**Status:** Accepted  
**Owners:** Framework architecture / space layer

## Context

The current tree search model is implemented by `tree_space` in `include/ctdp/space/tree_space.h` and the reduction-specific specialisation in `include/ctdp/space/reduction_tree_space.h`.

This model is explicitly **partition-rooted**:

- the root is a `partition_desc<N>` over a fixed set of items
- groups are identified by canonical labels `0..K-1`
- group membership is recovered via `group_lanes()`
- a `tree_point` contains the root partition plus one child plan per group
- a `tree_bridge` encodes root features first, then fixed-width child blocks in canonical group order

This representation is already coherent for the current use case:

- grouping reduction lanes by fusibility/legality
- generating per-group optimisation spaces
- composing child plans under a stable feature schema

Recent design discussion considered other possible tree representations, including:

- ordered / interval-rooted trees
- matrix-chain-style binary split trees
- fixed-tree + input-permutation search

These representations may become useful later, but they are not equivalent to the current partition-rooted model.

## Decision

We will keep the current **partition-rooted tree representation** as the canonical tree model.

We will **not** generalise `tree_space` into a universal tree abstraction at this stage.

If additional tree representations are needed later, they will be introduced as **separate sibling representations** with their own:

- point model
- legality model
- bridge / feature schema
- construction helpers
- solver integration

Only after at least two concrete non-partition tree representations exist should we consider extracting a shared abstraction.

## Consequences

### Positive

- Preserves a working, coherent model for reduction-tree search.
- Keeps `tree_space` and `reduction_tree_space` easy to reason about.
- Preserves the current bridge invariants:
  - root features first
  - canonical group order
  - fixed-width child blocks
  - zero-filled unused blocks
- Avoids premature type erasure, runtime indirection, or weakened contracts.
- Keeps grouping/fusibility semantics explicit and central.

### Negative

- The current tree model is intentionally not a generic home for all tree-like search problems.
- Ordered/interval tree problems will require a separate representation if adopted.
- Some future features, especially permutation inside the current partition tree family, remain out of scope unless their semantics are naturally partition-based.

## Invariants to Preserve

The following are design invariants for the current partition tree family:

1. **Root semantics**
   - The root is a `partition_desc<N>`.
   - Grouping is by partition membership, not interval contiguity.

2. **Point semantics**
   - `tree_point` means: root partition + per-group child plans.
   - Group plans are indexed by canonical group label order.

3. **Bridge semantics**
   - Root feature layout is stable.
   - Child blocks are stable in order and width within this representation.
   - Unused blocks are zero-filled.

4. **Factory / filter semantics**
   - Child factories and legality filters are pure and deterministic.
   - Group lane extraction is derived from canonical partition labels.

## Explicit Non-Goals (for now)

The following are out of scope for the current partition-rooted model:

- a generic root abstraction that subsumes partitions, intervals, and ordered trees
- embedding matrix-chain / interval-tree semantics into `tree_space`
- redesigning `tree_point` or `tree_bridge` to support hypothetical future ordered-tree features
- introducing permutation into the current partition-tree path unless the semantics are truly partition-native

## Implications for Permutation Work

This decision does **not** rule out permutation as a framework capability.

### Current branch harvest

The concrete value being taken now from the branch work is:

- first-class flat permutation support in the canonical `descriptor_space` /
  `feature_bridge` path
- tests and example coverage for that flat permutation descriptor integration
- architectural clarification that tree-internal permutation is *not* part of
  the current partition-rooted tree representation

The following are explicitly **not** part of the current harvest:

- permutation inside `reduction_tree_space`
- ordered / interval-tree semantics folded into `tree_space`
- any redesign of `tree_point` or `tree_bridge` to anticipate future ordered
  tree representations

It does mean:

- flat permutation search in `descriptor_space` remains fine
- flat permutation dimensions in reduction-policy spaces may still be viable
- permutation inside the current partition-rooted `reduction_tree_space` is not a near-term goal
- any future fixed-tree + input-permutation model should be introduced as a sibling representation, not by weakening the current partition contracts

## Deferred Alternatives

The following alternatives are deferred, not rejected:

1. **Interval / matrix-chain tree representation**
   - suitable for ordered contiguous binary trees
   - likely best introduced as a separate representation family

2. **Fixed-tree + input-permutation representation**
   - suitable when tree topology is fixed and only leaf assignment varies
   - should not be forced through the current partition-rooted tree API

3. **Common tree abstraction**
   - only worth considering after multiple concrete tree families exist and expose a real shared core

## Rule of Thumb

When evaluating future tree-related features, ask:

> Does this feature naturally belong to partition/group semantics?

- **Yes** → integrate into the current partition tree family.
- **No** → introduce it as a separate sibling representation.


