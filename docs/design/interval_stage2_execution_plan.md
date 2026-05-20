# Interval Stage 2 Execution Plan

**Status:** Draft  
**Date:** May 20, 2026  
**Scope:** Stage 2 implementation planning  
**Related:** `docs/adr_interval_rooted_representation_family.md`, `docs/design/sprint9_stage2_interval_representation_and_integration.md`, `docs/design/phase_a_interval_rooted_candidate.md`

## 1. Purpose

This note turns the Stage 2 interval direction into an implementable plan.

It assumes:

- Stage 1 interval substrate has landed
- partition-rooted trees remain canonical
- interval-rooted work is a sibling family
- Stage 2 is representation-first

## 2. Delivery strategy

Stage 2 should be executed in four phases.

- **Phase A** — public interval-rooted candidate and reconstruction surface
- **Phase B** — broader solver, recurrence, and memo support
- **Phase C** — interval-rooted encoding / model integration
- **Phase D** — adapters, cleanup, and optional convergence work

The key sequencing rule is:

> Do not broaden interval solver generality before the interval family has a stable public solution representation.

## 3. Phase A — Public candidate and reconstruction surface

### Goal

Make interval-rooted solutions first-class public values.

### Deliverables

- `interval_rooted_candidate<MaxN>` specification and implementation
- interval-rooted node/view accessors over that candidate
- reconstruction helpers from split choices or split tables
- invariants and tests for legal interval-rooted structure
- `plan<interval_rooted_candidate<...>>` usage or alias documentation

### Status snapshot (May 20, 2026)

The current implementation has landed the core Phase A public surface:

- canonical header `include/ctdp/solver/interval_rooted_candidate.h`
- umbrella exposure via `include/ctdp/solver/solver.h`
- legality / canonicality / equality semantics with deterministic traversal helpers
- reconstruction from callback, legacy split tables, and `plan<interval_split_candidate<MaxN>>`
- `reconstruct_interval_rooted_plan(...)` adapter preserving cost and stats
- public example path in `examples/matrix_chain_demo.cpp`

Phase A can now be treated as functionally landed, with only optional ergonomic/docs follow-ups remaining.

### Tickets

#### A1. Candidate contract

Define the canonical public type and its invariants.

Acceptance criteria:

- public type name agreed
- split convention agreed (`[i, j)` + absolute split `k`)
- root semantics agreed
- canonical equality semantics documented

Current status: **landed**

#### A2. Storage and views

Choose the storage strategy and view API.

Acceptance criteria:

- candidate storage remains value-semantic and testable
- leaf/internal node queries are deterministic
- no dependency on partition-tree types

Current status: **landed**

#### A3. Reconstruction helpers

Add helpers that materialize the candidate from solver decisions.

Acceptance criteria:

- reconstruction from split choices is deterministic
- reachable intervals are exactly the represented tree
- no stray represented intervals remain in canonical output

Current status: **landed**

#### A4. Traversal and testing

Add traversal and validation support.

Acceptance criteria:

- preorder/inorder/postorder or equivalent deterministic traversal exists
- structural legality tests cover balanced and unbalanced trees
- reconstruction laws are tested on small examples

Current status: **landed**

## 4. Phase B — Broader solver, recurrence, and memo support

### Goal

Expand interval-family solver power only after public representation exists.

### Deliverables

- reconstruction-aware recurrence protocol
- optional branch infeasibility support
- one additional memo backend if needed
- additional split policies where semantically justified

### Tickets

#### B1. Recurrence-with-choice protocol

Allow recurrences to return both value and reconstruction choice metadata.

Acceptance criteria:

- narrow Stage 1 recurrence remains supported
- richer recurrence path can drive materialized interval-rooted output
- compatibility story is documented

#### B2. Sparse memo backend

Introduce a sparse or map-based memo only if a concrete use case requires it.

Acceptance criteria:

- clear motivation over `triangular_memo`
- stable lookup/store semantics
- tests compare behavior against dense memo where both apply

#### B3. Partial / infeasible branch handling

Support recurrences where some splits are invalid after child evaluation.

Acceptance criteria:

- infeasible branch semantics are explicit
- stats semantics remain defined
- solver tests cover mixed feasible/infeasible split sets

#### B4. Additional split policies

Add selective policies only where they encode real legality or search intent.

Acceptance criteria:

- semantics are documented
- policy order is deterministic
- policies remain interval-family-owned

## 5. Phase C — Interval-rooted encoding and model integration

### Goal

Design interval-rooted integration honestly, with interval-owned feature contracts.

### Deliverables

- interval-rooted feature schema proposal
- encoding contract for interval-rooted public candidates
- model-integration strategy if needed by downstream search or learning work

### Tickets

#### C1. Feature schema design

Define what an interval-rooted solution exposes as features.

Acceptance criteria:

- schema does not depend on partition labels
- reconstruction-to-feature mapping is deterministic
- encoded width/structure is documented

#### C2. Encoding / bridge-like layer

Introduce an interval-family-specific encoder only if justified.

Acceptance criteria:

- no reuse of `tree_bridge` assumptions
- stable encoding order documented
- tests cover legal interval-rooted examples

#### C3. Learned-model integration

Only if a real consumer exists, define how interval-rooted features feed model layers.

Acceptance criteria:

- consumer API is known
- feature semantics are stable enough to justify model coupling
- no premature coupling to unrelated solver families

## 6. Phase D — Adapters, cleanup, and optional convergence

### Goal

Clean up coexistence after the interval family is stable.

### Deliverables

- compatibility adapters to/from specialized interval APIs if worthwhile
- namespace cleanup toward canonical `ctdp`
- decision on relationship to `interval_dp`
- only then, any exploration of shared abstractions

### Tickets

#### D1. `interval_dp` compatibility review

Decide whether `interval_dp` remains specialized or should delegate internally.

Acceptance criteria:

- comparison against Stage 2 representation is documented
- no performance regressions are introduced blindly
- migration, if any, is optional and explicit

#### D2. Namespace cleanup

Reduce visible `ct_dp` residue where canonical `ctdp` placement is ready.

Acceptance criteria:

- canonical public names exist first
- cleanup does not break supported users accidentally

#### D3. Shared abstraction review

Only after interval and partition families are both mature, review commonality.

Acceptance criteria:

- at least two concrete families expose a meaningful shared core
- abstraction removes real duplication rather than hiding semantic differences

## 7. Ordering constraints

The following ordering constraints should be treated as hard rules.

1. Phase A before Phase B
2. Phase B before any strong integration promises in Phase C
3. Phase D only after Phase A/B/C produce stable public surfaces
4. no tree-family common abstraction work before two stable sibling families exist

## 8. Recommended near-term backlog

The best near-term backlog is:

1. B1 — recurrence-with-choice
2. B2 — sparse memo backend (only if justified)
3. B3 — partial / infeasible branch handling
4. B4 — additional split policies (only where semantically real)

That sequence keeps Stage 2 product-shaped rather than infrastructure-shaped.

## 9. Exit criteria

Stage 2 should be considered coherent only when:

- interval-rooted public solution values exist
- reconstruction is stable and tested
- broader interval solver features serve those public values
- integration contracts, if added, are interval-family-owned
- coexistence with `interval_dp` is explicit rather than accidental


