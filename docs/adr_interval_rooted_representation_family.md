# ADR: Introduce Interval-Rooted Representation as a Sibling Family

**Decision:** Introduce interval-rooted ordered decomposition as a separate sibling representation family, not as an extension of the canonical partition-rooted tree family.  
**Date:** May 20, 2026  
**Status:** Proposed  
**Owners:** Solver architecture / space layer / interval work

## Context

Sprint 9 Stage 1 has now landed a narrow interval-solving substrate under the canonical `ctdp` surface:

- `ctdp::solver::interval_context`
- `ctdp::solver::plans::interval_partition_plan`
- `ctdp::solver::policies::all_binary_splits`
- `ctdp::solver::memo::triangular_memo<Value>`
- `ctdp::solver::algorithms::interval_solver`

That Stage 1 slice was intentionally narrow. It delivered the minimal substrate for ordered binary interval decomposition without attempting to change the framework's existing tree architecture.

The existing canonical tree family, per `docs/adr_partition_rooted_tree_canonical.md`, is explicitly **partition-rooted**:

- partition descriptors over a fixed item set
- canonical group labels
- per-group child plans
- fixed-width child feature blocks in canonical group order

Ordered interval decomposition is a different family:

- every node owns a contiguous interval `[i, j)`
- every internal node owns a split `k` with `i < k < j`
- children are ordered left/right
- legality is about interval contiguity and split validity, not partition membership

The main deferred Sprint 9 value now lies in:

- first-class interval-rooted public solution values
- reconstruction and materialization of interval solutions
- broader interval-family solver support
- interval-rooted integration work where justified

## Decision

We will introduce **interval-rooted representations** as a separate sibling architecture family.

We will **not** fold interval-rooted semantics into the current partition-rooted tree family.

Stage 2 interval work should therefore define its own:

- public candidate / plan types
- reconstruction helpers
- legality invariants
- optional feature / bridge schema
- solver-side integration path

The intended public direction is toward canonical `ctdp` placement, even if some implementation residue still exists under `ct_dp` during transition.

## Invariants to Preserve

The following invariants govern the Stage 2 interval-rooted family.

1. **Interval-rooted semantics**
   - Every represented node names a half-open interval `[i, j)`.
   - Every represented internal node has exactly one strictly interior split `k`.
   - Left and right children are ordered.

2. **Family separation**
   - Interval-rooted representations do not depend on partition labels.
   - Interval-rooted legality is not expressed in partition/group terms.
   - Interval-rooted feature schemas, if added later, are family-owned.

3. **Partition-family preservation**
   - `tree_space` remains partition-rooted.
   - `reduction_tree_space` remains partition-rooted.
   - `tree_point` and `tree_bridge` retain their current meaning.

4. **No false genericity**
   - We do not introduce a universal tree abstraction merely to house both partition-rooted and interval-rooted structures.
   - Shared abstractions should be considered only after multiple concrete sibling families exist and expose a genuine shared core.

## Consequences

### Positive

- Gives the deferred Sprint 9 work a coherent architectural home.
- Preserves the existing partition-tree ADR and its contracts.
- Allows honest interval-specific reconstruction, legality, and feature design.
- Avoids forcing ordered interval semantics through partition-centric APIs.

### Negative

- The framework will continue to have more than one tree-like representation family.
- Some concepts and helper APIs may be duplicated temporarily until a real shared abstraction emerges.
- Existing specialized interval APIs may coexist with richer interval-rooted types for some time.

## Explicit Non-Goals

The following are not decided or delivered by this ADR.

- replacing `interval_dp`
- automatic migration of all interval users to `interval_solver`
- universal solver abstraction across all DP families
- integration of fixed-tree + permutation into the interval family
- redesign of `tree_space`, `tree_point`, or `tree_bridge`

## Deferred Alternatives

1. **Treat interval-rooted problems as just another `tree_space` configuration**
   - rejected for now because it weakens the partition-rooted tree model and obscures different semantics

2. **Keep interval work solver-only forever**
   - rejected for now because the main deferred Sprint 9 value is public interval-rooted representation and reconstruction

3. **Introduce a universal tree abstraction immediately**
   - deferred because there is still only one mature canonical tree family and one emerging sibling family

## Rule of Thumb

When evaluating Stage 2 interval features, ask:

> Is this naturally an ordered contiguous interval decomposition concern, with its own reconstruction and legality rules?

- **Yes** → place it in the interval-rooted sibling family.
- **No** → keep it in the existing partition family or another future sibling family.

