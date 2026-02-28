# Weighted Graph Infrastructure — Design Rationale

**Decision:** Option C (weighted_view wrapper)  
**Date:** February 2026  
**Version:** v0.7.2 → v0.8  
**Status:** Implemented  

## Context

The CT-DP graph layer needed weighted edge support to unblock three
algorithms: shortest path (Dijkstra/Bellman-Ford), min-cut (Stoer-Wagner),
and bipartite matching (Hopcroft-Karp).  The existing `constexpr_graph`
stores unweighted CSR adjacency only.  The `property_map` header provides
node-indexed external annotations but has no edge equivalent.

## Options Considered

### Option A — BGL-style External Edge Maps Only

Pass graph + edge weight map as separate arguments to every algorithm.

**Advantages:**
- Zero changes to any existing type.
- Maximum flexibility — same graph, different weight interpretations.

**Disadvantages:**
- No concept enforces that graph and weight map agree on edge count.
  Mismatched sizes are a runtime/consteval failure, not a compile-time
  concept failure.
- Algorithm signatures become noisy: `dijkstra(graph, weights, source)`
  instead of `dijkstra(weighted_graph, source)`.
- Synchronisation burden on the caller — if the graph is rebuilt (e.g.
  after coarsening), the weight map must be rebuilt in lockstep.

### Option B — Weight Array Baked into CSR

Add a `weights_` array parallel to `targets_` inside `constexpr_graph`,
controlled by a `Weight` template parameter (defaulting to `void` for
unweighted).

**Advantages:**
- Maximum locality during constexpr evaluation — weight sits next to
  target in the same object.
- Simplest iteration: `out_neighbors` returns `(node_id, Weight)` pairs.
- Single object to pass around.

**Disadvantages:**
- **Dead storage:** All 17 existing graph headers and every unweighted
  algorithm (coloring, SCC, topological sort, connected components,
  fusion legality) would carry `MaxE × sizeof(Weight)` of unused array.
  At `MaxE = 256` and `Weight = double`, that's 2 KB of dead compile-time
  state per graph instance.
- **Template parameter proliferation:** `constexpr_graph<MaxV, MaxE>` becomes
  `constexpr_graph<MaxV, MaxE, Weight>`.  Every existing template
  specialisation, concept check, and friend declaration must be updated.
  `graph_builder`, `symmetric_graph`, `symmetric_graph_builder`, `coarsen`,
  `from_stencil`, `from_pipeline` — all would need the extra parameter.
- **Multiple weight interpretations:** If the same topology needs cost
  weights for Dijkstra and capacity weights for max-flow, two separate
  graph instances are required, duplicating the entire CSR structure.

### Option C — Weighted View Wrapper (chosen)

Edge weights live in `edge_property_map` (external, edge-indexed).
`weighted_view` bundles a graph reference + weight map reference into a
single object that models the `weighted_graph_queryable` concept.

**Advantages:**
- **constexpr_graph unchanged.** No dead storage, no template parameter
  creep, no impact on the 17 existing graph headers.
- **Unweighted algorithms unchanged.** Coloring, SCC, topological sort
  continue to constrain on `graph_queryable` — they never see weights.
- **Multiple weight interpretations.** Same graph topology can carry
  cost, capacity, and latency weights via different `edge_property_map`
  instances.  Each produces a different `weighted_view`.
- **Concept-enforced agreement.** `weighted_graph_queryable` requires
  both adjacency access AND weight access on the same object.  You
  can't accidentally pass a mismatched map.
- **Mirrors the space layer pattern.** `valid_view` wraps
  `descriptor_space` with a predicate filter.  `weighted_view` wraps
  `constexpr_graph` with edge weights.  Same wrapper-adds-capability
  philosophy throughout CT-DP.

**Disadvantages:**
- One accessor (`edge_begin_offset`) added to `constexpr_graph` — minimal
  and backward-compatible.
- Weight lookup during iteration is `edge_map[offset + i]` rather than a
  direct field access.  For constexpr evaluation at ≤256 nodes (the
  `ct_limits` ceiling), this is negligible.
- `weighted_view` stores pointers to graph and weight map, requiring that
  both outlive the view.  This matches `adjacency_range`'s existing
  lifetime model and is the standard constexpr pattern for composed views.

## Decision

**Option C** was chosen for consistency with the space layer's wrapper
model and for zero impact on the existing unweighted graph infrastructure.

## Binding Model — Graph ↔ Map Agreement

Concept-enforced availability of weight access; constructor enforces
size and topology agreement.

`edge_property_map` records the `topology_token` of the graph it was
built for (set automatically by all factory functions).  `weighted_view`'s
constructor checks:

1. `weights.size() == graph.edge_count()` — size agreement
2. `weights.token() == graph.token()` — topology agreement

A mismatch is a constexpr evaluation failure (throws `std::logic_error`
in consteval context, which terminates compilation).  Maps constructed
manually (without a token) have `token.value == 0` and skip the topology
check, preserving backward compatibility while still enforcing size.

The `topology_token` is a 64-bit FNV-1a hash of `(V, E, CSR offsets,
CSR targets)`, computed deterministically by `constexpr_graph::token()`.
Same input edges → same CSR ordering → same token.  Any structural
transform (coarsen, rebuild, filter) produces a different token.

## Concept Hierarchy

```
graph_queryable                    (existing — coloring, SCC, topo-sort)
  ├─ sized_graph                   (existing — adds edge_count)
  ├─ symmetric_graph_queryable     (existing — adds undirected guarantees)
  ├─ bipartite_graph_queryable     (NEW — adds left/right partition)
  └─ weighted_graph_queryable      (NEW — adds edge_weight, weighted_out_neighbors)
       └─ symmetric_weighted_queryable  (NEW — undirected + weighted)
```

Algorithms declare exactly which concept they need.  A coloring algorithm
constraining on `symmetric_graph_queryable` will reject a `weighted_view`
over a directed graph at compile time.  Hopcroft-Karp constraining on
`bipartite_graph_queryable` will reject any graph without an explicit
left/right partition.

## Edge Identity

The CSR position (0..E-1) serves as the edge identity, wrapped in the
`edge_id` strong type (mirroring `node_id`).  This prevents accidental
mixing of edge and node indices in algorithm code.

Edge IDs are NOT stable across graph construction or transformation.
Persisting edge IDs beyond the lifetime of the graph instance is
undefined behaviour.  The `topology_token` mechanism detects stale
or mismatched edge annotations at constexpr evaluation time.

The edge identity is deterministic because `graph_builder::finalise()`
sorts edges by `(src, dst)` before building the CSR.  Same input edges
in any insertion order → same CSR → same edge IDs → same token.

The "one true primitive" for edge access is `edge_range(u)` which
returns `(edge_id begin, edge_id end)`.  `edge_target(edge_id)` gives
the target node, and `edge_weight(edge_id)` (on weighted_view) gives
the weight.  `weighted_out_neighbors` is sugar that zips these into
`(target, weight, eid)` triples for convenient iteration.

## Symmetric Weight Invariant

For undirected weighted graphs (Stoer-Wagner, etc.), each undirected
edge `{u, v}` is stored as two directed edges `u→v` and `v→u` at
different CSR positions.

Symmetry of weight values is enforced through three mechanisms
(users choose one):

1. **Guaranteed by construction:** `make_symmetric_weight_map` calls
   `fn(min(u,v), max(u,v))` — canonicalised argument order ensures
   both directed edges receive identical weights.  This is the
   recommended path.

2. **Validated by check:** `verify_symmetric_weights(graph, map)`
   performs a constexpr O(E × max_degree) check that
   `weight(u→v) == weight(v→u)` for all edges.  Use after manual
   map construction to prove symmetry.

3. **Required by precondition:** Algorithms constraining on
   `symmetric_weighted_queryable` assume symmetry.  If the map
   was constructed via path (1) or validated via path (2), this
   holds.  Otherwise it is the caller's responsibility.

The `symmetric_weighted_queryable` concept checks structural capability
(undirected graph + weight access), not value-level symmetry.  The
concept tells you "this type *can* be symmetric"; the construction/
validation path tells you "this instance *is* symmetric".

## Files Added

| File | Purpose |
|------|---------|
| `include/ctdp/graph/edge_property_map.h` | Edge-indexed external property map with topology token binding |
| `include/ctdp/graph/weighted_view.h` | Wrapper + concepts + symmetric validation |
| `include/ctdp/graph/weighted_graph_builder.h` | Weight map factories (token-bound) |
| `tests/graph/test_weighted_view.cpp` | 25+ tests: token binding, mismatch detection, symmetric validation, edge ordering determinism |
| `include/ctdp/graph/bipartite_graph.h` | Bipartite graph type, builder, `bipartite_graph_queryable` concept |
| `include/ctdp/graph/bipartite_matching.h` | Hopcroft-Karp O(E√V) maximum cardinality matching |
| `tests/graph/test_bipartite_matching.cpp` | 20+ tests: construction, matching, verification, constexpr pipeline |
| `docs/weighted_graph_design.md` | This document |

## Files Modified

| File | Change |
|------|--------|
| `include/ctdp/graph/graph_concepts.h` | Added `edge_id` strong type, `topology_token` type |
| `include/ctdp/graph/constexpr_graph.h` | Added `edge_begin_offset(u)`, `edge_range(u)`, `edge_target(e)`, `token()` |
| `include/ctdp/graph/symmetric_graph.h` | Forwarded `edge_begin_offset(u)`, `edge_range(u)`, `edge_target(e)`, `token()` |
| `tests/CMakeLists.txt` | Registered `test_weighted_view` |

No existing tests, algorithms, or headers were changed.

## Unblocked Work

- **Shortest path** (Dijkstra, Bellman-Ford): constrain on `weighted_graph_queryable`, iterate via `weighted_out_neighbors`.
- **Min-cut** (Stoer-Wagner): constrain on `symmetric_weighted_queryable`, use canonicalised undirected weights.
- **Bipartite matching** (Hopcroft-Karp): DONE — `bipartite_graph.h` + `bipartite_matching.h`.  Dedicated `bipartite_graph` type enforces partition by construction (mirroring `symmetric_graph`'s construction-enforced symmetry).  `bipartite_graph_queryable` concept constrains the algorithm.
- **v0.8 calibration bridge**: edge feature encoding can follow the same external-map pattern.
