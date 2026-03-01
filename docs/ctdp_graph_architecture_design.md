# CT-DP Graph Library — Architecture & Design

**Andrew Drakeford · February 2026 · CT-DP v0.8**

---

## 1. Overview

The CT-DP graph library is a C++20 header-only library providing constexpr graph algorithms for compile-time dynamic programming. Every algorithm is fully constexpr: graph construction, analysis, and verification can all execute during compilation, with results checked by `static_assert`. The same code runs identically at runtime.

The library follows a traits-based architecture where algorithms are decoupled from storage via `graph_traits<G>`. This single customisation point tells algorithms how to allocate working arrays, what index type to use, and what the compile-time capacity bounds are. Three graph representations share a common algorithmic interface: constexpr (fixed-size, consteval-safe), runtime (vector-backed, dynamically constructed), and implicit (procedurally generated, zero storage).

| Metric | Value |
|--------|-------|
| Headers | 23 (graph) + 4 (engine bridge) + 1 (core) |
| Lines of code | ~4,800 |
| Graph representations | 4 (constexpr, symmetric, bipartite, runtime, implicit) |
| Algorithms | 8 (topo sort, CC, SCC, colouring, matching, Dijkstra, Stoer–Wagner, coarsen/fusion) |
| Test cases (ctest) | 536 |
| C++ standard | C++20 (concepts, constexpr, NTTP) |
| Compiler support | GCC 13+, Clang 17+, MSVC 19.36+ |

---

## 2. Architecture Layers

Each layer depends only on layers below it. Algorithms never depend on a specific graph representation, and the engine never depends on algorithm internals.

```
Engine Bridge           graph_to_space, graph_to_constraints, coloring_to_groups, graph_types
────────────────────────────────────────────────────────────────────────────────
Algorithms              topo_sort, cc, scc, coloring, matching, dijkstra, stoer_wagner, coarsen, fuse
────────────────────────────────────────────────────────────────────────────────
Traits + Concepts       graph_traits<G>, graph_queryable, sized_graph, symmetric_graph_queryable
────────────────────────────────────────────────────────────────────────────────
Representations         constexpr_graph, symmetric_graph, bipartite_graph, runtime_graph, implicit_graph
────────────────────────────────────────────────────────────────────────────────
Annotations             property_map, edge_property_map, weighted_view, kernel_info, capacity_guard
────────────────────────────────────────────────────────────────────────────────
Core                    constexpr_vector, rt_array, node_id, edge_id, topology_token
```

---

## 3. Traits System

### 3.1 Purpose

`graph_traits<G>` is the single customisation point that decouples algorithm storage from graph representation. The primary template is intentionally undefined; a compilation error for unsupported graph types gives a clear diagnostic. Specialisations exist for all concrete graph types.

Algorithms call `make_node_array<double>(g)` and never mention storage. Adding a new graph type requires only a traits specialisation — no modification to existing code.

### 3.2 Trait Members

| Member | Purpose |
|--------|---------|
| `is_constexpr_storage` | True for fixed-size `std::array` storage (constexpr-safe) |
| `max_nodes` | Compile-time upper bound on vertex count |
| `max_edges` | Compile-time upper bound on edge count |
| `node_index_type` | Integer type for node indices (currently `uint16_t` everywhere) |
| `node_array<T>` | Array type alias for node-indexed data |
| `edge_array<T>` | Array type alias for edge-indexed data |
| `make_node_array<T>(cap)` | Factory: construct a node-indexed array |
| `make_node_array<T>(cap, fill)` | Factory: construct a node-indexed array with fill value |
| `make_edge_array<T>(cap)` | Factory: construct an edge-indexed array |
| `make_edge_array<T>(cap, fill)` | Factory: construct an edge-indexed array with fill value |

Convenience aliases: `node_index_t<G>` for the node index type, `node_nil_v<G>` for the sentinel value (max of the index type). Free functions `make_node_array(g)` and `make_edge_array(g)` in `array_helpers.h` delegate to the traits.

### 3.3 Constexpr Graph Specialisation

```cpp
template <typename G>
struct graph_traits;  // primary — must be specialised

template <std::size_t MaxV, std::size_t MaxE>
struct graph_traits<constexpr_graph<MaxV, MaxE>> {
    static constexpr bool is_constexpr_storage = true;
    using node_index_type = std::uint16_t;

    template <typename T>
    using node_array = std::array<T, MaxV>;
    template <typename T>
    using edge_array = std::array<T, MaxE>;

    template <typename T>
    static constexpr node_array<T> make_node_array(std::size_t /*cap*/) {
        return node_array<T>{};
    }
    template <typename T>
    static constexpr node_array<T> make_node_array(std::size_t /*cap*/,
                                                    const T& fill) {
        node_array<T> a{};
        for (auto& x : a) x = fill;
        return a;
    }
    // make_edge_array analogous
};
```

### 3.4 Runtime Graph Specialisation

```cpp
template <>
struct graph_traits<runtime_graph> {
    static constexpr bool is_constexpr_storage = false;
    using node_index_type = std::uint16_t;

    template <typename T>
    using node_array = rt_array<T>;
    template <typename T>
    using edge_array = rt_array<T>;

    template <typename T>
    static node_array<T> make_node_array(std::size_t cap) {
        return node_array<T>(cap);
    }
    template <typename T>
    static node_array<T> make_node_array(std::size_t cap, const T& fill) {
        return node_array<T>(cap, fill);
    }
    // make_edge_array analogous
};
```

### 3.5 Free-Standing Helpers

```cpp
template <typename T, typename G>
constexpr auto make_node_array(const G& g) {
    return graph_traits<G>::template make_node_array<T>(g.node_capacity());
}

template <typename T, typename G>
constexpr auto make_node_array(const G& g, const T& fill) {
    return graph_traits<G>::template make_node_array<T>(g.node_capacity(), fill);
}
// same: make_edge_array<T>(g), make_edge_array<T>(g, fill)
```

### 3.6 Specialisation Summary

| Graph Type | is_constexpr_storage | max_nodes | max_edges | Note |
|-----------|---------------------|-----------|-----------|------|
| `constexpr_graph<V, E>` | true | V | E | Directed, CSR, fixed-size |
| `symmetric_graph<V, E>` | true | V | 2×E | Undirected wrapper, stores both directions |
| `bipartite_graph<L, R, E>` | true | L+R | E | Construction-enforced partition |
| `runtime_graph<V, E>` | false | V | E | Directed, CSR, vector-backed |

### 3.7 Boundaries

- Traits do not carry capacity values. `node_capacity()` is on the graph object.
- Traits do not drive algorithm selection. One algorithm body serves all scales.
- Traits do not (yet) drive `node_id` width at the strong-type level. `node_index_type` is in the traits and used in result types. The `node_id` strong type wrapper still uses `uint16_t` internally. Full widening is deferred — 65K nodes covers the engine use case and most practical runtime problems.

---

## 4. Concept Hierarchy

```
graph_queryable           node_count(), edge_count(), token()
  └─ sized_graph          + node_capacity(), edge_capacity()
       └─ adjacency_queryable     + degree(n), edge_range(n), edge_target(e)
            ├─ weighted_graph_queryable      + edge_weight(e), weighted_out_neighbors(u)
            ├─ symmetric_graph_queryable     + undirected guarantee
            ├─ symmetric_weighted_queryable  (both above)
            └─ bipartite_graph_queryable     + left_count(), right_count(),
                                               left_neighbors(), right_neighbors()
```

### 4.1 Concept Requirements

| Concept | Requirements | Satisfied By |
|---------|-------------|--------------|
| `graph_queryable` | `node_count()`, `out_neighbors(u)`, `out_degree(u)`, `max_out_degree()`, `has_node(u)`, `empty()` | All graph types |
| `sized_graph` | `graph_queryable` + `edge_count()`, `node_capacity()`, `edge_capacity()` | constexpr_graph, symmetric_graph, runtime_graph |
| `adjacency_queryable` | `sized_graph` + `degree(n)`, `edge_range(n)`, `edge_target(e)` | All CSR graph types |
| `weighted_graph_queryable` | `adjacency_queryable` + `edge_weight(e)`, `weighted_out_neighbors(u)` | weighted_view |
| `symmetric_graph_queryable` | `adjacency_queryable` + undirected guarantee | symmetric_graph |
| `symmetric_weighted_queryable` | undirected + weighted | weighted_view over symmetric_graph |
| `bipartite_graph_queryable` | `adjacency_queryable` + `left_count()`, `right_count()`, `left_neighbors()`, `right_neighbors()` | bipartite_graph |

### 4.2 Design Principles

`symmetric_graph_queryable` requires adjacency iteration plus a semantic guarantee of undirectedness. It does **not** require `reverse_edge(e)`. If a future algorithm requires reverse-edge lookup, it can constrain on a tighter `reverse_edge_queryable` concept rather than burdening all symmetric graphs.

`bipartite_graph_queryable` refines `adjacency_queryable` (not just `sized_graph`), because Hopcroft–Karp needs `edge_range()` and `edge_target()`.

Every concept that refines `adjacency_queryable` also implicitly refines `sized_graph` and `graph_queryable`. The lattice is strictly hierarchical.

---

## 5. Graph Representations

### 5.1 constexpr_graph\<MaxV, MaxE\>

The primary representation. A directed graph stored in Compressed Sparse Row (CSR) format using fixed-size `std::array` storage. Fully constexpr: can be constructed, queried, and passed to algorithms inside `consteval` contexts. Constructed via `graph_builder<MaxV, MaxE>` which enforces canonicalisation: edges sorted by (src, dst), duplicates removed, self-edges removed.

```cpp
constexpr auto g = []() {
    graph_builder<8, 16> b;
    auto a = b.add_node(); auto bb = b.add_node();
    b.add_edge(a, bb);
    return b.finalise();
}();
static_assert(topological_sort(g).is_dag);
```

### 5.2 symmetric_graph\<MaxV, MaxE\>

An undirected graph. Wraps a `constexpr_graph<MaxV, 2*MaxE>` storing both edge directions internally. `add_edge(u, v)` on the builder inserts both u→v and v→u. Provides `neighbors(u)`, `adjacent(u, v)`, `degree(u)`, and `max_degree()` satisfying `symmetric_graph_queryable`. Required by graph colouring and Stoer–Wagner min cut.

### 5.3 bipartite_graph\<MaxL, MaxR, MaxE\>

A bipartite graph with construction-enforced left/right partition, mirroring `symmetric_graph`'s construction-enforced symmetry. Satisfies `bipartite_graph_queryable`. Required by Hopcroft–Karp matching.

### 5.4 runtime_graph\<MaxV, MaxE\>

A runtime-constructed CSR graph using `std::vector` for internal storage. MaxV and MaxE are compile-time upper bounds that size the result arrays; the graph itself can be smaller. Constructed via `runtime_graph_builder<MaxV, MaxE>` with the same canonicalisation rules as `graph_builder`. The builder throws on capacity overflow, invalid node IDs, or edges on empty graphs.

A companion `symmetric_runtime_graph_builder<MaxV, MaxE>` inserts both edge directions for undirected runtime graphs.

All algorithms that accept `sized_graph` work with runtime_graph. The traits specialisation sets `is_constexpr_storage = false` but uses the same `std::array<T, MaxV>` result types, so constexpr and runtime graphs with the same MaxV produce identical result types.

```cpp
class runtime_graph {
    std::vector<std::size_t> offsets_;
    std::vector<uint16_t> targets_;
    std::size_t node_count_{};
    std::size_t edge_count_{};
    topology_token token_{};
public:
    std::size_t node_count() const { return node_count_; }
    std::size_t edge_count() const { return edge_count_; }
    std::size_t node_capacity() const { return node_count_; }
    std::size_t edge_capacity() const { return edge_count_; }
    topology_token token() const { return token_; }
    // ... same query interface as constexpr_graph
};
```

`node_capacity() == node_count()` for runtime graphs — they are exact-fit. Algorithms allocate working arrays to capacity before running, so exact-fit is correct.

### 5.5 implicit_graph

Procedurally-generated graphs that store no edges. A callable produces the adjacency list on demand. Satisfies `graph_queryable` but not `sized_graph` (no edge_count or capacity). Used by `from_stencil` for regular-grid stencil patterns where explicitly storing O(V×stencil_size) edges would waste memory.

---

## 6. Weighted Graph Infrastructure

### 6.1 Design

Edge weights live in `edge_property_map` (external, edge-indexed). `weighted_view` bundles a graph reference + weight map reference into a single object that models `weighted_graph_queryable`.

This design was chosen over two alternatives:

- **External edge maps only** (BGL-style): no concept enforcement of graph/map agreement; noisy algorithm signatures; synchronisation burden.
- **Weight array baked into CSR**: dead storage for unweighted algorithms; template parameter proliferation; no support for multiple weight interpretations on the same topology.

The `weighted_view` wrapper has zero impact on the existing unweighted graph infrastructure. Unweighted algorithms continue to constrain on `graph_queryable` and never see weights. The same graph topology can carry cost, capacity, and latency weights via different `edge_property_map` instances.

### 6.2 Binding Model

`edge_property_map` records the `topology_token` of the graph it was built for. `weighted_view`'s constructor checks:

1. `weights.size() == graph.edge_count()` — size agreement
2. `weights.token() == graph.token()` — topology agreement

A mismatch is a constexpr evaluation failure (throws `std::logic_error` in consteval context, which terminates compilation). Maps constructed manually (without a token) have `token.value == 0` and skip the topology check, preserving backward compatibility while still enforcing size.

### 6.3 Edge Identity

The CSR position (0..E-1) serves as the edge identity, wrapped in the `edge_id` strong type (mirroring `node_id`). Edge IDs are deterministic because `graph_builder::finalise()` sorts edges by `(src, dst)` before building the CSR. Same input edges in any insertion order → same CSR → same edge IDs → same token.

Edge IDs are NOT stable across graph construction or transformation. The `topology_token` mechanism detects stale or mismatched edge annotations at constexpr evaluation time.

The "one true primitive" for edge access is `edge_range(u)` returning `(edge_id begin, edge_id end)`. `edge_target(edge_id)` gives the target node, and `edge_weight(edge_id)` (on weighted_view) gives the weight. `weighted_out_neighbors(u)` is sugar that zips these into `(target, weight, eid)` triples.

### 6.4 Symmetric Weight Invariant

For undirected weighted graphs, each undirected edge `{u, v}` is stored as two directed edges at different CSR positions. Symmetry of weight values is enforced through three mechanisms:

1. **Guaranteed by construction:** `make_symmetric_weight_map` calls `fn(min(u,v), max(u,v))` — canonicalised argument order ensures both directed edges receive identical weights. This is the recommended path.

2. **Validated by check:** `verify_symmetric_weights(graph, map)` performs a constexpr O(E × max_degree) check that `weight(u→v) == weight(v→u)` for all edges.

3. **Required by precondition:** Algorithms constraining on `symmetric_weighted_queryable` assume symmetry.

The `symmetric_weighted_queryable` concept checks structural capability, not value-level symmetry. The concept tells you "this type *can* be symmetric"; the construction/validation path tells you "this instance *is* symmetric".

---

## 7. Topology Tokens

**What a token identifies:** the CSR topology — the node count, edge count, and the specific pattern of which edges connect which nodes, in CSR storage order. Two graphs with identical topology have identical tokens. Changing any edge or adding/removing a node changes the token.

**What a token does not identify:** edge weights, property map contents, or algorithmic results.

**Token lifecycle:**

- Computed at `finalise()` in both constexpr and runtime builders, via FNV-1a hash of the CSR structure.
- Wrappers (`weighted_view`, `symmetric_graph`, `bipartite_graph`) forward `token()` from their underlying graph unchanged.
- Algorithms receive tokens through the graph interface and treat them as opaque value types. Algorithms do not embed tokens in result types — results are pure value types.
- `edge_property_map` stores the topology token it was bound to at construction. If later paired with a different graph (different token), `weighted_view` construction detects the mismatch.

---

## 8. `rt_array<T>` — Runtime Container

A thin wrapper around `std::vector<T>` with fixed-capacity semantics, used as the node/edge array type for runtime graphs:

```cpp
template <typename T>
class rt_array {
    std::vector<T> data_;
public:
    explicit rt_array() = default;
    explicit rt_array(std::size_t n) : data_(n) {}
    rt_array(std::size_t n, const T& fill) : data_(n, fill) {}

    rt_array(rt_array&&) noexcept = default;
    rt_array& operator=(rt_array&&) noexcept = default;
    rt_array(const rt_array&) = default;
    rt_array& operator=(const rt_array&) = default;

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }
    std::size_t size() const { return data_.size(); }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
};
```

- No `push_back`, `resize`, or `emplace_back`. Fixed size after construction — same invariant as `std::array`.
- Default-constructed `rt_array` has `size() == 0`, making default-constructed runtime results safely empty rather than dangerously unsized.
- Explicit `data()` for interop with C APIs, span conversion, and test inspection.

---

## 9. Algorithms

Every algorithm follows a consistent pattern: accept a graph satisfying a concept, derive MaxV from `graph_traits<G>::max_nodes`, compute the result, verify correctness, and return a result struct. All algorithms are fully constexpr.

### 9.1 Algorithm Summary

| Algorithm | Header | Concept | Complexity | Verification |
|-----------|--------|---------|------------|-------------|
| Topological sort | topological_sort.h | sized_graph | O(V + E) | Kahn's; cycle detection |
| Connected components | connected_components.h | sized_graph | O(V + E) amort. | Union–Find |
| Strongly connected comp. | scc.h | sized_graph | O(V + E) | Iterative Tarjan |
| Graph colouring | graph_coloring.h | symmetric + sized | O(V log V + E) | Welsh–Powell; O(E) verify |
| Bipartite matching | bipartite_matching.h | bipartite_graph_queryable | O(E√V) | Hopcroft–Karp |
| Shortest path (Dijkstra) | shortest_path.h | weighted_graph_queryable | O((V+E) log V) | Triangle inequality |
| Min cut (Stoer–Wagner) | min_cut.h | symmetric_weighted | O(V³) | Canonical cut-weight |
| Coarsening / Fusion | coarsen.h, fuse_group.h | sized_graph | O(V + E) | Group contraction; DAG check |

### 9.2 Topological Sort

Kahn's algorithm. Returns `topo_result<MaxV>` containing `order` (a constexpr_vector of node_ids in topological order) and `is_dag` (false if a cycle is detected). Deterministic tie-breaking: smallest node ID first.

```cpp
constexpr auto r = topological_sort(g);
static_assert(r.is_dag);
static_assert(r.order[0] == node_id{0});
```

### 9.3 Connected Components

Union–Find with path compression and union-by-rank. Treats edges as undirected. Returns `components_result<MaxV>` with `component_of[v]` (dense component ID) and `component_count`.

### 9.4 Strongly Connected Components

Iterative Tarjan's algorithm (no recursion, constexpr-safe). Returns `scc_result<MaxV>` with `component_of[v]` and `component_count`. Components numbered in reverse topological order.

### 9.5 Graph Colouring

Welsh–Powell greedy colouring. Requires `symmetric_graph_queryable`. Sorts vertices by degree (descending), assigns the smallest available colour. Returns `coloring_result<MaxV>` with `color_of[v]`, `color_count`, and `verified` (O(E) check that no adjacent pair shares a colour).

MaxV is derived from the graph's traits: `graph_coloring(g)`, not `graph_coloring<4>(g)`.

### 9.6 Bipartite Matching

Hopcroft–Karp O(E√V) maximum cardinality matching. Requires `bipartite_graph_queryable`. The dedicated `bipartite_graph` type enforces partition by construction, mirroring `symmetric_graph`'s construction-enforced symmetry.

### 9.7 Dijkstra Shortest Path

**Concept:** `weighted_graph_queryable` (directed OK, symmetry not required)

**Algorithm:** Binary heap over traits-allocated node arrays. A single implementation serves all scales. The heap is a simple index-based binary min-heap: a `make_node_array<node_index_t<G>>(g)` position array and a `make_node_array<double>(g)` key array. Decrease-key via sift-up.

The heap is constexpr in C++20 (`std::swap` is constexpr, array mutation via `operator[]` is constexpr). At V ≤ 256 the heap has marginally more overhead than linear scan but this is undetectable in practice.

**Precondition:** all edge weights non-negative. Enforced via `throw std::domain_error` (compile-time diagnostic in constexpr per [expr.const], runtime exception otherwise).

**Weights** are supplied as a callable `WeightFn(node_id, node_id) → double`, decoupled from weight storage. Weights can come from a property map, a lambda, a lookup table, or a formula.

**Result type:**

```cpp
template <typename G>
struct shortest_path_result {
    using index_t = node_index_t<G>;
    typename graph_traits<G>::template node_array<double>  dist;
    typename graph_traits<G>::template node_array<index_t> pred;
    index_t source{};
    index_t node_count{};
    bool verified{false};
};
```

**Verification:** `verify_shortest_path(g, result)` checks in O(E):

1. `dist[source] == 0`
2. All finite `dist[v]` are non-negative
3. Edge feasibility: for every edge (u,v) with weight w, `dist[v] ≤ dist[u] + w`
4. Predecessor consistency: for every reachable v ≠ source, `pred[v]` is valid, reachable, and `dist[v] == dist[pred[v]] + weight(pred[v]→v)`. Predecessor chain terminates at source (no cycles).

Under non-negative weights, conditions 1–3 (feasible potential) plus condition 4 (tightness on spanning tree) imply optimality via LP duality.

**Edge cases:** empty graph (verified trivially), disconnected nodes (`dist = INFINITY`, `pred = NODE_NIL`), zero-weight edges (handled), self-loops (harmless), parallel edges (minimum dominates through relaxation).

```cpp
constexpr auto r = dijkstra(g, source, [](node_id u, node_id v) {
    return lookup_weight(u, v);
});
static_assert(r.verified);  // Triangle inequality proved at compile time
```

### 9.8 Stoer–Wagner Min-Cut

**Concept:** `symmetric_weighted_queryable` (undirected + weighted)

**Algorithm:** Stoer–Wagner with V−1 minimum-cut phases. Each phase performs a maximum adjacency ordering, identifies the cut-of-the-phase, then contracts the last two nodes.

**Node contraction** uses a dense V×V weight matrix. At V ≤ 64 (constexpr), this is `std::array<double, 64>` × 64 = 32 KiB — perfect cache locality and trivial merge logic. For runtime graphs the matrix uses `make_node_array<double>(g)` per row and scales naturally.

```
W = V×V zero-initialised matrix from graph adjacency
alive = all true

for phase = 0 to V-2:
    // Maximum adjacency ordering (linear scan for max key among alive nodes)
    // Cut of phase = key[last]
    // CONTRACT: merge last into second_last
    for v = 0 to V-1:
        W[second_last][v] += W[last][v]
        W[v][second_last] += W[v][last]
    alive[last] = false
```

Complexity: O(V²) per phase × (V−1) phases = O(V³). At V = 64 that's 262K iterations. At V = 1,000 it's ~10⁹, still under a second. If profiling shows this matters for large runtime graphs, a heap-based maximum-adjacency-ordering (O(E + V log V) per phase) can be added behind the same interface.

**Partition tracking:** a `node_array<index_t> merged_into` records which representative each node has been merged into. The full partition is reconstructed by walking `merged_into`.

**Weight precondition:** non-negative. Zero weights permitted. Ties in the maximum adjacency ordering broken by node index for determinism.

**Result type:**

```cpp
template <typename G>
struct min_cut_result {
    using index_t = node_index_t<G>;
    double cut_weight{};
    typename graph_traits<G>::template node_array<bool> partition;
    index_t node_count{};
    bool has_cut{false};
    bool verified{false};
};
```

`has_cut` is false for graphs with fewer than 2 nodes.

**Verification:** `verify_min_cut(g, result)` checks in O(E) that both partition sides are non-empty and the sum of crossing edge weights equals `cut_weight`. Summation uses canonical edge ordering (u < v dedup) for bitwise-identical IEEE 754 results.

### 9.9 Coarsening & Fusion

`fusion_legal` determines which adjacent kernel pairs can be fused based on kernel_info tags and fusability. `find_fusion_groups` performs greedy BFS-based merge. `coarsen` contracts the graph by merging all nodes in each group into a super-node, rewiring edges, and aggregating kernel_info. The coarsened graph is a DAG if the input was.

---

## 10. Result Types

### 10.1 Unified Pattern

Every algorithm result follows this structure:

```cpp
template <typename G>
struct algorithm_result {
    using index_t = node_index_t<G>;
    typename graph_traits<G>::template node_array<T> solution;
    index_t node_count{};
    bool verified{false};
};
```

### 10.2 Construction Contract

- **Constexpr graphs** (`is_constexpr_storage == true`): default construction is safe. Arrays are `std::array`, already sized.
- **Runtime graphs** (`is_constexpr_storage == false`): default-constructed results contain empty `rt_array`s with `size() == 0`. Writing to them is UB. **Runtime results must be created via factory functions.**

### 10.3 Factory Functions

```cpp
template <typename G>
constexpr auto make_shortest_path_result(const G& g) {
    shortest_path_result<G> r{};
    r.dist = make_node_array<double>(g, INFINITY);
    r.pred = make_node_array<node_index_t<G>>(g, NODE_NIL);
    r.node_count = static_cast<node_index_t<G>>(g.node_count());
    return r;
}

template <typename G>
constexpr auto make_min_cut_result(const G& g) {
    min_cut_result<G> r{};
    r.partition = make_node_array<bool>(g, false);
    r.node_count = static_cast<node_index_t<G>>(g.node_count());
    return r;
}
```

The algorithm verifier is called before returning. `result.verified = true` is set by the verifier, not by the algorithm.

### 10.4 Result Type Reference

| Result Type | Key Fields | Factory |
|-------------|-----------|---------|
| `topo_result<MaxV>` | order (constexpr_vector), is_dag | `make_topo_result(g)` |
| `components_result<MaxV>` | component_of[v], component_count | `make_components_result(g)` |
| `scc_result<MaxV>` | component_of[v], component_count | `make_scc_result(g)` |
| `coloring_result<MaxV>` | color_of[v], color_count, verified | `make_coloring_result(g)` |
| `shortest_path_result<MaxV>` | dist[v], pred[v], source, verified | `make_shortest_path_result(g)` |
| `min_cut_result<MaxV>` | cut_weight, partition[v], has_cut, verified | `make_min_cut_result(g)` |
| `fuse_group_result<MaxV>` | group_of[v], group_count, is_valid_dag | (from find_fusion_groups) |

---

## 11. Sentinel Values

| Sentinel | Value | Scope | Type Domain |
|----------|-------|-------|-------------|
| `NODE_NIL` | `std::numeric_limits<node_index_t<G>>::max()` | Project-wide | `node_index_t` (node indices) |
| `DIST_INF_SENTINEL` | `0xFFFE` | Internal to `bipartite_matching.h` | `uint16_t` (HK BFS levels) |
| `std::numeric_limits<double>::infinity()` | IEEE 754 +∞ | Project-wide | `double` (distances) |
| `false` | `0` | Result-local | `bool` (min-cut partition) |

When `node_index_t` widens to `uint32_t`, `NODE_NIL` becomes `0xFFFF'FFFF` automatically.

---

## 12. Constexpr Evaluation Limits

`ct_limits` documents tested constexpr bounds — advisory, not enforced:

```cpp
struct ct_limits {
    static constexpr std::size_t topo_sort_max = 512;
    static constexpr std::size_t coloring_max = 128;
    static constexpr std::size_t scc_max = 512;
    static constexpr std::size_t connected_components_max = 512;
    static constexpr std::size_t min_cut_max = 64;
    static constexpr std::size_t shortest_path_max = 256;
    static constexpr std::size_t bipartite_matching_max = 64;
    static constexpr std::size_t rcm_max = 256;
};
```

`capacity_guard` provides opt-in bounds checking:

```cpp
template <typename G>
constexpr void assert_ct_capacity(const G& g, std::size_t limit) {
    if constexpr (graph_traits<G>::is_constexpr_storage) {
        if (g.node_capacity() > limit)
            throw std::length_error("exceeds tested constexpr capacity");
    }
    // runtime graphs: no check
}
```

---

## 13. Annotations & Property Maps

`property_map<T, MaxV>` is a BGL-style external property map: a fixed-size array indexed by node_id. Graph topology is separate from semantic annotations. Provides O(1) access, constexpr construction, and factory functions.

`edge_property_map<Weight, MaxE>` is the edge-indexed counterpart, with topology token binding for graph/map agreement verification.

`kernel_info` describes the computational profile of a graph node: flops, bytes_read, bytes_written, kernel_tag, and fusability. Consumed by fusion legality, coarsening, roofline analysis, and the cost model. `kernel_map<MaxV>` is a type alias for `property_map<kernel_info, MaxV>`.

`capacity_guard` provides uniform compile-time and runtime capacity checks. Every algorithm that allocates MaxV-sized arrays passes through the guard, producing a clear diagnostic if the graph exceeds the array capacity.

---

## 14. Construction Factories

| Factory | Header | Output | Purpose |
|---------|--------|--------|---------|
| `graph_builder<V, E>` | graph_builder.h | `constexpr_graph<V, E>` | Directed graph from explicit add_node/add_edge |
| `symmetric_graph_builder<V, E>` | symmetric_graph.h | `symmetric_graph<V, E>` | Undirected; add_edge inserts both directions |
| `runtime_graph_builder<V, E>` | runtime_graph.h | `runtime_graph<V, E>` | Runtime directed; throws on overflow |
| `symmetric_runtime_graph_builder<V, E>` | runtime_graph.h | `runtime_graph<V, 2E>` | Runtime undirected; both directions |
| `from_pipeline<N>()` | from_pipeline.h | `constexpr_graph<N, N-1>` | Linear chain 0→1→…→N-1 |
| `from_stencil(dims, offsets)` | from_stencil.h | `implicit_graph` | Regular grid stencil pattern |

All builders share canonicalisation rules: edges sorted by (src, dst), duplicates removed, self-edges removed. The runtime builder additionally validates node IDs at insertion time and throws on overflow.

---

## 15. Engine Bridge

The engine bridge connects graph-level analysis to the DP solver. One-way dependency: engine → graph, never reverse.

### 15.1 Bridge Headers

| Header | Purpose |
|--------|---------|
| `graph_types.h` | Engine type aliases (engine_graph, engine_rt_graph, result types). Standard capacity MaxV=64, MaxE=256. |
| `graph_to_space.h` | Build schedule_space from graph + kernel_map. Topological sort, degree counts, descriptors. |
| `graph_to_constraints.h` | Extract dependency_set and resource_constraints for legality checking. |
| `coloring_to_groups.h` | Convert coloring_result to fuse_group_result (independent sets → parallel groups). |
| `build_fusion_graph.h` | `build_fusion_graph(kernel_list, cost_model)` → weighted graph |
| `interpret_results.h` | `interpret_shortest_path(result, kernels)` → `fusion_sequence`; `interpret_min_cut(result, kernels)` → `partition_decision` |

### 15.2 Engine Type Aliases

| Alias | Resolves To |
|-------|-------------|
| `engine_graph` | `constexpr_graph<64, 256>` |
| `engine_sym_graph` | `symmetric_graph<64, 256>` |
| `engine_rt_graph` | `runtime_graph<64, 256>` |
| `engine_topo_result` | `topo_result<64>` |
| `engine_cc_result` | `components_result<64>` |
| `engine_coloring_result` | `coloring_result<64>` |
| `engine_sp_result` | `shortest_path_result<64>` |
| `engine_min_cut_result` | `min_cut_result<64>` |

### 15.3 Bridge Semantics

The adapter constructs a graph via the builder, binds a weight map via `make_weight_map`, and returns `weighted_view`. Topology token flows through. Interpretation functions map node indices back to kernel identifiers.

Engine mapping is heuristic: shortest path provides an initial fusion ordering, min-cut suggests communication partitioning, colouring estimates register pressure. These are inputs to the cost model, not claimed as reductions of the optimisation problem to graph primitives.

---

## 16. Header Reference

### 16.1 Core

| Header | Purpose |
|--------|---------|
| `core/constexpr_vector.h` | Fixed-capacity vector for constexpr use. Underlies most result types. |
| `core/rt_array.h` | Fixed-capacity, heap-allocated array for runtime graphs. |

### 16.2 Graph Library

| Header | Category | Purpose |
|--------|----------|---------|
| `graph_concepts.h` | Foundation | node_id, edge_id, topology_token, concepts |
| `graph_traits.h` | Foundation | Primary template + specialisations, node_index_t\<G\>, node_nil_v\<G\> |
| `array_helpers.h` | Foundation | Free functions: make_node_array(g), make_edge_array(g) |
| `constexpr_graph.h` | Representation | CSR directed graph, fixed-size, fully constexpr |
| `graph_builder.h` | Representation | Builder for constexpr_graph with canonicalisation |
| `symmetric_graph.h` | Representation | Undirected graph wrapper with builder |
| `bipartite_graph.h` | Representation | Bipartite graph with L/R partition and builder |
| `runtime_graph.h` | Representation | Runtime CSR graph with builder + symmetric builder |
| `implicit_graph.h` | Representation | Procedurally-generated graph (no edge storage) |
| `graph_equal.h` | Representation | Structural equality comparison |
| `edge_property_map.h` | Annotation | Edge-indexed external property map with topology token |
| `weighted_view.h` | Annotation | Graph + weight map wrapper, concepts, symmetric validation |
| `weighted_graph_builder.h` | Annotation | Weight map factories (token-bound) |
| `property_map.h` | Annotation | External per-node property map |
| `kernel_info.h` | Annotation | Computational profile metadata |
| `capacity_guard.h` | Annotation | Compile-time/runtime capacity bounds check |
| `topological_sort.h` | Algorithm | Kahn's topological sort, O(V+E) |
| `connected_components.h` | Algorithm | Union–Find connected components, O(V+E) |
| `scc.h` | Algorithm | Iterative Tarjan SCC, O(V+E) |
| `graph_coloring.h` | Algorithm | Welsh–Powell greedy colouring, O(V log V + E) |
| `bipartite_matching.h` | Algorithm | Hopcroft-Karp O(E√V) maximum matching |
| `shortest_path.h` | Algorithm | Dijkstra with binary heap, O((V+E) log V) |
| `min_cut.h` | Algorithm | Stoer–Wagner min cut, O(V³) |
| `fusion_legal.h` | Transform | Fusion legality analysis from kernel_info |
| `fuse_group.h` | Transform | BFS-based fusion group identification |
| `coarsen.h` | Transform | Graph contraction by group |
| `from_pipeline.h` | Construction | Linear chain factory |
| `from_stencil.h` | Construction | Regular grid stencil factory |

### 16.3 Engine Bridge

| Header | Purpose |
|--------|---------|
| `graph_types.h` | Engine type aliases |
| `graph_to_space.h` | Build schedule_space from graph + kernel_map |
| `graph_to_constraints.h` | Extract dependency_set and resource_constraints |
| `coloring_to_groups.h` | Convert coloring_result to fuse_group_result |
| `build_fusion_graph.h` | Build weighted fusion dependency graph |
| `interpret_results.h` | Map algorithm results back to engine semantics |

---

## 17. Testing

### 17.1 Test Suite

| Test Suite | Type | Coverage |
|-----------|------|---------|
| `test_baseline_standalone.cpp` | Standalone | All algorithms at constexpr on constexpr_graph |
| `test_traits_checkpoint.cpp` | Standalone | rt_array, traits specialisations, concept checks |
| `test_step8_checkpoint.cpp` | Standalone | Retrofitted algorithms with traits-derived MaxV |
| `test_new_algos.cpp` | Standalone | Dijkstra, Stoer–Wagner (constexpr + runtime) |
| `test_runtime_graph.cpp` | Standalone | Runtime graph builder, all algorithms, constexpr/runtime parity |
| `test_engine_integration.cpp` | Standalone | Engine type aliases, bridge functions, array helpers |
| `test_red_team.cpp` | Standalone | 20 adversarial tests: empty, singleton, max capacity, dedup, errors |
| `test_full_include.cpp` | Header hygiene | All graph headers compile together |
| `graph_step{1-7}_test.cc` | Google Test | Core graph operations, stencil, pipeline, fusion, coarsening, colouring |
| `test_graph_coloring.cc` | Google Test | 18 colouring test cases including Petersen graph |
| `test_symmetric_graph.cc` | Google Test | Symmetric graph operations |
| `graph_hardening_test.cc` | Google Test | Capacity guards, property map bounds, builder validation |
| `test_graph_hardening_v2.cc` | Google Test | Edge rejection, SCC correctness, include hygiene |
| `test_weighted_view.cpp` | Google Test | 25+ tests: token binding, mismatch, symmetric validation |
| `test_bipartite_matching.cpp` | Google Test | 20+ tests: construction, matching, verification, constexpr pipeline |

### 17.2 Dijkstra Test Matrix

| Test | Validates |
|------|-----------|
| Linear chain | Basic relaxation, predecessor chain |
| Diamond graph | Shortest among alternatives |
| DAG | Directed graph correctness |
| Disconnected component | INFINITY for unreachable |
| Negative weight | `domain_error` (constexpr and runtime) |
| Single node | Trivial case |
| K₁₆ complete | Stress within ct_limits |
| Zero-weight cycle | Correct distances, no infinite loop |
| constexpr evaluation | Whole algorithm at compile time |
| Runtime 1000-node random | Heap correctness at scale |

### 17.3 Stoer–Wagner Test Matrix

| Test | Validates |
|------|-----------|
| Path P₄ | Min cut at end vertex |
| Cycle C₆ uniform | Min cut = 2 |
| Two clusters + bridge | Bridge is the cut |
| K₄ uniform | Min cut = 3 |
| Disconnected | Cut weight = 0 |
| Single / zero nodes | `has_cut = false` |
| Weighted diamond | Correct minimum |
| constexpr evaluation | Whole algorithm at compile time |
| Runtime 200-node random | Matrix contraction at scale |

---

## 18. Usage Patterns

### 18.1 Compile-Time Pipeline Analysis

```cpp
constexpr auto g = from_pipeline<5>();
constexpr auto kmap = make_uniform_kernel_map<5>(g, default_kernel_info);
constexpr auto topo = topological_sort(g);
static_assert(topo.is_dag);
constexpr auto space = build_schedule_space<5, 4>(g, kmap);
```

### 18.2 Runtime Graph from External Data

```cpp
runtime_graph_builder<64, 256> b;
for (auto& node : external_nodes) b.add_node();
for (auto& [u, v] : external_edges) b.add_edge(u, v);
auto g = b.finalise();
auto cc = connected_components(g);
```

### 18.3 Conflict Graph Colouring for Parallelism

```cpp
constexpr auto sg = build_row_conflict_graph(pattern);
constexpr auto cr = graph_coloring(sg);
constexpr auto fg = coloring_to_groups(cr);
// Same-colour nodes execute in parallel
```

### 18.4 Weighted Shortest Path

```cpp
constexpr auto r = dijkstra(g, source, [](node_id u, node_id v) {
    return lookup_weight(u, v);
});
static_assert(r.verified);
```

---

## 19. Design Decisions

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Storage decoupling | `graph_traits<G>` with make functions | Non-intrusive; centralised construction; no `is_same_v` dispatch |
| Capacity reporting | `node_capacity()` / `edge_capacity()` on graph | Traits provide types and construction; values from the graph |
| Array construction | Traits `make_*_array<T>(cap)` → free helper delegates | Single construction point per specialisation |
| Index type | `node_index_t<G>` from traits | Widening is one-line change, not codebase grep |
| `NODE_NIL` | `std::numeric_limits<node_index_t<G>>::max()` | Adapts automatically when index type widens |
| Traits vs inheritance | Traits-based | Constexpr-safe; zero overhead; open extension |
| MaxV as compile-time bound | Capacity limit, not size | Constexpr-constructible results; enables static_assert |
| Weighted infrastructure | weighted_view wrapper | Zero impact on unweighted code; multiple weight interpretations |
| Weight interface | Callable `WeightFn(node_id, node_id) → double` | Decouples algorithm from weight storage |
| Dijkstra heap | Constexpr binary heap | Single implementation for all scales |
| Stoer–Wagner contraction | Dense matrix | Cache-friendly; simple to verify |
| Cut-weight verification | Canonical CSR-order, u < v dedup | Bitwise IEEE 754 reproducibility |
| `symmetric_graph_queryable` | No `reverse_edge` requirement | Stoer–Wagner doesn't need it; avoids premature commitment |
| Min-cut trivial case | `has_cut` flag | Explicit; avoids INFINITY sentinel ambiguity |
| ct_limits | Advisory + opt-in guard | Friendly diagnostics without hard limits |
| Result construction | Factory-mandatory for runtime | Default-constructed runtime results safely empty |
| Engine bridge coupling | One-way (engine → graph) | Graph layer has zero engine dependencies |
| Engine graph mapping | Heuristic building blocks | Inputs to cost model, not claimed reductions |
| Bipartite concept | Refines `adjacency_queryable` | Consistent with traversal algorithms |
| Property map retrofit | Minimal first (separate rt type) | Unblocks algorithms without large refactor |
| `rt_array` API | Fixed-size, move-explicit | Enforces capacity invariant |
| Token semantics | Topology only; wrappers forward | Prevents token creep |

---

## 20. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.8 | Feb 2026 | Weighted graph infrastructure (weighted_view, edge_property_map, bipartite_graph, Hopcroft-Karp). Cap template unification (`cap_from<MaxV, MaxE>`). |
| 0.7.0 | Feb 2026 | Traits-based architecture (17-step plan). runtime_graph. Dijkstra, Stoer–Wagner. Engine bridge graph_types. graph_coloring API change (MaxV inferred). 3,500+ lines added. |
| 0.6.x | Jan 2026 | Graph colouring, symmetric graph, fusion groups, coarsening, engine bridge. SpMV domain. |
| 0.5.x | Dec 2025 | Core graph library: constexpr_graph, builder, topo sort, CC, SCC, property map, kernel_info, capacity_guard. |
