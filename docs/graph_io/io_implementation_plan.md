# CT-DP Graph I/O Layer — Implementation Plan

**Based on:** `ctdp_io_design.docx` (1 March 2026) vs repository `andyD123-DR3/tuple_algebra` at HEAD  
**Date:** 1 March 2026  
**Framework version:** v0.7.2 (per design doc); repo README states v0.4.0 (README out of date)

---

## 1. Current State Assessment

The design document specifies six implementation steps. The repository already contains substantial implementations for all six. The table below summarises per-step completion status.

| Design Step | Header(s) | Status | Remaining Work |
|---|---|---|---|
| Step 1: Capacity Policies | `capacity_types.h`, `graph_capacity.h` | **Complete** | None |
| Step 2: Deduction Aliases | `graph_traits.h` | **Complete** | None |
| Step 3: Constexpr Parser | `graph_io_parse.h`, `io_lexer.h` | **Complete** | Minor alignment issues |
| Step 4: Runtime I/O | `graph_io_stream.h` | **~85%** | Weighted readers, default cap, concept constraints |
| Step 5: DOT Writer | `graph_io_dot.h` | **Complete** | Minor concept constraint alignment |
| Step 6: Tests | `test_graph_io.cpp` + 3 others | **~80%** | CMake registration, coverage gaps |
| — Umbrella/Detail | `graph_io.h`, `graph_io_detail.h` | **Complete** | None |

**Overall: approximately 90% complete. Roughly 1–1.5 days of work remain.**

---

## 2. Detailed Gap Analysis

### 2.1 Step 1 — Capacity Policies: ✅ COMPLETE

Everything the design specifies is implemented and working:

- Named tiers: `cap::tiny` (8/24), `cap::small` (16/64), `cap::medium` (64/256), `cap::large` (256/1024), `cap::xlarge` (1024/4096). The design specifies only small/medium/large; the repo adds tiny and xlarge — a superset, fully compatible.
- `capacity_policy` concept with positive `max_v`/`max_e` checks.
- `cap_from<V, E>` bridge for raw numbers.
- Convenience aliases: `directed<Cap>`, `directed_builder<Cap>`, `undirected<Cap>`, `undirected_builder<Cap>`, `rt_graph<Cap>`, `rt_builder<Cap>`, `symmetric_rt_builder<Cap>`.

**No action required.**

### 2.2 Step 2 — Deduction Aliases: ✅ COMPLETE

All seven deduction aliases from the design are implemented in `graph_traits.h`: `topo_result_for<G>`, `components_result_for<G>`, `scc_result_for<G>`, `coloring_result_for<G>`, `shortest_path_result_for<G>`, `min_cut_result_for<G>`, `fuse_group_result_for<G>`. The `is_symmetric_graph_v<G>` trait is also present.

**No action required.**

### 2.3 Step 3 — Constexpr Parser: ✅ COMPLETE (minor alignment)

All three parser functions are implemented:

- `parse_directed<Cap>(string_view)` ✅
- `parse_symmetric<Cap>(string_view)` ✅
- `parse_weighted_directed<Cap>(string_view)` ✅ (returns `pair<graph, edge_property_map>`)

The design's `io_lexer.h` in `ctdp::io` is in place, with the forwarding header `graph_io_detail.h` bridging to `ctdp::graph::io::detail`.

**Minor alignment item:** The design specifies that `parse_directed` should accept and skip weight fields on edge lines (allowing the same file format for both weighted and unweighted consumption). Currently, `parse_directed` calls `skip_to_eol` after parsing `dst`, which effectively skips trailing weight tokens, so the intent is met. However, this behaviour is implicit rather than documented in the function's doc comment.

**Action:** Add a brief comment in `parse_directed` noting that trailing weight fields are silently skipped (design §5.1 / §12 "Weights ignored in topology-only parse"). No code change needed — just documentation alignment. Low priority.

### 2.4 Step 4 — Runtime I/O: ~85% — **Primary gap area**

**What exists:**

- `read_directed<Cap>(istream&)` ✅
- `read_symmetric<Cap>(istream&)` ✅
- `write(ostream&, G const&)` ✅ (unweighted)
- `write(ostream&, G const&, WeightFn)` ✅ (weighted)

**Gap 4a: Default capacity for runtime readers**

The design (§6.1) specifies that runtime readers should default to `cap::large` (256/1024) rather than `cap::medium` (64/256), reasoning that runtime readers handle unknown inputs and should avoid surprising capacity failures. The current code defaults to `cap::medium`.

This is a one-line fix per template but is a behavioural change — any existing code relying on `read_directed(stream)` without explicit cap will get a larger (but compatible) default.

**Gap 4b: Weighted stream readers**

The design's function summary (§4.2) lists only `read_directed` and `read_symmetric` — no explicit weighted reader variants. However, the round-trip design invariant (§6.2) implies that `write(stream, g, weight_fn)` output should be readable. Currently there is no `read_weighted_directed(istream&)` or `read_weighted_symmetric(istream&)` to close the round-trip for weighted graphs.

This requires two new functions in `graph_io_stream.h` that mirror the `parse_weighted_directed` logic but using `std::getline`/`std::stod`.

**Gap 4c: Tag-object capacity overloads**

The design (§6.1) mentions overloads that accept a policy tag object: `read_directed(stream, cap::small{})`. The current templates use `<Cap>` as a template parameter. The design's intent could be met with overloads that deduce `Cap` from a tag argument:

```cpp
template<capacity_policy Cap>
auto read_directed(std::istream& is, Cap /*tag*/) -> constexpr_graph<Cap> {
    return read_directed<Cap>(is);
}
```

This is a convenience addition — no new logic, just ergonomic wrappers.

**Gap 4d: Concept constraints on `write`**

The design (§6.3) specifies `template<graph_queryable G>` on the unweighted write and `template<adjacency_queryable G, typename WeightFn>` on the weighted write. The current code uses unconstrained `template<typename G>`. Adding concept constraints would give better diagnostics if someone passes a non-graph type.

The `graph_concepts.h` header already defines `graph_queryable` and `adjacency_queryable`.

### 2.5 Step 5 — DOT Writer: ✅ COMPLETE (minor alignment)

Both `write_dot` overloads (unweighted and weighted) are implemented. Directed/symmetric detection uses `is_symmetric_graph_v<G>` via `constexpr bool` rather than the design's `if constexpr (symmetric_graph_queryable<G>)` — the effect is identical since `is_symmetric_graph_v` is the underlying detection mechanism.

**Minor alignment:** Same concept-constraint gap as Step 4 — `write_dot` uses unconstrained `template<typename G>` where the design implies concept-constrained templates. Same fix approach.

### 2.6 Step 6 — Tests: ~80%

**What exists:**

| Test File | Tests | Static Asserts | In CMakeLists? |
|---|---|---|---|
| `test_graph_io.cpp` | 27 | 7 | ✅ Yes |
| `test_io_layering.cpp` | 2 | 6 | ✅ Yes |
| `test_capacity_policy.cpp` | 2 | ~30 | ❌ **Not registered** |
| `test_deduction_aliases.cpp` | 1 | ~25 | ❌ **Not registered** |

**Gap 6a: CMake registration** — `test_capacity_policy.cpp` and `test_deduction_aliases.cpp` exist but are not registered in `tests/CMakeLists.txt`. These tests will never run in CI.

**Gap 6b: Design-specified test coverage not yet present:**

- Weighted round-trip via streams (`write` with weights → `read_weighted_directed` → compare). Blocked on Gap 4b.
- Explicit capacity-overflow test for symmetric parser doubling (§5.4: verify 2×edges ≤ max_e).
- DOT output test for weighted symmetric graph (currently only directed weighted DOT is tested).
- Error-handling test for `read_symmetric` with capacity overflow.
- Tag-object overload tests (once Gap 4c is addressed).

---

## 3. Implementation Plan — Ordered Tasks

Tasks are ordered by dependency. Estimated total: **6–8 hours**.

### Task 1: Register Missing Tests in CMake  
**Effort:** 15 minutes  
**Files:** `tests/CMakeLists.txt`

Add `ctdp_add_test(test_capacity_policy graph/test_capacity_policy.cpp)` and `ctdp_add_test(test_deduction_aliases graph/test_deduction_aliases.cpp)`. Verify both compile and pass. Add them to the appropriate CTest label group.

**Acceptance:** `ctest --output-on-failure` passes with the two new test executables visible.

### Task 2: Change Runtime Reader Defaults to `cap::large`  
**Effort:** 15 minutes  
**Files:** `graph_io_stream.h`

Change the default template parameter from `cap::medium` to `cap::large` on `read_directed` and `read_symmetric`.

```cpp
template<capacity_policy Cap = cap::large>   // was cap::medium
[[nodiscard]] auto read_directed(std::istream& is) -> constexpr_graph<Cap>
```

**Acceptance:** Existing round-trip tests in `test_graph_io.cpp` still pass (they supply explicit caps). Add one new test that reads with default capacity and verifies it works for a graph exceeding medium limits.

### Task 3: Add Tag-Object Capacity Overloads  
**Effort:** 30 minutes  
**Files:** `graph_io_stream.h`

Add convenience overloads that deduce `Cap` from a tag argument:

```cpp
template<capacity_policy Cap>
[[nodiscard]] auto read_directed(std::istream& is, Cap) -> constexpr_graph<Cap> {
    return read_directed<Cap>(is);
}

template<capacity_policy Cap>
[[nodiscard]] auto read_symmetric(std::istream& is, Cap) -> symmetric_graph<Cap> {
    return read_symmetric<Cap>(is);
}
```

**Acceptance:** Test that `io::read_directed(stream, cap::small{})` compiles and produces the correct type.

### Task 4: Implement Weighted Stream Readers  
**Effort:** 1.5 hours  
**Files:** `graph_io_stream.h`

Implement `read_weighted_directed<Cap>(istream&)` and `read_weighted_symmetric<Cap>(istream&)` returning `pair<graph, edge_property_map>`, mirroring the logic in `parse_weighted_directed` but using `std::getline` and `std::stod` for runtime parsing. Include tag-object overloads.

The key challenge is matching edge weights to the graph's canonicalised (sorted) edge order, identical to the approach used in `parse_weighted_directed`. Factor the weight-matching logic into a shared helper if the code duplication is significant.

**Acceptance:**
- Weighted directed round-trip: `write(stream, g, wfn)` → `read_weighted_directed(stream)` → topology equal + weights equal.
- Weighted symmetric round-trip.
- Edges without weights default to 0.0.
- Error on malformed weight token.

### Task 5: Add Concept Constraints to Write/DOT Functions  
**Effort:** 30 minutes  
**Files:** `graph_io_stream.h`, `graph_io_dot.h`

Constrain `write` and `write_dot` templates:

```cpp
template<graph_queryable G>
void write(std::ostream& os, G const& g);

template<graph_queryable G, typename WeightFn>
    requires std::invocable<WeightFn, G const&, std::size_t>
void write(std::ostream& os, G const& g, WeightFn weight_fn);
```

Same pattern for `write_dot`. This is purely additive (SFINAE tightening) — all existing valid call sites remain valid.

**Acceptance:** Existing tests pass unchanged. Verify that passing a non-graph type to `write` produces a concept-related diagnostic rather than a cryptic template error.

### Task 6: Add Symmetric Capacity-Doubling Validation  
**Effort:** 30 minutes  
**Files:** `graph_io_parse.h`

The design (§5.4) requires that `parse_symmetric` verify `2 × declared_edges ≤ Cap::max_e` to catch the common error where users forget that symmetric parsing doubles edges. Currently the parser relies on the builder to detect overflow, which may produce a less clear error message.

Add an explicit check after accumulating edge count:

```cpp
if (2 * edge_count_so_far > Cap::max_e) {
    throw std::runtime_error(
        "parse_symmetric: doubled edge count exceeds capacity "
        "(each undirected edge produces two directed edges)");
}
```

This requires tracking edge count in the symmetric parser loop, which currently does not maintain a counter (edges go directly to the builder).

**Acceptance:** Test that parsing a symmetric graph where N edges × 2 > max_e produces the specific diagnostic message. Existing valid symmetric parse tests still pass.

### Task 7: Expand Test Coverage  
**Effort:** 2 hours  
**Files:** `test_graph_io.cpp` (expand existing), optionally `test_graph_io_weighted.cpp`

New tests to add, aligned with the design's §11.6 test matrix:

**Weighted round-trip tests (requires Task 4):**
- Write weighted directed → read weighted directed → compare topology + weights.
- Write weighted symmetric → read weighted symmetric → compare.
- Mixed: edges with and without weights in the same file.

**DOT tests:**
- Weighted symmetric DOT output: verify `graph` keyword, `--` edges, weight labels.
- Empty graph (0 edges) DOT output.

**Error handling:**
- Symmetric parse capacity overflow with doubling diagnostic (requires Task 6).
- Runtime reader: `read_directed` with malformed lines throws.
- Runtime reader: stream in fail state returns error.

**Tag-object overload tests (requires Task 3):**
- `read_directed(stream, cap::tiny{})` produces `constexpr_graph<cap::tiny>`.
- `read_symmetric(stream, cap::small{})` produces `symmetric_graph<cap::small>`.

**Capacity boundary tests:**
- Read a graph at exactly `cap::small::max_v` nodes — should succeed.
- Read a graph at `cap::small::max_v + 1` nodes — should fail.

**Acceptance:** All new tests pass. `ctest` runs clean. Approximate count: 12–15 new test cases.

### Task 8: Documentation Alignment  
**Effort:** 30 minutes  
**Files:** `graph_io.h` (umbrella header comments), `graph_io_stream.h`, `graph_io_parse.h`

Update header doc comments to match the design document's API surface:

- Document the `cap::large` default on runtime readers and the rationale.
- Document that `parse_directed`/`parse_symmetric` silently skip weight fields.
- Document the round-trip invariant as a design contract.
- Add a usage example showing `#embed` readiness (design §5.5) as a comment.
- Note that calibration sections (`[hardware]`, `[costs]`) are deferred to a future implementation.

**Acceptance:** Doc comments accurately reflect the API as implemented.

---

## 4. Dependency Graph

```
Task 1 (CMake registration)     ─── independent, do first
Task 2 (Default cap::large)     ─── independent
Task 3 (Tag overloads)          ─── independent
Task 5 (Concept constraints)    ─── independent
Task 6 (Symmetric doubling)     ─── independent
Task 4 (Weighted readers)       ─── after Task 2 (uses same defaults)
Task 7 (Tests)                  ─── after Tasks 2, 3, 4, 6
Task 8 (Documentation)          ─── after all code tasks
```

Tasks 1, 2, 3, 5, and 6 can be done in parallel. Task 4 depends on Task 2 (for default capacity consistency). Task 7 depends on the code it tests. Task 8 is documentation cleanup at the end.

---

## 5. Items Explicitly Deferred (per Design §14)

The following are called out in the design document as future extensions, **not** part of this implementation round:

- **Constexpr config parser** for `[hardware]`, `[costs]`, `[tile_costs]` calibration sections.
- **Sample selection algorithms** (Latin hypercube, adaptive sampling).
- **`ctdp-gen` CLI tool** for format conversion.
- **Binary serialisation** for large runtime graphs.
- **Bridge decoupling** of `schedule_space` capacity from graph `MaxV`.

These should not be pursued until the core I/O layer (Tasks 1–8) is confirmed stable.

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Default capacity change (Task 2) breaks downstream code | Runtime type change for default readers | Search for bare `read_directed(stream)` calls in examples/tests; update any that assume medium capacity |
| Concept constraints (Task 5) reject valid but unusual call sites | Compilation failure | Run full test suite after constraint addition; widen constraints if needed |
| Weighted reader edge-order matching is fragile | Incorrect weight mapping | Reuse the same matching algorithm as `parse_weighted_directed`; test exhaustively with canonical and non-canonical input orderings |
| Symmetric doubling check (Task 6) rejects graphs that currently work by accident | Breaking change for edge cases | Only trigger on actual overflow; the check is strictly tighter than the builder's internal check |

---

## 7. Verification Checklist

After all tasks are complete:

- [ ] `cmake --build . && ctest --output-on-failure` — all tests pass, zero warnings
- [ ] `test_capacity_policy` and `test_deduction_aliases` appear in test output
- [ ] Weighted round-trip test proves `write → read_weighted → compare` invariant
- [ ] DOT output for directed, symmetric, weighted directed, and weighted symmetric all verified
- [ ] Default capacity for runtime readers is `cap::large`
- [ ] Tag-object overloads `read_directed(stream, cap::small{})` work
- [ ] Concept constraints on `write`/`write_dot` produce clear diagnostics for non-graph types
- [ ] Symmetric parse produces clear diagnostic when doubled edges exceed capacity
- [ ] No regressions in existing 675+ tests
