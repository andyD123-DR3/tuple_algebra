# Phase A / A1 Concrete C++ API Shape — `interval_rooted_candidate`

**Status:** Draft  
**Date:** May 20, 2026  
**Scope:** Concrete header/API shape for Stage 2 Phase A / A1  
**Related:** `docs/design/phase_a_interval_rooted_candidate.md`, `docs/design/interval_stage2_execution_plan.md`, `docs/adr_interval_rooted_representation_family.md`, `include/ctdp/solver/spaces/interval_split_space.h`, `include/ctdp/solver/interval_context.h`

## 1. Purpose

This note refines the Phase A candidate specification into a concrete proposed C++ header/API shape.

It is intentionally still a design artifact, not an implementation commit.

The purpose is to answer the practical question:

> If Phase A / A1 were implemented next, what exact public header and type signatures should we aim for?

## 2. Recommended header placement

Recommended canonical header:

- `include/ctdp/solver/interval_rooted_candidate.h`

Recommended namespace:

- `ctdp::solver`

This keeps the interval-rooted family near the Stage 1 interval vocabulary already shipped under `ctdp::solver`.

## 3. Proposed public types

The concrete public types should be:

- `ctdp::solver::interval_rooted_candidate<MaxN>`
- `ctdp::solver::interval_rooted_node_ref<MaxN>`
- `ctdp::solver::interval_rooted_plan<MaxN>` as an alias only

Recommended plan alias:

```cpp
namespace ctdp::solver {

template<std::size_t MaxN>
using interval_rooted_plan = ctdp::plan<interval_rooted_candidate<MaxN>>;

} // namespace ctdp::solver
```

## 4. Proposed storage encoding

The candidate should use a dense table indexed by half-open intervals.

### 4.1 Table domain

For active leaf count `n`, legal interval coordinates are:

- `0 <= i < j <= n`

Recommended dense indexing domain for storage capacity:

- `0 <= i < j <= MaxN`

### 4.2 Encoded cell values

Recommended encoded representation:

- `0` = absent
- `1` = represented leaf
- `k + 2` = represented internal node split at absolute split point `k`

This gives a single compact `std::size_t` table with zero-initialization-friendly semantics.

### 4.3 Why this encoding

This encoding is recommended because it is:

- value-semantic
- compact
- easy to zero-initialize
- deterministic
- easy to compare under reachability-aware equality
- compatible with fixed-capacity framework style

## 5. Proposed header shape

The concrete API target should look approximately like this.

```cpp
#ifndef CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H
#define CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H

#include "../core/plan.h"
#include "interval_context.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ctdp::solver {

template<std::size_t MaxN>
struct interval_rooted_node_ref;

template<std::size_t MaxN>
struct interval_rooted_candidate {
    using interval_type = interval_context;
    using node_ref = interval_rooted_node_ref<MaxN>;

    static constexpr std::size_t max_size = MaxN;
    static constexpr std::size_t absent_code = 0;
    static constexpr std::size_t leaf_code = 1;

    std::size_t n{};
    std::array<std::size_t, (MaxN + 1) * (MaxN + 1)> split_or_tag{};

    [[nodiscard]] constexpr bool empty() const noexcept;
    [[nodiscard]] constexpr std::size_t size() const noexcept;
    [[nodiscard]] constexpr std::size_t leaf_count() const noexcept;

    [[nodiscard]] constexpr interval_type root_interval() const noexcept;

    [[nodiscard]] constexpr bool contains(std::size_t i, std::size_t j) const noexcept;
    [[nodiscard]] constexpr bool is_leaf(std::size_t i, std::size_t j) const noexcept;
    [[nodiscard]] constexpr bool is_internal(std::size_t i, std::size_t j) const noexcept;
    [[nodiscard]] constexpr std::size_t split(std::size_t i, std::size_t j) const noexcept;

    [[nodiscard]] constexpr interval_type left_interval(std::size_t i, std::size_t j) const noexcept;
    [[nodiscard]] constexpr interval_type right_interval(std::size_t i, std::size_t j) const noexcept;

    [[nodiscard]] constexpr std::optional<node_ref> find_node(std::size_t i, std::size_t j) const noexcept;
    [[nodiscard]] constexpr node_ref root() const noexcept;

    [[nodiscard]] constexpr bool is_legal() const noexcept;
    [[nodiscard]] constexpr bool is_canonical() const noexcept;

    [[nodiscard]] constexpr bool operator==(interval_rooted_candidate const& other) const noexcept;

private:
    [[nodiscard]] static constexpr std::size_t index(std::size_t i, std::size_t j) noexcept;
    [[nodiscard]] constexpr std::size_t code(std::size_t i, std::size_t j) const noexcept;
};

template<std::size_t MaxN>
struct interval_rooted_node_ref {
    using candidate_type = interval_rooted_candidate<MaxN>;
    using interval_type = interval_context;

    candidate_type const* candidate{};
    interval_type ctx{0, 1};

    [[nodiscard]] constexpr interval_type interval() const noexcept;
    [[nodiscard]] constexpr bool is_leaf() const noexcept;
    [[nodiscard]] constexpr bool is_internal() const noexcept;
    [[nodiscard]] constexpr std::size_t split() const noexcept;
    [[nodiscard]] constexpr interval_rooted_node_ref left() const noexcept;
    [[nodiscard]] constexpr interval_rooted_node_ref right() const noexcept;
};

template<std::size_t MaxN>
using interval_rooted_plan = ctdp::plan<interval_rooted_candidate<MaxN>>;

} // namespace ctdp::solver

#endif // CTDP_SOLVER_INTERVAL_ROOTED_CANDIDATE_H
```

## 6. Recommended method semantics

### 6.1 `empty`, `size`, `leaf_count`

Recommended semantics:

- `empty()` means `n == 0`
- `size()` returns active leaf count `n`
- `leaf_count()` returns `n`

This mirrors the existing style where `n` is the active problem size.

### 6.2 `root_interval()`

Recommended semantics:

- valid only conceptually for non-empty candidates
- returns `interval_context{0, n}`

Possible implementation policy:

- assert `n > 0`

### 6.3 `contains(i, j)`

Returns whether `[i, j)` is represented in the materialized decomposition.

This must mean:

- reachable and represented in the chosen tree
- not merely present as stale or arbitrary storage content

This is why `is_canonical()` and reconstruction discipline matter.

### 6.4 `is_leaf(i, j)`

Returns true iff the represented interval `[i, j)` is a leaf interval.

Recommended leaf law:

- only `[i, i+1)` may be leaf

### 6.5 `is_internal(i, j)` and `split(i, j)`

`is_internal(i, j)` is true iff `[i, j)` is represented and stores a split code.

`split(i, j)` returns the absolute split point `k`.

Recommended implementation law:

- decode from `split_or_tag[index(i, j)]` as `code - 2`

### 6.6 `left_interval` and `right_interval`

These are pure derived helpers from the stored split.

For internal `[i, j)` with split `k`:

- `left_interval(i, j)` returns `[i, k)`
- `right_interval(i, j)` returns `[k, j)`

## 7. Recommended node-ref/view shape

`interval_rooted_node_ref<MaxN>` should be a lightweight non-owning view.

Recommended fields:

- `candidate_type const* candidate`
- `interval_context ctx`

### 7.1 Why use `interval_context`

This keeps the Stage 2 candidate aligned with the existing Stage 1 vocabulary and avoids inventing a second interval record type.

### 7.2 Why not `std::optional<node_ref>` everywhere

Recommended API split:

- `find_node(i, j)` returns `std::optional<node_ref>`
- `root()` returns `node_ref` directly for non-empty canonical candidates
- child access from node refs (`left()`, `right()`) should assume the current node is valid and internal

This gives both safe lookup and a convenient tree-like consumer path.

## 8. Equality and canonicality guidance

### 8.1 Equality

Equality should be reachability-aware, not raw-table equality.

Two candidates are equal iff:

- `n` matches
- reachable represented structure matches
- corresponding internal intervals use the same split points

Unreachable storage garbage must not affect equality.

### 8.2 Canonicality

Recommended Phase A policy:

- expose `is_canonical()`
- require reconstruction helpers to produce canonical candidates
- allow implementation to zero unreachable cells during reconstruction

This keeps equality simpler in practice while preserving the stronger semantic rule.

## 9. Recommended companion helpers

Phase A / A1 likely also needs a small set of free helpers.

Recommended candidates:

```cpp
namespace ctdp::solver {

template<std::size_t MaxN>
[[nodiscard]] constexpr auto make_empty_interval_rooted_candidate()
    -> interval_rooted_candidate<MaxN>;

template<std::size_t MaxN>
[[nodiscard]] constexpr auto make_single_leaf_interval_rooted_candidate(std::size_t n)
    -> interval_rooted_candidate<MaxN>;

} // namespace ctdp::solver
```

These are optional, but they may help tests and reconstruction code.

## 10. Reconstruction-facing API boundary

Phase A / A1 does not need to solve recurrence-with-choice yet, but it should leave space for reconstruction.

Recommended future-facing hooks:

- candidate storage can be filled by reconstruction helpers
- representation uses half-open intervals and absolute splits throughout
- no method assumes the source was `interval_dp` specifically

That keeps the type usable both for:

- Stage 2 reconstruction from richer interval-solver paths
- adapters from legacy split-table outputs

## 11. Relationship to existing `interval_split_candidate`

The new API should not replace `interval_split_candidate` immediately.

The intended distinction is:

- `interval_split_candidate<MaxN>` = split decisions for subproblems in classic interval DP
- `interval_rooted_candidate<MaxN>` = materialized chosen decomposition tree

Both may coexist.

## 12. What Phase A / A1 should not add yet

This concrete API should avoid introducing, at this stage:

- feature encoding APIs
- learned-model hooks
- sparse memo interfaces
- universal traversal abstractions across tree families
- automatic `interval_solver` reconstruction support

Those belong in later Stage 2 phases.

## 13. Recommended next implementation step

If implementation starts after this note, the best next move is:

1. add `include/ctdp/solver/interval_rooted_candidate.h`
2. implement the candidate and node-ref types only
3. add focused legality / traversal / equality tests
4. leave reconstruction helpers for the next ticket unless the type definition naturally requires one minimal builder

## 14. Summary

The recommended concrete public API target for Phase A / A1 is:

- `ctdp::solver::interval_rooted_candidate<MaxN>`
- `ctdp::solver::interval_rooted_node_ref<MaxN>`
- `ctdp::solver::interval_rooted_plan<MaxN>` as an alias to `ctdp::plan<...>`

with:

- dense half-open interval table storage
- `0 / 1 / k+2` encoding for absent / leaf / split
- `interval_context` reused as the interval value type
- reachability-aware equality
- deterministic view-based tree access

That is concrete enough to implement without overcommitting to later Phase B/C/D concerns.

