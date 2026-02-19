# CT-DP: Compile-Time Dynamic Programming Framework

**Version 0.4.0** — C++20 header-only framework for compile-time optimization.

## Quick Start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure
```

## Libraries

| Library | Headers | Lines | Description |
|---------|--------:|------:|-------------|
| **core** | 5 | 1,365 | Constexpr containers, sorting, concepts, limits |
| **solver** | 6 | 1,244 | DP engine: plans, composition, Pareto sets, traversal |
| **algebra** | 6 | 937 | Tuple algebra for variadic reductions (P3666R2) |
| **graph** | 18 | 3,237 | Graph algorithms, fusion, scheduling, SpMV pipeline |

**Dependency flow:** `graph → solver → core`, `algebra` (standalone, `ct_dp::algebra` namespace)

## Core Library (`include/ctdp/core/`)

| Header | Purpose |
|--------|---------|
| `constexpr_vector.h` | Fixed-capacity dynamic array (std::array + size) |
| `constexpr_sort.h` | Heap sort, merge sort, unique — all constexpr |
| `constexpr_map.h` | Sorted associative map with binary search |
| `concepts.h` | `cost_value`, `cost_function`, `search_space`, `candidate` |
| `ct_limits.h` | Compile-time resource budgets with policy override |

## Solver Library (`include/ctdp/solver/`)

| Header | Purpose |
|--------|---------|
| `plan.h` | Central output type: candidate + cost + stats |
| `plan_compose.h` | Additive, max, custom plan composition |
| `plan_set.h` | Bounded Pareto frontier of non-dominated plans |
| `plan_traversal.h` | Generic iteration over plan assignments |
| `solve_stats.h` | Solver metrics (subproblems, cache hits, pruning) |
| `candidate_traits.h` | Extension point for candidate types |

## Algebra Library (`include/ct_dp/algebra/`)

P3666R2 tuple algebra for variadic single-pass reductions.

| Header | Purpose |
|--------|---------|
| `algebra.h` | Convenience header — includes all algebra components |
| `operations.h` | Typed ops: `identity_t`, `power_t<N>`, `plus_fn`, `min_fn`, `max_fn` |
| `elementwise.h` | `elementwise_binary_op`, `elementwise_unary_op` — per-lane combine |
| `fan_out.h` | `fan_out_unary_op` — scalar to tuple (moments pattern) |
| `tuple_select.h` | Lane projection: extract subset of tuple lanes |
| `tuple_fold.h` | Horizontal fold: reduce tuple to scalar |
| `make_reduction.h` | `reduction_lane`, `make_reduction` — assembles full pipeline |

## Graph Library (`include/ctdp/graph/`)

18 headers across 7 layers: Representation → Construction → Algorithms →
Annotation → Fusion → Engine Bridge → SpMV Demo.
See `docs/graph_library_reference.docx`.

## Tests

| Suite | Runtime Tests | Static Asserts |
|-------|-------------:|---------------:|
| Core | 58 | 155 |
| Solver | 50 | 99 |
| Algebra | 54 | 166 |
| Graph | 172 | 494 |
| **Total** | **334** | **914** |

## Requirements

- C++20 compiler (GCC 13+, Clang 16+, MSVC 19.36+)
- CMake 3.20+
- Internet connection (GTest fetched automatically)

## Documentation

- `docs/architecture.docx` — Framework design and data flow
- `docs/graph_library_reference.docx` — Graph API reference (18 headers)
