# CT-DP: Compile-Time Dynamic Programming Framework

**Version 0.8.0** — C++20 header-only framework for compile-time optimization.

## Quick Start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure
```

## Libraries

| Tier | Headers | Lines | Description |
|------|--------:|------:|-------------|
| **core** | 15 | 3,104 | Constexpr containers, sorting, concepts, limits, plan, solve_stats |
| **solver** | 32 | 4,634 | Algorithms (exhaustive, beam, local, interval DP), spaces, ML cost models, memo |
| **algebra** | 7 | 937 | Tuple algebra for variadic reductions (P3666R2) |
| **graph** | 41 | 7,923 | CSR storage, topo sort, SCC, coloring, min-cut, matching, RCM, I/O, transforms |
| **engine** | 8 | 1,483 | Bridges (graph→space, coloring→groups), instantiation, segmentation DP |
| **space** | 2 | 1,320 | Descriptor-based search spaces, feature bridge encoding |
| **bench** | 9 | 1,826 | Measurement kernel, perf counters, distribution fitting, environment |
| **calibrator** | 35 | 7,598 | Calibration harness, FIX parser domain, plans, wisdom, provenance |
| **domain** | 1 | 616 | SpMV graph pipeline |

**150 headers, 63 test files, 28 examples, ~63K lines total.**

### Dependency flow (CMake targets)

```
graph → core          engine → graph + solver       domain → engine + graph + solver
solver → core         space → solver                calibrator → bench
algebra (standalone)  bench (standalone)
```

All tiers are `INTERFACE` libraries; the framework is header-only.

## Algebra Library (`include/ct_dp/algebra/`)

P3666R2 tuple algebra for variadic single-pass reductions (7 headers).

## Graph Library (`include/ctdp/graph/`)

41 headers covering CSR representation, construction, 8 algorithm families
(topological sort, SCC, connected components, coloring, shortest path,
min-cut, bipartite matching, RCM bandwidth reduction), I/O (text, DOT),
transforms (contract, subgraph, transpose), weighted views, and merge rules.
See `docs/graph_library_reference.docx`.

## Space Library (`include/ctdp/space/`)

Descriptor-based compile-time search spaces. Each dimension is a typed
descriptor (positive_int, power_2, int_set, bool_flag, enum_vals).
`descriptor_space` provides enumeration, cardinality, per-dimension access,
and feature encoding via `feature_bridge` / `default_bridge(space)`.

## Calibrator Library (`include/ctdp/calibrator/`)

Runtime calibration pipeline: Scenario → calibration_harness → profile →
cost_model → plan → wisdom. Includes a complete FIX protocol parser
domain (15 headers) with expression-template instantiation, hardware
counter measurement, and SVR-based cost prediction.

## Requirements

- C++20 compiler (GCC 13+, Clang 16+, MSVC 19.36+)
- CMake 3.20+
- Internet connection (GTest fetched automatically)

## Documentation

- `docs/architecture.docx` — Framework design and data flow
- `docs/ctdp_architecture.md` — Architecture overview
- `docs/graph_library_reference.docx` — Graph API reference
- `docs/ctdp_graph_user_guide.md` — Graph library user guide
- `docs/design/epsilon_svr_design_note.md` — Epsilon-SVR implementation
- `docs/calibrator/` — Calibrator design and FIX parser explainer
