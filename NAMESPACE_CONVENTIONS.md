# Namespace & Include Path Conventions

## Two namespaces, by design

The project uses two top-level include paths and two C++ namespaces:

| Path prefix | C++ namespace | Scope |
|-------------|---------------|-------|
| `ctdp/` | `ctdp::` | Core framework: core/, solver/, graph/, engine/, instantiate/, tooling/ |
| `ct_dp/` | `ct_dp::algebra` | Algebra library (P3666R2 reference implementation) |

### Why two?

The algebra library (`ct_dp/algebra/`) was developed as a standalone reference
implementation of WG21 proposal P3666R2 (*Compile-Time Parallel Reduction*).
It has its own namespace (`ct_dp::algebra`) to allow independent distribution
and to keep the P3666R2 interface unchanged if the proposal's namespace evolves.

The algebra depends on nothing in the rest of the framework. It can be used
entirely on its own (as the `variadic_reduce` example demonstrates).

### Include conventions

```cpp
// Framework headers: use ctdp/ prefix
#include "ctdp/core/constexpr_vector.h"
#include "ctdp/solver/algorithms/interval_dp.h"
#include "ctdp/graph/topological_sort.h"

// Algebra headers: use ct_dp/ prefix
#include "ct_dp/algebra/algebra.h"
#include "ct_dp/algebra/make_reduction.h"

// Within a tier, headers use relative paths to peers:
// (inside solver/algorithms/exhaustive_search.h)
#include "../concepts.h"           // solver/concepts.h
#include "../../core/plan.h"       // core/plan.h
#include "../../core/ct_limits.h"  // core/ct_limits.h
```

### CMake targets

| Target | Provides | Depends on |
|--------|----------|------------|
| `ctdp_core` | core/ headers | — |
| `ctdp_solver` | solver/ headers | ctdp_core |
| `ctdp_algebra` | ct_dp/algebra/ headers | — |
| `ctdp_graph` | graph/ headers | ctdp_core |
| `ctdp_domain` | domain/ headers | ctdp_graph, ctdp_solver |
| `ctdp_all` | everything | all of the above |

### Domain-specific code

Application-specific construction utilities (e.g. SpMV graph builder) live
under `ctdp/domain/<app>/` rather than in the framework tiers, to keep the
analytics libraries general-purpose.

`kernel_info.h` remains in `graph/` — it is framework infrastructure used by
fusion_legal, coarsen, fuse_group, and the engine bridge. It is not
domain-specific despite originating from compute-graph applications.
