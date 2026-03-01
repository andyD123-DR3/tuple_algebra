# Graph Library Examples

Worked examples demonstrating every major graph algorithm in the CT-DP
graph library.  Each example solves a practical problem at **compile time**
with `static_assert` proofs, then prints the results at runtime.

## Examples

| Example | Algorithm | Problem | Key result |
|---------|-----------|---------|------------|
| `example_topo_sort` | Topological sort | Build system ordering | Legal build order with dependency proofs |
| `example_components_scc` | Connected components + SCC | Module dependency analysis | Find circular dependencies |
| `example_coloring` | Graph colouring | Exam scheduling | Minimum time slots with no conflicts |
| `example_shortest_path` | Dijkstra | Data-centre routing | Minimum-latency paths with route reconstruction |
| `example_min_cut` | Stoer–Wagner | Pipeline bottleneck | Identify bandwidth-limiting link |
| `example_bipartite` | Hopcroft–Karp | Engineer–project assignment | Maximum matching with perfect assignment proof |
| `example_rcm` | Reverse Cuthill–McKee | Sparse matrix bandwidth | Reorder nodes to reduce adjacency bandwidth |

## Building

### With CMake (recommended)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCTDP_BUILD_EXAMPLES=ON
cmake --build build
./build/example_topo_sort
```

### Standalone (single file, no dependencies beyond the library headers)

```bash
g++ -std=c++20 -O2 -I include -o example_topo_sort examples/graph/example_topo_sort.cpp
./example_topo_sort
```

## What each example demonstrates

**Topological sort** — 6 build targets with dependencies.  `static_assert`
proves that every dependency is built before its dependent.  The topological
order is computed entirely during compilation.

**Connected components + SCC** — 7 software modules with two circular
dependency cycles.  Weakly connected components show everything is reachable;
Tarjan's SCC finds the exact cycles (UI cycle of 3, backend cycle of 2).

**Graph colouring** — 6 exams with student-overlap conflicts.  Welsh–Powell
greedy colouring assigns slots so no student has two exams simultaneously.
`static_assert` proves the colouring is valid and uses at most 3 slots.

**Dijkstra** — 7-node data-centre network with latency weights.  Shortest
paths from the web server to every node, including path reconstruction.
`static_assert` proves each optimal latency value.  Demonstrates that the
cache route (10μs) beats the direct database route (15μs).

**Min cut (Stoer–Wagner)** — A two-cluster pipeline with a narrow
inter-cluster link.  The minimum cut of 20 MB/s identifies the bottleneck.
`static_assert` proves the cut weight and that the two clusters are on
opposite sides of the partition.

**Bipartite matching (Hopcroft–Karp)** — 4 engineers × 4 projects with
skill constraints.  Maximum matching finds a perfect assignment where
every engineer gets a project.  `static_assert` proves the matching is
perfect and verified.

**RCM** — Scrambled 4×4 grid with bandwidth 14 under identity ordering.
RCM reduces bandwidth to 4.  ASCII matrix visualisation shows the band
tightening.  A runtime SpMV conflict graph example shows bandwidth 7 → 2.
