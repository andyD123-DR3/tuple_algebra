# Cache Size Optimisation

**Type:** Standalone (no framework imports)
**Dimensions:** 1 (cache_size, values 1–8)
**Space:** 8 points
**Speedup:** varies by instance

## What It Demonstrates

Cost-as-simulation: the compile-time cost function is a full LRU simulation that shares its engine with the runtime cache. Single source of truth — the simulation uses exactly the same `lru_cache<N>` template as the production code.

More importantly, it demonstrates why analytical cost models get the wrong answer. The simulation evaluates single-pass behaviour. Real workloads have warm-start effects across invocations. The `phase_shift` instance exposes this gap: the simulation says size 3 suffices, but runtime measurement shows size 6 is dramatically faster because it stays warm across phase boundaries.

This is the motivation for learned cost models — replace simulation with measurement.

## Three Instances

| Instance | Access Pattern | CT Optimal | Why |
|----------|---------------|-----------|-----|
| Alternating | Functions 2,3 alternate; 0,1,4 rare | size=2 | Only 2 hot functions |
| Sequential | 0,1,2,3,4,5 cycling | size=6 | Needs all 6 to avoid thrashing |
| Phase shift | 0,1,2 then 3,4,5 | size=3 | 3 covers each phase independently |

## The Cost Model ≠ Reality Lesson

The phase_shift instance is the key teaching point:

- **CT simulation** counts misses in a single pass of 20 accesses. With size=3, each phase has perfect hits after warming up. Total misses ≈ 6 (3 cold starts per phase).
- **Runtime** iterates the pattern 50 times. With size=3, every phase transition evicts the previous phase's entries, causing 3 cold misses per transition × 50 iterations. With size=6, both phases' functions stay cached — zero transition misses after the first pass.

The simulation is honest (it correctly simulates one pass), but incomplete (it doesn't model multi-pass warm-start). This is not a bug — it's a fundamental limitation of analytical models that motivates calibration.

## Pipeline Steps

1. **Space** — cache sizes 1 through 8 (template parameter, not runtime value)
2. **Validity** — all sizes valid (no constraints)
3. **Search** — compile-time LRU simulation counts misses per access pattern
4. **Dispatch** — `lru_cache<N>` where N = optimal size (NTTP on template)
5. **Execute** — cached function dispatch using the same LRU engine

## Shared Engine Pattern

The `lru_cache<Capacity>` template is used twice:

1. **At compile time** in `simulate_misses<CacheSize>()` — runs the LRU simulation constexpr
2. **At runtime** in `cached_executor<CacheSize>` — provides the actual function cache

Same type, same logic, different execution context. This is the CT-DP guarantee: what you simulate is what you execute.

## Checklist Compliance

1. **No dead knobs** — cache_size directly determines the LRU capacity in both cost model and executor
2. **No lying metadata** — space has exactly 8 points, all valid
3. **Single solve** — one `solve_cache_opt<MaxSize>()` function, constexpr
4. **Search matches executor** — every size 1–8 has a corresponding `cached_executor<N>` instantiation
5. **Constraints structural** — no constraints (all sizes valid)
6. **Bridge from same space** — N/A (standalone)
7. **Non-trivial correctness** — three distinct access patterns with different working set sizes, runtime verification

## Build

```bash
g++ -std=c++20 -O2 standalone/cache_size_opt.cpp -o cache_size_opt
./cache_size_opt
```
