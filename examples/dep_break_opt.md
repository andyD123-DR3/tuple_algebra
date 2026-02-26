# Dependency Breaking Optimisation

**Type:** Standalone (no framework imports)
**Dimensions:** 3 (dep_strategy × unroll_factor × num_accumulators)
**Space:** 105 total (7 × 5 × 3)
**Speedup:** ~3.3× (REDUCTION_TREE with 8 accumulators vs scalar loop)

## What It Demonstrates

Same space, three different instances, three different optimal strategies. The solver's output depends entirely on the instance, not the space geometry.

Also demonstrates conditional dimension relevance: `unroll_factor` is meaningless for REDUCTION_TREE (it doesn't unroll — it uses multi-accumulator reduction). `num_accumulators` is meaningless for LOOP_UNROLLING. The space contains all 105 combinations; the cost function ignores irrelevant dimensions.

## Three Instances

| Instance | Properties | Optimal Strategy |
|----------|-----------|-----------------|
| Accumulation | associative, commutative, dep_distance=1 | REDUCTION_TREE, accum=8 |
| Recurrence d=3 | non-associative, non-commutative, dep_distance=3 | SOFTWARE_PIPELINING, unroll=3 |
| Prefix sum | non-associative, non-commutative, dep_distance=1 | SCALAR_EXPANSION, accum=8 |

## Validity Constraints

- REDUCTION_TREE requires `is_associative` (can't reorder non-associative ops)
- SOFTWARE_PIPELINING requires `dep_distance > 1` (no pipeline if adjacent dependency)
- SOFTWARE_PIPELINING requires `unroll_factor ≤ dep_distance` (can't pipeline past the dependency)
- LOOP_INTERCHANGE requires `dep_distance > 0` (no dependency to break)

These are structural constraints — they depend on the instance, not the cost function. Invalid combinations are rejected before cost evaluation.

## Pipeline Steps

1. **Space** — 7 strategies × 5 unroll factors × 3 accumulator counts = 105
2. **Validity** — `is_valid(pt, inst)` filters per instance. Accumulation: 45 feasible. Recurrence: 15 feasible. Prefix: 30 feasible.
3. **Search** — constexpr exhaustive search with strategy-weighted cost
4. **Dispatch** — `dep_config` struct as NTTP → `dep_executor<cfg>`
5. **Execute** — multi-accumulator reduction, pipelined loop, or scalar expansion

## The "Don't-Care" Pattern

When the solver selects REDUCTION_TREE with accum=8 and unroll=1, the unroll_factor=1 is irrelevant — the executor ignores it. This is valid: the dimension exists in the space, the cost function doesn't penalise any particular unroll value for REDUCTION_TREE, and the first feasible value wins. The alternative — having different spaces per strategy — would complicate the architecture for no benefit.

## Checklist Compliance

1. **No dead knobs** — every strategy value has its own `if constexpr` branch; accum and unroll affect the generated loop body
2. **No lying metadata** — total=105 for all instances; feasible count computed by enumeration
3. **Single solve** — one `solve()` function parameterised on instance
4. **Search matches executor** — REDUCTION_TREE, SOFTWARE_PIPELINING, SCALAR_EXPANSION, NONE all have branches
5. **Constraints structural** — associativity and dep_distance checked in `is_valid`, not in cost
6. **Bridge from same space** — N/A (standalone)
7. **Non-trivial correctness** — non-uniform data `(7*i+3)%100 * 0.001`, error reported vs `std::accumulate` reference

## Build

```bash
g++ -std=c++20 -O2 standalone/dep_break_opt.cpp -o dep_break_opt
./dep_break_opt
```
