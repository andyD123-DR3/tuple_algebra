# Cache-Line Layout Policy Search

**Type:** Framework (uses `ctdp_space` — `descriptor_space`, `default_bridge`, `exhaustive_search_with_cost`)
**Dimensions:** 4 policy dims (hot_line_w × mix_penalty × waste_w × conflict_w)
**Space:** 5 × 6 × 3 × 3 = 270 policies
**Speedup:** ~1.42× on hot-path pricing field access

## What It Demonstrates

The architecturally significant example. The problem — assigning 13 FIX protocol fields to cache lines — has a power-set state space (2^N) that isn't a Cartesian product. A subset DP (or greedy bin-packing) exploits this structure. The framework can't replace the DP. But it can calibrate it.

The framework searches over 270 cost-model policy configurations. For each policy, the bin-packing solver finds the optimal field-to-line assignment. The framework discovers which policy produces the best layout by comparing solver outputs.

This is the general pattern for structured subproblems:
- **Framework** owns the outer search (Cartesian product of policy weights)
- **Problem-specific solver** owns the inner search (bin packing / subset DP)
- The framework's value is in calibrating the inner solver's parameters

## FIX Protocol Session

13 fields totalling 108 bytes. 6 fields are "hot" (accessed on every market data message): `msg_type`, `msg_seq_num`, `sending_time`, `bid_price`, `ask_price`, `bid_size`.

The goal: pack fields onto ≤4 cache lines (4 × 64 = 256 bytes) such that hot fields land on as few lines as possible, hot and cold fields don't mix, and fields from the same semantic group stay together.

## Policy Dimensions

| Dimension | Values | Effect |
|-----------|--------|--------|
| `hot_line_w` | 1000, 5000, 10000, 25000, 50000 | Reward for consolidating hot fields |
| `mix_penalty` | 1000, 5000, 10000, 50000, 100000, 500000 | Penalty for hot+cold on same line |
| `waste_w` | 1, 5, 10 | Penalty per wasted byte |
| `conflict_w` | 100, 500, 2000 | Penalty for cross-group fields on same line |

Different weight combinations produce different layouts. High `mix_penalty` forces strict hot/cold separation even at the cost of more waste. Low `waste_w` allows sparse lines if it keeps hot fields together.

## Bridge Encoding

The framework encodes each policy point as a 17-feature vector (one-hot per `int_set` dimension: 5 + 6 + 3 + 3). This encoding is produced by `default_bridge(space)` and would feed a learned cost model in the calibration phase (future work). For now, it demonstrates that the bridge machinery works with `descriptor_space`.

## Pipeline Steps

1. **Space** — `descriptor_space("policy", ...)` creates a 4-dimensional space of 270 policy configs
2. **Validity** — no filtering needed (all policies are valid weight combinations)
3. **Search** — `space.enumerate([&](auto const& pt) { ... })` — for each policy, run the bin-packing solver and track the minimum-cost layout
4. **Dispatch** — optimal layout plan becomes `static constexpr` field offsets
5. **Execute** — `read_bid_price()` uses compile-time offsets into the cache-line-aligned struct

## Build Note

Needs `-fconstexpr-ops-limit=500000000` (GCC) or `-fconstexpr-steps=500000000` (Clang) because 270 bin-packing evaluations at compile time exceed default constexpr operation limits.

## Checklist Compliance

1. **No dead knobs** — every policy weight changes the bin-packing cost function, producing different layouts for different weight combinations
2. **No lying metadata** — `descriptor_space` truthfully reports cardinality=270
3. **Single solve** — one `solve_policy_search()` function, constexpr
4. **Search matches executor** — the executor reads fields at the offsets computed from the winning layout plan
5. **Constraints structural** — no invalid policies (all weight combinations are legal)
6. **Bridge from same space** — `default_bridge(make_policy_space())` encodes from the same space the search enumerates
7. **Non-trivial correctness** — test prices written via computed offsets, read back and compared vs naive struct access

## Build

```bash
g++ -std=c++20 -O2 -fconstexpr-ops-limit=500000000 \
    -I../../include framework/cache_line_layout.cpp -o cache_line_layout
./cache_line_layout
```

Or via CMake:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cache_line_layout
./build/cache_line_layout
```
