# FIX Parser p99 Calibration -- Explainer

**Example:** `examples/fix_p99_calibration.cpp`
**Headers:** `include/ctdp/bench/percentile.h`, `include/ctdp/calibrator/fix_et_parser.h`
**Tests:** `tests/bench/test_percentile.cpp` (25 cases)
**Date:** 3 March 2026
**Framework version:** v0.7.0+

---

## 1. What This Example Demonstrates

The FIX parser p99 calibration is a self-contained demonstration of
compile-time strategy selection optimised for *tail latency* rather than
mean latency.  It answers the question: given a 12-field FIX MarketData
message where each integer field can be parsed by one of four strategies
(Unrolled, SWAR, Loop, Generic), which of the 4^12 = 16,777,216 possible
configurations minimises p99 latency?

The example instantiates 241 expression-template parser configurations as
distinct template specialisations, measures each with 100,000 samples for
stable p99 estimation, and reports the full latency distribution.  It runs
in under 4 seconds on commodity hardware.

This is the same problem solved in Phase 6 of the CT-DP project for *mean*
latency, where the optimal plan SSSLSSSSSUSU achieved 27.8 ns -- a 2.11x
speedup over Lima & Tonetti's expert hand-tuned 59 ns.  The p99
investigation (Phase 10) revealed that the mean-optimal plan is *not*
p99-optimal, and that the p99 objective exposes fundamentally different
hardware effects.


## 2. Why p99 Is Hard

Three properties make p99 calibration qualitatively different from mean
calibration.

**Sample count.**  Mean stabilises with O(100) samples.  p99 requires
O(100,000).  At 10K samples, measured p99 values exhibit 2-3x variance
between runs of the same configuration.  At 100K, variance drops to +/-1-2 ns.
The example defaults to 100K and the `--quick` flag drops to 50K for faster
iteration.

**Additive cost models fail.**  The natural approach -- measure each
strategy in isolation, sum per-field costs to predict whole-message p99 --
produces predictions that are 73-87% wrong.  The error comes from treating
the expression-template parser as a sum of independent costs, when in fact
the compiler fuses the 12-field parse chain into a single function with
shared register allocation, instruction-level parallelism, and cache-line
amortisation.  These cross-field effects dominate at the tail.

**ML models fail.**  Neural networks trained on whole-message ET
measurements also fail.  The root cause is a hard collinearity constraint:
the strategy counts n_U + n_S + n_L + n_G == 12 for every configuration.
This produces a condition number of 6.9 x 10^16 in the feature matrix,
which destroys gradient signals.  Even with 2,000 training samples, the
best neural network achieved only R^2 = 0.54 out-of-sample -- insufficient
for reliable ranking of configurations separated by 1-2 ns.


## 3. What Works: Direct Measurement + Structured Search

The example uses three search strategies that together cover the space
efficiently without any cost model:

| Strategy | Configs | Purpose |
|---|---|---|
| **Baselines** (5) | All-U, All-S, All-L, All-G, Phase10g | Reference points and sanity check |
| **Simplex** (36) | All single-field mutations of Phase10g | Local search around known good point |
| **Uniform random** (200) | Compile-time SplitMix64, seed 42 | Unbiased global coverage |

Total: 241 configurations.  At 66 configs/second with 100K samples each,
the full sweep completes in under 4 seconds.

The simplex neighborhood consistently populates the top-10 by p99, which
confirms that the Phase10g region is locally optimal.  Random sampling
validates that no distant configuration dominates the simplex.  When
multiple configurations achieve similar p99 (within measurement noise),
this indicates a *performance plateau* -- the hardware lower bound for this
message format.


## 4. Architecture

### 4.1 Expression-Template Parser

`fix_et_parser.h` implements the template-specialised parser chain.  The
core is a recursive `parse_fields<Config, I>()` that processes field I with
the strategy specified by `Config[I]`, then calls itself for field I+1.
The compiler sees the entire chain as a single function body.

```
template<fix_config Config, int I = 0>
inline std::uint64_t parse_fields(const char* msg,
                                   const int* offsets,
                                   std::uint64_t acc) noexcept
{
    if constexpr (I < num_fields) {
        constexpr Strategy s = Config[I];
        constexpr int digits = field_digits[I];
        acc ^= parse_field<s, digits>(msg + offsets[I]);
        return parse_fields<Config, I + 1>(msg, offsets, acc);
    } else {
        return acc;
    }
}
```

Each `parse_field<S, Digits>()` dispatches to one of four implementations
(`parse_unrolled`, `parse_swar`, `parse_loop`, `parse_generic`), all
marked `[[gnu::always_inline]]` to ensure the compiler can optimise across
field boundaries.

The `fix_config` type is `std::array<Strategy, 12>` -- a value type that
can be a non-type template parameter in C++20.

### 4.2 Compile-Time Configuration Generation

Random configurations are generated at compile time using a constexpr
SplitMix64 PRNG:

```
constexpr auto random_configs = generate_random_configs<200>(42);
```

This is critical for measurement validity.  Each configuration becomes a
distinct template instantiation with its own generated machine code.
Runtime-parameterised approaches (e.g., a function pointer table) would
measure dispatch overhead rather than the actual optimised parse chain.

### 4.3 Dispatch Table

The calibration program needs to iterate over configurations at runtime
(to print progress, collect results) while calling template-specialised
code for each.  This is bridged by a dispatch table of function pointers:

```
struct config_entry {
    fix_config config;
    const char* group;
    percentile_result (*measure)(vector<string> const&, size_t);
};
```

The table is populated via variadic `make_random_entries(index_sequence<Is...>)`,
which expands each compile-time config into its own `measure_wrapper<Config>`
instantiation.

### 4.4 Percentile Statistics

`percentile.h` extends `bench/statistics.h` with tail-latency computation.
`compute_percentiles()` sorts once and extracts p50/p90/p95/p99/p99.9/max
in a single pass.  The `percentile_result` type carries `tail_ratio()`
(p99/p50) and `jitter_ns()` (p99 - p50) as derived metrics.


## 5. The FIX Message Model

The example models a 12-field FIX MarketData message with the following
digit counts per field:

| Index | Semantic | Digits |
|---|---|---|
| 0 | BodyLength | 3 |
| 1 | MsgSeqNum | 6 |
| 2 | AccountRef | 4 |
| 3 | Price | 8 |
| 4 | OrderQty | 6 |
| 5 | SymbolSfx | 4 |
| 6 | Side | 2 |
| 7 | TransactTime | 8 |
| 8 | TradeID | 6 |
| 9 | CheckSum | 3 |
| 10 | PartyRef | 4 |
| 11 | SecuritySfx | 4 |

Total: 58 digits per message.  Fields are separated by SOH (0x01)
delimiters.  The message pool is pre-generated with random digit values
to avoid allocation during measurement.


## 6. Four Parsing Strategies

**Unrolled (U).**  Compile-time digit count produces a straight-line
sequence of multiply-accumulate operations with no branches.  Largest code
footprint, highest ILP.

**SWAR (S).**  Processes 4 digits at a time using SIMD-Within-A-Register
techniques: load 4 bytes, subtract `0x30303030`, extract and combine.
Tail digits (1-3) handled by an `if constexpr` chain.

**Loop (L).**  A simple `for (int i = 0; i < N; ++i)` counted loop.
Smallest source code; the compiler decides whether to unroll.  In practice,
GCC at `-O2` partially unrolls for small N.

**Generic (G).**  Loop with per-character bounds checking (`if (c < '0' || c > '9') break`).  Most defensive, smallest code, highest branch cost.

All four strategies are correct for well-formed input (the test suite
verifies agreement across all strategies and digit counts).  They differ
only in the generated machine code shape, which affects instruction-cache
pressure, branch-predictor state, and ILP.


## 7. Key Findings from Phase 10

### 7.1 Template Specialisation vs Runtime Dispatch

Runtime dispatch (function pointer or switch per field) overestimates
latency by 1.2-1.9x compared to the expression-template approach.  The
gap arises because the compiler cannot optimise across a function-pointer
call: register allocation restarts at each field boundary, and instruction
scheduling cannot interleave work from adjacent fields.

Implication: *calibration must measure actual compiled instantiations*.
Measuring isolated strategies and summing costs gives the wrong answer.

### 7.2 Biased Sampling Destroys Model Training

Early experiments concentrated samples around known-good configurations
(mean strategy count S ~ 8.0 instead of the expected 3.0 for uniform
sampling).  This exacerbated the collinearity in the feature matrix and
produced neural networks that predicted the mean for every input (R^2 ~ 0).

The fix is simple: uniform random sampling over the full 4^12 space.  The
example uses a constexpr PRNG with a fixed seed for reproducibility.

### 7.3 Performance Plateaus

Multiple configurations achieve p99 within +/-1-2 ns of each other.  This
is the hardware floor for this message format on this microarchitecture.
The practical consequence is that the optimiser does not need to find *the*
optimal configuration -- it needs to find *any* configuration in the plateau
region.

This is good news for the framework: approximate search (beam search,
random sampling, local search) is sufficient.  Exhaustive search over all
16M configurations is unnecessary.

### 7.4 p99 Optimisation Finds Different Plans

The mean-optimal plan (SSSLSSSSSUSU) and the p99-optimal plan differ.
SWAR dominance at the mean reflects throughput advantages; at the tail,
Unrolled strategies win because their branchless execution avoids
worst-case branch misprediction.  The optimal p99 configurations tend
toward 8-9 Unrolled fields with SWAR for the widest fields (Price,
TransactTime at 8 digits).


## 8. Build and Run

```bash
# Full run: 241 configs x 100K samples ~ 4 seconds
cmake --build build --target fix_p99_calibration
./build/examples/fix_p99_calibration

# Quick mode: 91 configs x 50K samples ~ 0.7 seconds
./build/examples/fix_p99_calibration --quick

# CSV output for external analysis
./build/examples/fix_p99_calibration --csv > fix_p99_data.csv

# Custom sample count
./build/examples/fix_p99_calibration --samples 200000
```

CSV columns: `config,group,mean,p50,p90,p95,p99,p999,max,samples,tail_ratio`


## 9. Tests

`tests/bench/test_percentile.cpp` contains 25 GTest cases covering:

- Percentile computation: empty, single-element, median, min/max, p99 on
  large data, unsorted input, full distribution, tail ratio, jitter.
- FIX ET parser: config round-trip, all four parse strategies for
  correctness, cross-strategy agreement at multiple digit counts, message
  pool generation, ET parser chain, compile-time random config generation,
  PRNG determinism, measurement infrastructure.

Run:

```bash
cmake --build build --target test_percentile
./build/tests/test_percentile
```


## 10. Relationship to the CT-DP Framework

This example is a stepping stone toward the full calibrator integration
described in the Phase 2 design (`doc/calibrator/ctdp_calibrator_project_design_v2.md`).

**What it uses:**  `bench::result_token`, `bench::DoNotOptimize`,
`bench::ClobberMemory` for anti-elision.  The `percentile.h` header
extends the bench layer's statistics capability.

**What it demonstrates but does not yet integrate:**

- *Plan-based measurement.*  The example measures configs directly rather
  than going through the `Scenario` -> `calibration_harness` ->
  `calibration_dataset` -> `calibration_profile` -> `plan_builder` pipeline.
  The full integration would define a `fix_et_scenario` that satisfies the
  `Scenario` concept, with `point_type = fix_config` and
  `callable_type = fix_et_parser_tag`.

- *Beam search DP.*  The solver layer's `beam_search.h` could drive the
  configuration search with `measure_config<>` as the cost oracle, rather
  than the uniform random + simplex approach used here.

- *Percentile as cost metric.*  The existing `data_point` stores
  `median_ns` and `mad_ns`.  Extending it to store the full
  `percentile_result` (or at least p99) would allow the profile -> plan
  pipeline to optimise for tail latency natively.

These integration steps are planned for the calibrator Phase 2 workstream.


## 11. Lessons for the Conference Talk

This example encodes five lessons that are relevant to the C++ Online talk
(13 March 2026, "Pareto-Optimal Tiling and Kernel Fusion for Variadic
Reductions"):

1. **Problem formulation > optimisation cleverness.**  Lima & Tonetti
   optimised for an implicit objective.  Making the objective explicit
   (p99, not mean) and measuring it directly (100K samples, not 10K)
   changes the answer.

2. **Measurement discipline is non-negotiable.**  p99 requires 10-100x
   more samples than mean.  Without sufficient samples, the optimiser
   finds a configuration that is optimal *for the noise*, not for the
   hardware.

3. **When to use ML vs direct measurement.**  ML fails when: features are
   collinear, samples are insufficient, and the true function is complex.
   Direct measurement wins when measurement throughput is high enough
   (66+ configs/second here).

4. **Performance plateaus are the common case.**  When multiple
   configurations are hardware-equivalent, the optimiser's job is to find
   the plateau, not the exact minimum.  This justifies approximate search
   methods.

5. **Template specialisation is not optional.**  Runtime dispatch measures
   a different program than the one that will execute in production.
   Calibration must measure what ships.
