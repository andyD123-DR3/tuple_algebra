# CT-DP Calibrator Project — Two-Library Design (v2)

**Author:** Andrew Drakeford
**Date:** 1 March 2026
**Status:** Draft v2 — revised after LEWG-style and architectural review
**Supersedes:** ctdp_calibrator_project_design.md (v1)


## Revision History

v2.2 incorporates callable_type as fundamental architectural parameter.

16. **callable_type on Scenario.**  The plan's callable — the expression
    template, function object, or compiled kernel being measured — is an
    associated type on Scenario.  The calibrator must know this type to
    construct the executor and to ensure cost models are keyed correctly.

17. **Dataset and profile keyed by (Space, Callable).**  A profile fitted
    from FIX SWAR parser data must not be applied to a lookup parser plan
    even if they share the same space dimensions.  Type identity enforced
    at compile time.

18. **Data flow updated.**  §8 traces callable identity from Scenario
    through dataset to profile to plan.

v2.1 incorporates final sign-off review.  Key changes:

9. **Naming: reps / warmup_iters / measure_iters.**  Three distinct
   quantities with unambiguous names.  warmup_reps eliminated.

10. **mix_token helper.**  Cheap bit-mix prevents trivial-constant footgun.

11. **Metric::snapshot() contract.**  start() resets+arms, stop() latches,
    snapshot() returns latch (O(1), valid only after stop).

12. **wall_ns is always steady_clock.**  TSC cycles are an additional
    observable, not the timing label.

13. **cache_thrasher::thrash() ends with ClobberMemory.**  Non-elidable.

14. **Size invariant.**  raw_timings.size() == raw_snapshots.size() == reps.
    Enforced by construction, asserted in debug.

15. **Data flow section (§8).**  Concrete trace of how dimensions and
    counter values are captured, paired, and stored.

v2 incorporates feedback from two independent reviews.  Key changes:

1. **CSV stripped from Metric concept.**  `ctdp::bench::Metric` returns
   structured `snapshot_type`, not formatted strings.  All serialisation
   lives in `ctdp::calibrator`.

2. **`data_point` as atomic output.**  The calibrator's unit of output is
   `data_point = {space_point, observables}` — the configuration and its
   measurements are paired at collection and never split.

3. **Structured returns from measurement kernel.**  `measure_once` returns
   `{wall_ns, token, metric_snapshot}`.  No `char*` buffers, no CSV.

4. **Cache thrasher: primitive in bench, placement in calibrator.**
   `measure_once` accepts a `SetupCallable` hook for pre-measurement work
   (cache flush, TLB invalidation, etc.).

5. **Policy separated from primitives in environment.h.**  `pin_to(core_id)`
   and `restore_affinity()` are bench primitives.  Core selection heuristics
   move to calibrator.

6. **Flat namespaces:** `ctdp::bench` and `ctdp::calibrator`.

7. **Honest dependency statement.**  Platform APIs listed explicitly.

8. **Tier 0 corrected.**  `steady_clock` everywhere.  TSC and thread CPU
   time are platform enhancements, not baseline.


## 1. Motivation

The v4.1 calibrator prototype (5 files, ~1,650 lines) demonstrated that the
measurement infrastructure and the framework-facing calibration logic are
independently useful and have different dependency profiles.

The measurement layer — environment control, hardware counters, anti-elision
barriers, timing, statistics — is a general-purpose benchmarking toolkit.

The calibration layer — search-space traversal, feature encoding, dataset
assembly, profile emission — is specific to CT-DP's compile-time optimisation
pipeline.  Its fundamental output is the **data point**: the pairing of a
space configuration with its measured observables.  Crucially, a data point
belongs not just to a point in a space but to a specific **callable** at that
point — the expression template, function object, or compiled kernel that the
plan will dispatch to.  The same space can host multiple callables (a SWAR
parser and a lookup parser both live in a parser strategy space), and their
cost surfaces are independent.

Factoring into two libraries delivers three benefits:

1. **Reuse.**  `ctdp::bench` is useful outside CT-DP — any C++
   microbenchmarking task benefits from correct anti-elision, CPU pinning,
   perf counters, and robust statistics.

2. **Testability.**  Each library has a self-contained test surface.

3. **Build isolation.**  `ctdp::bench` depends only on the C++ standard
   library and platform OS APIs.  `ctdp::calibrator` depends on
   `ctdp::bench` and `ctdp::core` but not on the solver or graph libraries.


## 2. The Data Point and Callable Identity

The calibrator exists to produce data points.  A data point is the atomic,
indivisible record that pairs a space configuration with everything observed
when that configuration was measured:

```cpp
namespace ctdp::calibrator {

/// The atomic output of calibration.  Never split.
///
/// space_point:   the configuration that was measured (tile sizes, strategy
///                flags, field counts — whatever the Space defines)
/// observables:   everything measured about that configuration (wall time,
///                counter snapshots, derived statistics)
/// environment:   the conditions under which measurement occurred (cache
///                regime, platform, pinning state)
///
template <typename SpacePoint, typename MetricSnapshot>
struct data_point {
    SpacePoint                    space_point;
    double                        median_ns;
    double                        mad_ns;
    std::vector<double>           raw_timings;   // per-rep wall_ns
    std::vector<MetricSnapshot>   raw_snapshots; // per-rep metric samples
    environment_context           env;           // platform + regime
};

} // namespace ctdp::calibrator
```

The data point is constructed inside `calibration_harness::measure_point` at
the instant measurement completes.  The space point and the observables enter
the same struct before any other operation (serialisation, filtering,
aggregation) can occur.  They travel together from that point forward — into
the CSV writer, the ML pipeline, the profile fitter.

This is the central design invariant.  Everything else in the architecture
serves to produce, transport, or consume data points.

### 2.1 Callable Identity

A space defines the topology — the dimensions and their ranges.  But the
same space can host multiple applications.  A 3-dimensional space of
(strategy, field_count, delimiter) could describe a FIX parser, a JSON
parser, or a CSV parser.  Same dimensions, completely different kernels,
completely different cost surfaces.

In ctdp-core, a plan binds a Space to a Callable:

```cpp
// ctdp-core
template <typename Space, typename Callable>
struct plan {
    using point_type    = typename Space::point_type;
    using callable_type = Callable;
    // The solver selects the optimal point using a cost model
    // fitted to measurements of THIS Callable across THIS Space.
};
```

The Callable is the concrete type that gets invoked — an expression template,
a function object, a compiled kernel.  It is not metadata or provenance.
It is the thing being measured.  The calibrator must know this type because:

1. **Construction.**  The Scenario must construct or reference the executor
   that invokes specialisations of the Callable at each space point.

2. **Type safety.**  A cost model fitted from FIX SWAR parser measurements
   must not be applied to a lookup parser plan, even if both live in the
   same space.  The type system enforces this.

3. **Dispatch.**  The plan's `execute()` dispatches to a Callable
   specialisation.  The calibrator's `Scenario::execute()` invokes the
   same dispatch path so that measurements reflect real execution cost.

The callable_type is an associated type on Scenario.  It propagates through
the architecture:

```
Scenario<callable_type>
    → calibration_harness<Scenario>
        → calibration_dataset<Space, Callable, MetricSnapshot>
            → calibration_profile<Space, Callable>
                → plan<Space, Callable>  (consumes the profile)
```

Individual data_points do not carry the callable_type as a field — they
inherit it from the dataset they belong to, just as CSV rows inherit the
column schema.  A dataset is produced by one Scenario, and a Scenario has
one Callable, so the identity is unambiguous at the dataset level.


## 3. Library Structure

```
ctdp-bench/                          ctdp-calibrator/
  include/ctdp/bench/                  include/ctdp/calibrator/
    compiler_barrier.h                   scenario.h
    perf_counter.h                       data_point.h
    environment.h                        feature_encoder.h
    cache_thrasher.h                     sampler.h
    measurement_kernel.h                 calibration_dataset.h
    statistics.h                         calibration_harness.h
                                         calibration_profile.h
                                         ct_dp_emit.h
                                         plan_validate.h
                                         provenance.h
                                         csv_writer.h
                                         benchmark_explorer.h  (optional)
```

**ctdp-bench** depends on:
- C++ standard library
- OS APIs: pthreads + Linux syscalls (pinning, priority, perf_event_open);
  Win32 (SetThreadAffinityMask, thread priority).  Platform headers are
  isolated behind preprocessor guards.
- No mandatory external libraries.

**ctdp-calibrator** depends on ctdp-bench and ctdp-core.


## 4. ctdp-bench — Measurement Library

### 4.1 Scope

Everything needed to measure a callable reliably on commodity hardware.  No
knowledge of CT-DP spaces, feature vectors, cost models, or serialisation
formats.  Returns structured data only.

### 4.2 Metric Concept (revised — no CSV, explicit latch semantics)

```cpp
namespace ctdp::bench {

template <typename M>
concept Metric = requires(M& m) {
    typename M::snapshot_type;
    { m.start() }    -> std::same_as<void>;
    { m.stop()  }    -> std::same_as<void>;
    { m.snapshot() } -> std::same_as<typename M::snapshot_type>;
};

} // namespace ctdp::bench
```

The concept requires three operations with strict temporal semantics:

- **`start()`** — resets all internal state and arms the counters.  Any
  data from a previous start/stop cycle is discarded.

- **`stop()`** — disarms counters and latches the accumulated values.
  After stop(), the metric holds a frozen snapshot of the interval.

- **`snapshot()`** — returns the latched value.  O(1).  Valid only after
  `stop()`.  Calling `snapshot()` before `stop()` is undefined behaviour.
  Multiple calls to `snapshot()` without an intervening `start()/stop()`
  return the same value.

The snapshot is a structured value type — no string formatting, no buffers,
no CSV awareness.

Built-in implementations:

- **`counter_metric`** — wraps `perf_counter_group`.
  `snapshot_type = counter_snapshot`.  Contains raw counters (cycles,
  instructions, cache misses) and derived ratios (IPC, miss rates).

- **`null_metric`** — zero overhead.
  `snapshot_type = null_snapshot` (empty struct).

### 4.3 Measurement Kernel (revised — structured returns, setup hook)

#### Terminology

Three distinct quantities control the measurement loop:

- **`reps`** — number of independent repetitions.  Each rep produces one
  wall_ns value and one metric snapshot.  Statistical aggregation (median,
  MAD) operates across reps.

- **`warmup_iters`** — throw-away iterations at the start of *each* rep.
  Saturates the branch predictor and warms the instruction cache.  Not timed.

- **`measure_iters`** — iterations inside the timing bracket per rep.
  Each iteration calls fn() + DoNotOptimize + ClobberMemory.  wall_ns for
  the rep is the total elapsed time divided by measure_iters.

#### result_token contract

`result_token` is `uint64_t`.  The token returned by a callable must satisfy:

1. **Data-dependent on the work** — not a constant, not derived solely from
   input addresses.  The compiler must be unable to prove the value without
   executing the kernel.

2. **Cheap to compute** — the token should be a byproduct of the kernel
   (parsed field count, accumulator, checksum), not additional work.

3. **Observable** — `DoNotOptimize(tok)` forces the compiler to materialise
   the value, which transitively requires all work that produced it.

A `mix_token` helper provides a correct-by-default idiom:

```cpp
namespace ctdp::bench {

using result_token = std::uint64_t;

/// Cheap bit-mix.  Prevents returning trivial constants.
/// Usage: return mix_token(parsed_field_count);
inline constexpr result_token mix_token(result_token x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return x;
}

} // namespace ctdp::bench
```

#### wall_ns contract

`wall_ns` is always measured via `std::chrono::steady_clock`.  This is the
primary training label for ML cost models.  TSC cycles, thread CPU time, and
hardware counter values are additional observables inside the metric
snapshot — they do not replace or contaminate the wall_ns measurement.

#### Types

```cpp
namespace ctdp::bench {

template <typename MetricSnapshot>
struct measurement_result {
    double          wall_ns;
    result_token    last_token;
    MetricSnapshot  metric;
};

template <typename MetricSnapshot>
struct repeated_result {
    double                        median_ns;
    double                        mad_ns;
    std::vector<double>           all_ns;       // size == reps
    std::vector<MetricSnapshot>   all_snapshots; // size == reps
    result_token                  last_token;

    // Invariant: all_ns.size() == all_snapshots.size()
    // Enforced by construction in measure_repeated.
    // Asserted in debug builds.
};

} // namespace ctdp::bench
```

#### Functions

```cpp
namespace ctdp::bench {

/// Single measurement with setup hook.
///
/// Callable:       () -> result_token     (the kernel under test)
/// SetupCallable:  () -> void             (pre-measurement: cache flush, etc.)
/// Metric:         satisfies bench::Metric
///
/// Sequence per call:
///   1. warmup: warmup_iters × { fn() + DoNotOptimize + ClobberMemory }
///   2. setup()                          ← caller's hook (e.g. cache thrash)
///   3. metric.start()
///   4. t0 = steady_clock::now()
///   5. for measure_iters: fn() + DoNotOptimize(tok) + ClobberMemory()
///   6. t1 = steady_clock::now()
///   7. metric.stop()
///   8. wall_ns = (t1 - t0) / measure_iters
///   9. return {wall_ns, last_token, metric.snapshot()}
///
template <typename Callable, typename SetupCallable, Metric M>
auto measure_once(Callable&&      fn,
                  SetupCallable&& setup,
                  M&              metric,
                  std::size_t     warmup_iters,
                  std::size_t     measure_iters)
    -> measurement_result<typename M::snapshot_type>;

/// Repeated measurement with statistics.
///
/// Calls measure_once() `reps` times.  Returns median/MAD of wall_ns
/// plus all per-rep raw data (timings and metric snapshots).
///
/// Postcondition:
///   result.all_ns.size() == result.all_snapshots.size() == reps
///
template <typename Callable, typename SetupCallable, Metric M>
auto measure_repeated(Callable&&      fn,
                      SetupCallable&& setup,
                      M&              metric,
                      std::size_t     reps,
                      std::size_t     warmup_iters,
                      std::size_t     measure_iters)
    -> repeated_result<typename M::snapshot_type>;

} // namespace ctdp::bench
```

Key design decisions:

- **`result_token` stays `uint64_t`.**  The v4 red team restriction to a
  single scalar type prevents pointer-aliasing footguns and unwanted spills.
  Any kernel can fold its result into `uint64_t` via `mix_token`.

- **`SetupCallable` hook** enables cache thrashing, TLB invalidation, or any
  pre-measurement preparation without bench knowing what it is.  For
  warm-cache measurement, pass a no-op lambda `[]{}`.

- **ClobberMemory in warmup.**  Warmup iterations include ClobberMemory to
  match the instruction footprint of the measured loop.  The branch predictor
  sees the same instruction sequence it will see during measurement.

- **No measurement mode enum yet.**  The current loop is latency-mode
  (DoNotOptimize + ClobberMemory per iteration, creating serial dependency).
  Throughput mode (drop ClobberMemory) is a documented future extension, not
  an interface change — it would be a second function or a template parameter.

### 4.4 Headers

#### compiler_barrier.h (implemented, 89 lines)

`DoNotOptimize(uint64_t)` — single scalar type only.
`ClobberMemory()` — full compiler memory fence.
`result_token` — wrapper type.

Platform support: GCC/Clang inline asm, MSVC `_ReadWriteBarrier`.

#### perf_counter.h (implemented, 475 lines)

Two-tier hardware counters.

**Tier 0** (all platforms): `steady_clock` wall time.  Always available.

**Tier 0 x86 enhancement**: TSC reference cycles via RDTSC with correct
serialisation (lfence/rdtsc at start, rdtscp/lfence at stop).  Available
when `__x86_64__` or `_M_X64` defined.

**Tier 0 Linux enhancement**: thread CPU time via
`clock_gettime(CLOCK_THREAD_CPUTIME_ID)`.

**Tier 1** (Linux only): `perf_event_open` group — hw_cycles, instructions,
L1d misses, LLC misses, branch misses, dTLB misses.  Atomic group snapshot.
Graceful fallback to Tier 0 on restrictive paranoid level or VM.

All tiers guarded by preprocessor.  Including `perf_counter.h` on Windows
gives Tier 0 only — no linker errors, no missing headers.

`counter_snapshot` struct contains raw counters and derived ratios (IPC,
miss rates, frequency ratio, CPU utilisation).

#### environment.h (implemented, 290 lines — primitives only)

RAII environment control.  **Primitives only — no policy.**

- `pin_to(int core_id)` — pin current thread to specific core.
- `restore_affinity()` — restore saved affinity mask.
- `elevate_priority()` / `restore_priority()` — thread priority control.
- `detect_platform()` — CPU model, cache sizes, OS, counter tier.
- `environment_guard` — RAII composition of pin + priority + restore.

**Removed from bench:** `select_pin_cpu()` (the "quietest core" heuristic)
moves to `ctdp::calibrator` as policy.  Bench provides the mechanism,
calibrator decides which core.

#### cache_thrasher.h (extracted from harness)

Standalone cache thrashing primitive.  Sized from LLC capacity at
construction.  `thrash()` performs random-stride writes across 1.5× LLC
buffer and ends with `ClobberMemory()` to guarantee the writes are not
elided.  The final written element is passed through `DoNotOptimize` as
an additional guard.

Contract: after `thrash()` returns, the entire LLC has been evicted with
high probability.  The function is non-elidable by construction.

#### measurement_kernel.h (new — extracted from calibrator.h)

`measure_once` and `measure_repeated` as described in §4.3.

#### statistics.h (extracted from calibrator.h)

`median(span<double>)` and `mad(span<double>)` as free functions.


## 5. ctdp-calibrator — Framework-Facing Library

### 5.1 Scope

Knows about CT-DP spaces and plan types.  Orchestrates the measurement of a
combinatorial search space.  Produces data points.  Owns all serialisation.

### 5.2 The Scenario Concept

A Scenario binds a Space to a Callable and adapts them for measurement.
It is the type-level contract that says "this calibration run measures
*this kernel* across *this space*."

```cpp
namespace ctdp::calibrator {

template <typename S>
concept Scenario = requires(S& s, typename S::point_type const& pt) {
    typename S::point_type;
    typename S::callable_type;   // the application identity
    { s.name()     } -> std::convertible_to<std::string_view>;
    { s.points()   };  // iterable of point_type
    { s.prepare(pt) };
    { s.execute(pt) } -> std::convertible_to<bench::result_token>;
};

} // namespace ctdp::calibrator
```

The `callable_type` associated type identifies the concrete kernel being
measured.  It is the same type that appears in `plan<Space, Callable>`.
The calibrator propagates it to the dataset and profile so the type system
prevents mismatched cost models.

A concrete Scenario implementation holds (or references) the executor that
invokes specialisations of the Callable at each space point:

```cpp
// Example: FIX parser scenario
struct fix_parser_scenario {
    using point_type    = parser_point;
    using callable_type = fix_swar_parser;  // the kernel being calibrated

    auto name() const -> std::string_view { return "fix_swar_parser"; }
    auto points() const -> /* iterable */;
    void prepare(point_type const& pt);
    auto execute(point_type const& pt) -> bench::result_token;

private:
    dispatch_table<parser_space, fix_swar_parser> table_;
    std::vector<char> message_buffer_;
};
```

Note: feature encoding and CSV formatting are **not** part of Scenario.
Scenario defines the search space, identifies the callable, and executes
the kernel.  Encoding and serialisation are separate concerns handled by
the harness and writer.

The Scenario concept remains a single "all-in-one adapter" for now.
Decomposition into separate point-source, executor, and encoder types is a
documented future direction when more real scenarios reveal what varies
independently.

Note: both Scenario (via `points()`) and sampler.h can enumerate points.
The authoritative source is Scenario — `sampler.h` produces a batch by
sampling from `scenario.points()`.  Scenario defines the full space;
sampler selects a subset for calibration.

### 5.3 Feature Encoding

Feature encoding converts a `point_type` into a fixed-width numeric vector
for the ML pipeline:

```cpp
template <typename E, typename PointType>
concept FeatureEncoder = requires(E& enc, PointType const& pt) {
    { E::width } -> std::convertible_to<std::size_t>;
    { E::column_names } ;  // static constexpr array<string_view, width>
    { enc.encode(pt) } -> std::convertible_to<std::array<float, E::width>>;
};
```

Returns `std::array<float, W>`, not `char*` buffer.  Column names are
compile-time constants (`static constexpr std::array<std::string_view, W>`)
to avoid lifetime issues with dynamically constructed views.  The CSV writer
handles formatting.

### 5.4 The Calibration Harness

The composition point.  Templates on Scenario and bench::Metric.
Propagates the Scenario's callable_type to the dataset type so that
downstream consumers (profile fitter, solver, validator) can verify
type compatibility.

```cpp
namespace ctdp::calibrator {

template <Scenario S, bench::Metric M = bench::counter_metric>
class calibration_harness {
public:
    using point_type      = typename S::point_type;
    using callable_type   = typename S::callable_type;
    using snapshot_type   = typename M::snapshot_type;
    using data_point_type = data_point<point_type, snapshot_type>;

    struct config {
        std::size_t reps            = 10;   // statistical repetitions
        std::size_t warmup_iters    = 200;  // per-rep warmup iterations
        std::size_t measure_iters   = 1;    // iterations inside timing bracket
        bool        pin_cpu         = true;
        int         pin_cpu_id      = -1;   // -1 = auto (policy here)
        bool        boost_priority  = true;
        bool        flush_cache     = true;
        std::size_t llc_bytes       = 0;    // 0 = auto-detect
        bool        verbose         = true;
    };

    explicit calibration_harness(S& scenario,
                                 config const& cfg = {},
                                 M metric = M{});

    /// Run the full sweep.  Returns all data points.
    auto run() -> std::vector<data_point_type>;

private:
    /// Measure one space point.  Pairs result with point immediately.
    auto measure_point(point_type const& pt) -> data_point_type {
        // 1. Scenario prepares (Part B: space-aware)
        scenario_.prepare(pt);

        // 2. Create opaque callable for bench (the seam)
        auto fn = [&]() -> bench::result_token {
            return scenario_.execute(pt);
        };

        // 3. Create setup hook (cache thrash or no-op)
        auto setup = config_.flush_cache
            ? [this]() { thrasher_.thrash(); }
            : []() {};

        // 4. Measure (Part A: pure measurement)
        auto result = bench::measure_repeated(
            fn, setup, metric_,
            config_.reps,
            config_.warmup_iters,
            config_.measure_iters);

        // Invariant: per-rep vectors have identical size
        assert(result.all_ns.size() == result.all_snapshots.size());

        // 5. WELD: pair space point with observables (never split)
        return data_point_type{
            .space_point    = pt,
            .median_ns      = result.median_ns,
            .mad_ns         = result.mad_ns,
            .raw_timings    = std::move(result.all_ns),
            .raw_snapshots  = std::move(result.all_snapshots),
            .env            = current_env_context()
        };
    }

    S&                   scenario_;
    config               config_;
    M                    metric_;
    bench::cache_thrasher thrasher_;
};

} // namespace ctdp::calibrator
```

The harness's `run()` returns `vector<data_point_type>`.  What the caller
does with those data points — write CSV, fit a model, validate a plan — is
a separate decision.

### 5.5 CSV Writer (new — serialisation separated)

```cpp
namespace ctdp::calibrator {

/// Write data points to CSV.
///
/// Templated on FeatureEncoder (for point columns) and a snapshot
/// formatter (for metric columns).  Knows about data_point structure
/// but NOT about bench::Metric or measurement internals.
///
template <typename FeatureEncoder, typename SnapshotFormatter>
class csv_writer {
public:
    void write_header(std::ostream& os) const;
    void write_row(std::ostream& os,
                   data_point<...> const& dp,
                   std::size_t rep_index) const;
    void write_dataset(std::ostream& os,
                       std::span<const data_point<...>> points) const;
};

/// Built-in snapshot formatter for counter_snapshot.
struct counter_snapshot_formatter {
    static auto columns() -> std::span<const std::string_view>;
    static void format(std::ostream& os, bench::counter_snapshot const& snap);
};

} // namespace ctdp::calibrator
```

### 5.6 Remaining Headers

**calibration_dataset.h** — typed dataset container wrapping
`vector<data_point>` with schema metadata, provenance, and callable
identity.  Templated on `<Space, Callable, MetricSnapshot>`.  The
Callable type parameter is the compile-time key that prevents datasets
from being fed to the wrong cost model:

```cpp
template <typename Space, typename Callable, typename MetricSnapshot>
struct calibration_dataset {
    using point_type    = typename Space::point_type;
    using callable_type = Callable;

    std::vector<data_point<point_type, MetricSnapshot>> points;
    dataset_provenance                                  provenance;

    // Type identity: only a profile<Space, Callable> can consume this
};
```

**calibration_profile.h** — fitted model parameters consumed by solver
cost models.  Templated on `<Space, Callable>`.  A profile fitted from
`fix_swar_parser` data cannot be accidentally applied to a
`fix_lookup_parser` plan — the type mismatch is a compile error:

```cpp
template <typename Space, typename Callable>
struct calibration_profile {
    using callable_type = Callable;
    // model parameters, schema, provenance
};
```

**ct_dp_emit.h** — emits `constexpr` headers from fitted profiles.  The
emitted header encodes the Callable identity as a type alias or tag so
that the solver can verify compatibility at compile time.

**plan_validate.h** — re-measures individual plan steps using
`bench::measure_once`, compares predicted vs actual cost.  Templated on
`<Space, Callable>` to ensure the validator uses the same executor as
the calibrator:

```cpp
template <typename Space, typename Callable, bench::Metric M = bench::counter_metric>
auto validate_plan(
    plan<Space, Callable> const& p,
    Scenario auto& scenario,    // must have matching callable_type
    bench::environment_guard& env,
    double tolerance = 0.10
) -> validation_result;
```

**provenance.h** — platform fingerprint, build ID, schema versioning.

**benchmark_explorer.h** — optional GBench adapter for interactive
exploration.  Not part of production pipeline.  Requires linking
`libbenchmark`.


## 6. Dependency Graph

```
┌─────────────────────────────────────────────────────┐
│                 ctdp-calibrator                      │
│                                                     │
│  scenario.h ──────────► calibration_harness.h       │
│   (callable_type)              │                    │
│  data_point.h ────────►        │                    │
│  feature_encoder.h ───►        │                    │
│  sampler.h ───────────►        │                    │
│                                ▼                    │
│                         csv_writer.h                │
│                                │                    │
│  calibration_dataset.h ◄───────┘                    │
│   <Space, Callable, Snapshot>                       │
│                                                     │
│  calibration_profile.h ──► ct_dp_emit.h             │
│   <Space, Callable>                                 │
│                                                     │
│  plan_validate.h                                    │
│   <Space, Callable>                                 │
│                                                     │
│  provenance.h                                       │
│                                                     │
│  Depends on: ctdp-bench, ctdp-core                  │
└───────────────────────┬─────────────────────────────┘
                        │ uses
┌───────────────────────▼─────────────────────────────┐
│                   ctdp-bench                         │
│                                                     │
│  compiler_barrier.h     (DoNotOptimize, ClobberMem) │
│  perf_counter.h ──────► counter_metric              │
│  environment.h  ──────► environment_guard            │
│  cache_thrasher.h       (LLC-sized, random stride)  │
│  measurement_kernel.h ◄── statistics.h              │
│                                                     │
│  Depends on: C++ stdlib, OS platform APIs            │
│  Optional: perf_event_open (Linux, Tier 1)          │
│                                                     │
│  No knowledge of Space, Callable, or CT-DP types    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   ctdp-core                          │
│                                                     │
│  Space concepts     — defines dimensions + topology │
│  Callable types     — expression templates, kernels │
│  plan<Space, Callable> — optimal dispatch plan      │
│                                                     │
│  (consumed as-is — not part of this project)        │
└─────────────────────────────────────────────────────┘
```

The callable_type flows top-down:

```
ctdp-core defines:   Callable (the kernel type)
Scenario declares:   callable_type = Callable
Harness propagates:  callable_type from Scenario
Dataset carries:     <Space, Callable, Snapshot>
Profile carries:     <Space, Callable>
Plan consumes:       profile<Space, Callable> for cost model
Validator checks:    plan<Space, Callable> vs scenario<callable_type>
```


## 7. The Decoupling Seam

```
calibration_harness::measure_point(pt)
    │
    ├── scenario_.prepare(pt)                     // Part B: space-aware
    │
    ├── auto fn = [&]() { return                  // Lambda bridge
    │       scenario_.execute(pt); };
    │
    ├── auto setup = [&]() {                      // Setup hook
    │       thrasher_.thrash(); };                 //   (or no-op)
    │
    ├── bench::measure_repeated(fn, setup,        // Part A: pure measurement
    │       metric_, reps, warmup_iters, measure_iters)
    │       │
    │       ├── per rep:
    │       │   ├── warmup: warmup_iters × fn() + DoNotOptimize + ClobberMemory
    │       │   ├── setup()                       ← caller's hook
    │       │   ├── metric.start()
    │       │   ├── t0 = steady_clock::now()
    │       │   ├── measure_iters × fn() + DoNotOptimize + ClobberMemory
    │       │   ├── t1 = steady_clock::now()
    │       │   ├── metric.stop()
    │       │   └── record {wall_ns, token, metric.snapshot()}
    │       │
    │       └── return {median_ns, mad_ns, all_ns[], all_snapshots[]}
    │
    └── WELD: data_point{pt, observables, env}    // Part B: never split
```

Part A never sees `point_type`, feature vectors, or CSV.
Part B never manages timing, counter setup, or anti-elision.
The data point is assembled at the earliest possible moment.


## 8. Data Flow: From Dimensions to Stored Data Point

This section traces the concrete path of a single data point from space
configuration through measurement to storage.

### 8.1 The Space Point (dimensions)

A space point is a concrete configuration from the search space.  Its fields
are the independent variables — the dimensions of the experiment:

```cpp
// Example: FIX parser scenario
struct parser_point {
    Strategy    strategy;      // e.g. SWAR, Unrolled, Lookup
    int         field_count;   // e.g. 8
    char        delimiter;     // e.g. '|'
};
```

The scenario's `points()` method yields these.  The harness iterates them.

### 8.2 The Callable (application identity)

The space defines WHERE — but the callable defines WHAT.  Two scenarios
can share the same `parser_point` type but measure different kernels:

```cpp
struct fix_swar_scenario {
    using point_type    = parser_point;       // same space
    using callable_type = fix_swar_parser;    // different kernel
    // ...
};

struct fix_lookup_scenario {
    using point_type    = parser_point;       // same space
    using callable_type = fix_lookup_parser;  // different kernel
    // ...
};
```

At `parser_point{SWAR, 8, '|'}`, the SWAR scenario measures the SWAR
kernel's execution cost.  The lookup scenario measures the lookup kernel's
cost at the same point.  These are different observables — different cost
surfaces — and must not be mixed.

The callable_type does not appear as a field in the data_point.  It is a
type parameter on the dataset that contains the data points:

```cpp
calibration_dataset<parser_space, fix_swar_parser, counter_snapshot>
//                  ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
//                  Space         Callable          MetricSnapshot
```

Every data_point in this dataset was produced by measuring `fix_swar_parser`.
The type system guarantees this — the harness's Scenario has
`callable_type = fix_swar_parser`, and only data_points from that harness
enter this dataset.

### 8.3 The Measurement (observables)

For each space point, `bench::measure_repeated` is called `reps` times.
Each rep produces a wall_ns (from steady_clock) and a metric snapshot
(from the Metric).  `measure_repeated` returns all raw per-rep data plus
median/MAD statistics:

```cpp
// bench returns this — structured, no formatting
repeated_result<counter_snapshot> {
    .median_ns     = 14.2,
    .mad_ns        = 0.3,
    .all_ns        = {14.1, 14.2, 14.5, 14.0, 13.9, ...},  // reps entries
    .all_snapshots = {                                        // reps entries
        counter_snapshot{
            .ref_tsc        = 4200,
            .hw_cycles      = 4180,
            .instructions   = 12600,
            .l1d_misses     = 2,
            .llc_misses     = 0,
            .branch_misses  = 1,
            .dtlb_misses    = 0,
            // derived ratios
            .ipc            = 3.01,
            .l1d_miss_rate  = 0.00016,
            .llc_miss_rate  = 0.0,
            .branch_miss_rate = 0.00008,
            .freq_ratio     = 0.995,
            .cpu_utilisation = 0.98
        },
        counter_snapshot{ ... },   // rep 1
        ...                        // reps entries total
    },
    .last_token = 0x8a3f...
}
```

The size invariant holds: `all_ns.size() == all_snapshots.size() == reps`.

### 8.4 The Weld (data_point construction)

Inside `calibration_harness::measure_point`, the space point and the
observables are paired into a single value type.  This is the moment they
become inseparable:

```cpp
data_point<parser_point, counter_snapshot> {
    // THE DIMENSIONS — what was measured
    .space_point = parser_point{
        .strategy    = Strategy::SWAR,
        .field_count = 8,
        .delimiter   = '|'
    },

    // THE OBSERVABLES — what was observed
    .median_ns     = 14.2,
    .mad_ns        = 0.3,
    .raw_timings   = {14.1, 14.2, 14.5, 14.0, 13.9, ...},
    .raw_snapshots = {snap_rep0, snap_rep1, snap_rep2, ...},

    // THE CONDITIONS — where it was measured
    .env = environment_context{
        .cpu_model    = "Intel i9-13900K",
        .pinned_core  = 23,
        .l1d_bytes    = 49152,
        .l3_bytes     = 36700160,
        .os           = "Linux 6.5",
        .counter_tier = tier::tier1
    }
}
```

After construction, the data_point is a self-contained value.  It can be
copied, moved, stored in a vector, serialised, or passed to a fitter.
The dimensions and observables are never in separate tables, never joined
by index, never at risk of misalignment.

### 8.5 Serialisation (CSV)

The `csv_writer` flattens a data_point into rows.  One row per rep, with
the space point dimensions and environment repeated on every row:

```
strategy,field_count,delimiter,rep,wall_ns,median_ns,mad_ns,ref_tsc,hw_cycles,...,ipc,l1d_miss_rate,...
SWAR,8,|,0,14.1,14.2,0.3,4200,4180,...,3.01,0.00016,...
SWAR,8,|,1,14.2,14.2,0.3,4150,4130,...,3.05,0.00024,...
SWAR,8,|,2,14.5,14.2,0.3,4220,4200,...,2.98,0.00016,...
```

The feature encoder converts the space point into numeric columns (one-hot
for enums, direct for scalars).  The snapshot formatter converts the
counter_snapshot into its column set.  The csv_writer composes both.

The wall_ns column is the per-rep measurement from steady_clock — the
primary training label.  median_ns and mad_ns are repeated on every row
for convenience but are properties of the full data_point, not of
individual reps.  The counter columns (ref_tsc through cpu_utilisation)
are per-rep snapshots that serve as additional ML features alongside the
space-point dimensions.

### 8.6 Data ownership summary

```
Core defines:      Space (the coordinate system) + Callable (the kernel)
Space defines:      point_type (the independent variables / dimensions)
Scenario binds:     Space + Callable → callable_type + points() + execute()
Bench measures:     wall_ns + counter_snapshot (the dependent variables)
Harness welds:      data_point = {space_point, observables, env}
Dataset carries:    <Space, Callable, Snapshot> — type-safe container
Writer flattens:    CSV rows with dimensions, wall_ns, and counters
Fitter produces:    profile<Space, Callable> (the cost model)
Plan consumes:      profile<Space, Callable> for optimal dispatch
Validator checks:   plan<Space, Callable> predictions vs measurements
```

The Callable identity flows through the type system from Scenario to Plan.
It is never a runtime string or tag — it is a compile-time type parameter
that makes mismatched cost models a type error.


## 9. What Moves Where (from v4.1 prototype)

| Current location | Content | Destination |
|---|---|---|
| `compiler_barrier.h` | DoNotOptimize, ClobberMemory, result_token | **bench** — as-is |
| `perf_counter.h` | perf_counter_group, counter_snapshot | **bench** — as-is |
| `perf_counter.h` | counter_metric, null_metric | **bench** — revised (snapshot() replaces to_csv()) |
| `environment.h` | pin_to, restore, elevate, guard, detect | **bench** — primitives only |
| `environment.h` | select_pin_cpu (quietest core) | **calibrator** — policy |
| `calibrator.h` | Scenario concept | **calibrator** — scenario.h (encoding removed) |
| `calibrator.h` | Metric concept | **bench** — measurement_kernel.h (CSV removed) |
| `calibrator.h` | median(), mad() | **bench** — statistics.h |
| `calibrator.h` | calibrator_config | **calibrator** — calibration_harness.h |
| `calibrator.h` | calibrator class (harness + loop) | Split: inner loop → **bench**, orchestration → **calibrator** |
| `calibrator.h` | CSV header/row writing | **calibrator** — csv_writer.h |
| `calibrator_main.cpp` | example scenarios | **calibrator** — examples/ |
| `calibrator_main.cpp` | CLI + main | **calibrator** — examples/ |


## 10. Deferred Design Decisions (documented, not forgotten)

**Throughput measurement mode.**  Current loop is latency-mode (serial
dependency via ClobberMemory).  Throughput mode (drop ClobberMemory, allow
OoO overlap) is a future `measure_once` variant or template parameter.
Interface-compatible — does not require concept changes.

**Scenario decomposition.**  Scenario currently bundles space enumeration,
execution, and feature encoding.  Decomposition into separate point-source,
executor, and encoder concepts deferred until multiple real scenarios reveal
what varies independently.

**result_token flexibility.**  Fixed at `uint64_t`.  The v4 red team
restriction prevents pointer-aliasing footguns.  Any kernel can fold its
result into `uint64_t`.  Not templated.

**Binary dataset format.**  CSV is the initial output format.  Parquet,
protobuf, or memory-mapped binary formats are future alternatives.  The
`data_point` struct is format-agnostic by design — adding a new writer
does not change bench or the harness.

**perf_counter split.**  Currently a single header with preprocessor guards.
If it grows past ~600 lines, split into `perf_counter_linux.h` and
`perf_counter_portable.h`.


## 11. Implementation Plan

The prototype implements the bulk of both libraries.  The work is primarily
extraction, interface revision (Metric concept, structured returns), and the
new data_point/csv_writer types.

### Phase 1: ctdp-bench extraction (~4.5 hours from existing code)

The prototype files provide ~950 of the ~1,100 lines needed.  Phase 1
is extraction and interface revision, not greenfield development.
See `phase1_implementation_plan.md` for line-by-line mapping.

| Step | Description | Source | Effort |
|---|---|---|---|
| 1.1 | compiler_barrier.h — namespace + mix_token | Prototype as-is + 4 lines | 15 min |
| 1.2 | statistics.h — extract median/mad | calibrator.h:136–160 | 15 min |
| 1.3 | perf_counter.h — namespace change | Prototype as-is | 20 min |
| 1.4 | metric.h — revised concept (snapshot API), counter_metric, null_metric | calibrator.h:82–132, revised | 45 min |
| 1.5 | environment.h — extract select_pin_cpu to policy | Prototype minus policy | 30 min |
| 1.6 | cache_thrasher.h — new standalone (~40 lines) | New from v4 design | 20 min |
| 1.7 | measurement_kernel.h — measure_once, measure_repeated | calibrator.h:295–354, restructured | 60 min |
| 1.8 | Tests — barrier, counters, environment, measure_repeated | perf_counter_test.cpp + new | 60 min |

Deliverable: standalone ctdp-bench library (~1,100 lines, 7 headers),
compiles and tests independently with no CT-DP knowledge.

### Phase 2: ctdp-calibrator extraction (~1 day from existing code)

Once ctdp-bench is standalone, the calibrator is primarily the harness,
data_point type, csv_writer, and ported example scenarios.

| Step | Description | Source | Effort |
|---|---|---|---|
| 2.1 | Directory structure + CMakeLists | — | 15 min |
| 2.2 | data_point.h — atomic output type | New | 30 min |
| 2.3 | scenario.h — concept with callable_type | calibrator.h:63–80, revised | 20 min |
| 2.4 | calibration_harness.h — compose bench + scenario, produce data_points | calibrator.h:193–376, restructured | 2 hrs |
| 2.5 | csv_writer.h — serialise data_points + counter_snapshot_formatter | calibrator.h CSV code, restructured | 45 min |
| 2.6 | Port example scenarios + CLI main | calibrator_main.cpp (381 lines) | 45 min |
| 2.7 | Tests — mock scenario, data_point assembly, CSV round-trip | New | 45 min |

Deliverable: ctdp-calibrator links against ctdp-bench, produces CSV
from example scenarios.  callable_type propagated through all types.

### Phase 3: Remaining v4 headers (2–3 days)

| Step | Description | Effort | Input |
|---|---|---|---|
| 3.1 | feature_encoder.h — point_type → array<float, W> | 0.5d | v4 design §4 |
| 3.2 | sampler.h — deterministic batch extraction from space | 0.5d | v4 design §5 |
| 3.3 | provenance.h — platform fingerprint, schema IDs | 0.5d | v4 design §3.1 |
| 3.4 | calibration_dataset.h — typed container + schema metadata | 0.5d | v4 design §7 |
| 3.5 | calibration_profile.h + ct_dp_emit.h — profile type + constexpr emission | 0.5d | v4 design §8–9 |
| 3.6 | plan_validate.h — validate plans via bench::measure_once | 0.5d | v4 design §10 |

### Phase 4: Integration (1 day)

| Step | Description | Effort |
|---|---|---|
| 4.1 | End-to-end test: FIX parser scenario → data_points → CSV → profile | 0.5d |
| 4.2 | benchmark_explorer.h (optional GBench adapter) | 0.5d |

**Total: ~2 days for Phases 1–2** (working libraries from existing code),
**plus 3–4 days for Phases 3–4** (remaining v4 headers + integration).
Overall: **~5–6 days.**
