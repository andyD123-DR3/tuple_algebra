#ifndef CTDP_CALIBRATOR_FIX_BENCH_H
#define CTDP_CALIBRATOR_FIX_BENCH_H

// ============================================================
//  fix_bench.h  –  CT-DP FIX Parser Optimiser
//
//  Measurement harness: populates a DataPoint<N> corpus by
//  running the A1/A2/B three-run protocol for each plan.
//
//  Architecture — compile-time dispatch table (Trivial tier):
//    The ET parser templates on fix_config (a constexpr array).
//    We cannot select a template instantiation at runtime by
//    value, so we build a flat array of function pointers —
//    one per valid dense_id — at compile time, then index it
//    at runtime via SchemaIndex::encode(plan).
//
//    For N=4 (trivial_schema) the table has 64 entries and
//    instantiates 64 ET parser specialisations.  Compile time
//    is ~10–20s on a modern machine; the resulting binary is
//    the actual benchmark artifact for the talk.
//
//    For N=12 (full_schema, 4^12 = 16M plans) exhaustive
//    instantiation is impossible.  bench_corpus_n12 accepts
//    an explicit list of plans to measure — used by the beam
//    search loop which proposes candidate plans on demand.
//
//  Three-run protocol (calibrator_design_v2 §3-run):
//    A1: warm run (no pre-flush)
//    A2: immediate warm run (no flush between A1 and A2)
//    B:  post-break run (LLC thrash before measurement)
//
//    Instability ratio |A1−A2|/min(A1,A2) drives DataPointStatus.
//    B latency detects warm-cache bias: if B >> A1/A2 the plan
//    is sensitive to cold-start effects.
//
//  PerfSample counter slots (from config_metrics):
//    [0] instructions     — instructions per parse
//    [1] cycles           — TSC cycles per parse
//    [2] ipc              — instructions per cycle
//    [3] l1d_misses       — L1D read misses per parse
//    [4] branch_miss_rate — branch miss rate
//    [5] cache_miss_rate  — LL cache miss rate
//
//  Dependencies: fix_schema.h (top-level aggregator)
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/fix_schema.h>
#include <ctdp/bench/cache_thrasher.h>

#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  BenchConfig — tunable parameters for the measurement harness
// ─────────────────────────────────────────────────────────────────────────────

struct BenchConfig {
    measurement_config meas;          // passed to measure_config_with_counters
    bool   flush_before_b  = true;    // LLC thrash between A2 and B
    bool   use_fitted_p99  = true;    // use dist-fitted p99 (more stable than empirical)
    std::size_t llc_bytes  = 0;       // 0 = 8 MiB default in cache_thrasher

    // Progress callback: called after each plan is measured.
    // Arguments: (plans_done, plans_total)
    std::function<void(int, int)> progress_cb;
};

// ─────────────────────────────────────────────────────────────────────────────
//  adapter — config_metrics → PerfSample
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline PerfSample
metrics_to_perf_sample(const config_metrics& m, bool use_fitted) noexcept {
    PerfSample s{};
    s.p99_ns       = use_fitted ? m.fitted_p99 : m.timing.p99;
    if (s.p99_ns <= 0.0) s.p99_ns = m.timing.p99;   // fallback if fit failed
    s.counters[0]  = m.instructions;
    s.counters[1]  = m.cycles;
    s.counters[2]  = m.ipc;
    s.counters[3]  = m.l1d_misses;
    s.counters[4]  = m.branch_miss_rate;
    s.counters[5]  = m.cache_miss_rate;
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Compile-time dispatch table for N=4 (trivial_schema, 64 plans)
//
//  Each entry is a function pointer:
//    config_metrics(*)(const std::vector<std::string>&, const measurement_config&)
//  indexed by dense_id in [0, 64).
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

// One instantiation per dense_id.
template<std::size_t DenseId>
[[nodiscard]] config_metrics
measure_trivial(const std::vector<std::string>& msgs,
                const measurement_config&       cfg)
{
    constexpr auto plan    = trivial_index.decode(static_cast<int32_t>(DenseId));
    constexpr auto fix_cfg = schema_to_fix_config<4>(plan);
    return measure_config_with_counters<fix_cfg>(msgs, cfg);
}

using TrivialMeasureFn = config_metrics(*)(const std::vector<std::string>&,
                                           const measurement_config&);

// Build the 64-entry dispatch table via index_sequence expansion.
inline constexpr std::array<TrivialMeasureFn, 64>
trivial_dispatch_table =
    []<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array<TrivialMeasureFn, 64>{ measure_trivial<Is>... };
    }(std::make_index_sequence<64>{});

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
//  measure_one_trivial — A1/A2/B protocol for one N=4 plan
// ─────────────────────────────────────────────────────────────────────────────

inline RunTriple
measure_one_trivial(int32_t                         dense_id,
                    const std::vector<std::string>& messages,
                    const BenchConfig&               cfg,
                    bench::cache_thrasher&           thrasher)
{
    assert(trivial_index.is_valid_id(dense_id));
    auto fn = detail::trivial_dispatch_table[static_cast<std::size_t>(dense_id)];

    RunTriple rt{};
    // A1 — first warm run
    rt.a1 = metrics_to_perf_sample(fn(messages, cfg.meas), cfg.use_fitted_p99);

    // A2 — immediate back-to-back (no flush)
    rt.a2 = metrics_to_perf_sample(fn(messages, cfg.meas), cfg.use_fitted_p99);

    // Optional LLC thrash before B
    if (cfg.flush_before_b) thrasher.thrash();

    // B — post-break baseline
    rt.b  = metrics_to_perf_sample(fn(messages, cfg.meas), cfg.use_fitted_p99);

    return rt;
}

// ─────────────────────────────────────────────────────────────────────────────
//  bench_corpus — measure all plans in a DataPoint<4> corpus
//
//  corpus: skeleton produced by make_data_points_from_schema(trivial_schema).
//          All entries must have valid plans; status/runs will be overwritten.
//
//  After completion:
//    - dp.runs filled with A1/A2/B PerfSamples
//    - dp.status set via update_status()
//    - assign_folds() NOT called here — caller should do that after bench
//      so fold assignment only covers Ok points
// ─────────────────────────────────────────────────────────────────────────────

inline void
bench_corpus(std::span<DataPoint<4>>          corpus,
             const std::vector<std::string>&  messages,
             const BenchConfig&               cfg = {})
{
    bench::cache_thrasher thrasher(cfg.llc_bytes);
    const int total = static_cast<int>(corpus.size());

    for (int i = 0; i < total; ++i) {
        auto& dp = corpus[i];
        int32_t dense_id = trivial_index.encode(dp.plan);
        assert(dense_id >= 0 && "bench_corpus: plan not valid for trivial_schema");

        dp.runs = measure_one_trivial(dense_id, messages, cfg, thrasher);
        dp.update_status();

        if (cfg.progress_cb) cfg.progress_cb(i + 1, total);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  bench_corpus_n12 — measure a selected subset of N=12 plans
//
//  For full_schema (4^12 = 16M), exhaustive instantiation is impractical.
//  Instead, the beam-search loop calls this with a small candidate set
//  (e.g. 20–200 plans per beam iteration).
//
//  The measurement function is provided by the caller as a runtime functor
//  because we cannot build a 16M-entry dispatch table. The recommended
//  pattern is to precompile a fixed set of candidate parsers into a
//  separate translation unit and supply a dispatcher from there.
//
//  plan_fn: (const array<Strategy,12>&, const measurement_config&)
//              → config_metrics
// ─────────────────────────────────────────────────────────────────────────────

inline void
bench_corpus_n12(
    std::span<DataPoint<12>>                                    corpus,
    const std::vector<std::string>& /*messages*/,
    const BenchConfig&                                          cfg,
    std::function<config_metrics(const std::array<Strategy,12>&,
                                 const measurement_config&)>    plan_fn)
{
    bench::cache_thrasher thrasher(cfg.llc_bytes);
    const int total = static_cast<int>(corpus.size());

    for (int i = 0; i < total; ++i) {
        auto& dp = corpus[i];

        auto measure = [&](const measurement_config& m) {
            return plan_fn(dp.plan, m);
        };

        RunTriple rt{};
        rt.a1 = metrics_to_perf_sample(measure(cfg.meas), cfg.use_fitted_p99);
        rt.a2 = metrics_to_perf_sample(measure(cfg.meas), cfg.use_fitted_p99);
        if (cfg.flush_before_b) thrasher.thrash();
        rt.b  = metrics_to_perf_sample(measure(cfg.meas), cfg.use_fitted_p99);

        dp.runs = rt;
        dp.update_status();

        if (cfg.progress_cb) cfg.progress_cb(i + 1, total);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Default message pool helper
//
//  Generates a pool of synthetic FIX messages consistent with trivial_schema
//  (or full_schema) field widths, matching the format expected by the ET parser.
//  Delegates to generate_messages() from fix_et_parser.h.
//
//  pool_size: number of distinct messages (default 1024 — fits L1/L2, avoids
//             repeated-message branch-prediction effects)
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<std::string>
make_message_pool(std::size_t pool_size = 1024) {
    return generate_message_pool(pool_size);
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_BENCH_H
