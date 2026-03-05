// ============================================================
//  data_point.h  –  CT-DP FIX Parser Optimiser  (Phase 1 calibrator)
//
//  A DataPoint<N> represents one measured plan configuration:
//    • N-field strategy plan (one Strategy per field position)
//    • Three timed runs (A1, A2, B) each with p99 latency + 6 HW counters
//    • A DataPointStatus classifying measurement quality
//    • A fold_id for 5-fold CV partitioning
//    • Feature-encoding helpers consumed by counter_preprocessor.h
//
//  Protocol summary (calibrator_design_v2, §3-run protocol):
//    A1 – first measurement run
//    A2 – second measurement run, back-to-back with A1
//    B  – third measurement run after a deliberate break (context switch
//         or cache flush) to catch warm-cache bias
//
//  Status classification:
//    Ok          – |A1-A2|/min(A1,A2) ≤ SOFT_THRESH  (default 0.05)
//    Suspicious  – SOFT_THRESH < ratio ≤ HARD_THRESH  (default 0.20)
//                  Excluded from Stage 1 training; retained for inspection
//                  and potential CounterAdapter transfer
//    HardDeleted – ratio > HARD_THRESH or measurement failure
//                  Excluded from all training
//
//  Feature layout (one-hot strategy encoding, calibrator_design_v2 §encoding):
//    For each of the N field positions, a block of NUM_STRATEGIES bits is
//    set to 1 at the active strategy index, 0 elsewhere.
//    Total strategy features = N * NUM_STRATEGIES.
//    Counter features = NUM_COUNTERS (mean over A1+A2; B available separately).
//
//  Dependencies: none (std only)
//  C++ standard: C++20
//  Compiler: GCC 12+ / Clang 15+ / MSVC 19.34+
// ============================================================

#ifndef CTDP_CALIBRATOR_FIX_DATA_POINT_H
#define CTDP_CALIBRATOR_FIX_DATA_POINT_H

#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  Strategy — per-field parse strategy (must stay in sync with fix_field_descriptor.h)
// ─────────────────────────────────────────────────────────────────────────────

enum class Strategy : uint8_t {
    Unrolled = 0,  // U – fully unrolled digit loop (fastest for short fixed fields)
    SWAR     = 1,  // S – SWAR word-at-a-time (unconditional; any field width)
    Loop     = 2,  // L – scalar counted loop
    Generic  = 3,  // G – branch-on-delimiter fallback
    Count_   = 4   // sentinel — not a valid strategy
};

inline constexpr int NUM_STRATEGIES = static_cast<int>(Strategy::Count_);

constexpr std::string_view strategy_char(Strategy s) noexcept {
    switch (s) {
        case Strategy::Unrolled: return "U";
        case Strategy::SWAR:     return "S";
        case Strategy::Loop:     return "L";
        case Strategy::Generic:  return "G";
        default:                 return "?";
    }
}

constexpr std::optional<Strategy> strategy_from_char(char c) noexcept {
    switch (c) {
        case 'U': case 'u': return Strategy::Unrolled;
        case 'S': case 's': return Strategy::SWAR;
        case 'L': case 'l': return Strategy::Loop;
        case 'G': case 'g': return Strategy::Generic;
        default:            return std::nullopt;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  PerfSample — one timed run: tail latency + 6 hardware counter values
//
//  Counter slot assignment (matches perf_counter.h group layout):
//    [0] instructions
//    [1] cycles
//    [2] cache-references
//    [3] cache-misses
//    [4] branch-instructions
//    [5] branch-misses
//
//  p99_ns is nanoseconds, converted from RDTSC cycles by the measurement
//  harness using a pre-calibrated cycles-per-ns ratio.
//  A value of 0.0 indicates a measurement failure / not-yet-populated slot.
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr int NUM_COUNTERS = 6;

// Counter slot assignment (aligns with config_metrics per-parse fields):
//   [0] instructions     — instructions per parse
//   [1] cycles           — TSC cycles per parse
//   [2] ipc              — instructions per cycle
//   [3] l1d_misses       — L1D read misses per parse
//   [4] branch_miss_rate — branch miss rate (0.0–1.0)
//   [5] cache_miss_rate  — LL cache miss rate (0.0–1.0)
// All values are per-parse averages (doubles), matching config_metrics output.
// Zero-filled until populated by the measurement harness (fix_bench.h).
struct PerfSample {
    double                          p99_ns{0.0};
    std::array<double, NUM_COUNTERS> counters{};

    [[nodiscard]] constexpr bool valid() const noexcept { return p99_ns > 0.0; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  RunTriple — three PerfSamples for the A1/A2/B measurement protocol
// ─────────────────────────────────────────────────────────────────────────────

struct RunTriple {
    PerfSample a1{};   // first back-to-back run
    PerfSample a2{};   // second back-to-back run
    PerfSample b{};    // post-break baseline run

    [[nodiscard]] constexpr bool all_valid() const noexcept {
        return a1.valid() && a2.valid() && b.valid();
    }

    // Ratio |a1-a2| / min(a1,a2) — primary instability metric.
    // Returns +inf if either run is invalid.
    [[nodiscard]] double instability_ratio() const noexcept {
        if (!a1.valid() || !a2.valid()) return std::numeric_limits<double>::infinity();
        const double lo = std::min(a1.p99_ns, a2.p99_ns);
        if (lo == 0.0) return std::numeric_limits<double>::infinity();
        return std::abs(a1.p99_ns - a2.p99_ns) / lo;
    }

    // Best-of-three p99: min over all valid runs (conservative; avoids warm-cache
    // bias because B follows a break so an artificially low B is unlikely).
    [[nodiscard]] double best_p99_ns() const noexcept {
        double v = std::numeric_limits<double>::infinity();
        if (a1.valid()) v = std::min(v, a1.p99_ns);
        if (a2.valid()) v = std::min(v, a2.p99_ns);
        if (b.valid())  v = std::min(v, b.p99_ns);
        return v == std::numeric_limits<double>::infinity() ? 0.0 : v;
    }

    // Mean of A1+A2 p99 (used as Stage 1 training target; B is held out for
    // warm-vs-cold analysis).
    [[nodiscard]] double mean_ab_p99_ns() const noexcept {
        if (!a1.valid() || !a2.valid()) return 0.0;
        return 0.5 * (a1.p99_ns + a2.p99_ns);
    }

    // Mean counter vector over A1+A2 — Stage 2 features.
    [[nodiscard]] std::array<double, NUM_COUNTERS> mean_ab_counters() const noexcept {
        std::array<double, NUM_COUNTERS> out{};
        for (int i = 0; i < NUM_COUNTERS; ++i)
            out[i] = 0.5 * (a1.counters[i] + a2.counters[i]);
        return out;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  DataPointStatus — measurement quality classification
// ─────────────────────────────────────────────────────────────────────────────

enum class DataPointStatus : uint8_t {
    Ok          = 0,  // instability_ratio ≤ SOFT_THRESH
    Suspicious  = 1,  // SOFT_THRESH < ratio ≤ HARD_THRESH
    HardDeleted = 2   // ratio > HARD_THRESH or measurement failure
};

inline constexpr double SOFT_THRESH = 0.05;  // 5%  — suspect boundary
inline constexpr double HARD_THRESH = 0.20;  // 20% — hard-delete boundary

[[nodiscard]] inline DataPointStatus classify(const RunTriple& r) noexcept {
    if (!r.all_valid()) return DataPointStatus::HardDeleted;
    const double ratio = r.instability_ratio();
    if (ratio <= SOFT_THRESH) return DataPointStatus::Ok;
    if (ratio <= HARD_THRESH) return DataPointStatus::Suspicious;
    return DataPointStatus::HardDeleted;
}

// ─────────────────────────────────────────────────────────────────────────────
//  DataPoint<N> — the central calibrator record
//
//  Template parameter N: number of FIX fields in the schema being optimised.
//    N=4  — Trivial tier (BeginString, BodyLength, MsgType, CheckSum)
//    N=12 — Full NewOrderSingle tier (default for production calibration)
//
//  plan_id: packed uint32 encoding strategy[0..N) as base-4 digits.
//    plan_id = Σ strategy[i] * 4^i
//  This gives a unique dense index for N ≤ 16 (4^16 = 4G, fits uint32).
//
//  fold_id: 0..4, assigned by the corpus builder for 5-fold CV.
//    Constant once assigned; the SVR / counter_preprocessor.h training loop
//    iterates over folds using fold_id as the hold-out selector.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
    requires (N >= 1 && N <= 16)
struct DataPoint {
    // ── Identity ──────────────────────────────────────────────────────────────
    std::array<Strategy, N> plan{};      // per-field strategy selection
    uint32_t                plan_id{0};  // packed plan index (see above)
    uint8_t                 fold_id{0};  // 5-fold CV assignment [0,4]

    // ── Measurements ─────────────────────────────────────────────────────────
    RunTriple               runs{};
    DataPointStatus         status{DataPointStatus::Ok};

    // ── Construction helpers ──────────────────────────────────────────────────

    // Build plan_id from the plan array; call after constructing the plan.
    constexpr void update_plan_id() noexcept {
        uint32_t id = 0;
        uint32_t base = 1;
        for (int i = 0; i < N; ++i) {
            id += static_cast<uint32_t>(plan[i]) * base;
            base *= 4;
        }
        plan_id = id;
    }

    // Populate status from the run triple; call after all three runs are filled.
    void update_status() noexcept { status = classify(runs); }

    // ── Predicates ───────────────────────────────────────────────────────────

    [[nodiscard]] constexpr bool is_ok()           const noexcept { return status == DataPointStatus::Ok; }
    [[nodiscard]] constexpr bool is_suspicious()   const noexcept { return status == DataPointStatus::Suspicious; }
    [[nodiscard]] constexpr bool is_hard_deleted() const noexcept { return status == DataPointStatus::HardDeleted; }
    [[nodiscard]] constexpr bool is_usable()       const noexcept { return status == DataPointStatus::Ok; }

    // ── Latency accessors ────────────────────────────────────────────────────

    [[nodiscard]] double latency_estimate() const noexcept { return runs.mean_ab_p99_ns(); }
    [[nodiscard]] double best_latency()     const noexcept { return runs.best_p99_ns(); }

    // ── Feature encoding: one-hot strategy ───────────────────────────────────
    //
    //  Output layout (row-major):
    //    [field_0 × 4 bits | field_1 × 4 bits | … | field_{N-1} × 4 bits]
    //  Total = N * NUM_STRATEGIES floats.
    //
    //  Usage:
    //    std::array<float, 4*N> feats;
    //    dp.write_strategy_features(feats);

    static constexpr int STRATEGY_FEATURE_DIM = N * NUM_STRATEGIES;

    void write_strategy_features(std::span<float> out) const noexcept {
        assert(static_cast<int>(out.size()) >= STRATEGY_FEATURE_DIM);
        for (int fi = 0; fi < N; ++fi) {
            const int base = fi * NUM_STRATEGIES;
            for (int si = 0; si < NUM_STRATEGIES; ++si) {
                out[base + si] = (static_cast<int>(plan[fi]) == si) ? 1.0f : 0.0f;
            }
        }
    }

    [[nodiscard]] std::array<float, N * NUM_STRATEGIES> strategy_features() const noexcept {
        std::array<float, N * NUM_STRATEGIES> out{};
        write_strategy_features(out);
        return out;
    }

    // ── Feature encoding: counter features (Stage 2) ─────────────────────────
    //
    //  Returns mean A1/A2 counters as double[NUM_COUNTERS].
    //  These are the features for Stage 2 of the calibrator (SVR trained on
    //  measured counter vectors only, not on one-hot strategy encoding).

    [[nodiscard]] std::array<double, NUM_COUNTERS> counter_features() const noexcept {
        return runs.mean_ab_counters();
    }

    // ── Plan string  (e.g. "USGL" for N=4) ───────────────────────────────────

    [[nodiscard]] std::string plan_string() const {
        std::string s;
        s.reserve(N);
        for (int i = 0; i < N; ++i) s += strategy_char(plan[i]);
        return s;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience aliases
// ─────────────────────────────────────────────────────────────────────────────

using TrivialDataPoint = DataPoint<4>;   // BeginString/BodyLength/MsgType/CheckSum
using FullDataPoint    = DataPoint<12>;  // NewOrderSingle (12 required fields)

// ─────────────────────────────────────────────────────────────────────────────
//  make_data_point — factory from a plan string + fold assignment
//
//  plan_str: exactly N characters from {'U','S','L','G'} (case-insensitive)
//  Returns nullopt if the string is invalid length or contains unknown chars.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] std::optional<DataPoint<N>>
make_data_point(std::string_view plan_str, uint8_t fold_id = 0) noexcept {
    if (static_cast<int>(plan_str.size()) != N) return std::nullopt;
    DataPoint<N> dp{};
    for (int i = 0; i < N; ++i) {
        auto s = strategy_from_char(plan_str[i]);
        if (!s) return std::nullopt;
        dp.plan[i] = *s;
    }
    dp.fold_id = fold_id;
    dp.update_plan_id();
    return dp;
}

// ─────────────────────────────────────────────────────────────────────────────
//  plan_id_to_plan — inverse of update_plan_id; reconstructs strategy array
//  from a packed plan_id for a given N.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] constexpr std::array<Strategy, N>
plan_id_to_plan(uint32_t id) noexcept {
    std::array<Strategy, N> plan{};
    for (int i = 0; i < N; ++i) {
        plan[i] = static_cast<Strategy>(id % 4);
        id /= 4;
    }
    return plan;
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_DATA_POINT_H
