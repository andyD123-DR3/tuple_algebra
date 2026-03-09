#ifndef CTDP_FIX_EXPERIMENT_COMPILED_MEASURER_H
#define CTDP_FIX_EXPERIMENT_COMPILED_MEASURER_H

// experiments/fix_calibration/common/compiled_measurer.h
//
// Compile-time dispatch table for ET-instantiated FIX parser measurement.
//
// The plan IS a type: each fix_config is a distinct template instantiation.
// Runtime dispatch is forbidden — the whole point of the CT-DP framework
// is that the compiler sees the entire 12-field parse chain as a single
// function, enabling cross-field ILP.
//
// Two-phase usage:
//   Phase 1 (discovery): compiled_measurer measures the training pool
//     (200 configs, fixed seed, known at compile time).
//   Phase 2 (verification): compiled_measurer measures training pool
//     + beam-search winners from generated/X_candidates.h.
//
// Supports exactly one or two constexpr config arrays (pools).
// Phase 1 uses a single pool; Phase 2 uses two (training + candidates).
// Duplicate configs across pools are allowed (the first match wins).
//
// The measurement function per config is provided by a MeasureAdapter
// policy that wraps the real RDTSC infrastructure:
//
//   struct my_adapter {
//       // Runtime state: message pool, measurement params, etc.
//       std::vector<std::string> const* messages;
//       fix::measurement_config const* config;
//
//       // Per-config measurement — called through the dispatch table.
//       template<fix::fix_config Cfg>
//       measurement_result measure_one() const;
//   };
//
// The dispatch table is built at compile time.  Lookup is O(N) linear
// scan over the merged table — acceptable for N ≤ 300 total configs.
// A hash map is unnecessary complexity for this scale.
//
// Dependencies: baselines.h (measurement_result), fix_et_parser.h
// C++20

#include "baselines.h"

#include <ctdp/calibrator/fix_et_parser.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ctdp::fix_experiment {

namespace fix = ctdp::calibrator::fix;

// ─────────────────────────────────────────────────────────────────────
//  config_key — packed 24-bit structural identifier for dispatch lookup
// ─────────────────────────────────────────────────────────────────────

namespace compiled_detail {

// Strategy values must fit in 2 bits (0..3) for the packed key encoding.
static_assert(fix::NUM_STRATEGIES <= 4,
    "config_key assumes Strategy fits in 2 bits");
static_assert(fix::num_fields * 2 <= 32,
    "config_key assumes num_fields * 2 bits fit in uint32_t");

inline constexpr std::uint32_t config_key(const fix::fix_config& c) noexcept {
    std::uint32_t k = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(fix::num_fields); ++i)
        k |= (static_cast<std::uint32_t>(c[i]) & 0x3u) << (2u * i);
    return k;
}

} // namespace compiled_detail

// ─────────────────────────────────────────────────────────────────────
//  dispatch_entry — one row of the dispatch table
//
//  The function pointer takes a type-erased adapter pointer and returns
//  measurement_result.  The template instantiation for each Config
//  casts back to the concrete adapter type.
// ─────────────────────────────────────────────────────────────────────

struct dispatch_entry {
    std::uint32_t key;
    measurement_result (*measure_fn)(const void* adapter);
};

// ─────────────────────────────────────────────────────────────────────
//  make_dispatch_table — build from a constexpr config array
//
//  Returns a std::array of dispatch_entry, one per config.
//  MeasureAdapter must provide:
//    template<fix::fix_config Cfg>
//    measurement_result measure_one() const;
// ─────────────────────────────────────────────────────────────────────

namespace compiled_detail {

template<typename MeasureAdapter, fix::fix_config Cfg>
measurement_result dispatch_thunk(const void* adapter_ptr) {
    auto* adapter = static_cast<const MeasureAdapter*>(adapter_ptr);
    return adapter->template measure_one<Cfg>();
}

template<typename MeasureAdapter, auto const& Configs, std::size_t... Is>
constexpr auto make_table_impl(std::index_sequence<Is...>) {
    return std::array<dispatch_entry, sizeof...(Is)>{{
        dispatch_entry{
            config_key(Configs[Is]),
            &dispatch_thunk<MeasureAdapter, Configs[Is]>
        }...
    }};
}

template<typename MeasureAdapter, auto const& Configs>
constexpr auto make_dispatch_table() {
    return make_table_impl<MeasureAdapter, Configs>(
        std::make_index_sequence<Configs.size()>{});
}

} // namespace compiled_detail

// ─────────────────────────────────────────────────────────────────────
//  compiled_measurer — dispatch table over one or two config arrays
//
//  Single-pool usage (Phase 1):
//    compiled_measurer<MyAdapter, train_pool> m{adapter};
//
//  Dual-pool usage (Phase 2):
//    compiled_measurer<MyAdapter, train_pool, cand_pool> m{adapter};
//
//  Both satisfy the Measurer concept:
//    measurement_result m.measure(fix_config const&) const;
//
//  Throws std::out_of_range if the config is not in any table.
// ─────────────────────────────────────────────────────────────────────

template<typename MeasureAdapter, auto const& Pool1>
class compiled_measurer_1 {
    static constexpr auto table1_ =
        compiled_detail::make_dispatch_table<MeasureAdapter, Pool1>();

    MeasureAdapter adapter_;

public:
    explicit compiled_measurer_1(MeasureAdapter adapter)
        : adapter_(std::move(adapter)) {}

    [[nodiscard]] measurement_result measure(
        const fix::fix_config& cfg) const
    {
        auto key = compiled_detail::config_key(cfg);
        for (const auto& e : table1_) {
            if (e.key == key) return e.measure_fn(&adapter_);
        }
        throw std::out_of_range(
            "compiled_measurer: config not in dispatch table: "
            + fix::config_to_string(cfg));
    }

    static constexpr std::size_t table_size() { return table1_.size(); }
};

template<typename MeasureAdapter, auto const& Pool1, auto const& Pool2>
class compiled_measurer_2 {
    static constexpr auto table1_ =
        compiled_detail::make_dispatch_table<MeasureAdapter, Pool1>();
    static constexpr auto table2_ =
        compiled_detail::make_dispatch_table<MeasureAdapter, Pool2>();

    MeasureAdapter adapter_;

public:
    explicit compiled_measurer_2(MeasureAdapter adapter)
        : adapter_(std::move(adapter)) {}

    [[nodiscard]] measurement_result measure(
        const fix::fix_config& cfg) const
    {
        auto key = compiled_detail::config_key(cfg);
        for (const auto& e : table1_) {
            if (e.key == key) return e.measure_fn(&adapter_);
        }
        for (const auto& e : table2_) {
            if (e.key == key) return e.measure_fn(&adapter_);
        }
        throw std::out_of_range(
            "compiled_measurer: config not in dispatch table: "
            + fix::config_to_string(cfg));
    }

    static constexpr std::size_t table_size() {
        return table1_.size() + table2_.size();
    }
};

// ─────────────────────────────────────────────────────────────────────
//  Convenience alias: compiled_measurer<Adapter, Pool1 [, Pool2]>
//
//  Exactly one or two pools.  Three or more is not implemented.
//
//  Usage:
//    using measurer_t = compiled_measurer<my_adapter, train_pool>;
//    using measurer_t = compiled_measurer<my_adapter, train_pool, cands>;
// ─────────────────────────────────────────────────────────────────────

template<typename MeasureAdapter, auto const& Pool1>
using compiled_measurer_single = compiled_measurer_1<MeasureAdapter, Pool1>;

template<typename MeasureAdapter, auto const& Pool1, auto const& Pool2>
using compiled_measurer_dual = compiled_measurer_2<MeasureAdapter, Pool1, Pool2>;

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_COMPILED_MEASURER_H
