// experiments/fix_calibration/tests/test_compiled_measurer.cpp
//
// Unit tests for compiled_measurer.h dispatch table infrastructure.
//
// Uses a trivial MeasureAdapter that returns a measurement_result
// derived from the config_key, so we can verify dispatch correctness
// without any real RDTSC or parser infrastructure.

#include "compiled_measurer.h"
#include "experiment_config.h"  // training_pool

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fxe = ctdp::fix_experiment;
namespace fix = ctdp::calibrator::fix;

// =================================================================
// Trivial adapter: returns config_key as p50, config_key * 1.5 as p99
// =================================================================

struct trivial_adapter {
    template<fix::fix_config Cfg>
    fxe::measurement_result measure_one() const {
        auto key = static_cast<double>(
            fxe::compiled_detail::config_key(Cfg));
        return {key, key * 1.5};
    }
};

// =================================================================
// Small test pools
// =================================================================

inline constexpr auto small_pool_1 = std::array<fix::fix_config, 3>{{
    fix::all_unrolled,
    fix::all_swar,
    fix::all_loop,
}};

inline constexpr auto small_pool_2 = std::array<fix::fix_config, 2>{{
    fix::all_generic,
    // A mixed config: UUSLSUUSUUUU
    {fix::Strategy::Unrolled, fix::Strategy::Unrolled,
     fix::Strategy::SWAR,     fix::Strategy::Loop,
     fix::Strategy::SWAR,     fix::Strategy::Unrolled,
     fix::Strategy::Unrolled, fix::Strategy::SWAR,
     fix::Strategy::Unrolled, fix::Strategy::Unrolled,
     fix::Strategy::Unrolled, fix::Strategy::Unrolled},
}};

// Pool with a deliberate duplicate of all_unrolled (also in pool_1)
inline constexpr auto pool_with_dup = std::array<fix::fix_config, 2>{{
    fix::all_unrolled,  // duplicate of pool_1[0]
    fix::all_generic,
}};

// =================================================================
// Single-pool dispatch
// =================================================================

TEST(CompiledMeasurer, SinglePool_HitAllEntries) {
    fxe::compiled_measurer_single<trivial_adapter, small_pool_1>
        m{trivial_adapter{}};

    auto r0 = m.measure(fix::all_unrolled);
    auto r1 = m.measure(fix::all_swar);
    auto r2 = m.measure(fix::all_loop);

    // all_unrolled has key 0 -> p50 = 0.0
    EXPECT_DOUBLE_EQ(r0.p50_ns, 0.0);
    EXPECT_DOUBLE_EQ(r0.p99_ns, 0.0);

    // all_swar and all_loop should have non-zero keys
    EXPECT_GT(r1.p50_ns, 0.0);
    EXPECT_GT(r2.p50_ns, 0.0);

    // p99 = p50 * 1.5 (from adapter)
    EXPECT_DOUBLE_EQ(r1.p99_ns, r1.p50_ns * 1.5);
    EXPECT_DOUBLE_EQ(r2.p99_ns, r2.p50_ns * 1.5);
}

TEST(CompiledMeasurer, SinglePool_MissThrows) {
    fxe::compiled_measurer_single<trivial_adapter, small_pool_1>
        m{trivial_adapter{}};

    // all_generic is NOT in small_pool_1
    EXPECT_THROW(m.measure(fix::all_generic), std::out_of_range);
}

TEST(CompiledMeasurer, SinglePool_TableSize) {
    using M = fxe::compiled_measurer_single<trivial_adapter, small_pool_1>;
    EXPECT_EQ(M::table_size(), 3u);
}

// =================================================================
// Dual-pool dispatch
// =================================================================

TEST(CompiledMeasurer, DualPool_HitInPool1) {
    fxe::compiled_measurer_dual<trivial_adapter, small_pool_1, small_pool_2>
        m{trivial_adapter{}};

    auto r = m.measure(fix::all_swar);
    EXPECT_GT(r.p50_ns, 0.0);
    EXPECT_DOUBLE_EQ(r.p99_ns, r.p50_ns * 1.5);
}

TEST(CompiledMeasurer, DualPool_HitInPool2) {
    fxe::compiled_measurer_dual<trivial_adapter, small_pool_1, small_pool_2>
        m{trivial_adapter{}};

    auto r = m.measure(fix::all_generic);
    EXPECT_GT(r.p50_ns, 0.0);
    EXPECT_DOUBLE_EQ(r.p99_ns, r.p50_ns * 1.5);
}

TEST(CompiledMeasurer, DualPool_MissThrows) {
    // A config in neither pool
    fix::fix_config exotic = {
        fix::Strategy::Generic,  fix::Strategy::Loop,
        fix::Strategy::Generic,  fix::Strategy::Loop,
        fix::Strategy::Generic,  fix::Strategy::Loop,
        fix::Strategy::Generic,  fix::Strategy::Loop,
        fix::Strategy::Generic,  fix::Strategy::Loop,
        fix::Strategy::Generic,  fix::Strategy::Loop,
    };

    fxe::compiled_measurer_dual<trivial_adapter, small_pool_1, small_pool_2>
        m{trivial_adapter{}};

    EXPECT_THROW(m.measure(exotic), std::out_of_range);
}

TEST(CompiledMeasurer, DualPool_TableSize) {
    using M = fxe::compiled_measurer_dual<trivial_adapter, small_pool_1, small_pool_2>;
    EXPECT_EQ(M::table_size(), 5u);  // 3 + 2
}

// =================================================================
// Duplicate config across pools: first match wins
// =================================================================

TEST(CompiledMeasurer, DualPool_DuplicateFirstMatchWins) {
    // all_unrolled is in both small_pool_1 and pool_with_dup.
    // Both use the same trivial_adapter, so the result is the same.
    // The key test is that it does not throw or double-count.
    fxe::compiled_measurer_dual<trivial_adapter, small_pool_1, pool_with_dup>
        m{trivial_adapter{}};

    auto r = m.measure(fix::all_unrolled);
    EXPECT_DOUBLE_EQ(r.p50_ns, 0.0);  // key = 0

    // all_generic is only in pool_with_dup (pool 2)
    auto r2 = m.measure(fix::all_generic);
    EXPECT_GT(r2.p50_ns, 0.0);
}

// =================================================================
// config_key structural tests
// =================================================================

TEST(CompiledMeasurerKey, AllUnrolledIsZero) {
    EXPECT_EQ(fxe::compiled_detail::config_key(fix::all_unrolled), 0u);
}

TEST(CompiledMeasurerKey, AllGenericIsMaxPattern) {
    // Strategy::Generic = 3, all 12 fields: 0x00FFFFFF
    EXPECT_EQ(fxe::compiled_detail::config_key(fix::all_generic), 0x00FFFFFFu);
}

TEST(CompiledMeasurerKey, DifferentConfigsDifferentKeys) {
    auto k1 = fxe::compiled_detail::config_key(fix::all_unrolled);
    auto k2 = fxe::compiled_detail::config_key(fix::all_swar);
    auto k3 = fxe::compiled_detail::config_key(fix::all_loop);
    auto k4 = fxe::compiled_detail::config_key(fix::all_generic);
    EXPECT_NE(k1, k2);
    EXPECT_NE(k1, k3);
    EXPECT_NE(k1, k4);
    EXPECT_NE(k2, k3);
    EXPECT_NE(k2, k4);
    EXPECT_NE(k3, k4);
}

// =================================================================
// Large pool: training_pool from experiment_config.h
// =================================================================

TEST(CompiledMeasurer, TrainingPoolDispatch) {
    // Verifies that compiled_measurer can be instantiated over the
    // full 200-config training_pool without hitting compile limits.
    fxe::compiled_measurer_single<trivial_adapter, fxe::training_pool>
        m{trivial_adapter{}};

    EXPECT_EQ(m.table_size(), 200u);

    // Measure the first and last entries — should not throw
    auto r0 = m.measure(fxe::training_pool[0]);
    auto r199 = m.measure(fxe::training_pool[199]);
    EXPECT_DOUBLE_EQ(r0.p99_ns, r0.p50_ns * 1.5);
    EXPECT_DOUBLE_EQ(r199.p99_ns, r199.p50_ns * 1.5);
}
