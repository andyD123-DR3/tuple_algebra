// tests/bench/test_perf_counter.cpp
//
// Diagnostic unit tests for ctdp::bench perf_counter primitives.
//
// These tests exist specifically to catch the class of bug where
// rdtsc_start/end, thread_cycle_count, or perf_counter_group::snapshot()
// silently return zero on a given compiler/platform combination (e.g. MinGW
// using <intrin.h> instead of <x86intrin.h>).
//
// Each test prints a diagnostic value so failures are easy to interpret in CI.
//
// Platform behaviour:
//   Linux  : Tier 0 (RDTSC) + Tier 1 (perf_event_open) if paranoid <= 1
//   Windows/MSVC  : Tier 0 (RDTSC + QueryThreadCycleTime), Tier 1 unavailable
//   Windows/MinGW : same as MSVC if <x86intrin.h> include is correct

#include <ctdp/bench/perf_counter.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <thread>
#include <chrono>

namespace bench = ctdp::bench;

// ─────────────────────────────────────────────────────────────────────────────
//  Tier 0: RDTSC
// ─────────────────────────────────────────────────────────────────────────────

TEST(PerfCounterTier0, RdtscStartReturnsNonZero) {
    auto t = bench::rdtsc_start();
    EXPECT_GT(t, std::uint64_t{0})
        << "rdtsc_start() returned 0 -- likely missing x86intrin.h on MinGW "
           "or non-x86 build";
}

TEST(PerfCounterTier0, RdtscEndReturnsNonZero) {
    auto t = bench::rdtsc_end();
    EXPECT_GT(t, std::uint64_t{0})
        << "rdtsc_end() returned 0";
}

TEST(PerfCounterTier0, RdtscMonotonicallyIncreasing) {
    auto t0 = bench::rdtsc_start();
    // do a tiny bit of work so the counter advances
    volatile std::uint64_t sink = 0;
    for (int i = 0; i < 1000; ++i) sink += static_cast<std::uint64_t>(i);
    auto t1 = bench::rdtsc_end();
    (void)sink;
    EXPECT_GT(t1, t0)
        << "rdtsc_end() <= rdtsc_start() -- TSC is not monotone or both returned 0";
}

TEST(PerfCounterTier0, RdtscDeltaReasonable) {
    // Measure a 1ms sleep; expect ~1e6..1e10 raw TSC ticks (covers 1 MHz..10 GHz)
    auto t0 = bench::rdtsc_start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto t1 = bench::rdtsc_end();
    auto delta = t1 - t0;
    EXPECT_GT(delta, std::uint64_t{1'000'000})
        << "TSC delta over 1ms sleep was " << delta << " -- suspiciously small";
    EXPECT_LT(delta, std::uint64_t{10'000'000'000ULL})
        << "TSC delta over 1ms sleep was " << delta << " -- suspiciously large";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tier 0: Thread CPU time
// ─────────────────────────────────────────────────────────────────────────────

TEST(PerfCounterTier0, ThreadCpuTimeNsNonNegative) {
    double t = bench::thread_cpu_time_ns();
    EXPECT_GE(t, 0.0)
        << "thread_cpu_time_ns() returned negative value: " << t;
}

TEST(PerfCounterTier0, ThreadCpuTimeNsAdvances) {
    double t0 = bench::thread_cpu_time_ns();
    volatile std::uint64_t sink = 0;
    for (int i = 0; i < 100'000; ++i) sink += static_cast<std::uint64_t>(i);
    double t1 = bench::thread_cpu_time_ns();
    (void)sink;
    EXPECT_GT(t1, t0)
        << "thread_cpu_time_ns() did not advance after busy work "
           "(t0=" << t0 << " t1=" << t1 << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tier 0: Windows QueryThreadCycleTime
// ─────────────────────────────────────────────────────────────────────────────

#if defined(_WIN32)
TEST(PerfCounterTier0Windows, ThreadCycleCountNonZero) {
    auto c = bench::thread_cycle_count();
    EXPECT_GT(c, std::uint64_t{0})
        << "thread_cycle_count() (QueryThreadCycleTime) returned 0";
}

TEST(PerfCounterTier0Windows, ThreadCycleCountAdvances) {
    auto c0 = bench::thread_cycle_count();
    volatile std::uint64_t sink = 0;
    for (int i = 0; i < 100'000; ++i) sink += static_cast<std::uint64_t>(i);
    auto c1 = bench::thread_cycle_count();
    (void)sink;
    EXPECT_GT(c1, c0)
        << "thread_cycle_count() did not advance (c0=" << c0 << " c1=" << c1 << ")";
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  perf_counter_group: snapshot tsc_cycles
// ─────────────────────────────────────────────────────────────────────────────

TEST(PerfCounterGroup, SnapshotTscCyclesNonZero) {
    bench::perf_counter_group ctr;
    ctr.start();
    volatile std::uint64_t sink = 0;
    for (int i = 0; i < 100'000; ++i) sink += static_cast<std::uint64_t>(i);
    ctr.stop();
    (void)sink;
    auto snap = ctr.snapshot();
    EXPECT_GT(snap.tsc_cycles, std::uint64_t{0})
        << "snapshot().tsc_cycles == 0 after busy work -- "
           "RDTSC or QueryThreadCycleTime not working on this platform/compiler";
}

TEST(PerfCounterGroup, SnapshotTscCyclesReasonable) {
    // Run ~50000 iterations of simple arithmetic; expect 1k..100M cycles
    bench::perf_counter_group ctr;
    ctr.start();
    volatile std::uint64_t sink = 0;
    for (int i = 0; i < 50'000; ++i) sink += static_cast<std::uint64_t>(i);
    ctr.stop();
    (void)sink;
    auto snap = ctr.snapshot();
    EXPECT_GT(snap.tsc_cycles, std::uint64_t{1'000})
        << "tsc_cycles=" << snap.tsc_cycles << " suspiciously small";
    EXPECT_LT(snap.tsc_cycles, std::uint64_t{100'000'000})
        << "tsc_cycles=" << snap.tsc_cycles << " suspiciously large";
}

TEST(PerfCounterGroup, MultipleStartStopGiveConsistentResults) {
    bench::perf_counter_group ctr;

    volatile std::uint64_t sink = 0;

    ctr.start();
    for (int i = 0; i < 10'000; ++i) sink += static_cast<std::uint64_t>(i);
    ctr.stop();
    auto snap1 = ctr.snapshot();

    ctr.start();
    for (int i = 0; i < 10'000; ++i) sink += static_cast<std::uint64_t>(i);
    ctr.stop();
    auto snap2 = ctr.snapshot();

    (void)sink;

    // Both should be non-zero
    EXPECT_GT(snap1.tsc_cycles, std::uint64_t{0}) << "First measurement zero";
    EXPECT_GT(snap2.tsc_cycles, std::uint64_t{0}) << "Second measurement zero";

    // Neither should be wildly different from the other (within 100x)
    double ratio = snap1.tsc_cycles > snap2.tsc_cycles
        ? static_cast<double>(snap1.tsc_cycles) / static_cast<double>(snap2.tsc_cycles)
        : static_cast<double>(snap2.tsc_cycles) / static_cast<double>(snap1.tsc_cycles);
    EXPECT_LT(ratio, 100.0)
        << "Consecutive identical workloads gave wildly different cycle counts: "
        << snap1.tsc_cycles << " vs " << snap2.tsc_cycles;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tier 1 availability (informational — not a failure if unavailable)
// ─────────────────────────────────────────────────────────────────────────────

TEST(PerfCounterGroup, Tier1AvailabilityReported) {
    bench::perf_counter_group ctr;
    // Just confirm the flag is readable without crashing
    bool avail = ctr.tier1_available();
    // On Windows this should always be false (no perf_event_open)
    // On Linux it depends on kernel.perf_event_paranoid
#if defined(_WIN32)
    EXPECT_FALSE(avail)
        << "tier1_available() returned true on Windows -- unexpected";
#else
    // On Linux: either true (counters available) or false (restricted kernel)
    // We just print which case we're in
    if (avail) {
        SUCCEED() << "Tier 1 hardware counters available (perf_event_open succeeded)";
    } else {
        SUCCEED() << "Tier 1 not available (kernel.perf_event_paranoid too high "
                     "or missing CAP_PERFMON) -- run: "
                     "sudo sysctl -w kernel.perf_event_paranoid=1";
    }
#endif
}

TEST(PerfCounterGroup, Tier1InstructionCountNonZeroWhenAvailable) {
    bench::perf_counter_group ctr;
    if (!ctr.tier1_available()) {
        GTEST_SKIP() << "Tier 1 not available on this platform/config";
    }
    ctr.start();
    volatile std::uint64_t sink = 0;
    for (int i = 0; i < 50'000; ++i) sink += static_cast<std::uint64_t>(i);
    ctr.stop();
    (void)sink;
    auto snap = ctr.snapshot();
    EXPECT_GT(snap.instructions, std::uint64_t{0})
        << "instructions == 0 despite tier1_available() == true";
    EXPECT_GT(snap.instructions, std::uint64_t{10'000})
        << "instructions=" << snap.instructions << " suspiciously small for 50k-iter loop";
}
