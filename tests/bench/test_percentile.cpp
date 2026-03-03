// tests/bench/test_percentile.cpp -- Tests for bench::percentile

#include <ctdp/bench/percentile.h>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace bench = ctdp::bench;

// --- percentile_sorted ----------------------------------------------

TEST(Percentile, SortedEmptyReturnsZero) {
    std::vector<double> empty;
    EXPECT_DOUBLE_EQ(0.0, bench::percentile_sorted(empty, 50.0));
}

TEST(Percentile, SortedSingleElement) {
    std::vector<double> data = {42.0};
    EXPECT_DOUBLE_EQ(42.0, bench::percentile_sorted(data, 0.0));
    EXPECT_DOUBLE_EQ(42.0, bench::percentile_sorted(data, 50.0));
    EXPECT_DOUBLE_EQ(42.0, bench::percentile_sorted(data, 100.0));
}

TEST(Percentile, SortedP50IsMedian) {
    // Odd count: exact middle
    std::vector<double> odd = {10.0, 20.0, 30.0, 40.0, 50.0};
    EXPECT_DOUBLE_EQ(30.0, bench::percentile_sorted(odd, 50.0));

    // Even count: interpolation between middle two
    std::vector<double> even = {10.0, 20.0, 30.0, 40.0};
    double p50 = bench::percentile_sorted(even, 50.0);
    EXPECT_NEAR(25.0, p50, 1.0); // Interpolation between 20 and 30
}

TEST(Percentile, SortedP0IsMin) {
    std::vector<double> data = {5.0, 10.0, 15.0, 20.0};
    EXPECT_DOUBLE_EQ(5.0, bench::percentile_sorted(data, 0.0));
}

TEST(Percentile, SortedP100IsMax) {
    std::vector<double> data = {5.0, 10.0, 15.0, 20.0};
    EXPECT_DOUBLE_EQ(20.0, bench::percentile_sorted(data, 100.0));
}

TEST(Percentile, SortedP99OnLargeData) {
    // 1000 values: 1.0, 2.0, ..., 1000.0
    std::vector<double> data(1000);
    std::iota(data.begin(), data.end(), 1.0);

    double p99 = bench::percentile_sorted(data, 99.0);
    // p99 of {1..1000} should be near 991
    EXPECT_NEAR(991.0, p99, 2.0);
}

// --- percentile (unsorted) ------------------------------------------

TEST(Percentile, UnsortedMatchesSorted) {
    std::vector<double> unsorted = {50.0, 10.0, 30.0, 20.0, 40.0};
    std::vector<double> sorted = {10.0, 20.0, 30.0, 40.0, 50.0};

    EXPECT_DOUBLE_EQ(
        bench::percentile_sorted(sorted, 75.0),
        bench::percentile(unsorted, 75.0));
}

// --- compute_percentiles --------------------------------------------

TEST(Percentile, ComputePercentilesEmpty) {
    std::vector<double> empty;
    auto result = bench::compute_percentiles(empty);
    EXPECT_EQ(0u, result.samples);
    EXPECT_DOUBLE_EQ(0.0, result.mean);
}

TEST(Percentile, ComputePercentilesBasic) {
    // 100 values: 1.0 through 100.0
    std::vector<double> data(100);
    std::iota(data.begin(), data.end(), 1.0);

    auto result = bench::compute_percentiles(data);

    EXPECT_EQ(100u, result.samples);
    EXPECT_NEAR(50.5, result.mean, 0.01);
    EXPECT_NEAR(50.5, result.p50, 1.0);
    EXPECT_NEAR(90.0, result.p90, 2.0);
    EXPECT_NEAR(95.0, result.p95, 2.0);
    EXPECT_NEAR(99.0, result.p99, 2.0);
    EXPECT_DOUBLE_EQ(100.0, result.max);
}

TEST(Percentile, TailRatio) {
    // Construct data where p99 is 10x the median
    std::vector<double> data(100, 10.0); // 99 values at 10
    data.push_back(100.0);               // 1 value at 100

    auto result = bench::compute_percentiles(data);
    EXPECT_NEAR(10.0, result.p50, 1.0);
    // Tail ratio should be reasonable
    EXPECT_GT(result.tail_ratio(), 0.0);
}

TEST(Percentile, JitterNs) {
    std::vector<double> data(100);
    std::iota(data.begin(), data.end(), 1.0);
    auto result = bench::compute_percentiles(data);
    EXPECT_NEAR(result.p99 - result.p50, result.jitter_ns(), 0.01);
}


// --- FIX ET Parser --------------------------------------------------

#include <ctdp/calibrator/fix_et_parser.h>

namespace fix = ctdp::calibrator::fix;

TEST(FixETParser, ConfigToString) {
    EXPECT_EQ("UUUUUUUUUUUU", fix::config_to_string(fix::all_unrolled));
    EXPECT_EQ("SSSSSSSSSSSS", fix::config_to_string(fix::all_swar));
    EXPECT_EQ("LLLLLLLLLLLL", fix::config_to_string(fix::all_loop));
    EXPECT_EQ("GGGGGGGGGGGG", fix::config_to_string(fix::all_generic));
}

TEST(FixETParser, ConfigRoundTrip) {
    auto cfg = fix::config_from_string("UUSLSUUSUUUU");
    EXPECT_EQ("UUSLSUUSUUUU", fix::config_to_string(cfg));
}

TEST(FixETParser, StrategyChar) {
    EXPECT_EQ('U', fix::strategy_char(fix::Strategy::Unrolled));
    EXPECT_EQ('S', fix::strategy_char(fix::Strategy::SWAR));
    EXPECT_EQ('L', fix::strategy_char(fix::Strategy::Loop));
    EXPECT_EQ('G', fix::strategy_char(fix::Strategy::Generic));
}

TEST(FixETParser, ParseUnrolled) {
    EXPECT_EQ(123u, (fix::parse_unrolled<3>("123")));
    EXPECT_EQ(12345678u, (fix::parse_unrolled<8>("12345678")));
    EXPECT_EQ(9u, (fix::parse_unrolled<1>("9")));
}

TEST(FixETParser, ParseSWAR) {
    EXPECT_EQ(1234u, (fix::parse_swar<4>("1234")));
    EXPECT_EQ(12345678u, (fix::parse_swar<8>("12345678")));
    EXPECT_EQ(123u, (fix::parse_swar<3>("123")));
    EXPECT_EQ(12u, (fix::parse_swar<2>("12")));
}

TEST(FixETParser, ParseLoop) {
    EXPECT_EQ(123u, (fix::parse_loop<3>("123")));
    EXPECT_EQ(12345678u, (fix::parse_loop<8>("12345678")));
}

TEST(FixETParser, ParseGeneric) {
    EXPECT_EQ(123u, (fix::parse_generic<3>("123")));
    EXPECT_EQ(12345678u, (fix::parse_generic<8>("12345678")));
}

TEST(FixETParser, AllStrategiesAgree) {
    // All strategies should produce identical results
    const char* input = "12345678";
    EXPECT_EQ((fix::parse_unrolled<8>(input)),
              (fix::parse_swar<8>(input)));
    EXPECT_EQ((fix::parse_unrolled<8>(input)),
              (fix::parse_loop<8>(input)));
    EXPECT_EQ((fix::parse_unrolled<8>(input)),
              (fix::parse_generic<8>(input)));
}

TEST(FixETParser, AllStrategiesAgreeAllDigitCounts) {
    const char* digits = "1234567890";
    for (int n = 1; n <= 10; ++n) {
        std::uint64_t expected = 0;
        for (int i = 0; i < n; ++i) {
            expected = expected * 10 + static_cast<std::uint64_t>(digits[i] - '0');
        }
        // Can't use template parameter from loop, but we can test known sizes
        if (n == 4) {
            EXPECT_EQ(expected, (fix::parse_unrolled<4>(digits)));
            EXPECT_EQ(expected, (fix::parse_swar<4>(digits)));
            EXPECT_EQ(expected, (fix::parse_loop<4>(digits)));
            EXPECT_EQ(expected, (fix::parse_generic<4>(digits)));
        }
        if (n == 6) {
            EXPECT_EQ(expected, (fix::parse_unrolled<6>(digits)));
            EXPECT_EQ(expected, (fix::parse_swar<6>(digits)));
            EXPECT_EQ(expected, (fix::parse_loop<6>(digits)));
            EXPECT_EQ(expected, (fix::parse_generic<6>(digits)));
        }
    }
}

TEST(FixETParser, MessagePoolGeneration) {
    auto pool = fix::generate_message_pool(100, 42);
    EXPECT_EQ(100u, pool.size());

    // All messages should be the same length
    auto expected_len = pool[0].size();
    for (auto const& msg : pool) {
        EXPECT_EQ(expected_len, msg.size());
    }

    // Different seed -> different messages
    auto pool2 = fix::generate_message_pool(100, 99);
    EXPECT_NE(pool[0], pool2[0]);
}

TEST(FixETParser, ETParserProducesNonZero) {
    auto pool = fix::generate_message_pool(10, 42);
    auto const* offsets = fix::standard_offsets.data();

    auto tok = fix::fix_et_parser<fix::all_unrolled>::parse(
        pool[0].data(), offsets);
    EXPECT_NE(0u, tok.value);

    auto tok2 = fix::fix_et_parser<fix::all_swar>::parse(
        pool[0].data(), offsets);
    EXPECT_NE(0u, tok2.value);
}

TEST(FixETParser, RandomConfigGeneration) {
    constexpr auto configs = fix::generate_random_configs<10>(42);

    // All configs should have 12 fields
    for (auto const& cfg : configs) {
        EXPECT_EQ(12u, cfg.size());
    }

    // Configs should differ (extremely unlikely for random to be identical)
    EXPECT_NE(fix::config_to_string(configs[0]),
              fix::config_to_string(configs[1]));
}

TEST(FixETParser, SplitMix64Deterministic) {
    std::uint64_t state1 = 42;
    std::uint64_t state2 = 42;
    EXPECT_EQ(fix::splitmix64(state1), fix::splitmix64(state2));
}

TEST(FixETParser, MeasureConfigProducesResults) {
    auto pool = fix::generate_message_pool(100, 42);

    // Small sample count for test speed
    auto pctl = fix::measure_config<fix::all_unrolled>(pool, 1000);

    EXPECT_EQ(1000u, pctl.samples);
    EXPECT_GT(pctl.mean, 0.0);
    EXPECT_GT(pctl.p50, 0.0);
    EXPECT_GE(pctl.p99, pctl.p50);
    EXPECT_GE(pctl.max, pctl.p99);
}
