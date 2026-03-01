// ctdp-calibrator test suite
// Tests: data_point invariants, Scenario concept, calibration_harness,
//        csv_writer round-trip, mock scenarios.
//
// Build: g++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -O2
//        -I../../ctdp-bench/include -I../include
//        calibrator_test.cpp -o calibrator_test

#include <ctdp/calibrator/data_point.h>
#include <ctdp/calibrator/scenario.h>
#include <ctdp/calibrator/calibration_harness.h>
#include <ctdp/calibrator/csv_writer.h>

#include <ctdp/bench/compiler_barrier.h>
#include <ctdp/bench/metric.h>
#include <ctdp/bench/statistics.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// ═══════════════════════════════════════════════════════════════════
// Test infrastructure
// ═══════════════════════════════════════════════════════════════════

static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name)                                                        \
    do {                                                                  \
        ++tests_run;                                                      \
        std::cerr << "  [" << tests_run << "] " << #name << "... ";      \
    } while (0)

#define PASS()                                                            \
    do { ++tests_passed; std::cerr << "PASS\n"; } while (0)

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "FAIL\n    " << __FILE__ << ":" << __LINE__      \
                      << ": CHECK(" #cond ") failed\n";                   \
            return;                                                       \
        }                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                             \
    do {                                                                  \
        if (std::abs(static_cast<double>(a) - static_cast<double>(b))     \
            > static_cast<double>(eps)) {                                 \
            std::cerr << "FAIL\n    " << __FILE__ << ":" << __LINE__      \
                      << ": CHECK_NEAR(" #a "=" << (a) << ", " #b "="    \
                      << (b) << ", eps=" << (eps) << ") failed\n";        \
            return;                                                       \
        }                                                                 \
    } while (0)

// ═══════════════════════════════════════════════════════════════════
// Mock types for testing
// ═══════════════════════════════════════════════════════════════════

/// Simple space point: just an integer size
struct mock_point {
    int size = 0;

    friend bool operator==(mock_point const&, mock_point const&) = default;
};

/// Application identity tag
struct mock_callable_tag {};

/// Mock scenario: busy loop whose cost scales linearly with point.size
struct mock_scenario {
    using point_type    = mock_point;
    using callable_type = mock_callable_tag;

    std::vector<mock_point> points_;

    explicit mock_scenario(std::vector<mock_point> pts)
        : points_{std::move(pts)} {}

    [[nodiscard]] std::string_view name() const { return "mock_scenario"; }
    [[nodiscard]] auto points() const -> std::vector<mock_point> const& {
        return points_;
    }

    void prepare(mock_point const&) {
        // No state to set up in the mock
    }

    [[nodiscard]] ctdp::bench::result_token execute(mock_point const& pt) const {
        std::uint64_t acc = 0;
        for (int i = 0; i < pt.size * 100; ++i) {
            acc += static_cast<std::uint64_t>(i);
        }
        return ctdp::bench::result_token{acc};
    }
};

/// Trivial scenario: constant-time work, for testing harness mechanics
struct trivial_scenario {
    using point_type    = mock_point;
    using callable_type = mock_callable_tag;

    std::vector<mock_point> points_;

    explicit trivial_scenario(std::vector<mock_point> pts)
        : points_{std::move(pts)} {}

    [[nodiscard]] std::string_view name() const { return "trivial"; }
    [[nodiscard]] auto points() const -> std::vector<mock_point> const& {
        return points_;
    }

    void prepare(mock_point const&) {}

    [[nodiscard]] ctdp::bench::result_token execute(mock_point const&) const {
        return ctdp::bench::result_token{42};
    }
};

/// PointFormatter for mock_point
struct mock_point_formatter {
    [[nodiscard]] static auto csv_header() -> std::string {
        return "size";
    }
    [[nodiscard]] static auto to_csv(mock_point const& pt) -> std::string {
        return std::to_string(pt.size);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Compile-time concept checks
// ═══════════════════════════════════════════════════════════════════

static_assert(ctdp::calibrator::Scenario<mock_scenario>,
    "mock_scenario must satisfy Scenario concept");
static_assert(ctdp::calibrator::Scenario<trivial_scenario>,
    "trivial_scenario must satisfy Scenario concept");

static_assert(ctdp::calibrator::PointFormatter<mock_point_formatter, mock_point>,
    "mock_point_formatter must satisfy PointFormatter concept");
static_assert(ctdp::calibrator::SnapshotFormatter<
    ctdp::calibrator::counter_snapshot_formatter,
    ctdp::bench::counter_snapshot>,
    "counter_snapshot_formatter must satisfy SnapshotFormatter");
static_assert(ctdp::calibrator::SnapshotFormatter<
    ctdp::calibrator::null_snapshot_formatter,
    ctdp::bench::null_metric::null_snapshot>,
    "null_snapshot_formatter must satisfy SnapshotFormatter");

// ═══════════════════════════════════════════════════════════════════
// Test: data_point
// ═══════════════════════════════════════════════════════════════════

void test_data_point_default_invariant() {
    TEST(data_point_default_invariant);
    using dp_t = ctdp::calibrator::data_point<mock_point,
                                               ctdp::bench::null_metric::null_snapshot>;
    dp_t dp;
    CHECK(dp.invariant());  // both vectors empty → sizes match
    CHECK(dp.reps() == 0);
    CHECK_NEAR(dp.median_ns, 0.0, 1e-9);
    CHECK_NEAR(dp.relative_mad(), 0.0, 1e-9);
    PASS();
}

void test_data_point_with_data() {
    TEST(data_point_with_data);
    using snap_t = ctdp::bench::null_metric::null_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    dp_t dp;
    dp.space_point = mock_point{42};
    dp.median_ns   = 100.0;
    dp.mad_ns      = 5.0;
    dp.raw_timings   = {95.0, 98.0, 100.0, 102.0, 105.0};
    dp.raw_snapshots = {snap_t{}, snap_t{}, snap_t{}, snap_t{}, snap_t{}};

    CHECK(dp.invariant());
    CHECK(dp.reps() == 5);
    CHECK(dp.space_point.size == 42);
    CHECK_NEAR(dp.relative_mad(), 0.05, 1e-9);
    PASS();
}

void test_data_point_broken_invariant() {
    TEST(data_point_broken_invariant);
    using snap_t = ctdp::bench::null_metric::null_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    dp_t dp;
    dp.raw_timings   = {1.0, 2.0, 3.0};
    dp.raw_snapshots = {snap_t{}, snap_t{}};  // size mismatch!

    CHECK(!dp.invariant());
    PASS();
}

void test_data_point_counter_alias() {
    TEST(data_point_counter_alias);
    // Verify the convenience alias compiles
    using dp_t = ctdp::calibrator::counter_data_point<mock_point>;
    dp_t dp;
    dp.space_point = mock_point{7};
    CHECK(dp.invariant());
    CHECK(dp.reps() == 0);
    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: Scenario concept (compile-time — if it compiles, it passes)
// ═══════════════════════════════════════════════════════════════════

/// A type that should NOT satisfy Scenario — missing callable_type
struct bad_scenario_no_callable {
    using point_type = mock_point;
    // missing: callable_type

    std::string_view name() const { return "bad"; }
    auto points() const -> std::vector<mock_point> const& {
        static std::vector<mock_point> v;
        return v;
    }
    void prepare(mock_point const&) {}
    ctdp::bench::result_token execute(mock_point const&) const {
        return ctdp::bench::result_token{0};
    }
};

/// Missing execute()
struct bad_scenario_no_execute {
    using point_type = mock_point;
    using callable_type = mock_callable_tag;

    std::string_view name() const { return "bad"; }
    auto points() const -> std::vector<mock_point> const& {
        static std::vector<mock_point> v;
        return v;
    }
    void prepare(mock_point const&) {}
    // missing: execute()
};

void test_scenario_concept_negative() {
    TEST(scenario_concept_negative);
    static_assert(!ctdp::calibrator::Scenario<bad_scenario_no_callable>,
        "Missing callable_type should not satisfy Scenario");
    static_assert(!ctdp::calibrator::Scenario<bad_scenario_no_execute>,
        "Missing execute() should not satisfy Scenario");
    static_assert(!ctdp::calibrator::Scenario<int>,
        "int should not satisfy Scenario");
    PASS();
}

void test_scenario_name_and_points() {
    TEST(scenario_name_and_points);
    mock_scenario s{{mock_point{1}, mock_point{2}, mock_point{3}}};

    CHECK(s.name() == "mock_scenario");

    auto pts = s.points();
    CHECK(pts.size() == 3);
    CHECK(pts[0].size == 1);
    CHECK(pts[1].size == 2);
    CHECK(pts[2].size == 3);
    PASS();
}

void test_scenario_execute_returns_token() {
    TEST(scenario_execute_returns_token);
    mock_scenario s{{mock_point{10}}};
    s.prepare(mock_point{10});
    auto tok = s.execute(mock_point{10});
    // Token should be non-zero for size > 0
    CHECK(tok.value != 0);
    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: calibration_harness with null_metric (no perf_event needed)
// ═══════════════════════════════════════════════════════════════════

void test_harness_type_aliases() {
    TEST(harness_type_aliases);
    using harness_t = ctdp::calibrator::calibration_harness<
        mock_scenario, ctdp::bench::null_metric>;

    static_assert(std::is_same_v<harness_t::point_type, mock_point>);
    static_assert(std::is_same_v<harness_t::callable_type, mock_callable_tag>);
    static_assert(std::is_same_v<harness_t::snapshot_type,
                                  ctdp::bench::null_metric::null_snapshot>);
    PASS();
}

void test_harness_config_defaults() {
    TEST(harness_config_defaults);
    ctdp::calibrator::harness_config cfg;
    CHECK(cfg.reps == 10);
    CHECK(cfg.warmup_iters == 200);
    CHECK(cfg.measure_iters == 1);
    CHECK(cfg.pin_cpu == true);
    CHECK(cfg.pin_cpu_id == -1);
    CHECK(cfg.boost_priority == true);
    CHECK(cfg.flush_cache == true);
    CHECK(cfg.llc_bytes == 0);
    CHECK(cfg.verbose == true);
    PASS();
}

void test_harness_run_trivial() {
    TEST(harness_run_trivial);

    trivial_scenario scenario{{mock_point{1}, mock_point{2}, mock_point{3}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps          = 3;
    cfg.warmup_iters  = 5;
    cfg.measure_iters = 1;
    cfg.pin_cpu       = false;   // don't require root
    cfg.boost_priority = false;
    cfg.flush_cache    = false;  // speed up test
    cfg.verbose        = false;

    auto harness = ctdp::calibrator::calibration_harness<
        trivial_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto results = harness.run();

    CHECK(results.size() == 3);  // one data_point per space point

    for (auto const& dp : results) {
        CHECK(dp.invariant());
        CHECK(dp.reps() == 3);
        CHECK(dp.median_ns > 0.0);   // timing should be positive
        CHECK(dp.raw_timings.size() == 3);
        CHECK(dp.raw_snapshots.size() == 3);
    }

    // Points should be in order
    CHECK(results[0].space_point.size == 1);
    CHECK(results[1].space_point.size == 2);
    CHECK(results[2].space_point.size == 3);

    PASS();
}

void test_harness_run_scaling() {
    TEST(harness_run_scaling);

    // Larger size = more work = more time
    mock_scenario scenario{{mock_point{10}, mock_point{100}, mock_point{1000}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps          = 5;
    cfg.warmup_iters  = 10;
    cfg.measure_iters = 1;
    cfg.pin_cpu       = false;
    cfg.boost_priority = false;
    cfg.flush_cache    = false;
    cfg.verbose        = false;

    auto harness = ctdp::calibrator::calibration_harness<
        mock_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto results = harness.run();
    CHECK(results.size() == 3);

    // Timing should increase with size (loose check — at least monotone)
    // size=1000 should be measurably slower than size=10
    CHECK(results[2].median_ns > results[0].median_ns);

    PASS();
}

void test_harness_with_cache_flush() {
    TEST(harness_with_cache_flush);

    trivial_scenario scenario{{mock_point{1}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps          = 3;
    cfg.warmup_iters  = 2;
    cfg.measure_iters = 1;
    cfg.pin_cpu       = false;
    cfg.boost_priority = false;
    cfg.flush_cache    = true;   // test with cache flushing
    cfg.llc_bytes      = 1024 * 1024;  // 1 MiB (small for speed)
    cfg.verbose        = false;

    auto harness = ctdp::calibrator::calibration_harness<
        trivial_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto results = harness.run();
    CHECK(results.size() == 1);
    CHECK(results[0].invariant());
    CHECK(results[0].reps() == 3);
    CHECK(results[0].median_ns > 0.0);

    PASS();
}

void test_harness_make_factory() {
    TEST(harness_make_factory);

    trivial_scenario scenario{{mock_point{1}}};
    ctdp::calibrator::harness_config cfg;
    cfg.reps = 2;
    cfg.warmup_iters = 2;
    cfg.pin_cpu = false;
    cfg.boost_priority = false;
    cfg.flush_cache = false;
    cfg.verbose = false;

    auto harness = ctdp::calibrator::make_harness<
        trivial_scenario, ctdp::bench::null_metric>(
            std::move(scenario), cfg);

    auto results = harness.run();
    CHECK(results.size() == 1);
    CHECK(results[0].invariant());
    PASS();
}

void test_harness_measure_point_directly() {
    TEST(harness_measure_point_directly);

    trivial_scenario scenario{{mock_point{5}}};
    ctdp::calibrator::harness_config cfg;
    cfg.reps = 4;
    cfg.warmup_iters = 3;
    cfg.pin_cpu = false;
    cfg.boost_priority = false;
    cfg.flush_cache = false;
    cfg.verbose = false;

    auto harness = ctdp::calibrator::calibration_harness<
        trivial_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto env = ctdp::bench::capture_environment();
    auto dp = harness.measure_point(mock_point{5}, env);

    CHECK(dp.invariant());
    CHECK(dp.reps() == 4);
    CHECK(dp.space_point.size == 5);
    CHECK(dp.median_ns > 0.0);

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: calibration_harness with counter_metric
// ═══════════════════════════════════════════════════════════════════

void test_harness_with_counter_metric() {
    TEST(harness_with_counter_metric);

    mock_scenario scenario{{mock_point{100}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps          = 3;
    cfg.warmup_iters  = 5;
    cfg.measure_iters = 1;
    cfg.pin_cpu       = false;
    cfg.boost_priority = false;
    cfg.flush_cache    = false;
    cfg.verbose        = false;

    // Default metric = counter_metric
    auto harness = ctdp::calibrator::calibration_harness<mock_scenario>{
        std::move(scenario), cfg};

    auto results = harness.run();
    CHECK(results.size() == 1);
    CHECK(results[0].invariant());
    CHECK(results[0].reps() == 3);

    // Tier 0 counters (TSC, thread CPU) should always be populated
    for (auto const& snap : results[0].raw_snapshots) {
        CHECK(snap.tsc_cycles > 0);
        // thread_cpu_ns may be 0 on non-Linux, but TSC is always there
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: csv_writer
// ═══════════════════════════════════════════════════════════════════

void test_csv_header_format() {
    TEST(csv_header_format);

    auto hdr = ctdp::calibrator::counter_snapshot_formatter::csv_header();
    CHECK(hdr.find("tsc_cycles") != std::string::npos);
    CHECK(hdr.find("ipc") != std::string::npos);
    CHECK(hdr.find("cache_miss_rate") != std::string::npos);

    auto null_hdr = ctdp::calibrator::null_snapshot_formatter::csv_header();
    CHECK(null_hdr.empty());

    PASS();
}

void test_csv_write_null_metric() {
    TEST(csv_write_null_metric);

    using snap_t = ctdp::bench::null_metric::null_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    dp_t dp;
    dp.space_point = mock_point{42};
    dp.median_ns   = 100.0;
    dp.mad_ns      = 5.0;
    dp.raw_timings   = {95.0, 100.0, 105.0};
    dp.raw_snapshots = {snap_t{}, snap_t{}, snap_t{}};

    std::vector<dp_t> data = {dp};

    std::ostringstream oss;
    ctdp::calibrator::write_csv(
        oss,
        std::span<const dp_t>{data},
        mock_point_formatter{},
        ctdp::calibrator::null_snapshot_formatter{});

    auto csv = oss.str();

    // Check header
    CHECK(csv.find("size,median_ns,mad_ns,rep,wall_ns") == 0);

    // Should have 3 data rows (one per rep) + 1 header
    std::size_t newlines = static_cast<std::size_t>(
        std::count(csv.begin(), csv.end(), '\n'));
    CHECK(newlines == 4);  // header + 3 rows

    // Check first data row contains the point value
    CHECK(csv.find("\n42,") != std::string::npos);

    PASS();
}

void test_csv_write_counter_metric() {
    TEST(csv_write_counter_metric);

    using snap_t = ctdp::bench::counter_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    snap_t snap;
    snap.tsc_cycles = 1000;
    snap.instructions = 500;
    snap.ipc = 0.5;

    dp_t dp;
    dp.space_point = mock_point{7};
    dp.median_ns   = 50.0;
    dp.mad_ns      = 2.0;
    dp.raw_timings   = {48.0, 50.0, 52.0};
    dp.raw_snapshots = {snap, snap, snap};

    std::vector<dp_t> data = {dp};

    std::ostringstream oss;
    ctdp::calibrator::write_csv(
        oss,
        std::span<const dp_t>{data},
        mock_point_formatter{},
        ctdp::calibrator::counter_snapshot_formatter{});

    auto csv = oss.str();

    // Header should contain both point and snapshot columns
    CHECK(csv.find("size,") == 0);
    CHECK(csv.find("tsc_cycles") != std::string::npos);
    CHECK(csv.find("ipc") != std::string::npos);

    // 3 data rows + header
    std::size_t newlines = static_cast<std::size_t>(
        std::count(csv.begin(), csv.end(), '\n'));
    CHECK(newlines == 4);

    PASS();
}

void test_csv_write_summary() {
    TEST(csv_write_summary);

    using snap_t = ctdp::bench::null_metric::null_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    dp_t dp1;
    dp1.space_point = mock_point{10};
    dp1.median_ns = 100.0;
    dp1.mad_ns = 3.0;
    dp1.raw_timings   = {97.0, 100.0, 103.0};
    dp1.raw_snapshots = {snap_t{}, snap_t{}, snap_t{}};

    dp_t dp2;
    dp2.space_point = mock_point{20};
    dp2.median_ns = 200.0;
    dp2.mad_ns = 8.0;
    dp2.raw_timings   = {192.0, 200.0, 208.0};
    dp2.raw_snapshots = {snap_t{}, snap_t{}, snap_t{}};

    std::vector<dp_t> data = {dp1, dp2};

    std::ostringstream oss;
    ctdp::calibrator::write_csv_summary(
        oss,
        std::span<const dp_t>{data},
        mock_point_formatter{},
        ctdp::calibrator::null_snapshot_formatter{});

    auto csv = oss.str();

    // Header + 2 summary rows (one per point, not per rep)
    std::size_t newlines = static_cast<std::size_t>(
        std::count(csv.begin(), csv.end(), '\n'));
    CHECK(newlines == 3);

    // Should contain reps and relative_mad columns
    CHECK(csv.find("reps") != std::string::npos);
    CHECK(csv.find("relative_mad") != std::string::npos);

    PASS();
}

void test_csv_empty_dataset() {
    TEST(csv_empty_dataset);

    using snap_t = ctdp::bench::null_metric::null_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    std::vector<dp_t> data;  // empty

    std::ostringstream oss;
    ctdp::calibrator::write_csv(
        oss,
        std::span<const dp_t>{data},
        mock_point_formatter{},
        ctdp::calibrator::null_snapshot_formatter{});

    auto csv = oss.str();

    // Should still have the header
    CHECK(!csv.empty());
    CHECK(csv.find("size,median_ns") == 0);

    // Just 1 newline (the header)
    std::size_t newlines = static_cast<std::size_t>(
        std::count(csv.begin(), csv.end(), '\n'));
    CHECK(newlines == 1);

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: CSV round-trip (write + parse back)
// ═══════════════════════════════════════════════════════════════════

void test_csv_round_trip() {
    TEST(csv_round_trip);

    using snap_t = ctdp::bench::null_metric::null_snapshot;
    using dp_t   = ctdp::calibrator::data_point<mock_point, snap_t>;

    // Create data with known values
    dp_t dp;
    dp.space_point = mock_point{99};
    dp.median_ns   = 123.45;
    dp.mad_ns      = 6.78;
    dp.raw_timings   = {120.0, 123.45, 126.9};
    dp.raw_snapshots = {snap_t{}, snap_t{}, snap_t{}};

    std::vector<dp_t> data = {dp};

    std::ostringstream oss;
    ctdp::calibrator::write_csv(
        oss,
        std::span<const dp_t>{data},
        mock_point_formatter{},
        ctdp::calibrator::null_snapshot_formatter{});

    // Parse back line by line
    std::istringstream iss(oss.str());
    std::string line;

    // Header
    std::getline(iss, line);
    CHECK(line == "size,median_ns,mad_ns,rep,wall_ns");

    // Row 0
    std::getline(iss, line);
    CHECK(line.substr(0, 2) == "99");  // size=99

    // Row 1 — rep=1
    std::getline(iss, line);
    CHECK(line.find(",1,") != std::string::npos);

    // Row 2 — rep=2
    std::getline(iss, line);
    CHECK(line.find(",2,") != std::string::npos);

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: end-to-end (harness → csv_writer)
// ═══════════════════════════════════════════════════════════════════

void test_end_to_end_harness_to_csv() {
    TEST(end_to_end_harness_to_csv);

    mock_scenario scenario{{mock_point{10}, mock_point{50}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps          = 3;
    cfg.warmup_iters  = 5;
    cfg.measure_iters = 1;
    cfg.pin_cpu       = false;
    cfg.boost_priority = false;
    cfg.flush_cache    = false;
    cfg.verbose        = false;

    auto harness = ctdp::calibrator::calibration_harness<
        mock_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto results = harness.run();
    CHECK(results.size() == 2);

    // Write to CSV
    std::ostringstream oss;
    ctdp::calibrator::write_csv(
        oss,
        std::span<const decltype(results)::value_type>{results},
        mock_point_formatter{},
        ctdp::calibrator::null_snapshot_formatter{});

    auto csv = oss.str();

    // Header + 2 points × 3 reps = 7 lines
    std::size_t newlines = static_cast<std::size_t>(
        std::count(csv.begin(), csv.end(), '\n'));
    CHECK(newlines == 7);

    // Summary variant
    std::ostringstream oss2;
    ctdp::calibrator::write_csv_summary(
        oss2,
        std::span<const decltype(results)::value_type>{results},
        mock_point_formatter{},
        ctdp::calibrator::null_snapshot_formatter{});

    auto summary = oss2.str();
    // Header + 2 summary rows
    std::size_t summary_lines = static_cast<std::size_t>(
        std::count(summary.begin(), summary.end(), '\n'));
    CHECK(summary_lines == 3);

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: environment_context capture
// ═══════════════════════════════════════════════════════════════════

void test_environment_context_populated() {
    TEST(environment_context_populated);

    trivial_scenario scenario{{mock_point{1}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps = 2;
    cfg.warmup_iters = 2;
    cfg.pin_cpu = false;
    cfg.boost_priority = false;
    cfg.flush_cache = false;
    cfg.verbose = false;

    auto harness = ctdp::calibrator::calibration_harness<
        trivial_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto results = harness.run();
    CHECK(results.size() == 1);

    auto const& env = results[0].env;
    CHECK(env.llc_bytes > 0);  // should be auto-detected or defaulted

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: statistics integration (median/MAD populated correctly)
// ═══════════════════════════════════════════════════════════════════

void test_statistics_in_data_points() {
    TEST(statistics_in_data_points);

    trivial_scenario scenario{{mock_point{1}}};

    ctdp::calibrator::harness_config cfg;
    cfg.reps = 7;  // odd number for clear median
    cfg.warmup_iters = 3;
    cfg.measure_iters = 1;
    cfg.pin_cpu = false;
    cfg.boost_priority = false;
    cfg.flush_cache = false;
    cfg.verbose = false;

    auto harness = ctdp::calibrator::calibration_harness<
        trivial_scenario, ctdp::bench::null_metric>{
            std::move(scenario), cfg};

    auto results = harness.run();
    CHECK(results.size() == 1);

    auto const& dp = results[0];
    CHECK(dp.reps() == 7);
    CHECK(dp.median_ns > 0.0);
    CHECK(dp.mad_ns >= 0.0);  // MAD can be 0 if all timings identical

    // Verify median is actually the median of raw_timings
    auto sorted = dp.raw_timings;
    std::sort(sorted.begin(), sorted.end());
    double expected_median = sorted[3];  // 7 elements, median is index 3
    CHECK_NEAR(dp.median_ns, expected_median, 0.01);

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// Test: callable_type propagation
// ═══════════════════════════════════════════════════════════════════

struct custom_callable_tag {};
struct custom_point { int x = 0; };

struct typed_scenario {
    using point_type    = custom_point;
    using callable_type = custom_callable_tag;

    std::vector<custom_point> pts_{{custom_point{1}}};

    std::string_view name() const { return "typed"; }
    auto points() const -> std::vector<custom_point> const& { return pts_; }
    void prepare(custom_point const&) {}
    ctdp::bench::result_token execute(custom_point const&) const {
        return ctdp::bench::result_token{0};
    }
};

void test_callable_type_propagation() {
    TEST(callable_type_propagation);

    static_assert(ctdp::calibrator::Scenario<typed_scenario>);

    using harness_t = ctdp::calibrator::calibration_harness<
        typed_scenario, ctdp::bench::null_metric>;

    // Verify type aliases propagate callable_type
    static_assert(std::is_same_v<harness_t::callable_type, custom_callable_tag>);
    static_assert(std::is_same_v<harness_t::point_type, custom_point>);

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::cerr << "ctdp-calibrator test suite\n"
              << "═════════════════════════════════════════\n\n"
              << "data_point tests:\n";

    test_data_point_default_invariant();
    test_data_point_with_data();
    test_data_point_broken_invariant();
    test_data_point_counter_alias();

    std::cerr << "\nScenario concept tests:\n";
    test_scenario_concept_negative();
    test_scenario_name_and_points();
    test_scenario_execute_returns_token();

    std::cerr << "\ncalibration_harness tests:\n";
    test_harness_type_aliases();
    test_harness_config_defaults();
    test_harness_run_trivial();
    test_harness_run_scaling();
    test_harness_with_cache_flush();
    test_harness_make_factory();
    test_harness_measure_point_directly();
    test_harness_with_counter_metric();

    std::cerr << "\ncsv_writer tests:\n";
    test_csv_header_format();
    test_csv_write_null_metric();
    test_csv_write_counter_metric();
    test_csv_write_summary();
    test_csv_empty_dataset();
    test_csv_round_trip();

    std::cerr << "\nIntegration tests:\n";
    test_end_to_end_harness_to_csv();
    test_environment_context_populated();
    test_statistics_in_data_points();
    test_callable_type_propagation();

    std::cerr << "\n═════════════════════════════════════════\n"
              << tests_passed << "/" << tests_run << " tests passed\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
