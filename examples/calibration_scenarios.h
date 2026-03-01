// ctdp-calibrator example scenarios
// Ported from calibrator_main.cpp with restructured interfaces.
//
// Two scenarios:
//   1. memory_regime_scenario — measures array traversal at different sizes
//      to characterise L1/L2/L3/DRAM regime transitions
//   2. parser_strategy_scenario — measures different parsing strategies
//      across digit counts (CT-DP FIX parser use case)

#ifndef CTDP_CALIBRATOR_EXAMPLES_SCENARIOS_H
#define CTDP_CALIBRATOR_EXAMPLES_SCENARIOS_H

#include <ctdp/calibrator/scenario.h>
#include <ctdp/calibrator/csv_writer.h>
#include <ctdp/bench/compiler_barrier.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace ctdp::calibrator::examples {

// ═══════════════════════════════════════════════════════════════════
// Scenario 1: Memory Regime
// ═══════════════════════════════════════════════════════════════════

/// Space point for memory regime: working set size
struct memory_point {
    std::size_t bytes = 0;   ///< Working set size in bytes
    int         stride = 64; ///< Access stride in bytes (cache line)
};

/// Application identity: "array traversal at various sizes"
struct memory_traverse_tag {};

/// Scenario that measures sequential array traversal across
/// working set sizes that span L1 → L2 → L3 → DRAM boundaries.
class memory_regime_scenario {
public:
    using point_type    = memory_point;
    using callable_type = memory_traverse_tag;

    /// Construct with explicit points
    explicit memory_regime_scenario(std::vector<memory_point> pts)
        : points_{std::move(pts)} {}

    /// Construct with geometric progression of sizes
    /// @param min_kb  Smallest working set (KiB)
    /// @param max_kb  Largest working set (KiB)
    /// @param steps   Number of logarithmically-spaced points
    memory_regime_scenario(std::size_t min_kb, std::size_t max_kb,
                           int steps, int stride = 64) {
        double log_min = std::log2(static_cast<double>(min_kb));
        double log_max = std::log2(static_cast<double>(max_kb));
        double log_step = (log_max - log_min) / static_cast<double>(steps - 1);

        points_.reserve(static_cast<std::size_t>(steps));
        for (int i = 0; i < steps; ++i) {
            double log_val = log_min + static_cast<double>(i) * log_step;
            auto kb = static_cast<std::size_t>(std::pow(2.0, log_val));
            points_.push_back(memory_point{kb * 1024, stride});
        }
    }

    [[nodiscard]] std::string_view name() const { return "memory_regime"; }
    [[nodiscard]] auto points() const -> std::vector<memory_point> const& {
        return points_;
    }

    void prepare(memory_point const& pt) {
        // Allocate and initialise working set
        auto n = pt.bytes / sizeof(std::uint64_t);
        if (n == 0) n = 1;
        buffer_.resize(n);
        std::iota(buffer_.begin(), buffer_.end(), std::uint64_t{1});
    }

    [[nodiscard]] bench::result_token execute(memory_point const& pt) {
        auto stride_words = static_cast<std::size_t>(pt.stride)
                          / sizeof(std::uint64_t);
        if (stride_words == 0) stride_words = 1;

        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < buffer_.size(); i += stride_words) {
            acc += buffer_[i];
        }
        return bench::result_token{acc};
    }

private:
    std::vector<memory_point> points_;
    std::vector<std::uint64_t> buffer_;
};

/// PointFormatter for memory_point
struct memory_point_formatter {
    [[nodiscard]] static auto csv_header() -> std::string {
        return "bytes,stride";
    }
    [[nodiscard]] static auto to_csv(memory_point const& pt) -> std::string {
        return std::to_string(pt.bytes) + "," + std::to_string(pt.stride);
    }
};

// Verify concept
static_assert(Scenario<memory_regime_scenario>);
static_assert(PointFormatter<memory_point_formatter, memory_point>);

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: Parser Strategy
// ═══════════════════════════════════════════════════════════════════

/// Parsing strategies for integer fields
enum class parse_strategy : int {
    generic  = 0,  // Byte-at-a-time
    loop     = 1,  // Simple loop
    swar     = 2,  // SWAR (4 digits per iteration)
    unrolled = 3   // Fully unrolled
};

[[nodiscard]] inline std::string_view strategy_name(parse_strategy s) {
    switch (s) {
        case parse_strategy::generic:  return "generic";
        case parse_strategy::loop:     return "loop";
        case parse_strategy::swar:     return "swar";
        case parse_strategy::unrolled: return "unrolled";
    }
    return "unknown";
}

/// Space point: (digit_count, strategy) pair
struct parser_point {
    int            digits   = 0;
    parse_strategy strategy = parse_strategy::generic;
};

/// Application identity: "integer field parser"
struct integer_parser_tag {};

/// Scenario that measures different integer parsing strategies
/// across digit counts (1–12).
class parser_strategy_scenario {
public:
    using point_type    = parser_point;
    using callable_type = integer_parser_tag;

    /// Construct with the full cross-product of digits × strategies
    parser_strategy_scenario(int min_digits = 1, int max_digits = 12) {
        for (int d = min_digits; d <= max_digits; ++d) {
            for (auto s : {parse_strategy::generic, parse_strategy::loop,
                           parse_strategy::swar, parse_strategy::unrolled}) {
                points_.push_back(parser_point{d, s});
            }
        }
    }

    /// Construct with explicit points
    explicit parser_strategy_scenario(std::vector<parser_point> pts)
        : points_{std::move(pts)} {}

    [[nodiscard]] std::string_view name() const { return "parser_strategy"; }
    [[nodiscard]] auto points() const -> std::vector<parser_point> const& {
        return points_;
    }

    void prepare(parser_point const& pt) {
        // Generate a test string of the right length
        input_.clear();
        for (int i = 0; i < pt.digits; ++i) {
            input_ += static_cast<char>('1' + (i % 9));
        }
    }

    [[nodiscard]] bench::result_token execute(parser_point const& pt) {
        switch (pt.strategy) {
            case parse_strategy::generic:  return parse_generic(pt.digits);
            case parse_strategy::loop:     return parse_loop(pt.digits);
            case parse_strategy::swar:     return parse_swar(pt.digits);
            case parse_strategy::unrolled: return parse_unrolled(pt.digits);
        }
        return bench::result_token{0};
    }

private:
    std::vector<parser_point> points_;
    std::string input_;

    // ─── Strategy implementations ───────────────────────────────

    [[nodiscard]] bench::result_token parse_generic(int digits) const {
        std::uint64_t result = 0;
        for (int i = 0; i < digits && i < static_cast<int>(input_.size()); ++i) {
            result = result * 10 + static_cast<std::uint64_t>(input_[static_cast<std::size_t>(i)] - '0');
        }
        return bench::result_token{result};
    }

    [[nodiscard]] bench::result_token parse_loop(int digits) const {
        std::uint64_t result = 0;
        auto const* p = input_.data();
        int n = std::min(digits, static_cast<int>(input_.size()));
        for (int i = 0; i < n; ++i) {
            result = result * 10 + static_cast<std::uint64_t>(p[i] - '0');
        }
        return bench::result_token{result};
    }

    [[nodiscard]] bench::result_token parse_swar(int digits) const {
        // SWAR: process 4 digits at a time using multiply-accumulate
        std::uint64_t result = 0;
        auto const* p = input_.data();
        int n = std::min(digits, static_cast<int>(input_.size()));
        int i = 0;

        // Process 4 digits at a time
        for (; i + 4 <= n; i += 4) {
            std::uint32_t chunk = 0;
            std::memcpy(&chunk, p + i, 4);
            // ASCII to digits (subtract '0' from each byte)
            chunk -= 0x30303030u;
            // Combine: d0*1000 + d1*100 + d2*10 + d3 (little-endian)
            std::uint64_t val =
                static_cast<std::uint64_t>(chunk & 0xFF) * 1000 +
                static_cast<std::uint64_t>((chunk >> 8) & 0xFF) * 100 +
                static_cast<std::uint64_t>((chunk >> 16) & 0xFF) * 10 +
                static_cast<std::uint64_t>((chunk >> 24) & 0xFF);
            result = result * 10000 + val;
        }

        // Tail
        for (; i < n; ++i) {
            result = result * 10 + static_cast<std::uint64_t>(p[i] - '0');
        }
        return bench::result_token{result};
    }

    [[nodiscard]] bench::result_token parse_unrolled(int digits) const {
        // Fully unrolled: switch on digit count
        auto const* p = input_.data();
        int n = std::min(digits, static_cast<int>(input_.size()));
        std::uint64_t r = 0;

        // Duff's device style — fall through
        switch (n) {
            // Intentional fallthrough for each case
            default: [[fallthrough]];
            case 12: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n - 12)] - '0'); [[fallthrough]];
            case 11: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n - 11)] - '0'); [[fallthrough]];
            case 10: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n - 10)] - '0'); [[fallthrough]];
            case  9: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  9)] - '0'); [[fallthrough]];
            case  8: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  8)] - '0'); [[fallthrough]];
            case  7: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  7)] - '0'); [[fallthrough]];
            case  6: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  6)] - '0'); [[fallthrough]];
            case  5: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  5)] - '0'); [[fallthrough]];
            case  4: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  4)] - '0'); [[fallthrough]];
            case  3: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  3)] - '0'); [[fallthrough]];
            case  2: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  2)] - '0'); [[fallthrough]];
            case  1: r = r * 10 + static_cast<std::uint64_t>(p[static_cast<std::size_t>(n -  1)] - '0'); [[fallthrough]];
            case  0: break;
        }
        return bench::result_token{r};
    }
};

/// PointFormatter for parser_point
struct parser_point_formatter {
    [[nodiscard]] static auto csv_header() -> std::string {
        return "digits,strategy";
    }
    [[nodiscard]] static auto to_csv(parser_point const& pt) -> std::string {
        return std::to_string(pt.digits) + ","
             + std::string(strategy_name(pt.strategy));
    }
};

// Verify concepts
static_assert(Scenario<parser_strategy_scenario>);
static_assert(PointFormatter<parser_point_formatter, parser_point>);

} // namespace ctdp::calibrator::examples

#endif // CTDP_CALIBRATOR_EXAMPLES_SCENARIOS_H
