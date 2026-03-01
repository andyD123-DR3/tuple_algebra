// tests/solver/test_cost_table_io.cpp
// Tests for cost_table, cost_table_io_parse, cost_table_io_stream.
//
// Validates:
//   1.  Constexpr parse of basic cost table
//   2.  Constexpr verification with static_assert
//   3.  Constexpr parse with comments and blank lines
//   4.  Constexpr parse with strategies-before-elements order
//   5.  Constexpr parse with negative costs
//   6.  Constexpr parse with integer costs (no decimal point)
//   7.  Constexpr parse with rows out of order
//   8.  Runtime round-trip: write → read → compare
//   9.  Runtime round-trip with larger table
//  10.  Write output format verification
//  11.  Error: missing 'elements' line
//  12.  Error: missing 'strategies' line
//  13.  Error: 'row' before dimensions
//  14.  Error: row index out of range
//  15.  Error: duplicate 'elements' line
//  16.  Error: duplicate 'strategies' line
//  17.  Error: duplicate row index
//  18.  Error: incomplete — not all rows provided
//  19.  Error: unrecognised line (constexpr parse)
//  20.  Error: capacity overflow — elements
//  21.  Error: capacity overflow — strategies
//  22.  Windows \r\n line endings
//  23.  Tag-object read overload

#include "ctdp/solver/cost_table_io.h"
#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace ct_io = ctdp::cost_table_io;
using ctdp::cost_table;
namespace ct_cap = ctdp::ct_cap;
using ctdp::ct_cap_from;

// =============================================================================
// 1. Constexpr parse: basic 2×3 table
// =============================================================================

constexpr auto k_basic_text =
    "elements 2\n"
    "strategies 3\n"
    "row 0 1.0 2.0 3.0\n"
    "row 1 4.0 5.0 6.0\n";

constexpr auto k_basic = ct_io::parse<ct_cap::tiny>(k_basic_text);

TEST(CostTableIO, ConstexprParseBasic) {
    EXPECT_EQ(k_basic.num_elements(), 2u);
    EXPECT_EQ(k_basic.num_strategies(), 3u);
    EXPECT_DOUBLE_EQ(k_basic(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(k_basic(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(k_basic(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(k_basic(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(k_basic(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(k_basic(1, 2), 6.0);
}

// =============================================================================
// 2. Constexpr verification with static_assert
// =============================================================================

static_assert(k_basic.num_elements() == 2);
static_assert(k_basic.num_strategies() == 3);
static_assert(k_basic(0, 0) == 1.0);
static_assert(k_basic(1, 2) == 6.0);
static_assert(!k_basic.empty());

TEST(CostTableIO, StaticAssertVerification) {
    // The static_asserts above are the real test.  Runtime mirror for reporting.
    EXPECT_EQ(k_basic.num_elements(), 2u);
    EXPECT_EQ(k_basic(0, 0), 1.0);
    EXPECT_EQ(k_basic(1, 2), 6.0);
    EXPECT_FALSE(k_basic.empty());
}

// =============================================================================
// 3. Constexpr parse: comments and blank lines
// =============================================================================

constexpr auto k_comments_text =
    "# Cost table for GEMM tiling\n"
    "\n"
    "elements 2\n"
    "# Two strategies: unroll-4 and unroll-8\n"
    "strategies 2\n"
    "\n"
    "row 0 10.5 20.5\n"
    "# Element 1 has lower costs\n"
    "row 1 5.5 15.5\n";

constexpr auto k_comments = ct_io::parse<ct_cap::tiny>(k_comments_text);

TEST(CostTableIO, ConstexprParseCommentsAndBlanks) {
    EXPECT_EQ(k_comments.num_elements(), 2u);
    EXPECT_EQ(k_comments.num_strategies(), 2u);
    EXPECT_DOUBLE_EQ(k_comments(0, 0), 10.5);
    EXPECT_DOUBLE_EQ(k_comments(0, 1), 20.5);
    EXPECT_DOUBLE_EQ(k_comments(1, 0), 5.5);
    EXPECT_DOUBLE_EQ(k_comments(1, 1), 15.5);
}

// =============================================================================
// 4. Constexpr parse: strategies-before-elements order
// =============================================================================

constexpr auto k_reversed_text =
    "strategies 2\n"
    "elements 3\n"
    "row 0 1.0 2.0\n"
    "row 1 3.0 4.0\n"
    "row 2 5.0 6.0\n";

constexpr auto k_reversed = ct_io::parse<ct_cap::tiny>(k_reversed_text);

TEST(CostTableIO, ConstexprParseReversedDimensionOrder) {
    EXPECT_EQ(k_reversed.num_elements(), 3u);
    EXPECT_EQ(k_reversed.num_strategies(), 2u);
    EXPECT_DOUBLE_EQ(k_reversed(2, 1), 6.0);
}

// =============================================================================
// 5. Constexpr parse: negative costs
// =============================================================================

constexpr auto k_negative_text =
    "elements 2\n"
    "strategies 2\n"
    "row 0 -1.5 2.0\n"
    "row 1 3.0 -4.5\n";

constexpr auto k_negative = ct_io::parse<ct_cap::tiny>(k_negative_text);

TEST(CostTableIO, ConstexprParseNegativeCosts) {
    EXPECT_DOUBLE_EQ(k_negative(0, 0), -1.5);
    EXPECT_DOUBLE_EQ(k_negative(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(k_negative(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(k_negative(1, 1), -4.5);
}

// =============================================================================
// 6. Constexpr parse: integer costs (no decimal point)
// =============================================================================

constexpr auto k_integer_text =
    "elements 2\n"
    "strategies 2\n"
    "row 0 10 20\n"
    "row 1 30 40\n";

constexpr auto k_integer = ct_io::parse<ct_cap::tiny>(k_integer_text);

TEST(CostTableIO, ConstexprParseIntegerCosts) {
    EXPECT_DOUBLE_EQ(k_integer(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(k_integer(0, 1), 20.0);
    EXPECT_DOUBLE_EQ(k_integer(1, 0), 30.0);
    EXPECT_DOUBLE_EQ(k_integer(1, 1), 40.0);
}

// =============================================================================
// 7. Constexpr parse: rows out of sequential order
// =============================================================================

constexpr auto k_unordered_text =
    "elements 3\n"
    "strategies 2\n"
    "row 2 5.0 6.0\n"
    "row 0 1.0 2.0\n"
    "row 1 3.0 4.0\n";

constexpr auto k_unordered = ct_io::parse<ct_cap::tiny>(k_unordered_text);

TEST(CostTableIO, ConstexprParseRowsOutOfOrder) {
    EXPECT_DOUBLE_EQ(k_unordered(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(k_unordered(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(k_unordered(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(k_unordered(2, 1), 6.0);
}

// =============================================================================
// 8. Runtime round-trip: write → read → compare
// =============================================================================

TEST(CostTableIO, RuntimeRoundTrip) {
    constexpr auto ct1 = ct_io::parse<ct_cap::tiny>(
        "elements 2\n"
        "strategies 3\n"
        "row 0 1.5 2.5 3.5\n"
        "row 1 4.5 5.5 6.5\n");

    // Write to string.
    std::ostringstream oss;
    ct_io::write(oss, ct1);
    std::string text = oss.str();

    // Read back.
    std::istringstream iss{text};
    auto ct2 = ct_io::read<ct_cap::tiny>(iss);

    EXPECT_EQ(ct1, ct2);
}

// =============================================================================
// 9. Runtime round-trip: larger table
// =============================================================================

TEST(CostTableIO, RuntimeRoundTripLarger) {
    // Build a 5×4 table programmatically.
    cost_table<ct_cap::small> ct1(5, 4);
    double val = 0.0;
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ct1(i, j) = val;
            val += 1.5;
        }
    }

    std::ostringstream oss;
    ct_io::write(oss, ct1);

    std::istringstream iss{oss.str()};
    auto ct2 = ct_io::read<ct_cap::small>(iss);

    EXPECT_EQ(ct1, ct2);
}

// =============================================================================
// 10. Write output format verification
// =============================================================================

TEST(CostTableIO, WriteOutputFormat) {
    constexpr auto ct = ct_io::parse<ct_cap::tiny>(
        "elements 2\n"
        "strategies 2\n"
        "row 0 1.5 2.5\n"
        "row 1 3.5 4.5\n");

    std::ostringstream oss;
    ct_io::write(oss, ct);
    std::string text = oss.str();

    // Must contain the header lines and row lines.
    EXPECT_NE(text.find("elements 2"), std::string::npos);
    EXPECT_NE(text.find("strategies 2"), std::string::npos);
    EXPECT_NE(text.find("row 0"), std::string::npos);
    EXPECT_NE(text.find("row 1"), std::string::npos);
    EXPECT_NE(text.find("1.5"), std::string::npos);
}

// =============================================================================
// 11. Error: missing 'elements' line
// =============================================================================

TEST(CostTableIO, ErrorMissingElements) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "strategies 2\n"
            "row 0 1.0 2.0\n"),
        std::runtime_error);
}

// =============================================================================
// 12. Error: missing 'strategies' line
// =============================================================================

TEST(CostTableIO, ErrorMissingStrategies) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "row 0 1.0 2.0\n"),
        std::runtime_error);
}

// =============================================================================
// 13. Error: 'row' before dimensions
// =============================================================================

TEST(CostTableIO, ErrorRowBeforeDimensions) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "row 0 1.0 2.0\n"
            "elements 2\n"
            "strategies 2\n"),
        std::runtime_error);
}

// =============================================================================
// 14. Error: row index out of range
// =============================================================================

TEST(CostTableIO, ErrorRowIndexOutOfRange) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "strategies 2\n"
            "row 0 1.0 2.0\n"
            "row 5 3.0 4.0\n"),
        std::runtime_error);
}

// =============================================================================
// 15. Error: duplicate 'elements' line
// =============================================================================

TEST(CostTableIO, ErrorDuplicateElements) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "elements 3\n"
            "strategies 2\n"
            "row 0 1.0 2.0\n"
            "row 1 3.0 4.0\n"),
        std::runtime_error);
}

// =============================================================================
// 16. Error: duplicate 'strategies' line
// =============================================================================

TEST(CostTableIO, ErrorDuplicateStrategies) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "strategies 2\n"
            "strategies 3\n"
            "row 0 1.0 2.0\n"
            "row 1 3.0 4.0\n"),
        std::runtime_error);
}

// =============================================================================
// 17. Error: duplicate row index
// =============================================================================

TEST(CostTableIO, ErrorDuplicateRow) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "strategies 2\n"
            "row 0 1.0 2.0\n"
            "row 0 3.0 4.0\n"),
        std::runtime_error);
}

// =============================================================================
// 18. Error: incomplete — not all rows provided
// =============================================================================

TEST(CostTableIO, ErrorIncompleteRows) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 3\n"
            "strategies 2\n"
            "row 0 1.0 2.0\n"
            "row 1 3.0 4.0\n"),
        std::runtime_error);
}

// =============================================================================
// 19. Error: unrecognised line (constexpr parse)
// =============================================================================

TEST(CostTableIO, ErrorUnrecognisedLine) {
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "strategies 2\n"
            "foo bar\n"
            "row 0 1.0 2.0\n"
            "row 1 3.0 4.0\n"),
        std::runtime_error);
}

// =============================================================================
// 20. Error: capacity overflow — elements
// =============================================================================

TEST(CostTableIO, ErrorCapacityOverflowElements) {
    // ct_cap::tiny has max_elements = 8; requesting 100 should fail.
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 100\n"
            "strategies 2\n"),
        std::runtime_error);
}

// =============================================================================
// 21. Error: capacity overflow — strategies
// =============================================================================

TEST(CostTableIO, ErrorCapacityOverflowStrategies) {
    // ct_cap::tiny has max_strategies = 8; requesting 100 should fail.
    EXPECT_THROW(
        (void)ct_io::parse<ct_cap::tiny>(
            "elements 2\n"
            "strategies 100\n"),
        std::runtime_error);
}

// =============================================================================
// 22. Windows \r\n line endings
// =============================================================================

constexpr auto k_crlf_text =
    "elements 2\r\n"
    "strategies 2\r\n"
    "row 0 1.0 2.0\r\n"
    "row 1 3.0 4.0\r\n";

constexpr auto k_crlf = ct_io::parse<ct_cap::tiny>(k_crlf_text);

TEST(CostTableIO, WindowsLineEndings) {
    EXPECT_EQ(k_crlf.num_elements(), 2u);
    EXPECT_EQ(k_crlf.num_strategies(), 2u);
    EXPECT_DOUBLE_EQ(k_crlf(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(k_crlf(1, 1), 4.0);
}

// =============================================================================
// 23. Tag-object read overload
// =============================================================================

TEST(CostTableIO, TagObjectRead) {
    std::istringstream iss{
        "elements 2\n"
        "strategies 2\n"
        "row 0 1.0 2.0\n"
        "row 1 3.0 4.0\n"};
    auto ct = ct_io::read(iss, ct_cap::tiny{});

    // Verify correct type.
    static_assert(std::is_same_v<
        decltype(ct), cost_table<ct_cap::tiny>>);

    EXPECT_EQ(ct.num_elements(), 2u);
    EXPECT_EQ(ct.num_strategies(), 2u);
    EXPECT_DOUBLE_EQ(ct(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(ct(1, 1), 4.0);
}
