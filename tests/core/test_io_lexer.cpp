// tests/core/test_io_lexer.cpp — Tests for core/io_lexer.h
//
// Verifies the constexpr lexing primitives in ctdp::io work standalone,
// independent of any graph types.

#include <ctdp/core/io_lexer.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace lex = ctdp::io;

// ---- Compile-time verification ----
static_assert(lex::at_eol("\n", 0));
static_assert(lex::at_eol("", 0));
static_assert(lex::at_eol("abc\n", 3));
static_assert(lex::at_eol("# comment", 0));
static_assert(!lex::at_eol("x", 0));

static_assert(lex::skip_hspace("  x", 0) == 2);
static_assert(lex::skip_hspace("\tx", 0) == 1);
static_assert(lex::skip_hspace("x", 0) == 0);
static_assert(lex::skip_hspace("  ", 0) == 2);

static_assert(lex::skip_to_eol("ab\ncd", 0) == 3);
static_assert(lex::skip_to_eol("ab", 0) == 2);
static_assert(lex::skip_to_eol("\n", 0) == 1);

static_assert(lex::starts_with("nodes 5", 0, "nodes"));
static_assert(lex::starts_with("  edge 0 1", 2, "edge"));
static_assert(!lex::starts_with("edge", 0, "nodes"));
static_assert(!lex::starts_with("no", 0, "nodes"));

static_assert(lex::parse_uint("42abc", 0).first == 42);
static_assert(lex::parse_uint("42abc", 0).second == 2);
static_assert(lex::parse_uint("0", 0).first == 0);
static_assert(lex::parse_uint("12345", 0).first == 12345);

static_assert(lex::parse_double("3.14x", 0).first > 3.13);
static_assert(lex::parse_double("3.14x", 0).first < 3.15);
static_assert(lex::parse_double("-2.5", 0).first > -2.51);
static_assert(lex::parse_double("-2.5", 0).first < -2.49);
static_assert(lex::parse_double("7", 0).first == 7.0);

// ---- Runtime tests ----

TEST(IOLexer, SkipHspace) {
    EXPECT_EQ(lex::skip_hspace("  hello", 0), 2u);
    EXPECT_EQ(lex::skip_hspace("\t\thello", 0), 2u);
    EXPECT_EQ(lex::skip_hspace("hello", 0), 0u);
    EXPECT_EQ(lex::skip_hspace("  ", 0), 2u);
    // Does not skip newlines.
    EXPECT_EQ(lex::skip_hspace("\nhello", 0), 0u);
}

TEST(IOLexer, SkipToEol) {
    EXPECT_EQ(lex::skip_to_eol("abc\ndef", 0), 4u);
    EXPECT_EQ(lex::skip_to_eol("abc", 0), 3u);
    EXPECT_EQ(lex::skip_to_eol("\n", 0), 1u);
    EXPECT_EQ(lex::skip_to_eol("", 0), 0u);
}

TEST(IOLexer, AtEol) {
    EXPECT_TRUE(lex::at_eol("\n", 0));
    EXPECT_TRUE(lex::at_eol("\r", 0));
    EXPECT_TRUE(lex::at_eol("# comment", 0));
    EXPECT_TRUE(lex::at_eol("", 0));
    EXPECT_TRUE(lex::at_eol("x", 1));
    EXPECT_FALSE(lex::at_eol("x", 0));
}

TEST(IOLexer, ParseUint) {
    auto [v1, p1] = lex::parse_uint("42abc", 0);
    EXPECT_EQ(v1, 42u);
    EXPECT_EQ(p1, 2u);

    auto [v2, p2] = lex::parse_uint("0 rest", 0);
    EXPECT_EQ(v2, 0u);
    EXPECT_EQ(p2, 1u);

    EXPECT_THROW(lex::parse_uint("abc", 0), std::runtime_error);
    EXPECT_THROW(lex::parse_uint("", 0), std::runtime_error);
}

TEST(IOLexer, ParseDouble) {
    auto [v1, p1] = lex::parse_double("3.14x", 0);
    EXPECT_NEAR(v1, 3.14, 1e-9);
    EXPECT_EQ(p1, 4u);

    auto [v2, p2] = lex::parse_double("-2.5 rest", 0);
    EXPECT_NEAR(v2, -2.5, 1e-9);
    EXPECT_EQ(p2, 4u);

    auto [v3, p3] = lex::parse_double("7", 0);
    EXPECT_DOUBLE_EQ(v3, 7.0);
    EXPECT_EQ(p3, 1u);

    EXPECT_THROW(lex::parse_double("", 0), std::runtime_error);
    EXPECT_THROW(lex::parse_double("abc", 0), std::runtime_error);
}

TEST(IOLexer, StartsWith) {
    EXPECT_TRUE(lex::starts_with("nodes 5", 0, "nodes"));
    EXPECT_TRUE(lex::starts_with("  edge 0 1", 2, "edge"));
    EXPECT_FALSE(lex::starts_with("edge", 0, "nodes"));
    EXPECT_FALSE(lex::starts_with("no", 0, "nodes"));
    EXPECT_TRUE(lex::starts_with("", 0, ""));
}
