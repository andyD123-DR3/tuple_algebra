// tests/test_interval_split_space.cpp
// Tests for solver/spaces/interval_split_space.h

#include "ctdp/solver/spaces/interval_split_space.h"
#include <gtest/gtest.h>

using namespace ctdp;

// --- Construction ---
static_assert(interval_split_space<8>{.n = 6}.n == 6);
static_assert(interval_split_space<8>{.n = 6}.max_size == 8);
static_assert(interval_split_space<4>{.n = 1}.n == 1);

TEST(IntervalSplitSpace, Construction) {
    constexpr auto space = interval_split_space<8>{.n = 6};
    EXPECT_EQ(space.n, 6u);
    EXPECT_EQ(space.max_size, 8u);
}

// --- Size (Catalan) ---
static_assert(interval_split_space<4>{.n = 0}.size() == 1);
static_assert(interval_split_space<4>{.n = 1}.size() == 1);
static_assert(interval_split_space<8>{.n = 3}.size() == 2);  // C(2) = 2

TEST(IntervalSplitSpace, Size) {
    constexpr auto s0 = interval_split_space<4>{.n = 0};
    constexpr auto s1 = interval_split_space<4>{.n = 1};
    constexpr auto s3 = interval_split_space<8>{.n = 3};
    EXPECT_EQ(s0.size(), 1u);
    EXPECT_EQ(s1.size(), 1u);
    EXPECT_EQ(s3.size(), 2u);
}

// --- Candidate split accessor ---
constexpr auto make_test_candidate() {
    interval_split_candidate<4> c{};
    c.n = 3;
    c.optimal_split[0 * 4 + 2] = 1;  // split(0, 2) = 1
    return c;
}

static_assert(make_test_candidate().split(0, 2) == 1);
static_assert(make_test_candidate().n == 3);

TEST(IntervalSplitSpace, CandidateSplitAccessor) {
    constexpr auto c = make_test_candidate();
    EXPECT_EQ(c.split(0, 2), 1u);
}

// --- Candidate equality (active region only) ---
constexpr auto make_candidate_a() {
    interval_split_candidate<4> c{};
    c.n = 2;
    c.optimal_split[0 * 4 + 1] = 0;
    return c;
}

constexpr auto make_candidate_b() {
    interval_split_candidate<4> c{};
    c.n = 2;
    c.optimal_split[0 * 4 + 1] = 0;
    c.optimal_split[2 * 4 + 3] = 99;  // outside active region
    return c;
}

constexpr auto make_candidate_c() {
    interval_split_candidate<4> c{};
    c.n = 2;
    c.optimal_split[0 * 4 + 1] = 1;  // different split
    return c;
}

static_assert(make_candidate_a() == make_candidate_b());   // same active region
static_assert(make_candidate_a() != make_candidate_c());   // different split

TEST(IntervalSplitSpace, CandidateEquality) {
    constexpr auto a = make_candidate_a();
    constexpr auto b = make_candidate_b();
    constexpr auto c = make_candidate_c();
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// --- Different n means not equal ---
constexpr auto make_candidate_n2() {
    interval_split_candidate<4> c{};
    c.n = 2;
    return c;
}

constexpr auto make_candidate_n3() {
    interval_split_candidate<4> c{};
    c.n = 3;
    return c;
}

static_assert(make_candidate_n2() != make_candidate_n3());

TEST(IntervalSplitSpace, DifferentNNotEqual) {
    constexpr auto n2 = make_candidate_n2();
    constexpr auto n3 = make_candidate_n3();
    EXPECT_NE(n2, n3);
}

// --- search_space concept ---
static_assert(search_space<interval_split_space<8>>);
static_assert(search_space<interval_split_space<16>>);

TEST(IntervalSplitSpace, SearchSpaceConcept) {
    EXPECT_TRUE(search_space<interval_split_space<8>>);
}
