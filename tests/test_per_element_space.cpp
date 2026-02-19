// tests/test_per_element_space.cpp
// Tests for solver/spaces/per_element_space.h

#include "ctdp/solver/spaces/per_element_space.h"
#include <gtest/gtest.h>

using namespace ctdp;

enum class Strat { Fast, Small, Safe };
using PES3x3 = per_element_space<Strat, 3, 3>;
using PES2x2 = per_element_space<Strat, 2, 2>;

// --- Construction ---
constexpr auto make_space_3x3() {
    return PES3x3{.strategies = {Strat::Fast, Strat::Small, Strat::Safe}};
}

static_assert(make_space_3x3().dimension == 3);
static_assert(make_space_3x3().branching == 3);

TEST(PerElementSpace, Construction) {
    constexpr auto s = make_space_3x3();
    EXPECT_EQ(s.dimension, 3u);
    EXPECT_EQ(s.branching, 3u);
}

// --- Size ---
static_assert(make_space_3x3().size() == 27);  // 3^3

constexpr auto make_space_2x2() {
    return PES2x2{.strategies = {Strat::Fast, Strat::Small}};
}
static_assert(make_space_2x2().size() == 4);  // 2^2

TEST(PerElementSpace, Size) {
    constexpr auto s = make_space_3x3();
    EXPECT_EQ(s.size(), 27u);
}

// --- Strategy access ---
static_assert(make_space_3x3().strategy(0) == Strat::Fast);
static_assert(make_space_3x3().strategy(1) == Strat::Small);
static_assert(make_space_3x3().strategy(2) == Strat::Safe);

TEST(PerElementSpace, StrategyAccess) {
    constexpr auto s = make_space_3x3();
    EXPECT_EQ(s.strategy(0), Strat::Fast);
    EXPECT_EQ(s.strategy(2), Strat::Safe);
}

// --- Enumeration ---
constexpr auto count_enumerated_2x2() {
    auto s = make_space_2x2();
    std::size_t count = 0;
    s.enumerate([&](auto const&) { ++count; });
    return count;
}

static_assert(count_enumerated_2x2() == 4);

TEST(PerElementSpace, Enumeration) {
    constexpr auto count = count_enumerated_2x2();
    EXPECT_EQ(count, 4u);
}

// --- Concept satisfaction ---
static_assert(search_space<PES3x3>);
static_assert(factored_space<PES3x3>);

TEST(PerElementSpace, Concepts) {
    EXPECT_TRUE(search_space<PES3x3>);
    EXPECT_TRUE(factored_space<PES3x3>);
}
