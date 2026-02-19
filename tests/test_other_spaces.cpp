// tests/test_other_spaces.cpp
// Tests for cartesian_space and permutation_space.

#include "ctdp/solver/spaces/cartesian_space.h"
#include "ctdp/solver/spaces/permutation_space.h"
#include <gtest/gtest.h>

using namespace ctdp;

// =====================================================================
// cartesian_space
// =====================================================================

enum class Color { Red, Green, Blue };
enum class Size  { S, M, L };

constexpr auto make_cart_space() {
    cartesian_space<4, Color, Size> s{};
    std::get<0>(s.axes) = {Color::Red, Color::Green, Color::Blue};
    std::get<1>(s.axes) = {Size::S, Size::M, Size::L};
    return s;
}

// --- Size ---
constexpr auto cart_size() { return make_cart_space().size(); }
static_assert(cart_size() == 9);  // 3 × 3

TEST(CartesianSpace, Size) {
    EXPECT_EQ(cart_size(), 9u);
}

// --- Enumeration ---
constexpr auto cart_count() {
    auto s = make_cart_space();
    std::size_t count = 0;
    s.enumerate([&](auto const&) { ++count; });
    return count;
}

static_assert(cart_count() == 9);

TEST(CartesianSpace, Enumeration) {
    EXPECT_EQ(cart_count(), 9u);
}

// --- Neighbours ---
constexpr auto cart_neighbour_count() {
    auto s = make_cart_space();
    auto c = std::tuple{Color::Red, Size::M};
    std::size_t count = 0;
    s.neighbours(c, [&](auto const&) { ++count; });
    return count;
}

// Neighbours: 2 other colors + 2 other sizes = 4
static_assert(cart_neighbour_count() == 4);

TEST(CartesianSpace, Neighbours) {
    EXPECT_EQ(cart_neighbour_count(), 4u);
}

// --- Concept ---
using CartType = cartesian_space<4, Color, Size>;
static_assert(search_space<CartType>);

TEST(CartesianSpace, Concept) {
    EXPECT_TRUE(search_space<CartType>);
}

// =====================================================================
// permutation_space
// =====================================================================

// --- Size ---
static_assert(permutation_space<1>::size() == 1);
static_assert(permutation_space<2>::size() == 2);
static_assert(permutation_space<3>::size() == 6);
static_assert(permutation_space<4>::size() == 24);

TEST(PermutationSpace, Size) {
    EXPECT_EQ(permutation_space<4>::size(), 24u);
    EXPECT_EQ(permutation_space<5>::size(), 120u);
}

// --- Identity ---
static_assert(permutation_space<3>::identity() == std::array<std::size_t, 3>{0, 1, 2});

TEST(PermutationSpace, Identity) {
    constexpr auto id = permutation_space<3>::identity();
    EXPECT_EQ(id[0], 0u);
    EXPECT_EQ(id[1], 1u);
    EXPECT_EQ(id[2], 2u);
}

// --- Enumeration ---
constexpr auto perm_count_3() {
    permutation_space<3> s{};
    std::size_t count = 0;
    s.enumerate([&](auto const&) { ++count; });
    return count;
}

static_assert(perm_count_3() == 6);

TEST(PermutationSpace, Enumeration) {
    EXPECT_EQ(perm_count_3(), 6u);
}

// --- Neighbours (swap pairs) ---
constexpr auto perm_neighbour_count() {
    permutation_space<4> s{};
    auto id = permutation_space<4>::identity();
    std::size_t count = 0;
    s.neighbours(id, [&](auto const&) { ++count; });
    return count;
}

// (4 choose 2) = 6 swap neighbours
static_assert(perm_neighbour_count() == 6);

TEST(PermutationSpace, Neighbours) {
    EXPECT_EQ(perm_neighbour_count(), 6u);
}

// --- Neighbours exclude original ---
constexpr auto perm_no_self_neighbour() {
    permutation_space<3> s{};
    auto id = permutation_space<3>::identity();
    bool found_self = false;
    s.neighbours(id, [&](auto const& n) {
        if (n == id) found_self = true;
    });
    return !found_self;
}

static_assert(perm_no_self_neighbour());

TEST(PermutationSpace, NeighboursExcludeOriginal) {
    EXPECT_TRUE(perm_no_self_neighbour());
}

// --- Concept ---
static_assert(search_space<permutation_space<4>>);

TEST(PermutationSpace, Concept) {
    EXPECT_TRUE(search_space<permutation_space<4>>);
}
