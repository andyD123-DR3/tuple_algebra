#include "ctdp/solver/solver.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using ctdp::plan;
using ctdp::solver::interval_context;
using ctdp::solver::interval_rooted_candidate;
using ctdp::solver::interval_rooted_plan;

template<std::size_t MaxN>
constexpr std::size_t slot(std::size_t i, std::size_t j) {
    return i * (MaxN + 1) + j;
}

template<std::size_t MaxN>
constexpr void set_leaf(interval_rooted_candidate<MaxN>& c, std::size_t i) {
    c.split_or_tag[slot<MaxN>(i, i + 1)] = interval_rooted_candidate<MaxN>::leaf_code;
}

template<std::size_t MaxN>
constexpr void set_split(interval_rooted_candidate<MaxN>& c,
                         std::size_t i,
                         std::size_t j,
                         std::size_t k) {
    c.split_or_tag[slot<MaxN>(i, j)] = k + 2;
}

constexpr auto make_single_leaf_candidate() {
    interval_rooted_candidate<1> c{};
    c.n = 1;
    set_leaf(c, 0);
    return c;
}

constexpr auto make_balanced_candidate() {
    interval_rooted_candidate<4> c{};
    c.n = 4;
    set_split(c, 0, 4, 2);
    set_split(c, 0, 2, 1);
    set_split(c, 2, 4, 3);
    set_leaf(c, 0);
    set_leaf(c, 1);
    set_leaf(c, 2);
    set_leaf(c, 3);
    return c;
}

constexpr auto make_right_skewed_candidate() {
    interval_rooted_candidate<4> c{};
    c.n = 4;
    set_split(c, 0, 4, 1);
    set_split(c, 1, 4, 2);
    set_split(c, 2, 4, 3);
    set_leaf(c, 0);
    set_leaf(c, 1);
    set_leaf(c, 2);
    set_leaf(c, 3);
    return c;
}

constexpr auto make_noncanonical_balanced_candidate() {
    auto c = make_balanced_candidate();
    c.split_or_tag[slot<4>(0, 3)] = 3; // stray unreachable split k=1
    return c;
}

constexpr auto make_illegal_missing_child_candidate() {
    interval_rooted_candidate<4> c{};
    c.n = 4;
    set_split(c, 0, 4, 2);
    set_split(c, 0, 2, 1);
    set_leaf(c, 0);
    set_leaf(c, 1);
    return c;
}

constexpr auto make_illegal_large_leaf_candidate() {
    interval_rooted_candidate<2> c{};
    c.n = 2;
    c.split_or_tag[slot<2>(0, 2)] = interval_rooted_candidate<2>::leaf_code;
    return c;
}

static_assert(make_single_leaf_candidate().is_legal());
static_assert(make_balanced_candidate().is_legal());
static_assert(make_right_skewed_candidate().is_legal());
static_assert(make_balanced_candidate().contains(0, 4));
static_assert(make_balanced_candidate().is_internal(0, 4));
static_assert(make_single_leaf_candidate().is_leaf(0, 1));
static_assert(std::is_same_v<interval_rooted_plan<4>, plan<interval_rooted_candidate<4>>>);

TEST(IntervalRootedCandidate, EmptyCandidate) {
    constexpr interval_rooted_candidate<4> c{};

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);
    EXPECT_EQ(c.leaf_count(), 0u);
    EXPECT_TRUE(c.is_legal());
    EXPECT_TRUE(c.is_canonical());
    EXPECT_FALSE(c.contains(0, 1));
}

TEST(IntervalRootedCandidate, SingleLeafCandidate) {
    constexpr auto c = make_single_leaf_candidate();

    EXPECT_FALSE(c.empty());
    EXPECT_EQ(c.size(), 1u);
    EXPECT_EQ(c.root_interval().start(), 0u);
    EXPECT_EQ(c.root_interval().end(), 1u);
    EXPECT_TRUE(c.contains(0, 1));
    EXPECT_TRUE(c.is_leaf(0, 1));
    EXPECT_FALSE(c.is_internal(0, 1));
    EXPECT_TRUE(c.is_legal());
    EXPECT_TRUE(c.is_canonical());
}

TEST(IntervalRootedCandidate, BalancedCandidateQueriesAndViews) {
    constexpr auto c = make_balanced_candidate();

    ASSERT_TRUE(c.is_legal());
    ASSERT_TRUE(c.is_canonical());
    EXPECT_TRUE(c.contains(0, 4));
    EXPECT_TRUE(c.contains(0, 2));
    EXPECT_TRUE(c.contains(2, 4));
    EXPECT_TRUE(c.contains(0, 1));
    EXPECT_TRUE(c.contains(1, 2));
    EXPECT_TRUE(c.contains(2, 3));
    EXPECT_TRUE(c.contains(3, 4));
    EXPECT_FALSE(c.contains(0, 3));

    EXPECT_EQ(c.split(0, 4), 2u);
    EXPECT_EQ(c.left_interval(0, 4).start(), 0u);
    EXPECT_EQ(c.left_interval(0, 4).end(), 2u);
    EXPECT_EQ(c.right_interval(0, 4).start(), 2u);
    EXPECT_EQ(c.right_interval(0, 4).end(), 4u);

    auto root = c.root();
    EXPECT_EQ(root.interval().start(), 0u);
    EXPECT_EQ(root.interval().end(), 4u);
    EXPECT_TRUE(root.is_internal());
    EXPECT_EQ(root.split(), 2u);

    auto left = root.left();
    auto right = root.right();
    EXPECT_EQ(left.interval().start(), 0u);
    EXPECT_EQ(left.interval().end(), 2u);
    EXPECT_EQ(right.interval().start(), 2u);
    EXPECT_EQ(right.interval().end(), 4u);
    EXPECT_EQ(left.split(), 1u);
    EXPECT_EQ(right.split(), 3u);
}

TEST(IntervalRootedCandidate, RightSkewedCandidateIsLegal) {
    constexpr auto c = make_right_skewed_candidate();

    EXPECT_TRUE(c.is_legal());
    EXPECT_TRUE(c.is_canonical());
    EXPECT_EQ(c.split(0, 4), 1u);
    EXPECT_EQ(c.split(1, 4), 2u);
    EXPECT_EQ(c.split(2, 4), 3u);
}

TEST(IntervalRootedCandidate, FindNodeReturnsOptionalView) {
    constexpr auto c = make_balanced_candidate();

    auto root = c.find_node(0, 4);
    auto missing = c.find_node(0, 3);

    ASSERT_TRUE(root.has_value());
    EXPECT_FALSE(missing.has_value());
    EXPECT_TRUE(root->is_internal());
    EXPECT_EQ(root->split(), 2u);
}

TEST(IntervalRootedCandidate, IllegalCandidatesAreRejected) {
    constexpr auto missing_child = make_illegal_missing_child_candidate();
    constexpr auto large_leaf = make_illegal_large_leaf_candidate();

    EXPECT_FALSE(missing_child.is_legal());
    EXPECT_FALSE(missing_child.is_canonical());
    EXPECT_FALSE(large_leaf.is_legal());
    EXPECT_FALSE(large_leaf.is_canonical());
}

TEST(IntervalRootedCandidate, EqualityIgnoresUnreachableStorageButCanonicalityDoesNot) {
    constexpr auto canonical = make_balanced_candidate();
    constexpr auto noncanonical = make_noncanonical_balanced_candidate();

    EXPECT_EQ(canonical, noncanonical);
    EXPECT_TRUE(canonical.is_canonical());
    EXPECT_FALSE(noncanonical.is_canonical());
}

TEST(IntervalRootedCandidate, DifferentShapesAreNotEqual) {
    constexpr auto balanced = make_balanced_candidate();
    constexpr auto skewed = make_right_skewed_candidate();

    EXPECT_NE(balanced, skewed);
}

} // namespace

