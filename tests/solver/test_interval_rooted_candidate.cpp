#include "ctdp/solver/solver.h"

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using ctdp::plan;
using ctdp::solver::interval_context;
using ctdp::solver::interval_rooted_candidate;
using ctdp::solver::interval_rooted_plan;
using ctdp::solver::make_empty_interval_rooted_candidate;
using ctdp::solver::make_single_leaf_interval_rooted_candidate;
using ctdp::solver::reconstruct_interval_rooted_candidate;

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
static_assert(make_empty_interval_rooted_candidate<4>().empty());
static_assert(make_single_leaf_interval_rooted_candidate<4>().is_leaf(0, 1));

constexpr auto reconstruct_balanced_from_callback() {
    return reconstruct_interval_rooted_candidate<4>(4, [](std::size_t i, std::size_t j) constexpr {
        if (i == 0 && j == 4) return std::size_t{2};
        if (i == 0 && j == 2) return std::size_t{1};
        return std::size_t{3};
    });
}

constexpr auto reconstruct_right_skewed_from_callback() {
    return reconstruct_interval_rooted_candidate<4>(4, [](std::size_t i, [[maybe_unused]] std::size_t j) constexpr {
        return i + 1;
    });
}

constexpr auto reconstruct_from_legacy_split_candidate() {
    ctdp::interval_split_candidate<4> splits{};
    splits.n = 4;
    splits.optimal_split[0 * 4 + 3] = 1; // [0,4) -> k=2
    splits.optimal_split[0 * 4 + 1] = 0; // [0,2) -> k=1
    splits.optimal_split[2 * 4 + 3] = 2; // [2,4) -> k=3
    return reconstruct_interval_rooted_candidate(splits);
}

TEST(IntervalRootedCandidate, EmptyCandidate) {
    constexpr interval_rooted_candidate<4> c{};

    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.size(), 0u);
    EXPECT_EQ(c.leaf_count(), 0u);
    EXPECT_TRUE(c.is_legal());
    EXPECT_TRUE(c.is_canonical());
    EXPECT_FALSE(c.contains(0, 1));
}

TEST(IntervalRootedCandidate, FactoryHelpers) {
    constexpr auto empty = make_empty_interval_rooted_candidate<4>();
    constexpr auto single = make_single_leaf_interval_rooted_candidate<4>();

    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(empty.is_canonical());

    EXPECT_EQ(single.size(), 1u);
    EXPECT_TRUE(single.is_leaf(0, 1));
    EXPECT_TRUE(single.is_canonical());
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

TEST(IntervalRootedCandidate, IntervalContextOverloadsMatchCoordinateQueries) {
    constexpr auto c = make_balanced_candidate();

    constexpr interval_context root_ctx{0, 4};
    constexpr interval_context left_ctx{0, 2};
    constexpr interval_context leaf_ctx{0, 1};
    constexpr interval_context missing_ctx{0, 3};

    EXPECT_EQ(c.contains(root_ctx), c.contains(0, 4));
    EXPECT_EQ(c.contains(left_ctx), c.contains(0, 2));
    EXPECT_EQ(c.contains(leaf_ctx), c.contains(0, 1));
    EXPECT_EQ(c.contains(missing_ctx), c.contains(0, 3));

    EXPECT_EQ(c.is_internal(root_ctx), c.is_internal(0, 4));
    EXPECT_EQ(c.is_internal(left_ctx), c.is_internal(0, 2));
    EXPECT_EQ(c.is_leaf(leaf_ctx), c.is_leaf(0, 1));
    EXPECT_EQ(c.split(root_ctx), c.split(0, 4));
    EXPECT_EQ(c.left_interval(root_ctx).end(), c.left_interval(0, 4).end());
    EXPECT_EQ(c.right_interval(root_ctx).start(), c.right_interval(0, 4).start());
}

TEST(IntervalRootedCandidate, FindNodeIntervalContextOverloadIsErgonomic) {
    constexpr auto c = make_balanced_candidate();

    auto root = c.find_node(interval_context{0, 4});
    auto left = c.find_node(interval_context{0, 2});
    auto missing = c.find_node(interval_context{0, 3});

    ASSERT_TRUE(root.has_value());
    ASSERT_TRUE(left.has_value());
    EXPECT_FALSE(missing.has_value());
    EXPECT_EQ(root->split(), 2u);
    EXPECT_EQ(left->split(), 1u);
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

TEST(IntervalRootedCandidate, CanonicalizedRemovesUnreachableStorage) {
    constexpr auto noncanonical = make_noncanonical_balanced_candidate();
    constexpr auto canonical = make_balanced_candidate();

    auto normalized = noncanonical.canonicalized();

    EXPECT_EQ(normalized, canonical);
    EXPECT_TRUE(normalized.is_canonical());
    EXPECT_FALSE(noncanonical.is_canonical());
}

TEST(IntervalRootedCandidate, CanonicalizedIsNoOpForCanonicalCandidate) {
    constexpr auto canonical = make_balanced_candidate();

    auto normalized = canonical.canonicalized();

    EXPECT_EQ(normalized, canonical);
    EXPECT_TRUE(normalized.is_canonical());
}

TEST(IntervalRootedCandidate, CanonicalizedPreservesEmptyAndSingleLeafCandidates) {
    constexpr auto empty = make_empty_interval_rooted_candidate<4>();
    constexpr auto single = make_single_leaf_interval_rooted_candidate<4>();

    auto normalized_empty = empty.canonicalized();
    auto normalized_single = single.canonicalized();

    EXPECT_EQ(normalized_empty, empty);
    EXPECT_TRUE(normalized_empty.is_canonical());
    EXPECT_EQ(normalized_single, single);
    EXPECT_TRUE(normalized_single.is_canonical());
}

TEST(IntervalRootedCandidate, DifferentShapesAreNotEqual) {
    constexpr auto balanced = make_balanced_candidate();
    constexpr auto skewed = make_right_skewed_candidate();

    EXPECT_NE(balanced, skewed);
}

TEST(IntervalRootedCandidate, CallbackReconstructionBuildsCanonicalBalancedTree) {
    constexpr auto reconstructed = reconstruct_balanced_from_callback();
    constexpr auto expected = make_balanced_candidate();

    EXPECT_EQ(reconstructed, expected);
    EXPECT_TRUE(reconstructed.is_legal());
    EXPECT_TRUE(reconstructed.is_canonical());
    EXPECT_EQ(reconstructed.split(0, 4), 2u);
    EXPECT_EQ(reconstructed.split(0, 2), 1u);
    EXPECT_EQ(reconstructed.split(2, 4), 3u);
}

TEST(IntervalRootedCandidate, CallbackReconstructionBuildsCanonicalRightSkewedTree) {
    constexpr auto reconstructed = reconstruct_right_skewed_from_callback();
    constexpr auto expected = make_right_skewed_candidate();

    EXPECT_EQ(reconstructed, expected);
    EXPECT_TRUE(reconstructed.is_canonical());
}

TEST(IntervalRootedCandidate, ReconstructionFromLegacySplitCandidate) {
    constexpr auto reconstructed = reconstruct_from_legacy_split_candidate();
    constexpr auto expected = make_balanced_candidate();

    EXPECT_EQ(reconstructed, expected);
    EXPECT_TRUE(reconstructed.is_canonical());
}

TEST(IntervalRootedCandidate, ReconstructionFromIntervalDpPlan) {
    constexpr std::array<std::size_t, 5> dims{40, 20, 30, 10, 30};
    constexpr auto dp_result = ctdp::interval_dp(ctdp::interval_split_space<5>{.n = 4},
                                                 ctdp::make_chain_cost(dims));

    auto reconstructed = reconstruct_interval_rooted_candidate(dp_result);

    EXPECT_TRUE(reconstructed.is_legal());
    EXPECT_TRUE(reconstructed.is_canonical());
    EXPECT_EQ(reconstructed.size(), 4u);
    EXPECT_EQ(reconstructed.split(0, 4), 3u);
    EXPECT_EQ(reconstructed.split(0, 3), 1u);
    EXPECT_EQ(reconstructed.split(1, 3), 2u);
}

TEST(IntervalRootedCandidate, EmptyPreorderTraversalIsEmpty) {
    constexpr auto empty = make_empty_interval_rooted_candidate<4>();
    auto preorder = empty.preorder();

    EXPECT_EQ(preorder.count, 0u);
    EXPECT_EQ(preorder.begin(), preorder.end());
}

TEST(IntervalRootedCandidate, BalancedPreorderTraversalIsDeterministic) {
    constexpr auto candidate = make_balanced_candidate();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.preorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 4}, {0, 2}, {0, 1}, {1, 2}, {2, 4}, {2, 3}, {3, 4}
    };

    EXPECT_EQ(seen, expected);
}

TEST(IntervalRootedCandidate, RightSkewedPreorderTraversalIsDeterministic) {
    constexpr auto candidate = make_right_skewed_candidate();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.preorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 4}, {0, 1}, {1, 4}, {1, 2}, {2, 4}, {2, 3}, {3, 4}
    };

    EXPECT_EQ(seen, expected);
}

TEST(IntervalRootedCandidate, ReconstructedCandidatePreorderMatchesCanonicalOrder) {
    constexpr auto candidate = reconstruct_balanced_from_callback();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.preorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 4}, {0, 2}, {0, 1}, {1, 2}, {2, 4}, {2, 3}, {3, 4}
    };

    EXPECT_EQ(seen, expected);
}

TEST(IntervalRootedCandidate, EmptyInorderAndPostorderTraversalsAreEmpty) {
    constexpr auto empty = make_empty_interval_rooted_candidate<4>();
    auto inorder = empty.inorder();
    auto postorder = empty.postorder();

    EXPECT_EQ(inorder.count, 0u);
    EXPECT_EQ(inorder.begin(), inorder.end());
    EXPECT_EQ(postorder.count, 0u);
    EXPECT_EQ(postorder.begin(), postorder.end());
}

TEST(IntervalRootedCandidate, BalancedInorderTraversalIsDeterministic) {
    constexpr auto candidate = make_balanced_candidate();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.inorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 1}, {0, 2}, {1, 2}, {0, 4}, {2, 3}, {2, 4}, {3, 4}
    };

    EXPECT_EQ(seen, expected);
}

TEST(IntervalRootedCandidate, BalancedPostorderTraversalIsDeterministic) {
    constexpr auto candidate = make_balanced_candidate();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.postorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 1}, {1, 2}, {0, 2}, {2, 3}, {3, 4}, {2, 4}, {0, 4}
    };

    EXPECT_EQ(seen, expected);
    ASSERT_FALSE(seen.empty());
    EXPECT_EQ(seen.back(), std::make_pair(0u, 4u));
}

TEST(IntervalRootedCandidate, RightSkewedPostorderTraversalIsDeterministic) {
    constexpr auto candidate = make_right_skewed_candidate();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.postorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {2, 4}, {1, 4}, {0, 4}
    };

    EXPECT_EQ(seen, expected);
}

TEST(IntervalRootedCandidate, ReconstructedCandidatePostorderMatchesCanonicalOrder) {
    constexpr auto candidate = reconstruct_balanced_from_callback();

    std::vector<std::pair<std::size_t, std::size_t>> seen;
    for (auto const& node : candidate.postorder()) {
        seen.emplace_back(node.interval().start(), node.interval().end());
    }

    std::vector<std::pair<std::size_t, std::size_t>> expected{
        {0, 1}, {1, 2}, {0, 2}, {2, 3}, {3, 4}, {2, 4}, {0, 4}
    };

    EXPECT_EQ(seen, expected);
}

} // namespace








