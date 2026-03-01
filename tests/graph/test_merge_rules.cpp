// tests/graph/test_merge_rules.cpp — Tests for merge policy vocabulary
//
// Tests: all 10 named policies, merge_property group reduction,
// constexpr proofs (static_assert), edge cases.

#include <ctdp/graph/merge_rules.h>
#include <ctdp/graph/property_map.h>
#include <ctdp/graph/graph_concepts.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace ctdp::graph;

// =========================================================================
// Policy unit tests — constexpr proofs
// =========================================================================

namespace {

// max_of
static_assert(merge::max_of{}(3, 7) == 7);
static_assert(merge::max_of{}(9, 2) == 9);
static_assert(merge::max_of{}(5, 5) == 5);
static_assert(merge::max_of{}(0, 0) == 0);
static_assert(merge::max_of{}(1.5, 2.5) == 2.5);

// min_of
static_assert(merge::min_of{}(3, 7) == 3);
static_assert(merge::min_of{}(9, 2) == 2);
static_assert(merge::min_of{}(5, 5) == 5);
static_assert(merge::min_of{}(1.5, 2.5) == 1.5);

// sum
static_assert(merge::sum{}(3, 7) == 10);
static_assert(merge::sum{}(0, 0) == 0);
static_assert(merge::sum{}(100, 200) == 300);

// union_of (bitwise OR)
static_assert(merge::union_of{}(0b1010u, 0b0110u) == 0b1110u);
static_assert(merge::union_of{}(0u, 0u) == 0u);
static_assert(merge::union_of{}(0xFFu, 0x00u) == 0xFFu);

// intersect (bitwise AND)
static_assert(merge::intersect{}(0b1010u, 0b0110u) == 0b0010u);
static_assert(merge::intersect{}(0xFFu, 0x0Fu) == 0x0Fu);
static_assert(merge::intersect{}(0u, 0xFFu) == 0u);

// logical_and
static_assert(merge::logical_and{}(true, true) == true);
static_assert(merge::logical_and{}(true, false) == false);
static_assert(merge::logical_and{}(false, true) == false);
static_assert(merge::logical_and{}(false, false) == false);

// logical_or
static_assert(merge::logical_or{}(true, true) == true);
static_assert(merge::logical_or{}(true, false) == true);
static_assert(merge::logical_or{}(false, true) == true);
static_assert(merge::logical_or{}(false, false) == false);

// stricter (alias for min_of)
static_assert(merge::stricter{}(10, 20) == 10);
static_assert(merge::stricter{}(20, 10) == 10);

// first
static_assert(merge::first{}(42, 99) == 42);
static_assert(merge::first{}(1, 2) == 1);

// second
static_assert(merge::second{}(42, 99) == 99);
static_assert(merge::second{}(1, 2) == 2);

} // anonymous namespace

// =========================================================================
// Policy runtime tests
// =========================================================================

TEST(MergeRulesTest, MaxOf) {
    EXPECT_EQ(merge::max_of{}(3, 7), 7);
    EXPECT_EQ(merge::max_of{}(9, 2), 9);
    EXPECT_DOUBLE_EQ(merge::max_of{}(1.5, 2.5), 2.5);
}

TEST(MergeRulesTest, MinOf) {
    EXPECT_EQ(merge::min_of{}(3, 7), 3);
    EXPECT_EQ(merge::min_of{}(9, 2), 2);
}

TEST(MergeRulesTest, Sum) {
    EXPECT_EQ(merge::sum{}(100, 200), 300);
    EXPECT_EQ(merge::sum{}(0, 0), 0);
}

TEST(MergeRulesTest, UnionOf) {
    EXPECT_EQ(merge::union_of{}(0b1010u, 0b0110u), 0b1110u);
}

TEST(MergeRulesTest, Intersect) {
    EXPECT_EQ(merge::intersect{}(0b1010u, 0b0110u), 0b0010u);
}

TEST(MergeRulesTest, LogicalAnd) {
    EXPECT_TRUE(merge::logical_and{}(true, true));
    EXPECT_FALSE(merge::logical_and{}(true, false));
}

TEST(MergeRulesTest, LogicalOr) {
    EXPECT_TRUE(merge::logical_or{}(false, true));
    EXPECT_FALSE(merge::logical_or{}(false, false));
}

TEST(MergeRulesTest, First) {
    EXPECT_EQ(merge::first{}(42, 99), 42);
}

TEST(MergeRulesTest, Second) {
    EXPECT_EQ(merge::second{}(42, 99), 99);
}

TEST(MergeRulesTest, Fail) {
    EXPECT_THROW((void)merge::fail{}(1, 2), std::logic_error);
}

// =========================================================================
// merge_property — group reduction tests
// =========================================================================

namespace {

// Set up: 4 nodes, 2 groups: {0,1}=group0, {2,3}=group1
// Values: [10, 20, 30, 40]
constexpr auto make_test_pmap() {
    property_map<std::size_t, 8> pmap(4, 0);
    pmap[0] = 10;
    pmap[1] = 20;
    pmap[2] = 30;
    pmap[3] = 40;
    return pmap;
}

constexpr auto make_test_groups() {
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0;
    groups[1] = 0;
    groups[2] = 1;
    groups[3] = 1;
    return groups;
}

constexpr auto test_pmap = make_test_pmap();
constexpr auto test_groups = make_test_groups();

// merge with sum: group0 = 10+20 = 30, group1 = 30+40 = 70
constexpr auto merged_sum = merge_property(test_pmap, test_groups,
    std::size_t{2}, merge::sum{});
static_assert(merged_sum[0] == 30);
static_assert(merged_sum[1] == 70);

// merge with max_of: group0 = max(10,20) = 20, group1 = max(30,40) = 40
constexpr auto merged_max = merge_property(test_pmap, test_groups,
    std::size_t{2}, merge::max_of{});
static_assert(merged_max[0] == 20);
static_assert(merged_max[1] == 40);

// merge with min_of: group0 = min(10,20) = 10, group1 = min(30,40) = 30
constexpr auto merged_min = merge_property(test_pmap, test_groups,
    std::size_t{2}, merge::min_of{});
static_assert(merged_min[0] == 10);
static_assert(merged_min[1] == 30);

// merge with first: group0 = 10 (first encountered), group1 = 30
constexpr auto merged_first = merge_property(test_pmap, test_groups,
    std::size_t{2}, merge::first{});
static_assert(merged_first[0] == 10);
static_assert(merged_first[1] == 30);

// merge with second: group0 = 20 (last encountered), group1 = 40
constexpr auto merged_second = merge_property(test_pmap, test_groups,
    std::size_t{2}, merge::second{});
static_assert(merged_second[0] == 20);
static_assert(merged_second[1] == 40);

// Boolean property: [true, false, true, true]
constexpr auto make_bool_pmap() {
    property_map<bool, 8> pmap(4, true);
    pmap[1] = false;
    return pmap;
}
constexpr auto bool_pmap = make_bool_pmap();

// logical_and: group0 = true && false = false, group1 = true && true = true
constexpr auto merged_and = merge_property(bool_pmap, test_groups,
    std::size_t{2}, merge::logical_and{});
static_assert(merged_and[0] == false);
static_assert(merged_and[1] == true);

// logical_or: group0 = true || false = true, group1 = true || true = true
constexpr auto merged_or = merge_property(bool_pmap, test_groups,
    std::size_t{2}, merge::logical_or{});
static_assert(merged_or[0] == true);
static_assert(merged_or[1] == true);

// Single-node groups (identity contraction): each node its own group
constexpr auto make_id_groups() {
    property_map<std::uint16_t, 8> groups(4, 0);
    groups[0] = 0; groups[1] = 1; groups[2] = 2; groups[3] = 3;
    return groups;
}
constexpr auto id_groups = make_id_groups();
constexpr auto merged_id = merge_property(test_pmap, id_groups,
    std::size_t{4}, merge::sum{});
static_assert(merged_id[0] == 10);
static_assert(merged_id[1] == 20);
static_assert(merged_id[2] == 30);
static_assert(merged_id[3] == 40);

// All-one-group: merge everything
constexpr auto all_zero_groups = property_map<std::uint16_t, 8>(4, 0);
constexpr auto merged_all = merge_property(test_pmap, all_zero_groups,
    std::size_t{1}, merge::sum{});
static_assert(merged_all[0] == 100);  // 10+20+30+40

} // anonymous namespace

TEST(MergePropertyTest, SumAcrossGroups) {
    auto merged = merge_property(test_pmap, test_groups,
        std::size_t{2}, merge::sum{});
    EXPECT_EQ(merged[0], 30u);
    EXPECT_EQ(merged[1], 70u);
}

TEST(MergePropertyTest, MaxAcrossGroups) {
    auto merged = merge_property(test_pmap, test_groups,
        std::size_t{2}, merge::max_of{});
    EXPECT_EQ(merged[0], 20u);
    EXPECT_EQ(merged[1], 40u);
}

TEST(MergePropertyTest, MinAcrossGroups) {
    auto merged = merge_property(test_pmap, test_groups,
        std::size_t{2}, merge::min_of{});
    EXPECT_EQ(merged[0], 10u);
    EXPECT_EQ(merged[1], 30u);
}

TEST(MergePropertyTest, BooleanAndAcrossGroups) {
    auto merged = merge_property(bool_pmap, test_groups,
        std::size_t{2}, merge::logical_and{});
    EXPECT_FALSE(merged[0]);
    EXPECT_TRUE(merged[1]);
}

TEST(MergePropertyTest, SingletonGroups) {
    auto merged = merge_property(test_pmap, id_groups,
        std::size_t{4}, merge::sum{});
    EXPECT_EQ(merged[0], 10u);
    EXPECT_EQ(merged[1], 20u);
    EXPECT_EQ(merged[2], 30u);
    EXPECT_EQ(merged[3], 40u);
}

TEST(MergePropertyTest, AllOneGroup) {
    auto merged = merge_property(test_pmap, all_zero_groups,
        std::size_t{1}, merge::sum{});
    EXPECT_EQ(merged[0], 100u);
}

TEST(MergePropertyTest, FailPolicyThrows) {
    EXPECT_THROW(
        (void)merge_property(test_pmap, test_groups, std::size_t{2}, merge::fail{}),
        std::logic_error);
}

TEST(MergePropertyTest, BitfieldUnion) {
    property_map<unsigned, 8> flags(4, 0u);
    flags[0] = 0b0001u;
    flags[1] = 0b0010u;
    flags[2] = 0b0100u;
    flags[3] = 0b1000u;

    auto merged = merge_property(flags, test_groups,
        std::size_t{2}, merge::union_of{});
    EXPECT_EQ(merged[0], 0b0011u);
    EXPECT_EQ(merged[1], 0b1100u);
}

TEST(MergePropertyTest, BitfieldIntersect) {
    property_map<unsigned, 8> flags(4, 0u);
    flags[0] = 0b1111u;
    flags[1] = 0b0011u;
    flags[2] = 0b1100u;
    flags[3] = 0b1110u;

    auto merged = merge_property(flags, test_groups,
        std::size_t{2}, merge::intersect{});
    EXPECT_EQ(merged[0], 0b0011u);  // 1111 & 0011
    EXPECT_EQ(merged[1], 0b1100u);  // 1100 & 1110
}
