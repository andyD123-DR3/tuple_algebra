// tests/test_tree_space.cpp
// Tests for ctdp/space/tree_space.h — partition-rooted hierarchy (Layer 1)
//
// Coverage:
//   Section 1: tree_point structure (R4.3)
//   Section 2: group_lanes utility (R4.2)
//   Section 3: tree_space construction and cardinality (R4.1, R4.4)
//   Section 4: tree_space enumeration (R4.5)
//   Section 5: tree_bridge encoding (R4.6)
//   Section 6: Legality filtering (R4.7)
//   Section 7: Degenerate — flat (R4.8)
//   Section 8: Edge cases + space interface (R4.13)

#include "ctdp/space/tree_space.h"
#include "ctdp/space/descriptor.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <vector>

namespace {

using namespace ctdp::space;

// ── Mock child space for generic tests ──────────────────────────────
// A minimal space with known cardinality that satisfies the interface.

struct mock_child_space {
    using point_type = std::tuple<int>;
    int num_values = 3;

    std::size_t cardinality() const {
        return static_cast<std::size_t>(num_values);
    }
    template <typename F>
    void enumerate(F&& fn) const {
        for (int i = 0; i < num_values; ++i)
            fn(point_type{i});
    }
    std::string_view space_name() const { return "mock"; }
};

// Mock factory: returns mock_child_space with cardinality = lane count in group.
auto mock_factory() {
    return [](const auto& /*root*/,
              std::size_t /*group*/,
              std::span<const std::size_t> lanes) {
        return mock_child_space{static_cast<int>(lanes.size())};
    };
}

// Fixed-cardinality factory: always returns the same cardinality.
auto fixed_factory(int card) {
    return [card](const auto&, std::size_t, std::span<const std::size_t>) {
        return mock_child_space{card};
    };
}

auto allow_all_filter() {
    return [](const auto&) { return true; };
}

// ════════════════════════════════════════════════════════════════════════
// Section 1: tree_point (R4.3)
// ════════════════════════════════════════════════════════════════════════

TEST(TreePoint, Construction) {
    partition_value<3> pv{};
    pv.labels = {0, 0, 1};

    tree_point<3, std::tuple<int>> pt{pv, {{42}, {99}}};
    EXPECT_EQ(pt.root, pv);
    EXPECT_EQ(pt.group_plans.size(), 2u);
    EXPECT_EQ(std::get<0>(pt.group_plans[0]), 42);
    EXPECT_EQ(std::get<0>(pt.group_plans[1]), 99);
}

TEST(TreePoint, NumGroups) {
    partition_value<3> pv{};
    pv.labels = {0, 0, 1};
    tree_point<3, std::tuple<int>> pt{pv, {{1}, {2}}};
    EXPECT_EQ(pt.num_groups(), 2u);
}

TEST(TreePoint, Equality) {
    partition_value<3> a{}, b{};
    a.labels = {0, 0, 1};
    b.labels = {0, 0, 1};

    tree_point<3, std::tuple<int>> p1{a, {{1}, {2}}};
    tree_point<3, std::tuple<int>> p2{b, {{1}, {2}}};
    tree_point<3, std::tuple<int>> p3{a, {{1}, {3}}};

    EXPECT_EQ(p1, p2);
    EXPECT_NE(p1, p3);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: group_lanes (R4.2)
// ════════════════════════════════════════════════════════════════════════

TEST(GroupLanes, TwoGroups) {
    partition_value<5> pv{};
    pv.labels = {0, 0, 1, 1, 2};
    std::array<std::size_t, 5> buf{};

    EXPECT_EQ(group_lanes(pv, 0, buf), 2u);
    EXPECT_EQ(buf[0], 0u);
    EXPECT_EQ(buf[1], 1u);

    EXPECT_EQ(group_lanes(pv, 1, buf), 2u);
    EXPECT_EQ(buf[0], 2u);
    EXPECT_EQ(buf[1], 3u);

    EXPECT_EQ(group_lanes(pv, 2, buf), 1u);
    EXPECT_EQ(buf[0], 4u);
}

TEST(GroupLanes, AllOneGroup) {
    partition_value<3> pv{};
    pv.labels = {0, 0, 0};
    std::array<std::size_t, 3> buf{};

    EXPECT_EQ(group_lanes(pv, 0, buf), 3u);
    EXPECT_EQ(buf[0], 0u);
    EXPECT_EQ(buf[1], 1u);
    EXPECT_EQ(buf[2], 2u);
}

TEST(GroupLanes, AllSingletons) {
    partition_value<3> pv{};
    pv.labels = {0, 1, 2};
    std::array<std::size_t, 3> buf{};

    for (std::size_t g = 0; g < 3; ++g) {
        EXPECT_EQ(group_lanes(pv, g, buf), 1u);
        EXPECT_EQ(buf[0], g);
    }
}

TEST(GroupLanes, CallbackMatchesSpan) {
    partition_value<5> pv{};
    pv.labels = {0, 0, 1, 1, 2};

    for (std::size_t g = 0; g < 3; ++g) {
        std::array<std::size_t, 5> buf{};
        auto count = group_lanes(pv, g, buf);

        std::vector<std::size_t> cb_result;
        for_each_lane_in_group(pv, g, [&](std::size_t idx) {
            cb_result.push_back(idx);
        });

        EXPECT_EQ(cb_result.size(), count);
        for (std::size_t i = 0; i < count; ++i)
            EXPECT_EQ(cb_result[i], buf[i]);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: tree_space construction and cardinality (R4.1, R4.4)
// ════════════════════════════════════════════════════════════════════════

TEST(TreeSpace, Construction) {
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(2), allow_all_filter());

    EXPECT_EQ(ts.space_name(), "test");
}

TEST(TreeSpace, Cardinality_NoFilter) {
    // N=2, Bell(2)=2 partitions: [0,0] (1 group) and [0,1] (2 groups)
    // Factory returns cardinality 2 for all groups.
    // [0,0]: 1 group of 2 lanes → product = 2
    // [0,1]: 2 groups of 1 lane each → product = 2 * 2 = 4
    // Total = 6
    auto ts = make_tree_space<2>(
        "test", make_partition<2>("g"),
        fixed_factory(2), allow_all_filter());

    EXPECT_EQ(ts.cardinality(), 6u);
}

TEST(TreeSpace, Cardinality_WithFilter) {
    // N=2, filter rejects [0,0] (one group), accepts [0,1] (two groups)
    auto filter = [](const partition_value<2>& pv) {
        return pv.num_groups() > 1;
    };
    auto ts = make_tree_space<2>(
        "test", make_partition<2>("g"),
        fixed_factory(2), filter);

    // Only [0,1] passes: 2 groups, card = 2*2 = 4
    EXPECT_EQ(ts.cardinality(), 4u);
}

TEST(TreeSpace, Cardinality_HeterogeneousChildren) {
    // Factory returns cardinality = number of lanes in the group
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        mock_factory(), allow_all_filter());

    // Bell(3) = 5 partitions:
    // [0,0,0]: 1 group of 3 → card = 3
    // [0,0,1]: groups {0,1}(2) + {2}(1) → card = 2*1 = 2
    // [0,1,0]: groups {0,2}(2) + {1}(1) → card = 2*1 = 2
    // [0,1,1]: groups {0}(1) + {1,2}(2) → card = 1*2 = 2
    // [0,1,2]: singletons → card = 1*1*1 = 1
    // Total = 3 + 2 + 2 + 2 + 1 = 10
    EXPECT_EQ(ts.cardinality(), 10u);
}

// Recording factory: logs all calls for verification
struct call_record_3 {
    partition_value<3> root;
    std::size_t group;
    std::vector<std::size_t> lanes;
};

TEST(TreeSpace, RecordingFactory) {
    // Factory records all calls
    auto records = std::make_shared<std::vector<call_record_3>>();

    auto factory = [records](
        const partition_value<3>& root,
        std::size_t group,
        std::span<const std::size_t> lanes) {
            records->push_back({root, group, {lanes.begin(), lanes.end()}});
            return mock_child_space{1};  // cardinality 1 to keep it fast
    };

    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        factory, allow_all_filter());

    ts.enumerate([](const auto&) {});  // trigger factory calls

    // Verify each recorded call matches group_lanes
    for (const auto& rec : *records) {
        std::array<std::size_t, 3> expected_buf{};
        auto expected_count = group_lanes(rec.root, rec.group, expected_buf);
        EXPECT_EQ(rec.lanes.size(), expected_count);
        for (std::size_t i = 0; i < expected_count; ++i)
            EXPECT_EQ(rec.lanes[i], expected_buf[i]);
    }

    // Should have one call per group per partition = sum of num_groups
    // Bell(3): [0,0,0]=1, [0,0,1]=2, [0,1,0]=2, [0,1,1]=2, [0,1,2]=3
    // Total = 1+2+2+2+3 = 10
    EXPECT_EQ(records->size(), 10u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: tree_space enumeration (R4.5)
// ════════════════════════════════════════════════════════════════════════

TEST(TreeSpace, Enumerate_Count) {
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(2), allow_all_filter());

    std::size_t count = 0;
    ts.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, ts.cardinality());
}

TEST(TreeSpace, Enumerate_AllValid) {
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(2), allow_all_filter());

    ts.enumerate([](const auto& pt) {
        EXPECT_TRUE(pt.root.is_canonical());
        EXPECT_EQ(pt.num_groups(), pt.root.num_groups());
        EXPECT_GE(pt.num_groups(), 1u);
        EXPECT_LE(pt.num_groups(), 3u);
    });
}

TEST(TreeSpace, Enumerate_NoDuplicates) {
    auto ts = make_tree_space<2>(
        "test", make_partition<2>("g"),
        fixed_factory(2), allow_all_filter());

    using pt_type = typename decltype(ts)::point_type;
    std::vector<pt_type> points;
    ts.enumerate([&](const auto& pt) {
        points.push_back(pt);
    });

    // Check no duplicates
    for (std::size_t i = 0; i < points.size(); ++i)
        for (std::size_t j = i + 1; j < points.size(); ++j)
            EXPECT_NE(points[i], points[j]) << "Duplicate at " << i << " and " << j;
}

TEST(TreeSpace, Enumerate_CallbackLifetime) {
    auto ts = make_tree_space<2>(
        "test", make_partition<2>("g"),
        fixed_factory(2), allow_all_filter());

    using pt_type = typename decltype(ts)::point_type;
    std::vector<pt_type> stored;
    ts.enumerate([&](const auto& pt) {
        stored.push_back(pt);
    });

    // All stored points should still be valid after enumeration
    for (const auto& pt : stored) {
        EXPECT_TRUE(pt.root.is_canonical());
        EXPECT_EQ(pt.num_groups(), pt.root.num_groups());
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 5: tree_bridge encoding (R4.6)
// ════════════════════════════════════════════════════════════════════════

// Mock bridge: encodes tuple<int> as 1 feature (the value as double)
struct mock_bridge {
    std::size_t num_features() const { return 1; }
    void write_features(const std::tuple<int>& pt,
                        std::span<double> out) const {
        out[0] = static_cast<double>(std::get<0>(pt));
    }
};

// Helper: build a tree_bridge for mock spaces
auto mock_tree_bridge() {
    auto factory = fixed_factory(2);
    auto bridge_factory = [](const mock_child_space&) {
        return mock_bridge{};
    };
    return make_tree_bridge<3>(
        make_partition<3>("g"), factory, bridge_factory, 1);
}

TEST(TreeBridge, FeatureWidth) {
    auto tb = mock_tree_bridge();
    // root pairwise: 3*(3-1)/2 = 3, child blocks: 3 * 1 = 3
    EXPECT_EQ(tb.num_features(), 6u);
}

TEST(TreeBridge, Encode_SingleGroup) {
    auto tb = mock_tree_bridge();

    partition_value<3> pv{};
    pv.labels = {0, 0, 0};
    tree_point<3, std::tuple<int>> pt{pv, {{42}}};

    auto f = tb.encode(pt);
    EXPECT_EQ(f.size(), 6u);

    // Root: all pairs co-member → all 1.0
    EXPECT_DOUBLE_EQ(f[0], 1.0);
    EXPECT_DOUBLE_EQ(f[1], 1.0);
    EXPECT_DOUBLE_EQ(f[2], 1.0);

    // Group 0: real feature (42.0)
    EXPECT_DOUBLE_EQ(f[3], 42.0);

    // Groups 1, 2: zero-filled
    EXPECT_DOUBLE_EQ(f[4], 0.0);
    EXPECT_DOUBLE_EQ(f[5], 0.0);
}

TEST(TreeBridge, Encode_MultiGroup) {
    auto tb = mock_tree_bridge();

    partition_value<3> pv{};
    pv.labels = {0, 0, 1};
    tree_point<3, std::tuple<int>> pt{pv, {{10}, {20}}};

    auto f = tb.encode(pt);

    // Root: pairs (0,1)=same, (0,2)=diff, (1,2)=diff
    EXPECT_DOUBLE_EQ(f[0], 1.0);
    EXPECT_DOUBLE_EQ(f[1], 0.0);
    EXPECT_DOUBLE_EQ(f[2], 0.0);

    // Group 0: 10.0, Group 1: 20.0, Group 2: 0.0 (unused)
    EXPECT_DOUBLE_EQ(f[3], 10.0);
    EXPECT_DOUBLE_EQ(f[4], 20.0);
    EXPECT_DOUBLE_EQ(f[5], 0.0);
}

TEST(TreeBridge, WidthStable) {
    auto tb = mock_tree_bridge();

    // 1 group
    partition_value<3> pv1{};
    pv1.labels = {0, 0, 0};
    tree_point<3, std::tuple<int>> pt1{pv1, {{1}}};

    // 3 groups
    partition_value<3> pv3{};
    pv3.labels = {0, 1, 2};
    tree_point<3, std::tuple<int>> pt3{pv3, {{1}, {2}, {3}}};

    EXPECT_EQ(tb.encode(pt1).size(), tb.encode(pt3).size());
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: Legality filtering (R4.7)
// ════════════════════════════════════════════════════════════════════════

TEST(Filter, AllPass) {
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(1), allow_all_filter());

    // Bell(3)=5 partitions, all card=1 per group → total = 5
    EXPECT_EQ(ts.cardinality(), 5u);
}

TEST(Filter, NonePass) {
    auto filter = [](const partition_value<3>&) { return false; };
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(1), filter);

    EXPECT_EQ(ts.cardinality(), 0u);

    std::size_t count = 0;
    ts.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, 0u);
}

TEST(Filter, Selective) {
    // Only allow partitions with exactly 2 groups
    auto filter = [](const partition_value<3>& pv) {
        return pv.num_groups() == 2;
    };
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(1), filter);

    // Bell(3): [0,0,1](2), [0,1,0](2), [0,1,1](2) have 2 groups
    // Each has 2 groups with card=1 → product = 1 per partition
    // Total = 3
    EXPECT_EQ(ts.cardinality(), 3u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 7: Degenerate — flat (R4.8)
// ════════════════════════════════════════════════════════════════════════

TEST(Flat, Cardinality) {
    mock_child_space inner{5};
    auto flat = make_flat_tree_space("flat", inner);
    EXPECT_EQ(flat.cardinality(), 5u);
}

TEST(Flat, EnumerateCount) {
    mock_child_space inner{4};
    auto flat = make_flat_tree_space("flat", inner);

    std::size_t count = 0;
    flat.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, 4u);
}

TEST(Flat, Projection) {
    mock_child_space inner{3};
    auto flat = make_flat_tree_space("flat", inner);

    // Collect projected child points
    std::vector<std::tuple<int>> projected;
    flat.enumerate([&](const auto& pt) {
        EXPECT_EQ(pt.num_groups(), 1u);
        projected.push_back(pt.group_plans[0]);
    });

    // Collect inner points directly
    std::vector<std::tuple<int>> direct;
    inner.enumerate([&](const auto& pt) { direct.push_back(pt); });

    EXPECT_EQ(projected, direct);
}

// ════════════════════════════════════════════════════════════════════════
// Section 8: Edge cases + space interface (R4.13)
// ════════════════════════════════════════════════════════════════════════

TEST(Edge, ZeroCardinalityChild) {
    // Factory returns cardinality 0 for groups with 1 lane
    auto factory = [](const auto&, std::size_t,
                      std::span<const std::size_t> lanes) {
        return mock_child_space{lanes.size() > 1 ? 2 : 0};
    };

    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        factory, allow_all_filter());

    // [0,0,0]: 1 group of 3 → card = 2 (>1 lane → non-zero)
    // [0,0,1]: groups {0,1}(2 lanes→2) + {2}(1 lane→0) → product = 0
    // [0,1,0]: groups {0,2}(2→2) + {1}(1→0) → product = 0
    // [0,1,1]: groups {0}(1→0) + {1,2}(2→2) → product = 0
    // [0,1,2]: singletons → all 0 → product = 0
    // Total = 2
    EXPECT_EQ(ts.cardinality(), 2u);

    std::size_t count = 0;
    ts.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, 2u);
}

TEST(Edge, MaxGroupsRespected) {
    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        fixed_factory(1), allow_all_filter());

    ts.enumerate([](const auto& pt) {
        EXPECT_LE(pt.num_groups(), 3u);
    });
}

TEST(Interface, HasPointType) {
    using ts_type = decltype(make_tree_space<2>(
        "t", make_partition<2>("g"), fixed_factory(1), allow_all_filter()));
    using pt = typename ts_type::point_type;
    static_assert(std::is_same_v<pt, tree_point<2, std::tuple<int>>>);
}

TEST(Interface, HasCardinality) {
    auto ts = make_tree_space<2>(
        "t", make_partition<2>("g"), fixed_factory(1), allow_all_filter());
    EXPECT_GE(ts.cardinality(), 0u);
}

TEST(Interface, HasEnumerate) {
    auto ts = make_tree_space<2>(
        "t", make_partition<2>("g"), fixed_factory(1), allow_all_filter());
    ts.enumerate([](const auto&) {});
}

TEST(Interface, HasSpaceName) {
    auto ts = make_tree_space<2>(
        "test_name", make_partition<2>("g"), fixed_factory(1), allow_all_filter());
    EXPECT_EQ(ts.space_name(), "test_name");
}

TEST(Interface, FlatSameInterface) {
    auto flat = make_flat_tree_space("flat", mock_child_space{3});
    EXPECT_EQ(flat.space_name(), "flat");
    EXPECT_EQ(flat.cardinality(), 3u);
    flat.enumerate([](const auto&) {});
}

} // namespace
