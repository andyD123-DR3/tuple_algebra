// tests/test_space_v072.cpp
// Tests for ctdp/space/ layer (v0.7.2)
//
// 5 tests:
//   1. valid_view — filters enumeration, tier 1 only
//   2. point_at — closed-form indexing (tier 3)
//   3. factored — per-dimension enumeration
//   4. dispatch — NTTP instantiation utility
//   5. tier composition — filter(tier3) → tier1, exhaustive_search on both

#include "ctdp/space/descriptor.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace {

using namespace ctdp::space;

// ── Test enums ──────────────────────────────────────────────────────────────

enum class Color : int { RED, GREEN, BLUE };
enum class Size  : int { SMALL, LARGE };

// ── Helper: build a small test space ────────────────────────────────────────

constexpr auto make_test_space() {
    return descriptor_space("test",
        make_enum_vals("color", Color::RED, Color::GREEN, Color::BLUE),
        make_enum_vals("size", Size::SMALL, Size::LARGE),
        bool_flag("turbo")
    );
}

// 3 × 2 × 2 = 12 points

// ============================================================================
// Test 1: valid_view filters enumeration
// ============================================================================

TEST(SpaceV072, ValidViewFiltersEnumeration) {
    constexpr auto base = make_test_space();
    static_assert(base.cardinality() == 12);

    // Filter: only BLUE points
    auto valid = filter_valid(base, [](auto const& pt) {
        return std::get<0>(pt) == Color::BLUE;
    });

    // Count valid points
    std::size_t count = 0;
    valid.enumerate([&](auto const& pt) {
        EXPECT_EQ(std::get<0>(pt), Color::BLUE);
        ++count;
    });
    EXPECT_EQ(count, 4u);  // BLUE × {SMALL,LARGE} × {false,true}

    // valid_view rank is preserved
    static_assert(decltype(valid)::rank == 3);
}

// ============================================================================
// Test 2: point_at — closed-form indexing
// ============================================================================

TEST(SpaceV072, PointAtIndexing) {
    constexpr auto space = make_test_space();

    // Enumerate all points and verify point_at matches
    std::size_t i = 0;
    space.enumerate([&](auto const& pt) {
        auto indexed = space.point_at(i);
        EXPECT_EQ(std::get<0>(pt), std::get<0>(indexed));
        EXPECT_EQ(std::get<1>(pt), std::get<1>(indexed));
        EXPECT_EQ(std::get<2>(pt), std::get<2>(indexed));
        ++i;
    });
    EXPECT_EQ(i, space.cardinality());
}

// Compile-time version
static_assert([] {
    auto space = make_test_space();
    auto pt0 = space.point_at(0);
    // First point: RED, SMALL, false
    return std::get<0>(pt0) == Color::RED
        && std::get<1>(pt0) == Size::SMALL
        && std::get<2>(pt0) == false;
}());

// ============================================================================
// Test 3: factored — per-dimension enumeration
// ============================================================================

TEST(SpaceV072, FactoredAccess) {
    constexpr auto space = make_test_space();

    static_assert(space.num_dims() == 3);

    // Dimension 0: Color, 3 values
    EXPECT_EQ(space.dim_cardinality(0), 3u);
    EXPECT_EQ(space.dim_value<0>(0), Color::RED);
    EXPECT_EQ(space.dim_value<0>(1), Color::GREEN);
    EXPECT_EQ(space.dim_value<0>(2), Color::BLUE);

    // Dimension 1: Size, 2 values
    EXPECT_EQ(space.dim_cardinality(1), 2u);
    EXPECT_EQ(space.dim_value<1>(0), Size::SMALL);
    EXPECT_EQ(space.dim_value<1>(1), Size::LARGE);

    // Dimension 2: bool, 2 values
    EXPECT_EQ(space.dim_cardinality(2), 2u);
    EXPECT_EQ(space.dim_value<2>(0), false);
    EXPECT_EQ(space.dim_value<2>(1), true);

    // Dimension names
    auto names = space.dimension_names();
    EXPECT_EQ(names[0], "color");
    EXPECT_EQ(names[1], "size");
    EXPECT_EQ(names[2], "turbo");
}

// ============================================================================
// Test 4: dispatch — NTTP instantiation
// ============================================================================

// A simple config struct (structural type for NTTP)
struct test_config {
    Color color;
    Size size;
    bool turbo;

    constexpr bool operator==(test_config const&) const = default;
};

// An executor template parameterised on config
template<test_config Cfg>
struct test_executor {
    static constexpr int value() {
        int v = 0;
        if constexpr (Cfg.color == Color::RED)   v += 100;
        if constexpr (Cfg.color == Color::GREEN) v += 200;
        if constexpr (Cfg.color == Color::BLUE)  v += 300;
        if constexpr (Cfg.size == Size::LARGE)   v += 10;
        if constexpr (Cfg.turbo)                 v += 1;
        return v;
    }
};

TEST(SpaceV072, DispatchNTTP) {
    // BLUE, LARGE, turbo=true → 300 + 10 + 1 = 311
    constexpr test_config cfg{Color::BLUE, Size::LARGE, true};
    using executor = dispatch<test_executor, cfg>;
    static_assert(executor::value() == 311);
    EXPECT_EQ(executor::value(), 311);

    // RED, SMALL, turbo=false → 100 + 0 + 0 = 100
    constexpr test_config cfg2{Color::RED, Size::SMALL, false};
    using executor2 = dispatch<test_executor, cfg2>;
    static_assert(executor2::value() == 100);
    EXPECT_EQ(executor2::value(), 100);
}

// ============================================================================
// Test 5: tier composition — search on base and filtered space
// ============================================================================

TEST(SpaceV072, TierCompositionSearch) {
    constexpr auto base = make_test_space();

    // Cost: prefer BLUE + LARGE + turbo
    auto cost = [](auto const& pt) -> double {
        double c = 0.0;
        if (std::get<0>(pt) == Color::BLUE) c -= 100.0;
        if (std::get<1>(pt) == Size::LARGE) c -= 10.0;
        if (std::get<2>(pt)) c -= 1.0;
        return c;
    };

    // Search the full space
    auto full_result = exhaustive_search_with_cost(base, cost);
    EXPECT_EQ(full_result.evaluated, 12u);
    EXPECT_EQ(std::get<0>(full_result.best), Color::BLUE);
    EXPECT_EQ(std::get<1>(full_result.best), Size::LARGE);
    EXPECT_EQ(std::get<2>(full_result.best), true);
    EXPECT_DOUBLE_EQ(full_result.best_cost, -111.0);

    // Filter: only RED or GREEN (exclude BLUE)
    auto valid = filter_valid(base, [](auto const& pt) {
        return std::get<0>(pt) != Color::BLUE;
    });

    // Search the filtered space — should find GREEN+LARGE+turbo
    auto filtered_result = exhaustive_search_with_cost(valid, cost);
    EXPECT_EQ(filtered_result.evaluated, 8u);  // 2 colors × 2 sizes × 2 turbo
    EXPECT_EQ(std::get<0>(filtered_result.best), Color::GREEN);
    EXPECT_EQ(std::get<1>(filtered_result.best), Size::LARGE);
    EXPECT_EQ(std::get<2>(filtered_result.best), true);

    // Bridge: feature encoding
    auto bridge = make_bridge(base);
    EXPECT_EQ(bridge.feature_width(), 3u + 2u + 1u);  // one-hot(3) + one-hot(2) + binary

    double features[6];
    bridge.write_features(full_result.best, std::span<double>{features, 6});
    // BLUE = [0,0,1], LARGE = [0,1], turbo = [1]
    EXPECT_DOUBLE_EQ(features[0], 0.0);  // not RED
    EXPECT_DOUBLE_EQ(features[1], 0.0);  // not GREEN
    EXPECT_DOUBLE_EQ(features[2], 1.0);  // BLUE
    EXPECT_DOUBLE_EQ(features[3], 0.0);  // not SMALL
    EXPECT_DOUBLE_EQ(features[4], 1.0);  // LARGE
    EXPECT_DOUBLE_EQ(features[5], 1.0);  // turbo
}

// ── Concept static assertions ───────────────────────────────────────────────

// descriptor_space satisfies all four tiers
static_assert(search_space<decltype(make_test_space())>);
static_assert(countable_space<decltype(make_test_space())>);
static_assert(indexable_space<decltype(make_test_space())>);
static_assert(factored_space<decltype(make_test_space())>);

// valid_view satisfies only tier 1
using filtered_t = decltype(filter_valid(make_test_space(),
    [](auto const&) { return true; }));
static_assert(search_space<filtered_t>);
static_assert(!countable_space<filtered_t>);  // honest: no cardinality
static_assert(!indexable_space<filtered_t>);   // honest: no point_at
static_assert(!factored_space<filtered_t>);    // honest: no dim access

} // anonymous namespace
