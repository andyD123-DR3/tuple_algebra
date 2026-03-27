// tests/test_integration_fix.cpp
// Sprint 6 — Example 1: FIX parser search space equivalence
//
// Proves that descriptor_space with per-field enum_vals produces the
// same plan set as the existing Schema<4>::enumerate_plans(), and that
// feature_bridge encodes strategies correctly.
//
// Evidence type: Equivalence.

#include "ctdp/space/descriptor.h"
#include "ctdp/calibrator/fix/fix_field_descriptor.h"
#include <gtest/gtest.h>
#include <set>
#include <tuple>

namespace {

using namespace ctdp::space;
using namespace ctdp::calibrator::fix;
using S = Strategy;

// The descriptor_space representation of trivial_schema.
auto make_fix_space() {
    return descriptor_space("fix_trivial",
        make_enum_vals("Field0", {S::Unrolled, S::SWAR, S::Loop, S::Generic}),
        make_enum_vals("Field1", {S::Unrolled, S::SWAR, S::Loop, S::Generic}),
        make_enum_vals("Field2", {S::Unrolled}),
        make_enum_vals("Field3", {S::Unrolled, S::SWAR, S::Loop, S::Generic}));
}

using fix_space_t = decltype(make_fix_space());
using fix_point_t = typename fix_space_t::point_type;

// ════════════════════════════════════════════════════════════════════════
// R6.1: Cardinality equivalence
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationFix, SpaceCardinality) {
    auto space = make_fix_space();
    EXPECT_EQ(space.cardinality(), 64u);
    EXPECT_EQ(space.cardinality(),
        static_cast<std::size_t>(trivial_schema.plan_space_size()));
}

// ════════════════════════════════════════════════════════════════════════
// R6.2: Plan set equivalence (by actual Strategy enum values)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationFix, SpaceEnumeration) {
    auto space = make_fix_space();

    // Collect framework points
    std::set<fix_point_t> framework_set;
    space.enumerate([&](const auto& pt) { framework_set.insert(pt); });

    // Collect schema plans, converted to tuples
    auto schema_plans = trivial_schema.enumerate_plans();
    std::set<fix_point_t> schema_set;
    for (const auto& arr : schema_plans)
        schema_set.insert(std::make_tuple(arr[0], arr[1], arr[2], arr[3]));

    EXPECT_EQ(framework_set, schema_set);
}

// ════════════════════════════════════════════════════════════════════════
// R6.3: Feature encoding (one-hot per field)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationFix, FeatureEncoding) {
    auto space = make_fix_space();
    auto bridge = default_bridge(space);

    // Field0=SWAR(1), Field1=Loop(2), Field2=Unrolled(0), Field3=Generic(3)
    auto pt = std::make_tuple(S::SWAR, S::Loop, S::Unrolled, S::Generic);
    auto f = bridge.encode(pt);

    ASSERT_EQ(f.size(), 13u);

    // Field0 (4 slots): [0,1,0,0]
    EXPECT_DOUBLE_EQ(f[0], 0.0);
    EXPECT_DOUBLE_EQ(f[1], 1.0);
    EXPECT_DOUBLE_EQ(f[2], 0.0);
    EXPECT_DOUBLE_EQ(f[3], 0.0);

    // Field1 (4 slots): [0,0,1,0]
    EXPECT_DOUBLE_EQ(f[4], 0.0);
    EXPECT_DOUBLE_EQ(f[5], 0.0);
    EXPECT_DOUBLE_EQ(f[6], 1.0);
    EXPECT_DOUBLE_EQ(f[7], 0.0);

    // Field2 (1 slot): [1]
    EXPECT_DOUBLE_EQ(f[8], 1.0);

    // Field3 (4 slots): [0,0,0,1]
    EXPECT_DOUBLE_EQ(f[9],  0.0);
    EXPECT_DOUBLE_EQ(f[10], 0.0);
    EXPECT_DOUBLE_EQ(f[11], 0.0);
    EXPECT_DOUBLE_EQ(f[12], 1.0);
}

// ════════════════════════════════════════════════════════════════════════
// R6.4: Feature width
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationFix, FeatureWidth) {
    auto space = make_fix_space();
    auto bridge = default_bridge(space);
    EXPECT_EQ(bridge.num_features(), 13u);
}

// ════════════════════════════════════════════════════════════════════════
// R6.21: Bridge protocol
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationFix, BridgeProtocol) {
    auto space = make_fix_space();
    auto bridge = default_bridge(space);
    auto pt = std::make_tuple(S::Unrolled, S::Unrolled, S::Unrolled, S::Unrolled);
    auto features = bridge.encode(pt);
    EXPECT_EQ(features.size(), bridge.num_features());
}

} // namespace
