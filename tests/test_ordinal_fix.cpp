// tests/test_ordinal_fix.cpp
// Tests for Sprint 1: ordinal ordinate type and fix<I> alias
//
// Coverage:
//   Section 1: ordinal — concept satisfaction
//   Section 2: ordinal — basic contract (cardinality, membership, enumeration)
//   Section 3: ordinal — rank_of and less_than
//   Section 4: ordinal — degenerate cases
//   Section 5: ordinal — feature encoding (normalised rank)
//   Section 6: ordinal — encoding override (one-hot, raw)
//   Section 7: ordinal — in descriptor_space (cardinality, enumeration)
//   Section 8: ordinal — feature bridge integration
//   Section 9: ordinal + conditional_dim interaction
//   Section 10: fix<I> — rank reduction and enumeration
//   Section 11: fix<I> — embed round-trip
//   Section 12: enum_vals normalised encoding (retroactive support)
//   Section 13: Realistic scenario — reproducibility level dimension

#include "ctdp/space/conditional.h"
#include "ctdp/space/descriptor.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

namespace {

using namespace ctdp::space;

// ── Test enums (at namespace scope — Clang requirement) ─────────────

enum class Repro : int { nondet, deterministic, reproducible, bitwise };
enum class SimdTier : int { scalar, SSE, AVX2, AVX512 };
enum class MemLevel : int { L1, L2, L3, DRAM };
enum class Color : int { RED, GREEN, BLUE };
enum class Size : int { SMALL, LARGE };
enum class TreeShape : int { flat, binary, pairwise };

// ════════════════════════════════════════════════════════════════════════
// Section 1: Concept satisfaction
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, ConceptSatisfied_Repro) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    static_assert(dimension_descriptor<decltype(od)>);
}

TEST(Ordinal, ConceptSatisfied_SimdTier) {
    auto od = make_ordinal("simd", {SimdTier::scalar, SimdTier::SSE, SimdTier::AVX2, SimdTier::AVX512});
    static_assert(dimension_descriptor<decltype(od)>);
}

TEST(Ordinal, ConceptSatisfied_MemLevel) {
    auto od = make_ordinal("mem", {MemLevel::L1, MemLevel::L2, MemLevel::L3, MemLevel::DRAM});
    static_assert(dimension_descriptor<decltype(od)>);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: Basic contract
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, Cardinality) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    EXPECT_EQ(od.cardinality(), 4u);
}

TEST(Ordinal, ValueAt) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    EXPECT_EQ(od.value_at(0), Repro::nondet);
    EXPECT_EQ(od.value_at(1), Repro::deterministic);
    EXPECT_EQ(od.value_at(2), Repro::reproducible);
    EXPECT_EQ(od.value_at(3), Repro::bitwise);
}

TEST(Ordinal, Contains) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    EXPECT_TRUE(od.contains(Repro::nondet));
    EXPECT_TRUE(od.contains(Repro::bitwise));
    EXPECT_FALSE(od.contains(static_cast<Repro>(99)));
}

TEST(Ordinal, IndexOfRoundTrips) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    for (std::size_t i = 0; i < od.cardinality(); ++i) {
        EXPECT_EQ(od.index_of(od.value_at(i)), i);
    }
}

TEST(Ordinal, IndexOfMissing) {
    auto od = make_ordinal("repro", {Repro::nondet, Repro::deterministic});
    EXPECT_EQ(od.index_of(Repro::bitwise), 2u);  // out-of-range = cardinality
}

TEST(Ordinal, DefaultEncodingIsNormalised) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    EXPECT_EQ(od.default_encoding(), encoding_hint::normalised);
}

TEST(Ordinal, KindIsEnumSet) {
    auto od = make_ordinal("repro", {Repro::nondet, Repro::deterministic});
    EXPECT_EQ(decltype(od)::kind, dim_kind::enum_set);
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: rank_of and less_than
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, RankOf_FourLevels) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});

    EXPECT_DOUBLE_EQ(od.rank_of(Repro::nondet),         0.0);
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::deterministic),   1.0 / 3.0);
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::reproducible),    2.0 / 3.0);
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::bitwise),         1.0);
}

TEST(Ordinal, RankOf_TwoLevels) {
    auto od = make_ordinal("repro", {Repro::nondet, Repro::bitwise});
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::nondet),  0.0);
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::bitwise), 1.0);
}

TEST(Ordinal, RankOf_Missing) {
    auto od = make_ordinal("repro", {Repro::nondet, Repro::bitwise});
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::reproducible), -1.0);
}

TEST(Ordinal, LessThan) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});

    EXPECT_TRUE(od.less_than(Repro::nondet, Repro::bitwise));
    EXPECT_TRUE(od.less_than(Repro::deterministic, Repro::reproducible));
    EXPECT_FALSE(od.less_than(Repro::bitwise, Repro::nondet));
    EXPECT_FALSE(od.less_than(Repro::reproducible, Repro::reproducible));
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: Degenerate cases
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, SingleValue_Cardinality) {
    auto od = make_ordinal("trivial", {Repro::nondet});
    EXPECT_EQ(od.cardinality(), 1u);
}

TEST(Ordinal, SingleValue_RankIsZero) {
    auto od = make_ordinal("trivial", {Repro::nondet});
    EXPECT_DOUBLE_EQ(od.rank_of(Repro::nondet), 0.0);
}

TEST(Ordinal, EncodedAs_Override) {
    auto od = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});
    EXPECT_EQ(od.default_encoding(), encoding_hint::normalised);

    auto oh = od.encoded_as(encoding_hint::one_hot);
    EXPECT_EQ(oh.default_encoding(), encoding_hint::one_hot);
    // Cardinality and values unchanged
    EXPECT_EQ(oh.cardinality(), 4u);
    EXPECT_EQ(oh.value_at(0), Repro::nondet);
}

// ════════════════════════════════════════════════════════════════════════
// Section 5: Feature encoding — normalised rank
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, FeatureEncoding_Normalised_AllValues) {
    auto space = descriptor_space("test",
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise})
    );
    auto bridge = default_bridge(space);

    EXPECT_EQ(bridge.num_features(), 1u);

    auto f0 = bridge.encode(std::tuple{Repro::nondet});
    auto f1 = bridge.encode(std::tuple{Repro::deterministic});
    auto f2 = bridge.encode(std::tuple{Repro::reproducible});
    auto f3 = bridge.encode(std::tuple{Repro::bitwise});

    EXPECT_DOUBLE_EQ(f0[0], 0.0);
    EXPECT_DOUBLE_EQ(f1[0], 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(f2[0], 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(f3[0], 1.0);
}

TEST(Ordinal, FeatureEncoding_OrderPreserved) {
    auto space = descriptor_space("test",
        make_ordinal("simd",
            {SimdTier::scalar, SimdTier::SSE, SimdTier::AVX2, SimdTier::AVX512})
    );
    auto bridge = default_bridge(space);

    auto fs = bridge.encode(std::tuple{SimdTier::scalar});
    auto fa = bridge.encode(std::tuple{SimdTier::AVX2});
    auto f5 = bridge.encode(std::tuple{SimdTier::AVX512});

    EXPECT_LT(fs[0], fa[0]);
    EXPECT_LT(fa[0], f5[0]);
}

TEST(Ordinal, FeatureEncoding_WithOtherDims) {
    auto space = descriptor_space("test",
        power_2("tile", 8, 64),
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise})
    );
    auto bridge = default_bridge(space);

    // tile (log2 → 1) + repro (normalised → 1) = 2
    EXPECT_EQ(bridge.num_features(), 2u);

    auto features = bridge.encode(std::tuple{32, Repro::reproducible});
    EXPECT_DOUBLE_EQ(features[0], 5.0);         // log2(32)
    EXPECT_DOUBLE_EQ(features[1], 2.0 / 3.0);   // rank 2 of 4
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: Encoding override
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, EncodingOverride_OneHot) {
    auto space = descriptor_space("test",
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise})
            .encoded_as(encoding_hint::one_hot)
    );
    auto bridge = default_bridge(space);

    EXPECT_EQ(bridge.num_features(), 4u);

    auto features = bridge.encode(std::tuple{Repro::reproducible});
    EXPECT_DOUBLE_EQ(features[0], 0.0);
    EXPECT_DOUBLE_EQ(features[1], 0.0);
    EXPECT_DOUBLE_EQ(features[2], 1.0);
    EXPECT_DOUBLE_EQ(features[3], 0.0);
}

// ════════════════════════════════════════════════════════════════════════
// Section 7: ordinal in descriptor_space
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, InSpace_Cardinality) {
    auto space = descriptor_space("test",
        power_2("tile", 64, 256),
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise})
    );
    EXPECT_EQ(space.cardinality(), 3u * 4u);
}

TEST(Ordinal, InSpace_EnumerationCountMatches) {
    auto space = descriptor_space("test",
        make_int_set("vec", {1, 4, 8}),
        make_ordinal("mem", {MemLevel::L1, MemLevel::L2, MemLevel::L3, MemLevel::DRAM})
    );

    std::size_t count = 0;
    space.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, space.cardinality());
    EXPECT_EQ(count, 12u);
}

TEST(Ordinal, InSpace_PointAtRoundTrip) {
    auto space = descriptor_space("test",
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise}),
        bool_flag("turbo")
    );

    for (std::size_t i = 0; i < space.cardinality(); ++i) {
        auto pt = space.point_at(i);
        // Verify point is valid
        EXPECT_TRUE(space.point_at(i) == pt);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Section 8: ordinal + conditional_dim
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, WithConditionalDim_Active) {
    auto space = descriptor_space("red_opt",
        power_2("tile", 64, 256),
        conditional_dim(true,
            make_ordinal("repro",
                {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise}))
    );
    // 3 tiles × 4 repro (active) = 12
    EXPECT_EQ(space.cardinality(), 12u);
}

TEST(Ordinal, WithConditionalDim_Inactive) {
    auto space = descriptor_space("red_opt",
        power_2("tile", 64, 256),
        conditional_dim(false,
            make_ordinal("repro",
                {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise}))
    );
    // 3 tiles × 1 (inactive) = 3
    EXPECT_EQ(space.cardinality(), 3u);
}

TEST(Ordinal, WithConditionalDim_FeatureWidthStable) {
    auto repro_dim = make_ordinal("repro",
        {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise});

    auto space_on = descriptor_space("on",
        power_2("tile", 64, 256), conditional_dim(true, repro_dim));
    auto space_off = descriptor_space("off",
        power_2("tile", 64, 256), conditional_dim(false, repro_dim));

    auto bridge_on  = default_bridge(space_on);
    auto bridge_off = default_bridge(space_off);

    // Both should have same feature width: tile(1) + repro(1) = 2
    EXPECT_EQ(bridge_on.num_features(), bridge_off.num_features());
    EXPECT_EQ(bridge_on.num_features(), 2u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 9: fix<I> alias
// ════════════════════════════════════════════════════════════════════════

TEST(Fix, ReducesRank) {
    auto space = descriptor_space("test",
        make_enum_vals("color", {Color::RED, Color::GREEN, Color::BLUE}),
        make_enum_vals("size", {Size::SMALL, Size::LARGE}),
        bool_flag("turbo")
    );

    auto fixed = ctdp::space::section<0>(space, Color::BLUE);
    EXPECT_EQ(decltype(fixed)::rank, 2u);  // 3 - 1 = 2
}

TEST(Fix, CorrectEnumeration) {
    auto space = descriptor_space("test",
        make_enum_vals("color", {Color::RED, Color::GREEN, Color::BLUE}),
        make_enum_vals("size", {Size::SMALL, Size::LARGE}),
        bool_flag("turbo")
    );

    // Fix color to BLUE → size × turbo = 4 points
    auto fixed = ctdp::space::section<0>(space, Color::BLUE);
    std::size_t count = 0;
    fixed.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, 4u);
}

TEST(Fix, FixMiddleDimension) {
    auto space = descriptor_space("test",
        make_enum_vals("color", {Color::RED, Color::GREEN, Color::BLUE}),
        make_enum_vals("size", {Size::SMALL, Size::LARGE}),
        bool_flag("turbo")
    );

    // Fix size to LARGE → color × turbo = 6 points
    auto fixed = ctdp::space::section<1>(space, Size::LARGE);
    std::size_t count = 0;
    fixed.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, 6u);
}

TEST(Fix, FixLastDimension) {
    auto space = descriptor_space("test",
        make_enum_vals("color", {Color::RED, Color::GREEN, Color::BLUE}),
        make_enum_vals("size", {Size::SMALL, Size::LARGE}),
        bool_flag("turbo")
    );

    // Fix turbo to true → color × size = 6 points
    auto fixed = ctdp::space::section<2>(space, true);
    std::size_t count = 0;
    fixed.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, 6u);
}

TEST(Fix, EmbedRoundTrip) {
    auto space = descriptor_space("test",
        make_enum_vals("color", {Color::RED, Color::GREEN, Color::BLUE}),
        make_enum_vals("size", {Size::SMALL, Size::LARGE})
    );

    auto fixed = ctdp::space::section<0>(space, Color::GREEN);

    auto reduced = std::tuple{Size::LARGE};
    auto full = fixed.embed(reduced);
    EXPECT_EQ(std::get<0>(full), Color::GREEN);
    EXPECT_EQ(std::get<1>(full), Size::LARGE);
}

TEST(Fix, WithOrdinal) {
    auto space = descriptor_space("test",
        power_2("tile", 64, 256),
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise})
    );

    // Fix repro to deterministic → 1D: 3 tile values
    auto subproblem = ctdp::space::section<1>(space, Repro::deterministic);
    std::size_t count = 0;
    subproblem.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, 3u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 10: enum_vals with normalised encoding (retroactive)
// ════════════════════════════════════════════════════════════════════════

TEST(EnumVals, NormalisedEncoding_Override) {
    auto space = descriptor_space("test",
        make_enum_vals("color", {Color::RED, Color::GREEN, Color::BLUE})
            .encoded_as(encoding_hint::normalised)
    );
    auto bridge = default_bridge(space);

    EXPECT_EQ(bridge.num_features(), 1u);

    auto f0 = bridge.encode(std::tuple{Color::RED});
    auto f1 = bridge.encode(std::tuple{Color::GREEN});
    auto f2 = bridge.encode(std::tuple{Color::BLUE});

    EXPECT_DOUBLE_EQ(f0[0], 0.0);
    EXPECT_DOUBLE_EQ(f1[0], 0.5);
    EXPECT_DOUBLE_EQ(f2[0], 1.0);
}

// ════════════════════════════════════════════════════════════════════════
// Section 11: Realistic scenario — reproducibility as search dimension
// ════════════════════════════════════════════════════════════════════════

TEST(Ordinal, ReproScenario_FullPipeline) {
    // A reduction-opt space with tile, tree shape, and reproducibility level.
    // Tree shape is conditional on associativity; repro constrains it further.
    constexpr bool all_assoc = true;

    auto space = descriptor_space("reduction_opt",
        power_2("tile", 64, 256),
        conditional_dim(all_assoc,
            make_enum_vals("tree_shape",
                {TreeShape::flat, TreeShape::binary, TreeShape::pairwise})),
        make_ordinal("repro",
            {Repro::nondet, Repro::deterministic, Repro::reproducible, Repro::bitwise})
    );

    // 3 tiles × 3 tree shapes (active) × 4 repro = 36
    EXPECT_EQ(space.cardinality(), 36u);

    auto bridge = default_bridge(space);
    // tile(log2:1) + tree(one-hot:3) + repro(normalised:1) = 5
    EXPECT_EQ(bridge.num_features(), 5u);

    // Encode a specific point and verify
    auto pt = std::tuple{128, TreeShape::binary, Repro::reproducible};
    auto features = bridge.encode(pt);

    EXPECT_DOUBLE_EQ(features[0], 7.0);         // log2(128)
    EXPECT_DOUBLE_EQ(features[1], 0.0);          // one-hot: flat
    EXPECT_DOUBLE_EQ(features[2], 1.0);          // one-hot: binary ← selected
    EXPECT_DOUBLE_EQ(features[3], 0.0);          // one-hot: pairwise
    EXPECT_DOUBLE_EQ(features[4], 2.0 / 3.0);   // repro rank 2/3

    // Fix repro to bitwise → 2D subproblem: tile × tree = 9
    auto subproblem = ctdp::space::section<2>(space, Repro::bitwise);
    std::size_t count = 0;
    subproblem.enumerate([&](auto const&) { ++count; });
    EXPECT_EQ(count, 9u);
}

} // namespace
