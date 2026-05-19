// tests/test_integration_gemm.cpp
// Sprint 6 — Example 2: GEMM tiling search space + constraints
//
// Proves full set equivalence with gemm_tile_space, constraint filtering,
// fix<I> subproblem creation, and feature encoding.
//
// Evidence type: Equivalence (unconstrained), Composition (constrained).

#include "ctdp/space/descriptor.h"
#include "ctdp/space/counted_view.h"
#include "ctdp/space/space.h"
#include <gtest/gtest.h>
#include <set>
#include <tuple>

namespace {

using namespace ctdp::space;

auto make_gemm_space() {
    return descriptor_space("gemm",
        power_2("TM", 2, 64), power_2("TN", 2, 64), power_2("TK", 2, 64));
}

// ════════════════════════════════════════════════════════════════════════
// R6.5: Full set equivalence with gemm_tile_space
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, FullSetEquivalence) {
    auto space = make_gemm_space();

    using pt_t = std::tuple<int, int, int>;
    std::set<pt_t> framework_set;
    space.enumerate([&](const auto& pt) { framework_set.insert(pt); });

    std::set<pt_t> domain_set;
    gemm_tile_space::enumerate([&](const tile_shape& ts) {
        domain_set.insert({ts.tm, ts.tn, ts.tk});
    });

    EXPECT_EQ(framework_set.size(), 216u);
    EXPECT_EQ(framework_set, domain_set);
}

// ════════════════════════════════════════════════════════════════════════
// R6.6a: L1 constraint (working set model)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, L1Constraint) {
    auto space = make_gemm_space();
    auto l1_view = make_counted_view(space, [](const auto& pt) {
        auto tm = std::get<0>(pt);
        auto tn = std::get<1>(pt);
        auto tk = std::get<2>(pt);
        return static_cast<std::size_t>(tm * tn + tn * tk + tm * tk) <= 2048;
    });

    l1_view.enumerate([](const auto& pt) {
        auto tm = std::get<0>(pt);
        auto tn = std::get<1>(pt);
        auto tk = std::get<2>(pt);
        EXPECT_LE(static_cast<std::size_t>(tm * tn + tn * tk + tm * tk), 2048u);
    });
}

// ════════════════════════════════════════════════════════════════════════
// R6.6b: product_le constraint (raw tile product)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, ProductLeConstraint) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 512, "TM", "TN", "TK");
    auto product_view = make_counted_view(space, c);

    product_view.enumerate([](const auto& pt) {
        auto product = static_cast<std::size_t>(std::get<0>(pt)) *
                       static_cast<std::size_t>(std::get<1>(pt)) *
                       static_cast<std::size_t>(std::get<2>(pt));
        EXPECT_LE(product, 512u);
    });
}

// ════════════════════════════════════════════════════════════════════════
// R6.7: Constrained cardinality < unconstrained
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, ConstrainedFewer) {
    auto space = make_gemm_space();

    auto l1_view = make_counted_view(space, [](const auto& pt) {
        auto tm = std::get<0>(pt);
        auto tn = std::get<1>(pt);
        auto tk = std::get<2>(pt);
        return static_cast<std::size_t>(tm * tn + tn * tk + tm * tk) <= 2048;
    });

    auto c = make_product_le(space, 512, "TM", "TN", "TK");
    auto product_view = make_counted_view(space, c);

    EXPECT_LT(l1_view.cardinality(), 216u);
    EXPECT_LT(product_view.cardinality(), 216u);
}

// ════════════════════════════════════════════════════════════════════════
// R6.8: Feature encoding (log2 for power_2)
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, FeatureEncoding) {
    auto space = make_gemm_space();
    auto bridge = default_bridge(space);

    auto pt = std::tuple{16, 8, 4};
    auto f = bridge.encode(pt);

    ASSERT_EQ(f.size(), 3u);
    EXPECT_DOUBLE_EQ(f[0], 4.0);  // log2(16)
    EXPECT_DOUBLE_EQ(f[1], 3.0);  // log2(8)
    EXPECT_DOUBLE_EQ(f[2], 2.0);  // log2(4)
}

// ════════════════════════════════════════════════════════════════════════
// R6.9: fix<I> creates 2D subspace
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, FixDimension) {
    auto space = make_gemm_space();
    auto fixed = fix<2>(space, 8);  // fix TK=8

    EXPECT_EQ(decltype(fixed)::rank, 2u);
    auto names = fixed.dimension_names();
    EXPECT_EQ(names[0], "TM");
    EXPECT_EQ(names[1], "TN");
    EXPECT_EQ(fixed.cardinality(), 36u);
}

// ════════════════════════════════════════════════════════════════════════
// R6.10: counted_view composes with fix<I>
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, FixWithConstraint) {
    auto space = make_gemm_space();
    auto fixed = fix<2>(space, 8);

    auto c = make_product_le(fixed, 128, "TM", "TN");
    auto view = make_counted_view(fixed, c);

    EXPECT_GT(view.cardinality(), 0u);
    EXPECT_LE(view.cardinality(), 36u);

    view.enumerate([](const auto& pt) {
        auto product = static_cast<std::size_t>(std::get<0>(pt)) *
                       static_cast<std::size_t>(std::get<1>(pt));
        EXPECT_LE(product, 128u);
    });
}

// ════════════════════════════════════════════════════════════════════════
// R6.21: Bridge protocol
// ════════════════════════════════════════════════════════════════════════

TEST(IntegrationGemm, BridgeProtocol) {
    auto space = make_gemm_space();
    auto bridge = default_bridge(space);
    auto features = bridge.encode(std::tuple{16, 8, 4});
    EXPECT_EQ(features.size(), bridge.num_features());
}

} // namespace
