// tests/test_counted_view.cpp
// Tests for ctdp/space/counted_view.h — counted views and declarative constraints
//
// Coverage:
//   Section 1: counted_view basics (R5.1, R5.2, R5.3, R5.4, R5.5)
//   Section 2: make_product_le (R5.6)
//   Section 3: make_multiple_of (R5.7)
//   Section 4: make_all_of (R5.8)
//   Section 5: Inspectability (R5.9)
//   Section 6: Integration (R5.10, R5.11, R5.12, R5.13, R5.14)

#include "ctdp/space/counted_view.h"
#include "ctdp/space/descriptor.h"
#include "ctdp/space/reduction_tree_space.h"
#include "ct_dp/algebra/operations.h"
#include "ct_dp/algebra/make_reduction.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

using namespace ctdp::space;

// ── Helper: a gemm-like 3D space ────────────────────────────────────

auto make_gemm_space() {
    return descriptor_space("gemm",
        power_2("TM", 2, 64),
        power_2("TN", 2, 64),
        power_2("TK", 2, 64));
}

// ── Helper: counting wrapper to verify caching ──────────────────────

template <typename Space>
struct counting_space {
    using point_type = typename Space::point_type;
    static constexpr std::size_t rank = Space::rank;

    Space inner_;
    mutable std::size_t enumerate_calls_ = 0;

    counting_space(Space s) : inner_(std::move(s)) {}

    auto space_name() const { return inner_.space_name(); }
    auto dimension_names() const { return inner_.dimension_names(); }
    auto cardinality() const { return inner_.cardinality(); }

    template <typename F>
    void enumerate(F&& fn) const {
        ++enumerate_calls_;
        inner_.enumerate(std::forward<F>(fn));
    }

    int get_dim_as_int(const point_type& pt, std::string_view dim) const {
        return inner_.get_dim_as_int(pt, dim);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Section 1: counted_view basics (R5.1, R5.2, R5.3, R5.4, R5.5)
// ════════════════════════════════════════════════════════════════════════

TEST(CountedView, Cardinality) {
    auto space = make_gemm_space();
    auto view = make_counted_view(space, [](const auto& pt) {
        return std::get<0>(pt) <= 16;
    });

    std::size_t count = 0;
    view.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(view.cardinality(), count);
    EXPECT_LT(view.cardinality(), space.cardinality());
}

TEST(CountedView, CardinalityCached) {
    auto base = counting_space(make_gemm_space());
    auto view = make_counted_view(base, [](const auto&) { return true; });

    auto c1 = view.cardinality();
    auto c2 = view.cardinality();
    EXPECT_EQ(c1, c2);
    // Base enumerate should be called only once (for the first cardinality)
    EXPECT_EQ(view.base_.enumerate_calls_, 1u);
}

TEST(CountedView, Enumerate) {
    auto space = make_gemm_space();
    auto view = make_counted_view(space, [](const auto& pt) {
        return std::get<2>(pt) >= 16;  // TK >= 16
    });

    std::size_t count = 0;
    view.enumerate([&](const auto& pt) {
        EXPECT_GE(std::get<2>(pt), 16);
        ++count;
    });
    EXPECT_EQ(count, view.cardinality());
}

TEST(CountedView, ConceptSatisfied) {
    using space_t = decltype(make_gemm_space());
    using view_t = counted_view<space_t, bool(*)(const typename space_t::point_type&)>;
    static_assert(countable_space<view_t>);
}

TEST(CountedView, ConstrainedConceptSatisfied) {
    using space_t = decltype(make_gemm_space());
    using view_t = counted_view<space_t, bool(*)(const typename space_t::point_type&)>;
    static_assert(constrained_space<view_t>);
}

TEST(CountedView, BaseAccessor) {
    auto space = make_gemm_space();
    auto view = make_counted_view(space, [](const auto&) { return true; });

    EXPECT_EQ(view.base().cardinality(), space.cardinality());
    EXPECT_GE(view.base().cardinality(), view.cardinality());
}

TEST(CountedView, IsValid) {
    auto space = descriptor_space("t", make_int_set("x", {1, 2, 3, 4}));
    auto view = make_counted_view(space, [](const auto& pt) {
        return std::get<0>(pt) > 2;
    });

    EXPECT_TRUE(view.is_valid(std::tuple{3}));
    EXPECT_TRUE(view.is_valid(std::tuple{4}));
    EXPECT_FALSE(view.is_valid(std::tuple{1}));
    EXPECT_FALSE(view.is_valid(std::tuple{2}));
}

TEST(CountedView, ZeroCardinality) {
    auto space = descriptor_space("t", power_2("x", 2, 8));
    auto view = make_counted_view(space, [](const auto&) { return false; });

    EXPECT_EQ(view.cardinality(), 0u);
    EXPECT_EQ(view.cardinality(), 0u);  // cached

    std::size_t count = 0;
    view.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, 0u);

    EXPECT_FALSE(view.is_valid(std::tuple{4}));
}

TEST(CountedView, ZeroBaseCardinality) {
    auto space = descriptor_space("t", positive_int("x", 10, 5));  // hi < lo → card 0
    auto view = make_counted_view(space, [](const auto&) { return true; });

    EXPECT_EQ(view.cardinality(), 0u);
    EXPECT_EQ(view.cardinality(), 0u);  // cached

    std::size_t count = 0;
    view.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, 0u);
}

TEST(CountedView, RankForwarded) {
    auto space = make_gemm_space();
    auto view = make_counted_view(space, [](const auto&) { return true; });

    static_assert(decltype(view)::rank == 3);
    EXPECT_EQ(decltype(view)::rank, decltype(space)::rank);
}

TEST(CountedView, DimensionNamesForwarded) {
    auto space = make_gemm_space();
    auto view = make_counted_view(space, [](const auto&) { return true; });

    auto base_names = space.dimension_names();
    auto view_names = view.dimension_names();
    EXPECT_EQ(base_names, view_names);
}

TEST(CountedView, CopyCacheShared) {
    auto base = counting_space(make_gemm_space());
    auto view = make_counted_view(base, [](const auto&) { return true; });

    auto c1 = view.cardinality();
    EXPECT_EQ(view.base_.enumerate_calls_, 1u);

    auto view2 = view;  // copy
    EXPECT_EQ(view2.cardinality(), c1);  // uses copied cache
    // No additional enumerate calls (still 1 from original)
    EXPECT_EQ(view2.base_.enumerate_calls_, 1u);
}

// ════════════════════════════════════════════════════════════════════════
// Section 2: make_product_le (R5.6)
// ════════════════════════════════════════════════════════════════════════

TEST(ProductLe, FiltersCorrectly) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 512, "TM", "TN", "TK");
    auto view = make_counted_view(space, c);

    view.enumerate([](const auto& pt) {
        auto product = static_cast<std::size_t>(std::get<0>(pt)) *
                       static_cast<std::size_t>(std::get<1>(pt)) *
                       static_cast<std::size_t>(std::get<2>(pt));
        EXPECT_LE(product, 512u);
    });
}

TEST(ProductLe, ReducesCardinality) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 512, "TM", "TN", "TK");
    auto view = make_counted_view(space, c);

    EXPECT_LT(view.cardinality(), space.cardinality());
}

TEST(ProductLe, TightLimit) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 8, "TM", "TN", "TK");
    auto view = make_counted_view(space, c);

    // Only 2*2*2=8 fits
    EXPECT_EQ(view.cardinality(), 1u);
}

TEST(ProductLe, LooseLimit) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 1000000000, "TM", "TN", "TK");
    auto view = make_counted_view(space, c);

    EXPECT_EQ(view.cardinality(), space.cardinality());
}

TEST(ProductLe, BadName) {
    auto space = make_gemm_space();
    EXPECT_THROW(make_product_le(space, 512, "TM", "BOGUS", "TK"),
                 std::invalid_argument);
}

TEST(ProductLe, DuplicateName) {
    auto space = make_gemm_space();
    EXPECT_THROW(make_product_le(space, 512, "TM", "TM", "TK"),
                 std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════
// Section 3: make_multiple_of (R5.7)
// ════════════════════════════════════════════════════════════════════════

TEST(MultipleOf, FiltersCorrectly) {
    auto space = descriptor_space("t",
        make_int_set("width", {1, 2, 4, 8, 16}));
    auto c = make_multiple_of(space, "width", 4);
    auto view = make_counted_view(space, c);

    EXPECT_EQ(view.cardinality(), 3u);  // 4, 8, 16

    view.enumerate([](const auto& pt) {
        EXPECT_EQ(std::get<0>(pt) % 4, 0);
    });
}

TEST(MultipleOf, Factor1) {
    auto space = descriptor_space("t",
        make_int_set("width", {1, 2, 4, 8, 16}));
    auto c = make_multiple_of(space, "width", 1);
    auto view = make_counted_view(space, c);

    EXPECT_EQ(view.cardinality(), space.cardinality());
}

TEST(MultipleOf, NothingPasses) {
    auto space = descriptor_space("t",
        make_int_set("width", {1, 2, 4, 8, 16}));
    auto c = make_multiple_of(space, "width", 32);
    auto view = make_counted_view(space, c);

    EXPECT_EQ(view.cardinality(), 0u);
}

TEST(MultipleOf, FactorZero) {
    auto space = descriptor_space("t",
        make_int_set("width", {1, 2, 4}));
    EXPECT_THROW(make_multiple_of(space, "width", 0),
                 std::invalid_argument);
}

TEST(MultipleOf, BadName) {
    auto space = descriptor_space("t",
        make_int_set("width", {1, 2, 4}));
    EXPECT_THROW(make_multiple_of(space, "BOGUS", 4),
                 std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════
// Section 4: make_all_of (R5.8)
// ════════════════════════════════════════════════════════════════════════

TEST(AllOf, Composition) {
    auto space = make_gemm_space();
    auto c1 = make_product_le(space, 512, "TM", "TN", "TK");
    auto c2 = make_multiple_of(space, "TK", 8);
    auto c = make_all_of(c1, c2);
    auto view = make_counted_view(space, c);

    view.enumerate([](const auto& pt) {
        auto product = static_cast<std::size_t>(std::get<0>(pt)) *
                       static_cast<std::size_t>(std::get<1>(pt)) *
                       static_cast<std::size_t>(std::get<2>(pt));
        EXPECT_LE(product, 512u);
        EXPECT_EQ(std::get<2>(pt) % 8, 0);
    });
}

TEST(AllOf, MatchesManual) {
    auto space = make_gemm_space();
    auto c1 = make_product_le(space, 512, "TM", "TN", "TK");
    auto c2 = make_multiple_of(space, "TK", 8);
    auto c = make_all_of(c1, c2);
    auto view = make_counted_view(space, c);

    // Manual predicate
    auto manual = make_counted_view(space, [](const auto& pt) {
        auto product = static_cast<std::size_t>(std::get<0>(pt)) *
                       static_cast<std::size_t>(std::get<1>(pt)) *
                       static_cast<std::size_t>(std::get<2>(pt));
        return product <= 512 && std::get<2>(pt) % 8 == 0;
    });

    EXPECT_EQ(view.cardinality(), manual.cardinality());
}

// ════════════════════════════════════════════════════════════════════════
// Section 5: Inspectability (R5.9)
// ════════════════════════════════════════════════════════════════════════

TEST(Inspect, ProductLe_Kind) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 2048, "TM", "TN");
    EXPECT_EQ(c.kind(), constraint_kind::product_le);
}

TEST(Inspect, ProductLe_Limit) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 2048, "TM", "TN");
    EXPECT_EQ(c.limit(), 2048u);
}

TEST(Inspect, ProductLe_DimNames) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 2048, "TM", "TN", "TK");
    auto names = c.dim_names();
    EXPECT_EQ(names.size(), 3u);
    EXPECT_EQ(names[0], "TM");
    EXPECT_EQ(names[1], "TN");
    EXPECT_EQ(names[2], "TK");
}

TEST(Inspect, MultipleOf_Kind) {
    auto space = descriptor_space("t", make_int_set("w", {1, 2, 4}));
    auto c = make_multiple_of(space, "w", 2);
    EXPECT_EQ(c.kind(), constraint_kind::multiple_of);
}

TEST(Inspect, MultipleOf_Factor) {
    auto space = descriptor_space("t", make_int_set("w", {1, 2, 4}));
    auto c = make_multiple_of(space, "w", 2);
    EXPECT_EQ(c.factor(), 2);
}

TEST(Inspect, MultipleOf_DimName) {
    auto space = descriptor_space("t", make_int_set("w", {1, 2, 4}));
    auto c = make_multiple_of(space, "w", 2);
    EXPECT_EQ(c.dim_name(), "w");
}

TEST(Inspect, AllOf_Kind) {
    auto space = make_gemm_space();
    auto c = make_all_of(
        make_product_le(space, 512, "TM", "TN"),
        make_multiple_of(space, "TK", 4));
    EXPECT_EQ(c.kind(), constraint_kind::all_of);
}

TEST(Inspect, AllOf_NumChildren) {
    auto space = make_gemm_space();
    auto c = make_all_of(
        make_product_le(space, 512, "TM", "TN"),
        make_multiple_of(space, "TK", 4));
    EXPECT_EQ(c.num_children(), 2u);
}

TEST(Inspect, AllOf_ChildAccess) {
    auto space = make_gemm_space();
    auto c = make_all_of(
        make_product_le(space, 512, "TM", "TN"),
        make_multiple_of(space, "TK", 4));

    EXPECT_EQ(c.child<0>().kind(), constraint_kind::product_le);
    EXPECT_EQ(c.child<1>().kind(), constraint_kind::multiple_of);
}

// ════════════════════════════════════════════════════════════════════════
// Section 6: Integration (R5.10, R5.11, R5.12, R5.13, R5.14)
// ════════════════════════════════════════════════════════════════════════

TEST(Integration, Factory_MakeCountedView) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 512, "TM", "TN", "TK");
    auto view = make_counted_view(space, c);
    EXPECT_GT(view.cardinality(), 0u);
    EXPECT_LT(view.cardinality(), space.cardinality());
}

TEST(Integration, Bridge_FromBase) {
    auto space = make_gemm_space();
    auto c = make_product_le(space, 512, "TM", "TN", "TK");
    auto view = make_counted_view(space, c);
    auto bridge = default_bridge(view.base());

    view.enumerate([&](const auto& pt) {
        auto features = bridge.encode(pt);
        EXPECT_EQ(features.size(), bridge.num_features());
    });
}

TEST(Integration, ComposeWithFix) {
    auto space = make_gemm_space();
    // Fix TK to 8 (index 2)
    auto fixed = fix<2>(space, 8);

    // Build constraint against the FIXED 2D subspace
    auto c = make_product_le(fixed, 256, "TM", "TN");
    auto view = make_counted_view(fixed, c);

    EXPECT_GT(view.cardinality(), 0u);
    view.enumerate([](const auto& pt) {
        auto tm = std::get<0>(pt);
        auto tn = std::get<1>(pt);
        EXPECT_LE(static_cast<std::size_t>(tm) *
                  static_cast<std::size_t>(tn), 256u);
    });
}

TEST(Integration, BackwardsCompat_ValidView) {
    auto space = make_gemm_space();
    auto vv = make_valid_view(space, [](const auto& pt) {
        return std::get<0>(pt) <= 16;
    });

    std::size_t count = 0;
    vv.enumerate([&](const auto&) { ++count; });
    EXPECT_GT(count, 0u);
    EXPECT_LT(count, space.cardinality());
}

TEST(Integration, TreeSpace_CountedChildren) {
    using namespace ct_dp::algebra;

    auto red = make_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, min_fn{},
                       std::numeric_limits<int>::max()}
    );
    auto props = reduction_properties(red);

    // Factory that returns counted_view children (all groups same type)
    auto child_factory = [props](
        const auto& /*root*/,
        std::size_t /*group*/,
        std::span<const std::size_t> lane_indices) {
            auto gp = make_group_properties(props, lane_indices);
            auto child = make_reduction_opt_space(gp).space;
            // Wrap in counted_view with trivially-true predicate
            // so all groups have the same type
            return make_counted_view(child,
                [](const auto&) { return true; });
    };

    auto filter = make_fusibility_filter<3>(props);

    auto ts = make_tree_space<3>(
        "test", make_partition<3>("g"),
        child_factory, filter);

    EXPECT_GT(ts.cardinality(), 0u);

    std::size_t count = 0;
    ts.enumerate([&](const auto&) { ++count; });
    EXPECT_EQ(count, ts.cardinality());
}

} // namespace
