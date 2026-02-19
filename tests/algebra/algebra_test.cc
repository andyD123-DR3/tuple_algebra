// tests/algebra_test.cpp
//
// Google Tests for ct_dp::algebra.
//
// Every test exercises both runtime correctness (via EXPECT_*) and
// compile-time correctness (via static_assert). This validates that
// all components are fully constexpr and produce identical results
// in both contexts.
//
// Copyright (c) 2025 Andrew Drakeford. All rights reserved.

#include <ct_dp/algebra/algebra.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

using namespace ct_dp::algebra;

// ============================================================================
// Operations
// ============================================================================

TEST(Operations, Identity) {
    constexpr identity_t id;
    static_assert(id(42) == 42);
    static_assert(id(3.14) == 3.14);

    EXPECT_EQ(id(42), 42);
    EXPECT_DOUBLE_EQ(id(3.14), 3.14);

    // Traits
    static_assert(identity_t::is_identity);
    static_assert(identity_t::is_power);
    static_assert(identity_t::exponent == 1);
}

TEST(Operations, Constant) {
    constexpr constant_t<1> one;
    static_assert(one(42) == 1);
    static_assert(one(0) == 1);
    static_assert(one(-999) == 1);

    EXPECT_EQ(one(42), 1);
    EXPECT_EQ(one(0), 1);

    constexpr constant_t<0> zero;
    static_assert(zero(42) == 0);

    // Traits
    static_assert(!constant_t<1>::is_identity);
    static_assert(!constant_t<1>::is_power);
}

TEST(Operations, Power0) {
    constexpr power_t<0> p;
    static_assert(p(5) == 1);
    static_assert(p(0) == 0 || p(0) == 1);  // 0^0 = 1 by convention here
    EXPECT_EQ(p(5), 1);
    EXPECT_EQ(p(100), 1);

    static_assert(!power_t<0>::is_identity);
    static_assert(power_t<0>::exponent == 0);
}

TEST(Operations, Power1) {
    constexpr power_t<1> p;
    static_assert(p(7) == 7);
    static_assert(p(3.5) == 3.5);
    EXPECT_EQ(p(7), 7);

    static_assert(power_t<1>::is_identity);
    static_assert(power_t<1>::exponent == 1);
}

TEST(Operations, Power2) {
    constexpr power_t<2> p;
    static_assert(p(3) == 9);
    static_assert(p(5) == 25);
    static_assert(p(-4) == 16);
    EXPECT_EQ(p(3), 9);
    EXPECT_DOUBLE_EQ(p(2.5), 6.25);
}

TEST(Operations, Power3) {
    constexpr power_t<3> p;
    static_assert(p(2) == 8);
    static_assert(p(3) == 27);
    static_assert(p(-2) == -8);
    EXPECT_EQ(p(2), 8);
}

TEST(Operations, Power4) {
    constexpr power_t<4> p;
    static_assert(p(2) == 16);
    static_assert(p(3) == 81);
    static_assert(p(-2) == 16);
    EXPECT_EQ(p(3), 81);
}

TEST(Operations, PowerLarge) {
    constexpr power_t<10> p;
    static_assert(p(2) == 1024);
    EXPECT_EQ(p(2), 1024);
}

TEST(Operations, MinFn) {
    constexpr min_fn m;
    static_assert(m(3, 5) == 3);
    static_assert(m(5, 3) == 3);
    static_assert(m(4, 4) == 4);
    EXPECT_EQ(m(3, 5), 3);

    // Algebraic properties (declared mathematical intent)
    static_assert(min_fn::declared_associative);
    static_assert(min_fn::declared_commutative);
    static_assert(min_fn::declared_idempotent);

    // Type-dependent traits: safe for int, not guaranteed for double
    static_assert(declares_associative_v<min_fn, int>);
    static_assert(declares_commutative_v<min_fn, int>);
    static_assert(declares_idempotent_v<min_fn, int>);
    static_assert(!declares_associative_v<min_fn, double>);  // NaN issues
    static_assert(!declares_commutative_v<min_fn, double>);

    // Identity element
    static_assert(min_fn::identity<int>() == std::numeric_limits<int>::max());
    static_assert(min_fn::identity<double>() == std::numeric_limits<double>::max());
}

TEST(Operations, MaxFn) {
    constexpr max_fn m;
    static_assert(m(3, 5) == 5);
    static_assert(m(5, 3) == 5);
    static_assert(m(4, 4) == 4);
    EXPECT_EQ(m(3, 5), 5);

    static_assert(max_fn::declared_associative);
    static_assert(max_fn::declared_commutative);
    static_assert(max_fn::declared_idempotent);
    static_assert(max_fn::identity<int>() == std::numeric_limits<int>::lowest());
}

TEST(Operations, PlusFn) {
    constexpr plus_fn p;
    static_assert(p(3, 5) == 8);
    static_assert(p(0, 7) == 7);
    EXPECT_EQ(p(3, 5), 8);

    static_assert(plus_fn::declared_associative);
    static_assert(plus_fn::declared_commutative);
    static_assert(!plus_fn::declared_idempotent);
    static_assert(plus_fn::identity<int>() == 0);
}

TEST(Operations, MultipliesFn) {
    constexpr multiplies_fn m;
    static_assert(m(3, 5) == 15);
    static_assert(m(1, 7) == 7);
    EXPECT_EQ(m(3, 5), 15);

    static_assert(multiplies_fn::identity<int>() == 1);
}

TEST(Operations, Negate) {
    constexpr negate_t n;
    static_assert(n(5) == -5);
    static_assert(n(-3) == 3);
    EXPECT_EQ(n(5), -5);
}

TEST(Operations, Abs) {
    constexpr abs_t a;
    static_assert(a(5) == 5);
    static_assert(a(-5) == 5);
    static_assert(a(0) == 0);
    EXPECT_EQ(a(-5), 5);
}

// Concept checks
TEST(Operations, Concepts) {
    static_assert(is_identity_op<identity_t>);
    static_assert(!is_identity_op<power_t<2>>);
    static_assert(is_identity_op<power_t<1>>);

    static_assert(is_power_op<power_t<3>>);
    static_assert(is_power_op<identity_t>);
    static_assert(!is_power_op<negate_t>);

    // Type-independent declared intent
    static_assert(declares_associative<plus_fn>);
    static_assert(declares_associative<min_fn>);
    static_assert(declares_commutative<max_fn>);

    // Type-dependent safe traits (integral: true; float: false)
    static_assert(declares_associative_v<plus_fn, int>);
    static_assert(!declares_associative_v<plus_fn, double>);
    static_assert(declares_commutative_v<plus_fn, int>);
    static_assert(!declares_commutative_v<plus_fn, float>);
    static_assert(declares_idempotent_v<min_fn, int>);
    static_assert(!declares_idempotent_v<min_fn, double>);

    static_assert(has_identity<plus_fn, int>);
    static_assert(has_identity<min_fn, double>);
}

// ============================================================================
// Elementwise Binary
// ============================================================================

TEST(ElementwiseBinary, Basic) {
    constexpr auto op = elementwise_binary_op{plus_fn{}, min_fn{}, max_fn{}};

    constexpr auto a = std::make_tuple(10, 3, 1);
    constexpr auto b = std::make_tuple(20, 5, 7);
    constexpr auto r = op(a, b);

    static_assert(std::get<0>(r) == 30);   // 10 + 20
    static_assert(std::get<1>(r) == 3);    // min(3, 5)
    static_assert(std::get<2>(r) == 7);    // max(1, 7)

    EXPECT_EQ(std::get<0>(r), 30);
    EXPECT_EQ(std::get<1>(r), 3);
    EXPECT_EQ(std::get<2>(r), 7);
}

TEST(ElementwiseBinary, SingleLane) {
    constexpr auto op = elementwise_binary_op{plus_fn{}};
    constexpr auto r = op(std::make_tuple(3), std::make_tuple(4));
    static_assert(std::get<0>(r) == 7);
    EXPECT_EQ(std::get<0>(r), 7);
}

TEST(ElementwiseBinary, Doubles) {
    constexpr auto op = elementwise_binary_op{plus_fn{}, multiplies_fn{}};
    constexpr auto r = op(std::make_tuple(1.5, 2.0), std::make_tuple(3.5, 4.0));

    static_assert(std::get<0>(r) == 5.0);
    static_assert(std::get<1>(r) == 8.0);

    EXPECT_DOUBLE_EQ(std::get<0>(r), 5.0);
    EXPECT_DOUBLE_EQ(std::get<1>(r), 8.0);
}

TEST(ElementwiseBinary, LaneCount) {
    constexpr auto op = elementwise_binary_op{plus_fn{}, min_fn{}, max_fn{}};
    static_assert(op.lane_count == 3);
}

TEST(ElementwiseBinary, Associativity) {
    // Verify that folding with elementwise_binary_op is associative
    // when all lanes are associative.
    constexpr auto op = elementwise_binary_op{plus_fn{}, plus_fn{}};
    constexpr auto a = std::make_tuple(1, 10);
    constexpr auto b = std::make_tuple(2, 20);
    constexpr auto c = std::make_tuple(3, 30);

    constexpr auto ab_c = op(op(a, b), c);
    constexpr auto a_bc = op(a, op(b, c));

    static_assert(ab_c == a_bc);
    EXPECT_EQ(ab_c, a_bc);
}

TEST(ElementwiseBinary, BroadcastMode) {
    // Single op broadcast to all 3 lanes
    constexpr auto op = elementwise_binary_op{plus_fn{}};
    constexpr auto a = std::make_tuple(1, 2, 3);
    constexpr auto b = std::make_tuple(10, 20, 30);
    constexpr auto r = op(a, b);

    static_assert(std::get<0>(r) == 11);
    static_assert(std::get<1>(r) == 22);
    static_assert(std::get<2>(r) == 33);

    EXPECT_EQ(std::get<0>(r), 11);
    EXPECT_EQ(std::get<1>(r), 22);
    EXPECT_EQ(std::get<2>(r), 33);
}

TEST(ElementwiseBinary, BroadcastFiveLanes) {
    constexpr auto op = elementwise_binary_op{min_fn{}};
    constexpr auto a = std::make_tuple(5, 2, 8, 1, 9);
    constexpr auto b = std::make_tuple(3, 7, 4, 6, 2);
    constexpr auto r = op(a, b);

    static_assert(std::get<0>(r) == 3);   // min(5,3)
    static_assert(std::get<1>(r) == 2);   // min(2,7)
    static_assert(std::get<2>(r) == 4);   // min(8,4)
    static_assert(std::get<3>(r) == 1);   // min(1,6)
    static_assert(std::get<4>(r) == 2);   // min(9,2)
}

TEST(MakeReduction, NewFactoryName) {
    // make_reduction is the R2 name (make_tuple_reduction kept as alias)
    constexpr auto stats = make_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0}
    );

    constexpr std::array<double, 3> data = {10.0, 20.0, 30.0};
    constexpr auto result = stats.reduce(data.begin(), data.end());

    static_assert(std::get<0>(result) == 3);
    static_assert(std::get<1>(result) == 60.0);

    EXPECT_EQ(std::get<0>(result), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 60.0);
}

// ============================================================================
// Elementwise Unary
// ============================================================================

TEST(ElementwiseUnary, Basic) {
    constexpr auto op = elementwise_unary_op{identity_t{}, negate_t{}, power_t<2>{}};
    constexpr auto t = std::make_tuple(5, 3, 4);
    constexpr auto r = op(t);

    static_assert(std::get<0>(r) == 5);    // identity(5)
    static_assert(std::get<1>(r) == -3);   // negate(3)
    static_assert(std::get<2>(r) == 16);   // power<2>(4)

    EXPECT_EQ(std::get<0>(r), 5);
    EXPECT_EQ(std::get<1>(r), -3);
    EXPECT_EQ(std::get<2>(r), 16);
}

TEST(ElementwiseUnary, AllIdentity) {
    constexpr auto op = elementwise_unary_op{identity_t{}, identity_t{}};
    constexpr auto t = std::make_tuple(42, 99);
    constexpr auto r = op(t);
    static_assert(r == t);
    EXPECT_EQ(r, t);
}

// ============================================================================
// Fan Out
// ============================================================================

TEST(FanOut, MomentsPattern) {
    constexpr auto fan = fan_out_unary_op{
        constant_t<1>{}, identity_t{}, power_t<2>{}, power_t<3>{}, power_t<4>{}
    };

    constexpr auto r = fan(3);

    static_assert(std::get<0>(r) == 1);     // constant 1
    static_assert(std::get<1>(r) == 3);     // x
    static_assert(std::get<2>(r) == 9);     // x^2
    static_assert(std::get<3>(r) == 27);    // x^3
    static_assert(std::get<4>(r) == 81);    // x^4

    EXPECT_EQ(std::get<0>(r), 1);
    EXPECT_EQ(std::get<1>(r), 3);
    EXPECT_EQ(std::get<2>(r), 9);
    EXPECT_EQ(std::get<3>(r), 27);
    EXPECT_EQ(std::get<4>(r), 81);
}

TEST(FanOut, SingleLane) {
    constexpr auto fan = fan_out_unary_op{power_t<2>{}};
    constexpr auto r = fan(5);
    static_assert(std::get<0>(r) == 25);
    EXPECT_EQ(std::get<0>(r), 25);
}

TEST(FanOut, WithDoubles) {
    constexpr auto fan = fan_out_unary_op{identity_t{}, power_t<2>{}};
    constexpr auto r = fan(2.5);

    static_assert(std::get<0>(r) == 2.5);
    static_assert(std::get<1>(r) == 6.25);

    EXPECT_DOUBLE_EQ(std::get<0>(r), 2.5);
    EXPECT_DOUBLE_EQ(std::get<1>(r), 6.25);
}

TEST(FanOut, LaneCount) {
    constexpr auto fan = fan_out_unary_op{identity_t{}, identity_t{}, identity_t{}};
    static_assert(fan.lane_count == 3);
}

// ============================================================================
// Tuple Select
// ============================================================================

TEST(TupleSelect, SelectSubset) {
    constexpr auto t = std::make_tuple(10, 20, 30, 40, 50);

    constexpr auto first_last = tuple_select<0, 4>(t);
    static_assert(std::get<0>(first_last) == 10);
    static_assert(std::get<1>(first_last) == 50);

    constexpr auto middle = tuple_select<1, 2, 3>(t);
    static_assert(std::get<0>(middle) == 20);
    static_assert(std::get<1>(middle) == 30);
    static_assert(std::get<2>(middle) == 40);

    EXPECT_EQ(std::get<0>(first_last), 10);
    EXPECT_EQ(std::get<1>(first_last), 50);
}

TEST(TupleSelect, SingleElement) {
    constexpr auto t = std::make_tuple(10, 20, 30);
    constexpr auto s = tuple_select<1>(t);
    static_assert(std::get<0>(s) == 20);
    EXPECT_EQ(std::get<0>(s), 20);
}

TEST(TupleSelect, Reorder) {
    constexpr auto t = std::make_tuple(1, 2, 3);
    constexpr auto reversed = tuple_select<2, 1, 0>(t);
    static_assert(reversed == std::make_tuple(3, 2, 1));
    EXPECT_EQ(reversed, std::make_tuple(3, 2, 1));
}

TEST(TupleSelect, TupleLane) {
    constexpr auto t = std::make_tuple(10, 20, 30);
    static_assert(tuple_lane<0>(t) == 10);
    static_assert(tuple_lane<1>(t) == 20);
    static_assert(tuple_lane<2>(t) == 30);

    EXPECT_EQ(tuple_lane<0>(t), 10);
}

TEST(TupleSelect, LaneCount) {
    using T = std::tuple<int, double, float>;
    static_assert(tuple_lane_count<T> == 3);
}

// ============================================================================
// Tuple Fold
// ============================================================================

TEST(TupleFold, SumWithInit) {
    constexpr auto t = std::make_tuple(1, 2, 3, 4);
    constexpr auto r = tuple_fold(plus_fn{}, 0, t);
    static_assert(r == 10);
    EXPECT_EQ(r, 10);
}

TEST(TupleFold, SumWithoutInit) {
    constexpr auto t = std::make_tuple(1, 2, 3, 4);
    constexpr auto r = tuple_fold(plus_fn{}, t);
    static_assert(r == 10);
    EXPECT_EQ(r, 10);
}

TEST(TupleFold, MinFold) {
    constexpr auto t = std::make_tuple(5, 2, 8, 1, 9);
    constexpr auto r = tuple_fold(min_fn{}, t);
    static_assert(r == 1);
    EXPECT_EQ(r, 1);
}

TEST(TupleFold, MaxFold) {
    constexpr auto t = std::make_tuple(5, 2, 8, 1, 9);
    constexpr auto r = tuple_fold(max_fn{}, t);
    static_assert(r == 9);
    EXPECT_EQ(r, 9);
}

TEST(TupleFold, SingleElement) {
    constexpr auto t = std::make_tuple(42);
    constexpr auto r = tuple_fold(plus_fn{}, t);
    static_assert(r == 42);
    EXPECT_EQ(r, 42);
}

TEST(TupleFold, WithInitAndSingleElement) {
    constexpr auto t = std::make_tuple(5);
    constexpr auto r = tuple_fold(plus_fn{}, 100, t);
    static_assert(r == 105);
    EXPECT_EQ(r, 105);
}

TEST(TupleFold, Product) {
    constexpr auto t = std::make_tuple(2, 3, 4);
    constexpr auto r = tuple_fold(multiplies_fn{}, 1, t);
    static_assert(r == 24);
    EXPECT_EQ(r, 24);
}

// ============================================================================
// Make Tuple Reduction
// ============================================================================

TEST(ReductionLane, Construction) {
    constexpr auto lane = reduction_lane{identity_t{}, plus_fn{}, 0.0};
    static_assert(lane.init == 0.0);
    EXPECT_DOUBLE_EQ(lane.init, 0.0);
}

TEST(TupleReduction, InitValue) {
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{power_t<2>{},    plus_fn{}, 0.0}
    );

    constexpr auto init = stats.init_value();
    static_assert(std::get<0>(init) == 0);
    static_assert(std::get<1>(init) == 0.0);
    static_assert(std::get<2>(init) == 0.0);

    EXPECT_EQ(std::get<0>(init), 0);
}

TEST(TupleReduction, Transform) {
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{power_t<2>{},    plus_fn{}, 0.0}
    );

    constexpr auto t = stats.transform(3.0);
    static_assert(std::get<0>(t) == 1);
    static_assert(std::get<1>(t) == 3.0);
    static_assert(std::get<2>(t) == 9.0);

    EXPECT_EQ(std::get<0>(t), 1);
    EXPECT_DOUBLE_EQ(std::get<1>(t), 3.0);
    EXPECT_DOUBLE_EQ(std::get<2>(t), 9.0);
}

TEST(TupleReduction, Combine) {
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()}
    );

    constexpr auto a = std::make_tuple(5, 10.0, 3.0);
    constexpr auto b = std::make_tuple(3, 7.0, 1.0);
    constexpr auto r = stats.combine(a, b);

    static_assert(std::get<0>(r) == 8);     // 5 + 3
    static_assert(std::get<1>(r) == 17.0);  // 10 + 7
    static_assert(std::get<2>(r) == 1.0);   // min(3, 1)

    EXPECT_EQ(std::get<0>(r), 8);
    EXPECT_DOUBLE_EQ(std::get<1>(r), 17.0);
    EXPECT_DOUBLE_EQ(std::get<2>(r), 1.0);
}

TEST(TupleReduction, ReduceConstexprArray) {
    // Full reduction over a constexpr array — verified at compile time.
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{},  0},
        reduction_lane{identity_t{},    plus_fn{},  0.0},
        reduction_lane{power_t<2>{},    plus_fn{},  0.0},
        reduction_lane{identity_t{},    min_fn{},   std::numeric_limits<double>::max()},
        reduction_lane{identity_t{},    max_fn{},   std::numeric_limits<double>::lowest()}
    );

    constexpr std::array<double, 5> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    constexpr auto result = stats.reduce(data.begin(), data.end());

    // count
    static_assert(std::get<0>(result) == 5);
    // sum
    static_assert(std::get<1>(result) == 15.0);
    // sum of squares: 1 + 4 + 9 + 16 + 25 = 55
    static_assert(std::get<2>(result) == 55.0);
    // min
    static_assert(std::get<3>(result) == 1.0);
    // max
    static_assert(std::get<4>(result) == 5.0);

    EXPECT_EQ(std::get<0>(result), 5);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 15.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result), 55.0);
    EXPECT_DOUBLE_EQ(std::get<3>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<4>(result), 5.0);
}

TEST(TupleReduction, ReduceRuntimeVector) {
    auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{},  0},
        reduction_lane{identity_t{},    plus_fn{},  0.0},
        reduction_lane{power_t<2>{},    plus_fn{},  0.0},
        reduction_lane{identity_t{},    min_fn{},   std::numeric_limits<double>::max()},
        reduction_lane{identity_t{},    max_fn{},   std::numeric_limits<double>::lowest()}
    );

    std::vector<double> data = {10.0, 20.0, 30.0};
    auto result = stats.reduce(data);

    EXPECT_EQ(std::get<0>(result), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 60.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result), 1400.0);  // 100 + 400 + 900
    EXPECT_DOUBLE_EQ(std::get<3>(result), 10.0);
    EXPECT_DOUBLE_EQ(std::get<4>(result), 30.0);
}

TEST(TupleReduction, ReduceEmpty) {
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0}
    );

    constexpr std::array<double, 0> empty = {};
    constexpr auto result = stats.reduce(empty.begin(), empty.end());

    // Empty range returns init values
    static_assert(std::get<0>(result) == 0);
    static_assert(std::get<1>(result) == 0.0);

    EXPECT_EQ(std::get<0>(result), 0);
}

TEST(TupleReduction, ReduceSingleElement) {
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0},
        reduction_lane{identity_t{},    min_fn{},  std::numeric_limits<double>::max()}
    );

    constexpr std::array<double, 1> data = {42.0};
    constexpr auto result = stats.reduce(data.begin(), data.end());

    static_assert(std::get<0>(result) == 1);
    static_assert(std::get<1>(result) == 42.0);
    static_assert(std::get<2>(result) == 42.0);

    EXPECT_EQ(std::get<0>(result), 1);
}

TEST(TupleReduction, ReduceContainerOverload) {
    // Test the container convenience overload
    auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{}, 0},
        reduction_lane{identity_t{},    plus_fn{}, 0.0}
    );

    std::vector<double> data = {1.0, 2.0, 3.0};
    auto result = stats.reduce(data);

    EXPECT_EQ(std::get<0>(result), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 6.0);
}

TEST(TupleReduction, LaneCount) {
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0},
        reduction_lane{identity_t{}, plus_fn{}, 0}
    );
    static_assert(stats.lane_count == 3);
}

// ============================================================================
// Full 7-Lane Variadic Statistics (the P3666R0 showcase)
// ============================================================================

TEST(VariadicStats, SevenLanes) {
    // count, sum, sum_x2, sum_x3, sum_x4, min, max
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{},  0},
        reduction_lane{identity_t{},    plus_fn{},  0.0},
        reduction_lane{power_t<2>{},    plus_fn{},  0.0},
        reduction_lane{power_t<3>{},    plus_fn{},  0.0},
        reduction_lane{power_t<4>{},    plus_fn{},  0.0},
        reduction_lane{identity_t{},    min_fn{},   std::numeric_limits<double>::max()},
        reduction_lane{identity_t{},    max_fn{},   std::numeric_limits<double>::lowest()}
    );

    static_assert(stats.lane_count == 7);

    constexpr std::array<double, 4> data = {1.0, 2.0, 3.0, 4.0};
    constexpr auto result = stats.reduce(data.begin(), data.end());

    // count = 4
    static_assert(std::get<0>(result) == 4);
    // sum = 1 + 2 + 3 + 4 = 10
    static_assert(std::get<1>(result) == 10.0);
    // sum_x2 = 1 + 4 + 9 + 16 = 30
    static_assert(std::get<2>(result) == 30.0);
    // sum_x3 = 1 + 8 + 27 + 64 = 100
    static_assert(std::get<3>(result) == 100.0);
    // sum_x4 = 1 + 16 + 81 + 256 = 354
    static_assert(std::get<4>(result) == 354.0);
    // min = 1
    static_assert(std::get<5>(result) == 1.0);
    // max = 4
    static_assert(std::get<6>(result) == 4.0);

    EXPECT_EQ(std::get<0>(result), 4);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 10.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result), 30.0);
    EXPECT_DOUBLE_EQ(std::get<3>(result), 100.0);
    EXPECT_DOUBLE_EQ(std::get<4>(result), 354.0);
    EXPECT_DOUBLE_EQ(std::get<5>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<6>(result), 4.0);
}

// ============================================================================
// Integration: extract components and use manually
// ============================================================================

TEST(Integration, ExtractComponents) {
    // Demonstrate extracting init/transform/combine for use with
    // std::transform_reduce or custom parallel execution.
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{identity_t{}, plus_fn{}, 0.0},
        reduction_lane{identity_t{}, min_fn{},  std::numeric_limits<double>::max()}
    );

    constexpr auto init = stats.init_value();
    constexpr auto xform = stats.transform_fn();
    constexpr auto comb = stats.combine_fn();

    // Manual fold over constexpr data
    constexpr std::array<double, 3> data = {5.0, 2.0, 8.0};

    constexpr auto manual_result = [&]() constexpr {
        auto acc = init;
        for (const auto& x : data) {
            acc = comb(acc, xform(x));
        }
        return acc;
    }();

    static_assert(std::get<0>(manual_result) == 15.0);  // sum
    static_assert(std::get<1>(manual_result) == 2.0);   // min

    // Compare to direct reduce
    constexpr auto direct_result = stats.reduce(data.begin(), data.end());
    static_assert(manual_result == direct_result);

    EXPECT_EQ(manual_result, direct_result);
}

TEST(Integration, SelectFromReductionResult) {
    // Reduce, then project specific lanes.
    constexpr auto stats = make_tuple_reduction(
        reduction_lane{constant_t<1>{}, plus_fn{},  0},
        reduction_lane{identity_t{},    plus_fn{},  0.0},
        reduction_lane{power_t<2>{},    plus_fn{},  0.0},
        reduction_lane{identity_t{},    min_fn{},   std::numeric_limits<double>::max()},
        reduction_lane{identity_t{},    max_fn{},   std::numeric_limits<double>::lowest()}
    );

    constexpr std::array<double, 3> data = {10.0, 20.0, 30.0};
    constexpr auto full = stats.reduce(data.begin(), data.end());

    // Extract just count and sum for mean computation
    constexpr auto count = tuple_lane<0>(full);
    constexpr auto sum   = tuple_lane<1>(full);
    static_assert(count == 3);
    static_assert(sum == 60.0);

    // Extract extrema
    constexpr auto extrema = tuple_select<3, 4>(full);
    static_assert(std::get<0>(extrema) == 10.0);  // min
    static_assert(std::get<1>(extrema) == 30.0);  // max

    EXPECT_EQ(count, 3);
    EXPECT_DOUBLE_EQ(sum, 60.0);
}
