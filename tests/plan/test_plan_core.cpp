// ============================================================================
// CT-DP Phase 5 — Plan core static_assert test suite
// ============================================================================
// Compile-time tests. If this file compiles, the foundational headers are
// structurally correct. No runtime code.
// ============================================================================

#include "ct_dp/plan/plan_ast.hpp"

#include <tuple>
#include <type_traits>

namespace p = ct_dp::plan;
namespace w = ct_dp::plan::wrap;

// ----------------------------------------------------------------------------
// Test fixtures
// ----------------------------------------------------------------------------

struct kernel_A {};
struct kernel_B {};

struct delegate_to_compiler {};   // family: compute_strategy
struct canonical_reduce    {};    // family: reduce_topology

struct fusion_groups_t { constexpr fusion_groups_t() = default; };
inline constexpr fusion_groups_t fusion_groups{};

template<int D> struct domain {};
template<class Domain, int TX, int TY> struct tessellation {};

struct always_true_pred { static constexpr bool value = true; };
struct openmp_policy {};

// ----------------------------------------------------------------------------
// Sample plans
// ----------------------------------------------------------------------------

using bare_leaf = p::leaf<kernel_A, delegate_to_compiler>;

using bare_plan =
    p::nest_t<
        tessellation<domain<2>, 64, 16>,
        p::split_t<fusion_groups,
            p::leaf<kernel_A, delegate_to_compiler>,
            p::leaf<kernel_B, canonical_reduce>
        >
    >;

using one_wrapper   = w::vectorise<8, bare_plan>;
using two_wrappers  = w::unroll<4, w::vectorise<8, bare_plan>>;
using three_wrappers =
    w::parallelise<openmp_policy,
        w::unroll<4, w::vectorise<8, bare_plan>>
    >;

using with_choose = p::choose_t<always_true_pred, bare_leaf, bare_plan>;

// ----------------------------------------------------------------------------
// Plan universe
// ----------------------------------------------------------------------------

static_assert(p::is_plan_v<bare_leaf>);
static_assert(p::is_plan_v<bare_plan>);
static_assert(p::is_plan_v<one_wrapper>);
static_assert(p::is_plan_v<two_wrappers>);
static_assert(p::is_plan_v<three_wrappers>);
static_assert(p::is_plan_v<with_choose>);

static_assert(p::is_core_plan_v<bare_leaf>);
static_assert(p::is_core_plan_v<bare_plan>);
static_assert(!p::is_core_plan_v<one_wrapper>);
static_assert(!p::is_core_plan_v<two_wrappers>);

struct not_a_plan {};
static_assert(!p::is_plan_v<not_a_plan>);
static_assert(!p::is_plan_v<int>);

static_assert(p::plan<bare_leaf>);
static_assert(p::plan<two_wrappers>);

// ----------------------------------------------------------------------------
// is_core_plan agrees with exactly one shape trait per plan
// ----------------------------------------------------------------------------
// Guards against the dual-mechanism drift the baseline commit eliminates.

template<class P>
inline constexpr int shape_count_v =
    int(p::is_leaf_v<P>) + int(p::is_seq_v<P>) + int(p::is_nest_v<P>) +
    int(p::is_split_v<P>) + int(p::is_choose_v<P>);

static_assert(shape_count_v<bare_leaf> == 1);
static_assert(shape_count_v<bare_plan> == 1);
static_assert(shape_count_v<with_choose> == 1);
static_assert(shape_count_v<one_wrapper> == 0);     // wrapper: no core shape
static_assert(shape_count_v<not_a_plan> == 0);

// ----------------------------------------------------------------------------
// Shape traits
// ----------------------------------------------------------------------------

static_assert(p::is_leaf_v<bare_leaf>);
static_assert(!p::is_leaf_v<bare_plan>);
static_assert(p::is_nest_v<bare_plan>);
static_assert(!p::is_nest_v<bare_leaf>);
static_assert(p::is_choose_v<with_choose>);
static_assert(p::is_wrapper_v<one_wrapper>);
static_assert(p::is_wrapper_v<two_wrappers>);
static_assert(!p::is_wrapper_v<bare_plan>);

// ----------------------------------------------------------------------------
// plan_children totality
// ----------------------------------------------------------------------------
// The baseline commit's headline navigation guarantee: plan_children_t
// compiles for every plan in the universe.

static_assert(std::is_same_v<
    p::plan_children_t<bare_leaf>,
    std::tuple<>
>);

static_assert(std::is_same_v<
    p::plan_children_t<one_wrapper>,
    std::tuple<bare_plan>
>);

static_assert(std::is_same_v<
    p::plan_children_t<two_wrappers>,
    std::tuple<w::vectorise<8, bare_plan>>
>);

// Core composite children
static_assert(std::tuple_size_v<p::plan_children_t<bare_plan>> == 1);   // nest
static_assert(std::tuple_size_v<p::plan_children_t<with_choose>> == 2); // choose

// ----------------------------------------------------------------------------
// Wrapper navigation
// ----------------------------------------------------------------------------

static_assert(std::is_same_v<p::wrapper_inner_t<one_wrapper>, bare_plan>);
static_assert(std::is_same_v<
    p::wrapper_inner_t<two_wrappers>,
    w::vectorise<8, bare_plan>
>);

static_assert(std::is_same_v<p::strip_wrappers_t<one_wrapper>,    bare_plan>);
static_assert(std::is_same_v<p::strip_wrappers_t<two_wrappers>,   bare_plan>);
static_assert(std::is_same_v<p::strip_wrappers_t<three_wrappers>, bare_plan>);
static_assert(std::is_same_v<p::strip_wrappers_t<bare_plan>,      bare_plan>);

static_assert(std::is_same_v<
    p::rewrap_t<w::vectorise<8, bare_plan>, bare_leaf>,
    w::vectorise<8, bare_leaf>
>);

// ----------------------------------------------------------------------------
// Leaf / nest / choose navigation
// ----------------------------------------------------------------------------

static_assert(std::is_same_v<p::leaf_kernel_t<bare_leaf>, kernel_A>);
static_assert(std::is_same_v<
    p::leaf_attributes_t<bare_leaf>,
    std::tuple<delegate_to_compiler>
>);
static_assert(std::is_same_v<
    p::nest_tessellation_t<bare_plan>,
    tessellation<domain<2>, 64, 16>
>);
static_assert(std::is_same_v<p::choose_predicate_t<with_choose>, always_true_pred>);

// ----------------------------------------------------------------------------
// nth_child_t convenience
// ----------------------------------------------------------------------------

static_assert(std::is_same_v<p::nth_child_t<0, with_choose>, bare_leaf>);
static_assert(std::is_same_v<p::nth_child_t<1, with_choose>, bare_plan>);

// ----------------------------------------------------------------------------
// Empty seq<> is legal identity
// ----------------------------------------------------------------------------

using empty_seq = p::seq_t<>;
static_assert(p::is_plan_v<empty_seq>);
static_assert(p::is_seq_v<empty_seq>);
static_assert(empty_seq::arity == 0);
static_assert(std::is_same_v<p::plan_children_t<empty_seq>, std::tuple<>>);

// ----------------------------------------------------------------------------
// all_plans_v utility
// ----------------------------------------------------------------------------

static_assert(p::all_plans_v<bare_leaf, bare_plan, one_wrapper>);
static_assert(!p::all_plans_v<bare_leaf, not_a_plan>);

// ----------------------------------------------------------------------------
// Alias-name consistency spot checks
// ----------------------------------------------------------------------------
// Confirms the _t naming pass. If these break, the rename is incomplete.

static_assert(std::is_same_v<bare_leaf::kernel_t, kernel_A>);
static_assert(std::is_same_v<bare_plan::inner_t,
    p::split_t<fusion_groups,
        p::leaf<kernel_A, delegate_to_compiler>,
        p::leaf<kernel_B, canonical_reduce>
    >
>);
static_assert(std::is_same_v<with_choose::true_plan_t,  bare_leaf>);
static_assert(std::is_same_v<with_choose::false_plan_t, bare_plan>);

// GoogleTest provides main() via GTest::gtest_main (see tests/CMakeLists.txt
// ctdp_add_test helper). A dummy TEST case ensures gtest_discover_tests
// reports one passing case to CTest. The real validation is the 56
// static_asserts above, all evaluated at compile time.
#include <gtest/gtest.h>
TEST(PlanCore, StaticAssertsCompiledSuccessfully) { SUCCEED(); }
