// Counterexamples from the CT-DP semantic/graph red-team review.
// This suite is registered in default CI, not an optional research driver.
#include <ctdp/graph/graph_builder.h>
#include <ctdp/graph/kernel_info.h>
#include <ctdp/graph/fuse_group.h>
#include <ctdp/engine/bridge/graph_to_space.h>
#include <ctdp/engine/bridge/graph_to_constraints.h>
#include <ctdp/engine/bridge/segmentation_dp.h>
#include <ctdp/core/cost_vector.h>
#include <ctdp/core/pareto.h>
#include <ctdp/space/descriptor.h>
#include <ctdp/space/space.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace g = ctdp::graph;
namespace s = ctdp::space;

constexpr auto reversed_chain() {
    g::graph_builder<g::cap_from<3, 2>> b;
    auto n0=b.add_node(), n1=b.add_node(), n2=b.add_node();
    b.add_edge(n2,n1);
    b.add_edge(n1,n0);
    return b.finalise();
}
constexpr auto forward_chain() {
    g::graph_builder<g::cap_from<3, 2>> b;
    auto n0=b.add_node(), n1=b.add_node(), n2=b.add_node();
    b.add_edge(n0,n1);
    b.add_edge(n1,n2);
    return b.finalise();
}
constexpr auto reverse_g=reversed_chain();
constexpr auto forward_g=forward_chain();
constexpr auto reverse_k=g::make_uniform_kernel_map<3>(
    reverse_g, g::kernel_info{.tag=g::kernel_tag{1}, .flops=8, .is_fusable=true});
constexpr auto forward_k=g::make_uniform_kernel_map<3>(
    forward_g, g::kernel_info{.tag=g::kernel_tag{1}, .flops=8, .is_fusable=true});
constexpr auto reversed_space=g::build_schedule_space<3,2>(reverse_g,reverse_k);
constexpr auto forward_space=g::build_schedule_space<3,2>(forward_g,forward_k);
constexpr auto reversed_constraints=g::build_constraints<3,2>(reverse_g,reversed_space);
constexpr auto forward_constraints=g::build_constraints<3,2>(forward_g,forward_space);
static_assert(reversed_constraints.critical_path_length == 2,
    "Original vertex IDs must not affect critical path length");
static_assert(forward_constraints.critical_path_length ==
              reversed_constraints.critical_path_length);

// Two eligible input edges must fit an exact MaxE=2 capacity even when
// connected-component discovery treats eligibility as undirected.
constexpr auto fused_reverse = g::find_fusion_groups<3,2>(reverse_g,reverse_k);
static_assert(fused_reverse.group_count == 1);
static_assert(fused_reverse.fused_edge_count == 2);
static_assert(fused_reverse.is_valid_dag);

constexpr auto segment_solution = g::solve_segmented<3>(
    reversed_space,
    [](auto const& group) constexpr {
        g::segment_result<3> r;
        r.group_id=group.group_id;
        r.feasible=true;
        r.cost=4.0;
        return r;
    });
static_assert(segment_solution.segment_count == 3);
static_assert(g::validate_segmented(segment_solution,
    reversed_constraints.dependencies,reversed_space));
static_assert([]{
    auto invalid=segment_solution;
    invalid.all_feasible=false;
    return !g::validate_segmented(invalid,reversed_constraints.dependencies);
}());
static_assert([]{
    auto invalid=segment_solution;
    invalid.segment_cost[0]=std::numeric_limits<double>::quiet_NaN();
    return !g::validate_segmented(invalid,reversed_constraints.dependencies);
}());
static_assert([]{
    auto invalid=segment_solution;
    invalid.total_cost+=1.0;
    return !g::validate_segmented(invalid,reversed_constraints.dependencies);
}());
static_assert([]{
    auto limit=g::resource_constraint{.max_flops_per_group=3};
    return !g::validate_segmented(segment_solution,
        reversed_constraints.dependencies,reversed_space,limit);
}());

constexpr auto quiet_nan_value=std::numeric_limits<double>::quiet_NaN();
constexpr ctdp::cost_vector<2> bad{{quiet_nan_value,0.0}};
constexpr ctdp::cost_vector<2> good{{1.0,1.0}};
static_assert(!bad.is_finite());
static_assert(!bad.dominates(good));
static_assert(!good.dominates(bad));

constexpr auto descriptor=s::descriptor_space(
    "nan_cost_regression",s::make_int_set("choice",{0,1,2}));
constexpr auto nan_cost=[](auto const& point) constexpr {
    return std::get<0>(point)==0 ? quiet_nan_value : static_cast<double>(3-std::get<0>(point));
};
constexpr auto scored=s::exhaustive_search_with_cost(descriptor,nan_cost);
static_assert(scored.best_cost == 1.0 && std::get<0>(scored.best)==2);
constexpr auto multi_cost=[](auto const& pt) constexpr {
    if(std::get<0>(pt)==0) return ctdp::cost_vector<2>{{quiet_nan_value,0}};
    return ctdp::cost_vector<2>{{static_cast<double>(std::get<0>(pt)),4.0}};
};
constexpr auto pareto=ctdp::pareto_search<2,3>(descriptor,multi_cost);
static_assert(pareto.size()==1);
static_assert(std::get<0>(pareto[0].candidate)==1);

TEST(RedteamSoundness, CriticalPathRelabelInvariant) {
    EXPECT_EQ(forward_constraints.critical_path_length,2u);
    EXPECT_EQ(reversed_constraints.critical_path_length,2u);
}
TEST(RedteamSoundness, FusibilityCapacityDoesNotDoubleInputEdges) {
    EXPECT_EQ(fused_reverse.group_count,1u);
    EXPECT_EQ(fused_reverse.fused_edge_count,2u);
}
TEST(RedteamSoundness, SegmentedChecksActualResult) {
    EXPECT_TRUE(g::validate_segmented(segment_solution,
         reversed_constraints.dependencies,reversed_space));
    auto x=segment_solution;
    x.infeasible_count=1;
    EXPECT_FALSE(g::validate_segmented(x,reversed_constraints.dependencies));
    x=segment_solution;
    x.total_cost=quiet_nan_value;
    EXPECT_FALSE(g::validate_segmented(x,reversed_constraints.dependencies));
}
TEST(RedteamSoundness, NaNIsNotAValidPlanCost) {
    EXPECT_DOUBLE_EQ(scored.best_cost,1.0);
    EXPECT_EQ(std::get<0>(scored.best),2);
    EXPECT_EQ(pareto.size(),1u);
}
