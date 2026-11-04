// Shared optimisation protocol: three *different* CT-DP solver families.
// Standalone Release assertions, no testing framework or hardware required.
#include <ctdp/solver/cross_domain_protocol.h>
#include <ctdp/space/descriptor.h>
#include <ctdp/solver/algorithms/interval_dp.h>
#include <ctdp/solver/cost_models/chain.h>
#include <ctdp/solver/algorithms/compositional_pareto.h>
#include <ctdp/core/cost_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ci=ctdp::solver::integration;
static void require(bool b,const char* message) {
    if(!b) throw std::runtime_error(message);
}
struct numerical_contract { std::uint32_t id{}; };

constexpr bool descriptor_exhaustive_constexpr() {
    auto space=ctdp::space::descriptor_space("function.knobs",
        ctdp::space::positive_int("degree",1,7));
    const int target=4;
    const numerical_contract contract{3};
    auto p=ci::problem(target,space,contract);
    auto solver=ci::adapt_space_solver(
        [](auto const& native_space, int desired,
           numerical_contract const& c) constexpr {
            return ctdp::space::exhaustive_search_with_cost(native_space,
                [=](auto const& point) constexpr -> double {
                    const int x=std::get<0>(point);
                    const int d=x-desired;
                    return double(d*d + (c.id-3));
                });
        });
    auto out=ci::explore(p,solver,
        {"descriptor_exhaustive",ci::model_status::complete_for_declared_space});
    return std::get<0>(out.native.best)==4 &&
           out.native.best_cost==0.0 &&
           out.native.evaluated==7 &&
           out.search.family=="descriptor_exhaustive" &&
           out.search.status==ci::model_status::complete_for_declared_space;
}
static_assert(descriptor_exhaustive_constexpr(),
    "cross-domain protocol broke constexpr CT-DP descriptor search");

constexpr bool interval_constexpr() {
    constexpr std::array<std::size_t,7> dimensions{30,35,15,5,10,20,25};
    const ctdp::interval_split_space<7> space{6};
    const numerical_contract c{2};
    auto solver=ci::adapt_space_solver(
        [](auto const& interval_space,auto const& dims,
           numerical_contract const&) constexpr {
            return ctdp::interval_dp(interval_space,ctdp::make_chain_cost(dims));
        });
    auto out=ci::explore(ci::problem(dimensions,space,c),solver,
        {"interval_dp",ci::model_status::complete_for_declared_space});
    return out.native.predicted_cost==15125.0 &&
           out.native.stats.candidates_evaluated==35 &&
           out.native.params.n==6 && out.native.is_feasible();
}
static_assert(interval_constexpr(),
    "cross-domain protocol broke constexpr interval planning");

struct pareto_label {
    std::vector<int> steps;
    ctdp::cost_vector<2> objectives{};
};
static auto pareto_cost=[](pareto_label const& l){return l.objectives;};
static auto pareto_less=[](pareto_label const& a,pareto_label const& b){
    return a.steps<b.steps;
};
static auto pareto_priority=[](pareto_label const& l){
    return l.objectives[0]+l.objectives[1];
};
static auto pareto_expand=[](int n,auto const& children,auto const& emit) {
    if(n==1) { emit(pareto_label{}); return; }
    for(int r:{2,4}) {
        if(n%r!=0)continue;
        for(auto const& child:children(n/r)) {
            pareto_label candidate=child;
            candidate.steps.insert(candidate.steps.begin(),r);
            candidate.objectives[0]+=double(r);
            candidate.objectives[1]+=double(8/r);
            emit(std::move(candidate));
        }
    }
};
static void recursive_pareto_native() {
    const int n=8;
    const numerical_contract c{4};
    // A recursive domain's native space is its root key, not a Cartesian
    // descriptor. This is a genuine memoised/compositional CT-DP solve.
    auto solver=ci::adapt_space_solver(
        [](int root,int original,numerical_contract const& req) {
            require(root==original && req.id==4,"domain instance mismatch");
            return ctdp::solver::physical::compositional_pareto<
                int,pareto_label,2>(
                    root,{16,100,32},pareto_expand,pareto_cost,
                    pareto_less,pareto_priority);
        });
    auto out=ci::explore(ci::problem(n,n,c),solver,
        {"compositional_pareto",ci::model_status::unverified});
    require(!out.native.approximate,"untruncated Pareto search should be complete");
    require(!out.native.frontier.empty(),"native Pareto frontier lost");
    require(out.native.subproblems>1 && out.native.generated>1,
            "recursive native solver was not executed");
    require(out.search.status==ci::model_status::unverified,
            "generic protocol must not manufacture a model optimality proof");
}
static void bounded_pareto_status() {
    const int root=1;
    const numerical_contract c{4};
    auto bounded=[](int,auto const&,auto const& emit) {
        emit(pareto_label{{1},{{1,5}}});
        emit(pareto_label{{2},{{3,3}}});
        emit(pareto_label{{3},{{5,1}}});
    };
    auto solver=[&](auto const&) {
        return ctdp::solver::physical::compositional_pareto<int,pareto_label,2>(
            root,{2,50,10},bounded,pareto_cost,pareto_less,pareto_priority);
    };
    auto out=ci::explore(ci::problem(root,root,c),solver,
        {"pareto_bounded",ci::model_status::bounded_approximate});
    require(out.native.approximate && out.native.frontier.size()==2,
            "bounded native frontier/truncation flags lost");
    require(out.search.status==ci::model_status::bounded_approximate,
            "bounded status not preserved");
}
static void domain_isolation() {
    // Each domain may use exactly the same optimiser protocol while preserving
    // its own native instance, graph and numerical-contract types.
    struct shape { int m,n,k; };
    struct backend { int id; };
    const shape problem_shape{64,32,16};
    const int domain_space=12;
    const backend contract{7};
    auto s=ci::adapt_space_solver([](int space,shape const& i,backend const& c) {
        return i.m+i.n+i.k+space+c.id;
    });
    const auto r=ci::explore(ci::problem(problem_shape,domain_space,contract),
        s,{"matrix.domain",ci::model_status::unverified});
    static_assert(std::is_same_v<decltype(r.native),const int>,
                  "native solver result type changed");
    require(r.native==131,"typed problem forwarding changed");
}
int main() {
    require(descriptor_exhaustive_constexpr(),"descriptor constexpr failed");
    require(interval_constexpr(),"interval constexpr failed");
    recursive_pareto_native();
    bounded_pareto_status();
    domain_isolation();
}
