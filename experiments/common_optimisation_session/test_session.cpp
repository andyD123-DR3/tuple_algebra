#include <ctdp/solver/optimisation_session.h>
#include <ctdp/solver/algorithms/compositional_pareto.h>
#include <ctdp/space/descriptor.h>
#include <ctdp/core/cost_vector.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace so=ctdp::solver::session;
struct candidate {
    std::uint64_t id{};
    double work{},storage{};
};
struct native_report {
    std::vector<candidate> frontier;
    std::size_t evaluated{};
};
void check(bool ok,const char* why) {
    if(!ok)throw std::runtime_error(why);
}
template<class F>void rejected(F&& f) {
    bool did_throw=false;
    try{f();}catch(std::invalid_argument const&){did_throw=true;}
    catch(std::length_error const&){did_throw=true;}
    check(did_throw,"expected invalid request/evidence to be rejected");
}
so::identity same_scope() { return {11,22,33,44,55,66}; }
so::evidence passed(so::identity scope,double runtime,double prepare=0) {
    so::evidence e;
    e.level=so::evidence_level::numerically_qualified;
    e.observed=scope;
    e.actual_execution=e.complete_scope=e.independent_reference=e.numerical_pass=true;
    e.observed_inputs=128;
    e.observed_bytes=512;
    e.reference_witness=123;
    e.candidate_witness=234; // different digests allowed under bounded-ULP tolerances
    e.runtime_ns=runtime;
    e.preparation_ns=prepare;
    return e;
}
int main() {
    // Real CT-DP Pareto solver: its two incomparable domain-native labels
    // remain intact; a model-only result cannot be selected for execution.
    const auto model=so::run(so::request{same_scope(),7,16},
        [] {
            return ctdp::solver::physical::compositional_pareto<
                int,candidate,2>(0,{8,32,8},
                [](int,auto const&,auto const& emit) {
                    emit(candidate{10,2.0,6.0});
                    emit(candidate{20,6.0,2.0});
                    emit(candidate{30,7.0,7.0}); // dominated
                },
                [](candidate const& p)->ctdp::cost_vector<2> {
                    return {{p.work,p.storage}};
                },
                [](candidate const& x,candidate const& y){return x.id<y.id;},
                [](candidate const& p){return p.work+p.storage;});
        },
        [](auto const& r)->auto const& {return r.frontier;},
        [](candidate const& p){return p.id;},
        [](candidate const&)->so::evidence {return {};});
    check(model.native.frontier.size()==2,"CT-DP native Pareto frontier lost");
    check(model.candidates.size()==2 && model.selected()==nullptr,
          "model-only evidence promoted to qualified execution");
    check(model.native.dominated==1,"CT-DP solver diagnostics lost");

    // Existing space-descriptor search is also accepted without changing
    // search-space types. No hidden runtime type erasure or new solver.
    const auto scalar=so::run(so::request{same_scope(),1,4},
        [] {
            auto s=ctdp::space::descriptor_space(
                "matrix",ctdp::space::positive_int("tile",1,4));
            const auto best=ctdp::space::exhaustive_search_with_cost(
                s,[](auto const& p) {return double(std::get<0>(p)-3)*
                                                  double(std::get<0>(p)-3);});
            return native_report{{candidate{
                static_cast<std::uint64_t>(std::get<0>(best.best)),
                best.best_cost,0}},best.evaluated};
        },
        [](auto const& r)->auto const& {return r.frontier;},
        [](candidate const& p){return p.id;},
        [](candidate const&)->so::evidence{return {};});
    check(scalar.native.evaluated==4 && scalar.candidates[0].stable_id==3,
          "CT-DP native descriptor search not preserved");
    check(!scalar.selected(),"unverified descriptor winner must not be promoted");

    const so::request req{same_scope(),10,16};
    auto solve=[] {return native_report{{{40,1,1},{5,2,2},{80,3,3}}};};
    auto members=[](native_report const& r)->auto const&{return r.frontier;};
    auto identity=[](candidate const& p){return p.id;};
    // The fastest runtime candidate is not the cheapest over the full
    // invocation horizon if its preparation cost is high.
    auto qualified=so::run(req,solve,members,identity,[&](candidate const& c){
        if(c.id==40)return passed(req.required,2,100); // total 120
        if(c.id==5)return passed(req.required,4,10);   // total 50
        return passed(req.required,5,0);               // total 50, stable ID 5 wins
    });
    check(qualified.qualified_count==3 && qualified.selected() &&
          qualified.selected()->stable_id==5,
          "workload accounting or stable-ID tie break failed");
    check(*qualified.selected()->qualified_total_ns==50,
          "qualification total cost incorrect");

    // Measured but not independently qualified is never selected.
    auto measured=so::run(req,solve,members,identity,[](candidate const&) {
        so::evidence e;
        e.level=so::evidence_level::measured_unqualified;
        e.actual_execution=true;e.runtime_ns=1;
        return e;
    });
    check(!measured.selected() && measured.qualified_count==0,
          "measured-but-unqualified candidate was admitted");

    // A qualification from another input range, build or backend must fail.
    rejected([&]{so::run(req,solve,members,identity,[&](candidate const& c){
        auto e=passed(req.required,1);
        if(c.id==5)++e.observed.input_scope;
        return e;});});
    rejected([&]{so::run(req,solve,members,identity,[&](candidate const& c){
        auto e=passed(req.required,1);
        if(c.id==40)e.complete_scope=false;
        return e;});});
    rejected([&]{so::run(req,solve,members,identity,[&](candidate const& c){
        auto e=passed(req.required,1);
        if(c.id==40)e.numerical_pass=false;
        return e;});});
    rejected([&]{so::run(req,solve,members,identity,[&](candidate const& c){
        auto e=passed(req.required,1);
        if(c.id==40)e.runtime_ns=std::numeric_limits<double>::infinity();
        return e;});});
    rejected([&]{so::run(so::request{same_scope(),1,2},solve,members,identity,
        [](candidate const&)->so::evidence{return {};});});
    rejected([&]{so::run(req,
        []{return native_report{{{9,0,0},{9,1,1}}};},
        members,identity,[](candidate const&)->so::evidence{return {};});});
    rejected([&]{so::run(so::request{same_scope(),0,1},solve,members,identity,
        [](candidate const&)->so::evidence{return {};});});

    std::cout<<"CT-DP shared session PASS: native Pareto, descriptor search, "
             <<"qualified measured selection and fail-closed evidence\n";
}
