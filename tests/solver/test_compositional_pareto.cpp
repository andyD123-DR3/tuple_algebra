// Independent exhaustive oracle and adversarial admission tests for generic
// compositional Pareto DP. All assertions are live in Release builds.
#include <ctdp/solver/algorithms/compositional_pareto.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
namespace p=ctdp::solver::physical;

struct choice {
    std::vector<std::size_t> factors{};
    ctdp::cost_vector<4> objectives{};
};
static void require(bool b,char const* reason) {
    if(!b)throw std::runtime_error(reason);
}
template<class F>
static void rejects(F&& f,char const* reason) {
    bool caught=false;
    try{f();}catch(std::exception const&){caught=true;}
    require(caught,reason);
}
static auto costs=[](choice const& x){return x.objectives;};
static auto order=[](choice const& a,choice const& b) {
    return a.factors<b.factors;
};
static auto priority=[](choice const& x) {
    return x.objectives[0]+20*x.objectives[1]+
        x.objectives[2]/16+x.objectives[3]/64;
};
static auto expansion=[](std::size_t remaining,auto const& child,
                          auto const& emit) {
    if(remaining==1) {
        emit(choice{});
        return;
    }
    for(std::size_t radix:{2u,4u,7u}) {
        if(remaining%radix)continue;
        const auto& subplans=child(remaining/radix);
        const auto m=remaining/radix;
        const double work=radix==2?4.0:radix==4?11.0:105.0;
        const double span=radix==2?4.0:radix==4?10.0:35.0;
        for(auto const& c:subplans) {
            choice next;
            next.factors.push_back(radix);
            next.factors.insert(next.factors.end(),c.factors.begin(),c.factors.end());
            next.objectives={{
                radix*c.objectives[0]+work*m,
                c.objectives[1]+span,
                c.objectives[2]+(remaining+radix*radix)*4.0,
                radix*c.objectives[3]+remaining*4.0
            }};
            emit(std::move(next));
        }
    }
};
static choice independent_model(std::vector<std::size_t> factors) {
    choice result{factors,{}};
    std::size_t current=1;
    for(std::size_t i=factors.size();i>0;--i) {
        const auto radix=factors[i-1];
        const auto m=current;
        const auto parent=m*radix;
        const double work=radix==2?4.0:radix==4?11.0:105.0;
        const double span=radix==2?4.0:radix==4?10.0:35.0;
        result.objectives={{
            radix*result.objectives[0]+work*m,
            result.objectives[1]+span,
            result.objectives[2]+(parent+radix*radix)*4.0,
            radix*result.objectives[3]+parent*4.0
        }};
        current=parent;
    }
    return result;
}
static std::vector<choice> exhaustive(std::size_t n) {
    std::vector<choice> out;
    std::vector<std::size_t> sequence;
    auto visit=[&](auto&& self,std::size_t remaining)->void {
        if(remaining==1){out.push_back(independent_model(sequence));return;}
        for(std::size_t r:{2u,4u,7u}) {
            if(remaining%r)continue;
            sequence.push_back(r);self(self,remaining/r);sequence.pop_back();
        }
    };
    visit(visit,n);
    return out;
}
static void exact_oracle() {
    for(auto n:std::array<std::size_t,7>{14,28,56,112,224,256,512}) {
        const auto result=p::compositional_pareto<std::size_t,choice,4>(
            n,{128,10000,100},expansion,costs,order,priority);
        require(!result.approximate && !result.frontier.empty(),
                "CT-DP physical DP: exact model run was approximated");
        const auto all=exhaustive(n);
        std::set<std::vector<std::size_t>> valid;
        for(auto const& x:all)valid.insert(x.factors);
        for(auto const& item:result.frontier) {
            require(valid.count(item.factors)!=0,
                    "CT-DP physical DP: invented factorisation");
            require(item.objectives==independent_model(item.factors).objectives,
                    "CT-DP physical DP: incorrect monotone recurrence");
            for(auto const& other:all)
                require(!other.objectives.dominates(item.objectives),
                        "CT-DP physical DP: globally dominated state admitted");
        }
        // Each globally non-dominated model objective should have exactly
        // one canonical stable-factorisation representative.
        for(auto const& candidate:all) {
            const bool dominated=std::any_of(all.begin(),all.end(),
                [&](choice const& other) {
                    return other.objectives.dominates(candidate.objectives);
                });
            if(dominated)continue;
            const auto best=std::min_element(all.begin(),all.end(),
                [&](choice const& a,choice const& b) {
                    if(a.objectives==candidate.objectives &&
                       b.objectives!=candidate.objectives)return true;
                    if(b.objectives==candidate.objectives &&
                       a.objectives!=candidate.objectives)return false;
                    return a.factors<b.factors;
                });
            if(best->objectives==candidate.objectives &&
               best->factors!=candidate.factors)continue;
            require(std::any_of(result.frontier.begin(),result.frontier.end(),
                [&](choice const& item) {
                    return item.objectives==candidate.objectives &&
                        item.factors==candidate.factors;
                }),"CT-DP physical DP: lost a nondominated objective");
        }
    }
}
static void bounded_and_adversarial() {
    auto forced=[](int,auto const&,auto const& emit) {
        emit(choice{{1},{{1,4,1,4}}});
        emit(choice{{2},{{4,1,4,1}}});
        emit(choice{{3},{{2,3,2,3}}});
        emit(choice{{4},{{3,2,3,2}}});
    };
    auto r=p::compositional_pareto<int,choice,4>(
        99,{2,100,10},forced,costs,order,priority);
    require(r.approximate && r.budget_pruned==2 &&
            r.frontier.size()==2 &&
            r.frontier[0].factors==std::vector<std::size_t>{1} &&
            r.frontier[1].factors==std::vector<std::size_t>{2},
            "CT-DP physical DP: explicit objective extrema were not kept");
    const auto repeat=p::compositional_pareto<int,choice,4>(
        99,{2,100,10},forced,costs,order,priority);
    require(repeat.frontier[0].factors==r.frontier[0].factors &&
            repeat.frontier[1].factors==r.frontier[1].factors,
            "CT-DP physical DP: unstable beam pruning");
    rejects([&] {
        (void)p::compositional_pareto<int,choice,4>(
            99,{1,100,10},forced,costs,order,priority);
    },"CT-DP physical DP: dropped objective extrema silently");
    rejects([&] {
        (void)p::compositional_pareto<int,choice,4>(
            99,{4,2,10},forced,costs,order,priority);
    },"CT-DP physical DP: truncated generation silently");
    rejects([&] {
        auto bad=[](int key,auto const& child,auto const& emit) {
            if(key==0){emit(choice{});return;}
            (void)child(key);
        };
        (void)p::compositional_pareto<int,choice,4>(
            99,{8,100,10},bad,costs,order,priority);
    },"CT-DP physical DP: cyclic graph admitted");
    rejects([&] {
        auto invalid=[](int,auto const&,auto const& emit) {
            emit(choice{{1},{{1,std::numeric_limits<double>::quiet_NaN(),1,1}}});
        };
        (void)p::compositional_pareto<int,choice,4>(
            99,{8,100,10},invalid,costs,order,priority);
    },"CT-DP physical DP: NaN cost admitted");
    rejects([&] {
        auto invalid=[](int,auto const&,auto const& emit) {
            emit(choice{{1},{{1,-2,1,1}}});
        };
        (void)p::compositional_pareto<int,choice,4>(
            99,{8,100,10},invalid,costs,order,priority);
    },"CT-DP physical DP: negative resource cost admitted");
}
int main() {
    exact_oracle();
    bounded_and_adversarial();
    std::puts("CT-DP compositional physical Pareto: PASS");
}
