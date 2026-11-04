#pragma once
// Domain-neutral, model-only compositional Pareto DP over a bounded DAG.
//
// Reuses CT-DP's established cost_vector<D> and its exact dominance
// semantics. The domain supplies admissible transitions, physical-interface
// preconditions, objective recurrence, and a stable candidate ordering.
// No model score is numerical or hardware admission. Only an external
// actually-executed and fully verified lowering can grant that status.
//
// Callback interfaces:
//   expand(key, child(key)->const vector<Label>&, emit(Label))
//   cost(Label)->ctdp::cost_vector<D>
//   stable_less(a,b)->bool (strict stable physical ID/factor sequence)
//   priority(Label)->double (reproducible, finite model-only beam ordering)
#include <ctdp/core/cost_vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ctdp::solver::physical {

struct pareto_limits {
    std::size_t maximum_frontier{8};
    std::size_t maximum_generated{10000};
    std::size_t maximum_subproblems{4096};
};
template<class Label>
struct pareto_report {
    std::vector<Label> frontier{};
    std::size_t subproblems{},generated{},dominated{},budget_pruned{};
    bool approximate{};
};

// A bounded memoised AND-OR recurrence. Key must form a deterministic
// std::less ordering; the caller's expansion must be finite, and cycles
// are rejected as an invalid physical decomposition graph.
// The parent requires compatible interfaces BEFORE calling emit.
//
// Exact only when no frontier was truncated AND the domain has a
// monotone/compositional cost recurrence that justifies dominance pruning.
// Even an exact model frontier is NOT measured-performance optimality.
template<class Key,class Label,std::size_t D,
         class Expand,class Costs,class StableLess,class Priority>
[[nodiscard]] pareto_report<Label> compositional_pareto(
    Key root,pareto_limits limits,Expand&& expand,Costs&& costs,
    StableLess&& stable_less,Priority&& priority) {
    static_assert(D>0);
    if(limits.maximum_frontier==0 || limits.maximum_frontier>4096 ||
       limits.maximum_generated==0 || limits.maximum_generated>1000000 ||
       limits.maximum_subproblems==0 || limits.maximum_subproblems>100000)
        throw std::invalid_argument("CT-DP physical Pareto: invalid bounds");

    using points=std::vector<Label>;
    std::map<Key,points> memo;
    std::set<Key> active;
    pareto_report<Label> report{};

    auto solve=[&](auto&& self,Key key)->points const& {
        if(auto it=memo.find(key);it!=memo.end())return it->second;
        if(!active.insert(key).second)
            throw std::invalid_argument("CT-DP physical Pareto: cyclic plan DAG");
        if(memo.size()+active.size()>limits.maximum_subproblems)
            throw std::length_error("CT-DP physical Pareto: subproblem cap");

        points possible;
        auto child=[&](Key next)->points const& {return self(self,next);};
        auto emit=[&](Label next) {
            if(report.generated==limits.maximum_generated)
                throw std::length_error("CT-DP physical Pareto: generation cap");
            const auto score=costs(next);
            if(!score.is_finite())
                throw std::invalid_argument("CT-DP physical Pareto: nonfinite model");
            for(std::size_t i=0;i<D;++i)
                if(score[i]<0)
                    throw std::invalid_argument("CT-DP physical Pareto: negative cost");
            const auto value=priority(next);
            if(!std::isfinite(value))
                throw std::invalid_argument("CT-DP physical Pareto: invalid priority");
            ++report.generated;
            possible.push_back(std::move(next));
        };
        expand(key,child,emit);
        if(possible.empty())
            throw std::invalid_argument("CT-DP physical Pareto: empty subproblem");

        std::stable_sort(possible.begin(),possible.end(),
            [&](Label const& lhs,Label const& rhs) {
                const auto a=priority(lhs),b=priority(rhs);
                return a<b || (a==b && stable_less(lhs,rhs));
            });
        // All dominance and stable-order callbacks must observe intact
        // labels. Moving a retained vector-owning label here can change its
        // cost/order and incorrectly dominate later, incomparable candidates.
        std::vector<std::size_t> retained_indices;
        for(std::size_t i=0;i<possible.size();++i) {
            const auto ci=costs(possible[i]);
            bool inferior=false;
            for(std::size_t j=0;j<possible.size();++j) {
                if(i==j)continue;
                const auto cj=costs(possible[j]);
                if(cj.dominates(ci) ||
                   (ci==cj && (stable_less(possible[j],possible[i]) ||
                               (!stable_less(possible[i],possible[j]) && j<i)))) {
                    inferior=true;break;
                }
            }
            if(inferior)++report.dominated;
            else retained_indices.push_back(i);
        }
        points frontier;
        frontier.reserve(retained_indices.size());
        for(auto i:retained_indices)frontier.push_back(std::move(possible[i]));

        if(frontier.size()>limits.maximum_frontier) {
            std::vector<bool> retained(frontier.size());
            for(std::size_t dim=0;dim<D;++dim) {
                auto best=std::min_element(frontier.begin(),frontier.end(),
                    [&](Label const& a,Label const& b) {
                        const auto x=costs(a)[dim],y=costs(b)[dim];
                        return x<y || (x==y && stable_less(a,b));
                    });
                retained[static_cast<std::size_t>(best-frontier.begin())]=true;
            }
            auto count=static_cast<std::size_t>(
                std::count(retained.begin(),retained.end(),true));
            if(count>limits.maximum_frontier)
                throw std::invalid_argument(
                    "CT-DP physical Pareto: budget below objective extrema");
            for(std::size_t i=0;i<frontier.size() &&
                                   count<limits.maximum_frontier;++i)
                if(!retained[i]){retained[i]=true;++count;}
            points bounded;
            bounded.reserve(count);
            for(std::size_t i=0;i<frontier.size();++i)
                if(retained[i])bounded.push_back(std::move(frontier[i]));
            report.budget_pruned+=frontier.size()-bounded.size();
            report.approximate=true;
            frontier=std::move(bounded);
        }
        active.erase(key);
        auto [inserted,ok]=memo.emplace(std::move(key),std::move(frontier));
        (void)ok;
        ++report.subproblems;
        return inserted->second;
    };
    report.frontier=solve(solve,std::move(root));
    return report;
}

} // namespace ctdp::solver::physical
