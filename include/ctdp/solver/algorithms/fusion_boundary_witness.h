#ifndef CTDP_SOLVER_ALGORITHMS_FUSION_BOUNDARY_WITNESS_H
#define CTDP_SOLVER_ALGORITHMS_FUSION_BOUNDARY_WITNESS_H

// Deliberately small, deterministic analytical counterexample generators.
// These are COST-MODEL counterexamples, not claims about native speedups or
// proof of equivalence for arbitrary user programs. Pure producer duplication
// is an explicit precondition of lifting two chain solutions to one fork.
#include "fusion_chain_probe.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ctdp::solver::fusion_probe {

// Smallest branching graph: one producer P with two consumers L and R.
// "duplicate": evaluate P in each independently fused P->L / P->R chain.
// "materialize": evaluate P once, keep its value and feed both consumers.
// Materialization cost is the complete extra storage/reads cost in this model;
// fusion_gain includes ALL producer-consumer fusion savings for that branch.
// No implicit free sharing or reduction of producer invocation count.
struct fork_cost {
    std::uint16_t producer{},left{},right{},materialization{};
    std::uint16_t gain_left{},gain_right{};
    bool producer_is_pure{true};
};
enum class fork_strategy : std::uint8_t {
    none,duplicate_producer,materialize_once
};
struct fork_solution {
    bool feasible{};
    cost_type total{unavailable};
    fork_strategy strategy{fork_strategy::none};
};
[[nodiscard]] constexpr bool valid_fork(fork_cost x) noexcept {
    return x.producer>0 && x.left>0 && x.right>0 &&
        x.gain_left<=static_cast<unsigned>(x.producer)+x.left &&
        x.gain_right<=static_cast<unsigned>(x.producer)+x.right;
}
[[nodiscard]] constexpr cost_type materialization_cost(fork_cost x) noexcept {
    if(!valid_fork(x))return unavailable;
    return cost_type{x.producer}+x.left+x.right+x.materialization;
}
[[nodiscard]] constexpr cost_type duplication_cost(fork_cost x) noexcept {
    if(!valid_fork(x) || !x.producer_is_pure)return unavailable;
    return (cost_type{x.producer}+x.left-x.gain_left)+
           (cost_type{x.producer}+x.right-x.gain_right);
}

// This is the problematic auxiliary-problem lift. It solves the two chains
// separately with the EXACT chain DP, then duplicates their producer when
// both solutions are realised in the original fork.
[[nodiscard]] constexpr fork_solution lift_independent_chains(fork_cost x) {
    if(!valid_fork(x) || !x.producer_is_pure)return {};
    chain_table<2> left{2},right{2};
    left.set(0,1,x.producer);
    left.set(1,2,x.left);
    left.set(0,2,cost_type{x.producer}+x.left-x.gain_left);
    right.set(0,1,x.producer);
    right.set(1,2,x.right);
    right.set(0,2,cost_type{x.producer}+x.right-x.gain_right);
    const auto l=solve_chain(left),r=solve_chain(right);
    if(!l.feasible || !r.feasible || l.cost>=unavailable-r.cost)return {};
    return {true,l.cost+r.cost,fork_strategy::duplicate_producer};
}

// Global model admits only the two explicitly named physical strategies.
// It does NOT silently assume that two chains can share their producer.
// When the producer is impure, duplication is illegal; materialization
// remains a legal single evaluation if its other environmental constraints
// are satisfied by the caller.
[[nodiscard]] constexpr fork_solution solve_fork_model(fork_cost x) noexcept {
    const auto shared=materialization_cost(x);
    const auto duplicated=duplication_cost(x);
    if(shared==unavailable && duplicated==unavailable)return {};
    if(duplicated<shared)
        return {true,duplicated,fork_strategy::duplicate_producer};
    return {true,shared,fork_strategy::materialize_once};
}

struct fork_counterexample {
    fork_cost problem{};
    cost_type independent_chains{};
    cost_type global_best{};
    cost_type regret{};
    unsigned model_weight{};
};
template<std::size_t MaxCount>
struct fork_corpus {
    std::array<fork_counterexample,MaxCount> cases{};
    std::size_t count{};
    std::size_t candidates_examined{};
};

// Within the explicit bounded grammar, enumerate by:
//   (number of vertices = 3), total positive parameter weight, then
//   lexicographic parameter tuple. Symmetric consumer permutations are
//   canonically identified. Gains and materialization must be positive:
//   the minimal witness cannot exploit "fusion offers no saving" or "free
//   materialization". The first returned instance is minimal UNDER THESE
//   LIMITS, not among arbitrary possible optimisation problems.
template<std::size_t MaxCount>
[[nodiscard]] constexpr fork_corpus<MaxCount> find_fork_counterexamples(
        unsigned max_weight=19) {
    fork_corpus<MaxCount> result{};
    if constexpr(MaxCount==0)return result;
    for(unsigned weight=6;weight<=max_weight;++weight)
        for(unsigned producer=1;producer<=8;++producer)
            for(unsigned left=1;left<=4;++left)
                for(unsigned right=1;right<=4;++right)
                    for(unsigned memory=1;memory<=3;++memory)
                        for(unsigned gl=1;gl<=3;++gl)
                            for(unsigned gr=1;gr<=3;++gr) {
                                if(left>right || (left==right && gl>gr) ||
                                   producer+left+right+memory+gl+gr!=weight)
                                    continue;
                                const fork_cost x{
                                    static_cast<std::uint16_t>(producer),
                                    static_cast<std::uint16_t>(left),
                                    static_cast<std::uint16_t>(right),
                                    static_cast<std::uint16_t>(memory),
                                    static_cast<std::uint16_t>(gl),
                                    static_cast<std::uint16_t>(gr),true};
                                if(!valid_fork(x))continue;
                                ++result.candidates_examined;
                                const auto lifted=lift_independent_chains(x);
                                const auto global=solve_fork_model(x);
                                if(!lifted.feasible || !global.feasible ||
                                   global.strategy!=fork_strategy::materialize_once ||
                                   lifted.total<=global.total)continue;
                                result.cases[result.count++]={
                                    x,lifted.total,global.total,
                                    lifted.total-global.total,weight};
                                if(result.count==MaxCount)return result;
                            }
    return result;
}

struct layout_counterexample {
    std::array<std::uint16_t,2> producer{};
    std::array<std::uint16_t,2> consumer{};
    cost_type greedy_prefix{},stateful_best{},regret{};
    unsigned model_weight{};
};
[[nodiscard]] constexpr layout_counterexample find_layout_counterexample(
        unsigned max_weight=32) {
    // The prefix-greedy solver discards the more expensive layout 1 output
    // of the first operation. No conversion between layouts is available.
    // A state-aware DP may retain that layout for a cheaper consumer.
    for(unsigned weight=4;weight<=max_weight;++weight)
        for(unsigned pa=1;pa<=8;++pa)
            for(unsigned pb=1;pb<=8;++pb)
                for(unsigned ca=1;ca<=8;++ca)
                    for(unsigned cb=1;cb<=8;++cb) {
                        if(pa+pb+ca+cb!=weight || pa>=pb)continue;
                        layout_chain<2> model{2};
                        model.set(0,0,pa);
                        model.set(0,1,pb);
                        model.set(1,0,ca);
                        model.set(1,1,cb);
                        const auto best=solve_layout(model);
                        const cost_type greedy=pa+ca;
                        if(best.feasible && best.layouts==3u &&
                           best.cost<greedy)
                            return {{static_cast<std::uint16_t>(pa),
                                     static_cast<std::uint16_t>(pb)},
                                    {static_cast<std::uint16_t>(ca),
                                     static_cast<std::uint16_t>(cb)},
                                    greedy,best.cost,greedy-best.cost,weight};
                    }
    return {};
}

} // namespace ctdp::solver::fusion_probe
#endif // CTDP_SOLVER_ALGORITHMS_FUSION_BOUNDARY_WITNESS_H
