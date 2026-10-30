#ifndef CTDP_SOLVER_ALGORITHMS_FUSION_CHAIN_PROBE_H
#define CTDP_SOLVER_ALGORITHMS_FUSION_CHAIN_PROBE_H

// Research-only exact reference model for contiguous fusion partitions.
// Assumptions: a pure linear single-consumer chain; each fused interval has
// one independently known finite nonnegative total cost; segment costs add;
// there are NO inter-segment layouts, global register limits, aliases, live
// shared values or unmodelled observations. The model is deliberately NOT a
// calibrated hardware-performance predictor or a numerical proof.
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ctdp::solver::fusion_probe {

using cost_type = std::uint64_t;
inline constexpr cost_type unavailable = std::numeric_limits<cost_type>::max();

template<std::size_t MaxN = 12>
struct chain_table {
    static_assert(MaxN>0 && MaxN<=16, "bounded cut-mask oracle supports at most 16 nodes");
    std::size_t nodes{};
    std::array<std::array<cost_type,MaxN+1>,MaxN+1> intervals{};

    constexpr explicit chain_table(std::size_t count=0):nodes(count) {
        if(count>MaxN)throw std::invalid_argument("fusion chain: capacity exceeded");
        for(auto& row:intervals)row.fill(unavailable);
    }
    constexpr void set(std::size_t begin,std::size_t end,cost_type total) {
        if(begin>=end || end>nodes || total==unavailable)
            throw std::invalid_argument("fusion chain: invalid interval or cost");
        intervals[begin][end]=total;
    }
    [[nodiscard]] constexpr cost_type get(
        std::size_t begin,std::size_t end) const noexcept {
        return begin<end && end<=nodes ? intervals[begin][end] : unavailable;
    }
};

struct chain_solution {
    bool feasible{};
    cost_type cost{unavailable};
    // Bit k means a cut AFTER operation k, so a set bit separates segments.
    std::uint64_t cuts{};
    std::size_t segments{};
    std::size_t transitions_evaluated{};
};

// O(n^2) shortest-prefix recurrence: dp[j] = min_i dp[i] + C(i,j).
// The cut mask is a canonical deterministic tie-breaker. Overflowed sums,
// absent intervals and unavailable prefixes are not admissible.
template<std::size_t MaxN>
[[nodiscard]] constexpr chain_solution solve_chain(
        chain_table<MaxN> const& instance) {
    if(instance.nodes>MaxN)
        throw std::invalid_argument("fusion chain: invalid length");
    std::array<chain_solution,MaxN+1> dp{};
    dp[0]={true,0,0,0,0};
    std::size_t transitions=0;
    for(std::size_t j=1;j<=instance.nodes;++j) {
        for(std::size_t i=0;i<j;++i) {
            const auto segment=instance.get(i,j);
            if(!dp[i].feasible || segment==unavailable)continue;
            if(dp[i].cost>=unavailable-segment)continue;
            ++transitions;
            const auto cost=dp[i].cost+segment;
            const auto mask=dp[i].cuts | (i ? (std::uint64_t{1}<<(i-1)) : 0);
            if(!dp[j].feasible || cost<dp[j].cost ||
               (cost==dp[j].cost && mask<dp[j].cuts)) {
                dp[j]={true,cost,mask,dp[i].segments+1,0};
            }
        }
    }
    auto out=dp[instance.nodes];
    out.transitions_evaluated=transitions;
    return out;
}

// State-aware reference case. A dependency edge transports a two-valued
// boundary layout. A transition may be forbidden (unavailable) or have a
// specified conversion cost. The naive prefix-minimum without layout state
// is NOT an exact solver for this class.
template<std::size_t MaxN = 12>
struct layout_chain {
    static_assert(MaxN>0 && MaxN<=16);
    std::size_t nodes{};
    std::array<std::array<cost_type,2>,MaxN> execution{};
    std::array<std::array<cost_type,2>,2> conversion{{
        {{0,unavailable}},{{unavailable,0}}
    }};

    constexpr explicit layout_chain(std::size_t count=0):nodes(count) {
        if(count>MaxN)throw std::invalid_argument("fusion layout: capacity exceeded");
        for(auto& row:execution)row.fill(unavailable);
    }
    constexpr void set(std::size_t node,std::size_t layout,cost_type total) {
        if(node>=nodes || layout>=2 || total==unavailable)
            throw std::invalid_argument("fusion layout: invalid operation cost");
        execution[node][layout]=total;
    }
    constexpr void set_conversion(std::size_t before,std::size_t after,cost_type cost) {
        if(before>=2 || after>=2)
            throw std::invalid_argument("fusion layout: invalid conversion");
        conversion[before][after]=cost;
    }
};

struct layout_solution {
    bool feasible{};
    cost_type cost{unavailable};
    // Bit i indicates that operation i uses layout 1 instead of layout 0.
    std::uint64_t layouts{};
};

// A genuine two-state DP: different boundary layouts are NOT dominated by
// cost alone, because their future conversion/consumer costs differ.
template<std::size_t MaxN>
[[nodiscard]] constexpr layout_solution solve_layout(
        layout_chain<MaxN> const& instance) {
    if(instance.nodes>MaxN)
        throw std::invalid_argument("fusion layout: invalid length");
    if(instance.nodes==0)return {true,0,0};
    std::array<layout_solution,2> prev{};
    for(std::size_t state=0;state<2;++state) {
        const auto value=instance.execution[0][state];
        if(value!=unavailable)
            prev[state]={true,value,static_cast<std::uint64_t>(state)};
    }
    for(std::size_t node=1;node<instance.nodes;++node) {
        std::array<layout_solution,2> next{};
        for(std::size_t dest=0;dest<2;++dest) {
            const auto cost=instance.execution[node][dest];
            if(cost==unavailable)continue;
            for(std::size_t source=0;source<2;++source) {
                const auto transfer=instance.conversion[source][dest];
                if(!prev[source].feasible || transfer==unavailable)continue;
                if(prev[source].cost>=unavailable-transfer)continue;
                const auto partial=prev[source].cost+transfer;
                if(partial>=unavailable-cost)continue;
                const auto total=partial+cost;
                const auto bits=prev[source].layouts |
                    (static_cast<std::uint64_t>(dest)<<node);
                if(!next[dest].feasible || total<next[dest].cost ||
                   (total==next[dest].cost && bits<next[dest].layouts))
                    next[dest]={true,total,bits};
            }
        }
        prev=next;
    }
    const auto a=prev[0],b=prev[1];
    if(!a.feasible)return b;
    if(!b.feasible)return a;
    return b.cost<a.cost || (b.cost==a.cost && b.layouts<a.layouts) ? b:a;
}

} // namespace ctdp::solver::fusion_probe
#endif // CTDP_SOLVER_ALGORITHMS_FUSION_CHAIN_PROBE_H
