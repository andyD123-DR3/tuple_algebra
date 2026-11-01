#pragma once
// Opt-in domain-neutral recursive physical-planning primitives.
//
// This header does NOT lower an algorithm, prove a transformation, run an
// executor, or grant numerical/device admission. Clients supply domain facts,
// structural split choices, full-state boundary witnesses, actual physical
// lowerings, and complete execution/observation evidence. The last step uses
// DN's existing evidence gate (contract.hpp / ctdp_selector.hpp).
//
// Shared by deterministic sparse SpMV and stateful program replay, with no
// default production dispatch changes.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ctdp::solver::physical {

struct interval {
    std::size_t first{},last{}; // Half open, source-coordinate order.
    [[nodiscard]] constexpr bool valid() const noexcept {
        return first<last;
    }
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return last-first; // Call only after valid() / range admission.
    }
};
struct bounds {
    std::size_t minimum_leaf_size{1};
    std::size_t maximum_depth{6};
    std::size_t maximum_leaves{128};
};
inline void validate(bounds const& cfg,interval root,std::size_t depth=0) {
    if(!root.valid() || cfg.minimum_leaf_size==0 ||
       cfg.maximum_depth>32 || depth>cfg.maximum_depth ||
       cfg.maximum_leaves==0 || cfg.maximum_leaves>4096 ||
       root.size()<cfg.minimum_leaf_size)
        throw std::invalid_argument("DN recursive plan: invalid region/bounds");
}

// Append an entire bounded recursive subtree WITHOUT publishing it. The
// splitter returns an optional absolute cut coordinate or nullopt to emit
// a leaf. The lowerer returns the domain's physical leaf payload and must
// preserve first/last. Only the client's full-composite actual-execution
// verification can grant numerical/semantic admissibility.
// Leaf: a type with publicly accessible first,last. Splitter: (interval,
// depth) -> optional<size_t>. Lowerer: (interval,depth) -> Leaf.
//
// Invalid cuts, underfilled children, excessive depth/leaf counts and
// non-contiguous lowerer results are errors rather than silently truncated
// search. The caller keeps the old executable until qualification completes.
template<class Leaf,class Splitter,class Lowerer>
void append_subtree(std::vector<Leaf>& output, interval root,
                    std::size_t depth,bounds const& cfg,
                    Splitter&& split,Lowerer&& lower) {
    validate(cfg,root,depth);
    auto visit=[&](auto&& self,interval current,std::size_t level)->void {
        if(output.size()>=cfg.maximum_leaves)
            throw std::length_error("DN recursive plan: leaf cap exceeded");
        auto emit=[&] {
            Leaf made=lower(current,level);
            if(made.first!=current.first || made.last!=current.last)
                throw std::logic_error(
                    "DN recursive plan: physical lowerer changed source range");
            output.push_back(std::move(made));
        };
        if(level>=cfg.maximum_depth ||
           current.size()/2<cfg.minimum_leaf_size) {
            emit();return;
        }
        const std::optional<std::size_t> pivot=split(current,level);
        if(!pivot){emit();return;}
        if(*pivot<=current.first || *pivot>=current.last ||
           *pivot-current.first<cfg.minimum_leaf_size ||
           current.last-*pivot<cfg.minimum_leaf_size)
            throw std::invalid_argument(
                "DN recursive plan: illegal or underfilled split");
        self(self,interval{current.first,*pivot},level+1);
        self(self,interval{*pivot,current.last},level+1);
    };
    visit(visit,root,depth);
}

// Validate a complete executable *source-ordered* region partition. This
// does not prove that an individual leaf preserves observations.
template<class Region>
void require_partition(std::span<const Region> regions,interval root) {
    if(!root.valid() || regions.empty())
        throw std::invalid_argument("DN recursive plan: empty partition");
    std::size_t cursor=root.first;
    for(auto const& r:regions) {
        if(r.first!=cursor || r.last<=r.first || r.last>root.last)
            throw std::invalid_argument(
                "DN recursive plan: gap, overlap, or bad region");
        cursor=r.last;
    }
    if(cursor!=root.last)
        throw std::invalid_argument("DN recursive plan: incomplete partition");
}

// Fusion is NOT simply "adjacent rows have the same type". A stateful
// implementation may need to materialise and restore live state, and a
// numerical implementation may require a different reduction tree. The
// domain witness must certify that removal of THIS boundary is legal, then
// the domain-specific join callback must actually reconstruct the payload.
// Full output verification still follows any fusion.
struct fusion_witness {
    bool same_physical_representation{};
    bool boundary_state_compatible{};
    bool preserves_observations{};
    [[nodiscard]] constexpr bool admitted() const noexcept {
        return same_physical_representation &&
               boundary_state_compatible && preserves_observations;
    }
};

template<class Region,class Witness,class Join>
[[nodiscard]] std::vector<Region> fuse_adjacent(
    std::span<const Region> source,interval root,
    Witness&& witness,Join&& join) {
    require_partition(source,root);
    std::vector<Region> result;
    result.reserve(source.size());
    for(auto const& next:source) {
        if(!result.empty() && witness(result.back(),next).admitted()) {
            const auto first=result.back().first;
            Region merged=join(result.back(),next);
            if(merged.first!=first || merged.last!=next.last)
                throw std::logic_error(
                    "DN recursive plan: invalid physical fusion result");
            result.back()=std::move(merged);
        } else result.push_back(next);
    }
    require_partition(std::span<const Region>(result),root);
    return result;
}

// Search economics is a domain-independent *heuristic* for whether to
// attempt more work. Structural estimates and this gate are never proof of
// numerical equivalence or hardware capability.
struct probe_context {
    std::size_t remaining_calls{},attempted{};
    double incumbent_ns{},initial_setup_ns{},spent_ns{};
    double exploration_fraction{0.25},total_budget_ns{1.e9};
};
[[nodiscard]] inline bool affordable(probe_context const& x) noexcept {
    if(x.remaining_calls==0 || !std::isfinite(x.incumbent_ns) ||
       x.incumbent_ns<=0 || !std::isfinite(x.initial_setup_ns) ||
       x.initial_setup_ns<0 || !std::isfinite(x.spent_ns) || x.spent_ns<0 ||
       !std::isfinite(x.exploration_fraction) ||
       x.exploration_fraction<=0 || x.exploration_fraction>1 ||
       !std::isfinite(x.total_budget_ns) || x.total_budget_ns<=0 ||
       x.spent_ns>=x.total_budget_ns)
        return false;
    const double estimate=x.attempted?
        x.spent_ns/static_cast<double>(x.attempted):x.initial_setup_ns;
    const double remaining_work=x.incumbent_ns*
        static_cast<double>(x.remaining_calls);
    const double required=x.spent_ns+estimate;
    const double allowance=x.exploration_fraction*remaining_work;
    return std::isfinite(estimate) && std::isfinite(required) &&
           std::isfinite(allowance) && required<allowance;
}

// Apply ONLY AFTER the domain has actually lowered and qualified the
// complete candidate with the DN evidence boundary. This decision concerns
// amortisation, not numerical semantics. The action includes all unsuccessful
// candidate search, preparation and calibration performed for this proposal.
struct adoption_context {
    std::size_t remaining_calls{};
    double incumbent_median_ns{},incumbent_mad_ns{};
    double trial_median_ns{},trial_mad_ns{};
    double mad_penalty{},action_wall_ns{};
    double minimum_step_gain_fraction{0.01},gain_margin{0.10};
};
struct adoption_result {
    bool adopt{};
    double future_savings_ns{};
};
[[nodiscard]] inline adoption_result should_adopt(
    adoption_context const& x) noexcept {
    if(x.remaining_calls==0 || !std::isfinite(x.incumbent_median_ns) ||
       x.incumbent_median_ns<=0 || !std::isfinite(x.trial_median_ns) ||
       x.trial_median_ns<=0 || !std::isfinite(x.incumbent_mad_ns) ||
       x.incumbent_mad_ns<0 || !std::isfinite(x.trial_mad_ns) ||
       x.trial_mad_ns<0 || !std::isfinite(x.mad_penalty) ||
       x.mad_penalty<0 || !std::isfinite(x.action_wall_ns) ||
       x.action_wall_ns<0 || !std::isfinite(x.gain_margin) ||
       x.gain_margin<0 || !std::isfinite(x.minimum_step_gain_fraction) ||
       x.minimum_step_gain_fraction<0 || x.minimum_step_gain_fraction>=1)
        return {};
    const auto before=x.incumbent_median_ns+
        x.mad_penalty*x.incumbent_mad_ns;
    const auto after=x.trial_median_ns+x.mad_penalty*x.trial_mad_ns;
    const auto saved=before-after;
    const auto future=saved*static_cast<double>(x.remaining_calls);
    const auto threshold=x.action_wall_ns*(1+x.gain_margin);
    if(!std::isfinite(before) || !std::isfinite(after) ||
       !std::isfinite(saved) || !std::isfinite(future) ||
       !std::isfinite(threshold))
        return {};
    return {saved>before*x.minimum_step_gain_fraction &&
                future>threshold,future};
}

} // namespace ctdp::solver::physical
