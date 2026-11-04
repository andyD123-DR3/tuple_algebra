#pragma once
// Common CT-DP optimisation-session boundary (C++20, header-only).
//
// The domain owns its native space, solver, plan type, physical lowerings,
// numerical oracle and semantics. This protocol only preserves the native
// solver report, records candidate evidence and selects *actually qualified*
// physical alternatives. Model-only Pareto frontiers are never promoted.
//
// The four callbacks are:
//   solve()                 -> native solver report (any type)
//   proposals(native)       -> range of domain-native plan objects
//   stable_id(plan)         -> persistent uint64_t physical/plan identifier
//   assess(plan)            -> evidence (domain evaluates/qualifies)
// No forced Cartesian enumeration, type erasure, or scalarisation of Pareto
// objectives is introduced; constexpr solvers remain independently usable.
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ctdp::solver::session {

enum class evidence_level : std::uint8_t {
    model_only,
    measured_unqualified,
    numerically_qualified
};

// All values are opaque domain-owned hashes/IDs. Identity includes input
// scope: a candidate verified for a smoke domain cannot qualify a larger one.
struct identity {
    std::uint64_t expression{};
    std::uint64_t observation{};
    std::uint64_t environment{};
    std::uint64_t input_scope{};
    std::uint64_t target{};
    std::uint64_t compiler_build{};
    friend constexpr bool operator==(identity const&,identity const&) = default;
};

struct request {
    identity required{};
    std::size_t invocations{1};
    std::size_t maximum_candidates{10000};
};

struct evidence {
    evidence_level level{evidence_level::model_only};
    identity observed{};
    bool actual_execution{};
    bool complete_scope{};
    bool independent_reference{};
    bool numerical_pass{};
    std::size_t observed_inputs{};
    std::size_t observed_bytes{};
    // Source-controlled hashes, not a proof in themselves.
    std::uint64_t reference_witness{},candidate_witness{};
    // Costs are from a real execution when level != model_only.
    double runtime_ns{},preparation_ns{},calibration_ns{};
};

inline void validate_request(request const& r) {
    if(!r.invocations || !r.maximum_candidates ||
       r.maximum_candidates>1000000)
        throw std::invalid_argument("CT-DP session: invalid bounds/workload");
}

// A qualified result is a declaration by the *domain's* independent verifier,
// not a proof supplied by CT-DP. Strict identity matching prevents reusing
// measurements from another input scope, build, device or FP environment.
[[nodiscard]] inline bool admitted(evidence const& e,request const& r) {
    if(e.level!=evidence_level::numerically_qualified)return false;
    if(e.observed!=r.required || !e.actual_execution || !e.complete_scope ||
       !e.independent_reference || !e.numerical_pass ||
       !e.observed_inputs || !e.observed_bytes)return false;
    return std::isfinite(e.runtime_ns) && e.runtime_ns>0 &&
           std::isfinite(e.preparation_ns) && e.preparation_ns>=0 &&
           std::isfinite(e.calibration_ns) && e.calibration_ns>=0;
}

[[nodiscard]] inline double total_cost_ns(evidence const& e,request const& r) {
    const double total=e.preparation_ns+e.calibration_ns+
        static_cast<double>(r.invocations)*e.runtime_ns;
    if(!std::isfinite(total) || total<=0)
        throw std::invalid_argument("CT-DP session: invalid total workload cost");
    return total;
}

template<class Plan>
struct candidate_record {
    Plan plan;
    std::uint64_t stable_id{};
    evidence observed{};
    std::optional<double> qualified_total_ns{};
};

template<class NativeReport,class Plan>
struct result {
    NativeReport native;    // full original statistics and Pareto frontier
    std::vector<candidate_record<Plan>> candidates;
    std::optional<std::size_t> selected_index; // qualified physical only
    std::size_t qualified_count{};
    [[nodiscard]] candidate_record<Plan> const* selected() const noexcept {
        return selected_index ? &candidates[*selected_index] : nullptr;
    }
};

// General orchestration: the solver, candidate range, evidence provider and
// identity mapping remain domain-native. Only the lifecycle is shared.
// Invalid claims of qualification fail closed instead of being downgraded.
template<class Solve,class Proposals,class StableId,class Assess>
[[nodiscard]] auto run(request const& req,Solve&& solve,
                       Proposals&& proposals,StableId&& stable_id,
                       Assess&& assess) {
    validate_request(req);
    auto native=std::invoke(std::forward<Solve>(solve));
    auto&& found=std::invoke(std::forward<Proposals>(proposals),
                            std::as_const(native));
    using plan_type=std::ranges::range_value_t<decltype(found)>;
    static_assert(std::copy_constructible<plan_type>,
                  "CT-DP session: domain plans must be copyable");
    std::vector<candidate_record<plan_type>> records;
    std::optional<std::size_t> selected{};
    std::size_t qualified_count=0;
    double best=std::numeric_limits<double>::infinity();
    std::uint64_t best_id=0;
    for(auto const& plan:found) {
        if(records.size()==req.maximum_candidates)
            throw std::length_error("CT-DP session: candidate budget exhausted");
        const auto id=static_cast<std::uint64_t>(std::invoke(stable_id,plan));
        for(auto const& prev:records)
            if(prev.stable_id==id)
                throw std::invalid_argument("CT-DP session: duplicate stable ID");
        auto e=std::invoke(assess,plan);
        if(e.level==evidence_level::numerically_qualified && !admitted(e,req))
            throw std::invalid_argument("CT-DP session: invalid qualification");
        if(e.level==evidence_level::measured_unqualified &&
           (!e.actual_execution || !std::isfinite(e.runtime_ns) ||
            e.runtime_ns<=0))
            throw std::invalid_argument("CT-DP session: invalid measurement");
        std::optional<double> total{};
        if(e.level==evidence_level::numerically_qualified) {
            total=total_cost_ns(e,req);
            ++qualified_count;
            if(!selected || *total<best || (*total==best && id<best_id)) {
                best=*total;
                best_id=id;
                selected=records.size();
            }
        }
        records.push_back({plan,id,e,total});
    }
    return result<decltype(native),plan_type>{
        std::move(native),std::move(records),selected,qualified_count};
}

} // namespace ctdp::solver::session
