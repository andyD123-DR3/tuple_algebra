#ifndef CTDP_CALIBRATOR_FIX_DP_SEARCH_H
#define CTDP_CALIBRATOR_FIX_DP_SEARCH_H

// ============================================================
//  fix_dp_search.h  –  CT-DP FIX Parser Optimiser  (Phase 2)
//
//  DP search over a measured cost table.
//
//  Two algorithms:
//
//  1. exhaustive_search<N>(cost_table)
//     Reads the full CostTable<N,double> and returns the plan
//     with minimum measured p99.  Optimal by construction.
//     Used for trivial_schema (64 plans — completes in microseconds).
//
//  2. beam_dp<N>(cost_table, beam_width)
//     Staged per-field beam search.  At each stage i (field index):
//       - Take the current beam of partial plans (fixed fields 0..i-1)
//       - Expand each candidate over all allowed strategies for field i
//       - Look up cost in cost_table (plan must be fully specified —
//         unmeasured entries are skipped)
//       - Keep the beam_width lowest-cost complete plans
//     Returns sorted results with rank annotations.
//     For N=4 with beam_width=4 this explores all 64 plans in 4 passes.
//     For N=12 with beam_width=4 this explores 4×12=48 candidate plans
//     instead of 4^12 = 16M.
//
//  Cost function contract:
//    cost_table[dense_id] == 0.0  →  unmeasured / skipped
//    cost_table[dense_id]  > 0.0  →  measured p99 in nanoseconds
//
//  DpResult<N>:
//    optimal_plan   — best array<Strategy,N> found
//    optimal_cost   — p99 in nanoseconds
//    optimal_id     — dense_id in SchemaIndex
//    rank_in_corpus — 1-based rank among all measured plans
//    n_measured     — how many plans had cost > 0
//    n_evaluated    — how many plans the algorithm examined
//    solver         — "exhaustive" or "beam_dp(w=K)"
//
//  BeamDpOnline<N>:
//    For full_schema (N=12) where we cannot pre-fill all 16M entries.
//    Takes a measurement callback; measures only the plans it visits.
//    The callback signature:
//      config_metrics callback(const array<Strategy,N>& plan)
//    Results are stored in a provided CostTable (sparse, unmeasured = 0).
//
//  Dependencies: fix_schema.h
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/fix_schema.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  DpResult<N>
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
struct DpResult {
    std::array<Strategy, N> optimal_plan{};
    double   optimal_cost   {0.0};   // p99 ns of the optimal plan
    int32_t  optimal_id     {-1};    // dense_id in SchemaIndex
    int      rank_in_corpus {0};     // 1-based rank among measured plans
    int      n_measured     {0};     // plans with cost > 0 in table
    int      n_evaluated    {0};     // plans examined by the algorithm
    std::string solver;              // "exhaustive" or "beam_dp(w=K)"

    [[nodiscard]] bool found()     const noexcept { return optimal_id >= 0; }
    [[nodiscard]] std::string plan_str() const    { return plan_string<N>(optimal_plan); }

    // All measured plans sorted ascending by cost (populated by exhaustive only)
    struct RankedPlan {
        std::array<Strategy, N> plan{};
        double cost{0.0};
        int32_t dense_id{-1};
    };
    std::vector<RankedPlan> ranked_corpus;   // empty for beam_dp
};

// ─────────────────────────────────────────────────────────────────────────────
//  exhaustive_search<N>
//
//  Scans every entry in cost_table, collects measured plans (cost > 0),
//  sorts them ascending, and returns DpResult with full ranked_corpus.
//  Optimal for small N (trivial_schema: 64 plans, sub-microsecond).
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] DpResult<N>
exhaustive_search(const CostTable<N, double>& cost_table)
{
    const SchemaIndex<N>& idx = cost_table.index();
    DpResult<N> result;
    result.solver = "exhaustive";

    // Collect all measured plans
    for (int32_t id = 0; id < idx.size(); ++id) {
        double cost = cost_table[id];
        if (cost <= 0.0) continue;
        ++result.n_measured;
        result.ranked_corpus.push_back({
            idx.decode(id), cost, id
        });
    }
    result.n_evaluated = result.n_measured;

    if (result.ranked_corpus.empty()) return result;

    // Sort ascending by cost
    std::sort(result.ranked_corpus.begin(), result.ranked_corpus.end(),
        [](const auto& a, const auto& b){ return a.cost < b.cost; });

    // Optimal is rank 1
    const auto& best = result.ranked_corpus.front();
    result.optimal_plan  = best.plan;
    result.optimal_cost  = best.cost;
    result.optimal_id    = best.dense_id;
    result.rank_in_corpus = 1;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  beam_dp<N>
//
//  Staged per-field beam search on a pre-filled CostTable.
//
//  Algorithm:
//    beam = { partial plan with no fields fixed }
//    for field i = 0..N-1:
//      expand: for each partial plan in beam, try all allowed strategies
//              for field i, forming a fully-determined plan
//              (fields beyond i default to the first allowed strategy)
//      evaluate: look up cost_table[encode(plan)]
//      prune: keep best beam_width complete plans
//    return minimum-cost plan from final beam
//
//  For trivial_schema (beam_width=64): equivalent to exhaustive.
//  For full_schema (beam_width=4): O(N × beam_width × max_strategies) lookups.
//
//  Plans with cost == 0.0 (unmeasured) are treated as +∞ and skipped.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] DpResult<N>
beam_dp(const CostTable<N, double>& cost_table,
        int                          beam_width = 4)
{
    const SchemaIndex<N>& idx = cost_table.index();

    struct Candidate {
        std::array<Strategy, N> plan{};
        double cost{std::numeric_limits<double>::infinity()};
        int32_t dense_id{-1};
    };

    // Seed beam with one candidate per allowed strategy at field 0
    std::vector<Candidate> beam;
    beam.reserve(static_cast<std::size_t>(beam_width));

    // Build seed: one candidate per strategy allowed at field 0,
    // all other fields filled with the first allowed strategy for that field
    std::array<Strategy, N> default_plan{};
    for (int fi = 0; fi < N; ++fi)
        default_plan[fi] = idx.local_to_strategy(fi, 0);

    for (int si = 0; si < idx.n_allowed(0); ++si) {
        Candidate c;
        c.plan         = default_plan;
        c.plan[0]      = idx.local_to_strategy(0, si);
        c.dense_id     = idx.encode(c.plan);
        c.cost         = (c.dense_id >= 0) ? cost_table[c.dense_id] : 0.0;
        if (c.cost <= 0.0) c.cost = std::numeric_limits<double>::infinity();
        beam.push_back(c);
    }

    int n_evaluated = static_cast<int>(beam.size());

    // Expand field by field
    for (int fi = 1; fi < N; ++fi) {
        std::vector<Candidate> next;
        next.reserve(static_cast<std::size_t>(beam_width));

        for (const auto& parent : beam) {
            for (int si = 0; si < idx.n_allowed(fi); ++si) {
                Candidate c;
                c.plan     = parent.plan;
                c.plan[fi] = idx.local_to_strategy(fi, si);
                c.dense_id = idx.encode(c.plan);
                c.cost     = (c.dense_id >= 0) ? cost_table[c.dense_id] : 0.0;
                if (c.cost <= 0.0) c.cost = std::numeric_limits<double>::infinity();
                next.push_back(c);
                ++n_evaluated;
            }
        }

        // Prune to beam_width best
        if (static_cast<int>(next.size()) > beam_width) {
            std::partial_sort(next.begin(),
                next.begin() + beam_width,
                next.end(),
                [](const Candidate& a, const Candidate& b){
                    return a.cost < b.cost;
                });
            next.resize(static_cast<std::size_t>(beam_width));
        }
        beam = std::move(next);
    }

    // Result
    std::ostringstream ss;
    ss << "beam_dp(w=" << beam_width << ")";

    DpResult<N> result;
    result.solver      = ss.str();
    result.n_evaluated = n_evaluated;

    // Count measured plans in the full table for n_measured
    for (int32_t id = 0; id < idx.size(); ++id)
        if (cost_table[id] > 0.0) ++result.n_measured;

    if (beam.empty()) return result;

    // Sort beam ascending; best is at front
    std::sort(beam.begin(), beam.end(),
        [](const Candidate& a, const Candidate& b){ return a.cost < b.cost; });

    const auto& best = beam.front();
    if (std::isinf(best.cost)) return result;   // no measured plan found

    result.optimal_plan  = best.plan;
    result.optimal_cost  = best.cost;
    result.optimal_id    = best.dense_id;

    // Compute rank among all measured plans
    int rank = 1;
    for (int32_t id = 0; id < idx.size(); ++id)
        if (cost_table[id] > 0.0 && cost_table[id] < best.cost)
            ++rank;
    result.rank_in_corpus = rank;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BeamDpOnline<N>
//
//  For full_schema (N=12) where pre-filling all 4^12 entries is infeasible.
//  Runs the same staged beam expansion but calls a measurement callback
//  for each plan it visits.  Results are stored in the provided CostTable
//  (sparse — only visited plans are filled).
//
//  Usage:
//    CostTable<12, double> sparse(full_index, 0.0);
//    BeamDpOnline<12> searcher(full_index, beam_width=4);
//    auto result = searcher.run(sparse, [](const array<Strategy,12>& plan){
//        return measure_one_plan(plan);   // returns p99 in ns
//    });
//
//  The callback returns double (measured p99).  A return of 0.0 means
//  measurement failed; that candidate is treated as +∞.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
class BeamDpOnline {
public:
    using MeasureFn = std::function<double(const std::array<Strategy, N>&)>;
    using ProgressFn = std::function<void(int /*done*/, int /*total*/)>;

    explicit BeamDpOnline(const SchemaIndex<N>& index,
                          int                   beam_width = 4)
        : index_{&index}
        , beam_width_{beam_width}
    {}

    void set_progress_cb(ProgressFn cb) { progress_cb_ = std::move(cb); }

    [[nodiscard]] DpResult<N>
    run(CostTable<N, double>& cost_table,
        MeasureFn              measure_fn)
    {
        const SchemaIndex<N>& idx = *index_;

        struct Candidate {
            std::array<Strategy, N> plan{};
            double cost{std::numeric_limits<double>::infinity()};
            int32_t dense_id{-1};
        };

        // Estimate total measurements for progress: N stages × beam_width × 4
        const int est_total = N * beam_width_ * NUM_STRATEGIES;
        int done = 0;

        auto measure_and_store = [&](Candidate& c) {
            int32_t id = idx.encode(c.plan);
            c.dense_id = id;
            if (id < 0) {
                c.cost = std::numeric_limits<double>::infinity();
                return;
            }
            // Check if already measured (re-use)
            if (cost_table[id] > 0.0) {
                c.cost = cost_table[id];
                return;
            }
            double ns = measure_fn(c.plan);
            if (ns > 0.0) cost_table[id] = ns;
            c.cost = (ns > 0.0) ? ns : std::numeric_limits<double>::infinity();
            if (progress_cb_) progress_cb_(++done, est_total);
        };

        // Seed
        std::array<Strategy, N> default_plan{};
        for (int fi = 0; fi < N; ++fi)
            default_plan[fi] = idx.local_to_strategy(fi, 0);

        std::vector<Candidate> beam;
        beam.reserve(static_cast<std::size_t>(beam_width_));
        for (int si = 0; si < idx.n_allowed(0); ++si) {
            Candidate c;
            c.plan    = default_plan;
            c.plan[0] = idx.local_to_strategy(0, si);
            measure_and_store(c);
            beam.push_back(c);
        }

        int n_evaluated = static_cast<int>(beam.size());

        // Staged expansion
        for (int fi = 1; fi < N; ++fi) {
            // Prune beam first
            if (static_cast<int>(beam.size()) > beam_width_) {
                std::partial_sort(beam.begin(),
                    beam.begin() + beam_width_,
                    beam.end(),
                    [](const Candidate& a, const Candidate& b){
                        return a.cost < b.cost;
                    });
                beam.resize(static_cast<std::size_t>(beam_width_));
            }

            std::vector<Candidate> next;
            next.reserve(static_cast<std::size_t>(
                beam.size() * static_cast<std::size_t>(idx.n_allowed(fi))));

            for (const auto& parent : beam) {
                for (int si = 0; si < idx.n_allowed(fi); ++si) {
                    Candidate c;
                    c.plan     = parent.plan;
                    c.plan[fi] = idx.local_to_strategy(fi, si);
                    measure_and_store(c);
                    next.push_back(c);
                    ++n_evaluated;
                }
            }
            beam = std::move(next);
        }

        // Build result
        std::ostringstream ss;
        ss << "beam_dp_online(w=" << beam_width_ << ")";

        DpResult<N> result;
        result.solver      = ss.str();
        result.n_evaluated = n_evaluated;

        // Count measured in sparse table
        for (int32_t id = 0; id < idx.size(); ++id)
            if (cost_table[id] > 0.0) ++result.n_measured;

        if (beam.empty()) return result;

        std::sort(beam.begin(), beam.end(),
            [](const Candidate& a, const Candidate& b){
                return a.cost < b.cost;
            });

        const auto& best = beam.front();
        if (std::isinf(best.cost)) return result;

        result.optimal_plan  = best.plan;
        result.optimal_cost  = best.cost;
        result.optimal_id    = best.dense_id;

        // Rank among all visited plans
        int rank = 1;
        for (int32_t id = 0; id < idx.size(); ++id)
            if (cost_table[id] > 0.0 && cost_table[id] < best.cost)
                ++rank;
        result.rank_in_corpus = rank;

        return result;
    }

private:
    const SchemaIndex<N>* index_;
    int                   beam_width_;
    ProgressFn            progress_cb_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  verify_beam_matches_exhaustive<N>
//
//  Debug utility: run both exhaustive and beam_dp on the same cost_table
//  and assert they find the same optimal plan.
//  Returns true if they agree, false otherwise.
//  Only valid when the cost_table is fully populated.
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
[[nodiscard]] bool
verify_beam_matches_exhaustive(const CostTable<N, double>& cost_table,
                                int                         beam_width)
{
    auto ex   = exhaustive_search<N>(cost_table);
    auto beam = beam_dp<N>(cost_table, beam_width);
    if (!ex.found() || !beam.found()) return ex.found() == beam.found();
    return ex.optimal_id == beam.optimal_id;
}

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_DP_SEARCH_H
