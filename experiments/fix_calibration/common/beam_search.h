#ifndef EXPERIMENT_FIX_CALIBRATION_BEAM_SEARCH_H
#define EXPERIMENT_FIX_CALIBRATION_BEAM_SEARCH_H

// experiments/fix_calibration/common/beam_search.h
//
// Left-to-right constructive beam search over the 12-field x 4-strategy
// FIX parser configuration space.
//
// Algorithm:
//   1. Seed the beam: deduplicate seeds, evaluate each unique config,
//      keep top beam_width.
//   2. For each field position 0..11 (left to right):
//      a. Expand: for each beam entry, try all 4 strategies at that field.
//      b. Deduplicate (by packed structural key).
//      c. Score every unique candidate with the model.
//      d. Sort by (score, config_key), keep top beam_width.
//   3. Return the final beam (sorted, best-first).
//
// This is a constructive search over the field-assignment sequence:
// once a field has been processed, later rounds do not revisit it.
// For separable objectives this recovers the global optimum.  For
// interacting objectives it is heuristic -- the beam width controls
// how much interaction is captured.
//
// Ordering semantics:
//   Beam entries are sorted by a total order: (score, config_key).
//   config_key is a packed 24-bit integer (2 bits per field, 12 fields).
//   This makes truncation at the beam_width boundary fully deterministic
//   across compilers, standard libraries, and platforms.
//
// Preconditions:
//   beam_width > 0  (asserted; zero width is a configuration error)
//   seeds may be empty (returns empty beam, zero evals)
//
// The Model concept requires only:
//   double model.predict(const fix_point& p) const;
// where lower score = better (latency minimisation).
//
// Factored from examples/fix_p99_beam_search.cpp so that:
//   - experiment_runner.h can call beam_search() with any trained model
//   - test_beam_search.cpp can test with a trivial mock model
//   - the same code path is used in all experiment programs

#include "feature_extractors.h"              // fix_point, fix_config, NUM_FIELDS
#include <ctdp/calibrator/fix/data_point.h>  // Strategy, NUM_STRATEGIES
#include <ctdp/calibrator/fix_et_parser.h>   // config_to_string (display only)

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace ctdp::fix_experiment {

// -- config_key: packed structural identifier -----------------------------
// 12 fields x 2 bits = 24 bits.  Fits in uint32_t.
// Used for deduplication and deterministic tie-breaking.

namespace detail {

inline constexpr std::uint32_t config_key(const fix_config& c) noexcept {
    std::uint32_t k = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(NUM_FIELDS); ++i)
        k |= (static_cast<std::uint32_t>(c[i]) & 0x3u) << (2u * i);
    return k;
}

} // namespace detail

// -- beam_entry: one candidate in the beam --------------------------------

struct beam_entry {
    fix_config    config;
    std::uint32_t key;      // detail::config_key(config)
    double        score;    // model-predicted cost (lower = better)
};

// Total order: (score, key).  Deterministic truncation semantics.
inline bool beam_less(const beam_entry& a, const beam_entry& b) noexcept {
    if (a.score < b.score) return true;
    if (b.score < a.score) return false;
    return a.key < b.key;
}

// -- beam_result: complete output of a beam search run --------------------

struct beam_result {
    std::vector<beam_entry> beam;           // sorted best-first
    std::size_t             total_evaluated; // total model.predict() calls
};

// -- beam_search ----------------------------------------------------------

/// Left-to-right constructive beam search.
///
/// \tparam Model   Any type with `double predict(const fix_point&) const`
/// \param  model   Trained model (SVR, linear, or mock)
/// \param  seeds   Initial configurations to populate the beam (duplicates
///                 are removed; each unique seed is evaluated exactly once)
/// \param  beam_width  Maximum entries retained per round (must be > 0)
/// \return beam_result with sorted beam and evaluation count
///
template<typename Model>
beam_result beam_search(
        const Model& model,
        const std::vector<fix_config>& seeds,
        std::size_t beam_width)
{
    assert(beam_width > 0 && "beam_width must be positive");

    std::size_t evals = 0;

    // Phase 1: deduplicate and evaluate seeds, keep top beam_width
    std::unordered_set<std::uint32_t> seed_seen;
    seed_seen.reserve(seeds.size());
    std::vector<beam_entry> beam;
    beam.reserve(seeds.size());

    for (const auto& cfg : seeds) {
        auto k = detail::config_key(cfg);
        if (!seed_seen.insert(k).second) continue;  // skip duplicate

        double s = model.predict(fix_point{cfg});
        beam.push_back(beam_entry{cfg, k, s});
        ++evals;
    }
    std::sort(beam.begin(), beam.end(), beam_less);
    if (beam.size() > beam_width)
        beam.resize(beam_width);

    // Phase 2: field-by-field expansion (left to right, no revisiting)
    for (int field = 0; field < NUM_FIELDS; ++field) {
        std::unordered_set<std::uint32_t> seen;
        seen.reserve(beam.size() * static_cast<std::size_t>(NUM_STRATEGIES));
        std::vector<beam_entry> candidates;
        candidates.reserve(beam.size() * static_cast<std::size_t>(NUM_STRATEGIES));

        for (const auto& entry : beam) {
            for (int s = 0; s < NUM_STRATEGIES; ++s) {
                fix_config trial = entry.config;
                trial[static_cast<std::size_t>(field)] =
                    static_cast<Strategy>(s);

                auto k = detail::config_key(trial);
                if (!seen.insert(k).second) continue;

                double sc = model.predict(fix_point{trial});
                candidates.push_back(beam_entry{trial, k, sc});
                ++evals;
            }
        }

        std::sort(candidates.begin(), candidates.end(), beam_less);
        if (candidates.size() > beam_width)
            candidates.resize(beam_width);
        beam = std::move(candidates);
    }

    return beam_result{std::move(beam), evals};
}

} // namespace ctdp::fix_experiment

#endif // EXPERIMENT_FIX_CALIBRATION_BEAM_SEARCH_H
