#ifndef CTDP_CALIBRATOR_SAMPLER_H
#define CTDP_CALIBRATOR_SAMPLER_H

// ctdp::calibrator::sampler — Deterministic batch extraction from space
//
// Design v2.2 §5.2 note:
//   "Both Scenario (via points()) and sampler.h can enumerate points.
//    The authoritative source is Scenario — sampler.h produces a batch
//    by sampling from scenario.points()."
//
// Three sampling strategies:
//   full_sampler     — use all points (identity pass-through)
//   stride_sampler   — every Nth point (systematic sampling)
//   random_sampler   — deterministic pseudo-random subset (seeded)
//
// All samplers produce a vector<point_type> that can be fed to a
// calibration_harness or used directly.

#include "scenario.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ctdp::calibrator {

// ─── Full sampler (identity) ─────────────────────────────────────

/// Returns all points from the scenario.  The trivial case.
struct full_sampler {
    template <Scenario S>
    [[nodiscard]] auto sample(S const& scenario) const
        -> std::vector<typename S::point_type>
    {
        auto const& pts = scenario.points();
        return {pts.begin(), pts.end()};
    }
};

// ─── Stride sampler (systematic) ─────────────────────────────────

/// Returns every Nth point.  Deterministic.
/// If the total point count <= max_points, returns all of them.
struct stride_sampler {
    std::size_t max_points = 100;  ///< Maximum number of points to return

    template <Scenario S>
    [[nodiscard]] auto sample(S const& scenario) const
        -> std::vector<typename S::point_type>
    {
        auto const& pts = scenario.points();
        auto total = static_cast<std::size_t>(
            std::distance(pts.begin(), pts.end()));

        if (total <= max_points) {
            return {pts.begin(), pts.end()};
        }

        // Stride to get approximately max_points samples
        std::size_t stride = total / max_points;
        if (stride == 0) stride = 1;

        std::vector<typename S::point_type> result;
        result.reserve(max_points);

        std::size_t idx = 0;
        for (auto it = pts.begin(); it != pts.end() && result.size() < max_points;
             ++it, ++idx) {
            if (idx % stride == 0) {
                result.push_back(*it);
            }
        }
        return result;
    }
};

// ─── Random sampler (deterministic pseudo-random) ────────────────

/// Selects a deterministic pseudo-random subset using a seeded PRNG.
/// Reproducible: same seed + same points → same sample.
///
/// Uses Fisher–Yates shuffle on indices, then takes the first N.
struct random_sampler {
    std::size_t max_points = 100;
    std::uint64_t seed     = 42;

    template <Scenario S>
    [[nodiscard]] auto sample(S const& scenario) const
        -> std::vector<typename S::point_type>
    {
        auto const& pts = scenario.points();

        // Collect all points into a vector for index-based access
        std::vector<typename S::point_type> all(pts.begin(), pts.end());
        auto total = all.size();

        if (total <= max_points) {
            return all;
        }

        // Fisher–Yates partial shuffle: only need first max_points elements
        auto rng_state = seed;
        auto n = max_points;
        for (std::size_t i = 0; i < n; ++i) {
            // SplitMix64-style step
            rng_state += 0x9e3779b97f4a7c15ULL;
            std::uint64_t z = rng_state;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z ^= (z >> 31);

            std::size_t j = i + static_cast<std::size_t>(z % (total - i));
            std::swap(all[i], all[j]);
        }

        all.resize(n);
        return all;
    }
};

// ─── Sampler concept ─────────────────────────────────────────────

/// A Sampler produces a subset of points from a Scenario.
template <typename Samp, typename Scen>
concept Sampler = Scenario<Scen> &&
    requires(Samp const& samp, Scen const& scen) {
        { samp.sample(scen) }
            -> std::same_as<std::vector<typename Scen::point_type>>;
    };

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_SAMPLER_H
