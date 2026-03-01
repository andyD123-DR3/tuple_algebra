#ifndef CTDP_CALIBRATOR_SCENARIO_H
#define CTDP_CALIBRATOR_SCENARIO_H

// ctdp::calibrator::scenario — Scenario concept
//
// A Scenario binds a Space (iterable of point_type) to a Callable
// (the application identity). It is the user-facing abstraction for
// declaring "what to measure".
//
// callable_type propagates through the framework:
//   Scenario → harness → dataset → profile → plan
// This prevents mismatched cost models via the type system.
//
// Note: feature encoding and CSV formatting are NOT part of Scenario.
// Those live in csv_writer and (Phase 3) feature_encoder.
//
// Extracted from calibrator.h:63–80 with the addition of callable_type
// (design v2.2 §3).

#include <ctdp/bench/compiler_barrier.h>

#include <concepts>
#include <ranges>
#include <string_view>

namespace ctdp::calibrator {

/// A Scenario binds a point space to a callable for measurement.
///
/// Required associated types:
///   point_type     — the space point (e.g., struct{int n; Strategy s;})
///   callable_type  — the application identity tag (propagates to plan)
///
/// Required expressions:
///   s.name()       — human-readable scenario name
///   s.points()     — iterable of point_type
///   s.prepare(pt)  — set up state for measuring at this point
///   s.execute(pt)  — run the benchmark; return result_token
template <typename S>
concept Scenario = requires(S& s, typename S::point_type const& pt) {
    typename S::point_type;
    typename S::callable_type;
    { s.name()       } -> std::convertible_to<std::string_view>;
    { s.points()     } -> std::ranges::range;
    { s.prepare(pt)  };
    { s.execute(pt)  } -> std::convertible_to<bench::result_token>;
} && std::same_as<
    std::ranges::range_value_t<decltype(std::declval<S&>().points())>,
    typename S::point_type
>;

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_SCENARIO_H
