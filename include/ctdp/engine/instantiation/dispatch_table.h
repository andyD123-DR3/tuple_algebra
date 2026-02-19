// engine/instantiation/dispatch_table.h — Positional strategy dispatch
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// A plan assigns a strategy to each position.  Dispatch translates
// (position, strategy) → implementation.  Two patterns arise:
//
// 1. UNIFORM:  every position uses the same strategy→impl mapping.
//    Example: SpMV rows all share {CSR, DIA, ELL} → {csr_kernel, ...}.
//    Storage: one strategy_map shared across all positions.
//
// 2. POSITIONAL: each position has its own strategy→impl mapping.
//    Example: FIX parser fields have different available strategies.
//    Storage: array of strategy_maps, one per position.
//
// Both satisfy the dispatchable concept:
//   dt.dispatch(position, strategy) → Impl const&
//
// USAGE:
// ```cpp
// // Uniform: same implementations for all positions
// auto dt = make_uniform_dispatch<Strat, FnPtr>(
//     std::pair{Strat::Fast, &fast_fn},
//     std::pair{Strat::Safe, &safe_fn}
// );
// auto fn = dt.dispatch(/*pos=*/2, Strat::Fast);  // → &fast_fn
//
// // Positional: different per position
// auto dt = make_positional_dispatch<Strat, FnPtr, 3>(
//     make_strategy_map<Strat, FnPtr>(...),  // position 0
//     make_strategy_map<Strat, FnPtr>(...),  // position 1
//     make_strategy_map<Strat, FnPtr>(...)   // position 2
// );
// ```

#ifndef CTDP_ENGINE_DISPATCH_TABLE_H
#define CTDP_ENGINE_DISPATCH_TABLE_H

#include "strategy_map.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

namespace ctdp {

// =========================================================================
// Concept: dispatchable
// =========================================================================

/// A dispatch table maps (position, strategy) → implementation.
template<typename D, typename Strategy>
concept dispatchable = requires(D const& d, std::size_t pos, Strategy s) {
    { d.dispatch(pos, s) };
};

// =========================================================================
// uniform_dispatch: same mapping for all positions
// =========================================================================

/// Dispatch table where every position shares the same strategy→impl map.
///
/// This is the common case: all positions offer the same strategies
/// (e.g., all SpMV rows can choose CSR/DIA/ELL).
///
/// Template parameters:
/// - Strategy: enum type
/// - Impl: implementation type
/// - MaxS: max strategies per position
template<typename Strategy, typename Impl, std::size_t MaxS>
struct uniform_dispatch {
    strategy_map<Strategy, Impl, MaxS> map_{};

    /// Look up the implementation for a strategy (position-independent).
    [[nodiscard]] constexpr Impl const&
    dispatch([[maybe_unused]] std::size_t pos, Strategy s) const {
        return map_[s];
    }

    /// Direct access to the underlying strategy_map.
    [[nodiscard]] constexpr strategy_map<Strategy, Impl, MaxS> const&
    map() const { return map_; }

    /// Check if a strategy is bound.
    [[nodiscard]] constexpr bool
    contains(Strategy s) const { return map_.contains(s); }
};

// =========================================================================
// positional_dispatch: per-position mapping
// =========================================================================

/// Dispatch table with a separate strategy→impl map per position.
///
/// Used when different positions have different available strategies
/// (e.g., FIX parser fields with heterogeneous strategy sets).
///
/// Template parameters:
/// - Strategy: enum type
/// - Impl: implementation type
/// - MaxS: max strategies per position
/// - N: number of positions
template<typename Strategy, typename Impl, std::size_t MaxS, std::size_t N>
struct positional_dispatch {
    std::array<strategy_map<Strategy, Impl, MaxS>, N> maps_{};

    /// Look up the implementation for (position, strategy).
    [[nodiscard]] constexpr Impl const&
    dispatch(std::size_t pos, Strategy s) const {
        return maps_[pos][s];
    }

    /// Access the strategy_map for a specific position.
    [[nodiscard]] constexpr strategy_map<Strategy, Impl, MaxS> const&
    map_at(std::size_t pos) const { return maps_[pos]; }

    /// Bind an implementation for a specific (position, strategy).
    constexpr void
    bind(std::size_t pos, Strategy s, Impl impl) {
        maps_[pos].bind(s, impl);
    }

    /// Check if (position, strategy) is bound.
    [[nodiscard]] constexpr bool
    contains(std::size_t pos, Strategy s) const {
        return maps_[pos].contains(s);
    }
};

// =========================================================================
// Factory: uniform_dispatch from pairs
// =========================================================================

/// Build a uniform dispatch table from (strategy, impl) pairs.
///
/// Example:
/// ```cpp
/// auto dt = make_uniform_dispatch<Strat, FnPtr>(
///     std::pair{Strat::Fast,   &fast_fn},
///     std::pair{Strat::Medium, &medium_fn}
/// );
/// ```
template<typename Strategy, typename Impl, typename... Pairs>
[[nodiscard]] constexpr auto
make_uniform_dispatch(Pairs&&... pairs) {
    uniform_dispatch<Strategy, Impl, sizeof...(Pairs)> result;
    result.map_ = make_strategy_map<Strategy, Impl>(
        std::forward<Pairs>(pairs)...);
    return result;
}

// =========================================================================
// Factory: positional_dispatch from per-position strategy_maps
// =========================================================================

/// Build a positional dispatch table from N strategy_maps.
///
/// Example:
/// ```cpp
/// auto dt = make_positional_dispatch(
///     make_strategy_map<Strat, FnPtr>(pair1, pair2),  // pos 0
///     make_strategy_map<Strat, FnPtr>(pair3, pair4)   // pos 1
/// );
/// ```
template<typename Strategy, typename Impl, std::size_t MaxS>
[[nodiscard]] constexpr auto
make_positional_dispatch(strategy_map<Strategy, Impl, MaxS> const& single) {
    positional_dispatch<Strategy, Impl, MaxS, 1> result;
    result.maps_[0] = single;
    return result;
}

template<typename Strategy, typename Impl, std::size_t MaxS,
         typename... Rest>
[[nodiscard]] constexpr auto
make_positional_dispatch(strategy_map<Strategy, Impl, MaxS> const& first,
                         Rest const&... rest) {
    constexpr std::size_t N = 1 + sizeof...(Rest);
    positional_dispatch<Strategy, Impl, MaxS, N> result;
    result.maps_[0] = first;
    std::size_t idx = 1;
    ((result.maps_[idx++] = rest), ...);
    return result;
}

} // namespace ctdp

#endif // CTDP_ENGINE_DISPATCH_TABLE_H
