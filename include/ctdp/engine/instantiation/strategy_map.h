// engine/instantiation/strategy_map.h — Strategy → Implementation mapping
// Part of the compile-time DP library (C++20)
//
// DESIGN RATIONALE:
// The solver produces a plan<Candidate> where each position is assigned
// a Strategy enum value.  Deployment needs to map that enum to an actual
// implementation: a function pointer, a lambda, an index, a codegen tag,
// or any other type the user chooses.
//
// strategy_map<Strategy, Impl, MaxS> is that mapping.  It is:
// - Constexpr:  can be built and queried at compile time
// - Value-typed: copyable, storable in arrays
// - Generic:    Impl is whatever the user needs
// - Small:      linear scan (MaxS ≤ ~16 typical for strategy enums)
//
// USAGE:
// ```cpp
// enum class Strat { Fast, Medium, Safe };
// using FnPtr = double(*)(double const*, std::size_t);
//
// constexpr auto smap = make_strategy_map<Strat, FnPtr>(
//     std::pair{Strat::Fast,   &compute_fast},
//     std::pair{Strat::Medium, &compute_medium},
//     std::pair{Strat::Safe,   &compute_safe}
// );
//
// auto fn = smap[Strat::Fast];  // → &compute_fast
// ```

#ifndef CTDP_ENGINE_STRATEGY_MAP_H
#define CTDP_ENGINE_STRATEGY_MAP_H

#include <array>
#include <cstddef>
#include <utility>

namespace ctdp {

// =========================================================================
// strategy_map: constexpr map from Strategy → Impl
// =========================================================================

/// A compile-time map from strategy enum values to implementation objects.
///
/// Template parameters:
/// - Strategy: the enum type (e.g., Strat, SpmvFormat)
/// - Impl: the implementation type (function pointer, lambda, index, etc.)
/// - MaxS: maximum number of strategies (capacity)
///
/// Lookup is O(MaxS) linear scan.  For strategy enums with < 16 values
/// this is optimal (no hashing overhead, cache-friendly, constexpr-safe).
template<typename Strategy, typename Impl, std::size_t MaxS>
struct strategy_map {
    struct entry {
        Strategy key{};
        Impl     value{};
    };

    std::array<entry, MaxS> entries_{};
    std::size_t size_ = 0;

    // --- Construction ---

    constexpr strategy_map() = default;

    /// Bind a strategy to an implementation.
    /// Overwrites if already bound.
    constexpr void bind(Strategy s, Impl impl) {
        // Check for existing binding.
        for (std::size_t i = 0; i < size_; ++i) {
            if (entries_[i].key == s) {
                entries_[i].value = impl;
                return;
            }
        }
        // New binding.
        if (size_ < MaxS) {
            entries_[size_] = entry{s, impl};
            ++size_;
        }
        // Silent no-op if full.  In practice, MaxS == number of enum values.
    }

    // --- Lookup ---

    /// Look up the implementation for a strategy.
    /// Returns default-constructed Impl if not found.
    [[nodiscard]] constexpr Impl const& operator[](Strategy s) const {
        for (std::size_t i = 0; i < size_; ++i) {
            if (entries_[i].key == s) {
                return entries_[i].value;
            }
        }
        // Not found — return first entry as fallback (or UB if empty).
        // In well-formed usage, every strategy in the plan is bound.
        return entries_[0].value;
    }

    /// Check if a strategy has a binding.
    [[nodiscard]] constexpr bool contains(Strategy s) const {
        for (std::size_t i = 0; i < size_; ++i) {
            if (entries_[i].key == s) return true;
        }
        return false;
    }

    /// Number of bound strategies.
    [[nodiscard]] constexpr std::size_t size() const { return size_; }
};

// =========================================================================
// Factory: make_strategy_map from pairs
// =========================================================================

/// Build a strategy_map from (Strategy, Impl) pairs.
///
/// Example:
/// ```cpp
/// auto smap = make_strategy_map<Strat, FnPtr>(
///     std::pair{Strat::Fast,   &fast_fn},
///     std::pair{Strat::Medium, &medium_fn},
///     std::pair{Strat::Safe,   &safe_fn}
/// );
/// ```
template<typename Strategy, typename Impl, typename... Pairs>
[[nodiscard]] constexpr auto
make_strategy_map(Pairs&&... pairs) {
    strategy_map<Strategy, Impl, sizeof...(Pairs)> result;
    (result.bind(pairs.first, pairs.second), ...);
    return result;
}

// =========================================================================
// Concept: strategy_mappable
// =========================================================================

/// Check if a type provides strategy → impl lookup.
template<typename M, typename Strategy>
concept strategy_mappable = requires(M const& m, Strategy s) {
    { m[s] };
    { m.contains(s) } -> std::same_as<bool>;
};

} // namespace ctdp

#endif // CTDP_ENGINE_STRATEGY_MAP_H
