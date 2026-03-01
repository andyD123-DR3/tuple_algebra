// solver/cost_table.h — Constexpr 2D cost table for DP planning
// Part of the compile-time DP library (C++20)
//
// A cost_table maps (element, strategy) pairs to floating-point costs.
// The table is dense: every element has a cost for every strategy.
// Dimensions are set at construction; capacity is bounded by the Cap policy.
//
// TEXT FORMAT (for I/O — see cost_table_io_parse.h / cost_table_io_stream.h):
//   elements N
//   strategies M
//   row 0  c0 c1 c2 ... c(M-1)
//   row 1  c0 c1 c2 ... c(M-1)
//   ...
//
// DESIGN DECISIONS:
//   - Flat row-major storage: std::array<double, MaxE * MaxS>.
//     Compile-time-friendly, cache-friendly for row scans.
//   - Capacity concept (cost_table_capacity) mirrors graph::capacity_policy.
//   - Named tiers (ct_cap::tiny through ct_cap::large) for ergonomic use.
//   - `constexpr` not `consteval` — same code at compile time and runtime.

#ifndef CTDP_SOLVER_COST_TABLE_H
#define CTDP_SOLVER_COST_TABLE_H

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ctdp {

// =============================================================================
// cost_table_capacity concept
// =============================================================================

/// A type satisfies cost_table_capacity if it provides positive max_elements
/// and max_strategies.
template<typename C>
concept cost_table_capacity = requires {
    { C::max_elements }   -> std::convertible_to<std::size_t>;
    { C::max_strategies } -> std::convertible_to<std::size_t>;
} && (C::max_elements > 0) && (C::max_strategies > 0);

// =============================================================================
// Named capacity tiers
// =============================================================================

namespace ct_cap {

/// 8 elements × 8 strategies — unit tests, tiny examples.
struct tiny {
    static constexpr std::size_t max_elements   = 8;
    static constexpr std::size_t max_strategies = 8;
};

/// 16 elements × 16 strategies — small demos.
struct small {
    static constexpr std::size_t max_elements   = 16;
    static constexpr std::size_t max_strategies = 16;
};

/// 64 elements × 32 strategies — moderate planning problems.
struct medium {
    static constexpr std::size_t max_elements   = 64;
    static constexpr std::size_t max_strategies = 32;
};

/// 256 elements × 64 strategies — large calibration tables.
struct large {
    static constexpr std::size_t max_elements   = 256;
    static constexpr std::size_t max_strategies = 64;
};

} // namespace ct_cap

// =============================================================================
// ct_cap_from<E, S> — inline capacity from raw numbers
// =============================================================================

template<std::size_t MaxE, std::size_t MaxS = MaxE>
struct ct_cap_from {
    static constexpr std::size_t max_elements   = MaxE;
    static constexpr std::size_t max_strategies = MaxS;
};

// =============================================================================
// cost_table<Cap>
// =============================================================================

/// Dense 2D cost table: num_elements rows × num_strategies columns.
///
/// Storage is flat row-major: element i, strategy j → data_[i * MaxS + j].
/// All MaxE × MaxS entries exist (zero-initialised); only the active
/// rectangle [0, num_elements) × [0, num_strategies) is meaningful.
///
/// Example:
/// ```cpp
/// constexpr auto ct = []() {
///     cost_table<ct_cap::tiny> t(3, 2);
///     t(0, 0) = 1.0; t(0, 1) = 2.0;
///     t(1, 0) = 3.0; t(1, 1) = 4.0;
///     t(2, 0) = 5.0; t(2, 1) = 6.0;
///     return t;
/// }();
/// static_assert(ct(1, 0) == 3.0);
/// ```
template<cost_table_capacity Cap = ct_cap::medium>
class cost_table {
public:
    static constexpr std::size_t max_elements   = Cap::max_elements;
    static constexpr std::size_t max_strategies = Cap::max_strategies;

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Default: 0 × 0 table.
    constexpr cost_table() noexcept = default;

    /// Construct with given dimensions.  Throws if exceeds capacity.
    constexpr cost_table(std::size_t elems, std::size_t strats)
        : num_elements_{elems}, num_strategies_{strats}
    {
        if (elems > max_elements) {
            throw std::runtime_error("cost_table: element count exceeds capacity");
        }
        if (strats > max_strategies) {
            throw std::runtime_error("cost_table: strategy count exceeds capacity");
        }
    }

    // -----------------------------------------------------------------
    // Dimensions
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr std::size_t num_elements()   const noexcept { return num_elements_; }
    [[nodiscard]] constexpr std::size_t num_strategies() const noexcept { return num_strategies_; }
    [[nodiscard]] constexpr bool        empty()          const noexcept { return num_elements_ == 0; }

    // -----------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------

    /// Access cost(element, strategy).  Bounds-checked.
    [[nodiscard]] constexpr double& operator()(std::size_t elem, std::size_t strat) {
        check_bounds(elem, strat);
        return data_[elem * max_strategies + strat];
    }

    [[nodiscard]] constexpr double operator()(std::size_t elem, std::size_t strat) const {
        check_bounds(elem, strat);
        return data_[elem * max_strategies + strat];
    }

    // -----------------------------------------------------------------
    // Row access (pointer to first strategy cost for element i)
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr double const* row(std::size_t elem) const {
        if (elem >= num_elements_) {
            throw std::runtime_error("cost_table::row: element out of range");
        }
        return data_.data() + elem * max_strategies;
    }

    // -----------------------------------------------------------------
    // Equality
    // -----------------------------------------------------------------

    [[nodiscard]] constexpr bool operator==(cost_table const& other) const noexcept {
        if (num_elements_ != other.num_elements_ ||
            num_strategies_ != other.num_strategies_) {
            return false;
        }
        for (std::size_t i = 0; i < num_elements_; ++i) {
            for (std::size_t j = 0; j < num_strategies_; ++j) {
                if (data_[i * max_strategies + j] !=
                    other.data_[i * max_strategies + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] constexpr bool operator!=(cost_table const& other) const noexcept {
        return !(*this == other);
    }

private:
    std::array<double, max_elements * max_strategies> data_{};
    std::size_t num_elements_   = 0;
    std::size_t num_strategies_ = 0;

    constexpr void check_bounds(std::size_t elem, std::size_t strat) const {
        if (elem >= num_elements_) {
            throw std::runtime_error("cost_table: element index out of range");
        }
        if (strat >= num_strategies_) {
            throw std::runtime_error("cost_table: strategy index out of range");
        }
    }
};

} // namespace ctdp

#endif // CTDP_SOLVER_COST_TABLE_H
