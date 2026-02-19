// ctdp/solver/cost_models/chain.h
// Compile-time dynamic programming framework — Analytics: Solver
// Interval cost model: combine(i, k, j) + leaf(i).
// Satisfies: interval_cost concept.
//
// Primary use: matrix chain multiplication.
// The domain closes over dimensions at construction via make_chain_cost().

#ifndef CTDP_SOLVER_COST_MODELS_CHAIN_H
#define CTDP_SOLVER_COST_MODELS_CHAIN_H

#include "../concepts.h"
#include <array>
#include <cstddef>

namespace ctdp {

// ---------------------------------------------------------------------------
// chain_cost: wraps a combine function for interval DP.
//
// Follows right-start convention: combine(i, mid, j) where mid is the
// start index of the right subproblem.
// ---------------------------------------------------------------------------
template<typename CombineFn>
struct chain_cost {
    CombineFn combine_fn;

    [[nodiscard]] constexpr auto combine(std::size_t i, std::size_t mid,
                                          std::size_t j) const
        -> decltype(combine_fn(i, mid, j))
    {
        return combine_fn(i, mid, j);
    }

    [[nodiscard]] constexpr auto leaf(std::size_t /*i*/) const -> double {
        return 0.0;
    }
};

// Deduction guide
template<typename CombineFn>
chain_cost(CombineFn) -> chain_cost<CombineFn>;

// ---------------------------------------------------------------------------
// make_chain_cost: factory that closes over a dimension array.
//
// Matrix chain: dims = [p0, p1, ..., pn], n matrices.
// Matrix i has dimensions dims[i] × dims[i+1].
//
// combine(i, mid, j): cost of multiplying the left result (dims[i] × dims[mid])
//   with the right result (dims[mid] × dims[j+1]) = dims[i] × dims[mid] × dims[j+1].
//
// mid is the start of the right half (right-start convention).
// ---------------------------------------------------------------------------
template<std::size_t MaxN>
[[nodiscard]] constexpr auto make_chain_cost(
    std::array<std::size_t, MaxN> const& dims)
{
    return chain_cost{
        [dims](std::size_t i, std::size_t mid, std::size_t j) constexpr -> double {
            return static_cast<double>(dims[i] * dims[mid] * dims[j + 1]);
        }
    };
}

// Verify interval_cost satisfaction
namespace detail {
    struct chain_cost_test_ {
        constexpr auto combine(std::size_t, std::size_t, std::size_t) const
            -> double { return 0.0; }
        constexpr auto leaf(std::size_t) const -> double { return 0.0; }
    };
    static_assert(interval_cost<chain_cost_test_>);
}

} // namespace ctdp

#endif // CTDP_SOLVER_COST_MODELS_CHAIN_H
