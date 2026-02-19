// ctdp/solver/memo/cost_purity.h
// Compile-time dynamic programming framework — Analytics: Solver
// Declared-purity trait for cost functions.
//
// Purity is like noexcept: you opt IN, you're responsible.
//
// Default: IMPURE (false).  The cache will not memoise results unless
// the cost function explicitly declares purity.  This is the safe
// default — a forgotten opt-in means slower (re-evaluated) but correct;
// a forgotten opt-out means cached stale results (silent bug).
//
// Opt in:
//   template<> struct ctdp::cost_purity<my_pure_cost> : std::true_type {};
//
// Or for an entire template family:
//   template<typename T>
//   struct ctdp::cost_purity<additive_cost<T>> : std::true_type {};

#ifndef CTDP_SOLVER_MEMO_COST_PURITY_H
#define CTDP_SOLVER_MEMO_COST_PURITY_H

#include <type_traits>

namespace ctdp {

template<typename Cost>
struct cost_purity : std::false_type {};

template<typename Cost>
inline constexpr bool cost_is_pure_v = cost_purity<Cost>::value;

} // namespace ctdp

#endif // CTDP_SOLVER_MEMO_COST_PURITY_H
