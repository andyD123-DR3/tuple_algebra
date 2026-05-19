// ctdp/solver/memo/triangular_memo.h
// Canonical Stage 1 dense interval memo.

#ifndef CTDP_SOLVER_MEMO_TRIANGULAR_MEMO_H
#define CTDP_SOLVER_MEMO_TRIANGULAR_MEMO_H

#include "ct_dp/solver/triangular_memo.hpp"

namespace ctdp::solver::memo {

template <class Value>
using triangular_memo = ::ct_dp::solver::triangular_memo<Value>;

} // namespace ctdp::solver::memo

#endif // CTDP_SOLVER_MEMO_TRIANGULAR_MEMO_H

