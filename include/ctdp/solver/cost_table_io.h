// solver/cost_table_io.h — I/O layer: constexpr parsing, runtime read/write
// Part of the compile-time DP library (C++20)
//
// Umbrella header.  Includes all cost-table I/O sub-headers.
// For compilation-time-sensitive translation units, prefer individual headers:
//
//   cost_table_io_parse.h   — constexpr parsing only (no iostream)
//   cost_table_io_stream.h  — runtime stream-based read/write
//
// TEXT FORMAT:
//   # comment
//   elements N
//   strategies M
//   row I  c0 c1 c2 ... c(M-1)
//
// Lines starting with # are comments.  Blank lines are ignored.
// Element indices are zero-based unsigned integers.
// Costs are decimal numbers (integer or integer.fraction, optional leading minus).

#ifndef CTDP_SOLVER_COST_TABLE_IO_H
#define CTDP_SOLVER_COST_TABLE_IO_H

#include "cost_table_io_parse.h"
#include "cost_table_io_stream.h"

#endif // CTDP_SOLVER_COST_TABLE_IO_H
