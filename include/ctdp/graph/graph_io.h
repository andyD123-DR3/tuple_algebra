// graph/graph_io.h — I/O layer: constexpr parsing, runtime read/write, DOT export
// Part of the compile-time DP library (C++20)
//
// Umbrella header.  Includes all graph I/O sub-headers for backward
// compatibility.  For compilation-time-sensitive translation units,
// prefer including individual headers:
//
//   graph_io_parse.h   — constexpr parsing only (no iostream)
//   graph_io_stream.h  — runtime stream-based read/write
//   graph_io_dot.h     — Graphviz DOT export
//
// TEXT FORMAT:
//   # comment
//   nodes N              (must appear before any edge lines)
//   edge SRC DST         (directed edge)
//   edge SRC DST WEIGHT  (weighted edge — only in parse_weighted variants)
//   symmetric            (flag: parse as undirected — only in parse_symmetric)
//
// Lines starting with # are comments.  Blank lines are ignored.
// Node IDs are zero-based unsigned integers.  Weights are decimal numbers
// (integer or integer.fraction, optional leading minus).

#ifndef CTDP_GRAPH_IO_H
#define CTDP_GRAPH_IO_H

#include "graph_io_detail.h"
#include "graph_io_parse.h"
#include "graph_io_stream.h"
#include "graph_io_dot.h"

#endif // CTDP_GRAPH_IO_H
