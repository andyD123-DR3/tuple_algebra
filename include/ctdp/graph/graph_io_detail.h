// graph/graph_io_detail.h — Forwarding header for constexpr parsing primitives
// Part of the compile-time DP library (C++20)
//
// The canonical implementation now lives in <ctdp/core/io_lexer.h> under
// namespace ctdp::io.  This header forwards all symbols into the original
// ctdp::graph::io::detail namespace so that existing consumers
// (graph_io_parse.h, graph_io_stream.h) continue to compile unchanged.

#ifndef CTDP_GRAPH_IO_DETAIL_H
#define CTDP_GRAPH_IO_DETAIL_H

#include <ctdp/core/io_lexer.h>

namespace ctdp::graph::io {
namespace detail {

using ctdp::io::skip_hspace;
using ctdp::io::skip_to_eol;
using ctdp::io::at_eol;
using ctdp::io::parse_uint;
using ctdp::io::parse_double;
using ctdp::io::starts_with;

} // namespace detail
} // namespace ctdp::graph::io

#endif // CTDP_GRAPH_IO_DETAIL_H
