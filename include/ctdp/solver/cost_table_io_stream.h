// solver/cost_table_io_stream.h — Runtime stream-based cost-table I/O
// Part of the compile-time DP library (C++20)
//
// Stream-based reading and writing of cost tables in the text format.
// Uses <istream> and <ostream> — NOT <iostream> — to avoid pulling
// in static initialisation of std::cin/cout/cerr/clog.
//
// For constexpr parsing from string literals, use cost_table_io_parse.h
// instead (no stream dependency at all).

#ifndef CTDP_SOLVER_COST_TABLE_IO_STREAM_H
#define CTDP_SOLVER_COST_TABLE_IO_STREAM_H

#include "cost_table.h"
#include <ctdp/core/io_lexer.h>

#include <cstddef>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ctdp::cost_table_io {

// =============================================================================
// Runtime I/O: stream-based writing
// =============================================================================

/// Write a cost table in the text format (matches parse format for round-trip).
///
/// Output:
///   elements N
///   strategies M
///   row 0  1.5 2.0 3.0
///   row 1  4.0 5.0 6.0
///   ...
template<cost_table_capacity Cap>
void write(std::ostream& os, cost_table<Cap> const& ct) {
    os << "elements " << ct.num_elements() << '\n';
    os << "strategies " << ct.num_strategies() << '\n';

    for (std::size_t i = 0; i < ct.num_elements(); ++i) {
        os << "row " << i;
        for (std::size_t j = 0; j < ct.num_strategies(); ++j) {
            os << ' ' << ct(i, j);
        }
        os << '\n';
    }
}

// =============================================================================
// Runtime I/O: stream-based reading
// =============================================================================

/// Read a cost table from a stream.  Same format as parse().
///
/// Default capacity is ct_cap::large — generous to avoid surprising failures
/// when reading tables of unknown size.  Pass an explicit policy for tighter
/// control: read<ct_cap::small>(is).
template<cost_table_capacity Cap = ct_cap::large>
[[nodiscard]] auto read(std::istream& is)
    -> cost_table<Cap>
{
    bool have_elements   = false;
    bool have_strategies = false;
    std::size_t n_elem = 0;
    std::size_t n_strat = 0;
    std::size_t rows_seen = 0;

    cost_table<Cap> ct;
    std::string line;

    while (std::getline(is, line)) {
        std::string_view sv(line);
        std::size_t pos = io::skip_hspace(sv, 0);

        if (io::at_eol(sv, pos)) continue;

        if (io::starts_with(sv, pos, "elements")) {
            if (have_elements) {
                throw std::runtime_error("cost_table read: duplicate 'elements' line");
            }
            pos += 8;
            pos = io::skip_hspace(sv, pos);
            auto [n, next] = io::parse_uint(sv, pos);
            if (n > Cap::max_elements) {
                throw std::runtime_error("cost_table read: element count exceeds capacity");
            }
            n_elem = n;
            have_elements = true;
            if (have_strategies) {
                ct = cost_table<Cap>(n_elem, n_strat);
            }
            continue;
        }

        if (io::starts_with(sv, pos, "strategies")) {
            if (have_strategies) {
                throw std::runtime_error("cost_table read: duplicate 'strategies' line");
            }
            pos += 10;
            pos = io::skip_hspace(sv, pos);
            auto [m, next] = io::parse_uint(sv, pos);
            if (m > Cap::max_strategies) {
                throw std::runtime_error("cost_table read: strategy count exceeds capacity");
            }
            n_strat = m;
            have_strategies = true;
            if (have_elements) {
                ct = cost_table<Cap>(n_elem, n_strat);
            }
            continue;
        }

        if (io::starts_with(sv, pos, "row")) {
            if (!have_elements || !have_strategies) {
                throw std::runtime_error("cost_table read: 'row' before 'elements'/'strategies'");
            }
            pos += 3;
            pos = io::skip_hspace(sv, pos);
            auto [idx, p1] = io::parse_uint(sv, pos);
            pos = p1;

            if (idx >= n_elem) {
                throw std::runtime_error("cost_table read: row index out of range");
            }

            for (std::size_t j = 0; j < n_strat; ++j) {
                pos = io::skip_hspace(sv, pos);
                auto [val, p2] = io::parse_double(sv, pos);
                pos = p2;
                ct(idx, j) = val;
            }

            ++rows_seen;
            continue;
        }

        // Unknown lines are silently skipped in stream reading
        // (lenient, matching graph stream reader behaviour).
    }

    if (!have_elements) {
        throw std::runtime_error("cost_table read: missing 'elements' line");
    }
    if (!have_strategies) {
        throw std::runtime_error("cost_table read: missing 'strategies' line");
    }
    if (rows_seen != n_elem) {
        throw std::runtime_error("cost_table read: incomplete — not all rows provided");
    }

    return ct;
}

/// Read with capacity deduced from a tag object.
template<cost_table_capacity Cap>
[[nodiscard]] auto read(std::istream& is, Cap /*tag*/)
    -> cost_table<Cap>
{
    return read<Cap>(is);
}

} // namespace ctdp::cost_table_io

#endif // CTDP_SOLVER_COST_TABLE_IO_STREAM_H
