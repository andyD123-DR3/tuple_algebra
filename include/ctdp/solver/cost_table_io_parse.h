// solver/cost_table_io_parse.h — Constexpr cost-table parser
// Part of the compile-time DP library (C++20)
//
// Constexpr parsing of cost tables from string_view text.
// No iostream dependency — constexpr-only consumers pay for nothing.
//
// TEXT FORMAT:
//   # comment
//   elements N              (number of elements / rows)
//   strategies M            (number of strategies / columns)
//   row I  c0 c1 c2 ... c(M-1)   (costs for element I)
//
// Lines starting with # are comments.  Blank lines are ignored.
// Element indices are zero-based unsigned integers.
// Costs are decimal numbers (integer or integer.fraction, optional leading minus).
// Rows may appear in any order; each element index must appear exactly once.

#ifndef CTDP_SOLVER_COST_TABLE_IO_PARSE_H
#define CTDP_SOLVER_COST_TABLE_IO_PARSE_H

#include "cost_table.h"
#include <ctdp/core/io_lexer.h>

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace ctdp::cost_table_io {

// =============================================================================
// Constexpr parsing
// =============================================================================

/// Parse a cost table from a text description (constexpr-safe).
///
/// The text must contain `elements N` and `strategies M` lines before
/// any `row` lines.  Each `row I c0 c1 ...` line supplies M cost values
/// for element I.
///
/// Example:
/// ```cpp
/// constexpr auto ct = cost_table_io::parse<ct_cap::tiny>(
///     "elements 2\n"
///     "strategies 3\n"
///     "row 0 1.0 2.0 3.0\n"
///     "row 1 4.0 5.0 6.0\n"
/// );
/// static_assert(ct.num_elements() == 2);
/// static_assert(ct(0, 1) == 2.0);
/// ```
template<cost_table_capacity Cap = ct_cap::medium>
[[nodiscard]] constexpr auto parse(std::string_view text)
    -> cost_table<Cap>
{
    bool have_elements   = false;
    bool have_strategies = false;
    std::size_t n_elem = 0;
    std::size_t n_strat = 0;
    std::size_t rows_seen = 0;

    // Deferred construction: we build once we know dimensions.
    cost_table<Cap> ct;

    // Track which rows we've seen (for duplicate detection).
    // Use a small fixed array — capacity bounded by Cap::max_elements.
    bool row_seen[Cap::max_elements]{};

    std::size_t pos = 0;

    while (pos < text.size()) {
        pos = io::skip_hspace(text, pos);

        // Skip \r (Windows line endings).
        if (pos < text.size() && text[pos] == '\r') {
            ++pos;
            continue;
        }

        // Blank line or comment.
        if (io::at_eol(text, pos)) {
            pos = io::skip_to_eol(text, pos);
            continue;
        }

        // "elements N"
        if (io::starts_with(text, pos, "elements")) {
            if (have_elements) {
                throw std::runtime_error("cost_table parse: duplicate 'elements' line");
            }
            pos += 8;
            pos = io::skip_hspace(text, pos);
            auto [n, next] = io::parse_uint(text, pos);
            pos = next;
            if (n > Cap::max_elements) {
                throw std::runtime_error("cost_table parse: element count exceeds capacity");
            }
            n_elem = n;
            have_elements = true;

            // If we now have both dimensions, construct the table.
            if (have_strategies) {
                ct = cost_table<Cap>(n_elem, n_strat);
            }

            pos = io::skip_to_eol(text, pos);
            continue;
        }

        // "strategies M"
        if (io::starts_with(text, pos, "strategies")) {
            if (have_strategies) {
                throw std::runtime_error("cost_table parse: duplicate 'strategies' line");
            }
            pos += 10;
            pos = io::skip_hspace(text, pos);
            auto [m, next] = io::parse_uint(text, pos);
            pos = next;
            if (m > Cap::max_strategies) {
                throw std::runtime_error("cost_table parse: strategy count exceeds capacity");
            }
            n_strat = m;
            have_strategies = true;

            if (have_elements) {
                ct = cost_table<Cap>(n_elem, n_strat);
            }

            pos = io::skip_to_eol(text, pos);
            continue;
        }

        // "row I c0 c1 ..."
        if (io::starts_with(text, pos, "row")) {
            if (!have_elements || !have_strategies) {
                throw std::runtime_error("cost_table parse: 'row' before 'elements'/'strategies'");
            }
            pos += 3;
            pos = io::skip_hspace(text, pos);
            auto [idx, p1] = io::parse_uint(text, pos);
            pos = p1;

            if (idx >= n_elem) {
                throw std::runtime_error("cost_table parse: row index out of range");
            }
            if (row_seen[idx]) {
                throw std::runtime_error("cost_table parse: duplicate row index");
            }

            for (std::size_t j = 0; j < n_strat; ++j) {
                pos = io::skip_hspace(text, pos);
                auto [val, p2] = io::parse_double(text, pos);
                pos = p2;
                ct(idx, j) = val;
            }

            row_seen[idx] = true;
            ++rows_seen;

            pos = io::skip_to_eol(text, pos);
            continue;
        }

        // Unknown line — error.
        throw std::runtime_error("cost_table parse: unrecognised line");
    }

    if (!have_elements) {
        throw std::runtime_error("cost_table parse: missing 'elements' line");
    }
    if (!have_strategies) {
        throw std::runtime_error("cost_table parse: missing 'strategies' line");
    }
    if (rows_seen != n_elem) {
        throw std::runtime_error("cost_table parse: incomplete — not all rows provided");
    }

    return ct;
}

} // namespace ctdp::cost_table_io

#endif // CTDP_SOLVER_COST_TABLE_IO_PARSE_H
