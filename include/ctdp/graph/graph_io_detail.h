// graph/graph_io_detail.h — Constexpr parsing primitives
// Part of the compile-time DP library (C++20)
//
// Low-level character-by-character parsing of integers, doubles,
// and whitespace from string_view.  Fully constexpr, no graph
// type dependencies, no iostream.
//
// These utilities are consumed by the graph I/O parsers and runtime
// readers, but have no graph-specific knowledge.  This separation
// enables a future move to core::io without touching graph code.

#ifndef CTDP_GRAPH_IO_DETAIL_H
#define CTDP_GRAPH_IO_DETAIL_H

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ctdp::graph::io {
namespace detail {

/// Skip whitespace (space, tab) but NOT newlines.
constexpr std::size_t skip_hspace(std::string_view sv, std::size_t pos) noexcept {
    while (pos < sv.size() && (sv[pos] == ' ' || sv[pos] == '\t'))
        ++pos;
    return pos;
}

/// Skip to end of line (past '\n' or to end of string).
constexpr std::size_t skip_to_eol(std::string_view sv, std::size_t pos) noexcept {
    while (pos < sv.size() && sv[pos] != '\n')
        ++pos;
    if (pos < sv.size()) ++pos;  // skip the '\n'
    return pos;
}

/// Returns true if pos is at end-of-line or end-of-string.
constexpr bool at_eol(std::string_view sv, std::size_t pos) noexcept {
    return pos >= sv.size() || sv[pos] == '\n' || sv[pos] == '\r' || sv[pos] == '#';
}

/// Parse unsigned integer.  Returns {value, new_pos}.
/// Throws if no digits found.
constexpr std::pair<std::size_t, std::size_t>
parse_uint(std::string_view sv, std::size_t pos) {
    if (pos >= sv.size() || sv[pos] < '0' || sv[pos] > '9') {
        throw std::runtime_error("parse_uint: expected digit");
    }
    std::size_t val = 0;
    while (pos < sv.size() && sv[pos] >= '0' && sv[pos] <= '9') {
        val = val * 10 + static_cast<std::size_t>(sv[pos] - '0');
        ++pos;
    }
    return {val, pos};
}

/// Parse a double: optional '-', digits, optional '.digits'.
/// Returns {value, new_pos}.  Throws if no digits found.
constexpr std::pair<double, std::size_t>
parse_double(std::string_view sv, std::size_t pos) {
    if (pos >= sv.size()) {
        throw std::runtime_error("parse_double: unexpected end");
    }

    double sign = 1.0;
    if (sv[pos] == '-') {
        sign = -1.0;
        ++pos;
    }

    if (pos >= sv.size() || sv[pos] < '0' || sv[pos] > '9') {
        throw std::runtime_error("parse_double: expected digit");
    }

    double integer_part = 0.0;
    while (pos < sv.size() && sv[pos] >= '0' && sv[pos] <= '9') {
        integer_part = integer_part * 10.0 + static_cast<double>(sv[pos] - '0');
        ++pos;
    }

    double frac_part = 0.0;
    if (pos < sv.size() && sv[pos] == '.') {
        ++pos;
        double place = 0.1;
        while (pos < sv.size() && sv[pos] >= '0' && sv[pos] <= '9') {
            frac_part += static_cast<double>(sv[pos] - '0') * place;
            place *= 0.1;
            ++pos;
        }
    }

    return {sign * (integer_part + frac_part), pos};
}

/// Test whether sv at pos starts with the given prefix.
constexpr bool starts_with(std::string_view sv, std::size_t pos,
                           std::string_view prefix) noexcept {
    if (pos + prefix.size() > sv.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (sv[pos + i] != prefix[i]) return false;
    }
    return true;
}

} // namespace detail
} // namespace ctdp::graph::io

#endif // CTDP_GRAPH_IO_DETAIL_H
