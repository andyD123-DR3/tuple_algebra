#ifndef CTDP_CALIBRATOR_WISDOM_H
#define CTDP_CALIBRATOR_WISDOM_H

// ctdp::calibrator::wisdom — FFTW-style plan serialisation
//
// Serialise and deserialise plans as human-readable text files.
// The "wisdom" pattern (from FFTW):
//   1. Calibrate + optimise → plan
//   2. Save wisdom to file
//   3. Next build loads wisdom, skips recalibration
//
// Wisdom files are text-based with key=value pairs, designed for:
//   - Version control (human-readable diffs)
//   - Cross-compilation (generate on host, consume on target)
//   - Reproducibility (provenance included)
//
// Format:
//   # ctdp wisdom v1
//   # solver: exhaustive_search
//   # scenario: parser_strategy
//   # entries: 12
//   # optimal_cost: 10.000000
//   ---
//   rank cost_ns pareto [user fields...]
//   0 10.000000 1 <point serialisation>
//   1 12.000000 0 <point serialisation>
//   ...
//
// Usage:
//   // Save
//   write_wisdom(file, plan, serialiser);
//
//   // Load
//   auto entries = read_wisdom(file);  // generic key-value entries
//   // Then reconstruct plan with domain-specific point parser

#include "plan.h"
#include "provenance.h"

#include <cstddef>
#include <cstdio>
#include <functional>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ctdp::calibrator {

// ─── Wisdom file format version ──────────────────────────────────

inline constexpr int wisdom_format_version = 1;

// ─── Wisdom entry (type-erased for loading) ──────────────────────

/// A parsed wisdom entry: rank, cost, Pareto flag, and the raw
/// point serialisation string (to be parsed by domain-specific code).
struct wisdom_entry {
    std::size_t rank           = 0;
    double      cost_ns        = 0.0;
    bool        pareto_optimal = false;
    std::string point_str;        ///< Raw point serialisation
    std::vector<double> objectives;
};

/// Parsed wisdom file metadata.
struct wisdom_metadata {
    int         version       = 0;
    std::string solver;
    std::string scenario;
    std::size_t entry_count   = 0;
    double      optimal_cost  = 0.0;
    std::string hostname;
    std::string compiler;

    /// All header key-value pairs (for extensibility).
    std::vector<std::pair<std::string, std::string>> headers;
};

/// Result of reading a wisdom file.
struct wisdom_file {
    wisdom_metadata              metadata;
    std::vector<wisdom_entry>    entries;

    [[nodiscard]] bool valid() const noexcept {
        return metadata.version == wisdom_format_version
            && !entries.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return entries.size();
    }
};

// ─── Write wisdom ────────────────────────────────────────────────

/// Write a plan as a wisdom file.
///
/// @param out          Output stream
/// @param p            The plan to serialise
/// @param serialiser   Converts point_type → string (one line, no newlines)
///
template <typename Space, typename Callable, typename Serialiser>
void write_wisdom(
    std::ostream& out,
    plan<Space, Callable> const& p,
    Serialiser const& serialiser)
{
    // Header
    out << "# ctdp wisdom v" << wisdom_format_version << "\n";
    out << "# solver: " << p.solver_name << "\n";
    if (!p.provenance.scenario_name.empty()) {
        out << "# scenario: " << p.provenance.scenario_name << "\n";
    }
    if (!p.provenance.hostname.empty()) {
        out << "# hostname: " << p.provenance.hostname << "\n";
    }
    if (!p.provenance.compiler.empty()) {
        out << "# compiler: " << p.provenance.compiler << "\n";
    }
    out << "# entries: " << p.entries.size() << "\n";
    out << "# evaluated: " << p.evaluated_points << "\n";
    if (!p.entries.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", p.entries.front().cost_ns);
        out << "# optimal_cost: " << buf << "\n";
    }
    out << "---\n";

    // Data rows
    for (std::size_t i = 0; i < p.entries.size(); ++i) {
        auto const& e = p.entries[i];
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", e.cost_ns);
        out << i << " " << buf << " " << (e.pareto_optimal ? 1 : 0);

        // Objectives (if more than just cost_ns)
        if (e.objectives.size() > 1) {
            for (std::size_t j = 1; j < e.objectives.size(); ++j) {
                char obuf[64];
                std::snprintf(obuf, sizeof(obuf), " %.6f", e.objectives[j]);
                out << obuf;
            }
        }

        out << " " << serialiser(e.point) << "\n";
    }
}

/// Write wisdom with a lambda serialiser.
template <typename Space, typename Callable, typename Fn>
void write_wisdom_fn(
    std::ostream& out,
    plan<Space, Callable> const& p,
    Fn&& fn)
{
    write_wisdom(out, p, std::forward<Fn>(fn));
}

// ─── Read wisdom ─────────────────────────────────────────────────

/// Read a wisdom file into type-erased entries.
/// The caller must parse point_str with domain-specific logic.
///
[[nodiscard]] inline wisdom_file read_wisdom(std::istream& in) {
    wisdom_file result;
    std::string line;
    bool in_header = true;

    while (std::getline(in, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Header section
        if (in_header) {
            if (line == "---") {
                in_header = false;
                continue;
            }
            if (line.size() >= 2 && line[0] == '#' && line[1] == ' ') {
                auto content = line.substr(2);
                auto colon = content.find(": ");
                if (colon != std::string::npos) {
                    auto key = content.substr(0, colon);
                    auto val = content.substr(colon + 2);
                    result.metadata.headers.emplace_back(key, val);

                    if (key == "ctdp wisdom v") {
                        // Handled below
                    } else if (key == "solver") {
                        result.metadata.solver = val;
                    } else if (key == "scenario") {
                        result.metadata.scenario = val;
                    } else if (key == "hostname") {
                        result.metadata.hostname = val;
                    } else if (key == "compiler") {
                        result.metadata.compiler = val;
                    } else if (key == "entries") {
                        result.metadata.entry_count =
                            static_cast<std::size_t>(std::stoul(val));
                    } else if (key == "optimal_cost") {
                        result.metadata.optimal_cost = std::stod(val);
                    }
                }
                // Check for version line
                if (content.find("ctdp wisdom v") == 0) {
                    auto vstr = content.substr(13);
                    result.metadata.version = std::stoi(vstr);
                }
            }
            continue;
        }

        // Data section: rank cost_ns pareto [extra_objectives...] point_str
        std::istringstream iss(line);
        wisdom_entry entry;
        int pareto_int = 0;
        if (!(iss >> entry.rank >> entry.cost_ns >> pareto_int)) {
            continue;  // skip malformed lines
        }
        entry.pareto_optimal = (pareto_int != 0);
        entry.objectives.push_back(entry.cost_ns);

        // Read rest of line as point_str (everything after the 3 fields)
        // First, consume any extra numeric fields (objectives)
        // Then the remainder is the point serialisation.
        std::string remaining;
        std::getline(iss >> std::ws, remaining);
        entry.point_str = remaining;

        result.entries.push_back(std::move(entry));
    }

    return result;
}

/// Read wisdom from a string.
[[nodiscard]] inline wisdom_file read_wisdom_string(std::string_view text) {
    std::istringstream iss{std::string(text)};
    return read_wisdom(iss);
}

// ─── Reconstruct plan from wisdom ────────────────────────────────

/// Reconstruct a plan from wisdom entries + a point parser.
///
/// @param wf          Parsed wisdom file
/// @param parser      Converts string → point_type (or nullopt on failure)
///
template <typename Space, typename Callable, typename Parser>
[[nodiscard]] auto reconstruct_plan(
    wisdom_file const& wf,
    Parser const& parser)
    -> plan<Space, Callable>
{
    using point_type = typename Space::point_type;

    plan<Space, Callable> result;
    result.solver_name     = wf.metadata.solver;
    result.evaluated_points = wf.entries.size();

    // Reconstruct provenance
    result.provenance.scenario_name = wf.metadata.scenario;
    result.provenance.hostname      = wf.metadata.hostname;
    result.provenance.compiler      = wf.metadata.compiler;

    result.entries.reserve(wf.entries.size());
    for (auto const& we : wf.entries) {
        auto pt_opt = parser(we.point_str);
        if (!pt_opt) continue;  // skip unparseable entries

        result.entries.push_back(plan_entry<point_type>{
            .point          = *pt_opt,
            .cost_ns        = we.cost_ns,
            .objectives     = we.objectives,
            .pareto_optimal = we.pareto_optimal
        });
    }

    return result;
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_WISDOM_H
