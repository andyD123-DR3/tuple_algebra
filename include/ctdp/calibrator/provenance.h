#ifndef CTDP_CALIBRATOR_PROVENANCE_H
#define CTDP_CALIBRATOR_PROVENANCE_H

// ctdp::calibrator::provenance — Platform fingerprint + schema IDs
//
// Design v2.2 §5.6:
//   Platform fingerprint, build ID, schema versioning.
//   Captures conditions under which calibration was performed so that
//   profiles can be matched to the hardware they were measured on.
//
// dataset_provenance carries:
//   - platform identity (CPU model, microarchitecture, cache hierarchy)
//   - OS/kernel info
//   - compiler info (for reproducibility)
//   - calibration timestamp
//   - schema version (for forward compatibility of stored datasets)
//
// The provenance is attached to calibration_dataset and calibration_profile.

#include <ctdp/bench/environment.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <string>

namespace ctdp::calibrator {

/// Schema version for serialised datasets and profiles.
/// Increment when the binary/CSV format changes.
struct schema_version {
    std::uint16_t major = 1;
    std::uint16_t minor = 0;

    [[nodiscard]] bool compatible_with(schema_version const& other) const noexcept {
        return major == other.major;  // minor changes are backward-compatible
    }

    [[nodiscard]] auto to_string() const -> std::string {
        return std::to_string(major) + "." + std::to_string(minor);
    }

    friend bool operator==(schema_version const&, schema_version const&) = default;
};

/// Current schema version for this build.
inline constexpr schema_version current_schema{1, 0};

/// Full provenance record for a calibration run.
///
/// Captures everything needed to determine whether a stored dataset
/// or profile was produced under compatible conditions.
struct dataset_provenance {
    // ─── Platform identity ───────────────────────────────────────
    std::string cpu_model;          ///< e.g. "Intel i9-13900K"
    std::string microarchitecture;  ///< e.g. "Raptor Lake", "Zen 4"
    std::size_t l1d_bytes   = 0;   ///< L1 data cache per core
    std::size_t l2_bytes    = 0;   ///< L2 cache per core
    std::size_t l3_bytes    = 0;   ///< L3 (LLC) total
    std::size_t num_cores   = 0;   ///< Number of online CPUs

    // ─── OS/kernel ───────────────────────────────────────────────
    std::string hostname;
    std::string kernel_version;
    std::string os_name;            ///< e.g. "Linux", "Windows"

    // ─── Build info ──────────────────────────────────────────────
    std::string compiler;           ///< e.g. "GCC 14.1.0"
    std::string build_flags;        ///< e.g. "-O2 -march=native"

    // ─── Calibration run ─────────────────────────────────────────
    std::string timestamp;          ///< ISO 8601 UTC
    std::string scenario_name;      ///< From Scenario::name()
    std::size_t total_points = 0;   ///< Number of space points measured
    std::size_t reps_per_point = 0; ///< Repetitions per point

    // ─── Schema ──────────────────────────────────────────────────
    schema_version schema = current_schema;

    // ─── Counter tier ────────────────────────────────────────────
    bool tier1_counters = false;    ///< Whether perf_event_open was available
};

/// Capture provenance from the current environment + scenario metadata.
///
/// Extends bench::capture_environment() with additional platform detection
/// and schema tagging.
template <typename Scenario>
[[nodiscard]] inline dataset_provenance capture_provenance(
    Scenario const& scenario,
    bench::environment_context const& env,
    std::size_t reps)
{
    dataset_provenance prov;

    // From environment_context
    prov.hostname       = env.hostname;
    prov.kernel_version = env.kernel_version;
    prov.l3_bytes       = env.llc_bytes;
    prov.tier1_counters = env.tier1_counters;
    prov.num_cores      = static_cast<std::size_t>(bench::cpu_count());

    // CPU model from /proc/cpuinfo (Linux)
#ifdef __linux__
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    prov.cpu_model = line.substr(pos + 2);
                }
                break;
            }
        }
    }
    prov.os_name = "Linux";
#elif defined(_WIN32)
    prov.os_name = "Windows";
#elif defined(__APPLE__)
    prov.os_name = "macOS";
#else
    prov.os_name = "Unknown";
#endif

    // Compiler
#if defined(__GNUC__) && !defined(__clang__)
    prov.compiler = "GCC " + std::to_string(__GNUC__) + "."
                  + std::to_string(__GNUC_MINOR__) + "."
                  + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(__clang__)
    prov.compiler = "Clang " + std::to_string(__clang_major__) + "."
                  + std::to_string(__clang_minor__) + "."
                  + std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
    prov.compiler = "MSVC " + std::to_string(_MSC_VER);
#else
    prov.compiler = "Unknown";
#endif

    // Timestamp (UTC, ISO 8601)
    {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        char buf[64] = {};
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ",
                      std::gmtime(&time_t));
        prov.timestamp = buf;
    }

    // Scenario metadata
    prov.scenario_name  = std::string(scenario.name());
    prov.reps_per_point = reps;

    prov.schema = current_schema;

    return prov;
}

/// Write provenance as CSV comment header (lines starting with #).
inline void write_provenance_header(std::ostream& os,
                                     dataset_provenance const& prov)
{
    os << "# ctdp-calibrator dataset v" << prov.schema.to_string() << "\n";
    os << "# scenario: " << prov.scenario_name << "\n";
    os << "# timestamp: " << prov.timestamp << "\n";
    os << "# cpu: " << prov.cpu_model << "\n";
    os << "# os: " << prov.os_name << " " << prov.kernel_version << "\n";
    os << "# compiler: " << prov.compiler << "\n";
    os << "# hostname: " << prov.hostname << "\n";
    os << "# llc_bytes: " << prov.l3_bytes << "\n";
    os << "# cores: " << prov.num_cores << "\n";
    os << "# tier1_counters: " << (prov.tier1_counters ? "yes" : "no") << "\n";
    os << "# reps_per_point: " << prov.reps_per_point << "\n";
    os << "# total_points: " << prov.total_points << "\n";
}

} // namespace ctdp::calibrator

#endif // CTDP_CALIBRATOR_PROVENANCE_H
