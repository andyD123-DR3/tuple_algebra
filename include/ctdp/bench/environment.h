#ifndef CTDP_BENCH_ENVIRONMENT_H
#define CTDP_BENCH_ENVIRONMENT_H

// ctdp::bench::environment — CPU affinity and priority primitives
//
// Primitives only — no policy. The calibrator layer decides WHICH cpu
// to pin to and WHETHER to elevate priority. This layer provides:
//   pin_to(cpu_id)          — set thread affinity
//   restore_affinity()      — restore original affinity mask
//   cpu_count()             — number of online CPUs
//   environment_guard        — RAII pin + priority elevation
//
// Design: select_pin_cpu() auto-select policy was deliberately removed.
// That belongs in calibration_harness::config, not in the bench layer.

#include <cstddef>
#include <string>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#include <sys/resource.h>
#include <fstream>
#include <sstream>
#endif

namespace ctdp::bench {

// ─── CPU affinity primitives ────────────────────────────────────────

/// Pin the calling thread to a specific CPU core.
/// @return true on success
inline bool pin_to(int cpu_id) noexcept {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask) == 0;
#else
    (void)cpu_id;
    return false;
#endif
}

/// Restore the calling thread's affinity to all online CPUs.
/// @return true on success
inline bool restore_affinity() noexcept {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int n = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    for (int i = 0; i < n; ++i) CPU_SET(i, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask) == 0;
#else
    return false;
#endif
}

/// Number of online CPUs
[[nodiscard]] inline int cpu_count() noexcept {
#ifdef __linux__
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#else
    return 1;
#endif
}

/// Get the current CPU the calling thread is running on
[[nodiscard]] inline int current_cpu() noexcept {
#ifdef __linux__
    return sched_getcpu();
#else
    return 0;
#endif
}

// ─── Priority elevation ─────────────────────────────────────────────

/// Try to set the calling process to high priority (nice -20).
/// @return the previous nice value, or 0 if unavailable
inline int elevate_priority() noexcept {
#ifdef __linux__
    int prev = getpriority(PRIO_PROCESS, 0);
    setpriority(PRIO_PROCESS, 0, -20); // best-effort
    return prev;
#else
    return 0;
#endif
}

/// Restore priority to a previous level
inline void restore_priority(int prev_nice) noexcept {
#ifdef __linux__
    setpriority(PRIO_PROCESS, 0, prev_nice);
#else
    (void)prev_nice;
#endif
}

// ─── LLC size detection ─────────────────────────────────────────────

/// Auto-detect LLC (last-level cache) size in bytes.
/// Falls back to 8 MiB if detection fails.
[[nodiscard]] inline std::size_t detect_llc_bytes() noexcept {
#ifdef __linux__
    // Try sysfs: look for highest-index cache
    for (int idx = 4; idx >= 0; --idx) {
        std::string path = "/sys/devices/system/cpu/cpu0/cache/index"
                         + std::to_string(idx) + "/size";
        std::ifstream f(path);
        if (f.is_open()) {
            std::string line;
            std::getline(f, line);
            if (!line.empty()) {
                std::size_t val = 0;
                std::size_t mult = 1;
                char suffix = line.back();
                if (suffix == 'K' || suffix == 'k') {
                    mult = 1024;
                    line.pop_back();
                } else if (suffix == 'M' || suffix == 'm') {
                    mult = 1024 * 1024;
                    line.pop_back();
                }
                try {
                    val = static_cast<std::size_t>(std::stoul(line)) * mult;
                } catch (...) {
                    continue;
                }
                // LLC is typically > 1 MiB; skip L1/L2 if they're small
                if (val >= 1024 * 1024) return val;
            }
        }
    }
#endif
    return 8u * 1024u * 1024u; // 8 MiB fallback
}

// ─── environment_context ────────────────────────────────────────────

/// Captures the platform and environment at measurement time.
/// Stored alongside data_points for reproducibility.
struct environment_context {
    int         pinned_cpu     = -1;    ///< CPU core used (-1 = not pinned)
    int         nice_level     = 0;     ///< Process nice level during measurement
    std::size_t llc_bytes      = 0;     ///< LLC size used for cache thrashing
    bool        tier1_counters = false; ///< Whether perf_event was available
    std::string hostname;               ///< Machine identifier
    std::string kernel_version;         ///< uname -r
};

/// Populate an environment_context with current system info
[[nodiscard]] inline environment_context capture_environment() {
    environment_context ctx;
    ctx.pinned_cpu = current_cpu();
    ctx.llc_bytes  = detect_llc_bytes();

#ifdef __linux__
    // Hostname
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf)) == 0) ctx.hostname = buf;

    // Kernel version
    std::ifstream f("/proc/version");
    if (f.is_open()) {
        std::getline(f, ctx.kernel_version);
        // Trim to just the version string
        auto pos = ctx.kernel_version.find(' ');
        if (pos != std::string::npos) {
            auto pos2 = ctx.kernel_version.find(' ', pos + 1);
            if (pos2 != std::string::npos) {
                auto pos3 = ctx.kernel_version.find(' ', pos2 + 1);
                ctx.kernel_version = ctx.kernel_version.substr(
                    pos2 + 1, pos3 - pos2 - 1);
            }
        }
    }
#endif

    return ctx;
}

// ─── environment_guard (RAII) ───────────────────────────────────────

/// RAII guard that pins the thread and elevates priority on construction,
/// restores both on destruction.
class environment_guard {
public:
    /// @param cpu_id  CPU to pin to (-1 = don't pin)
    /// @param boost   Whether to elevate process priority
    explicit environment_guard(int cpu_id = -1, bool boost = false)
        : did_pin_{false}, did_boost_{false}, prev_nice_{0}
    {
        if (cpu_id >= 0) {
            did_pin_ = pin_to(cpu_id);
        }
        if (boost) {
            prev_nice_ = elevate_priority();
            did_boost_ = true;
        }
    }

    ~environment_guard() noexcept {
        if (did_pin_)   restore_affinity();
        if (did_boost_) restore_priority(prev_nice_);
    }

    // Non-copyable, non-movable (RAII scope guard)
    environment_guard(environment_guard const&) = delete;
    environment_guard& operator=(environment_guard const&) = delete;
    environment_guard(environment_guard&&) = delete;
    environment_guard& operator=(environment_guard&&) = delete;

    [[nodiscard]] bool pinned() const noexcept { return did_pin_; }
    [[nodiscard]] bool boosted() const noexcept { return did_boost_; }

private:
    bool did_pin_;
    bool did_boost_;
    int  prev_nice_;
};

} // namespace ctdp::bench

#endif // CTDP_BENCH_ENVIRONMENT_H
