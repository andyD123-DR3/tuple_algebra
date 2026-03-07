#ifndef CTDP_BENCH_PERF_COUNTER_H
#define CTDP_BENCH_PERF_COUNTER_H

// ctdp::bench::perf_counter -- Hardware performance counter abstraction
//
// Three tiers:
//   Tier 0: Always available -- RDTSC cycles + thread CPU time
//   Tier 1: Linux perf_event_open -- instructions, cache refs/misses,
//           branches, branch misses
//           Windows: QueryThreadCycleTime (thread cycles) +
//                    GetThreadTimes (kernel/user CPU time)
//   Tier 2: Linux L1D/L1I/DTLB miss rates (separate counter group)
//
// Windows Tier 1 notes:
//   QueryThreadCycleTime counts CPU cycles consumed by this thread only
//   (excludes time the thread is descheduled). More stable than wall-clock
//   RDTSC for latency measurement on a loaded system.
//   Instructions and cache counters require VTune or a kernel driver on
//   Windows -- not available here. IPC column will show 0.
//
// counter_snapshot: raw counters + derived ratios
// perf_counter_group: RAII group managing OS counter handles

#include <cstdint>
#include <cstring>
#include <string>
#include <array>

#if defined(_MSC_VER) || defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>   // __rdtsc
#endif

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#endif

#ifdef __x86_64__
#include <x86intrin.h>
#endif

namespace ctdp::bench {

// --- counter_snapshot -----------------------------------------------

/// Raw counters harvested from a single measurement interval.
struct counter_snapshot {
    // Tier 0 -- always available
    std::uint64_t tsc_cycles       = 0;   ///< RDTSC delta
    double        thread_cpu_ns    = 0.0; ///< CLOCK_THREAD_CPUTIME_NS delta

    // Tier 1 -- Linux perf_event (0 if unavailable)
    std::uint64_t instructions     = 0;
    std::uint64_t cache_references = 0;
    std::uint64_t cache_misses     = 0;
    std::uint64_t branches         = 0;
    std::uint64_t branch_misses    = 0;

    // Tier 2 -- Cache hierarchy (separate counter group, 0 if unavailable)
    std::uint64_t l1d_read_access  = 0;
    std::uint64_t l1d_read_miss    = 0;
    std::uint64_t l1i_read_access  = 0;   ///< icache
    std::uint64_t l1i_read_miss    = 0;   ///< icache misses
    std::uint64_t dtlb_read_access = 0;
    std::uint64_t dtlb_read_miss   = 0;

    // Derived ratios -- computed after harvesting via compute_derived()
    double ipc             = 0.0;   ///< instructions / tsc_cycles
    double cache_miss_rate = 0.0;   ///< cache_misses / cache_references
    double branch_miss_rate= 0.0;   ///< branch_misses / branches
    double cycles_per_ns   = 0.0;   ///< tsc_cycles / wall_ns
    double instructions_per_ns = 0.0;
    double cache_refs_per_ns   = 0.0;
    double branches_per_ns     = 0.0;

    // Tier 2 derived
    double l1d_miss_rate   = 0.0;   ///< l1d_read_miss / l1d_read_access
    double l1i_miss_rate   = 0.0;   ///< l1i_read_miss / l1i_read_access
    double dtlb_miss_rate  = 0.0;   ///< dtlb_read_miss / dtlb_read_access

    /// Compute derived ratios from raw counters.
    /// @param wall_ns wall-clock nanoseconds for the measurement interval
    void compute_derived(double wall_ns) noexcept {
        ipc = (tsc_cycles > 0)
            ? static_cast<double>(instructions) / static_cast<double>(tsc_cycles)
            : 0.0;
        cache_miss_rate = (cache_references > 0)
            ? static_cast<double>(cache_misses) / static_cast<double>(cache_references)
            : 0.0;
        branch_miss_rate = (branches > 0)
            ? static_cast<double>(branch_misses) / static_cast<double>(branches)
            : 0.0;

        if (wall_ns > 0.0) {
            cycles_per_ns       = static_cast<double>(tsc_cycles) / wall_ns;
            instructions_per_ns = static_cast<double>(instructions) / wall_ns;
            cache_refs_per_ns   = static_cast<double>(cache_references) / wall_ns;
            branches_per_ns     = static_cast<double>(branches) / wall_ns;
        }

        // Tier 2 derived
        l1d_miss_rate = (l1d_read_access > 0)
            ? static_cast<double>(l1d_read_miss) / static_cast<double>(l1d_read_access)
            : 0.0;
        l1i_miss_rate = (l1i_read_access > 0)
            ? static_cast<double>(l1i_read_miss) / static_cast<double>(l1i_read_access)
            : 0.0;
        dtlb_miss_rate = (dtlb_read_access > 0)
            ? static_cast<double>(dtlb_read_miss) / static_cast<double>(dtlb_read_access)
            : 0.0;
    }
};

// --- RDTSC ----------------------------------------------------------

/// Read Time Stamp Counter -- serialising variant
[[nodiscard]] inline std::uint64_t rdtsc_start() noexcept {
#ifdef __x86_64__
    unsigned int aux;
    return __rdtscp(&aux);
#elif defined(_M_X64) || defined(_M_AMD64)
    return __rdtsc();
#else
    return 0;
#endif
}

/// Read Time Stamp Counter -- post-measurement (no serialisation needed)
[[nodiscard]] inline std::uint64_t rdtsc_end() noexcept {
#ifdef __x86_64__
    unsigned int aux;
    return __rdtscp(&aux);
#elif defined(_M_X64) || defined(_M_AMD64)
    return __rdtsc();
#else
    return 0;
#endif
}

// --- Thread CPU time ------------------------------------------------

/// Get thread CPU time in nanoseconds.
/// Linux: CLOCK_THREAD_CPUTIME_ID (nanosecond resolution).
/// Windows: GetThreadTimes (100ns resolution, kernel+user).
[[nodiscard]] inline double thread_cpu_time_ns() noexcept {
#ifdef __linux__
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
#elif defined(_WIN32)
    FILETIME creation, exit, kernel, user;
    if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) {
        // FILETIME is in 100ns units
        ULARGE_INTEGER k, u;
        k.LowPart  = kernel.dwLowDateTime;  k.HighPart = kernel.dwHighDateTime;
        u.LowPart  = user.dwLowDateTime;    u.HighPart = user.dwHighDateTime;
        return static_cast<double>(k.QuadPart + u.QuadPart) * 100.0;
    }
    return 0.0;
#else
    return 0.0;
#endif
}

/// Get thread cycle count (Windows only: QueryThreadCycleTime).
/// More stable than wall-clock RDTSC on a loaded system because it
/// excludes cycles consumed while the thread is descheduled.
/// Returns 0 on non-Windows platforms (use RDTSC there).
[[nodiscard]] inline std::uint64_t thread_cycle_count() noexcept {
#if defined(_WIN32)
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return static_cast<std::uint64_t>(cycles);
#else
    return 0;
#endif
}

// --- perf_counter_group ---------------------------------------------

/// RAII group for Linux perf_event counters.
/// Tier 0: always works (RDTSC + thread CPU).
/// Tier 1: requires perf_event_open (may need CAP_SYS_ADMIN or
///          perf_event_paranoid <= 1).
class perf_counter_group {
public:
    perf_counter_group() noexcept {
#ifdef __linux__
        open_counters();
#endif
    }

    ~perf_counter_group() noexcept {
#ifdef __linux__
        close_counters();
#endif
    }

    // Non-copyable, movable
    perf_counter_group(perf_counter_group const&) = delete;
    perf_counter_group& operator=(perf_counter_group const&) = delete;
    perf_counter_group(perf_counter_group&&) noexcept = default;
    perf_counter_group& operator=(perf_counter_group&&) noexcept = default;

    /// Whether Tier 1 (perf_event) counters are available
    [[nodiscard]] bool tier1_available() const noexcept { return tier1_ok_; }

    /// Start counting
    void start() noexcept {
        // Tier 0
        start_cpu_ns_ = thread_cpu_time_ns();
        start_tsc_    = rdtsc_start();
#if defined(_WIN32)
        start_thread_cycles_ = thread_cycle_count();
#endif
#ifdef __linux__
        if (tier1_ok_) {
            // Reset and enable the group leader
            ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
            ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        }
#endif
    }

    /// Stop counting
    void stop() noexcept {
        // Tier 0
        end_tsc_    = rdtsc_end();
        end_cpu_ns_ = thread_cpu_time_ns();
#if defined(_WIN32)
        end_thread_cycles_ = thread_cycle_count();
#endif
#ifdef __linux__
        if (tier1_ok_) {
            ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            read_counters();
        }
#endif
    }

    /// Harvest a snapshot of the last start/stop interval
    [[nodiscard]] counter_snapshot snapshot() const noexcept {
        counter_snapshot snap;
        snap.tsc_cycles    = end_tsc_ - start_tsc_;
        snap.thread_cpu_ns = end_cpu_ns_ - start_cpu_ns_;

#if defined(_WIN32)
        // Windows Tier 1: thread cycles (excludes descheduled time)
        // Use as tsc_cycles override — more accurate for per-thread latency.
        if (end_thread_cycles_ > start_thread_cycles_) {
            snap.tsc_cycles = end_thread_cycles_ - start_thread_cycles_;
        }
        // tier1_ok_ stays false — instructions/cache not available
#endif

        if (tier1_ok_) {
            snap.instructions     = values_[0];
            snap.cache_references = values_[1];
            snap.cache_misses     = values_[2];
            snap.branches         = values_[3];
            snap.branch_misses    = values_[4];
        }

        return snap;
    }

private:
    // Tier 0 state
    std::uint64_t start_tsc_ = 0, end_tsc_ = 0;
    double start_cpu_ns_ = 0.0, end_cpu_ns_ = 0.0;

#if defined(_WIN32)
    std::uint64_t start_thread_cycles_ = 0, end_thread_cycles_ = 0;
#endif

    // Tier 1 state
    bool tier1_ok_ = false;

    static constexpr int kNumCounters = 5;
    std::array<int, kNumCounters> fds_{};
    std::array<std::uint64_t, kNumCounters> values_{};

#ifdef __linux__
    static long perf_event_open(struct perf_event_attr* attr, pid_t pid,
                                int cpu, int group_fd, unsigned long flags) {
        return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
    }

    void open_counters() noexcept {
        fds_.fill(-1);
        values_.fill(0);

        struct perf_event_attr attrs[kNumCounters];
        std::memset(attrs, 0, sizeof(attrs));

        // Instructions
        attrs[0].type           = PERF_TYPE_HARDWARE;
        attrs[0].config         = PERF_COUNT_HW_INSTRUCTIONS;
        attrs[0].size           = sizeof(struct perf_event_attr);
        attrs[0].disabled       = 1;
        attrs[0].exclude_kernel = 1;
        attrs[0].exclude_hv     = 1;

        // Cache references
        attrs[1].type           = PERF_TYPE_HARDWARE;
        attrs[1].config         = PERF_COUNT_HW_CACHE_REFERENCES;
        attrs[1].size           = sizeof(struct perf_event_attr);
        attrs[1].exclude_kernel = 1;
        attrs[1].exclude_hv     = 1;

        // Cache misses
        attrs[2].type           = PERF_TYPE_HARDWARE;
        attrs[2].config         = PERF_COUNT_HW_CACHE_MISSES;
        attrs[2].size           = sizeof(struct perf_event_attr);
        attrs[2].exclude_kernel = 1;
        attrs[2].exclude_hv     = 1;

        // Branches
        attrs[3].type           = PERF_TYPE_HARDWARE;
        attrs[3].config         = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
        attrs[3].size           = sizeof(struct perf_event_attr);
        attrs[3].exclude_kernel = 1;
        attrs[3].exclude_hv     = 1;

        // Branch misses
        attrs[4].type           = PERF_TYPE_HARDWARE;
        attrs[4].config         = PERF_COUNT_HW_BRANCH_MISSES;
        attrs[4].size           = sizeof(struct perf_event_attr);
        attrs[4].exclude_kernel = 1;
        attrs[4].exclude_hv     = 1;

        // Open group leader
        fds_[0] = static_cast<int>(
            perf_event_open(&attrs[0], 0, -1, -1, 0));
        if (fds_[0] < 0) {
            tier1_ok_ = false;
            return;
        }

        // Open remaining counters in the group
        for (int i = 1; i < kNumCounters; ++i) {
            fds_[i] = static_cast<int>(
                perf_event_open(&attrs[i], 0, -1, fds_[0], 0));
            if (fds_[i] < 0) {
                close_counters();
                tier1_ok_ = false;
                return;
            }
        }

        tier1_ok_ = true;
    }

    void close_counters() noexcept {
        for (auto& fd : fds_) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
        }
        tier1_ok_ = false;
    }

    void read_counters() noexcept {
        for (int i = 0; i < kNumCounters; ++i) {
            std::uint64_t val = 0;
            if (fds_[i] >= 0) {
                auto bytes = read(fds_[i], &val, sizeof(val));
                (void)bytes;
            }
            values_[static_cast<std::size_t>(i)] = val;
        }
    }
#else
    void open_counters() noexcept { tier1_ok_ = false; }
    void close_counters() noexcept {}
    void read_counters() noexcept {}
#endif
};

} // namespace ctdp::bench

// =====================================================================
// Tier 2: Cache hierarchy counters (L1D, icache, DTLB)
// =====================================================================
//
// Separate group from Tier 1 because hardware typically supports only
// 4 general-purpose PMU counters. Run as a second measurement pass.
//
// Events (all user-mode only):
//   L1D read accesses, L1D read misses
//   L1I read accesses, L1I read misses  (icache)
//   DTLB read accesses, DTLB read misses
//
// These use PERF_TYPE_HW_CACHE with the packed config:
//   config = cache_id | (op << 8) | (result << 16)

namespace ctdp::bench {

class cache_hierarchy_group {
public:
    cache_hierarchy_group() noexcept {
#ifdef __linux__
        open_counters();
#endif
    }

    ~cache_hierarchy_group() noexcept {
#ifdef __linux__
        close_counters();
#endif
    }

    cache_hierarchy_group(cache_hierarchy_group const&) = delete;
    cache_hierarchy_group& operator=(cache_hierarchy_group const&) = delete;
    cache_hierarchy_group(cache_hierarchy_group&&) noexcept = default;
    cache_hierarchy_group& operator=(cache_hierarchy_group&&) noexcept = default;

    [[nodiscard]] bool available() const noexcept { return ok_; }

    void start() noexcept {
#ifdef __linux__
        if (ok_) {
            ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
            ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        }
#endif
    }

    void stop() noexcept {
#ifdef __linux__
        if (ok_) {
            ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            read_counters();
        }
#endif
    }

    /// Fill the Tier 2 fields of an existing counter_snapshot.
    void fill_snapshot(counter_snapshot& snap) const noexcept {
        if (ok_) {
            snap.l1d_read_access  = values_[0];
            snap.l1d_read_miss    = values_[1];
            snap.l1i_read_access  = values_[2];
            snap.l1i_read_miss    = values_[3];
            snap.dtlb_read_access = values_[4];
            snap.dtlb_read_miss   = values_[5];
        }
    }

private:
    bool ok_ = false;
    static constexpr int kNumCounters = 6;
    std::array<int, kNumCounters> fds_{};
    std::array<std::uint64_t, kNumCounters> values_{};

#ifdef __linux__
    static long perf_event_open(struct perf_event_attr* attr, pid_t pid,
                                int cpu, int group_fd, unsigned long flags) {
        return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
    }

    static constexpr std::uint64_t hw_cache_config(
            std::uint64_t cache_id,
            std::uint64_t op,
            std::uint64_t result) noexcept {
        return cache_id | (op << 8) | (result << 16);
    }

    void open_counters() noexcept {
        fds_.fill(-1);
        values_.fill(0);

        // Event configs: {cache_id, op, result}
        struct cache_event {
            std::uint64_t cache_id;
            std::uint64_t op;
            std::uint64_t result;
        };

        constexpr cache_event events[kNumCounters] = {
            {PERF_COUNT_HW_CACHE_L1D,  PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS},
            {PERF_COUNT_HW_CACHE_L1D,  PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS},
            {PERF_COUNT_HW_CACHE_L1I,  PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS},
            {PERF_COUNT_HW_CACHE_L1I,  PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS},
            {PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS},
            {PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS},
        };

        for (int i = 0; i < kNumCounters; ++i) {
            struct perf_event_attr attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.type           = PERF_TYPE_HW_CACHE;
            attr.config         = hw_cache_config(
                events[i].cache_id, events[i].op, events[i].result);
            attr.size           = sizeof(struct perf_event_attr);
            attr.exclude_kernel = 1;
            attr.exclude_hv     = 1;

            if (i == 0) {
                attr.disabled = 1;
                fds_[0] = static_cast<int>(
                    perf_event_open(&attr, 0, -1, -1, 0));
            } else {
                fds_[i] = static_cast<int>(
                    perf_event_open(&attr, 0, -1, fds_[0], 0));
            }

            if (fds_[i] < 0) {
                close_counters();
                ok_ = false;
                return;
            }
        }
        ok_ = true;
    }

    void close_counters() noexcept {
        for (auto& fd : fds_) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
        }
        ok_ = false;
    }

    void read_counters() noexcept {
        for (int i = 0; i < kNumCounters; ++i) {
            std::uint64_t val = 0;
            if (fds_[i] >= 0) {
                auto bytes = read(fds_[i], &val, sizeof(val));
                (void)bytes;
            }
            values_[static_cast<std::size_t>(i)] = val;
        }
    }
#else
    void open_counters() noexcept { ok_ = false; }
    void close_counters() noexcept {}
    void read_counters() noexcept {}
#endif
};

} // namespace ctdp::bench

#endif // CTDP_BENCH_PERF_COUNTER_H
