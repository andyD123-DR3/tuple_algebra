#pragma once

#include "spmv_design_space/enums.hpp"

#include <cstddef>
#include <sstream>
#include <string>

namespace ctdp::spmv_dsl {

struct hardware_profile {
    std::size_t max_simd_lanes = 8;
    std::size_t cache_line_bytes = 64;
    std::size_t l1_bytes = 32 * 1024;
    std::size_t max_worker_threads = 0; // 0 means use the runtime/search default.
    bool gather_is_expensive = true;
    bool task_runtime_available = true;
    bool allow_parallel_candidates = true;
};

constexpr std::size_t declared_lanes_for(simd_kind simd) noexcept {
    switch (simd) {
    case simd_kind::scalar: return 1;
    case simd_kind::lanes4: return 4;
    case simd_kind::lanes8: return 8;
    }
    return 1;
}

constexpr bool declared_parallel_threading(threading_kind threading) noexcept {
    switch (threading) {
    case threading_kind::serial: return false;
    case threading_kind::static_blocks:
    case threading_kind::recursive_tasks:
    case threading_kind::colour_phases:
        return true;
    }
    return false;
}

inline std::string describe_hardware_profile(const hardware_profile& hardware) {
    std::ostringstream os;
    os << "simd_lanes<= " << hardware.max_simd_lanes
       << ", cache_line=" << hardware.cache_line_bytes << "B"
       << ", l1=" << hardware.l1_bytes << "B"
       << ", workers=";
    if (hardware.max_worker_threads == 0) {
        os << "auto";
    } else {
        os << hardware.max_worker_threads;
    }
    os << ", gather=" << (hardware.gather_is_expensive ? "expensive" : "cheap")
       << ", task_runtime=" << (hardware.task_runtime_available ? "available" : "unavailable")
       << ", parallel_candidates=" << (hardware.allow_parallel_candidates ? "enabled" : "disabled");
    return os.str();
}

} // namespace ctdp::spmv_dsl
