#ifndef CTDP_BENCH_ITT_SCOPE_H
#define CTDP_BENCH_ITT_SCOPE_H

// ctdp::bench::itt_scope -- RAII wrapper for Intel ITT pause/resume
//
// When CTDP_VTUNE is defined (CMake option CTDP_VTUNE=ON and ITT SDK found):
//   itt_scope::resume() / pause() bracket the measurement loop so VTune
//   collects only during the region of interest, not warmup or setup.
//
// When CTDP_VTUNE is not defined:
//   All calls compile to no-ops -- zero overhead, no ITT dependency.
//
// Usage:
//   ctdp::bench::itt_scope itt;   // constructed in paused state
//   itt.resume();
//   for (...) { /* measured work */ }
//   itt.pause();
//
// Running under VTune:
//   vtune -collect hardware-event-based-sampling \
//         -knob sampling-mode=hw                  \
//         -- fix_counter_profile.exe
//
// The ITT resume/pause tells VTune exactly which samples to attribute
// to your parse loop, giving clean per-plan counter data.

#ifdef CTDP_VTUNE
#include <ittnotify.h>
#endif

namespace ctdp::bench {

class itt_scope {
public:
    /// Construct in paused state.
    /// If CTDP_VTUNE is not defined this is a no-op.
    itt_scope() noexcept {
#ifdef CTDP_VTUNE
        __itt_pause();
#endif
    }

    ~itt_scope() noexcept {
#ifdef CTDP_VTUNE
        __itt_pause();
#endif
    }

    // Non-copyable
    itt_scope(const itt_scope&)            = delete;
    itt_scope& operator=(const itt_scope&) = delete;

    /// Begin ITT collection.
    void resume() noexcept {
#ifdef CTDP_VTUNE
        __itt_resume();
#endif
    }

    /// Suspend ITT collection.
    void pause() noexcept {
#ifdef CTDP_VTUNE
        __itt_pause();
#endif
    }

    /// Returns true if ITT is active (CTDP_VTUNE was defined at compile time).
    [[nodiscard]] static constexpr bool active() noexcept {
#ifdef CTDP_VTUNE
        return true;
#else
        return false;
#endif
    }
};

} // namespace ctdp::bench

#endif // CTDP_BENCH_ITT_SCOPE_H
