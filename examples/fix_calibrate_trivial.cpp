// fix_calibrate_trivial.cpp — Phase 3 live calibration runner
//
// Measures all 64 Trivial-schema (N=4) plans on this machine via the
// A1/A2/B protocol, populates CostTable<4,double>, runs exhaustive_search
// and beam_dp<4>, and prints the ranked results.
//
// HARDWARE TOOL — do not run on CI / shared build machines.
// For best results pin to a quiet core:
//   Linux:   taskset -c 2 ./fix_calibrate_trivial
//   Windows: start /affinity 4 fix_calibrate_trivial.exe

#include <ctdp/calibrator/fix/fix_calibration_driver.h>
#include <cstdio>

int main() {
    using namespace ctdp::calibrator::fix;

    CalibrationConfig cal_cfg;
    cal_cfg.message_pool_size = 1024;
    cal_cfg.beam_width        = 4;
    cal_cfg.verbose           = true;

    BenchConfig bench_cfg;
    bench_cfg.flush_before_b = true;
    bench_cfg.use_fitted_p99 = true;

    auto result = run_trivial_calibration(cal_cfg, bench_cfg);

    result.print_summary();
    result.print_top_n(10);

    return result.exhaustive.found() ? 0 : 1;
}
