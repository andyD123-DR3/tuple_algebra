#ifndef CTDP_FIX_EXPERIMENT_RDTSC_ADAPTER_H
#define CTDP_FIX_EXPERIMENT_RDTSC_ADAPTER_H

// experiments/fix_calibration/common/rdtsc_adapter.h
//
// MeasureAdapter for real RDTSC measurement through the ET parser.
// Satisfies the compiled_measurer policy:
//
//   template<fix::fix_config Cfg>
//   measurement_result measure_one() const;
//
// Wraps measure_config_batched<Cfg>() → measurement_result{p50, p99}.
// Shared across all experiment programs A–F.

#include "baselines.h"
#include "experiment_config.h"

#include <ctdp/calibrator/fix_et_parser.h>

#include <string>
#include <vector>

namespace ctdp::fix_experiment {

namespace fix = ctdp::calibrator::fix;

/// MeasureAdapter: wraps RDTSC measurement for compiled_measurer dispatch.
struct rdtsc_adapter {
    std::vector<std::string> const& messages;
    fix::measurement_config         mconfig;

    template<fix::fix_config Cfg>
    measurement_result measure_one() const {
        auto pctl = fix::measure_config_batched<Cfg>(messages, mconfig);
        return measurement_result{pctl.p50, pctl.p99};
    }
};

/// Create a default measurement config from experiment constants.
inline fix::measurement_config default_measurement_config() {
    fix::measurement_config cfg;
    cfg.samples       = SAMPLES;
    cfg.batch_size    = BATCH_SIZE;
    cfg.warmup_parses = WARMUP;
    return cfg;
}

} // namespace ctdp::fix_experiment

#endif // CTDP_FIX_EXPERIMENT_RDTSC_ADAPTER_H
