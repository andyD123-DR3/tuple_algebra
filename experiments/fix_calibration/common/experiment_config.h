#ifndef EXPERIMENT_FIX_CALIBRATION_CONFIG_H
#define EXPERIMENT_FIX_CALIBRATION_CONFIG_H

// experiments/fix_calibration/common/experiment_config.h
//
// Shared constants for all experiment programs.
// Fixed seeds ensure determinism: same seed = same configs = same
// training data = same beam winners.

#include <cstddef>
#include <cstdint>
#include <ctdp/calibrator/fix/data_point.h>

namespace ctdp::fix_experiment {

using Strategy = ctdp::calibrator::fix::Strategy;

// -- Random seeds (deterministic config generation) ---------------
inline constexpr std::uint64_t TRAIN_SEED     = 12345;
inline constexpr std::uint64_t VERIFY_SEED    = 67890;
inline constexpr std::uint64_t MSG_POOL_SEED  = 42;    // training messages
inline constexpr std::uint64_t EVAL_POOL_SEED = 99;    // verification messages

// -- Training and search parameters ------------------------------
inline constexpr std::size_t N_TRAIN    = 200;
inline constexpr std::size_t BEAM_WIDTH = 20;
inline constexpr std::size_t N_BEAM     = 20;

// -- Measurement parameters --------------------------------------
inline constexpr std::size_t SAMPLES    = 50000;
inline constexpr std::size_t POOL_SIZE  = 5000;
inline constexpr std::size_t BATCH_SIZE = 64;
inline constexpr std::size_t WARMUP     = 4000;

// -- Consensus data (from beam search analysis) ------------------
// Used by Program F for the exhaustive reduced-space sweep.
// Each entry: (position, fixed strategy).
struct consensus_entry {
    int      position;
    Strategy strategy;
};

inline constexpr consensus_entry CONSENSUS_FIXED[] = {
    { 0, Strategy::Unrolled},   // 3-digit field, 100% U
    { 2, Strategy::Unrolled},   // 4-digit field, 100% U
    { 3, Strategy::SWAR},       // 8-digit field, 100% S
    { 4, Strategy::SWAR},       // 6-digit field, 100% S
    { 5, Strategy::SWAR},       // 4-digit field, 100% S
    {10, Strategy::SWAR},       // 4-digit field,  85% S
    {11, Strategy::SWAR},       // 4-digit field, 100% S
};

inline constexpr int CONSENSUS_UNCERTAIN[] = { 1, 6, 7, 8, 9 };
inline constexpr int N_FIXED     = 7;
inline constexpr int N_UNCERTAIN = 5;
inline constexpr int N_SWEEP     = 1024;  // 4^5

} // namespace ctdp::fix_experiment

#endif // EXPERIMENT_FIX_CALIBRATION_CONFIG_H
