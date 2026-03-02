#ifndef CTDP_CALIBRATOR_H
#define CTDP_CALIBRATOR_H

// ctdp::calibrator — convenience header
//
// Includes the complete calibration pipeline:
//   Scenario → Harness → Dataset → Profile → CostModel →
//   Solver → Plan → DispatchTable → Wisdom
//
// For fine-grained control, include individual headers instead.

// ─── Core types ──────────────────────────────────────────────────
#include <ctdp/calibrator/data_point.h>
#include <ctdp/calibrator/scenario.h>
#include <ctdp/calibrator/provenance.h>

// ─── Measurement ─────────────────────────────────────────────────
#include <ctdp/calibrator/calibration_harness.h>

// ─── Dataset + encoding ──────────────────────────────────────────
#include <ctdp/calibrator/calibration_dataset.h>
#include <ctdp/calibrator/feature_encoder.h>
#include <ctdp/calibrator/sampler.h>
#include <ctdp/calibrator/csv_writer.h>

// ─── Profile fitting ─────────────────────────────────────────────
#include <ctdp/calibrator/calibration_profile.h>
#include <ctdp/calibrator/ct_dp_emit.h>

// ─── Solver ──────────────────────────────────────────────────────
#include <ctdp/calibrator/cost_model.h>
#include <ctdp/calibrator/solver.h>
#include <ctdp/calibrator/plan.h>
#include <ctdp/calibrator/plan_builder.h>

// ─── Code instantiation ─────────────────────────────────────────
#include <ctdp/calibrator/dispatch_table.h>
#include <ctdp/calibrator/plan_emit.h>
#include <ctdp/calibrator/plan_validate.h>
#include <ctdp/calibrator/wisdom.h>

// ─── Optional (requires libbenchmark) ────────────────────────────
// #include <ctdp/calibrator/benchmark_explorer.h>

#endif // CTDP_CALIBRATOR_H
