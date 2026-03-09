#!/usr/bin/env bash
# experiments/fix_calibration/scripts/run_all.sh
#
# Run all six experiment programs (A–F) in sequence.
#
# Modes:
#   --mock     Run all programs in mock mode (no RDTSC, fast CI check)
#   --real     Two-phase: Phase 1 (A–E discovery) then Phase 2 (A–E verify + F sweep)
#   --phase1   Phase 1 only: A–E discovery, writes generated headers (F skipped)
#   --phase2   Phase 2 only: A–E verification + F sweep (assumes headers exist)
#
# Note: F is the exhaustive consensus-subspace sweep.  It runs only in
# --mock, --phase2, and --real (after Phase 1).  It is skipped in
# --phase1 because it is not a discovery step.
#
# Usage:
#   cd <repo_root>
#   bash experiments/fix_calibration/scripts/run_all.sh --mock
#   bash experiments/fix_calibration/scripts/run_all.sh --real

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

PROGRAMS_AE="A B C D E"
RESULTS_DIR="$EXPERIMENT_DIR/results"
BUILD_DIR="$REPO_ROOT/build_exp"

usage() {
    echo "Usage: $0 [--mock | --real | --phase1 | --phase2]"
    echo ""
    echo "  --mock     Run all programs with mock measurer (fast, no RDTSC)"
    echo "  --real     Full two-phase: Phase 1 then Phase 2 for A–E, then F"
    echo "  --phase1   Phase 1 only: A–E discovery, writes generated headers"
    echo "  --phase2   Phase 2 only: A–E verification + F sweep"
    exit 1
}

[ $# -ge 1 ] || usage
MODE="$1"

# ── Helpers ───────────────────────────────────────────────────────────

clean_results() {
    # Remove stale results to prevent contamination from previous runs.
    if [ -d "$RESULTS_DIR" ]; then
        echo "  Cleaning stale results in $RESULTS_DIR/"
        rm -f "$RESULTS_DIR"/*.json
    fi
    mkdir -p "$RESULTS_DIR"
}

configure_build() {
    local mock_flag="$1"
    echo "=== Configuring (mock=$mock_flag) ==="
    cmake -B "$BUILD_DIR" \
        -DCTDP_BUILD_EXPERIMENTS=ON \
        -DCTDP_BUILD_TESTS=OFF \
        -DCTDP_BUILD_EXAMPLES=OFF \
        -DCTDP_FIX_EXPERIMENT_MOCK="$mock_flag" \
        "$REPO_ROOT" 2>&1 | tail -3
}

run_target() {
    local target="$1"
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Building + running: $target"
    echo "──────────────────────────────────────────────────"
    cmake --build "$BUILD_DIR" --config Release --target "$target" 2>&1 | tail -3

    # Find the built executable.
    # Use a known set of paths (handles single-config and multi-config generators).
    local exe=""
    local search_base="$BUILD_DIR/experiments/fix_calibration"
    for candidate in \
        "$search_base/$target" \
        "$search_base/$target.exe" \
        "$search_base/Release/$target.exe" \
        "$search_base/Debug/$target.exe" \
        "$search_base/RelWithDebInfo/$target.exe"; do
        if [ -f "$candidate" ]; then
            exe="$candidate"
            break
        fi
    done

    if [ -z "$exe" ]; then
        echo "ERROR: cannot find built executable for $target" >&2
        echo "  Searched: $search_base/{,$target,Release/,Debug/}$target{,.exe}" >&2
        return 1
    fi

    # Run from the experiment directory so results/ is written there.
    (cd "$EXPERIMENT_DIR" && "$exe")
}

check_verify_targets() {
    # Warn if generated headers are missing for Phase 2.
    local missing=0
    for p in $PROGRAMS_AE; do
        local header="$EXPERIMENT_DIR/generated/${p}_candidates.h"
        if [ ! -f "$header" ]; then
            echo "WARNING: Missing generated header: $header" >&2
            missing=$((missing + 1))
        fi
    done
    if [ $missing -gt 0 ]; then
        echo "ERROR: $missing generated headers missing. Run --phase1 first." >&2
        exit 1
    fi
}

run_comparison() {
    if ls "$RESULTS_DIR"/*.json >/dev/null 2>&1; then
        echo ""
        echo "=== Comparison ==="
        if command -v python3 &>/dev/null; then
            python3 "$SCRIPT_DIR/compare_results.py" "$RESULTS_DIR"
        else
            echo "  (python3 not found, skipping comparison)"
        fi
    fi
}

# ── Main ──────────────────────────────────────────────────────────────

case "$MODE" in
    --mock)
        clean_results
        configure_build ON
        for p in $PROGRAMS_AE; do
            run_target "fix_experiment_$p"
        done
        run_target "fix_experiment_F"
        ;;

    --phase1)
        # Phase 1: A–E discovery only.  F is not a discovery step.
        configure_build OFF
        for p in $PROGRAMS_AE; do
            run_target "fix_experiment_$p"
        done
        echo ""
        echo "=== Phase 1 complete for A–E ==="
        echo "Generated headers in: $EXPERIMENT_DIR/generated/"
        echo "Run --phase2 (or --real) for verification + F sweep."
        ;;

    --phase2)
        check_verify_targets
        clean_results
        configure_build OFF
        for p in $PROGRAMS_AE; do
            run_target "fix_experiment_${p}_verify"
        done
        run_target "fix_experiment_F"
        ;;

    --real)
        echo "=== Phase 1: Discovery ==="
        configure_build OFF
        for p in $PROGRAMS_AE; do
            run_target "fix_experiment_$p"
        done

        echo ""
        echo "=== Phase 2: Verification ==="
        check_verify_targets
        clean_results
        # Reconfigure so CMake picks up any new generated headers
        configure_build OFF
        for p in $PROGRAMS_AE; do
            run_target "fix_experiment_${p}_verify"
        done
        run_target "fix_experiment_F"
        ;;

    *)
        usage
        ;;
esac

echo ""
echo "=== All programs complete ==="
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "(no JSON files found)"

run_comparison
