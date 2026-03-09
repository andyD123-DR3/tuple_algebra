#!/usr/bin/env python3
"""
experiments/fix_calibration/scripts/compare_results.py

Compare experiment results across Programs A–F.

Reads results/*.json and produces:
  1. Summary table: best candidate vs baselines per program
  2. Best candidate per program with config and tail ratio
  3. Model quality comparison (A–E surrogate programs only)
  4. Baseline measurement consistency check with tolerance
  5. Cross-program candidate overlap analysis

Usage:
    python3 compare_results.py results/
    python3 compare_results.py results/ --csv    # machine-readable output
"""

import csv
import io
import json
import math
import os
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────

def is_exhaustive(report: dict) -> bool:
    """Robustly detect exhaustive mode (handles bool, string, missing)."""
    v = report.get("exhaustive", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return False


def load_results(results_dir: str) -> dict:
    """Load all program JSON files from the results directory.

    Fails on duplicate program_ids rather than silently overwriting.
    """
    reports = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        pid = data.get("program_id", path.stem)
        if pid in reports:
            print(f"ERROR: duplicate program_id '{pid}' in "
                  f"{path} and earlier file", file=sys.stderr)
            sys.exit(1)
        reports[pid] = data
    return reports


def get_best_candidate(report: dict) -> dict:
    """Return the best candidate by measured p50, or empty dict."""
    candidates = report.get("candidates", [])
    best_idx = report.get("best_measured_index", 0)
    if candidates and best_idx < len(candidates):
        return candidates[best_idx]
    return {}


def get_best_baseline(report: dict) -> tuple:
    """Return (name, p50) of the best baseline, or ("", inf)."""
    best_p50 = float("inf")
    best_name = ""
    for bl in report.get("baselines", []):
        p50 = bl.get("measured_p50_ns", float("inf"))
        if p50 < best_p50:
            best_p50 = p50
            best_name = bl.get("name", "")
    return best_name, best_p50


# ── Output sections ──────────────────────────────────────────────────

def print_separator(width=78):
    print("=" * width)


def print_summary_table(reports: dict):
    """Print summary: best candidate p50/p99 vs best baseline per program."""
    print()
    print_separator()
    print("  EXPERIMENT SUMMARY")
    print_separator()
    print()
    fmt = "  {:<4s} {:<15s} {:<14s} {:>4s} {:>7s} {:>7s} {:>8s} {:>8s}"
    print(fmt.format(
        "Prog", "Target", "Extractor", "Feat",
        "CV R²", "RMSE", "Best p50", "Best p99"))
    print("  " + "-" * 74)

    for pid in sorted(reports.keys()):
        r = reports[pid]
        exhaustive = is_exhaustive(r)
        model = r.get("model", {})
        cv_r2 = model.get("cv_r2", float("nan"))
        cv_rmse = model.get("cv_rmse", 0.0)
        n_features = model.get("n_features", 0)

        best_p50 = r.get("best_p50_ns", 0.0)
        best_p99 = r.get("best_p99_ns", 0.0)

        target = r.get("target_description", "")[:15]
        extractor = r.get("extractor_name", "")[:14]

        r2_str = "  N/A" if exhaustive else f"{cv_r2:7.4f}"
        rmse_str = "  N/A" if exhaustive else f"{cv_rmse:7.4f}"
        feat_str = "  —" if exhaustive else f"{n_features:4d}"
        print(f"  {pid:<4s} {target:<15s} {extractor:<14s} "
              f"{feat_str} {r2_str} {rmse_str} "
              f"{best_p50:8.2f} {best_p99:8.2f}")
    print()


def write_csv(reports: dict, out=sys.stdout):
    """Write machine-readable CSV (no preamble, proper escaping)."""
    writer = csv.writer(out)
    writer.writerow([
        "program", "target", "extractor", "features", "cv_r2", "cv_rmse",
        "best_p50", "best_p99", "best_config",
        "best_baseline_p50", "best_baseline_name", "speedup"])

    for pid in sorted(reports.keys()):
        r = reports[pid]
        exhaustive = is_exhaustive(r)
        model = r.get("model", {})
        cv_r2 = model.get("cv_r2", float("nan"))
        cv_rmse = model.get("cv_rmse", 0.0)
        n_features = model.get("n_features", 0)

        best_p50 = r.get("best_p50_ns", 0.0)
        best_p99 = r.get("best_p99_ns", 0.0)

        best_cand = get_best_candidate(r)
        best_config = best_cand.get("config_label", "")

        bl_name, bl_p50 = get_best_baseline(r)
        speedup = bl_p50 / best_p50 if best_p50 > 0 else 0

        writer.writerow([
            pid,
            r.get("target_description", ""),
            r.get("extractor_name", ""),
            n_features if not exhaustive else 0,
            f"{cv_r2:.4f}" if not math.isnan(cv_r2) else "nan",
            f"{cv_rmse:.4f}",
            f"{best_p50:.2f}",
            f"{best_p99:.2f}",
            best_config,
            f"{bl_p50:.2f}",
            bl_name,
            f"{speedup:.3f}"])


def print_winner_table(reports: dict):
    """Show the best config from each program side by side."""
    print_separator()
    print("  BEST CANDIDATE PER PROGRAM")
    print_separator()
    print()
    fmt = "  {:<4s} {:>8s} {:>8s} {:>6s}  {:<14s} {:<s}"
    print(fmt.format("Prog", "p50 (ns)", "p99 (ns)", "Tail", "Config", "Target"))
    print("  " + "-" * 66)

    for pid in sorted(reports.keys()):
        r = reports[pid]
        best = get_best_candidate(r)
        if not best:
            continue
        p50 = best.get("measured_p50_ns", 0)
        p99 = best.get("measured_p99_ns", 0)
        tail = p99 / p50 if p50 > 0 else 0
        config = best.get("config_label", "")
        target = r.get("target_description", "")
        print(f"  {pid:<4s} {p50:8.2f} {p99:8.2f} {tail:6.3f}  {config:<14s} {target}")
    print()


def print_model_comparison(reports: dict):
    """Compare model quality across surrogate programs (A–E)."""
    surrogate = {pid: r for pid, r in reports.items()
                 if not is_exhaustive(r)}
    if not surrogate:
        return

    print_separator()
    print("  MODEL QUALITY (surrogate programs only)")
    print_separator()
    print()
    fmt = "  {:<4s} {:>7s} {:>7s} {:>4s} {:>8s} {:>14s}"
    print(fmt.format("Prog", "CV R²", "RMSE", "Feat", "Training", "Extractor"))
    print("  " + "-" * 52)

    for pid in sorted(surrogate.keys()):
        model = surrogate[pid].get("model", {})
        print(f"  {pid:<4s} "
              f"{model.get('cv_r2', 0):7.4f} "
              f"{model.get('cv_rmse', 0):7.4f} "
              f"{model.get('n_features', 0):4d} "
              f"{model.get('n_training', 0):8d} "
              f"{surrogate[pid].get('extractor_name', ''):>14s}")
    print()


def print_baseline_check(reports: dict, tolerance_ns: float = 0.01):
    """Check baseline measurement consistency across programs.

    Prints a table of p50 values and flags inconsistencies beyond
    the given tolerance (in nanoseconds).
    """
    print_separator()
    print("  BASELINE MEASUREMENTS")
    print_separator()
    print()

    # Collect all baseline measurements
    baseline_data = {}  # name -> {program -> (p50, p99)}
    for pid, r in sorted(reports.items()):
        for bl in r.get("baselines", []):
            name = bl.get("name", "")
            if not name:
                continue
            if name not in baseline_data:
                baseline_data[name] = {}
            baseline_data[name][pid] = (
                bl.get("measured_p50_ns", 0),
                bl.get("measured_p99_ns", 0))

    programs = sorted(reports.keys())
    header = "  {:<16s}".format("Baseline")
    for p in programs:
        header += f" {p:>8s}"
    header += "  Check"
    print(header + "  (p50 ns)")
    print("  " + "-" * (16 + 9 * len(programs) + 10))

    all_consistent = True
    for name in sorted(baseline_data.keys()):
        row = f"  {name:<16s}"
        values = []
        for p in programs:
            if p in baseline_data[name]:
                v = baseline_data[name][p][0]
                values.append(v)
                row += f" {v:8.2f}"
            else:
                row += "      N/A"

        # Consistency check
        if len(values) >= 2:
            spread = max(values) - min(values)
            if spread > tolerance_ns:
                row += f"  DIFFERS ({spread:.2f} ns)"
                all_consistent = False
            else:
                row += "  ok"
        print(row)

    print()
    if all_consistent:
        print("  Baseline consistency: PASS")
    else:
        print("  Baseline consistency: DIFFERS — check measurement conditions")
    print()


def print_candidate_overlap(reports: dict):
    """Analyse overlap between top candidates across programs."""
    print_separator()
    print("  CANDIDATE OVERLAP (top-5 configs)")
    print_separator()
    print()

    # Collect top-5 configs per program (skip empty labels)
    program_configs = {}
    for pid, r in sorted(reports.items()):
        candidates = r.get("candidates", [])
        configs = set()
        for c in candidates[:5]:
            label = c.get("config_label", "")
            if label:
                configs.add(label)
        program_configs[pid] = configs

    programs = sorted(program_configs.keys())
    if len(programs) < 2:
        print("  (need at least 2 programs for overlap)")
        return

    fmt = "  {:<5s}"
    header = fmt.format("")
    for p in programs:
        header += f" {p:>4s}"
    print(header)
    print("  " + "-" * (5 + 5 * len(programs)))

    for p1 in programs:
        row = fmt.format(p1)
        for p2 in programs:
            if p1 == p2:
                row += f" {len(program_configs[p1]):4d}"
            else:
                overlap = len(program_configs[p1] & program_configs[p2])
                row += f" {overlap:4d}"
        print(row)
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [--csv]", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    csv_mode = "--csv" in sys.argv

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    reports = load_results(results_dir)
    if not reports:
        print(f"No JSON results found in {results_dir}/", file=sys.stderr)
        sys.exit(1)

    if csv_mode:
        # CSV mode: clean output only, no preamble
        write_csv(reports)
    else:
        print(f"Loaded {len(reports)} program results: "
              f"{', '.join(sorted(reports.keys()))}")

        print_summary_table(reports)
        print_winner_table(reports)
        print_model_comparison(reports)
        print_baseline_check(reports)
        print_candidate_overlap(reports)


if __name__ == "__main__":
    main()
