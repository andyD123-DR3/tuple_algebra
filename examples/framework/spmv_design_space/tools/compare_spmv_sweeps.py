#!/usr/bin/env python3
"""Compare SpMV selected-plan sweep summaries across platforms.

Input files are the *_selected.csv files emitted by spmv_design_space_demo
when --summary-prefix is supplied.  The report groups rows by problem size
and observation/search mode, then shows whether the strict observation hash
matches while allowing the selected plan to differ by platform.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Key = Tuple[str, str, str, str, str]


def read_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row.setdefault("source_file", str(path))
                rows.append(row)
    return rows


def key_for(row: Dict[str, str]) -> Key:
    return (
        row.get("problem_kind", ""),
        row.get("width", ""),
        row.get("height", ""),
        row.get("observation_mode", ""),
        row.get("search_mode", ""),
    )


def short_plan(row: Dict[str, str]) -> str:
    return " / ".join(
        part for part in [
            row.get("layout", ""),
            row.get("decomposition", ""),
            row.get("simd", ""),
            row.get("fusion", ""),
            row.get("reduction", ""),
        ] if part
    )


def markdown_report(rows: List[Dict[str, str]]) -> str:
    groups: Dict[Key, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[key_for(row)].append(row)

    out: List[str] = []
    out.append("# SpMV cross-platform selected-plan comparison")
    out.append("")
    out.append(
        "This report compares selected strict plans emitted by `spmv_design_space_demo`. "
        "Different platforms may choose different legal plans; the strict reproducibility "
        "claim is that the observed solver-state hash matches for the same problem and observation mode."
    )
    out.append("")
    out.append("| Problem | Size | Observation | Search | Platforms | Hash match | Distinct plans | Hash |")
    out.append("|---|---:|---|---|---|---|---:|---|")

    for key in sorted(groups, key=lambda k: (k[0], int(k[1] or 0), int(k[2] or 0), k[3], k[4])):
        group = sorted(groups[key], key=lambda r: r.get("platform", ""))
        hashes = {r.get("observation_hash", "") for r in group if r.get("observation_hash", "")}
        plans = {r.get("selected", "") for r in group if r.get("selected", "")}
        platforms = ", ".join(r.get("platform", "<unknown>") for r in group)
        hash_value = next(iter(hashes)) if len(hashes) == 1 else "multiple"
        out.append(
            f"| {key[0]} | {key[1]}x{key[2]} | {key[3]} | {key[4]} | "
            f"{platforms} | {'yes' if len(hashes) == 1 else 'NO'} | {len(plans)} | `{hash_value}` |"
        )

    out.append("")
    out.append("## Per-platform selected plans")
    out.append("")
    out.append("| Size | Platform | Median ns | Plan shape | Observation hash | Full selected name |")
    out.append("|---:|---|---:|---|---|---|")
    for key in sorted(groups, key=lambda k: (k[0], int(k[1] or 0), int(k[2] or 0), k[3], k[4])):
        for row in sorted(groups[key], key=lambda r: r.get("platform", "")):
            out.append(
                f"| {row.get('width', '')}x{row.get('height', '')} "
                f"| {row.get('platform', '')} "
                f"| {row.get('median_ns', '')} "
                f"| {short_plan(row)} "
                f"| `{row.get('observation_hash', '')}` "
                f"| `{row.get('selected', '')}` |"
            )

    out.append("")
    out.append(
        "Interpretation: a `Hash match` of `yes` with more than one distinct plan is the key "
        "heterogeneous-reproducibility evidence: implementation strategy changed, but the strict observed solver state did not."
    )
    out.append("")
    return "\n".join(out)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="*_selected.csv files to compare")
    parser.add_argument("-o", "--output", type=Path, help="Markdown report path; stdout if omitted")
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    if not rows:
        print("no selected-plan rows found", file=sys.stderr)
        return 1

    text = markdown_report(rows)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
