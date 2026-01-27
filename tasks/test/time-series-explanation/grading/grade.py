#!/usr/bin/env python3
"""
Grade time-series explanation task runs by parsing CSV logs and refreshing task tables.

This script mirrors the grading utilities available for other tasks:

Usage patterns:

- Grade everything (real + synthetic tables) using defaults:
    ./grading/grade.py
- Override log directory or markdown path:
    ./grading/grade.py --logs logs/custom --md task_description.md
- Focus on one table:
    ./grading/grade.py --table real --real-explainer my_method
- Inspect metrics without touching markdown:
    ./grading/grade.py --json-out summary.json --no-markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import StatisticsError, fmean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


REAL_TOPK_DEFAULT = 0.1
REAL_TOPK_EPS = 1e-6


@dataclass(frozen=True)
class RealColumnSpec:
    dataset_key: str  # lower-case token used for file matching (e.g. "pam")
    baseline: str     # "Average" or "Zeros"
    label: str        # Column label in markdown table


REAL_COLUMNS: Tuple[RealColumnSpec, ...] = (
    RealColumnSpec("pam", "Average", "PAM Avg."),
    RealColumnSpec("pam", "Zeros", "PAM Zero"),
    RealColumnSpec("boiler", "Average", "Boiler Avg."),
    RealColumnSpec("boiler", "Zeros", "Boiler Zero"),
    RealColumnSpec("epilepsy", "Average", "Epilepsy Avg."),
    RealColumnSpec("epilepsy", "Zeros", "Epilepsy Zero"),
    RealColumnSpec("wafer", "Average", "Wafer Avg."),
    RealColumnSpec("wafer", "Zeros", "Wafer Zero"),
    RealColumnSpec("freezer", "Average", "Freezer Avg."),
    RealColumnSpec("freezer", "Zeros", "Freezer Zero"),
)


REAL_DATASET_DISPLAY = {
    "pam": "PAM",
    "boiler": "Boiler",
    "epilepsy": "Epilepsy",
    "wafer": "Wafer",
    "freezer": "Freezer",
}


@dataclass(frozen=True)
class SyntheticTableSpec:
    name: str                # identifier used in CLI (e.g. "switchfeature")
    heading_label: str       # markdown label preceding the table (e.g. "**Switch-Feature**")
    dataset_token: str       # substring expected in filenames (e.g. "switch_feature")
    display_name: str        # human readable name for JSON payload


SYNTHETIC_TABLES: Tuple[SyntheticTableSpec, ...] = (
    SyntheticTableSpec("switchfeature", "**Switch-Feature**", "switch_feature", "Switch-Feature"),
    SyntheticTableSpec("state", "**State**", "hmm", "State"),
)


TABLE_CHOICES = {spec.name for spec in SYNTHETIC_TABLES} | {"real"}


@dataclass
class MetricAggregate:
    mean: float
    se: float
    count: int

    def formatted(self) -> str:
        return f"{self.mean:.3f}±{self.se:.3f}"


@dataclass
class RealSummary:
    aggregates: Dict[Tuple[str, str], MetricAggregate]
    available_explainers: Set[str]


@dataclass
class SyntheticSummary:
    metrics: Dict[str, Dict[str, MetricAggregate]]
    available_explainers: Set[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    task_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        type=Path,
        default=task_root / "logs",
        help="Directory containing CSV outputs (default: tasks/.../logs)",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=task_root / "task_description.md",
        help="Markdown file to update (default: task_description.md)",
    )
    parser.add_argument(
        "--method",
        default=os.environ.get("METHOD_NAME", "Your Method"),
        help="Label for the markdown row (default: METHOD_NAME env or 'Your Method')",
    )
    parser.add_argument(
        "--real-explainer",
        dest="real_explainers",
        action="append",
        help="Explainer label(s) to grade for real datasets (case-insensitive). Can repeat.",
    )
    parser.add_argument(
        "--synthetic-explainer",
        dest="synthetic_explainers",
        action="append",
        help="Explainer label(s) to grade for synthetic datasets. Can repeat.",
    )
    parser.add_argument(
        "--synthetic-baseline",
        default=os.environ.get("SYNTHETIC_BASELINE", "Average"),
        help="Baseline column to use for synthetic tables (default: Average)",
    )
    parser.add_argument(
        "--topk",
        type=float,
        default=REAL_TOPK_DEFAULT,
        help="Masking ratio (Topk column) to use for real datasets (default: 0.1).",
    )
    parser.add_argument(
        "--table",
        dest="tables",
        action="append",
        choices=sorted(TABLE_CHOICES | {"all"}),
        help="Which table(s) to refresh (choices: real, switchfeature, state, all). Default: all",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write a JSON summary of the collected metrics.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip markdown updates; still prints summary and JSON if requested.",
    )
    return parser.parse_args(argv)


def resolve_explainers(raw: Iterable[str] | None, env_var: str, default: str) -> Set[str]:
    if raw:
        return {item.strip().lower() for item in raw if item.strip()}
    env_value = os.environ.get(env_var)
    if env_value:
        return {item.strip().lower() for item in env_value.split(",") if item.strip()}
    return {default.lower()} if default else set()


def resolve_tables(tables: Iterable[str] | None) -> List[str]:
    if not tables:
        return ["real", "switchfeature", "state"]
    resolved: List[str] = []
    for entry in tables:
        if entry == "all":
            return ["real", "switchfeature", "state"]
        if entry not in TABLE_CHOICES:
            continue
        if entry not in resolved:
            resolved.append(entry)
    return resolved


def aggregate(values: List[float]) -> Optional[MetricAggregate]:
    if not values:
        return None
    mean_val = fmean(values)
    if len(values) == 1:
        se_val = 0.0
    else:
        try:
            se_val = stdev(values) / math.sqrt(len(values))
        except StatisticsError:
            se_val = 0.0
    return MetricAggregate(mean=mean_val, se=se_val, count=len(values))


def iter_csv_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*.csv"):
        if path.is_file():
            yield path


def collect_real_summary(
    logs_dir: Path,
    explainers: Set[str],
    topk_target: float,
    topk_eps: float,
) -> RealSummary:
    aggregates: Dict[Tuple[str, str], MetricAggregate] = {}
    available_explainers: Set[str] = set()

    for spec in REAL_COLUMNS:
        dataset_matches: List[float] = []
        dataset_token = spec.dataset_key.lower()
        for csv_path in iter_csv_files(logs_dir):
            lower_path = str(csv_path).lower()
            if dataset_token not in lower_path:
                continue
            try:
                with csv_path.open("r", newline="") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        if len(row) < 10:
                            continue
                        baseline = row[2].strip()
                        if baseline.lower() != spec.baseline.lower():
                            continue
                        try:
                            topk_val = float(row[3])
                        except (ValueError, IndexError):
                            continue
                        if not math.isfinite(topk_val) or abs(topk_val - topk_target) > topk_eps:
                            continue
                        explainer = row[4].strip().lower()
                        if explainer:
                            available_explainers.add(explainer)
                        if explainers and explainer not in explainers:
                            continue
                        try:
                            value = float(row[9])
                        except (ValueError, IndexError):
                            continue
                        if math.isfinite(value):
                            dataset_matches.append(value)
            except OSError:
                continue
        agg = aggregate(dataset_matches)
        if agg is not None:
            aggregates[(spec.dataset_key, spec.baseline)] = agg

    return RealSummary(aggregates=aggregates, available_explainers=available_explainers)


def collect_synthetic_summary(
    logs_dir: Path,
    explainers: Set[str],
    baseline: str,
) -> SyntheticSummary:
    metrics: Dict[str, Dict[str, MetricAggregate]] = {}
    available_explainers: Set[str] = set()

    for spec in SYNTHETIC_TABLES:
        values_by_metric = {
            "cpd": [],
            "aup": [],
            "aur": [],
        }
        dataset_token = spec.dataset_token.lower()

        for csv_path in iter_csv_files(logs_dir):
            lower_path = str(csv_path).lower()
            if dataset_token not in lower_path:
                continue
            try:
                with csv_path.open("r", newline="") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        if len(row) < 11:
                            continue
                        row_baseline = row[2].strip()
                        if row_baseline.lower() != baseline.lower():
                            continue
                        explainer = row[3].strip().lower()
                        if explainer:
                            available_explainers.add(explainer)
                        if explainers and explainer not in explainers:
                            continue
                        try:
                            cpd_val = float(row[7])
                            aup_val = float(row[9])
                            aur_val = float(row[10])
                        except (ValueError, IndexError):
                            continue
                        if math.isfinite(cpd_val):
                            values_by_metric["cpd"].append(cpd_val)
                        if math.isfinite(aup_val):
                            values_by_metric["aup"].append(aup_val)
                        if math.isfinite(aur_val):
                            values_by_metric["aur"].append(aur_val)
            except OSError:
                continue

        metrics[spec.name] = {
            metric: agg
            for metric, agg in (
                (metric_name, aggregate(values))
                for metric_name, values in values_by_metric.items()
            )
            if agg is not None
        }

    return SyntheticSummary(metrics=metrics, available_explainers=available_explainers)


def build_real_row(method_label: str, summary: RealSummary) -> Optional[str]:
    cells: List[str] = []
    for spec in REAL_COLUMNS:
        agg = summary.aggregates.get((spec.dataset_key, spec.baseline))
        if agg is None or not math.isfinite(agg.mean) or not math.isfinite(agg.se):
            return None
        cells.append(agg.formatted())
    return f"| {method_label} | " + " | ".join(cells) + " |"


def build_synthetic_row(method_label: str, summary: SyntheticSummary, table_name: str) -> Optional[str]:
    metrics = summary.metrics.get(table_name, {})
    required_keys = ("cpd", "aup", "aur")
    formatted: List[str] = []
    for key in required_keys:
        agg = metrics.get(key)
        if agg is None or not math.isfinite(agg.mean) or not math.isfinite(agg.se):
            return None
        formatted.append(agg.formatted())
    return f"| {method_label} | " + " | ".join(formatted) + " |"


def replace_table_row(
    md_path: Path,
    method_label: str,
    new_row: str,
    header_hint: str,
    anchor_label: Optional[str] = None,
) -> bool:
    original = md_path.read_text(encoding="utf-8")
    lines = original.splitlines()

    # Locate table start
    start_idx = None
    if anchor_label:
        for idx, line in enumerate(lines):
            if line.strip() == anchor_label.strip():
                start_idx = idx + 1
                break
    if start_idx is None:
        start_idx = 0

    header_idx = None
    for idx in range(start_idx, len(lines)):
        if header_hint in lines[idx]:
            header_idx = idx
            break
        if anchor_label and lines[idx].strip().startswith("**") and lines[idx].strip() != anchor_label.strip():
            # Hit the next section without finding the table.
            break
    if header_idx is None:
        return False

    # Determine table block
    data_start = header_idx + 2  # skip header + separator
    end_idx = data_start
    while end_idx < len(lines) and lines[end_idx].startswith("|"):
        end_idx += 1

    target_prefix = f"| {method_label} "
    replaced = False
    for idx in range(data_start, end_idx):
        if lines[idx].startswith(target_prefix):
            lines[idx] = new_row
            replaced = True
            break

    if not replaced:
        lines.insert(end_idx, new_row)
        end_idx += 1

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def build_json_summary(
    method_label: str,
    real_summary: Optional[RealSummary],
    synthetic_summary: Optional[SyntheticSummary],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "method": method_label,
        "real": {},
        "synthetic": {},
    }
    if real_summary is not None:
        real_payload: Dict[str, object] = {}
        for spec in REAL_COLUMNS:
            agg = real_summary.aggregates.get((spec.dataset_key, spec.baseline))
            if agg is None:
                continue
            dataset_name = REAL_DATASET_DISPLAY[spec.dataset_key]
            entry = real_payload.setdefault(dataset_name, {})
            entry[spec.baseline] = {
                "mean": agg.mean,
                "se": agg.se,
                "count": agg.count,
            }
        payload["real"] = real_payload
        payload["real_available_explainers"] = sorted(real_summary.available_explainers)
    if synthetic_summary is not None:
        syn_payload: Dict[str, object] = {}
        for spec in SYNTHETIC_TABLES:
            metrics = synthetic_summary.metrics.get(spec.name, {})
            syn_payload[spec.display_name] = {
                metric_name: {
                    "mean": agg.mean,
                    "se": agg.se,
                    "count": agg.count,
                }
                for metric_name, agg in metrics.items()
            }
        payload["synthetic"] = syn_payload
        payload["synthetic_available_explainers"] = sorted(synthetic_summary.available_explainers)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    logs_dir = args.logs
    method_label: str = args.method
    tables = resolve_tables(args.tables)

    real_explainers = resolve_explainers(args.real_explainers, "REAL_EXPLAINER", "your_method")
    synthetic_explainers = resolve_explainers(
        args.synthetic_explainers,
        "SYNTHETIC_EXPLAINER",
        "your_method",
    )

    real_summary: Optional[RealSummary] = None
    synthetic_summary: Optional[SyntheticSummary] = None

    if "real" in tables:
        real_summary = collect_real_summary(
            logs_dir=logs_dir,
            explainers=real_explainers,
            topk_target=args.topk,
            topk_eps=REAL_TOPK_EPS,
        )
        row = build_real_row(method_label, real_summary)
        if row:
            if not args.no_markdown:
                updated = replace_table_row(
                    args.md,
                    method_label=method_label,
                    new_row=row,
                    header_hint="PAM Avg.",
                )
                if updated:
                    print("Updated real-world table.")
                else:
                    print("Warning: real-world table header not found; skipped markdown update.")
        else:
            print("Warning: incomplete real-world metrics; not updating markdown table.")
            missing = sorted(real_summary.available_explainers) if real_summary else []
            if missing:
                print(f"  Available explainer labels in logs: {missing}")

    requested_synthetic_specs = [spec for spec in SYNTHETIC_TABLES if spec.name in tables]
    if requested_synthetic_specs:
        synthetic_summary = collect_synthetic_summary(
            logs_dir=logs_dir,
            explainers=synthetic_explainers,
            baseline=args.synthetic_baseline,
        )
        for spec in requested_synthetic_specs:
            row = build_synthetic_row(method_label, synthetic_summary, spec.name)
            if row and not args.no_markdown:
                updated = replace_table_row(
                    args.md,
                    method_label=method_label,
                    new_row=row,
                    header_hint="| Method | CPD",
                    anchor_label=spec.heading_label,
                )
                if updated:
                    print(f"Updated synthetic table for {spec.display_name}.")
                else:
                    print(f"Warning: synthetic table for {spec.display_name} not found; skipped markdown.")
            elif row is None:
                print(f"Warning: incomplete metrics for {spec.display_name}; not updating markdown table.")
                available = sorted(synthetic_summary.available_explainers)
                if available:
                    print(f"  Available explainer labels in logs: {available}")

    summary = build_json_summary(method_label, real_summary, synthetic_summary)
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
