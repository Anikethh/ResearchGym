#!/usr/bin/env python3
"""
Grade improving-replay-buffers runs by parsing JSON summaries and refreshing markdown tables.

Expected log format
-------------------
Each run should emit one or more JSON files under ``logs/`` (default search root).
Every JSON file may contain either a single record or a list of records. Each record
must expose at least the keys below (additional fields are ignored):

{
    "environment": "quadruped-walk-v0",          # or "env", "task", "env_name"
    "category": "dmc_state",                     # or "suite", "domain"
    "seed": 0,                                   # optional; aids deduplication
    "metrics": {
        "average_return": 712.4,                 # required – numeric
        "dynamics_mse_log": -1.82,               # optional
        "dormant_ratio": 0.07                    # optional
    }
}

The grader aggregates per-environment metrics across seeds, computes mean/stdev,
emits a JSON summary (for reproducibility), and optionally updates the two tables
in ``task_description.md``. Use ``--no-markdown`` to skip markdown edits.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import StatisticsError, fmean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_METHOD_LABEL = "Your Method"
SUMMARY_NAME = "grading_summary.json"


def canonical_token(value: str) -> str:
    """Normalize identifiers for fuzzy matching."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


@dataclass(frozen=True)
class EnvSpec:
    key: str
    category: str
    display: str
    aliases: Tuple[str, ...]

    @property
    def alias_tokens(self) -> Tuple[str, ...]:
        return tuple(canonical_token(name) for name in self.aliases if name)

    @property
    def display_token(self) -> str:
        return canonical_token(self.display)


ENV_SPECS: Dict[str, EnvSpec] = {
    "dmc_state::quadruped_walk": EnvSpec(
        key="dmc_state::quadruped_walk",
        category="dmc_state",
        display="Quadruped-Walk",
        aliases=("quadruped_walk", "quadruped-walk", "quadruped-walk-v0", "quadrupedwalk"),
    ),
    "dmc_state::cheetah_run": EnvSpec(
        key="dmc_state::cheetah_run",
        category="dmc_state",
        display="Cheetah-Run",
        aliases=("cheetah_run", "cheetah-run", "cheetah-run-v0", "cheetahrun"),
    ),
    "dmc_state::reacher_hard": EnvSpec(
        key="dmc_state::reacher_hard",
        category="dmc_state",
        display="Reacher-Hard",
        aliases=("reacher_hard", "reacher-hard", "reacher-hard-v0", "reacherhard"),
    ),
    "dmc_state::finger_turn_hard": EnvSpec(
        key="dmc_state::finger_turn_hard",
        category="dmc_state",
        display="Finger-Turn-Hard",
        aliases=("finger_turn_hard", "finger-turn-hard", "finger-turn-hard-v0", "fingerturnhard"),
    ),
    "dmc_pixel::walker_walk": EnvSpec(
        key="dmc_pixel::walker_walk",
        category="dmc_pixel",
        display="Walker-Walk",
        aliases=("walker_walk", "walker-walk", "walker-walk-pixel", "walker-walk-v0", "pixelwalkerwalk"),
    ),
    "dmc_pixel::cheetah_run": EnvSpec(
        key="dmc_pixel::cheetah_run",
        category="dmc_pixel",
        display="Cheetah-Run",
        aliases=("cheetah_run_pixel", "cheetah-run-pixel", "pixelcheetahrun"),
    ),
    "gym::walker2d_v2": EnvSpec(
        key="gym::walker2d_v2",
        category="gym",
        display="Walker2d-v2",
        aliases=("walker2d_v2", "walker2d-v2", "walker2dv2"),
    ),
    "gym::halfcheetah_v2": EnvSpec(
        key="gym::halfcheetah_v2",
        category="gym",
        display="HalfCheetah-v2",
        aliases=("halfcheetah_v2", "halfcheetah-v2", "halfcheetahv2"),
    ),
    "gym::hopper_v2": EnvSpec(
        key="gym::hopper_v2",
        category="gym",
        display="Hopper-v2",
        aliases=("hopper_v2", "hopper-v2", "hopperv2"),
    ),
}


ALIAS_LOOKUP: Dict[str, List[str]] = defaultdict(list)
for spec_key, spec in ENV_SPECS.items():
    for token in spec.alias_tokens:
        ALIAS_LOOKUP[token].append(spec_key)


DMC_TABLE_COLUMNS: Tuple[str, ...] = (
    "dmc_state::quadruped_walk",
    "dmc_state::cheetah_run",
    "dmc_state::reacher_hard",
    "dmc_state::finger_turn_hard",
    "dmc_pixel::walker_walk",
    "dmc_pixel::cheetah_run",
)

GYM_TABLE_COLUMNS: Tuple[str, ...] = (
    "gym::walker2d_v2",
    "gym::halfcheetah_v2",
    "gym::hopper_v2",
)


CATEGORY_ALIASES: Dict[str, str] = {
    "dmc_state": "dmc_state",
    "dmc": "dmc_state",
    "state": "dmc_state",
    "states": "dmc_state",
    "dmcontrol_state": "dmc_state",
    "dmcontrol": "dmc_state",
    "dmc_pixel": "dmc_pixel",
    "pixel": "dmc_pixel",
    "pixels": "dmc_pixel",
    "vision": "dmc_pixel",
    "visual": "dmc_pixel",
    "gym": "gym",
    "mujoco": "gym",
}


ENV_KEYS = ("environment", "env", "env_name", "task", "name")
CATEGORY_KEYS = ("category", "suite", "domain", "group", "table")
SEED_KEYS = ("seed", "run_seed", "random_seed")
METRIC_CONTAINER_KEYS = ("metrics", "evaluation", "eval", "results")
AVERAGE_RETURN_KEYS = (
    "average_return",
    "avg_return",
    "return_mean",
    "mean_return",
    "return_avg",
    "final_average_return",
)
DYNAMICS_MSE_KEYS = (
    "dynamics_mse_log",
    "dynamics_mse",
    "dyn_mse_log",
    "dyn_mse",
    "model_error_log",
)
DORMANT_RATIO_KEYS = ("dormant_ratio", "inactive_fraction", "inactive_ratio")


@dataclass
class MetricRecord:
    spec_key: str
    average_return: float
    dynamics_mse_log: Optional[float]
    dormant_ratio: Optional[float]
    seed: Optional[int]
    source: Path


@dataclass
class StatBlock:
    mean: float
    std: float
    count: int

    def formatted(self) -> str:
        return f"{self.mean:.2f} +/- {self.std:.2f}"


@dataclass
class EnvSummary:
    spec: EnvSpec
    returns: Optional[StatBlock]
    dynamics_mse_log: Optional[StatBlock]
    dormant_ratio: Optional[StatBlock]
    samples: List[MetricRecord]


def canonicalize_category(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    token = canonical_token(str(raw))
    return CATEGORY_ALIASES.get(token, token or None)


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("mean", "value", "avg", "average"):
            if key in value:
                parsed = parse_float(value[key])
                if parsed is not None:
                    return parsed
    return None


def first_present(mapping: Dict[str, object], keys: Iterable[str]) -> Optional[object]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def resolve_env_spec(env_value: Optional[str], category_value: Optional[str]) -> Optional[EnvSpec]:
    if not env_value:
        return None
    token = canonical_token(env_value)
    candidates = ALIAS_LOOKUP.get(token, [])
    normalized_category = canonicalize_category(category_value)
    if normalized_category:
        candidates = [
            key for key in candidates if ENV_SPECS[key].category == normalized_category
        ] or candidates
    if len(candidates) == 1:
        return ENV_SPECS[candidates[0]]
    if not candidates and normalized_category:
        scoped = [
            spec
            for spec in ENV_SPECS.values()
            if spec.category == normalized_category
        ]
    elif not candidates:
        scoped = list(ENV_SPECS.values())
    else:
        scoped = [ENV_SPECS[key] for key in candidates]
    for spec in scoped:
        if spec.display_token and spec.display_token in token:
            return spec
    if not candidates and normalized_category:
        for spec in scoped:
            for alias_token in spec.alias_tokens:
                if alias_token and alias_token in token:
                    return spec
    return None


def extract_metrics(record: Dict[str, object]) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[float], Optional[float], Optional[float]]:
    env_raw = first_present(record, ENV_KEYS)
    category_raw = first_present(record, CATEGORY_KEYS)
    seed_raw = first_present(record, SEED_KEYS)
    seed_value: Optional[int] = None
    if isinstance(seed_raw, int):
        seed_value = seed_raw
    elif isinstance(seed_raw, str):
        try:
            seed_value = int(seed_raw.strip())
        except ValueError:
            seed_value = None
    metrics_obj = first_present(record, METRIC_CONTAINER_KEYS)
    metrics_dict: Dict[str, object]
    if isinstance(metrics_obj, dict):
        metrics_dict = metrics_obj
    else:
        metrics_dict = record
    avg_val = None
    for key in AVERAGE_RETURN_KEYS:
        if key in metrics_dict:
            avg_val = parse_float(metrics_dict[key])
            if avg_val is not None:
                break
    dyn_val = None
    for key in DYNAMICS_MSE_KEYS:
        if key in metrics_dict:
            dyn_val = parse_float(metrics_dict[key])
            if dyn_val is not None:
                break
    dormant_val = None
    for key in DORMANT_RATIO_KEYS:
        if key in metrics_dict:
            dormant_val = parse_float(metrics_dict[key])
            if dormant_val is not None:
                break
    return (
        env_raw if isinstance(env_raw, str) else None,
        category_raw if isinstance(category_raw, str) else None,
        seed_value,
        parse_float(avg_val),
        parse_float(dyn_val),
        parse_float(dormant_val),
    )


def iter_payload_records(payload: object) -> Iterable[Dict[str, object]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("records", "runs", "entries"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        yield item
                return
    if isinstance(payload, dict):
        yield payload


def collect_metric_records(logs_dir: Path) -> Tuple[List[MetricRecord], List[str]]:
    records: List[MetricRecord] = []
    warnings: List[str] = []
    if not logs_dir.exists():
        warnings.append(f"Logs directory does not exist: {logs_dir}")
        return records, warnings
    json_paths = sorted(p for p in logs_dir.rglob("*.json") if p.name != SUMMARY_NAME)
    if not json_paths:
        warnings.append(f"No JSON files found under {logs_dir}")
        return records, warnings
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Failed to parse {path}: {exc}")
            continue
        for entry in iter_payload_records(payload):
            env_raw, category_raw, seed, avg_return, dyn_mse, dormant = extract_metrics(entry)
            if avg_return is None:
                continue
            spec = resolve_env_spec(env_raw, category_raw)
            if spec is None:
                warnings.append(f"Unrecognized environment '{env_raw}' (category: {category_raw}) in {path}")
                continue
            records.append(
                MetricRecord(
                    spec_key=spec.key,
                    average_return=float(avg_return),
                    dynamics_mse_log=float(dyn_mse) if dyn_mse is not None else None,
                    dormant_ratio=float(dormant) if dormant is not None else None,
                    seed=seed,
                    source=path,
                )
            )
    return records, warnings


def summarize_records(records: Sequence[MetricRecord]) -> Dict[str, EnvSummary]:
    grouped: Dict[str, List[MetricRecord]] = defaultdict(list)
    for record in records:
        grouped[record.spec_key].append(record)
    summaries: Dict[str, EnvSummary] = {}
    for spec_key, samples in grouped.items():
        spec = ENV_SPECS[spec_key]
        returns_values = [item.average_return for item in samples]
        returns_stats = compute_stats(returns_values)
        dyn_values = [item.dynamics_mse_log for item in samples if item.dynamics_mse_log is not None]
        dyn_stats = compute_stats(dyn_values)
        dormant_values = [item.dormant_ratio for item in samples if item.dormant_ratio is not None]
        dormant_stats = compute_stats(dormant_values)
        summaries[spec_key] = EnvSummary(
            spec=spec,
            returns=returns_stats,
            dynamics_mse_log=dyn_stats,
            dormant_ratio=dormant_stats,
            samples=list(samples),
        )
    return summaries


def compute_stats(values: Sequence[Optional[float]]) -> Optional[StatBlock]:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None
    try:
        mean_val = fmean(cleaned)
    except StatisticsError:
        mean_val = sum(cleaned) / len(cleaned)
    if len(cleaned) >= 2:
        try:
            std_val = stdev(cleaned)
        except StatisticsError:
            std_val = 0.0
    else:
        std_val = 0.0
    return StatBlock(mean=mean_val, std=std_val, count=len(cleaned))


def build_json_summary(
    summaries: Dict[str, EnvSummary],
    warnings: Sequence[str],
    method: str,
    logs_dir: Path,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "logs_dir": str(logs_dir.resolve()),
        "warnings": list(warnings),
        "environments": {},
    }
    for spec_key, summary in sorted(summaries.items()):
        spec = summary.spec
        env_entry: Dict[str, object] = {}
        if summary.returns:
            env_entry["average_return"] = {
                "mean": summary.returns.mean,
                "std": summary.returns.std,
                "count": summary.returns.count,
            }
        if summary.dynamics_mse_log:
            env_entry["dynamics_mse_log"] = {
                "mean": summary.dynamics_mse_log.mean,
                "std": summary.dynamics_mse_log.std,
                "count": summary.dynamics_mse_log.count,
            }
        if summary.dormant_ratio:
            env_entry["dormant_ratio"] = {
                "mean": summary.dormant_ratio.mean,
                "std": summary.dormant_ratio.std,
                "count": summary.dormant_ratio.count,
            }
        env_entry["samples"] = [
            {
                "seed": record.seed,
                "average_return": record.average_return,
                "dynamics_mse_log": record.dynamics_mse_log,
                "dormant_ratio": record.dormant_ratio,
                "source": str(record.source),
            }
            for record in summary.samples
        ]
        payload["environments"][spec.display] = env_entry
    return payload


def format_value(block: Optional[StatBlock]) -> str:
    if block is None:
        return "--"
    return block.formatted()


def update_markdown(
    md_path: Path,
    method_label: str,
    dmc_values: Sequence[str],
    gym_values: Sequence[str],
) -> bool:
    try:
        contents = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Markdown file not found: {md_path}") from None
    lines = contents.splitlines()
    changed = False
    changed |= replace_table_row(
        lines=lines,
        header_prefix="| Environment |",
        method_label=method_label,
        fallback_label=DEFAULT_METHOD_LABEL,
        new_row=build_row(method_label, dmc_values),
    )
    changed |= replace_table_row(
        lines=lines,
        header_prefix="| Environment   |",
        method_label=method_label,
        fallback_label=DEFAULT_METHOD_LABEL,
        new_row=build_row(method_label, gym_values),
    )
    if changed:
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def build_row(method_label: str, values: Sequence[str]) -> str:
    cells = " | ".join(values)
    return f"| {method_label} | {cells} |"


def replace_table_row(
    lines: List[str],
    header_prefix: str,
    method_label: str,
    fallback_label: str,
    new_row: str,
) -> bool:
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            header_idx = idx
            break
    if header_idx is None:
        return False
    # Skip header and alignment row
    row_start = header_idx + 2
    insert_idx = None
    target_idx = None
    for idx in range(row_start, len(lines)):
        stripped = lines[idx].strip()
        if not stripped.startswith("|"):
            insert_idx = idx
            break
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        if not cells:
            continue
        first_cell = cells[0]
        if first_cell == method_label:
            target_idx = idx
            break
        if first_cell == fallback_label and target_idx is None:
            target_idx = idx
    if target_idx is None:
        if insert_idx is None:
            lines.append(new_row)
        else:
            lines.insert(insert_idx, new_row)
    else:
        if lines[target_idx] != new_row:
            lines[target_idx] = new_row
        else:
            return False
    return True


def resolve_default_json_out(logs_dir: Path) -> Path:
    return logs_dir / SUMMARY_NAME


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Grade improving replay buffers results")
    parser.add_argument(
        "--logs",
        type=Path,
        default=task_dir / "logs",
        help="Directory containing JSON metric files (default: tasks/.../logs)",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=task_dir / "task_description.md",
        help="Markdown file to update (default: task_description.md)",
    )
    parser.add_argument(
        "--method",
        default=os.environ.get("METHOD_NAME", DEFAULT_METHOD_LABEL),
        help="Label for the markdown table row (default: env METHOD_NAME or 'Your Method')",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the grading summary JSON (default: <logs>/grading_summary.json)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip updating markdown tables",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    method_label = args.method.strip() or DEFAULT_METHOD_LABEL
    logs_dir = args.logs.resolve()
    records, record_warnings = collect_metric_records(logs_dir)
    summaries = summarize_records(records)
    missing_keys = [
        ENV_SPECS[key].display for key in sorted(ENV_SPECS) if key not in summaries
    ]
    if missing_keys:
        print("Missing metrics for:", ", ".join(missing_keys))
    else:
        print("Collected metrics for all tracked environments.")
    if record_warnings:
        for warning in record_warnings:
            print(f"[warning] {warning}")
    json_summary = build_json_summary(
        summaries=summaries,
        warnings=record_warnings,
        method=method_label,
        logs_dir=logs_dir,
    )
    json_path = args.json_out.resolve() if args.json_out else resolve_default_json_out(logs_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    print(f"Wrote JSON summary to {json_path}")
    if not args.no_markdown:
        dmc_values = [format_value(summaries.get(key).returns if key in summaries else None) for key in DMC_TABLE_COLUMNS]
        gym_values = [format_value(summaries.get(key).returns if key in summaries else None) for key in GYM_TABLE_COLUMNS]
        try:
            if update_markdown(args.md.resolve(), method_label, dmc_values, gym_values):
                print(f"Updated {args.md}")
            else:
                print("Markdown did not require changes.")
        except FileNotFoundError as exc:
            print(exc)
    else:
        print("Skipped markdown update (--no-markdown set).")


if __name__ == "__main__":
    main()
