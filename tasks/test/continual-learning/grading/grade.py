#!/usr/bin/env python3
"""
Grade continual-learning runs by parsing log files and updating the task tables.

Usage tips for agents:
- Run without flags to refresh every table once you have logs for all subtasks.
- Add repeated `--table` flags (choices: imagenet_r, imagenet_a, cifar100, cub200)
  to update a single table. Each table groups one logical sub-task:
    * imagenet_r  -> ImageNet-R splits (N = 5 / 10 / 20)
    * imagenet_a  -> ImageNet-A split (N = 10)
    * cifar100    -> CIFAR100 split (N = 10)
    * cub200      -> CUB200 split (N = 10)
- The script always writes a JSON summary (before touching markdown) so intermediates
  can be inspected programmatically. Use `--json-out` to override its path.

Make sure your training configs encode the desired sub-task (e.g. ImageNet-R N=5 vs
N=20) because the grader only parses the resulting logs; it does not rerun training.
"""

import argparse
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class RunMetrics:
    acc: float  # final overall accuracy after last task ("total")
    aaa: float  # average anytime accuracy (mean over all per-step "Average Accuracy (CNN)" values)
    seed: Optional[int]
    path: str


DatasetKey = Tuple[str, Optional[int]]  # (dataset_name, num_tasks_or_None)


@dataclass(frozen=True)
class TableConfig:
    name: str
    dataset_keys: Tuple[DatasetKey, ...]
    table_header_regex: str



TABLE_CONFIGS: Dict[str, TableConfig] = {
    "imagenet_r": TableConfig(
        name="imagenet_r",
        dataset_keys=(
            ("ImageNet-R", 5),
            ("ImageNet-R", 10),
            ("ImageNet-R", 20),
        ),
        table_header_regex=r"(\| Method \| ImageNet-R \(N\s*=\s*5\) Acc ↑ AAA↑ \| ImageNet-R \(N\s*=\s*10\) Acc ↑ AAA ↑ \| ImageNet-R \(N\s*=\s*20\) Acc ↑ AAA ↑ \|\n\|\s*:---\s*\|.*?\n)",
    ),
    "imagenet_a": TableConfig(
        name="imagenet_a",
        dataset_keys=(("ImageNet-A", 10),),
        table_header_regex=r"(\| Method \| ImageNet-A \(N\s*=\s*10\) Acc ↑ AAA ↑ \|\n\|\s*:---\s*\|.*?\n)",
    ),
    "cifar100": TableConfig(
        name="cifar100",
        dataset_keys=(("CIFAR100", 10),),
        table_header_regex=r"(\| Method \| CIFAR100 Acc ↑ AAA ↑ \|\n\|\s*:---\s*\|.*?\n)",
    ),
    "cub200": TableConfig(
        name="cub200",
        dataset_keys=(("CUB200", 10),),
        table_header_regex=r"(\| Method \| CUB200 Acc ↑ AAA ↑ \|\n\|\s*:---\s*\|.*?\n)",
    ),
}

TABLE_ORDER: Tuple[str, ...] = tuple(TABLE_CONFIGS.keys())


def resolve_tables(requested: Optional[List[str]]) -> List[str]:
    if not requested or "all" in requested:
        return list(TABLE_ORDER)
    order_index = {name: idx for idx, name in enumerate(TABLE_ORDER)}
    resolved: List[str] = []
    for name in requested:
        if name == "all":
            continue
        if name not in TABLE_CONFIGS:
            raise ValueError(f"Unknown table '{name}'")
        if name not in resolved:
            resolved.append(name)
    resolved.sort(key=lambda item: order_index[item])
    return resolved


def tables_to_dataset_keys(tables: Iterable[str]) -> List[DatasetKey]:
    keys: List[DatasetKey] = []
    seen = set()
    for table_name in tables:
        config = TABLE_CONFIGS[table_name]
        for key in config.dataset_keys:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


ACC_TOTAL_RE = re.compile(
    r"CNN:\s*\{[^}]*'total':\s*(?:np\.float64\()?([0-9]+(?:\.[0-9]+)?)(?:\))?",
    re.IGNORECASE,
)
AVG_ACC_RE = re.compile(r"Average Accuracy \(CNN\):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
SEED_LINE_RE = re.compile(r"\bseed\s*[:=]\s*(\d+)|\bseed\s*\[(\d+)\]", re.IGNORECASE)
NB_TASKS_RE = re.compile(r"\bnb_tasks\b\s*[:=]\s*(\d+)", re.IGNORECASE)


def parse_seed(contents: str) -> Optional[int]:
    """Parse the last seed value from the provided text.

    Many runs append to the same log file; the last seed entry denotes the
    latest run block. We return the last valid integer seed encountered.
    """
    last_seed: Optional[int] = None
    for m in SEED_LINE_RE.finditer(contents):
        g = m.group(1) or m.group(2)
        if g is not None:
            try:
                last_seed = int(g)
            except ValueError:
                continue
    return last_seed


def _latest_run_block(contents: str) -> str:
    """Extract the content corresponding to the latest run appended to a log file.

    We heuristically delimit runs by the "seed:" entries logged at the start of
    each training invocation. We take the text from the last seed occurrence to
    the end of the file. If no seed marker exists, we fall back to the entire
    contents.
    """
    markers = list(SEED_LINE_RE.finditer(contents))
    if markers:
        return contents[markers[-1].start() :]
    return contents


def parse_log_metrics(path: str, expected_n_tasks: Optional[int] = None) -> Optional[RunMetrics]:
    """Parse metrics from a single log file.

    - Prefer extracting metrics from the last complete run segment delimited by
      seed markers. A segment is the text between successive seed entries.
    - If `expected_n_tasks` is provided, require that the chosen segment
      contains at least that many "Average Accuracy (CNN)" entries.
    - AAA is taken from the last "Average Accuracy (CNN)" line of the segment
      (which itself reflects the mean across tasks so far).
    - If seed markers are absent, fall back to using the tail of the file.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    seed_marks = list(SEED_LINE_RE.finditer(text))
    chosen_segment: Optional[Tuple[int, int]] = None

    if seed_marks:
        # Build segments [start, end) between seed markers
        boundaries: List[Tuple[int, int]] = []
        for i, m in enumerate(seed_marks):
            start = m.start()
            end = seed_marks[i + 1].start() if i + 1 < len(seed_marks) else len(text)
            boundaries.append((start, end))

        # Choose the last segment that has sufficient AAA entries
        for start, end in reversed(boundaries):
            segment = text[start:end]
            acc_values = ACC_TOTAL_RE.findall(segment)
            avg_values = AVG_ACC_RE.findall(segment)
            if not acc_values or not avg_values:
                continue
            try:
                avg_series_all = [float(x) for x in avg_values]
            except ValueError:
                continue
            if expected_n_tasks is not None and len(avg_series_all) < expected_n_tasks:
                continue
            chosen_segment = (start, end)
            break

    if chosen_segment is None:
        # Fallback: use latest run block by last seed marker or entire file
        block = _latest_run_block(text)
        acc_values = ACC_TOTAL_RE.findall(block)
        avg_values = AVG_ACC_RE.findall(block)
        if not acc_values or not avg_values:
            return None
        try:
            final_acc = float(acc_values[-1])
            avg_series_all = [float(x) for x in avg_values]
        except ValueError:
            return None
        if expected_n_tasks is not None:
            if len(avg_series_all) < expected_n_tasks:
                return None
        final_aaa = float(avg_values[-1])
        seed = parse_seed(block)
        return RunMetrics(acc=final_acc, aaa=final_aaa, seed=seed, path=path)

    # Compute metrics from the chosen segment
    start, end = chosen_segment
    segment = text[start:end]
    acc_values = ACC_TOTAL_RE.findall(segment)
    avg_values = AVG_ACC_RE.findall(segment)
    if not acc_values or not avg_values:
        return None
    try:
        final_acc = float(acc_values[-1])
        final_aaa = float(avg_values[-1])
    except ValueError:
        return None

    # Attribute seed to the run that produced the last AAA: pick the nearest
    # preceding seed marker in the entire file
    seed: Optional[int] = None
    try:
        last_avg_match = list(AVG_ACC_RE.finditer(segment))[-1]
        last_avg_pos_abs = start + last_avg_match.start()
        preceding = [m for m in seed_marks if m.start() <= last_avg_pos_abs]
        if preceding:
            sm = preceding[-1]
            g = sm.group(1) or sm.group(2)
            seed = int(g) if g is not None else None
    except Exception:
        # Fallback to first seed within the segment
        m = SEED_LINE_RE.search(segment)
        if m:
            g = m.group(1) or m.group(2)
            try:
                seed = int(g) if g is not None else None
            except Exception:
                seed = None

    return RunMetrics(acc=final_acc, aaa=final_aaa, seed=seed, path=path)


def _infer_nb_tasks_from_latest_block(path: str) -> Optional[int]:
    """Read the file and attempt to parse `nb_tasks` from the latest run block."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    block = _latest_run_block(text)
    m = NB_TASKS_RE.search(block)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def discover_logs(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".log"):
                paths.append(os.path.join(dirpath, fn))
    return paths


def identify_dataset_and_split_from_filename(filename: str) -> Optional[DatasetKey]:
    name = os.path.basename(filename)
    lower = name.lower()

    # ImageNet-R splits
    if "inr" in lower:
        # Match T5/T10/T20
        m = re.search(r"t(\d+)", lower)
        if m:
            return ("ImageNet-R", int(m.group(1)))
        # Fallback if unspecified
        return ("ImageNet-R", None)

    # ImageNet-A
    if re.search(r"\bina\b|imagenet-a|imageneta", lower):
        return ("ImageNet-A", 10)

    # CIFAR-100
    if re.search(r"c100|cifar100|cifar-?100", lower):
        return ("CIFAR100", 10)

    # CUB-200
    if re.search(r"cub|cub200|cub-200", lower):
        return ("CUB200", 10)

    return None


def group_runs_by_dataset(log_paths: List[str]) -> Dict[DatasetKey, List[RunMetrics]]:
    grouped: Dict[DatasetKey, List[RunMetrics]] = {}
    for p in log_paths:
        key = identify_dataset_and_split_from_filename(p)
        if key is None:
            # Try to infer from folder structure: e.g., logs/{model}/{dataset}/{init}/{inc}/...
            parts = p.replace("\\", "/").split("/")
            try:
                idx = max(i for i, x in enumerate(parts) if x == "logs")
                dataset_hint = parts[idx + 2] if len(parts) > idx + 2 else ""
                inc_hint = parts[idx + 4] if len(parts) > idx + 4 else ""
                dataset_readable = dataset_hint
                if "inr" in dataset_hint.lower() or "imagenet-r" in dataset_hint.lower():
                    dataset_readable = "ImageNet-R"
                elif "ina" in dataset_hint.lower() or "imagenet-a" in dataset_hint.lower():
                    dataset_readable = "ImageNet-A"
                elif "cifar" in dataset_hint.lower():
                    dataset_readable = "CIFAR100"
                elif "cub" in dataset_hint.lower():
                    dataset_readable = "CUB200"
                # Prefer nb_tasks from log (latest block); fall back to inc_hint
                n_tasks = _infer_nb_tasks_from_latest_block(p)
                if n_tasks is None:
                    try:
                        # Sometimes folder encodes number of tasks; otherwise this may be increment size
                        n_tasks = int(inc_hint)
                    except Exception:
                        n_tasks = None
                key = (dataset_readable, n_tasks)
            except Exception:
                key = None

        # Expected number of tasks, if we can infer it (prefer nb_tasks parsed from log)
        expected_n: Optional[int] = None
        if key is not None:
            expected_n = key[1]

        metrics = parse_log_metrics(p, expected_n_tasks=expected_n)
        if key is not None and metrics is not None:
            grouped.setdefault(key, []).append(metrics)
    return grouped


def mean_se(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], 0.0)
    mean = statistics.fmean(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        stdev = 0.0
    se = stdev / math.sqrt(len(values))
    return (mean, se)


def format_pair(acc_mean: float, acc_se: float, aaa_mean: float, aaa_se: float) -> str:
    return f"{acc_mean:.2f} ({acc_se:.2f}) {aaa_mean:.2f} ({aaa_se:.2f})"

def summarize_runs(runs: List[RunMetrics]) -> Optional[Dict[str, float]]:
    if not runs:
        return None
    accs = [r.acc for r in runs]
    aa_as = [r.aaa for r in runs]
    acc_m, acc_se = mean_se(accs)
    aaa_m, aaa_se = mean_se(aa_as)
    return {
        "acc_mean": acc_m,
        "acc_se": acc_se,
        "aaa_mean": aaa_m,
        "aaa_se": aaa_se,
    }


def fmt_or_dashes(summary: Optional[Dict[str, float]]) -> str:
    if summary is None:
        return "-- --"
    values = (
        summary.get("acc_mean"),
        summary.get("acc_se"),
        summary.get("aaa_mean"),
        summary.get("aaa_se"),
    )
    if not all(v is not None and math.isfinite(v) for v in values):
        return "-- --"
    return format_pair(
        summary["acc_mean"],
        summary["acc_se"],
        summary["aaa_mean"],
        summary["aaa_se"],
    )


def has_finite_metrics(summary: Optional[Dict[str, float]]) -> bool:
    if summary is None:
        return False
    return all(
        summary.get(field) is not None and math.isfinite(summary[field])
        for field in ("acc_mean", "acc_se", "aaa_mean", "aaa_se")
    )


def build_table_row(table: TableConfig, results: Dict[DatasetKey, List[RunMetrics]]) -> Optional[str]:
    columns: List[str] = []
    all_complete = True
    for key in table.dataset_keys:
        summary = summarize_runs(results.get(key, []))
        is_complete = has_finite_metrics(summary)
        all_complete = all_complete and is_complete
        columns.append(fmt_or_dashes(summary))
    if not all_complete:
        # Only write a row if we have complete metrics for all splits in the table
        return None
    joined = " | ".join(columns)
    return f"| Your Method | {joined} |"


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return value


def build_json_summary(tables: Iterable[str], results: Dict[DatasetKey, List[RunMetrics]]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tables": [],
    }
    for table_name in tables:
        config = TABLE_CONFIGS[table_name]
        table_entry = {
            "table": table_name,
            "datasets": [],
        }
        for key in config.dataset_keys:
            runs = results.get(key, [])
            summary = summarize_runs(runs)
            metrics = (
                None
                if summary is None
                else {
                    "acc_mean": safe_float(summary["acc_mean"]),
                    "acc_se": safe_float(summary["acc_se"]),
                    "aaa_mean": safe_float(summary["aaa_mean"]),
                    "aaa_se": safe_float(summary["aaa_se"]),
                }
            )
            seeds = sorted({r.seed for r in runs if r.seed is not None})
            table_entry["datasets"].append(
                {
                    "dataset": key[0],
                    "num_tasks": key[1],
                    "num_runs": len(runs),
                    "metrics": metrics,
                    "seeds": seeds,
                    "log_paths": sorted(r.path for r in runs),
                }
            )
        payload["tables"].append(table_entry)
    return payload


def update_task_description(md_path: str, results: Dict[DatasetKey, List[RunMetrics]], tables_to_update: Iterable[str]) -> bool:
    with open(md_path, "r", encoding="utf-8") as f:
        original_md = f.read()

    md = original_md

    for table_name in tables_to_update:
        config = TABLE_CONFIGS[table_name]
        new_row = build_table_row(config, results)
        if new_row is None:
            continue
        md = _replace_your_method_row(
            md,
            table_title_regex=config.table_header_regex,
            new_row=new_row,
        )

    if md != original_md:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        return True
    return False


def _replace_your_method_row(md: str, table_title_regex: str, new_row: str) -> str:
    # Find the table block that begins with table_title_regex and replace the line that starts with "| Your Method |"
    header_pattern = re.compile(table_title_regex, re.IGNORECASE | re.DOTALL)
    match = header_pattern.search(md)
    if not match:
        return md
    start = match.start(1)
    end = match.end(1)

    rest = md[end:]
    lines = rest.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("| your method |"):
            # If this is the initial placeholder row (no digits), replace it.
            # Otherwise, append a new row immediately after it to preserve prior results.
            if re.search(r"\d", line):
                # Already populated once; append a new row below
                lines.insert(i + 1, new_row)
            else:
                lines[i] = new_row
            new_rest = "\n".join(lines)
            return md[:end] + new_rest
        if not line.startswith("|"):
            break
    new_rest = new_row + "\n" + rest
    return md[:end] + new_rest


def main():
    parser = argparse.ArgumentParser(description="Grade continual-learning results by parsing logs and updating task_description.md")
    # Defaults expect to be invoked via grading/grade.sh which passes explicit paths
    parser.add_argument("--logs", type=str, default="../logs", help="Directory containing .log files (recursively searched)")
    parser.add_argument("--md", type=str, default="../task_description.md", help="Path to task_description.md to update")
    parser.add_argument(
        "--table",
        action="append",
        choices=list(TABLE_ORDER) + ["all"],
        help="Restrict grading to specific table(s); repeatable. Defaults to all tables.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        help="Optional path to write the intermediate metrics JSON. Defaults to <logs>/grade_summary.json.",
    )
    args = parser.parse_args()

    logs_dir = os.path.abspath(args.logs)
    md_path = os.path.abspath(args.md)
    tables = resolve_tables(args.table)
    target_keys = tables_to_dataset_keys(tables)

    log_paths = discover_logs(logs_dir)
    if not log_paths:
        print(f"No .log files found under {logs_dir}")
        return

    grouped = group_runs_by_dataset(log_paths)

    print("Parsed results (by dataset, split):")
    for key in sorted(target_keys, key=lambda x: (x[0], x[1] or 0)):
        runs = grouped.get(key, [])
        summary = summarize_runs(runs)
        if summary is None:
            print(f"- {key[0]} N={key[1] if key[1] is not None else '?'}: no runs found")
            continue
        acc_m = summary["acc_mean"]
        acc_se = summary["acc_se"]
        aaa_m = summary["aaa_mean"]
        aaa_se = summary["aaa_se"]
        print(
            f"- {key[0]} N={key[1] if key[1] is not None else '?'}: Acc={acc_m:.2f} (SE {acc_se:.2f}), "
            f"AAA={aaa_m:.2f} (SE {aaa_se:.2f}) from {len(runs)} run(s)"
        )

    summary_payload = build_json_summary(tables, grouped)
    json_path = (
        os.path.abspath(args.json_out)
        if args.json_out
        else os.path.join(logs_dir, "grade_summary.json")
    )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Wrote JSON summary to {json_path}")

    if update_task_description(md_path, grouped, tables):
        print(f"Updated {md_path} with 'Your Method' rows for: {', '.join(tables)}.")
    else:
        print("No markdown changes were necessary for the requested tables.")


if __name__ == "__main__":
    main()
