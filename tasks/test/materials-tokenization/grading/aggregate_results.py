#!/usr/bin/env python3
"""Aggregate materials-tokenization evaluation outputs and update task tables.

The script scans prediction directories for per-subtask metrics, writes a JSON
summary, and (optionally) refreshes the "Your Method" rows in task_description.md.
Use `--tables` to control which tables are touched and `--no_markdown` when you
only want the JSON artefact.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import statistics
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

CLASSIFICATION_TABLE_HINT = "Table 2"
GENERATION_TABLE_HINT = "Table 1"

CLASSIFICATION_COLUMN_KEYS = [
    "ner_sofc_val",
    "ner_sofc_test",
    "ner_matscholar_val",
    "ner_matscholar_test",
    "sf_val",
    "sf_test",
    "rc_val",
    "rc_test",
    "pc_val",
    "pc_test",
]

GENERATION_COLUMN_KEYS = ["NER", "RC", "EAE", "PC", "SAR", "SC", "SF", "Overall"]


@dataclass
class Cell:
    display: str
    values: List[float]
    updated: bool


@dataclass
class TableSummary:
    micro: Dict[str, Cell]
    macro: Dict[str, Cell]


@dataclass
class AggregationSummary:
    method: str
    preds_dir: str
    seeds: List[int]
    timestamp: str
    tables: List[str]
    classification: Optional[TableSummary]
    generation: Optional[TableSummary]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--task_dir", required=True, type=Path)
    parser.add_argument("--preds_dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=("classification", "generation"),
        default=["classification", "generation"],
        help="Tables to refresh",
    )
    parser.add_argument("--json_out", type=Path, help="Optional output path for the summary JSON")
    parser.add_argument("--no_markdown", action="store_true", help="Skip editing task_description.md")
    return parser


def fmt(mean: float, std: float) -> str:
    return f"{mean * 100:.1f} ±{std * 100:.1f}"


def compute_cell(values: List[float]) -> Cell:
    has_values = len(values) > 0
    if not has_values:
        return Cell(display="--", values=[], updated=False)
    mean_val = statistics.mean(values)
    std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
    return Cell(display=fmt(mean_val, std_val), values=values, updated=True)


def load_pickle(path: Path) -> Optional[object]:
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[aggregate] Failed to load pickle {path}: {exc}")
        return None


def load_ner_metrics(preds_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    result: Dict[str, Dict[str, List[float]]] = {}
    datasets = ["sofc", "sofc_slot", "matscholar"]
    for ds in datasets:
        ds_dir = preds_dir / "ner" / ds
        if not ds_dir.is_dir():
            continue
        if ds == "matscholar":
            pkl_path = next((ds_dir / name for name in os.listdir(ds_dir) if name.startswith("res_") and name.endswith("_bert-crf.pkl")), None)
            if pkl_path and pkl_path.exists():
                df = load_pickle(pkl_path)
                if df is None:
                    continue
                res = {
                    "val_micro": list(df.loc['Val micro_f1']) if 'Val micro_f1' in df.index else [],
                    "test_micro": list(df.loc['Test micro_f1']) if 'Test micro_f1' in df.index else [],
                    "val_macro": list(df.loc['Val macro_f1']) if 'Val macro_f1' in df.index else [],
                    "test_macro": list(df.loc['Test macro_f1']) if 'Test macro_f1' in df.index else [],
                }
                result[ds] = res
        else:
            per_key_fold_seed: Dict[str, List[List[float]]] = {
                'val_micro': [],
                'test_micro': [],
                'val_macro': [],
                'test_macro': [],
            }
            for fold in range(1, 6):
                fold_dir = ds_dir / f"cv_{fold}"
                if not fold_dir.is_dir():
                    continue
                pkl_path = next((fold_dir / name for name in os.listdir(fold_dir) if name.startswith("res_") and name.endswith("_bert-crf.pkl")), None)
                if not pkl_path or not pkl_path.exists():
                    continue
                df = load_pickle(pkl_path)
                if df is None:
                    continue
                for key, idx in [
                    ('val_micro', 'Val micro_f1'),
                    ('test_micro', 'Test micro_f1'),
                    ('val_macro', 'Val macro_f1'),
                    ('test_macro', 'Test macro_f1'),
                ]:
                    if idx in df.index:
                        vals = list(df.loc[idx])
                        if vals:
                            per_key_fold_seed[key].append(vals)
            agg: Dict[str, List[float]] = {}
            for key, fold_lists in per_key_fold_seed.items():
                if not fold_lists:
                    agg[key] = []
                    continue
                num_seeds = max(len(fl) for fl in fold_lists)
                per_seed_avgs: List[float] = []
                for s in range(num_seeds):
                    seed_vals = [fl[s] for fl in fold_lists if len(fl) > s]
                    if seed_vals:
                        per_seed_avgs.append(sum(seed_vals) / len(seed_vals))
                agg[key] = per_seed_avgs
            result[ds] = agg
    return result


def load_rc_metrics(preds_dir: Path) -> Dict[str, List[float]]:
    rc_dir = preds_dir / "relation_classification"
    if not rc_dir.is_dir():
        return {}
    pkl_path = next((rc_dir / name for name in os.listdir(rc_dir) if name.startswith("res_") and name.endswith(".pkl")), None)
    if not pkl_path or not pkl_path.exists():
        return {}
    df = load_pickle(pkl_path)
    if df is None:
        return {}
    return {
        'val_micro': list(df.loc['Val micro_f1']) if 'Val micro_f1' in df.index else [],
        'test_micro': list(df.loc['Test micro_f1']) if 'Test micro_f1' in df.index else [],
        'val_macro': list(df.loc['Val macro_f1']) if 'Val macro_f1' in df.index else [],
        'test_macro': list(df.loc['Test macro_f1']) if 'Test macro_f1' in df.index else [],
    }


def load_pc_summary(preds_dir: Path) -> Optional[Dict[str, List[float]]]:
    path = preds_dir / "cls" / "pc_summary.json"
    if not path.exists():
        return None
    with path.open() as fh:
        data = json.load(fh)
    return {
        'val': data.get('val_accuracy_list', []),
        'test': data.get('test_accuracy_list', []),
    }


def load_generation_summary(preds_dir: Path) -> Optional[Dict[str, Dict[str, List[float]]]]:
    path = preds_dir / "generation" / "gen_summary.json"
    if not path.exists():
        return None
    with path.open() as fh:
        data = json.load(fh)
    return data


def build_classification_summary(
    preds_dir: Path,
) -> TableSummary:
    ner = load_ner_metrics(preds_dir)
    rc = load_rc_metrics(preds_dir)
    pc = load_pc_summary(preds_dir)

    micro: Dict[str, Cell] = {
        "ner_sofc_val": compute_cell(ner.get('sofc', {}).get('val_micro', [])),
        "ner_sofc_test": compute_cell(ner.get('sofc', {}).get('test_micro', [])),
        "ner_matscholar_val": compute_cell(ner.get('matscholar', {}).get('val_micro', [])),
        "ner_matscholar_test": compute_cell(ner.get('matscholar', {}).get('test_micro', [])),
        "sf_val": compute_cell(ner.get('sofc_slot', {}).get('val_micro', [])),
        "sf_test": compute_cell(ner.get('sofc_slot', {}).get('test_micro', [])),
        "rc_val": compute_cell(rc.get('val_micro', []) if rc else []),
        "rc_test": compute_cell(rc.get('test_micro', []) if rc else []),
        "pc_val": compute_cell(pc['val'] if pc else []),
        "pc_test": compute_cell(pc['test'] if pc else []),
    }

    macro: Dict[str, Cell] = {
        "ner_sofc_val": compute_cell(ner.get('sofc', {}).get('val_macro', [])),
        "ner_sofc_test": compute_cell(ner.get('sofc', {}).get('test_macro', [])),
        "ner_matscholar_val": compute_cell(ner.get('matscholar', {}).get('val_macro', [])),
        "ner_matscholar_test": compute_cell(ner.get('matscholar', {}).get('test_macro', [])),
        "sf_val": compute_cell(ner.get('sofc_slot', {}).get('val_macro', [])),
        "sf_test": compute_cell(ner.get('sofc_slot', {}).get('test_macro', [])),
        "rc_val": compute_cell(rc.get('val_macro', []) if rc else []),
        "rc_test": compute_cell(rc.get('test_macro', []) if rc else []),
        "pc_val": compute_cell(pc['val'] if pc else []),
        "pc_test": compute_cell(pc['test'] if pc else []),
    }

    return TableSummary(micro=micro, macro=macro)


def compute_generation_cells(data: Mapping[str, Mapping[str, List[float]]]) -> TableSummary:
    micro_cells: Dict[str, Cell] = {}
    macro_cells: Dict[str, Cell] = {}
    task_order = ["NER", "RC", "EAE", "PC", "SAR", "SC", "SF"]

    micro_means: List[float] = []
    macro_means: List[float] = []

    for task in task_order:
        task_micro = compute_cell(data.get('micro', {}).get(task, []) if data else [])
        task_macro = compute_cell(data.get('macro', {}).get(task, []) if data else [])
        micro_cells[task] = task_micro
        macro_cells[task] = task_macro
        if task_micro.updated:
            micro_means.append(statistics.mean(task_micro.values))
        if task_macro.updated:
            macro_means.append(statistics.mean(task_macro.values))

    def overall(means: List[float]) -> Cell:
        if not means:
            return Cell(display="--", values=[], updated=False)
        mean_val = statistics.mean(means)
        return Cell(display=fmt(mean_val, 0.0), values=means, updated=True)

    micro_cells["Overall"] = overall(micro_means)
    macro_cells["Overall"] = overall(macro_means)

    return TableSummary(micro=micro_cells, macro=macro_cells)


def build_generation_summary(preds_dir: Path) -> Optional[TableSummary]:
    data = load_generation_summary(preds_dir)
    if not data:
        return None
    return compute_generation_cells(data)


def ensure_parent(path: Path) -> None:
    if path and path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def extract_section(md: str, table_hint: str) -> tuple[int, int, str]:
    start = md.find(table_hint)
    if start == -1:
        start = 0
    next_idx = md.find('Table', start + len(table_hint))
    if next_idx == -1:
        next_idx = len(md)
    section = md[start:next_idx]
    return start, next_idx, section


def parse_existing_row(section: str, metric_label: str | None, expected_len: int) -> tuple[List[str], Optional[re.Match[str]]]:
    if metric_label:
        pattern = re.compile(rf"^\|\s*Your Method\s*\|\s*{re.escape(metric_label)}\s*\|[^\n]*$", re.MULTILINE)
    else:
        pattern = re.compile(r"^\|\s*Your Method\s*\|[^\n]*$", re.MULTILINE)
    match = pattern.search(section)
    if not match:
        return ["--"] * expected_len, None
    row = match.group(0)
    parts = [part.strip() for part in row.strip().split('|')]
    if metric_label:
        cells = parts[3:-1]
    else:
        cells = parts[2:-1]
    if len(cells) != expected_len:
        # Fallback to placeholder length
        return ["--"] * expected_len, match
    return cells, match


def build_row(metric_label: str | None, cells: Sequence[str]) -> str:
    payload = " | ".join(cells)
    if metric_label:
        return f"| Your Method | {metric_label} | {payload} |\n"
    return f"| Your Method | {payload} |\n"


def merge_cells(existing: List[str], columns: Sequence[str], updates: Mapping[str, Cell]) -> List[str]:
    merged = existing[:]
    for idx, key in enumerate(columns):
        cell = updates.get(key)
        if cell and cell.updated:
            merged[idx] = cell.display
    return merged


def upsert_row(md: str, table_hint: str, metric_label: str, columns: Sequence[str], updates: Mapping[str, Cell]) -> str:
    start, end, section = extract_section(md, table_hint)
    existing_cells, match = parse_existing_row(section, metric_label, len(columns))
    merged_cells = merge_cells(existing_cells, columns, updates)
    new_row = build_row(metric_label, merged_cells)
    if match:
        section = section[:match.start()] + new_row + section[match.end():]
    else:
        if section and not section.endswith('\n'):
            section += '\n'
        section += new_row
    return md[:start] + section + md[end:]


def summarise_to_dict(summary: AggregationSummary) -> Dict[str, object]:
    def table_to_dict(table: Optional[TableSummary]) -> Optional[Dict[str, object]]:
        if table is None:
            return None
        return {
            'micro': {
                key: {
                    'display': cell.display,
                    'values': cell.values,
                    'updated': cell.updated,
                }
                for key, cell in table.micro.items()
            },
            'macro': {
                key: {
                    'display': cell.display,
                    'values': cell.values,
                    'updated': cell.updated,
                }
                for key, cell in table.macro.items()
            },
        }

    return {
        'method': summary.method,
        'preds_dir': summary.preds_dir,
        'seeds': summary.seeds,
        'timestamp': summary.timestamp,
        'tables': summary.tables,
        'classification': table_to_dict(summary.classification),
        'generation': table_to_dict(summary.generation),
    }


def main() -> None:
    args = parse_args().parse_args()
    preds_dir = args.preds_dir.resolve()
    task_dir = args.task_dir.resolve()
    md_path = task_dir / "task_description.md"

    classification_summary: Optional[TableSummary] = None
    generation_summary: Optional[TableSummary] = None

    if "classification" in args.tables:
        classification_summary = build_classification_summary(preds_dir)
    if "generation" in args.tables:
        generation_summary = build_generation_summary(preds_dir)

    summary = AggregationSummary(
        method=args.method,
        preds_dir=str(preds_dir),
        seeds=args.seeds,
        timestamp=datetime.utcnow().isoformat(),
        tables=args.tables,
        classification=classification_summary,
        generation=generation_summary,
    )

    summary_dict = summarise_to_dict(summary)
    json_path = args.json_out or (preds_dir / "grading_summary.json")
    ensure_parent(json_path)
    with json_path.open("w") as fh:
        json.dump(summary_dict, fh, indent=2)
    print(f"[aggregate] Wrote summary JSON to {json_path}")

    if args.no_markdown:
        print("[aggregate] --no_markdown set; skipping table updates")
        return

    if not md_path.exists():
        raise FileNotFoundError(f"Cannot locate task_description.md at {md_path}")

    with md_path.open() as fh:
        md_content = fh.read()

    if "classification" in args.tables and classification_summary:
        md_content = upsert_row(md_content, CLASSIFICATION_TABLE_HINT, "Micro-F1", CLASSIFICATION_COLUMN_KEYS, classification_summary.micro)
        md_content = upsert_row(md_content, CLASSIFICATION_TABLE_HINT, "Macro-F1", CLASSIFICATION_COLUMN_KEYS, classification_summary.macro)
        print("[aggregate] Updated classification table")

    if "generation" in args.tables and generation_summary:
        md_content = upsert_row(md_content, GENERATION_TABLE_HINT, "Micro-F1", GENERATION_COLUMN_KEYS, generation_summary.micro)
        md_content = upsert_row(md_content, GENERATION_TABLE_HINT, "Macro-F1", GENERATION_COLUMN_KEYS, generation_summary.macro)
        print("[aggregate] Updated generation table")

    with md_path.open("w") as fh:
        fh.write(md_content)


if __name__ == "__main__":
    main()
