#!/usr/bin/env python3
"""Run the materials tokenization evaluation suite with modular table control.

Usage patterns for agents:

- Run the full pipeline (all subtasks, both tables):
    ./grading/grade.py
- Limit execution/updating to specific tables (choices: generation, classification):
    ./grading/grade.py --table generation
- Re-run just one subtask and refresh the affected table columns:
    ./grading/grade.py --subtask ner_sofc --table classification
- Inspect the intermediate metrics without touching markdown:
    ./grading/grade.py --table generation --aggregate-only

Every invocation writes results under grading/runs/<timestamp>/ and emits a
JSON summary before markdown updates so agents can programmatically inspect
metrics. Subtask identifiers are shared with downstream scripts; make sure your
training configs align (e.g. generate the correct dataset folder names).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set

# Subtask identifiers available via --subtask
CLASSIFICATION_SUBTASKS = {
    "ner_sofc",
    "ner_sofc_slot",
    "ner_matscholar",
    "rc",
    "pc",
}
CLASSIFICATION_DEFAULT_ORDER = [
    "ner_sofc",
    "ner_sofc_slot",
    "ner_matscholar",
    "rc",
    "pc",
]
GENERATION_SUBTASK = "generation"
ALL_SUBTASKS = CLASSIFICATION_SUBTASKS | {GENERATION_SUBTASK}

TABLE_CHOICES = {"generation", "classification", "all"}

NER_DATASET_METADATA = {
    "ner_sofc": {
        "dataset": "sofc",
        "folds": [1, 2, 3, 4, 5],
    },
    "ner_sofc_slot": {
        "dataset": "sofc_slot",
        "folds": [1, 2, 3, 4, 5],
    },
    "ner_matscholar": {
        "dataset": "matscholar",
        "folds": None,
    },
}

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_LM_LRS = ["2e-5", "3e-5", "5e-5"]
DEFAULT_METHOD = "YourMethod"
DEFAULT_MODEL_NAME = "MatSciBERT_100000"
DEFAULT_HF_MODEL = "allenai/scibert_scivocab_uncased"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the materials-tokenization task directory (defaults to repo layout)",
    )
    parser.add_argument(
        "--table",
        dest="tables",
        action="append",
        choices=sorted(TABLE_CHOICES),
        help="Tables to refresh (generation, classification, all). Default: both",
    )
    parser.add_argument(
        "--subtask",
        dest="subtasks",
        action="append",
        choices=sorted(ALL_SUBTASKS),
        help="Subtasks to execute. Defaults depend on tables (classification runs every NER/RC/PC).",
    )
    parser.add_argument(
        "--method",
        default=os.environ.get("METHOD_NAME", DEFAULT_METHOD),
        help="Method label for the markdown tables (default from METHOD_NAME env or 'YourMethod')",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Identifier consumed by classification scripts (MODEL_NAME)",
    )
    parser.add_argument(
        "--hf-model",
        default=os.environ.get("HF_MODEL", DEFAULT_HF_MODEL),
        help="HF backbone model id passed to evaluation scripts",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[int(s) for s in os.environ.get("SEEDS", " ".join(map(str, DEFAULT_SEEDS))).split()],
        help="Deterministic seed list (defaults to SEEDS env or 42 43 44)",
    )
    parser.add_argument(
        "--lm-lrs",
        nargs="+",
        default=os.environ.get("LM_LRS", " ".join(DEFAULT_LM_LRS)).split(),
        help="Learning-rate sweep used by classification scripts (LM_LRS env)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Override the timestamped run directory (under grading/runs by default)",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        help="Override predictions directory (defaults to <run-dir>/preds)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Override model artifacts directory (defaults to <run-dir>/models)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Override HF cache directory (defaults to <run-dir>/.cache)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write the intermediate grading summary JSON",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running subtasks; aggregate existing outputs into JSON/markdown",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Produce JSON summary only (skip task_description.md updates)",
    )
    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="Value to export as CUDA_VISIBLE_DEVICES for child processes",
    )
    return parser.parse_args(argv)


def resolve_tables(raw_tables: Iterable[str] | None) -> List[str]:
    if not raw_tables:
        return ["classification", "generation"]
    tables = []
    for entry in raw_tables:
        if entry == "all":
            return ["classification", "generation"]
        tables.append(entry)
    # preserve request order but ensure uniqueness
    seen: Set[str] = set()
    ordered: List[str] = []
    for table in tables:
        if table not in seen:
            ordered.append(table)
            seen.add(table)
    return ordered


def resolve_subtasks(raw: Iterable[str] | None, tables: Sequence[str]) -> List[str]:
    if raw:
        # maintain order, enforce uniqueness
        seen: Set[str] = set()
        ordered = []
        for item in raw:
            if item in ALL_SUBTASKS and item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    subtasks: List[str] = []
    if "classification" in tables:
        subtasks.extend(CLASSIFICATION_DEFAULT_ORDER)
    if "generation" in tables:
        subtasks.append(GENERATION_SUBTASK)
    return subtasks


def make_run_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    grading_dir = Path(__file__).resolve().parent
    if args.run_dir:
        run_dir = args.run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = grading_dir / "runs" / timestamp
    preds_dir = args.preds_dir or (run_dir / "preds")
    models_dir = args.models_dir or (run_dir / "models")
    cache_dir = args.cache_dir or (run_dir / ".cache")
    log_dir = run_dir / "logs"
    for path in (run_dir, preds_dir, models_dir, cache_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    return run_dir, preds_dir, models_dir, cache_dir, log_dir


def run_command(cmd: Sequence[str], cwd: Path, env: dict) -> None:
    cmd_display = " ".join(cmd)
    print(f"[grading] cd {cwd} && {cmd_display}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_ner_subtask(
    dataset_key: str,
    folds: List[int] | None,
    model_name: str,
    hf_model: str,
    lm_lrs: Sequence[str],
    seeds: Sequence[int],
    models_dir: Path,
    preds_dir: Path,
    cache_dir: Path,
    env: dict,
    eval_dir: Path,
    pretrained_path: str | None,
) -> None:
    ner_dir = eval_dir / "classification" / "ner"
    base_args = [
        "python",
        "-u",
        "ner.py",
        "--model_name",
        model_name,
        "--model_save_dir",
        str(models_dir / "ner"),
        "--preds_save_dir",
        str(preds_dir / "ner"),
        "--cache_dir",
        str(cache_dir),
        "--architecture",
        "bert-crf",
        "--dataset_name",
        dataset_key,
        "--hf_model_name",
        hf_model,
    ]
    if lm_lrs:
        base_args += ["--lm_lrs", *lm_lrs]
    if seeds:
        base_args += ["--seeds", *[str(s) for s in seeds]]
    if pretrained_path:
        base_args += ["--pretrained_path", pretrained_path]

    if folds:
        for fold in folds:
            cmd = [*base_args, "--fold_num", str(fold)]
            print(f"[grading][ner] dataset={dataset_key} fold={fold}")
            run_command(cmd, ner_dir, env)
    else:
        print(f"[grading][ner] dataset={dataset_key}")
        run_command(base_args, ner_dir, env)


def run_relation_classification(
    model_name: str,
    hf_model: str,
    lm_lrs: Sequence[str],
    seeds: Sequence[int],
    models_dir: Path,
    preds_dir: Path,
    cache_dir: Path,
    env: dict,
    eval_dir: Path,
    pretrained_path: str | None,
) -> None:
    rc_dir = eval_dir / "classification" / "relation_classification"
    cmd = [
        "python",
        "-u",
        "relation_classification.py",
        "--model_name",
        model_name,
        "--model_save_dir",
        str(models_dir / "relation_classification"),
        "--preds_save_dir",
        str(preds_dir / "relation_classification"),
        "--cache_dir",
        str(cache_dir),
        "--hf_model_name",
        hf_model,
    ]
    if lm_lrs:
        cmd += ["--lm_lrs", *lm_lrs]
    if seeds:
        cmd += ["--seeds", *[str(s) for s in seeds]]
    if pretrained_path:
        cmd += ["--pretrained_path", pretrained_path]
    run_command(cmd, rc_dir, env)


def run_paragraph_classification(
    hf_model: str,
    seeds: Sequence[int],
    models_dir: Path,
    preds_dir: Path,
    cache_dir: Path,
    env: dict,
    grading_dir: Path,
    eval_dir: Path,
) -> None:
    dataset_dir = eval_dir / "classification" / "cls" / "datasets" / "glass_non_glass"
    cmd = [
        "python",
        "-u",
        str(grading_dir / "run_pc.py"),
        "--dataset_dir",
        str(dataset_dir),
        "--output_dir",
        str(models_dir / "cls"),
        "--preds_dir",
        str(preds_dir / "cls"),
        "--cache_dir",
        str(cache_dir),
        "--model_name",
        hf_model,
        "--seeds",
        *[str(s) for s in seeds],
    ]
    run_command(cmd, grading_dir, env)


def run_generation(
    hf_model: str,
    seeds: Sequence[int],
    models_dir: Path,
    preds_dir: Path,
    cache_dir: Path,
    env: dict,
    eval_dir: Path,
) -> None:
    gen_dir = eval_dir / "generation"
    cmd = [
        "python",
        "-u",
        "main.py",
        "--basemodel",
        "WordPiece",
        "--hf_model_name",
        hf_model,
        "--output_dir",
        str(models_dir / "generation"),
        "--preds_dir",
        str(preds_dir / "generation"),
        "--cache_dir",
        str(cache_dir),
        "--seeds",
        *[str(s) for s in seeds],
    ]
    run_command(cmd, gen_dir, env)


def invoke_aggregator(
    grading_dir: Path,
    task_dir: Path,
    preds_dir: Path,
    method: str,
    seeds: Sequence[int],
    tables: Sequence[str],
    json_out: Path | None,
    update_markdown: bool,
) -> Path:
    aggregator = grading_dir / "aggregate_results.py"
    cmd = [
        "python",
        "-u",
        str(aggregator),
        "--task_dir",
        str(task_dir),
        "--preds_dir",
        str(preds_dir),
        "--method",
        method,
        "--seeds",
        *[str(s) for s in seeds],
        "--tables",
        *tables,
    ]
    if json_out:
        cmd += ["--json_out", str(json_out)]
    if not update_markdown:
        cmd.append("--no_markdown")
    run_command(cmd, grading_dir, os.environ.copy())
    return json_out or (preds_dir / ".." / "grading_summary.json").resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    task_dir = args.task_dir.resolve()
    grading_dir = Path(__file__).resolve().parent
    eval_dir = task_dir / "eval"

    tables = resolve_tables(args.tables)
    subtasks = resolve_subtasks(args.subtasks, tables)

    run_dir, preds_dir, models_dir, cache_dir, _ = make_run_dirs(args)

    env = os.environ.copy()
    if args.cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    pretrained = env.get("CKP_DST")

    print(f"[grading] Task dir: {task_dir}")
    print(f"[grading] Run dir: {run_dir}")
    print(f"[grading] Method: {args.method}")
    print(f"[grading] Tables: {tables}")
    print(f"[grading] Subtasks: {subtasks}")

    if not args.aggregate_only:
        for subtask in subtasks:
            if subtask in NER_DATASET_METADATA:
                meta = NER_DATASET_METADATA[subtask]
                run_ner_subtask(
                    dataset_key=meta["dataset"],
                    folds=meta["folds"],
                    model_name=args.model_name,
                    hf_model=args.hf_model,
                    lm_lrs=args.lm_lrs,
                    seeds=args.seeds,
                    models_dir=models_dir,
                    preds_dir=preds_dir,
                    cache_dir=cache_dir,
                    env=env,
                    eval_dir=eval_dir,
                    pretrained_path=pretrained,
                )
            elif subtask == "rc":
                run_relation_classification(
                    model_name=args.model_name,
                    hf_model=args.hf_model,
                    lm_lrs=args.lm_lrs,
                    seeds=args.seeds,
                    models_dir=models_dir,
                    preds_dir=preds_dir,
                    cache_dir=cache_dir,
                    env=env,
                    eval_dir=eval_dir,
                    pretrained_path=pretrained,
                )
            elif subtask == "pc":
                run_paragraph_classification(
                    hf_model=args.hf_model,
                    seeds=args.seeds,
                    models_dir=models_dir,
                    preds_dir=preds_dir,
                    cache_dir=cache_dir,
                    env=env,
                    grading_dir=grading_dir,
                    eval_dir=eval_dir,
                )
            elif subtask == GENERATION_SUBTASK:
                run_generation(
                    hf_model=args.hf_model,
                    seeds=args.seeds,
                    models_dir=models_dir,
                    preds_dir=preds_dir,
                    cache_dir=cache_dir,
                    env=env,
                    eval_dir=eval_dir,
                )
            else:
                raise ValueError(f"Unhandled subtask '{subtask}'")
    else:
        print("[grading] Skipping execution (--aggregate-only)")

    summary_path = args.json_out or (run_dir / "grading_summary.json")
    invoke_aggregator(
        grading_dir=grading_dir,
        task_dir=task_dir,
        preds_dir=preds_dir,
        method=args.method,
        seeds=args.seeds,
        tables=tables,
        json_out=summary_path,
        update_markdown=not args.no_markdown,
    )
    print(f"[grading] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
