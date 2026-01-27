#!/usr/bin/env python3
"""
Grade interpreting-diffusion-transformers results and refresh task_description.md tables.

Usage examples:
  ./grading/grade.py --subtask pascal_single --results results.json
  ./grading/grade.py --subtask imagenet --results results.json --arch "Flux DiT"
  ./grading/grade.py --subtask pascal_multi --results results.json --no-markdown

Flags:
  --subtask {imagenet,pascal_single,pascal_multi}
  --results PATH            JSON file with metrics (see schema below)
  --md PATH                 Path to task_description.md (default: ../task_description.md)
  --arch NAME               Architecture label to match/update in the table (default: first "Your Method" row)
  --method NAME             Method name (default: "Your Method")
  --no-markdown             Skip writing back to markdown; just print summary
  --json-out PATH           Optional JSON summary output

Expected JSON schema (only the fields needed for the chosen subtask are required):
{
  "method": "Your Method",          # optional override
  "arch": "Flux DiT",               # optional override
  "imagenet": {"acc": 0.0, "miou": 0.0, "map": 0.0},
  "pascal_single": {"acc": 0.0, "miou": 0.0, "map": 0.0},
  "pascal_multi": {"acc": 0.0, "miou": 0.0}
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

ROW_SEP = "|"


def fmt_val(v: object) -> str:
    if v is None:
        return "---"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def parse_table(block: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in block:
        if not line.strip().startswith(ROW_SEP):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def serialize_table(rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(r) + " |" for r in rows]


def replace_row(rows: List[List[str]], header: List[str], target_idx: int, new_row: List[str]) -> List[List[str]]:
    return rows[:target_idx] + [new_row] + rows[target_idx + 1 :]


def find_table(lines: List[str], anchor: str) -> Optional[tuple[int, int]]:
    start = None
    end = None
    for i, line in enumerate(lines):
        if anchor.lower() in line.lower():
            # table starts after a blank line following anchor or immediately next line starting with |
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(ROW_SEP):
                j += 1
            start = j
            break
    if start is None:
        return None
    end = start
    while end < len(lines) and lines[end].strip().startswith(ROW_SEP):
        end += 1
    return start, end


def load_results(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_table(table_rows: List[List[str]], subtask: str, method: str, arch: Optional[str], metrics: Dict[str, float]) -> List[List[str]]:
    header = table_rows[0]
    data_rows = table_rows[2:] if len(table_rows) > 2 else []  # skip header + align row

    def match(row: List[str]) -> bool:
        if not row:
            return False
        if row[0].strip().lower() != method.lower():
            return False
        if arch is not None and len(row) > 1 and row[1].strip().lower() != arch.lower():
            return False
        return True

    idx = None
    for i, row in enumerate(data_rows):
        if match(row):
            idx = i
            break

    if subtask == "imagenet":
        cols = ["ImageNet Acc $\uparrow$", "ImageNet mIoU $\uparrow$", "ImageNet mAP $\uparrow$"]
    elif subtask == "pascal_single":
        cols = ["PascalVOC Acc $\uparrow$", "PascalVOC mIoU $\uparrow$", "PascalVOC mAP $\uparrow$"]
    else:  # pascal_multi
        cols = ["Acc $\uparrow$", "mIoU $\uparrow$"]

    header_to_idx = {h.strip(): j for j, h in enumerate(header)}

    def apply(row: List[str]) -> List[str]:
        row = row.copy()
        for k, name in zip(["acc", "miou", "map"], cols):
            if name not in header_to_idx:
                continue
            if k not in metrics:
                continue
            row[header_to_idx[name]] = fmt_val(metrics[k])
        return row

    if idx is not None:
        data_rows[idx] = apply(data_rows[idx])
    else:
        # build new row with placeholders
        new = ["---" for _ in header]
        new[0] = method
        if arch is not None and len(header) > 1:
            new[1] = arch
        new = apply(new)
        data_rows.append(new)

    return table_rows[:2] + data_rows if len(table_rows) > 2 else [header] + data_rows


def write_table(lines: List[str], span: tuple[int, int], new_rows: List[List[str]]) -> List[str]:
    start, end = span
    out = lines[:start] + serialize_table(new_rows) + lines[end:]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=["imagenet", "pascal_single", "pascal_multi"], required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--md", type=Path, default=Path(__file__).resolve().parent.parent / "task_description.md")
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--method", type=str, default="Your Method")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    res = load_results(args.results)
    method = res.get("method", args.method)
    arch = res.get("arch", args.arch)

    metrics_key = {
        "imagenet": "imagenet",
        "pascal_single": "pascal_single",
        "pascal_multi": "pascal_multi",
    }[args.subtask]
    metrics = res.get(metrics_key) or {}

    lines = args.md.read_text(encoding="utf-8").splitlines()

    anchor = "Table 1." if args.subtask in {"imagenet", "pascal_single"} else "Table 2."
    span = find_table(lines, anchor)
    if not span:
        raise RuntimeError(f"Could not find {anchor} in {args.md}")

    table_block = lines[span[0]: span[1]]
    table_rows = parse_table(table_block)
    if not table_rows:
        raise RuntimeError("Failed to parse markdown table")

    updated_rows = update_table(table_rows, args.subtask, method, arch, metrics)
    updated_lines = write_table(lines, span, updated_rows)

    summary = {"subtask": args.subtask, "method": method, "arch": arch, "metrics": metrics}

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_markdown:
        args.md.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()