#!/usr/bin/env python3
"""Grade cross-modal retrieval runs by parsing evaluation outputs and updating task tables.

Usage examples for agents:

- Grade every table using default run discovery:
    ./grading/grade.py
- Limit to the Query-Shift tables only:
    ./grading/grade.py --table qs_image --table qs_text
- Inspect results without touching markdown:
    ./grading/grade.py --no-markdown --json-out /tmp/cross_modal_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


@dataclass
class MetricEntry:
    value: float
    severity: Optional[int]
    run_path: str


@dataclass(frozen=True)
class ColumnDefinition:
    key: str
    display: str
    metric: str


@dataclass(frozen=True)
class RowSpec:
    key: str
    group: str
    display_name: str
    variant: str
    category: str
    columns: Tuple[ColumnDefinition, ...]
    table_header_regex: str
    anchor_pattern: str


@dataclass
class RowResult:
    spec: RowSpec
    values: Dict[str, float]
    average: float
    sources: Dict[str, List[MetricEntry]]


@dataclass
class RunRecord:
    config_path: Path
    evaluate_path: Path
    config: Dict[str, object]
    metrics: Dict[str, float]
    category: str
    variant: str
    dataset: Optional[str] = None
    corruption_key: Optional[str] = None
    severity: Optional[int] = None


IMAGE_COLUMNS: Tuple[ColumnDefinition, ...] = (
    ColumnDefinition("gaussian_noise", "Gauss.", "txt_r1"),
    ColumnDefinition("shot_noise", "Shot", "txt_r1"),
    ColumnDefinition("impulse_noise", "Impul.", "txt_r1"),
    ColumnDefinition("speckle_noise", "Speckle", "txt_r1"),
    ColumnDefinition("defocus_blur", "Defoc.", "txt_r1"),
    ColumnDefinition("glass_blur", "Glass", "txt_r1"),
    ColumnDefinition("motion_blur", "Motion", "txt_r1"),
    ColumnDefinition("zoom_blur", "Zoom", "txt_r1"),
    ColumnDefinition("snow", "Snow", "txt_r1"),
    ColumnDefinition("frost", "Frost", "txt_r1"),
    ColumnDefinition("fog", "Fog", "txt_r1"),
    ColumnDefinition("brightness", "Brit.", "txt_r1"),
    ColumnDefinition("contrast", "Contr.", "txt_r1"),
    ColumnDefinition("elastic_transform", "Elastic", "txt_r1"),
    ColumnDefinition("pixelate", "Pixel", "txt_r1"),
    ColumnDefinition("jpeg_compression", "JPEG", "txt_r1"),
)

TEXT_COLUMNS: Tuple[ColumnDefinition, ...] = (
    ColumnDefinition("annotation_OcrAug", "OCR", "img_r1"),
    ColumnDefinition("annotation_RandomCharAug_insert", "CI", "img_r1"),
    ColumnDefinition("annotation_RandomCharAug_substitute", "CR", "img_r1"),
    ColumnDefinition("annotation_RandomCharAug_swap", "CS", "img_r1"),
    ColumnDefinition("annotation_RandomCharAug_delete", "CD", "img_r1"),
    ColumnDefinition("annotation_sr", "SR", "img_r1"),
    ColumnDefinition("annotation_ri", "RI", "img_r1"),
    ColumnDefinition("annotation_rs", "RS", "img_r1"),
    ColumnDefinition("annotation_rd", "RD", "img_r1"),
    ColumnDefinition("annotation_ip", "IP", "img_r1"),
    ColumnDefinition("annotation_formal", "Formal", "img_r1"),
    ColumnDefinition("annotation_casual", "Casual", "img_r1"),
    ColumnDefinition("annotation_passive", "Passive", "img_r1"),
    ColumnDefinition("annotation_active", "Active", "img_r1"),
    ColumnDefinition("annotation_back_trans", "Backtrans", "img_r1"),
)

QGS_COLUMNS: Tuple[ColumnDefinition, ...] = (
    ColumnDefinition("flickr:txt_r1", "Base2Flickr I2TR@1", "txt_r1"),
    ColumnDefinition("flickr:img_r1", "Base2Flickr T2IR@1", "img_r1"),
    ColumnDefinition("coco:txt_r1", "Base2COCO I2TR@1", "txt_r1"),
    ColumnDefinition("coco:img_r1", "Base2COCO T2IR@1", "img_r1"),
    ColumnDefinition("fashion:txt_r1", "Base2Fashion I2TR@1", "txt_r1"),
    ColumnDefinition("fashion:img_r1", "Base2Fashion T2IR@1", "img_r1"),
    ColumnDefinition("nocaps_id:txt_r1", "Base2Nocaps(ID) I2TR@1", "txt_r1"),
    ColumnDefinition("nocaps_id:img_r1", "Base2Nocaps(ID) T2IR@1", "img_r1"),
    ColumnDefinition("nocaps_nd:txt_r1", "Base2Nocaps(ND) I2TR@1", "txt_r1"),
    ColumnDefinition("nocaps_nd:img_r1", "Base2Nocaps(ND) T2IR@1", "img_r1"),
    ColumnDefinition("nocaps_od:txt_r1", "Base2Nocaps(OD) I2TR@1", "txt_r1"),
    ColumnDefinition("nocaps_od:img_r1", "Base2Nocaps(OD) T2IR@1", "img_r1"),
)

REID_COLUMNS: Tuple[ColumnDefinition, ...] = (
    ColumnDefinition("icfg", "CUHK2ICFG T2IR@1", "img_r1"),
    ColumnDefinition("cuhk", "ICFG2CUHK T2IR@1", "img_r1"),
)

TABLE1_HEADER_REGEX = r"(\| Query Shift \| Gauss\..*?\| Avg\. \|\n\|---\|.*?\n)"
TABLE2_HEADER_REGEX = r"(\| Query Shift \| OCR \| CI .*?\| Avg\. \|\n\|---\|.*?\n)"
TABLE3_HEADER_REGEX = r"(\| Query Shift \| Base2Flickr I2TR@1 .*?\| Avg\. \|\n\|---\|.*?\n)"
TABLE4_HEADER_REGEX = r"(\| Query Shift \| CUHK2ICFG T2IR@1 .*?\| Avg\. \|\n\|---\|.*?\n)"

ALL_ROW_ORDER: List[str] = [
    "qs_image_blip_base",
    "qs_image_blip_large",
    "qs_text_blip_base",
    "qs_text_blip_large",
    "qgs_clip_base",
    "qgs_blip_base",
    "reid_clip",
]

ROW_SPECS: Dict[str, RowSpec] = {
    "qs_image_blip_base": RowSpec(
        key="qs_image_blip_base",
        group="qs_image",
        display_name="Table 1 · BLIP ViT-B/16",
        variant="blip_base",
        category="qs_image",
        columns=IMAGE_COLUMNS,
        table_header_regex=TABLE1_HEADER_REGEX,
        anchor_pattern="| **BLIP ViT-B/16** |",
    ),
    "qs_image_blip_large": RowSpec(
        key="qs_image_blip_large",
        group="qs_image",
        display_name="Table 1 · BLIP ViT-L/16",
        variant="blip_large",
        category="qs_image",
        columns=IMAGE_COLUMNS,
        table_header_regex=TABLE1_HEADER_REGEX,
        anchor_pattern="| **BLIP ViT-L/16** |",
    ),
    "qs_text_blip_base": RowSpec(
        key="qs_text_blip_base",
        group="qs_text",
        display_name="Table 2 · BLIP ViT-B/16",
        variant="blip_base",
        category="qs_text",
        columns=TEXT_COLUMNS,
        table_header_regex=TABLE2_HEADER_REGEX,
        anchor_pattern="| **BLIP ViT-B/16** |",
    ),
    "qs_text_blip_large": RowSpec(
        key="qs_text_blip_large",
        group="qs_text",
        display_name="Table 2 · BLIP ViT-L/16",
        variant="blip_large",
        category="qs_text",
        columns=TEXT_COLUMNS,
        table_header_regex=TABLE2_HEADER_REGEX,
        anchor_pattern="| **BLIP ViT-L/16** |",
    ),
    "qgs_clip_base": RowSpec(
        key="qgs_clip_base",
        group="qgs",
        display_name="Table 3 · CLIP ViT-B/16",
        variant="clip",
        category="qgs",
        columns=QGS_COLUMNS,
        table_header_regex=TABLE3_HEADER_REGEX,
        anchor_pattern="| **CLIP ViT-B/16** |",
    ),
    "qgs_blip_base": RowSpec(
        key="qgs_blip_base",
        group="qgs",
        display_name="Table 3 · BLIP ViT-B/16",
        variant="blip_base",
        category="qgs",
        columns=QGS_COLUMNS,
        table_header_regex=TABLE3_HEADER_REGEX,
        anchor_pattern="| **BLIP ViT-B/16** |",
    ),
    "reid_clip": RowSpec(
        key="reid_clip",
        group="reid",
        display_name="Table 4 · CLIP ViT-B/16",
        variant="clip_reid",
        category="reid",
        columns=REID_COLUMNS,
        table_header_regex=TABLE4_HEADER_REGEX,
        anchor_pattern="| **CLIP ViT-B/16** |",
    ),
}

TABLE_GROUPS: Dict[str, List[str]] = {
    "qs_image": ["qs_image_blip_base", "qs_image_blip_large"],
    "qs_text": ["qs_text_blip_base", "qs_text_blip_large"],
    "qgs": ["qgs_clip_base", "qgs_blip_base"],
    "reid": ["reid_clip"],
    "all": list(ALL_ROW_ORDER),
}

QGS_DATASET_ALIASES = {
    "flickr": "flickr",
    "coco": "coco",
    "fashion_gen": "fashion",
    "fashion": "fashion",
    "fashion_gen_detail": "fashion",
    "fashion_gen_detail_corrupt": "fashion",
    "fashion_gen_detail_full": "fashion",
    "nocaps_in_domain": "nocaps_id",
    "nocaps_near_domain": "nocaps_nd",
    "nocaps_out_domain": "nocaps_od",
}

REID_DATASET_ALIASES = {
    "icfg_pedes": "icfg",
    "cuhk_pedes": "cuhk",
}


class RunAggregator:
    def __init__(self) -> None:
        self.qs_image: Dict[str, Dict[str, Dict[int, List[MetricEntry]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.qs_text: Dict[str, Dict[str, Dict[int, List[MetricEntry]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.qgs_txt: Dict[str, Dict[str, List[MetricEntry]]] = defaultdict(lambda: defaultdict(list))
        self.qgs_img: Dict[str, Dict[str, List[MetricEntry]]] = defaultdict(lambda: defaultdict(list))
        self.reid: Dict[str, List[MetricEntry]] = defaultdict(list)
        self.row_missing: Dict[str, List[str]] = defaultdict(list)
        self.warnings: List[str] = []
        self.ingested: int = 0

    def add(self, record: RunRecord) -> bool:
        if record.category == "qs_image":
            if record.corruption_key is None or record.severity is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: missing corruption metadata for QS image"
                )
                return False
            value = record.metrics.get("txt_r1")
            if value is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: txt_r1 metric not found"
                )
                return False
            entry = MetricEntry(value=float(value), severity=record.severity, run_path=str(record.evaluate_path))
            self.qs_image[record.variant][record.corruption_key][record.severity].append(entry)
            self.ingested += 1
            return True

        if record.category == "qs_text":
            if record.corruption_key is None or record.severity is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: missing corruption metadata for QS text"
                )
                return False
            value = record.metrics.get("img_r1")
            if value is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: img_r1 metric not found"
                )
                return False
            entry = MetricEntry(value=float(value), severity=record.severity, run_path=str(record.evaluate_path))
            self.qs_text[record.variant][record.corruption_key][record.severity].append(entry)
            self.ingested += 1
            return True

        if record.category == "qgs":
            if record.dataset is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: dataset metadata missing for QGS"
                )
                return False
            txt_value = record.metrics.get("txt_r1")
            img_value = record.metrics.get("img_r1")
            if txt_value is not None:
                self.qgs_txt[record.variant][record.dataset].append(
                    MetricEntry(value=float(txt_value), severity=None, run_path=str(record.evaluate_path))
                )
            if img_value is not None:
                self.qgs_img[record.variant][record.dataset].append(
                    MetricEntry(value=float(img_value), severity=None, run_path=str(record.evaluate_path))
                )
            if txt_value is None and img_value is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: neither txt_r1 nor img_r1 present"
                )
                return False
            self.ingested += 1
            return True

        if record.category == "reid":
            if record.dataset is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: dataset metadata missing for ReID"
                )
                return False
            value = record.metrics.get("img_r1")
            if value is None:
                self.warnings.append(
                    f"Skipping {record.evaluate_path}: img_r1 metric not found"
                )
                return False
            self.reid[record.dataset].append(
                MetricEntry(value=float(value), severity=None, run_path=str(record.evaluate_path))
            )
            self.ingested += 1
            return True

        self.warnings.append(
            f"Skipping {record.evaluate_path}: unhandled category '{record.category}'"
        )
        return False

    def build_rows(self, row_keys: Sequence[str]) -> List[RowResult]:
        results: List[RowResult] = []
        for row_key in row_keys:
            spec = ROW_SPECS[row_key]
            builder = {
                "qs_image": self._build_qs_image,
                "qs_text": self._build_qs_text,
                "qgs": self._build_qgs,
                "reid": self._build_reid,
            }.get(spec.category)
            if builder is None:
                self.warnings.append(f"No builder registered for category '{spec.category}'")
                continue
            result = builder(spec)
            if result is not None:
                results.append(result)
        return results

    def _build_qs_image(self, spec: RowSpec) -> Optional[RowResult]:
        variant_data = self.qs_image.get(spec.variant)
        if not variant_data:
            self.row_missing[spec.key] = ["no runs"]
            return None
        values: Dict[str, float] = {}
        sources: Dict[str, List[MetricEntry]] = {}
        missing: List[str] = []
        for col in spec.columns:
            corruption = variant_data.get(col.key)
            if not corruption:
                missing.append(col.display)
                continue
            best_severity = max(corruption)
            entries = corruption[best_severity]
            metric_values = [entry.value for entry in entries]
            if not metric_values:
                missing.append(col.display)
                continue
            values[col.key] = statistics.fmean(metric_values)
            sources[col.display] = entries
        if missing:
            self.row_missing[spec.key] = missing
            return None
        self.row_missing.pop(spec.key, None)
        average = statistics.fmean(values.values())
        return RowResult(spec=spec, values=values, average=average, sources=sources)

    def _build_qs_text(self, spec: RowSpec) -> Optional[RowResult]:
        variant_data = self.qs_text.get(spec.variant)
        if not variant_data:
            self.row_missing[spec.key] = ["no runs"]
            return None
        values: Dict[str, float] = {}
        sources: Dict[str, List[MetricEntry]] = {}
        missing: List[str] = []
        for col in spec.columns:
            corruption = variant_data.get(col.key)
            if not corruption:
                missing.append(col.display)
                continue
            best_severity = max(corruption)
            entries = corruption[best_severity]
            metric_values = [entry.value for entry in entries]
            if not metric_values:
                missing.append(col.display)
                continue
            values[col.key] = statistics.fmean(metric_values)
            sources[col.display] = entries
        if missing:
            self.row_missing[spec.key] = missing
            return None
        self.row_missing.pop(spec.key, None)
        average = statistics.fmean(values.values())
        return RowResult(spec=spec, values=values, average=average, sources=sources)

    def _build_qgs(self, spec: RowSpec) -> Optional[RowResult]:
        variant_txt = self.qgs_txt.get(spec.variant, {})
        variant_img = self.qgs_img.get(spec.variant, {})
        if not variant_txt and not variant_img:
            self.row_missing[spec.key] = ["no runs"]
            return None
        values: Dict[str, float] = {}
        sources: Dict[str, List[MetricEntry]] = {}
        missing: List[str] = []
        for col in spec.columns:
            try:
                dataset_key, metric_name = col.key.split(":", 1)
            except ValueError:
                self.warnings.append(f"Invalid column key '{col.key}' for QGS table")
                continue
            if metric_name == "txt_r1":
                entries = variant_txt.get(dataset_key)
            else:
                entries = variant_img.get(dataset_key)
            if not entries:
                missing.append(col.display)
                continue
            values[col.key] = statistics.fmean(entry.value for entry in entries)
            sources[col.display] = entries
        if missing:
            self.row_missing[spec.key] = missing
            return None
        self.row_missing.pop(spec.key, None)
        average = statistics.fmean(values.values())
        return RowResult(spec=spec, values=values, average=average, sources=sources)

    def _build_reid(self, spec: RowSpec) -> Optional[RowResult]:
        values: Dict[str, float] = {}
        sources: Dict[str, List[MetricEntry]] = {}
        missing: List[str] = []
        for col in spec.columns:
            entries = self.reid.get(col.key)
            if not entries:
                missing.append(col.display)
                continue
            values[col.key] = statistics.fmean(entry.value for entry in entries)
            sources[col.display] = entries
        if missing:
            self.row_missing[spec.key] = missing
            return None
        self.row_missing.pop(spec.key, None)
        average = statistics.fmean(values.values())
        return RowResult(spec=spec, values=values, average=average, sources=sources)


def format_metric(value: float) -> str:
    return f"{value:.1f}"


def parse_metrics(path: Path) -> Optional[Dict[str, float]]:
    lines: List[str]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    records: List[Dict[str, float]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            filtered: Dict[str, float] = {}
            for key, value in payload.items():
                try:
                    filtered[key] = float(value)
                except (TypeError, ValueError):
                    continue
            records.append(filtered)
    if not records:
        return None
    return records[-1]


def normalize_variant(config: Dict[str, object]) -> str:
    backbone = str(config.get("backbone", "")).lower()
    vit = str(config.get("vit", "")).lower()
    if backbone == "blip":
        if vit in {"large", "base"}:
            return f"blip_{vit}"
        return "blip_base"
    if backbone == "clip":
        return f"clip_{vit}" if vit else "clip"
    if backbone == "clip_reid":
        return "clip_reid"
    return backbone or "unknown"


def parse_coco_ip(image_root: str) -> Tuple[Optional[str], Optional[int]]:
    name = Path(image_root).name.lower()
    if not name:
        return None, None
    if name.startswith("coco_ip_"):
        name = name[len("coco_ip_") :]
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None, None
    corruption, severity_str = parts
    try:
        severity = int(severity_str)
    except ValueError:
        return None, None
    return corruption, severity


def parse_coco_tp(ann_root: str) -> Tuple[Optional[str], Optional[int]]:
    trimmed = ann_root.rstrip("/\\")
    if not trimmed:
        return None, None
    ann_path = Path(trimmed)
    leaf = ann_path.name
    parent = ann_path.parent.name if ann_path.parent else ""
    if leaf.isdigit():
        try:
            severity = int(leaf)
        except ValueError:
            severity = None
        corruption = parent
    else:
        severity = None
        corruption = leaf
    return corruption or None, severity


def normalize_qgs_dataset(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = QGS_DATASET_ALIASES.get(str(raw).lower())
    return key


def normalize_reid_dataset(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return REID_DATASET_ALIASES.get(str(raw).lower())


def build_run_record(config_path: Path, evaluate_path: Path, config: Dict[str, object], metrics: Dict[str, float]) -> Optional[RunRecord]:
    variant = normalize_variant(config)
    image_root = str(config.get("image_root") or "")
    ann_root = str(config.get("ann_root") or "")
    dataset_raw = config.get("dataset")
    dataset = str(dataset_raw) if dataset_raw is not None else None

    lower_image = image_root.lower()
    lower_ann = ann_root.lower()

    if "coco-ip" in lower_image and dataset == "coco":
        corruption, severity = parse_coco_ip(image_root)
        if corruption is None or severity is None:
            return None
        if not variant.startswith("blip_"):
            return None
        return RunRecord(
            config_path=config_path,
            evaluate_path=evaluate_path,
            config=config,
            metrics=metrics,
            category="qs_image",
            variant=variant,
            dataset="coco",
            corruption_key=corruption,
            severity=severity,
        )

    if "coco-tp" in lower_ann and dataset == "coco":
        corruption, severity = parse_coco_tp(ann_root)
        if corruption is None or severity is None:
            return None
        if not variant.startswith("blip_"):
            return None
        return RunRecord(
            config_path=config_path,
            evaluate_path=evaluate_path,
            config=config,
            metrics=metrics,
            category="qs_text",
            variant=variant,
            dataset="coco",
            corruption_key=corruption,
            severity=severity,
        )

    if variant == "clip_reid":
        dataset_key = normalize_reid_dataset(dataset)
        if dataset_key is None:
            return None
        return RunRecord(
            config_path=config_path,
            evaluate_path=evaluate_path,
            config=config,
            metrics=metrics,
            category="reid",
            variant=variant,
            dataset=dataset_key,
        )

    dataset_key = normalize_qgs_dataset(dataset)
    if dataset_key is not None:
        return RunRecord(
            config_path=config_path,
            evaluate_path=evaluate_path,
            config=config,
            metrics=metrics,
            category="qgs",
            variant=variant,
            dataset=dataset_key,
        )

    return None


def discover_runs(roots: Sequence[Path]) -> Tuple[List[RunRecord], List[str]]:
    records: List[RunRecord] = []
    warnings: List[str] = []
    for root in roots:
        if not root.exists():
            warnings.append(f"Search root {root} does not exist; skipping.")
            continue
        for evaluate_path in root.rglob("evaluate.txt"):
            config_path = evaluate_path.with_name("config.yaml")
            if not config_path.exists():
                warnings.append(f"Missing config.yaml alongside {evaluate_path}; skipping run.")
                continue
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to load {config_path}: {exc}")
                continue
            if not isinstance(config, dict):
                warnings.append(f"Unexpected config format in {config_path}; skipping.")
                continue
            metrics = parse_metrics(evaluate_path)
            if not metrics:
                warnings.append(f"No metrics parsed from {evaluate_path}; skipping.")
                continue
            record = build_run_record(config_path, evaluate_path, config, metrics)
            if record is None:
                warnings.append(f"Unrecognized run layout for {evaluate_path}; skipping.")
                continue
            records.append(record)
    return records, warnings


def resolve_row_keys(raw_tables: Optional[Sequence[str]]) -> List[str]:
    if not raw_tables:
        raw_tables = ["all"]
    resolved: List[str] = []
    for entry in raw_tables:
        key = entry.lower()
        if key == "all":
            for row in ALL_ROW_ORDER:
                if row not in resolved:
                    resolved.append(row)
            continue
        if key in TABLE_GROUPS:
            for row in TABLE_GROUPS[key]:
                if row not in resolved:
                    resolved.append(row)
            continue
        if key in ROW_SPECS and key not in resolved:
            resolved.append(key)
            continue
        raise ValueError(f"Unknown table identifier '{entry}'.")
    return resolved


def format_row(result: RowResult, method_label: str) -> str:
    cells = [method_label]
    for col in result.spec.columns:
        value = result.values.get(col.key)
        cells.append(format_metric(value) if value is not None else "--")
    cells.append(format_metric(result.average))
    return "| " + " | ".join(cells) + " |"


def replace_table_row(md: str, spec: RowSpec, new_row: str) -> Tuple[str, bool]:
    pattern = re.compile(spec.table_header_regex, re.IGNORECASE | re.DOTALL)
    match = pattern.search(md)
    if not match:
        return md, False
    header_end = match.end(1)
    remainder = md[header_end:]
    leading_gap_len = 0
    while leading_gap_len < len(remainder) and remainder[leading_gap_len] in "\n\r":
        leading_gap_len += 1
    table_start = header_end + leading_gap_len
    pos = table_start
    while pos < len(md) and md[pos] == "|":
        newline = md.find("\n", pos)
        if newline == -1:
            pos = len(md)
            break
        pos = newline + 1
    table_end = pos
    table_block = md[table_start:table_end]
    line_ending = table_block.endswith("\n")
    lines = table_block.splitlines()
    anchor_lower = spec.anchor_pattern.lower()
    anchor_index = None
    for idx, line in enumerate(lines):
        if anchor_lower in line.lower():
            anchor_index = idx
    if anchor_index is None:
        return md, False
    for idx in range(anchor_index + 1, len(lines)):
        if lines[idx].strip().lower().startswith("| your method |"):
            if re.search(r"\d", lines[idx]):
                lines.insert(idx + 1, new_row)
            else:
                lines[idx] = new_row
            break
    else:
        insertion_point = anchor_index + 1
        lines.insert(insertion_point, new_row)
    rebuilt = "\n".join(lines) + ("\n" if line_ending else "")
    new_md = md[:table_start] + rebuilt + md[table_end:]
    return new_md, True


def build_summary(
    results: Sequence[RowResult],
    method_label: str,
    warnings: Sequence[str],
    missing: Dict[str, List[str]],
    ingested_runs: int,
    searched_roots: Sequence[Path],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "method": method_label,
        "ingested_runs": ingested_runs,
        "roots": [str(p) for p in searched_roots],
        "rows": [],
    }
    if warnings:
        payload["warnings"] = list(warnings)
    if missing:
        payload["missing"] = {row: cols for row, cols in missing.items() if cols}
    for result in results:
        columns_payload: List[Dict[str, object]] = []
        for col in result.spec.columns:
            value = result.values.get(col.key)
            column_entry: Dict[str, object] = {
                "id": col.key,
                "label": col.display,
                "value": value,
            }
            source_entries = []
            for entry in result.sources.get(col.display, []):
                source_entries.append(
                    {
                        "value": entry.value,
                        "severity": entry.severity,
                        "path": entry.run_path,
                    }
                )
            if source_entries:
                column_entry["sources"] = source_entries
            columns_payload.append(column_entry)
        payload["rows"].append(
            {
                "row": result.spec.key,
                "display": result.spec.display_name,
                "average": result.average,
                "columns": columns_payload,
            }
        )
    return payload


def update_markdown(md_path: Path, results: Sequence[RowResult], method_label: str) -> bool:
    try:
        md_text = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot update markdown; {md_path} does not exist") from None
    changed = False
    current_text = md_text
    for result in results:
        row_text = format_row(result, method_label)
        current_text, updated = replace_table_row(current_text, result.spec, row_text)
        changed = changed or updated
    if changed:
        md_path.write_text(current_text, encoding="utf-8")
    return changed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade cross-modal retrieval results")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the cross-modal-retrieval task directory",
    )
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Root directory to search for evaluate.txt files (repeatable)",
    )
    parser.add_argument(
        "--table",
        dest="tables",
        action="append",
        help="Tables to refresh (choices: qs_image, qs_text, qgs, reid, all, or row keys)",
    )
    parser.add_argument(
        "--method",
        default=os.environ.get("METHOD_NAME", "Your Method"),
        help="Label to use in the markdown tables",
    )
    parser.add_argument(
        "--md",
        type=Path,
        help="Override path to task_description.md",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write the JSON summary",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip updating task_description.md",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    task_dir = args.task_dir.resolve()
    md_path = args.md.resolve() if args.md else task_dir / "task_description.md"
    if args.roots:
        roots = [Path(p).resolve() for p in args.roots]
    else:
        roots = [task_dir / "output"]
    row_keys = resolve_row_keys(args.tables)
    records, discovery_warnings = discover_runs(roots)
    aggregator = RunAggregator()
    for record in records:
        aggregator.add(record)
    results = aggregator.build_rows(row_keys)
    missing = {row: cols for row, cols in aggregator.row_missing.items() if cols}
    summary = build_summary(
        results,
        args.method,
        [*discovery_warnings, *aggregator.warnings],
        missing,
        aggregator.ingested,
        roots,
    )

    if args.json_out:
        json_path = args.json_out
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        json_path = task_dir / "grading" / "runs" / timestamp / "summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote JSON summary to {json_path}")

    if missing:
        for row_key, cols in missing.items():
            spec = ROW_SPECS[row_key]
            print(f"Missing data for {spec.display_name}: {', '.join(cols)}")

    if not args.no_markdown and results:
        if update_markdown(md_path, results, args.method):
            print(f"Updated {md_path}")
        else:
            print("Markdown did not require changes.")
    elif args.no_markdown:
        print("Skipped markdown update per --no-markdown")
    else:
        print("No results available to update markdown.")


if __name__ == "__main__":
    main()
