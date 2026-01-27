"""Post-process Codex JSONL output to detect blocked content mentions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_text(event: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    for key in ("content", "message", "text"):
        val = event.get(key)
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    texts.append(item)

    payload = event.get("payload")
    if isinstance(payload, dict):
        for key in ("content", "message", "text"):
            val = payload.get(key)
            if isinstance(val, str):
                texts.append(val)

    return texts


def post_process_output(jsonl_path: Path, blocked_patterns: list[str]) -> dict[str, Any]:
    """Post-process JSONL to detect blocked content mentions."""
    violations: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return {"violations": violations, "clean": True}

    patterns = [p.lower() for p in blocked_patterns if p]
    if not patterns:
        return {"violations": violations, "clean": True}

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                event = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            for text in _extract_text(event):
                lower = text.lower()
                for pattern in patterns:
                    if pattern in lower:
                        violations.append({
                            "pattern": pattern,
                            "event_type": event.get("type"),
                            "snippet": text[:200],
                        })

    return {"violations": violations, "clean": len(violations) == 0}


def write_violations(log_dir: Path, violations: dict[str, Any]) -> Path:
    """Write violations to blocked_violations.json."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "blocked_violations.json"
    path.write_text(json.dumps(violations, indent=2))
    return path
