from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def setup_file_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_file) for h in logger.handlers):
        logger.addHandler(fh)
    return logger


class CostTracker:
    def __init__(self, out_path: Path) -> None:
        self._path = out_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.totals: Dict[str, Any] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "by_model": {},
        }
        self._flush()

    def add(self, model: str, in_tokens: int, out_tokens: int) -> None:
        self.totals["requests"] += 1
        self.totals["input_tokens"] += int(in_tokens)
        self.totals["output_tokens"] += int(out_tokens)
        m = self.totals["by_model"].setdefault(model, {"requests": 0, "input_tokens": 0, "output_tokens": 0})
        m["requests"] += 1
        m["input_tokens"] += int(in_tokens)
        m["output_tokens"] += int(out_tokens)
        self._flush()

    def _flush(self) -> None:
        self._path.write_text(json.dumps(self.totals, indent=2))


