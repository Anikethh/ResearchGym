#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    logs = here / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / "simplecil.log"
    cfg = here / "exps" / "simplecil.json"

    with open(log_file, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(
            ["python", str(here / "main.py"), "--config", str(cfg)],
            cwd=str(here),
            stdout=out,
            stderr=out,
        )
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())

