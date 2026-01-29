from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DB_NAME = "db.jsonl"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class Candidate:
    id: str
    parent_id: Optional[str]
    note: str
    metrics: Dict[str, float]
    created_at: str

@dataclass
class EvoState:
    best_id: Optional[str]
    best_metric: Optional[float]
    primary_metric: str


def _db_dir_from_env() -> Path:
    root = Path(os.environ.get("CODE_DIR", ".")).resolve()
    path = Path(os.environ.get("RG_EVO_DB_PATH", root / ".rg_evo" / DEFAULT_DB_NAME))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _primary_metric() -> str:
    return os.environ.get("RG_EVO_PRIMARY_METRIC", "score")


def load_db(db_path: Optional[Path] = None) -> tuple[List[Candidate], EvoState]:
    path = db_path or _db_dir_from_env()
    candidates: List[Candidate] = []
    best_id: Optional[str] = None
    best_metric: Optional[float] = None
    primary = _primary_metric()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "type" in obj and obj.get("type") == "state":
                    best_id = obj.get("best_id")
                    best_metric = obj.get("best_metric")
                    primary = obj.get("primary_metric", primary)
                    continue
                candidates.append(
                    Candidate(
                        id=obj["id"],
                        parent_id=obj.get("parent_id"),
                        note=obj.get("note", ""),
                        metrics=obj.get("metrics", {}),
                        created_at=obj.get("created_at", _now_iso()),
                    )
                )
            except Exception:
                continue
    return candidates, EvoState(best_id=best_id, best_metric=best_metric, primary_metric=primary)


def _persist(candidates: List[Candidate], state: EvoState, db_path: Optional[Path] = None) -> None:
    path = db_path or _db_dir_from_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(c)) for c in candidates]
    lines.append(
        json.dumps({
            "type": "state",
            "best_id": state.best_id,
            "best_metric": state.best_metric,
            "primary_metric": state.primary_metric,
            "updated_at": _now_iso(),
        })
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_candidate(metrics: Dict[str, float], note: str = "", parent_id: Optional[str] = None, db_path: Optional[Path] = None) -> Candidate:
    candidates, state = load_db(db_path)
    cid = uuid.uuid4().hex
    cand = Candidate(id=cid, parent_id=parent_id, note=note, metrics=metrics, created_at=_now_iso())
    candidates.append(cand)
    primary = state.primary_metric or _primary_metric()
    metric_value = metrics.get(primary)
    if metric_value is not None:
        if state.best_metric is None or metric_value > state.best_metric:
            state.best_metric = metric_value
            state.best_id = cid
    _persist(candidates, state, db_path)
    return cand


def summarize(candidates: List[Candidate], state: EvoState) -> Dict:
    return {
        "count": len(candidates),
        "primary_metric": state.primary_metric,
        "best_id": state.best_id,
        "best_metric": state.best_metric,
    }
