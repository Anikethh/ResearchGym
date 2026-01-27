from __future__ import annotations

import json
from typing import Optional

from inspect_ai.tool import Tool, tool

from evo_database import record_candidate, load_db, summarize


@tool(name="record_candidate")
def record_candidate_tool(timeout: int | None = None, user: str | None = None) -> Tool:
    """Record an evaluated candidate in the evolution database.

    Parameters:
      metrics_json (str): JSON object of metrics (e.g., {"score": 0.75}).
      note (str, optional): Short note about the change.
      parent_id (str, optional): Parent candidate id.
    Returns: Summary with candidate id and best id.
    """

    async def execute(metrics_json: str, note: str = "", parent_id: Optional[str] = None) -> str:
        try:
            metrics = json.loads(metrics_json)
            if not isinstance(metrics, dict):
                return "error: metrics_json must be a JSON object"
        except Exception as exc:
            return f"error: invalid metrics_json ({exc})"
        cand = record_candidate(metrics=metrics, note=note, parent_id=parent_id)
        candidates, state = load_db()
        summary = summarize(candidates, state)
        summary.update({"candidate_id": cand.id})
        return json.dumps(summary, indent=2)

    return execute


@tool(name="list_candidates")
def list_candidates_tool(timeout: int | None = None, user: str | None = None) -> Tool:
    """List candidates recorded in the evolution database."""

    async def execute() -> str:
        cands, state = load_db()
        rows = []
        for c in cands[-50:]:
            rows.append({
                "id": c.id,
                "parent_id": c.parent_id,
                "metrics": c.metrics,
                "note": c.note,
                "created_at": c.created_at,
            })
        out = {"candidates": rows, **summarize(cands, state)}
        return json.dumps(out, indent=2)

    return execute


@tool(name="best_candidate")
def best_candidate_tool(timeout: int | None = None, user: str | None = None) -> Tool:
    """Return the current best candidate summary."""

    async def execute() -> str:
        cands, state = load_db()
        best = None
        if state.best_id:
            best = next((c for c in cands if c.id == state.best_id), None)
        payload = summarize(cands, state)
        if best:
            payload["best"] = {
                "id": best.id,
                "metrics": best.metrics,
                "note": best.note,
                "created_at": best.created_at,
            }
        return json.dumps(payload, indent=2)

    return execute
