from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Observation:
    message: str
    info: Dict[str, Any] | None = None


class AgenticEnv:
    """
    Minimal gym-like environment for agentic systems operating on research tasks.

    Contract (stable, minimal for v0):
    - reset(task_dir): initializes an isolated working directory for the agent to modify
    - step(action): accepts dict-based actions; must handle at least {"type": "finish", "payload": {...}}
    - workspace_dir: property returning the mutable workspace path for the agent
    - logs_dir: property returning where environment logs should be written
    """

    def __init__(self, base_runs_dir: Path) -> None:
        self._base_runs_dir = Path(base_runs_dir)
        self._run_dir: Optional[Path] = None
        self._workspace_dir: Optional[Path] = None
        self._logs_dir: Optional[Path] = None

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            raise RuntimeError("Environment not reset")
        return self._run_dir

    @property
    def workspace_dir(self) -> Path:
        if self._workspace_dir is None:
            raise RuntimeError("Environment not reset")
        return self._workspace_dir

    @property
    def logs_dir(self) -> Path:
        if self._logs_dir is None:
            raise RuntimeError("Environment not reset")
        return self._logs_dir

    def reset(self, task_dir: Path, run_group: str, run_id: str) -> Observation:
        task_dir = Path(task_dir).resolve()
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        self._run_dir = self._base_runs_dir / run_group / run_id
        self._workspace_dir = self._run_dir / "workspace"
        self._logs_dir = self._run_dir / "logs"
        for d in [self._run_dir, self._workspace_dir, self._logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # standard submissions folder (PaperBench-style)
        (self._run_dir / "submissions").mkdir(parents=True, exist_ok=True)

        # Record minimal metadata now; richer metadata will be appended by the runner
        meta = {
            "task_dir": str(task_dir),
            "run_group": run_group,
            "run_id": run_id,
        }
        (self._run_dir / "metadata.json").write_text(__import__("json").dumps(meta, indent=2))
        # Initialize status
        (self._run_dir / "status.json").write_text(__import__("json").dumps({"status": "created"}, indent=2))

        return Observation(message="reset", info={"run_dir": str(self._run_dir)})

    def step(self, action: Dict[str, Any]) -> Observation:
        if action.get("type") == "finish":
            return Observation(message="finished", info=action.get("payload", {}))
        return Observation(message="noop", info=action)


