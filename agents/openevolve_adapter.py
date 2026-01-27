from __future__ import annotations

"""Adapter wiring the vendored OpenEvolve agent into ResearchGym."""

import json
import os
import shutil
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger


@dataclass
class OpenEvolveConfig:
    """Configuration required to launch an OpenEvolve run."""

    task_id: str
    model: str
    api_base: str
    time_limit_hours: float
    budget_limit: float
    max_iterations: int = 100
    checkpoint_interval: int = 25
    evaluator_timeout: int = 10800
    parallel_evaluations: int = 1


class OpenEvolveAdapter:
    """Bridge between ResearchGym and the vendored OpenEvolve project."""

    INITIAL_FILENAME = "openevolve_initial.py"
    EVALUATOR_FILENAME = "openevolve_evaluator.py"
    CONFIG_FILENAME = "openevolve_config.yaml"

    def __init__(self, env: AgenticEnv, run_group: str, run_id: str) -> None:
        self.env = env
        self.run_group = run_group
        self.run_id = run_id
        self.logger = setup_file_logger(
            name=f"openevolve-adapter:{run_id}",
            log_file=self.env.logs_dir / "adapter.log",
        )
        self.run_logger = setup_file_logger(
            name=f"openevolve:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )
        self.dataset_info: Dict[str, Optional[int]] = {}
        self.baseline_config: Optional[Path] = None

    # ------------------------------------------------------------------ utils
    def _workspace_input(self) -> Path:
        return self.env.workspace_dir / "input"

    def _openevolve_workspace(self) -> Path:
        return self.env.workspace_dir / "openevolve"

    def _detect_default_experiment(self, task_root: Path) -> Optional[Path]:
        """Heuristically pick a baseline config for continual-learning."""

        candidate = task_root / "exps" / "simplecil.json"
        if candidate.exists():
            return candidate
        return None

    def _detect_datasets(self, task_root: Path) -> Dict[str, Optional[int]]:
        """Infer dataset splits from logs directory structure."""

        logs_dir = task_root / "logs"
        mapping: Dict[str, Optional[int]] = {}
        if not logs_dir.exists():
            return mapping
        for path in logs_dir.rglob("*.log"):
            lower = path.name.lower()
            if "inr" in lower:
                if "t20" in lower:
                    mapping["imagenet_r"] = 20
                elif "t10" in lower:
                    mapping.setdefault("imagenet_r", 10)
                elif "t5" in lower:
                    mapping.setdefault("imagenet_r", 5)
                else:
                    mapping.setdefault("imagenet_r", None)
            elif "ina" in lower or "imageneta" in lower:
                mapping["imagenet_a"] = 10
            elif "c100" in lower or "cifar" in lower:
                mapping["cifar100"] = 10
            elif "cub" in lower:
                mapping["cub200"] = 10
        return mapping

    def _choose_primary_metric(self, summary: Dict) -> float:
        best = 0.0
        for table in summary.get("tables", []):
            for dataset in table.get("datasets", []):
                metrics = dataset.get("metrics") or {}
                for key in ("aaa_mean", "acc_mean"):
                    value = metrics.get(key)
                    if isinstance(value, (int, float)) and value > best:
                        best = float(value)
        return best

    # ---------------------------------------------------------------- workspace
    def prepare_workspace(self, task_dir: Path) -> Dict[str, Path]:
        """Copy task payload and synthesize OpenEvolve artefacts.

        Returns mapping with paths to generated initial program, evaluator, and config.
        """

        input_dir = self._workspace_input()
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            task_dir,
            input_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "idea_hint.txt"),
        )

        openevolve_dir = self._openevolve_workspace()
        openevolve_dir.mkdir(parents=True, exist_ok=True)

        # Ensure a task description exists at workspace root for graders that expect it
        task_description_src = input_dir / "task_description.md"
        task_description_dst = self.env.workspace_dir / "task_description.md"
        if task_description_src.exists() and not task_description_dst.exists():
            shutil.copy(task_description_src, task_description_dst)

        initial_program = openevolve_dir / self.INITIAL_FILENAME
        evaluator_file = openevolve_dir / self.EVALUATOR_FILENAME
        config_file = openevolve_dir / self.CONFIG_FILENAME

        dataset_info = self._detect_datasets(input_dir)
        self.dataset_info = dataset_info
        dataset_json = json.dumps(dataset_info)

        baseline_cfg = self._detect_default_experiment(input_dir)
        self.baseline_config = baseline_cfg
        baseline_cfg_rel = None if baseline_cfg is None else baseline_cfg.relative_to(input_dir)

        # Generate initial program runner
        initial_program.write_text(
            textwrap.dedent(
                f"""
                import json
                import os
                import subprocess
                import sys
                import time
                from pathlib import Path

                WORKSPACE = Path(__file__).resolve().parents[1] / "input"
                TASK_ROOT = WORKSPACE

                BASELINE_CONFIG = "{baseline_cfg_rel.as_posix() if baseline_cfg_rel else ''}"
                TIME_LIMIT_HOURS = float(os.environ.get("RG_TIME_LIMIT_HOURS", "0"))
                BUDGET_LIMIT = float(os.environ.get("RG_BUDGET_LIMIT", "0"))

                def _load_cost_tracker() -> float:
                    cost_file = Path(os.environ.get("RG_LOG_DIR", WORKSPACE / "logs")) / "costs.json"
                    if not cost_file.exists():
                        return 0.0
                    try:
                        data = json.loads(cost_file.read_text())
                        return float(data.get("usd", 0.0))
                    except Exception:
                        return 0.0

                def within_budget() -> bool:
                    if BUDGET_LIMIT <= 0:
                        return True
                    return _load_cost_tracker() <= BUDGET_LIMIT

                def within_time(start_ts: float) -> bool:
                    if TIME_LIMIT_HOURS <= 0:
                        return True
                    return (time.time() - start_ts) <= (TIME_LIMIT_HOURS * 3600.0)

                def main() -> int:
                    run_script = TASK_ROOT / "run.py"
                    if not run_script.exists():
                        raise FileNotFoundError(f"Expected run.py at {{run_script}}")
                    cfg = TASK_ROOT / BASELINE_CONFIG if BASELINE_CONFIG else None
                    if cfg and not cfg.exists():
                        raise FileNotFoundError(f"Expected baseline config at {{cfg}}")

                    cmd = [sys.executable, str(run_script)]
                    if cfg:
                        cmd.extend(["--config", str(cfg)])

                    start_ts = time.time()
                    proc = subprocess.Popen(cmd, cwd=str(TASK_ROOT))
                    try:
                        while True:
                            rc = proc.poll()
                            if rc is not None:
                                return rc
                            if not within_budget() or not within_time(start_ts):
                                proc.terminate()
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                return 99
                            time.sleep(5)
                    finally:
                        if proc.poll() is None:
                            proc.terminate()

                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        # Generate evaluator shim
        evaluator_file.write_text(
            textwrap.dedent(
                f"""
                import json
                import os
                import subprocess
                import time
                from pathlib import Path

                WORKSPACE = Path(__file__).resolve().parents[1] / "input"
                LOGS_DIR = WORKSPACE / "logs"
                GRADER = WORKSPACE / "grading" / "grade.py"
                DATASET_META = {dataset_json}
                BASELINE_CONFIG = "{baseline_cfg_rel.as_posix() if baseline_cfg_rel else ''}"

                TIME_LIMIT_HOURS = float(os.environ.get("RG_TIME_LIMIT_HOURS", "0"))
                BUDGET_LIMIT = float(os.environ.get("RG_BUDGET_LIMIT", "0"))

                def _load_cost_tracker() -> float:
                    cost_file = Path(os.environ.get("RG_LOG_DIR", LOGS_DIR)) / "costs.json"
                    if not cost_file.exists():
                        return 0.0
                    try:
                        data = json.loads(cost_file.read_text())
                        return float(data.get("usd", 0.0))
                    except Exception:
                        return 0.0

                def within_budget() -> bool:
                    if BUDGET_LIMIT <= 0:
                        return True
                    return _load_cost_tracker() <= BUDGET_LIMIT

                def within_time(start_ts: float) -> bool:
                    if TIME_LIMIT_HOURS <= 0:
                        return True
                    return (time.time() - start_ts) <= (TIME_LIMIT_HOURS * 3600.0)

                def _ensure_logs() -> None:
                    has_logs = LOGS_DIR.exists() and any(LOGS_DIR.rglob("*.log"))
                    if has_logs:
                        return
                    run_py = WORKSPACE / "run.py"
                    if not run_py.exists():
                        return
                    cfg = WORKSPACE / BASELINE_CONFIG if BASELINE_CONFIG else None
                    cmd = [
                        "python",
                        str(run_py),
                    ] + (["--config", str(cfg)] if cfg and cfg.exists() else [])
                    start_ts = time.time()
                    proc = subprocess.Popen(cmd, cwd=str(WORKSPACE))
                    while proc.poll() is None:
                        if not within_budget() or not within_time(start_ts):
                            proc.terminate()
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            break

                def evaluate(_program_path: str) -> dict:
                    _ensure_logs()
                    if not within_budget():
                        return {{"combined_score": 0.0, "meta": {{"reason": "budget_exhausted"}}}}

                    summary_path = LOGS_DIR / "grade_summary.json"
                    summary_path.unlink(missing_ok=True)

                    cmd = [
                        "python",
                        str(GRADER),
                        "--logs",
                        str(LOGS_DIR),
                        "--json-out",
                        str(summary_path),
                    ]

                    start_ts = time.time()
                    proc = subprocess.Popen(cmd, cwd=str(GRADER.parent.parent))
                    while proc.poll() is None:
                        if not within_budget() or not within_time(start_ts):
                            proc.terminate()
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            return {{"combined_score": 0.0, "meta": {{"reason": "limits_exceeded"}}}}

                    returncode = proc.wait()
                    if returncode != 0:
                        raise RuntimeError(f"Grading failed with return code {{returncode}}")

                    if not summary_path.exists():
                        raise FileNotFoundError(f"Expected summary at {{summary_path}}")

                    metrics = json.loads(summary_path.read_text())
                    tables = metrics.get("tables", [])
                    best = 0.0
                    for table in tables:
                        for dataset in table.get("datasets", []):
                            data_metrics = dataset.get("metrics") or {{}}
                            for key in ("aaa_mean", "acc_mean"):
                                value = data_metrics.get(key)
                                if isinstance(value, (int, float)) and value > best:
                                    best = float(value)
                    metrics.setdefault("meta", {{}})
                    metrics["meta"].update({{"datasets": DATASET_META}})
                    metrics["combined_score"] = best
                    return metrics

                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        return {
            "initial_program": initial_program,
            "evaluator": evaluator_file,
            "config": config_file,
        }

    # ----------------------------------------------------------------- command
    def build_config(self, cfg: OpenEvolveConfig, artefacts: Dict[str, Path]) -> Path:
        """Emit an OpenEvolve YAML generated from the provided config."""

        output_dir = self.env.run_dir / "openevolve"
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_info = self._detect_datasets(self._workspace_input())

        # Resolve model name (strip provider prefixes like "openai/azure/")
        def _resolve_model_name(model: str) -> str:
            try:
                parts = [p for p in str(model).split("/") if p]
                return parts[-1] if parts else str(model)
            except Exception:
                return str(model)

        resolved_model = _resolve_model_name(cfg.model)

        # Compose YAML including LLM + evaluator + database config so OpenEvolve initializes correctly
        db_path = (output_dir / "db").as_posix()
        yaml_payload = textwrap.dedent(
            f"""
            max_iterations: {cfg.max_iterations}
            checkpoint_interval: {cfg.checkpoint_interval}
            log_level: "DEBUG"
            random_seed: 42

            budget_limit_usd: {cfg.budget_limit}
            time_limit_hours: {cfg.time_limit_hours}
            task_id: "{cfg.task_id}"
            dataset_meta: {json.dumps(self.dataset_info)}
            baseline_config: "{self.baseline_config.as_posix() if self.baseline_config else ''}"

            llm:
              api_base: "{cfg.api_base}"
              models:
                - name: "{resolved_model}"
                  weight: 1.0
              evaluator_models:
                - name: "{resolved_model}"
                  weight: 1.0

            evaluator:
              cascade_evaluation: false
              timeout: {cfg.evaluator_timeout}
              parallel_evaluations: {cfg.parallel_evaluations}

            database:
              db_path: "{db_path}"
              log_prompts: true
            """
        ).strip()

        artefacts["config"].write_text(yaml_payload + "\n", encoding="utf-8")
        return artefacts["config"]

    def build_command(self, openevolve_root: Path, artefacts: Dict[str, Path], cfg: OpenEvolveConfig) -> Observation:
        """Return command/env tuple for run planning."""

        python = os.environ.get("PYTHON", "python")
        entry = openevolve_root / "openevolve-run.py"
        # Ensure the CLI receives explicit model/base overrides (for cases where YAML is minimal)
        # Map model to OpenEvolve model name (strip provider prefixes)
        try:
            parts = [p for p in str(cfg.model).split("/") if p]
            primary_model = parts[-1] if parts else str(cfg.model)
        except Exception:
            primary_model = str(cfg.model)

        cmd = [
            python,
            str(entry),
            str(artefacts["initial_program"]),
            str(artefacts["evaluator"]),
            "--config",
            str(artefacts["config"]),
            "--iterations",
            str(cfg.max_iterations),
            "--api-base",
            str(cfg.api_base),
            "--primary-model",
            str(primary_model),
        ]

        env = {
            "RG_LOG_DIR": str(self.env.logs_dir),
            "RG_RUN_DIR": str(self.env.run_dir),
            "RG_BUDGET_LIMIT": str(cfg.budget_limit),
            "RG_TIME_LIMIT_HOURS": str(cfg.time_limit_hours),
            "OPENEVOLVE_OUTPUT_DIR": str(self.env.run_dir / "openevolve"),
            "OPENAI_API_BASE": cfg.api_base,
        }

        return Observation(
            message="planned",
            info={
                "command": cmd,
                "env": env,
            },
        )

    def run(
        self,
        cfg: OpenEvolveConfig,
        openevolve_root: Path,
        task_dir: Path,
        dry_run: bool = True,
    ) -> Observation:
        artefacts = self.prepare_workspace(task_dir)
        self.build_config(cfg, artefacts)
        plan = self.build_command(openevolve_root, artefacts, cfg)

        if dry_run:
            return plan

        # Runtime execution (best effort) – currently unimplemented.
        raise NotImplementedError("Blocking execution is not yet implemented for OpenEvolveAdapter")

