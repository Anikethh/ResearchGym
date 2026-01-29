from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger


@dataclass
class RGAgentConfig:
    task_id: str
    model: str
    time_hours: float
    iterative: bool = False
    disallow_submit: bool = False
    code_only: bool = False
    time_limit_secs: int = 3600
    budget_limit: float = 10.0  # Set to 0 for no limit
    idea_hint: bool = False


class RGAgentAdapter:
    def __init__(self, env: AgenticEnv, run_group: str, run_id: str) -> None:
        self.env = env
        self.run_group = run_group
        self.run_id = run_id
        self.run_logger = setup_file_logger(
            name=f"rg-agent:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )
        self.logger = setup_file_logger(
            name=f"rg-agent-adapter:{run_id}",
            log_file=self.env.logs_dir / "adapter.log",
        )

    def prepare_workspace(self, task_dir: Path) -> None:
        input_dir = self.env.workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            task_dir,
            input_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "requirements.txt", "install.sh", "idea_hint.txt"),
        )

    def build_command(self, agent_root: Path, cfg: RGAgentConfig) -> list[str]:
        # Run the vendored RGAgent start.py as a module with env vars
        python = sys.executable
        entry = agent_root / "start.py"
        cmd = [python, str(entry)]
        return cmd

    def run(self, cfg: RGAgentConfig, agent_root: Path, dry_run: bool = True) -> Observation:
        cmd = self.build_command(agent_root, cfg)
        self.logger.info(f"RGAgent command: {' '.join(cmd)}")
        self.run_logger.info(f"planning: {' '.join(cmd)}")
        env = os.environ.copy()
        input_dir = self.env.workspace_dir / "input"
        instructions_txt = input_dir / "instructions.txt"
        task_desc_md = input_dir / "task_description.md"
        instructions_file: Optional[Path] = None
        if instructions_txt.exists():
            instructions_file = instructions_txt
        elif task_desc_md.exists():
            instructions_file = task_desc_md
        # Provide environment used by RGAgent
        env.update(
            {
                "WORKSPACE_BASE": str(self.env.workspace_dir),
                "CODE_DIR": str(self.env.workspace_dir / "input"),
                # Keep async job outputs inside the agent-visible CODE_DIR
                "RG_ASYNC_JOBS_DIR": str(self.env.workspace_dir / "input" / "async_jobs"),
                "AGENT_DIR": str(agent_root),
                "RG_RUN_DIR": str(self.env.run_dir),
                "MAX_TIME_IN_HOURS": str(cfg.time_hours),
                "MODEL": cfg.model,
                "ITERATIVE_AGENT": "true" if cfg.iterative else "false",
                "DISALLOW_SUBMIT": "true" if cfg.disallow_submit else "false",
                "PB_CODE_ONLY": "true" if cfg.code_only else "false",
                # Pass the correct log directory to RGAgent
                "RG_LOG_DIR": str(self.env.logs_dir),
                # Pass budget limit for real-time monitoring
                "RG_BUDGET_LIMIT": str(cfg.budget_limit),
                # Stream per-iteration metadata so interrupted runs can recover usage stats
                "RG_METADATA_STREAM_PATH": str(self.env.logs_dir / "metadata_stream.jsonl"),
                "RG_IDEA_HINT": "true" if cfg.idea_hint else "false",
            }
        )
        if instructions_file:
            env["RG_INSTRUCTIONS_FILE"] = str(instructions_file)

        # Map GEMINI_API_KEY -> GOOGLE_API_KEY if needed for inspect_ai's google provider
        if "GOOGLE_API_KEY" not in env and env.get("GEMINI_API_KEY"):
            env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]
        # Auto-enable web_search tool when provider credentials are present
        try:
            has_exa = bool(env.get("EXA_API_KEY"))
            has_google_cse = bool(env.get("GOOGLE_CSE_API_KEY") and env.get("GOOGLE_CSE_ID"))
            if has_exa:
                env["USE_EXA_SEARCH"] = "true"
                # Ensure google flag is explicitly false to avoid ambiguity
                env.setdefault("USE_GOOGLE_WEB_SEARCH", "false")
            elif has_google_cse:
                env["USE_GOOGLE_WEB_SEARCH"] = "true"
                env.setdefault("USE_EXA_SEARCH", "false")
            else:
                # Neither provider configured; keep both disabled unless user overrides
                env.setdefault("USE_EXA_SEARCH", "false")
                env.setdefault("USE_GOOGLE_WEB_SEARCH", "false")
        except Exception:
            # Fail-safe: don't block run on env inspection
            pass
        if dry_run:
            return Observation(
                message="planned",
                info={
                    "command": cmd,
                    # Only include non-secret env used to wire the agent/runtime
                    "env": {
                        k: env[k]
                        for k in [
                            "WORKSPACE_BASE",
                            "CODE_DIR",
                            "AGENT_DIR",
                            "MAX_TIME_IN_HOURS",
                            "MODEL",
                            "ITERATIVE_AGENT",
                            "DISALLOW_SUBMIT",
                            "PB_CODE_ONLY",
                            "RG_INSTRUCTIONS_FILE",
                    "RG_LOG_DIR",
                    "RG_RUN_DIR",
                    "RG_BUDGET_LIMIT",
                    "RG_IDEA_HINT",
                    # Web search feature flags (no secrets)
                    "USE_EXA_SEARCH",
                    "USE_GOOGLE_WEB_SEARCH",
                        ]
                        if k in env
                    },
                },
            )

        proc = subprocess.Popen(
            cmd,
            cwd=str(agent_root),
            env=env,
            stdout=open(self.env.logs_dir / "rg_agent.stdout.log", "w"),
            stderr=open(self.env.logs_dir / "rg_agent.stderr.log", "w"),
        )
        rc = proc.wait(timeout=cfg.time_limit_secs)
        return Observation(message="completed", info={"returncode": rc})


