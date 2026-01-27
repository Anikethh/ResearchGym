"""Adapter for Claude Code agent integration with ResearchGym.

Follows the existing adapter pattern used by BasicAgent, ML-Master, etc.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger

from .config import ClaudeCodeConfig, load_blocked_urls


class ClaudeCodeAdapter:
    """Adapter for running Claude Code agent within ResearchGym.

    Handles:
    - Workspace preparation (copying task files)
    - Environment configuration
    - Agent execution via subprocess or direct runner call
    - Logging setup
    """

    def __init__(self, env: AgenticEnv, run_group: str, run_id: str) -> None:
        """Initialize the adapter.

        Args:
            env: AgenticEnv instance with run directories
            run_group: Run group identifier
            run_id: Unique run identifier
        """
        self.env = env
        self.run_group = run_group
        self.run_id = run_id
        self.run_logger = setup_file_logger(
            name=f"claude-code:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )
        self.logger = setup_file_logger(
            name=f"claude-code-adapter:{run_id}",
            log_file=self.env.logs_dir / "adapter.log",
        )

    def prepare_workspace(self, task_dir: Path, include_idea_hint: bool = False) -> None:
        """Copy task files to workspace.

        Args:
            task_dir: Path to source task directory
            include_idea_hint: If False (default), exclude idea_hint.txt from copy
        """
        input_dir = self.env.workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Build ignore patterns
        ignore_patterns = [
            ".git",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            # Don't copy blocklist to workspace - agent shouldn't see it
            "blocked_urls.yaml",
            # Already installed by runtime, don't confuse agent
            "requirements.txt",
            "install.sh",
        ]
        if not include_idea_hint:
            ignore_patterns.append("idea_hint.txt")

        # Copy task files, excluding certain patterns
        shutil.copytree(
            task_dir,
            input_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*ignore_patterns),
        )

        self.logger.info(f"Prepared workspace at {input_dir}")

    def build_command(self, cfg: ClaudeCodeConfig) -> list[str]:
        """Build the command to run the Claude Code agent.

        Args:
            cfg: Agent configuration

        Returns:
            Command as list of strings
        """
        # Run via the runner module as a package (to support relative imports)
        python = sys.executable

        cmd = [
            python,
            "-m", "ResearchGym.agents.ClaudeCode.run_cli",
            "--workspace", str(self.env.workspace_dir),
            "--log-dir", str(self.env.logs_dir),
            "--model", cfg.model,
            "--time-hours", str(cfg.time_hours),
            "--budget-limit", str(cfg.budget_limit),
        ]

        if cfg.extended_continue:
            cmd.append("--extended-continue")

        if cfg.use_api:
            cmd.append("--use-api")

        if cfg.blocked_urls:
            # Write URLs to file to avoid Windows shell quote mangling
            blocked_urls_file = self.env.logs_dir / "blocked_urls_input.json"
            blocked_urls_file.parent.mkdir(parents=True, exist_ok=True)
            with open(blocked_urls_file, "w") as f:
                json.dump(cfg.blocked_urls, f)
            cmd.extend(["--blocked-urls-file", str(blocked_urls_file)])

        return cmd

    def build_env(self, cfg: ClaudeCodeConfig) -> dict[str, str]:
        """Build environment variables for the run.

        Args:
            cfg: Agent configuration

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()

        # Ensure ResearchGym is importable in subprocess
        # adapter.py is at ResearchGym/agents/ClaudeCode/adapter.py
        # so repo root is 3 levels up
        repo_root = Path(__file__).parent.parent.parent.parent
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(repo_root)

        # Add venv bin to PATH so agent's pip/python commands use correct environment
        # This mirrors BasicAgent's approach in _execute.py
        venv_bin = os.path.dirname(sys.executable)
        existing_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}{os.pathsep}{existing_path}"

        # Also set VIRTUAL_ENV for tools that check it
        venv_root = Path(sys.executable).parent.parent
        env["VIRTUAL_ENV"] = str(venv_root)

        # Core ResearchGym variables
        env.update({
            "WORKSPACE_BASE": str(self.env.workspace_dir),
            "CODE_DIR": str(self.env.workspace_dir / "input"),
            "RG_RUN_DIR": str(self.env.run_dir),
            "RG_LOG_DIR": str(self.env.logs_dir),
            "RG_BUDGET_LIMIT": str(cfg.budget_limit),
            "RG_TIME_HOURS": str(cfg.time_hours),
            "RG_MODEL": cfg.model,
        })

        # Add any custom env vars from config
        env.update(cfg.env)

        return env

    def run(
        self,
        cfg: ClaudeCodeConfig,
        task_dir: Path,
        dry_run: bool = False,
    ) -> Observation:
        """Run the Claude Code agent.

        Args:
            cfg: Agent configuration
            task_dir: Path to task directory (for loading blocked URLs if not in config)
            dry_run: If True, return plan without executing

        Returns:
            Observation with run results
        """
        # Load blocked URLs from task if not already in config
        if not cfg.blocked_urls:
            cfg.blocked_urls = load_blocked_urls(task_dir)
            if cfg.blocked_urls:
                self.logger.info(f"Loaded {len(cfg.blocked_urls)} blocked URLs from task")

        cmd = self.build_command(cfg)
        env = self.build_env(cfg)

        self.logger.info(f"Claude Code command: {' '.join(cmd)}")
        self.run_logger.info(f"planning: {' '.join(cmd)}")

        if dry_run:
            return Observation(
                message="planned",
                info={
                    "command": cmd,
                    "env": {
                        k: env[k]
                        for k in [
                            "WORKSPACE_BASE",
                            "CODE_DIR",
                            "RG_RUN_DIR",
                            "RG_LOG_DIR",
                            "RG_BUDGET_LIMIT",
                            "RG_TIME_HOURS",
                            "RG_MODEL",
                        ]
                        if k in env
                    },
                    "config": cfg.to_dict(),
                },
            )

        # Execute the agent
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.env.workspace_dir),
                env=env,
                stdout=open(self.env.logs_dir / "claude_code.stdout.log", "w"),
                stderr=open(self.env.logs_dir / "claude_code.stderr.log", "w"),
            )

            # Wait with timeout (time limit + buffer)
            timeout_secs = int(cfg.time_hours * 3600) + 300  # +5 min buffer
            rc = proc.wait(timeout=timeout_secs)

            # Load results from cost summary
            cost_summary_path = self.env.logs_dir / "cost_summary.json"
            results: dict[str, Any] = {"returncode": rc}
            if cost_summary_path.exists():
                with open(cost_summary_path) as f:
                    results["cost_summary"] = json.load(f)

            return Observation(
                message="completed" if rc == 0 else "error",
                info=results,
            )

        except subprocess.TimeoutExpired:
            proc.kill()
            self.logger.error("Process timed out and was killed")
            return Observation(
                message="timeout",
                info={"error": "Process exceeded time limit"},
            )

        except Exception as e:
            self.logger.error(f"Error running agent: {e}", exc_info=True)
            return Observation(
                message="error",
                info={"error": str(e)},
            )

    def run_direct(
        self,
        cfg: ClaudeCodeConfig,
        task_dir: Path,
    ) -> dict[str, Any]:
        """Run the agent directly (not via subprocess).

        Useful for debugging or when subprocess isolation isn't needed.

        Args:
            cfg: Agent configuration
            task_dir: Path to task directory

        Returns:
            Dictionary with run results
        """
        from .runner import run_agent_sync

        # Load blocked URLs from task if not already in config
        if not cfg.blocked_urls:
            cfg.blocked_urls = load_blocked_urls(task_dir)

        self.logger.info("Running Claude Code agent directly")
        self.logger.info(f"  Model: {cfg.model}")
        self.logger.info(f"  Time: {cfg.time_hours}h")
        self.logger.info(f"  Budget: ${cfg.budget_limit}")

        return run_agent_sync(
            config=cfg,
            workspace_dir=self.env.workspace_dir,
            log_dir=self.env.logs_dir,
        )
