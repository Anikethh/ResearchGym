"""Adapter for Codex CLI agent integration with ResearchGym.

Follows the existing adapter pattern used by BasicAgent, Claude Code, etc.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger

from .config import CodexConfig, load_blocked_urls


class CodexAdapter:
    """Adapter for running Codex CLI agent within ResearchGym.

    Handles:
    - Workspace preparation (copying task files)
    - Environment configuration
    - Agent execution via subprocess
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
            name=f"codex:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )
        self.logger = setup_file_logger(
            name=f"codex-adapter:{run_id}",
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

        # Install task requirements so agent doesn't have to deal with pip/venv issues
        requirements_txt = task_dir / "requirements.txt"
        if requirements_txt.exists():
            self._install_requirements(requirements_txt)

        # Run install.sh if present
        install_sh = task_dir / "install.sh"
        if install_sh.exists():
            self._run_install_script(install_sh, input_dir)

        self.logger.info(f"Prepared workspace at {input_dir}")

    def _install_requirements(self, requirements_path: Path) -> None:
        """Install requirements using uv pip."""
        import subprocess
        self.logger.info(f"Installing requirements from {requirements_path}")
        try:
            # Use uv pip install which works with uv-managed venvs
            result = subprocess.run(
                ["uv", "pip", "install", "-r", str(requirements_path)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )
            if result.returncode != 0:
                self.logger.warning(f"uv pip install failed: {result.stderr[:500]}")
            else:
                self.logger.info("Requirements installed successfully")
        except Exception as e:
            self.logger.warning(f"Failed to install requirements: {e}")

    def _run_install_script(self, install_sh: Path, cwd: Path) -> None:
        """Run install.sh script."""
        import subprocess
        self.logger.info(f"Running install script: {install_sh}")
        try:
            # Try git bash first, fall back to sh
            bash_paths = [
                "C:/Program Files/Git/bin/bash.exe",
                "bash",
                "sh",
            ]
            for bash in bash_paths:
                try:
                    result = subprocess.run(
                        [bash, str(install_sh)],
                        cwd=str(cwd),
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode == 0:
                        self.logger.info("Install script completed successfully")
                        return
                    else:
                        self.logger.warning(f"Install script failed: {result.stderr[:500]}")
                except FileNotFoundError:
                    continue
        except Exception as e:
            self.logger.warning(f"Failed to run install script: {e}")

    def build_command(self, cfg: CodexConfig) -> list[str]:
        """Build the command to run the Codex CLI agent.

        Args:
            cfg: Agent configuration

        Returns:
            Command as list of strings
        """
        python = sys.executable
        cmd = [
            python,
            "-m", "ResearchGym.agents.Codex.run_cli",
            "--workspace", str(self.env.workspace_dir),
            "--log-dir", str(self.env.logs_dir),
            "--model", cfg.model,
            "--time-hours", str(cfg.time_hours),
            "--budget-limit", str(cfg.budget_limit),
            "--approval-mode", cfg.approval_mode,
            "--sandbox-mode", cfg.sandbox_mode,
        ]

        if cfg.resume_session_id:
            cmd.extend(["--resume-session", cfg.resume_session_id])

        if cfg.provider:
            cmd.extend(["--provider", cfg.provider])

        return cmd

    def build_env(self, cfg: CodexConfig) -> dict[str, str]:
        """Build environment variables for the run.

        Args:
            cfg: Agent configuration

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()

        # Ensure ResearchGym is importable in subprocess
        repo_root = Path(__file__).parent.parent.parent.parent
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(repo_root)

        # Add venv bin to PATH so agent's pip/python commands use correct environment
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
            "RG_PROVIDER": cfg.provider or "auto",
        })

        # Add any custom env vars from config
        env.update(cfg.env)

        return env

    def run(
        self,
        cfg: CodexConfig,
        task_dir: Path,
        dry_run: bool = False,
    ) -> Observation:
        """Run the Codex CLI agent.

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
                # Note: Codex CLI doesn't have built-in URL blocking
                # Would need external proxy or wrapper to enforce this

        cmd = self.build_command(cfg)
        env_vars = self.build_env(cfg)

        self.logger.info(f"Codex command: {' '.join(cmd[:10])}...")  # Truncate long prompt
        self.run_logger.info(f"planning: codex --model {cfg.model}")

        if dry_run:
            return Observation(
                message="planned",
                info={
                    "command": cmd,
                    "env": {
                        k: env_vars[k]
                        for k in [
                            "WORKSPACE_BASE",
                            "CODE_DIR",
                            "RG_RUN_DIR",
                            "RG_LOG_DIR",
                            "RG_BUDGET_LIMIT",
                            "RG_TIME_HOURS",
                            "RG_MODEL",
                        ]
                        if k in env_vars
                    },
                    "config": cfg.to_dict(),
                },
            )

        # Execute the agent
        stdout_log = self.env.logs_dir / "codex.stdout.log"
        stderr_log = self.env.logs_dir / "codex.stderr.log"

        proc: subprocess.Popen | None = None
        try:
            with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.env.workspace_dir),
                    env=env_vars,
                    stdout=out,
                    stderr=err,
                )

                # Wait with timeout (time limit + buffer)
                timeout_secs = int(cfg.time_hours * 3600) + 300  # +5 min buffer
                rc = proc.wait(timeout=timeout_secs)

            # Load results
            results: dict[str, Any] = {"returncode": rc}
            cost_summary_path = self.env.logs_dir / "cost_summary.json"
            if cost_summary_path.exists():
                try:
                    results["cost_summary"] = json.loads(cost_summary_path.read_text(encoding="utf-8"))
                except Exception as e:
                    self.logger.warning(f"Could not parse cost_summary.json: {e}")

            return Observation(
                message="completed" if rc == 0 else "error",
                info=results,
            )

        except KeyboardInterrupt:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self.logger.warning("Codex run interrupted by user")
            return Observation(
                message="interrupted",
                info={"error": "Run interrupted by user"},
            )

        except subprocess.TimeoutExpired:
            proc.kill()
            self.logger.error("Process timed out and was killed")
            return Observation(
                message="timeout",
                info={"error": "Process exceeded time limit"},
            )

        except FileNotFoundError:
            self.logger.error("codex CLI not found")
            return Observation(
                message="error",
                info={"error": "codex CLI not found. Install with: npm install -g @openai/codex"},
            )

        except Exception as e:
            self.logger.error(f"Error running agent: {e}", exc_info=True)
            return Observation(
                message="error",
                info={"error": str(e)},
            )
