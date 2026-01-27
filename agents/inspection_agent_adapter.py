from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ResearchGym.utils.logging import setup_file_logger


@dataclass
class InspectionAgentConfig:
    run_dir: Path
    task_dir: Path
    model: str = "openai/gpt-4o"
    budget_limit: float = 2.0
    output_dir: Optional[Path] = None  # If None, auto-generated under inspections/


@dataclass
class InspectionResult:
    verdict: str  # "PASS", "SUSPICIOUS", "NEEDS_REVIEW"
    confidence: float
    summary: str
    violations: list
    evidence: list
    report_path: Optional[Path] = None
    cost_usd: float = 0.0
    tokens: int = 0


class InspectionAgentAdapter:
    """Adapter for the InspectionAgent that verifies completed runs."""

    def __init__(self, run_dir: Path, output_dir: Optional[Path] = None) -> None:
        self.run_dir = Path(run_dir).resolve()

        # Determine output directory (outside of run_dir to keep results immutable)
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            # Auto-generate: inspections/{task_name}/{run_name}/
            # e.g., results/continual-learning/001 -> inspections/continual-learning/001
            project_root = Path(__file__).resolve().parent.parent
            # Extract relative path from results/ if present
            try:
                rel_path = self.run_dir.relative_to(project_root / "results")
            except ValueError:
                # Not under results/, use run_dir name
                rel_path = Path(self.run_dir.name)
            self.output_dir = project_root / "inspections" / rel_path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_file_logger(
            name=f"inspection-agent:{self.run_dir.name}",
            log_file=self.output_dir / "adapter.log",
        )

    def resolve_task_dir(self) -> Path:
        """Read metadata.json to find the original task directory."""
        metadata_path = self.run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.run_dir}")

        metadata = json.loads(metadata_path.read_text())
        task_dir = metadata.get("task_dir")
        if not task_dir:
            raise ValueError("task_dir not found in metadata.json")

        task_path = Path(task_dir)
        if not task_path.exists():
            raise FileNotFoundError(f"Task directory not found: {task_path}")

        return task_path

    def build_command(self, agent_root: Path) -> list[str]:
        """Build the command to run the inspection agent."""
        python = sys.executable
        entry = agent_root / "start.py"
        return [python, str(entry)]

    def run(
        self,
        cfg: InspectionAgentConfig,
        agent_root: Path,
        dry_run: bool = False,
    ) -> InspectionResult:
        """Run the inspection agent on the specified run directory."""
        cmd = self.build_command(agent_root)
        self.logger.info(f"InspectionAgent command: {' '.join(cmd)}")

        # Build environment
        env = os.environ.copy()
        env.update({
            "INSPECTION_RUN_DIR": str(cfg.run_dir),
            "INSPECTION_TASK_DIR": str(cfg.task_dir),
            "INSPECTION_OUTPUT_DIR": str(self.output_dir),  # Where to write reports
            "INSPECTION_TRANSCRIPT_PATH": str(cfg.run_dir / "transcript.json"),
            "INSPECTION_BUDGET_LIMIT": str(cfg.budget_limit),
            "MODEL": cfg.model,
            # Also set CODE_DIR for bash/read_file_chunk tools to work
            "CODE_DIR": str(cfg.run_dir / "workspace" / "input"),
        })

        # Map GEMINI_API_KEY -> GOOGLE_API_KEY if needed
        if "GOOGLE_API_KEY" not in env and env.get("GEMINI_API_KEY"):
            env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]

        # Map AZUREAI_OPENAI_* -> AZURE_OPENAI_* for inspect_ai compatibility
        if "AZURE_OPENAI_API_KEY" not in env and env.get("AZUREAI_OPENAI_API_KEY"):
            env["AZURE_OPENAI_API_KEY"] = env["AZUREAI_OPENAI_API_KEY"]
        if "AZURE_OPENAI_ENDPOINT" not in env and env.get("AZUREAI_OPENAI_BASE_URL"):
            env["AZURE_OPENAI_ENDPOINT"] = env["AZUREAI_OPENAI_BASE_URL"]

        # Unbuffered output for streaming
        env["PYTHONUNBUFFERED"] = "1"

        if dry_run:
            self.logger.info("Dry run - not executing")
            return InspectionResult(
                verdict="DRY_RUN",
                confidence=0.0,
                summary="Dry run mode - inspection not executed",
                violations=[],
                evidence=[],
            )

        # Execute inspection
        self.logger.info("Starting inspection...")
        stdout_log = self.output_dir / "inspection.stdout.log"
        stderr_log = self.output_dir / "inspection.stderr.log"

        with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(agent_root),
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
            )
            # No timeout - agent should finish with finish_inspection()
            # But set a generous timeout just in case (30 minutes)
            try:
                rc = proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.logger.error("Inspection timed out after 30 minutes")
                return InspectionResult(
                    verdict="NEEDS_REVIEW",
                    confidence=0.0,
                    summary="Inspection timed out",
                    violations=[],
                    evidence=[],
                )

        self.logger.info(f"Inspection completed with return code: {rc}")

        # Read the inspection report if it exists
        report_path = self.output_dir / "inspection_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                cost_info = report.get("cost", {})
                return InspectionResult(
                    verdict=report.get("verdict", "NEEDS_REVIEW"),
                    confidence=report.get("confidence", 0.0),
                    summary=report.get("summary", ""),
                    violations=report.get("violations", []),
                    evidence=report.get("evidence", []),
                    report_path=report_path,
                    cost_usd=cost_info.get("cost_usd", 0.0),
                    tokens=cost_info.get("total_tokens", 0),
                )
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse inspection report: {e}")

        # No report generated
        return InspectionResult(
            verdict="NEEDS_REVIEW",
            confidence=0.0,
            summary="Inspection completed but no report was generated",
            violations=[],
            evidence=[],
        )
