from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import platform
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables from .env file
load_dotenv()

# Ensure package import works when running this file directly
CURRENT_FILE = Path(__file__).resolve()
PKG_ROOT = CURRENT_FILE.parent
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ResearchGym.environment import AgenticEnv
from ResearchGym.utils.logging import setup_file_logger
from ResearchGym.agents.ml_master_adapter import MLMasterAdapter, MLMasterConfig
from ResearchGym.agents.ai_scientist_adapter import AIScientistAdapter, AIScientistConfig
from ResearchGym.environment.runtime.docker_runner import plan_docker_command
from ResearchGym.environment.runtime.uv_runner import detect_task_overlay, plan_uv_commands
from ResearchGym.agents.basic_agent_adapter import BasicAgentAdapter, BasicAgentConfig
from ResearchGym.agents.basic_agent_evolution_adapter import BasicAgentEvolutionAdapter, BasicAgentEvolutionConfig
from ResearchGym.agents.ClaudeCode.adapter import ClaudeCodeAdapter
from ResearchGym.agents.ClaudeCode.config import ClaudeCodeConfig
from ResearchGym.agents.Codex.adapter import CodexAdapter
from ResearchGym.agents.Codex.config import CodexConfig, PROVIDER_SUBSCRIPTION


def _gen_ids() -> tuple[str, str]:
    run_group = time.strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:8]
    return run_group, run_id

def _ensure_git_bash_in_path() -> None:
    """On Windows, ensure RG_BASH_PATH points to Git Bash and PATH includes it."""
    if os.name != "nt":
        return

    def _is_stub(path: str) -> bool:
        lowered = path.lower()
        return (
            "windowsapps\\bash.exe" in lowered
            or "system32\\bash.exe" in lowered
            or "sysnative\\bash.exe" in lowered
        )

    if os.environ.get("RG_BASH_PATH"):
        return

    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin",
        r"C:\Program Files\Git\usr\bin",
    ]

    for cand in candidates:
        if not cand:
            continue
        path_obj = Path(cand)
        if path_obj.is_dir():
            path_obj = path_obj / "bash.exe"
        if path_obj.exists() and not _is_stub(str(path_obj)):
            os.environ["RG_BASH_PATH"] = str(path_obj)
            parent = str(path_obj.parent)
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{parent}{os.pathsep}{current_path}"
            return


def _ensure_git_initialized(repo_dir: Path, logger=None) -> None:
    """Ensure a git repository exists at repo_dir.

    This initializes git for each run's task workspace so that agents can make
    commits during the run. Safe to call repeatedly.
    """
    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        if (repo_dir / ".git").exists():
            return
        # Initialize repository quietly
        subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Ensure user identity is set locally to avoid commit failures
        email_probe = subprocess.run(["git", "config", "user.email"], cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8')
        if email_probe.returncode != 0 or not email_probe.stdout.strip():
            subprocess.run(["git", "config", "user.email", "agent@researchgym.local"], cwd=str(repo_dir), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        name_probe = subprocess.run(["git", "config", "user.name"], cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8')
        if name_probe.returncode != 0 or not name_probe.stdout.strip():
            subprocess.run(["git", "config", "user.name", "ResearchGym Agent"], cwd=str(repo_dir), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if logger:
            try:
                logger.info(f"Initialized git repository at {repo_dir}")
            except Exception:
                pass
    except Exception as e:
        # Do not fail the run if git is not available; just warn.
        msg = f"Warning: failed to initialize git repo at {repo_dir}: {e}"
        print(msg)
        if logger:
            try:
                logger.warning(msg)
            except Exception:
                pass


@dataclass
class ResumeUsage:
    hours_spent: float
    budget_spent: float
    log_files: List[str]


def _parse_iso_timestamp(raw: str) -> datetime:
    if not raw:
        raise ValueError("missing timestamp")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def _list_inspect_logs(log_dir: Path) -> List[Path]:
    if not log_dir.exists():
        return []
    candidates = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=False)
    if candidates:
        return candidates
    return sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=False)


def _collect_resume_usage(run_dir: Path) -> ResumeUsage:
    log_dir = run_dir / "logs"
    stdout_log = log_dir / "exec.stdout.log"
    if not stdout_log.exists():
        raise FileNotFoundError(f"Missing exec stdout log for resume: {stdout_log}")

    inspect_logs = []
    for candidate in _list_inspect_logs(log_dir):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        stats = data.get("stats") or {}
        started = stats.get("started_at")
        completed = stats.get("completed_at")
        if not started or not completed:
            continue
        inspect_logs.append((candidate, started, completed))
    if not inspect_logs:
        raise FileNotFoundError(f"No inspect logs (.json/.eval) with timing metadata found in {log_dir}")

    total_hours = 0.0
    for _, start_raw, end_raw in inspect_logs:
        start_dt = _parse_iso_timestamp(start_raw)
        end_dt = _parse_iso_timestamp(end_raw)
        delta = (end_dt - start_dt).total_seconds()
        if delta > 0:
            total_hours += delta / 3600.0

    last_cost = None
    cost_pattern = re.compile(r"Session total cost:\s*\$([0-9]+(?:\.[0-9]+)?)")
    with stdout_log.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = cost_pattern.search(line)
            if match:
                try:
                    last_cost = float(match.group(1))
                except ValueError:
                    continue
    if last_cost is None:
        raise RuntimeError(f"Could not parse session cost from {stdout_log}")

    return ResumeUsage(
        hours_spent=total_hours,
        budget_spent=last_cost,
        log_files=[path.name for path, _, _ in inspect_logs],
    )


def _collect_claude_code_resume_usage(run_dir: Path, original_hours: float | None = None) -> ResumeUsage:
    """Collect resume usage from ClaudeCode's cost_summary.json.

    Unlike BasicAgent which uses inspect logs, ClaudeCode stores usage
    in cost_summary.json created by its CostTracker.

    Args:
        run_dir: Path to the run directory to collect usage from
        original_hours: Original time budget in hours (e.g., 24). If provided,
            calculates cumulative time spent using remaining_seconds for accurate
            multi-generation resume tracking.
    """
    log_dir = run_dir / "logs"
    cost_summary_path = log_dir / "cost_summary.json"

    if not cost_summary_path.exists():
        raise FileNotFoundError(f"Missing cost_summary.json for resume: {cost_summary_path}")

    try:
        data = json.loads(cost_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in cost_summary.json: {e}")

    # Extract time spent (in hours)
    # For multi-generation resumes, use remaining_seconds to get cumulative time
    time_info = data.get("time", {})
    remaining_seconds = time_info.get("remaining_seconds")

    if original_hours is not None and remaining_seconds is not None and remaining_seconds >= 0:
        # Calculate cumulative time spent: original_budget - remaining
        # This correctly accounts for ALL previous sessions, not just the immediate one
        original_seconds = original_hours * 3600
        hours_spent = (original_seconds - remaining_seconds) / 3600.0
    else:
        # Fallback to active_seconds (first-generation resume or missing data)
        active_seconds = time_info.get("active_seconds", 0.0)
        hours_spent = active_seconds / 3600.0

    # Extract cost spent
    budget_spent = data.get("total_cost_usd", 0.0)

    return ResumeUsage(
        hours_spent=hours_spent,
        budget_spent=budget_spent,
        log_files=["cost_summary.json"],
    )


def _collect_codex_resume_usage(run_dir: Path, original_hours: float | None = None) -> ResumeUsage:
    """Collect resume usage from Codex cost_summary.json.

    Args:
        run_dir: Path to the run directory to collect usage from
        original_hours: Original time budget in hours (e.g., 24). If provided,
            calculates cumulative time spent using remaining_seconds for accurate
            multi-generation resume tracking.
    """
    log_dir = run_dir / "logs"
    cost_summary_path = log_dir / "cost_summary.json"

    if not cost_summary_path.exists():
        raise FileNotFoundError(f"Missing cost_summary.json for resume: {cost_summary_path}")

    try:
        data = json.loads(cost_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in cost_summary.json: {e}")

    # Extract time spent (in hours)
    # For multi-generation resumes, use remaining_seconds to get cumulative time
    time_info = data.get("time", {})
    remaining_seconds = time_info.get("remaining_seconds")

    if original_hours is not None and remaining_seconds is not None and remaining_seconds >= 0:
        # Calculate cumulative time spent: original_budget - remaining
        original_seconds = original_hours * 3600
        hours_spent = (original_seconds - remaining_seconds) / 3600.0
    else:
        # Fallback to active_seconds (first-generation resume or missing data)
        active_seconds = time_info.get("active_seconds", 0.0)
        hours_spent = active_seconds / 3600.0

    budget_spent = data.get("total_cost_usd", 0.0)

    return ResumeUsage(
        hours_spent=hours_spent,
        budget_spent=budget_spent,
        log_files=["cost_summary.json"],
    )


def _get_claude_code_session_id(run_dir: Path) -> str | None:
    """Get session ID from ClaudeCode's session.json for resume."""
    session_path = run_dir / "logs" / "session.json"
    if not session_path.exists():
        return None

    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
        return data.get("session_id")
    except (json.JSONDecodeError, OSError):
        return None


def _get_codex_session_id(run_dir: Path) -> str | None:
    """Get session ID from Codex JSONL output for resume."""
    output_path = run_dir / "logs" / "codex_output.jsonl"
    if not output_path.exists():
        return None

    try:
        with output_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                session_id = data.get("session_id")
                if session_id:
                    return session_id
    except Exception:
        return None

    return None


def _encode_claude_project_path(workspace_path: Path) -> str:
    """Encode a workspace path the way Claude stores it in ~/.claude/projects/."""
    path_str = str(workspace_path.resolve())
    # Replace \ / : with - (Claude does NOT collapse multiple dashes)
    return path_str.replace('\\', '-').replace('/', '-').replace(':', '-')


def _copy_claude_session_file(
    old_workspace: Path,
    new_workspace: Path,
    session_id: str,
) -> bool:
    """Copy Claude session file from old workspace to new workspace.

    Claude stores sessions in ~/.claude/projects/<encoded-cwd>/<session_id>.jsonl
    To resume a session in a new workspace, we need to copy the session file.

    Args:
        old_workspace: Original workspace directory (as used by Claude cwd)
        new_workspace: New workspace directory
        session_id: Session ID to copy

    Returns:
        True if copy succeeded, False otherwise
    """
    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        print(f"Warning: ~/.claude/projects/ not found")
        return False

    old_encoded = _encode_claude_project_path(old_workspace)
    new_encoded = _encode_claude_project_path(new_workspace)

    old_session_dir = claude_projects / old_encoded
    new_session_dir = claude_projects / new_encoded

    if not old_session_dir.exists():
        print(f"Warning: Original session directory not found: {old_session_dir}")
        return False

    session_file = old_session_dir / f"{session_id}.jsonl"
    if not session_file.exists():
        print(f"Warning: Session file not found: {session_file}")
        return False

    # Create new session directory and copy the file
    new_session_dir.mkdir(parents=True, exist_ok=True)
    new_session_file = new_session_dir / f"{session_id}.jsonl"

    try:
        shutil.copy2(session_file, new_session_file)
        print(f"  Copied session file to new workspace")
        return True
    except Exception as e:
        print(f"Warning: Failed to copy session file: {e}")
        return False


def _derive_resume_run_id(run_group_dir: Path, parent_run_id: str) -> str:
    # Extract base ID if parent is already a resume (e.g., "abc123_resume-01" -> "abc123")
    resume_match = re.match(r"^(.+?)_resume-\d+$", parent_run_id)
    base_id = resume_match.group(1) if resume_match else parent_run_id
    base = f"{base_id}_resume"
    suffix = 1
    while True:
        candidate = f"{base}-{suffix:02d}"
        candidate_path = run_group_dir / candidate
        if not candidate_path.exists():
            return candidate
        suffix += 1


def _replicate_run_state(src: Path, dst: Path, symlink_workspace: bool = False) -> None:
    """Copy run state from src to dst for resume.

    Args:
        src: Source run directory
        dst: Destination run directory
        symlink_workspace: If True, symlink workspace/input instead of copying.
            Useful when workspace has many files (models, datasets).
    """
    # Directories to skip during resume (regenerated automatically or too large)
    skip_dirs = {".uv_cache", "__pycache__", ".venv", "venv"}

    for item in src.iterdir():
        if item.name in skip_dirs:
            continue
        target = dst / item.name

        # Handle workspace specially - can symlink to avoid copying large dirs
        if item.name == "workspace" and symlink_workspace:
            # Symlink workspace/input to original (Windows needs special handling)
            src_input = item / "input"
            dst_input = target / "input"
            if src_input.exists():
                target.mkdir(parents=True, exist_ok=True)
                # Remove dst_input if it exists (created by env.reset)
                if dst_input.exists():
                    shutil.rmtree(dst_input)
                try:
                    # Try symlink first (requires admin or dev mode on Windows)
                    dst_input.symlink_to(src_input, target_is_directory=True)
                except OSError:
                    # Fallback: use junction on Windows
                    import subprocess
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(dst_input), str(src_input)],
                        capture_output=True
                    )
            continue

        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _write_usage_summary(run_dir: Path, agent_type: str = "basic-agent") -> None:
    """Write usage summary, trying agent-specific format first."""
    try:
        if agent_type == "claude-code":
            # Try claude-code format (cost_summary.json)
            usage = _collect_claude_code_resume_usage(run_dir)
        elif agent_type == "codex":
            usage = _collect_codex_resume_usage(run_dir)
        else:
            # Try inspect log format (basic-agent)
            usage = _collect_resume_usage(run_dir)
    except Exception as exc:
        print(f"Warning: unable to write usage summary for {run_dir}: {exc}")
        return
    summary_path = run_dir / "usage_summary.json"
    payload = {
        "hours_spent": usage.hours_spent,
        "budget_spent": usage.budget_spent,
        "log_files": usage.log_files,
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    summary_path.write_text(json.dumps(payload, indent=2))


IDEA_HINT_HEADER = "## Idea Hint"


def _load_idea_hint_text(task_dir: Path) -> str:
    hint_path = task_dir / "idea_hint.txt"
    if not hint_path.exists():
        raise FileNotFoundError(f"--idea_hint enabled but idea_hint.txt not found at {hint_path}")
    idea_text = hint_path.read_text(encoding="utf-8").strip()
    if not idea_text:
        raise ValueError(f"--idea_hint enabled but {hint_path} is empty.")
    return idea_text


def _inject_idea_hint_section(target_file: Path, idea_hint: str) -> Path:
    """Append or replace the idea hint section in the given file."""
    existing = target_file.read_text(encoding="utf-8")
    section = f"{IDEA_HINT_HEADER}\n\n{idea_hint.strip()}\n"
    pattern = re.compile(r"^## Idea Hint\b.*?(?=^#|\Z)", re.MULTILINE | re.DOTALL)
    if pattern.search(existing):
        updated = pattern.sub(section + "\n", existing)
    else:
        base = existing.rstrip()
        updated = f"{base}\n\n{section}" if base else section
    target_file.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return target_file


def _apply_idea_hint(input_dir: Path, idea_hint: Optional[str]) -> Optional[Path]:
    """Apply idea_hint to task_description.md under input_dir."""
    if not idea_hint:
        return None
    target = input_dir / "task_description.md"
    if not target.exists():
        raise FileNotFoundError(f"--idea_hint enabled but task_description.md not found under {input_dir}")
    _inject_idea_hint_section(target, idea_hint)
    return target


def _apply_idea_hint_or_exit(input_dir: Path, idea_hint: Optional[str]) -> Optional[Path]:
    """Wrapper that exits cleanly on idea-hint failures."""
    if not idea_hint:
        return None
    try:
        return _apply_idea_hint(input_dir, idea_hint)
    except Exception as exc:
        print(f"Failed to apply idea hint: {exc}")
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent against a ResearchGym task")
    parser.add_argument("task_dir", type=Path, help="Path to task directory (e.g., tasks/continual-learning)")
    parser.add_argument(
        "agent",
        choices=["ml-master", "ai-scientist", "basic-agent", "basic-agent-evolution", "openevolve", "claude-code", "codex"],
        help="Agent to run",
    )
    parser.add_argument("--runs_dir", type=Path, default=PKG_ROOT / "runs", dest="runs_dir")
    parser.add_argument("--dry_run", action="store_true", help="Do not execute, only plan")

    # Resume options
    parser.add_argument("--resume", type=Path, default=None, help="Absolute path to existing run_dir to resume (e.g., .../runs/2025-09-11/<run_id>)")
    parser.add_argument("--resume_with_instruction", action="store_true", help="When resuming, start a fresh conversation with a minimal resume note prepended to the original instructions")
    parser.add_argument("--idea_hint", action="store_true", help="Append tasks/<task>/idea_hint.txt to the task description used for prompting")

    # ML-Master minimal args
    # Model selection argument
    parser.add_argument("--model", type=str, 
                       help="Model name to use (e.g., 'gemini-2.5-pro', 'gpt-5')")
    
    # Generic agent config (model/backends)
    parser.add_argument("--code_model", type=str, default="")
    parser.add_argument("--code_temp", type=float, default=0.5)
    parser.add_argument("--code_base_url", type=str, default="")
    parser.add_argument("--code_api_key", type=str, default="")
    parser.add_argument("--feedback_model", type=str, default="")
    parser.add_argument("--feedback_temp", type=float, default=0.5)
    parser.add_argument("--feedback_base_url", type=str, default="")
    parser.add_argument("--feedback_api_key", type=str, default="")
    parser.add_argument("--ml_master_root", type=Path, default=None)
    parser.add_argument("--desc_file", type=Path, default=None)
    parser.add_argument("--runtime", choices=["docker", "uv"], default="uv")
    parser.add_argument("--image", type=str, default="researchgym-base:latest")
    parser.add_argument("--gpus", action="store_true")
    # AI-Scientist args
    parser.add_argument("--ai_scientist_root", type=Path, default=None)
    parser.add_argument("--ai_desc_file", type=Path, default=None)
    parser.add_argument("--ai_steps", type=int, default=5)
    parser.add_argument("--ai_workers", type=int, default=3)
    parser.add_argument("--ai_code_model", type=str, default="openai/gpt-5")
    parser.add_argument("--ai_feedback_model", type=str, default="openai/gpt-5")
    parser.add_argument("--ai_vlm_model", type=str, default="openai/gpt-5")
    parser.add_argument("--ai_code_temp", type=float, default=1.0)
    parser.add_argument("--ai_feedback_temp", type=float, default=0.5)
    parser.add_argument("--ai_vlm_temp", type=float, default=0.5)
    parser.add_argument("--ai_report_model", type=str, default="openai/gpt-5")
    parser.add_argument("--ai_report_temp", type=float, default=1.0)
    parser.add_argument("--ai_hours", type=float, default=3.0, help="Max wall-clock hours for AI-Scientist runs")
    
    # LiteLLM configuration
    parser.add_argument("--litellm_base_url", type=str, default="", help="LiteLLM base URL for proxying API calls")
    parser.add_argument("--litellm_prelude", type=str, default="", help="LiteLLM prelude commands")

    # BasicAgent args
    parser.add_argument("--basic_agent_root", type=Path, default=None)
    parser.add_argument("--basic_agent_evolution_root", type=Path, default=None)
    # OpenEvolve args (initially minimal)
    parser.add_argument("--openevolve_root", type=Path, default=None)
    parser.add_argument("--openevolve_iterations", type=int, default=100)
    parser.add_argument("--basic_hours", type=float, default=0.25, help="Max wall-clock hours for BasicAgent")
    parser.add_argument("--basic_iterative", action="store_true", help="Use iterative BasicAgent loop")
    parser.add_argument("--basic_disallow_submit", action="store_true", help="Hide end_task tool")
    parser.add_argument("--basic_code_only", action="store_true", help="Code-only system message variant")
    parser.add_argument("--budget_limit", type=float, default=10.0, help="Maximum budget in USD for LLM API calls (0 for no limit)")

    # ClaudeCode args
    parser.add_argument("--claude_hours", type=float, default=0.25, help="Max wall-clock hours for ClaudeCode agent")

    # Codex args
    parser.add_argument("--codex_hours", type=float, default=0.25, help="Max wall-clock hours for Codex agent")
    parser.add_argument("--codex_model", type=str, default="", help="Codex model override (e.g., gpt-5-codex)")
    parser.add_argument(
        "--codex_reasoning_effort",
        type=str,
        default="xhigh",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level for Codex (minimal, low, medium, high, xhigh). Default: xhigh"
    )
    parser.add_argument(
        "--codex_subscription",
        action="store_true",
        help="Use ChatGPT subscription instead of API (requires prior 'codex login')"
    )

    args = parser.parse_args()
    _ensure_git_bash_in_path()

    # Determine default models based on --model argument and Azure configuration
    def get_default_models():
        if args.model:
            # Use the exact model name provided by the user
            return args.model, args.model
        else:
            # Default based on Azure availability
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
            azure_available = azure_endpoint and azure_api_key
            
            if azure_available:
                return "gpt-5", "gpt-5"
            else:
                return "gemini/gemini-2.5-flash-lite", "gemini/gemini-2.5-flash-lite"
    
    default_code_model, default_feedback_model = get_default_models()
    
    # Set default models if not provided
    if not args.code_model:
        args.code_model = default_code_model
    if not args.feedback_model:
        args.feedback_model = default_feedback_model

    runs_dir_abs = args.runs_dir.resolve()
    env = AgenticEnv(base_runs_dir=args.runs_dir)
    task_dir_resolved = args.task_dir.resolve()

    # Allow enabling idea hints via CLI flag or env var (useful in Docker where envs are easier to pass)
    idea_hint_enabled = args.idea_hint or os.environ.get("RG_IDEA_HINT", "false").strip().lower() == "true"

    idea_hint_text: Optional[str] = None
    if idea_hint_enabled:
        try:
            idea_hint_text = _load_idea_hint_text(task_dir_resolved)
        except Exception as exc:
            print(f"Failed to load idea hint: {exc}")
            sys.exit(2)

    # Plan run identifiers and environment initialization (supports resume)
    is_resuming = args.resume is not None
    resume_dir = None
    resume_usage = None
    resumed_into_new_dir = False
    if is_resuming:
        resume_dir = args.resume.resolve()
        if not resume_dir.exists() or not resume_dir.is_dir():
            print(f"--resume path does not exist or is not a directory: {resume_dir}")
            sys.exit(2)
        resume_run_group = resume_dir.parent.name
        resume_run_id = resume_dir.name
        try:
            resume_dir.relative_to(runs_dir_abs)
        except Exception:
            print(f"--resume path {resume_dir} is not under --runs_dir {runs_dir_abs}. Set --runs_dir appropriately or pass a path under it.")
            sys.exit(2)
        meta_path = resume_dir / "metadata.json"
        parent_meta = {}
        if meta_path.exists():
            try:
                parent_meta = json.loads(meta_path.read_text())
                prior_task_dir = Path(parent_meta.get("task_dir", "")).resolve() if parent_meta.get("task_dir") else None
                if prior_task_dir and prior_task_dir != task_dir_resolved:
                    print(f"Task dir mismatch. metadata.json has {prior_task_dir}, CLI provided {task_dir_resolved}.")
                    sys.exit(2)
            except Exception:
                parent_meta = {}

        if args.agent == "basic-agent":
            transcript_path = resume_dir / "transcript.json"
            if not transcript_path.exists():
                print(f"Transcript file not found at {transcript_path}. Cannot resume without conversation history.")
                sys.exit(2)
            log_dir = resume_dir / "logs"
            has_json_logs = any(log_dir.glob("*.json"))
            has_eval_logs = any(log_dir.glob("*.eval"))
            if not has_json_logs or not has_eval_logs:
                missing = []
                if not has_json_logs:
                    missing.append(".json")
                if not has_eval_logs:
                    missing.append(".eval")
                print(f"Required log files {', '.join(missing)} missing under {log_dir}. Restore the logs before resuming.")
                sys.exit(2)
            try:
                resume_usage = _collect_resume_usage(resume_dir)
            except Exception as exc:
                print(f"Unable to resume: {exc}")
                sys.exit(2)
            requested_total_hours = args.basic_hours
            if requested_total_hours <= 0:
                print(f"--basic_hours must represent the TOTAL desired hours when resuming. Received {requested_total_hours}.")
                sys.exit(2)
            hours_remaining = requested_total_hours - resume_usage.hours_spent
            if hours_remaining <= 0:
                print(
                    f"No time remaining: already used {resume_usage.hours_spent:.2f}h, "
                    f"requested total {requested_total_hours:.2f}h."
                )
                sys.exit(2)
            total_budget = args.budget_limit
            if total_budget > 0:
                budget_remaining = total_budget - resume_usage.budget_spent
                if budget_remaining <= 0:
                    print(
                        f"No budget remaining: already used ${resume_usage.budget_spent:.2f}, "
                        f"requested total ${total_budget:.2f}."
                    )
                    sys.exit(2)
            else:
                budget_remaining = 0.0

            args.basic_hours = hours_remaining
            if total_budget > 0:
                args.budget_limit = budget_remaining

            run_group = resume_run_group
            run_id = _derive_resume_run_id(runs_dir_abs / run_group, resume_run_id)
            obs = env.reset(task_dir=args.task_dir, run_group=run_group, run_id=run_id)
            new_run_dir = env.run_dir
            _replicate_run_state(resume_dir, new_run_dir)
            resumed_into_new_dir = True

            resume_generation = parent_meta.get("resume_generation", 0) + 1
            meta = {
                "task_dir": str(task_dir_resolved),
                "run_group": run_group,
                "run_id": run_id,
                "resumed_from": str(resume_dir),
                "resumed_from_run_group": resume_run_group,
                "resumed_from_run_id": resume_run_id,
                "resume_generation": resume_generation,
                "resume_total_hours": requested_total_hours,
                "resume_hours_consumed": resume_usage.hours_spent,
                "resume_hours_remaining": hours_remaining,
                "resume_total_budget": total_budget,
                "resume_budget_consumed": resume_usage.budget_spent,
                "resume_budget_remaining": budget_remaining if total_budget > 0 else None,
            }
            (new_run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            (new_run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "resuming",
                        "resumed_from": str(resume_dir),
                        "remaining_hours": hours_remaining,
                        "remaining_budget": budget_remaining if total_budget > 0 else "unlimited",
                    },
                    indent=2,
                )
            )
            print(
                f"Created resumed run {run_group}/{run_id} from {resume_dir} "
                f"(used {resume_usage.hours_spent:.2f}h / ${resume_usage.budget_spent:.2f})."
            )
            if total_budget > 0:
                print(
                    f"Total targets -> {requested_total_hours:.2f}h & ${total_budget:.2f}; "
                    f"remaining allocation -> {hours_remaining:.2f}h & ${budget_remaining:.2f}."
                )
            else:
                print(
                    f"Total target -> {requested_total_hours:.2f}h; remaining allocation -> {hours_remaining:.2f}h (budget unlimited)."
                )

        elif args.agent == "claude-code":
            # ClaudeCode uses session.json + cost_summary.json for resume
            # Can also resume with just transcript.json (transcript seeding fallback)
            session_id = _get_claude_code_session_id(resume_dir)
            cost_summary_path = resume_dir / "logs" / "cost_summary.json"
            transcript_path = resume_dir / "logs" / "transcript.json"
            if not cost_summary_path.exists():
                print(f"cost_summary.json not found at {cost_summary_path}. Cannot resume without usage data.")
                sys.exit(2)
            if not session_id and not transcript_path.exists():
                print(f"Neither session.json nor transcript.json found in {resume_dir / 'logs'}. Cannot resume.")
                sys.exit(2)
            if not session_id:
                print(f"  Note: No session_id found, will use transcript seeding for resume.")

            requested_total_hours = args.claude_hours

            try:
                # Pass original_hours for accurate cumulative time tracking across multi-gen resumes
                resume_usage = _collect_claude_code_resume_usage(resume_dir, original_hours=requested_total_hours)
            except Exception as exc:
                print(f"Unable to resume: {exc}")
                sys.exit(2)
            if requested_total_hours <= 0:
                print(f"--claude_hours must represent the TOTAL desired hours when resuming. Received {requested_total_hours}.")
                sys.exit(2)
            hours_remaining = requested_total_hours - resume_usage.hours_spent
            if hours_remaining <= 0:
                print(
                    f"No time remaining: already used {resume_usage.hours_spent:.2f}h, "
                    f"requested total {requested_total_hours:.2f}h."
                )
                sys.exit(2)
            total_budget = args.budget_limit
            if total_budget > 0:
                budget_remaining = total_budget - resume_usage.budget_spent
                if budget_remaining <= 0:
                    print(
                        f"No budget remaining: already used ${resume_usage.budget_spent:.2f}, "
                        f"requested total ${total_budget:.2f}."
                    )
                    sys.exit(2)
            else:
                budget_remaining = 0.0

            # Update args with remaining time
            # NOTE: budget_limit stays as TOTAL budget (not remaining)
            # CostTracker inherits previous costs and compares against total
            args.claude_hours = hours_remaining
            # args.budget_limit stays as total_budget (not budget_remaining)

            run_group = resume_run_group
            run_id = _derive_resume_run_id(runs_dir_abs / run_group, resume_run_id)
            obs = env.reset(task_dir=args.task_dir, run_group=run_group, run_id=run_id)
            new_run_dir = env.run_dir
            # Use symlink for workspace to avoid copying large dirs (models, datasets)
            _replicate_run_state(resume_dir, new_run_dir, symlink_workspace=True)
            resumed_into_new_dir = True

            # Store session_id for later use in dispatch
            # Use placeholder if no session_id but transcript exists (for transcript seeding)
            args._claude_resume_session_id = session_id or "transcript-seeding"

            resume_generation = parent_meta.get("resume_generation", 0) + 1
            meta = {
                "task_dir": str(task_dir_resolved),
                "run_group": run_group,
                "run_id": run_id,
                "resumed_from": str(resume_dir),
                "resumed_from_run_group": resume_run_group,
                "resumed_from_run_id": resume_run_id,
                "resume_generation": resume_generation,
                "resume_total_hours": requested_total_hours,
                "resume_hours_consumed": resume_usage.hours_spent,
                "resume_hours_remaining": hours_remaining,
                "resume_total_budget": total_budget,
                "resume_budget_consumed": resume_usage.budget_spent,
                "resume_budget_remaining": budget_remaining if total_budget > 0 else None,
                "resume_session_id": session_id,
            }
            (new_run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            (new_run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "resuming",
                        "resumed_from": str(resume_dir),
                        "remaining_hours": hours_remaining,
                        "remaining_budget": budget_remaining if total_budget > 0 else "unlimited",
                        "session_id": session_id,
                    },
                    indent=2,
                )
            )
            print(
                f"Created resumed run {run_group}/{run_id} from {resume_dir} "
                f"(used {resume_usage.hours_spent:.2f}h / ${resume_usage.budget_spent:.2f})."
            )
            print(f"Resuming session: {session_id}")
            if total_budget > 0:
                print(
                    f"Total targets -> {requested_total_hours:.2f}h & ${total_budget:.2f}; "
                    f"remaining allocation -> {hours_remaining:.2f}h & ${budget_remaining:.2f}."
                )
            else:
                print(
                    f"Total target -> {requested_total_hours:.2f}h; remaining allocation -> {hours_remaining:.2f}h (budget unlimited)."
                )

        elif args.agent == "codex":
            session_id = _get_codex_session_id(resume_dir)
            cost_summary_path = resume_dir / "logs" / "cost_summary.json"
            transcript_path = resume_dir / "logs" / "codex_output.jsonl"
            if not cost_summary_path.exists():
                print(f"cost_summary.json not found at {cost_summary_path}. Cannot resume without usage data.")
                sys.exit(2)
            if not transcript_path.exists() and not session_id:
                print(f"No codex_output.jsonl or session_id found in {resume_dir / 'logs'}. Cannot resume.")
                sys.exit(2)

            requested_total_hours = args.codex_hours

            try:
                # Pass original_hours for accurate cumulative time tracking across multi-gen resumes
                resume_usage = _collect_codex_resume_usage(resume_dir, original_hours=requested_total_hours)
            except Exception as exc:
                print(f"Unable to resume: {exc}")
                sys.exit(2)
            if requested_total_hours <= 0:
                print(f"--codex_hours must represent the TOTAL desired hours when resuming. Received {requested_total_hours}.")
                sys.exit(2)
            hours_remaining = requested_total_hours - resume_usage.hours_spent
            if hours_remaining <= 0:
                print(
                    f"No time remaining: already used {resume_usage.hours_spent:.2f}h, "
                    f"requested total {requested_total_hours:.2f}h."
                )
                sys.exit(2)

            total_budget = args.budget_limit
            if total_budget > 0:
                budget_remaining = total_budget - resume_usage.budget_spent
                if budget_remaining <= 0:
                    print(
                        f"No budget remaining: already used ${resume_usage.budget_spent:.2f}, "
                        f"requested total ${total_budget:.2f}."
                    )
                    sys.exit(2)
            else:
                budget_remaining = 0.0

            args.codex_hours = hours_remaining

            run_group = resume_run_group
            run_id = _derive_resume_run_id(runs_dir_abs / run_group, resume_run_id)
            obs = env.reset(task_dir=args.task_dir, run_group=run_group, run_id=run_id)
            new_run_dir = env.run_dir
            _replicate_run_state(resume_dir, new_run_dir, symlink_workspace=True)
            resumed_into_new_dir = True

            # For Codex, prefer transcript seeding over native resume
            # Native resume only if no transcript exists
            use_native_resume = bool(session_id and not transcript_path.exists())
            args._codex_resume_session_id = session_id if use_native_resume else None
            # Always pass inherited cost path for cost tracking continuity
            args._codex_inherited_cost_path = cost_summary_path

            resume_generation = parent_meta.get("resume_generation", 0) + 1
            meta = {
                "task_dir": str(task_dir_resolved),
                "run_group": run_group,
                "run_id": run_id,
                "resumed_from": str(resume_dir),
                "resumed_from_run_group": resume_run_group,
                "resumed_from_run_id": resume_run_id,
                "resume_generation": resume_generation,
                "resume_total_hours": requested_total_hours,
                "resume_hours_consumed": resume_usage.hours_spent,
                "resume_hours_remaining": hours_remaining,
                "resume_total_budget": total_budget,
                "resume_budget_consumed": resume_usage.budget_spent,
                "resume_budget_remaining": budget_remaining if total_budget > 0 else None,
                "resume_session_id": session_id,
            }
            (new_run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            (new_run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "resuming",
                        "resumed_from": str(resume_dir),
                        "remaining_hours": hours_remaining,
                        "remaining_budget": budget_remaining if total_budget > 0 else "unlimited",
                        "session_id": session_id,
                    },
                    indent=2,
                )
            )

            if transcript_path.exists():
                prev_transcript = new_run_dir / "logs" / "previous_transcript.jsonl"
                prev_transcript.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(transcript_path, prev_transcript)
                print(f"  Copied transcript for seeding: {prev_transcript}")
                # Also copy cost summary for inheritance detection
                if cost_summary_path.exists():
                    prev_cost = new_run_dir / "logs" / "previous_cost_summary.json"
                    shutil.copy2(cost_summary_path, prev_cost)
                    print(f"  Copied cost summary for inheritance: {prev_cost}")
            elif use_native_resume:
                print(f"  Using native resume session: {session_id}")

            print(
                f"Created resumed run {run_group}/{run_id} from {resume_dir} "
                f"(used {resume_usage.hours_spent:.2f}h / ${resume_usage.budget_spent:.2f})."
            )
            if total_budget > 0:
                print(
                    f"Total targets -> {requested_total_hours:.2f}h & ${total_budget:.2f}; "
                    f"remaining allocation -> {hours_remaining:.2f}h & ${budget_remaining:.2f}."
                )
            else:
                print(
                    f"Total target -> {requested_total_hours:.2f}h; remaining allocation -> {hours_remaining:.2f}h (budget unlimited)."
                )

        else:
            run_group, run_id = resume_run_group, resume_run_id
            obs = env.reset(task_dir=args.task_dir, run_group=run_group, run_id=run_id)
            if env.run_dir.resolve() != resume_dir:
                print(f"Resolved env.run_dir {env.run_dir} differs from --resume path {resume_dir}. Check --runs_dir.")
                sys.exit(2)
            (env.run_dir / "status.json").write_text(json.dumps({"status": "resuming"}, indent=2))
            print(f"Resuming run: group={run_group} id={run_id}")
    else:
        run_group, run_id = _gen_ids()
        obs = env.reset(task_dir=args.task_dir, run_group=run_group, run_id=run_id)

    logger = setup_file_logger("runner", env.logs_dir / "runner.log")
    logger.info(f"Run created: group={run_group} id={run_id}")
    (env.run_dir / "status.json").write_text(json.dumps({"status": "initialized"}, indent=2))
    print(f"Run group: {run_group}")
    print(f"Run id: {run_id}")
    print(f"Run dir: {env.run_dir}")

    # Ensure a git repo exists in the workspace input for this run
    try:
        input_dir = env.workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _ensure_git_initialized(input_dir, logger)
    except Exception as e:
        # Non-fatal; proceed even if git is unavailable
        logger.warning(f"Could not initialize git in workspace: {e}")

    if args.agent == "ml-master":
        adapter = MLMasterAdapter(env=env, run_group=run_group, run_id=run_id)
        # Resolve ML-Master root if not provided
        ml_root = args.ml_master_root
        if ml_root is None:
            candidates = [
                PKG_ROOT / "agents" / "ML-Master",
                PKG_ROOT / "agents" / "ml-master",
                PKG_ROOT / "agents" / "ML-Master-main",
                PKG_ROOT / "vendor" / "ML-Master",
            ]
            ml_root = next((c for c in candidates if (c / "main_mcts.py").exists()), None)
            if ml_root is None:
                print("Could not auto-detect ML-Master. Set --ml_master_root to the ML-Master directory.")
                print("Tried candidates:")
                for c in candidates:
                    print(f" - {c}")
                sys.exit(2)
        print(f"Using ML-Master root: {ml_root}")
        # When resuming, do not copy/overlay the task into the existing workspace
        if not is_resuming:
            adapter.prepare_workspace(task_dir=args.task_dir)
        idea_desc_path = _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name
        desc_file = idea_desc_path or (args.desc_file if args.desc_file else (args.task_dir / "description.md"))
        
        # Setup LiteLLM proxy if using non-OpenAI models without custom base URLs
        proxy_url = None
        def needs_proxy(model: str) -> bool:
            if not model or "/" not in model:
                return False
            provider = model.split("/")[0]
            return provider not in ["openai", "azure"]
        
        if (needs_proxy(args.code_model) or needs_proxy(args.feedback_model)) and not args.code_base_url and not args.feedback_base_url:
            try:
                proxy_url = adapter.setup_litellm_proxy(args.code_model, args.feedback_model)
                logger.info(f"LiteLLM proxy started at {proxy_url}")
            except Exception as e:
                logger.error(f"Failed to setup LiteLLM proxy: {e}")
                print(f"Failed to setup LiteLLM proxy: {e}")
                sys.exit(1)
        
        # Determine base URLs and API keys for each model
        code_base_url = args.code_base_url or args.litellm_base_url
        code_api_key = args.code_api_key
        feedback_base_url = args.feedback_base_url or args.litellm_base_url  
        feedback_api_key = args.feedback_api_key
        
        # Use proxy for non-OpenAI models if proxy is active
        if proxy_url:
            if needs_proxy(args.code_model):
                code_base_url = proxy_url
                code_api_key = "sk-1234"
            if needs_proxy(args.feedback_model):
                feedback_base_url = proxy_url
                feedback_api_key = "sk-1234"
        
        cfg = MLMasterConfig(
            task_id=task_id,
            desc_file=desc_file if desc_file.exists() else None,
            code_model=args.code_model,
            code_temp=args.code_temp,
            code_base_url=code_base_url,
            code_api_key=code_api_key,
            feedback_model=args.feedback_model,
            feedback_temp=args.feedback_temp,
            feedback_base_url=feedback_base_url,
            feedback_api_key=feedback_api_key,
        )
        # Build the inner agent command (python main_mcts.py ...)
        agent_plan = adapter.run(cfg=cfg, ml_master_root=ml_root, dry_run=True)
        inner_cmd = agent_plan.info.get("command", [])

        # Detect task overlay env files
        overlay = detect_task_overlay(args.task_dir)
        runtime_plan = {}
        if args.runtime == "docker":
            # Build a prelude to apply task overlays inside the container
            docker_install_steps = []
            if overlay.get("install_sh"):
                docker_install_steps.append("chmod +x /task/install.sh || true && bash /task/install.sh")
            elif overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            # Provide apply_patch in container pointing to vendored script
            prelude_parts.append("echo '#!/bin/bash' > /usr/local/bin/apply_patch && echo 'python /agent/apply_patch.py \"$@\"' >> /usr/local/bin/apply_patch && chmod +x /usr/local/bin/apply_patch")
            # also provide applypatch alias for model artifacts
            prelude_parts.append("ln -sf /usr/local/bin/apply_patch /usr/local/bin/applypatch || cp /usr/local/bin/apply_patch /usr/local/bin/applypatch")
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts
            docker_prelude = " && ".join(prelude_parts) if prelude_parts else None
            # Ensure pip root-user warning is silenced inside container
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && {docker_prelude}"
                if docker_prelude
                else "export PIP_ROOT_USER_ACTION=ignore"
            )

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=ml_root,
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "GEMINI_API_KEY",
                    "OPENROUTER_API_KEY",
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_API_VERSION",
                ],
                prelude=docker_prelude,
                task_dir=args.task_dir,
            )
        else:
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-ml-master",
                task_overlay=overlay,
                project_root=ml_root,
                command=inner_cmd,
            )
        combined = {"agent": agent_plan.info, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
    elif args.agent == "ai-scientist":
        adapter = AIScientistAdapter(env=env, run_group=run_group, run_id=run_id)
        ai_root = args.ai_scientist_root
        if ai_root is None:
            candidates = [
                PKG_ROOT / "agents" / "AI-Scientist-v2",
                PKG_ROOT / "vendor" / "AI-Scientist-v2",
            ]
            ai_root = next((c for c in candidates if (c / "launch_scientist_bfts.py").exists()), None)
            if ai_root is None:
                print("Could not auto-detect AI-Scientist-v2. Set --ai_scientist_root to the directory.")
                for c in candidates:
                    print(f" - {c}")
                sys.exit(2)
        print(f"Using AI-Scientist root: {ai_root}")
        adapter.prepare_workspace(task_dir=args.task_dir)
        idea_desc_path = _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name

        # Setup LiteLLM proxy if needed
        proxy_url = None
        def needs_proxy(model: str) -> bool:
            if not model or "/" not in model:
                return False
            provider = model.split("/")[0]
            return provider not in ["openai", "azure"]

        model_list = [args.ai_code_model, args.ai_feedback_model, args.ai_vlm_model]
        if any(needs_proxy(m) for m in model_list) and not args.litellm_base_url:
            try:
                proxy_url = adapter.setup_litellm_proxy(model_list)
            except Exception as e:
                print(f"Failed to setup LiteLLM proxy: {e}")
                sys.exit(1)

        # Build config
        cfg = AIScientistConfig(
            task_id=task_id,
            desc_file=idea_desc_path or args.ai_desc_file or (args.task_dir / "task_description.md"),
            steps=args.ai_steps,
            num_workers=args.ai_workers,
            code_model=args.ai_code_model,
            feedback_model=args.ai_feedback_model,
            vlm_model=args.ai_vlm_model,
            code_temp=args.ai_code_temp,
            feedback_temp=args.ai_feedback_temp,
            vlm_temp=args.ai_vlm_temp,
            base_url=args.litellm_base_url or proxy_url or "",
            report_model=args.ai_report_model,
            report_temp=args.ai_report_temp,
            time_limit_secs=int(args.ai_hours * 60 * 60) + 60,
            cost_limit=float(os.getenv("RG_BUDGET_LIMIT", str(args.budget_limit))) if hasattr(args, "budget_limit") else 10.0,
        )

        # Build inner command and environment without using adapter dry-run
        inner_cmd = adapter.build_command(cfg=cfg, ai_root=ai_root)
        inner_env = getattr(adapter, "_last_env_hint", {})

        # Always enable transcript persistence for Basic-Agent runs so future
        # resumes have a transcript to seed from.
        try:
            transcript_path = env.run_dir / "transcript.json"
            inner_env["RG_TRANSCRIPT_PATH"] = str(transcript_path)
        except Exception:
            pass

        overlay = detect_task_overlay(args.task_dir)
        if args.runtime == "docker":
            docker_install_steps = []
            # Install task overlays if present
            if overlay.get("install_sh"):
                docker_install_steps.append("chmod +x /task/install.sh || true && bash /task/install.sh")
            elif overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts
            docker_prelude = " && ".join(prelude_parts) if prelude_parts else None
            # Ensure pip root-user warning is silenced inside container
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && {docker_prelude}"
                if docker_prelude
                else "export PIP_ROOT_USER_ACTION=ignore"
            )

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=ai_root,
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY","AZURE_OPENAI_API_KEY","AZURE_OPENAI_ENDPOINT",
                    "ANTHROPIC_API_KEY","GOOGLE_API_KEY","GEMINI_API_KEY","OPENROUTER_API_KEY","S2_API_KEY",
                    "AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_REGION_NAME"
                ],
                prelude=docker_prelude,
                task_dir=args.task_dir,
            )
        else:
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-ai-scientist",
                task_overlay=overlay,
                project_root=ai_root,
                command=inner_cmd,
                env_vars=inner_env or None,
            )

        combined = {"agent": {"command": inner_cmd, "env": inner_env}, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
        (env.run_dir / "status.json").write_text(json.dumps({"status": "planned"}, indent=2))
        print(f"Plan written: {plan_path}")

        if not args.dry_run:
            (env.run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2))
            stdout_log = env.logs_dir / "exec.stdout.log"
            stderr_log = env.logs_dir / "exec.stderr.log"
            if args.runtime == "docker":
                docker_cli = runtime_plan.get("docker_cli")
                if not docker_cli:
                    print("Docker plan missing 'docker_cli'")
                    sys.exit(2)
                with open(stdout_log, "w", encoding='utf-8') as out, open(stderr_log, "w", encoding='utf-8') as err:
                    proc = subprocess.Popen(docker_cli, stdout=out, stderr=err, shell=platform.system() == "Windows")
                    rc = proc.wait()
            else:
                shell_cmd = runtime_plan.get("shell")
                if not shell_cmd:
                    print("UV plan missing 'shell'")
                    sys.exit(2)
                # Enforce wall-clock time limit similar to BasicAgent
                max_secs = int(args.ai_hours * 3600)
                start_time = time.time()
                with open(stdout_log, "w", encoding='utf-8') as out, open(stderr_log, "w", encoding='utf-8') as err:
                    proc = subprocess.Popen(shell_cmd, stdout=out, stderr=err, shell=platform.system() == "Windows")
                    rc = None
                    try:
                        while True:
                            rc = proc.poll()
                            if rc is not None:
                                break
                            elapsed = time.time() - start_time
                            if max_secs > 0 and elapsed >= max_secs:
                                try:
                                    proc.terminate()
                                    proc.wait(timeout=10)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                                rc = -9
                                break
                            time.sleep(1)
                    except KeyboardInterrupt:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        rc = -2
            (env.run_dir / "status.json").write_text(json.dumps({"status": "completed", "returncode": rc}, indent=2))
            # Post-run: integrate AI-Scientist costs into ResearchGym tracker
            try:
                summary = adapter._get_ai_scientist_cost_summary()
                adapter._integrate_ai_scientist_costs(summary)
                print(f"Integrated AI-Scientist costs: total=${summary.get('total_cost_usd', 0.0):.6f}")
            except Exception as e:
                print(f"Warning: could not integrate AI-Scientist costs: {e}")


    elif args.agent == "basic-agent":
        adapter = BasicAgentAdapter(env=env, run_group=run_group, run_id=run_id)
        agent_root = args.basic_agent_root
        if agent_root is None:
            candidates = [
                PKG_ROOT / "agents" / "BasicAgent",
            ]
            agent_root = next((c for c in candidates if (c / "start.py").exists()), None)
            if agent_root is None:
                print("Could not auto-detect BasicAgent. Set --basic_agent_root to the directory.")
                for c in candidates:
                    print(f" - {c}")
                sys.exit(2)
        print(f"Using BasicAgent root: {agent_root}")
        if not (is_resuming and resumed_into_new_dir):
            adapter.prepare_workspace(task_dir=args.task_dir)
        else:
            print("Resume detected: reusing cloned workspace without re-syncing task files.")
        _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name

        # Build config
        ba_cfg = BasicAgentConfig(
            task_id=task_id,
            model=args.model or default_code_model,
            time_hours=args.basic_hours,
            iterative=args.basic_iterative,
            disallow_submit=args.basic_disallow_submit,
            code_only=args.basic_code_only,
            time_limit_secs=int(args.basic_hours * 60 * 60) + 60,
            budget_limit=args.budget_limit,
            idea_hint=bool(idea_hint_text),
        )

        agent_plan = adapter.run(cfg=ba_cfg, agent_root=agent_root, dry_run=True)
        inner_cmd = agent_plan.info.get("command", [])
        inner_env = agent_plan.info.get("env", {})
        if is_resuming:
            inner_env["RG_EXTENDED_CONTINUE"] = "true"

        # Ensure transcript persistence for all runs (not just resume)
        try:
            transcript_path = env.run_dir / "transcript.json"
            if "RG_TRANSCRIPT_PATH" not in inner_env or not inner_env["RG_TRANSCRIPT_PATH"]:
                inner_env["RG_TRANSCRIPT_PATH"] = str(transcript_path)
        except Exception:
            pass

        # If resuming with instruction, point RG_INSTRUCTIONS_FILE to a minimal resume file
        if is_resuming and args.resume_with_instruction:
            input_dir = env.workspace_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            base_text = ""
            candidate_paths = []
            existing_env_path = inner_env.get("RG_INSTRUCTIONS_FILE")
            if existing_env_path:
                candidate_paths.append(Path(existing_env_path))
            candidate_paths.append(input_dir / "instructions.txt")
            candidate_paths.append(input_dir / "task_description.md")
            for path in candidate_paths:
                try:
                    resolved = Path(path)
                    if resolved.exists():
                        base_text = resolved.read_text(encoding="utf-8")
                        if base_text:
                            break
                except Exception:
                    continue
            resume_instr = input_dir / "instructions_resume.txt"
            header = "You are resuming in the same workspace. Inspect files under CODE_DIR and continue from your last step.\n\n"
            try:
                resume_instr.write_text(header + base_text)
                inner_env["RG_INSTRUCTIONS_FILE"] = str(resume_instr)
            except Exception:
                # If write fails, fall back silently
                pass

        # If strict resume (no instruction), wire transcript seeding as well
        if is_resuming and not args.resume_with_instruction:
            transcript_path = env.run_dir / "transcript.json"
            inner_env["RG_RESUME_CONTEXT_FILE"] = str(transcript_path)
            # RG_TRANSCRIPT_PATH already set above for persistence

        # Detect task overlays from the original task directory (not the filtered copy)
        # so we still install requirements/install.sh even when we hide them from the agent.
        # For docker we need to mount the real task dir (the filtered workspace copy omits
        # install.sh/requirements.txt), otherwise the prelude can't find them.
        task_mount = args.task_dir.resolve() if args.runtime == "docker" else env.workspace_dir / "input"
        overlay = detect_task_overlay(args.task_dir)
        if args.runtime == "docker":
            docker_install_steps = []
            # Ensure BasicAgent dependencies are available inside the container when requirements exist
            docker_install_steps.append("pip install -r /agent/requirements.txt")
            if overlay.get("install_sh"):
                docker_install_steps.append("chmod +x /task/install.sh || true && bash /task/install.sh")
            elif overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts
            # Inject BasicAgent env exports so tools know workspace locations inside container
            docker_env_exports = None
            if inner_env:
                inner_env["DISABLE_BROWSER"] = "true"
                # Remap host paths to container mounts
                def _remap_path(val: str) -> str:
                    try:
                        if isinstance(val, str):
                            mapped = val
                            mapped = mapped.replace(str(env.run_dir), "/run")
                            mapped = mapped.replace(str(agent_root), "/agent")
                            return mapped
                    except Exception:
                        pass
                    return val

                remapped = {k: _remap_path(v) for k, v in inner_env.items()}
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in remapped.items()])
                # Ensure GOOGLE_API_KEY is set when only GEMINI_API_KEY is provided
                exports = exports + " && export GOOGLE_API_KEY=${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
                docker_env_exports = exports
            docker_prelude = " && ".join([p for p in [docker_env_exports, " && ".join(prelude_parts) if prelude_parts else None] if p]) if (docker_env_exports or prelude_parts) else None
            # Pre-setup MuJoCo for RL tasks (mujoco-py expects binaries in ~/.mujoco/mujoco210)
            # The RL image may already have /root/.mujoco/mujoco210 - use it if present
            mujoco_setup_cmd = (
                "if [ ! -d /root/.mujoco/mujoco210 ]; then "
                "mkdir -p /opt/mujoco && "
                "curl -sL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C /opt/mujoco && "
                "mkdir -p /root/.mujoco && ln -sfn /opt/mujoco/mujoco210 /root/.mujoco/mujoco210; "
                "fi"
            )
            mujoco_env = "export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
            # Ensure pip root-user warning is silenced inside container
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && {mujoco_env} && {mujoco_setup_cmd} && {docker_prelude}"
                if docker_prelude
                else f"export PIP_ROOT_USER_ACTION=ignore && {mujoco_env} && {mujoco_setup_cmd}"
            )

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=agent_root,
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY","AZURE_OPENAI_API_KEY","AZURE_OPENAI_ENDPOINT", "AZUREAI_OPENAI_API_KEY", "AZUREAI_OPENAI_BASE_URL", "SEMANTIC_SCHOLAR_API_KEY",
                    "ANTHROPIC_API_KEY","GOOGLE_API_KEY","GEMINI_API_KEY","OPENROUTER_API_KEY","EXA_API_KEY"
                ],
                prelude=docker_prelude,
                task_dir=task_mount,
            )
        else:
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-basic-agent",
                task_overlay=overlay,
                project_root=agent_root,
                command=inner_cmd,
            )
            # Inject apply_patch into UV venv PATH by creating a small wrapper before agent run
            shell_cmd = runtime_plan.get("shell")
            if shell_cmd and isinstance(shell_cmd, list) and len(shell_cmd) == 3:
                if platform.system() == "Windows":
                    # Safely rewrite the generated .cmd script so it has ONLY the current run's env
                    # block (prevents stale env from previous runs overriding CODE_DIR/RG_LOG_DIR).
                    try:
                        script_path = Path(runtime_plan["shell"][2])
                        if script_path.exists():
                            lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                            # Find the env insertion region: between `setlocal` and the first step marker
                            setlocal_idx = next((i for i, l in enumerate(lines) if l.strip().lower() == "setlocal"), None)
                            step_idx = None
                            for i, l in enumerate(lines):
                                if i <= (setlocal_idx or -1):
                                    continue
                                if l.startswith("echo [RG step "):
                                    step_idx = i
                                    break
                            # Build a fresh env block for this run
                            env_lines = [f"set {k}={v}" for k, v in (inner_env or {}).items()]
                            # Helpful diagnostics without leaking secrets
                            env_lines += [
                                "if defined OPENAI_API_KEY echo [RG env] OPENAI_API_KEY set",
                                "if defined AZURE_OPENAI_API_KEY echo [RG env] AZURE_OPENAI_API_KEY set",
                                "if defined AZUREAI_OPENAI_API_KEY echo [RG env] AZUREAI_OPENAI_API_KEY set",
                                "if defined AZURE_OPENAI_ENDPOINT echo [RG env] AZURE_OPENAI_ENDPOINT set",
                                "if defined AZUREAI_OPENAI_BASE_URL echo [RG env] AZUREAI_OPENAI_BASE_URL set",
                            ]
                            # Ensure UTF-8 I/O for Python
                            env_lines += [
                                "set PYTHONIOENCODING=utf-8",
                                "set PYTHONUTF8=1",
                            ]
                            if setlocal_idx is not None and step_idx is not None and step_idx > setlocal_idx:
                                # Replace the entire region between setlocal and the first step with our env
                                new_lines = []
                                new_lines += lines[: setlocal_idx + 1]
                                new_lines += env_lines
                                new_lines += lines[step_idx:]
                                lines = new_lines
                            else:
                                # Fallback: prepend after setlocal if present, else prepend at top
                                insert_at = (setlocal_idx + 1) if setlocal_idx is not None else 0
                                lines = lines[:insert_at] + env_lines + lines[insert_at:]

                            # Optionally ensure apply_patch wrapper exists before agent exec
                                try:
                                    ap_py = json.dumps(str(agent_root / "apply_patch.py"))
                                    wrapper = [
                                        # Windows CMD wrappers for apply_patch/applypatch
                                        "echo @echo off > \"%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd\"",
                                        f"echo \"%VIRTUAL_ENV%\\Scripts\\python.exe\" {ap_py} %* >> \"%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd\"",
                                        "copy /Y \"%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd\" \"%VIRTUAL_ENV%\\Scripts\\applypatch.cmd\" >nul",
                                        # Git Bash friendly wrappers (no extension) so `bash` can find them
                                        "echo #!/usr/bin/env bash > \"%VIRTUAL_ENV%\\Scripts\\apply_patch\"",
                                        f"echo \"%VIRTUAL_ENV%\\Scripts\\python.exe\" {ap_py} \"$@\" >> \"%VIRTUAL_ENV%\\Scripts\\apply_patch\"",
                                        "copy /Y \"%VIRTUAL_ENV%\\Scripts\\apply_patch\" \"%VIRTUAL_ENV%\\Scripts\\applypatch\" >nul",
                                    ]
                                    # Insert wrapper right before the agent start line if we can find it
                                    exec_line = next((i for i, l in enumerate(lines) if l.strip().endswith("BasicAgent\\start.py")), None)
                                    if exec_line is not None:
                                        lines = lines[:exec_line] + wrapper + lines[exec_line:]
                                except Exception:
                                    pass

                            script_path.write_text("\n".join(lines), encoding="utf-8")
                    except Exception:
                        pass
            else:
                # Append after PATH export so $VIRTUAL_ENV is set (Unix)
                wrapper = "echo '#!/bin/bash' > \"$VIRTUAL_ENV\"/bin/apply_patch && echo 'python " + json.dumps(str(agent_root / "apply_patch.py")) + " \"$@\"' >> \"$VIRTUAL_ENV\"/bin/apply_patch && chmod +x \"$VIRTUAL_ENV\"/bin/apply_patch && (ln -sf \"$VIRTUAL_ENV\"/bin/apply_patch \"$VIRTUAL_ENV\"/bin/applypatch || cp \"$VIRTUAL_ENV\"/bin/apply_patch \"$VIRTUAL_ENV\"/bin/applypatch)"
                runtime_plan["shell"][2] = runtime_plan["shell"][2].replace('export PATH="$VIRTUAL_ENV/bin:$PATH"', 'export PATH="$VIRTUAL_ENV/bin:$PATH" && ' + wrapper)
            # Always disable interactive browser tools for BasicAgent in UV runs
            try:
                inner_env["DISABLE_BROWSER"] = "true"
            except Exception:
                pass
            if inner_env:
                shell_cmd = runtime_plan.get("shell")
                if shell_cmd and isinstance(shell_cmd, list) and len(shell_cmd) == 3:
                    # If the Windows runtime uses a .cmd script path, environment has been injected directly
                    # into the script above. For non-Windows (or if using inline command strings), prepend exports.
                    if platform.system() != "Windows":
                        exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in inner_env.items()])
                        runtime_plan["shell"][2] = runtime_plan["shell"][2].replace(" && uv venv", f" && {exports} && uv venv")

        combined = {"agent": agent_plan.info, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
        (env.run_dir / "status.json").write_text(json.dumps({"status": "planned"}, indent=2))
        print(f"Plan written: {plan_path}")

        if not args.dry_run:
            (env.run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2))
            stdout_log = env.logs_dir / "exec.stdout.log"
            stderr_log = env.logs_dir / "exec.stderr.log"
            if args.runtime == "docker":
                docker_cli = runtime_plan.get("docker_cli")
                if not docker_cli:
                    print("Docker plan missing 'docker_cli'")
                    sys.exit(2)
                # Append when resuming to preserve prior logs
                with open(stdout_log, "a" if is_resuming else "w", encoding='utf-8') as out, open(stderr_log, "a" if is_resuming else "w", encoding='utf-8') as err:
                    proc = subprocess.Popen(docker_cli, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            else:
                shell_cmd = runtime_plan.get("shell")
                if not shell_cmd:
                    print("UV plan missing 'shell'")
                    sys.exit(2)
                # Append when resuming to preserve prior logs
                with open(stdout_log, "a" if is_resuming else "w", encoding='utf-8') as out, open(stderr_log, "a" if is_resuming else "w", encoding='utf-8') as err:
                    proc = subprocess.Popen(shell_cmd, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            (env.run_dir / "status.json").write_text(json.dumps({"status": "completed", "returncode": rc}, indent=2))
            _write_usage_summary(env.run_dir)

    elif args.agent == "basic-agent-evolution":
        adapter = BasicAgentEvolutionAdapter(env=env, run_group=run_group, run_id=run_id)
        agent_root = args.basic_agent_evolution_root
        if agent_root is None:
            candidates = [
                PKG_ROOT / "agents" / "BasicAgentEvolution",
            ]
            agent_root = next((c for c in candidates if (c / "start.py").exists()), None)
            if agent_root is None:
                print("Could not auto-detect BasicAgentEvolution. Set --basic_agent_evolution_root to the directory.")
                for c in candidates:
                    print(f" - {c}")
                sys.exit(2)
        print(f"Using BasicAgentEvolution root: {agent_root}")
        adapter.prepare_workspace(task_dir=args.task_dir)
        _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name

        bae_cfg = BasicAgentEvolutionConfig(
            task_id=task_id,
            model=args.model or default_code_model,
            time_hours=args.basic_hours,
            iterative=args.basic_iterative,
            disallow_submit=args.basic_disallow_submit,
            code_only=args.basic_code_only,
            time_limit_secs=int(args.basic_hours * 60 * 60) + 60,
            budget_limit=args.budget_limit,
            idea_hint=bool(idea_hint_text),
        )

        agent_plan = adapter.run(cfg=bae_cfg, agent_root=agent_root, dry_run=True)
        inner_cmd = agent_plan.info.get("command", [])
        inner_env = agent_plan.info.get("env", {})

        if is_resuming and args.resume_with_instruction:
            input_dir = env.workspace_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            default_instr = input_dir / "instructions.txt"
            fallback_md = input_dir / "task_description.md"
            base_text = ""
            if default_instr.exists():
                try:
                    base_text = default_instr.read_text()
                except Exception:
                    base_text = ""
            if not base_text and fallback_md.exists():
                try:
                    base_text = fallback_md.read_text()
                except Exception:
                    base_text = ""
            resume_instr = input_dir / "instructions_resume.txt"
            header = "You are resuming in the same workspace. Inspect files under CODE_DIR and continue from your last step.\n\n"
            try:
                resume_instr.write_text(header + base_text)
                inner_env["RG_INSTRUCTIONS_FILE"] = str(resume_instr)
            except Exception:
                pass

        if is_resuming and not args.resume_with_instruction:
            transcript_path = env.run_dir / "transcript.json"
            inner_env["RG_RESUME_CONTEXT_FILE"] = str(transcript_path)

        task_mount = env.workspace_dir / "input"
        # Detect overlays from the source task dir so installs still run even though
        # requirements/install are not copied into the agent-visible workspace.
        overlay = detect_task_overlay(args.task_dir)
        if args.runtime == "docker":
            docker_install_steps = []
            docker_install_steps.append("pip install -r /agent/requirements.txt")
            if overlay.get("install_sh"):
                docker_install_steps.append("chmod +x /task/install.sh || true && bash /task/install.sh")
            elif overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts
            docker_env_exports = None
            if inner_env:
                inner_env["DISABLE_BROWSER"] = "true"

                def _remap_path(val: str) -> str:
                    try:
                        if isinstance(val, str):
                            mapped = val
                            mapped = mapped.replace(str(env.run_dir), "/run")
                            mapped = mapped.replace(str(agent_root), "/agent")
                            return mapped
                    except Exception:
                        pass
                    return val

                remapped = {k: _remap_path(v) for k, v in inner_env.items()}
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in remapped.items()])
                exports = exports + " && export GOOGLE_API_KEY=${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
                docker_env_exports = exports
            docker_prelude = " && ".join(
                [p for p in [docker_env_exports, " && ".join(prelude_parts) if prelude_parts else None] if p]
            ) if (docker_env_exports or prelude_parts) else None
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && {docker_prelude}"
                if docker_prelude
                else "export PIP_ROOT_USER_ACTION=ignore"
            )

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=agent_root,
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY","AZURE_OPENAI_API_KEY","AZURE_OPENAI_ENDPOINT",
                    "AZUREAI_OPENAI_API_KEY","AZUREAI_OPENAI_BASE_URL","AZUREAI_OPENAI_API_VERSION",
                    "ANTHROPIC_API_KEY","GOOGLE_API_KEY","GEMINI_API_KEY","OPENROUTER_API_KEY","EXA_API_KEY"
                ],
                prelude=docker_prelude,
                task_dir=task_mount,
            )
        else:
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-basic-agent-evolution",
                task_overlay=overlay,
                project_root=agent_root,
                command=inner_cmd,
            )
            shell_cmd = runtime_plan.get("shell")
            if shell_cmd and isinstance(shell_cmd, list) and len(shell_cmd) == 3:
                if platform.system() == "Windows":
                    try:
                        script_path = Path(runtime_plan["shell"][2])
                        if script_path.exists():
                            lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                            setlocal_idx = next((i for i, l in enumerate(lines) if l.strip().lower() == "setlocal"), None)
                            step_idx = None
                            for i, l in enumerate(lines):
                                if i <= (setlocal_idx or -1):
                                    continue
                                if l.startswith("echo [RG step "):
                                    step_idx = i
                                    break
                            env_lines = [f"set {k}={v}" for k, v in (inner_env or {}).items()]
                            env_lines += [
                                "if defined OPENAI_API_KEY echo [RG env] OPENAI_API_KEY set",
                                "if defined AZURE_OPENAI_API_KEY echo [RG env] AZURE_OPENAI_API_KEY set",
                                "if defined AZUREAI_OPENAI_API_KEY echo [RG env] AZUREAI_OPENAI_API_KEY set",
                                "if defined AZURE_OPENAI_ENDPOINT echo [RG env] AZURE_OPENAI_ENDPOINT set",
                                "if defined AZUREAI_OPENAI_BASE_URL echo [RG env] AZUREAI_OPENAI_BASE_URL set",
                            ]
                            env_lines += [
                                "set PYTHONIOENCODING=utf-8",
                                "set PYTHONUTF8=1",
                            ]
                            if setlocal_idx is not None and step_idx is not None and step_idx > setlocal_idx:
                                new_lines = []
                                new_lines += lines[: setlocal_idx + 1]
                                new_lines += env_lines
                                new_lines += lines[step_idx:]
                                lines = new_lines
                            else:
                                insert_at = (setlocal_idx + 1) if setlocal_idx is not None else 0
                                lines = lines[:insert_at] + env_lines + lines[insert_at:]

                            try:
                                ap_py = json.dumps(str(agent_root / "apply_patch.py"))
                                wrapper = [
                                    "echo @echo off > \"%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd\"",
                                    f"echo \"%VIRTUAL_ENV%\\Scripts\\python.exe\" {ap_py} %* >> \"%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd\"",
                                    "copy /Y \"%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd\" \"%VIRTUAL_ENV%\\Scripts\\applypatch.cmd\" >nul",
                                    "echo #!/usr/bin/env bash > \"%VIRTUAL_ENV%\\Scripts\\apply_patch\"",
                                    f"echo \"%VIRTUAL_ENV%\\Scripts\\python.exe\" {ap_py} \"$@\" >> \"%VIRTUAL_ENV%\\Scripts\\apply_patch\"",
                                    "copy /Y \"%VIRTUAL_ENV%\\Scripts\\apply_patch\" \"%VIRTUAL_ENV%\\Scripts\\applypatch\" >nul",
                                ]
                                exec_line = next(
                                    (i for i, l in enumerate(lines) if l.strip().endswith("BasicAgentEvolution\\start.py")), None
                                )
                                if exec_line is not None:
                                    lines = lines[:exec_line] + wrapper + lines[exec_line:]
                            except Exception:
                                pass

                            script_path.write_text("\n".join(lines), encoding="utf-8")
                    except Exception:
                        pass
            else:
                wrapper = (
                    "echo '#!/bin/bash' > \"$VIRTUAL_ENV\"/bin/apply_patch && "
                    "echo 'python " + json.dumps(str(agent_root / "apply_patch.py")) + " \"$@\"' >> \"$VIRTUAL_ENV\"/bin/apply_patch && "
                    "chmod +x \"$VIRTUAL_ENV\"/bin/apply_patch && "
                    "(ln -sf \"$VIRTUAL_ENV\"/bin/apply_patch \"$VIRTUAL_ENV\"/bin/applypatch || cp \"$VIRTUAL_ENV\"/bin/apply_patch \"$VIRTUAL_ENV\"/bin/applypatch)"
                )
                runtime_plan["shell"][2] = runtime_plan["shell"][2].replace(
                    'export PATH="$VIRTUAL_ENV/bin:$PATH"',
                    'export PATH="$VIRTUAL_ENV/bin:$PATH" && ' + wrapper,
                )
            try:
                inner_env["DISABLE_BROWSER"] = "true"
            except Exception:
                pass
            if inner_env:
                shell_cmd = runtime_plan.get("shell")
                if shell_cmd and isinstance(shell_cmd, list) and len(shell_cmd) == 3:
                    if platform.system() != "Windows":
                        exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in inner_env.items()])
                        runtime_plan["shell"][2] = runtime_plan["shell"][2].replace(" && uv venv", f" && {exports} && uv venv")

        combined = {"agent": agent_plan.info, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
        (env.run_dir / "status.json").write_text(json.dumps({"status": "planned"}, indent=2))
        print(f"Plan written: {plan_path}")

        if not args.dry_run:
            (env.run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2))
            stdout_log = env.logs_dir / "exec.stdout.log"
            stderr_log = env.logs_dir / "exec.stderr.log"
            if args.runtime == "docker":
                docker_cli = runtime_plan.get("docker_cli")
                if not docker_cli:
                    print("Docker plan missing 'docker_cli'")
                    sys.exit(2)
                with open(stdout_log, "a" if is_resuming else "w", encoding='utf-8') as out, open(
                    stderr_log, "a" if is_resuming else "w", encoding='utf-8'
                ) as err:
                    proc = subprocess.Popen(docker_cli, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            else:
                shell_cmd = runtime_plan.get("shell")
                if not shell_cmd:
                    print("UV plan missing 'shell'")
                    sys.exit(2)
                with open(stdout_log, "a" if is_resuming else "w", encoding='utf-8') as out, open(
                    stderr_log, "a" if is_resuming else "w", encoding='utf-8'
                ) as err:
                    proc = subprocess.Popen(shell_cmd, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            (env.run_dir / "status.json").write_text(json.dumps({"status": "completed", "returncode": rc}, indent=2))

    elif args.agent == "openevolve":
        from ResearchGym.agents.openevolve_adapter import OpenEvolveAdapter, OpenEvolveConfig

        adapter = OpenEvolveAdapter(env=env, run_group=run_group, run_id=run_id)
        openevolve_root = args.openevolve_root
        if openevolve_root is None:
            candidates = [
                PKG_ROOT / "agents" / "openevolve",
                PKG_ROOT.parent / "openevolve",
            ]
            openevolve_root = next((c for c in candidates if (c / "openevolve-run.py").exists()), None)
            if openevolve_root is None:
                print("Could not auto-detect OpenEvolve. Set --openevolve_root to the project directory.")
                for c in candidates:
                    print(f" - {c}")
                sys.exit(2)
        print(f"Using OpenEvolve root: {openevolve_root}")

        adapter.prepare_workspace(task_dir=args.task_dir)
        _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name

        op_cfg = OpenEvolveConfig(
            task_id=task_id,
            model=args.model or default_code_model,
            api_base=args.code_base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            time_limit_hours=args.basic_hours,
            budget_limit=args.budget_limit,
            max_iterations=args.openevolve_iterations,
        )

        agent_plan = adapter.run(cfg=op_cfg, openevolve_root=openevolve_root, task_dir=args.task_dir, dry_run=True)
        inner_cmd = agent_plan.info.get("command", [])
        inner_env = agent_plan.info.get("env", {})

        # Propagate API credentials from ML config to OpenEvolve environment if missing
        for key in [
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_CSE_API_KEY",
            "GOOGLE_CSE_ID",
        ]:
            if key not in inner_env and os.environ.get(key):
                inner_env[key] = os.environ[key]

        overlay = detect_task_overlay(args.task_dir)
        if args.runtime == "docker":
            docker_install_steps = []
            if overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts
            docker_env_exports = None
            if inner_env:
                def _remap_path(val: str) -> str:
                    try:
                        if isinstance(val, str):
                            mapped = val.replace(str(env.run_dir), "/run").replace(str(openevolve_root), "/agent")
                            return mapped
                    except Exception:
                        pass
                    return val

                remapped = {k: _remap_path(v) for k, v in inner_env.items()}
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in remapped.items()])
                docker_env_exports = exports
            prelude_parts = [p for p in [docker_env_exports, " && ".join(prelude_parts) if prelude_parts else None] if p]
            docker_prelude = " && ".join(prelude_parts) if prelude_parts else None
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && {docker_prelude}"
                if docker_prelude
                else "export PIP_ROOT_USER_ACTION=ignore"
            )

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=openevolve_root,
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY",
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_API_VERSION",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "GEMINI_API_KEY",
                    "OPENROUTER_API_KEY",
                ],
                prelude=docker_prelude,
                task_dir=args.task_dir,
            )
        else:
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-openevolve",
                task_overlay=overlay,
                project_root=openevolve_root,
                command=inner_cmd,
            )
            # Inject apply_patch equivalent and set env exports for the UV run
            shell_cmd = runtime_plan.get("shell")
            if shell_cmd and isinstance(shell_cmd, list) and len(shell_cmd) == 3:
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in inner_env.items()]) if inner_env else ""
                if exports:
                    runtime_plan["shell"][2] = runtime_plan["shell"][2].replace(" && uv venv", f" && {exports} && uv venv")
            elif inner_env:
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in inner_env.items()])
                runtime_plan["shell"][2] = runtime_plan["shell"][2].replace(" && uv venv", f" && {exports} && uv venv")

        combined = {"agent": agent_plan.info, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
        (env.run_dir / "status.json").write_text(json.dumps({"status": "planned"}, indent=2))
        print(f"Plan written: {plan_path}")

        final_status = "planned"
        final_payload: Dict[str, str] = {"status": final_status}
        if not args.dry_run:
            (env.run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2))
            stdout_log = env.logs_dir / "exec.stdout.log"
            stderr_log = env.logs_dir / "exec.stderr.log"
            stdout_log.parent.mkdir(parents=True, exist_ok=True)
            stderr_log.parent.mkdir(parents=True, exist_ok=True)
            if args.runtime == "docker":
                docker_cli = runtime_plan.get("docker_cli")
                if not docker_cli:
                    print("Docker plan missing 'docker_cli'")
                    sys.exit(2)
                with open(stdout_log, "w", encoding="utf-8") as out, open(stderr_log, "w", encoding="utf-8") as err:
                    proc = subprocess.Popen(docker_cli, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            else:
                shell_cmd = runtime_plan.get("shell")
                if not shell_cmd:
                    print("UV plan missing 'shell'")
                    sys.exit(2)
                with open(stdout_log, "w", encoding="utf-8") as out, open(stderr_log, "w", encoding="utf-8") as err:
                    proc = subprocess.Popen(shell_cmd, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            final_status = "completed" if rc == 0 else "failed"
            final_payload = {"status": final_status, "returncode": rc}
            (env.run_dir / "status.json").write_text(json.dumps({"status": final_status, "returncode": rc}, indent=2))

        fin = env.step({"type": "finish", "payload": final_payload})
        logger.info(f"Finished: {fin.info}")
        (env.run_dir / "status.json").write_text(json.dumps({"status": fin.info.get('status', final_status)}, indent=2))
        return

    elif args.agent == "claude-code":
        adapter = ClaudeCodeAdapter(env=env, run_group=run_group, run_id=run_id)
        claude_code_root = PKG_ROOT / "agents" / "ClaudeCode"
        # When resuming, do not copy/overlay the task into the existing workspace
        if not is_resuming:
            adapter.prepare_workspace(task_dir=args.task_dir, include_idea_hint=idea_hint_enabled)
        _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name

        # Build config
        cc_cfg = ClaudeCodeConfig(
            model=args.model or "claude-opus-4-5-20251101",
            time_hours=args.claude_hours,
            budget_limit=args.budget_limit,
            use_api=True,
        )

        print(f"ClaudeCode agent configured:")
        print(f"  Model: {cc_cfg.model}")
        print(f"  Time: {cc_cfg.time_hours}h")
        print(f"  Budget: ${cc_cfg.budget_limit}")
        print(f"  Workspace: {env.workspace_dir}")

        # Get command and env from adapter (dry_run mode)
        agent_plan = adapter.run(cfg=cc_cfg, task_dir=args.task_dir, dry_run=True)
        inner_cmd = agent_plan.info.get("command", [])
        inner_env = agent_plan.info.get("env", {})

        # Add resume session ID if resuming
        resume_session_id = getattr(args, "_claude_resume_session_id", None)
        if resume_session_id:
            is_real_session = resume_session_id != "transcript-seeding"

            # Always pass --resume-session so runner knows to use transcript seeding
            inner_cmd = list(inner_cmd)  # Make a copy
            inner_cmd.extend(["--resume-session", resume_session_id])

            if is_real_session:
                print(f"  Resume session: {resume_session_id}")
                # Copy Claude session file from old workspace to new workspace
                # Claude stores sessions in ~/.claude/projects/<encoded-cwd>/<session_id>.jsonl
                old_workspace_input = resume_dir / "workspace" / "input"
                new_workspace_input = env.workspace_dir / "input"
                _copy_claude_session_file(old_workspace_input, new_workspace_input, resume_session_id)
            else:
                print(f"  Resume mode: transcript seeding (no session ID)")

            # Copy transcript for transcript seeding (used instead of SDK resume)
            old_transcript = resume_dir / "logs" / "transcript.json"
            new_transcript = env.logs_dir / "previous_transcript.json"
            if old_transcript.exists():
                shutil.copy2(old_transcript, new_transcript)
                print(f"  Copied transcript for seeding: {new_transcript}")
            else:
                print(f"  Warning: Previous transcript not found at {old_transcript}")

        # Detect task overlay for requirements installation
        overlay = detect_task_overlay(args.task_dir)

        # For Docker, mount original task dir (not filtered workspace) so install.sh/requirements.txt are accessible
        task_mount = args.task_dir.resolve() if args.runtime == "docker" else env.workspace_dir / "input"

        if args.runtime == "docker":
            # Build Docker prelude with install steps and env exports
            # For ClaudeCode, we mount the entire PKG_ROOT (ResearchGym package) at /agent
            # so that module imports like `python -m ResearchGym.agents.ClaudeCode.run_cli` work
            docker_install_steps = []
            # Install ClaudeCode agent dependencies inside the container
            # Since we mount PKG_ROOT at /agent, requirements are at /agent/agents/ClaudeCode/requirements.txt
            docker_install_steps.append("pip install -r /agent/agents/ClaudeCode/requirements.txt")
            if overlay.get("install_sh"):
                docker_install_steps.append("chmod +x /task/install.sh || true && bash /task/install.sh")
            elif overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts

            # Inject ClaudeCode env exports so the runner knows workspace locations inside container
            docker_env_exports = None
            if inner_env:
                inner_env["DISABLE_BROWSER"] = "true"
                # Remap host paths to container mounts
                # PKG_ROOT is mounted at /agent, so paths need to be remapped accordingly
                def _remap_path_claude(val: str) -> str:
                    try:
                        if isinstance(val, str):
                            mapped = val
                            mapped = mapped.replace(str(env.run_dir), "/run")
                            mapped = mapped.replace(str(PKG_ROOT), "/agent")
                            return mapped
                    except Exception:
                        pass
                    return val

                remapped = {k: _remap_path_claude(v) for k, v in inner_env.items()}
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in remapped.items()])
                # Ensure GOOGLE_API_KEY is set when only GEMINI_API_KEY is provided
                exports = exports + " && export GOOGLE_API_KEY=${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
                docker_env_exports = exports

            docker_prelude = " && ".join([p for p in [docker_env_exports, " && ".join(prelude_parts) if prelude_parts else None] if p]) if (docker_env_exports or prelude_parts) else None
            # Ensure pip root-user warning is silenced inside container
            # Also set PYTHONPATH so module imports work: python -m ResearchGym.agents.ClaudeCode.run_cli
            # The PKG_ROOT is mounted at /agent, but imports expect /ResearchGym, so create a symlink
            symlink_cmd = "ln -sfn /agent /ResearchGym"
            # Claude Code CLI refuses --dangerously-skip-permissions when running as root
            # Create a non-root user and run the agent as that user
            # We do setup as root (pip install, symlink), then switch to non-root for the agent
            create_user_cmd = (
                "id -u agentuser &>/dev/null || useradd -m -s /bin/bash agentuser && "
                "chown -R agentuser:agentuser /run/workspace /run/logs 2>/dev/null || true"
            )
            # Pre-setup MuJoCo for RL tasks (mujoco-py expects binaries in ~/.mujoco/mujoco210)
            # The RL image may already have /root/.mujoco/mujoco210 - make it readable by agentuser
            # If not present, download to /opt/mujoco and symlink
            # Also fix numpy version (mujoco_py requires numpy<2.0) and make generated dir writable
            mujoco_setup_cmd = (
                "if [ -d /root/.mujoco/mujoco210 ]; then "
                "chmod 755 /root /root/.mujoco 2>/dev/null || true; "
                "else "
                "mkdir -p /opt/mujoco && "
                "curl -sL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C /opt/mujoco && "
                "mkdir -p /root/.mujoco && ln -sfn /opt/mujoco/mujoco210 /root/.mujoco/mujoco210; "
                "fi && "
                "mkdir -p /home/agentuser/.mujoco && "
                "ln -sfn /root/.mujoco/mujoco210 /home/agentuser/.mujoco/mujoco210 && "
                "chown -h agentuser:agentuser /home/agentuser/.mujoco /home/agentuser/.mujoco/mujoco210 && "
                "chmod -R 777 /opt/py310/lib/python3.10/site-packages/mujoco_py 2>/dev/null || true && "
                "pip install 'numpy<2.0' -q 2>/dev/null || true"
            )
            # Set LD_LIBRARY_PATH for mujoco binaries
            mujoco_env = "export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && export PYTHONPATH=/ && {mujoco_env} && {symlink_cmd} && {create_user_cmd} && {mujoco_setup_cmd} && {docker_prelude}"
                if docker_prelude
                else f"export PIP_ROOT_USER_ACTION=ignore && export PYTHONPATH=/ && {mujoco_env} && {symlink_cmd} && {create_user_cmd} && {mujoco_setup_cmd}"
            )
            # The actual command will be run as agentuser via runuser/su
            # We'll wrap the command execution in the prelude

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=PKG_ROOT,  # Mount full package for module imports to work
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                    "AZUREAI_OPENAI_API_KEY", "AZUREAI_OPENAI_BASE_URL", "SEMANTIC_SCHOLAR_API_KEY",
                    "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "EXA_API_KEY"
                ],
                prelude=docker_prelude,
                task_dir=task_mount,
                run_command_as_user="agentuser",  # Claude CLI refuses --dangerously-skip-permissions as root
            )
        else:
            # Plan UV commands - creates venv and installs task dependencies
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-claude-code",
                task_overlay=overlay,
                project_root=claude_code_root,
                command=inner_cmd,
            )

        # On Windows, inject environment variables into the generated .cmd script (UV runtime only)
        shell_cmd = runtime_plan.get("shell")
        if args.runtime != "docker" and shell_cmd and isinstance(shell_cmd, list) and len(shell_cmd) == 3 and platform.system() == "Windows":
            try:
                script_path = Path(shell_cmd[2])
                if script_path.exists():
                    lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    # Find insertion point after setlocal
                    setlocal_idx = next((i for i, l in enumerate(lines) if l.strip().lower() == "setlocal"), None)
                    step_idx = None
                    for i, l in enumerate(lines):
                        if i <= (setlocal_idx or -1):
                            continue
                        if l.startswith("echo [RG step "):
                            step_idx = i
                            break

                    # Build env block
                    env_lines = [f"set {k}={v}" for k, v in (inner_env or {}).items()]
                    # Add PYTHONPATH for ResearchGym imports
                    repo_root = PKG_ROOT.parent
                    env_lines.append(f"set PYTHONPATH={repo_root}")
                    # Ensure UTF-8 for Python
                    env_lines.extend([
                        "set PYTHONIOENCODING=utf-8",
                        "set PYTHONUTF8=1",
                    ])
                    # Propagate API keys
                    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                        if os.environ.get(key):
                            env_lines.append(f"set {key}={os.environ[key]}")

                    if setlocal_idx is not None and step_idx is not None and step_idx > setlocal_idx:
                        new_lines = lines[: setlocal_idx + 1] + env_lines + lines[step_idx:]
                        lines = new_lines
                    else:
                        insert_at = (setlocal_idx + 1) if setlocal_idx is not None else 0
                        lines = lines[:insert_at] + env_lines + lines[insert_at:]

                    script_path.write_text("\n".join(lines), encoding="utf-8")
            except Exception as e:
                print(f"Warning: could not inject env vars into script: {e}")

        # Save plan
        combined = {"agent": agent_plan.info, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
        (env.run_dir / "status.json").write_text(json.dumps({"status": "planned"}, indent=2))
        print(f"Plan written: {plan_path}")

        if not args.dry_run:
            (env.run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2))
            stdout_log = env.logs_dir / "exec.stdout.log"
            stderr_log = env.logs_dir / "exec.stderr.log"

            # Append logs when resuming to preserve previous run context
            log_mode = "a" if is_resuming else "w"

            if args.runtime == "docker":
                docker_cli = runtime_plan.get("docker_cli")
                if not docker_cli:
                    print("Docker plan missing 'docker_cli'")
                    sys.exit(2)
                with open(stdout_log, log_mode, encoding="utf-8") as out, open(stderr_log, log_mode, encoding="utf-8") as err:
                    if is_resuming:
                        out.write(f"\n{'='*60}\n")
                        out.write(f"=== RESUME RUN: {run_id} ===\n")
                        out.write(f"{'='*60}\n\n")
                        err.write(f"\n{'='*60}\n")
                        err.write(f"=== RESUME RUN: {run_id} ===\n")
                        err.write(f"{'='*60}\n\n")
                    proc = subprocess.Popen(docker_cli, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            else:
                shell_cmd = runtime_plan.get("shell")
                if not shell_cmd:
                    print("UV plan missing 'shell'")
                    sys.exit(2)
                with open(stdout_log, log_mode, encoding="utf-8") as out, open(stderr_log, log_mode, encoding="utf-8") as err:
                    if is_resuming:
                        out.write(f"\n{'='*60}\n")
                        out.write(f"=== RESUME RUN: {run_id} ===\n")
                        out.write(f"{'='*60}\n\n")
                        err.write(f"\n{'='*60}\n")
                        err.write(f"=== RESUME RUN: {run_id} ===\n")
                        err.write(f"{'='*60}\n\n")
                    proc = subprocess.Popen(shell_cmd, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()

            final_status = "completed" if rc == 0 else "failed"
            (env.run_dir / "status.json").write_text(json.dumps({"status": final_status, "returncode": rc}, indent=2))
            print(f"ClaudeCode finished: {final_status} (rc={rc})")
            _write_usage_summary(env.run_dir, agent_type="claude-code")

        fin = env.step({"type": "finish", "payload": {"status": "planned" if args.dry_run else final_status}})
        logger.info(f"Finished: {fin.info}")
        return

    elif args.agent == "codex":
        adapter = CodexAdapter(env=env, run_group=run_group, run_id=run_id)
        codex_root = PKG_ROOT / "agents" / "Codex"
        # When resuming, do not copy/overlay the task into the existing workspace
        if not is_resuming:
            adapter.prepare_workspace(task_dir=args.task_dir, include_idea_hint=idea_hint_enabled)
        _apply_idea_hint_or_exit(env.workspace_dir / "input", idea_hint_text)
        task_id = args.task_dir.resolve().name

        resume_session_id = getattr(args, "_codex_resume_session_id", None)
        inherited_cost_path = getattr(args, "_codex_inherited_cost_path", None)

        # Build config kwargs - only include provider if explicitly set
        config_kwargs = {
            "model": args.codex_model or CodexConfig().model,
            "time_hours": args.codex_hours,
            "budget_limit": args.budget_limit,
            "reasoning_effort": args.codex_reasoning_effort,
            "resume_session_id": resume_session_id,
            "inherited_cost_path": inherited_cost_path,
        }
        if args.codex_subscription:
            config_kwargs["provider"] = PROVIDER_SUBSCRIPTION

        codex_cfg = CodexConfig(**config_kwargs)

        print("Codex agent configured:")
        print(f"  Model: {codex_cfg.model}")
        print(f"  Provider: {codex_cfg.provider}" + (" (ChatGPT subscription)" if args.codex_subscription else ""))
        print(f"  Time: {codex_cfg.time_hours}h")
        print(f"  Budget: ${codex_cfg.budget_limit}")
        print(f"  Reasoning effort: {codex_cfg.reasoning_effort}")
        print(f"  Workspace: {env.workspace_dir}")
        if resume_session_id:
            print(f"  Resume session: {resume_session_id}")

        # Get command and env from adapter (dry_run mode)
        agent_plan = adapter.run(cfg=codex_cfg, task_dir=args.task_dir, dry_run=True)
        inner_cmd = agent_plan.info.get("command", [])
        inner_env = agent_plan.info.get("env", {})

        # Detect task overlay for requirements installation
        overlay = detect_task_overlay(args.task_dir)

        # For Docker, mount original task dir (not filtered workspace) so install.sh/requirements.txt are accessible
        task_mount = args.task_dir.resolve() if args.runtime == "docker" else env.workspace_dir / "input"

        if args.runtime == "docker":
            # Build Docker prelude with install steps and env exports
            # For Codex, we mount the entire PKG_ROOT (ResearchGym package) at /agent
            # so that module imports like `python -m ResearchGym.agents.Codex.run_cli` work
            docker_install_steps = []
            # Install Node.js 18+ and Codex CLI (required for Codex agent)
            # Ubuntu's default nodejs is too old (12.x), need NodeSource for 18+
            docker_install_steps.append("apt-get update -qq && apt-get install -qq -y curl >/dev/null 2>&1")
            docker_install_steps.append("curl -fsSL https://deb.nodesource.com/setup_18.x | bash - >/dev/null 2>&1")
            docker_install_steps.append("apt-get install -qq -y nodejs >/dev/null 2>&1")
            docker_install_steps.append("npm install -g @openai/codex >/dev/null 2>&1")
            # Install Codex agent dependencies inside the container
            # Since we mount PKG_ROOT at /agent, requirements are at /agent/agents/Codex/requirements.txt
            docker_install_steps.append("pip install -r /agent/agents/Codex/requirements.txt")
            if overlay.get("install_sh"):
                docker_install_steps.append("chmod +x /task/install.sh || true && bash /task/install.sh")
            elif overlay.get("requirements"):
                docker_install_steps.append("pip install -r /task/requirements.txt")
            if overlay.get("setup_py"):
                docker_install_steps.append("pip install -e /task")

            prelude_parts = []
            if args.litellm_prelude:
                prelude_parts.append(args.litellm_prelude)
            if docker_install_steps:
                prelude_parts = docker_install_steps + prelude_parts

            # Inject Codex env exports so the runner knows workspace locations inside container
            docker_env_exports = None
            if inner_env:
                inner_env["DISABLE_BROWSER"] = "true"
                # Remap host paths to container mounts
                # PKG_ROOT is mounted at /agent, so paths need to be remapped accordingly
                def _remap_path_codex(val: str) -> str:
                    try:
                        if isinstance(val, str):
                            mapped = val
                            mapped = mapped.replace(str(env.run_dir), "/run")
                            mapped = mapped.replace(str(PKG_ROOT), "/agent")
                            return mapped
                    except Exception:
                        pass
                    return val

                remapped = {k: _remap_path_codex(v) for k, v in inner_env.items()}
                exports = " && ".join([f"export {k}={json.dumps(v)}" for k, v in remapped.items()])
                # Ensure GOOGLE_API_KEY is set when only GEMINI_API_KEY is provided
                exports = exports + " && export GOOGLE_API_KEY=${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
                docker_env_exports = exports

            docker_prelude = " && ".join([p for p in [docker_env_exports, " && ".join(prelude_parts) if prelude_parts else None] if p]) if (docker_env_exports or prelude_parts) else None
            # Ensure pip root-user warning is silenced inside container
            # Also set PYTHONPATH so module imports work: python -m ResearchGym.agents.Codex.run_cli
            # The PKG_ROOT is mounted at /agent, but imports expect /ResearchGym, so create a symlink
            symlink_cmd = "ln -sfn /agent /ResearchGym"
            # Pre-setup MuJoCo for RL tasks (mujoco-py expects binaries in ~/.mujoco/mujoco210)
            # The RL image may already have /root/.mujoco/mujoco210 - use it if present
            mujoco_setup_cmd = (
                "if [ ! -d /root/.mujoco/mujoco210 ]; then "
                "mkdir -p /opt/mujoco && "
                "curl -sL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C /opt/mujoco && "
                "mkdir -p /root/.mujoco && ln -sfn /opt/mujoco/mujoco210 /root/.mujoco/mujoco210; "
                "fi"
            )
            mujoco_env = "export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
            docker_prelude = (
                f"export PIP_ROOT_USER_ACTION=ignore && export PYTHONPATH=/ && {mujoco_env} && {symlink_cmd} && {mujoco_setup_cmd} && {docker_prelude}"
                if docker_prelude
                else f"export PIP_ROOT_USER_ACTION=ignore && export PYTHONPATH=/ && {mujoco_env} && {symlink_cmd} && {mujoco_setup_cmd}"
            )

            runtime_plan = plan_docker_command(
                image=args.image,
                run_dir=env.run_dir,
                ml_master_root=PKG_ROOT,  # Mount full package for module imports to work
                command=inner_cmd,
                gpus=args.gpus,
                env_keys=[
                    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                    "AZUREAI_OPENAI_API_KEY", "AZUREAI_OPENAI_BASE_URL", "SEMANTIC_SCHOLAR_API_KEY",
                    "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "EXA_API_KEY"
                ],
                prelude=docker_prelude,
                task_dir=task_mount,
            )
        else:
            # Plan UV commands - creates venv and installs task dependencies
            runtime_plan = plan_uv_commands(
                cache_dir=env.run_dir / ".uv_cache",
                venv_name=f"{task_id}-{run_id}-codex",
                task_overlay=overlay,
                project_root=codex_root,
                command=inner_cmd,
            )

        # Save plan
        combined = {"agent": agent_plan.info, "runtime": runtime_plan, "overlay": overlay}
        plan_path = env.run_dir / "plan.json"
        plan_path.write_text(json.dumps(combined, indent=2))
        (env.run_dir / "status.json").write_text(json.dumps({"status": "planned"}, indent=2))
        print(f"Plan written: {plan_path}")

        final_status = "planned"
        result_payload: Dict[str, str] = {"status": final_status}
        if not args.dry_run:
            (env.run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2))
            stdout_log = env.logs_dir / "exec.stdout.log"
            stderr_log = env.logs_dir / "exec.stderr.log"

            # Append logs when resuming to preserve previous run context
            log_mode = "a" if is_resuming else "w"

            if args.runtime == "docker":
                docker_cli = runtime_plan.get("docker_cli")
                if not docker_cli:
                    print("Docker plan missing 'docker_cli'")
                    sys.exit(2)
                with open(stdout_log, log_mode, encoding="utf-8") as out, open(stderr_log, log_mode, encoding="utf-8") as err:
                    if is_resuming:
                        out.write(f"\n{'='*60}\n")
                        out.write(f"=== RESUME RUN: {run_id} ===\n")
                        out.write(f"{'='*60}\n\n")
                        err.write(f"\n{'='*60}\n")
                        err.write(f"=== RESUME RUN: {run_id} ===\n")
                        err.write(f"{'='*60}\n\n")
                    proc = subprocess.Popen(docker_cli, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()
            else:
                shell_cmd = runtime_plan.get("shell")
                if not shell_cmd:
                    print("UV plan missing 'shell'")
                    sys.exit(2)
                with open(stdout_log, log_mode, encoding="utf-8") as out, open(stderr_log, log_mode, encoding="utf-8") as err:
                    if is_resuming:
                        out.write(f"\n{'='*60}\n")
                        out.write(f"=== RESUME RUN: {run_id} ===\n")
                        out.write(f"{'='*60}\n\n")
                        err.write(f"\n{'='*60}\n")
                        err.write(f"=== RESUME RUN: {run_id} ===\n")
                        err.write(f"{'='*60}\n\n")
                    proc = subprocess.Popen(shell_cmd, stdout=out, stderr=err, shell=False)
                    rc = proc.wait()

            final_status = "completed" if rc == 0 else "failed"
            result_payload = {"status": final_status, "returncode": str(rc)}
            (env.run_dir / "status.json").write_text(json.dumps(result_payload, indent=2))
            print(f"Codex finished: {final_status} (rc={rc})")
            _write_usage_summary(env.run_dir, agent_type="codex")

        fin = env.step({"type": "finish", "payload": result_payload})
        logger.info(f"Finished: {fin.info}")
        return

    # Always finish for now
    fin = env.step({"type": "finish", "payload": {"status": "planned" if args.dry_run else "submitted"}})
    logger.info(f"Finished: {fin.info}")
    (env.run_dir / "status.json").write_text(json.dumps({"status": fin.info.get('status', 'finished')}, indent=2))


if __name__ == "__main__":
    main()

