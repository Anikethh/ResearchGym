"""Runner for Codex CLI agent.

Wraps the Codex CLI (codex-cli) for autonomous operation within ResearchGym.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CodexConfig, PROVIDER_AZURE, PROVIDER_SUBSCRIPTION, SANDBOX_DANGER_FULL_ACCESS, SANDBOX_WORKSPACE_WRITE, SANDBOX_BYPASS, generate_codex_config_toml
from .cost_tracker import CodexCostTracker, BudgetExceeded
from .messages import (
    AUTONOMOUS_SYSTEM_PROMPT,
    DEFAULT_CONTINUE_MESSAGE,
    get_continue_message,
    get_periodic_status_message,
)
from .post_processor import post_process_output, write_violations


logger = logging.getLogger(__name__)


@dataclass
class CodexResult:
    """Result from a Codex CLI run."""
    status: str  # completed, timeout, error, budget_exceeded
    return_code: int
    elapsed_seconds: float
    output_file: Path | None = None
    session_id: str | None = None
    error: str | None = None
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# =============================================================================
# Workspace Isolation via Subst Drive (Windows)
# =============================================================================

# Preferred drive letters for subst (avoid common ones like C, D, E)
SUBST_DRIVE_CANDIDATES = "RQZYXWVUTSPONMLKJIHGF"


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def get_used_drive_letters() -> set[str]:
    """Get set of drive letters currently in use."""
    if not is_windows():
        return set()

    used = set()
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if os.path.exists(f"{letter}:"):
            used.add(letter)
    return used


def find_available_drive_letter() -> str | None:
    """Find an available drive letter for subst."""
    if not is_windows():
        return None

    used = get_used_drive_letters()
    for letter in SUBST_DRIVE_CANDIDATES:
        if letter not in used:
            return letter
    return None


def create_subst_drive(target_path: Path) -> str | None:
    """Create a subst drive mapping to target path."""
    if not is_windows():
        logger.debug("Subst not available on non-Windows platforms")
        return None

    letter = find_available_drive_letter()
    if not letter:
        logger.warning("No available drive letters for subst")
        return None

    try:
        result = subprocess.run(
            ["subst", f"{letter}:", str(target_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(f"subst failed: {result.stderr}")
            return None

        logger.info(f"Created subst drive {letter}: -> {target_path}")
        return letter
    except Exception as e:
        logger.warning(f"Failed to create subst drive: {e}")
        return None


def remove_subst_drive(letter: str) -> bool:
    """Remove a subst drive mapping."""
    if not is_windows() or not letter:
        return False

    try:
        result = subprocess.run(
            ["subst", f"{letter}:", "/d"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info(f"Removed subst drive {letter}:")
            return True
        logger.warning(f"Failed to remove subst {letter}: {result.stderr}")
        return False
    except Exception as e:
        logger.warning(f"Error removing subst drive: {e}")
        return False


# Transient errors that should trigger retry
TRANSIENT_ERRORS = (
    "connection",
    "timeout",
    "overloaded",
    "rate limit",
    "503",
    "502",
    "500",
    "429",
    "network",
    "reset by peer",
    "broken pipe",
    "eof",
    # Stream/connection interruption errors
    "disconnected",
    "response.failed",
    "stream disconnected",
    "turn.failed",
    "reconnecting",
    # Additional API errors
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "capacity",
)

# Retry configuration - aggressive settings for rate-limited APIs
# Mirrors BasicAgent's tenacity retry settings for long-running research tasks
MAX_RETRIES = 30  # Up from 5 - allows for extended rate limit periods
INITIAL_RETRY_DELAY = 30  # seconds
MAX_RETRY_DELAY = 3600  # 1 hour max backoff (up from 5 min) - matches BasicAgent


def is_transient_error(error: str) -> bool:
    """Check if an error is transient and should be retried."""
    error_str = (error or "").lower()
    for pattern in TRANSIENT_ERRORS:
        if pattern in error_str:
            return True
    return False


def build_codex_command(
    config: CodexConfig,
    workspace_dir: Path,
    prompt: str,
    output_format: str = "json",
    codex_bin: str = "codex",
) -> tuple[list[str], str]:
    """Build the Codex CLI command.

    Args:
        config: Codex configuration
        workspace_dir: Working directory for the agent
        prompt: The task prompt/instructions
        output_format: Output format (json, text, stream-json)
        codex_bin: Path to codex binary

    Returns:
        Tuple of (command list, prompt string) - prompt should be passed via stdin
    """
    # Check if we're resuming a session
    if config.resume_session_id:
        # Use codex exec resume <session_id> "continue prompt"
        cmd = [codex_bin, "exec", "resume", config.resume_session_id]

        # Set approval mode (skip if using bypass - it already implies no approval)
        if config.sandbox_mode != SANDBOX_BYPASS:
            if config.approval_mode == "full-auto":
                cmd.append("--full-auto")
            elif config.approval_mode == "auto-edit":
                cmd.append("--auto-edit")

        # Set model (can override on resume)
        cmd.extend(["--model", config.model])

        # Reasoning effort (for o-series and reasoning models)
        # Use -c config override for broader compatibility
        if config.reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{config.reasoning_effort}"'])

        # Sandbox mode
        if config.sandbox_mode:
            if config.sandbox_mode == SANDBOX_BYPASS:
                cmd.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                cmd.extend(["--sandbox", config.sandbox_mode])
                # Enable network access for workspace-write mode (needed for huggingface, pip, etc.)
                if config.sandbox_mode == SANDBOX_WORKSPACE_WRITE:
                    cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])

        # Output format
        if output_format and output_format != "text":
            cmd.append("--json")

        # For resume, prompt is short enough to pass as argument
        cmd.append(prompt)
        return cmd, ""  # Empty string means no stdin needed

    # Normal (non-resume) command
    cmd = [codex_bin, "exec"]

    # Set approval mode (skip if using bypass - it already implies no approval)
    if config.sandbox_mode != SANDBOX_BYPASS:
        if config.approval_mode == "full-auto":
            cmd.append("--full-auto")
        elif config.approval_mode == "auto-edit":
            cmd.append("--auto-edit")

    # Set model
    cmd.extend(["--model", config.model])

    # Reasoning effort (for o-series and reasoning models)
    # Use -c config override for broader compatibility
    if config.reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{config.reasoning_effort}"'])

    # Sandbox mode
    if config.sandbox_mode:
        if config.sandbox_mode == SANDBOX_BYPASS:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", config.sandbox_mode])
            # Enable network access for workspace-write mode (needed for huggingface, pip, etc.)
            if config.sandbox_mode == SANDBOX_WORKSPACE_WRITE:
                cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])

    # Web search tool (enabled via --enable feature flag)
    if config.web_search:
        cmd.extend(["--enable", "web_search"])

    # Output format
    if output_format and output_format != "text":
        cmd.append("--json")

    # Prompt passed via stdin to avoid shell/argument length limits
    return cmd, prompt


def build_resume_command(
    config: CodexConfig,
    session_id: str,
    continue_message: str,
    codex_bin: str = "codex",
) -> list[str]:
    """Build a Codex CLI resume command for continuing a session.

    Args:
        config: Codex configuration
        session_id: Session ID to resume
        continue_message: Message to prompt continuation
        codex_bin: Path to codex binary

    Returns:
        Command as list of strings
    """
    # NOTE: Flags must be between "exec" and "resume" per Codex CLI docs
    cmd = [codex_bin, "exec"]

    # Set approval mode (skip if using bypass - it already implies no approval)
    if config.sandbox_mode != SANDBOX_BYPASS:
        if config.approval_mode == "full-auto":
            cmd.append("--full-auto")
        elif config.approval_mode == "auto-edit":
            cmd.append("--auto-edit")

    # Set model
    cmd.extend(["--model", config.model])

    # Reasoning effort (for o-series and reasoning models)
    # Use -c config override for broader compatibility
    if config.reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{config.reasoning_effort}"'])

    # Sandbox mode
    if config.sandbox_mode:
        if config.sandbox_mode == SANDBOX_BYPASS:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", config.sandbox_mode])
            # Enable network access for workspace-write mode (needed for huggingface, pip, etc.)
            if config.sandbox_mode == SANDBOX_WORKSPACE_WRITE:
                cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])

    # Web search tool (enabled via --enable feature flag)
    if config.web_search:
        cmd.extend(["--enable", "web_search"])

    # JSON output
    cmd.append("--json")

    # Resume with session ID (flags must come BEFORE "resume")
    cmd.extend(["resume", session_id])

    # Add continue prompt
    cmd.append(continue_message)

    return cmd


def _extract_event_text(event: dict[str, Any]) -> list[str]:
    """Extract text content from a Codex CLI event.

    Codex CLI outputs events with structure like:
    - item.completed: {"item": {"type": "reasoning", "text": "..."}}
    - item.started/updated: {"item": {"type": "command_execution", "command": "...", "aggregated_output": "..."}}
    - item.updated: {"item": {"type": "todo_list", "items": [{"text": "..."}]}}

    Skips error/status events that shouldn't be included in transcript context.
    """
    texts: list[str] = []

    # Skip event types that shouldn't be in transcript context
    event_type = event.get("type", "")
    skip_types = {
        "error",           # "Reconnecting... 1/5" messages
        "turn.failed",     # Stream disconnects
        "turn.started",    # Turn lifecycle
        "thread.started",  # Thread lifecycle
        "response.failed", # API failures
    }
    if event_type in skip_types:
        return texts

    # Check top-level text fields (legacy format)
    for key in ("content", "message", "text"):
        val = event.get(key)
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    texts.append(item)

    # Check payload (legacy format)
    payload = event.get("payload")
    if isinstance(payload, dict):
        for key in ("content", "message", "text"):
            val = payload.get(key)
            if isinstance(val, str):
                texts.append(val)

    # Check item structure (Codex CLI format)
    item = event.get("item")
    if isinstance(item, dict):
        item_type = item.get("type", "")

        # Reasoning items have direct text
        if item_type == "reasoning":
            text_val = item.get("text")
            if isinstance(text_val, str) and text_val.strip():
                texts.append(text_val)

        # Command executions have command and aggregated_output
        elif item_type == "command_execution":
            cmd = item.get("command")
            if isinstance(cmd, str) and cmd.strip():
                texts.append(f"[Command] {cmd}")
            output = item.get("aggregated_output")
            if isinstance(output, str) and output.strip():
                # Truncate very long outputs
                if len(output) > 2000:
                    output = output[:2000] + "... [truncated]"
                texts.append(f"[Output] {output}")

        # Todo list items
        elif item_type == "todo_list":
            items_list = item.get("items", [])
            if items_list:
                todo_texts = []
                for todo_item in items_list:
                    if isinstance(todo_item, dict):
                        todo_text = todo_item.get("text", "")
                        completed = todo_item.get("completed", False)
                        if todo_text:
                            status = "[x]" if completed else "[ ]"
                            todo_texts.append(f"  {status} {todo_text}")
                if todo_texts:
                    texts.append("[Todo List]\n" + "\n".join(todo_texts))

        # File edit items
        elif item_type == "file_edit":
            filepath = item.get("filepath")
            if isinstance(filepath, str):
                texts.append(f"[File Edit] {filepath}")

        # Generic item text fallback
        else:
            text_val = item.get("text")
            if isinstance(text_val, str) and text_val.strip():
                texts.append(text_val)

    return texts


def extract_transcript_context(transcript_path: Path, max_chars: int = 140000) -> str:
    """Extract context from a Codex JSONL transcript.

    Codex CLI has ~272K token limit. 140K chars (~35-45K tokens) leaves room for
    task description, system prompts, and the session's own growing context.
    250K caused context_length_exceeded when sessions ran long (compaction failed).
    If still over limit, keeps beginning AND end to preserve both
    initial understanding and recent progress.
    """
    if not transcript_path.exists():
        return ""

    parts: list[str] = []
    try:
        with transcript_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                for text in _extract_event_text(event):
                    if text:
                        parts.append(text)
    except Exception:
        return ""

    context = "\n\n".join(parts).strip()
    if len(context) > max_chars:
        # Keep both beginning (initial understanding) and end (recent progress)
        # Split 30% beginning, 70% end since recent work is more important
        begin_chars = int(max_chars * 0.3)
        end_chars = max_chars - begin_chars - 100  # 100 chars for separator
        context = (
            context[:begin_chars] +
            "\n\n[... earlier session context omitted for brevity ...]\n\n" +
            context[-end_chars:]
        )
    return context


def build_resume_prompt(previous_transcript: Path, continue_message: str) -> str:
    """Build prompt with previous transcript context and continue message."""
    context = extract_transcript_context(previous_transcript)
    if not context:
        return ""

    return f"""# Previous Session Context

You are resuming a previous research session. Below is the context from your prior work:

{context}

---

# Continue Instructions

{continue_message}

Please continue where you left off. Review your previous progress and proceed with the next steps.
"""


def build_task_prompt(workspace_dir: Path, log_dir: Path | None = None) -> str:
    """Build the task prompt from workspace files.

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Combined prompt string
    """
    input_dir = workspace_dir / "input"

    # Look for task description
    task_desc = ""
    for desc_file in ["task_description.md", "instructions.txt", "README.md"]:
        desc_path = input_dir / desc_file
        if desc_path.exists():
            task_desc = desc_path.read_text(encoding="utf-8")
            break

    resume_block = ""
    if log_dir:
        prev_transcript = log_dir / "previous_transcript.jsonl"
        if prev_transcript.exists():
            resume_block = build_resume_prompt(prev_transcript, DEFAULT_CONTINUE_MESSAGE).strip()

    # Build autonomous operation prompt
    resume_section = f"\n\n{resume_block}\n" if resume_block else "\n"
    prompt = f"""{AUTONOMOUS_SYSTEM_PROMPT.strip()}{resume_section}

## Working Directory
Your working directory is: {input_dir}

## Task Description
{task_desc}

## Instructions
1. Work autonomously to complete the research task
2. Read and understand the existing code and data
3. Implement improvements and run experiments
4. Commit your changes regularly using git
5. Document your methodology and findings

Start by exploring the codebase and understanding the task requirements.
"""
    return prompt


def run_codex_cli(
    config: CodexConfig,
    workspace_dir: Path,
    log_dir: Path,
    timeout_seconds: float | None = None,
) -> CodexResult:
    """Run Codex CLI with the given configuration.

    Implements real-time budget enforcement by monitoring turn.completed usage
    in the streaming JSON output.
    """
    subst_letter = create_subst_drive(workspace_dir)
    run_workspace_dir = Path(f"{subst_letter}:") if subst_letter else workspace_dir

    # When subst drive is active, it provides path isolation, so we can relax
    # Codex's sandbox to avoid conflicts with subst drive recognition
    if subst_letter:
        msg = f"Subst drive {subst_letter}: active, bypassing sandbox (subst provides isolation)"
        print(msg, file=sys.stderr, flush=True)
        logger.info(msg)
        config.sandbox_mode = SANDBOX_BYPASS

    try:
        # Setup environment
        env = os.environ.copy()
        env.update(config.env)

        if config.provider == PROVIDER_SUBSCRIPTION:
            # Clear API keys so Codex uses ChatGPT subscription login
            env.pop("OPENAI_API_KEY", None)
            env.pop("AZURE_OPENAI_API_KEY", None)
            logger.info("Using ChatGPT subscription mode (API keys cleared)")
        elif config.provider == PROVIDER_AZURE:
            config_path = config.config_path or (log_dir / ".codex" / "config.toml")
            generate_codex_config_toml(config, config_path)
            env["CODEX_HOME"] = str(config_path.parent)

        # Use Git Bash instead of PowerShell on Windows (more familiar to the model)
        if sys.platform == "win32":
            git_bash_paths = [
                Path("C:/Program Files/Git/bin/bash.exe"),
                Path("C:/Program Files (x86)/Git/bin/bash.exe"),
            ]
            for bash_path in git_bash_paths:
                if bash_path.exists():
                    env["SHELL"] = str(bash_path)
                    break

        # Ensure workspace is the cwd
        input_dir = run_workspace_dir / "input"

        # Log files
        stdout_log = log_dir / "codex.stdout.log"
        stderr_log = log_dir / "codex.stderr.log"
        output_log = log_dir / "codex_output.jsonl"
        cost_log = log_dir / "cost_summary.json"

        # Token/cost tracking for real-time budget enforcement
        session_id = None
        already_elapsed = 0.0
        cost_tracker = CodexCostTracker(budget_limit=config.budget_limit, model=config.model)

        # Load prior costs if resuming (native session resume OR transcript seeding)
        # For transcript seeding, look for previous_cost_summary.json alongside the transcript
        prev_cost_file = log_dir / "previous_cost_summary.json"
        prior_cost_path = config.inherited_cost_path or (
            cost_log if config.resume_session_id else (
                prev_cost_file if prev_cost_file.exists() else None
            )
        )
        if prior_cost_path and prior_cost_path.exists():
            try:
                with open(prior_cost_path) as f:
                    prior_data = json.load(f)
                cost_tracker = CodexCostTracker.load(
                    prior_cost_path,
                    budget_limit=config.budget_limit,
                    model=config.model,
                )
                already_elapsed = prior_data.get("time", {}).get("active_seconds", 0.0)
                if timeout_seconds:
                    timeout_seconds = max(0, timeout_seconds - already_elapsed)
                logger.info(
                    f"Loaded prior costs from {prior_cost_path}: "
                    f"${cost_tracker.inherited_cost_usd:.4f} inherited, "
                    f"{already_elapsed:.1f}s elapsed"
                )
            except Exception as e:
                logger.warning(f"Could not load prior costs from {prior_cost_path}: {e}")
        else:
            logger.info(f"No prior costs to load (path={prior_cost_path})")
        cost_tracker.time_limit_seconds = timeout_seconds

        def _run_attempt(cmd: list[str], append_logs: bool, stdin_input: str = "") -> CodexResult:
            nonlocal session_id
            log_mode = "a" if append_logs else "w"
            turn_failed_error: str | None = None  # Track turn.failed events for retry logic
            original_session_id: str | None = None  # Track first session for context loss detection
            saw_turn_failed: bool = False  # Track if we saw turn.failed (to detect context loss)

            try:
                with open(stdout_log, log_mode, encoding="utf-8", errors="replace") as stdout_f, \
                     open(stderr_log, log_mode, encoding="utf-8", errors="replace") as stderr_f, \
                     open(output_log, log_mode, encoding="utf-8", errors="replace") as output_f:

                    # Use stdin for prompt if provided (avoids command-line length limits)
                    stdin_pipe = subprocess.PIPE if stdin_input else None

                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(input_dir),
                        env=env,
                        stdin=stdin_pipe,
                        stdout=subprocess.PIPE,
                        stderr=stderr_f,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )

                    # Send prompt via stdin if provided
                    if stdin_input:
                        proc.stdin.write(stdin_input)
                        proc.stdin.close()

                    # Process streaming output
                    while True:
                        # Check timeout
                        if cost_tracker.is_over_time_limit():
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()

                            elapsed = cost_tracker.get_wall_clock_time()
                            return CodexResult(
                                status="timeout",
                                return_code=-9,
                                elapsed_seconds=elapsed,
                                output_file=output_log,
                                session_id=session_id,
                                total_cost_usd=cost_tracker.total_cost_usd,
                                total_input_tokens=cost_tracker.total_input_tokens,
                                total_output_tokens=cost_tracker.total_output_tokens,
                            )

                        # Read output line
                        line = proc.stdout.readline()
                        if not line:
                            if proc.poll() is not None:
                                break
                            continue

                        # Write to logs
                        stdout_f.write(line)
                        stdout_f.flush()

                        # Try to parse as JSON for structured output
                        try:
                            data = json.loads(line.strip())
                            output_f.write(json.dumps(data) + "\n")
                            output_f.flush()

                            # Extract session ID if present
                            if "session_id" in data:
                                session_id = data["session_id"]
                                if not original_session_id:
                                    original_session_id = session_id
                            # Also check thread_id as session identifier
                            if "thread_id" in data:
                                session_id = data["thread_id"]
                                if not original_session_id:
                                    original_session_id = session_id

                            # =========================================================
                            # COST ESTIMATION: Track content for real-time estimates
                            # =========================================================
                            event_type = data.get("type", "")

                            # Extract content for estimation
                            item = data.get("item", {})
                            content_chars = 0

                            # Count chars from various content fields
                            if "aggregated_output" in item:
                                content_chars += len(item["aggregated_output"] or "")
                            if "content" in data:
                                content_chars += len(str(data["content"]))
                            if "message" in data:
                                content_chars += len(str(data["message"]))

                            # Track reasoning text (major part of output tokens)
                            item_type = item.get("type", "")
                            if item_type == "reasoning":
                                text = item.get("text", "")
                                if text:
                                    content_chars += len(text)

                            # Track file write commands - content is in the command itself
                            # PowerShell heredocs: @'...'@ or Set-Content/Out-File with content
                            if item_type == "command_execution":
                                cmd = item.get("command", "")
                                # Heredoc file writes have the file content in the command
                                if "@'" in cmd or "Set-Content" in cmd or "Out-File" in cmd:
                                    content_chars += len(cmd)

                            # Add to estimates (output chars from agent)
                            if content_chars > 0:
                                cost_tracker.add_estimated_chars(output_chars=content_chars)

                            # =========================================================
                            # TURN FAILED: Track for retry logic (Codex CLI may exit 0)
                            # =========================================================
                            if event_type == "turn.failed":
                                error_info = data.get("error", {})
                                turn_failed_error = error_info.get("message") if isinstance(error_info, dict) else str(error_info)
                                saw_turn_failed = True
                                logger.warning(f"turn.failed event received: {turn_failed_error}")

                            # =========================================================
                            # CONTEXT LOSS DETECTION: Kill if Codex starts fresh thread
                            # =========================================================
                            # When Codex's internal retries fail, it may start a NEW thread
                            # instead of exiting. This loses context. Detect and kill so our
                            # retry logic can do proper session resume.
                            if event_type == "thread.started":
                                new_thread_id = data.get("thread_id")
                                if saw_turn_failed and original_session_id and new_thread_id != original_session_id:
                                    logger.warning(f"Codex started fresh thread {new_thread_id} after turn.failed (was {original_session_id}), killing for proper resume")
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=10)
                                    except subprocess.TimeoutExpired:
                                        proc.kill()
                                    cost_tracker.save(cost_log)
                                    elapsed = cost_tracker.get_wall_clock_time()
                                    return CodexResult(
                                        status="error",
                                        return_code=-2,
                                        elapsed_seconds=elapsed,
                                        output_file=output_log,
                                        session_id=original_session_id,  # Use ORIGINAL for resume
                                        error=f"context_loss: Codex started fresh thread after failure",
                                        total_cost_usd=cost_tracker.total_cost_usd,
                                        total_input_tokens=cost_tracker.total_input_tokens,
                                        total_output_tokens=cost_tracker.total_output_tokens,
                                    )

                            # =========================================================
                            # ACTUAL USAGE: Replace estimates when turn.completed
                            # =========================================================
                            if event_type == "turn.completed":
                                usage = data.get("usage", {})
                                if usage:
                                    curr_input = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                                    curr_output = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                                    curr_cached = usage.get("cached_input_tokens", 0) or usage.get("cache_read_input_tokens", 0)

                                    try:
                                        cost_tracker.record_cumulative(
                                            input_tokens=curr_input,
                                            output_tokens=curr_output,
                                            cached_tokens=curr_cached,
                                        )
                                    except BudgetExceeded as exc:
                                        proc.terminate()
                                        try:
                                            proc.wait(timeout=10)
                                        except subprocess.TimeoutExpired:
                                            proc.kill()

                                        cost_tracker.save(cost_log)
                                        elapsed = cost_tracker.get_wall_clock_time()
                                        return CodexResult(
                                            status="budget_exceeded",
                                            return_code=-8,
                                            elapsed_seconds=elapsed,
                                            output_file=output_log,
                                            session_id=session_id,
                                            error=str(exc),
                                            total_cost_usd=cost_tracker.total_cost_usd,
                                            total_input_tokens=cost_tracker.total_input_tokens,
                                            total_output_tokens=cost_tracker.total_output_tokens,
                                        )

                            # =========================================================
                            # GRACEFUL STOP: Check if approaching limits
                            # =========================================================
                            should_stop, stop_reason = cost_tracker.should_graceful_stop()
                            if should_stop:
                                logger.warning(f"Graceful stop triggered: {stop_reason}")
                                # Terminate gracefully - Codex should clean up
                                proc.terminate()
                                try:
                                    proc.wait(timeout=30)  # Give more time for graceful cleanup
                                except subprocess.TimeoutExpired:
                                    proc.kill()

                                cost_tracker.save(cost_log)
                                elapsed = cost_tracker.get_wall_clock_time()
                                return CodexResult(
                                    status="graceful_stop",
                                    return_code=-7,
                                    elapsed_seconds=elapsed,
                                    output_file=output_log,
                                    session_id=session_id,
                                    error=f"Graceful stop: {stop_reason}",
                                    total_cost_usd=cost_tracker.get_effective_cost(),
                                    total_input_tokens=cost_tracker.total_input_tokens,
                                    total_output_tokens=cost_tracker.total_output_tokens,
                                )

                            # Save cost summary periodically
                            cost_tracker.save(cost_log)

                        except json.JSONDecodeError:
                            pass

                    rc = proc.wait()
                    elapsed = cost_tracker.get_wall_clock_time()

                    # Final cost summary save
                    summary = cost_tracker.get_summary()
                    summary.update({
                        "session_id": session_id,
                        "status": "completed" if rc == 0 else "error",
                        "elapsed_seconds": elapsed,
                    })
                    cost_log.parent.mkdir(parents=True, exist_ok=True)
                    cost_log.write_text(json.dumps(summary, indent=2))

                    error_text = None
                    if rc != 0 and stderr_log.exists():
                        try:
                            lines = stderr_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                            error_text = "\n".join(lines[-20:]) if lines else None
                        except Exception:
                            error_text = None

                    # Treat turn.failed as error (Codex CLI may exit 0 or non-0 after exhausting retries)
                    # This enables our retry logic (30 retries with exponential backoff) to kick in
                    # IMPORTANT: Use turn_failed_error for retry detection even when rc != 0
                    # because stderr may contain non-transient errors (like 401) that mask
                    # the actual transient disconnect error
                    if turn_failed_error:
                        if rc == 0:
                            logger.warning(f"Codex exited 0 but turn.failed detected, treating as error for retry")
                        else:
                            logger.warning(f"Codex exited {rc} with turn.failed: {turn_failed_error}")
                        return CodexResult(
                            status="error",
                            return_code=rc,
                            elapsed_seconds=elapsed,
                            output_file=output_log,
                            session_id=session_id,
                            error=turn_failed_error,  # Use turn.failed error for retry detection
                            total_cost_usd=cost_tracker.total_cost_usd,
                            total_input_tokens=cost_tracker.total_input_tokens,
                            total_output_tokens=cost_tracker.total_output_tokens,
                        )

                    return CodexResult(
                        status="completed" if rc == 0 else "error",
                        return_code=rc,
                        elapsed_seconds=elapsed,
                        output_file=output_log,
                        session_id=session_id,
                        error=error_text,
                        total_cost_usd=cost_tracker.total_cost_usd,
                        total_input_tokens=cost_tracker.total_input_tokens,
                        total_output_tokens=cost_tracker.total_output_tokens,
                    )

            except FileNotFoundError:
                elapsed = cost_tracker.get_wall_clock_time()
                return CodexResult(
                    status="error",
                    return_code=-1,
                    elapsed_seconds=elapsed,
                    error="codex CLI not found. Install with: npm install -g @openai/codex",
                )

            except Exception as e:
                elapsed = cost_tracker.get_wall_clock_time()
                return CodexResult(
                    status="error",
                    return_code=-1,
                    elapsed_seconds=elapsed,
                    error=str(e),
                )

        codex_path = shutil.which("codex")
        if not codex_path:
            raise FileNotFoundError(f"codex CLI not found in PATH: {env.get('PATH')}")

        def _run_with_retries(cmd: list[str], append_logs: bool, stdin_input: str = "") -> CodexResult:
            """Run command with retry logic for transient errors."""
            retry_count = 0
            retry_delay = INITIAL_RETRY_DELAY
            current_cmd = cmd
            current_stdin = stdin_input

            while True:
                use_append = append_logs or retry_count > 0
                result = _run_attempt(current_cmd, append_logs=use_append, stdin_input=current_stdin)

                if result.status != "error":
                    return result
                if retry_count >= MAX_RETRIES or not is_transient_error(result.error or ""):
                    if retry_count > 0 and result.error:
                        result.error = f"{result.error} (after {retry_count} retries)"
                    return result

                retry_count += 1

                # Try to resume: native session resume > transcript-seeded > fail
                if result.session_id:
                    logger.info(f"Transient error, resuming session {result.session_id}, retry {retry_count}/{MAX_RETRIES} in {retry_delay}s")
                    current_cmd = build_resume_command(config, result.session_id, "Continue.", codex_path)
                    current_stdin = ""
                elif output_log.exists() and output_log.stat().st_size > 0:
                    logger.info(f"Transient error, transcript-seeded resume, retry {retry_count}/{MAX_RETRIES} in {retry_delay}s")
                    resume_context = build_resume_prompt(output_log, "Continue from where you left off.")
                    current_cmd, current_stdin = build_codex_command(
                        config=config,
                        workspace_dir=run_workspace_dir,
                        prompt=resume_context,
                        output_format="stream-json",
                        codex_bin=codex_path,
                    )
                else:
                    logger.error("Transient error but no session or transcript to resume from, giving up")
                    return result

                cost_tracker.add_retry_time(retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                cost_tracker.prev_input_tokens = 0
                cost_tracker.prev_output_tokens = 0
                cost_tracker.prev_cached_tokens = 0

        def _should_continue() -> bool:
            """Check if we have time and budget to continue.

            Uses the same graceful stop thresholds as the inner loop to avoid
            triggering a continuation when we're about to stop anyway.
            """
            # Check graceful stop first (uses proportional thresholds)
            should_stop, reason = cost_tracker.should_graceful_stop()
            if should_stop:
                logger.debug(f"_should_continue: graceful stop triggered ({reason})")
                return False

            # Also check hard limits
            remaining_time = cost_tracker.get_remaining_time()
            if remaining_time is not None and remaining_time <= 0:
                return False
            if config.budget_limit > 0:
                remaining_budget = cost_tracker.get_remaining_budget()
                if remaining_budget <= 0:
                    return False
            return True

        # =====================================================================
        # INITIAL RUN: Build full task prompt and run codex exec
        # =====================================================================
        initial_prompt = build_task_prompt(run_workspace_dir, log_dir)
        initial_cmd, stdin_prompt = build_codex_command(
            config=config,
            workspace_dir=run_workspace_dir,
            prompt=initial_prompt,
            output_format="stream-json",
            codex_bin=codex_path,
        )

        # Add initial prompt to input estimates
        cost_tracker.add_estimated_chars(input_chars=len(stdin_prompt))

        logger.info("Starting initial Codex run")
        logger.info(f"Command: {' '.join(initial_cmd)}")
        logger.info(f"Prompt length: {len(stdin_prompt)} chars (via stdin)")
        # Write command to file for debugging (logs may be buffered)
        (log_dir / "codex_command.txt").write_text(" ".join(initial_cmd))
        result = _run_with_retries(initial_cmd, append_logs=False, stdin_input=stdin_prompt)

        # Post-process for blocked URLs
        if config.blocked_urls and output_log.exists():
            try:
                violations = post_process_output(output_log, config.blocked_urls)
                write_violations(log_dir, violations)
            except Exception:
                pass

        # If not completed successfully, return immediately
        if result.status != "completed":
            return result

        # =====================================================================
        # CONTINUATION LOOP: Use native resume to continue SAME session
        # =====================================================================
        continuation_count = 0
        while _should_continue() and result.session_id:
            continuation_count += 1

            # Build continue message with status update
            remaining_time = cost_tracker.get_remaining_time()
            total_time = cost_tracker.time_limit_seconds or 0
            elapsed_time = cost_tracker.get_active_time()

            # Use extended continue message for longer runs
            extended = total_time > 12 * 3600  # 12+ hours
            continue_msg = get_continue_message(extended=extended)

            # Add status info to the continue prompt
            status_info = get_periodic_status_message(
                elapsed_seconds=elapsed_time,
                total_seconds=total_time,
                cost_usd=cost_tracker.total_cost_usd,
                budget_limit_usd=config.budget_limit,
            )
            full_continue_prompt = f"{status_info}\n\n{continue_msg}"

            # Use NATIVE resume to continue the SAME session
            resume_cmd = build_resume_command(
                config=config,
                session_id=result.session_id,
                continue_message=full_continue_prompt,
                codex_bin=codex_path,
            )

            logger.info(f"Continuation {continuation_count}: resuming session {result.session_id}")
            result = _run_with_retries(resume_cmd, append_logs=True)

            # Post-process for blocked URLs
            if config.blocked_urls and output_log.exists():
                try:
                    violations = post_process_output(output_log, config.blocked_urls)
                    write_violations(log_dir, violations)
                except Exception:
                    pass

            # If not completed successfully, return
            if result.status != "completed":
                return result

        # Save final transcript for potential cross-process resume
        prev_transcript = log_dir / "previous_transcript.jsonl"
        try:
            if output_log.exists():
                prev_transcript.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_log, prev_transcript)
        except Exception:
            pass

        return result
    finally:
        if subst_letter:
            remove_subst_drive(subst_letter)


def parse_codex_costs(output_file: Path) -> dict[str, Any]:
    """Parse cost information from Codex output.

    Codex CLI outputs usage information in stream-json format.

    Args:
        output_file: Path to the JSONL output file

    Returns:
        Dictionary with cost summary
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    turns = 0

    if not output_file or not output_file.exists():
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "total_turns": 0,
        }

    try:
        with open(output_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Look for usage data in various formats
                    if "usage" in data:
                        usage = data["usage"]
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_input_tokens += usage.get("prompt_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                        total_output_tokens += usage.get("completion_tokens", 0)
                        turns += 1

                    # Look for cost data
                    if "cost" in data:
                        total_cost += data["cost"]

                except json.JSONDecodeError:
                    continue

    except Exception:
        pass

    # Estimate cost if not provided (rough o3 pricing)
    if total_cost == 0 and (total_input_tokens > 0 or total_output_tokens > 0):
        # o3 pricing estimates (may need adjustment)
        input_cost_per_1k = 0.01
        output_cost_per_1k = 0.03
        total_cost = (
            (total_input_tokens / 1000) * input_cost_per_1k +
            (total_output_tokens / 1000) * output_cost_per_1k
        )

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost,
        "total_turns": turns,
    }
