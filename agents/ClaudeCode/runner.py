"""Main runner for Claude Code agent.

Executes the Claude Code agent using the claude-agent-sdk,
with hooks for autonomous operation, cost tracking, and URL blocking.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Python 3.10 compatibility for asyncio.timeout (added in 3.11)
# =============================================================================
if sys.version_info >= (3, 11):
    asyncio_timeout = asyncio.timeout
else:
    @asynccontextmanager
    async def asyncio_timeout(delay: float | None):
        """Fallback timeout context manager for Python < 3.11.

        Note: This is a simplified implementation that doesn't support
        rescheduling. For our use case (long-running agent with backup timeout),
        this is sufficient.
        """
        if delay is None:
            yield
            return

        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        handle = loop.call_later(delay, task.cancel)
        try:
            yield
        except asyncio.CancelledError:
            raise asyncio.TimeoutError() from None
        finally:
            handle.cancel()

# Note: claude_agent_sdk import will fail if not installed
# This is handled gracefully with try/except
try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookMatcher,
        ResultMessage,
    )
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    ClaudeAgentOptions = None
    ClaudeSDKClient = None
    HookMatcher = None
    ResultMessage = None

from .config import ClaudeCodeConfig
from .cost_tracker import CostTracker, BudgetExceeded, TimeLimitExceeded
from .hooks import make_continue_hook, make_url_filter_hook, make_path_guard_hook, get_blocked_attempts
from .messages import AUTONOMOUS_SYSTEM_PROMPT, get_continue_message

logger = logging.getLogger(__name__)


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
    # Check which drives exist
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
    """Create a subst drive mapping to target path.

    Args:
        target_path: Path to map the drive to

    Returns:
        Drive letter (e.g., "R") if successful, None if failed
    """
    if not is_windows():
        logger.debug("Subst not available on non-Windows platforms")
        return None

    letter = find_available_drive_letter()
    if not letter:
        logger.warning("No available drive letters for subst")
        return None

    try:
        # Create the subst mapping
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
    """Remove a subst drive mapping.

    Args:
        letter: Drive letter to remove (e.g., "R")

    Returns:
        True if successful
    """
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
        else:
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
)

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 30  # seconds
MAX_RETRY_DELAY = 300  # 5 minutes max backoff


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    for pattern in TRANSIENT_ERRORS:
        if pattern in error_str or pattern in error_type:
            return True

    # Also check for common HTTP/connection error types
    if "http" in error_type or "connection" in error_type or "timeout" in error_type:
        return True

    return False


def check_sdk_available() -> bool:
    """Check if claude-agent-sdk is available."""
    if not CLAUDE_SDK_AVAILABLE:
        raise ImportError(
            "claude-agent-sdk is not installed. "
            "Install it with: pip install claude-agent-sdk"
        )
    return True


def build_task_prompt(workspace_dir: Path) -> str:
    """Build the task prompt from workspace files.

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Combined prompt string
    """
    prompt_parts = []

    # Add autonomous operation instructions
    prompt_parts.append(AUTONOMOUS_SYSTEM_PROMPT.strip())
    prompt_parts.append("\n---\n")

    # Load task description
    task_desc_path = workspace_dir / "input" / "task_description.md"
    if task_desc_path.exists():
        prompt_parts.append("# Task Description\n")
        prompt_parts.append(task_desc_path.read_text())
    else:
        # Try alternate location
        alt_path = workspace_dir / "task_description.md"
        if alt_path.exists():
            prompt_parts.append("# Task Description\n")
            prompt_parts.append(alt_path.read_text())

    # Load instructions if separate
    instructions_path = workspace_dir / "input" / "instructions.txt"
    if instructions_path.exists():
        prompt_parts.append("\n---\n")
        prompt_parts.append("# Additional Instructions\n")
        prompt_parts.append(instructions_path.read_text())

    return "\n".join(prompt_parts)


def check_transcript_end_status(transcript: list[dict]) -> tuple[bool, str | None]:
    """Check how a transcript ended.

    Args:
        transcript: List of message dicts from transcript.json

    Returns:
        Tuple of (ended_cleanly, subtype)
        - ended_cleanly: True if ended with ResultMessage subtype=success
        - subtype: The subtype from ResultMessage if found
    """
    if not transcript:
        return False, None

    # Find the last ResultMessage
    for msg in reversed(transcript):
        if msg.get("type") == "ResultMessage":
            subtype = msg.get("subtype")
            return subtype == "success", subtype

    return False, None


def build_resume_context(transcript_path: Path, max_chars: int = 50000) -> tuple[str, bool]:
    """Build resume context from previous transcript.

    Extracts key information from the transcript to seed a new session:
    - Assistant's text responses (reasoning, plans)
    - Tool calls and results (what was done)
    - Truncates if too long
    - If session ended cleanly (subtype=success), appends a continue message

    Args:
        transcript_path: Path to transcript.json from previous run
        max_chars: Maximum characters for context (default 50k)

    Returns:
        Tuple of (context_string, ended_cleanly)
        - context_string: Formatted context for resume prompt
        - ended_cleanly: True if previous session ended with subtype=success
    """
    if not transcript_path.exists():
        logger.warning(f"Transcript not found: {transcript_path}")
        return "", False

    try:
        transcript = json.loads(transcript_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to parse transcript: {e}")
        return "", False

    # Check how the previous session ended
    ended_cleanly, subtype = check_transcript_end_status(transcript)
    if ended_cleanly:
        logger.info("Previous session ended cleanly (subtype=success), will append continue message")
    else:
        logger.info(f"Previous session ended abruptly (subtype={subtype}), resuming from crash")

    context_parts = []
    context_parts.append("# Previous Session Context\n")
    context_parts.append("Below is the conversation history from your previous session. Continue from where you left off.\n")
    context_parts.append("---\n")

    for msg in transcript:
        msg_type = msg.get("type", "")
        content = msg.get("content", "")

        # Skip system messages and ResultMessages
        if msg_type in ("SystemMessage", "ResultMessage"):
            continue

        # Skip error-related entries (rate limits, API errors, etc.)
        # These shouldn't pollute the resume context
        content_str = str(content).lower() if content else ""
        if any(err in content_str for err in [
            "rate limit", "ratelimit", "<synthetic>", "apierror",
            "overloaded", "429", "500", "503", "timeout"
        ]):
            continue

        # Format based on message type
        if msg_type == "AssistantMessage":
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        # Extract text from TextBlock or ToolUseBlock string repr
                        if "TextBlock(text=" in item:
                            # Extract text content
                            text = item.split("TextBlock(text=", 1)[-1]
                            text = text.rsplit(")", 1)[0]
                            text = text.strip("'\"")
                            context_parts.append(f"Assistant: {text}\n")
                        elif "ToolUseBlock" in item:
                            # Extract tool name and input
                            if "name='" in item:
                                tool_name = item.split("name='")[1].split("'")[0]
                                context_parts.append(f"Assistant: [Called {tool_name}]\n")
            elif isinstance(content, str):
                context_parts.append(f"Assistant: {content}\n")

        elif msg_type == "UserMessage":
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and "ToolResultBlock" in item:
                        # Summarize tool result (truncate long outputs)
                        if "content='" in item:
                            result = item.split("content='")[1].split("', is_error")[0]
                            if len(result) > 500:
                                result = result[:500] + "..."
                            context_parts.append(f"Tool Result: {result}\n")

    # If session ended cleanly, append a continue message from the user
    # This signals to the agent that it should continue working
    if ended_cleanly:
        context_parts.append("\n---\n")
        context_parts.append("User: " + get_continue_message(extended=True) + "\n")

    context = "\n".join(context_parts)

    # Truncate if too long - keep beginning AND end (30/70 split)
    # Recent work is more important than early exploration
    if len(context) > max_chars:
        begin_chars = int(max_chars * 0.3)
        end_chars = max_chars - begin_chars - 100  # 100 chars for separator
        context = (
            context[:begin_chars] +
            "\n\n[... earlier session context omitted for brevity ...]\n\n" +
            context[-end_chars:]
        )
        logger.info(f"Resume context truncated to {max_chars} chars (30/70 split)")

    logger.info(f"Built resume context: {len(context)} chars from {len(transcript)} messages")
    return context, ended_cleanly


def serialize_message(message: Any) -> dict[str, Any]:
    """Serialize a message from the SDK to a dictionary.

    Args:
        message: Message object from claude-agent-sdk

    Returns:
        Dictionary representation
    """
    msg_type = getattr(message, "type", type(message).__name__)

    result = {
        "type": msg_type,
        "timestamp": datetime.now().isoformat(),
    }

    # Handle different message types
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            result["content"] = content
        elif isinstance(content, list):
            result["content"] = [
                c.model_dump() if hasattr(c, "model_dump") else str(c) for c in content
            ]
        else:
            result["content"] = str(content)

    if hasattr(message, "model"):
        result["model"] = message.model

    if hasattr(message, "usage"):
        result["usage"] = message.usage

    if hasattr(message, "total_cost_usd"):
        result["total_cost_usd"] = message.total_cost_usd

    if hasattr(message, "duration_ms"):
        result["duration_ms"] = message.duration_ms

    if hasattr(message, "subtype"):
        result["subtype"] = message.subtype

    if hasattr(message, "is_error"):
        result["is_error"] = message.is_error

    if hasattr(message, "result"):
        result["result"] = message.result

    if hasattr(message, "session_id"):
        result["session_id"] = message.session_id

    # Capture message ID for usage deduplication (same ID = same usage)
    if hasattr(message, "id"):
        result["id"] = message.id

    return result


def save_transcript(path: Path, messages: list[dict[str, Any]]) -> None:
    """Save transcript to JSON file.

    Args:
        path: Path to save to
        messages: List of serialized messages
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)


def discover_session_id_from_claude_projects(workspace_dir: Path) -> str | None:
    """Discover session ID from ~/.claude/projects/ folder.

    Claude Code stores sessions in ~/.claude/projects/<encoded-path>/<session-id>.jsonl
    The path encoding replaces path separators and colons with dashes.

    Args:
        workspace_dir: The workspace directory used as cwd for the agent

    Returns:
        Session ID string if found, None otherwise
    """
    claude_projects_dir = Path.home() / ".claude" / "projects"
    if not claude_projects_dir.exists():
        return None

    # Encode the workspace path the way Claude does:
    # Replace \ / : with - (Claude does NOT collapse multiple dashes)
    workspace_str = str(workspace_dir.resolve())
    encoded_path = workspace_str.replace('\\', '-').replace('/', '-').replace(':', '-')

    project_dir = claude_projects_dir / encoded_path
    if not project_dir.exists():
        # Try to find a matching folder (case insensitive on Windows)
        for folder in claude_projects_dir.iterdir():
            if folder.is_dir() and folder.name.lower() == encoded_path.lower():
                project_dir = folder
                break
        else:
            return None

    # Find the most recent .jsonl file (session file)
    jsonl_files = list(project_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Get the most recently modified session file
    latest_session = max(jsonl_files, key=lambda p: p.stat().st_mtime)

    # The filename (without extension) is the session ID
    session_id = latest_session.stem

    # Verify it looks like a UUID
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    if uuid_pattern.match(session_id):
        return session_id

    return None


async def run_agent(
    config: ClaudeCodeConfig,
    workspace_dir: Path,
    log_dir: Path,
    resume_session_id: str | None = None,
) -> dict[str, Any]:
    """Run the Claude Code agent.

    Args:
        config: Agent configuration
        workspace_dir: Path to workspace directory
        log_dir: Path to log directory
        resume_session_id: Session ID to resume (optional)

    Returns:
        Dictionary with run results
    """
    check_sdk_available()

    # Initialize tracking with time limit
    time_limit_secs = config.time_hours * 3600
    cost_summary_path = log_dir / "cost_summary.json"
    transcript: list[dict[str, Any]] = []

    # Load existing cost tracker if resuming
    if resume_session_id and cost_summary_path.exists():
        try:
            cost_tracker = CostTracker.load(cost_summary_path)

            # Get already-spent resources (for logging)
            inherited_cost = cost_tracker.inherited_cost
            already_elapsed = cost_tracker.get_summary()["time"]["active_seconds"]

            # Update budget limit from config (total budget, not remaining)
            # CostTracker.load() already inherits total_cost, so budget check works correctly
            cost_tracker.budget_limit = config.budget_limit

            # Update time limit: remaining = new config time - already elapsed
            # Reset start_time to now, set time_limit to remaining
            cost_tracker.start_time = time.time()
            cost_tracker.time_limit_seconds = max(0, time_limit_secs - already_elapsed)

            # Update model if config changed
            cost_tracker.model = config.model

            logger.info(
                f"Resumed cost tracker: inherited ${inherited_cost:.2f} from previous sessions, "
                f"{already_elapsed:.0f}s elapsed. "
                f"Budget limit: ${config.budget_limit:.2f}, "
                f"Remaining time: {cost_tracker.time_limit_seconds:.0f}s"
            )
        except Exception as e:
            logger.warning(f"Could not load cost tracker for resume: {e}, starting fresh")
            cost_tracker = CostTracker(
                budget_limit=config.budget_limit,
                model=config.model,
                time_limit_seconds=time_limit_secs,
            )
    else:
        cost_tracker = CostTracker(
            budget_limit=config.budget_limit,
            model=config.model,
            time_limit_seconds=time_limit_secs,
        )

    # Set up paths
    transcript_path = log_dir / "transcript.json"
    cost_summary_path = log_dir / "cost_summary.json"
    blocked_urls_log_path = log_dir / "blocked_urls.json"
    blocked_paths_log_path = log_dir / "blocked_paths.json"

    # Create workspace isolation via subst drive (Windows only)
    # This hides the real path structure from the agent
    workspace_input = workspace_dir / "input"
    subst_drive = create_subst_drive(workspace_input)
    if subst_drive:
        agent_cwd = f"{subst_drive}:\\"
        logger.info(f"Workspace isolated at {subst_drive}:")
    else:
        agent_cwd = str(workspace_input)
        logger.info(f"Workspace at {agent_cwd} (no isolation)")

    # Create hooks - pass cost_tracker for time-aware continue decisions
    continue_hook = make_continue_hook(
        cost_tracker=cost_tracker,
        start_time=cost_tracker.start_time,
        time_limit_hours=config.time_hours,
        extended=config.extended_continue,
        status_interval=config.status_interval,
    )

    url_filter_hook = make_url_filter_hook(
        blocked_urls=config.blocked_urls,
        log_blocked=True,
    )

    # Create path guard hook if we have workspace isolation
    path_guard_hook = None
    if subst_drive:
        path_guard_hook = make_path_guard_hook(
            allowed_drive=subst_drive,
            workspace_path=str(workspace_input),
        )

    # Build options
    options_dict = {
        "cwd": agent_cwd,
        "model": config.model,
        "allowed_tools": config.allowed_tools,
        "disallowed_tools": config.disallowed_tools,
        "permission_mode": "bypassPermissions",
        "hooks": {
            "Stop": [HookMatcher(hooks=[continue_hook])],
        },
    }

    # Build PreToolUse hooks list
    pre_tool_hooks = []
    if config.blocked_urls:
        pre_tool_hooks.append(
            HookMatcher(
                matcher="WebFetch|WebSearch",
                hooks=[url_filter_hook],
            )
        )
    if path_guard_hook:
        pre_tool_hooks.append(
            HookMatcher(
                matcher="Read|Write|Edit|Glob|Grep|Bash",
                hooks=[path_guard_hook],
            )
        )
    if pre_tool_hooks:
        options_dict["hooks"]["PreToolUse"] = pre_tool_hooks

    # NOTE: We don't use SDK resume (options_dict["resume"]) because it doesn't work
    # reliably (see GitHub issue #12730). Instead, we use transcript seeding.

    options = ClaudeAgentOptions(**options_dict)

    # Build prompt - use transcript seeding if resuming
    if resume_session_id:
        # Try to find transcript from previous run (should be copied to log_dir)
        prev_transcript_path = log_dir / "previous_transcript.json"

        # Build context from transcript (returns tuple: context, ended_cleanly)
        if prev_transcript_path.exists():
            resume_context, prev_ended_cleanly = build_resume_context(prev_transcript_path)
        else:
            resume_context, prev_ended_cleanly = "", False

        # Build full prompt with task + context
        base_prompt = build_task_prompt(workspace_dir)
        remaining_hours = config.time_hours

        if resume_context:
            # If previous session ended cleanly, the context already has a continue message appended
            # If it crashed/was interrupted, we tell the agent to check workspace and continue
            if prev_ended_cleanly:
                resume_instruction = (
                    f"You have {remaining_hours} more hours. The previous session completed a phase "
                    f"but there's more work to do. Continue improving the solution."
                )
            else:
                resume_instruction = (
                    f"You have {remaining_hours} more hours. The previous session was interrupted. "
                    f"Continue from where you left off."
                )

            prompt = (
                f"{base_prompt}\n\n"
                f"---\n\n"
                f"# RESUMING FROM PREVIOUS SESSION\n\n"
                f"{resume_instruction}\n\n"
                f"{resume_context}"
            )
            logger.info(
                f"Resuming with transcript seeding (context: {len(resume_context)} chars, "
                f"clean_stop: {prev_ended_cleanly})"
            )
        else:
            prompt = (
                f"{base_prompt}\n\n"
                f"---\n\n"
                f"# RESUMING FROM PREVIOUS SESSION\n\n"
                f"You have {remaining_hours} more hours. A previous session worked on this task. "
                f"Check the workspace for any existing work (git log, created files) and continue."
            )
            logger.warning("Resuming without transcript context (transcript not found)")
    else:
        prompt = build_task_prompt(workspace_dir)

    logger.info(f"Starting Claude Code agent run")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Time limit: {config.time_hours} hours")
    logger.info(f"  Budget limit: ${config.budget_limit}")
    logger.info(f"  Blocked URLs: {len(config.blocked_urls)}")
    logger.info(f"  Workspace: {workspace_dir}")

    result = {
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_cost_usd": 0.0,
        "total_turns": 0,
        "session_id": None,
        "error": None,
        "time": {
            "wall_clock_seconds": 0.0,
            "active_seconds": 0.0,
            "retry_seconds": 0.0,
        },
    }

    # Retry loop for transient errors
    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY
    current_session_id = resume_session_id
    continuation_count = 0  # Track how many times we've continued after voluntary stops

    try:
        while True:
            try:
                # Run with asyncio timeout as backup (use wall clock + buffer for safety)
                # The actual time limit enforcement uses active time in the loop
                async with asyncio_timeout(time_limit_secs + 3600):  # +1hr buffer for retries
                    # Use resume if we have a session from a previous attempt
                    if current_session_id and current_session_id != resume_session_id:
                        options_dict["resume"] = current_session_id
                        options = ClaudeAgentOptions(**options_dict)
                        prompt = "Continue from where you left off."
                        logger.info(f"Resuming session {current_session_id} after retry")

                    # Track if we should continue with a new query after this client session
                    should_continue_outer = False

                    async with ClaudeSDKClient(options) as client:
                        await client.query(prompt)
                        should_exit = False  # Track if we should exit the loop
                        graceful_stop_triggered = False  # Track if we've called interrupt()

                        async for message in client.receive_messages():
                            # Serialize and save message
                            serialized = serialize_message(message)
                            transcript.append(serialized)

                            # Save transcript incrementally
                            save_transcript(transcript_path, transcript)

                            # Capture session ID from any message that has it (for resume support)
                            # Session ID is typically available in early messages
                            message_type = type(message).__name__
                            if hasattr(message, "session_id") and message.session_id:
                                if not result.get("session_id"):
                                    logger.info(f"Captured session ID: {message.session_id}")
                                result["session_id"] = message.session_id
                                current_session_id = message.session_id

                            # Estimate costs during run (real usage comes from ResultMessage at end)
                            if message_type in ("AssistantMessage", "UserMessage"):
                                content = getattr(message, "content", None)
                                if content is None:
                                    content = serialized.get("content", "")
                                msg_model = getattr(message, "model", None) or serialized.get("model")
                                try:
                                    cost_tracker.record_estimated_message(
                                        message_type=message_type,
                                        content=content,
                                        model=msg_model,
                                    )
                                    # Save cost summary periodically
                                    cost_tracker.save(cost_summary_path)
                                except BudgetExceeded as e:
                                    logger.warning(f"Budget exceeded: {e}")
                                    result["status"] = "budget_exceeded"
                                    result["error"] = str(e)
                                    should_exit = True

                            # Check for graceful stop (budget/time threshold)
                            # Must interrupt mid-stream because agent doesn't naturally stop
                            if not graceful_stop_triggered and not should_exit:
                                should_stop, stop_reason = cost_tracker.should_graceful_stop()
                                if should_stop:
                                    logger.info(f"Graceful stop triggered: {stop_reason}")
                                    logger.info("Calling interrupt() to stop agent...")
                                    try:
                                        await client.interrupt()
                                        graceful_stop_triggered = True
                                        result["status"] = "graceful_stop"
                                        result["error"] = f"Graceful stop: {stop_reason}"
                                        # Don't set should_exit yet - wait for ResultMessage
                                    except Exception as e:
                                        logger.warning(f"interrupt() failed: {e}")
                                        should_exit = True

                            # Check for retry/rate limit events in message
                            # SDK may emit these as error messages or specific event types
                            if hasattr(message, "is_error") and message.is_error:
                                error_content = str(getattr(message, "content", ""))
                                if "rate" in error_content.lower() or "retry" in error_content.lower():
                                    # Estimate retry wait time from message if available
                                    # Default to 30 seconds if we can't determine
                                    retry_wait = 30.0
                                    if hasattr(message, "retry_after"):
                                        retry_wait = float(message.retry_after)
                                    cost_tracker.add_retry_time(retry_wait)
                                    logger.info(f"Rate limit detected, adding {retry_wait}s to retry time")

                            # ResultMessage contains real usage data (after interrupt or completion)
                            if message_type == "ResultMessage" or hasattr(message, "total_cost_usd"):
                                real_cost = getattr(message, "total_cost_usd", None)
                                subtype = getattr(message, "subtype", "unknown")
                                logger.info(f"ResultMessage: subtype={subtype}, cost=${real_cost}")

                                try:
                                    cost_tracker.record_from_result_message(message)
                                except BudgetExceeded as e:
                                    logger.warning(f"Budget exceeded: {e}")
                                    result["status"] = "budget_exceeded"
                                    result["error"] = str(e)
                                    should_exit = True

                                # Save cost summary
                                cost_tracker.save(cost_summary_path)

                                # Check if we should continue (subtype=success but time/budget remaining)
                                # The Stop hook's continue_: True doesn't work reliably, so we implement
                                # our own continuation loop here
                                if subtype == "success" and result["status"] not in ("budget_exceeded", "time_limit", "graceful_stop"):
                                    # Use graceful stop check for consistency
                                    should_stop, stop_reason = cost_tracker.should_graceful_stop()
                                    if not should_stop:
                                        # Still have time and budget - continue
                                        remaining_time = cost_tracker.get_remaining_time() or 0
                                        remaining_budget = cost_tracker.get_remaining_budget()
                                        logger.info(
                                            f"Agent stopped but resources remain: "
                                            f"{remaining_time/3600:.2f}h, ${remaining_budget:.2f}. Continuing..."
                                        )
                                        # Break inner loop to start new query
                                        should_continue_outer = True
                                        break
                                    else:
                                        logger.info(f"Agent stopped, graceful stop triggered: {stop_reason}. Ending.")
                                        result["status"] = "graceful_stop"
                                        result["error"] = f"Graceful stop: {stop_reason}"
                                        should_exit = True
                                else:
                                    # Error or limits reached - exit
                                    should_exit = True

                            # Track subagent costs from Task tool results
                            # The SDK emits SubagentStop events or ToolResult events with subagent costs
                            if message_type in ("SubagentStop", "TaskOutput"):
                                subagent_cost = getattr(message, "total_cost_usd", None)
                                subagent_usage = getattr(message, "usage", None)
                                if subagent_cost is not None:
                                    try:
                                        cost_tracker.record_subagent_cost(
                                            cost_usd=subagent_cost,
                                            usage=subagent_usage,
                                            subagent_type=getattr(message, "subagent_type", "unknown"),
                                        )
                                        logger.debug(f"Recorded subagent cost: ${subagent_cost:.4f}")
                                    except BudgetExceeded as e:
                                        logger.warning(f"Budget exceeded from subagent: {e}")
                                        result["status"] = "budget_exceeded"
                                        result["error"] = str(e)
                                        should_exit = True
                                    cost_tracker.save(cost_summary_path)

                            # Also check for tool results that may contain subagent costs
                            # (Task tool returns TaskOutput with usage info)
                            if message_type == "ToolResult" and hasattr(message, "tool_name"):
                                if message.tool_name == "Task":
                                    tool_result = getattr(message, "result", None)
                                    if tool_result and hasattr(tool_result, "total_cost_usd"):
                                        try:
                                            cost_tracker.record_subagent_cost(
                                                cost_usd=tool_result.total_cost_usd,
                                                usage=getattr(tool_result, "usage", None),
                                                subagent_type=getattr(tool_result, "subagent_type", "Task"),
                                            )
                                        except BudgetExceeded as e:
                                            logger.warning(f"Budget exceeded from Task tool: {e}")
                                            result["status"] = "budget_exceeded"
                                            result["error"] = str(e)
                                            should_exit = True
                                        cost_tracker.save(cost_summary_path)

                            if should_exit:
                                break

                # Check if we should continue with a new query
                # (agent stopped voluntarily but we have time/budget remaining)
                if should_continue_outer and current_session_id:
                    continuation_count += 1
                    # Prepare for next iteration with continuation prompt
                    options_dict["resume"] = current_session_id
                    options = ClaudeAgentOptions(**options_dict)
                    prompt = get_continue_message(extended=config.extended_continue)
                    logger.info(
                        f"Continuation #{continuation_count}: resuming session {current_session_id}"
                    )
                    # Don't break - continue the while loop
                    continue

                if result["status"] == "running":
                    result["status"] = "completed"

                # Success - exit retry loop
                break

            except asyncio.TimeoutError:
                logger.warning("Asyncio timeout reached")
                result["status"] = "timeout"
                result["error"] = "Asyncio timeout exceeded"
                break

            except BudgetExceeded as e:
                logger.warning(f"Budget exceeded: {e}")
                result["status"] = "budget_exceeded"
                result["error"] = str(e)
                break

            except TimeLimitExceeded as e:
                logger.warning(f"Time limit exceeded: {e}")
                result["status"] = "time_limit"
                result["error"] = str(e)
                break

            except Exception as e:
                # Check if this is a transient error we should retry
                if is_transient_error(e) and retry_count < MAX_RETRIES:
                    retry_count += 1
                    logger.warning(
                        f"Transient error (attempt {retry_count}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )

                    # Track retry time (excluded from time limit)
                    cost_tracker.add_retry_time(retry_delay)
                    await asyncio.sleep(retry_delay)

                    # Exponential backoff with cap
                    retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)

                    # Check if we've exceeded time limit during retries
                    if cost_tracker.is_over_time_limit():
                        logger.warning("Time limit reached during retry wait")
                        result["status"] = "time_limit"
                        result["error"] = f"Time limit reached after {retry_count} retries"
                        break

                    # Continue to next retry attempt
                    continue

                # Non-transient error or max retries exceeded
                logger.error(f"Error during run: {e}", exc_info=True)
                result["status"] = "error"
                result["error"] = str(e)
                if retry_count > 0:
                    result["error"] += f" (after {retry_count} retries)"
                break

    finally:
        # Final saves
        result["end_time"] = datetime.now().isoformat()
        result["total_cost_usd"] = cost_tracker.total_cost
        result["total_turns"] = len(cost_tracker.turns)
        result["continuation_count"] = continuation_count
        result["time"] = {
            "wall_clock_seconds": cost_tracker.get_wall_clock_time(),
            "active_seconds": cost_tracker.get_active_time(),
            "retry_seconds": cost_tracker.total_retry_time,
        }

        # Save final transcript
        save_transcript(transcript_path, transcript)

        # Save final cost summary
        cost_tracker.save(cost_summary_path)

        # Save blocked URLs log
        blocked_attempts = get_blocked_attempts(url_filter_hook)
        if blocked_attempts:
            with open(blocked_urls_log_path, "w") as f:
                json.dump(blocked_attempts, f, indent=2)

        # Save blocked paths log
        if path_guard_hook:
            blocked_paths = get_blocked_attempts(path_guard_hook)
            if blocked_paths:
                with open(blocked_paths_log_path, "w") as f:
                    json.dump(blocked_paths, f, indent=2)
                logger.info(f"Blocked {len(blocked_paths)} path access attempts")

        # Clean up subst drive
        if subst_drive:
            remove_subst_drive(subst_drive)

        # Save session ID for resume support
        # First try from captured messages, then discover from ~/.claude/projects/
        session_id = result.get("session_id")
        if not session_id:
            # Try to discover from Claude's local project storage
            # Try both the subst drive path and the original path
            if subst_drive:
                session_id = discover_session_id_from_claude_projects(Path(f"{subst_drive}:\\"))
            if not session_id:
                cwd_used = workspace_dir / "input"
                session_id = discover_session_id_from_claude_projects(cwd_used)
            if session_id:
                result["session_id"] = session_id
                logger.info(f"Discovered session ID from Claude projects: {session_id}")

        if session_id:
            session_path = log_dir / "session.json"
            with open(session_path, "w") as f:
                json.dump({
                    "session_id": session_id,
                    "status": result["status"],
                    "end_time": result["end_time"],
                }, f, indent=2)
            logger.info(f"Session ID saved: {session_id}")

        logger.info(f"Run completed with status: {result['status']}")
        logger.info(f"  Total cost: ${result['total_cost_usd']:.2f}")
        logger.info(f"  Total turns: {result['total_turns']}")
        if continuation_count > 0:
            logger.info(f"  Continuations: {continuation_count}")
        logger.info(
            f"  Time - wall: {result['time']['wall_clock_seconds']:.0f}s, "
            f"active: {result['time']['active_seconds']:.0f}s, "
            f"retry: {result['time']['retry_seconds']:.0f}s"
        )

    return result


def run_agent_sync(
    config: ClaudeCodeConfig,
    workspace_dir: Path,
    log_dir: Path,
    resume_session_id: str | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for run_agent.

    Args:
        config: Agent configuration
        workspace_dir: Path to workspace directory
        log_dir: Path to log directory
        resume_session_id: Session ID to resume (optional)

    Returns:
        Dictionary with run results
    """
    return asyncio.run(
        run_agent(
            config=config,
            workspace_dir=workspace_dir,
            log_dir=log_dir,
            resume_session_id=resume_session_id,
        )
    )
