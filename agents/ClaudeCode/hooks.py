"""Hooks for Claude Code agent.

Implements:
- Stop hook: Injects continue messages, enforces time/budget limits
- PreToolUse hook: URL blocking for WebFetch/WebSearch

Time tracking follows BasicAgent pattern:
- Uses "active time" (wall clock minus retry time) for limit enforcement
- Retry time is tracked separately and excluded from the time limit
"""

import time
from typing import Any, Callable

from .cost_tracker import CostTracker
from .messages import (
    get_continue_message,
    get_periodic_status_message,
    get_time_limit_message,
    get_time_warning_message,
    get_budget_limit_message,
)


def make_continue_hook(
    cost_tracker: CostTracker,
    start_time: float,
    time_limit_hours: float,
    extended: bool = False,
    status_interval: int = 5,
) -> Callable:
    """Create a Stop hook that keeps the agent running.

    This hook fires when Claude stops responding. It:
    1. Checks time and budget limits (using active time, not wall clock)
    2. Injects continue messages to keep agent working
    3. Provides periodic status updates

    Time limit is enforced using "active time" which excludes retry wait time.
    This follows the BasicAgent pattern where rate limit retries don't count
    against the time budget.

    Args:
        cost_tracker: CostTracker instance for budget and time checking
        start_time: Unix timestamp when run started (used for logging)
        time_limit_hours: Maximum active runtime in hours
        extended: If True, use more aggressive continue messages
        status_interval: Steps between status updates

    Returns:
        Async hook function
    """
    step_count = [0]  # Mutable counter
    warning_sent = [False]  # Track if we've sent the wrap-up warning
    time_limit_secs = time_limit_hours * 3600
    # Warn when 2 minutes remaining OR 10% of time, whichever is larger
    warning_threshold_secs = max(120, time_limit_secs * 0.1)

    async def stop_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Stop hook implementation."""
        # Prevent infinite loops - if already in a stop hook, don't continue
        if context.get("stop_hook_active"):
            return {}

        step_count[0] += 1

        # Use active time (excluding retry time) for limit checking
        active_time = cost_tracker.get_active_time()
        remaining = time_limit_secs - active_time

        # For status messages, show wall clock elapsed
        wall_clock = cost_tracker.get_wall_clock_time()

        # Check time limit (using active time)
        if remaining <= 0:
            return {
                "continue_": False,  # NOTE: SDK uses continue_ (with underscore)
                "stopReason": "Time limit reached",
                "systemMessage": get_time_limit_message(),
            }

        # Send wrap-up warning when time is almost up (but still continue)
        # This gives the agent time to commit, save, and finish gracefully
        if remaining <= warning_threshold_secs and not warning_sent[0]:
            warning_sent[0] = True
            return {
                "continue_": True,
                "systemMessage": get_time_warning_message(remaining),
            }

        # Check budget limit
        if cost_tracker.total_cost >= cost_tracker.budget_limit:
            return {
                "continue_": False,
                "stopReason": "Budget limit reached",
                "systemMessage": get_budget_limit_message(
                    cost_tracker.total_cost, cost_tracker.budget_limit
                ),
            }

        # Build continue message
        if step_count[0] % status_interval == 0:
            # Periodic status update - show active time for accuracy
            msg = get_periodic_status_message(
                elapsed_seconds=active_time,
                total_seconds=time_limit_secs,
                cost_usd=cost_tracker.total_cost,
                budget_limit_usd=cost_tracker.budget_limit,
            )
            # Add retry time info if significant
            retry_time = cost_tracker.total_retry_time
            if retry_time > 60:  # More than 1 minute of retries
                msg += f"\n(Note: {retry_time:.0f}s spent on rate limit retries, not counted against time limit)"
        else:
            # Regular continue message
            msg = get_continue_message(extended=extended)

        return {
            "continue_": True,
            "systemMessage": msg,
        }

    return stop_hook


def make_url_filter_hook(
    blocked_urls: list[str],
    log_blocked: bool = True,
) -> Callable:
    """Create a PreToolUse hook that blocks specific URLs.

    This hook intercepts WebFetch and WebSearch tool calls and
    denies access to blocked URLs (e.g., original paper, GitHub repo).

    Args:
        blocked_urls: List of URL patterns to block (substring match)
        log_blocked: If True, print blocked URL attempts

    Returns:
        Async hook function
    """
    blocked_attempts: list[dict[str, Any]] = []

    async def url_filter_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """URL filter hook implementation."""
        tool_name = input_data.get("tool_name", "")
        hook_event_name = input_data.get("hook_event_name", "PreToolUse")

        # Only process WebFetch and WebSearch
        if tool_name not in ("WebFetch", "WebSearch"):
            return {}

        tool_input = input_data.get("tool_input", {})

        # Check WebFetch URL
        if tool_name == "WebFetch":
            url = tool_input.get("url", "")
            for blocked_pattern in blocked_urls:
                if blocked_pattern in url:
                    if log_blocked:
                        print(f"[URL BLOCKED] WebFetch blocked: {url}")
                    blocked_attempts.append(
                        {
                            "tool": "WebFetch",
                            "url": url,
                            "pattern": blocked_pattern,
                            "timestamp": time.time(),
                        }
                    )
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": hook_event_name,
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Access to {url} is blocked for this evaluation. "
                                "This URL may contain information about the original solution."
                            ),
                        }
                    }

        # Check WebSearch query (optional - can block searches for paper titles etc.)
        if tool_name == "WebSearch":
            query = tool_input.get("query", "")
            # For now, just log searches - can add blocking logic if needed
            # This is less strict since search results are indirect
            pass

        return {}

    # Attach blocked attempts list for later inspection
    url_filter_hook.blocked_attempts = blocked_attempts  # type: ignore

    return url_filter_hook


def get_blocked_attempts(hook: Callable) -> list[dict[str, Any]]:
    """Get list of blocked URL attempts from a URL filter hook."""
    return getattr(hook, "blocked_attempts", [])


def make_path_guard_hook(
    allowed_drive: str,
    workspace_path: str,
) -> Callable:
    """Create a PreToolUse hook that blocks access outside the workspace.

    This hook intercepts file/bash operations and blocks any that try to
    access paths outside the allowed workspace drive.

    Args:
        allowed_drive: Drive letter that workspace is mounted on (e.g., "R")
        workspace_path: Original workspace path (for logging)

    Returns:
        Async hook function
    """
    blocked_attempts: list[dict[str, Any]] = []
    drive_prefix = f"{allowed_drive}:"
    drive_prefix_lower = drive_prefix.lower()

    # Patterns that suggest accessing outside workspace
    suspicious_patterns = [
        "ResearchGym",
        "/runs/",
        "\\runs\\",
        "/results/",
        "\\results\\",
        "/tasks/",
        "\\tasks\\",
    ]

    async def path_guard_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Path guard hook implementation."""
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        hook_event_name = input_data.get("hook_event_name", "PreToolUse")

        def is_path_allowed(path: str) -> bool:
            """Check if a path is within allowed workspace."""
            if not path:
                return True

            path_lower = path.lower()

            # Check for suspicious patterns that indicate benchmark structure
            for pattern in suspicious_patterns:
                if pattern.lower() in path_lower:
                    return False

            # Allow relative paths (no drive letter or /)
            if not path.startswith("/") and ":" not in path[:3]:
                # But block .. traversal
                if ".." in path:
                    return False
                return True

            # Allow paths on the workspace drive
            if path_lower.startswith(drive_prefix_lower):
                return True
            if path_lower.startswith(f"/{allowed_drive.lower()}/"):
                return True

            # Block absolute paths on other drives or roots
            return False

        def check_bash_command(command: str) -> tuple[bool, str]:
            """Check if bash command accesses forbidden paths.

            Returns (allowed, reason).
            """
            # Check for suspicious patterns in the command
            for pattern in suspicious_patterns:
                if pattern in command:
                    return False, f"Command contains forbidden pattern: {pattern}"

            # Block find/ls on root or other drives
            if "find /" in command or "find \\" in command:
                cmd_lower = command.lower()
                allowed_find = (
                    f"find {drive_prefix.lower()}" in cmd_lower or  # find R:\ or R:/
                    f"find /{allowed_drive.lower()}/" in cmd_lower or  # find /r/path
                    f"find /{allowed_drive.lower()} " in cmd_lower or  # find /r -name
                    "find . " in cmd_lower or "find ./" in cmd_lower  # find . (current dir)
                )
                if not allowed_find:
                    return False, "find on root/other drives not allowed"

            # Block explicit absolute paths to other drives
            # Look for patterns like /c/, /e/, C:\, E:\, etc.
            import re
            other_drive_pattern = re.compile(r'[/\\][a-df-z][/\\]|[A-DF-Z]:[/\\]', re.IGNORECASE)
            # Exclude our allowed drive
            matches = other_drive_pattern.findall(command)
            for match in matches:
                match_drive = match[1].upper() if match[0] in '/\\' else match[0].upper()
                if match_drive != allowed_drive:
                    return False, f"Access to drive {match_drive} not allowed"

            return True, ""

        # Check file operation tools
        if tool_name in ("Read", "Write", "Edit", "Glob", "Grep"):
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if not is_path_allowed(file_path):
                blocked_attempts.append({
                    "tool": tool_name,
                    "path": file_path,
                    "timestamp": time.time(),
                })
                return {
                    "hookSpecificOutput": {
                        "hookEventName": hook_event_name,
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Access denied: path outside workspace. "
                            f"Use relative paths or paths on {drive_prefix}"
                        ),
                    }
                }

        # Check Bash commands
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            allowed, reason = check_bash_command(command)
            if not allowed:
                blocked_attempts.append({
                    "tool": "Bash",
                    "command": command[:200],  # Truncate for logging
                    "reason": reason,
                    "timestamp": time.time(),
                })
                return {
                    "hookSpecificOutput": {
                        "hookEventName": hook_event_name,
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Command blocked: {reason}. "
                            f"Use relative paths or paths on {drive_prefix}"
                        ),
                    }
                }

        return {}

    # Attach blocked attempts list for inspection
    path_guard_hook.blocked_attempts = blocked_attempts  # type: ignore

    return path_guard_hook
