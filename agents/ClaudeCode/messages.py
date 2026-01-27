"""Continue messages and status templates for Claude Code agent.

These messages are injected via Stop hooks to keep the agent working
autonomously until time/budget limits are reached.
"""


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# Default continue message (when agent stops without tool calls)
# Used for standard 12-hour runs
DEFAULT_CONTINUE_MESSAGE = """Please proceed to the next step using your best judgement. If you believe you are finished, double check your work to continue to refine and improve your submission."""

# Extended continue message for longer runs (24+ hours)
# More aggressive about continuing work
EXTENDED_CONTINUE_MESSAGE = """Please proceed to the next step using your best judgement. You should always strive to improve performance further, if you have ideas on how to improve the current proposed solution, you can do that. Otherwise, if you feel the current solution is sub-optimal, feel free to propose completely new ideas to improve performance further. If you are absolutely sure that you have reached the best possible performance, and verified with results on ALL sub-tasks, only then you can choose to end the task. You should be ambitious, improve the performance and keep continuing even if you beat the baseline."""


def get_continue_message(extended: bool = False) -> str:
    """Get the appropriate continue message.

    Args:
        extended: If True, use the more aggressive extended message

    Returns:
        Continue message string
    """
    return EXTENDED_CONTINUE_MESSAGE if extended else DEFAULT_CONTINUE_MESSAGE


def get_periodic_status_message(
    elapsed_seconds: float,
    total_seconds: float,
    cost_usd: float,
    budget_limit_usd: float,
) -> str:
    """Generate periodic status update message.

    Injected every N steps to keep agent informed of progress.

    Args:
        elapsed_seconds: Time elapsed since start
        total_seconds: Total time limit
        cost_usd: Current cumulative cost
        budget_limit_usd: Budget limit

    Returns:
        Status message string
    """
    elapsed_str = format_time(elapsed_seconds)
    total_str = format_time(total_seconds)
    remaining_str = format_time(max(0, total_seconds - elapsed_seconds))

    return f"""Info: {elapsed_str} time elapsed out of {total_str} ({remaining_str} remaining).
Budget: ${cost_usd:.2f} / ${budget_limit_usd:.2f} used.
Remember, you only have to stop working when the time limit has been reached.

Note: Don't forget to git commit regularly!"""


def get_time_warning_message(remaining_seconds: float) -> str:
    """Message when time is almost up - agent should wrap up gracefully."""
    remaining_str = format_time(remaining_seconds)
    return f"""⚠️ TIME WARNING: Only {remaining_str} remaining!

Please wrap up NOW:
1. Git commit any uncommitted changes immediately
2. Save your current progress
3. Write a brief summary of what you accomplished

This is your final warning - the session will end soon. Finishing gracefully ensures your work is saved and the session can be resumed later."""


def get_time_limit_message() -> str:
    """Message when time limit is reached."""
    return "Time limit reached. Session ending. Your work has been saved."


def get_budget_limit_message(cost_usd: float, budget_limit_usd: float) -> str:
    """Message when budget limit is reached."""
    return f"Budget limit reached (${cost_usd:.2f} / ${budget_limit_usd:.2f}). Please wrap up your current work and provide a final summary of what was accomplished."


# System prompt additions for autonomous operation
AUTONOMOUS_SYSTEM_PROMPT = """
You are an autonomous research agent working on a research task. You should:

1. Work continuously until the time limit or budget limit is reached
2. Make your own decisions without waiting for user input
3. Commit your work regularly using git
4. Run experiments, analyze results, and iterate on your approach
5. Document your methodology and findings
6. Prioritize items marked [Primary subtask] - focus on these first before other subtasks

You will receive periodic status updates about time and budget remaining.
Use all available time to improve your solution.
"""
