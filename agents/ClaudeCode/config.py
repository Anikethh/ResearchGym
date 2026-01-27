"""Configuration for Claude Code agent runs."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default tools to allow
DEFAULT_ALLOWED_TOOLS = [
    # File operations
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    # Execution
    "Bash",
    # Web (with URL blocking via hooks)
    "WebFetch",
    "WebSearch",
    # Organization
    "TodoWrite",
    # Task tool for subagents
    "Task",
]

# Tools to disallow for autonomous operation
DEFAULT_DISALLOWED_TOOLS = [
    "AskUserQuestion",  # No user interaction
]

# Default model
DEFAULT_MODEL = "claude-opus-4-5-20251101"

# Default limits
DEFAULT_BUDGET_LIMIT = 10.0  # USD
DEFAULT_TIME_HOURS = 12  # hours


@dataclass
class ClaudeCodeConfig:
    """Configuration for a Claude Code agent run.

    Attributes:
        model: Model identifier to use
        time_hours: Maximum runtime in hours
        budget_limit: Maximum cost in USD
        blocked_urls: URLs to block via hooks/permissions
        use_api: If True, use API key auth; else subscription
        extended_continue: If True, use more aggressive continue messages
        allowed_tools: Tools to allow (None = use defaults)
        disallowed_tools: Tools to disallow (None = use defaults)
        status_interval: Steps between status updates
        env: Additional environment variables
    """

    model: str = DEFAULT_MODEL
    time_hours: float = DEFAULT_TIME_HOURS
    budget_limit: float = DEFAULT_BUDGET_LIMIT
    blocked_urls: list[str] = field(default_factory=list)
    use_api: bool = True
    extended_continue: bool = False
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    status_interval: int = 5
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-enable extended continue for long runs
        if self.time_hours >= 24 and not self.extended_continue:
            self.extended_continue = True

        # Use defaults if not specified
        if self.allowed_tools is None:
            self.allowed_tools = DEFAULT_ALLOWED_TOOLS.copy()
        if self.disallowed_tools is None:
            self.disallowed_tools = DEFAULT_DISALLOWED_TOOLS.copy()

    def get_permissions_deny(self) -> list[str]:
        """Generate permissions.deny patterns for blocked URLs.

        Returns list of patterns for ClaudeAgentOptions.
        """
        patterns = []
        for url in self.blocked_urls:
            # Create WebFetch deny pattern
            patterns.append(f"WebFetch(url:*{url}*)")
        return patterns

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "time_hours": self.time_hours,
            "budget_limit": self.budget_limit,
            "blocked_urls": self.blocked_urls,
            "use_api": self.use_api,
            "extended_continue": self.extended_continue,
            "allowed_tools": self.allowed_tools,
            "disallowed_tools": self.disallowed_tools,
            "status_interval": self.status_interval,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaudeCodeConfig":
        """Create config from dictionary."""
        return cls(
            model=data.get("model", DEFAULT_MODEL),
            time_hours=data.get("time_hours", DEFAULT_TIME_HOURS),
            budget_limit=data.get("budget_limit", DEFAULT_BUDGET_LIMIT),
            blocked_urls=data.get("blocked_urls", []),
            use_api=data.get("use_api", True),
            extended_continue=data.get("extended_continue", False),
            allowed_tools=data.get("allowed_tools"),
            disallowed_tools=data.get("disallowed_tools"),
            status_interval=data.get("status_interval", 5),
            env=data.get("env", {}),
        )


def load_blocked_urls(task_dir: Path) -> list[str]:
    """Load blocked URLs from task's blocked_urls.yaml.

    Supports multiple YAML formats:
    - blocked_urls: [...]  (legacy)
    - urls: [...]          (current)
    - patterns: [...]      (current, for wildcards)

    Args:
        task_dir: Path to task directory

    Returns:
        List of blocked URL patterns
    """
    blocklist_path = task_dir / "blocked_urls.yaml"

    if not blocklist_path.exists():
        return []

    with open(blocklist_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    # Combine all URL sources
    urls = []
    urls.extend(data.get("blocked_urls", []))  # legacy format
    urls.extend(data.get("urls", []))          # current format
    urls.extend(data.get("patterns", []))      # wildcard patterns
    return urls


def get_api_key() -> str | None:
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


def is_api_mode() -> bool:
    """Check if running in API mode (vs subscription)."""
    return get_api_key() is not None
