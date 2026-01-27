"""Claude Code agent scaffold for ResearchGym.

This module provides integration with Claude Code (via claude-agent-sdk)
for running autonomous research agents on ResearchGym benchmark tasks.
"""

from .adapter import ClaudeCodeAdapter
from .config import ClaudeCodeConfig
from .cost_tracker import CostTracker, BudgetExceeded, TimeLimitExceeded, UnknownModelError
from .runner import run_agent

__all__ = [
    "ClaudeCodeAdapter",
    "ClaudeCodeConfig",
    "CostTracker",
    "BudgetExceeded",
    "TimeLimitExceeded",
    "UnknownModelError",
    "run_agent",
]
