"""Cost tracking for Codex CLI runs.

Tracks cumulative token usage from JSONL events and computes costs per model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


class BudgetExceeded(Exception):
    """Raised when the budget limit is exceeded."""


class UnknownModelError(Exception):
    """Raised when model pricing is not found."""


# Pricing per 1M tokens (January 2026)
# Source: https://platform.openai.com/docs/pricing
# IMPORTANT: Do not add fallback/default pricing. Unknown models should fail explicitly.
MODEL_PRICING = {
    "gpt-5-codex": {"input": 1.25, "output": 10.00, "cached": 0.3125},
    "gpt-5": {"input": 1.25, "output": 10.00, "cached": 0.3125},
    "gpt-5.2-codex": {"input": 1.75, "output": 14.00, "cached": 0.175},
    "o3": {"input": 2.00, "output": 8.00, "cached": 0.50},
    "o3-mini": {"input": 0.55, "output": 2.20, "cached": 0.14},
}


def _get_pricing(model: str) -> dict[str, float]:
    model_lower = model.lower()
    if model_lower in MODEL_PRICING:
        return MODEL_PRICING[model_lower]
    for key in MODEL_PRICING:
        if model_lower.startswith(key):
            return MODEL_PRICING[key]
    available = ", ".join(sorted(MODEL_PRICING.keys()))
    raise UnknownModelError(
        f"Unknown model '{model}'. No pricing data available. "
        f"Add pricing to MODEL_PRICING in cost_tracker.py. "
        f"Available models: {available}"
    )


def _calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    model: str,
) -> float:
    pricing = _get_pricing(model)
    regular_input = max(0, input_tokens - cached_tokens)
    return (
        (regular_input / 1_000_000) * pricing["input"]
        + (cached_tokens / 1_000_000) * pricing["cached"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )


@dataclass
class TurnUsage:
    """Per-turn usage derived from cumulative token counts."""

    turn_number: int
    timestamp: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    cumulative_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
            "cumulative_cost_usd": self.cumulative_cost_usd,
        }


# Estimate budget multiplier: estimates don't account for caching, so actual
# costs are ~20% lower. Apply this to budget limit, not to cost estimate.
# E.g., $10 budget → stop at ~$12 estimated cost.
ESTIMATE_BUDGET_MULTIPLIER = 1.2

# Thresholds for graceful stop
GRACEFUL_STOP_BUDGET_THRESHOLD = 0.05  # $0.05 remaining
# Time threshold: 10% of time limit, clamped between 30s and 300s
GRACEFUL_STOP_TIME_PERCENT = 0.10  # 10% of time limit
GRACEFUL_STOP_TIME_MIN = 30  # At least 30 seconds
GRACEFUL_STOP_TIME_MAX = 300  # At most 5 minutes


@dataclass
class CodexCostTracker:
    """Track cumulative token usage and cost for Codex CLI.

    Cost tracking:
    - total_cost_usd: Cumulative cost across all sessions (inherited on resume)
    - session_cost_usd: Cost for THIS session only (resets on resume)
    - inherited_cost_usd: Cost inherited from previous sessions
    - pending_estimated_cost_usd: Estimated costs during streaming, replaced by actual
    """

    budget_limit: float
    model: str
    total_cost_usd: float = 0.0
    session_cost_usd: float = 0.0  # Cost for this session only
    inherited_cost_usd: float = 0.0  # Cost from previous sessions
    pending_estimated_cost_usd: float = 0.0  # Estimates to be replaced by actual
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    prev_input_tokens: int = 0
    prev_output_tokens: int = 0
    prev_cached_tokens: int = 0
    turns: list[TurnUsage] = field(default_factory=list)
    # Time tracking
    start_time: float = field(default_factory=time.time)
    total_retry_time: float = 0.0
    time_limit_seconds: float | None = None
    # Estimate tracking (before turn.completed arrives)
    estimated_input_chars: int = 0
    estimated_output_chars: int = 0
    estimated_cost_usd: float = 0.0
    has_actual_usage: bool = False  # True once we get real usage data

    def get_wall_clock_time(self) -> float:
        return time.time() - self.start_time

    def get_active_time(self) -> float:
        return self.get_wall_clock_time() - self.total_retry_time

    def get_remaining_time(self) -> float | None:
        if self.time_limit_seconds is None:
            return None
        return max(0.0, self.time_limit_seconds - self.get_active_time())

    def add_retry_time(self, seconds: float) -> None:
        self.total_retry_time += seconds

    def is_over_time_limit(self) -> bool:
        if self.time_limit_seconds is None:
            return False
        return self.get_active_time() >= self.time_limit_seconds

    def add_estimated_chars(self, input_chars: int = 0, output_chars: int = 0) -> None:
        """Add estimated character counts for cost estimation.

        Uses chars/4 as token estimate, with ESTIMATE_MULTIPLIER for safety.
        """
        self.estimated_input_chars += input_chars
        self.estimated_output_chars += output_chars
        self._update_estimated_cost()

    def _update_estimated_cost(self) -> None:
        """Recalculate estimated cost from character counts.

        Updates both estimated_cost_usd (total estimate) and
        pending_estimated_cost_usd (to be replaced by actual when turn completes).

        Note: Does NOT apply multiplier here - multiplier is applied to budget
        limit when checking, since estimates don't account for caching.
        """
        # Estimate tokens as chars / 4
        est_input_tokens = self.estimated_input_chars // 4
        est_output_tokens = self.estimated_output_chars // 4

        # Calculate base cost (no cached tokens in estimates)
        base_cost = _calculate_cost(
            input_tokens=est_input_tokens,
            output_tokens=est_output_tokens,
            cached_tokens=0,
            model=self.model,
        )
        self.estimated_cost_usd = base_cost
        self.pending_estimated_cost_usd = self.estimated_cost_usd

    def get_effective_cost(self) -> float:
        """Get the current best cost estimate.

        Returns total actual cost plus any pending estimates that haven't
        been confirmed yet. This ensures budget enforcement catches costs
        even before turn.completed events arrive.
        """
        return self.total_cost_usd + self.pending_estimated_cost_usd

    def get_remaining_budget(self) -> float:
        """Get remaining budget based on effective cost.

        Uses ESTIMATE_BUDGET_MULTIPLIER to compute effective limit.
        """
        if self.budget_limit <= 0:
            return float("inf")
        effective_limit = self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER
        return max(0.0, effective_limit - self.get_effective_cost())

    def should_graceful_stop(self) -> tuple[bool, str]:
        """Check if we should trigger a graceful stop.

        Returns:
            Tuple of (should_stop, reason)

        Note: For budget checks with estimates, we use ESTIMATE_BUDGET_MULTIPLIER
        to allow estimates to exceed nominal budget by ~20% (since estimates
        don't account for caching, actual costs are typically lower).
        """
        # Check budget (use multiplier for estimate-based checks)
        effective_cost = self.get_effective_cost()
        effective_limit = self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER
        remaining = effective_limit - effective_cost
        if remaining <= GRACEFUL_STOP_BUDGET_THRESHOLD:
            return True, f"budget_low (${remaining:.2f} remaining of ${effective_limit:.2f} effective limit)"

        # Check time (threshold is 10% of time limit, clamped between 30s and 300s)
        remaining_time = self.get_remaining_time()
        if remaining_time is not None and self.time_limit_seconds is not None:
            time_threshold = max(
                GRACEFUL_STOP_TIME_MIN,
                min(GRACEFUL_STOP_TIME_MAX, self.time_limit_seconds * GRACEFUL_STOP_TIME_PERCENT)
            )
            if remaining_time <= time_threshold:
                return True, f"time_low ({remaining_time:.0f}s remaining, threshold {time_threshold:.0f}s)"

        return False, ""

    def is_over_budget(self) -> bool:
        """Check if effective cost exceeds budget.

        Uses ESTIMATE_BUDGET_MULTIPLIER to allow estimates to exceed nominal
        budget by ~20% (since estimates don't account for caching).
        """
        if self.budget_limit <= 0:
            return False
        effective_limit = self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER
        return self.get_effective_cost() >= effective_limit

    def record_cumulative(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
    ) -> TurnUsage | None:
        """Record usage from cumulative token totals.

        Codex JSONL usage is cumulative; we compute deltas per turn.
        This REPLACES any pending estimated costs with the actual cost.
        """
        delta_input = input_tokens - self.prev_input_tokens
        delta_output = output_tokens - self.prev_output_tokens
        delta_cached = max(0, cached_tokens - self.prev_cached_tokens)

        if delta_input <= 0 and delta_output <= 0:
            return None

        turn_cost = _calculate_cost(
            input_tokens=delta_input,
            output_tokens=delta_output,
            cached_tokens=delta_cached,
            model=self.model,
        )

        # REPLACE pending estimates with actual cost (not add)
        self.total_cost_usd += turn_cost
        self.session_cost_usd += turn_cost  # Track session cost
        self.pending_estimated_cost_usd = 0.0  # Clear - actual replaces estimates
        self.estimated_input_chars = 0  # Reset estimate tracking
        self.estimated_output_chars = 0
        self.estimated_cost_usd = 0.0

        self.total_input_tokens = input_tokens
        self.total_output_tokens = output_tokens
        self.total_cached_tokens = cached_tokens
        self.prev_input_tokens = input_tokens
        self.prev_output_tokens = output_tokens
        self.prev_cached_tokens = cached_tokens
        self.has_actual_usage = True  # Mark that we have real usage data

        turn = TurnUsage(
            turn_number=len(self.turns) + 1,
            timestamp=datetime.utcnow().isoformat() + "Z",
            input_tokens=delta_input,
            output_tokens=delta_output,
            cached_tokens=delta_cached,
            cost_usd=turn_cost,
            cumulative_cost_usd=self.total_cost_usd,
        )
        self.turns.append(turn)

        if self.budget_limit > 0 and self.total_cost_usd >= self.budget_limit:
            raise BudgetExceeded(
                f"Budget limit ${self.budget_limit:.2f} exceeded. "
                f"Total cost: ${self.total_cost_usd:.2f}"
            )

        return turn

    def get_summary(self) -> dict[str, Any]:
        # Calculate remaining against RAW budget limit (not effective limit)
        # The 1.2x multiplier is only for internal stop decisions, not user-facing numbers
        effective_cost = self.get_effective_cost()
        remaining_budget = max(0.0, self.budget_limit - effective_cost)

        return {
            "budget_limit_usd": self.budget_limit,
            "total_cost_usd": self.total_cost_usd,
            "session_cost_usd": self.session_cost_usd,  # Cost for THIS session only
            "inherited_cost_usd": self.inherited_cost_usd,  # Cost from previous sessions
            "pending_estimated_cost_usd": self.pending_estimated_cost_usd,  # Unconfirmed estimates
            "effective_cost_usd": effective_cost,
            "remaining_budget_usd": remaining_budget,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "model": self.model,
            "total_turns": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
            "time": {
                "wall_clock_seconds": self.get_wall_clock_time(),
                "active_seconds": self.get_active_time(),
                "retry_seconds": self.total_retry_time,
                "time_limit_seconds": self.time_limit_seconds,
                "remaining_seconds": self.get_remaining_time(),
            },
            "estimates": {
                "input_chars": self.estimated_input_chars,
                "output_chars": self.estimated_output_chars,
                "estimated_cost_usd": self.estimated_cost_usd,
                "has_actual_usage": self.has_actual_usage,
                "budget_multiplier": ESTIMATE_BUDGET_MULTIPLIER,
                "effective_budget_limit": self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.get_summary(), indent=2))

    @classmethod
    def load(cls, path: Path, budget_limit: float, model: str) -> "CodexCostTracker":
        """Load cost tracker from JSON file (for resume).

        On resume:
        - total_cost_usd is inherited (not reset) so cumulative tracking continues
        - session_cost_usd starts at 0 for the new session
        - inherited_cost_usd tracks what came from previous sessions
        - pending_estimated_cost_usd is cleared (previous session's estimates are gone)
        """
        data = json.loads(path.read_text())

        # Get the previous total cost
        # If total_cost is 0 but pending/estimated costs exist (e.g., crashed before usage recorded),
        # use estimates as conservative estimate of actual spend
        prev_total = data.get("total_cost_usd", 0.0)
        prev_pending = data.get("pending_estimated_cost_usd", 0.0)
        prev_estimated = data.get("estimates", {}).get("estimated_cost_usd", 0.0)
        if prev_total == 0.0 and (prev_pending > 0 or prev_estimated > 0):
            # Run crashed before actual costs were recorded, use estimates
            prev_total = max(prev_pending, prev_estimated)

        tracker = cls(budget_limit=budget_limit, model=model)
        tracker.total_cost_usd = prev_total  # Inherit cumulative cost
        tracker.session_cost_usd = 0.0  # New session starts fresh
        tracker.inherited_cost_usd = prev_total  # Track what we inherited
        tracker.pending_estimated_cost_usd = 0.0  # Clear - new session
        tracker.total_input_tokens = data.get("total_input_tokens", 0)
        tracker.total_output_tokens = data.get("total_output_tokens", 0)
        tracker.total_cached_tokens = data.get("total_cached_tokens", 0)
        tracker.prev_input_tokens = tracker.total_input_tokens
        tracker.prev_output_tokens = tracker.total_output_tokens
        tracker.prev_cached_tokens = tracker.total_cached_tokens
        tracker.has_actual_usage = prev_total > 0  # Had usage if cost > 0

        # Reconstruct turns from previous sessions
        for turn_data in data.get("turns", []):
            turn = TurnUsage(
                turn_number=turn_data["turn_number"],
                timestamp=turn_data["timestamp"],
                input_tokens=turn_data.get("input_tokens", 0),
                output_tokens=turn_data.get("output_tokens", 0),
                cached_tokens=turn_data.get("cached_tokens", 0),
                cost_usd=turn_data.get("cost_usd", 0.0),
                cumulative_cost_usd=turn_data.get("cumulative_cost_usd", 0.0),
            )
            tracker.turns.append(turn)

        return tracker
