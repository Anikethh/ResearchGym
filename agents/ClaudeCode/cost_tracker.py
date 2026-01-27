"""Cost and time tracking for Claude Code agent runs.

Tracks:
- Per-message token usage and costs, with budget enforcement
- Wall clock time (total elapsed)
- Active LLM time (wall clock minus retry time)
- Retry time (excluded from time limit calculations)

Supports both API mode (with total_cost_usd) and subscription mode
(manual calculation from token counts).
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Global Claude spend tracking (temporary - remove after runs)
try:
    from utils.global_claude_tracker import track_spend as _global_track_spend
except ImportError:
    _global_track_spend = None


# Estimated costs (chars/4) don't account for prompt caching, so they
# overestimate by roughly this factor. Apply as multiplier to budget limit
# so we don't stop prematurely. E.g., $10 limit → stop at $12 estimated.
ESTIMATE_BUDGET_MULTIPLIER = 1.2

# Thresholds for graceful stop - gives agent time to wrap up cleanly
GRACEFUL_STOP_BUDGET_THRESHOLD = 0.05  # $0.05 remaining
# Time threshold: 10% of time limit, clamped between 30s and 300s
GRACEFUL_STOP_TIME_PERCENT = 0.10  # 10% of time limit
GRACEFUL_STOP_TIME_MIN = 30  # At least 30 seconds
GRACEFUL_STOP_TIME_MAX = 300  # At most 5 minutes


class BudgetExceeded(Exception):
    """Raised when the budget limit is exceeded."""

    pass


class TimeLimitExceeded(Exception):
    """Raised when the active time limit is exceeded."""

    pass


# Pricing per 1M tokens (January 2026)
# Source: https://platform.claude.com/docs/en/about-claude/pricing
# IMPORTANT: Do not add fallback/default pricing. Unknown models should fail explicitly.
MODEL_PRICING = {
    # Claude Opus 4.5 (latest - reduced pricing from Opus 4.1)
    "claude-opus-4-5-20251101": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,  # 90% discount
        "cache_write": 6.25,  # 25% premium
    },
    # Claude Opus 4.1 (legacy)
    "claude-opus-4-1": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.5,
        "cache_write": 18.75,
    },
    # Claude Sonnet 4.5
    "claude-sonnet-4-5-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
        # Note: Long context (>200K tokens): input 6.0, output 22.5
    },
    # Claude Sonnet 4
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    # Claude Haiku 4.5
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    # Claude Haiku 3.5 (legacy)
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.0,
        "cache_read": 0.08,
        "cache_write": 1.0,
    },
}

FULL_OUTPUT_PATTERN = re.compile(r"Full output:\s*([^\]\r\n]+)")


class UnknownModelError(Exception):
    """Raised when model pricing is not found."""
    pass


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses ~4 characters per token as approximation.
    Accuracy: ~80% (varies by content type).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_tokens_from_text(text: str) -> int:
    tokens = estimate_tokens(text)
    extra_tokens = 0
    seen_paths: set[str] = set()
    for match in FULL_OUTPUT_PATTERN.finditer(text):
        path_str = match.group(1).strip().rstrip("]'\"")
        if not path_str or path_str in seen_paths:
            continue
        seen_paths.add(path_str)
        try:
            size_bytes = Path(path_str).stat().st_size
        except OSError:
            continue
        if size_bytes > 0:
            extra_tokens += max(1, size_bytes // 4)
    return tokens + extra_tokens


def _estimate_tokens_from_content(content: Any, seen: set[int] | None = None) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _estimate_tokens_from_text(content)
    if isinstance(content, bytes):
        return max(1, len(content) // 4)
    if isinstance(content, (list, tuple)):
        return sum(_estimate_tokens_from_content(item, seen) for item in content)
    if isinstance(content, dict):
        tokens = 0
        for key in ("text", "content", "input", "output", "result", "error"):
            if key in content:
                tokens += _estimate_tokens_from_content(content[key], seen)
        if tokens:
            return tokens
        try:
            return _estimate_tokens_from_text(json.dumps(content, ensure_ascii=True))
        except TypeError:
            return _estimate_tokens_from_text(str(content))

    if seen is None:
        seen = set()
    obj_id = id(content)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    tokens = 0
    for attr in ("text", "content", "input"):
        if hasattr(content, attr):
            tokens += _estimate_tokens_from_content(getattr(content, attr), seen)
    if tokens:
        return tokens
    return _estimate_tokens_from_text(str(content))


@dataclass
class TurnUsage:
    """Token usage for a single turn."""

    turn_number: int
    timestamp: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    duration_ms: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "duration_ms": self.duration_ms,
            "model": self.model,
        }


@dataclass
class CostTracker:
    """Tracks costs, token usage, and time across an agent run.

    Supports:
    - Per-turn token tracking (input, output, cached, reasoning)
    - Budget enforcement with configurable limit
    - Both API mode (using total_cost_usd) and subscription mode (manual calc)
    - Time tracking: wall clock, active LLM time, and retry time
    - Incremental saving to JSON

    Time tracking follows BasicAgent pattern:
    - Wall clock time: Total elapsed time since start
    - Retry time: Time spent waiting on rate limit retries (excluded from limit)
    - Active time: Wall clock - retry time (used for time limit enforcement)

    Cost tracking:
    - total_cost: Cumulative cost across all sessions (inherited on resume)
    - session_cost: Cost for THIS session only (resets on resume)
    - pending_estimated_cost: Estimated costs during streaming, replaced by actual when ResultMessage arrives
    """

    budget_limit: float
    model: str = "claude-opus-4-5-20251101"
    total_cost: float = 0.0
    session_cost: float = 0.0  # Cost for this session only
    pending_estimated_cost: float = 0.0  # Estimates to be replaced by actual
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_reasoning_tokens: int = 0
    turns: list[TurnUsage] = field(default_factory=list)
    # Subagent cost tracking
    total_subagent_cost: float = 0.0
    subagent_costs: list[dict[str, Any]] = field(default_factory=list)
    # Time tracking
    start_time: float = field(default_factory=time.time)
    total_retry_time: float = 0.0  # Accumulated retry wait time
    time_limit_seconds: float | None = None  # Optional time limit
    # Track cost inherited from previous session (for resume)
    inherited_cost: float = 0.0

    def get_wall_clock_time(self) -> float:
        """Get total wall clock time elapsed since start."""
        return time.time() - self.start_time

    def get_active_time(self) -> float:
        """Get active LLM time (wall clock minus retry time)."""
        return self.get_wall_clock_time() - self.total_retry_time

    def get_remaining_time(self) -> float | None:
        """Get remaining active time before limit (or None if no limit)."""
        if self.time_limit_seconds is None:
            return None
        return max(0, self.time_limit_seconds - self.get_active_time())

    def add_retry_time(self, seconds: float) -> None:
        """Add retry wait time (excluded from active time).

        Call this when a rate limit retry occurs.

        Args:
            seconds: Time spent waiting for retry
        """
        self.total_retry_time += seconds

    def is_over_time_limit(self) -> bool:
        """Check if active time has exceeded the time limit."""
        if self.time_limit_seconds is None:
            return False
        return self.get_active_time() >= self.time_limit_seconds

    def check_time_limit(self) -> None:
        """Check time limit and raise exception if exceeded."""
        if self.is_over_time_limit():
            raise TimeLimitExceeded(
                f"Time limit {self.time_limit_seconds:.0f}s exceeded. "
                f"Active time: {self.get_active_time():.0f}s, "
                f"Retry time: {self.total_retry_time:.0f}s"
            )

    def get_remaining_budget(self) -> float:
        """Get remaining budget accounting for pending estimates.

        Uses ESTIMATE_BUDGET_MULTIPLIER on budget limit since estimates
        don't account for prompt caching (actual costs are ~20% lower).
        """
        effective_limit = self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER
        effective_cost = self.total_cost + self.pending_estimated_cost
        return max(0.0, effective_limit - effective_cost)

    def should_graceful_stop(self) -> tuple[bool, str]:
        """Check if we should trigger a graceful stop.

        Returns:
            Tuple of (should_stop, reason)

        Graceful stop triggers when:
        - Remaining budget <= $0.05
        - Remaining time <= 10% of time limit (clamped 30-300s)
        """
        # Check budget
        remaining_budget = self.get_remaining_budget()
        if remaining_budget <= GRACEFUL_STOP_BUDGET_THRESHOLD:
            effective_limit = self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER
            return True, f"budget_low (${remaining_budget:.2f} remaining of ${effective_limit:.2f} effective limit)"

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

    def _get_pricing(self, model: str | None = None) -> dict[str, float]:
        """Get pricing for a model.

        Raises:
            UnknownModelError: If model pricing is not found.
        """
        model_name = model or self.model

        # Handle synthetic/placeholder models from rate limiting
        # These are not billable - return zero pricing
        if model_name in ("<synthetic>", "synthetic", ""):
            return {
                "input": 0.0,
                "output": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            }

        # Try exact match first
        if model_name in MODEL_PRICING:
            return MODEL_PRICING[model_name]

        # Try prefix match (e.g., "claude-sonnet-4-20250514-v2" matches "claude-sonnet-4-20250514")
        for key in MODEL_PRICING:
            if model_name.startswith(key):
                return MODEL_PRICING[key]

        # No fallback - fail explicitly
        available = ", ".join(sorted(MODEL_PRICING.keys()))
        raise UnknownModelError(
            f"Unknown model '{model_name}'. No pricing data available. "
            f"Add pricing to MODEL_PRICING in cost_tracker.py. "
            f"Available models: {available}"
        )

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model: str | None = None,
    ) -> float:
        """Calculate cost from token counts."""
        pricing = self._get_pricing(model)

        # Regular input tokens (not cached)
        regular_input = max(0, input_tokens - cache_read_tokens)

        cost = (
            (regular_input / 1_000_000) * pricing["input"]
            + (cache_read_tokens / 1_000_000) * pricing["cache_read"]
            + (cache_write_tokens / 1_000_000) * pricing["cache_write"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        return cost

    def record_turn(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float | None = None,
        duration_ms: int = 0,
        model: str | None = None,
    ) -> TurnUsage:
        """Record a turn's usage and cost from ResultMessage.

        This REPLACES any pending estimated costs with the actual cost.
        Call this when you receive a ResultMessage with real usage data.

        Args:
            input_tokens: Total input tokens
            output_tokens: Total output tokens
            cache_read_tokens: Tokens read from cache
            cache_write_tokens: Tokens written to cache
            reasoning_tokens: Extended thinking tokens (if applicable)
            cost_usd: Pre-calculated cost (from API). If None, calculates manually.
            duration_ms: Turn duration in milliseconds
            model: Model used for this turn (defaults to tracker's model)

        Returns:
            TurnUsage object with recorded data

        Raises:
            BudgetExceeded: If cumulative cost exceeds budget_limit
        """
        turn_model = model or self.model

        # Use provided cost or calculate
        if cost_usd is not None:
            turn_cost = cost_usd
        else:
            turn_cost = self._calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                model=turn_model,
            )

        # REPLACE pending estimates with actual cost (not add)
        # This fixes the double-counting bug
        self.total_cost += turn_cost  # Add actual
        self.session_cost += turn_cost  # Track session cost
        self.pending_estimated_cost = 0.0  # Clear estimates - actual replaces them

        # Update token totals (these are actual from API, not estimates)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_cache_write_tokens += cache_write_tokens
        self.total_reasoning_tokens += reasoning_tokens

        # Create turn record
        turn = TurnUsage(
            turn_number=len(self.turns) + 1,
            timestamp=datetime.now().isoformat(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=turn_cost,
            cumulative_cost_usd=self.total_cost,
            duration_ms=duration_ms,
            model=turn_model,
        )
        self.turns.append(turn)

        # Global spend tracking (temporary - remove after runs)
        if _global_track_spend:
            try:
                _global_track_spend(
                    model=turn_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    cost_usd=turn_cost,
                    context="ClaudeCode",
                )
            except Exception:
                pass  # Don't fail main tracking if global tracking fails

        # Check budget (no multiplier needed now - using actual costs)
        if self.total_cost >= self.budget_limit:
            raise BudgetExceeded(
                f"Budget limit ${self.budget_limit:.2f} exceeded. "
                f"Total cost: ${self.total_cost:.2f}"
            )

        return turn

    def record_from_result_message(self, result_message: Any) -> TurnUsage:
        """Record usage from a Claude SDK ResultMessage.

        Args:
            result_message: ResultMessage from claude-agent-sdk

        Returns:
            TurnUsage object with recorded data
        """
        usage = getattr(result_message, "usage", {}) or {}

        return self.record_turn(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            cost_usd=getattr(result_message, "total_cost_usd", None),
            duration_ms=getattr(result_message, "duration_ms", 0),
            model=getattr(result_message, "model", None),
        )

    def record_estimated_message(
        self,
        message_type: str,
        content: Any,
        model: str | None = None,
    ) -> None:
        """Record estimated usage from message content.

        Since the SDK doesn't expose per-message usage, we estimate tokens
        from the message content. This is approximate but good enough for
        budget enforcement.

        These estimates are tracked in pending_estimated_cost and will be
        REPLACED (not added) when the actual ResultMessage arrives.

        Args:
            message_type: Type of message ("AssistantMessage", "UserMessage", etc)
            content: Message content (string, list of blocks, dict, or SDK objects)
            model: Model used (defaults to tracker's model)

        Raises:
            BudgetExceeded: If cumulative estimated cost exceeds budget_limit

        Note:
            - AssistantMessage content → output tokens
            - UserMessage content → input tokens (tool results)
            - If content includes "Full output: <path>", estimate from that file size
            - Doesn't account for prompt caching (actual costs ~1.5x lower)
            - Budget enforcement uses ESTIMATE_BUDGET_MULTIPLIER to compensate
        """
        tokens = _estimate_tokens_from_content(content)

        if message_type == "AssistantMessage":
            # Assistant output
            cost = self._calculate_cost(
                input_tokens=0,
                output_tokens=tokens,
                model=model,
            )
        elif message_type == "UserMessage":
            # Tool results (input to next turn)
            cost = self._calculate_cost(
                input_tokens=tokens,
                output_tokens=0,
                model=model,
            )
        else:
            # System messages, etc - minimal cost
            return

        # Track as pending estimate (will be replaced by actual when ResultMessage arrives)
        self.pending_estimated_cost += cost

        # Check budget using total + pending estimates
        # (use multiplier since estimates don't account for caching)
        current_cost = self.total_cost + self.pending_estimated_cost
        effective_limit = self.budget_limit * ESTIMATE_BUDGET_MULTIPLIER
        if current_cost >= effective_limit:
            raise BudgetExceeded(
                f"Budget limit ${self.budget_limit:.2f} exceeded "
                f"(effective: ${effective_limit:.2f} with {ESTIMATE_BUDGET_MULTIPLIER}x multiplier). "
                f"Estimated cost: ${current_cost:.2f}"
            )

    def record_subagent_cost(
        self,
        cost_usd: float,
        usage: dict[str, Any] | None = None,
        subagent_type: str = "unknown",
        duration_ms: int = 0,
    ) -> None:
        """Record cost from a subagent (Task tool) execution.

        Subagent costs are tracked separately but included in total_cost
        for budget enforcement.

        Args:
            cost_usd: Cost in USD from the subagent
            usage: Token usage dict from the subagent (optional)
            subagent_type: Type/name of the subagent
            duration_ms: Execution duration in milliseconds

        Raises:
            BudgetExceeded: If cumulative cost exceeds budget_limit
        """
        usage = usage or {}

        # Record subagent cost entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "subagent_type": subagent_type,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
        }
        self.subagent_costs.append(entry)

        # Update totals
        self.total_subagent_cost += cost_usd
        self.total_cost += cost_usd
        self.session_cost += cost_usd  # Track in session cost too

        # Also aggregate token counts from subagents
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        self.total_cache_write_tokens += usage.get("cache_creation_input_tokens", 0)

        # Check budget (subagent costs are actual, no multiplier needed)
        if self.total_cost >= self.budget_limit:
            raise BudgetExceeded(
                f"Budget limit ${self.budget_limit:.2f} exceeded. "
                f"Total cost: ${self.total_cost:.2f} "
                f"(main: ${self.total_cost - self.total_subagent_cost:.2f}, "
                f"subagents: ${self.total_subagent_cost:.2f})"
            )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the cost and time tracking."""
        wall_clock = self.get_wall_clock_time()
        active_time = self.get_active_time()

        # Calculate main agent cost (excluding subagents)
        main_agent_cost = self.total_cost - self.total_subagent_cost

        return {
            "budget_limit_usd": self.budget_limit,
            "total_cost_usd": self.total_cost,
            "session_cost_usd": self.session_cost,  # Cost for THIS session only
            "inherited_cost_usd": self.inherited_cost,  # Cost from previous sessions
            "pending_estimated_cost_usd": self.pending_estimated_cost,  # Unconfirmed estimates
            "main_agent_cost_usd": main_agent_cost,
            "subagent_cost_usd": self.total_subagent_cost,
            "remaining_budget_usd": max(0, self.budget_limit - self.total_cost - self.pending_estimated_cost),
            "total_turns": len(self.turns),
            "total_subagent_calls": len(self.subagent_costs),
            "model": self.model,
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "cache_read": self.total_cache_read_tokens,
                "cache_write": self.total_cache_write_tokens,
                "reasoning": self.total_reasoning_tokens,
            },
            "time": {
                "wall_clock_seconds": wall_clock,
                "active_seconds": active_time,
                "retry_seconds": self.total_retry_time,
                "time_limit_seconds": self.time_limit_seconds,
                "remaining_seconds": self.get_remaining_time(),
            },
            "turns": [turn.to_dict() for turn in self.turns],
            "subagent_costs": self.subagent_costs,
        }

    def save(self, path: Path) -> None:
        """Save cost summary to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CostTracker":
        """Load cost tracker from JSON file (for resume).

        On resume:
        - total_cost is inherited (not reset) so cumulative tracking continues
        - session_cost starts at 0 for the new session
        - inherited_cost tracks what came from previous sessions
        - pending_estimated_cost is cleared (previous session's estimates are gone)
        """
        with open(path) as f:
            data = json.load(f)

        # Get the previous total cost
        # If total_cost is 0 but pending estimates exist (e.g., crashed before ResultMessage),
        # use pending estimates as conservative estimate of actual spend
        prev_total = data.get("total_cost_usd", 0.0)
        prev_pending = data.get("pending_estimated_cost_usd", 0.0)
        if prev_total == 0.0 and prev_pending > 0:
            # Run crashed before actual costs were recorded, use estimates
            prev_total = prev_pending

        tracker = cls(
            budget_limit=data["budget_limit_usd"],
            model=data.get("model", "claude-opus-4-5-20251101"),
            total_cost=prev_total,  # Inherit cumulative cost
            session_cost=0.0,  # New session starts fresh
            inherited_cost=prev_total,  # Track what we inherited
            pending_estimated_cost=0.0,  # Clear - new session
            total_input_tokens=data.get("total_tokens", {}).get("input", 0),
            total_output_tokens=data.get("total_tokens", {}).get("output", 0),
            total_cache_read_tokens=data.get("total_tokens", {}).get("cache_read", 0),
            total_cache_write_tokens=data.get("total_tokens", {}).get("cache_write", 0),
            total_reasoning_tokens=data.get("total_tokens", {}).get("reasoning", 0),
            total_subagent_cost=data.get("subagent_cost_usd", 0.0),
            subagent_costs=data.get("subagent_costs", []),
        )

        # Reconstruct turns from previous sessions
        for turn_data in data.get("turns", []):
            turn = TurnUsage(
                turn_number=turn_data["turn_number"],
                timestamp=turn_data["timestamp"],
                input_tokens=turn_data.get("input_tokens", 0),
                output_tokens=turn_data.get("output_tokens", 0),
                cache_read_tokens=turn_data.get("cache_read_tokens", 0),
                cache_write_tokens=turn_data.get("cache_write_tokens", 0),
                reasoning_tokens=turn_data.get("reasoning_tokens", 0),
                cost_usd=turn_data.get("cost_usd", 0.0),
                cumulative_cost_usd=turn_data.get("cumulative_cost_usd", 0.0),
                duration_ms=turn_data.get("duration_ms", 0),
                model=turn_data.get("model", ""),
            )
            tracker.turns.append(turn)

        return tracker
