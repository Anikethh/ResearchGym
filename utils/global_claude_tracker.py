"""
Global Claude API spend tracker.
Tracks all Claude API usage across runs for accountability.

TEMPORARY FILE - Remove after runs are complete.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt
    def _lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def _unlock_file(f):
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl
    def _lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def _unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# Storage location - project root
SPEND_FILE = Path(__file__).parent.parent / ".claude_api_spend.json"

# Claude model pricing (per 1M tokens) - Jan 2026
CLAUDE_PRICING = {
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-3-5-20241022": {"input": 0.8, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
    # Aliases
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-3-5-haiku": {"input": 0.8, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
}


def _load_data() -> dict:
    """Load existing spend data or return empty structure."""
    if SPEND_FILE.exists():
        try:
            return json.loads(SPEND_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "lifetime_total_usd": 0.0,
        "first_tracked": None,
        "last_updated": None,
        "by_model": {},
        "by_day": {},
        "calls": []
    }


def _save_data(data: dict):
    """Save spend data with file locking."""
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEND_FILE, "w") as f:
        _lock_file(f)
        try:
            json.dump(data, f, indent=2)
        finally:
            _unlock_file(f)


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Calculate cost for Claude API usage."""
    # Normalize model name
    model_lower = model.lower()
    pricing = None
    for key, prices in CLAUDE_PRICING.items():
        if key in model_lower or model_lower in key:
            pricing = prices
            break

    if not pricing:
        # Default to sonnet pricing if unknown
        pricing = CLAUDE_PRICING["claude-sonnet-4-5-20250514"]

    cost = (
        (input_tokens * pricing["input"] / 1_000_000) +
        (output_tokens * pricing["output"] / 1_000_000) +
        (cache_read_tokens * pricing["cache_read"] / 1_000_000) +
        (cache_write_tokens * pricing["cache_write"] / 1_000_000)
    )
    return cost


def track_spend(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: Optional[float] = None,
    context: str = "",
    run_id: str = "",
):
    """
    Track a Claude API call.

    Args:
        model: Model name (e.g., "claude-sonnet-4-5-20250514")
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        cache_read_tokens: Cache read tokens
        cache_write_tokens: Cache write tokens
        cost_usd: Pre-calculated cost (if None, will calculate)
        context: Description of what this call was for
        run_id: Associated run ID if any
    """
    if cost_usd is None:
        cost_usd = calculate_cost(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)

    now = datetime.now().isoformat()
    today = datetime.now().strftime("%Y-%m-%d")

    data = _load_data()

    # Update totals
    data["lifetime_total_usd"] += cost_usd
    if data["first_tracked"] is None:
        data["first_tracked"] = now
    data["last_updated"] = now

    # Update by_model
    if model not in data["by_model"]:
        data["by_model"][model] = {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
    data["by_model"][model]["cost_usd"] += cost_usd
    data["by_model"][model]["calls"] += 1
    data["by_model"][model]["input_tokens"] += input_tokens
    data["by_model"][model]["output_tokens"] += output_tokens

    # Update by_day
    if today not in data["by_day"]:
        data["by_day"][today] = {"cost_usd": 0.0, "calls": 0}
    data["by_day"][today]["cost_usd"] += cost_usd
    data["by_day"][today]["calls"] += 1

    # Add to call log (keep last 1000)
    data["calls"].append({
        "timestamp": now,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "context": context,
        "run_id": run_id,
    })
    if len(data["calls"]) > 1000:
        data["calls"] = data["calls"][-1000:]

    _save_data(data)
    return cost_usd


def get_total_spend() -> float:
    """Get lifetime total spend."""
    return _load_data()["lifetime_total_usd"]


def get_summary() -> str:
    """Get a formatted summary of spend."""
    data = _load_data()

    lines = [
        "=" * 50,
        "CLAUDE API SPEND TRACKER",
        "=" * 50,
        f"Lifetime Total: ${data['lifetime_total_usd']:.4f}",
        f"First Tracked:  {data['first_tracked'] or 'N/A'}",
        f"Last Updated:   {data['last_updated'] or 'N/A'}",
        "",
        "By Model:",
    ]

    for model, stats in sorted(data["by_model"].items()):
        lines.append(f"  {model}:")
        lines.append(f"    Cost: ${stats['cost_usd']:.4f} ({stats['calls']} calls)")
        lines.append(f"    Tokens: {stats.get('input_tokens', 0):,} in / {stats.get('output_tokens', 0):,} out")

    lines.append("")
    lines.append("Recent Days:")
    for day in sorted(data["by_day"].keys(), reverse=True)[:7]:
        stats = data["by_day"][day]
        lines.append(f"  {day}: ${stats['cost_usd']:.4f} ({stats['calls']} calls)")

    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        if SPEND_FILE.exists():
            SPEND_FILE.unlink()
            print("Spend data reset.")
        else:
            print("No spend data to reset.")
    else:
        print(get_summary())
