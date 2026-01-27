#!/usr/bin/env python
"""Parse AI-Scientist exec stdout logs and estimate total LLM cost.

Usage
-----
python utils/parse_total_cost.py runs/2025-09-25/a98c75fe/logs/exec.stdout.log
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, Tuple

# Pricing table (per 1M tokens) aligned with ai_scientist/utils/cost_tracker.py
MODEL_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-5": {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.00,
    },
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    },
    "o1": {
        "input": 15.00,
        "cached_input": 7.50,
        "output": 60.00,
    },
    "o1-mini": {
        "input": 3.00,
        "cached_input": 1.50,
        "output": 12.00,
    },
    "o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
    },
}

MODEL_ALIAS = (
    ("gpt-5", "gpt-5"),
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4o", "gpt-4o"),
    ("o1-mini", "o1-mini"),
    ("o1", "o1"),
    ("o3-mini", "o3-mini"),
)

MODEL_LINE_RE = re.compile(r"Token usage tracking: (?P<model>\S+)")
TOKEN_LINE_RE = re.compile(
    r"Input: (?P<input>\d+), Output: (?P<output>\d+), Cached: (?P<cached>\d+), Reasoning: (?P<reasoning>\d+)"
)


def normalize_model(model: str) -> str:
    model = model.strip()
    if model in MODEL_PRICES:
        return model
    for needle, alias in MODEL_ALIAS:
        if needle in model:
            return alias
    return model


def iter_cost_entries(lines: Iterable[str]) -> Iterable[Tuple[str, int, int, int, int]]:
    """Yield (model, input, output, cached, reasoning) tuples from log lines."""
    current_model: str | None = None
    for line in lines:
        model_match = MODEL_LINE_RE.search(line)
        if model_match:
            current_model = normalize_model(model_match.group("model"))
            continue

        token_match = TOKEN_LINE_RE.search(line)
        if token_match and current_model:
            yield (
                current_model,
                int(token_match.group("input")),
                int(token_match.group("output")),
                int(token_match.group("cached")),
                int(token_match.group("reasoning")),
            )


def compute_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    prices = MODEL_PRICES.get(model)
    if not prices:
        raise KeyError(f"No pricing information for model '{model}'")

    # All prices are per 1M tokens
    per_million = 1_000_000
    cost = 0.0
    cost += input_tokens * prices.get("input", 0.0) / per_million
    cost += cached_tokens * prices.get("cached_input", 0.0) / per_million
    cost += output_tokens * prices.get("output", 0.0) / per_million
    return cost


def parse_log(path: pathlib.Path) -> Tuple[Dict[str, Dict[str, float]], float]:
    per_model = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost": 0.0})
    total_cost = 0.0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for model, input_tokens, output_tokens, cached_tokens, reasoning_tokens in iter_cost_entries(fh):
            # Reasoning tokens currently not billed for the supported models; keep them for completeness
            stats = per_model[model]
            stats["calls"] += 1
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["cached_tokens"] += cached_tokens
            stats.setdefault("reasoning_tokens", 0)
            stats["reasoning_tokens"] += reasoning_tokens

            try:
                cost = compute_cost(model, input_tokens, output_tokens, cached_tokens)
            except KeyError:
                # Skip models without pricing info but surface once to stderr
                if not stats.get("_warned", False):
                    print(f"[WARN] pricing missing for model '{model}'. Skipping cost aggregation.", file=sys.stderr)
                    stats["_warned"] = True
                continue

            stats["cost"] += cost
            total_cost += cost

    # Strip helper flag if present
    for stats in per_model.values():
        stats.pop("_warned", None)

    return per_model, total_cost


def format_usd(value: float) -> str:
    return f"$ {value:,.4f}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate total LLM cost from exec stdout log.")
    parser.add_argument("log_path", type=pathlib.Path, help="Path to exec.stdout.log")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.log_path.exists():
        print(f"Log file not found: {args.log_path}", file=sys.stderr)
        return 1

    per_model, total_cost = parse_log(args.log_path)

    if not per_model:
        print("No token usage records found in log.")
        return 0

    print(f"Parsed log: {args.log_path}")
    print("\nPer-model summary:")
    for model, stats in sorted(per_model.items()):
        print(
            f"- {model}: calls={stats['calls']}, input={stats['input_tokens']:,} tok, "
            f"output={stats['output_tokens']:,} tok, cached={stats['cached_tokens']:,} tok, "
            f"cost={format_usd(stats['cost'])}"
        )

    print("\nEstimated total cost:")
    print(f"  {format_usd(total_cost)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
