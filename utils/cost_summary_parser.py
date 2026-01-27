#!/usr/bin/env python3
"""
AI Scientist Cost Summary Parser
Extracts and summarizes token usage and cost information from AI Scientist execution logs.
This version uses robust, unicode-insensitive patterns and produces clean text output.
"""

import re
import json
from pathlib import Path
from typing import Dict


def parse_cost_summary_from_log(log_file_path: Path) -> Dict:
    """
    Parse AI Scientist execution log to extract comprehensive cost and token usage information.

    Args:
        log_file_path: Path to the exec.stdout.log file

    Returns:
        Dictionary containing cost summary information
    """
    if not log_file_path.exists():
        return {"error": f"Log file not found: {log_file_path}"}

    try:
        raw_content = log_file_path.read_text(encoding="utf-8", errors="ignore")
        # Strip ANSI escape codes and Rich-style markup so that regex parsing is stable
        content = re.sub(r"\x1b\[[0-9;]*m", "", raw_content)
        content = re.sub(r"\[[^\]]+\]", "", content)

        # Split into logical lines and drop empties
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return {"error": "Log file was empty after cleaning"}

        # Collect per-call token usage and running cost samples
        token_entries = []
        for idx, line in enumerate(lines):
            if "Token usage tracking:" not in line:
                continue
            model_fragment = line.split("Token usage tracking:", 1)[-1].strip()
            model = model_fragment.lstrip("✓•*-▶").strip() or "Unknown"

            tokens_line = ""
            for look_ahead in range(idx + 1, min(idx + 4, len(lines))):
                if "Input:" in lines[look_ahead]:
                    tokens_line = lines[look_ahead]
                    break
            if not tokens_line:
                continue

            def _extract(label: str) -> int:
                match = re.search(rf"{label}:\s*(\d+)", tokens_line)
                return int(match.group(1)) if match else 0

            entry = (
                model,
                _extract("Input"),
                _extract("Output"),
                _extract("Cached"),
                _extract("Reasoning"),
            )
            token_entries.append(entry)

        cost_matches = [float(match) for match in re.findall(r"Total cost so far:\s*\$([0-9.]+)", content)]
        final_cost = cost_matches[-1] if cost_matches else 0.0
        total_requests = max(len(cost_matches), len(token_entries))

        if final_cost == 0.0 and not token_entries:
            return {"error": "No cost or token usage information found in log"}

        total_input = sum(entry[1] for entry in token_entries)
        total_output = sum(entry[2] for entry in token_entries)
        total_cached = sum(entry[3] for entry in token_entries)
        total_reasoning = sum(entry[4] for entry in token_entries)
        total_tokens = total_input + total_output + total_cached + total_reasoning
        representative_model = token_entries[-1][0] if token_entries else "Unknown"

        summary: Dict = {
            "total_cost_usd": final_cost,
            "total_requests": total_requests,
            "total_entries": total_requests,
            "total_interactions": total_requests,
            "total_tokens": total_tokens,
            "model": representative_model,
            "token_breakdown": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cached_tokens": total_cached,
                "reasoning_tokens": total_reasoning,
            },
            "efficiency_metrics": {
                "cost_per_1k_tokens": (final_cost / total_tokens * 1000) if total_tokens > 0 else 0.0,
                "cost_per_request": final_cost / total_requests if total_requests > 0 else 0.0,
            },
            "cost_progression": cost_matches[-10:],
        }

        # Aggregate totals by model for downstream token accounting
        by_model: Dict[str, Dict] = {}
        for model, in_tokens, out_tokens, cached_tokens, reasoning_tokens in token_entries:
            entry = by_model.setdefault(
                model,
                {
                    "cost_usd": 0.0,
                    "tokens": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "reasoning_tokens": 0,
                        "total_tokens": 0,
                    },
                    "timing": {"num_requests": 0},
                },
            )
            entry["tokens"]["input_tokens"] += in_tokens
            entry["tokens"]["output_tokens"] += out_tokens
            entry["tokens"]["cached_input_tokens"] += cached_tokens
            entry["tokens"]["reasoning_tokens"] += reasoning_tokens
            entry["tokens"]["total_tokens"] += in_tokens + out_tokens + cached_tokens + reasoning_tokens
            entry["timing"]["num_requests"] += 1

        for model, data in by_model.items():
            summary[model] = data

        return summary

    except Exception as e:
        return {"error": f"Failed to parse log file: {e}"}


def format_cost_summary(summary: Dict) -> str:
    """Format cost summary for display."""
    if "error" in summary:
        return f"Error: {summary['error']}"

    lines = [
        "AI-Scientist Execution Summary:",
        "=" * 50,
        f"Total Cost: ${summary['total_cost_usd']:.6f}",
        f"Total Requests: {summary['total_requests']:,}",
        f"Total Tokens: {summary['total_tokens']:,}",
        f"Model: {summary['model']}",
        "",
        "Token Breakdown:",
        f"  Input tokens: {summary['token_breakdown']['input_tokens']:,}",
        f"  Output tokens: {summary['token_breakdown']['output_tokens']:,}",
        f"  Cached tokens: {summary['token_breakdown']['cached_tokens']:,}",
        f"  Reasoning tokens: {summary['token_breakdown']['reasoning_tokens']:,}",
        "",
        "Efficiency Metrics:",
        f"  Cost per 1K tokens: ${summary['efficiency_metrics']['cost_per_1k_tokens']:.4f}",
        f"  Cost per request: ${summary['efficiency_metrics']['cost_per_request']:.6f}",
        "=" * 50,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python cost_summary_parser.py <path_to_exec_stdout_log>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    summary = parse_cost_summary_from_log(log_path)
    print(format_cost_summary(summary))

    # Also save JSON summary
    json_path = log_path.parent / "ai_scientist_cost_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed summary saved to: {json_path}")
