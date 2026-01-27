#!/usr/bin/env python3
"""
Run the InspectionAgent to verify completed ResearchGym runs.

Usage:
    python run_inspector.py results/continual-learning/001 --model openai/azure/gpt-5
    python run_inspector.py results/*/001 results/*/002 results/*/003 --model openai/azure/gpt-5
    python run_inspector.py results/continual-learning/* --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure package import works when running this file directly
CURRENT_FILE = Path(__file__).resolve()
PKG_ROOT = CURRENT_FILE.parent
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ResearchGym.agents.inspection_agent_adapter import InspectionAgentAdapter, InspectionAgentConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a completed ResearchGym run for cheating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_inspector.py runs/2025-01-15/abc123
    python run_inspector.py runs/2025-01-15/abc123 --model google/gemini-2.0-flash
    python run_inspector.py runs/2025-01-15/abc123 --dry_run
        """,
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help="Path(s) to completed run directories",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/azure/gpt-5",
        help="Model to use for inspection (default: openai/azure/gpt-5)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="Budget limit in dollars (default: 2.0)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if inspection_report.json exists",
    )

    args = parser.parse_args()
    agent_root = Path(__file__).parent / "agents" / "InspectionAgent"
    if not agent_root.exists():
        print(f"Error: InspectionAgent not found at {agent_root}")
        return 1

    results = []
    total_cost = 0.0
    for run_dir in args.run_dirs:
        rc, cost = inspect_run(run_dir.resolve(), args, agent_root)
        results.append((run_dir, rc))
        total_cost += cost

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for run_dir, rc in results:
            status = {0: "PASS", 1: "NEEDS_REVIEW", 2: "SUSPICIOUS", -1: "SKIPPED", -2: "ERROR"}.get(rc, "UNKNOWN")
            print(f"  {run_dir.name}: {status}")
        print(f"\nTotal cost: ${total_cost:.4f}")

    return max(r[1] for r in results) if results else 0


def inspect_run(run_dir: Path, args, agent_root: Path) -> tuple[int, float]:
    """Inspect a single run. Returns (exit_code, cost)."""
    # Validate run directory
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return -2, 0.0

    if not (run_dir / "metadata.json").exists():
        print(f"Skipping {run_dir.name}: missing metadata.json")
        return -1, 0.0

    # Create adapter
    adapter = InspectionAgentAdapter(run_dir)

    # Check if already inspected
    report_path = adapter.output_dir / "inspection_report.json"
    if report_path.exists() and not args.force:
        report = json.loads(report_path.read_text())
        cost = report.get("cost", {}).get("cost_usd", 0.0)
        print(f"Already inspected: {report.get('verdict')} (${cost:.4f}) (use --force to re-run)")
        return {
            "PASS": 0, "SUSPICIOUS": 2, "NEEDS_REVIEW": 1
        }.get(report.get("verdict"), 1), cost

    # Resolve task directory from metadata
    try:
        task_dir = adapter.resolve_task_dir()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return -2, 0.0

    print(f"Task: {task_dir.name}")
    print(f"Output: {adapter.output_dir}")
    print(f"Model: {args.model}")

    # Create config
    cfg = InspectionAgentConfig(
        run_dir=run_dir,
        task_dir=task_dir,
        model=args.model,
        budget_limit=args.budget,
    )

    # Run inspection
    if args.dry_run:
        print("  [dry run - skipped]")
        return -1, 0.0

    print("Running...")
    result = adapter.run(cfg, agent_root, dry_run=False)

    print(f"Verdict: {result.verdict} (confidence: {result.confidence:.2f})")
    if result.cost_usd > 0:
        print(f"Cost: ${result.cost_usd:.4f} ({result.tokens:,} tokens)")
    if result.violations:
        for v in result.violations:
            print(f"  - [{v.get('severity')}] {v.get('type')}: {v.get('description', '')[:80]}")

    if result.verdict == "PASS":
        return 0, result.cost_usd
    elif result.verdict == "SUSPICIOUS":
        return 2, result.cost_usd
    else:
        return 1, result.cost_usd


if __name__ == "__main__":
    sys.exit(main())
