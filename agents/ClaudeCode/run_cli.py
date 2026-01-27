#!/usr/bin/env python3
"""CLI entry point for Claude Code agent runner.

Used by the adapter to run the agent in a subprocess.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("claude-code-cli")


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude Code agent for ResearchGym"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Path to workspace directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to log directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-5-20251101",
        help="Model to use",
    )
    parser.add_argument(
        "--time-hours",
        type=float,
        default=12.0,
        help="Time limit in hours",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=10.0,
        help="Budget limit in USD",
    )
    parser.add_argument(
        "--extended-continue",
        action="store_true",
        help="Use extended continue messages",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API key authentication (vs subscription)",
    )
    parser.add_argument(
        "--blocked-urls",
        type=str,
        default="[]",
        help="JSON array of blocked URL patterns (deprecated, use --blocked-urls-file)",
    )
    parser.add_argument(
        "--blocked-urls-file",
        type=str,
        default=None,
        help="Path to JSON file containing blocked URL patterns",
    )
    parser.add_argument(
        "--resume-session",
        type=str,
        default=None,
        help="Session ID to resume",
    )

    args = parser.parse_args()

    # Parse blocked URLs - prefer file over inline JSON (avoids Windows shell quote issues)
    blocked_urls = []
    if args.blocked_urls_file:
        try:
            with open(args.blocked_urls_file) as f:
                blocked_urls = json.load(f)
            logger.info(f"Loaded {len(blocked_urls)} blocked URLs from {args.blocked_urls_file}")
        except Exception as e:
            logger.error(f"Failed to load blocked-urls-file: {e}")
    elif args.blocked_urls != "[]":
        try:
            blocked_urls = json.loads(args.blocked_urls)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for blocked-urls: {args.blocked_urls}")

    # Import here to avoid import errors if SDK not installed
    from .config import ClaudeCodeConfig
    from .runner import run_agent_sync

    config = ClaudeCodeConfig(
        model=args.model,
        time_hours=args.time_hours,
        budget_limit=args.budget_limit,
        blocked_urls=blocked_urls,
        use_api=args.use_api,
        extended_continue=args.extended_continue,
    )

    logger.info("Starting Claude Code agent")
    logger.info(f"  Workspace: {args.workspace}")
    logger.info(f"  Log dir: {args.log_dir}")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Time: {config.time_hours}h")
    logger.info(f"  Budget: ${config.budget_limit}")
    logger.info(f"  Blocked URLs: {len(blocked_urls)}")

    try:
        result = run_agent_sync(
            config=config,
            workspace_dir=args.workspace,
            log_dir=args.log_dir,
            resume_session_id=args.resume_session,
        )

        logger.info(f"Run completed: {result['status']}")
        logger.info(f"  Total cost: ${result['total_cost_usd']:.2f}")
        logger.info(f"  Total turns: {result['total_turns']}")

        # Exit with appropriate code
        if result["status"] in ("completed", "time_limit"):
            sys.exit(0)
        elif result["status"] == "budget_exceeded":
            sys.exit(2)
        else:
            sys.exit(1)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure claude-agent-sdk is installed")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
