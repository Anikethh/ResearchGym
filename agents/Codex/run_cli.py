#!/usr/bin/env python3
"""CLI entry point for Codex agent runner.

Used by the adapter or manual invocation to run the Codex CLI wrapper.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import CodexConfig, PROVIDER_AZURE, PROVIDER_OPENAI, PROVIDER_SUBSCRIPTION, get_azure_credentials
from .runner import run_codex_cli


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("codex-cli")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run Codex CLI agent for ResearchGym",
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
        default=CodexConfig().model,
        help="Model to use",
    )
    parser.add_argument(
        "--time-hours",
        type=float,
        default=CodexConfig().time_hours,
        help="Time limit in hours",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=CodexConfig().budget_limit,
        help="Budget limit in USD",
    )
    parser.add_argument(
        "--approval-mode",
        type=str,
        default=CodexConfig().approval_mode,
        help="Approval mode (full-auto, auto-edit, suggest)",
    )
    parser.add_argument(
        "--sandbox-mode",
        type=str,
        default=CodexConfig().sandbox_mode,
        help="Sandbox mode (workspace-write, read-only, danger-full-access)",
    )
    parser.add_argument(
        "--resume-session",
        type=str,
        default=None,
        help="Session ID to resume with codex exec resume",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider mode (openai, azure, subscription). Default: auto-detect",
    )

    args = parser.parse_args()

    config = CodexConfig(
        model=args.model,
        time_hours=args.time_hours,
        budget_limit=args.budget_limit,
        approval_mode=args.approval_mode,
        sandbox_mode=args.sandbox_mode,
        resume_session_id=args.resume_session,
        provider=args.provider if args.provider else PROVIDER_OPENAI,
    )

    logger.info("Starting Codex CLI agent")
    logger.info(f"  Workspace: {args.workspace}")
    logger.info(f"  Log dir: {args.log_dir}")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Time: {config.time_hours}h")
    logger.info(f"  Budget: ${config.budget_limit}")

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    azure_key, azure_endpoint = get_azure_credentials()
    logger.info(f"  Provider: {config.provider}")

    # Validate credentials based on provider
    if config.provider == PROVIDER_SUBSCRIPTION:
        logger.info("  Using ChatGPT subscription (no API key required)")
    elif config.provider == PROVIDER_AZURE:
        if not config.azure_endpoint:
            logger.error("Azure provider selected but AZURE_OPENAI_ENDPOINT/AZUREAI_OPENAI_BASE_URL is not set.")
            sys.exit(1)
        if not azure_key:
            logger.error("Azure provider selected but AZURE_OPENAI_API_KEY/AZUREAI_OPENAI_API_KEY is not set.")
            sys.exit(1)
    else:
        if not openai_key:
            logger.error("OpenAI provider selected but OPENAI_API_KEY/CODEX_API_KEY is not set.")
            sys.exit(1)

    try:
        result = run_codex_cli(
            config=config,
            workspace_dir=args.workspace,
            log_dir=args.log_dir,
            timeout_seconds=config.time_hours * 3600,
        )

        logger.info(f"Run completed: {result.status}")
        logger.info(f"  Total cost: ${result.total_cost_usd:.2f}")
        if result.status == "error" and result.error:
            logger.error(f"Codex error: {result.error}")

        if result.status in ("completed", "timeout"):
            sys.exit(0)
        if result.status == "budget_exceeded":
            sys.exit(2)
        sys.exit(1)

    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
