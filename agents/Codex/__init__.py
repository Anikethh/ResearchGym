"""Codex CLI agent integration for ResearchGym.

Wraps OpenAI's Codex CLI (codex-cli) for autonomous operation.

Supports both OpenAI API and Azure OpenAI:
- OpenAI: Set OPENAI_API_KEY
- Azure: Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT

Note: Codex CLI currently only supports API mode, not ChatGPT Plus subscription
for automated/headless operation.
"""

from .adapter import CodexAdapter
from .config import (
    CodexConfig,
    PROVIDER_OPENAI,
    PROVIDER_AZURE,
    detect_provider,
)

__all__ = [
    "CodexAdapter",
    "CodexConfig",
    "PROVIDER_OPENAI",
    "PROVIDER_AZURE",
    "detect_provider",
]
