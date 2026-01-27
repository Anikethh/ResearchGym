"""Configuration for Codex CLI agent runs.

Codex CLI supports multiple authentication modes:
- API mode: Uses OPENAI_API_KEY or AZURE_OPENAI_API_KEY
- Subscription mode: Uses ChatGPT Plus/Pro subscription (requires browser login)

For Azure OpenAI, configure ~/.codex/config.toml:
    [model_providers.azure]
    base_url = "https://YOUR_PROJECT.openai.azure.com/openai"
    env_key = "AZURE_OPENAI_API_KEY"
    query_params = { api-version = "2025-04-01-preview" }
    wire_api = "responses"
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default model for Codex CLI
DEFAULT_MODEL = "gpt-5.2-codex"

# Default limits
DEFAULT_TIME_HOURS = 12
DEFAULT_BUDGET_LIMIT = 10.0  # USD (not directly enforced by Codex CLI)

# Approval modes
APPROVAL_MODE_FULL_AUTO = "full-auto"  # No human approval needed
APPROVAL_MODE_AUTO_EDIT = "auto-edit"  # Auto-approve file edits
APPROVAL_MODE_SUGGEST = "suggest"  # Suggest changes only

# Sandbox modes
SANDBOX_WORKSPACE_WRITE = "workspace-write"
SANDBOX_READ_ONLY = "read-only"
SANDBOX_DANGER_FULL_ACCESS = "danger-full-access"
SANDBOX_BYPASS = "bypass"  # Full bypass via --dangerously-bypass-approvals-and-sandbox

# Provider modes
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE = "azure"
PROVIDER_SUBSCRIPTION = "subscription"  # ChatGPT Plus/Pro subscription (no API key needed)

# Reasoning effort levels (for o-series and reasoning models)
# See: https://developers.openai.com/codex/config-advanced/
REASONING_EFFORT_MINIMAL = "minimal"
REASONING_EFFORT_LOW = "low"
REASONING_EFFORT_MEDIUM = "medium"  # Default for most models
REASONING_EFFORT_HIGH = "high"
REASONING_EFFORT_XHIGH = "xhigh"  # Extra high - for non-latency-sensitive tasks

VALID_REASONING_EFFORTS = [
    REASONING_EFFORT_MINIMAL,
    REASONING_EFFORT_LOW,
    REASONING_EFFORT_MEDIUM,
    REASONING_EFFORT_HIGH,
    REASONING_EFFORT_XHIGH,
]

DEFAULT_REASONING_EFFORT = REASONING_EFFORT_XHIGH


@dataclass
class CodexConfig:
    """Configuration for a Codex CLI agent run.

    Attributes:
        model: Model identifier/deployment name (e.g., o3, o4-mini, gpt-5)
        provider: Model provider (openai, azure)
        time_hours: Maximum runtime in hours (external enforcement)
        budget_limit: Maximum cost in USD (external enforcement)
        approval_mode: Approval policy (full-auto, auto-edit, suggest)
        reasoning_effort: Reasoning effort level (minimal, low, medium, high, xhigh)
        blocked_urls: URLs to block (not directly supported by Codex CLI)
        writable_roots: Directories the agent can write to
        azure_endpoint: Azure OpenAI endpoint (if using Azure)
        azure_api_version: Azure API version (default: 2025-04-01-preview)
        config_path: Path to custom config.toml (optional)
        resume_session_id: Session ID to resume (optional)
        inherited_cost_path: Path to previous cost_summary.json for resume (optional)
        env: Additional environment variables
    """

    model: str = DEFAULT_MODEL
    provider: str = PROVIDER_OPENAI
    time_hours: float = DEFAULT_TIME_HOURS
    budget_limit: float = DEFAULT_BUDGET_LIMIT
    approval_mode: str = APPROVAL_MODE_FULL_AUTO
    sandbox_mode: str = SANDBOX_WORKSPACE_WRITE
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    web_search: bool = True  # Enable web search tool
    blocked_urls: list[str] = field(default_factory=list)
    writable_roots: list[str] = field(default_factory=list)
    azure_endpoint: str | None = None
    azure_api_version: str = ""
    config_path: Path | None = None
    resume_session_id: str | None = None
    inherited_cost_path: Path | None = None  # For transcript-seeded resumes
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect Azure provider from environment."""
        if self.provider == PROVIDER_OPENAI:
            # Auto-switch to Azure if Azure env vars are set but OpenAI isn't
            if not os.environ.get("OPENAI_API_KEY"):
                if (
                    os.environ.get("AZURE_OPENAI_API_KEY")
                    or os.environ.get("AZURE_OPENAI_ENDPOINT")
                    or os.environ.get("AZUREAI_OPENAI_API_KEY")
                    or os.environ.get("AZUREAI_OPENAI_BASE_URL")
                ):
                    self.provider = PROVIDER_AZURE

        if self.provider == PROVIDER_AZURE and not self.azure_endpoint:
            self.azure_endpoint = (
                os.environ.get("AZURE_OPENAI_ENDPOINT")
                or os.environ.get("AZUREAI_OPENAI_BASE_URL")
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "provider": self.provider,
            "time_hours": self.time_hours,
            "budget_limit": self.budget_limit,
            "approval_mode": self.approval_mode,
            "sandbox_mode": self.sandbox_mode,
            "reasoning_effort": self.reasoning_effort,
            "blocked_urls": self.blocked_urls,
            "writable_roots": self.writable_roots,
            "azure_endpoint": self.azure_endpoint,
            "azure_api_version": self.azure_api_version,
            "resume_session_id": self.resume_session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodexConfig":
        """Create config from dictionary."""
        return cls(
            model=data.get("model", DEFAULT_MODEL),
            provider=data.get("provider", PROVIDER_OPENAI),
            time_hours=data.get("time_hours", DEFAULT_TIME_HOURS),
            budget_limit=data.get("budget_limit", DEFAULT_BUDGET_LIMIT),
            approval_mode=data.get("approval_mode", APPROVAL_MODE_FULL_AUTO),
            sandbox_mode=data.get("sandbox_mode", SANDBOX_WORKSPACE_WRITE),
            reasoning_effort=data.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
            blocked_urls=data.get("blocked_urls", []),
            writable_roots=data.get("writable_roots", []),
            azure_endpoint=data.get("azure_endpoint"),
            azure_api_version=data.get("azure_api_version", ""),
            config_path=Path(data["config_path"]) if data.get("config_path") else None,
            resume_session_id=data.get("resume_session_id"),
            env=data.get("env", {}),
        )


def load_blocked_urls(task_dir: Path) -> list[str]:
    """Load blocked URLs from task's blocked_urls.yaml.

    Args:
        task_dir: Path to task directory

    Returns:
        List of blocked URL patterns
    """
    blocklist_path = task_dir / "blocked_urls.yaml"

    if not blocklist_path.exists():
        return []

    with open(blocklist_path) as f:
        data = yaml.safe_load(f)

    return data.get("blocked_urls", []) if data else []


def get_openai_api_key() -> str | None:
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY")


def get_azure_credentials() -> tuple[str | None, str | None]:
    """Get Azure OpenAI credentials from environment.

    Returns:
        Tuple of (api_key, endpoint)
    """
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZUREAI_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZUREAI_OPENAI_BASE_URL")
    return api_key, endpoint


def generate_codex_config_toml(
    config: CodexConfig,
    output_path: Path,
) -> Path:
    """Generate a config.toml file for Codex CLI.

    This is useful for Azure OpenAI or custom provider configuration.

    Args:
        config: CodexConfig with provider settings
        output_path: Where to write the config.toml

    Returns:
        Path to the generated config file
    """
    lines = []

    # Model setting
    lines.append(f'model = "{config.model}"')

    if config.provider == PROVIDER_AZURE:
        lines.append(f'model_provider = "azure"')

    # Enable experimental Windows sandbox for proper sandbox support on Windows
    lines.append('')
    lines.append('[features]')
    lines.append('experimental_windows_sandbox = true')

    if config.provider == PROVIDER_AZURE:
        lines.append('')
        lines.append('[model_providers.azure]')
        lines.append('name = "Azure OpenAI"')

        # Azure endpoint - ensure /openai prefix exists, but don't double-append
        endpoint = config.azure_endpoint or ""
        if endpoint:
            if "/openai" not in endpoint:
                if endpoint.endswith("/"):
                    endpoint = endpoint + "openai"
                else:
                    endpoint = endpoint + "/openai"

        lines.append(f'base_url = "{endpoint}"')
        lines.append('env_key = "AZURE_OPENAI_API_KEY"')
        if config.azure_api_version:
            lines.append(f'query_params = {{ api-version = "{config.azure_api_version}" }}')
        lines.append('wire_api = "responses"')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')

    return output_path


def detect_provider() -> str:
    """Auto-detect which provider to use based on environment.

    Returns:
        Provider string (openai or azure)
    """
    # Check for Azure first (more specific)
    if (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZUREAI_OPENAI_API_KEY")
    ) and (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZUREAI_OPENAI_BASE_URL")
    ):
        return PROVIDER_AZURE

    # Fall back to OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return PROVIDER_OPENAI

    # Default to OpenAI (will fail later if no key)
    return PROVIDER_OPENAI
