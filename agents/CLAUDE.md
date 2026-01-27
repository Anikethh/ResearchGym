# Coding Preferences

- No fallback/default values that hide errors. Surface real errors explicitly.
- Verify external data (pricing, API responses) against official sources before using.

# Project Structure

- `ClaudeCode/` - Claude Agent SDK wrapper (claude-agent-sdk)
- `Codex/` - OpenAI Codex CLI wrapper (@openai/codex)
- `BasicAgent/` - Reference implementation in main ResearchGym

# Key Files

- `ClaudeCode/cost_tracker.py` - Claude model pricing (MODEL_PRICING dict)
- `Codex/runner.py` - OpenAI model pricing (OPENAI_MODEL_PRICING dict)

# Testing

Not yet established. SDK integrations are untested pending runtime validation.

# Known Limitations

## PyTorch on Windows

- Windows CUDA wheels require explicit `+cu124` suffix: `torch==2.5.1+cu124`, not just `torch==2.5.1`
- The `--extra-index-url` alone does not force CUDA version selection on Windows
- Must also match sympy version requirement (torch 2.5.1+cu124 requires sympy==1.13.1)

## Codex Cost Estimation

- File write tokens are NOT tracked in char-based estimates. The `file_change` events in Codex JSONL output only contain file paths, not the actual content. The tokens used to generate file content don't appear in any tracked field.
- Actual token usage is only recorded when `turn.completed` events occur. If a turn fails or is interrupted, we only have char-based estimates which undercount file writes.
- This means cost estimates may be lower than actual costs for file-heavy workloads.
