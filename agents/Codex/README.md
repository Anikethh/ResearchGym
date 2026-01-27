Codex Agent (Codex CLI)
=======================

This adapter runs OpenAI's Codex CLI in non-interactive mode and integrates
with ResearchGym's cost, time, resume, and sandbox controls.

Prerequisites
-------------
- Install Codex CLI: `npm install -g @openai/codex`
- Set an API key:
  - OpenAI: `OPENAI_API_KEY`
  - Azure: `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`

Basic Usage
-----------
From the repo root:

```
python run_agent.py tasks/test/continual-learning codex --runtime uv --codex_hours 0.25
```

Optional Overrides
------------------
- Model: `--codex_model gpt-5-codex`
- Budget: `--budget_limit 5.0`
- Hours: `--codex_hours 24`
- Reasoning effort: `--codex_reasoning_effort high`
- Sandbox: use `sandbox_mode` in `agents/Codex/config.py` or pass `--sandbox-mode` to the CLI entry point.

Reasoning Effort
----------------
Controls how much "thinking" time the model spends on responses. Higher effort = better
quality but slower and more expensive. Available values:

| Value    | Description                                              |
|----------|----------------------------------------------------------|
| minimal  | Fastest, least reasoning depth                           |
| low      | Quick responses with basic reasoning                     |
| medium   | Balanced (default for most models)                       |
| high     | Deeper reasoning, good for complex tasks                 |
| xhigh    | Maximum reasoning depth (default)                        |

Example with xhigh reasoning effort for 24 hours:
```
python run_agent.py tasks/test/continual-learning codex --runtime uv \
    --codex_hours 24 --budget_limit 20 --codex_reasoning_effort high
```

For non-latency-sensitive research tasks, `high` or `xhigh` is recommended.

Outputs
-------
Logs and summaries are written under `runs/<date>/<run_id>/logs/`:
- `codex_output.jsonl` (JSONL stream)
- `cost_summary.json` (token/cost/time summary)
- `blocked_violations.json` (blocked URL/content mentions, when configured)

Resume
------
Resume via `--resume` uses transcript seeding by default and will copy the
previous JSONL transcript into the new run logs.
