# ClaudeCode Agent

Claude Agent SDK wrapper for ResearchGym.

## Usage

```bash
# Fresh run
python run_agent.py tasks/test/continual-learning claude-code --runtime uv --model claude-haiku-4-5 --claude_hours 0.1

# Resume (not fully working - see limitations below)
python run_agent.py tasks/test/continual-learning claude-code --runtime uv --model claude-haiku-4-5 --resume runs/2026-01-17/<run_id> --claude_hours 0.2
```

## Architecture

### Files

- `runner.py` - Main agent runner, handles SDK integration, cost tracking, graceful shutdown
- `config.py` - Configuration dataclass for ClaudeCode runs
- `cost_tracker.py` - Tracks API costs, time limits, budget enforcement
- `hooks.py` - Stop hook (continue messages), PreToolUse hook (URL blocking)
- `messages.py` - System prompts and status messages
- `cli.py` - CLI entry point for the agent

### Key Features

1. **Cost Tracking**: Real usage from ResultMessage via `interrupt()`, estimates during run
2. **Time Limits**: Active time tracking (excludes retry wait time), triggers interrupt on limit
3. **Budget Limits**: Hard budget cap with BudgetExceeded exception
4. **URL Blocking**: PreToolUse hook blocks access to specified URLs (paper links, original repos)
5. **Resume**: Transcript seeding from previous run's transcript.json

### Session Storage

Claude Code stores sessions at:
```
~/.claude/projects/<encoded-cwd>/<session_id>.jsonl
```

Path encoding: `E:\ResearchGym\runs\...\workspace\input` → `E--ResearchGym-runs-...-workspace-input`
- Replace `\`, `/`, `:` with `-`
- No dash collapsing (important!)

### Run Artifacts

```
runs/<date>/<run_id>/
├── logs/
│   ├── transcript.json    # All SDK messages serialized
│   ├── cost_summary.json  # Cost tracking data
│   ├── session.json       # Session ID for resume
│   ├── exec.stdout.log
│   └── exec.stderr.log
├── workspace/
│   └── input/             # Task files + agent work
└── plan.json
```

---

## Resume Support - Implementation Details

### What We Implemented

1. **Session ID Discovery** (`discover_session_id_from_claude_projects`)
   - Encodes workspace path to match Claude's folder naming
   - Finds `.jsonl` file in `~/.claude/projects/<encoded-path>/`
   - Returns filename (without extension) as session_id
   - Called at end of run AND before graceful wrap-up

2. **Session File Copying** (`_copy_claude_session_file` in run_agent.py)
   - When resuming, workspace path changes → different Claude projects folder
   - Copies session `.jsonl` from old workspace folder to new workspace folder
   - Required because SDK looks for session in cwd-specific folder

3. **Resume Usage Collection** (`_collect_claude_code_resume_usage` in run_agent.py)
   - Reads `cost_summary.json` from previous run
   - Extracts time used, cost spent
   - Used to calculate remaining budget/time

4. **Graceful Wrap-up** (runner.py lines 599-628)
   - When `is_over_time_limit()` returns True:
     - Discovers session ID if not already captured
     - Sends wrap-up query: "Time limit reached. Commit any uncommitted changes and stop."
     - 60 second timeout for wrap-up
     - Logs "Graceful wrap-up completed" on success

   **Note on Transcript Seeding:** Graceful wrap-up can be harmful for resume because it adds intervention messages ("Time limit reached. Commit...") and agent responses to the transcript. When resuming, these get included in the context, creating unnatural conversation flow. The agent sees itself "stopping" then being asked to continue. Consider disabling for transcript-based resume or filtering out wrap-up messages.

5. **Resume Prompt** (runner.py line 358)
   - On resume: "Session resumed. You have X more hours. Continue where you left off."
   - Minimal to avoid confusion since agent has full context

6. **Time Calculation on Resume**
   - `remaining = config_time - already_elapsed`
   - Pass total desired time (e.g., 0.1h), system calculates leftover

---

## Resume Support - Known Limitations

### Why Resume Fails

Per [GitHub Issue #12730](https://github.com/anthropics/claude-code/issues/12730):
- Sessions terminated abruptly don't sync to Anthropic's backend
- Makes them non-resumable even with correct session ID

**Observed behavior:**
```
Resuming session 9f874ffd-... from previous run
Captured session ID: 48d12bc6-...  ← NEW session created!
Fatal error: Command failed with exit code 1
```

The SDK ignores the resume session ID and creates a new session, then crashes.

### Approaches Tried

1. **MCP Finish Tool**
   - Goal: Add custom `finish` tool agent can call to end gracefully
   - Implementation: `@tool` decorator + `create_sdk_mcp_server`
   - Result: `CLIConnectionError: ProcessTransport is not ready for writing`
   - SDK v0.1.20 may not support in-process MCP servers properly

2. **Stop Hook for Graceful End**
   - Goal: Inject wrap-up message when time approaching
   - Problem: Stop hook only fires when agent voluntarily stops (between turns)
   - Can't inject messages during continuous tool execution
   - Time limit check happens in runner message loop, not hook

3. **Graceful Wrap-up Query**
   - Goal: Send separate query asking agent to commit before exit
   - Implementation: Works! Agent receives message and can commit
   - Problem: Doesn't fix resume - session still doesn't sync to backend

4. **Session File Copying**
   - Goal: Copy session from old workspace to new workspace
   - Implementation: Works! File is copied correctly
   - Problem: SDK still can't resume - backend issue, not local file issue

### Current State

| Feature | Status |
|---------|--------|
| Fresh runs | ✅ Working |
| Cost tracking | ✅ Working |
| Time limits | ✅ Working |
| Graceful wrap-up | ✅ Working (agent gets message) |
| Session ID saved | ✅ Working |
| Resume (transcript seeding) | ✅ Implemented |

### Resume Implementation (Transcript Seeding)

Since SDK resume doesn't work reliably, we use transcript seeding instead:

1. **On resume, run_agent.py:**
   - Copies `transcript.json` from previous run to `previous_transcript.json`
   - Still passes session_id (used as a flag to trigger resume logic)

2. **runner.py `build_resume_context()`:**
   - Parses the previous transcript
   - Extracts assistant text, tool calls, and results
   - Truncates to 50k chars if needed
   - Returns formatted context string

3. **Resume prompt includes:**
   - Full task description (same as fresh run)
   - Previous session context from transcript
   - Remaining time information

This gives the agent full context to continue, without relying on SDK resume.

### Future Improvements

1. **Wait for SDK Fix**
   - Monitor GitHub issue #12730
   - SDK may add proper session sync in future versions

2. **Smart Context Summarization**
   - Instead of truncating, summarize long transcripts
   - Prioritize recent messages and key decisions

3. **Filter Wrap-up Messages from Resume Context**
   - Graceful wrap-up adds "Time limit reached" messages to transcript
   - These create unnatural resume context (agent sees itself stopping)
   - Could filter out wrap-up messages when building resume context
   - Or disable graceful wrap-up entirely when transcript seeding is used

---

## Cost Tracking

### How It Works

Uses `ClaudeSDKClient` with `interrupt()` to get real usage data:

1. **During run**: Estimated costs (chars/4) for real-time visibility
2. **On time limit**: Call `interrupt()` to trigger `ResultMessage`
3. **ResultMessage**: Contains actual `total_cost_usd` and `usage` dict

```python
# ResultMessage after interrupt():
ResultMessage(
    subtype="error_during_execution",
    total_cost_usd=0.201,
    usage={
        "input_tokens": 75,
        "output_tokens": 11515,
        "cache_read_input_tokens": 845418,
        "cache_creation_input_tokens": 26636,
    }
)
```

### What We Get

| Metric | Source |
|--------|--------|
| input_tokens | Real (from ResultMessage) |
| output_tokens | Real (from ResultMessage) |
| cache_read | Real (from ResultMessage) |
| cache_write | Real (from ResultMessage) |
| total_cost_usd | Real (from ResultMessage) |

### Limitations

- **No per-message usage**: SDK doesn't expose `.usage` on `AssistantMessage`
- **Real data only at end**: Must wait for `ResultMessage` after interrupt
- **Estimated during run**: Uses chars/4 approximation for real-time display
