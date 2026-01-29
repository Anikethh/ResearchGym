# RGAgent

The reference agent implementation for ResearchGym, built on the [inspect_ai](https://ukgovernmentbeis.github.io/inspect_ai/) framework from UK AI Safety Institute.

## Overview

RGAgent is designed for autonomous AI research tasks. It provides:
- Comprehensive file and code manipulation tools
- Background job execution for long-running experiments
- Optional web search capabilities
- Built-in cost tracking with budget enforcement
- Two solver modes: iterative (simple) and plus (advanced)

## Tools

### Code Execution

#### `bash(command: str)`
Execute shell commands in a sandboxed environment.

```python
# Run a Python script
bash("python train.py --epochs 10")

# Install dependencies
bash("pip install numpy pandas")

# Check GPU availability
bash("nvidia-smi")
```

- **Working directory**: `CODE_DIR` (task workspace)
- **Environment**: PATH includes venv, UTF-8 encoding, no interactive prompts
- **Platform support**: Unix and Windows (Git Bash)

#### `python(code: str)`
Execute Python code directly.

```python
python("""
import numpy as np
result = np.mean([1, 2, 3, 4, 5])
print(f"Mean: {result}")
""")
```

- Returns stdout and stderr combined
- Runs within `CODE_DIR` context

### File Operations

#### `read_file_chunk(file: str, start_line: int, max_lines: int)`
Read a portion of a file with line numbers.

```python
# Read first 50 lines
read_file_chunk("train.py", start_line=1, max_lines=50)

# Read lines 100-150
read_file_chunk("model.py", start_line=100, max_lines=50)
```

- **Max lines per call**: 50
- **Output**: Numbered lines with total file length
- **Paths**: Relative to `CODE_DIR` or absolute

#### `search_file(file: str, query: str, context_lines: int, max_matches: int, page: int)`
Search for text within a file.

```python
# Find all occurrences of "learning_rate"
search_file("config.py", query="learning_rate", context_lines=2, max_matches=5)
```

- **Case-insensitive**: Yes
- **Context**: Shows lines before/after each match
- **Pagination**: Use `page` parameter for many matches

#### `write_file(path: str, content: str)`
Create or overwrite a file.

```python
write_file("experiment.py", """
import torch
model = torch.nn.Linear(10, 1)
""")
```

- **Security**: Path must be within `CODE_DIR`
- **Behavior**: Overwrites existing files, creates parent directories
- **Availability**: Enabled by default for non-OpenAI models

#### `replace(path: str, old_string: str, new_string: str, expected_replacements: int)`
Make precise text replacements.

```python
# Change learning rate
replace("config.py",
    old_string="learning_rate = 0.001",
    new_string="learning_rate = 0.0001",
    expected_replacements=1)
```

- **Matching**: Exact string match required
- **Validation**: Fails if `old_string` not found or count mismatch
- **Create files**: Pass empty `old_string` to create new file
- **Availability**: Enabled by default for non-OpenAI models

### Background Jobs

For long-running experiments (training, evaluation), use async tools to avoid blocking.

#### `start_async(command: str, workdir: str)`
Launch a background process.

```python
result = start_async("python train.py --epochs 100", workdir=".")
# Returns: {"job_id": "abc123", "pid": 12345, "log_file": "async_jobs/abc123/stdout_stderr.log"}
```

#### `check_async(job_id: str, tail_lines: int, sleep_minutes: float)`
Check job status and view recent output.

```python
check_async("abc123", tail_lines=50, sleep_minutes=5)
# Returns: {"status": "running", "returncode": null, "log_tail": "..."}
```

- **sleep_minutes**: Cooperative delay before checking (for throttling)

#### `cancel_async(job_id: str)`
Terminate a running job.

```python
cancel_async("abc123")
# Returns: {"status": "terminated", "log_file": "..."}
```

**Job storage**: Metadata and logs in `CODE_DIR/async_jobs/{job_id}/`

### Web Tools (Optional)

#### `web_search(query: str)`
Search the web using Exa or Google Custom Search.

```python
web_search("transformer attention mechanism paper")
```

**Enable**: Set `USE_EXA_SEARCH=true` or `USE_GOOGLE_WEB_SEARCH=true`

#### `web_browser(url: str)`
Fetch and read web page content.

**Enable**: Set `ENABLE_BROWSER=true`
**Disable**: Set `DISABLE_BROWSER=true` (default)

### Task Control

#### `end_task(submission: str)`
Signal task completion and submit results.

```python
end_task("Final accuracy: 92.5% on test set. See results/final_metrics.json")
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_DIR` | `.` | Agent workspace root |
| `AGENT_DIR` | `.` | RGAgent installation directory |
| `MAX_TIME_IN_HOURS` | `0.25` | Maximum runtime |
| `MODEL` | `gpt-4o-mini` | LLM model identifier |
| `ITERATIVE_AGENT` | `false` | Use simple iterative mode |
| `DISALLOW_SUBMIT` | `false` | Prevent early task termination |

### Tool Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RG_ENABLE_ASYNC` | `true` | Enable background job tools |
| `RG_ENABLE_REPLACE` | varies | Override replace tool availability |
| `RG_ENABLE_WRITE_FILE` | varies | Override write_file availability |
| `USE_EXA_SEARCH` | `false` | Enable Exa web search |
| `USE_GOOGLE_WEB_SEARCH` | `false` | Enable Google web search |
| `ENABLE_BROWSER` | `false` | Enable web browser tool |

### Cost & Budget

| Variable | Default | Description |
|----------|---------|-------------|
| `RG_BUDGET_LIMIT` | `0` (unlimited) | Cost budget in USD |
| `RG_LOG_DIR` | `./logs` | Directory for cost logs |

### Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `RG_IDEA_HINT` | `false` | Provide idea hint to agent |
| `RG_EXTENDED_CONTINUE` | `false` | Use extended continue message |
| `RG_INSTRUCTIONS_FILE` | - | Override instruction file path |
| `RG_LOG_BUFFER` | `1` | Log buffer flush threshold |

## Solver Modes

### Iterative Mode (`ITERATIVE_AGENT=true`)
Simple single-tool-per-turn agent for basic tasks.
- Generates response, calls one tool, receives result, repeats
- Lower token usage, suitable for straightforward tasks
- Tools: bash, read_file_chunk, replace (optional), write_file (optional), async tools

### Plus Mode (default)
Advanced multi-turn agent with research-focused enhancements.
- Handoff mechanism for long conversations (context pruning with summary)
- Multi-hypothesis evaluation framework
- Literature survey capabilities
- GPU detection and optimization hints
- Tools: All iterative tools + python, search_file, web tools

## Cost Tracking

RGAgent includes built-in cost tracking with per-model pricing:

| Model | Input ($/1M) | Output ($/1M) | Cached ($/1M) |
|-------|--------------|---------------|---------------|
| GPT-5 | $1.25 | $10.00 | $0.125 |
| GPT-4o | $2.50 | $10.00 | $1.25 |
| Claude Sonnet | $3.00 | $15.00 | $0.30 |
| Gemini 2.5 Flash | $0.10 | $0.40 | $0.025 |

Cost logs are written to `RG_LOG_DIR/cost_summary.json`.

## Sandbox Environment

- **Type**: Local sandbox via inspect_ai (Docker optional)
- **Working directory**: `CODE_DIR` (task workspace)
- **Path security**: All file operations bounded to `CODE_DIR`
- **Process isolation**: Unix signals / Windows process groups
- **Environment setup**:
  - `DEBIAN_FRONTEND=noninteractive`
  - `GIT_TERMINAL_PROMPT=0`
  - `PYTHONIOENCODING=utf-8`
  - Virtual environment PATH integration

## File Structure

```
RGAgent/
├── start.py                    # Entry point
├── _basic_agent_iterative.py   # Simple solver
├── _basic_agent_plus.py        # Advanced solver
├── _execute.py                 # bash/python tools
├── _file_reader.py             # read/search tools
├── _write_file.py              # write_file tool
├── _replace.py                 # replace tool
├── _async_jobs.py              # background job tools
├── cost_tracker.py             # Cost calculation
├── cost_logger.py              # Cost logging
├── utils.py                    # Utilities
├── templates.py                # Prompt templates
├── apply_patch.py              # Patch utility
└── inspect_ai/                 # Vendored framework
```

## Usage Examples

### Quick Test Run
```bash
python run_agent.py tasks/test/continual-learning rg-agent \
    --runtime uv \
    --model google/gemini-2.5-flash-lite \
    --basic_hours 0.1
```

### Production Run
```bash
python run_agent.py tasks/test/continual-learning rg-agent \
    --runtime docker \
    --image researchgym-rg-agent:latest \
    --model openai/gpt-4o \
    --basic_hours 12
```

### With Web Search
```bash
export EXA_API_KEY=your_key_here
python run_agent.py tasks/test/materials-tokenization rg-agent \
    --runtime uv \
    --model anthropic/claude-sonnet-4-20250514 \
    --basic_hours 6
```

## Troubleshooting

### Tool not available
Check environment variables for tool enablement. Some tools are disabled by default for certain providers.

### Context length exceeded
RGAgent Plus mode handles this with automatic pruning and handoff. Consider using iterative mode for simpler tasks.

### Async jobs not working
Verify `RG_ENABLE_ASYNC=true` and check `async_jobs/` directory permissions.

### Cost budget exceeded
Agent will stop when `RG_BUDGET_LIMIT` is reached. Check `logs/cost_summary.json` for breakdown.
