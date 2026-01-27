# ResearchGym

**Evaluating LLM Agents on Open-Ended AI Research Tasks**

ResearchGym is a benchmark for evaluating the ability of LLM agents to perform autonomous AI research. Unlike code completion or bug-fixing benchmarks, ResearchGym tasks require agents to understand research problems, design novel approaches, implement solutions, and run experiments—mirroring the full cycle of ML research.

Each task provides a research problem statement, a pruned code repository (evaluation scripts, datasets, baselines—no solution code), and baseline scores to beat. Agents run autonomously for 12-24 hours with a fixed API budget and are evaluated on objective score improvements over baselines.

---

## Leaderboard

*Coming soon.*

| Agent | Model | Avg. Score | Tasks Improved | Runtime | Cost |
|-------|-------|------------|----------------|---------|------|
| — | — | — | — | — | — |

---

## Overview

ResearchGym addresses a key gap in AI evaluation: measuring an agent's ability to conduct **open-ended research** rather than solve well-defined programming tasks.

### Key Features

- **Research-Grade Tasks**: 5 test tasks sourced from ACL/ICML/ICLR 2024-2025 oral and spotlight papers, spanning vision, NLP, and RL
- **Objective Evaluation**: All tasks have quantitative metrics (accuracy, F1, mIoU, etc.) with established baselines to beat
- **Autonomous Operation**: Agents run for 12-24 hours without human intervention, making real research decisions
- **Budget Constraints**: $10 default API budget enforces efficient reasoning and experimentation
- **Multiple Runtimes**: Local execution with `uv` for development, Docker containers for reproducible production runs
- **Multi-Agent Support**: Includes BasicAgent (inspect_ai), ML-Master (MCTS), AI-Scientist-v2 (evolutionary), and OpenEvolve (AlphaEvolve-style)

---

## Tasks

### Test Set (5 Tasks)

| Task | Domain | Description | Primary Metric | Baseline |
|------|--------|-------------|----------------|----------|
| **continual-learning** | Vision | Develop scalable continual learning methods for foundation models without rehearsal | Accuracy, AAA | InfLoRA: 86.75% (CIFAR-100) |
| **cross-modal-retrieval** | Vision-Language | Address query shift in cross-modal retrieval with online adaptation | Recall@1 | EATA: 54.4% avg |
| **improving-replay-buffers** | Reinforcement Learning | Design memory systems for efficient experience replay without overfitting | Avg. Return | SynthER: 727 (Quadruped-Walk) |
| **materials-tokenization** | NLP/Science | Develop tokenization strategies preserving domain-specific material terminology | Micro-F1, Macro-F1 | PickyBPE: 75.6% |
| **time-series-explanation** | XAI | Create directionally-aware explanations for time series predictions | CPD, AUP, AUR | IG: 0.573 CPD |

### Task Structure

Each task directory contains:
```
tasks/test/<task-name>/
├── task_description.md    # Research problem, baselines, evaluation criteria
├── requirements.txt       # Task-specific dependencies
├── install.sh            # Setup script (optional)
├── grading/              # Evaluation scripts
│   ├── grade.sh
│   └── ...
├── idea_hint.txt         # Optional hints for agents
└── <baseline-code>/      # Starter code (no solutions)
```

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for production runs)
- API keys for your preferred LLM provider

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_ORG/ResearchGym.git
   cd ResearchGym
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure API keys** (create `.env` in project root):
   ```bash
   # At least one LLM provider is required
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here

   # Optional: for specific tasks or features
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   HF_TOKEN=your_huggingface_token
   SEMANTIC_SCHOLAR_API_KEY=your_key_here
   ```

### Docker Setup (Production)

Build the required Docker images:
```bash
# Base image
docker build -f environment/containers/Dockerfile.base -t researchgym-base:latest .

# Agent images
docker build -f environment/containers/Dockerfile.basic-agent -t researchgym-basic-agent:latest .
docker build -f environment/containers/Dockerfile.basic-agent-rl -t researchgym-basic-agent-rl:latest .
docker build -f environment/containers/Dockerfile.ml-master -t researchgym-ml-master:latest .
```

---

## Quick Start

### Run BasicAgent on a Task

```bash
# Quick test (6 minutes, lightweight model)
python run_agent.py tasks/test/continual-learning basic-agent \
    --runtime uv \
    --model google/gemini-2.5-flash-lite \
    --basic_hours 0.1

# Standard run (12 hours, production model)
python run_agent.py tasks/test/continual-learning basic-agent \
    --runtime uv \
    --model openai/gpt-4o \
    --basic_hours 12

# Production run with Docker
python run_agent.py tasks/test/continual-learning basic-agent \
    --runtime docker \
    --image researchgym-basic-agent:latest \
    --model anthropic/claude-sonnet-4-20250514 \
    --basic_hours 24
```

### Resume a Run

```bash
python run_agent.py tasks/test/continual-learning basic-agent \
    --runtime uv \
    --resume runs/2025-01-15/abcd1234 \
    --basic_hours 2
```

### Dry Run (Plan Only)

```bash
python run_agent.py tasks/test/continual-learning basic-agent \
    --runtime uv \
    --dry_run
```

### GPU-Enabled Tasks (RL)

```bash
python run_agent.py tasks/test/improving-replay-buffers basic-agent \
    --runtime docker \
    --image researchgym-basic-agent-rl:latest \
    --model openai/gpt-4o \
    --basic_hours 12 \
    --gpus
```

---

## Agents

ResearchGym includes multiple agent implementations. See [agents/README.md](agents/README.md) for detailed documentation.

| Agent | Description | Status |
|-------|-------------|--------|
| **[BasicAgent](agents/BasicAgent/README.md)** | Reference implementation using [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) with comprehensive tools | Production |
| **[InspectionAgent](agents/InspectionAgent/README.md)** | Post-run verification agent for detecting cheating/violations | Production |
| **ML-Master** | MCTS-based multi-agent tree search for research exploration | Functional |
| **AI-Scientist-v2** | Multi-worker evolutionary algorithm with code generation | Functional |
| **OpenEvolve** | Google DeepMind AlphaEvolve-style evolution for optimization | Functional |
| **ClaudeCode** | Claude Agent SDK wrapper | Experimental |
| **Codex** | OpenAI Codex CLI wrapper | Experimental |

### BasicAgent Tools

BasicAgent provides a comprehensive toolkit for autonomous research:

| Category | Tools |
|----------|-------|
| **Execution** | `bash()`, `python()` |
| **File Ops** | `read_file_chunk()`, `search_file()`, `write_file()`, `replace()` |
| **Background Jobs** | `start_async()`, `check_async()`, `cancel_async()` |
| **Web** (optional) | `web_search()`, `web_browser()` |
| **Control** | `end_task()` |

See [BasicAgent README](agents/BasicAgent/README.md) for full documentation.

### Running Different Agents

```bash
# ML-Master
python run_agent.py tasks/test/continual-learning ml-master \
    --runtime docker \
    --image researchgym-ml-master:latest

# AI-Scientist-v2
python run_agent.py tasks/test/continual-learning ai-scientist \
    --runtime uv \
    --ai_steps 10 \
    --ai_workers 2
```

### Inspecting Completed Runs

After runs complete, verify them for cheating/violations:

```bash
# Inspect a single run
python run_inspector.py results/continual-learning/001 --model openai/gpt-4o --budget 2.0

# Dry run (show config without executing)
python run_inspector.py results/continual-learning/001 --dry_run

# Inspect all 001-003 runs for all tasks
for task in continual-learning cross-modal-retrieval improving-replay-buffers materials-tokenization time-series-explanation; do
  for i in 001 002 003; do
    python run_inspector.py results/$task/$i --model openai/gpt-4o
  done
done
```

See [InspectionAgent README](agents/InspectionAgent/README.md) for details on verdicts, violations, and output format.

---

## Supported Models

ResearchGym uses [LiteLLM](https://docs.litellm.ai/docs/providers) for multi-provider support:

| Provider | Example Models |
|----------|---------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/o1` |
| Anthropic | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-3-haiku` |
| Google | `google/gemini-2.5-flash-lite`, `google/gemini-1.5-pro` |
| Azure | `openai/azure/gpt-4o` (requires `AZURE_OPENAI_*` env vars) |

See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for the full list.

---

## Run Artifacts

Each run produces artifacts in `runs/{YYYY-MM-DD}/{run_id}/`:

```
runs/2025-01-15/abc123/
├── workspace/input/       # Task files and agent workspace
├── logs/
│   ├── adapter.log       # Adapter-level logging
│   ├── agent.log         # Agent execution log
│   ├── exec.stdout.log   # Full command output
│   ├── transcript.json   # Conversation history
│   └── cost_summary.json # Per-turn API costs
├── metadata.json         # Run configuration
├── status.json           # Final status
└── plan.json             # Execution plan
```

---

## Adding New Tasks

1. Create a task directory:
   ```
   tasks/test/<your-task>/
   ├── task_description.md
   ├── requirements.txt
   ├── grading/
   │   └── grade.sh
   └── <baseline-code>/
   ```

2. Write `task_description.md` following the template:
   - Research Goal
   - Experimental Settings
   - Evaluation Metrics
   - Baseline Results (tables with scores to beat)
   - Hints (optional)

3. Implement `grade.sh` to evaluate submissions and output scores.

---

## Adding New Agents

1. Create an adapter in `agents/` following the pattern:
   ```python
   @dataclass
   class YourAgentConfig:
       # Agent-specific parameters
       pass

   class YourAgentAdapter:
       def prepare_workspace(self, task_dir: Path) -> None:
           """Copy task files to workspace."""
           pass

       def run(self, cfg, agent_root, dry_run) -> Observation:
           """Execute agent and return command + environment."""
           pass
   ```

2. Add CLI arguments in `run_agent.py` (~line 294-340).

3. Add dispatch logic in `main()` (~line 1364+).

See `agents/basic_agent_adapter.py` for a complete reference implementation.

---

## Architecture

```
run_agent.py          # Main entry point
├── Agent Adapter     # Prepares workspace, configures agent
├── Runtime           # UV (local) or Docker (containerized)
│   ├── uv_runner.py
│   └── docker_runner.py
└── AgenticEnv        # Gym-like interface for run management
```

**Key flows**:
1. CLI parses arguments and selects agent/runtime
2. Adapter copies task files and configures environment
3. Runtime executes agent with API key injection
4. Cost tracking enforces budget limits
5. Grading scripts evaluate final submission

---

## Citation

If you use ResearchGym in your research, please cite:

```bibtex
@misc{researchgym2025,
  title={ResearchGym: A Benchmark for Evaluating LLM Agents on Open-Ended AI Research Tasks},
  author={},
  year={2025},
  url={https://github.com/YOUR_ORG/ResearchGym}
}
```

---

## License

*License to be determined.*

---

## Acknowledgments

ResearchGym tasks are derived from research papers published at ACL, ICML, and ICLR 2024-2025. We thank the original authors for making their code and datasets available.

This project draws inspiration from:
- [PaperBench](https://github.com/openai/preparedness/tree/main/project/paperbench) (OpenAI)
- [RE-Bench](https://github.com/METR/RE-Bench) (METR)
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) (Princeton NLP)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- **New tasks**: Research problems from recent ML papers
- **New agents**: Alternative agent architectures and scaffolding
- **Evaluation**: Improved grading scripts and metrics
- **Documentation**: Tutorials, examples, and guides
