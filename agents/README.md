# Agents

This directory contains all agent implementations for ResearchGym. Each agent follows the **Adapter Pattern** to integrate with the `run_agent.py` orchestrator.

## Available Agents

| Agent | Directory | Status | Description |
|-------|-----------|--------|-------------|
| **BasicAgent** | `BasicAgent/` | Production | Reference implementation using [inspect_ai](https://ukgovernmentbeis.github.io/inspect_ai/) with comprehensive file/code tools |
| **InspectionAgent** | `InspectionAgent/` | Production | Post-run verification agent that detects cheating or rule violations |
| **ML-Master** | `ML-Master/` | Functional | MCTS-based multi-agent tree search for research exploration |
| **AI-Scientist-v2** | `AI-Scientist-v2/` | Functional | Multi-worker evolutionary algorithm with code generation |
| **OpenEvolve** | `openevolve/` | Functional | Google DeepMind AlphaEvolve-style evolution for optimization |
| **ClaudeCode** | `ClaudeCode/` | Experimental | Claude Agent SDK wrapper (not runtime-tested) |
| **Codex** | `Codex/` | Experimental | OpenAI Codex CLI wrapper (partial implementation) |

## Agent Architecture

All agents follow a common adapter pattern:

```python
@dataclass
class AgentConfig:
    """Agent-specific configuration parameters."""
    task_id: str
    model: str
    # ... additional parameters

class AgentAdapter:
    """Adapter connecting agent to ResearchGym runtime."""

    def prepare_workspace(self, task_dir: Path) -> None:
        """Copy task files to agent workspace."""
        pass

    def run(self, cfg: AgentConfig, agent_root: Path, dry_run: bool) -> Observation:
        """Execute agent and return command + environment."""
        pass
```

## Quick Comparison

| Feature | BasicAgent | ML-Master | AI-Scientist | OpenEvolve |
|---------|------------|-----------|--------------|------------|
| **Framework** | inspect_ai | Custom MCTS | Multi-worker | Evolutionary |
| **File Tools** | Yes | Yes | Yes | Limited |
| **Code Execution** | bash, python | bash | bash, python | python |
| **Web Search** | Optional | No | Yes | No |
| **Async Jobs** | Yes | No | No | No |
| **Cost Tracking** | Built-in | Built-in | Built-in | Built-in |

## Detailed Documentation

- **[BasicAgent](BasicAgent/README.md)** - Full tool reference, configuration, and usage
- **[InspectionAgent](InspectionAgent/README.md)** - Post-run verification and cheating detection

## Adding New Agents

1. Create a directory under `agents/` with your agent name
2. Implement the adapter following `basic_agent_adapter.py` as template
3. Add CLI arguments in `run_agent.py` (~line 294-340)
4. Add dispatch logic in `main()` (~line 1364+)

See [Adding New Agents](../README.md#adding-new-agents) in the main README.
