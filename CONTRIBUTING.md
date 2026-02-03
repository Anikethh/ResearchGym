# Contributing to ResearchGym

We welcome contributions! This guide covers how to contribute code, tasks, and documentation.

## Quick Start

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes and test
4. Submit a pull request

## Code Style

- Python 3.12+ with type hints
- 4-space indentation
- `snake_case` for functions, `PascalCase` for classes
- Use `utils.logging` instead of print statements
- No fallback values that hide errors

## Adding New Tasks

Tasks are the core of ResearchGym. Each task represents an open research problem from a recent ML paper.

### Task Requirements

1. **Source**: Must be from a peer-reviewed venue (ACL/ICML/ICLR/NeurIPS/etc.)
2. **Objective metric**: Must have quantitative evaluation (accuracy, F1, mIoU, etc.)
3. **Reproducible baselines**: Must include working baseline code with documented scores
4. **No solution code**: Task should provide the problem setup, not the paper's solution

### Task Structure

```
tasks/test/<task-name>/
├── task_description.md    # Required: Problem statement
├── requirements.txt       # Required: Dependencies
├── install.sh            # Optional: Setup script
├── grading/
│   └── grade.sh          # Required: Evaluation script
├── idea_hint.txt         # Optional: Hints for agents
└── <baseline-code>/      # Baseline implementation
```

### task_description.md Template

```markdown
## Research Goal
[1-2 paragraphs: What problem are we solving? Why does it matter?]

## Experimental Settings
- **Datasets**: [List datasets with sizes]
- **Baselines**: [List baseline methods]
- **Hardware**: [GPU requirements if any]

## Evaluation Metrics
- [Metric 1]: [Description]
- [Metric 2]: [Description]

## Data Setup
[Commands to download/prepare data]

## Baseline Results (to beat)

| Method | Metric1 | Metric2 |
|--------|---------|---------|
| Baseline1 | X.XX | X.XX |
| Your Method | -- | -- |

## Hint
[Optional: Guidance without giving away the solution]
```

### Grading Script

`grading/grade.sh` must:
- Output JSON to stdout: `{"metric_name": value, ...}`
- Return exit code 0 on success
- Work within the task workspace

Example:
```bash
#!/bin/bash
cd "$(dirname "$0")/.."
python evaluate.py --output-format json
```

### Submitting a Task

1. Create task directory following the structure above
2. Verify baselines run and produce expected scores
3. Test with RGAgent for ~1 hour to ensure it's tractable
4. Submit PR with:
   - Task files
   - Paper reference (title, authors, venue, year)
   - Why this task is suitable for ResearchGym

## Adding New Agents

1. Create adapter in `agents/` following `rg_agent_adapter.py`
2. Implement required methods:
   ```python
   def prepare_workspace(self, task_dir: Path) -> None
   def run(self, cfg, agent_root, dry_run) -> Observation
   ```
3. Add CLI arguments in `run_agent.py`
4. Add README documenting tools and configuration
5. Submit PR with example run demonstrating it works

## Pull Request Process

1. Ensure code passes linting
2. Update documentation if needed
3. Add yourself to contributors (optional)
4. Request review

## Reporting Issues

- **Bugs**: Include reproduction steps, error messages, environment info
- **Tasks**: Propose via issue first before implementing
- **Features**: Describe use case and proposed solution

## Questions?

Open a GitHub issue or discussion.
