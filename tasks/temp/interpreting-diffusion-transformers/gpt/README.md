## Skeleton Repository

This is a neutral skeleton derived from an interpretability project. All method-specific algorithms and assets have been removed. Infrastructure for setup, baselines, and evaluation remains to help you build new approaches for zero-shot image segmentation and related interpretability tasks.

### Install
```bash
pip install -e .
```

### Usage
- Baseline evaluations and scripts are under `experiments/` (e.g., CLIP/Chefer, DINO, DAAM-SDXL wrappers).
- Method-specific pipelines are intentionally omitted. Use the provided baselines as references or starting points for your own ideas.

See `task_description.md` for the problem scope and `MODIFICATIONS.md` for what was removed or preserved.