## Continual Learning

This repository provides a starting point for class-incremental learning research.

### Usage
- Configure experiments with JSON files in `exps/` (e.g., `simplecil.json`).
- Run a baseline example:
```bash
bash run.sh
```

Datasets should be available at `/data/` or adjust dataset paths in configs.

### Contents
- `utils/`: data pipeline, model factory, training utilities
- `models/`: baseline continual learning methods (e.g., simple CIL, iCaRL, DER, COIL, FOSTER, L2P variants)
- `backbone/`: standard backbones and prompt/adapter variants as used by baselines
- `exps/`: experiment configurations for baselines
