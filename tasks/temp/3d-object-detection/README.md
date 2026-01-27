## 3D Object Detection

This repository provides a scaffold for indoor 3D object detection experiments built on top of the OpenMMLab MMDetection3D stack and TR3D baselines.

### Environment
- Installation follows the upstream MMDetection3D and TR3D requirements.
- Build and extension setup are managed via `setup.py` and the `mmdet3d` package.

### Data Preparation
- Follow the MMDetection3D data preparation guides:
  - `data/scannet/README.md`
  - `data/sunrgbd/README.md`

### Sparse Supervision Splits
- You can generate sparse splits using `tools/create_sparse_infos.py`.

### Training
- Use standard training entrypoints:
  - Single GPU: `python tools/train.py <config.py>`
  - Distributed: `bash tools/dist_train.sh <config.py> <num_gpus>`

Baseline example configs are provided under `projects_sparse/configs/baseline/`:
- `projects_sparse/configs/baseline/tr3d_scannet-3d-18class.py`
- `projects_sparse/configs/baseline/tr3d_sunrgbd-3d-10class.py`

### Evaluation / Testing
- Single GPU: `python tools/test.py <config.py> <checkpoint.pth> --eval mAP`
- Distributed: `bash tools/dist_test.sh <config.py> <checkpoint.pth> <num_gpus>`

### Evaluation Metrics

The repository includes comprehensive evaluation metrics for:
- Indoor scenes (mAP@0.25, mAP@0.5)
- Outdoor scenes (3D AP at various IoU thresholds)
- Instance segmentation metrics

### Repository Layout
- `mmdet3d/`: core library (models, datasets, ops, utils)
- `projects_sparse/`: baseline configs and dataset wrappers for sparse supervision
  - `configs/baseline/`: reference baseline TR3D configs
  - `sparse/pre_processing/`: dataset adapters/utilities for sparse supervision
- `tools/`: training, testing, conversion, and analysis scripts
- `configs/`: base runtime and schedules
- `tests/`: unit tests and sample data for utilities

### Notes
- You can add your own methods by creating new heads/detectors under `mmdet3d/models/*` and referencing them from your configs.
