# Diffusion Transformer Interpretability Research Skeleton

This repository provides a clean, unbiased starting point for developing new interpretability methods for multi-modal diffusion transformers (DiTs). The skeleton preserves essential infrastructure while removing method-specific implementations.

## Repository Structure

### Baseline Segmentation Framework
- `baseline_segmentation/` - Framework for implementing and evaluating segmentation-based interpretability methods
  - `segmentation_base.py` - Abstract base class for segmentation models
  - `evaluation_utils.py` - Evaluation metrics and utilities
  - `dino_baseline.py` - DINO-based baseline implementation
  - `data_processing.py` - ImageNet segmentation data processing utilities

### Evaluation Datasets
- Support for ImageNet-Segmentation benchmark evaluation
- PascalVOC segmentation evaluation framework
- Standard evaluation metrics: mIoU, pixel accuracy, mAP

### Baseline Methods Included
- DINO self-attention visualization
- Infrastructure for CLIP-based methods
- Evaluation framework for comparing interpretability methods

## Setup

Install the package locally:
```bash
pip install -e .
```

## Experiments

The `experiments/` directory contains evaluation scripts for:
- ImageNet segmentation benchmarks
- PascalVOC segmentation evaluation  
- Baseline method comparisons

Example experiment structure:
```bash
cd experiments/imagenet_segmentation
python run_experiment.py --segmentation_model DINO
```

## Data Setup

For ImageNet-Segmentation evaluation, download the required data:
```bash
cd baseline_segmentation/data
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
```

## Development Guidelines

This skeleton is designed for developing new interpretability methods for diffusion transformers. The structure supports:

1. **New Method Implementation**: Inherit from `SegmentationAbstractClass` to implement your method
2. **Evaluation**: Use existing evaluation utilities and benchmarks
3. **Comparison**: Compare against included baseline methods

## License

[Original license preserved]