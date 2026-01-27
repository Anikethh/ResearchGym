# Monotonic Modeling Skeleton

Neutral, method-agnostic skeleton for studying monotonic probability modeling. It preserves data loading, training/evaluation utilities, and several standard baseline models (e.g., positive-weight MLPs and classic monotonic network variants). Implementations specific to any particular paper method have been removed.

## Requirements

- Python >= 3.8
- Install deps: `pip install -r requirements.txt`

## Datasets

CSV files are under `data/` with train/test splits for: `compas`, `loan`, `adult`, `diabetes`, `blog`, `auto`.

## Usage

Run baselines on a dataset:

`python run.py -d {dataset} -s {random_seed}`

Where `dataset` ∈ {`compas`, `loan`, `adult`, `diabetes`, `blog`, `auto`} and `random_seed` is an integer.

## Notes

- This repository is a starting point for new methods; plug in your own models in `model.py` without relying on removed method-specific code.
- Configuration such as batch size, learning rate, and epochs is in `config.py`.
