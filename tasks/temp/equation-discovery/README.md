# Scientific Equation Discovery Skeleton

This repository is a neutral skeleton for building solutions to scientific equation discovery and symbolic regression tasks. It preserves infrastructure (data layout, environment files, and specs) while removing method-specific code and paper content.

## Installation

To run, create a conda environment and install the dependencies provided in the `requirements.txt` or `environment.yml`:

```
conda create -n eqdisc python=3.11.7
conda activate eqdisc
pip install -r requirements.txt
```

or

```
conda env create -f environment.yml
conda activate llmsr
```

Note: Requires Python ≥ 3.9

## Datasets

Example datasets are provided under `data/`. Each subfolder contains `train.csv`, `test_id.csv`, and `test_ood.csv`.

## Usage

Run the loader stub to verify data access:

```
python main.py --problem_name oscillator1
```

Extend `main.py` with your own training/evaluation pipeline or add baselines under a new `baselines/` directory.

## Specs

The `specs/` directory contains problem templates that describe desired inputs/outputs for equation discovery tasks. They are preserved for reference and adaptation but do not enforce any specific method.

## License 

This repository retains the original license.
