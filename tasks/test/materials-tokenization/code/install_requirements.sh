#!/bin/sh

# Non-interactive, generic environment setup for this repository's utilities and evaluation.
# Adjust names/versions as needed for your system.

ENV_NAME="materials-nlp"
PY_VER="3.8"

# Create and activate a conda environment
conda create -n "$ENV_NAME" python="$PY_VER" -y
. "$(conda info --base)"/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Core Python deps
conda install -y numpy==1.20.3 pandas==1.2.4 scikit-learn=0.23.2

# PyTorch (CPU by default; switch channel/cu version per your GPU)
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# Project Python deps
pip install -r requirements.txt

echo "Environment '$ENV_NAME' is ready."
