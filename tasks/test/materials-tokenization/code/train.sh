#!/bin/bash
set -euo pipefail

# Preprocess raw .pt batches into normalized .pt files (see preprocess.py)
python preprocess.py

# Tokenize preprocessed .pt files into fixed-length input tensors using a vocab
# Expect a vocab at repo root. You can place your own vocab.txt there.
python tokenization.py

# Train a baseline masked language model with the provided dataset utilities.
# This is a placeholder; integrate your own training loop or model.
echo "Baseline training placeholder: integrate your model here."
