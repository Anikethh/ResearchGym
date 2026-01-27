#!/usr/bin/env bash

#!/usr/bin/env bash

# Neutral CLIP baseline training (no method-specific SVD)
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --name clip_vitl14_baseline \
  --wang2020_data_path /path/to/CNNDetection/dataset/ \
  --data_mode wang2020 \
  --arch CLIP:ViT-L/14 \
  --batch_size 48 \
  --loadSize 256 \
  --cropSize 224 \
  --lr 0.0002 

## Example for SigLIP baseline
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name siglip_vitl16_baseline \
#   --wang2020_data_path /path/to/CNNDetection/dataset/ \
#   --data_mode wang2020 \
#   --arch CLIP:ViT-L/14 \
#   --batch_size 48 \
#   --loadSize 256 \
#   --cropSize 256 \
#   --lr 0.0002 

## Example for BEiT-v2 baseline (requires adapting models/__init__.py)
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name beitv2_vitl16_baseline \
#   --wang2020_data_path /path/to/CNNDetection/dataset/ \
#   --data_mode wang2020 \
#   --arch CLIP:ViT-B/16 \
#   --batch_size 48 \
#   --loadSize 256 \
#   --cropSize 224 \
#   --lr 0.0002 