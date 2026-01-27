# Multi-label classification evaluation skeleton (CLIP-based baselines)


```bash
clip: OpenAI CLIP and helper modules

configs: minimal configuration files

data: dataset loaders and utilities (COCO, VOC, NUSWIDE, etc.)

maple_clip: reference prompt-learning baseline (MaPLe)

scripts: convenience shell scripts

utils: helper utilities (metrics, logging)
```

# Training and testing

```bash
# args: <gpu_id> <perform_tpt> <dataset> <is_bind>
# run a CLIP/TPT baseline (is_bind must be 0 in this skeleton)
bash scripts/test_tpt_clip.sh 0 1 coco2014 0
```

Notes:
- Set DATASETS root in scripts or pass via --data.
- Do not enable caption-binding flags; method-specific code has been removed.