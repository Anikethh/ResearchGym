## AIGI Detection

This repository is a starting point for AI-generated image (AIGI) detection research.

### What’s included
- DeepfakeBench-style training/evaluation pipeline under `DeepfakeBench/`
- A simple baseline detector (`baseline`) configured to use a standard backbone (default `xception`)
- A generic benchmarking folder `UniversalFakeDetect_Benchmark/` with CLIP baselines

### Quick inference (single image or folder)
```
cd DeepfakeBench/
python3 training/demo.py \
  --detector_config training/config/detector/method.yaml \
  --weights ./training/weights/{CHECKPOINT}.pth \
  --image {IMAGE_PATH or IMAGE_FOLDER}
```
If running on face images, you may optionally add `--landmark_model ./preprocessing/shape_predictor_81_face_landmarks.dat` to crop faces before inference.

### Train and test (DeepfakeBench)
Train:
- For multiple GPUs:
```
python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py \
--detector_path ./training/config/detector/method.yaml \
--train_dataset FaceForensics++ \
--test_dataset Celeb-DF-v2 \
--ddp
```
- For a single GPU:
```
python3 training/train.py \
--detector_path ./training/config/detector/method.yaml \
--train_dataset FaceForensics++ \
--test_dataset Celeb-DF-v2 \
```
Test:
```
python3 training/test.py \
  --detector_path ./training/config/detector/method.yaml \
  --test_dataset simswap_ff blendface_ff uniface_ff fomm_ff deepfacelab \
  --weights_path ./training/weights/{CKPT}.pth
```

### CLIP baselines (UniversalFakeDetect_Benchmark)
Use `script/train.sh` and `script/test.sh` as templates for neutral CLIP baselines (method-specific flags removed). Update model names/paths as needed.

### Downloading datasets

You can download the processed datasets (have already finished preprocessing such as frame extraction and face cropping) from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). For evaluating more diverse fake methods (such as SimSwap, BlendFace, DeepFaceLab, etc), you are recommended to use the [DF40 dataset](https://github.com/YZY-stack/DF40) (with 40 distinct forgery methods implemented).

### 3. Rearrangement 

> "Rearrangment" here means that we need to **create a *JSON file* for each dataset for collecting all frames within different folders**. Please refer to **DeepfakeBench** and **DF40** for the provided JSON files for each dataset.

After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 

### Notes
- Default config `training/config/detector/method.yaml` points to a neutral `baseline` detector with `xception` backbone. 

### Dataset setup (paths and env vars)
- DeepfakeBench datasets: place under `DeepfakeBench/training/datasets` and JSONs under `DeepfakeBench/preprocessing/dataset_json`. You can override with env vars:
  - `DFB_DATA_ROOT` (defaults to `./DeepfakeBench/training/datasets`)
  - `DFB_JSON_ROOT` (defaults to `./DeepfakeBench/preprocessing/dataset_json`)
- UniversalFakeDetect (Wang2020) dataset root via env var `UFD_DATA_ROOT` (expected layout: `test/<subset>/{0_real,1_fake}/...`).

### Eval protocols
- Protocol-1 (Cross-dataset): Train on FF++ (c23). Test on CDF-v2, DFD, DFDC, DFDCP, DFo, WDF, FFIW. Report video-level AUC (averaging 32 frames per video at inference).
- Protocol-2 (Cross-method/DF40): Train on FF++ (c23). Test on DF40 manipulations. Report video-level AUC.

See `tasks/test/aigi-detection/task_description.md` for baseline tables and metrics.

