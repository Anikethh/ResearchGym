## Cross-Modal Retrieval Starter Repository

### Features
- Dataset loaders for common benchmarks (e.g., COCO, Flickr, Fashion-Gen, ReID variants)
- Embedding extraction using BLIP and CLIP backbones
- Evaluation utilities for Recall@K and mAP
- Optional baseline adapters (e.g., Tent, SAR, READ, SHOT) for test-time adaptation experiments

### File Organization

```
├── data
│   ├── coco_dataset.py           # dataloaders
│   ├── ...
│   └── utils.py
├── configs
│   ├── QS                     # Config files for QS setting
│   ├── QGS                    # Config files for QGS setting
│   └── Reid                   # Config files for Reid (QGS) setting
├── dataset
│   ├── coco				 # coco dataset
│   ├── coco-IP				 # coco dataset with image corruptions
│   │   ├── COCO_IP_brightness_1
│   │   ├── ...
│   ├── coco-TP				 # coco dataset with text corruptions
│   │   ├── annotation_OcrAug
│   │   ├── ...
│   ├── ...
│   └── Fashion-Gen
├── model
│   ├── tta_baselines
│   │   ├── your_method.py			 # method
│   │   ├── read.py			 
│   │   ├── sar.py			 
│   │   ├── tent.py
│   │   ├── ...
│   └── blip_tta.py			 # model
├── output                    # output log
├── weights                   # pretrained model weight
│   ├── blip
│   ├── fashion_pretrained
│   └── reid
└── main.py	
	demo script for evaluation
```

### Benchmarks

You can download the benchmarks via [Google Cloud](https://drive.google.com/drive/folders/17QyJ9Y52XB67jektNvhUk6jKmIQnoatF?usp=sharing).

More details abouth how to get Flickr-C and COCO-C benchmarks can be found in [this repository](https://github.com/Jielin-Qiu/MM_Robustness). Specifically, for image corruptions, we employ image_perturbation/perturb_COCO_IP.py to construct image corruptions for the COCO dataset and image_perturbation/perturb_Flickr30K_IP.py for the Flickr dataset. For text corruptions, we utilize text_perturbation/perturb_COCO_TP.py to construct text corruptions for the COCO dataset and text_perturbation/perturb_Flickr30K_TP.py for the Flickr dataset.

The original datasets can be downloaded from the original website:
- Flickr30K: https://shannon.cs.illinois.edu/DenotationGraph/
- COCO: https://cocodataset.org/#home
- FashionGen: import kagglehub \n\n path = kagglehub.dataset_download("bothin/fashiongen-validation")



### Pretrained Model

The proposed proposed method should support mainstream pre-trained models, including BLIP and CLIP.

- **BLIP** can be downloaded from [BLIP](https://github.com/salesforce/BLIP).
- **CLIP** can be accessed via [Hugging Face OpenAI](https://huggingface.co/openai).

The CLIP model can be directly downloaded through the code. For other models, download link: [Google Cloud](https://drive.google.com/drive/folders/1tVuNY9l520BY6SvDdzzmNhF_wCrIaUei?usp=sharing).

### Reproduce

To run the **TTA process**, use the following commands:

#### Text Retrieval

- **BLIP Base & CLIP**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --master_port 10000 main.py --retrieval i2t --config your-config-path --method your_method
  ```

- **BLIP Large**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 10000 main.py --retrieval i2t --config your-config-path --method your_method
  ```

#### Image Retrieval

  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 --master_port 10000 main.py --retrieval t2i --config your-config-path --method your_method
  ```

#### Parameter

- **`--method`**: You can choose from existing baselines for evaluation or reproduce other methods directly in `./model/tta_baselines/`. Below are some provided SOTA baselines:  
  - **Tent** ([GitHub](https://github.com/DequanWang/tent))
  
  - **SAR** ([GitHub](https://github.com/mr-eggplant/SAR))

  - **READ** ([GitHub](https://github.com/XLearning-SCU/2024-ICLR-READ))
  
  - To use Tent, set `--method tent`. (Vanilla TTA approach with an entropy-based objective)

- **`--retrieval`**:
  - `i2t`: Image-to-text retrieval
  - `t2i`: Text-to-image retrieval
  - `None`: Disables TTA

## Acknowledgements
The setup is based on [MM_Robustness](https://github.com/Jielin-Qiu/MM_Robustness) and [BLIP](https://github.com/salesforce/BLIP) licensed under Apache 2.0.

### Grading

Use `./grading/grade.py` (or the wrapper `./grading/grade.sh`) after your runs finish to aggregate `evaluate.txt` files and refresh the `task_description.md` tables. By default the script searches the task’s `output/` directory, writes a JSON summary under `grading/runs/`, and updates the placeholder rows.

Examples:

```bash
# Refresh every table with the method label pulled from METHOD_NAME
./grading/grade.py

# Only rebuild the query-gallery shift table without touching markdown
./grading/grade.py --table qgs --no-markdown --json-out /tmp/qgs.json
```
