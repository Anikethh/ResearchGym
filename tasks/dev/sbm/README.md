
# Spurious Bias Benchmarks

## Preparation

### Download datasets
- [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) ([metadata](https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/celeba_metadata.csv))
- ImageNet [(train](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) [,val)](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
- CivilComments: The dataset will be automatically downloaded from`wilds`
- MultiNLI: Call the function `download` in `data/multinli.py` to download the dataset.

Unzip the dataset files into individual folders.

In the `config.py` file, set each value in `dataset_paths` to your corresponding dataset folder. 

### Prepare `metadata.csv` for each dataset
- Waterbirds, CelebA, CivilComments, and MultiNLI provide `metadata.csv` files.
- For the ImageNet-9 and ImageNet-A datasets, run the following code
    ```python
    from data.in9_data import prepare_imagenet9_metadata, prepare_imageneta_metadata
    base_dir = "path/to/imagenet/folder"
    prepare_imagenet9_metadata(base_dir)
    data_root = "path/to/imagenet-a/folder"
    prepare_imageneta_metadata(data_root)
    ````

## ERM training
Train a baseline ERM model.
```python
python main.py --dataset waterbirds\
               --save_folder /p/spurious/spurious_exprs\
               --backbone resnet50\
               --batch_size 32\
               --pretrained True\
               --mode train\
               --epoch 100\
               --optimizer sgd\
               --optimizer_kwargs lr=0.003 weight_decay=0.0001 momentum=0.9\
               --scheduler cosine\
               --scheduler_kwargs T_max=100\
               --gpu 0\
               --seed 0\
               --split_train 1.0\
               --train_split train\
               --test_split test\
               --algorithm erm
```

## Baselines

This skeleton preserves standard training and evaluation utilities for:
- `erm`: empirical risk minimization
- `dfr`: deep feature reweighting (last-layer retraining)
- `afr`: adaptive feature reweighting
- `jtt`: just train twice

See `algorithms/` for available baselines and usage.
