import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
from tqdm import tqdm
from .register import register_dataset, register_transform
from utils import log
# https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
# https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
# celeba_metadata: https://github.com/PolinaKirichenko/deep_feature_reweighting

@register_dataset(['waterbirds', 'celeba'])
class BiasedDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, sel_indexes=None, seed=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise (f"Unknown split {split}")
        
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
      
        total = len(metadata_df)
        split_info = metadata_df["split"].values
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        split_total = len(self.metadata_df)

        if sel_indexes is not None:
            if seed is None:
                raise ValueError("seed is not specified")
            ratio = len(sel_indexes) / len(self.metadata_df)
            assert ratio <= 1.0, "incorrect sel_indexes"
            self.metadata_df = self.metadata_df.iloc[sel_indexes]
        
        self.basedir = basedir
        self.transform = transform

        self.y_array = self.metadata_df["y"].values
        self.confounder_array = self.metadata_df["place"].values

        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.confounder_array).size

        self.group_array = (
            self.y_array * self.n_places + self.confounder_array
        ).astype("int")
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
            (
                torch.arange(self.n_groups).unsqueeze(1)
                == torch.from_numpy(self.group_array)
            )
            .sum(1)
            .float()
        )
        self.y_counts = (
            (
                torch.arange(self.n_classes).unsqueeze(1)
                == torch.from_numpy(self.y_array)
            )
            .sum(1)
            .float()
        )

        self.filename_array = self.metadata_df["img_filename"].values

        group_str = self.get_group_info()
        if sel_indexes is None:
            log(f"{split}: {split_total} ({split_total/total*100:.2f}%)\n{group_str}")
        else:
            log(f"{split} ({ratio*100:.2f}%): {len(self.y_array)} ({len(self.y_array)/split_total*100:.2f}%)\n{group_str}")
        
    def __len__(self):
        return len(self.filename_array)

    def get_group_info(self):
        total = sum(self.group_counts)
        msg = ''
        for g in range(len(self.group_counts)):
            y = g // self.n_places
            p = g % self.n_places
            gcount = self.group_counts[g]
            msg += f"Group{g} (y={y}, a={p}): {int(gcount)} ({gcount/total*100:.2f}%)\n"
        return msg

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        a = self.confounder_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, y, g, a
      
