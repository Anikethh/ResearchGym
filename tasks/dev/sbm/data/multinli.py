"""Dataset and DataModule for the MultiNLI dataset."""

# Imports Python builtins.
import os
import os.path as osp
import sys

# Imports Python packages.
import numpy as np
import pandas as pd
import wget

# Imports PyTorch packages.
import torch
from torchvision.datasets.utils import (
    extract_archive,
)
from torch.utils.data import Dataset
from tqdm import tqdm
from .register import register_dataset, register_transform
from utils import log

def download(data_root):
    multinli_dir = osp.join(data_root, "multinli")
    if not osp.isdir(multinli_dir):
        os.makedirs(multinli_dir)

        url = (
            "https://github.com/kohpangwei/group_DRO/raw/"
            "f7eae929bf4f9b3c381fae6b1b53ab4c6c911a0e/"
            "dataset_metadata/multinli/metadata_random.csv"
        )
        wget.download(url, out=multinli_dir)

        url = "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz"
        wget.download(url, out=multinli_dir)
        extract_archive(osp.join(multinli_dir, "multinli_bert_features.tar.gz"))

        url = (
            "https://raw.githubusercontent.com/izmailovpavel/"
            "spurious_feature_learning/6d098440c697a1175de6a24"
            "d7a46ddf91786804c/dataset_files/utils_glue.py"
        )
        wget.download(url, out=multinli_dir)

@register_dataset('multinli')
class MultiNLIDataset(Dataset):
    """Dataset for the MultiNLI dataset."""

    def __init__(self, basedir, split="train", transform=None, sel_indexes=None, seed=None):
        
        self.load_data(basedir, split, sel_indexes, seed)
    
    def load_data(self, multinli_dir, split, sel_indexes, seed):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise (f"Unknown split {split}")

        sys.path.append(multinli_dir)
        metadata_df = pd.read_csv(osp.join(multinli_dir, "metadata_random.csv"))
        splits = np.asarray(metadata_df["split"].values)
        self.data_indices = np.argwhere(splits == split_i).flatten()
        split_count = len(self.data_indices)
        if sel_indexes is not None:
            if seed is None:
                raise ValueError("seed is not specified")
            ratio = len(sel_indexes) / len(self.data_indices)
            assert ratio <= 1.0, "incorrect sel_indexes" 
            self.data_indices = self.data_indices[sel_indexes]
        
        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]

        features_array = sum([torch.load(osp.join(multinli_dir, name))
                              for name in bert_filenames], start=[])

        all_input_ids = torch.tensor([
            f.input_ids for f in features_array
        ]).long()
        all_input_masks = torch.tensor([
            f.input_mask for f in features_array
        ]).long()
        all_segment_ids = torch.tensor([
            f.segment_ids for f in features_array
        ]).long()
        total = len(all_input_ids)
        self.data = torch.stack((
            all_input_ids,
            all_input_masks,
            all_segment_ids,
        ), dim=2)[self.data_indices]
        
        
        self.y_array = np.asarray(metadata_df["gold_label"].values)[self.data_indices]
        self.n_classes = len(np.unique(self.y_array))
        
        spurious = np.asarray(metadata_df["sentence2_has_negation"].values)[self.data_indices]
        no_negation = np.argwhere(spurious == 0).flatten()
        negation = np.argwhere(spurious == 1).flatten()
        contradiction = np.argwhere(self.y_array == 0).flatten()
        entailment = np.argwhere(self.y_array == 1).flatten()
        neutral = np.argwhere(self.y_array == 2).flatten()

        self.groups = [
            np.intersect1d(contradiction, no_negation),
            np.intersect1d(contradiction, negation),
            np.intersect1d(entailment, no_negation),
            np.intersect1d(entailment, negation),
            np.intersect1d(neutral, no_negation),
            np.intersect1d(neutral, negation),
        ]

        # Adds group indices into targets for metrics.
        self.group_array = []
        self.confounder_array = []
        g2conf = {0:0,1:1,2:0,3:1,4:0,5:1}
        self.n_confounders = 2
        comb2g = {(0,0):0, (0,1):1, (1,0):2, (1,1):3, (2,0):4, (2,1):5}
        for j, t in enumerate(self.y_array):
            g = comb2g[(t,spurious[j])]
            # g = [k for k, group in enumerate(self.groups) if j in group][0]
            self.group_array.append(g)
            self.confounder_array.append(g2conf[g])
        self.confounder_array = np.array(self.confounder_array)
        group_str = self.get_group_info()
        if sel_indexes is None:
            log(f"{split}: {split_count} ({split_count/total*100:.2f}%)\n{group_str}")
        else:
            log(f"{split} ({ratio*100:.2f}%): {len(self.y_array)} ({len(self.y_array)/split_count*100:.2f}%)\n{group_str}")

    def __len__(self):
        return len(self.y_array)

    def get_group_info(self):
        total = len(self.group_array)
        msg = ''
        for g in range(len(self.groups)):
            y = g // self.n_confounders
            p = g % self.n_confounders
            gcount = len(self.groups[g])
            msg += f"Group{g} (y={y}, a={p}): {gcount} ({gcount/total*100:.2f}%)\n"
        return msg

    def __getitem__(self, idx):
        text, y = self.data[idx], self.y_array[idx]
        y = self.y_array[idx]
        g = self.group_array[idx]
        a = self.confounder_array[idx]
        return text, y, g, a
    