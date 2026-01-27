"""Dataset and DataModule for the CivilComments dataset."""

# Imports Python packages.
import numpy as np
import os.path as osp
import pickle
from transformers import BertTokenizer
import wilds

# Imports PyTorch packages.
import torch

from torch.utils.data import Dataset
from .register import register_dataset, register_transform
from utils import log

def _to_np(x):
    """Converts torch.Tensor input to numpy array."""

    return x.cpu().detach().numpy()

def to_np(x):
    """Converts input to numpy array.

    Args:
        x: A torch.Tensor, np.ndarray, or list.

    Returns:
        The input converted to a numpy array.

    Raises:
        ValueError: The input cannot be converted to a numpy array.
    """

    if not len(x):
        return np.array([])
    elif isinstance(x, torch.Tensor):
        return _to_np(x)
    elif isinstance(x, (np.ndarray, list)):
        if isinstance(x[0], torch.Tensor):
            return _to_np(torch.tensor(x))
        else:
            return np.asarray(x)
    else:
        raise ValueError("Input cannot be converted to numpy array.")

@register_dataset('civilcomments')
class CivilCommentsDataset(Dataset):
    """Dataset for the CivilComments dataset."""

    def __init__(self, basedir, split="train", transform=None, sel_indexes=None, seed=None):
        self.load_data(basedir, split, sel_indexes, seed)

    def load_data(self, basedir, split, sel_indexes, seed):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise (f"Unknown split {split}")
        dataset = wilds.get_dataset(
            dataset="civilcomments",
            download=True,
            root_dir=basedir,
        )

        spurious_names = ["male", "female", "LGBTQ", "black", "white",
                          "christian", "muslim", "other_religions"]
        column_names = dataset.metadata_fields
        spurious_cols = [column_names.index(name) for name in spurious_names]

        spurious = to_np(dataset._metadata_array[:, spurious_cols].sum(-1).clip(max=1))
        splits = dataset._split_array
        prefix = osp.join(basedir, "civilcomments_v1.0")
        data_file = osp.join(prefix, "civilcomments_token_data.pt")
        targets_file = osp.join(prefix, "civilcomments_token_targets.pt")


        if not osp.isfile(data_file):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            def tokenize(text):
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=220,
                    return_tensors="pt",
                )

                return torch.squeeze(torch.stack((
                    tokens["input_ids"], tokens["attention_mask"], 
                    tokens["token_type_ids"]), dim=2), dim=0)

            data = []
            targets = []
            ln = len(dataset)
            for j, d in enumerate(dataset):
                print(f"Caching {j}/{ln}")
                # print(d[0])
                data.append(tokenize(d[0]))
                targets.append(d[1])
            data = torch.stack(data)
            targets = torch.stack(targets)
            torch.save(data, data_file)
            torch.save(targets, targets_file)
        
        total = len(splits)
        self.data_indices = np.argwhere(splits == split_i).flatten()
        split_count = len(self.data_indices)
        if sel_indexes is not None:
            if seed is None:
                raise ValueError("seed is not specified")
            ratio = len(sel_indexes) / len(self.data_indices)
            assert ratio <= 1.0, "incorrect sel_indexes" 
            self.data_indices = self.data_indices[sel_indexes]



        self.data = torch.tensor(torch.load(data_file).numpy()[self.data_indices])
        self.y_array = torch.load(targets_file).numpy()[self.data_indices]
        self.n_classes = len(np.unique(self.y_array))

        
        spurious = spurious[self.data_indices]
        self.n_confounders = 2
        self.groups = [
            np.intersect1d((~self.y_array+2).nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d((~self.y_array+2).nonzero()[0], spurious.nonzero()[0]),
            np.intersect1d(self.y_array.nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d(self.y_array.nonzero()[0], spurious.nonzero()[0]),
        ]
        self.group_array = []
        self.confounder_array = []
        g2conf = {0:0,1:1,2:0,3:1}
        # Adds group indices into targets for metrics.
        for j, t in enumerate(self.y_array):
            g = t * 2 + spurious[j]
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
        total = len(self.y_array)
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
