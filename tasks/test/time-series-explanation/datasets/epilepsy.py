import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import torch as th
import warnings

from datetime import timedelta
from getpass import getpass
from sklearn.impute import SimpleImputer

from tint.utils import get_progress_bars

from datasets.dataset import DataModule, Dataset

import sys, os


from txai.utils.data.preprocess import process_Epilepsy

file_dir = os.path.dirname(__file__)

class Epilepsy(DataModule):
    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "epilepsy",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )
        
        self._mean = None
        self._std = None
        
    def prepare_data(self):
        if not os.path.exists(
            os.path.join(self.data_dir, "all_epilepsy.pt")
        ) or not os.path.join(
            self.data_dir, "split_{}.npy".format(self.fold + 1)
        ):
            raise RuntimeError("No data exists or wrong path")
        
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train = Dataset(self.preprocess("train"))
            self.val = Dataset(self.preprocess("val"))
            
        if stage == "test" or stage is None:
            self.test = Dataset(self.preprocess("test"))

        if stage == "predict" or stage is None:
            self.predict = Dataset(self.preprocess("test"))

    def preprocess(self, split: str = "train") -> dict:
        
        # (train, val, test)
        # total_data = process_PAM(split_no = self.fold + 1, base_path = self.data_dir, gethalf = True)
        total_data = process_Epilepsy(split_no = self.fold + 1,base_path = self.data_dir)
        
        data_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        data = total_data[data_dict[split]]
    
        
        features = data.X.transpose(0, 1)
        
        if split == "train":
            self._mean = features.mean(dim=(0, 1), keepdim=True)
            self._std = features.std(dim=(0, 1), keepdim=True)
            
        EPS = 1e-5
        features = (features - self._mean) / (self._std + EPS)

        return {
            "x": features.float(),
            "y": data.y.long(),
            "mask": th.ones_like(features)
        }