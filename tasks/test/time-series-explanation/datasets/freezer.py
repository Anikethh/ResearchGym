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
from tslearn.utils import _load_txt_uea

from datasets.dataset import DataModule, Dataset

import sys, os

file_dir = os.path.dirname(__file__)

class Freezer(DataModule):
    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "FreezerRegularTrain",
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
            os.path.join(self.data_dir, "FreezerRegularTrain_TRAIN.txt")
        ):
            raise RuntimeError("No data exists or wrong path")
        
    def preprocess(self, split: str = "train") -> dict:
        dataset_name = "FreezerRegularTrain"
        X_train, y_train = _load_txt_uea(
                    os.path.join(self.data_dir, dataset_name + "_TRAIN.txt")
        )
        X_test, y_test = _load_txt_uea(
            os.path.join(self.data_dir, dataset_name + "_TEST.txt")
        )
        print(X_train.shape)
        print(y_train.shape)
        if split == "train":
            features = th.Tensor(X_train)
            y = th.Tensor(y_train)
            self._mean = features.mean(dim=(0, 1), keepdim=True)
            self._std = features.std(dim=(0, 1), keepdim=True)
        elif split == "test":
            features = th.Tensor(X_test)
            y = th.Tensor(y_test)
        else:
            raise RuntimeError
     
        EPS = 1e-5
        features = (features - self._mean) / (self._std + EPS)
        
        y = y - 1

        return {
            "x": features.float(),
            "y": y.long(),
            "mask": th.ones_like(features)
        }

        