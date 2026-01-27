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

file_dir = os.path.dirname(__file__)

class PAMchunk:
    '''
    Class to hold chunks of PAM data
    '''
    def __init__(self, train_tensor, static, time, y):
        self.X = train_tensor
        self.static = None if static is None else static
        self.time = time
        self.y = y

    def choose_random(self):
        n_samp = len(self.X)           
        idx = random.choice(np.arange(n_samp))
        
        static_idx = None if self.static is None else self.static[idx]
        print('In chunk', self.time.shape)
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor

def getStats(P_tensor):
    N, T, F = P_tensor.shape
    if isinstance(P_tensor, np.ndarray):
        Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    else:
        Pf = P_tensor.permute(2, 0, 1).reshape(F, -1).detach().clone().cpu().numpy()
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f][0], eps])
    return mf, stdf

def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = th.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = th.Tensor(P_tensor)

    P_time = th.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = th.Tensor(y_tensor[:, 0]).type(th.LongTensor)
    return P_tensor, None, P_time, y_tensor

def process_PAM(split_no = 1, base_path = "", gethalf = False):
    
    split_path = os.path.join(
            base_path, 'splits/PAMAP2_split_{}.npy'.format(split_no)
    )
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle=True)

    Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)

    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    y = arr_outcomes[:, -1].reshape((-1, 1))

    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    #return Ptrain, Pval, Ptest, ytrain, yval, ytest

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_tensor = Ptrain
    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
    Pval_tensor = Pval_tensor.permute(1, 0, 2)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

    if gethalf:
        Ptrain_tensor = Ptrain_tensor[:,:,:(Ptrain_tensor.shape[-1] // 2)]
        Pval_tensor = Pval_tensor[:,:,:(Pval_tensor.shape[-1] // 2)]
        Ptest_tensor = Ptest_tensor[:,:,:(Ptest_tensor.shape[-1] // 2)]

    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    train_chunk = PAMchunk(Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor)
    val_chunk = PAMchunk(Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor)
    test_chunk = PAMchunk(Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor)

    return train_chunk, val_chunk, test_chunk


class PAM(DataModule):
    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "PAM",
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
            os.path.join(self.data_dir, "processed_data/PTdict_list.npy")
        ) or not os.path.join(
            self.data_dir, "splits/PAMAP2_split_{}.npy".format(self.fold + 1)
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
        total_data = process_PAM(split_no = self.fold + 1, base_path = self.data_dir, gethalf = True)
        data_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        data = total_data[data_dict[split]]
    
        
        features = data.X.transpose(0, 1)
        
        # print(features.mean(dim=(0, 1), keepdim=True))
        # print(features.std(dim=(0, 1), keepdim=True))
        # print(data.time)
        # raise RuntimeError
        
        if split == "train":
            self._mean = features.mean(dim=(0, 1), keepdim=True)
            self._std = features.std(dim=(0, 1), keepdim=True)
            
        EPS = 1e-5
        features = (features - self._mean) / (self._std + EPS)
            
        # print(features.shape)
        # print(data.y.shape)
        # print(th.ones_like(features).shape)
        
        # print(data_dict[split])
        

        return {
            "x": features.float(),
            "y": data.y.long(),
            "mask": th.ones_like(features)
        }