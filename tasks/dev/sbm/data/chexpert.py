
import os
import pandas as pd
from pathlib import Path
from utils import log
from .register import register_dataset
import torch
import numpy as np
from PIL import Image
import warnings
from sklearn.model_selection import train_test_split

def generate_metadata_chexpert(data_path, test_pct=0.15, val_pct=0.1):
    print("Generating metadata for CheXpert No Finding prediction...")
    chexpert_dir = Path(data_path)
    assert (chexpert_dir/'train.csv').is_file()
    assert (chexpert_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'CHEXPERT DEMO.xlsx').is_file()

    df = pd.concat([pd.read_csv(chexpert_dir/'train.csv'), pd.read_csv(chexpert_dir/'valid.csv')], ignore_index=True)

    df['filename'] = df['Path'].astype(str).apply(lambda x: x[x.index('/')+1:])
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)
    df = df[df.Sex.isin(['Male', 'Female'])]
    details = pd.read_excel(chexpert_dir/'CHEXPERT DEMO.xlsx', engine='openpyxl')[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    df = pd.merge(df, details, on='subject_id', how='inner').reset_index(drop=True)

    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
        return 2

    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    attr_mapping = {'Male_0': 0, 'Female_0': 1, 'Male_1': 2, 'Female_1': 3, 'Male_2': 4, 'Female_2': 5}
    df['a'] = (df['Sex'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['y'] = df['No Finding'].fillna(0.0).astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42, stratify=df['a'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df.loc[train_val_idx, 'a'])

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    # (chexpert_dir/'subpop_bench_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(chexpert_dir, "metadata_no_finding.csv"), index=False)
    return df

@register_dataset("chexpert")
class Chexpert(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir,
        split,
        transform=None,
        sel_indexes=None,
        seed=None
    ):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise (f"Unknown split {split}")

        metadata_path = os.path.join(basedir, "metadata_no_finding.csv")
        # if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        # else:
            # metadata_df = generate_metadata_chexpert(basedir)
        
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        total = len(metadata_df)
        split_total = len(self.metadata_df)

        if sel_indexes is not None:
            if seed is None:
                raise ValueError("seed is not specified")
            self.metadata_df = self.metadata_df.iloc[sel_indexes]
            ratio = len(sel_indexes) / split_total
            assert ratio <= 1.0, "incorrect sel_indexes" 


        self.y_array = self.metadata_df["y"].values
        self.filename_array = self.metadata_df["filename"].values
        self.n_classes = np.unique(self.y_array).size

        self.basedir = basedir
        self.transform = transform

        self.split = split

        self.confounder_array = self.metadata_df["a"].values
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

        group_str = self.get_group_info()
        if sel_indexes is None:
            log(f"{split}: {split_total} ({split_total/total*100:.2f}%)\n{group_str}")
        else:
            log(f"{split} ({ratio*100:.2f}%): {len(self.y_array)} ({len(self.y_array)/split_total*100:.2f}%)\n{group_str}")

    def get_group_info(self):
        total = sum(self.group_counts)
        msg = ''
        for g in range(len(self.group_counts)):
            y = g // self.n_places
            p = g % self.n_places
            gcount = self.group_counts[g]
            msg += f"Group{g} (y={y}, a={p}): {int(gcount)} ({gcount/total*100:.2f}%)\n"
        return msg

    def __getitem__(self, index):
        path, target = self.filename_array[index], self.y_array[index]
        img_path = os.path.join(self.basedir, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        g = self.group_array[index]
        a = self.confounder_array[index]
        return img, target, g, a
      

    def __len__(self):
        return len(self.y_array)