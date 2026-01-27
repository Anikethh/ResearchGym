from config import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler

import numpy as np
import pickle
import os
import glob
import torch

class PredDataset(Dataset):
    def __init__(
        self,
        dataset,
        pred_labels,

    ):
        self.n_classes = dataset.n_classes
       
        self.dataset = dataset
        self.pred_labels = pred_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset[idx]
        pred_label = self.pred_labels[idx]
        return *items, pred_label
    

class MisDataset(Dataset):
    def __init__(
        self,
        dataset,
        mis_labels,
        indexes,

    ):
        self.n_classes = dataset.n_classes
       
        self.dataset = dataset
        self.indexes = indexes
        self.mis_labels = mis_labels

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        items = self.dataset[self.indexes[idx]]
        mis_label = self.mis_labels[idx]
        return *items, mis_label


class JointDataset(Dataset):
    def __init__(
        self,
        datasets,
    ):
        
        self.datasets = datasets
        self.n_datasets = len(datasets)

    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        assert len(idx) == self.n_datasets, "wrong indexes"
        output = []
        for n,i in enumerate(idx):
            output.extend(self.datasets[n][i])
        return output

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

class SubsetDataset(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


def get_split_indexes(dataset : Dataset, ratio : float):
    """
    Get the indexes of the first and second subsets of the dataset. 
    The ratio specifies the first subset's proportion of the dataset.

    :param dataset: target dataset
    :param ratio: split ratio

    :return indexes_first_subset, indexes_second_subset
    """
    assert ratio > 0 and ratio < 1, "ratio is not within (0,1)"
    num = len(dataset)
    indexes = np.arange(num)
    np.random.shuffle(indexes)
    num1 = int(num*ratio)
    return indexes[0:num1], indexes[num1:]

def prepare_data(config):
    from .register import dataset_dict, transform_funcs
    dataset = config.dataset
    if dataset not in dataset_dict:
        raise ValueError(f"{dataset} not defined/registered")
    if dataset not in dataset_paths:
        raise ValueError(f"path to {dataset} not provided. Please add in config.py")
    
    dataset_class = dataset_dict[dataset]
    basedir = dataset_paths[dataset]
    if dataset in transform_funcs:
        eval_transform = transform_funcs[dataset](config.resolution, False, False)
        train_transform = transform_funcs[dataset](config.resolution, True, True)
    else:
        train_transform = None
        eval_transform = None

    train_dataset = dataset_class(basedir, "train", train_transform)
    train_no_aug_dataset = dataset_class(basedir, "train", eval_transform)
    val_dataset = dataset_class(basedir, "val", eval_transform)

    # build the test dataset
    if dataset == 'imagenet-9':
        test_dataset = dataset_class(dataset_paths['imagenet-a'], "test", eval_transform)
    else:
        test_dataset = dataset_class(basedir, "test", eval_transform)
    datasets =  {
            "train": train_dataset,
            "train_no_aug": train_no_aug_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

    if dataset == 'imagenet-bg':
        test_dict = {}
        test_dict["mixed_rand"] = dataset_class(basedir, "mixed_rand", eval_transform)
        test_dict["only_fg"] = dataset_class(basedir, "only_fg", eval_transform)
        test_dict["no_fg"] = dataset_class(basedir, "no_fg", eval_transform)
        datasets.update(test_dict)
    

    if config.split_train < 1.0:
        train_split_path = os.path.join(basedir, f"train_split_{config.split_train:.2f}_seed{config.seed}.pickle")
        if os.path.exists(train_split_path):
            with open(train_split_path, "rb") as f:
                train_indexes1, train_indexes2 = pickle.load(f)
        else:
            train_indexes1, train_indexes2 = get_split_indexes(train_dataset, config.split_train)
            with open(train_split_path, "wb") as f:
                pickle.dump((train_indexes1, train_indexes2), f)
        train_subset1 = dataset_class(basedir, "train", train_transform, train_indexes1, config.seed)
        train_subset2 = dataset_class(basedir, "train", train_transform, train_indexes2, config.seed)

        train_no_aug_subset1 = dataset_class(basedir, "train", eval_transform, train_indexes1, config.seed)
        train_no_aug_subset2 = dataset_class(basedir, "train", eval_transform, train_indexes2, config.seed)
        datasets.update({"train_subset1": train_subset1, 
                        "train_subset2": train_subset2, 
                        "train_no_aug_subset1": train_no_aug_subset1, 
                        "train_no_aug_subset2": train_no_aug_subset2,
                        "train_indexes1": train_indexes1,
                        "train_indexes2": train_indexes2,
                        })
    if config.split_val < 1.0:
        val_split_path = os.path.join(basedir, f"val_split_{config.split_val:.2f}_seed{config.seed}.pickle")
        if os.path.exists(val_split_path):
            with open(val_split_path, "rb") as f:
                val_indexes1, val_indexes2 = pickle.load(f)
        else:
            val_indexes1, val_indexes2 = get_split_indexes(val_dataset, config.split_val)
            with open(val_split_path, "wb") as f:
                pickle.dump((val_indexes1, val_indexes2), f)

        val_subset1 = dataset_class(basedir, "val", eval_transform, val_indexes1, config.seed)
        val_subset2 = dataset_class(basedir, "val", eval_transform, val_indexes2, config.seed)
        if "no_aug" in config.train_split:
            train_val_dataset = dataset_class(basedir, "train", eval_transform)
        else:
            train_val_dataset = dataset_class(basedir, "train", train_transform)
        train_val_dataset.y_array = np.concatenate([train_val_dataset.y_array, val_subset1.y_array])
        train_val_dataset.group_array = np.concatenate([train_val_dataset.group_array, val_subset1.group_array])
        train_val_dataset.confounder_array = np.concatenate([train_val_dataset.confounder_array, val_subset1.confounder_array])
        if config.dataset in ["civilcomments", "multinli"]:
            train_val_dataset.data = torch.cat([train_val_dataset.data, val_subset1.data])
        else:
            train_val_dataset.filename_array = np.concatenate([train_val_dataset.filename_array, val_subset1.filename_array])
        
        datasets.update({"val_subset1": val_subset1, 
                        "val_subset2": val_subset2, 
                        "val_indexes1": val_indexes1,
                        "val_indexes2": val_indexes2,
                        })
        if "no_aug" in config.train_split:
             datasets.update({"train_val_no_aug": train_val_dataset})
        else:
            datasets.update({"train_val": train_val_dataset})
    dataloaders = {}
    for k in datasets:
        if "indexes" in k:
            continue
        if "train" in k and "no_aug" not in k:
            loader = DataLoader(
                datasets[k],
                batch_size=config.batch_size,
                pin_memory=True,
                shuffle=True,
                num_workers=config.num_workers,
            )
        else:
            loader = DataLoader(
                datasets[k],
                batch_size=config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=config.num_workers,
            )
        dataloaders[k] = loader
    
    return datasets, dataloaders

# def get_waterbirds_loader(batch_size, vlm):
#     train_transform = get_transform_cub(
#         target_resolution=(224, 224), train=True, augment_data=True
#     )
#     test_transform = get_transform_cub(
#         target_resolution=(224, 224), train=False, augment_data=False
#     )
#     if vlm == "blip":
#         concept_path = WATERBIRDS_CONCEPT_PATH_BLIP
#         assert "blip" in concept_path, "not blip-generated embeddings"
#     elif vlm == "vit-gpt2":
#         concept_path = WATERBIRDS_CONCEPT_PATH_VIT_GPT2
#         assert "vit-gpt2" in concept_path, "not vit-gpt2-generated embeddings"
#     assert "waterbird" in concept_path, "not from the waterbird dataset"
#     assert "waterbird" in WATERBIRDS_DATA_FOLDER, "WATERBIRDS_DATA_FOLDER is incorrect"
#     trainset = BiasedDataset(
#         basedir=WATERBIRDS_DATA_FOLDER,
#         split="train",
#         transform=train_transform,
#         concept_embed=concept_path
#     )
#     trainset_ref = BiasedDataset(
#         basedir=WATERBIRDS_DATA_FOLDER,
#         split="train",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     train_idx_dataset = IdxDataset(trainset_ref)
#     train_loader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=True,
#         num_workers=4,
#     )
#     idx_train_loader = DataLoader(
#         train_idx_dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     valset = BiasedDataset(
#         basedir=WATERBIRDS_DATA_FOLDER,
#         split="val",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     val_loader = DataLoader(
#         valset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )

#     testset = BiasedDataset(
#         basedir=WATERBIRDS_DATA_FOLDER,
#         split="test",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     test_loader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     return train_loader, idx_train_loader, val_loader, test_loader


# def get_celeba_loader(batch_size, vlm, sampling=False):
#     train_transform = get_transform_cub(
#         target_resolution=(224, 224), train=True, augment_data=True
#     )
#     test_transform = get_transform_cub(
#         target_resolution=(224, 224), train=False, augment_data=False
#     )
#     if vlm == "blip":
#         concept_path = CELEBA_CONCEPT_PATH_BLIP
#         assert "blip" in concept_path, "not blip-generated embeddings"
#     elif vlm == "vit-gpt2":
#         concept_path = CELEBA_CONCEPT_PATH_VIT_GPT2
#         assert "vit-gpt2" in concept_path, "not vit-gpt2-generated embeddings"
#     assert "celeba" in concept_path, "not from the celeba dataset"
#     assert "celeba" in CELEBA_DATA_FOLDER, "CELEBA_DATA_FOLDER is incorrect"
#     trainset = BiasedDataset(
#         basedir=CELEBA_DATA_FOLDER,
#         split="train",
#         transform=train_transform,
#         concept_embed=concept_path
#     )
#     trainset_ref = BiasedDataset(
#         basedir=CELEBA_DATA_FOLDER,
#         split="train",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     train_idx_dataset = IdxDataset(trainset_ref)
#     train_loader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=True,
#         num_workers=4,
#     )
#     if sampling == True:
#         idx_train_loader = DataLoader(
#             train_idx_dataset,
#             batch_size=batch_size,
#             sampler=RandomSampler(
#                 train_idx_dataset, num_samples=batch_size*300),
#             pin_memory=True,
#             num_workers=4,
#         )
#     else:
#         idx_train_loader = DataLoader(
#             train_idx_dataset,
#             batch_size=batch_size,
#             pin_memory=True,
#             num_workers=4,
#         )
#     valset = BiasedDataset(
#         basedir=CELEBA_DATA_FOLDER,
#         split="val",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     val_loader = DataLoader(
#         valset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )

#     testset = BiasedDataset(
#         basedir=CELEBA_DATA_FOLDER,
#         split="test",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     test_loader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     return train_loader, idx_train_loader, val_loader, test_loader


# def get_nico_loader(batch_size, vlm="vit-gpt2"):
#     train_transform = get_transform_nico(train=True, augment_data=True)
#     test_transform = get_transform_nico(train=False, augment_data=False)
#     if vlm == "blip":
#         concept_path = NICO_CONCEPT_PATH_BLIP
#         assert "blip" in concept_path, "not blip-generated embeddings"
#     elif vlm == "vit-gpt2":
#         concept_path = NICO_CONCEPT_PATH_VIT_GPT2
#         assert "vit-gpt2" in concept_path, "not vit-gpt2-generated embeddings"
#     assert "NICO" in concept_path, "not from the nico dataset"
#     assert "NICO" in NICO_DATA_FOLDER, "NICO_DATA_FOLDER is incorrect"
#     trainset = NICO_dataset(
#         basedir=NICO_DATA_FOLDER,
#         split="train",
#         balance_factor=0.02,
#         training_dist=TRAINING_DIST,
#         transform=train_transform,
#         concept_embed=concept_path
#     )
#     trainset_ref = NICO_dataset(
#         basedir=NICO_DATA_FOLDER,
#         split="train",
#         balance_factor=0.02,
#         training_dist=TRAINING_DIST,
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     train_idx_dataset = IdxDataset(trainset_ref)
#     train_loader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=True,
#         num_workers=4,
#     )
#     idx_train_loader = DataLoader(
#         train_idx_dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     valset = NICO_dataset(
#         basedir=NICO_DATA_FOLDER,
#         split="val",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     val_loader = DataLoader(
#         valset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )

#     testset = NICO_dataset(
#         basedir=NICO_DATA_FOLDER,
#         split="test",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     test_loader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     return train_loader, idx_train_loader, val_loader, test_loader


# def get_imagenet9_loader(batch_size, vlm="vit-gpt2"):
#     train_transform = get_imagenet_transform(train=True, augment_data=True)
#     test_transform = get_imagenet_transform(train=False, augment_data=False)
#     if vlm == "blip":
#         concept_path = IMAGENET9_CONCEPT_PATH_BLIP
#         assert "blip" in concept_path, "not blip-generated embeddings"
#     elif vlm == "vit-gpt2":
#         concept_path = IMAGENET9_CONCEPT_PATH_VIT_GPT2
#         assert "vit-gpt2" in concept_path, "not vit-gpt2-generated embeddings"
#     assert "imagenet" in concept_path, "not from the imagenet dataset"
#     assert "imagenet" in IMAGENET9_DATA_FOLDER, "IMAGENET9_DATA_FOLDER is incorrect"
#     trainset = ImageNet9(
#         basedir=IMAGENET9_DATA_FOLDER,
#         split="train",
#         transform=train_transform,
#         concept_embed=concept_path
#     )
#     trainset_ref = ImageNet9(
#         basedir=IMAGENET9_DATA_FOLDER,
#         split="train",
#         transform=test_transform,
#         concept_embed=concept_path
#     )
#     train_idx_dataset = IdxDataset(trainset_ref)
#     train_loader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=True,
#         num_workers=12,
#     )
#     idx_train_loader = DataLoader(
#         train_idx_dataset,
#         # sampler=RandomSampler(train_idx_dataset, num_samples=batch_size*2),
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=12,
#     )
#     valset = ImageNet9(
#         basedir=IMAGENET9_DATA_FOLDER,
#         split="val",
#         transform=test_transform,
#         concept_embed=concept_path,
#         cluster_file=IMAGENET9_VAL_CLUSTERS
#     )
#     val_loader = DataLoader(
#         valset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=12,
#     )

#     testset = ImageNetA(
#         basedir=IMAGENETA_DATA_FOLDER,
#         transform=test_transform,
#     )
#     test_loader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=12,
#     )
#     return train_loader, idx_train_loader, val_loader, test_loader


# def get_civilcomments_loader(batch_size):
#     assert "civil" in CIVIL_DATA_FOLDER, "CIVIL_DATA_FOLDER is incorrect"
#     trainset = CivilCommentsDataset(
#         basedir=CIVIL_DATA_FOLDER,
#         split="train",
#     )
#     trainset_ref = CivilCommentsDataset(
#         basedir=CIVIL_DATA_FOLDER,
#         split="train",
#     )
#     train_idx_dataset = IdxDataset(trainset_ref)
#     train_loader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=True,
#         num_workers=4,
#     )
   
#     idx_train_loader = DataLoader(
#         train_idx_dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     valset = CivilCommentsDataset(
#         basedir=CIVIL_DATA_FOLDER,
#         split="val",
#     )
#     val_loader = DataLoader(
#         valset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )

#     testset = CivilCommentsDataset(
#         basedir=CIVIL_DATA_FOLDER,
#         split="test",
#     )
#     test_loader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     return train_loader, idx_train_loader, val_loader, test_loader



# def get_multinli_loader(batch_size):
#     assert "multinli" in MULTINLI_DATA_FOLDER, "MULTINLI_DATA_FOLDER is incorrect"
#     trainset = MultiNLIDataset(
#         basedir=MULTINLI_DATA_FOLDER,
#         split="train",
#     )
#     trainset_ref = MultiNLIDataset(
#         basedir=MULTINLI_DATA_FOLDER,
#         split="train",
#     )
#     train_idx_dataset = IdxDataset(trainset_ref)
#     train_loader = DataLoader(
#         trainset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=True,
#         num_workers=4,
#     )
   
#     idx_train_loader = DataLoader(
#         train_idx_dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     valset = MultiNLIDataset(
#         basedir=MULTINLI_DATA_FOLDER,
#         split="val",
#     )
#     val_loader = DataLoader(
#         valset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )

#     testset = MultiNLIDataset(
#         basedir=MULTINLI_DATA_FOLDER,
#         split="test",
#     )
#     test_loader = DataLoader(
#         testset,
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=4,
#     )
#     return train_loader, idx_train_loader, val_loader, test_loader

# def get_dataloader(dataset, batch_size, vlm="vit-gpt2", sampling=False):
#     if dataset == "waterbirds":
#         loaders = get_waterbirds_loader(batch_size, vlm)
#     elif dataset == "celeba":
#         loaders = get_celeba_loader(batch_size, vlm, sampling)
#     elif dataset == "nico":
#         loaders = get_nico_loader(batch_size, vlm)
#     elif dataset == "imagenet-9":
#         loaders = get_imagenet9_loader(batch_size, vlm)
#     elif dataset == "multinli":
#         loaders = get_multinli_loader(batch_size)
#     elif dataset == "civilcomments":
#         loaders = get_civilcomments_loader(batch_size)
#     else:
#         raise ValueError(
#             r"dataset must be from {waterbirds, celeba, bar, imagenet-a, imagenet-9}")
#     return loaders
