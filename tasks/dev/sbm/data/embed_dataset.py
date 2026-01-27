import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.special import softmax
from collections import Counter
# from sklearn.preprocessing import StandardScaler


def embedding_ops(
    embedding1,
    label1,
    scores1,
    embedding2,
    label2,
    scores2
):
    embedding1 = embedding1.copy()
    embedding2 = embedding2.copy()

    temp = embedding1.copy()
    op = np.random.choice(1)

    if op == 0:
        embedding1[spurious2] = embedding1[spurious2]
        embedding2[spurious1] = temp[spurious1]
    elif op == 1:
        embedding1[spurious1] = 0
        embedding2[spurious2] = 0
    elif op == 2:
        embedding1[core1] = 0
        embedding2[core2] = 0
    return embedding1, label1, embedding2, label2, op


def collate_fn(data: list[tuple[np.ndarray, int, np.ndarray, np.ndarray]], spu_dims, operation):
    """Transform a list of tuples into a batch

    Args:
        data (list[tuple[np.ndarray, int, np.ndarray, np.ndarray]]): a list of tuples sampled from a dataset

    Returns:
        tensor: a list of tensor data
    """
    # data = a list of tuples
    batch_size = len(data)
    batch1 = [data[i] for i in range(0, batch_size, 2)]
    batch2 = [data[i] for i in range(1, batch_size, 2)]

    embeds = []
    labels = []
    groups = []

    for i in range(len(batch1)):
        embed1, label1, group1 = batch1[i]
        embed2, label2, group2 = batch2[i]

        # if np.random.rand() > 0.5:
        #     spu_dim1 = spu_dims[label1]
        #     spu_dim2 = spu_dims[label2]
        #     # min_num_dims = min(len(spu_dim1), len(spu_dim2))
        #     # np.random.shuffle(spu_dim1)
        #     # np.random.shuffle(spu_dim2)
        #     # spu_dim1 = spu_dim1[0:min_num_dims]
        #     # spu_dim2 = spu_dim2[0:min_num_dims]
        #     embed1_ = embed1 * 1.0
        #     embed2_ = embed2 * 1.0

        #     embed1_[spu_dim1] = embed2[spu_dim2]
        #     embed2_[spu_dim2] = embed1[spu_dim1]
        # else:
        embed1_ = embed1 * 1.0
        embed2_ = embed2 * 1.0


        embeds.append(embed1_)
        embeds.append(embed2_)

        labels.append(label1)
        labels.append(label2)

        groups.append(group1)
        groups.append(group2)

    embeds = torch.stack(embeds, dim=0)
    labels = torch.tensor(labels)
    groups = torch.tensor(groups)

    return embeds, labels, groups


class EmbedSampler:
    def __init__(self, labels, n_batches, batch_size, pred_probs=None, temp=10):
        self.labels = labels
        self.classes = np.unique(labels)
        all_indexes = np.arange(len(labels))
        self.cls_indexes = {c:all_indexes[labels == c] for c in self.classes}
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.total_data = len(labels)
        # self.pred_probs = [1-pred_probs[labels == c, 0]
        #                    for c in self.classes]
        # self.sel_indexes = [self.cls_indexes[c][np.argsort(
        #     1-self.pred_probs[c])[0:200]] for c in self.classes]
        # self.pred_probs = [self.pred_probs[i]/self.pred_probs[i].sum()
        #                    for i in range(len(self.pred_probs))]
        # self.pred_probs = [softmax(self.pred_probs[i]*temp)
        #                    for i in range(len(self.pred_probs))]

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for n in range(self.n_batches):
            # batch = []
            # num_half = self.batch_size // 2
            # sel_classes = np.random.choice(
            #     self.classes, num_half, replace=True)
            # for c in sel_classes:
            #     batch.append(np.random.choice(
            #         self.cls_indexes[c], 2, p=self.pred_probs[c], replace=False))
            #     # batch.append(np.random.choice(
            #     #     self.cls_indexes[c], 2, replace=False))

            # batch = np.concatenate(batch)
            # np.random.shuffle(batch)

            # select class pairs
            num_half = self.batch_size // 2
            sel_classes = []
            for i in range(num_half):
                sel_cls = np.random.choice(
                    self.classes, 2, replace=True)
                sel_classes.append(sel_cls)
            sel_classes = np.concatenate(sel_classes)
            np.random.shuffle(sel_classes)
            counter = Counter(sel_classes)
            batch = np.zeros_like(sel_classes)
            for c in counter:
                # samples = np.random.choice(
                #     self.cls_indexes[c], counter[c], p=self.pred_probs[c], replace=False)
                samples = np.random.choice(
                    self.cls_indexes[c], counter[c], replace=True)
                batch[sel_classes == c] = samples

            yield torch.tensor(batch, dtype=torch.long)


class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 0
    def fit(self, x):
        num, dim = x.shape
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
    def transform(self, x):
        return (x - self.mean.reshape(1,-1))/(self.std.reshape(1,-1)+1e-10)

class EmbedDataset(Dataset):
    def __init__(
        self,
        embeddings,
        labels,
        groups,
        split,
        confounder_array,
        indexes=None,
        standarize=False,
        scaler = None,
    ):
        if indexes is not None:
            embeddings = embeddings[indexes]
            labels = labels[indexes]
            groups = groups[indexes]
            confounder_array = confounder_array[indexes]
            self.indexes = indexes
        self.n_classes = len(np.unique(labels))
        if standarize and (split == "train" or "subset1" in split):
            self.scaler = Scaler()
            self.scaler.fit(embeddings)
        else:
            self.scaler = scaler
        if self.scaler:
            self.embeddings = self.scaler.transform(embeddings)
        else:
            self.embeddings = embeddings

        self.y_array = torch.tensor(labels, dtype=torch.long)
        self.group_array = torch.tensor(groups, dtype=torch.long)
        self.confounder_array = torch.tensor(confounder_array, dtype=torch.long)
    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        embed, label, group = self.embeddings[idx], self.y_array[idx], self.group_array[idx]
        embed_ = torch.tensor(embed.copy(), dtype=torch.float)
        attr = self.confounder_array[idx]
        return embed_, label, group, attr



