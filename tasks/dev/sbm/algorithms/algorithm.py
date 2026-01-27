import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
from data.data_utils import prepare_data
from .utils import extract_feature_info
from data.embed_dataset import EmbedDataset
from data.sampler import GroupBalancedSampler, ClassBalancedSampler, RandomSampler, JointRandomSampler, ShortcutSampler
from torch.utils.data import DataLoader
from utils import log
from data.data_utils import IdxDataset, MisDataset, JointDataset, PredDataset, SubsetDataset

class Algorithm(torch.nn.Module):
    """
    Base class for algorithms
    """
    def __init__(self, config):
        super(Algorithm, self).__init__()
        self.config = config
        self.datasets, self.dataloaders = prepare_data(config)

    def _init_training(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def train(self, config):
        raise NotImplementedError
    
    def separate_samples_loss(self, dataset, mis_ratio=0):
        """
        For each class, separate samples into low-loss group (1-mis_ratio) and high-loss group (mis_ratio).
        """
        self.model.eval()
        index_dict = {c:[] for c in range(self.n_classes)}
        loss_dict = {c:[] for c in range(self.n_classes)}
        pred_dict = {c:[] for c in range(self.n_classes)}
        res_dict = {c:[] for c in range(self.n_classes)}
        ent_dict = {c:[] for c in range(self.n_classes)}
        all_pred_labels = []
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        index_dataset = IdxDataset(dataset)
        self.index_loader = DataLoader(
                            index_dataset,
                            batch_size=self.config.batch_size,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=self.config.num_workers,
                        )

        with torch.no_grad():
            for idx, x, y, groups, attrs in tqdm(self.index_loader, desc="Identifying shortcuts", leave=False):
                if self.config.last_layer:
                    logits = self.model.fc(x.to(self.device))
                else:
                    logits = self.model(x.to(self.device))
                losses = criterion(logits, y.to(self.device)).detach().cpu()
                res = (torch.argmax(logits, dim=1) == y.to(self.device))
                ent = (- F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1).detach().cpu()
                pred_labels = torch.argmax(logits, dim=1)
                # ent = torch.gather(F.softmax(logits, dim=-1),1,pred_labels.unsqueeze(1)).squeeze()
                all_pred_labels.append(pred_labels.detach().cpu())
                for i in range(len(y)):
                    loss_dict[y[i].item()].append(losses[i].item())
                    index_dict[y[i].item()].append(idx[i].item())
                    pred_dict[y[i].item()].append(pred_labels[i].item())
                    res_dict[y[i].item()].append(res[i].item())
                    ent_dict[y[i].item()].append(ent[i].item())
        all_pred_labels = torch.cat(all_pred_labels)
        mis_index_dict = {}
        cor_index_dict = {}
        mis_pred_label_dict = {}
        cor_pred_label_dict = {}

        for c in loss_dict:
            losses = torch.tensor(loss_dict[c])
            ents = torch.tensor(ent_dict[c])
            cls_indexes = torch.tensor(index_dict[c])
            res = torch.tensor(res_dict[c])
            pred_labels = torch.tensor(pred_dict[c])
            if mis_ratio > 0:
                sort_indexes = torch.argsort(losses)
                
                sel_num = int(len(sort_indexes) * mis_ratio)
                # cor_sel_indexes = torch.arange(len(losses))[res]
                # mis_sel_indexes = torch.arange(len(losses))[~res]
                # if len(mis_sel_indexes) == 0:
                num = int(len(sort_indexes) * 0.5)
                cor_sel_indexes = sort_indexes[0:num]
                mis_sel_indexes = sort_indexes[-num:]

                cls_cor_indexes = cls_indexes[cor_sel_indexes]
                cls_mis_indexes = cls_indexes[mis_sel_indexes]
                cor_pred_labels = pred_labels[cor_sel_indexes]
                mis_pred_labels = pred_labels[mis_sel_indexes]
                


                # cor_index_dict[c] = cls_cor_indexes
                # mis_index_dict[c] = cls_mis_indexes
                # cor_pred_label_dict[c] = pred_labels[cor_sel_indexes]
                # mis_pred_label_dict[c] = pred_labels[mis_sel_indexes]

                idx_ent_cor = torch.argsort(ents[cor_sel_indexes])
                idx_ent_mis = torch.argsort(ents[mis_sel_indexes])
                cor_index_dict[c] = cls_cor_indexes[idx_ent_cor[0:sel_num]]
                mis_index_dict[c] = cls_mis_indexes[idx_ent_mis[0:sel_num]]
                cor_pred_label_dict[c] = cor_pred_labels[idx_ent_cor[0:sel_num]]
                mis_pred_label_dict[c] = mis_pred_labels[idx_ent_mis[0:sel_num]]
            else:
                cls_cor_indexes = cls_indexes[res]
                cls_mis_indexes = cls_indexes[~res]

                cor_index_dict[c] = cls_cor_indexes
                mis_index_dict[c] = cls_mis_indexes
                cor_pred_label_dict[c] = pred_labels[res]
                mis_pred_label_dict[c] = pred_labels[~res]

                # idx_ent_cor = torch.argsort(ents[res])
                # idx_ent_mis = torch.argsort(ents[~res])
                # ratio = len(idx_ent_cor) / len(cls_indexes)
                # sel_num_cor = int(len(cls_cor_indexes) * mis_ratio)
                # sel_num_mis = int(len(cls_mis_indexes) * mis_ratio)
                # cor_index_dict[c] = cls_cor_indexes[idx_ent_cor[0:sel_num_cor]]
                # mis_index_dict[c] = cls_mis_indexes[idx_ent_mis[0:sel_num_mis]]
                # cor_pred_label_dict[c] = pred_labels[idx_ent_cor[0:sel_num_cor]]
                # mis_pred_label_dict[c] = pred_labels[idx_ent_mis[0:sel_num_mis]]
        return mis_index_dict, cor_index_dict, mis_pred_label_dict, cor_pred_label_dict, all_pred_labels


    def update_train_loaders(self, train_split):
        train_split = self._get_split(train_split)
        mis_index_dict, cor_index_dict, mis_pred_label_dict, cor_pred_label_dict, all_pred_labels = self.separate_samples_loss(self.datasets[train_split], self.config.mis_ratios[0])
        mis_indexes = torch.cat([mis_index_dict[c] for c in mis_index_dict])
        cor_indexes = torch.cat([cor_index_dict[c] for c in cor_index_dict])
        mis_pred_labels = torch.cat([mis_pred_label_dict[c] for c in mis_pred_label_dict])
        cor_pred_labels = torch.cat([cor_pred_label_dict[c] for c in cor_pred_label_dict])
        
        mis_str_stats = ' '.join([f'class{c}:{len(mis_index_dict[c])}' for c in mis_index_dict])
        cor_str_stats = ' '.join([f'class{c}:{len(cor_index_dict[c])}' for c in cor_index_dict])
        log(f"mis_indexes: {mis_str_stats}")
        log(f"cor_indexes: {cor_str_stats}")
        if self.config.use_shortcutwalk_dataset:
            self.total_data = len(mis_indexes)+len(cor_indexes)
        else:
            self.total_data = len(self.datasets[train_split])
        self.config.num_batches = (len(mis_pred_labels) + len(cor_pred_labels)) // self.config.batch_size + 1
        
        train_dataset = self.datasets[train_split]
        shortcut_sampler = ShortcutSampler(mis_index_dict, cor_index_dict, mis_pred_label_dict, cor_pred_label_dict, self.config.num_batches, self.config.batch_size)
        mis_subset = MisDataset(train_dataset, mis_pred_labels, mis_indexes)
        cor_subset = MisDataset(train_dataset, cor_pred_labels, cor_indexes)
        pred_subset = PredDataset(train_dataset, all_pred_labels)

        joint_dataset = JointDataset([mis_subset, cor_subset])
        shortcutwalk_dataset = SubsetDataset(train_dataset, torch.cat([mis_indexes, cor_indexes]))
        self.datasets.update({"embed_train_joint":joint_dataset})
        self.datasets.update({"embed_shortcutwalk_train":shortcutwalk_dataset})
        

        shortcutwalk_loader = DataLoader(shortcutwalk_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)

        mis_batch_sampler = ClassBalancedSampler(train_dataset.y_array[mis_indexes], self.config.num_batches, self.config.batch_size)
        cor_batch_sampler = ClassBalancedSampler(train_dataset.y_array[cor_indexes], self.config.num_batches, self.config.batch_size)

        mis_batch_sampler_random = RandomSampler(train_dataset.y_array[mis_indexes], self.config.num_batches, self.config.batch_size)
        cor_batch_sampler_random = RandomSampler(train_dataset.y_array[cor_indexes], self.config.num_batches, self.config.batch_size)
        joint_sampler = JointRandomSampler(train_dataset.y_array[mis_indexes], train_dataset.y_array[cor_indexes], self.config.num_batches, self.config.batch_size)

        mis_subset_loader_cb = DataLoader(mis_subset, batch_sampler=mis_batch_sampler, pin_memory=True, num_workers=self.config.num_workers)
        cor_subset_loader_cb = DataLoader(cor_subset, batch_sampler=cor_batch_sampler, pin_memory=True, num_workers=self.config.num_workers)

        mis_subset_loader = DataLoader(mis_subset, batch_sampler=mis_batch_sampler_random, pin_memory=True, num_workers=self.config.num_workers)
        cor_subset_loader = DataLoader(cor_subset, batch_sampler=cor_batch_sampler_random, pin_memory=True, num_workers=self.config.num_workers)

        joint_loader = DataLoader(joint_dataset, batch_sampler=joint_sampler, pin_memory=True, num_workers=self.config.num_workers)
        shortcut_loader = DataLoader(pred_subset, batch_sampler=shortcut_sampler, pin_memory=True, num_workers=self.config.num_workers)
        
        if self.config.last_layer:
            train_loaders = {
                "embed_train_mis": mis_subset_loader,
                "embed_train_cor": cor_subset_loader,
                "embed_train_mis_cb": mis_subset_loader_cb,
                "embed_train_cor_cb": cor_subset_loader_cb,
                "embed_train_joint": joint_loader,
                "embed_shortcut_train": shortcut_loader,
                "embed_shortcutwalk_train": shortcutwalk_loader,
            }
        else:
            train_loaders = {
                "train_mis":mis_subset_loader,
                "train_cor":cor_subset_loader,
                "train_mis_cb":mis_subset_loader_cb,
                "train_cor_cb":cor_subset_loader_cb,
                "train_joint": joint_loader,
                "shortcut_train": shortcut_loader,
                "shortcutwalk_train": shortcutwalk_loader,
            }
        self.dataloaders.update(train_loaders)
    
    def _get_split(self, split):
        if self.config.last_layer:
            return "embed_" + split
        else:
            return split

    def evaluate(self, split):
        loader = self.dataloaders[split]
        eval_metrics = ["acc", "worst_cls_acc", "worst_group_acc", "avg_cls_diff", "unbiased_group_acc"]
        self.model.eval()
        pred_res = []
        labels = []
        groups = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=split, leave=False):
                x, y, g, p = batch
                x, y = x.to(self.device), y.to(self.device)
                if "embed" in split:
                    logits = self.model.fc(x)
                else:
                    logits = self.model(x)
                res = (torch.argmax(logits, dim=-1) == y).float()
                pred_res.append(res.cpu())
                labels.append(y.cpu())
                groups.append(g)
        pred_res = torch.cat(pred_res).numpy()
        labels = torch.cat(labels).numpy()
        groups = torch.cat(groups).numpy()
        acc = pred_res.mean()
        n_classes = len(np.unique(labels))
        
        cls_accs = np.array([pred_res[labels==c].mean() for c in range(n_classes)])
        cls_diffs = []
        for c in range(n_classes):
            for m in range(c+1, n_classes):
                cls_diffs.append(abs(cls_accs[c] - cls_accs[m]))
        cls_diffs = np.array(cls_diffs)
        avg_cls_diff = cls_diffs.mean()
        worst_cls_acc = cls_accs.min()
        if len(groups.shape) == 1:
            groups = groups.reshape(-1,1)
        
        worst_group_acc = 0
        unbiased_group_acc = 0
        for g_iter in range(groups.shape[1]):
            n_groups = len(np.unique(groups[:,g_iter]))
            group_accs = []
            for g in range(n_groups):
                pred_group = pred_res[groups[:,g_iter]==g]
                if len(pred_group) > 0:
                    group_accs.append(pred_group.mean())
            group_accs = np.array(group_accs)
            worst_group_acc += group_accs.min()
            unbiased_group_acc += group_accs.mean()
        worst_group_acc /= groups.shape[1]
        unbiased_group_acc /= groups.shape[1]

        results = [acc, worst_cls_acc, worst_group_acc, avg_cls_diff, unbiased_group_acc]
        result_dict = {f"{split}_{k}":results[i] for i, k in enumerate(eval_metrics)}
        return result_dict

    def get_embed_loaders(self, train_split):
        # precompute embeddings
        train_split = "embed_" + train_split
        erm_folder = os.path.dirname(self.config.erm_model)
        embeds_train, labels_train, pred_probs_train, groups_train = extract_feature_info(self.model,self.dataloaders["train_no_aug"], erm_folder, "train_no_aug", device=self.device)
        embeds_val, labels_val, pred_probs_val, groups_val = extract_feature_info(self.model,self.dataloaders["val"], erm_folder, "val", device=self.device)
        embeds_test, labels_test, pred_probs_test, groups_test = extract_feature_info(self.model,self.dataloaders["test"], erm_folder, "test", device=self.device)
        
        embed_datasets = {}
        if train_split == "embed_train_no_aug":
            embed_datasets["embed_train_no_aug"] = EmbedDataset(embeds_train, labels_train, groups_train, "train_no_aug", self.datasets["train_no_aug"].confounder_array, standarize=self.config.process_embeddings)
            scaler = embed_datasets["embed_train_no_aug"].scaler
            embed_datasets["embed_val"] = EmbedDataset(embeds_val, labels_val, groups_val, "val", self.datasets["val"].confounder_array, scaler=scaler)
            
        elif self.config.split_train < 1.0:
            if train_split == "embed_train_no_aug_subset1":
                embed_datasets["embed_train_no_aug_subset1"] = EmbedDataset(embeds_train, labels_train, groups_train, "train_no_aug_subset1", self.datasets["train_no_aug"].confounder_array, self.datasets["train_indexes1"], standarize=self.config.process_embeddings)
                scaler = embed_datasets["embed_train_no_aug_subset1"].scaler
                embed_datasets["embed_train_no_aug_subset2"] = EmbedDataset(embeds_train, labels_train, groups_train, "train_no_aug_subset2", self.datasets["train_no_aug"].confounder_array, self.datasets["train_indexes2"], scaler=scaler)
            if train_split == "embed_train_no_aug_subset2":
                embed_datasets["embed_train_no_aug_subset2"] = EmbedDataset(embeds_train, labels_train, groups_train, "train_no_aug_subset2", self.datasets["train_no_aug"].confounder_array, self.datasets["train_indexes2"], standarize=self.config.process_embeddings)
                scaler = embed_datasets["embed_train_no_aug_subset2"].scaler
                embed_datasets["embed_train_no_aug_subset1"] = EmbedDataset(embeds_train, labels_train, groups_train, "train_no_aug_subset1", self.datasets["train_no_aug"].confounder_array, self.datasets["train_indexes1"], scaler=scaler)
            embed_datasets["embed_val"] = EmbedDataset(embeds_val, labels_val, groups_val, "val", self.datasets["val"].confounder_array, scaler=scaler)
        elif self.config.split_val < 1.0:
            if train_split == "embed_val_subset1": 
                embed_datasets["embed_val_subset1"] = EmbedDataset(embeds_val, labels_val, groups_val, "val_subset1", self.datasets["val"].confounder_array, self.datasets["val_indexes1"], standarize=self.config.process_embeddings)
                scaler = embed_datasets["embed_val_subset1"].scaler
                embed_datasets["embed_val_subset2"] = EmbedDataset(embeds_val, labels_val, groups_val, "val_subset2", self.datasets["val"].confounder_array, self.datasets["val_indexes2"], scaler=scaler)
            elif train_split == "embed_train_val_subset1":
                num_train = len(self.datasets["val_indexes1"])
                train_indexes = np.arange(len(embeds_train))
                np.random.shuffle(train_indexes)
                train_indexes = train_indexes[0:num_train]
                embeds_train_val = np.concatenate([embeds_train[train_indexes], embeds_val[self.datasets["val_indexes1"]]])
                labels_train_val = np.concatenate([labels_train[train_indexes], labels_val[self.datasets["val_indexes1"]]])
                groups_train_val = np.concatenate([groups_train[train_indexes], groups_val[self.datasets["val_indexes1"]]])
                conf_train_val = np.concatenate([self.datasets["train"].confounder_array[train_indexes], self.datasets["val_subset1"].confounder_array])
                embed_datasets["embed_train_val_subset1"] = EmbedDataset(embeds_train_val, labels_train_val, groups_train_val, "embed_train_val_subset1", conf_train_val, standarize=self.config.process_embeddings)
                scaler = embed_datasets["embed_train_val_subset1"].scaler
                embed_datasets["embed_val_subset2"] = EmbedDataset(embeds_val, labels_val, groups_val, "val_subset2", self.datasets["val"].confounder_array, self.datasets["val_indexes2"], scaler=scaler)

        
        embed_datasets["embed_test"] = EmbedDataset(embeds_test, labels_test, groups_test, "test", self.datasets["test"].confounder_array, scaler=scaler)
        self.datasets.update(embed_datasets)

        
        embed_test_loader = DataLoader(embed_datasets["embed_test"], shuffle=False, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_workers)
        embed_loaders = {
            "embed_test":embed_test_loader
        }
        if "train" in train_split:
            embed_val_loader = DataLoader(embed_datasets["embed_val"], shuffle=False, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_workers)
            embed_loaders.update({"embed_val":embed_val_loader})
        elif self.config.split_val < 1.0:
            embed_val_loader2 = DataLoader(embed_datasets["embed_val_subset2"], shuffle=False, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_workers)
            if train_split == "embed_val_subset1": 
                embed_val_loader1 = DataLoader(embed_datasets["embed_val_subset1"], shuffle=False, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_workers)
                embed_loaders.update(
                    {"embed_val_subset1": embed_val_loader1, "embed_val_subset2": embed_val_loader2}
                )
            elif train_split == "embed_train_val_subset1":
                embed_train_val_loader = DataLoader(embed_datasets["embed_train_val_subset1"], shuffle=False, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_workers)
                embed_loaders.update({"embed_train_val_subset1":embed_train_val_loader, "embed_val_subset2":embed_val_loader2})
        train_dataset = embed_datasets[train_split]

        if self.config.group_balanced:
            train_sampler = GroupBalancedSampler(train_dataset.group_array, self.config.num_batches, self.config.batch_size)
            embed_train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=self.config.num_workers)
            embed_loaders[train_split] = embed_train_loader
        elif self.config.class_balanced and (not self.config.group_balanced):
            train_sampler = ClassBalancedSampler(train_dataset.y_array, self.config.num_batches, self.config.batch_size)
            embed_train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=self.config.num_workers)
            embed_loaders[train_split] = embed_train_loader
        else:
            embed_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config.batch_size, pin_memory=True, num_workers=self.config.num_workers)
            embed_loaders[train_split] = embed_train_loader

        self.dataloaders.update(embed_loaders)
        return embed_loaders

    def load_check_point(self, check_point):
        if not os.path.exists(check_point):
            raise ValueError(f"check point file {check_point} does not exist")
        saved_dict = torch.load(check_point, map_location="cpu")
        return saved_dict

    
