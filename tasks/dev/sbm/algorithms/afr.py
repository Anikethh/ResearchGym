import torch
from .algorithm import Algorithm
from data.data_utils import prepare_data
from tqdm import tqdm
from utils import AverageMeter, BestMetric, Timer, time_str, log
from data.embed_dataset import EmbedDataset
from .utils import init_optimizer, init_scheduler
from .register import register_algorithm
from data.sampler import GroupBalancedSampler, ClassBalancedSampler
from data.data_utils import IdxDataset
import numpy as np
import os
from torch.utils.data import DataLoader
from models.classifier import Classifier

class Model(torch.nn.Module):
    def __init__(self, w0, b0):
        super().__init__()
        self.w0 = w0
        self.b0 = b0
        self.linear = torch.nn.Linear(w0.shape[1], w0.shape[0], bias=True)
        
    def forward(self, x):
        y_old = x @ self.w0.t() + self.b0
        y_new = self.linear(x)
        return y_old + y_new

def wxe_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    l = weights * ce
    return l.sum()


     
@register_algorithm("afr")
class AFR(Algorithm):
    def __init__(self, config):
        super(AFR, self).__init__(config)
        self._init_model()
        self._init_training()
        
    def _init_model(self):
        self.n_classes = self.datasets["train"].n_classes
        self.device = f"cuda:{self.config.gpu}"
        self.model = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        if self.config.check_point:
            log(f"loading the model checkpoint from {self.config.erm_model}")
            if len(self.config.erm_model) > 0:
                log(f"ignoring the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.check_point)
            self.model.load_state_dict(saved_dict["model_sd"])
        
        if len(self.config.check_point) == 0 and len(self.config.erm_model) > 0:
            log(f"loading the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.erm_model)
            self.model.load_state_dict(saved_dict["model_sd"])

        self.model.to(self.device)
        afr_model = Model(self.model.fc.weight.detach(), self.model.fc.bias.detach())
        afr_model.to(self.device)
        del self.model.fc
        self.model.fc = afr_model


    def _init_training(self):
        self.optimizer_cls = init_optimizer(self.model.fc, self.config.optimizer_cls, self.config.optimizer_cls_kwargs)
        self.scheduler_cls = None
      
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.config.split_val < 1.0:
            self.sel_metrics = [("embed_val_subset2_acc", True), ("embed_val_subset2_worst_cls_acc", True), ("embed_val_subset2_worst_group_acc", True), ("embed_val_subset2_avg_cls_diff", False)]
        else:
            self.sel_metrics = [("embed_val_acc", True), ("embed_val_worst_cls_acc", True), ("embed_val_worst_group_acc", True), ("embed_val_avg_cls_diff", False)]
        self.best_meters = {m:BestMetric(max_val) for m, max_val in self.sel_metrics}


    def compute_afr_weights(self, train_loader, gamma, balance_classes, group_uniform):
        class_label = []
        weights = []
        groups = []

        with torch.no_grad():
            for batch in tqdm(train_loader, leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device) 
                erm_logits = self.model.fc(x)
                class_label.append(y.cpu())
                groups.append(g.cpu())
                p = erm_logits.softmax(-1)
                y_onehot = torch.zeros_like(erm_logits).scatter_(-1, y.unsqueeze(-1), 1)
                p_true = (p * y_onehot).sum(-1)
                weights.append((-gamma * p_true).exp())
        weights = torch.cat(weights)
        class_label = torch.cat(class_label)
        groups = torch.cat(groups)
        n_classes = torch.unique(class_label).numel()
        if balance_classes:
            class_count = []
            for y in range(n_classes):
                class_count.append((class_label == y).sum())
            for y in range(1, n_classes):
                weights[class_label == y] *= class_count[0] / class_count[y]
        weights /= weights.sum()
        if group_uniform:
            # uniform total weights per group
            group_counts = torch.bincount(groups)
            for g in range(len(group_counts)):
                weights[groups == g] = 1 / group_counts[g] / len(group_counts)
        return weights
    
    def train(self, output_dir, split="train"):
        timer = Timer()
        criterion = wxe_fn
        embed_loaders = self.get_embed_loaders(split)
        if self.config.use_shortcutwalk_dataset:
            self.update_train_loaders(split)
            train_loader = self.dataloaders["embed_shortcutwalk_train"]
        else:
            train_loader = embed_loaders["embed_"+split]
        idxdataset = IdxDataset(train_loader.dataset)
        idx_loader = DataLoader(idxdataset, batch_size=train_loader.batch_size, shuffle=False)
        weights = self.compute_afr_weights(idx_loader, self.config.afr_gamma, self.config.balance_classes, self.config.group_uniform)
        
        best_val = 0.0
        for epoch in range(1, self.config.afr_epochs+1):
            self.model.fc.train()
            
            epoch_meters = {k:AverageMeter() for k in ["loss", "acc"]}
            if self.scheduler_cls:
                curr_lr = self.scheduler_cls.get_last_lr()[0]
            else:
                curr_lr = self.config.optimizer_cls_kwargs["lr"]


            for batch in tqdm(idx_loader, desc=split, leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer_cls.zero_grad()
                logits = self.model.fc(x)
                loss = criterion(logits, y, weights[idx])
                reg = self.model.fc.linear.weight.pow(2).sum() + self.model.fc.linear.bias.pow(2).sum()
                loss += self.config.afr_reg_coeff * reg
                loss.backward()
                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.fc.parameters(), 1.)
                self.optimizer_cls.step()
             
                epoch_meters["loss"].update(loss.item(), x.size(0))
                acc = (torch.argmax(logits, dim=-1) == y).sum() / len(y)
                epoch_meters["acc"].update(acc.item(), len(y))
            if self.scheduler_cls:
                self.scheduler_cls.step()
            
            train_state = {k:epoch_meters[k].avg for k in epoch_meters}
            
            if epoch % self.config.eval_freq == 0:
                if self.config.split_val < 1.0:
                    result_dict = self.evaluate("embed_val_subset2")
                else:
                    result_dict = self.evaluate("embed_val")
                for metric, _ in self.sel_metrics:
                    if self.best_meters[metric].add(result_dict[metric]):
                        self.save(epoch, self.best_meters[metric].get(), os.path.join(output_dir, f"best_{metric}_model.pt"))
                
                msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
                msg += ', '
                msg += ', '.join([f"{k}:{v:.6f}" for k,v in result_dict.items()])
                elapsed_time = timer.t()
                est_all_time = elapsed_time / epoch * self.config.epoch
                log(f"[Epoch {epoch}] {msg}, lr:{curr_lr:.6f} ({time_str(elapsed_time)}/{time_str(est_all_time)})")

            if self.config.save_freq > 0 and epoch % self.config.save_freq == 0:
                self.save(epoch, self.best_meters[self.sel_metrics[0][0]].get(), os.path.join(output_dir, f"model_epoch{epoch}.pt"))

        self.save(epoch, self.best_meters[self.sel_metrics[0][0]].get(), os.path.join(output_dir, "latest_model.pt"))

    def save(self, epoch, sel_metric, file_path):
        save_dict = {}
        save_dict["model_sd"] = self.model.state_dict()
        save_dict["sel_metric"] = sel_metric
        save_dict["config"] = self.config
        save_dict["optimizer"] = self.optimizer_cls.state_dict()
        save_dict["scheduler"] = self.scheduler_cls.state_dict() if self.scheduler_cls else None
        save_dict["epoch"] = epoch
        torch.save(save_dict, file_path)
    

    def test(self, output_dir, split=["test"], result_path=""):
        model_info = f"afr {self.config.dataset} {self.config.backbone} {self.config.train_split} train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f}  class_balanced:{self.config.balance_classes} group_uniform:{self.config.group_uniform} gamma:{self.config.afr_gamma:.6f} reg_coeff:{self.config.afr_reg_coeff:.6f}  seed:{self.config.seed}"
        if len(result_path) > 0:
            with open(result_path, "a") as fout:
                fout.write(model_info)
                fout.write('\n')
        model_paths = []
        for metric, _ in self.sel_metrics:
            model_path = os.path.join(output_dir, f"best_{metric}_model.pt")
            model_paths.append((model_path,metric))
        model_paths.append((os.path.join(output_dir, "latest_model.pt"),"latest"))
        for model_path, metric in model_paths:
            saved_dict = self.load_check_point(model_path)
            model_dict = saved_dict["model_sd"]
            sel_metric_val = saved_dict["sel_metric"]
            self.model.load_state_dict(model_dict)
            for sp in split:
                results = self.evaluate("embed_"+sp)
                result_str = f"[{sp} ({metric}:{sel_metric_val:.6f})]: " + ', '.join([f"{k}:{results[k]:.6f}" for k in results])
                log(result_str) 
                if len(result_path) > 0:
                    with open(result_path, "a") as fout:
                        fout.write(result_str)
                        fout.write('\n')

    