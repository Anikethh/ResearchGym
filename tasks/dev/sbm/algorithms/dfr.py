import torch
from .algorithm import Algorithm
from data.data_utils import prepare_data
from tqdm import tqdm
from utils import AverageMeter, BestMetric, Timer, time_str, log
from data.embed_dataset import EmbedDataset
from .utils import init_optimizer, init_scheduler
from .register import register_algorithm
from data.sampler import GroupBalancedSampler, ClassBalancedSampler
import numpy as np
import os
from torch.utils.data import DataLoader
from models.classifier import Classifier

@register_algorithm("dfr")
class DFR(Algorithm):
    def __init__(self, config):
        super(DFR, self).__init__(config)
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

    def _init_training(self):
        self.optimizer_cls = init_optimizer(self.model.fc, self.config.optimizer_cls, self.config.optimizer_cls_kwargs)
        self.scheduler_cls = None
      
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.config.split_val < 1.0:
            self.sel_metrics = [("embed_val_subset2_acc", True), ("embed_val_subset2_worst_cls_acc", True), ("embed_val_subset2_worst_group_acc", True), ("embed_val_subset2_avg_cls_diff", False)]
        else:
            self.sel_metrics = [("embed_val_acc", True), ("embed_val_worst_cls_acc", True), ("embed_val_worst_group_acc", True), ("embed_val_avg_cls_diff", False)]
        self.best_meters = {m:BestMetric(max_val) for m, max_val in self.sel_metrics}


    def train(self, output_dir, split="train"):
        timer = Timer()
    
        embed_loaders = self.get_embed_loaders(split)
        if self.config.use_shortcutwalk_dataset:
            self.update_train_loaders(split)
            train_loader = self.dataloaders["embed_shortcutwalk_train"]
        else:
            train_loader = embed_loaders["embed_"+split]
        self.model.eval()
        best_val = 0.0
        for epoch in range(1, self.config.dfr_epochs+1):
            self.model.fc.train()
            
            epoch_meters = {k:AverageMeter() for k in ["loss", "acc"]}
            if self.scheduler_cls:
                curr_lr = self.scheduler_cls.get_last_lr()[0]
            else:
                curr_lr = self.config.optimizer_cls_kwargs["lr"]

            for batch in tqdm(train_loader, desc=split, leave=False):
                x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)

                # num_classes = len(torch.unique(y))
                # assert np.array([abs((y==c).to(torch.float).sum().item()-len(y)/num_classes) for c in torch.unique(y)]).max() < 1, "not class balanced"
                # num_groups = len(torch.unique(g))
                # assert np.array([abs((g==c).to(torch.float).sum().item()-len(g)/num_groups) for c in torch.unique(g)]).max() < 1, "not group balanced"

                self.optimizer_cls.zero_grad()
                logits = self.model.fc(x)
                loss = self.criterion(logits, y) + self.config.dfr_reg * torch.norm(self.model.fc.weight, 1)
                loss.backward()
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
        model_info = f"dfr {self.config.dataset} {self.config.backbone} {self.config.train_split} train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f}  class_balanced:{self.config.class_balanced} group_balanced:{self.config.group_balanced} dfr_reg:{self.config.dfr_reg:.6f}  seed:{self.config.seed}"
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
           
