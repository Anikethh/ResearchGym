import torch
from .algorithm import Algorithm
from data.data_utils import prepare_data
from tqdm import tqdm
from utils import AverageMeter, BestMetric, Timer, time_str, log
from .utils import init_optimizer, init_scheduler
from .register import register_algorithm
import numpy as np
import os
from models.classifier import Classifier

@register_algorithm("erm")
class ERM(Algorithm):
    def __init__(self, config):
        super(ERM, self).__init__(config)
        self._init_model()
        self._init_training()
        
    def _init_training(self):
        self._optimizer = init_optimizer(self.model, self.config.optimizer, self.config.optimizer_kwargs)
        self._scheduler = init_scheduler(self._optimizer, self.config.scheduler, self.config.scheduler_kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.sel_metrics = [("val_acc", True), ("val_worst_cls_acc", True), ("val_worst_group_acc", True), ("val_avg_cls_diff", False)]
        self.best_meters = {m:BestMetric(max_val) for m, max_val in self.sel_metrics}

    def _init_model(self):
        self.n_classes = self.datasets["train"].n_classes
        self.device = f"cuda:{self.config.gpu}"
        self.model = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        if self.config.check_point:
            saved_dict = self.load_check_point(self.config.check_point)
            model_sd = saved_dict["model_sd"]
            self.model.load_state_dict(model_sd)
        self.model.to(self.device)

    def train(self, output_dir, split="train"):
        timer = Timer()
    
        train_loader = self.dataloaders[split]
        
        best_val = 0.0
        for epoch in range(1, self.config.epoch+1):
            self.model.train()
            
            epoch_meters = {k:AverageMeter() for k in ["loss", "acc"]}
            if self._scheduler:
                curr_lr = self._scheduler.get_last_lr()[0]
            else:
                curr_lr = self.config.optimizer_kwargs["lr"]

            for batch in tqdm(train_loader, desc=split, leave=False):
                x, y, _, _ = batch
                x, y = x.to(self.device), y.to(self.device)

                self._optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self._optimizer.step()

                epoch_meters["loss"].update(loss.item(), x.size(0))
                acc = (torch.argmax(logits, dim=-1) == y).sum() / len(y)
                epoch_meters["acc"].update(acc.item(), len(y))
            if self._scheduler:
                self._scheduler.step()
            
            train_state = {k:epoch_meters[k].avg for k in epoch_meters}
            
            if epoch % self.config.eval_freq == 0:
                result_dict = self.evaluate("val")
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
                self.save(epoch, self.best_meters["val_acc"].get(), os.path.join(output_dir, f"model_epoch{epoch}.pt"))

        self.save(epoch, self.best_meters["val_acc"].get(), os.path.join(output_dir, "latest_model.pt"))

    def save(self, epoch, sel_metric, file_path):
        save_dict = {}
        save_dict["model_sd"] = self.model.state_dict()
        save_dict["sel_metric"] = sel_metric
        save_dict["config"] = self.config
        save_dict["optimizer"] = self._optimizer.state_dict()
        save_dict["scheduler"] = self._scheduler.state_dict() if self._scheduler else None
        save_dict["epoch"] = epoch
        torch.save(save_dict, file_path)
    

    def test(self, output_dir, split=["test"], result_path=""):
        model_info = f"erm {self.config.dataset} {self.config.backbone} {self.config.train_split} train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f} seed:{self.config.seed}"
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
                results = self.evaluate(sp)
                result_str = f"[{sp} ({metric}:{sel_metric_val:.6f})]: " + ', '.join([f"{k}:{results[k]:.6f}" for k in results])
                log(result_str) 
                if len(result_path) > 0:
                    with open(result_path, "a") as fout:
                        fout.write(result_str)
                        fout.write('\n')
           
