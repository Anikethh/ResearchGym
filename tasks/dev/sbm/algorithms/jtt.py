from .algorithm import Algorithm
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from utils import AverageMeter, BestMetric, Timer, time_str, log
from .utils import init_optimizer, init_scheduler
from .register import register_algorithm
from data.data_utils import prepare_data
from models.classifier import Classifier
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Subset

@register_algorithm("jtt")
class JTT(Algorithm):
    def __init__(self, config):
        super(JTT, self).__init__(config)
        self.config = config
        # Set default values if not in config
        if not hasattr(self.config, 'first_stage_epochs'):
            self.config.first_stage_epochs = 1  # Default to 50% of total epochs
        if not hasattr(self.config, 'jtt_lambda'):
            self.config.jtt_lambda = 100.0
            
        # Calculate actual first stage epochs
        self.first_stage_epochs = self.config.first_stage_epochs
            
        self.datasets, self.dataloaders = prepare_data(config)
        self._init_model()
        self._init_training()

    def _init_model(self):
        """Initialize both stage 1 and stage 2 models"""
        self.n_classes = self.datasets["train"].n_classes
        self.device = f"cuda:{self.config.gpu}"
        
        # Initialize both models
        self.model1 = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        self.model2 = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        
        if self.config.check_point:
            saved_dict = self.load_check_point(self.config.check_point)
            if "model1_sd" in saved_dict:  # Resume from JTT checkpoint
                self.model1.load_state_dict(saved_dict["model1_sd"])
                self.model2.load_state_dict(saved_dict["model2_sd"])
            else:  # Resume from regular checkpoint
                self.model1.load_state_dict(saved_dict["model_sd"])
        
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model = self.model1  # Current active model

    def _init_training(self):
        """Initialize optimizers and training components"""
        # Initialize optimizers for both models
        self.optimizer1 = init_optimizer(self.model1, self.config.optimizer, self.config.optimizer_kwargs)
        self.optimizer2 = init_optimizer(self.model2, self.config.optimizer, self.config.optimizer_kwargs)
        
        # Initialize schedulers
        self.scheduler1 = init_scheduler(self.optimizer1, self.config.scheduler, self.config.scheduler_kwargs)
        self.scheduler2 = init_scheduler(self.optimizer2, self.config.scheduler, self.config.scheduler_kwargs)
        
        # Set current active optimizer/scheduler
        self._optimizer = self.optimizer1
        self._scheduler = self.scheduler1
        
        # Loss function (no reduction)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Setup metrics tracking
        self.sel_metrics = [("val_acc", True), ("val_worst_cls_acc", True), 
                          ("val_worst_group_acc", True), ("val_avg_cls_diff", False)]
        self.best_meters = {m: BestMetric(max_val) for m, max_val in self.sel_metrics}

    def _identify_error_set(self, train_loader):
        """Identify examples misclassified by first model"""
        self.model1.eval()
        error_indices = []
        total_indices = range(len(train_loader.dataset))
        
        with torch.no_grad():
            for batch_idx, (x, y, _, _) in enumerate(tqdm(train_loader, desc='Finding errors')):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model1(x)
                preds = torch.argmax(logits, dim=1)
                errors = (preds != y)
                
                # Get original indices for misclassified examples
                error_idx = torch.nonzero(errors).cpu()
                if len(error_idx) > 0:  # Only process if there are errors
                    error_idx = error_idx.squeeze()  # Squeeze but keep as tensor
                    # Handle case where there's only one error
                    if error_idx.ndim == 0:
                        error_idx = error_idx.unsqueeze(0)
                    batch_start_idx = batch_idx * train_loader.batch_size
                    error_indices.extend((batch_start_idx + error_idx).tolist())

        return error_indices, total_indices

    def _create_upsampled_loader(self, error_indices, train_loader):
        """Create new dataloader that upsamples error examples"""
        dataset = train_loader.dataset
        
        # Create list of indices with errors repeated jtt_lambda times
        upsampled_indices = list(range(len(dataset)))
        for _ in range(int(self.config.jtt_lambda - 1)):
            upsampled_indices.extend(error_indices)
            
        # Create new sampler and dataloader
        class UpsampleSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices
            def __iter__(self):
                np.random.shuffle(self.indices)
                return iter(self.indices)
            def __len__(self):
                return len(self.indices)
                
        upsampled_sampler = UpsampleSampler(upsampled_indices)
        
        upsampled_loader = DataLoader(
            dataset,
            batch_size=train_loader.batch_size,
            sampler=upsampled_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=True)
            
        return upsampled_loader

    def train(self, output_dir, split="train"):
        """Full training procedure"""
        timer = Timer()
        train_loader = self.dataloaders[split]

        # Stage 1: Train first model normally
        log("Starting Stage 1: Training identification model...")
        self.model = self.model1
        self._optimizer = self.optimizer1
        self._scheduler = self.scheduler1
        
        for epoch in range(1, self.first_stage_epochs + 1):
            # Regular training epoch
            self.model.train()
            epoch_meters = {k: AverageMeter() for k in ["loss", "acc"]}
            
            for x, y, _ , _ in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                self._optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self._optimizer.step()
                
                acc = (torch.argmax(logits, dim=1) == y).float().mean()
                epoch_meters["loss"].update(loss.item(), x.size(0))
                epoch_meters["acc"].update(acc.item(), x.size(0))
                
            if self._scheduler:
                self._scheduler.step()

            # Log progress
            if epoch % self.config.eval_freq == 0:
                val_metrics = self.evaluate("val")
                for metric, _ in self.sel_metrics:
                    if self.best_meters[metric].add(val_metrics[metric]):
                        self.save(epoch, val_metrics[metric], 
                                os.path.join(output_dir, f"stage1_best_{metric}_model.pt"))
                
                train_state = {k: v.avg for k, v in epoch_meters.items()}
                msg = ', '.join([f"{k}:{v:.6f}" for k,v in {**train_state, **val_metrics}.items()])
                log(f"[Stage 1 Epoch {epoch}] {msg}")

        # Save stage 1 model
        self.save(self.first_stage_epochs, self.best_meters["val_acc"].get(), 
                 os.path.join(output_dir, "stage1_final_model.pt"))

        # Identify error set
        error_indices, total_indices = self._identify_error_set(train_loader)
        log(f"Found {len(error_indices)} misclassified examples")
        
        # Create upsampled loader
        upsampled_loader = self._create_upsampled_loader(error_indices, train_loader)

        # Stage 2: Train second model on upsampled dataset
        log("Starting Stage 2: Training final model...")
        self.model = self.model2  
        self._optimizer = self.optimizer2
        self._scheduler = self.scheduler2
        self.best_meters = {m: BestMetric(max_val) for m, max_val in self.sel_metrics}
        
        remaining_epochs = self.config.epoch - self.first_stage_epochs
        for epoch in range(1, remaining_epochs + 1):
            self.model.train()
            epoch_meters = {k: AverageMeter() for k in ["loss", "acc"]}
            
            for x, y, _, _ in tqdm(upsampled_loader, desc=f'Epoch {epoch}', leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                self._optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self._optimizer.step()
                
                acc = (torch.argmax(logits, dim=1) == y).float().mean()
                epoch_meters["loss"].update(loss.item(), x.size(0))
                epoch_meters["acc"].update(acc.item(), x.size(0))
                
            if self._scheduler:
                self._scheduler.step()

            # Log progress
            if epoch % self.config.eval_freq == 0:
                val_metrics = self.evaluate("val")
                for metric, _ in self.sel_metrics:
                    if self.best_meters[metric].add(val_metrics[metric]):
                        self.save(epoch, val_metrics[metric],
                                os.path.join(output_dir, f"stage2_best_{metric}_model.pt"))
                        # Also save as overall best
                        self.save(epoch, val_metrics[metric],
                                os.path.join(output_dir, f"best_{metric}_model.pt"))
                
                train_state = {k: v.avg for k, v in epoch_meters.items()}
                msg = ', '.join([f"{k}:{v:.6f}" for k,v in {**train_state, **val_metrics}.items()])
                log(f"[Stage 2 Epoch {epoch}] {msg}")

        # Save final model
        self.save(self.config.epoch, self.best_meters["val_acc"].get(),
                 os.path.join(output_dir, "latest_model.pt"))

    def save(self, epoch, sel_metric, file_path):
        """Save model checkpoint"""
        save_dict = {
            "model1_sd": self.model1.state_dict(),
            "model2_sd": self.model2.state_dict(),
            "sel_metric": sel_metric,
            "config": self.config,
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "scheduler1": self.scheduler1.state_dict() if self.scheduler1 else None,
            "scheduler2": self.scheduler2.state_dict() if self.scheduler2 else None,
            "epoch": epoch
        }
        torch.save(save_dict, file_path)

    def test(self, output_dir, test_splits=["test"], result_path=None):
        """Test the model"""
        # Convert test_splits to list if string is provided
        if isinstance(test_splits, str):
            test_splits = [test_splits]
            
        # Model info for logging
        model_info = (f"jtt {self.config.dataset} {self.config.backbone} {self.config.train_split} "
                     f"train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f} "
                     f"seed:{self.config.seed} jtt_lambda:{self.config.jtt_lambda:.2f}")
        
        if result_path is not None:
            with open(result_path, "a") as f:
                f.write(model_info + '\n')

        # Collect all model paths to evaluate
        model_paths = []
        for metric, _ in self.sel_metrics:
            model_path = os.path.join(output_dir, f"best_{metric}_model.pt")
            model_paths.append((model_path, metric))
        model_paths.append((os.path.join(output_dir, "latest_model.pt"), "latest"))

        # Evaluate each model checkpoint
        for model_path, metric in model_paths:
            try:
                saved_dict = self.load_check_point(model_path)
                self.model2.load_state_dict(saved_dict["model2_sd"])
                self.model = self.model2
            except Exception as e:
                log(f"Warning: Failed to load model from {model_path}: {str(e)}")
                continue
                
            # Test on each split
            for split in test_splits:
                if split not in self.dataloaders:
                    log(f"Warning: Split {split} not found in dataloaders")
                    continue
                    
                try:
                    results = self.evaluate(split)
                    result_str = f"[{split} ({metric})]: " + ', '.join([f"{k}:{v:.6f}" for k,v in results.items()])
                    log(result_str)
                    if result_path is not None:
                        with open(result_path, "a") as f:
                            f.write(result_str + '\n')
                except Exception as e:
                    log(f"Error evaluating split {split}: {str(e)}")
                    continue