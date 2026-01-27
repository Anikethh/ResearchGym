import sys
from os import path
from pathlib import Path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))


import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os
from utils.tools import print_results

from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from argparse import ArgumentParser
from tqdm import tqdm
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime, KernelShap, DeepLiftShap
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    Occlusion,
    FeatureAblation,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)

from real.cumulative_difference import cumulative_difference
from tint.models import MLP, RNN

from real.classifier import MimicClassifierNet


def main(
    explainers: List[str],
    data: str,
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    is_train: bool = True,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    num_segments: int = 50,
    min_seg_len: int = 1,
    max_seg_len: int = 48,
    mask_lr: float = 0.1,
    output_file: str = "results.csv",
    model_type: str = "state",
    testbs: int = 0,
    top: int = 50,
    skip_train_timex: bool = True,
    prob: float = 0.1 ,
):
    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    if data == "mimic3":
        datamodule = Mimic3(n_folds=5, fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=32,
            # feature_size=31,
            n_state=2,
            n_timesteps=48,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 32
        num_classes = 2
        max_len = 48
        
    elif data == "PAM":
        datamodule = PAM(fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=17,
            n_state=8,
            n_timesteps=600,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 17
        num_classes = 8
        max_len = 600
        
    elif data == "boiler":
        datamodule = Boiler(fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=20,
            n_state=2,
            n_timesteps=36,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 20
        num_classes = 2
        max_len = 36
    
    elif data == "epilepsy":
        datamodule = Epilepsy(fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=1,
            n_state=2,
            n_timesteps=178,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 1
        num_classes = 2
        max_len = 178
    
    elif data == "freezer":
        datamodule = Freezer(n_folds=5, fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=1,
            n_state=2,
            n_timesteps=301,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        
        num_features = 1
        num_classes = 2
        max_len = 301
    
    elif data == "wafer":
        datamodule = Wafer(n_folds=5, fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=1,
            n_state=2,
            n_timesteps=152,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 1
        num_classes = 2
        max_len = 152

    # Train classifier
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    if is_train:
        trainer.fit(classifier, datamodule=datamodule)
        if not os.path.exists("./model/{}/".format(data)):
            os.makedirs("./model/{}/".format(data))
        th.save(classifier.state_dict(), "./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed))
    else:
        classifier.load_state_dict(th.load("./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed)))
    # Get data for explainers
    with lock:
        x_train = datamodule.preprocess(split="train")["x"].to(device)
        x_test = datamodule.preprocess(split="test")["x"].to(device)
        y_train = datamodule.preprocess(split="train")["y"].to(device)
        y_test = datamodule.preprocess(split="test")["y"].to(device)
        mask_train = datamodule.preprocess(split="train")["mask"].to(device)
        mask_test = datamodule.preprocess(split="test")["mask"].to(device)

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()
    
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(x_test, mask_test)
    test_loader = DataLoader(test_dataset, batch_size=testbs, shuffle=False)
    
    if model_type == "state":
        temporal_additional_forward_args = (False, False, False)
    else:
        temporal_additional_forward_args = (False, False, False)
    
    data_mask=mask_test
    data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = MaskNet(
            forward_func=classifier.predict,
            perturbation="fade_moving_average",
            keep_ratio=list(np.arange(0.1, 0.7, 0.1)),
            deletion_mode=True,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
            loss="cross_entropy",
        )
        explainer = DynaMask(classifier.predict)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            additional_forward_args=(data_mask, timesteps, False),
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

    if "gate_mask" in explainers:
        trainer = Trainer(
            max_epochs=200,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
            # precision=16, # if OOM occurs
        )
        mask = GateMaskNet(
            forward_func=classifier.predict,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            loss="cross_entropy",
            optim="adam",
            lr = mask_lr,
            # lr=0.01,
        )
        explainer = GateMask(classifier.predict)
        _attr = explainer.attribute(
            x_test,
            # additional_forward_args=(True,) (return_all = True) is it really considered?
            additional_forward_args=(data_mask, timesteps, False),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.5,
        )
        attr["gate_mask"] = _attr.to(device)

    if "extremal_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = ExtremalMaskNet(
            forward_func=classifier.predict,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            loss="cross_entropy",
            optim="adam",
            lr=mask_lr,
        )
        explainer = ExtremalMask(classifier.predict)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(data_mask, timesteps, False),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask"] = _attr.to(device)
  
    if "fit" in explainers:
        from attribution.winit import FIT
        
        skip_training = skip_train_timex # consider this
        
        generator_path = Path("./generator/") / data / f"{model_type}_split_{fold}"
        generator_path.mkdir(parents=True, exist_ok=True)
        explainer = FIT(
            classifier,
            device=device,
            datamodule=datamodule,
            data_name=data,
            feature_size=num_features,
            path=generator_path,
            cv=fold,
        )
        
        if skip_training:
            explainer.load_generators()
        else:
            explainer.train_generators(300)
        
        fit = []

        for batch in tqdm(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(x_batch)
            
            fit.append(attr_batch)
        
        attr["fit"] = th.Tensor(np.concatenate(fit, axis=0)) 

    if "winit" in explainers:
        from attribution.winit import WinIT
        
        skip_training = skip_train_timex # consider this
        
        generator_path = Path("./generator/") / data / f"{model_type}_split_{fold}"
        generator_path.mkdir(parents=True, exist_ok=True)
        explainer = WinIT(
            classifier,
            device=device,
            datamodule=datamodule,
            data_name=data,
            feature_size=num_features,
            path=generator_path,
            cv=fold,
        )
        
        if skip_training:
            explainer.load_generators()
        else:
            explainer.train_generators(300)
        
        winit = []

        for batch in tqdm(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(x_batch)
            
            winit.append(attr_batch)
        
        attr["winit"] = th.Tensor(np.concatenate(winit, axis=0)) 

    ####  deeplift classfiier.predict error occur
    if "deeplift_abs" in explainers:
        explainer = DeepLift(classifier) # change forward function to self.net(*args, **kwargs).softmax(-1)

        deeplift = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)
            
            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
            ).abs()
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_abs"] = th.cat(deeplift, dim=0)
        
    ####  deeplift classfiier.predict error occur
    if "deeplift_signed" in explainers:
        explainer = DeepLift(classifier)

        deeplift = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)
            
            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_signed"] = th.cat(deeplift, dim=0)

    if "gradientshap_abs" in explainers:
        explainer = GradientShap(classifier.predict)

        gradientshap = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            
            attr_batch = explainer.attribute(
                    x_batch,
                    baselines=(th.cat([x_batch * 0, x_batch])),
                    target=partial_targets,
                    n_samples=50,
                    stdevs=0.0001,
                    additional_forward_args=(data_mask, timesteps, False),
                ).abs()
            
            
            # Append the IG attributes of the current batch to the list
            gradientshap.append(attr_batch.cpu())  # Move to CPU if necessary
        
        # Concatenate all batch IG attributes into a single tensor
        attr["gradientshap_abs"] = th.cat(gradientshap, dim=0)
        
    if "gradientshap_signed" in explainers:
        explainer = GradientShap(classifier.predict)

        gradientshap = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            
            attr_batch = explainer.attribute(
                    x_batch,
                    baselines=(th.cat([x_batch * 0, x_batch])),
                    target=partial_targets,
                    n_samples=50,
                    stdevs=0.0001,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            
            
            # Append the IG attributes of the current batch to the list
            gradientshap.append(attr_batch.cpu())  # Move to CPU if necessary
        
        # Concatenate all batch IG attributes into a single tensor
        attr["gradientshap_signed"] = th.cat(gradientshap, dim=0)
        
    if "integrated_gradients_base" in explainers:
        explainer = IntegratedGradients(classifier.predict)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_base"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_base_abs" in explainers:
        explainer = IntegratedGradients(classifier.predict)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            ).abs()
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_base_abs"] = th.cat(integrated_gradients, dim=0)

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier.predict))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier.predict, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            additional_forward_args=(data_mask, None, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            task="binary",
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier.predict))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            additional_forward_args=(data_mask, timesteps, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            attributions_fn=abs,
            show_progress=True,
        ).abs()
        
    if "timex" in explainers:
        from attribution.timex import TimeXExplainer
        explainer = TimeXExplainer(
            model=classifier.predict,
            device=x_test.device,
            num_features=num_features,
            num_classes=num_classes,
            max_len=max_len,
            data_name=data,
            split=fold,
            is_timex=True,
        )
        
        explainer.train_timex(x_train, y_train, x_test, y_test, "./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed), skip_train_timex)
            
        timex_results = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(
                x_batch,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
            timex_results.append(attr_batch.detach().cpu())
        
        
        attr["timex"] = th.cat(timex_results, dim=0)
        
    if "timex++" in explainers:
        from attribution.timex import TimeXExplainer
        explainer = TimeXExplainer(
            model=classifier.predict,
            device=x_test.device,
            num_features=num_features,
            num_classes=num_classes,
            max_len=max_len,
            data_name=data,
            split=fold,
            is_timex=False,
        )
        
        explainer.train_timex(x_train, y_train, x_test, y_test, "./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed), skip_train_timex)
            
        timex_results = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(
                x_batch,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
            timex_results.append(attr_batch.detach().cpu())
        
        
        attr["timex++"] = th.cat(timex_results, dim=0)
        

    

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)
    
    # print

    # Dict for baselines
    baselines_dict = {0: "Average", 1: "Zeros"}
    
    # ## data_mask=mask_test.to("cpu")
    # data_mask = mask_test.to(x_test.device)
    # data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):
            for topk in areas:
                for k, v in attr.items():
                    cum_diff, AUCC, cum_50_diff, _ = cumulative_difference(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        top=args.top,
                        testbs=testbs,
                        additional_forward_args=(mask_test, None, False),
                    )
                    
                    
                    
                    total_acc = 0.0
                    total_comp = 0.0
                    total_ce = 0.0
                    total_lodds = 0.0
                    total_suff = 0.0
                    total_samples = 0

                    # 2. Loop over batches
                    for batch_idx, batch in enumerate(test_loader):
                        # batch = (input_tensor, data_mask, ...)
                        x_batch = batch[0].to(device)
                        data_mask_batch = batch[1].to(device)
                        batch_size = x_batch.shape[0]

                        # If timesteps is sized for the entire dataset, slice for this batch
                        # Example (adjust accordingly if needed):
                        timesteps_batch = timesteps[batch_idx * batch_size : batch_idx * batch_size + batch_size]

                        # Prepare baselines for the batch
                        # If baselines is a tensor like x_avg, slice it for the batch dimension
                        if isinstance(baselines, th.Tensor):
                            baselines_batch = baselines[batch_idx * batch_size : batch_idx * batch_size + batch_size]
                            baselines_batch = baselines_batch.to(device)
                        else:
                            # e.g., if baselines=0.0 or a scalar, you might just keep it as-is
                            # Or replicate it: baselines_batch = torch.zeros_like(x_batch)
                            baselines_batch = baselines

                        # Similarly slice the attribution tensor 'v'
                        v_batch = v[batch_idx * batch_size : batch_idx * batch_size + batch_size].to(device)

                        # 3. Compute metrics for this batch
                        acc = accuracy(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        comp = comprehensiveness(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        ce = cross_entropy(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        l_odds = log_odds(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        suff = sufficiency(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )

                        # 4. Accumulate results (multiply by batch_size if metrics are averages)
                        #    If your metric function already returns a sum, you may not need to multiply.
                        total_acc += acc * batch_size
                        total_comp += comp * batch_size
                        total_ce += ce * batch_size
                        total_lodds += l_odds * batch_size
                        total_suff += suff * batch_size
                        total_samples += batch_size
                        
                    mean_acc = total_acc / total_samples
                    mean_comp = total_comp / total_samples
                    mean_ce = total_ce / total_samples
                    mean_lodds = total_lodds / total_samples
                    mean_suff = total_suff / total_samples

                    fp.write(str(seed) + ",")
                    fp.write(str(fold) + ",")
                    fp.write(baselines_dict[i] + ",")
                    fp.write(str(topk) + ",")
                    fp.write(k + ",")
                    fp.write(str(lambda_1) + ",")
                    fp.write(str(lambda_2) + ",")
                    fp.write(str(lambda_3) + ",")
                    fp.write(f"{cum_50_diff:.4},")
                    fp.write(f"{cum_diff:.4},")
                    fp.write(f"{AUCC:.4},")
                    fp.write(f"{mean_acc:.4},")
                    fp.write(f"{mean_comp:.4},")
                    fp.write(f"{mean_ce:.4},")
                    fp.write(f"{mean_lodds:.4},")
                    fp.write(f"{mean_suff:.4}")
                    fp.write("\n")

    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    for key in attr.keys():
        result = attr[key]
        if isinstance(result, tuple): result = result[0]
        np.save('./results/{}_{}_{}_result_{}_{}.npy'.format(data, model_type, key, fold, seed), result.detach().cpu().numpy())
    
    print(f"{explainers} done")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "gate_mask"
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mimic3",
        help="real world data",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train thr rnn classifier.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=0.001,   # 0.01
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-3",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--mask_lr",
        type=float,
        default=0.01,   
        help="learning rate for mask based method",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.1,   
        help="asff",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results_gate.csv",
        help="Where to save the results.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="state",
        choices=["state", "mtand", "seft", "transformer", "cnn"],
    )
    parser.add_argument(
        "--testbs",
        type=int,
        default=200
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=50
    )
    parser.add_argument(
        "--min_seg_len",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max_seg_len",
        type=int,
        default=48
    )
    parser.add_argument(
        "--skip_train_timex",
        action='store_true'
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    print(f"set seed as {seed}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(
        explainers=args.explainers,
        data=args.data,
        areas=args.areas,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        is_train=args.train,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        num_segments=args.num_segments,
        min_seg_len=args.min_seg_len,
        max_seg_len=args.max_seg_len,
        mask_lr=args.mask_lr,
        output_file=args.output_file,
        model_type=args.model_type,
        testbs=args.testbs,
        top=args.top,
        skip_train_timex=args.skip_train_timex,
        prob=args.prob
    )

