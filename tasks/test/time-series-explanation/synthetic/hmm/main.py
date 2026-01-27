import multiprocessing as mp
import os
from pytorch_lightning.callbacks import EarlyStopping
from os import path
from pathlib import Path
import sys
sys.path.append(path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) )))

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List
# from utils.tools import print_results
from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from tint.datasets import HMM
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)
from tint.models import MLP, RNN

from attribution.gatemasknn import *
from attribution.gate_mask import GateMask
from synthetic.hmm.classifier import StateClassifierNet

from synthetic.switchstate.cumulative_difference import cumulative_difference


def main(
    explainers: List[str],
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    is_train: bool = True,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    output_file: str = "results.csv",
):
    # Get accelerator and device
    accelerator = device.split(":")[0]
    print(accelerator)
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    hmm = HMM(n_folds=5, fold=fold, seed=seed, data_dir="data/hmm")

    # Create classifier
    classifier = StateClassifierNet(
        feature_size=3,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    if is_train:
        trainer.fit(classifier, datamodule=hmm)
        if not os.path.exists("./model/hmm/"):
            os.makedirs("./model/hmm/")
        th.save(classifier.state_dict(), "../../model/hmm/classifier_{}_{}".format(fold, seed))
    else:
        classifier.load_state_dict(th.load("./model/hmm/classifier_{}_{}".format(fold, seed)))

    # Get data for explainers
    with lock:
        x_train = hmm.preprocess(split="train")["x"].to(device)
        x_test = hmm.preprocess(split="test")["x"].to(device)
        y_test = hmm.preprocess(split="test")["y"].to(device)
        true_saliency = hmm.true_saliency(split="test").to(device)

    print("==============The sum of true_saliency is", true_saliency.sum(), "==============\n" + 70 * "=")
    # if not os.path.exists("./results/"):
    #     os.makedirs("./results/")
    # np.save('./results/true.npy', true_saliency.detach().numpy())

    # Switch to eval
    classifier.eval()
    classifier.zero_grad()

    # Set model to device
    classifier.to(device)
    
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            task="binary",
            show_progress=True,
        ).abs().to(device)

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
            forward_func=classifier,
            perturbation="gaussian_blur",
            sigma_max=1,
            keep_ratio=list(np.arange(0.25, 0.35, 0.05)),
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=100,
            time_reg_factor=1.0,
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

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
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=1.0,
            lambda_2=1.0,
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask"] = _attr.to(device)

    if "gate_mask" in explainers:
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
        mask = GateMaskNet(
            forward_func=classifier,
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
            optim="adam",
            lr=0.01,
        )
        explainer = GateMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.5,
        )
        attr["gate_mask"] = _attr.to(device)

    if "fit" in explainers:
        generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
        trainer = Trainer(
            max_epochs=200,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=10,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        explainer = Fit(
            classifier,
            generator=generator,
            features=x_test,
            trainer=trainer,
        )
        attr["fit"] = explainer.attribute(x_test, show_progress=True)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier))
        
        gradients = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            attr_batch = explainer.attribute(
                x_batch,
                baselines=th.cat([x_batch * 0, x_batch]),
                n_samples=50,
                stdevs=0.0001,
                task="binary",
                show_progress=True,
            ).abs()
            
            gradients.append(attr_batch)
            
        attr["gradient_shap"] = th.cat(gradients, dim=0).to(device)
            
        classifier.to(device)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            internal_batch_size=200,
            task="binary",
            show_progress=True,
        ).abs()

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    # Proprietary method removed for skeletonization.

    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)

    baselines_dict = {0: "Average", 1: "Zeros"}

    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):    
            for k, v in attr.items():
                cum_diff, AUCC, cum_50_diff, _ = cumulative_difference(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=0.1,
                    top=0,
                    testbs=x_test.shape[0],
                    additional_forward_args=(None, None, True),
                )
                fp.write(str(seed) + ",")
                fp.write(str(fold) + ",")
                fp.write(baselines_dict[i] + ",")
                fp.write(k + ",")
                fp.write(str(lambda_1) + ",")
                fp.write(str(lambda_2) + ",")
                fp.write(f"{cum_50_diff:.4},")
                fp.write(f"{cum_diff:.4},")
                fp.write(f"{AUCC:.4},")
                v = v.to(device)
                fp.write(f"{aup(v, true_saliency):.4},")
                fp.write(f"{aur(v, true_saliency):.4},")
                fp.write(f"{information(v, true_saliency):.4},")
                fp.write(f"{entropy(v, true_saliency):.4},")
                fp.write(f"{roc_auc(v, true_saliency):.4},")
                fp.write(f"{auprc(v, true_saliency):.4}")
                fp.write("\n")

    # if not os.path.exists("./results/"):
    #     os.makedirs("./results/")
    # for key in attr.keys():
    #     result = attr[key]
    #     np.save('./results/{}_result_{}_{}.npy'.format(key, fold, seed), result.detach().numpy())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "occlusion",
            "augmented_occlusion",
            "integrated_gradients",
            "gradient_shap",
            "deep_lift",
            "lime",
            "fit",
            "dyna_mask",
            "extremal_mask",
            "gate_mask"
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
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
        default=0,
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
        default=2,  #2 is best
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=1,
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results.csv",
        help="Where to save the results.",
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
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
        is_train=args.train,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        output_file=args.output_file,
    )
