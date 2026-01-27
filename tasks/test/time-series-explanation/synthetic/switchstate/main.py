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
from switchloader import Switch
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
from classifier import SpikeClassifierNet
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
    switch = Switch(n_folds=5, fold=fold, seed=seed, data_dir="data/switchstate")

    # Create classifier
    classifier = SpikeClassifierNet(
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
        trainer.fit(classifier, datamodule=switch)
        if not os.path.exists("./model/"):
            os.makedirs("./model/")
        th.save(classifier.state_dict(), "./model/switch_feature/classifier_{}_{}".format(fold, seed))
    else:
        classifier.load_state_dict(th.load("./model/switch_feature/classifier_{}_{}".format(fold, seed)))

    # Get data for explainers
    with lock:
        x_train = switch.preprocess(split="train")["x"].to(device)
        x_test = switch.preprocess(split="test")["x"].to(device)
        y_test = switch.preprocess(split="test")["y"].to(device)
        true_saliency = switch.true_saliency(split="test").to(device)

    print("==============The sum of true_saliency is", true_saliency.sum(), "==============\n" + 70 * "=")

    # # Switch to eval
    classifier.eval()
    classifier.zero_grad()

    # Set model to device
    classifier.to(device)

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
        ).abs()

    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=1000,
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
            keep_ratio=list(np.arange(0.25, 0.35, 0.01)),
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=100,
            time_reg_factor=1.0,
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(None, None, True),
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
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(None, None, True),
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
            additional_forward_args=(None, None, True),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.8,
        )
        attr["gate_mask"] = _attr.to(device)
        # print_results(attr["gate_mask"], true_saliency)

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
        attr["fit"] = explainer.attribute(x_test, 
                                          show_progress=True)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier.cpu()))
        attr["gradient_shap"] = explainer.attribute(
            x_test.cpu(),
            baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
            n_samples=50,
            stdevs=0.0001,
            task="binary",
            show_progress=True,
        ).abs().to(device)
        classifier.to(device)

    # if "integrated_gradients" in explainers:
    #     explainer = TimeForwardTunnel(IntegratedGradients(classifier))
    #     attr["integrated_gradients"] = explainer.attribute(
    #         x_test,
    #         baselines=x_test * 0,
    #         internal_batch_size=200,
    #         task="binary",
    #         show_progress=True,
    #     ).abs()
    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            internal_batch_size=200,
            task="binary",
            show_progress=True,
        ).abs().to(device)
        
    # Proprietary method removed for skeletonization.


    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            additional_forward_args=(None, None, False),
            show_progress=True,
        ).abs().to(device)

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, 
                is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs().to(device)

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            additional_forward_args=(None, None, False),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs().to(device)
        
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
                fp.write(f"{aup(v, true_saliency):.4},")
                fp.write(f"{aur(v, true_saliency):.4},")
                fp.write(f"{information(v, true_saliency):.4},")
                fp.write(f"{entropy(v, true_saliency):.4},")
                fp.write(f"{roc_auc(v, true_saliency):.4},")
                fp.write(f"{auprc(v, true_saliency):.4}")
                fp.write("\n")


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
            "extremal_mask",  # tensor(13723.2715, grad_fn=<SumBackward0>) tensor(0.2366, grad_fn=<MeanBackward0>)
            "gate_mask",# tensor(14289.1562) tensor(0.4865, grad_fn=<MeanBackward0>) tensor(0.0310, gra>) 1.1 1 tensor(0.1030, grad_fn=<MseLossBackward0>)
            # "our"
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
        help="Train the rnn classifier.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=1,
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=2,
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


    # #
    # from utils.tools import process_results_by_file
    # process_results_by_file(5, args.explainers)
