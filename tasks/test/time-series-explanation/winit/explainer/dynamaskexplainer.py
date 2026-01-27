from __future__ import annotations

import logging

import numpy as np
import torch

from winit.explainer.attribution.mask_group import MaskGroup
from winit.explainer.attribution.perturbation import GaussianBlur, FadeMovingAverage
from winit.explainer.dynamaskutils.losses import log_loss_multiple, cross_entropy_multiple
from winit.explainer.explainers import BaseExplainer


class DynamaskExplainer(BaseExplainer):
    """
    The explainer for Dynamask. The code was modified from the Dynamask repository.
    https://github.com/JonathanCrabbe/Dynamask

    As dynamask "training" does not have early stopping, we vectorize the dynamask
    generation to make it about 100x faster.
    """

    def __init__(
        self,
        device,
        area_list=None,
        num_epoch=200,
        num_class=1,
        blur_type="gaussian",
        deletion_mode=False,
        size_reg_factor_dilation=100,
        time_reg_factor=1,
        loss="logloss",
        use_last_timestep_only=False,
        **kwargs,
    ):
        super().__init__(device)
        if blur_type == "gaussian":
            self.pert = GaussianBlur(
                self.device, sigma_max=1.0
            )  # This is the perturbation operator
        elif blur_type == "fadema":
            self.pert = FadeMovingAverage(self.device)
        else:
            raise Exception("Unknown blur_type " + blur_type)
        self.blur_type = blur_type

        # This is the list of masks area to consider
        self.area_list = area_list if area_list is not None else np.arange(0.25, 0.35, 0.01)

        self.num_epoch = num_epoch
        self.num_class = num_class
        self.deletion_mode = deletion_mode
        self.size_reg_factor_dilation = size_reg_factor_dilation
        self.time_reg_factor = time_reg_factor
        if loss == "logloss":
            self.loss = log_loss_multiple
        elif loss == "ce":
            self.loss = cross_entropy_multiple
        else:
            raise RuntimeError(f"Unrecognized loss {loss}")
        self.loss_str = loss
        self.use_last_timestep_only = use_last_timestep_only

        if len(kwargs):
            log = logging.getLogger(DynamaskExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x):
        self.base_model.eval()
        self.base_model.zero_grad()
        return self._attribute_multiple(x)

    def _attribute_multiple(self, x):

        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        def f(x_in):
            # x_in (num_sample, num_time, num_feature)
            num_sample, num_times, num_features = x_in.shape
            x_in = x_in.permute(0, 2, 1)
            out = self.base_model(x_in, return_all=True)  # (num_sample, num_state=1, num_time)
            out = self.base_model.activation(out).reshape(num_sample, num_times)
            if self.num_class == 1:
                # stack
                out = torch.stack([1 - out, out], dim=2)
            else:
                out = out.reshape(num_sample, self.num_class, num_times).permute(0, 2, 1)

            if self.use_last_timestep_only:
                out = out[:, -1:, :]
            return out  # (num_sample, num_time, 2 or num_state)

        x = x.permute(0, 2, 1)

        # Fit the group of mask:
        mask_group = MaskGroup(
            self.pert, self.device, verbose=False, deletion_mode=self.deletion_mode
        )
        mask_group.fit_multiple(
            X=x,
            f=f,
            use_last_timestep_only=self.use_last_timestep_only,
            loss_function_multiple=self.loss,
            area_list=self.area_list,
            learning_rate=1.0,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=self.size_reg_factor_dilation,
            initial_mask_coeff=0.5,
            n_epoch=self.num_epoch,
            momentum=1.0,
            time_reg_factor=self.time_reg_factor,
        )

        # Extract the extremal mask:
        y_test = f(x).unsqueeze(0)
        thresh = cross_entropy_multiple(y_test, y_test)  # This is what we call epsilon in the paper
        mask = mask_group.get_extremal_mask_multiple(thresholds=thresh)
        mask_saliency = mask.permute(0, 2, 1)

        torch.backends.cudnn.enabled = orig_cudnn_setting

        return mask_saliency.detach().cpu().numpy()

    @staticmethod
    def print_mask_saliency(mask):
        mask = mask.detach().cpu().numpy()
        for j in mask:
            xs, ys = np.where(j > 0)
            d = {}
            for i, y in enumerate(ys):
                if y not in d.keys():
                    d[y] = []
                d[y].append(xs[i])
            print(d)

    def get_name(self):
        builder = ["dynamask", self.blur_type]
        if self.deletion_mode:
            builder.append("deletion")
        if self.loss_str != "logloss":
            builder.append("celoss")
        builder.extend(["timereg", str(self.time_reg_factor)])
        builder.extend(["sizereg", str(self.size_reg_factor_dilation)])
        if len(self.area_list) == 1:
            area_str = str(self.area_list[0])
        else:
            area_str = f"{self.area_list[0]}-{self.area_list[-1]}"
        builder.extend(["area", area_str])
        builder.extend(["epoch", str(self.num_epoch)])
        if self.use_last_timestep_only:
            builder.append("lastonly")
        return "_".join(builder)
