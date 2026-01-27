"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, device, args, metric_logger):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            loss, ema, reset_flag, outputs = forward_and_adapt_sar(x, device, args, metric_logger, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss
        
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, device, args, metric_logger, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """

    optimizer.zero_grad()

    # First forward pass
    outputs = model.module.forward_output(x, device, args)

    # Adaptation step
    entropys = softmax_entropy(outputs)

    # Calculate the threshold for the top `margin`% smallest entropys
    k = int(len(entropys) * margin)
    top_k_values, _ = torch.topk(entropys, k, largest=False)
    threshold = top_k_values[-1]  # The k-th smallest entropy value

    # Filter out elements with entropy less than or equal to the threshold
    filter_ids_1 = torch.where(entropys <= threshold)
    

    if filter_ids_1[0].numel() > 0:  # Ensure there are valid elements after filtering
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
    else:
        loss = torch.tensor(0.0, device=entropys.device, requires_grad=True)  # Set loss to 0 if no valid samples

    loss.backward()
    optimizer.first_step(zero_grad=True)  # Compute \hat{\epsilon(\Theta)} for first-order approximation

    # Second forward pass
    outputs2 = model.module.forward_output(x, device, args)
    entropys2 = softmax_entropy(outputs2)

    if filter_ids_1[0].numel() > 0:  # Ensure there are valid elements after the first filtering
        entropys2 = entropys2[filter_ids_1]  
        filter_ids_2 = torch.where(entropys2 < threshold)  # Re-filter reliable samples after model update

        if filter_ids_2[0].numel() > 0:  # Ensure there are valid elements after re-filtering
            loss_second = entropys2[filter_ids_2].mean(0)
        else:
            loss_second = torch.tensor(0.0, device=entropys2.device, requires_grad=True)  # Set loss to 0 if no valid samples
    else:
        loss_second = torch.tensor(0.0, device=entropys2.device, requires_grad=True)  # Set loss to 0 if no valid samples after first filtering

    if not torch.isnan(loss_second):
        ema = update_ema(ema, loss_second.item())  # Record moving average loss values for model recovery

    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            # print("ema < 0.2, now reset the model")
            reset_flag = True

    metric_logger.update(loss_total=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # print(loss)
    return loss, ema, reset_flag, outputs


def check_model(model):
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"

"""
from https://github.com/davda54/sam
"""
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups