import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, device, args, metric_logger, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()

        if if_adapt:
            #print("adaptating..")
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, device, args, metric_logger, self.model, self.optimizer)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, device, args, metric_logger, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model.module.forward_output(x, device, args)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    metric_logger.update(loss_total=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return outputs
