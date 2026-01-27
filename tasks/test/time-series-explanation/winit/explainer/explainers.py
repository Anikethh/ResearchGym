from __future__ import annotations

import abc
import logging

import numpy as np
import torch
from captum.attr import IntegratedGradients, DeepLift, GradientShap

from winit.explainer.generator.generator import GeneratorTrainingResults
from winit.models import TorchModel
from winit.utils import resolve_device


class BaseExplainer(abc.ABC):
    """
    A base class for explainer.
    """

    def __init__(self, device=None):
        """
        Constructor.

        Args:
            device:
               The torch device.
        """
        self.base_model: TorchModel | None = None
        self.device = resolve_device(device)

    @abc.abstractmethod
    def attribute(self, x):
        """
        The attribution method that the explainer will give.
        Args:
            x:
                The input tensor.

        Returns:
            The attribution with respect to x. The shape should be the same as x, or it could
            be one dimension greater than x if there is aggregation needed.

        """

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults | None:
        """
        If the explainer or attribution method needs a generator, this will train the generator.

        Args:
            train_loader:
                The dataloader for training
            valid_loader:
                The dataloader for validation.
            num_epochs:
                The number of epochs.

        Returns:
            The training results for the generator, if applicable. This includes the
            training curves.

        """
        return None

    def test_generators(self, test_loader) -> float | None:
        """
        If the explainer or attribution method needs a generator, this will return the performance
        of the generator on the test set.

        Args:
            test_loader:
                The dataloader for testing.

        Returns:
            The test result (MSE) for the generator, if applicable.

        """
        return None

    def load_generators(self) -> None:
        """
        If the explainer or attribution method needs a generator, this will load the generator from
        the disk.
        """

    def set_model(self, model, set_eval=True) -> None:
        """
        Set the base model the explainer wish to explain.

        Args:
            model:
                The base model.
            set_eval:
                Indicating whether we set to eval mode for the explainer. Note that in some cases
                like Dynamask or FIT, they do not set the model to eval mode.
        """
        self.base_model = model
        if set_eval:
            self.base_model.eval()
        self.base_model.to(self.device)

    @abc.abstractmethod
    def get_name(self):
        """
        Return the name of the explainer.
        """


class MockExplainer(BaseExplainer):
    """
    Class for mock explainer. The mock explainer returns all the attributes to 0.
    """

    def __init__(self):
        super().__init__()

    def attribute(self, x, **kwargs):
        return np.zeros(x.shape)

    def get_name(self):
        return "mock"


class DeepLiftExplainer(BaseExplainer):
    """
    The explainer for the DeepLIFT method using zeros as the baseline and captum for the
    implementation.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model)
        self.explainer = DeepLift(self.base_model)

    def attribute(self, x):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        assert self.base_model.num_states == 1, "TODO: Implement retrospective for > 1 class"
        score = self.explainer.attribute(x, baselines=(x * 0), additional_forward_args=(False))
        score = abs(score.detach().cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "deeplift"


class FOExplainer(BaseExplainer):
    """
    The explainer for feature occlusion. The implementation is simplified from the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, device, n_samples=10, **kwargs):
        super().__init__(device)
        self.n_samples = n_samples
        if len(kwargs) > 0:
            log = logging.getLogger(FOExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x):
        self.base_model.eval()
        self.base_model.zero_grad()

        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)

        for t in range(1, t_len):
            p_y_t = self.base_model.predict(x[:, :, : t + 1], return_all=False)
            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(self.n_samples):
                    x_hat[:, i, t] = torch.Tensor(np.random.uniform(-3, +3, size=(len(x),)))
                    y_hat_t = self.base_model.predict(x_hat, return_all=False)
                    kl = torch.abs(y_hat_t - p_y_t)
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl
        return score

    def get_name(self):
        if self.n_samples != 10:
            return f"fo_sample_{self.n_samples}"
        return "fo"


class AFOExplainer(BaseExplainer):
    """
    The explainer for augmented feature occlusion. The implementation is simplified from
    the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, device, train_loader, n_samples=10, **kwargs):
        super().__init__(device)
        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])
        self.n_samples = n_samples
        if len(kwargs) > 0:
            log = logging.getLogger(AFOExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x):
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)

        self.base_model.eval()
        self.base_model.zero_grad()

        for t in range(1, t_len):
            p_y_t = self.base_model.predict(x[:, :, : t + 1], return_all=False)
            for i in range(n_features):
                feature_dist = np.array(self.data_distribution[:, i, :]).reshape(-1)
                x_hat = x[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(self.n_samples):
                    x_hat[:, i, t] = torch.Tensor(
                        np.random.choice(feature_dist, size=(len(x),))
                    ).to(self.device)
                    y_hat_t = self.base_model.predict(x_hat, return_all=False)
                    kl = torch.abs((y_hat_t[:, :]) - (p_y_t[:, :]))
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl
        return score

    def get_name(self):
        if self.n_samples != 10:
            return f"afo_sample_{self.n_samples}"
        return "afo"


class IGExplainer(BaseExplainer):
    """
    The explainer for integrated gradients using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model, set_eval=set_eval)
        self.explainer = IntegratedGradients(self.base_model)

    def attribute(self, x):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        score = self.explainer.attribute(x, baselines=(x * 0), additional_forward_args=(False))
        score = np.abs(score.detach().cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "ig"


class GradientShapExplainer(BaseExplainer):
    """
    The explainer for gradient shap using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model, set_eval=set_eval)
        self.explainer = GradientShap(self.base_model)

    def attribute(self, x):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x = x.to(self.device)
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        score = self.explainer.attribute(
            x, n_samples=50, stdevs=0.0001, baselines=torch.cat([x * 0, x * 1]), additional_forward_args=(False)
        )
        score = abs(score.cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "gradientshap"
