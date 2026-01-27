from __future__ import annotations

import logging
import pathlib
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from winit.explainer.explainers import BaseExplainer
from winit.explainer.generator.generator import (
    FeatureGenerator,
    BaseFeatureGenerator,
    GeneratorTrainingResults,
)
from winit.explainer.generator.jointgenerator import JointFeatureGenerator


class WinITExplainer(BaseExplainer):
    """
    The explainer for our method WinIT
    """

    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: DataLoader | None = None,
        window_size: int = 10,
        num_samples: int = 3,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        random_state: int | None = None,
        **kwargs,
    ):
        """
        Construtor

        Args:
            device:
                The torch device.
            num_features:
                The number of features.
            data_name:
                The name of the data.
            path:
                The path indicating where the generator to be saved.
            train_loader:
                The train loader if we are using the data distribution instead of a generator
                for generating counterfactual. Default=None.
            window_size:
                The window size for the WinIT
            num_samples:
                The number of Monte-Carlo samples for generating counterfactuals.
            conditional:
                Indicate whether the individual feature generator we used are conditioned on
                the current features. Default=False
            joint:
                Indicate whether we are using the joint generator.
            metric:
                The metric for the measures of comparison of the two distributions for i(S)_a^b
            random_state:
                The random state.
            **kwargs:
                There should be no additional kwargs.
        """
        super().__init__(device)
        self.window_size = window_size
        self.num_samples = num_samples
        self.num_features = num_features
        self.data_name = data_name
        self.joint = joint
        self.conditional = conditional
        self.metric = metric

        self.generators: BaseFeatureGenerator | None = None
        self.path = path
        if train_loader is not None:
            self.data_distribution = (
                torch.stack([x['x'] for x in train_loader.dataset]).transpose(1, 2).detach().cpu().numpy()
            )
        else:
            self.data_distribution = None
        self.rng = np.random.default_rng(random_state)

        self.log = logging.getLogger(WinITExplainer.__name__)
        if len(kwargs):
            self.log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def _model_predict(self, x):
        p = self.base_model.predict(x.transpose(1, 2), return_all=False)

        return p

    def attribute(self, x):
        """
        Compute the WinIT attribution.

        Args:
            x:
                The input Tensor of shape (batch_size, num_features, num_times)

        Returns:
            The attribution Tensor of shape (batch_size, num_features, num_times, window_size)
            The (i, j, k, l)-entry is the importance of observation (i, j, k - window_size + l + 1)
            to the prediction at time k

        """
        x = x.transpose(1, 2)
        self.base_model.eval()
        self.base_model.zero_grad()

        with torch.no_grad():
            tic = time()

            batch_size, num_features, num_timesteps = x.shape
            scores = []

            for t in range(num_timesteps):
                window_size = min(t, self.window_size)

                if t == 0:
                    scores.append(np.zeros((batch_size, num_features, self.window_size)))
                    continue

                # x = (num_sample, num_feature, n_timesteps)
                p_y = self._model_predict(x[:, :, : t + 1])

                iS_array = np.zeros((num_features, window_size, batch_size), dtype=float)
                for n in range(window_size):
                    time_past = t - n
                    time_forward = n + 1
                    counterfactuals = self._generate_counterfactuals(
                        time_forward, x[:, :, :time_past], x[:, :, time_past : t + 1]
                    )
                    # counterfactual shape = (num_feat, num_samples, batch_size, time_forward)
                    for f in range(num_features):
                        # repeat input for num samples
                        x_hat_in = (
                            x[:, :, : t + 1].unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
                        )  # (ns, bs, f, time)
                        # replace unknown with counterfactuals
                        x_hat_in[:, :, f, time_past : t + 1] = counterfactuals[f, :, :, :]

                        # Compute Q = p(y_t | tilde(X)^S_{t-n:t})
                        p_y_hat = self._model_predict(
                            x_hat_in.reshape(self.num_samples * batch_size, num_features, t + 1)
                        )

                        # Compute P = p(y_t | X_{1:t})
                        p_y_exp = (
                            p_y.unsqueeze(0)
                            .repeat(self.num_samples, 1, 1)
                            .reshape(self.num_samples * batch_size, p_y.shape[-1])
                        )
                        iSab_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                            self.num_samples, batch_size
                        )
                        iSab = torch.mean(iSab_sample, dim=0).detach().cpu().numpy()
                        # For KL, the metric can be unbounded. We clip it for numerical stability.
                        iSab = np.clip(iSab, -1e6, 1e6)
                        iS_array[f, n, :] = iSab

                # Compute the I(S) array
                b = iS_array[:, 1:, :] - iS_array[:, :-1, :]
                iS_array[:, 1:, :] = b

                score = iS_array[:, ::-1, :].transpose(2, 0, 1)  # (bs, nfeat, time)

                # Pad the scores when time forward is less than window size.
                if score.shape[2] < self.window_size:
                    score = np.pad(score, ((0, 0), (0, 0), (self.window_size - score.shape[2], 0)))
                scores.append(score)
            print(f"Batch done: Time elapsed: {(time() - tic):.4f}")

            scores = np.stack(scores).transpose((1, 2, 0, 3))  # (bs, fts, ts, window_size)
            return scores

    def _compute_metric(self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric for comparisons of two distributions.

        Args:
            p_y_exp:
                The current expected distribution. Shape = (batch_size, num_states)
            p_y_hat:
                The modified (counterfactual) distribution. Shape = (batch_size, num_states)

        Returns:
            The result Tensor of shape (batch_size).

        """
        if self.metric == "kl":
            return torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp), -1)
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            diff = torch.abs(p_y_hat - p_y_exp)
            return torch.sum(diff, -1)
        raise Exception(f"unknown metric. {self.metric}")

    def _init_generators(self):
        if self.joint:
            gen_path = self.path / "joint_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = JointFeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=self.num_features * 3,
                prediction_size=self.window_size,
                data=self.data_name,
            )
        else:
            gen_path = self.path / "feature_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = FeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=50,
                prediction_size=self.window_size,
                conditional=self.conditional,
                data=self.data_name,
            )

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults:
        self._init_generators()
        return self.generators.train_generator(train_loader, valid_loader, num_epochs)

    def test_generators(self, test_loader) -> float:
        test_loss = self.generators.test_generator(test_loader)
        print(f"Generator Test MSE Loss: {test_loss}")
        return test_loss

    def load_generators(self) -> None:
        self._init_generators()
        self.generators.load_generator()

    def _generate_counterfactuals(
        self, time_forward: int, x_in: torch.Tensor, x_current: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate the counterfactuals.

        Args:
            time_forward:
                Number of timesteps of counterfactuals we wish to generate.
            x_in:
                The past Tensor. Shape = (batch_size, num_features, num_times)
            x_current:
                The current Tensor if a conditional generator is used.
                Shape = (batch_size, num_features, time_forward). If the generator is not
                conditional, x_current is None.

        Returns:
            Counterfactual of shape (num_features, num_samples, batch_size, time_forward)

        """
        # x_in shape (bs, num_feature, num_time)
        # x_current shape (bs, num_feature, time_forward)
        # return counterfactuals shape (num_feature, num_samples, batchsize, time_forward)
        batch_size, _, num_time = x_in.shape
        if self.data_distribution is not None:
            # Random sample instead of using generator
            counterfactuals = torch.zeros(
                (self.num_features, self.num_samples, batch_size, time_forward), device=self.device
            )
            for f in range(self.num_features):
                values = self.data_distribution[:, f, :].reshape(-1)
                counterfactuals[f, :, :, :] = torch.Tensor(
                    self.rng.choice(values, size=(self.num_samples, batch_size, time_forward)),
                ).to(self.device)
            return counterfactuals

        if isinstance(self.generators, FeatureGenerator):
            mu, std = self.generators.forward(x_current, x_in, deterministic=True)
            mu = mu[:, :, :time_forward]
            std = std[:, :, :time_forward]  # (bs, f, time_forward)
            counterfactuals = mu.unsqueeze(0) + torch.randn(
                self.num_samples, batch_size, self.num_features, time_forward, device=self.device
            ) * std.unsqueeze(0)
            return counterfactuals.permute(2, 0, 1, 3)

        if isinstance(self.generators, JointFeatureGenerator):
            counterfactuals = torch.zeros(
                (self.num_features, self.num_samples, batch_size, time_forward), device=self.device
            )
            for f in range(self.num_features):
                mu_z, std_z = self.generators.get_z_mu_std(x_in)
                gen_out, _ = self.generators.forward_conditional_multisample_from_z_mu_std(
                    x_in,
                    x_current,
                    list(set(range(self.num_features)) - {f}),
                    mu_z,
                    std_z,
                    self.num_samples,
                )
                # gen_out shape (ns, bs, num_feature, time_forward)
                counterfactuals[f, :, :, :] = gen_out[:, :, f, :]
            return counterfactuals

        raise ValueError("Unknown generator or no data distribution provided.")

    def get_name(self):
        builder = ["winit", "window", str(self.window_size)]
        if self.num_samples != 3:
            builder.extend(["samples", str(self.num_samples)])
        if self.conditional:
            builder.append("cond")
        if self.joint:
            builder.append("joint")
        builder.append(self.metric)
        if self.data_distribution is not None:
            builder.append("usedatadist")
        return "_".join(builder)
