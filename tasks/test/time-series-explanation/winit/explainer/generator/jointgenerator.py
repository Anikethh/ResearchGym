from __future__ import annotations

import itertools
import logging
import pathlib
from time import time

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import constraints

from winit.explainer.generator.generator import BaseFeatureGenerator, GeneratorTrainingResults


class JointFeatureGenerator(BaseFeatureGenerator):
    """
    The joint feature generator. Modified from the FIT repo.
    """

    def __init__(
        self,
        feature_size: int,
        device,
        gen_path: pathlib.Path,
        hidden_size: int,
        latent_size: int = 100,
        prediction_size: int = 1,
        data: str = "mimic",
    ):
        """
        Constructor

        Args:
            feature_size:
                The number of features.
            device:
                The torch device
            gen_path:
                The generator path
            hidden_size:
                The hidden size of the generator RNN.
            latent_size:
                The size of the latent nodes.
            prediction_size:
                The prediction window size.
            data:
                The data name.

        """
        super().__init__(feature_size, device, prediction_size, gen_path)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = self.prediction_size * self.feature_size
        self.data = data
        self.cov_noise = (
            torch.eye(self.output_size, requires_grad=False).unsqueeze(0).to(self.device)
        )

        # Generates the parameters of the distribution
        self.rnn = torch.nn.GRU(self.feature_size, self.hidden_size)
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if "weight" in p:
                    torch.nn.init.normal_(self.rnn.__getattr__(p), 0.0, 0.02)

        mid_layer_size = 100 if self.data == "mimic" else 10

        self.dist_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, mid_layer_size),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(num_features=mid_layer_size),
            torch.nn.Linear(mid_layer_size, self.latent_size * 2),
        )

        self.cov_generator = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, 10),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(num_features=10),
            torch.nn.Linear(10, self.output_size * self.output_size),
            torch.nn.ReLU(),
        )
        self.mean_generator = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, 10),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(num_features=10),
            torch.nn.Linear(10, self.output_size),
        )

        self.log = logging.getLogger(JointFeatureGenerator.__name__)

    def likelihood_distribution(self, mu, std, cov_noise_level=1e-4):
        """
        Return the likelihood distribution of the output given the latent distribution Z. We
        sample 1 from the latent distribution, run the "decoder network" to obtain the mean and
        the covariance of the output variable.

        Args:
            mu:
                The mean of the latent distribution. Shape = (batch_size, latent_size)
            std:
                The std of the latent distribution. Shape = (batch_size, latent_size)
            cov_noise_level:
                Mathematically, our result should generate positive definite covariance matrix.
                However, due to numerical instability, it could fail. We thus add a small noise of
                identity matrix to it.

        Returns:
            The mu and the std of the output distribution. Both of shape
            (batch_size, num_features * prediction_size)

        """

        # sample Z from the distribution
        Z = mu + std * torch.randn_like(mu).to(self.device)
        # Generate the distribution P(X|H,Z)
        mean = self.mean_generator(Z)
        cov_noise = torch.eye(self.output_size).unsqueeze(0).repeat(len(Z), 1, 1) * cov_noise_level
        cov_noise = cov_noise.to(self.device)
        A = self.cov_generator(Z).view(-1, self.output_size, self.output_size)
        covariance = torch.bmm(A, torch.transpose(A, 1, 2)) + cov_noise
        count_loop = 0
        while True:
            valid = constraints.positive_definite.check(covariance)
            if valid.all():
                break
            else:
                error_index = torch.where(~valid)[0]
                covariance[error_index, :, :] = covariance[error_index, :, :] + self.cov_noise * cov_noise_level
                self.log.warning(
                    f"Covariance matrix is not positive definite at {len(error_index)} indices."
                )
                self.log.warning(f"Adding {cov_noise_level}I to the matrix at those indices")
                count_loop += 1
                print(f"count_loop={count_loop}")
                if count_loop > 20:
                    covariance[error_index, :, :] = self.cov_noise
                    self.log.warning(f"Attempt to add more noise failed. Setting that covariance to I")
                    valid_loop = constraints.positive_definite.check(covariance)
                    np.save(f"debug.array.{error_index}.npy",
                            covariance[error_index, :, :].detach().cpu().numpy())
                    if valid_loop.all():
                        break
                    else:
                        self.log.warning(f"Should not be here.")
                        break

        return mean, covariance

    def likelihood_distribution_multisample(self, mu, std, n_samples, cov_noise_level=1e-4):
        """
        Return the likelihood distribution of the output given the latent distribution Z. We
        sample from the latent distribution, run the "decoder network" to obtain the mean and
        the covariance of the output variable.

        Only used at inference.

        Args:
            mu:
                The mean of the latent distribution. Shape = (batch_size, latent_size)
            std:
                The std of the latent distribution. Shape = (batch_size, latent_size)
            n_samples:
                The number of samples to sample from Z.
            cov_noise_level:
                Mathematically, our result should generate positive definite covariance matrix.
                However, due to numerical instability, it could fail. We thus add a small noise of
                identity matrix to it.

        Returns:
            The mu and the std of the output distribution. Both of shape
            (num_samples * batch_size, num_features * prediction_size)

        """
        # sample Z from the distribution
        rand = torch.randn((n_samples, *mu.shape)).to(self.device)
        Z = mu.unsqueeze(0) + std.unsqueeze(0) * rand
        Z = Z.reshape(-1, Z.shape[-1])
        # Generate the distribution P(X|H,Z)
        mean = self.mean_generator(Z)

        A = self.cov_generator(Z).view(-1, self.output_size, self.output_size)
        covariance = torch.bmm(A, torch.transpose(A, 1, 2)) + self.cov_noise * cov_noise_level
        count_loop = 0
        while True:
            valid = constraints.positive_definite.check(covariance)
            if valid.all():
                break
            else:
                error_index = torch.where(~valid)[0]
                covariance[error_index, :, :] = (
                    covariance[error_index, :, :] + self.cov_noise * cov_noise_level
                )
                self.log.warning(
                    f"Covariance matrix is not positive definite at {len(error_index)} indices."
                )
                self.log.warning(
                    f"Adding {cov_noise_level}I to the matrix at" f"those indices"
                )
                count_loop += 1
                if count_loop > 20:
                    covariance[error_index, :, :] = self.cov_noise
                    self.log.warning(f"attempt to add more noise failed. Setting covariance to I")

        return mean, covariance

    def get_z_mu_std(self, past):
        """
        Return the mean and the std of the distribution of the latent variables given the input x.

        Args:
            past:
                The past time series. (batch_size, num_features, num_times)

        Returns:
            A tuple of mu and std of shape (batch_size, latent_size)
        """
        past = past.permute(2, 0, 1)
        _, encoding = self.rnn(past.to(self.device))
        H = encoding.view(encoding.size(1), -1)
        # Find the distribution of the latent variable Z
        mu_std = self.dist_predictor(H)
        mu = mu_std[:, : mu_std.shape[1] // 2]
        std = mu_std[:, mu_std.shape[1] // 2 :]
        return mu, std

    def forward_conditional_multisample_from_z_mu_std(
        self, past, current, sig_inds, mu_z, std_z, n_samples
    ):
        """
        Sample from the distribution of the latent variable Z, run the "decoder" to get the
        distribution X, and sample from it. Return both the sampled and the mu.

        Only used during inference.

        Args:
            past:
                The past time series. Shape = (batch_size, num_features, num_times)
            current:
                The current time series for conditioned on.
                Shape (batch_size, num_features, time_forward)
            sig_inds:
                The list of features to be conditioned on.
            mu_z:
                The mean of the latent variables. Shape = (batch_size, latent_size)
            std_z:
                The std of the latent variables. Shape = (batch_size, latent_size)
            n_samples:
                The number of samples we are sampling from the latent distribution.

        Returns:
            A tuple.
            The first contains the full sample modified.
            Shape = (num_samples, batch_size, num_features, time_forward)
            The second is the mean of the output distribution.
            Shape = (num_samples, batch_size, (num_features - len(sig_inds)) * time_forward)

        """
        if self.feature_size == len(sig_inds):
            return current, current
        current = current.to(self.device)
        if len(current.shape) == 1:
            assert current.shape[0] == self.feature_size
            current = current[None, :, None]
        if len(current.shape) == 2:
            if current.shape[1] == self.feature_size:
                current = current[:, :, None]
            elif current.shape[0] == self.feature_size and current.shape[1] == self.prediction_size:
                current = current[None, :, :]
            else:
                raise RuntimeError("current shape is incompatible!")
        orig_shape = current.shape
        pred_size = orig_shape[-1] if len(orig_shape) > 2 else 1

        current = current.reshape(current.shape[0], -1)

        mean, covariance = self.likelihood_distribution_multisample(
            mu_z, std_z, n_samples
        )  # P(X_t|X_0:t-1)
        old_sig_inds = sig_inds
        sig_inds = [
            list(range(sig_ind * pred_size, (sig_ind + 1) * pred_size)) for sig_ind in sig_inds
        ]
        sig_inds = list(itertools.chain.from_iterable(sig_inds))
        sig_inds_comp = list(set(range(current.shape[1])) - set(sig_inds))

        full_sample = current.unsqueeze(0).repeat(n_samples, 1, 1).reshape(-1, current.shape[-1])

        ind_len = len(sig_inds)
        ind_len_not = len(sig_inds_comp)
        x_ind = full_sample[:, sig_inds].view(-1, ind_len)
        mean_1 = mean[:, sig_inds_comp].view(-1, ind_len_not)
        cov_1_2 = covariance[:, sig_inds_comp, :][:, :, sig_inds].view(-1, ind_len_not, ind_len)
        cov_2_2 = covariance[:, sig_inds, :][:, :, sig_inds].view(-1, ind_len, ind_len)
        cov_1_1 = covariance[:, sig_inds_comp, :][:, :, sig_inds_comp].view(
            -1, ind_len_not, ind_len_not
        )
        # intermediate, _ = torch.linalg.solve(cov_1_2.transpose(1, 2), cov_2_2)
        # intermediate = intermediate.transpose(1, 2)
        intermediate = torch.bmm(cov_1_2, torch.inverse(cov_2_2))
        mean_cond = mean_1 + torch.bmm(
            (intermediate), (x_ind - mean[:, sig_inds]).view(-1, ind_len, 1)
        ).squeeze(-1)
        covariance_cond = cov_1_1 - torch.bmm(intermediate, torch.transpose(cov_1_2, 2, 1))

        # covariance_cond may not be positive definite due to numerical instability.
        error_indexes = []
        while True:
            try:
                # P(x_{-i,t}|x_{i,t})
                valid = constraints.positive_definite.check(covariance_cond)
                if not valid.all():
                    error_index = torch.where(~valid)[0].detach().cpu().numpy()
                    error_indexes.extend(error_index)
                    covariance_cond[error_index, :, :] = torch.eye(covariance_cond.shape[1])[
                        None, :, :
                    ].to(self.device)
                likelihood = MultivariateNormal(loc=mean_cond, covariance_matrix=covariance_cond)
                sample = likelihood.rsample()
                full_sample[:, sig_inds_comp] = sample
                full_sample = full_sample.reshape(n_samples, *orig_shape)
                break
            except RuntimeError as e:
                self.log.warning("SHOULD NOT GET HERE!")
                # Very rarely, covariance_cond matrix will be singular causing the above to fail
                error_index = int(str(e).split(" ")[3].split(":")[0])
                error_indexes.append(error_index)
                covariance_cond[error_index] = torch.eye(covariance_cond.shape[1]).to(self.device)

        if len(error_indexes) > 0:
            # So, we will carry forward in this case
            for error_index in error_indexes:
                original_sample_index = error_index // orig_shape[0]
                original_batch_index = error_index % orig_shape[0]
                # full_sample.shape = (n_samples, batch_size, feature_size, time)
                full_sample[original_sample_index, original_batch_index, old_sig_inds, :] = past[
                    original_batch_index, old_sig_inds, -1:
                ]

            self.log.warning(f"WARNING: Carrying forward instead on {len(error_indexes)} index.")
            full_sample = full_sample.view(n_samples, *orig_shape)

        return full_sample, mean[:, sig_inds_comp].reshape(n_samples, -1, ind_len_not)

    def forward(self, past):
        mu, std = self.get_z_mu_std(past)
        mean, covariance = self.likelihood_distribution(mu, std)
        likelihood = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        )
        return likelihood.rsample()

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        num: int,
        run_train: bool = True,
        optimizer: Optimizer | None = None,
        mse_loss: bool = None,
    ) -> float:
        if run_train:
            self.train()
            if optimizer is None:
                raise ValueError("optimizer is none in train mode.")
        else:
            self.eval()
        epoch_loss = 0

        for i, (signals, _, _) in enumerate(dataloader):
            signals = signals.transpose(1,2)
            
            if num == 1:
                timepoints = [signals.shape[2] - self.prediction_size]
            else:
                timepoints = [
                    int(tt)
                    for tt in np.logspace(
                        1.0, np.log10(signals.shape[2] - self.prediction_size), num=num
                    )
                ]

            for t in timepoints:
                if run_train:
                    optimizer.zero_grad()
                label = (
                    signals[:, :, t : t + self.prediction_size]
                    .reshape(signals.shape[0], -1)
                    .to(self.device)
                )

                mu, std = self.get_z_mu_std(signals[:, :, :t])
                mean, covariance = self.likelihood_distribution(mu, std)
                dist = MultivariateNormal(loc=mean, covariance_matrix=covariance)
                if mse_loss:
                    prediction = dist.rsample()
                    loss = torch.nn.MSELoss(reduction="none")(prediction, label)
                    loss = loss.mean()
                else:
                    loss = -dist.log_prob(label).mean()
                epoch_loss = epoch_loss + loss.item()
                if run_train:
                    loss.backward(retain_graph=True)
                    optimizer.step()
        return float(epoch_loss) / len(dataloader)

    def train_generator(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs,
        **kwargs,
    ) -> GeneratorTrainingResults:
        """
        Train the generator.

        Args:
            train_loader:
                The train loader
            valid_loader:
                The validation loader
            num_epochs:
                Number of epochs for the training
            **kwargs:
                Additional kwargs.
        Returns:
            The training results for the generator.
        """
        tic = time()
        self.to(self.device)

        # Overwrite default learning parameters if values are passed
        default_params = {"lr": 0.001, "weight_decay": 0}
        for k, v in kwargs.items():
            if k in default_params.keys():
                default_params[k] = v

        parameters = self.parameters()
        optimizer = torch.optim.Adam(
            parameters, lr=default_params["lr"], weight_decay=default_params["weight_decay"]
        )

        best_loss = 1000000
        best_epoch = -1

        train_loss_trends = np.zeros((1, num_epochs + 1))
        valid_loss_trends = np.zeros((1, num_epochs + 1))
        best_epochs = np.zeros(1, dtype=int)

        for epoch in range(num_epochs + 1):
            self._run_one_epoch(train_loader, 3, True, optimizer, mse_loss=False)
            train_loss = self._run_one_epoch(train_loader, 10, False, None, mse_loss=False)
            valid_loss = self._run_one_epoch(valid_loader, 10, False, None, mse_loss=False)
            train_loss_trends[0, epoch] = train_loss
            valid_loss_trends[0, epoch] = valid_loss

            if self.verbose and epoch % self.verbose_eval == 0:
                print(f"\nEpoch {epoch}")
                print(f"Generator Training Loss   ===> {train_loss}")
                print(f"Generator Validation Loss ===> {valid_loss}")

            if self.early_stopping:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    torch.save(self.state_dict(), str(self._get_model_file_name()))
                    if self.verbose:
                        print(f"save ckpt:in epoch {epoch}")
        best_epochs[0] = best_epoch

        print(f"Joint generator test loss = {best_loss:.6f}   Time elapsed: {time() - tic}")
        return GeneratorTrainingResults(
            self.get_name(), train_loss_trends, valid_loss_trends, best_epochs
        )

    def test_generator(self, test_loader) -> float:
        """
        Test the generator

        Args:
            test_loader:
                The test data

        Returns:
            The test MSE result for the generator.

        """
        self.to(self.device)
        test_loss = self._run_one_epoch(test_loader, 10, False, None, mse_loss=True)
        return test_loss

    def _get_model_file_name(self) -> pathlib.Path:
        self.gen_path.mkdir(parents=True, exist_ok=True)
        return self.gen_path / f"len_{self.prediction_size}.pt"

    def load_generator(self):
        """
        Load the generator from the disk.
        """
        self.load_state_dict(torch.load(str(self._get_model_file_name())))
        self.to(self.device)

    @staticmethod
    def get_name() -> str:
        return "joint_generator"
