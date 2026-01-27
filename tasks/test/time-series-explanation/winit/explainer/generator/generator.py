from __future__ import annotations

import abc
import dataclasses
import logging
import pathlib
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclasses.dataclass
class GeneratorTrainingResults:
    name: str
    train_loss_trends: np.ndarray  # (num_features, epoch)
    valid_loss_trends: np.ndarray  # (num_features, epoch)
    best_epochs: np.ndarray  # (num_features)


class BaseFeatureGenerator(torch.nn.Module, abc.ABC):
    """
    Abstract class for a base feature generator.
    """

    def __init__(
        self,
        feature_size: int,
        device,
        prediction_size: int,
        gen_path: pathlib.Path,
        verbose: bool = True,
        verbose_eval: int = 10,
        early_stopping: bool = True,
    ):
        """
        Constructor

        Args:
            feature_size:
                The number of features.
            device:
                The torch device
            prediction_size:
                The output window size.
            gen_path:
                The path to the generator checkpoints to be saved.
            verbose:
                Verbosity.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
                Apply early stopping
        """
        super().__init__()
        self.feature_size = feature_size
        self.device = device
        self.prediction_size = prediction_size
        self.gen_path = gen_path
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.early_stopping = early_stopping

    @abc.abstractmethod
    def train_generator(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
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

    @abc.abstractmethod
    def test_generator(self, test_loader: DataLoader) -> float:
        """
        Test the generator

        Args:
            test_loader:
                The test data

        Returns:
            The test MSE result for the generator.

        """

    @abc.abstractmethod
    def load_generator(self):
        """
        Load the generator from the disk.
        """

    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        """
        Return the name of the generator.
        """


class IndividualFeatureGenerator(torch.nn.Module):
    """
    A class that describe the feature generator for 1 feature.
    """

    def __init__(
        self,
        feature_size: int,
        feature_index: int,
        device,
        gen_path: pathlib.Path,
        hidden_size: int = 100,
        prediction_size: int = 1,
        conditional: bool = True,
        data: str = "mimic",
    ):
        """
        Constructor

        Args:
            feature_size:
                The number of features.
            feature_index:
                The feature index for the generator
            device:
                The torch device
            gen_path:
                The generator path
            hidden_size:
                The hidden size of the generator RNN.
            prediction_size:
                The prediction window size.
            conditional:
                Whether the generator is conditional on present observations as well.
            data:
                The data name.
        """
        super().__init__()
        self.feature_index = feature_index
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.conditional = conditional
        self.prediction_size = prediction_size
        self.gen_path = gen_path
        self.device = device
        self.data = data

        self.rnn = torch.nn.GRU(self.feature_size, self.hidden_size)
        # f_size is the size of the input to the regressor, it equals the hidden size of
        # the recurrent model if observation is conditioned on the past only
        # If it is also conditioned on current observations of other dimensions
        # the size will be hidden_size+number of other dimensions
        f_size = self.hidden_size
        if conditional:
            f_size = f_size + (self.feature_size - 1) * self.prediction_size

        if self.data == "mimic":
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(f_size, 200),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=200),
                torch.nn.Linear(200, self.prediction_size * 2),
            )
        else:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(f_size, 50),
                torch.nn.Tanh(),
                torch.nn.BatchNorm1d(num_features=50),
                torch.nn.Linear(50, self.prediction_size * 2),
            )

    def forward(self, x, past, deterministic=False):
        """
        The forward function.

        Args:
            x:
                The current tensor shape = (num_samples, num_features, time_forward). If
                time_forward is less than predictionsize, padding will be applied. Only used in
                conditional generators.
            past:
                The past tensor. Shape = (num_samples, num_features, time_past)
            deterministic:
                See return.

        Returns:
            If deterministic, the output will be a tuple of mu and std describing the normal
            distribution. Shape = (num_samples, prediction_size)
            Otherwise, the output will be sampled from this distribution, together with mu.
            Shape = (num_samples, prediction_size)
        """
        past = past.permute(2, 0, 1)
        _, encoding = self.rnn(past.to(self.device))
        if self.conditional:
            if len(x.shape) == 2:
                x = x.unsqueeze(2)
            x = torch.cat(
                (x[:, : self.feature_index, :], x[:, self.feature_index + 1 :, :]), 1
            ).to(self.device)
            if x.shape[2] != self.prediction_size:
                # not enough current observations
                num_to_pad: int = self.prediction_size - x.shape[2]
                x = torch.nn.functional.pad(x, [0, num_to_pad])
            x = x.reshape(x.shape[0], -1)
            x = torch.cat((encoding.view(encoding.size(1), -1), x), 1)
        else:
            x = encoding.view(encoding.size(1), -1)

        mu_std = self.predictor(x)
        mu = mu_std[:, 0 : mu_std.shape[1] // 2]  # (num_samples, prediction_size)
        std = mu_std[:, mu_std.shape[1] // 2 :]
        if deterministic:
            return mu, std
        else:
            reparam_samples = mu + std * torch.randn_like(mu).to(self.device)
            return reparam_samples, mu

    def forward_all(self, x, deterministic=False):
        """
        Run forward at all timesteps.

        Args:
            x:
                The whole time series. Shape = (num_samples, num_features, num_times)
            deterministic:
                See return.

        Returns:
            If deterministic, the output will be a tuple of mu and std describing the normal
            distribution. Shape = (num_time, num_samples, prediction_size)
            Otherwise, the output will be sampled from this distribution, together with mu.
            Shape = (num_time, num_samples, prediction_size)
        """
        x = x.permute(2, 0, 1)  # (t, bs, f)
        all_encoding, _ = self.rnn(x.to(self.device))
        # all_encoding.shape = (t, bs, h)
        if self.conditional:
            x = torch.cat(
                (x[:, :, : self.feature_index], x[:, :, self.feature_index + 1 :]), 2
            ).to(
                self.device
            )  # (t, bs, f-1)
            x = torch.nn.functional.pad(
                x, [0, 0, 0, 0, 0, self.prediction_size - 1]
            )  # (t + p - 1, bs, f - 1)
            x = x.unfold(0, self.prediction_size, 1)  # (t, bs, f - 1, p)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # (t, bs, (f-1)*p)
            x = torch.cat([all_encoding, x], 2)  # (t, bs, (h + (f-1)*p)
        else:
            x = all_encoding

        mu_std = self.predictor(x.reshape(-1, x.shape[2])).reshape(
            x.shape[0], x.shape[1], -1
        )
        mu = mu_std[:, :, 0 : mu_std.shape[2] // 2]  # (t, bs, window)
        std = mu_std[:, :, mu_std.shape[2] // 2 :]  # (t, bs, window)
        if deterministic:
            return mu, std
        else:
            reparam_samples = mu + std * torch.randn_like(mu).to(self.device)
            return reparam_samples, mu

    def run_one_epoch(
        self, dataloader: DataLoader, run_train: bool, optimizer=None
    ) -> float:
        """
        Run one epoch of training or inference.

        Args:
            dataloader:
                The data loader.
            run_train:
                Whether in training mode or not. If not, optimizer is not needed.
            optimizer:
                The optimizer when it is on training mode.

        Returns:
            The average loss for the epoch.
        """
        if run_train:
            self.train()
            if optimizer is None:
                raise ValueError("optimizer is none in train mode.")
        else:
            self.eval()
        epoch_loss = 0
        for i, (signals, _, _) in enumerate(dataloader):
            signals = signals.transpose(1,2)
            label = signals[:, self.feature_index, :].transpose(0, 1)
            if run_train:
                optimizer.zero_grad()
            prediction, mus = self.forward_all(signals)
            total_time = len(prediction)
            all_reconstruction_loss = []
            for j in range(self.prediction_size):
                pred_slice = prediction[: total_time - j - 1, :, j]
                label_slice = label[j + 1 :, :]
                reconstruction_loss = F.mse_loss(
                    pred_slice, label_slice.to(self.device), reduction="none"
                )
                all_reconstruction_loss.append(reconstruction_loss.flatten())
            total_reconstruction_loss = torch.mean(
                torch.cat(all_reconstruction_loss, dim=0)
            )
            epoch_loss = epoch_loss + total_reconstruction_loss.item()
            if run_train:
                total_reconstruction_loss.backward()
                optimizer.step()
        return float(epoch_loss) / len(dataloader)

    def _get_model_file_name(self) -> pathlib.Path:
        self.gen_path.mkdir(parents=True, exist_ok=True)
        return (
            self.gen_path
            / f"feature_{self.feature_index}_len_{self.prediction_size}_cond_{self.conditional}.pt"
        )


class FeatureGenerator(BaseFeatureGenerator):
    """
    A class for a collection of individual feature generator for all features.
    """

    def __init__(
        self,
        feature_size,
        device,
        gen_path: pathlib.Path,
        hidden_size=100,
        prediction_size=1,
        conditional=True,
        data="mimic",
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
            prediction_size:
                The prediction window size.
            conditional:
                Whether the generator is conditional on present observations as well.
            data:
                The data name.

        """
        super().__init__(feature_size, device, prediction_size, gen_path)
        modules = []
        for feature_index in range(feature_size):
            modules.append(
                IndividualFeatureGenerator(
                    feature_size=feature_size,
                    feature_index=feature_index,
                    device=device,
                    gen_path=gen_path,
                    hidden_size=hidden_size,
                    prediction_size=prediction_size,
                    conditional=conditional,
                    data=data,
                )
            )

        self.hidden_size = hidden_size
        self.conditional = conditional
        self.data = data
        self.models = torch.nn.ModuleList(modules)
        self.log = logging.getLogger(FeatureGenerator.__name__)

    def forward(self, x, past, deterministic=False):
        """
        Run forward for each individual feature generator.

        Args:
            x:
                The current tensor shape = (num_samples, num_features, time_forward). If
                time_forward is less than predictionsize, padding will be applied. Only used in
                conditional generators.
            past:
                The past tensor. Shape = (num_samples, num_features, time_past)
            deterministic:
                See return.

        Returns:
            If deterministic, the output will be a tuple of mu and std describing the normal
            distribution. Shape = (num_samples, num_features, prediction_size)
            Otherwise, the output will be sampled from this distribution, together with mu.
            Shape = (num_samples, num_features, prediction_size)

        """
        firsts, seconds = [], []
        for model in self.models:
            first, second = model.forward(x, past, deterministic)
            firsts.append(first)
            seconds.append(second)
        firsts = torch.stack(firsts, dim=1)
        seconds = torch.stack(seconds, dim=1)
        return firsts, seconds  # (bs, feat, window)

    def forward_all(self, x, deterministic=False):
        """
        Run forward at all timesteps.

        Args:
            x:
                The whole time series. Shape = (num_samples, num_features, num_times)
            deterministic:
                See return.

        Returns:
            If deterministic, the output will be a tuple of mu and std describing the normal
            distribution. Shape = (num_time, num_samples, num_features, prediction_size)
            Otherwise, the output will be sampled from this distribution, together with mu.
            Shape = (num_time, num_samples, num_features, prediction_size)
        """
        firsts, seconds = [], []
        for model in self.models:
            first, second = model.forward_all(x, deterministic)
            firsts.append(first)
            seconds.append(second)
        firsts = torch.stack(firsts, dim=2)
        seconds = torch.stack(seconds, dim=2)
        return firsts, seconds  # (t, bs, f, window)

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
        train_loss_trends = np.zeros((self.feature_size, num_epochs + 1))
        valid_loss_trends = np.zeros((self.feature_size, num_epochs + 1))
        best_epochs = np.zeros(self.feature_size, dtype=int)

        for feature_to_predict, generator_model in enumerate(self.models):
            tic = time()

            generator_model.to(self.device)

            # Overwrite default learning parameters if values are passed
            default_params = {
                "lr": 0.0001,
                "weight_decay": 1e-3,
                "generator_type": "RNN_generator",
            }
            for k, v in kwargs.items():
                if k in default_params.keys():
                    default_params[k] = v

            parameters = generator_model.parameters()
            optimizer = torch.optim.Adam(
                parameters,
                lr=default_params["lr"],
                weight_decay=default_params["weight_decay"],
            )

            best_loss = 1000000
            best_epoch = -1
            for epoch in range(num_epochs + 1):
                generator_model.run_one_epoch(train_loader, True, optimizer)
                train_loss = generator_model.run_one_epoch(train_loader, False, None)
                valid_loss = generator_model.run_one_epoch(valid_loader, False, None)
                train_loss_trends[feature_to_predict, epoch] = train_loss
                valid_loss_trends[feature_to_predict, epoch] = valid_loss

                if self.verbose and epoch % self.verbose_eval == 0:
                    print(f"\nEpoch {epoch}")
                    print(f"Generator Training Loss   ===> {train_loss}")
                    print(f"Generator Validation Loss ===> {valid_loss}")

                if self.early_stopping:
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_epoch = epoch
                        torch.save(
                            generator_model.state_dict(),
                            str(generator_model._get_model_file_name()),
                        )
                        if self.verbose:
                            print(f"save ckpt:in epoch {epoch}")

            print(f"***** Training feature {feature_to_predict} *****")
            print(
                f"Validation loss: {valid_loss_trends[feature_to_predict, -1]} "
                f"Time elapsed: {time() - tic}"
            )
            best_epochs[feature_to_predict] = best_epoch
        return GeneratorTrainingResults(
            self.get_name(), train_loss_trends, valid_loss_trends, best_epochs
        )

    def test_generator(self, test_loader, **kwargs) -> float:
        """
        Test the generator

        Args:
            test_loader:
                The test data

        Returns:
            The test MSE result for the generator.

        """
        test_losses = []
        for generator_model in self.models:
            generator_model.to(self.device)
            test_loss = generator_model.run_one_epoch(test_loader, False, None)
            test_losses.append(test_loss)
        return np.mean(test_losses).item()

    def load_generator(self):
        """
        Load the generator from the disk.
        """
        for generator_model in self.models:
            generator_model.load_state_dict(
                torch.load(str(generator_model._get_model_file_name()))
            )
            generator_model.to(self.device)

    @staticmethod
    def get_name() -> str:
        return "feature_generator"
