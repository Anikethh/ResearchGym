from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from winit.dataloader import WinITDataset
from winit.models import StateClassifier, ConvClassifier, TorchModel
from winit.utils import resolve_device


@dataclasses.dataclass(frozen=True)
class EpochResult:
    epoch_loss: float
    accuracy: float
    auc: float
    precision: float
    recall: float

    def __str__(self):
        return f"Loss: {self.epoch_loss}, Acc: {100 * self.accuracy:.2f}%, Auc: {self.auc:.4f}"


class ModelTrainer:
    """
    A class for training one model.
    """

    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        batch_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        model_file_name: pathlib.Path,
        model_type: str = "GRU",
        device: str | torch.device | None = None,
        verbose_eval: int = 10,
        early_stopping: bool = False,
    ):
        """
        Constructor

        Args:
            feature_size:
               The number of features.
            num_classes:
               The number of classes (output nodes, i.e. num_states)
            batch_size:
               The batch size for training the model
            hidden_size:
               The hidden size for the model
            dropout:
               The dropout rate of the model in case of GRU or LSTM
            num_layers:
               The number of layers of the models in case of GRU or LSTM.
            model_file_name:
               The model file name (pathlib.Path)
            model_type:
               The model type. Can be "GRU", "LSTM" or "CONV"
            device:
               The torch device.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
               Whether apply early stopping or not.
        """
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.model_type = model_type
        self.model: TorchModel | None = None

        if model_type in ["GRU", "LSTM"] or model_type is None:
            self.model = StateClassifier(
                feature_size=self.feature_size,
                num_states=self.num_classes,
                hidden_size=hidden_size,
                device=self.device,
                rnn=model_type,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "CONV":
            self.model = ConvClassifier(
                feature_size=self.feature_size,
                num_states=self.num_classes,
                hidden_size=hidden_size,
                kernel_size=10,
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid model type ({model_type}). Must be ('GRU', 'LSTM', 'CONV')")

        self.verbose_eval = verbose_eval
        self.early_stopping = early_stopping
        self.model_file_name = model_file_name
        self.model_file_name.parent.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(ModelTrainer.__name__)

    def train_model(
        self,
        train_loader,
        valid_loader,
        test_loader,
        num_epochs,
        lr=0.001,
        weight_decay=0.001,
        use_all_times: bool = True,
    ) -> None:
        """
        Train the model on train loader, perform validation on valid loader. Evaluate
        the model on test loader after the model has been trained.

        Args:
            train_loader:
               DataLoader for training
            valid_loader:
               DataLoader for validation
            test_loader:
               DataLoader for testing
            num_epochs:
               Number of epochs of training
            lr:
               Learning rate
            weight_decay:
               Weight decay
            use_all_times:
               Use all the timestep for training. Otherwise, only use the last timestep.

        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_results_trend = []
        valid_results_trend = []

        best_auc = 0
        self.model.to(self.device)
        for epoch in range(num_epochs):
            train_results = self._run_one_epoch(
                train_loader, run_train=True, optimizer=optimizer, use_all_times=use_all_times
            )
            valid_results = self._run_one_epoch(
                valid_loader, run_train=False, optimizer=None, use_all_times=use_all_times
            )
            train_results_trend.append(train_results)
            valid_results_trend.append(valid_results)

            if epoch % self.verbose_eval == 0:
                self.log.info(f"Epoch {epoch + 1}")
                self.log.info(f"Training   ===> {train_results}")
                self.log.info(f"Validation ===> {valid_results}")

            if self.early_stopping:
                if valid_results.auc > best_auc:
                    best_auc = valid_results.auc
                    torch.save(self.model.state_dict(), str(self.model_file_name))

        if self.early_stopping:
            # retrieve best model
            self.load_model()
        else:
            # save the model at the end only
            torch.save(self.model.state_dict(), str(self.model_file_name))

        test_results = self._run_one_epoch(
            test_loader, run_train=False, optimizer=None, use_all_times=use_all_times
        )

        self.log.info(f"Test ===> {test_results}")

    def load_model(self) -> None:
        """
        Load the model for the file.
        """
        self.model.load_state_dict(
            torch.load(str(self.model_file_name), map_location=torch.device(self.device))
        )

    def get_test_results(self, test_loader, use_all_times: bool) -> EpochResult:
        """
        Return the test results on the test set.

        Args:
            test_loader:
                The test loader
            use_all_times:
                Whether to output the test results only on the last timesteps.

        Returns:
            An EpochResult object containing all the test results.
        """
        test_results = self._run_one_epoch(
            test_loader, run_train=False, optimizer=None, use_all_times=use_all_times
        )
        return test_results

    def run_inference(
        self, data: DataLoader | torch.Tensor, with_activation=True, return_all=True
    ) -> np.ndarray:
        """
        Run inference.

        Args:
            data:
                The data to be run. Shape of the batch = (num_samples, num_features, num_times)
            with_activation:
                Whether activation should be used.
            return_all:
                Whether return all the timesteps or the last one.

        Returns:
            A numpy array of shape (num_samples, num_classes, num_times) if return_all is True.
            Otherwise, returns a numpy array of shape (num_samples, num_classes)

        """
        self.model.eval()
        self.model.to(self.device)
        outputs = []

        for x in data:
            x = x[0]
            x = x.to(self.device)
            output = self.model(x, return_all=return_all)
            if with_activation:
                output = self.model.activation(output)
            output = output.detach().cpu().numpy()
            outputs.append(output)
        return np.concatenate(outputs, axis=0)

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        run_train: bool = True,
        optimizer: Optimizer = None,
        use_all_times: bool = True,
    ) -> EpochResult:
        """
        Run one epoch of forward pass. Run backward and update if run_train is true.
        Return the average epoch loss, and other metrics after the epoch.

        Args:
            dataloader:
                DataLoader for the data.
            run_train:
                Indicates whether training mode is on. If True, optimizer must not be None.
            optimizer:
                The optimizer for running train. Note that this will be ignored if run_train
                is False.
            use_all_times:
                Indicates whether using all timesteps for training or testing.

        Returns:
            An EpochResult object containing the epoch loss and other metrics.

        """
        multiclass = self.num_classes > 1
        self.model = self.model.to(self.device)
        if run_train:
            self.model.train()
            if optimizer is None:
                raise ValueError("optimizer is none in train mode.")
        else:
            self.model.eval()
        epoch_loss = 0
        if multiclass:
            loss_criterion = torch.nn.CrossEntropyLoss()
        else:
            loss_criterion = torch.nn.BCEWithLogitsLoss()
        all_labels, all_probs = [], []
        for signals, labels in dataloader:
            if run_train:
                optimizer.zero_grad()
            signals = torch.Tensor(signals.float()).to(self.device)
            labels = torch.Tensor(labels.long() if multiclass else labels.float()).to(self.device)
            output = self.model(signals, return_all=use_all_times)
            prob = self.model.activation(output)

            if use_all_times and not multiclass:
                output = output[:, 0, :]
                prob = prob[:, 0, :]
            if not use_all_times:
                if labels.ndim > 1:
                    labels = labels[:, -1:]
                else:
                    labels = labels[:, None]

            loss = loss_criterion(output, labels)
            epoch_loss += loss.item()
            if run_train:
                loss.backward()
                optimizer.step()

            all_probs.append(prob.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        # compile results
        all_labels = np.concatenate(all_labels).astype(int)
        all_probs = np.concatenate(all_probs)
        epoch_loss = epoch_loss / len(dataloader)

        auc = (
            0
            if len(np.unique(all_labels)) < 2 or multiclass
            else roc_auc_score(all_labels.reshape(-1), all_probs.reshape(-1))
        )
        all_preds = (all_probs > 0.5).astype(int)
        recall = (
            0
            if multiclass
            else recall_score(all_labels.reshape(-1), all_preds.reshape(-1), zero_division=0)
        )
        precision = (
            0
            if multiclass
            else precision_score(all_labels.reshape(-1), all_preds.reshape(-1), zero_division=0)
        )
        accuracy = 0 if multiclass else float(np.mean(all_labels == all_preds))
        return EpochResult(
            epoch_loss=epoch_loss,
            accuracy=accuracy,
            auc=auc,
            precision=precision,
            recall=recall,
        )


class ModelTrainerWithCv:
    """
    A class for training a separate model for each CV for a dataset.
    """

    def __init__(
        self,
        dataset: WinITDataset,
        ckpt_path: pathlib.Path,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        model_type: str = "GRU",
        device: str | torch.device | None = None,
        verbose_eval: int = 10,
        early_stopping: bool = False,
    ):
        """
        Constructor

        Args:
            dataset:
                The specified dataset.
            hidden_size:
                The size of the hidden units of the models.
            dropout:
                The dropout rate of the models.
            num_layers:
                The number of layers of the models if model type is RNN or LSTM.
            model_type:
                The model type of the models. It can be "RNN", "LSTM" or "CONV".
            device:
                The torch device.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
               Whether apply early stopping or not.
            ckpt_path:
               The ckeckpoint path.
        """
        self.dataset = dataset
        self.model_path = ckpt_path / dataset.get_name()
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_args = {
            "batch_size": dataset.batch_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_layers": num_layers,
            "model_type": model_type,
        }
        self.model_trainers: Dict[int, ModelTrainer] = {}
        self.log = logging.getLogger(ModelTrainerWithCv.__name__)

        for cv in self.dataset.cv_to_use():
            self.model_trainers[cv] = ModelTrainer(
                dataset.feature_size,
                dataset.num_classes,
                dataset.batch_size,
                hidden_size,
                dropout,
                num_layers,
                self._model_file_name(cv),
                model_type,
                device,
                verbose_eval,
                early_stopping,
            )

    def train_models(
        self, num_epochs, lr=0.001, weight_decay=0.001, use_all_times: bool = True
    ) -> None:
        """
        Train the models on the dataset for each CV. Evaluate the model on the test set afterwards.

        Args:
            num_epochs:
               Number of epochs of training
            lr:
               Learning rate
            weight_decay:
               Weight decay
            use_all_times:
               Use all the timestep for training. Otherwise, only use the last timestep.
        """
        for cv, model_trainer in self.model_trainers.items():
            self.log.info(f"Training model for cv={cv}")
            model_trainer.train_model(
                self.dataset.train_loaders[cv],
                self.dataset.valid_loaders[cv],
                self.dataset.test_loader,
                num_epochs,
                lr,
                weight_decay,
                use_all_times,
            )

    def load_model(self) -> None:
        """
        Load all the models from the disk.
        """
        for cv, model_trainer in self.model_trainers.items():
            model_trainer.load_model()

    def get_test_results(self, use_all_times: bool) -> Dict[int, EpochResult]:
        """
        Return the results of each model on the test sets.

        Args:
            use_all_times:
                Indicates whether we use all timesteps for test results.
        Returns:
            A dictionary from CV to EpochResult indicating the test results for each CV.
        """
        accuracies = {}
        for cv, model_trainer in self.model_trainers.items():
            accuracies[cv] = model_trainer.get_test_results(
                self.dataset.test_loader, use_all_times=use_all_times
            )
        return accuracies

    def run_inference(
        self,
        data: torch.Tensor | Dict[int, torch.Tensor] | None,
        with_activation=True,
        return_all=True,
    ) -> Dict[int, np.ndarray]:
        """
        Run inference.

        Args:
            data:
                The data to be run. Shape of the batch = (num_samples, num_features, num_times).
                If it is a Tensor, it will be run inference on the data for all CVs. If it is a
                dictionary of CV to Tensor, it will run inference on the data with the model
                corresponding to the CV. If it is None, the inference will be run on the test set
                of the dataset.
            with_activation:
                Whether activation should be used.
            return_all:
                Whether return all the timesteps or the last one.

        Returns:
            A dictionary of CV to numpy arrays of shape (num_samples, num_classes, num_times)
            if return_all is True. Otherwise, the numpy arrays will be of shape
            (num_samples, num_classes).

        """
        if isinstance(data, dict):
            return_dict = {}
            for cv, model_trainer in self.model_trainers.items():
                data_cv = data[cv]
                if isinstance(data_cv, torch.Tensor):
                    data_cv = DataLoader(TensorDataset(data_cv), batch_size=self.dataset.testbs)
                return_dict[cv] = model_trainer.run_inference(data_cv, with_activation, return_all)
            return return_dict

        if data is None:
            data = self.dataset.test_loader
        elif isinstance(data, torch.Tensor):
            data = DataLoader(TensorDataset(data), batch_size=self.dataset.testbs)

        return {
            cv: model_trainer.run_inference(data, with_activation, return_all)
            for cv, model_trainer in self.model_trainers.items()
        }

    def _model_file_name(self, cv) -> pathlib.Path:
        return self.model_path / f"{self._model_name()}_{cv}.pt"

    def _model_name(self) -> str:
        shortened_args = {
            "bs": self.model_args["batch_size"],
            "hid": self.model_args["hidden_size"],
            "drop": self.model_args["dropout"],
        }

        num_layers = self.model_args.get("num_layers")
        if num_layers is not None and num_layers != 1:
            shortened_args["lay"] = num_layers
        rnn_type = self.model_args.get("rnn_type")
        str_list = ["model"]
        if rnn_type is not None and rnn_type != "gru":
            str_list.append(rnn_type)
        str_list.extend([f"{key}_{value}" for key, value in shortened_args.items()])
        return "_".join(str_list)
