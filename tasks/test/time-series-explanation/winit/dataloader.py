from __future__ import annotations

import abc
import os
import pathlib
import pickle
from typing import List

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset


class WinITDataset(abc.ABC):
    """
    Dataset abstract class that needed to run using our code.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int,
        testbs: int | None,
        deterministic: bool,
        cv_to_use: List[int] | int | None,
        seed: int | None,
    ):
        """
        Constructor

        Args:
            data_path:
                The path of the data.
            batch_size:
                The batch size of the train loader and valid loader.
            testbs:
                The batch size for the test loader.
            deterministic:
                Indicate whether deterministic algorithm is to be used. GLOBAL behaviour!
            cv_to_use:
                Indicate which cv to use. CV are from 0 to 4.
            seed:
                The random seed.

        """
        self.data_path = data_path
        self.train_loaders: List[DataLoader] | None = None
        self.valid_loaders: List[DataLoader] | None = None
        self.test_loader: DataLoader | None = None
        self.feature_size: int | None = None

        self.seed = seed
        self.batch_size = batch_size
        self.testbs = testbs
        self._cv_to_use = cv_to_use

        torch.set_printoptions(precision=8)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

    def _get_loaders(
        self,
        train_data: np.ndarray,
        train_label: np.ndarray,
        test_data: np.ndarray,
        test_label: np.ndarray,
    ):
        """
        Get the train loader, valid loader and the test loaders. The "train_data" and "train_label"
        will be split to be the training set and the validation set.

        Args:
            train_data:
                The train data
            train_label:
                The train label
            test_data:
                The test data
            test_label:
                The test label
        """
        feature_size = train_data.shape[1]
        train_tensor_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
        test_tensor_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
        kf = KFold(n_splits=5)
        train_loaders = []
        valid_loaders = []
        for train_indices, valid_indices in kf.split(train_data):
            train_subset = Subset(train_tensor_dataset, train_indices)
            valid_subset = Subset(train_tensor_dataset, valid_indices)
            train_loaders.append(DataLoader(train_subset, batch_size=self.batch_size))
            valid_loaders.append(DataLoader(valid_subset, batch_size=self.batch_size))
        testbs = self.testbs if self.testbs is not None else len(test_data)
        test_loader = DataLoader(test_tensor_dataset, batch_size=testbs)
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.test_loader = test_loader
        self.feature_size = feature_size

    @abc.abstractmethod
    def load_data(self) -> None:
        """
        Load the data from the file.
        """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the dataset.
        """

    @property
    @abc.abstractmethod
    def data_type(self) -> str:
        """
        Return the type of the dataset. (Not currently used)
        """

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Return the number of classes
        """

    def num_cv(self) -> int:
        """
        Return the total number of CV
        """
        if self.train_loaders is None:
            return 0
        return len(self.train_loaders)

    def cv_to_use(self) -> List[int]:
        """
        Return a list of CV to use.
        """
        if self.train_loaders is None:
            return [0]
        num_cv = self.num_cv()
        if self._cv_to_use is None:
            return list(range(num_cv))
        if isinstance(self._cv_to_use, int) and 0 <= self._cv_to_use < num_cv:
            return [self._cv_to_use]
        if isinstance(self._cv_to_use, list) and all(0 <= c < num_cv for c in self._cv_to_use):
            return self._cv_to_use
        raise ValueError("CV to use range is invalid.")

    def get_train_loader(self, cv: int) -> DataLoader | None:
        """
        Get the train loader to the corresponding CV.
        """
        if self.train_loaders is None:
            return None
        return self.train_loaders[cv]

    def get_valid_loader(self, cv: int) -> DataLoader | None:
        """
        Get the valid loader to the corresponding CV.
        """
        if self.valid_loaders is None:
            return None
        return self.valid_loaders[cv]

    def get_test_loader(self) -> DataLoader | None:
        """
        Return the test loader.
        """
        return self.test_loader


class Mimic(WinITDataset):
    """
    The pre-processed Mimic mortality dataset.
    Num Features = 31, Num Times = 48, Num Classes = 1
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name

    def load_data(self, train_ratio=0.8):
        with (self.data_path / self.file_name).open("rb") as f:
            data = pickle.load(f)
        feature_size = len(data[0][0])

        n_train = int(train_ratio * len(data))

        X = np.array([x for (x, y, z) in data])
        train_data = X[0:n_train]
        test_data = X[n_train:]
        train_label = np.array([y for (x, y, z) in data[0:n_train]])
        test_label = np.array([y for (x, y, z) in data[n_train:]])

        train_data, test_data = self.normalize(train_data, test_data, feature_size)

        self._get_loaders(train_data, train_label, test_data, test_label)

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack([x.T for x in train_data], axis=0)
        num_timesteps = train_data.shape[-1]
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (num_timesteps, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in train_data
            ]
        )
        test_data = np.array(
            [
                np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std)
                for x in test_data
            ]
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "mimic"

    @property
    def data_type(self) -> str:
        return "mimic"

    @property
    def num_classes(self) -> int:
        return 1


class SimulatedData(WinITDataset, abc.ABC):
    """
    An abstract class for simulated data.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int,
        testbs: int | None,
        deterministic: bool,
        file_name_prefix: str,
        ground_truth_prefix: str,
        cv_to_use: List[int] | int | None,
        seed: int | None,
    ):
        """
        Constructor

        Args:
            data_path:
                The path of the data.
            batch_size:
                The batch size of the train loader and valid loader.
            testbs:
                The batch size for the test loader.
            deterministic:
                Indicate whether deterministic algorithm is to be used. GLOBAL behaviour!
            file_name_prefix:
                The file name prefix for the train and the test data. The names of the files will
                be [PREFIX]x_train.pkl, [PREFIX]y_train.pkl, [PREFIX]x_test.pkl and
                [PREFIX]y_test.pkl.
            ground_truth_prefix:
                The ground truth importance file prefix. The file name will be [PREFIX]_test.pkl
            cv_to_use:
                Indicate which cv to use. CV are from 0 to 4.
            seed:
                The random seed.
        """
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name_prefix = file_name_prefix
        self.ground_truth_prefix = ground_truth_prefix

    def load_data(self) -> None:
        with (self.data_path / f"{self.file_name_prefix}x_train.pkl").open("rb") as f:
            train_data = pickle.load(f)
        with (self.data_path / f"{self.file_name_prefix}y_train.pkl").open("rb") as f:
            train_label = pickle.load(f)
        with (self.data_path / f"{self.file_name_prefix}x_test.pkl").open("rb") as f:
            test_data = pickle.load(f)
        with (self.data_path / f"{self.file_name_prefix}y_test.pkl").open("rb") as f:
            test_label = pickle.load(f)

        rng = np.random.default_rng(seed=self.seed)
        perm = rng.permutation(train_data.shape[0])
        train_data = train_data[perm]
        train_label = train_label[perm]

        self._get_loaders(train_data, train_label, test_data, test_label)

    @property
    def num_classes(self) -> int:
        return 1

    def load_ground_truth_importance(self) -> np.ndarray:
        with open(os.path.join(self.data_path, self.ground_truth_prefix + "_test.pkl"), "rb") as f:
            gt = pickle.load(f)
        return gt


class SimulatedState(SimulatedData):
    """
    Simulated State data
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/simulated_state_data"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "state_dataset_",
        ground_truth_prefix: str = "state_dataset_importance",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        return "simulated_state"

    @property
    def data_type(self) -> str:
        return "state"


class SimulatedSwitch(SimulatedData):
    """
    Simulated Switch data
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/simulated_switch_data"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "state_dataset_",
        ground_truth_prefix: str = "state_dataset_importance",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        return "simulated_switch"

    @property
    def data_type(self) -> str:
        return "switch"


class SimulatedSpike(SimulatedData):
    """
    Simulated Spike data, with possible delay involved.
    """

    def __init__(
        self,
        data_path: pathlib.Path = None,
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "",
        ground_truth_prefix: str = "gt",
        delay: int = 0,
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        if data_path is None:
            if delay > 0:
                data_path = pathlib.Path(f"./data/simulated_spike_data_delay_{delay}")
            elif delay == 0:
                data_path = pathlib.Path("./data/simulated_spike_data")
            else:
                raise ValueError("delay must be non-negative.")
        self.delay = delay

        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        if self.delay == 0:
            return "simulated_spike"
        return f"simulated_spike_delay_{self.delay}"

    @property
    def data_type(self) -> str:
        return "spike"
