from __future__ import annotations

import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from winit.dataloader import WinITDataset
from winit.utils import aggregate_scores


class BoxPlotter:
    """
    A class for plotting various box plots as plotExampleBox in FIT repo.
    """

    def __init__(
        self,
        dataset: WinITDataset,
        base_plot_path: pathlib.Path,
        num_to_plot: int,
        explainer_name: str,
    ):
        """
        Constructor.

        Args:
            dataset:
                The dataset.
            base_plot_path:
                The base plot path for the plots
            num_to_plot:
                The number of samples to plot.
            explainer_name:
                The name of the explainer.
        """
        self.plot_path = base_plot_path / dataset.get_name()
        self.num_to_plot = num_to_plot
        self.explainer_name = explainer_name
        testset = list(dataset.test_loader.dataset)
        self.x_test = torch.stack(([x[0] for x_ind, x in enumerate(testset)])).cpu().numpy()
        self.y_test = torch.stack(([x[1] for x_ind, x in enumerate(testset)])).cpu().numpy()
        self.plot_path.mkdir(parents=True, exist_ok=True)

    def plot_importances(self, importances: Dict[int, np.ndarray], aggregate_method: str) -> None:
        """
        Plot the importance for all cv and save it to files.

        Args:
            importances:
                A dictionary from CV to feature importances.
            aggregate_method:
                The aggregation method for WinIT.
        """
        for cv, importance_unaggregated in importances.items():
            importance_scores = aggregate_scores(importance_unaggregated, aggregate_method)
            for i in range(self.num_to_plot):
                prefix = f"{self.explainer_name}_{aggregate_method}_cv_{cv}_attributions"
                self._plot_example_box(
                    np.abs(importance_scores[i]), self._get_plot_file_name(i, prefix)
                )

    def plot_ground_truth_importances(self, ground_truth_importance: np.ndarray):
        """
        Plot the ground truth importances for all cv and save it to files.

        Args:
            ground_truth_importance:
                The ground truth importance.
        """
        prefix = "ground_truth_attributions"
        for i in range(self.num_to_plot):
            self._plot_example_box(ground_truth_importance[i], self._get_plot_file_name(i, prefix))

    def plot_labels(self):
        """
        Plot the labels and save it to files. If the label is one-dimensional, skip the plotting.
        """
        if self.y_test.ndim != 2:
            return
        for i in range(self.num_to_plot):
            self._plot_example_box(
                self.y_test[i : i + 1], self._get_plot_file_name(i, prefix="labels")
            )

    def plot_x_pred(
        self,
        x: np.ndarray | Dict[int, np.ndarray] | None,
        preds: Dict[int, np.ndarray],
        prefix: str = None,
    ):
        """
        Plot the data and the corresponding predictions and save it to files.

        Args:
            x:
                The data. Can be a numpy array of a dictionary of CV to numpy arrays.
            preds:
                The predictions. A dictionary of CV to numpy arrays. (In case of only 1 data,
                the predictions are the predictions of the model of the corresponding CV
                on the same data.
            prefix:
                The prefix of the name of the files to be saved.
        """
        if x is None:
            x = self.x_test

        if isinstance(x, np.ndarray):
            for i in range(self.num_to_plot):
                filename_prefix = prefix if prefix is not None else "data"
                self._plot_example_box(x[i], self._get_plot_file_name(i, prefix=filename_prefix))
        elif isinstance(x, dict):
            for cv, xin in x.items():
                for i in range(self.num_to_plot):
                    filename_prefix = f"data_cv_{cv}" if prefix is None else f"{prefix}_cv_{cv}"
                    self._plot_example_box(
                        xin[i], self._get_plot_file_name(i, prefix=filename_prefix)
                    )

        for cv, pred in preds.items():
            for i in range(self.num_to_plot):
                filename_prefix = f"preds_cv_{cv}" if prefix is None else f"{prefix}_cv_{cv}"
                self._plot_example_box(
                    pred[i].reshape(1, -1), self._get_plot_file_name(i, filename_prefix)
                )

    def _get_plot_file_name(self, i: int, prefix: str) -> pathlib.Path:
        return self.plot_path / f"{prefix}_{i}.png"

    @staticmethod
    def _plot_example_box(input_array, save_location: pathlib.Path):
        fig, ax = plt.subplots()
        plt.axis("off")

        ax.imshow(input_array, interpolation="nearest", cmap="gray")

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(str(save_location), bbox_inches="tight", pad_inches=0)

        plt.close()
