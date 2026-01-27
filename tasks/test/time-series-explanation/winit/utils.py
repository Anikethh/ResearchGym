from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd
import torch


def resolve_device(device: torch.device | str | None) -> torch.device:
    """
    Resolve the torch device.
    """
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        return torch.device(device)


def aggregate_scores(scores: np.ndarray, aggregate_method: str) -> np.ndarray:
    """
    Run aggregation of the WinIT importance scores. If the importance scores is rank 3, return
    the scores.

    Args:
        scores:
            The input importance scores. Shape = (num_samples, num_features, num_times, window_size)
            or (num_samples, num_features, num_times).
        aggregate_method:
            The aggregation method of WinIT

    Returns:
        The aggregate scores as numpy array.
    """
    if scores.ndim == 3:
        return scores

    num_samples, num_features, num_times, window_size = scores.shape
    # where scores[i, :, k] is the importance score window with shape (num_features, window_size)
    # for the prediction at time (k). So scores[i, j, k, l] is the importance of observation
    # (i, j, k - window_size + l + 1) to the prediction at time (k)
    aggregated_scores = np.zeros((num_samples, num_features, num_times))
    for t in range(num_times):
        # windows where obs is included
        relevant_windows = np.arange(t, min(t + window_size, num_times))
        # relative position of obs within window
        relevant_obs = -relevant_windows + t - 1
        relevant_scores = scores[:, :, relevant_windows, relevant_obs]
        relevant_scores = np.nan_to_num(relevant_scores)
        if aggregate_method == "absmax":
            score_max = relevant_scores.max(axis=-1)
            score_min = relevant_scores.min(axis=-1)
            aggregated_scores[:, :, t] = np.where(-score_min > score_max, score_min, score_max)
        elif aggregate_method == "max":
            aggregated_scores[:, :, t] = relevant_scores.max(axis=-1)
        elif aggregate_method == "mean":
            aggregated_scores[:, :, t] = relevant_scores.mean(axis=-1)
        else:
            raise NotImplementedError(f"Aggregation method {aggregate_method} unrecognized")

    return aggregated_scores


def append_df_to_csv(df: pd.DataFrame, csv_path: pathlib.Path) -> int:
    """
    Write intermediate results to CSV. If there is no existing file, create a new one. If there
    is, and if the old csv file have the same columns as the current one, append it. If the
    columns are different or there are some IO Error, write to a different file.

    Args:
        df:
            The dataframe we wish to write.
        csv_path:
            The path and the filename of the file.

    Returns:
        An Error code.
        0 - Successful appending
        1 - Columns do not match
        2 - File exists but read failed or some other reasons
        3 - File does not exist or is not a file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.is_file():
        try:
            old_df = pd.read_csv(csv_path)
            if (old_df.columns == df.columns).all():
                old_df.append(df).to_csv(str(csv_path), index=False)
                return 0
            else:
                error_code = 1
        except Exception:
            error_code = 2
    else:
        error_code = 3
    log = logging.getLogger("Utils")
    if error_code == 3:
        log.info(f"Creating {csv_path}")
        df.to_csv(csv_path, index=False)
    else:
        i = 1
        while True:
            old_name = csv_path.stem
            new_path = csv_path.with_name(f"{old_name} ({i}){csv_path.suffix}")
            if not new_path.exists():
                break
            i += 1
        log.info(f"Writing to {new_path} instead of appending")
        df.to_csv(new_path, index=False)
    return error_code
