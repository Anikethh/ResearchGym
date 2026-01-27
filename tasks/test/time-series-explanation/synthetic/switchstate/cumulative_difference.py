import torch
import numpy as np
from scipy.integrate import trapezoid

from captum.log import log_usage
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
    _validate_input,
)
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable, Tuple, Union, cast

from tint.utils import add_noise_to_inputs, _expand_baselines


@torch.no_grad()
def cumulative_difference(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    testbs: int=32,
    stdevs: Union[float, Tuple[float, ...]] = 0.0,
    draw_baseline_from_distrib: bool = False,
    topk: float = 0.2,
    top: int = 0,
    largest: bool = True,
    weight_fn: Callable[
        [Tuple[Tensor, ...], Tuple[Tensor, ...]], Tensor
    ] = None,
    classification: bool = True,
    **kwargs,
) -> float:
    # Format data
    return_all = additional_forward_args[2]
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    additional_forward_args = _format_additional_forward_args(
        additional_forward_args
    )
    attributions = _format_tensor_into_tuples(attributions)  # type: ignore
    if baselines is not None:
        baselines = _format_baseline(
            baselines, cast(Tuple[Tensor, ...], inputs)
        )
        _validate_input(
            inputs,
            baselines,
            draw_baseline_from_distrib=draw_baseline_from_distrib,
        )
        
    num_steps = int(topk * inputs[0][0].numel()) if top == 0 else top
    
    assert 0 < topk < 1 or top > 0, "topk must be a float between 0 and 1 or specify top > 0"
    if not largest:
        topk = 1.0 - topk

    # Initialize cumulative prediction differences (one value per step)
    cumulative_differences = [0.0 for _ in range(num_steps)]

    # Total number of samples in the dataset
    total_samples = inputs[0].shape[0]

    # Loop over batches
    for start_idx in range(0, total_samples, testbs):
        end_idx = min(start_idx + testbs, total_samples)

        batch_inputs = tuple(inp[start_idx:end_idx] for inp in inputs)
        batch_attributions = tuple(attr[start_idx:end_idx] for attr in attributions)

        # Get top-k indices for each batch
        topk_indices = tuple(
            torch.topk(
                attr.reshape(len(attr), -1),
                num_steps,
                sorted=True,
                largest=largest,
            ).indices.to(attr.device)
            for attr in batch_attributions
        )

        topk_indices = tuple(
            topk.to(input.device) for topk, input in zip(topk_indices, batch_inputs)
        )

        # First step: Compare original inputs with inputs with only top-1 value removed
        if baselines is None:
            raise RuntimeError
        else:
            batch_baselines = tuple(
                baseline[start_idx:end_idx] if not isinstance(baseline, (int, float)) else baseline
                for baseline in baselines
            )

            inputs_pert_first = tuple(
                inp.reshape(len(inp), -1)
                .scatter(
                    -1,
                    topk_idx[:, :1],
                    baseline if isinstance(baseline, (int, float))
                    else baseline.reshape(len(baseline), -1).gather(-1, topk_idx[:, :1]),
                )
                .reshape(inp.shape)
                for inp, baseline, topk_idx in zip(batch_inputs, batch_baselines, topk_indices)
            )

        # Compute predictions for the first step
        logits_orig = _run_forward(
            forward_func=forward_func,
            inputs=batch_inputs,
            target=None,
            additional_forward_args=(None, None, return_all),
        )
        logits_first = _run_forward(
            forward_func=forward_func,
            inputs=inputs_pert_first,
            target=None,
            additional_forward_args=(None, None, return_all),
        )

        prob_orig = logits_orig.softmax(-1)
        prob_first = logits_first.softmax(-1)
        # print(logits_orig.shape)
        # raise RuntimeError

        # Compute and store the first step difference
        step_diff_first = torch.abs(prob_orig - prob_first).mean(dim=2).mean(dim=1).sum().item()
        cumulative_differences[0] += step_diff_first
        
        prob_before = prob_first

        for step in range(1, num_steps):
            inputs_pert_step = tuple(
                inp.reshape(len(inp), -1)
                .scatter(
                    -1,
                    topk_idx[:, : step + 1],
                    baseline if isinstance(baseline, (int, float))
                    else baseline.reshape(len(baseline), -1).gather(-1, topk_idx[:, : step + 1]),
                )
                .reshape(inp.shape)
                for inp, baseline, topk_idx in zip(batch_inputs, batch_baselines, topk_indices)
            )

            logits_step = _run_forward(
                forward_func=forward_func,
                inputs=inputs_pert_step,
                target=None,
                additional_forward_args=(None, None, return_all),
            )

            prob_step = logits_step.softmax(-1)

            # Compute cumulative difference for the batch up to this step
            step_diff = torch.abs(prob_before - prob_step).mean(dim=2).mean(dim=1).sum().item()
            cumulative_differences[step] += step_diff
            
            prob_before = prob_step

    # Compute mean cumulative prediction differences for each step
    mean_cumulative_differences = [
        cumulative_differences[step] / total_samples for step in range(num_steps)
    ]

    # Normalize x and y for AUCC calculation
    normalized_x = np.linspace(0, 1, len(mean_cumulative_differences))
    # normalized_y = np.cumsum(mean_cumulative_differences) / sum(mean_cumulative_differences) if sum(mean_cumulative_differences) > 0 else np.zeros(len(mean_cumulative_differences))
    normalized_y = np.cumsum(mean_cumulative_differences)
    
    # Compute AUCC using the trapezoidal rule
    aucc = trapezoid(normalized_y, normalized_x)
    
    # print(mean_cumulative_differences)
    # print(np.cumsum(mean_cumulative_differences))

    return sum(mean_cumulative_differences), aucc, sum(mean_cumulative_differences[:50]), mean_cumulative_differences