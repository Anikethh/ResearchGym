import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1))