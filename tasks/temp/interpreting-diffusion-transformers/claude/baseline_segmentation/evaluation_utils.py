"""
Evaluation utilities for segmentation tasks.
"""
import torch
import numpy as np
from sklearn.metrics import average_precision_score

def linear_normalization(x, dim):
    """Linearly normalize tensor along specified dimension."""
    # Subtract the minimum to shift all values to non-negative range
    x_min = torch.min(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_min
    # Sum the values along the specified dimension
    x_sum = torch.sum(x_shifted, dim=dim, keepdim=True)
    # Avoid division by zero by setting sums of zero to one
    x_sum = torch.where(x_sum == 0, torch.ones_like(x_sum), x_sum)
    # Normalize by dividing by the sum
    return x_shifted / x_sum

################################## Metrics ##################################

def get_ap_scores(predict, target, ignore_index=-1):
    """Calculate average precision scores for segmentation."""
    total = []
    for pred, tgt in zip(predict, target):
        pred = pred.reshape(-1)
        tgt = tgt.reshape(-1)
        if ignore_index >= 0:
            ignore_indices = tgt == ignore_index
            pred = pred[~ignore_indices]
            tgt = tgt[~ignore_indices]
        if len(np.unique(tgt)) == 1:
            continue
        total.append(average_precision_score(tgt, pred))
    return total

def batch_pix_accuracy(predict, target, labeled):
    """Calculate pixel accuracy for a batch."""
    labeled = labeled.bool()
    assert predict.shape == target.shape, "predict and target shapes must match"
    
    predict = predict.bool()
    target = target.bool()
    
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct pixels cannot exceed labeled pixels"
    return pixel_correct, pixel_labeled

def batch_intersection_union(predict, target, num_class, labeled):
    """Calculate intersection and union for a batch."""
    labeled = labeled.bool()
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection cannot exceed union"
    return area_inter, area_union
