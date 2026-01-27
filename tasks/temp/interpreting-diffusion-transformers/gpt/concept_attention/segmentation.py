"""
Neutral segmentation interfaces for baseline evaluations.
"""
from abc import ABC
from typing import List, Tuple

import torch
import torchvision.transforms.functional as F


class SegmentationAbstractClass(ABC):
    """
    Base interface: subclasses should implement `segment_individual_image` and
    return (coefficients, optional_reconstruction).
    Coefficients must be a tensor shaped [num_concepts, H, W].
    """

    def segment_individual_image(self, image, concepts: List[str], caption: str, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        images,
        target_concepts: List[str] | None,
        concepts: List[str],
        captions: List[str],
        mean_value_threshold: bool = True,
        apply_blur: bool = False,
        **kwargs,
    ) -> Tuple[list, list, list]:
        if not isinstance(images, list):
            images = [images]

        all_masks, all_coefficients, reconstructed_images = [], [], []

        for index, image in enumerate(images):
            coefficients, reconstructed_image = self.segment_individual_image(
                image, concepts, captions[index], **kwargs
            )
            if apply_blur:
                coefficients = F.gaussian_blur(coefficients.unsqueeze(0), kernel_size=3, sigma=1.0).squeeze()

            if target_concepts is None:
                mean_values = torch.mean(coefficients, dim=(1, 2), keepdim=True)
                masks = coefficients > mean_values
                all_masks.append(masks)
                all_coefficients.append(coefficients)
                reconstructed_images.append(reconstructed_image)
            else:
                target_index = concepts.index(target_concepts[index])
                threshold = coefficients[target_index].mean() if mean_value_threshold else 0.0
                mask = (coefficients[target_index] > threshold).cpu().numpy()
                all_masks.append(mask)
                all_coefficients.append(coefficients[target_index].detach().cpu().numpy())
                reconstructed_images.append(reconstructed_image)

        return all_masks, all_coefficients, reconstructed_images
