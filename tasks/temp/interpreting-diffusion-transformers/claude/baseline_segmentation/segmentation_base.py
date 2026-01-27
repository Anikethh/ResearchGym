"""
Abstract base class for segmentation models.
"""
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any

class SegmentationAbstractClass(ABC):
    """
    Abstract base class for segmentation models used in interpretability evaluation.
    """

    @abstractmethod
    def segment_individual_image(self, image, concepts: List[str], caption: str, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Segment an individual image based on provided concepts.
        
        Args:
            image: Input image (PIL Image or torch.Tensor)
            concepts: List of concept strings to segment
            caption: Text caption for the image
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (segmentation_scores, additional_info)
            - segmentation_scores: Tensor of shape (num_concepts, H, W) with scores for each concept
            - additional_info: Optional additional information
        """
        pass
