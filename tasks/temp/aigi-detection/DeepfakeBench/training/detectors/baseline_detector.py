import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import BACKBONE, DETECTOR

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='baseline')
class BaselineDetector(nn.Module):
    """
    Neutral baseline detector using a standard CNN backbone (e.g., Xception).
    - Expects input dict with key 'image' (B x C x H x W) and 'label' (B).
    - Produces prediction dict with 'cls' logits, 'prob' probabilities, and 'feat' features.
    """

    def __init__(self, config=None):
        super(BaselineDetector, self).__init__()
        self.config = config or {}

        # Select backbone from config (defaults to xception)
        backbone_name = (self.config.get('backbone_name') or 'xception').lower()
        backbone_cfg = self.config.get('backbone_config') or {
            'mode': 'original',
            'num_classes': 2,
            'inc': 3,
            'dropout': False,
        }

        if backbone_name not in BACKBONE.data:
            raise ValueError(f"Backbone '{backbone_name}' not registered. Available: {list(BACKBONE.data.keys())}")

        # Build backbone from registry
        self.backbone: nn.Module = BACKBONE[backbone_name](backbone_cfg)
        self.loss_func = nn.CrossEntropyLoss()

    def features(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Use backbone's feature extractor
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        # Use backbone's classifier head to produce logits for 2 classes
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)

        # Optional per-class losses for analysis
        mask_real = label == 0
        mask_fake = label == 1
        loss_real = self.loss_func(pred[mask_real], label[mask_real]) if mask_real.any() else torch.zeros((), device=pred.device)
        loss_fake = self.loss_func(pred[mask_fake], label[mask_fake]) if mask_fake.any() else torch.zeros((), device=pred.device)

        return {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }

    def get_train_metrics(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Basic train metrics are computed upstream via metrics.calculate_metrics_for_train
        # This function exists to match the interface.
        from metrics.base_metrics_class import calculate_metrics_for_train
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: Dict[str, torch.Tensor], inference: bool = False) -> Dict[str, torch.Tensor]:
        feats = self.features(data_dict)
        logits = self.classifier(feats)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {'cls': logits, 'prob': prob, 'feat': feats}


