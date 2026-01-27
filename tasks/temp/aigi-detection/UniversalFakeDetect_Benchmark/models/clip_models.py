import math 
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig


class ClipModel(nn.Module):
    def __init__(self, name, opt, num_classes=1):
        super(ClipModel, self).__init__()
        self.model = CLIPModel.from_pretrained(name)
        
        for name, param in self.model.vision_model.named_parameters():
            print('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.model.vision_model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
        
        self.fc = nn.Linear( 1024, num_classes )

    def forward(self, x, return_feature=False):
        features = self.model.vision_model(x)['pooler_output']
        
        if return_feature:
            return features
        return self.fc(features)


        # Strip method-specific SVD adaptation to keep this as a neutral CLIP baseline wrapper