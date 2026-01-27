from .register import model_dict
import torch
import torch.nn as nn
import math

def get_backbone(backbone, pretrained=True):
    if backbone not in model_dict:
        raise ValueError("Backbone not supported")
    network = model_dict[backbone](pretrained=pretrained)
    return network

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super(Classifier, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)
        num_features = self.backbone.num_features
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, get_fea=False):
        fea = self.backbone(x)
        logits = self.fc(fea)
        if get_fea:
            return logits, fea
        else:
            return logits

