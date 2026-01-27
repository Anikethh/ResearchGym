"""Load models from the VISSL package.
Code adapted from 
https://github.com/facebookresearch/vissl/blob/main/extra_scripts/convert_vissl_to_torchvision.py
"""

import torch
import torchvision
from torch.hub import load_state_dict_from_url
from .register import register_model
import timm

def _replace_fc(model, output_dim):
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, output_dim)
    return model


SIMCLR_RN50_URL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch"
BARLOWTWINS_RN50_URL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch"


def replace_module_prefix(state_dict, prefix, replace_with=""):
    state_dict = {
        (key.replace(prefix, replace_with, 1)
         if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def get_torchvision_state_dict(url):
    model = load_state_dict_from_url(url)
    model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]

    return replace_module_prefix(model_trunk, "_feature_blocks.")


def imagenet_resnet50_simclr(output_dim):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    model.load_state_dict(get_torchvision_state_dict(SIMCLR_RN50_URL))
    model.fc.in_features = 2048
    return _replace_fc(model, output_dim)


def imagenet_resnet50_barlowtwins(output_dim):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    import vissl
    model.load_state_dict(get_torchvision_state_dict(BARLOWTWINS_RN50_URL))
    model.fc.in_features = 2048
    return _replace_fc(model, output_dim)

@register_model("densenet121_in1k")
def imagenet_densenet121_pretrained(pretrained=True):
    return timm.create_model('densenet121.tv_in1k', pretrained=pretrained, num_classes=0)

@register_model("resnet_sup_in21k")
def resnet_sup_in21k(pretrained=True):
    return timm.create_model('tresnet_m.miil_in21k', pretrained=pretrained, num_classes=0) # https://github.com/Alibaba-MIIL/ImageNet21K

@register_model("vit_sup_in1k")
def vit_sup_in1k(pretrained=True):
    return timm.create_model('vit_base_patch32_224', pretrained=pretrained, num_classes=0)  # https://arxiv.org/abs/2106.10270

@register_model("vit_sup_in21k")
def vit_sup_in1k(pretrained=True):
    return timm.create_model('vit_base_patch32_224.augreg_in21k', pretrained=pretrained, num_classes=0)

@register_model("vit_clip_oai")
def vit_sup_in1k(pretrained=True):
    return timm.create_model('vit_base_patch32_clip_224.openai', pretrained=pretrained, num_classes=0) 

@register_model("vit_clip_laion")
def vit_sup_in1k(pretrained=True):
    return timm.create_model('vit_base_patch32_clip_224.laion2b', pretrained=pretrained, num_classes=0) 

@register_model("vit_dino_in1k")
def vit_sup_in1k(pretrained=True):
    return timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained, num_classes=0) # https://github.com/facebookresearch/dino
