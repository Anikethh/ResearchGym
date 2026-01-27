from .register import register_transform
from utils import log
import torchvision.transforms as transforms
import torch

@register_transform(['waterbirds', 'celeba', 'chexpert'])
def get_transform(target_resolution, train, augment_data, scale = 256.0 / 224.0):
    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose(
            [
                transforms.Resize(int(target_resolution * scale)),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
    return transform



IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# https://github.com/aliasgharkhani/Masktune/blob/master/src/methods/in9l_train.py
@register_transform('imagenet-bg')
def get_imagenet_bg_transforms(target_resolution, train, augment_data, scale = 256.0 / 224.0):
    if (not train) or (not augment_data):
        # Special transforms for ImageNet(s)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'],
                    IMAGENET_PCA['eigvec'])
        ])
    else:
        transform = transforms.Compose([
                transforms.Resize(int(target_resolution * scale)),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.4717, 0.4499, 0.3837], [
                    0.2600, 0.2516, 0.2575]),
            ])
    return transform




@register_transform(["imagenet-9", "imagenet-a"])
def get_imagenet_transform(resolution, train, augment_data=True, scale=256.0 / 224.0):
    if train and augment_data:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(int(resolution*scale)),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return transform

@register_transform('nico')
def get_transform_nico(resolution, train, augment_data=True, scale=256.0 / 224.0):
    mean = [0.52418953, 0.5233741, 0.44896784]
    std = [0.21851876, 0.2175944, 0.22552039]
    if train and augment_data:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomCrop(resolution, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([resolution, resolution]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform