"""
    Here the experiment is to see if we can get good performance on the 
    zero-shot ImageNet segmentation task. 
"""
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import einops
import os
# from nltk.corpus import wordnet as wn
# import nltk
# nltk.download('wordnet')
from PIL import Image
import argparse


from concept_attention.binary_segmentation_baselines.chefer_clip_vit_baselines import CheferAttentionGradCAMSegmentationModel, CheferFullLRPSegmentationModel, CheferLRPSegmentationModel, CheferLastLayerAttentionSegmentationModel, CheferLastLayerLRPSegmentationModel, CheferRolloutSegmentationModel, CheferTransformerAttributionSegmentationModel

from concept_attention.binary_segmentation_baselines.daam_sd2 import DAAMStableDiffusion2SegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sdxl import DAAMStableDiffusionXLSegmentationModel
from concept_attention.binary_segmentation_baselines.dino import DINOSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_value_space import RawValueSpaceSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionSegmentationModel

from concept_attention.utils import batch_intersection_union, batch_pix_accuracy, get_ap_scores

from data_processing import ImagenetSegmentation
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--segmentation_model", 
        type=str, 
        # default="RawOutputSpace",
        default="DAAMSDXL",
        # default="CheferAttentionGradCAM",
        choices=[
            "RawOutputSpace", 
            "RawValueSpace", 
            "RawCrossAttention",
            "CheferLRP",
            "CheferRollout",
            "CheferLastLayerAttention",
            "CheferAttentionGradCAM",
            "CheferTransformerAttribution",
            "CheferFullLRP",
            "CheferLastLayerLRP",
            "DAAMSDXL",
            "DAAMSD2",
            "DINO"
        ]
    )
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--normalize_concepts", action="store_true")
    parser.add_argument("--softmax", action="store_true")
    parser.add_argument("--offload", default=False)
    parser.add_argument("--concept_cross_attention", action="store_true")
    parser.add_argument("--concept_self_attention", action="store_true")
    parser.add_argument("--downscale_for_eval", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="flux-schnell")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--noise_timestep", type=int, default=3)
    parser.add_argument("--apply_blur", action="store_true")
    parser.add_argument("--background_concepts", type=str, nargs="+", default=["background", "floor", "tree", "person", "grass", "face"])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(14, 18)))
    # parser.add_argument("--resize_ablation", action="store_true")
    parser.add_argument("--image_save_dir", type=str, default="results/segmentation_predictions/daam_segmentations")

    args = parser.parse_args()

    if not os.path.exists(args.image_save_dir):
        os.makedirs(args.image_save_dir)

    dataset = ImagenetSegmentation()

    # Load up the model
    if args.segmentation_model == "RawCrossAttention":
        segmentation_model = RawCrossAttentionSegmentationModel(
            model_name=args.model_name,
            offload=args.offload,
            device=args.device,
        )
    elif args.segmentation_model == "RawValueSpace":
        segmentation_model = RawValueSpaceSegmentationModel(
            model_name=args.model_name,
            offload=args.offload,
            device=args.device,
        )
    elif args.segmentation_model == "RawOutputSpace":
        segmentation_model = RawOutputSpaceSegmentationModel(
            model_name=args.model_name,
            offload=args.offload,
            device=args.device,
        )
    elif args.segmentation_model == "CheferLRP":
        segmentation_model = CheferLRPSegmentationModel()
    elif args.segmentation_model == "CheferRollout":
        segmentation_model = CheferRolloutSegmentationModel()
    elif args.segmentation_model == "CheferLastLayerAttention":
        segmentation_model = CheferLastLayerAttentionSegmentationModel()
    elif args.segmentation_model == "CheferAttentionGradCAM":
        segmentation_model = CheferAttentionGradCAMSegmentationModel()
    elif args.segmentation_model == "CheferTransformerAttribution":
        segmentation_model = CheferTransformerAttributionSegmentationModel()
    elif args.segmentation_model == "CheferFullLRP":
        segmentation_model = CheferFullLRPSegmentationModel()
    elif args.segmentation_model == "CheferLastLayerLRP":
        segmentation_model = CheferLastLayerLRPSegmentationModel()
    elif args.segmentation_model == "DAAMSDXL":
        segmentation_model = DAAMStableDiffusionXLSegmentationModel()
    elif args.segmentation_model == "DAAMSD2":
        segmentation_model = DAAMStableDiffusion2SegmentationModel()
    elif args.segmentation_model == "DINO":
        segmentation_model = DINOSegmentationModel()
    else:
        raise ValueError(f"Segmentation model {args.segmentation_model} not recognized.")
    # # Make transforms for the images and labels (masks)
    image_transforms = transforms.Compose([
        transforms.Resize((512, 512), Image.BICUBIC),
        transforms.ToTensor(),
    ])
    label_transforms = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
        # transforms.Resize((64, 64), Image.NEAREST),
        transforms.ToTensor()
    ])
    # Iterate through the first N images, run concept encoding
    total_correct = 0.0
    total_label = 0.0
    total_inter = 0.0
    total_union = 0.0
    total_ap = []
    for index in range(len(dataset)):
        img, labels, simplified_name = dataset[index]
        # Apply transformations
        img = image_transforms(img)
        labels = label_transforms(labels)
        # Run the segmentation model
        mask, coefficients, reconstructed_image = segmentation_model(
            img,
            target_concepts=[simplified_name],
            concepts=[simplified_name] + args.background_concepts,
            captions=[f"a {simplified_name}"],
            # l1_penalty=0.001,
            stop_after_multimodal_attentions=True,
            mean_value_threshold=True,
            offload=False,
            num_samples=args.num_samples,
            device=args.device,
            num_steps=args.num_steps,
            noise_timestep=args.noise_timestep,
            normalize_concepts=args.normalize_concepts,
            softmax=args.softmax,
            layers=args.layers,
            apply_blur=args.apply_blur,
            # target_space=args.target_space,
            joint_attention_kwargs={
                "concept_cross_attention": args.concept_cross_attention,
                "concept_self_attention": args.concept_self_attention
            }
        )
        mask = mask[0]
        coefficients = coefficients[0]
        if len(coefficients.shape) == 1:    
            coefficients = einops.rearrange(
                coefficients,
                "(h w) -> h w",
                h=64,
                w=64
            )
        # Rescale coefficients to max
        coefficients = (coefficients - coefficients.min()) / (coefficients.max() - coefficients.min())
        coefficients = torch.Tensor(coefficients)
        # Downscale then upscale the coefficients
        if args.downscale_for_eval:
            # The 14x14 mimics the eval size for Chefer et. al. 
            coefficients = torch.nn.functional.interpolate(
                coefficients.unsqueeze(0).unsqueeze(0),
                size=(14, 14),
                mode="nearest"
            ).squeeze()
        # Upscale coefficients to 224 x 224
        coefficients = torch.nn.functional.interpolate(
            coefficients.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            # size=(64, 64),
            mode="nearest"
        ).squeeze()
        coefficients = coefficients.detach().cpu().numpy()
        # Upscale the mask to 224 x 224
        mask = torch.Tensor(mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            # size=(64, 64),
            mode="nearest"
        ).squeeze()
        mask = mask.detach().cpu().numpy()
        reconstructed_image = reconstructed_image[0] if isinstance(reconstructed_image, list) else reconstructed_image
        labels = labels.bool().detach().cpu().numpy().squeeze()
        unpadded_target = torch.Tensor(labels)
        unpadded_coefficients = torch.Tensor(coefficients)
        mask = torch.Tensor(mask)
        unpadded_mask = torch.stack((1 - mask, mask))
        unpadded_target = torch.stack((1 - unpadded_target, unpadded_target))
        current_correct, current_label = batch_pix_accuracy(unpadded_mask, unpadded_target) # (batch_size, h * w)
        total_correct += current_correct
        total_label += current_label
        # Add a 1 - mask and 1 - label
        current_inter, current_union = batch_intersection_union(unpadded_mask, unpadded_target, nclass=2)
        total_inter += current_inter
        total_union += current_union
        unpadded_coefficients = torch.stack((1 - unpadded_coefficients, unpadded_coefficients)).unsqueeze(0)
        labels = torch.Tensor(labels).unsqueeze(0)
        ap_score = np.nan_to_num(
            get_ap_scores(unpadded_coefficients, labels)
        )
        total_ap += [ap_score]
        pixAcc = (
            np.float64(1.0)
            * total_correct
            / (np.spacing(1, dtype=np.float64) + total_label)
        )
        IoU = (
            np.float64(1.0)
            * total_inter
            / (np.spacing(1, dtype=np.float64) + total_union)
        )
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        print(f"Current pixelwise accuracy: {pixAcc: .4f}, Current average IoU: {mIoU: .4f}, Current mAP: {mAp :.4f}")
        # Plot the results
        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        axs[0].imshow(img.permute(1, 2, 0))
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].set_title(f"Segmentation Mask (IoU: {current_inter[0] / current_union[0]:.2f}, Acc: {current_correct / current_label:.2f}), AP: {ap_score[0]:.2f}")
        axs[1].axis("off")
        axs[2].imshow(coefficients)
        axs[2].set_title("Coefficients")
        axs[2].axis("off")
        if reconstructed_image is not None:
            axs[3].imshow(reconstructed_image)
            axs[3].set_title("Reconstructed Image")
            axs[3].axis("off")
        axs[4].imshow(labels.squeeze().numpy())
        axs[4].set_title("Ground Truth")
        axs[4].axis("off")

        plt.savefig(f"{args.image_save_dir}/imagenet_segmentation_{index}.png", dpi=300)
        plt.close()