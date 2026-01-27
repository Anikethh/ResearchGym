from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from concept_attention.binary_segmentation_baselines.clip_text_span_baseline import CLIPTextSpanSegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sdxl import DAAMStableDiffusionXLSegmentationModel

if __name__ == "__main__":
    os.makedirs("results/daam_sdxl", exist_ok=True)
    image = Image.open("data/dragon_image.png").convert("RGB")
    # image = Image.open("data/dog.png").convert("RGB")
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = image_transform(image)
    # Load up the flux cross attention segmentation model
    # prompt = "a photo of a dog by a ball on the grass with the sky in the background"
    prompt = ""
    concepts = ["dog", "tree", "ball", "grass", "sky", "background"]
    concepts = ["dragon", "rock", "sky", "sun", "clouds"]
    # concepts = ["dog"]

    segmentation_model = DAAMStableDiffusionXLSegmentationModel(device="cuda:2")

    coefficients, _ = segmentation_model.segment_individual_image(
        image,
        concepts,
        caption=prompt,
        device="cuda:2",
        num_samples=1,
        # num_inference_steps=50
    )
    
    # (
    #     [image],
    #     ["dog", "tree", "ball", "grass", "sky"],
    #     concepts,
    #     captions=[prompt],
    #     layers=layers,
    #     device="cuda:2"
    # )

    # Bilinearly interpolate the coefficients to 64x64
    coefficients = F.interpolate(coefficients.unsqueeze(1), size=(64, 64), mode="bilinear").squeeze(1)
    # Plot the coefficients
    vmin = coefficients.min()
    vmax = coefficients.max()
    for concept_index, concept in enumerate(concepts):
        concept_heatmap = coefficients[concept_index].cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Removes padding
        plt.imshow(concept_heatmap, cmap="plasma", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"results/daam_sdxl/{concept}.png", bbox_inches="tight", pad_inches=0)