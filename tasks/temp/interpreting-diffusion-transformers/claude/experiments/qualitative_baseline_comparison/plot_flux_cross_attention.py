
from PIL import Image
import einops
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionSegmentationModel

if __name__ == "__main__":
    os.makedirs("results/flux_cross_attention", exist_ok=True)
    image = Image.open("data/dragon_image.png")
    # Load up the flux cross attention segmentation model
    prompt = "A fire breathing dragon."
    concepts = ["dog", "tree", "ball", "grass", "sky", "background"]
    concepts = ["dragon", "flames", "sky", "rock"]
    concepts = ["dragon", "rock", "sky", "sun", "clouds"]
    layers = list(range(15, 19))

    segmentation_model = RawCrossAttentionSegmentationModel(
        model_name="flux-schnell",
        device="cuda:2",
        offload=False,
    )

    coefficients, _ = segmentation_model.segment_individual_image(
        image,
        concepts,
        caption=prompt,
        device="cuda:2",
        softmax=True,
        layers=layers,
        num_samples=5
    )
    
    # (
    #     [image],
    #     ["dog", "tree", "ball", "grass", "sky"],
    #     concepts,
    #     captions=[prompt],
    #     layers=layers,
    #     device="cuda:2"
    # )


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
        plt.savefig(f"results/flux_cross_attention/{concept}.png", bbox_inches="tight", pad_inches=0)