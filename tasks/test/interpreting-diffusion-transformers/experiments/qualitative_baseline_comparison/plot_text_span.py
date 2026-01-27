from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from concept_attention.binary_segmentation_baselines.clip_text_span_baseline import CLIPTextSpanSegmentationModel

if __name__ == "__main__":
    os.makedirs("results/clip_text_span", exist_ok=True)
    image = Image.open("data/dragon_image.png")
    # Load up the flux cross attention segmentation model
    prompt = "A fire breathing dragon"
    # parser.add_argument('--seed', type=int, default=6)
    # concepts = ["dog", "tree", "ball", "grass", "sky", "background"]
    concepts = ["dragon", "rock", "sky", "sun", "clouds"]

    segmentation_model = CLIPTextSpanSegmentationModel(device="cuda:2")

    coefficients, _ = segmentation_model.segment_individual_image(
        image,
        concepts,
        caption=prompt,
        device="cuda:2",
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
    coefficients = F.interpolate(coefficients.unsqueeze(1), size=(64, 64), mode="bilinear")
    # Plot the coefficients
    vmin = coefficients.min()
    vmax = coefficients.max()
    for concept_index, concept in enumerate(concepts):
        concept_heatmap = coefficients[concept_index].cpu().numpy().squeeze()

        plt.figure(figsize=(10, 10))
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Removes padding
        plt.imshow(concept_heatmap, cmap="plasma", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"results/clip_text_span/{concept}.png", bbox_inches="tight", pad_inches=0)