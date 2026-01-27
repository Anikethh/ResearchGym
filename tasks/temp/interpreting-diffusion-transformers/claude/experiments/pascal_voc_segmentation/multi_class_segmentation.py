import einops
from torchvision.transforms import functional as F

from concept_attention.segmentation import generate_concept_basis_and_image_representation


class FluxMultiClassSegmentation():

    def __init__(self, generator):
        self.generator = generator

    def __call__(
        self,
        image, 
        caption: str, 
        background_concepts: list[str],
        target_concepts: list[str],
        noise_timestep: int = 2,
        layers: list[int] = list(range(0, 19)),
        normalize_concepts: bool = True,
        num_steps: int = 4,
        seed: int = 0,
        concept_scale_values: list[float] = [1.0],
        model_name: str = "flux-schnell",
        offload: bool = False,
        device: str = "cuda:1",
        target_space: str = "output",
        height: int = 1024,
        width: int = 1024,
        stop_after_multimodal_attentions: bool = False,
        num_samples: int = 1,
        apply_blur: bool = False,
        # softmax=True,
        joint_attention_kwargs: dict = None,
    ):
        all_concepts = background_concepts + target_concepts
        image_vectors, concept_vectors, reconstructed_image = generate_concept_basis_and_image_representation(
            image=image,
            caption=caption,
            concepts=all_concepts,
            noise_timestep=noise_timestep,
            layers=layers,
            normalize_concepts=normalize_concepts,
            num_steps=num_steps,
            seed=seed,
            model_name=model_name,
            offload=offload,
            device=device,
            target_space=target_space,
            height=height,
            width=width,
            generator=self.generator,
            stop_after_multimodal_attentions=stop_after_multimodal_attentions,
            num_samples=num_samples,
            joint_attention_kwargs=joint_attention_kwargs,
            # **kwargs
        )
        # Form concept heatmaps 
        concept_heatmaps = einops.einsum(
            image_vectors,
            concept_vectors,
            "patches dim, concepts dim -> patches concepts",
        )
        concept_heatmaps = einops.rearrange(
            concept_heatmaps,
            "(h w) concepts -> concepts h w",
            h=64,
            w=64,
        )
        # Apply a gaussian blur here to the coefficients
        if apply_blur:
            concept_heatmaps = F.gaussian_blur(concept_heatmaps, (3, 3), 1.0)

        # Increase the intensity of the background heatmaps by a factor of background_scale
        # concept_heatmaps[:len(background_concepts)] *= background_scale
        # Get the argmax across all concepts
        predicted_concepts = concept_heatmaps.argmax(dim=0)

        return predicted_concepts, concept_heatmaps, reconstructed_image