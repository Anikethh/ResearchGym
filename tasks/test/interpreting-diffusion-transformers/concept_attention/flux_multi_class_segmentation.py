"""
Minimal multi-class wrapper to reuse the FluxGenerator and raw attention/value/output spaces for Pascal VOC.
This adapter runs the generator, collects the requested target space, and produces per-concept masks.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import einops
from PIL import Image

from concept_attention.image_generator import FluxGenerator
from concept_attention.utils import embed_concepts, linear_normalization
from concept_attention.segmentation import add_noise_to_image, encode_image
from concept_attention.flux.src.flux.sampling import prepare, unpack


class FluxMultiClassSegmentation:
    def __init__(self, generator: FluxGenerator) -> None:
        self.generator = generator
        self.is_schnell = "schnell" in getattr(generator, "model_name", "")

    def __call__(
        self,
        image,
        background_concepts: List[str],
        target_concepts: List[str],
        caption: str,
        target_space: str = "cross_attention",
        device: str = "cuda",
        offload: bool = False,
        num_samples: int = 1,
        num_steps: int = 4,
        noise_timestep: int = 2,
        width: int = 1024,
        height: int = 1024,
        layers: Optional[List[int]] = None,
        normalize_concepts: bool = True,
        softmax: bool = False,
        joint_attention_kwargs=None,
        seed: int = 4,
        **kwargs,
    ):
        concepts = background_concepts + target_concepts

        encoded_image_without_noise = encode_image(
            image,
            self.generator.ae,
            offload=offload,
            device=device,
        )

        all_maps = []
        for i in range(num_samples):
            encoded_image, timesteps = add_noise_to_image(
                encoded_image_without_noise,
                num_steps=num_steps,
                noise_timestep=noise_timestep,
                seed=seed + i,
                width=width,
                height=height,
                device=device,
                is_schnell=self.is_schnell,
            )

            if offload:
                self.generator.t5, self.generator.clip = self.generator.t5.to(device), self.generator.clip.to(device)
            inp = prepare(t5=self.generator.t5, clip=self.generator.clip, img=encoded_image, prompt=caption)
            concept_embeddings, concept_ids, concept_vec = embed_concepts(self.generator.clip, self.generator.t5, concepts)
            inp["concepts"] = concept_embeddings.to(encoded_image.device)
            inp["concept_ids"] = concept_ids.to(encoded_image.device)
            inp["concept_vec"] = concept_vec.to(encoded_image.device)

            if offload:
                self.generator.t5, self.generator.clip = self.generator.t5.cpu(), self.generator.clip.cpu()
                torch.cuda.empty_cache()
                self.generator.model = self.generator.model.to(device)

            guidance_vec = torch.full((encoded_image.shape[0],), 0.0, device=encoded_image.device, dtype=encoded_image.dtype)
            t_curr = timesteps[0]
            t_prev = timesteps[1]
            t_vec = torch.full((encoded_image.shape[0],), t_curr, dtype=encoded_image.dtype, device=encoded_image.device)

            if target_space == "output":
                pred, concept_maps, _ = self.generator.model(
                    img=inp["img"],
                    img_ids=inp["img_ids"],
                    txt=inp["txt"],
                    txt_ids=inp["txt_ids"],
                    concepts=inp["concepts"],
                    concept_ids=inp["concept_ids"],
                    concept_vec=inp["concept_vec"],
                    y=inp["concept_vec"],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    stop_after_multimodal_attentions=False,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            elif target_space == "value":
                pred = self.generator.model(
                    img=inp["img"],
                    img_ids=inp["img_ids"],
                    txt=inp["txt"],
                    txt_ids=inp["txt_ids"],
                    concepts=inp["concepts"],
                    concept_ids=inp["concept_ids"],
                    concept_vec=inp["concept_vec"],
                    y=inp["concept_vec"],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    stop_after_multimodal_attentions=True,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                concept_maps = None
            else:  # cross_attention
                pred, concept_maps, _ = self.generator.model(
                    img=inp["img"],
                    img_ids=inp["img_ids"],
                    txt=inp["txt"],
                    txt_ids=inp["txt_ids"],
                    concepts=inp["concepts"],
                    concept_ids=inp["concept_ids"],
                    concept_vec=inp["concept_vec"],
                    y=inp["concept_vec"],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    stop_after_multimodal_attentions=True,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if target_space == "value":
                concept_vectors = []
                image_vectors = []
                for block in self.generator.model.double_blocks:
                    image_vecs = torch.stack(block.image_value_vectors).squeeze(1)
                    concept_vecs = torch.stack(block.concept_value_vectors).squeeze(1)
                    block.clear_cached_vectors()
                    image_vectors.append(image_vecs)
                    concept_vectors.append(concept_vecs)
                image_vectors = torch.stack(image_vectors).to(torch.float32)
                concept_vectors = torch.stack(concept_vectors).to(torch.float32)
                concept_maps = einops.einsum(
                    concept_vectors,
                    image_vectors,
                    "layers timesteps heads concepts dims, layers timesteps heads pixels dims -> layers timesteps heads concepts pixels",
                )
            elif target_space == "output":
                if not isinstance(concept_maps, torch.Tensor):
                    raise RuntimeError("Expected concept heatmaps from model")
            else:
                if not isinstance(concept_maps, torch.Tensor):
                    raise RuntimeError("Expected cross attention maps from model")

            if normalize_concepts:
                concept_maps = linear_normalization(concept_maps, dim=-2)
            if softmax:
                concept_maps = torch.nn.functional.softmax(concept_maps, dim=-2)
            if layers is not None:
                concept_maps = concept_maps[layers]
            concept_maps = einops.reduce(
                concept_maps,
                "layers time heads concepts patches -> concepts patches",
                reduction="mean",
            )
            concept_maps = einops.rearrange(concept_maps, "concepts (h w) -> concepts h w", h=64, w=64)

            all_maps.append(concept_maps)

        concept_maps = torch.stack(all_maps, dim=0).mean(0)
        predicted = concept_maps.argmax(0)

        reconstructed_image = None
        if target_space == "output":
            img_latents = inp["img"] + (t_prev - t_curr) * pred
            img_latents = unpack(img_latents.float(), height, width)
            with torch.autocast(device_type=self.generator.device.type, dtype=torch.bfloat16):
                img = self.generator.ae.decode(img_latents)
            if self.generator.offload:
                self.generator.ae.decoder.cpu()
                torch.cuda.empty_cache()
            img = img.clamp(-1, 1)
            img = einops.rearrange(img[0], "c h w -> h w c")
            reconstructed_image = Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy())

        return predicted, concept_maps, reconstructed_image

