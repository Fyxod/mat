"""Phase 1C multi-timestep white-box objectives for InstructPix2Pix.

These objectives still optimize only the geometric warp.  They reuse the frozen
InstructPix2Pix VAE/text/UNet components, but compare several fixed
noise/timestep predictions instead of a single timestep proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .config import ModelSettings
from .data import pil_to_tensor
from .instruct_pipeline import (
    InternalReference,
    _encode_prompt_without_cfg,
    _unet_prediction,
    encode_image_latent,
)

PHASE1C_OBJECTIVES = {
    "multi_timestep_edit_direction",
    "multi_timestep_unet_prediction",
    "multi_timestep_hybrid",
}


@dataclass
class MultiTimestepReference:
    prompt: str
    prompt_embedding: torch.Tensor
    empty_embedding: torch.Tensor
    original_latent: torch.Tensor
    references: list[InternalReference]
    timestep_indices: tuple[int, ...]
    model_settings: ModelSettings


def _sanitize_indices(indices: Sequence[int] | None, step_count: int) -> tuple[int, ...]:
    raw = tuple(indices or (3, 6, 10, 14, 18))
    if not raw:
        raw = (6, 10, 14)
    clipped: list[int] = []
    for index in raw:
        value = min(max(0, int(index)), step_count - 1)
        if value not in clipped:
            clipped.append(value)
    return tuple(clipped)


def prepare_multi_timestep_reference(
    pipe,
    original: Image.Image,
    prompt: str,
    settings: ModelSettings,
    device: torch.device,
    timestep_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, MultiTimestepReference]:
    """Build deterministic clean references for multiple denoising timesteps."""
    original_tensor = pil_to_tensor(original, device)
    with torch.no_grad():
        prompt_embedding = _encode_prompt_without_cfg(pipe, prompt, device).detach()
        empty_embedding = _encode_prompt_without_cfg(pipe, "", device).detach()
        original_latent = encode_image_latent(pipe, original_tensor).detach()
        pipe.scheduler.set_timesteps(settings.num_inference_steps, device=device)
        steps = pipe.scheduler.timesteps
        indices = _sanitize_indices(timestep_indices, len(steps))
        references: list[InternalReference] = []
        for offset, index in enumerate(indices):
            timestep = steps[index]
            generator = torch.Generator(device=device).manual_seed(settings.seed + 1009 * (offset + 1))
            fixed_noise = torch.randn(
                original_latent.shape,
                generator=generator,
                device=device,
                dtype=pipe.unet.dtype,
            ) * pipe.scheduler.init_noise_sigma
            placeholder = InternalReference(
                prompt=prompt,
                prompt_embedding=prompt_embedding,
                empty_embedding=empty_embedding,
                original_latent=original_latent,
                fixed_noise=fixed_noise,
                timestep=timestep,
                clean_prompt_prediction=torch.empty(0, device=device),
                clean_empty_prediction=torch.empty(0, device=device),
                model_settings=settings,
            )
            clean_prompt = _unet_prediction(pipe, original_latent, prompt_embedding, placeholder).detach()
            clean_empty = _unet_prediction(pipe, original_latent, empty_embedding, placeholder).detach()
            references.append(
                InternalReference(
                    prompt=prompt,
                    prompt_embedding=prompt_embedding,
                    empty_embedding=empty_embedding,
                    original_latent=original_latent,
                    fixed_noise=fixed_noise,
                    timestep=timestep,
                    clean_prompt_prediction=clean_prompt,
                    clean_empty_prediction=clean_empty,
                    model_settings=settings,
                )
            )

    return original_tensor, MultiTimestepReference(
        prompt=prompt,
        prompt_embedding=prompt_embedding,
        empty_embedding=empty_embedding,
        original_latent=original_latent,
        references=references,
        timestep_indices=indices,
        model_settings=settings,
    )


def multi_timestep_internal_objective(
    pipe,
    perturbed: torch.Tensor,
    reference: MultiTimestepReference,
    objective_name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a Phase 1C objective to maximize and compatible diagnostic terms."""
    if objective_name not in PHASE1C_OBJECTIVES:
        raise ValueError(f"Unsupported Phase 1C objective: {objective_name}")

    perturbed_latent = encode_image_latent(pipe, perturbed)
    direction_losses: list[torch.Tensor] = []
    unet_losses: list[torch.Tensor] = []
    direction_cosines: list[torch.Tensor] = []

    for item in reference.references:
        prompt_prediction = _unet_prediction(pipe, perturbed_latent, item.prompt_embedding, item)
        empty_prediction = _unet_prediction(pipe, perturbed_latent, item.empty_embedding, item)
        clean_direction = item.clean_prompt_prediction - item.clean_empty_prediction
        perturbed_direction = prompt_prediction - empty_prediction
        direction_losses.append(F.mse_loss(perturbed_direction.float(), clean_direction.float()))
        unet_losses.append(F.mse_loss(prompt_prediction.float(), item.clean_prompt_prediction.float()))
        direction_cosines.append(
            F.cosine_similarity(
                perturbed_direction.float().flatten(1),
                clean_direction.float().flatten(1),
                dim=1,
            ).mean()
        )

    edit_direction_mse = torch.stack(direction_losses).mean()
    unet_prediction_mse = torch.stack(unet_losses).mean()
    edit_direction_cosine = torch.stack(direction_cosines).mean()
    vae_conditioning_mse = F.mse_loss(perturbed_latent.float(), reference.original_latent.float())

    if objective_name == "multi_timestep_edit_direction":
        objective = edit_direction_mse
    elif objective_name == "multi_timestep_unet_prediction":
        objective = unet_prediction_mse
    elif objective_name == "multi_timestep_hybrid":
        objective = edit_direction_mse + 0.5 * unet_prediction_mse + 0.25 * vae_conditioning_mse
    else:
        raise ValueError(f"Unsupported Phase 1C objective: {objective_name}")

    return objective, {
        "edit_direction_mse": edit_direction_mse,
        "edit_direction_cosine": edit_direction_cosine,
        "unet_prediction_mse": unet_prediction_mse,
        "vae_conditioning_mse": vae_conditioning_mse,
        "perturbed_latent": perturbed_latent,
        "phase1c_timestep_count": torch.tensor(len(reference.references), device=perturbed.device),
    }


def is_phase1c_objective(name: str) -> bool:
    return name in PHASE1C_OBJECTIVES


__all__ = [
    "MultiTimestepReference",
    "PHASE1C_OBJECTIVES",
    "is_phase1c_objective",
    "multi_timestep_internal_objective",
    "prepare_multi_timestep_reference",
]
