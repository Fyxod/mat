"""True InstructPix2Pix component access for prompt-conditioned white-box gradients.

The implementation deliberately calls the model's VAE, text encoder, scheduler and
UNet directly. It does not use a transfer model or a detached image edit as the
optimization objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .config import ModelSettings
from .data import pil_to_tensor


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported torch_dtype {name!r}") from error


def load_instruct_pix2pix(settings: ModelSettings, device: torch.device):
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
    except ImportError as error:
        raise RuntimeError(
            "Diffusers is not installed. Run scripts/install_linux_a6000.sh before an A6000 run."
        ) from error

    if device.type != "cuda":
        raise RuntimeError(
            "True white-box Phase 1 is CUDA-only by design. Use the A6000 workflow, not a CPU fallback."
        )
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        settings.model_id,
        torch_dtype=_dtype(settings.torch_dtype),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    for module_name in ("vae", "text_encoder", "unet"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return pipe


def assert_whitebox_contract(pipe) -> dict[str, Any]:
    vae_channels = int(pipe.vae.config.latent_channels)
    unet_channels = int(pipe.unet.config.in_channels)
    if unet_channels != 2 * vae_channels:
        raise RuntimeError(
            f"Unexpected InstructPix2Pix channel contract: VAE={vae_channels}, UNet={unet_channels}."
        )
    frozen = all(not parameter.requires_grad for module in (pipe.vae, pipe.text_encoder, pipe.unet) for parameter in module.parameters())
    if not frozen:
        raise RuntimeError("A model parameter is trainable; refusing to run an invalid white-box attack.")
    return {
        "vae_latent_channels": vae_channels,
        "unet_in_channels": unet_channels,
        "model_weights_frozen": frozen,
        "objective_mode": "true_instruct_pix2pix_internal_whitebox",
    }


@dataclass
class InternalReference:
    prompt: str
    prompt_embedding: torch.Tensor
    empty_embedding: torch.Tensor
    original_latent: torch.Tensor
    fixed_noise: torch.Tensor
    timestep: torch.Tensor
    clean_prompt_prediction: torch.Tensor
    clean_empty_prediction: torch.Tensor
    model_settings: ModelSettings


def _encode_prompt_without_cfg(pipe, prompt: str, device: torch.device) -> torch.Tensor:
    # Diffusers' InstructPix2Pix pipeline exposes _encode_prompt. Passing
    # do_classifier_free_guidance=False returns exactly one text-conditioning branch.
    return pipe._encode_prompt(prompt, device, 1, False)


def encode_image_latent(pipe, image_tensor: torch.Tensor) -> torch.Tensor:
    image = (image_tensor * 2.0 - 1.0).to(device=image_tensor.device, dtype=pipe.vae.dtype)
    latent = pipe.vae.encode(image).latent_dist.mode()
    return latent.to(dtype=pipe.unet.dtype)


def _unet_prediction(pipe, image_latent: torch.Tensor, embedding: torch.Tensor, reference: InternalReference) -> torch.Tensor:
    noisy = pipe.scheduler.scale_model_input(reference.fixed_noise, reference.timestep)
    sample = torch.cat([noisy.to(dtype=pipe.unet.dtype), image_latent.to(dtype=pipe.unet.dtype)], dim=1)
    return pipe.unet(
        sample,
        reference.timestep,
        encoder_hidden_states=embedding,
        return_dict=False,
    )[0]


def prepare_internal_reference(
    pipe,
    original: Image.Image,
    prompt: str,
    settings: ModelSettings,
    device: torch.device,
    objective_timestep_index: int,
) -> tuple[torch.Tensor, InternalReference]:
    """Build the fixed prompt/noise/timestep reference used for one attack start."""
    original_tensor = pil_to_tensor(original, device)
    with torch.no_grad():
        prompt_embedding = _encode_prompt_without_cfg(pipe, prompt, device).detach()
        empty_embedding = _encode_prompt_without_cfg(pipe, "", device).detach()
        original_latent = encode_image_latent(pipe, original_tensor).detach()
        pipe.scheduler.set_timesteps(settings.num_inference_steps, device=device)
        steps = pipe.scheduler.timesteps
        timestep = steps[min(max(0, objective_timestep_index), len(steps) - 1)]
        generator = torch.Generator(device=device).manual_seed(settings.seed)
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
    reference = InternalReference(
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
    return original_tensor, reference


def internal_objective(pipe, perturbed: torch.Tensor, reference: InternalReference, objective_name: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return an objective to maximize and all primary white-box terms."""
    perturbed_latent = encode_image_latent(pipe, perturbed)
    prompt_prediction = _unet_prediction(pipe, perturbed_latent, reference.prompt_embedding, reference)
    empty_prediction = _unet_prediction(pipe, perturbed_latent, reference.empty_embedding, reference)

    clean_direction = reference.clean_prompt_prediction - reference.clean_empty_prediction
    perturbed_direction = prompt_prediction - empty_prediction
    edit_direction_mse = F.mse_loss(perturbed_direction.float(), clean_direction.float())
    edit_direction_cosine = F.cosine_similarity(
        perturbed_direction.float().flatten(1), clean_direction.float().flatten(1), dim=1
    ).mean()
    unet_prediction_mse = F.mse_loss(prompt_prediction.float(), reference.clean_prompt_prediction.float())
    vae_conditioning_mse = F.mse_loss(perturbed_latent.float(), reference.original_latent.float())

    if objective_name == "edit_direction":
        objective = edit_direction_mse
    elif objective_name == "unet_prediction":
        objective = unet_prediction_mse
    elif objective_name == "vae_conditioning":
        objective = vae_conditioning_mse
    elif objective_name == "hybrid_edit_unet_vae":
        objective = edit_direction_mse + 0.5 * unet_prediction_mse + 0.25 * vae_conditioning_mse
    else:
        raise ValueError(f"Unsupported internal objective: {objective_name}")

    return objective, {
        "edit_direction_mse": edit_direction_mse,
        "edit_direction_cosine": edit_direction_cosine,
        "unet_prediction_mse": unet_prediction_mse,
        "vae_conditioning_mse": vae_conditioning_mse,
        "perturbed_latent": perturbed_latent,
    }


@torch.no_grad()
def generate_edit(pipe, image: Image.Image, prompt: str, settings: ModelSettings, device: torch.device) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    output = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        image_guidance_scale=settings.image_guidance_scale,
        generator=generator,
    )
    return output.images[0].convert("RGB")
