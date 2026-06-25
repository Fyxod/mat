"""Final-edit-aware semantic scoring for Phase 1 geometric attacks.

The old Phase 1 score measured output pixel disruption.  This module keeps
those raw metrics, but adds an optional CLIP prompt-margin check so candidates
are ranked by visible requested-edit weakening rather than by pixel change
alone.
"""
from __future__ import annotations

import gc
import platform
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


PROMPT_TEXT_PAIRS: dict[str, dict[str, list[str]]] = {
    "add_black_sunglasses": {
        "positive": [
            "a portrait of a person wearing black sunglasses",
            "a face wearing sunglasses",
        ],
        "negative": [
            "a portrait of a person without sunglasses",
            "a face without sunglasses",
        ],
    },
    "add_round_glasses": {
        "positive": [
            "a portrait of a person wearing round glasses",
            "a face wearing eyeglasses",
        ],
        "negative": [
            "a portrait of a person without glasses",
            "a face without eyeglasses",
        ],
    },
    "add_a_small_beard": {
        "positive": [
            "a portrait of a person with a beard",
            "a face with facial hair",
        ],
        "negative": [
            "a portrait of a clean shaven person",
            "a face without a beard",
        ],
    },
    "add_headphones": {
        "positive": [
            "a portrait of a person wearing headphones",
            "a face wearing headphones",
        ],
        "negative": [
            "a portrait of a person without headphones",
        ],
    },
    "add_a_small_earring": {
        "positive": [
            "a portrait of a person wearing an earring",
        ],
        "negative": [
            "a portrait of a person without earrings",
        ],
    },
    "make_the_person_smile_slightly": {
        "positive": [
            "a portrait of a smiling person",
            "a face with a slight smile",
        ],
        "negative": [
            "a portrait of a neutral expression person",
        ],
    },
}


def prompt_text_pairs(prompt: str) -> dict[str, list[str]]:
    slug = _slug(prompt)
    if slug in PROMPT_TEXT_PAIRS:
        return PROMPT_TEXT_PAIRS[slug]
    return {
        "positive": [prompt, "a portrait where the requested edit is visible"],
        "negative": ["a portrait without the requested edit"],
    }


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass
class ClipSemanticScorer:
    model: Any
    processor: Any
    device: str
    warning: str | None = None

    @classmethod
    def load_optional(
        cls,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> "ClipSemanticScorer | None":
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id).to(resolved_device)
            model.eval()
            if diagnostics is not None:
                diagnostics.update({
                    "available": True,
                    "model_id": model_id,
                    "requested_device": device,
                    "resolved_device": resolved_device,
                    "error": None,
                })
            return cls(model=model, processor=processor, device=resolved_device)
        except Exception as error:
            if diagnostics is not None:
                diagnostics.update({
                    "available": False,
                    "model_id": model_id,
                    "requested_device": device,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                })
            return None

    def positive_margin(self, image: Image.Image, prompt: str) -> float:
        import torch

        pairs = prompt_text_pairs(prompt)
        texts = pairs["positive"] + pairs["negative"]
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            similarities = (image_features @ text_features.T).float().cpu().numpy()[0]
        positive = similarities[: len(pairs["positive"])]
        negative = similarities[len(pairs["positive"]) :]
        return float(np.mean(positive) - np.mean(negative))


def score_final_edit_case(
    prompt: str,
    original: Image.Image,
    clean_edit: Image.Image,
    perturbed: Image.Image,
    perturbed_edit: Image.Image,
    input_metrics: dict[str, Any],
    output_metrics: dict[str, Any],
    displacement_metrics: dict[str, Any],
    optional_clip_model: ClipSemanticScorer | None = None,
) -> dict[str, Any]:
    """Score whether the requested edit survives on clean input but weakens after warp."""
    _ = original, perturbed  # kept in the signature for future face/identity hooks
    target_input_ssim = _float(
        displacement_metrics.get("target_input_ssim", input_metrics.get("target_input_ssim", 0.90)),
        0.90,
    )
    max_disp_budget = _float(
        displacement_metrics.get("max_disp_px_budget", displacement_metrics.get("budget_max_disp_px", 0.0)),
        0.0,
    )
    input_ssim = _float(input_metrics.get("ssim", input_metrics.get("input_ssim")), 0.0)
    output_ssim = _float(output_metrics.get("ssim", output_metrics.get("output_ssim")), 1.0)
    output_l2 = _float(output_metrics.get("l2", output_metrics.get("output_l2")), 0.0)
    max_disp = _float(displacement_metrics.get("max_disp_px"), 0.0)

    output_disruption_score = (1.0 - output_ssim) + min(output_l2 / 0.25, 2.0)
    input_damage_penalty = max(0.0, target_input_ssim - input_ssim) * 8.0
    geometry_excess_penalty = (
        0.0 if max_disp_budget <= 0.0 else max(0.0, max_disp - max_disp_budget) / max(max_disp_budget, 1e-8)
    )
    input_preservation_score = _clip01((input_ssim - target_input_ssim) / max(1e-6, 1.0 - target_input_ssim))

    clip_available = optional_clip_model is not None
    clean_margin = 0.0
    perturbed_margin = 0.0
    clip_warning = None
    if optional_clip_model is not None:
        try:
            clean_margin = optional_clip_model.positive_margin(clean_edit, prompt)
            perturbed_margin = optional_clip_model.positive_margin(perturbed_edit, prompt)
        except Exception as error:
            clip_available = False
            clip_warning = str(error)
            clean_margin = 0.0
            perturbed_margin = 0.0

    semantic_drop = float(clean_margin - perturbed_margin) if clip_available else 0.0
    clean_semantic_success_score = float(clean_margin) if clip_available else 0.0
    perturbed_semantic_success_score = float(perturbed_margin) if clip_available else 0.0
    edit_failure_score = max(0.0, semantic_drop)
    final_attack_score = (
        2.0 * semantic_drop
        + 1.0 * output_disruption_score
        - 2.0 * input_damage_penalty
        - 1.0 * geometry_excess_penalty
    )

    reasons: list[str] = []
    if input_ssim < target_input_ssim:
        reasons.append(f"input_ssim_below_target:{input_ssim:.4f}<{target_input_ssim:.4f}")
    if max_disp_budget > 0.0 and max_disp > max_disp_budget + 1e-4:
        reasons.append(f"max_disp_exceeds_budget:{max_disp:.4f}>{max_disp_budget:.4f}")
    if not clip_available:
        reasons.append("clip_unavailable_semantic_fallback")
    elif clean_margin <= 0.0:
        reasons.append(f"clean_edit_clip_margin_nonpositive:{clean_margin:.4f}")
    elif semantic_drop <= 0.0:
        reasons.append(f"semantic_drop_nonpositive:{semantic_drop:.4f}")
    elif semantic_drop < 0.02:
        reasons.append(f"semantic_drop_small:{semantic_drop:.4f}")

    if input_ssim < target_input_ssim or (max_disp_budget > 0.0 and max_disp > max_disp_budget + 1e-4):
        decision_label = "reject_input_damage"
    elif clip_available and clean_margin <= 0.0:
        decision_label = "reject_clean_failed"
    elif clip_available and clean_margin > 0.0 and semantic_drop >= 0.04 and perturbed_margin < clean_margin:
        decision_label = "strong_candidate"
    elif clip_available and clean_margin > 0.0 and semantic_drop >= 0.015:
        decision_label = "weak_candidate"
    elif output_disruption_score > 0.08:
        decision_label = "metric_only_candidate"
    else:
        decision_label = "weak_candidate"

    if clip_warning:
        reasons.append(f"clip_warning:{clip_warning[:160]}")

    return {
        "clean_semantic_success_score": float(clean_semantic_success_score),
        "perturbed_semantic_success_score": float(perturbed_semantic_success_score),
        "semantic_drop": float(semantic_drop),
        "edit_failure_score": float(edit_failure_score),
        "identity_or_face_health_available": False,
        "input_preservation_score": float(input_preservation_score),
        "output_disruption_score": float(output_disruption_score),
        "input_damage_penalty": float(input_damage_penalty),
        "geometry_excess_penalty": float(geometry_excess_penalty),
        "final_attack_score": float(final_attack_score),
        "decision_label": decision_label,
        "decision_reasons": reasons,
        "clean_clip_positive_margin": float(clean_margin),
        "perturbed_clip_positive_margin": float(perturbed_margin),
        "clip_semantic_drop": float(semantic_drop),
        "clip_available": bool(clip_available),
        "clip_warning": clip_warning,
        "target_input_ssim": float(target_input_ssim),
        "max_disp_px_budget": float(max_disp_budget),
    }


def diagnose_clip_load(
    model_id: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> dict[str, Any]:
    """Load CLIP with explicit step-by-step diagnostics.

    This is intentionally separate from the scorer's permissive fallback path:
    Phase 1C uses it as a preflight gate so expensive GPU work does not proceed
    with semantic scoring silently disabled.
    """
    result: dict[str, Any] = {
        "available": False,
        "model_id": model_id,
        "requested_device": device,
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "steps": [],
    }

    def step(name: str, status: str, **extra: Any) -> None:
        result["steps"].append({"name": name, "status": status, **extra})

    try:
        import torch

        result["torch_version"] = getattr(torch, "__version__", "unknown")
        result["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        result["cuda_available"] = bool(torch.cuda.is_available())
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        result["resolved_device"] = resolved_device
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA CLIP scoring, but torch.cuda.is_available() is false.")
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
        step("import_torch", "ok")
    except Exception as error:
        step("import_torch", "failed", error_type=type(error).__name__, error=str(error))
        result.update({"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        return result

    try:
        import transformers
        from transformers import CLIPModel, CLIPProcessor

        result["transformers_version"] = getattr(transformers, "__version__", "unknown")
        step("import_transformers_clip", "ok")
    except Exception as error:
        step("import_transformers_clip", "failed", error_type=type(error).__name__, error=str(error))
        result.update({"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        return result

    processor = None
    model = None
    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        step("load_processor", "ok")
        model = CLIPModel.from_pretrained(model_id).to(result["resolved_device"])
        model.eval()
        step("load_model", "ok")
        # Tiny real forward pass catches tokenizer/image-processor/device errors.
        from PIL import Image
        import torch

        image = Image.new("RGB", (224, 224), "white")
        inputs = processor(text=["a portrait", "a landscape"], images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(result["resolved_device"]) for key, value in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        result["test_image_feature_shape"] = list(image_features.shape)
        result["test_text_feature_shape"] = list(text_features.shape)
        step("test_forward", "ok")
        result["available"] = True
        return result
    except Exception as error:
        step("load_or_forward", "failed", error_type=type(error).__name__, error=str(error))
        result.update({"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        return result
    finally:
        del processor
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "ClipSemanticScorer",
    "PROMPT_TEXT_PAIRS",
    "diagnose_clip_load",
    "prompt_text_pairs",
    "score_final_edit_case",
]
