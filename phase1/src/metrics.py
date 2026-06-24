"""Image and attack-score metrics used for filtering and reporting."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from .masks import face_detected


def _arrays(left: Image.Image, right: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    width = min(left.width, right.width)
    height = min(left.height, right.height)
    a = np.asarray(left.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
    b = np.asarray(right.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
    return a, b


def image_metrics(left: Image.Image, right: Image.Image) -> dict[str, float]:
    a, b = _arrays(left, right)
    mse = float(np.mean((a - b) ** 2))
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        ssim = float(max(0.0, 1.0 - math.sqrt(mse) * 4.0))
    return {
        "psnr": float(psnr),
        "ssim": ssim,
        "l2": float(math.sqrt(mse)),
        "mean_abs": float(np.abs(a - b).mean()),
        "max_abs": float(np.abs(a - b).max()),
    }


def image_health(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = array.mean(axis=-1)
    return {
        "mean_brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "near_black_fraction": float((gray < 0.03).mean()),
        "near_white_fraction": float((gray > 0.97).mean()),
    }


def clean_baseline_metrics(
    original: Image.Image,
    edited: Image.Image,
    identity_similarity: float | None = None,
) -> dict[str, Any]:
    metrics = image_metrics(original, edited)
    original_face = face_detected(original)
    edited_face = face_detected(edited)
    health = image_health(edited)
    return {
        **metrics,
        "face_detected_original": original_face,
        "face_detected_edited": edited_face,
        "identity_similarity_sface_if_available": identity_similarity,
        "identity_distance_sface_if_available": (
            None if identity_similarity is None else float(1.0 - identity_similarity)
        ),
        **health,
    }


def prompt_quality(metrics: dict[str, Any], minimum_ssim: float, minimum_contrast: float, min_identity: float) -> dict[str, Any]:
    face_original = metrics.get("face_detected_original")
    face_edited = metrics.get("face_detected_edited")
    identity = metrics.get("identity_similarity_sface_if_available")
    severe_color_collapse = (
        metrics["contrast"] < minimum_contrast
        or metrics["near_black_fraction"] > 0.92
        or metrics["near_white_fraction"] > 0.92
    )
    reject_reasons: list[str] = []
    if face_original is True and face_edited is False:
        reject_reasons.append("edited_face_not_detected")
    if metrics["ssim"] < minimum_ssim:
        reject_reasons.append("ssim_below_threshold")
    if severe_color_collapse:
        reject_reasons.append("global_color_or_contrast_collapse")
    if identity is not None and identity < min_identity:
        reject_reasons.append("identity_similarity_below_threshold")

    edit_strength = min(metrics["l2"] / 0.10, 1.0)
    excessive_drift = max(0.0, (0.55 - metrics["ssim"]) / 0.55)
    face_score = 1.0 if face_edited is True else (0.50 if face_edited is None else 0.0)
    identity_score = 0.75 if identity is None else max(0.0, min(1.0, float(identity)))
    artifact_penalty = 1.0 if severe_color_collapse else 0.0
    score = identity_score + face_score + edit_strength - excessive_drift - artifact_penalty
    status = "rejected" if reject_reasons else ("weak" if score < 1.2 else "keep")
    return {
        "clean_quality_score": float(score),
        "filter_status": status,
        "filter_reasons": ";".join(reject_reasons),
        "edit_strength_score": float(edit_strength),
        "excessive_drift_penalty": float(excessive_drift),
        "artifact_penalty": float(artifact_penalty),
    }


def attack_score(
    input_metrics: dict[str, float],
    output_metrics: dict[str, float],
    displacement: dict[str, float],
    target_input_ssim: float,
    max_disp_px: float,
) -> dict[str, float]:
    normalized_output_l2 = min(float(output_metrics["l2"]) / 0.25, 2.0)
    output_disruption = (1.0 - float(output_metrics["ssim"])) + normalized_output_l2
    input_penalty = max(0.0, target_input_ssim - float(input_metrics["ssim"])) * 8.0
    input_penalty += max(0.0, float(displacement["max_disp_px"]) - max_disp_px) * 2.0
    return {
        "output_disruption": float(output_disruption),
        "normalized_output_l2": float(normalized_output_l2),
        "input_damage_penalty": float(input_penalty),
        "attack_score": float(output_disruption - input_penalty),
    }
