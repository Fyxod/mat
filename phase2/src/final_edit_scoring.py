"""Final-edit-aware scoring for Phase 2 CEM candidates."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from phase1.src.metrics import image_metrics
from phase1.src.semantic_scoring import ClipSemanticScorer, score_final_edit_case


HEADPHONE_CROP_TEXTS = {
    "positive": [
        "a person wearing headphones",
        "headphones on a head",
        "over-ear headphones",
    ],
    "negative": [
        "a person without headphones",
        "ears without headphones",
        "a head without headphones",
    ],
}

HEADPHONE_CROPS = {
    "left": (0.02, 0.20, 0.42, 0.84),
    "right": (0.58, 0.20, 0.98, 0.84),
    "top": (0.18, 0.02, 0.82, 0.45),
}


def load_clip_scorer(config: dict[str, Any]) -> ClipSemanticScorer | None:
    clip = dict(config.get("clip", {}))
    diagnostics: dict[str, Any] = {}
    scorer = ClipSemanticScorer.load_optional(
        model_id=str(clip.get("model_id", "openai/clip-vit-base-patch32")),
        device=clip.get("device"),
        use_safetensors=bool(clip.get("use_safetensors", True)),
        revision=clip.get("revision"),
        diagnostics=diagnostics,
    )
    if scorer is None and bool(config.get("scoring", {}).get("require_clip", True)):
        raise RuntimeError(
            "Phase 2 requires CLIP semantic scoring, but CLIP could not load. "
            f"Diagnostics: {diagnostics}"
        )
    return scorer


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left = int(max(0, min(width - 1, round(box[0] * width))))
    top = int(max(0, min(height - 1, round(box[1] * height))))
    right = int(max(left + 1, min(width, round(box[2] * width))))
    bottom = int(max(top + 1, min(height, round(box[3] * height))))
    return image.convert("RGB").crop((left, top, right, bottom))


def _clip_margin_with_texts(
    clip_scorer: ClipSemanticScorer,
    image: Image.Image,
    *,
    positive: list[str],
    negative: list[str],
) -> float:
    import torch

    texts = positive + negative
    with torch.no_grad():
        inputs = clip_scorer.processor(
            text=texts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(clip_scorer.device) for key, value in inputs.items()}
        outputs = clip_scorer.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            return_dict=True,
        )
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        similarities = (image_features @ text_features.T).float().cpu().numpy()[0]
    positive_values = similarities[: len(positive)]
    negative_values = similarities[len(positive) :]
    return float(np.mean(positive_values) - np.mean(negative_values))


def _headphone_crop_semantics(
    *,
    prompt: str,
    clean_edit: Image.Image,
    perturbed_edit: Image.Image,
    scoring_config: dict[str, Any],
    clip_scorer: ClipSemanticScorer | None,
) -> dict[str, Any]:
    if "headphone" not in prompt.lower() or not bool(scoring_config.get("crop_clip_scoring", False)):
        return {"crop_clip_available": False, "crop_clip_warning": None}
    if clip_scorer is None:
        return {"crop_clip_available": False, "crop_clip_warning": "clip_scorer_unavailable"}
    try:
        payload: dict[str, Any] = {"crop_clip_available": True, "crop_clip_warning": None}
        perturbed_margins: list[float] = []
        crop_drops: list[float] = []
        for name, box in HEADPHONE_CROPS.items():
            clean_margin = _clip_margin_with_texts(
                clip_scorer,
                _crop(clean_edit, box),
                positive=HEADPHONE_CROP_TEXTS["positive"],
                negative=HEADPHONE_CROP_TEXTS["negative"],
            )
            perturbed_margin = _clip_margin_with_texts(
                clip_scorer,
                _crop(perturbed_edit, box),
                positive=HEADPHONE_CROP_TEXTS["positive"],
                negative=HEADPHONE_CROP_TEXTS["negative"],
            )
            drop = float(clean_margin - perturbed_margin)
            payload[f"{name}_crop_clean_margin"] = float(clean_margin)
            payload[f"{name}_crop_perturbed_margin"] = float(perturbed_margin)
            payload[f"{name}_crop_semantic_drop"] = drop
            perturbed_margins.append(float(perturbed_margin))
            crop_drops.append(drop)
        payload["left_crop_semantic_drop"] = float(payload.get("left_crop_semantic_drop", 0.0))
        payload["right_crop_semantic_drop"] = float(payload.get("right_crop_semantic_drop", 0.0))
        payload["top_crop_semantic_drop"] = float(payload.get("top_crop_semantic_drop", 0.0))
        payload["min_crop_margin"] = float(min(perturbed_margins)) if perturbed_margins else 0.0
        payload["mean_crop_margin"] = float(np.mean(perturbed_margins)) if perturbed_margins else 0.0
        payload["min_crop_semantic_drop"] = float(min(crop_drops)) if crop_drops else 0.0
        payload["mean_crop_semantic_drop"] = float(np.mean(crop_drops)) if crop_drops else 0.0
        return payload
    except Exception as error:
        return {"crop_clip_available": False, "crop_clip_warning": str(error)}


def score_phase2_candidate(
    *,
    prompt: str,
    original: Image.Image,
    clean_edit: Image.Image,
    perturbed: Image.Image,
    perturbed_edit: Image.Image,
    displacement_metrics: dict[str, Any],
    scoring_config: dict[str, Any],
    clip_scorer: ClipSemanticScorer | None,
) -> dict[str, Any]:
    input_values = image_metrics(original, perturbed)
    output_values = image_metrics(clean_edit, perturbed_edit)
    semantic = score_final_edit_case(
        prompt=prompt,
        original=original,
        clean_edit=clean_edit,
        perturbed=perturbed,
        perturbed_edit=perturbed_edit,
        input_metrics=input_values,
        output_metrics=output_values,
        displacement_metrics=displacement_metrics,
        optional_clip_model=clip_scorer,
    )
    if bool(scoring_config.get("require_clip", True)) and not bool(semantic.get("clip_available", False)):
        raise RuntimeError(
            "Phase 2 requires working CLIP semantic scoring, but candidate scoring fell back to metric-only. "
            f"warning={semantic.get('clip_warning')!r}"
        )
    output_disruption = (1.0 - float(output_values["ssim"])) + min(float(output_values["l2"]) / 0.25, 2.0)
    semantic_drop = float(semantic.get("semantic_drop", 0.0))
    crop_semantic = _headphone_crop_semantics(
        prompt=prompt,
        clean_edit=clean_edit,
        perturbed_edit=perturbed_edit,
        scoring_config=scoring_config,
        clip_scorer=clip_scorer,
    )
    crop_semantic["full_image_semantic_drop"] = float(semantic_drop)
    target_input_ssim = float(displacement_metrics.get("target_input_ssim", 0.90))
    max_disp_budget = float(displacement_metrics.get("max_disp_px_budget", displacement_metrics.get("budget_max_disp_px", 0.0)))
    input_damage_penalty = max(0.0, target_input_ssim - float(input_values["ssim"])) * 8.0
    max_disp = float(displacement_metrics.get("max_disp_px", 0.0))
    geometry_excess_penalty = 0.0 if max_disp_budget <= 0.0 else max(0.0, max_disp - max_disp_budget) / max(max_disp_budget, 1e-8)
    weights = {
        "semantic_drop_weight": float(scoring_config.get("semantic_drop_weight", 4.0)),
        "output_disruption_weight": float(scoring_config.get("output_disruption_weight", 1.0)),
        "input_damage_penalty_weight": float(scoring_config.get("input_damage_penalty_weight", 3.0)),
        "geometry_excess_penalty_weight": float(scoring_config.get("geometry_excess_penalty_weight", 1.0)),
    }
    phase2_score = (
        weights["semantic_drop_weight"] * max(0.0, semantic_drop)
        + weights["output_disruption_weight"] * output_disruption
        - weights["input_damage_penalty_weight"] * input_damage_penalty
        - weights["geometry_excess_penalty_weight"] * geometry_excess_penalty
    )

    reasons = list(semantic.get("decision_reasons", []))
    clean_margin = float(semantic.get("clean_clip_positive_margin", 0.0))
    perturbed_margin = float(semantic.get("perturbed_clip_positive_margin", 0.0))
    visible_drop = float(scoring_config.get("visible_failure_semantic_drop", 0.055))
    visible_perturbed_margin = float(scoring_config.get("visible_failure_perturbed_margin", -0.020))
    strong_drop = float(scoring_config.get("strong_semantic_drop", 0.04))
    weak_drop = float(scoring_config.get("weak_semantic_drop", 0.015))
    input_ok = float(input_values["ssim"]) >= target_input_ssim and (max_disp_budget <= 0.0 or max_disp <= max_disp_budget + 1e-4)
    if not input_ok:
        decision_label = "reject_input_damage"
    elif clean_margin <= 0.0:
        decision_label = "metric_only_candidate"
        if "clean_edit_clip_margin_nonpositive" not in ";".join(reasons):
            reasons.append(f"clean_edit_clip_margin_nonpositive:{clean_margin:.4f}")
    elif semantic_drop >= visible_drop and perturbed_margin <= visible_perturbed_margin:
        decision_label = "visible_failure_candidate"
    elif semantic_drop >= strong_drop and perturbed_margin < clean_margin:
        decision_label = "strong_candidate"
    elif semantic_drop >= weak_drop and perturbed_margin < clean_margin:
        decision_label = "weak_candidate"
    else:
        decision_label = "metric_only_candidate"
        if semantic_drop <= 0.0:
            reasons.append(f"semantic_drop_nonpositive:{semantic_drop:.4f}")
        else:
            reasons.append(f"semantic_drop_small:{semantic_drop:.4f}")

    return {
        **{f"input_{key}": value for key, value in input_values.items()},
        **{f"output_{key}": value for key, value in output_values.items()},
        **semantic,
        **crop_semantic,
        "decision_label": decision_label,
        "decision_reasons": reasons,
        "phase2_final_score": float(phase2_score),
        "final_attack_score": float(phase2_score),
        "phase2_output_disruption": float(output_disruption),
        "output_disruption_score": float(output_disruption),
        "input_damage_penalty": float(input_damage_penalty),
        "geometry_excess_penalty": float(geometry_excess_penalty),
        "phase2_score_formula": (
            f"{weights['semantic_drop_weight']}*max(0,semantic_drop)"
            f"+{weights['output_disruption_weight']}*output_disruption"
            f"-{weights['input_damage_penalty_weight']}*input_damage_penalty"
            f"-{weights['geometry_excess_penalty_weight']}*geometry_excess_penalty"
        ),
        **weights,
    }


__all__ = ["load_clip_scorer", "score_phase2_candidate"]
