"""Phase 4 final-edit scoring with existence-probe labels."""
from __future__ import annotations

from typing import Any

from PIL import Image

from phase2.src.final_edit_scoring import load_clip_scorer, score_phase2_candidate


def score_phase4_candidate(
    *,
    prompt: str,
    original: Image.Image,
    clean_edit: Image.Image,
    perturbed: Image.Image,
    perturbed_edit: Image.Image,
    displacement_metrics: dict[str, Any],
    scoring_config: dict[str, Any],
    clip_scorer: Any,
    budget_name: str,
) -> dict[str, Any]:
    semantic = score_phase2_candidate(
        prompt=prompt,
        original=original,
        clean_edit=clean_edit,
        perturbed=perturbed,
        perturbed_edit=perturbed_edit,
        displacement_metrics=displacement_metrics,
        scoring_config=scoring_config,
        clip_scorer=clip_scorer,
    )
    semantic_drop = float(semantic.get("semantic_drop", 0.0))
    perturbed_margin = float(semantic.get("perturbed_clip_positive_margin", 0.0))
    clean_margin = float(semantic.get("clean_clip_positive_margin", 0.0))
    input_ssim = float(semantic.get("input_ssim", 0.0))
    max_disp = float(displacement_metrics.get("max_disp_px", 0.0))
    target_ssim = float(displacement_metrics.get("target_input_ssim", 0.88))
    max_disp_budget = float(displacement_metrics.get("max_disp_px_budget", displacement_metrics.get("budget_max_disp_px", 0.0)))
    input_ok = input_ssim >= target_ssim and (max_disp_budget <= 0.0 or max_disp <= max_disp_budget + 1e-4)
    reasons = list(semantic.get("decision_reasons", []))
    visible_drop = float(scoring_config.get("visible_failure_semantic_drop", 0.07))
    visible_margin = float(scoring_config.get("visible_failure_perturbed_margin", -0.015))
    existence_drop = float(scoring_config.get("existence_success_semantic_drop", 0.07))
    existence_margin = float(scoring_config.get("existence_success_perturbed_margin", -0.015))
    existence_min_ssim = float(scoring_config.get("existence_success_min_input_ssim", 0.75))
    strong_drop = float(scoring_config.get("strong_semantic_drop", 0.05))
    weak_drop = float(scoring_config.get("weak_semantic_drop", 0.02))

    if (
        input_ok
        and clean_margin > 0.0
        and semantic_drop >= visible_drop
        and perturbed_margin <= visible_margin
    ):
        label = "visible_failure_candidate"
    elif (
        str(budget_name) == "existence_probe"
        and clean_margin > 0.0
        and semantic_drop >= existence_drop
        and perturbed_margin <= existence_margin
        and input_ssim >= existence_min_ssim
        and (max_disp_budget <= 0.0 or max_disp <= max_disp_budget + 1e-4)
    ):
        label = "existence_success_candidate"
    elif not input_ok:
        label = "reject_input_damage"
    elif clean_margin <= 0.0:
        label = "metric_only_candidate"
        reasons.append(f"clean_edit_clip_margin_nonpositive:{clean_margin:.4f}")
    elif semantic_drop >= strong_drop and perturbed_margin <= 0.0:
        label = "strong_candidate"
    elif semantic_drop >= weak_drop and perturbed_margin < clean_margin:
        label = "weak_candidate"
    else:
        label = "metric_only_candidate"
        if semantic_drop <= 0.0:
            reasons.append(f"semantic_drop_nonpositive:{semantic_drop:.4f}")
        else:
            reasons.append(f"semantic_drop_small:{semantic_drop:.4f}")

    semantic["decision_label"] = label
    semantic["decision_reasons"] = reasons
    semantic["phase4_final_score"] = float(semantic.get("phase2_final_score", semantic.get("final_attack_score", 0.0)))
    semantic["phase4_score_formula"] = semantic.get("phase2_score_formula")
    semantic["phase4_budget_name"] = budget_name
    return semantic


__all__ = ["load_clip_scorer", "score_phase4_candidate"]

