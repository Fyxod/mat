"""Phase 3 breadth-search runner.

Phase 3 searches for vulnerable image/prompt/settings combinations by scoring
actual InstructPix2Pix final edits.  It reuses the Phase 2 CEM and final-edit
scoring stack, but broadens the matrix across multiple face IDs and clean edit
settings.  No pixel noise, patches, finetuning, LoRA, or model-weight updates
are introduced.
"""
from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from typing import Any

from phase1.src.config import ModelSettings
from phase1.src.data import load_rgb
from phase1.src.instruct_pipeline import generate_edit
from phase1.src.metrics import image_metrics
from phase1.src.reporting import save_rgb
from phase2.src.final_edit_scoring import load_clip_scorer
from phase2.src.phase2_runner import SerialEvaluator, _run_combo

from .reporting import attach_phase3_paths, clean_report, clean_sheet, write_breadth_aggregate
from .utils import (
    configured_prompts,
    detect_phase3_faces,
    done_path,
    load_phase3_config,
    load_prompt_bank,
    mark_done,
    mark_failed,
    outputs_root,
    phase3_model_payload,
    project_root,
    prompt_is_compatible,
    prompt_slug,
    prompt_type,
    read_csv,
    read_json,
    relative_path,
    setting_slug,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


REQUIRED_PHASE3_FACE_IDS = {f"face_{index:03d}" for index in range(2, 9)}


def _require_phase3_bundle(root: Path, faces: list[dict[str, Any]], missing: list[str]) -> None:
    detected = {str(face.get("face_id")) for face in faces}
    missing_ids = sorted(REQUIRED_PHASE3_FACE_IDS - detected)
    messages: list[str] = []
    if missing_ids:
        messages.append("Missing new Phase 3 face IDs: " + ", ".join(missing_ids))
    if missing:
        messages.append("Missing required Phase 3 files:\n" + "\n".join(f"- {item}" for item in missing))
    if messages:
        raise FileNotFoundError(
            "Phase 3 breadth search requires the new image bundle under data/face_002..face_008.\n"
            + "\n\n".join(messages)
        )


def inspect_phase3_images(root_value: str | Path) -> dict[str, Any]:
    root = project_root(root_value)
    faces, missing = detect_phase3_faces(root)
    detected = [str(face.get("face_id")) for face in faces]
    payload = {
        "detected_face_count": len(faces),
        "detected_face_ids": detected,
        "missing_files": missing,
        "required_new_faces_present": sorted(REQUIRED_PHASE3_FACE_IDS).copy(),
        "prompt_restricted_faces": [
            {
                "face_id": face.get("face_id"),
                "existing_glasses": bool(face.get("existing_glasses", False)),
                "existing_facial_hair": bool(face.get("existing_facial_hair", False)),
            }
            for face in faces
            if bool(face.get("prompt_restricted", False))
        ],
    }
    return payload


def _case_id(face_id: str, prompt: str, image_guidance_scale: float) -> str:
    return f"{face_id}__{prompt_slug(prompt)}__{setting_slug(image_guidance_scale)}"


def _clean_case_folder(output: Path, face_id: str, prompt: str, image_guidance_scale: float) -> Path:
    return output / "cases" / face_id / prompt_slug(prompt) / setting_slug(image_guidance_scale)


def _prompt_priority(prompt: str, prompts: list[dict[str, Any]]) -> int:
    for index, item in enumerate(prompts, 1):
        if str(item.get("prompt")) == prompt:
            return int(item.get("priority", index))
    return 999


def _clean_quality(
    *,
    clean_clip_margin: float,
    clean_metrics: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, str]:
    clean_cfg = dict(config.get("clean_discovery", {}))
    min_margin = float(clean_cfg.get("min_clean_clip_margin", 0.0))
    min_ssim = float(clean_cfg.get("min_clean_ssim", 0.45))
    if clean_clip_margin <= min_margin:
        return "reject_clean_semantic_margin", f"clean_clip_margin_nonpositive:{clean_clip_margin:.4f}"
    if float(clean_metrics.get("ssim", 0.0)) < min_ssim:
        return "reject_clean_global_shift", f"clean_ssim_below_threshold:{float(clean_metrics.get('ssim', 0.0)):.4f}"
    return "clean_success_candidate", "clean_clip_margin_positive_and_no_coarse_collapse"


def _select_for_breadth(
    successes: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    prompt_bank: dict[str, Any],
) -> list[dict[str, Any]]:
    clean_cfg = dict(config.get("clean_discovery", {}))
    max_selected = int(clean_cfg.get("max_selected_cases", 24))
    caps = {str(key): int(value) for key, value in dict(clean_cfg.get("prompt_type_caps", {})).items()}
    if not successes:
        return []
    unique_faces = sorted({str(row.get("face_id")) for row in successes})
    max_per_face = max(2, int(math.ceil(max_selected / max(1, len(unique_faces))) + 1))
    type_order = ["headphones", "glasses", "beard", "clothing", "smile", "other"]
    type_counts: dict[str, int] = {}
    face_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    remaining = sorted(successes, key=lambda row: float(row.get("clean_clip_margin", 0.0)), reverse=True)

    # Round-robin over prompt types first so the first run stays broad even when
    # one prompt has uniformly higher CLIP margins.
    while remaining and len(selected) < max_selected:
        changed = False
        for prompt_kind in type_order:
            if len(selected) >= max_selected:
                break
            if type_counts.get(prompt_kind, 0) >= caps.get(prompt_kind, max_selected):
                continue
            candidates = [
                row for row in remaining
                if str(row.get("prompt_type", prompt_type(str(row.get("prompt", "")), prompt_bank))) == prompt_kind
                and face_counts.get(str(row.get("face_id")), 0) < max_per_face
            ]
            if not candidates:
                continue
            row = candidates[0]
            selected.append(row)
            remaining.remove(row)
            type_counts[prompt_kind] = type_counts.get(prompt_kind, 0) + 1
            face_counts[str(row.get("face_id"))] = face_counts.get(str(row.get("face_id")), 0) + 1
            changed = True
        if not changed:
            for row in list(remaining):
                if len(selected) >= max_selected:
                    break
                prompt_kind = str(row.get("prompt_type", prompt_type(str(row.get("prompt", "")), prompt_bank)))
                if type_counts.get(prompt_kind, 0) >= caps.get(prompt_kind, max_selected):
                    continue
                selected.append(row)
                remaining.remove(row)
                type_counts[prompt_kind] = type_counts.get(prompt_kind, 0) + 1
                face_counts[str(row.get("face_id"))] = face_counts.get(str(row.get("face_id")), 0) + 1
            break
    return selected


def run_phase3_prompt_discovery(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    output = outputs_root(root) / "clean_discovery"
    if done_path(output).exists() and (output / "phase3_clean_selected_cases.csv").exists() and not force:
        rows = read_csv(output / "phase3_clean_selected_cases.csv")
        print(f"Phase 3 clean discovery already complete: {len(rows)} selected cases")
        return rows

    config = load_phase3_config(root)
    prompt_bank = load_prompt_bank(root)
    prompts = configured_prompts(root)
    faces, missing = detect_phase3_faces(root)
    _require_phase3_bundle(root, faces, missing)
    output.mkdir(parents=True, exist_ok=True)

    clip_scorer = load_clip_scorer(config)
    image_guidance_scales = [float(value) for value in config.get("image_guidance_scales", [1.5])]
    model_payload = phase3_model_payload(config, image_guidance_scale=image_guidance_scales[0])
    evaluator = SerialEvaluator(model_payload, str(config.get("parallel", {}).get("cuda_device", "cuda:0")))

    discovery_rows: list[dict[str, Any]] = []
    clean_success_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    began = time.monotonic()
    try:
        for face in faces:
            face_id = str(face["face_id"])
            image_path = root / str(face["image_path"])
            for prompt_item in prompts:
                prompt = str(prompt_item["prompt"])
                compatible, compatibility_reason = prompt_is_compatible(face, prompt, prompt_bank)
                for image_guidance_scale in image_guidance_scales:
                    case_id = _case_id(face_id, prompt, image_guidance_scale)
                    prompt_kind = str(prompt_item.get("prompt_type", prompt_type(prompt, prompt_bank)))
                    common = {
                        "phase": "phase3_clean_discovery",
                        "case_id": case_id,
                        "face_id": face_id,
                        "image_path": relative_path(image_path, root),
                        "prompt": prompt,
                        "prompt_slug": prompt_slug(prompt),
                        "prompt_type": prompt_kind,
                        "prompt_priority": int(prompt_item.get("priority", _prompt_priority(prompt, prompts))),
                        "guidance_scale": float(config.get("model", {}).get("guidance_scale", 7.5)),
                        "image_guidance_scale": float(image_guidance_scale),
                        "num_inference_steps": int(config.get("model", {}).get("num_inference_steps", 20)),
                        "seed": int(config.get("model", {}).get("seed", 1234)),
                        "priority_group": str(face.get("priority_group", "")),
                        "existing_glasses": bool(face.get("existing_glasses", False)),
                        "existing_facial_hair": bool(face.get("existing_facial_hair", False)),
                    }
                    if not compatible:
                        skipped_rows.append({
                            **common,
                            "clean_output_path": "",
                            "reject_reason": compatibility_reason,
                            "clean_quality_label": "skip_incompatible_prompt",
                            "reason_selected": "",
                            "selected_for_breadth_probe": False,
                        })
                        continue

                    folder = _clean_case_folder(output, face_id, prompt, image_guidance_scale)
                    folder.mkdir(parents=True, exist_ok=True)
                    original_path = folder / "original.png"
                    clean_path = folder / "clean_edit.png"
                    original = load_rgb(image_path, size=(int(config.get("model", {}).get("width", 512)), int(config.get("model", {}).get("height", 512))))
                    save_rgb(original_path, original)
                    settings = ModelSettings.from_mapping(phase3_model_payload(config, image_guidance_scale=image_guidance_scale))
                    if force or not clean_path.exists():
                        clean_edit = generate_edit(evaluator.pipe, original, prompt, settings, evaluator.device)
                        save_rgb(clean_path, clean_edit)
                    else:
                        clean_edit = load_rgb(clean_path, size=original.size)
                    clean_metrics = image_metrics(original, clean_edit)
                    clean_margin = float(clip_scorer.positive_margin(clean_edit, prompt)) if clip_scorer is not None else 0.0
                    clean_quality_label, reason = _clean_quality(
                        clean_clip_margin=clean_margin,
                        clean_metrics=clean_metrics,
                        config=config,
                    )
                    row = {
                        **common,
                        "clean_output_path": relative_path(clean_path, root),
                        "clean_original_path": relative_path(original_path, root),
                        "clean_clip_margin": clean_margin,
                        "clean_ssim": clean_metrics.get("ssim"),
                        "clean_psnr": clean_metrics.get("psnr"),
                        "clean_l2": clean_metrics.get("l2"),
                        "clean_quality_label": clean_quality_label,
                        "reason_selected": reason if clean_quality_label == "clean_success_candidate" else "",
                        "reject_reason": "" if clean_quality_label == "clean_success_candidate" else reason,
                        "selected_for_breadth_probe": False,
                    }
                    discovery_rows.append(row)
                    write_json(folder / "metrics.json", {"clean": clean_metrics, "clean_clip_margin": clean_margin})
                    write_json(folder / "case_row.json", row)
                    if clean_quality_label == "clean_success_candidate":
                        clean_success_rows.append(row)
                    else:
                        rejected_rows.append(row)
    except Exception as error:
        mark_failed(output, error)
        raise
    finally:
        evaluator.close()

    selected_rows = _select_for_breadth(clean_success_rows, config=config, prompt_bank=prompt_bank)
    selected_ids = {str(row["case_id"]) for row in selected_rows}
    updated_discovery: list[dict[str, Any]] = []
    unselected_success: list[dict[str, Any]] = []
    for row in discovery_rows:
        payload = dict(row)
        if str(payload["case_id"]) in selected_ids:
            payload["selected_for_breadth_probe"] = True
            payload["reason_selected"] = "selected_for_phase3b_breadth_probe"
        elif payload.get("clean_quality_label") == "clean_success_candidate":
            payload["selected_for_breadth_probe"] = False
            payload["reason_selected"] = "clean_success_not_selected_due_first_run_cap_or_diversity"
            unselected_success.append(payload)
        updated_discovery.append(payload)
    selected_rows = [row for row in updated_discovery if bool(row.get("selected_for_breadth_probe", False))]

    write_csv(output / "phase3_clean_discovery.csv", updated_discovery + skipped_rows)
    write_csv(output / "phase3_clean_selected_cases.csv", selected_rows)
    write_csv(output / "phase3_clean_rejected_cases.csv", rejected_rows)
    write_csv(output / "phase3_clean_unselected_success_cases.csv", unselected_success)
    write_csv(output / "phase3_clean_skipped_incompatible_cases.csv", skipped_rows)
    clean_sheet(root, selected_rows if selected_rows else clean_success_rows, output / "phase3_clean_sheet.jpg")
    report = clean_report(
        detected_faces=faces,
        discovery_rows=updated_discovery,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        skipped_rows=skipped_rows,
    )
    write_text(output / "phase3_clean_report.md", report)
    summary = {
        "phase": "phase3_clean_discovery",
        "detected_face_count": len(faces),
        "detected_face_ids": [face["face_id"] for face in faces],
        "new_image_folder_count": len([face for face in faces if str(face.get("face_id")) != "face_001"]),
        "primary_faces": [face["face_id"] for face in faces if face.get("priority_group") in {"primary", "baseline_reference"}],
        "prompt_restricted_faces": [face["face_id"] for face in faces if bool(face.get("prompt_restricted", False))],
        "evaluated_clean_cases": len(updated_discovery),
        "selected_clean_cases": len(selected_rows),
        "rejected_clean_cases": len(rejected_rows),
        "skipped_incompatible_cases": len(skipped_rows),
        "elapsed_seconds": float(time.monotonic() - began),
        "updated_at": utc_now(),
    }
    write_json(output / "phase3_clean_summary.json", summary)
    mark_done(output, summary)
    print(f"Phase 3 clean discovery selected {len(selected_rows)} / {len(updated_discovery)} clean cases")
    print("Inspect phase3/outputs/clean_discovery/phase3_clean_sheet.jpg")
    return selected_rows


def _case_prompt_info(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = phase3_model_payload(config, image_guidance_scale=float(row["image_guidance_scale"]))
    payload.update({
        "prompt": str(row["prompt"]),
        "prompt_slug": str(row.get("prompt_slug", prompt_slug(str(row["prompt"])))),
        "baseline_output_path": str(row["clean_output_path"]),
    })
    return payload


def _augment_candidate_row(root: Path, row: dict[str, Any], clean_case: dict[str, Any]) -> dict[str, Any]:
    payload = attach_phase3_paths(root, row)
    payload.update({
        "phase": "phase3_breadth_probe",
        "case_id": clean_case.get("case_id"),
        "face_id": clean_case.get("face_id"),
        "image_path": clean_case.get("image_path"),
        "clean_output_path": clean_case.get("clean_output_path"),
        "prompt_type": clean_case.get("prompt_type"),
        "guidance_scale": clean_case.get("guidance_scale"),
        "image_guidance_scale": clean_case.get("image_guidance_scale"),
        "clean_discovery_margin": clean_case.get("clean_clip_margin"),
        "phase3_final_score": payload.get("phase2_final_score", payload.get("final_attack_score", 0.0)),
    })
    folder_value = payload.get("candidate_folder")
    if folder_value:
        folder = root / str(folder_value)
        candidate_path = folder / "candidate_row.json"
        existing = read_json(candidate_path, {})
        if isinstance(existing, dict):
            existing.update(payload)
            write_json(candidate_path, existing)
    return payload


def run_phase3_breadth_probe(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    output = outputs_root(root) / "breadth_probe"
    if done_path(output).exists() and (output / "phase3_all_candidates.csv").exists() and not force:
        rows = read_csv(output / "phase3_all_candidates.csv")
        print(f"Phase 3 breadth probe already complete: {len(rows)} candidates")
        return rows

    config = load_phase3_config(root)
    clean_output = outputs_root(root) / "clean_discovery"
    selected_cases = read_csv(clean_output / "phase3_clean_selected_cases.csv")
    minimum = int(config.get("clean_discovery", {}).get("minimum_cases_for_breadth_probe", 8))
    output.mkdir(parents=True, exist_ok=True)
    if len(selected_cases) < minimum:
        message = (
            f"Phase 3B breadth probe skipped: only {len(selected_cases)} clean-success cases were selected; "
            f"minimum configured threshold is {minimum}."
        )
        write_json(output / "phase3_summary.json", {
            "skipped": True,
            "reason": "too_few_clean_success_cases",
            "selected_clean_cases": len(selected_cases),
            "minimum_cases_for_breadth_probe": minimum,
        })
        write_text(output / "phase3_decision_report.md", "# Phase 3 breadth probe decision report\n\n" + message + "\n")
        mark_done(output, {"status": "skipped", "reason": "too_few_clean_success_cases"})
        print(message)
        return []

    clip_scorer = load_clip_scorer(config)
    model_payload = phase3_model_payload(config, image_guidance_scale=float(selected_cases[0]["image_guidance_scale"]))
    evaluator = SerialEvaluator(model_payload, str(config.get("parallel", {}).get("cuda_device", "cuda:0")))
    rows: list[dict[str, Any]] = []
    began = time.monotonic()
    try:
        for case_index, clean_case in enumerate(selected_cases):
            prompt_info = _case_prompt_info(clean_case, config)
            case_output = output / "cases" / str(clean_case["face_id"]) / setting_slug(float(clean_case["image_guidance_scale"]))
            case_config = {
                **config,
                "input_image": str(clean_case["image_path"]),
                "geometry": dict(config.get("geometry", {})),
                "scoring": dict(config.get("scoring", {})),
            }
            for budget_index, budget in enumerate(config.get("budgets", [])):
                combo_rows = _run_combo(
                    root=root,
                    phase="phase3_breadth_probe",
                    output=case_output,
                    prompt_info=prompt_info,
                    prompt_priority=int(clean_case.get("prompt_priority", case_index + 1)),
                    prompt_index=case_index,
                    budget=dict(budget),
                    budget_index=budget_index,
                    seed_index=0,
                    cem_config=dict(config.get("breadth_probe", {})),
                    config=case_config,
                    region_config=dict(config.get("prompt_regions", {})),
                    evaluator=evaluator,
                    pool=None,
                    clip_scorer=clip_scorer,
                    force=force,
                )
                rows.extend(_augment_candidate_row(root, row, clean_case) for row in combo_rows)
    except Exception as error:
        write_text(output / "phase3_error_traceback.txt", traceback.format_exc())
        mark_failed(output, error)
        raise
    finally:
        evaluator.close()

    ranked = write_breadth_aggregate(root, output, rows)
    summary_path = output / "phase3_summary.json"
    summary = read_json(summary_path, {})
    if isinstance(summary, dict):
        summary.update({
            "phase": "phase3_breadth_probe",
            "selected_clean_cases": len(selected_cases),
            "budgets": [budget.get("name") for budget in config.get("budgets", [])],
            "elapsed_seconds": float(time.monotonic() - began),
            "updated_at": utc_now(),
        })
        write_json(summary_path, summary)
    mark_done(output, {"candidate_count": len(ranked), "selected_clean_cases": len(selected_cases)})
    print(f"Phase 3 breadth probe candidates: {len(ranked)}")
    print("Inspect phase3/outputs/breadth_probe/phase3_semantic_top_sheet.jpg")
    return ranked


def summarize_phase3(root_value: str | Path) -> dict[str, Any]:
    root = project_root(root_value)
    clean_summary = read_json(outputs_root(root) / "clean_discovery" / "phase3_clean_summary.json", {})
    breadth_summary = read_json(outputs_root(root) / "breadth_probe" / "phase3_summary.json", {})
    clean_selected = read_csv(outputs_root(root) / "clean_discovery" / "phase3_clean_selected_cases.csv")
    semantic_top = read_csv(outputs_root(root) / "breadth_probe" / "phase3_semantic_top_candidates.csv")
    strong_rows = read_csv(outputs_root(root) / "breadth_probe" / "phase3_strong_candidates.csv")
    phase2c_summary = read_json(root / "phase2" / "outputs" / "phase2c_probe" / "phase2c_summary.json", {})

    payload = {
        "phase2c_negative": True,
        "phase2c_candidate_count": phase2c_summary.get("candidate_count"),
        "phase3_clean_selected_cases": len(clean_selected),
        "phase3_breadth_candidate_count": breadth_summary.get("candidate_count", 0),
        "phase3_decision_counts": breadth_summary.get("decision_counts", {}),
        "phase3_strong_candidate_count": len(strong_rows),
        "updated_at": utc_now(),
    }
    lines = [
        "# Phase 3 final report",
        "",
        "## Phase 2C handoff",
        "",
        "Phase 2C is treated as negative for the exact InstructPix2Pix / face_001 / add-headphones / region-local TPS-DCT-mesh setup. Do not spend more A6000 time deepening that line unless explicitly requested.",
        "",
        "## Phase 3 clean discovery",
        "",
    ]
    if clean_summary:
        lines.extend([
            f"- Detected face folders: {clean_summary.get('detected_face_count')}",
            f"- New image folders: {clean_summary.get('new_image_folder_count')}",
            f"- Selected clean-success cases for breadth probe: {clean_summary.get('selected_clean_cases')}",
            f"- Rejected clean cases: {clean_summary.get('rejected_clean_cases')}",
            f"- Prompt-restricted faces: {clean_summary.get('prompt_restricted_faces')}",
        ])
    else:
        lines.append("- Clean discovery has not run yet.")
    lines.extend(["", "## Phase 3 breadth probe", ""])
    if breadth_summary:
        if breadth_summary.get("skipped"):
            lines.append(f"- Skipped: {breadth_summary.get('reason')}")
        else:
            lines.append(f"- Candidate count: {breadth_summary.get('candidate_count')}")
            lines.append(f"- Decision counts: {breadth_summary.get('decision_counts')}")
            for row in semantic_top[:8]:
                lines.append(
                    "- "
                    f"{row.get('decision_label')}: {row.get('face_id')} / {row.get('prompt')} / igs={row.get('image_guidance_scale')} "
                    f"semantic_drop={float(row.get('semantic_drop', 0.0)):.4f}, "
                    f"score={float(row.get('phase2_final_score', 0.0)):.4f}"
                )
    else:
        lines.append("- Breadth probe has not run yet.")
    lines.extend([
        "",
        "## Decision rule",
        "",
        "If Phase 3 finds visible/strong candidates, run a narrow Phase 3C only on those image/prompt/settings cases. If it finds only weak or metric-only rows, inspect the sheet and avoid automatic deepening.",
    ])
    summary_dir = outputs_root(root) / "summaries"
    write_json(summary_dir / "phase3_final_summary.json", payload)
    write_text(summary_dir / "phase3_final_report.md", "\n".join(lines).rstrip() + "\n")

    phase2_summary_dir = root / "phase2" / "outputs" / "summaries"
    write_text(
        phase2_summary_dir / "phase3_handoff_report.md",
        "# Phase 3 handoff from Phase 2C\n\n"
        "Phase 2C is negative for the saturated face_001/headphones setup. Phase 3 broadens the search across images, prompts, and image-guidance settings while preserving geometry-only perturbations.\n",
    )
    mark_done(summary_dir, {"phase3_summary": True})
    print("Wrote phase3/outputs/summaries/phase3_final_report.md")
    return payload


__all__ = [
    "inspect_phase3_images",
    "run_phase3_breadth_probe",
    "run_phase3_prompt_discovery",
    "summarize_phase3",
]

