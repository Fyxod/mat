"""Phase 4 landmark semantic geometry runner."""
from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from phase1.src.config import ModelSettings
from phase1.src.data import load_rgb
from phase1.src.instruct_pipeline import generate_edit, load_instruct_pix2pix
from phase1.src.metrics import image_metrics
from phase1.src.reporting import image_sheet, save_rgb
from phase2.src.cem import CEMState, CandidateSample

from .landmark_geometry import LandmarkGeometryCodec, apply_flow, displacement_stats, flow_to_image
from .landmarks import (
    draw_transformed_landmarks_overlay,
    is_real_landmark_record,
    mediapipe_backend_report,
    save_landmarks_for_face,
)
from .reporting import attach_candidate_paths, candidate_sheet, copy_candidate_best, write_aggregate_outputs
from .scoring import load_clip_scorer, score_phase4_candidate
from .semantic_actions import actions_for_prompt, prompt_type
from .utils import (
    done_path,
    face_ids_from_data,
    landmark_json_path,
    landmark_output_folder,
    load_action_config,
    load_landmark_statuses,
    load_parallel_config,
    load_phase4_config,
    mark_done,
    mark_failed,
    outputs_root,
    phase4_model_payload,
    project_root,
    prompt_slug,
    read_csv,
    read_json,
    relative_path,
    setting_slug,
    status_has_real_landmarks,
    successful_landmark_faces,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


_WORKER_PIPE = None
_WORKER_DEVICE = None


def _torch_device(name: str):
    import torch

    return torch.device(name)


def _worker_init(model_payload: dict[str, Any], device_name: str) -> None:
    global _WORKER_PIPE, _WORKER_DEVICE
    _WORKER_DEVICE = _torch_device(device_name)
    _WORKER_PIPE = load_instruct_pix2pix(ModelSettings.from_mapping(model_payload), _WORKER_DEVICE)


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class SerialEvaluator:
    def __init__(self, model_payload: dict[str, Any], device_name: str) -> None:
        self.device = _torch_device(device_name)
        self.pipe = load_instruct_pix2pix(ModelSettings.from_mapping(model_payload), self.device)

    def evaluate(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [_evaluate_base_candidate(job, self.pipe, self.device) for job in jobs]

    def close(self) -> None:
        del self.pipe
        _clear_cuda()


def _landmark_overlay_sheet(root: Path, rows: list[dict[str, Any]], destination: Path) -> None:
    labels: list[str] = []
    images = []
    for row in rows:
        face_id = str(row.get("face_id", ""))
        overlay = landmark_output_folder(root, face_id) / "landmarks_overlay.jpg"
        if not overlay.exists():
            continue
        detector = str(row.get("detector", "unknown"))
        count = int(row.get("landmark_count", 0) or 0)
        real = "real" if status_has_real_landmarks(row) else "not-real"
        warnings = list(row.get("overlay_sanity_warnings", []) or [])
        labels.append(f"{face_id}\n{detector}\n{count} pts / {real}" + (f"\nwarn={len(warnings)}" if warnings else ""))
        images.append(load_rgb(overlay))
    if images:
        image_sheet(destination, [(labels, images)], columns=min(4, max(1, len(images))), cell_width=180, cell_height=180)


def _landmark_report(rows: list[dict[str, Any]], *, require_real_landmarks: bool) -> str:
    real = [row for row in rows if status_has_real_landmarks(row)]
    templates = [row for row in rows if str(row.get("detector")) == "template"]
    failed = [row for row in rows if not bool(row.get("success", False))]
    lines = [
        "# Phase 4 real landmark detection report",
        "",
        f"- Require real landmarks: {require_real_landmarks}",
        f"- Total faces checked: {len(rows)}",
        f"- Real MediaPipe landmark faces: {len(real)}",
        f"- Template fallback faces: {len(templates)}",
        f"- Failed faces: {len(failed)}",
        "",
        "## Backend check",
        "",
        "```json",
    ]
    import json

    lines.append(json.dumps(mediapipe_backend_report(), indent=2))
    lines.extend(["```", "", "## Per-face status", ""])
    for row in rows:
        warnings = list(row.get("overlay_sanity_warnings", []) or [])
        suffix = f"; warnings={warnings}" if warnings else ""
        lines.append(
            "- "
            f"{row.get('face_id')}: success={row.get('success')} detector={row.get('detector')} "
            f"count={row.get('landmark_count')} real={status_has_real_landmarks(row)} "
            f"failure={row.get('failure_reason')}{suffix}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _candidate_name(sample: CandidateSample) -> str:
    return "cand_000_initial" if sample.initial else f"cand_{sample.candidate_index:03d}_g{sample.generation:02d}_m{sample.member:02d}"


def _model_payload(config: dict[str, Any], image_guidance_scale: float) -> dict[str, Any]:
    return phase4_model_payload(config, image_guidance_scale=image_guidance_scale)


def detect_phase4_landmarks(
    root_value: str | Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    require_real_landmarks: bool = False,
) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase4_config(root)
    detector_mode = str(config.get("landmarks", {}).get("detector", "mediapipe_or_template"))
    prefer_mediapipe = detector_mode != "template"
    rows: list[dict[str, Any]] = []
    face_ids = face_ids_from_data(root)
    if limit is not None:
        face_ids = face_ids[: int(limit)]
    output = outputs_root(root) / "landmarks"
    legacy_status = output / "landmark_status.csv"
    if require_real_landmarks and force and legacy_status.exists() and not (output / "legacy_template_landmark_status_before_phase4c.csv").exists() and not dry_run:
        shutil.copy2(legacy_status, output / "legacy_template_landmark_status_before_phase4c.csv")
    for face_id in face_ids:
        folder = landmark_output_folder(root, face_id)
        status_path = folder / "status.json"
        if status_path.exists() and not force and not dry_run:
            rows.append(read_json(status_path, {}))
            continue
        image_path = root / "data" / face_id / "instruct_512.png"
        status = save_landmarks_for_face(
            root=root,
            face_id=face_id,
            image_path=image_path,
            output=folder,
            prefer_mediapipe=prefer_mediapipe,
            require_real_landmarks=require_real_landmarks,
            dry_run=dry_run,
        )
        rows.append(status)
    success = [row for row in rows if bool(row.get("success", False))]
    real_success = [row for row in rows if status_has_real_landmarks(row)]
    templates = [row for row in rows if str(row.get("detector")) == "template"]
    if not dry_run:
        write_csv(output / "landmark_status.csv", rows)
        if require_real_landmarks:
            write_csv(output / "real_landmark_status.csv", rows)
            write_text(output / "real_landmark_detection_report.md", _landmark_report(rows, require_real_landmarks=True))
            _landmark_overlay_sheet(root, rows, output / "real_landmarks_overlay_sheet.jpg")
        write_json(output / "landmark_summary.json", {
            "face_count": len(rows),
            "successful_faces": [row.get("face_id") for row in success],
            "success_count": len(success),
            "real_landmark_faces": [row.get("face_id") for row in real_success],
            "real_landmark_count": len(real_success),
            "template_fallback_faces": [row.get("face_id") for row in templates],
            "template_fallback_count": len(templates),
            "detectors": sorted({str(row.get("detector")) for row in rows if row.get("detector")}),
            "updated_at": utc_now(),
        })
        min_success = int(config.get("landmarks", {}).get("min_successful_faces", 3))
        if require_real_landmarks:
            if len(real_success) < min_success:
                raise RuntimeError(f"Only {len(real_success)} real MediaPipe landmark detections succeeded; need at least {min_success}.")
        elif len(success) < min_success:
            raise RuntimeError(f"Only {len(success)} Phase 4 landmark detections succeeded; need at least {min_success}.")
    print(f"Phase 4 landmarks: {len(success)} / {len(rows)} successful")
    print(f"Real MediaPipe landmarks detected for {len(real_success)} faces.")
    print(f"Template fallback used for {len(templates)} faces.")
    return rows


def _selected_cases(
    root: Path,
    config: dict[str, Any],
    action_config: dict[str, Any],
    *,
    require_real_landmarks: bool = False,
) -> list[dict[str, Any]]:
    selection = dict(config.get("case_selection", {}))
    source_csv = root / str(selection.get("source_csv", "phase3/outputs/clean_discovery/phase3_clean_selected_cases.csv"))
    rows = read_csv(source_csv)
    success_faces = successful_landmark_faces(root, require_real=require_real_landmarks)
    if not success_faces:
        if require_real_landmarks:
            raise RuntimeError("No real MediaPipe Phase 4 landmarks found. Run detect_phase4_landmarks --require-real-landmarks --force first.")
        raise RuntimeError("No successful Phase 4 landmarks found. Run phase4.scripts.detect_phase4_landmarks first.")
    allowed = {str(prompt).lower() for prompt in selection.get("allowed_prompts", [])}
    preferred_faces = [str(face_id) for face_id in selection.get("preferred_faces", [])]
    prompt_priority = [str(prompt).lower() for prompt in selection.get("prompt_priority", [])]
    face_rank = {face_id: index for index, face_id in enumerate(preferred_faces)}
    prompt_rank = {prompt: index for index, prompt in enumerate(prompt_priority)}
    candidates = [
        row for row in rows
        if str(row.get("face_id")) in success_faces
        and (not allowed or str(row.get("prompt", "")).lower() in allowed)
    ]
    for row in candidates:
        row["phase4_prompt_type"] = prompt_type(str(row.get("prompt", "")), action_config)
        row["clean_clip_margin_float"] = float(row.get("clean_clip_margin", 0.0))
    targets = {str(key): int(value) for key, value in dict(selection.get("minimum_type_targets", {})).items()}
    max_cases = int(selection.get("max_cases", 12))
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str, str]] = set()
    for kind, count in targets.items():
        subset = sorted(
            [row for row in candidates if row.get("phase4_prompt_type") == kind],
            key=lambda row: (
                face_rank.get(str(row.get("face_id")), 999),
                prompt_rank.get(str(row.get("prompt", "")).lower(), 999),
                -float(row.get("clean_clip_margin_float", 0.0)),
            ),
        )
        seen_faces: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for row in subset:
            face_id = str(row.get("face_id"))
            if face_id not in seen_faces:
                ordered.append(row)
                seen_faces.add(face_id)
        ordered.extend([row for row in subset if row not in ordered])
        for row in ordered[:count]:
            key = (str(row.get("face_id")), str(row.get("prompt")), str(row.get("image_guidance_scale")))
            if key not in used_keys and len(selected) < max_cases:
                selected.append(row)
                used_keys.add(key)
    remaining = sorted(
        candidates,
        key=lambda row: (
            prompt_rank.get(str(row.get("prompt", "")).lower(), 999),
            face_rank.get(str(row.get("face_id")), 999),
            -float(row.get("clean_clip_margin_float", 0.0)),
        ),
    )
    for row in remaining:
        if len(selected) >= max_cases:
            break
        key = (str(row.get("face_id")), str(row.get("prompt")), str(row.get("image_guidance_scale")))
        if key not in used_keys:
            selected.append(row)
            used_keys.add(key)
    return selected


def _custom_initial(theta: np.ndarray, method: str, methods: list[str], candidate_index: int = 0) -> CandidateSample:
    method_value = method if method in methods else methods[0]
    return CandidateSample(
        theta=np.asarray(theta, dtype=np.float32),
        method=method_value,
        method_index=methods.index(method_value),
        generation=0,
        member=0,
        candidate_index=candidate_index,
        initial=True,
    )


def _make_job(
    *,
    root: Path,
    phase: str,
    combo_folder: Path,
    sample: CandidateSample,
    case: dict[str, Any],
    budget: dict[str, Any],
    config: dict[str, Any],
    action_config: dict[str, Any],
    codec: LandmarkGeometryCodec,
    seed: int,
) -> dict[str, Any]:
    face_id = str(case["face_id"])
    prompt = str(case["prompt"])
    candidate_name = _candidate_name(sample)
    image_guidance_scale = float(case["image_guidance_scale"])
    return {
        "phase": phase,
        "root": str(root),
        "candidate_name": candidate_name,
        "candidate_index": int(sample.candidate_index),
        "generation": int(sample.generation),
        "member": int(sample.member),
        "initial_candidate": bool(sample.initial),
        "candidate_folder_abs": str(combo_folder / "candidates" / candidate_name),
        "face_id": face_id,
        "case_id": str(case.get("case_id", f"{face_id}__{prompt_slug(prompt)}__{setting_slug(image_guidance_scale)}")),
        "image_path": str(root / str(case["image_path"])),
        "clean_edit_path": str(root / str(case["clean_output_path"])),
        "landmarks_path": str(landmark_json_path(root, face_id)),
        "prompt": prompt,
        "prompt_slug": prompt_slug(prompt),
        "prompt_type": prompt_type(prompt, action_config),
        "image_guidance_scale": image_guidance_scale,
        "guidance_scale": float(config.get("model", {}).get("guidance_scale", 7.5)),
        "budget_name": str(budget["name"]),
        "max_disp_px": float(budget["max_disp_px"]),
        "target_input_ssim": float(budget["target_input_ssim"]),
        "seed": int(seed),
        "model_settings": _model_payload(config, image_guidance_scale),
        "geometry_method": sample.method,
        "geometry_config": dict(config.get("geometry", {})),
        "theta": [float(value) for value in sample.theta],
        "theta_names": codec.parameter_names,
        "action_names": list(codec.action_names),
    }


def _evaluate_base_candidate(job: dict[str, Any], pipe: Any, device: Any) -> dict[str, Any]:
    began = time.monotonic()
    folder = Path(job["candidate_folder_abs"])
    folder.mkdir(parents=True, exist_ok=True)
    original = load_rgb(Path(job["image_path"]), size=(int(job["model_settings"].get("width", 512)), int(job["model_settings"].get("height", 512))))
    clean_edit = load_rgb(Path(job["clean_edit_path"]), size=original.size)
    landmarks = read_json(Path(job["landmarks_path"]), {})
    codec = LandmarkGeometryCodec.from_config(
        action_names=list(job["action_names"]),
        geometry_config=dict(job["geometry_config"]),
    )
    flow, geometry_info = codec.decode(
        np.asarray(job["theta"], dtype=np.float32),
        method=str(job["geometry_method"]),
        landmarks=landmarks,
        height=original.height,
        width=original.width,
        max_disp_px=float(job["max_disp_px"]),
    )
    perturbed = apply_flow(original, flow)
    settings = ModelSettings.from_mapping(job["model_settings"])
    perturbed_edit = generate_edit(pipe, perturbed, str(job["prompt"]), settings, device)
    input_values = image_metrics(original, perturbed)
    output_values = image_metrics(clean_edit, perturbed_edit)
    displacement = {
        **displacement_stats(flow),
        "target_input_ssim": float(job["target_input_ssim"]),
        "max_disp_px_budget": float(job["max_disp_px"]),
        "budget_max_disp_px": float(job["max_disp_px"]),
    }
    save_rgb(folder / "original.png", original)
    save_rgb(folder / "clean_edit.png", clean_edit)
    save_rgb(folder / "original_edited.png", clean_edit)
    save_rgb(folder / "perturbed.png", perturbed)
    save_rgb(folder / "perturbed_edit.png", perturbed_edit)
    save_rgb(folder / "perturbed_edited.png", perturbed_edit)
    save_rgb(folder / "flow.png", flow_to_image(flow))
    landmark_source = Path(job["landmarks_path"]).parent / "landmarks_overlay.jpg"
    if landmark_source.exists():
        shutil.copy2(landmark_source, folder / "landmarks_overlay_original.jpg")
    else:
        save_rgb(folder / "landmarks_overlay_original.jpg", original)
    region_source = Path(job["landmarks_path"]).parent / "regions_overlay.jpg"
    if region_source.exists():
        shutil.copy2(region_source, folder / "regions_overlay_original.jpg")
    draw_transformed_landmarks_overlay(
        perturbed,
        landmarks,
        flow,
        folder / "landmarks_overlay_perturbed.jpg",
        title=f"{job['face_id']} {job['geometry_method']}",
    )
    row = {
        "phase": job["phase"],
        "face_id": job["face_id"],
        "case_id": job["case_id"],
        "prompt": job["prompt"],
        "prompt_slug": job["prompt_slug"],
        "prompt_type": job["prompt_type"],
        "image_guidance_scale": job["image_guidance_scale"],
        "guidance_scale": job["guidance_scale"],
        "budget": job["budget_name"],
        "target_input_ssim": job["target_input_ssim"],
        "max_disp_px_budget": job["max_disp_px"],
        "seed": job["seed"],
        "candidate_name": job["candidate_name"],
        "candidate_index": job["candidate_index"],
        "generation": job["generation"],
        "member": job["member"],
        "initial_candidate": job["initial_candidate"],
        "geometry_method": job["geometry_method"],
        "landmark_detector": landmarks.get("detector"),
        "landmark_count": int(landmarks.get("landmark_count", 0)),
        "real_landmarks": is_real_landmark_record(landmarks),
        "candidate_folder_abs": str(folder),
        "elapsed_seconds": float(time.monotonic() - began),
        "theta_names": job["theta_names"],
        "theta_values": job["theta"],
        **{f"input_{key}": value for key, value in input_values.items()},
        **{f"output_{key}": value for key, value in output_values.items()},
        **displacement,
    }
    write_json(folder / "metrics.json", {"input": input_values, "output": output_values, "displacement": displacement})
    write_json(folder / "geometry_params.json", geometry_info)
    write_json(folder / "action_vector.json", {"names": job["theta_names"], "values": job["theta"], "geometry_info": geometry_info})
    write_json(folder / "base_row.json", row)
    return row


def _evaluate_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_PIPE is None or _WORKER_DEVICE is None:
        raise RuntimeError("Phase 4 worker was not initialized with an InstructPix2Pix pipeline.")
    return _evaluate_base_candidate(job, _WORKER_PIPE, _WORKER_DEVICE)


def _score_and_persist(root: Path, row: dict[str, Any], scoring_config: dict[str, Any], clip_scorer: Any) -> dict[str, Any]:
    folder = Path(row["candidate_folder_abs"])
    original = load_rgb(folder / "original.png")
    clean_edit = load_rgb(folder / "clean_edit.png", size=original.size)
    perturbed = load_rgb(folder / "perturbed.png", size=original.size)
    perturbed_edit = load_rgb(folder / "perturbed_edit.png", size=original.size)
    metrics = read_json(folder / "metrics.json", {})
    semantic = score_phase4_candidate(
        prompt=str(row["prompt"]),
        original=original,
        clean_edit=clean_edit,
        perturbed=perturbed,
        perturbed_edit=perturbed_edit,
        displacement_metrics=dict(metrics.get("displacement", {})),
        scoring_config=scoring_config,
        clip_scorer=clip_scorer,
        budget_name=str(row["budget"]),
    )
    merged = {**row, **semantic}
    merged.pop("candidate_folder_abs", None)
    merged = attach_candidate_paths(root, merged, folder)
    write_json(folder / "semantic_score.json", semantic)
    write_json(folder / "score.json", {"phase4_final_score": merged["phase4_final_score"], "decision_label": merged["decision_label"]})
    write_json(folder / "candidate_row.json", merged)
    candidate_sheet(folder, merged)
    mark_done(folder, {"candidate_name": merged["candidate_name"], "phase4_final_score": merged["phase4_final_score"], "decision_label": merged["decision_label"]})
    return merged


def _load_completed_candidate(folder: Path) -> dict[str, Any] | None:
    row_path = folder / "candidate_row.json"
    if done_path(folder).exists() and row_path.exists():
        return read_json(row_path, None)
    return None


def _evaluate_samples(
    *,
    root: Path,
    samples: list[CandidateSample],
    evaluator: SerialEvaluator | None,
    pool: Any | None,
    scoring_config: dict[str, Any],
    clip_scorer: Any,
    job_kwargs: dict[str, Any],
) -> list[tuple[CandidateSample, dict[str, Any]]]:
    cached: list[tuple[CandidateSample, dict[str, Any]]] = []
    jobs: list[dict[str, Any]] = []
    sample_by_name: dict[str, CandidateSample] = {}
    for sample in samples:
        job = _make_job(sample=sample, **job_kwargs)
        folder = Path(job["candidate_folder_abs"])
        existing = _load_completed_candidate(folder)
        if existing is not None:
            cached.append((sample, existing))
            continue
        jobs.append(job)
        sample_by_name[job["candidate_name"]] = sample
    base_rows: list[dict[str, Any]] = []
    if jobs:
        if pool is not None:
            base_rows = list(pool.map(_evaluate_candidate_job, jobs))
        elif evaluator is not None:
            base_rows = evaluator.evaluate(jobs)
        else:
            raise RuntimeError("No Phase 4 evaluator is available.")
    completed = cached[:]
    for row in base_rows:
        scored = _score_and_persist(root, row, scoring_config, clip_scorer)
        completed.append((sample_by_name[row["candidate_name"]], scored))
    completed.sort(key=lambda item: item[0].candidate_index)
    return completed


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("decision_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _combo_seed(base_seed: int, case_index: int, budget_index: int) -> int:
    return int(base_seed + case_index * 1000 + budget_index * 100)


def _run_combo(
    *,
    root: Path,
    phase: str,
    output: Path,
    case: dict[str, Any],
    case_index: int,
    budget: dict[str, Any],
    budget_index: int,
    cem_config: dict[str, Any],
    config: dict[str, Any],
    action_config: dict[str, Any],
    evaluator: SerialEvaluator | None,
    pool: Any | None,
    clip_scorer: Any,
    force: bool = False,
    initial_theta: list[float] | None = None,
    initial_method: str | None = None,
) -> list[dict[str, Any]]:
    prompt = str(case["prompt"])
    face_id = str(case["face_id"])
    image_guidance_scale = float(case["image_guidance_scale"])
    combo_folder = output / "cases" / face_id / setting_slug(image_guidance_scale) / prompt_slug(prompt) / str(budget["name"]) / "seed_00"
    rows_path = combo_folder / "candidates.csv"
    if done_path(combo_folder).exists() and rows_path.exists() and not force:
        return [dict(row) for row in read_csv(rows_path)]
    combo_folder.mkdir(parents=True, exist_ok=True)
    action_names = actions_for_prompt(prompt, action_config)
    codec = LandmarkGeometryCodec.from_config(action_names=action_names, geometry_config=dict(config.get("geometry", {})))
    methods = list(config.get("geometry", {}).get("methods", ["landmark_semantic_tps", "landmark_piecewise_affine"]))
    seed = _combo_seed(int(cem_config.get("seed", 41001)), case_index, budget_index)
    state = CEMState(
        dimension=codec.dimension,
        methods=methods,
        seed=seed,
        sample_std=float(cem_config.get("sample_std", 0.9)),
        min_std=float(cem_config.get("min_std", 0.08)),
        std_smoothing=float(cem_config.get("std_smoothing", 0.65)),
    )
    if initial_theta is not None:
        trimmed = np.asarray(initial_theta[: codec.dimension], dtype=np.float32)
        if len(trimmed) < codec.dimension:
            trimmed = np.pad(trimmed, (0, codec.dimension - len(trimmed))).astype(np.float32)
        state.mean = trimmed.astype(np.float32)
    rows: list[dict[str, Any]] = []
    index = 0
    job_kwargs = {
        "root": root,
        "phase": phase,
        "combo_folder": combo_folder,
        "case": case,
        "budget": budget,
        "config": config,
        "action_config": action_config,
        "codec": codec,
        "seed": seed,
    }
    if bool(cem_config.get("eval_initial", True)):
        if initial_theta is None:
            sample = state.initial(candidate_index=index)
        else:
            sample = _custom_initial(state.mean, initial_method or methods[0], methods, candidate_index=index)
        rows.extend(row for _, row in _evaluate_samples(
            root=root,
            samples=[sample],
            evaluator=evaluator,
            pool=pool,
            scoring_config=dict(config.get("scoring", {})),
            clip_scorer=clip_scorer,
            job_kwargs=job_kwargs,
        ))
        index += 1
    generation_summaries: list[dict[str, Any]] = []
    for generation in range(1, int(cem_config["generations"]) + 1):
        began = time.monotonic()
        samples = state.sample_generation(generation, int(cem_config["population"]), index)
        evaluated = _evaluate_samples(
            root=root,
            samples=samples,
            evaluator=evaluator,
            pool=pool,
            scoring_config=dict(config.get("scoring", {})),
            clip_scorer=clip_scorer,
            job_kwargs=job_kwargs,
        )
        generation_rows = [row for _, row in evaluated]
        rows.extend(generation_rows)
        ranked_pairs = sorted(evaluated, key=lambda item: float(item[1].get("phase4_final_score", -1e9)), reverse=True)
        elite_pairs = ranked_pairs[: int(cem_config["elite"])]
        state_payload = state.update([sample for sample, _ in elite_pairs])
        summary = {
            "generation": generation,
            "face_id": face_id,
            "prompt": prompt,
            "budget": budget["name"],
            "population": int(cem_config["population"]),
            "elite": int(cem_config["elite"]),
            "best_score_this_generation": float(ranked_pairs[0][1].get("phase4_final_score", 0.0)) if ranked_pairs else None,
            "best_candidate_this_generation": ranked_pairs[0][1].get("candidate_name") if ranked_pairs else None,
            "decision_counts": _decision_counts(generation_rows),
            "elite_candidate_names": [row.get("candidate_name") for _, row in elite_pairs],
            "cem_state": state_payload,
            "elapsed_seconds": float(time.monotonic() - began),
        }
        generation_summaries.append(summary)
        write_json(combo_folder / f"generation_{generation:02d}.json", summary)
        index += len(samples)
    ranked = sorted(rows, key=lambda row: float(row.get("phase4_final_score", -1e9)), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    write_csv(rows_path, ranked)
    write_json(combo_folder / "generation_summaries.json", generation_summaries)
    if ranked:
        copy_candidate_best(root / str(ranked[0]["candidate_folder"]), combo_folder / "best")
    mark_done(combo_folder, {"candidate_count": len(ranked), "best_score": float(ranked[0].get("phase4_final_score", 0.0)) if ranked else None})
    return ranked


def _run_matrix(
    *,
    root: Path,
    phase: str,
    output: Path,
    cases: list[dict[str, Any]],
    budgets: list[dict[str, Any]],
    cem_config: dict[str, Any],
    config: dict[str, Any],
    action_config: dict[str, Any],
    evaluator: SerialEvaluator | None,
    pool: Any | None,
    clip_scorer: Any,
    force: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for budget_index, budget in enumerate(budgets):
            rows.extend(_run_combo(
                root=root,
                phase=phase,
                output=output,
                case=case,
                case_index=case_index,
                budget=budget,
                budget_index=budget_index,
                cem_config=cem_config,
                config=config,
                action_config=action_config,
                evaluator=evaluator,
                pool=pool,
                clip_scorer=clip_scorer,
                force=force,
            ))
    return rows


def _parallel_config(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    parallel = load_parallel_config(root)
    parallel.update(dict(config.get("parallel", {})))
    return parallel


def _open_evaluator(root: Path, config: dict[str, Any], cases: list[dict[str, Any]]):
    parallel = _parallel_config(root, config)
    device_name = str(parallel.get("cuda_device", "cuda:0"))
    model_payload = _model_payload(config, float(cases[0].get("image_guidance_scale", 1.5)))
    if bool(parallel.get("parallel_experimental", False)):
        ctx = mp.get_context(str(parallel.get("worker_start_method", "spawn")))
        pool = ctx.Pool(processes=int(parallel.get("parallel_workers", 2)), initializer=_worker_init, initargs=(model_payload, device_name))
        return None, pool
    return SerialEvaluator(model_payload, device_name), None


def run_phase4a_existence_probe(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase4_config(root)
    action_config = load_action_config(root)
    output = outputs_root(root) / "phase4a_existence_probe"
    if done_path(output).exists() and (output / "phase4a_all_candidates.csv").exists() and not force:
        rows = read_csv(output / "phase4a_all_candidates.csv")
        print(f"Phase 4A already complete: {len(rows)} candidates")
        return rows
    cases = _selected_cases(root, config, action_config)
    if not cases:
        raise RuntimeError("Phase 4A has no selected landmark-success clean cases.")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "phase4a_selected_cases.csv", cases)
    clip_scorer = load_clip_scorer(config)
    evaluator = None
    pool = None
    try:
        evaluator, pool = _open_evaluator(root, config, cases)
        rows = _run_matrix(
            root=root,
            phase="phase4a_existence_probe",
            output=output,
            cases=cases,
            budgets=list(config.get("budgets", [])),
            cem_config=dict(config.get("phase4a", {})),
            config=config,
            action_config=action_config,
            evaluator=evaluator,
            pool=pool,
            clip_scorer=clip_scorer,
            force=force,
        )
    except Exception as error:
        parallel = _parallel_config(root, config)
        if pool is not None and bool(parallel.get("retry_serial_on_parallel_failure", True)):
            write_text(
                output / "phase4a_parallel_fallback.md",
                "# Phase 4A parallel fallback\n\nParallel execution failed; retrying serial.\n\n"
                f"```\n{traceback.format_exc()}\n```\n",
            )
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
            pool = None
            evaluator = SerialEvaluator(_model_payload(config, float(cases[0].get("image_guidance_scale", 1.5))), str(parallel.get("cuda_device", "cuda:0")))
            rows = _run_matrix(
                root=root,
                phase="phase4a_existence_probe",
                output=output,
                cases=cases,
                budgets=list(config.get("budgets", [])),
                cem_config=dict(config.get("phase4a", {})),
                config=config,
                action_config=action_config,
                evaluator=evaluator,
                pool=None,
                clip_scorer=clip_scorer,
                force=force,
            )
        else:
            mark_failed(output, error)
            write_text(output / "phase4a_error_traceback.txt", traceback.format_exc())
            raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if evaluator is not None:
            evaluator.close()
    ranked = write_aggregate_outputs(root, output, "phase4a", rows)
    mark_done(output, {"candidate_count": len(ranked), "selected_case_count": len(cases)})
    print(f"Phase 4A candidates: {len(ranked)}")
    print("Inspect phase4/outputs/phase4a_existence_probe/phase4a_semantic_top_sheet.jpg")
    return ranked


def _real_landmark_status_rows(root: Path) -> list[dict[str, Any]]:
    status_path = outputs_root(root) / "landmarks" / "real_landmark_status.csv"
    rows = read_csv(status_path)
    if rows:
        return rows
    return list(load_landmark_statuses(root).values())


def _ensure_phase4c_real_landmarks(root: Path, config: dict[str, Any]) -> None:
    min_faces = int(config.get("landmarks", {}).get("min_successful_faces", 3))
    rows = _real_landmark_status_rows(root)
    real = [row for row in rows if status_has_real_landmarks(row)]
    if len(real) < min_faces:
        raise RuntimeError(
            f"Phase 4C requires at least {min_faces} real MediaPipe landmark faces, "
            f"but found {len(real)}. Run detect_phase4_landmarks --require-real-landmarks --force first."
        )


def run_phase4c_real_landmark_sanity(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase4_config(root, "phase4c_real_landmark_sanity.json")
    if not config:
        raise RuntimeError("Missing phase4/configs/phase4c_real_landmark_sanity.json")
    action_config = load_action_config(root)
    output = outputs_root(root) / "phase4c_real_landmark_sanity"
    if done_path(output).exists() and (output / "phase4c_all_candidates.csv").exists() and not force:
        rows = read_csv(output / "phase4c_all_candidates.csv")
        print(f"Phase 4C already complete: {len(rows)} candidates")
        return rows
    _ensure_phase4c_real_landmarks(root, config)
    cases = _selected_cases(root, config, action_config, require_real_landmarks=True)
    if not cases:
        raise RuntimeError("Phase 4C has no selected clean-success cases with real MediaPipe landmarks.")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "phase4c_selected_cases.csv", cases)
    clip_scorer = load_clip_scorer(config)
    evaluator = None
    pool = None
    rows: list[dict[str, Any]] = []
    try:
        evaluator, pool = _open_evaluator(root, config, cases)
        rows = _run_matrix(
            root=root,
            phase="phase4c_real_landmark_sanity",
            output=output,
            cases=cases,
            budgets=list(config.get("budgets", [])),
            cem_config=dict(config.get("phase4c", {})),
            config=config,
            action_config=action_config,
            evaluator=evaluator,
            pool=pool,
            clip_scorer=clip_scorer,
            force=force,
        )
    except Exception as error:
        parallel = _parallel_config(root, config)
        if pool is not None and bool(parallel.get("retry_serial_on_parallel_failure", True)):
            write_text(
                output / "phase4c_parallel_fallback.md",
                "# Phase 4C parallel fallback\n\nParallel execution failed; retrying serial.\n\n"
                f"```\n{traceback.format_exc()}\n```\n",
            )
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
            pool = None
            evaluator = SerialEvaluator(_model_payload(config, float(cases[0].get("image_guidance_scale", 1.5))), str(parallel.get("cuda_device", "cuda:0")))
            rows = _run_matrix(
                root=root,
                phase="phase4c_real_landmark_sanity",
                output=output,
                cases=cases,
                budgets=list(config.get("budgets", [])),
                cem_config=dict(config.get("phase4c", {})),
                config=config,
                action_config=action_config,
                evaluator=evaluator,
                pool=None,
                clip_scorer=clip_scorer,
                force=force,
            )
        else:
            mark_failed(output, error)
            write_text(output / "phase4c_error_traceback.txt", traceback.format_exc())
            raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if evaluator is not None:
            evaluator.close()
    ranked = write_aggregate_outputs(root, output, "phase4c", rows)
    mark_done(output, {"candidate_count": len(ranked), "selected_case_count": len(cases)})
    print(f"Phase 4C candidates: {len(ranked)}")
    print("Inspect phase4/outputs/phase4c_real_landmark_sanity/phase4c_semantic_top_sheet.jpg")
    return ranked


def _phase4b_sources(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    output = outputs_root(root) / "phase4a_existence_probe"
    rows = read_csv(output / "phase4a_visible_failure_candidates.csv") + read_csv(output / "phase4a_existence_success_candidates.csv")
    ranked = sorted(rows, key=lambda row: (float(row.get("semantic_drop", 0.0)), float(row.get("phase4_final_score", 0.0))), reverse=True)
    max_cases = int(config.get("phase4b", {}).get("max_source_cases", 3))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in ranked:
        key = (str(row.get("face_id")), str(row.get("prompt")), str(row.get("image_guidance_scale")))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= max_cases:
            break
    return selected


def run_phase4b_tighten(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase4_config(root)
    action_config = load_action_config(root)
    output = outputs_root(root) / "phase4b_tighten"
    sources = _phase4b_sources(root, config)
    if not sources:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "phase4b_summary.json", {"skipped": True, "reason": "phase4a_no_visible_or_existence_success", "candidate_count": 0})
        write_text(output / "phase4b_decision_report.md", "# Phase 4B decision report\n\nPhase 4A found no visible or existence success candidates. Phase 4B is skipped.\n")
        mark_done(output, {"status": "skipped", "reason": "phase4a_no_visible_or_existence_success"})
        print("Phase 4B skipped: no Phase 4A visible/existence success candidates")
        return []
    if done_path(output).exists() and (output / "phase4b_all_candidates.csv").exists() and not force:
        rows = read_csv(output / "phase4b_all_candidates.csv")
        print(f"Phase 4B already complete: {len(rows)} candidates")
        return rows
    clip_scorer = load_clip_scorer(config)
    evaluator = None
    pool = None
    rows: list[dict[str, Any]] = []
    try:
        evaluator, pool = _open_evaluator(root, config, sources)
        for case_index, source in enumerate(sources):
            theta_values = read_json(root / str(source.get("candidate_folder", "")) / "action_vector.json", {}).get("values", [])
            method = str(source.get("geometry_method", "landmark_semantic_tps"))
            for budget_index, budget in enumerate(config.get("phase4b", {}).get("budgets", [])):
                rows.extend(_run_combo(
                    root=root,
                    phase="phase4b_tighten",
                    output=output,
                    case=source,
                    case_index=case_index,
                    budget=budget,
                    budget_index=budget_index,
                    cem_config=dict(config.get("phase4b", {})),
                    config=config,
                    action_config=action_config,
                    evaluator=evaluator,
                    pool=pool,
                    clip_scorer=clip_scorer,
                    force=force,
                    initial_theta=theta_values,
                    initial_method=method,
                ))
    except Exception as error:
        parallel = _parallel_config(root, config)
        if pool is not None and bool(parallel.get("retry_serial_on_parallel_failure", True)):
            write_text(
                output / "phase4b_parallel_fallback.md",
                "# Phase 4B parallel fallback\n\nParallel execution failed; retrying serial.\n\n"
                f"```\n{traceback.format_exc()}\n```\n",
            )
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
            pool = None
            evaluator = SerialEvaluator(_model_payload(config, float(sources[0].get("image_guidance_scale", 1.5))), str(parallel.get("cuda_device", "cuda:0")))
            rows = []
            for case_index, source in enumerate(sources):
                theta_values = read_json(root / str(source.get("candidate_folder", "")) / "action_vector.json", {}).get("values", [])
                method = str(source.get("geometry_method", "landmark_semantic_tps"))
                for budget_index, budget in enumerate(config.get("phase4b", {}).get("budgets", [])):
                    rows.extend(_run_combo(
                        root=root,
                        phase="phase4b_tighten",
                        output=output,
                        case=source,
                        case_index=case_index,
                        budget=budget,
                        budget_index=budget_index,
                        cem_config=dict(config.get("phase4b", {})),
                        config=config,
                        action_config=action_config,
                        evaluator=evaluator,
                        pool=None,
                        clip_scorer=clip_scorer,
                        force=force,
                        initial_theta=theta_values,
                        initial_method=method,
                    ))
        else:
            mark_failed(output, error)
            write_text(output / "phase4b_error_traceback.txt", traceback.format_exc())
            raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if evaluator is not None:
            evaluator.close()
    ranked = write_aggregate_outputs(root, output, "phase4b", rows)
    write_csv(output / "phase4b_selected_sources.csv", sources)
    mark_done(output, {"candidate_count": len(ranked), "source_count": len(sources)})
    print(f"Phase 4B candidates: {len(ranked)}")
    return ranked


def summarize_phase4(root_value: str | Path) -> dict[str, Any]:
    root = project_root(root_value)
    phase4a = read_json(outputs_root(root) / "phase4a_existence_probe" / "phase4a_summary.json", {})
    phase4b = read_json(outputs_root(root) / "phase4b_tighten" / "phase4b_summary.json", {})
    payload = {
        "phase4a_decision_counts": phase4a.get("decision_counts", {}),
        "phase4a_visible_failure_count": phase4a.get("visible_failure_count", 0),
        "phase4a_existence_success_count": phase4a.get("existence_success_count", 0),
        "phase4b_skipped": bool(phase4b.get("skipped", False)),
        "phase4b_decision_counts": phase4b.get("decision_counts", {}),
        "updated_at": utc_now(),
    }
    lines = [
        "# Phase 4 landmark semantic geometry summary",
        "",
        "This summary is generated only after Phase 4 runs. The implementation itself does not change Phase 1/2/3 outputs.",
        "",
        "## Phase 4A",
        "",
    ]
    if phase4a:
        lines.append(f"- Decision counts: {phase4a.get('decision_counts', {})}")
        lines.append(f"- Visible failures: {phase4a.get('visible_failure_count', 0)}")
        lines.append(f"- Existence successes: {phase4a.get('existence_success_count', 0)}")
    else:
        lines.append("- Phase 4A has not run yet.")
    lines.extend(["", "## Phase 4B", ""])
    if phase4b:
        if phase4b.get("skipped"):
            lines.append(f"- Skipped: {phase4b.get('reason')}")
        else:
            lines.append(f"- Decision counts: {phase4b.get('decision_counts', {})}")
    else:
        lines.append("- Phase 4B has not run yet.")
    summary_dir = outputs_root(root) / "summaries"
    write_json(summary_dir / "phase4_summary.json", payload)
    write_text(summary_dir / "phase4_summary.md", "\n".join(lines).rstrip() + "\n")
    mark_done(summary_dir, {"phase4_summary": True})
    print("Wrote phase4/outputs/summaries/phase4_summary.md")
    return payload


def summarize_phase4c(root_value: str | Path) -> dict[str, Any]:
    root = project_root(root_value)
    output = outputs_root(root) / "phase4c_real_landmark_sanity"
    summary = read_json(output / "phase4c_summary.json", {})
    selected_cases = read_csv(output / "phase4c_selected_cases.csv")
    real_status_rows = _real_landmark_status_rows(root)
    real_faces = [row.get("face_id") for row in real_status_rows if status_has_real_landmarks(row)]
    counts = dict(summary.get("decision_counts", {}))
    visible = int(summary.get("visible_failure_count", 0) or 0)
    existence = int(summary.get("existence_success_count", 0) or 0)
    strong = int(counts.get("strong_candidate", 0) or 0)
    conclusion = (
        "Phase 4C found at least one real-landmark visible/existence/strong candidate. Do not broaden automatically; inspect and decide whether to tighten."
        if (visible or existence or strong)
        else "Real MediaPipe landmarks were fixed and tested in a small sanity subset. They still did not produce visible final-edit failures."
    )
    payload = {
        "real_landmark_faces": real_faces,
        "selected_case_count": len(selected_cases),
        "decision_counts": counts,
        "visible_failure_count": visible,
        "existence_success_count": existence,
        "strong_candidate_count": strong,
        "conclusion": conclusion,
        "updated_at": utc_now(),
    }
    lines = [
        "# Phase 4C real MediaPipe landmark sanity summary",
        "",
        f"- Real MediaPipe landmark faces: {real_faces}",
        f"- Selected cases: {len(selected_cases)}",
        f"- Decision counts: {counts}",
        f"- Visible failures: {visible}",
        f"- Existence successes: {existence}",
        f"- Strong candidates: {strong}",
        "",
        conclusion,
        "",
    ]
    if not summary:
        lines.append("Phase 4C summary artifacts were not found yet.")
    summary_dir = outputs_root(root) / "summaries"
    write_json(summary_dir / "phase4c_summary.json", payload)
    write_text(summary_dir / "phase4c_summary.md", "\n".join(lines).rstrip() + "\n")
    print("Wrote phase4/outputs/summaries/phase4c_summary.md")
    return payload


__all__ = [
    "detect_phase4_landmarks",
    "run_phase4a_existence_probe",
    "run_phase4b_tighten",
    "run_phase4c_real_landmark_sanity",
    "summarize_phase4",
    "summarize_phase4c",
]
