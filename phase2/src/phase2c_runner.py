"""Phase 2C targeted headphone failure probe.

Phase 2C is intentionally small and diagnostic.  It keeps the perturbation
space geometry-only, but stops doing broad prompt sweeps.  The only question is
whether the weak Phase 2B headphone signal can be pushed into an actual
visible final-edit failure.
"""
from __future__ import annotations

import math
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from phase1.src.config import ModelSettings
from phase1.src.data import load_rgb
from phase1.src.instruct_pipeline import generate_edit
from phase1.src.metrics import image_metrics
from phase1.src.reporting import save_rgb

from .cem import CEMState, CandidateSample
from .final_edit_scoring import load_clip_scorer
from .geometry_regions import GeometryCodec, apply_flow, displacement_stats, flow_to_image
from .phase2_runner import (
    SerialEvaluator,
    _decision_counts,
    _evaluate_samples,
    _model_payload,
    _prepare_clean_images,
    _score_and_persist,
)
from .reporting import copy_candidate_best, top_sheet, write_aggregate_outputs
from .utils import (
    done_path,
    load_phase2c_config,
    mark_done,
    mark_failed,
    outputs_root,
    project_root,
    read_csv,
    read_json,
    relative_path,
    selected_prompt_map,
    slug,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _budget_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in config.get("budgets", [])}


def _cap_flow(flow: np.ndarray, max_disp_px: float) -> np.ndarray:
    magnitude = np.sqrt(np.sum(flow * flow, axis=2))
    scale = np.minimum(1.0, (0.999 * float(max_disp_px)) / np.maximum(magnitude, 1e-6))
    return (flow * scale[:, :, None]).astype(np.float32)


def _placeholder_sheet(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 48), message, fill=(20, 20, 20))
    image.save(path, quality=92)


def _region_variants(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for name, item in dict(config.get("region_variants", {})).items():
        payload[str(name)] = [dict(region) for region in item.get("regions", [])]
    if "headphones_full" not in payload:
        payload["headphones_full"] = [
            {"center_x": 0.25, "center_y": 0.48, "radius_x": 0.16, "radius_y": 0.30},
            {"center_x": 0.75, "center_y": 0.48, "radius_x": 0.16, "radius_y": 0.30},
            {"center_x": 0.50, "center_y": 0.25, "radius_x": 0.34, "radius_y": 0.15},
        ]
    return payload


def _load_phase2b_sources(root: Path, *, limit: int) -> list[dict[str, Any]]:
    candidates_path = outputs_root(root) / "phase2b_cem" / "phase2b_semantic_top_candidates.csv"
    rows = read_csv(candidates_path)
    sources: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("prompt", "")).lower() != "add headphones":
            continue
        if str(row.get("decision_label", "")) not in {"weak_candidate", "strong_candidate", "visible_failure_candidate"}:
            continue
        folder = root / str(row.get("candidate_folder") or row.get("best_folder") or "")
        if not folder.exists():
            continue
        theta = read_json(folder / "theta.json", {})
        values = theta.get("values", [])
        if not values:
            continue
        geometry = read_json(folder / "geometry_parameters.json", {})
        regions = geometry.get("regions") or []
        method = str(theta.get("method") or row.get("geometry_method") or "region_local_tps")
        sources.append(
            {
                "rank": len(sources) + 1,
                "row": row,
                "folder": folder,
                "theta": np.asarray(values, dtype=np.float32),
                "theta_payload": theta,
                "method": method,
                "regions": [dict(region) for region in regions],
                "geometry_info": dict(geometry.get("info", {})),
                "source_budget_max_disp_px": _float(
                    row.get("max_disp_px_budget", row.get("budget_max_disp_px")),
                    _float(row.get("max_disp_px"), 4.0),
                ),
            }
        )
        if len(sources) >= limit:
            break
    return sources


def _source_flow(
    *,
    source: dict[str, Any],
    codec: GeometryCodec,
    original: Image.Image,
    regions: list[dict[str, Any]],
    method: str,
    max_disp_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    return codec.decode(
        np.asarray(source["theta"], dtype=np.float32),
        method=method,
        height=original.height,
        width=original.width,
        max_disp_px=float(max_disp_px),
        regions=regions,
    )


def _evaluate_fixed_flow_candidate(
    *,
    root: Path,
    folder: Path,
    candidate_name: str,
    prompt_info: dict[str, Any],
    budget: dict[str, Any],
    evaluator: SerialEvaluator,
    original_path: Path,
    clean_edit_path: Path,
    flow: np.ndarray,
    geometry_info: dict[str, Any],
    regions: list[dict[str, Any]],
    theta_payload: dict[str, Any],
    scoring_config: dict[str, Any],
    clip_scorer: Any,
    phase2c_part: str,
    candidate_index: int,
    seed: int,
    source: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if done_path(folder).exists() and (folder / "candidate_row.json").exists() and not force:
        return read_json(folder / "candidate_row.json", {})

    began = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    original = load_rgb(original_path)
    clean_edit = load_rgb(clean_edit_path, size=original.size)
    capped_flow = _cap_flow(np.asarray(flow, dtype=np.float32), float(budget["max_disp_px"]))
    perturbed = apply_flow(original, capped_flow)
    settings = ModelSettings.from_mapping(prompt_info)
    perturbed_edit = generate_edit(evaluator.pipe, perturbed, str(prompt_info["prompt"]), settings, evaluator.device)
    input_values = image_metrics(original, perturbed)
    output_values = image_metrics(clean_edit, perturbed_edit)
    displacement = {
        **displacement_stats(capped_flow),
        "target_input_ssim": float(budget["target_input_ssim"]),
        "max_disp_px_budget": float(budget["max_disp_px"]),
        "budget_max_disp_px": float(budget["max_disp_px"]),
    }

    save_rgb(folder / "original.png", original)
    save_rgb(folder / "original_edited.png", clean_edit)
    save_rgb(folder / "perturbed.png", perturbed)
    save_rgb(folder / "perturbed_edited.png", perturbed_edit)
    save_rgb(folder / "flow.png", flow_to_image(capped_flow))
    source_row = source.get("row", {}) if source else {}
    row = {
        "phase": "phase2c_probe",
        "phase2c_part": phase2c_part,
        "prompt": str(prompt_info["prompt"]),
        "prompt_slug": str(prompt_info["prompt_slug"]),
        "prompt_priority": 1,
        "budget": str(budget["name"]),
        "target_input_ssim": float(budget["target_input_ssim"]),
        "max_disp_px_budget": float(budget["max_disp_px"]),
        "seed": int(seed),
        "candidate_name": candidate_name,
        "candidate_index": int(candidate_index),
        "generation": int(extra.get("generation", 0) if extra else 0),
        "member": int(extra.get("member", candidate_index) if extra else candidate_index),
        "initial_candidate": False,
        "region_key": str(extra.get("region_variant", "headphones_full") if extra else "headphones_full"),
        "geometry_method": str(geometry_info.get("geometry_method", theta_payload.get("method", "unknown"))),
        "candidate_folder_abs": str(folder),
        "source_candidate_name": source_row.get("candidate_name"),
        "source_candidate_folder": relative_path(source["folder"], root) if source else None,
        "source_semantic_drop": source_row.get("semantic_drop"),
        "source_phase2_final_score": source_row.get("phase2_final_score"),
        **geometry_info,
        **displacement,
        **(extra or {}),
        "elapsed_seconds": float(time.monotonic() - began),
    }
    write_json(folder / "metrics.json", {"input": input_values, "output": output_values, "displacement": displacement})
    write_json(folder / "geometry_parameters.json", {"info": geometry_info, "regions": regions})
    write_json(folder / "theta.json", theta_payload)
    write_json(folder / "base_row.json", row)
    return _score_and_persist(root, row, scoring_config, clip_scorer)


def _run_amplification(
    *,
    root: Path,
    output: Path,
    config: dict[str, Any],
    sources: list[dict[str, Any]],
    prompt_info: dict[str, Any],
    evaluator: SerialEvaluator,
    clip_scorer: Any,
    original_path: Path,
    clean_edit_path: Path,
    force: bool,
) -> list[dict[str, Any]]:
    part_config = dict(config.get("amplification", {}))
    budget_by_name = _budget_map(config)
    budgets = [budget_by_name[name] for name in part_config.get("budgets", []) if name in budget_by_name]
    scale_factors = [float(value) for value in part_config.get("scale_factors", [1.0])]
    selected_sources = sources[: int(part_config.get("top_candidates", 15))]
    codec = GeometryCodec.from_config(dict(config.get("geometry", {})))
    original = load_rgb(original_path)
    rows: list[dict[str, Any]] = []
    index = 0
    for source in selected_sources:
        regions = source["regions"] or _region_variants(config)["headphones_full"]
        method = source["method"]
        base_flow, info = _source_flow(
            source=source,
            codec=codec,
            original=original,
            regions=regions,
            method=method,
            max_disp_px=float(source["source_budget_max_disp_px"]),
        )
        for budget in budgets:
            for scale_factor in scale_factors:
                index += 1
                scale_tag = str(scale_factor).replace(".", "p")
                candidate_name = f"amp_{source['rank']:02d}_{source['row'].get('candidate_name', 'source')}_x{scale_tag}_{budget['name']}"
                folder = output / "amplification" / f"source_{source['rank']:02d}_{slug(str(source['row'].get('candidate_name', 'source')))}" / str(budget["name"]) / f"scale_{scale_tag}"
                geometry_info = {
                    **info,
                    "geometry_method": method,
                    "phase2c_geometry_source": "phase2b_theta_scaled_flow",
                    "phase2c_applied_scale": float(scale_factor),
                }
                theta_payload = {
                    **source["theta_payload"],
                    "phase2c_source": relative_path(source["folder"], root),
                    "phase2c_applied_scale": float(scale_factor),
                    "phase2c_projection_budget": str(budget["name"]),
                }
                rows.append(
                    _evaluate_fixed_flow_candidate(
                        root=root,
                        folder=folder,
                        candidate_name=candidate_name,
                        prompt_info=prompt_info,
                        budget=budget,
                        evaluator=evaluator,
                        original_path=original_path,
                        clean_edit_path=clean_edit_path,
                        flow=base_flow * float(scale_factor),
                        geometry_info=geometry_info,
                        regions=regions,
                        theta_payload=theta_payload,
                        scoring_config=dict(config.get("scoring", {})),
                        clip_scorer=clip_scorer,
                        phase2c_part="amplification",
                        candidate_index=index,
                        seed=24001,
                        source=source,
                        extra={"phase2c_scale_factor": float(scale_factor), "region_variant": "source_regions"},
                        force=force,
                    )
                )
    return rows


def _run_region_ablations(
    *,
    root: Path,
    output: Path,
    config: dict[str, Any],
    sources: list[dict[str, Any]],
    prompt_info: dict[str, Any],
    evaluator: SerialEvaluator,
    clip_scorer: Any,
    original_path: Path,
    clean_edit_path: Path,
    force: bool,
) -> list[dict[str, Any]]:
    part_config = dict(config.get("region_ablations", {}))
    variants = _region_variants(config)
    variant_names = [str(name) for name in part_config.get("region_variants", []) if str(name) in variants]
    budget_by_name = _budget_map(config)
    budgets = [budget_by_name[name] for name in part_config.get("budgets", []) if name in budget_by_name]
    scale_factor = float(part_config.get("scale_factor", 1.75))
    selected_sources = sources[: int(part_config.get("top_candidates", 5))]
    codec = GeometryCodec.from_config(dict(config.get("geometry", {})))
    original = load_rgb(original_path)
    rows: list[dict[str, Any]] = []
    index = 0
    for source in selected_sources:
        method = source["method"]
        for variant_name in variant_names:
            regions = variants[variant_name]
            for budget in budgets:
                index += 1
                base_flow, info = _source_flow(
                    source=source,
                    codec=codec,
                    original=original,
                    regions=regions,
                    method=method,
                    max_disp_px=float(budget["max_disp_px"]),
                )
                candidate_name = f"abl_{source['rank']:02d}_{variant_name}_{budget['name']}"
                folder = output / "region_ablations" / variant_name / f"source_{source['rank']:02d}_{slug(str(source['row'].get('candidate_name', 'source')))}" / str(budget["name"])
                geometry_info = {
                    **info,
                    "geometry_method": method,
                    "phase2c_geometry_source": "phase2b_theta_region_ablation",
                    "phase2c_applied_scale": float(scale_factor),
                    "region_variant": variant_name,
                }
                theta_payload = {
                    **source["theta_payload"],
                    "phase2c_source": relative_path(source["folder"], root),
                    "phase2c_region_variant": variant_name,
                    "phase2c_ablation_scale": float(scale_factor),
                }
                rows.append(
                    _evaluate_fixed_flow_candidate(
                        root=root,
                        folder=folder,
                        candidate_name=candidate_name,
                        prompt_info=prompt_info,
                        budget=budget,
                        evaluator=evaluator,
                        original_path=original_path,
                        clean_edit_path=clean_edit_path,
                        flow=base_flow * scale_factor,
                        geometry_info=geometry_info,
                        regions=regions,
                        theta_payload=theta_payload,
                        scoring_config=dict(config.get("scoring", {})),
                        clip_scorer=clip_scorer,
                        phase2c_part="region_ablations",
                        candidate_index=index,
                        seed=24011,
                        source=source,
                        extra={"phase2c_scale_factor": float(scale_factor), "region_variant": variant_name},
                        force=force,
                    )
                )
    return rows


def _init_cem_state_from_source(
    *,
    source: dict[str, Any],
    codec: GeometryCodec,
    methods: list[str],
    seed: int,
    cem_config: dict[str, Any],
) -> CEMState:
    state = CEMState(
        dimension=codec.dimension,
        methods=methods,
        seed=seed,
        sample_std=float(cem_config.get("sample_std", 0.45)),
        min_std=float(cem_config.get("min_std", 0.05)),
        std_smoothing=float(cem_config.get("std_smoothing", 0.65)),
    )
    theta = np.asarray(source["theta"], dtype=np.float32)
    if theta.shape[0] == codec.dimension:
        state.mean = np.clip(theta, -2.5, 2.5).astype(np.float32)
    source_method = str(source.get("method", methods[0]))
    if source_method in methods:
        probs = np.full(len(methods), 0.10 / max(1, len(methods) - 1), dtype=np.float64)
        probs[methods.index(source_method)] = 0.90
        state.method_probs = probs / probs.sum()
    return state


def _run_semantic_heavy_cem_combo(
    *,
    root: Path,
    output: Path,
    config: dict[str, Any],
    source: dict[str, Any],
    prompt_info: dict[str, Any],
    budget: dict[str, Any],
    region_variant: str,
    regions: list[dict[str, Any]],
    evaluator: SerialEvaluator,
    clip_scorer: Any,
    original_path: Path,
    clean_edit_path: Path,
    variant_index: int,
    budget_index: int,
    force: bool,
) -> list[dict[str, Any]]:
    cem_config = dict(config.get("semantic_heavy_cem", {}))
    methods = [str(item) for item in cem_config.get("geometry_methods", ["region_local_tps", "combined_tps_dct", "combined_all"])]
    geometry_config = {**dict(config.get("geometry", {})), "methods": methods}
    codec = GeometryCodec.from_config(geometry_config)
    combo_folder = output / "semantic_heavy_cem" / region_variant / str(budget["name"]) / "seed_00"
    rows_path = combo_folder / "candidates.csv"
    if done_path(combo_folder).exists() and rows_path.exists() and not force:
        return [dict(row) for row in read_csv(rows_path)]

    seed = int(cem_config.get("seed", 24001)) + variant_index * 100 + budget_index
    state = _init_cem_state_from_source(source=source, codec=codec, methods=methods, seed=seed, cem_config=cem_config)
    source_method = str(source.get("method", methods[0]))
    initial_method = source_method if source_method in methods else methods[0]
    rows: list[dict[str, Any]] = []
    index = 0
    job_kwargs = {
        "root": root,
        "phase": "phase2c_probe",
        "combo_folder": combo_folder,
        "prompt_info": prompt_info,
        "prompt_priority": 1,
        "budget": budget,
        "original_path": original_path,
        "clean_edit_path": clean_edit_path,
        "codec": codec,
        "geometry_config": geometry_config,
        "region_key": region_variant,
        "regions": regions,
        "seed": seed,
    }
    if bool(cem_config.get("eval_initial", True)):
        initial = CandidateSample(
            theta=state.mean.copy(),
            method=initial_method,
            method_index=methods.index(initial_method),
            generation=0,
            member=0,
            candidate_index=index,
            initial=True,
        )
        rows.extend(
            {
                **row,
                "phase2c_part": "semantic_heavy_cem",
                "region_variant": region_variant,
                "source_candidate_name": source["row"].get("candidate_name"),
                "source_candidate_folder": relative_path(source["folder"], root),
            }
            for _, row in _evaluate_samples(
                root=root,
                samples=[initial],
                evaluator=evaluator,
                pool=None,
                scoring_config=dict(config.get("scoring", {})),
                clip_scorer=clip_scorer,
                job_kwargs=job_kwargs,
            )
        )
        index += 1

    generation_summaries: list[dict[str, Any]] = []
    for generation in range(1, int(cem_config.get("generations", 4)) + 1):
        began = time.monotonic()
        samples = state.sample_generation(generation, int(cem_config.get("population", 16)), index)
        evaluated = _evaluate_samples(
            root=root,
            samples=samples,
            evaluator=evaluator,
            pool=None,
            scoring_config=dict(config.get("scoring", {})),
            clip_scorer=clip_scorer,
            job_kwargs=job_kwargs,
        )
        generation_rows = []
        for _, row in evaluated:
            row["phase2c_part"] = "semantic_heavy_cem"
            row["region_variant"] = region_variant
            row["source_candidate_name"] = source["row"].get("candidate_name")
            row["source_candidate_folder"] = relative_path(source["folder"], root)
            generation_rows.append(row)
        rows.extend(generation_rows)
        ranked_pairs = sorted(evaluated, key=lambda item: float(item[1].get("phase2_final_score", -1e9)), reverse=True)
        elite_pairs = ranked_pairs[: int(cem_config.get("elite", 4))]
        state_payload = state.update([sample for sample, _ in elite_pairs])
        summary = {
            "generation": generation,
            "prompt": prompt_info["prompt"],
            "budget": budget["name"],
            "region_variant": region_variant,
            "population": int(cem_config.get("population", 16)),
            "elite": int(cem_config.get("elite", 4)),
            "best_score_this_generation": float(ranked_pairs[0][1].get("phase2_final_score", 0.0)) if ranked_pairs else None,
            "best_candidate_this_generation": ranked_pairs[0][1].get("candidate_name") if ranked_pairs else None,
            "decision_counts": _decision_counts(generation_rows),
            "elite_candidate_names": [row.get("candidate_name") for _, row in elite_pairs],
            "cem_state": state_payload,
            "elapsed_seconds": float(time.monotonic() - began),
        }
        generation_summaries.append(summary)
        write_json(combo_folder / f"generation_{generation:02d}.json", summary)
        index += len(samples)

    ranked = sorted(rows, key=lambda row: float(row.get("phase2_final_score", -1e9)), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    write_csv(rows_path, ranked)
    write_json(combo_folder / "generation_summaries.json", generation_summaries)
    if ranked:
        best_folder = root / str(ranked[0]["candidate_folder"])
        copy_candidate_best(best_folder, combo_folder / "best")
    mark_done(combo_folder, {"candidate_count": len(ranked), "best_score": float(ranked[0].get("phase2_final_score", 0.0)) if ranked else None})
    return ranked


def _run_semantic_heavy_cem(
    *,
    root: Path,
    output: Path,
    config: dict[str, Any],
    sources: list[dict[str, Any]],
    prompt_info: dict[str, Any],
    evaluator: SerialEvaluator,
    clip_scorer: Any,
    original_path: Path,
    clean_edit_path: Path,
    force: bool,
) -> list[dict[str, Any]]:
    cem_config = dict(config.get("semantic_heavy_cem", {}))
    variants = _region_variants(config)
    variant_names = [str(name) for name in cem_config.get("region_variants", []) if str(name) in variants]
    budget_by_name = _budget_map(config)
    budgets = [budget_by_name[name] for name in cem_config.get("budgets", []) if name in budget_by_name]
    init_sources = sources[: max(1, int(cem_config.get("init_top_candidates", 4)))]
    rows: list[dict[str, Any]] = []
    for variant_index, variant_name in enumerate(variant_names):
        for budget_index, budget in enumerate(budgets):
            source = init_sources[(variant_index + budget_index) % len(init_sources)]
            rows.extend(
                _run_semantic_heavy_cem_combo(
                    root=root,
                    output=output,
                    config=config,
                    source=source,
                    prompt_info=prompt_info,
                    budget=budget,
                    region_variant=variant_name,
                    regions=variants[variant_name],
                    evaluator=evaluator,
                    clip_scorer=clip_scorer,
                    original_path=original_path,
                    clean_edit_path=clean_edit_path,
                    variant_index=variant_index,
                    budget_index=budget_index,
                    force=force,
                )
            )
    return rows


def phase2c_expected_evaluations(config: dict[str, Any]) -> dict[str, int]:
    amplification = dict(config.get("amplification", {}))
    ablations = dict(config.get("region_ablations", {}))
    cem = dict(config.get("semantic_heavy_cem", {}))
    amp_count = int(amplification.get("top_candidates", 15)) * len(amplification.get("scale_factors", [])) * len(amplification.get("budgets", []))
    abl_count = int(ablations.get("top_candidates", 5)) * len(ablations.get("region_variants", [])) * len(ablations.get("budgets", []))
    cem_count = len(cem.get("region_variants", [])) * len(cem.get("budgets", [])) * (
        (1 if bool(cem.get("eval_initial", True)) else 0) + int(cem.get("generations", 4)) * int(cem.get("population", 16))
    )
    return {
        "amplification": int(amp_count),
        "region_ablations": int(abl_count),
        "semantic_heavy_cem": int(cem_count),
        "total": int(amp_count + abl_count + cem_count),
    }


def _phase2c_conclusion(rows: list[dict[str, Any]]) -> str:
    if any(row.get("decision_label") == "visible_failure_candidate" for row in rows):
        return "Phase 2C found a visible headphone edit failure."
    return (
        "Phase 2C did not find a visible headphone edit failure. The best candidates only weaken or shift the "
        "headphones. Do not spend more A6000 time on the same InstructPix2Pix/headphones setup."
    )


def _write_phase2c_outputs(root: Path, output: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = write_aggregate_outputs(root, output, "phase2c", rows)
    visible = [row for row in ranked if row.get("decision_label") == "visible_failure_candidate"]
    semantic_ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("semantic_drop", 0.0)),
            float(row.get("phase2_final_score", row.get("final_attack_score", 0.0))),
        ),
        reverse=True,
    )
    write_csv(output / "phase2c_visible_failure_candidates.csv", visible)
    if visible:
        top_sheet(root, visible, output / "phase2c_visible_failure_sheet.jpg", score_key="semantic_drop")
    else:
        _placeholder_sheet(
            output / "phase2c_visible_failure_sheet.jpg",
            "No visible_failure_candidate rows. Inspect semantic top sheet for weak headphone-shift candidates.",
        )
    counts = Counter(str(row.get("decision_label", "unknown")) for row in rows)
    part_counts = Counter(str(row.get("phase2c_part", "unknown")) for row in rows)
    conclusion = _phase2c_conclusion(rows)
    summary = {
        "candidate_count": len(rows),
        "expected_evaluations": phase2c_expected_evaluations(config),
        "decision_counts": dict(counts),
        "part_counts": dict(part_counts),
        "visible_failure_count": len(visible),
        "best_rows": ranked[: min(24, len(ranked))],
        "best_semantic_rows": semantic_ranked[: min(24, len(semantic_ranked))],
        "conclusion": conclusion,
        "updated_at": utc_now(),
    }
    write_json(output / "phase2c_summary.json", summary)

    lines = [
        "# Phase 2C targeted headphone failure probe decision report",
        "",
        conclusion,
        "",
        "## Scope",
        "",
        "- Prompt: add headphones",
        "- Perturbation type: geometric coordinate warps only",
        "- Parts: amplification, headphone-region ablations, semantic-heavy CEM",
        "- A row is not a true success unless the clean edit clearly adds headphones and the perturbed edit does not clearly add headphones.",
        "",
        "## Counts",
        "",
        f"- Total candidates: {len(rows)}",
        f"- Visible failure candidates: {counts.get('visible_failure_candidate', 0)}",
        f"- Strong semantic candidates: {counts.get('strong_candidate', 0)}",
        f"- Weak semantic candidates: {counts.get('weak_candidate', 0)}",
        f"- Metric-only candidates: {counts.get('metric_only_candidate', 0)}",
        f"- Rejected for input damage: {counts.get('reject_input_damage', 0)}",
        f"- Part counts: {dict(part_counts)}",
        "",
        "## Best semantic rows",
        "",
    ]
    for row in semantic_ranked[:12]:
        lines.append(
            "- "
            f"{row.get('phase2c_part')} / {row.get('geometry_method')} / {row.get('region_variant', row.get('region_key'))} / {row.get('budget')} "
            f"{row.get('candidate_name')}: label={row.get('decision_label')}, "
            f"score={float(row.get('phase2_final_score', 0.0)):.4f}, "
            f"semantic_drop={float(row.get('semantic_drop', 0.0)):.4f}, "
            f"perturbed_margin={float(row.get('perturbed_clip_positive_margin', 0.0)):.4f}, "
            f"input_ssim={float(row.get('input_ssim', 0.0)):.4f}, "
            f"max_disp={float(row.get('max_disp_px', 0.0)):.2f}"
        )
    lines.extend(
        [
            "",
            "## Visual inspection rule",
            "",
            "If the perturbed edit still clearly has headphones, call it weak or metric-only even if semantic_drop improved.",
        ]
    )
    write_text(output / "phase2c_decision_report.md", "\n".join(lines) + "\n")
    return ranked


def run_phase2c_probe(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase2c_config(root)
    output = outputs_root(root) / "phase2c_probe"
    if done_path(output).exists() and (output / "phase2c_all_candidates.csv").exists() and not force:
        rows = read_csv(output / "phase2c_all_candidates.csv")
        print(f"Phase 2C already complete: {len(rows)} candidates")
        return rows

    output.mkdir(parents=True, exist_ok=True)
    try:
        top_n = max(
            int(dict(config.get("amplification", {})).get("top_candidates", 15)),
            int(dict(config.get("region_ablations", {})).get("top_candidates", 5)),
            int(dict(config.get("semantic_heavy_cem", {})).get("init_top_candidates", 4)),
        )
        sources = _load_phase2b_sources(root, limit=top_n)
        if not sources:
            raise RuntimeError("No usable weak Phase 2B headphone candidates found for Phase 2C initialization.")
        selected = selected_prompt_map(root)
        prompt_info = selected[str(config.get("prompt", "add headphones"))]
        clip_scorer = load_clip_scorer(config)
        device_name = str(dict(config.get("parallel", {})).get("cuda_device", "cuda:0"))
        evaluator = SerialEvaluator(_model_payload(prompt_info), device_name)
        try:
            input_image = root / str(config.get("input_image", "data/face_001/instruct_512.png"))
            baseline_folder = output / "_baseline" / str(prompt_info["prompt_slug"])
            original_path, clean_edit_path = _prepare_clean_images(
                root=root,
                combo_folder=baseline_folder,
                prompt_info=prompt_info,
                input_image=input_image,
                pipe=evaluator.pipe,
                device=evaluator.device,
            )
            rows: list[dict[str, Any]] = []
            rows.extend(
                _run_amplification(
                    root=root,
                    output=output,
                    config=config,
                    sources=sources,
                    prompt_info=prompt_info,
                    evaluator=evaluator,
                    clip_scorer=clip_scorer,
                    original_path=original_path,
                    clean_edit_path=clean_edit_path,
                    force=force,
                )
            )
            rows.extend(
                _run_region_ablations(
                    root=root,
                    output=output,
                    config=config,
                    sources=sources,
                    prompt_info=prompt_info,
                    evaluator=evaluator,
                    clip_scorer=clip_scorer,
                    original_path=original_path,
                    clean_edit_path=clean_edit_path,
                    force=force,
                )
            )
            rows.extend(
                _run_semantic_heavy_cem(
                    root=root,
                    output=output,
                    config=config,
                    sources=sources,
                    prompt_info=prompt_info,
                    evaluator=evaluator,
                    clip_scorer=clip_scorer,
                    original_path=original_path,
                    clean_edit_path=clean_edit_path,
                    force=force,
                )
            )
        finally:
            evaluator.close()
        ranked = _write_phase2c_outputs(root, output, config, rows)
        mark_done(output, {"candidate_count": len(ranked), "conclusion": _phase2c_conclusion(ranked)})
        print(f"Phase 2C probe rows: {len(ranked)}")
        print("Inspect phase2/outputs/phase2c_probe/phase2c_semantic_top_sheet.jpg")
        return ranked
    except Exception as error:
        mark_failed(output, error)
        write_text(output / "phase2c_failure_traceback.txt", traceback.format_exc())
        raise


def summarize_phase2c(root_value: str | Path) -> dict[str, Any]:
    root = project_root(root_value)
    output = outputs_root(root) / "phase2c_probe"
    summary_dir = outputs_root(root) / "summaries"
    phase2a = read_json(outputs_root(root) / "phase2a_probe" / "phase2a_summary.json", {})
    phase2b = read_json(outputs_root(root) / "phase2b_cem" / "phase2b_summary.json", {})
    phase2c = read_json(output / "phase2c_summary.json", {})
    config = load_phase2c_config(root)
    if not phase2c:
        phase2c = {
            "candidate_count": 0,
            "expected_evaluations": phase2c_expected_evaluations(config),
            "decision_counts": {},
            "conclusion": "Phase 2C has not completed yet.",
        }
    payload = {
        "phase2a_decision_counts": phase2a.get("decision_counts", {}),
        "phase2b_decision_counts": phase2b.get("decision_counts", {}),
        "phase2c_decision_counts": phase2c.get("decision_counts", {}),
        "phase2c_candidate_count": phase2c.get("candidate_count", 0),
        "phase2c_expected_evaluations": phase2c.get("expected_evaluations", phase2c_expected_evaluations(config)),
        "phase2c_conclusion": phase2c.get("conclusion"),
        "updated_at": utc_now(),
    }
    lines = [
        "# Phase 2C targeted headphone failure probe report",
        "",
        str(phase2c.get("conclusion", "Phase 2C has not completed yet.")),
        "",
        "## Handoff from Phase 2B",
        "",
        "Phase 2B improved the weak `add headphones` signal, but the perturbed edits still visibly retained headphones. It is preserved as diagnostic data and should not be treated as a strong attack success.",
        "",
        "## Counts",
        "",
        f"- Phase 2A decision counts: {phase2a.get('decision_counts', {})}",
        f"- Phase 2B decision counts: {phase2b.get('decision_counts', {})}",
        f"- Phase 2C decision counts: {phase2c.get('decision_counts', {})}",
        f"- Phase 2C part counts: {phase2c.get('part_counts', {})}",
        f"- Phase 2C expected evaluations: {phase2c.get('expected_evaluations', phase2c_expected_evaluations(config))}",
        "",
        "## Files to inspect",
        "",
        "- `phase2/outputs/phase2c_probe/phase2c_visible_failure_sheet.jpg`",
        "- `phase2/outputs/phase2c_probe/phase2c_semantic_top_sheet.jpg`",
        "- `phase2/outputs/phase2c_probe/phase2c_decision_report.md`",
        "",
        "## Decision rule",
        "",
        "If the perturbed edit still clearly has headphones, call it weak or metric-only even if semantic_drop improved.",
    ]
    write_json(summary_dir / "phase2c_final_summary.json", payload)
    write_text(summary_dir / "phase2c_final_report.md", "\n".join(lines) + "\n")

    phase2_final = [
        "# Phase 2 final-edit geometric CEM report",
        "",
        "## Phase 1 handoff",
        "",
        "Phase 1A/1B/1C are preserved as diagnostic internal-objective results. The current conclusion is not to run more Phase 1C/1D with the same objective family.",
        "",
        "## Phase 2A probe",
        "",
        f"- Decision counts: {phase2a.get('decision_counts', {})}" if phase2a else "- Phase 2A has not been run yet.",
        "",
        "## Phase 2B",
        "",
        f"- Decision counts: {phase2b.get('decision_counts', {})}" if phase2b else "- Phase 2B has not been run yet.",
        "- Visual verdict: weak headphone weakening/shift only, not a convincing visible edit failure.",
        "",
        "## Phase 2C",
        "",
        str(phase2c.get("conclusion", "Phase 2C has not completed yet.")),
        f"- Decision counts: {phase2c.get('decision_counts', {})}",
        "",
        "## Interpretation rule",
        "",
        "If the final clean and perturbed edited images look the same, treat the row as metric-only even if the CSV score is high.",
    ]
    write_text(summary_dir / "phase2_final_report.md", "\n".join(phase2_final) + "\n")
    mark_done(summary_dir, {"phase2c_summary": True})
    print("Wrote phase2/outputs/summaries/phase2c_final_report.md")
    return payload


__all__ = ["phase2c_expected_evaluations", "run_phase2c_probe", "summarize_phase2c"]
