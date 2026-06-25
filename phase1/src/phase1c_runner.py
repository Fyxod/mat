"""Phase 1C/1D final-edit-aligned geometric white-box workflow."""
from __future__ import annotations

import csv
import dataclasses
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .config import AttackSettings
from .data import load_rgb, tensor_to_pil
from .geometry import displacement_stats, flow_to_image, jacobian_penalty
from .identity import identity_panel
from .instruct_pipeline import (
    assert_whitebox_contract,
    generate_edit,
    load_instruct_pix2pix,
    prepare_internal_reference,
)
from .metrics import attack_score, image_metrics
from .masks import resolved_face_mask
from .optimizer import optimize_geometry
from .parallel_jobs import run_serial_or_parallel
from .phase1c_objectives import PHASE1C_OBJECTIVES, prepare_multi_timestep_reference
from .reporting import attack_sheet, copy_if_exists, image_sheet, save_rgb, write_csv, write_json
from .runners import _device, _model_settings, _original, _settings_for_selection
from .semantic_scoring import ClipSemanticScorer, diagnose_clip_load, score_final_edit_case
from .utils import (
    append_run_note,
    is_complete,
    mark_done,
    mark_failed,
    outputs_root,
    read_json,
    relative_path,
    require_selected_prompts,
)


def _load_phase1c_config(root: Path) -> dict[str, Any]:
    return read_json(root / "phase1" / "configs" / "phase1c_final_edit_aligned.json")


def _load_parallel_config(root: Path) -> dict[str, Any]:
    return read_json(root / "phase1" / "configs" / "phase1c_parallel.json", {})


def _load_semantic_config(root: Path) -> dict[str, Any]:
    return read_json(root / "phase1" / "configs" / "phase1c_semantic_scoring.json", {})


def _semantic_scorer(root: Path) -> ClipSemanticScorer | None:
    semantic = _load_semantic_config(root)
    if not bool(semantic.get("clip_enabled", True)):
        return None
    diagnostics: dict[str, Any] = {}
    scorer = ClipSemanticScorer.load_optional(
        model_id=str(semantic.get("clip_model_id", "openai/clip-vit-base-patch32")),
        device=semantic.get("clip_device"),
        use_safetensors=bool(semantic.get("clip_use_safetensors", True)),
        diagnostics=diagnostics,
    )
    if scorer is None:
        write_json(outputs_root(root) / "summaries" / "clip_semantic_load_failure.json", diagnostics)
        error = diagnostics.get("error", "unknown error")
        append_run_note(
            root,
            "Phase 1C semantic scoring",
            f"CLIP could not be loaded ({error}); semantic scoring will mark rows as metric-only fallback instead of strong.",
        )
    return scorer


def check_clip_semantic_preflight(root: Path, *, require_available: bool = False) -> dict[str, Any]:
    """Diagnose CLIP scoring and optionally fail before expensive Phase 1C work."""
    semantic = _load_semantic_config(root)
    output = outputs_root(root) / "summaries" / "clip_semantic_diagnostics.json"
    if not bool(semantic.get("clip_enabled", True)):
        payload = {"available": False, "disabled": True, "reason": "clip_enabled=false"}
        write_json(output, payload)
        if require_available:
            raise RuntimeError("CLIP semantic scoring is disabled in phase1c_semantic_scoring.json.")
        return payload
    payload = diagnose_clip_load(
        model_id=str(semantic.get("clip_model_id", "openai/clip-vit-base-patch32")),
        device=semantic.get("clip_device"),
        use_safetensors=bool(semantic.get("clip_use_safetensors", True)),
    )
    write_json(output, payload)
    if payload.get("available"):
        append_run_note(
            root,
            "Phase 1C semantic preflight",
            f"CLIP semantic scoring is available with {payload.get('model_id')} on {payload.get('resolved_device')}.",
        )
    else:
        append_run_note(
            root,
            "Phase 1C semantic preflight",
            f"CLIP semantic scoring is unavailable: {payload.get('error', 'unknown error')}. Diagnostics: phase1/outputs/summaries/clip_semantic_diagnostics.json",
        )
        if require_available:
            raise RuntimeError(
                "CLIP semantic scoring is unavailable; refusing to run expensive Phase 1C screening. "
                "Inspect phase1/outputs/summaries/clip_semantic_diagnostics.json, fix the environment/cache, "
                "then rerun the CLIP check."
            )
    return payload


def _phase1c_attack_settings(
    config: dict[str, Any],
    objective: str,
    budget: dict[str, Any],
    start_seed_index: int,
    *,
    iterations: int | None = None,
    learning_rate: float | None = None,
    checkpoint_every: int | None = None,
) -> AttackSettings:
    return AttackSettings(
        objective=objective,
        max_disp_px=float(budget["max_disp_px"]),
        target_input_ssim=float(budget["target_input_ssim"]),
        iterations=int(iterations if iterations is not None else config["iterations"]),
        learning_rate=float(learning_rate if learning_rate is not None else config["learning_rate"]),
        objective_scale=float(config.get("objective_scale", 1.0)),
        objective_gradient_balance=float(config.get("objective_gradient_balance", 1.25)),
        objective_scale_min=float(config.get("objective_scale_min", 1e-6)),
        objective_scale_max=float(config.get("objective_scale_max", 1e8)),
        lambda_visual=float(config["lambda_visual"]),
        lambda_disp=float(config["lambda_disp"]),
        lambda_smooth=float(config["lambda_smooth"]),
        lambda_fold=float(config["lambda_fold"]),
        checkpoint_every=int(checkpoint_every if checkpoint_every is not None else config["checkpoint_every"]),
        objective_timestep_index=int(config.get("objective_timestep_index", 10)),
        objective_timestep_indices=tuple(int(item) for item in config.get("objective_timestep_indices", [3, 6, 10, 14, 18])),
        geometry_method=str(config["geometry_method"]),
        dct_size=int(config["dct_size"]),
        tps_size=int(config["tps_size"]),
        face_mask=dict(config["face_mask"]),
        seed=1234 + start_seed_index * 1009,
    )


def _prepare_reference(pipe, original: Image.Image, selection: dict[str, Any], attack: AttackSettings, root: Path):
    model_settings = _settings_for_selection(root, selection)
    if attack.objective in PHASE1C_OBJECTIVES:
        return prepare_multi_timestep_reference(
            pipe,
            original,
            selection["prompt"],
            model_settings,
            _device(),
            attack.objective_timestep_indices,
        )
    return prepare_internal_reference(
        pipe,
        original,
        selection["prompt"],
        model_settings,
        _device(),
        attack.objective_timestep_index,
    )


def _attack_output_record(
    root: Path,
    selection: dict[str, Any],
    attack: AttackSettings,
    start: int,
    iteration: int,
    input_values: dict[str, float],
    output_values: dict[str, float],
    displacement: dict[str, float],
    old_score: dict[str, float],
    semantic: dict[str, Any],
    folder: Path,
    budget_name: str,
    combo_id: str,
) -> dict[str, Any]:
    return {
        "prompt": selection["prompt"],
        "prompt_slug": selection["prompt_slug"],
        "objective": attack.objective,
        "budget": budget_name,
        "combo_id": combo_id,
        "target_input_ssim": attack.target_input_ssim,
        "start": start,
        "iter": iteration,
        "input_psnr": input_values["psnr"],
        "input_ssim": input_values["ssim"],
        "input_l2": input_values["l2"],
        "output_psnr": output_values["psnr"],
        "output_ssim": output_values["ssim"],
        "output_l2": output_values["l2"],
        **displacement,
        **old_score,
        **semantic,
        "path_original": relative_path(folder / "original.png", root),
        "path_original_edited": relative_path(folder / "original_edited.png", root),
        "path_perturbed": relative_path(folder / "perturbed.png", root),
        "path_perturbed_edited": relative_path(folder / "perturbed_edited.png", root),
        "path_flow": relative_path(folder / "flow.png", root),
    }


def _run_phase1c_attack_start(
    root: Path,
    pipe,
    clip_scorer: ClipSemanticScorer | None,
    original: Image.Image,
    clean_edit: Image.Image,
    selection: dict[str, Any],
    attack: AttackSettings,
    start: int,
    folder: Path,
    force: bool,
    budget_name: str,
    combo_id: str,
    fallback_objective_scale: float | None,
    objective_timestep_indices_fallback: list[int] | tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = folder / "start_result.json"
    if is_complete(folder) and result_path.exists() and not force:
        payload = read_json(result_path)
        if clip_scorer is not None and any(not bool(row.get("clip_available", False)) for row in payload.get("checkpoint_rows", [])):
            payload = _rescore_cached_phase1c_start(root, folder, clip_scorer)
        return payload["best_row"], payload["checkpoint_rows"]

    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        face_mask, face_mask_source = resolved_face_mask(original)
        attack = dataclasses.replace(attack, face_mask=face_mask)
        model_settings = _settings_for_selection(root, selection)
        try:
            original_tensor, reference = _prepare_reference(pipe, original, selection, attack, root)
            optimization = optimize_geometry(pipe, original_tensor, reference, attack)
        except RuntimeError as error:
            fallback_indices = tuple(int(item) for item in (objective_timestep_indices_fallback or ()))
            current_indices = tuple(attack.objective_timestep_indices or ())
            if (
                attack.objective in PHASE1C_OBJECTIVES
                and fallback_indices
                and fallback_indices != current_indices
                and "out of memory" in str(error).lower()
            ):
                torch.cuda.empty_cache()
                append_run_note(
                    root,
                    "Phase 1C timestep fallback",
                    f"{combo_id}/start_{start:02d} hit CUDA OOM with timesteps {current_indices}; retrying {fallback_indices}.",
                )
                attack = dataclasses.replace(attack, objective_timestep_indices=fallback_indices)
                original_tensor, reference = _prepare_reference(pipe, original, selection, attack, root)
                optimization = optimize_geometry(pipe, original_tensor, reference, attack)
            else:
                raise
        retry_used = False
        if (
            fallback_objective_scale is not None
            and optimization.history
            and optimization.history[-1].get("objective", 0.0)
            <= optimization.history[0].get("objective", 0.0) * 1.01
        ):
            retry_used = True
            attack = dataclasses.replace(attack, objective_scale=fallback_objective_scale)
            optimization = optimize_geometry(pipe, original_tensor, reference, attack)

        save_rgb(folder / "original.png", original)
        save_rgb(folder / "original_edited.png", clean_edit)
        write_json(
            folder / "config_resolved.json",
            {
                "selection": selection,
                "attack": attack.payload(),
                "budget_name": budget_name,
                "combo_id": combo_id,
                "face_mask_source": face_mask_source,
                "model": model_settings.payload(),
                "whitebox_contract": assert_whitebox_contract(pipe),
                "objective_scale_fallback_used": retry_used,
                "effective_objective_scale": optimization.effective_objective_scale,
                "initial_objective_grad_norm": optimization.initial_objective_grad_norm,
                "initial_regularizer_grad_norm": optimization.initial_regularizer_grad_norm,
                "objective_gradient_probe_multiplier": optimization.objective_gradient_probe_multiplier,
                "semantic_clip_available": clip_scorer is not None,
            },
        )
        write_csv(folder / "history.csv", optimization.history)

        checkpoint_rows: list[dict[str, Any]] = []
        for iteration, snapshot in sorted(optimization.snapshots.items()):
            checkpoint = folder / "checkpoints" / f"iter_{iteration:03d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            perturbed = tensor_to_pil(snapshot.image)
            flow = flow_to_image(snapshot.displacement)
            perturbed_edit = generate_edit(pipe, perturbed, selection["prompt"], model_settings, _device())
            save_rgb(checkpoint / "original.png", original)
            save_rgb(checkpoint / "original_edited.png", clean_edit)
            save_rgb(checkpoint / "perturbed.png", perturbed)
            save_rgb(checkpoint / "perturbed_edited.png", perturbed_edit)
            save_rgb(checkpoint / "flow.png", flow)
            input_values = image_metrics(original, perturbed)
            output_values = image_metrics(clean_edit, perturbed_edit)
            _, jacobian = jacobian_penalty(snapshot.displacement)
            displacement = {**displacement_stats(snapshot.displacement), **jacobian}
            old_score = attack_score(
                input_values, output_values, displacement, attack.target_input_ssim, attack.max_disp_px
            )
            semantic = score_final_edit_case(
                selection["prompt"],
                original,
                clean_edit,
                perturbed,
                perturbed_edit,
                input_values,
                output_values,
                {
                    **displacement,
                    "target_input_ssim": attack.target_input_ssim,
                    "max_disp_px_budget": attack.max_disp_px,
                },
                optional_clip_model=clip_scorer,
            )
            metrics = {
                "input": input_values,
                "output": output_values,
                "displacement": displacement,
            }
            row = _attack_output_record(
                root,
                selection,
                attack,
                start,
                iteration,
                input_values,
                output_values,
                displacement,
                old_score,
                semantic,
                checkpoint,
                budget_name,
                combo_id,
            )
            write_json(checkpoint / "metrics.json", metrics)
            write_json(checkpoint / "score.json", old_score)
            write_json(checkpoint / "semantic_score.json", semantic)
            checkpoint_rows.append(row)

        optimized_rows = [row for row in checkpoint_rows if int(row["iter"]) > 0]
        budget_admissible_rows = [
            row for row in optimized_rows
            if float(row["input_ssim"]) >= attack.target_input_ssim
            and float(row["max_disp_px"]) <= attack.max_disp_px + 1e-4
            and float(row["foldover_fraction_det_below_0"]) == 0.0
        ]
        candidate_pool = budget_admissible_rows or optimized_rows or checkpoint_rows
        best_row = max(candidate_pool, key=lambda row: float(row.get("final_attack_score", row["attack_score"])))
        best_checkpoint = root / best_row["path_perturbed"].replace("perturbed.png", "")
        best_folder = folder / "best"
        best_folder.mkdir(parents=True, exist_ok=True)
        for filename in (
            "original.png",
            "original_edited.png",
            "perturbed.png",
            "perturbed_edited.png",
            "flow.png",
            "metrics.json",
            "score.json",
            "semantic_score.json",
        ):
            copy_if_exists(best_checkpoint / filename, best_folder / filename)
        best_snapshot = optimization.snapshots[int(best_row["iter"])]
        torch.save(
            {"state_dict": best_snapshot.state_dict, "iteration": best_snapshot.iteration},
            best_folder / "theta.pt",
        )
        attack_sheet(
            best_folder / "sheet.jpg",
            load_rgb(best_folder / "original.png"),
            load_rgb(best_folder / "original_edited.png"),
            load_rgb(best_folder / "perturbed.png"),
            load_rgb(best_folder / "perturbed_edited.png"),
            load_rgb(best_folder / "flow.png"),
            f"final={float(best_row['final_attack_score']):.3f}, {best_row['decision_label']}",
        )
        initial_row = next((row for row in checkpoint_rows if int(row["iter"]) == 0), checkpoint_rows[0])
        best_row = {
            **best_row,
            "elapsed_seconds": time.monotonic() - started,
            "best_folder": relative_path(best_folder, root),
            "objective_scale_used": optimization.effective_objective_scale,
            "initial_attack_score": initial_row["attack_score"],
            "attack_score_gain_from_initial": float(best_row["attack_score"]) - float(initial_row["attack_score"]),
            "initial_final_attack_score": initial_row.get("final_attack_score", 0.0),
            "final_attack_score_gain_from_initial": float(best_row["final_attack_score"]) - float(initial_row.get("final_attack_score", 0.0)),
            "budget_admissible": bool(budget_admissible_rows),
        }
        write_csv(folder / "checkpoint_rows.csv", checkpoint_rows)
        write_json(result_path, {"best_row": best_row, "checkpoint_rows": checkpoint_rows})
        mark_done(folder, {"best_final_attack_score": best_row["final_attack_score"], "best_iter": best_row["iter"]})
        return best_row, checkpoint_rows
    except Exception as error:
        mark_failed(folder, error)
        raise


def _rescore_cached_phase1c_start(root: Path, folder: Path, clip_scorer: ClipSemanticScorer) -> dict[str, Any]:
    """Recompute semantic scores for a completed Phase 1C start from saved images."""
    result_path = folder / "start_result.json"
    payload = read_json(result_path)
    checkpoint_rows = payload.get("checkpoint_rows", [])
    rescored_rows: list[dict[str, Any]] = []
    for row in checkpoint_rows:
        paths = {
            "original": root / row["path_original"],
            "original_edited": root / row["path_original_edited"],
            "perturbed": root / row["path_perturbed"],
            "perturbed_edited": root / row["path_perturbed_edited"],
        }
        if not all(path.exists() for path in paths.values()):
            rescored_rows.append(row)
            continue
        original = load_rgb(paths["original"])
        clean_edit = load_rgb(paths["original_edited"])
        perturbed = load_rgb(paths["perturbed"])
        perturbed_edit = load_rgb(paths["perturbed_edited"])
        input_values = image_metrics(original, perturbed)
        output_values = image_metrics(clean_edit, perturbed_edit)
        displacement = {
            key: row[key]
            for key in (
                "max_disp_px",
                "mean_disp_px",
                "p95_disp_px",
                "foldover_fraction_det_below_0",
                "jacobian_det_mean",
                "jacobian_det_min",
                "low_det_fraction_below_0p2",
                "target_input_ssim",
                "max_disp_px_budget",
            )
            if key in row
        }
        if "max_disp_px_budget" not in displacement:
            displacement["max_disp_px_budget"] = 6.0 if str(row.get("budget", "")).lower() == "strong" else 4.0
        if "target_input_ssim" not in displacement:
            displacement["target_input_ssim"] = row.get("target_input_ssim", 0.90)
        semantic = score_final_edit_case(
            row.get("prompt", row.get("prompt_slug", "")),
            original,
            clean_edit,
            perturbed,
            perturbed_edit,
            input_values,
            output_values,
            displacement,
            optional_clip_model=clip_scorer,
        )
        updated = {**row, **semantic}
        write_json((root / row["path_perturbed"]).parent / "semantic_score.json", semantic)
        rescored_rows.append(updated)

    optimized_rows = [row for row in rescored_rows if int(row.get("iter", 0)) > 0]
    budget_rows = [
        row for row in optimized_rows
        if float(row.get("input_ssim", 0.0)) >= float(row.get("target_input_ssim", 0.90))
        and float(row.get("max_disp_px", 999.0)) <= float(row.get("max_disp_px_budget", 999.0)) + 1e-4
        and float(row.get("foldover_fraction_det_below_0", 0.0)) == 0.0
    ]
    pool = budget_rows or optimized_rows or rescored_rows
    best_row = max(pool, key=lambda item: float(item.get("final_attack_score", item.get("attack_score", 0.0))))
    initial_row = next((row for row in rescored_rows if int(row.get("iter", 0)) == 0), rescored_rows[0])
    best_folder = folder / "best"
    best_folder.mkdir(parents=True, exist_ok=True)
    best_checkpoint = (root / best_row["path_perturbed"]).parent
    for filename in (
        "original.png",
        "original_edited.png",
        "perturbed.png",
        "perturbed_edited.png",
        "flow.png",
        "metrics.json",
        "score.json",
        "semantic_score.json",
    ):
        copy_if_exists(best_checkpoint / filename, best_folder / filename)
    attack_sheet(
        best_folder / "sheet.jpg",
        load_rgb(best_folder / "original.png"),
        load_rgb(best_folder / "original_edited.png"),
        load_rgb(best_folder / "perturbed.png"),
        load_rgb(best_folder / "perturbed_edited.png"),
        load_rgb(best_folder / "flow.png"),
        f"final={float(best_row.get('final_attack_score', 0.0)):.3f}, {best_row.get('decision_label', '')}",
    )
    best_row = {
        **best_row,
        "best_folder": relative_path(best_folder, root),
        "initial_final_attack_score": initial_row.get("final_attack_score", 0.0),
        "final_attack_score_gain_from_initial": float(best_row.get("final_attack_score", 0.0)) - float(initial_row.get("final_attack_score", 0.0)),
        "budget_admissible": bool(budget_rows),
        "semantic_rescored_from_cache": True,
    }
    write_csv(folder / "checkpoint_rows.csv", rescored_rows)
    new_payload = {"best_row": best_row, "checkpoint_rows": rescored_rows}
    write_json(result_path, new_payload)
    return new_payload


def _selected_phase1c_prompts(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = set(config.get("prompts", []))
    selected = require_selected_prompts(root)
    if wanted:
        selected = [item for item in selected if item["prompt_slug"] in wanted or item["prompt"] in wanted]
    if not selected:
        raise RuntimeError("Phase 1C selected no prompts. Check phase1c_final_edit_aligned.json.")
    return selected


def _make_screening_jobs(root: Path, output: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    selected = _selected_phase1c_prompts(root, config)
    jobs: list[dict[str, Any]] = []
    for selection in selected:
        for objective in config["objectives"]:
            for budget in config["budgets"]:
                for start in range(int(config["starts"])):
                    combo_id = f"{selection['prompt_slug']}__{objective}__{budget['name']}"
                    folder = output / selection["prompt_slug"] / objective / budget["name"] / f"start_{start:02d}"
                    jobs.append(
                        {
                            "selection": selection,
                            "objective": objective,
                            "budget": budget,
                            "start": start,
                            "start_seed_index": start + 100 * len(jobs),
                            "folder": relative_path(folder, root),
                            "budget_name": budget["name"],
                            "combo_id": combo_id,
                            "config": config,
                        }
                    )
    return jobs


def _make_phase1d_jobs(root: Path, output: Path, config: dict[str, Any], combinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {item["prompt_slug"]: item for item in require_selected_prompts(root)}
    jobs: list[dict[str, Any]] = []
    for combo_index, combo in enumerate(combinations):
        budget = {"name": combo["budget"], "max_disp_px": combo["max_disp_px_budget"], "target_input_ssim": combo["target_input_ssim"]}
        selection = selected[combo["prompt_slug"]]
        for start in range(int(config["phase1d"]["starts"])):
            combo_id = f"{combo_index + 1:02d}_{selection['prompt_slug']}__{combo['objective']}__{combo['budget']}"
            folder = output / combo_id / f"start_{start:02d}"
            jobs.append(
                {
                    "selection": selection,
                    "objective": combo["objective"],
                    "budget": budget,
                    "start": start,
                    "start_seed_index": start + 1000 * (combo_index + 1),
                    "folder": relative_path(folder, root),
                    "budget_name": combo["budget"],
                    "combo_id": combo_id,
                    "config": {
                        **config,
                        "iterations": int(config["phase1d"]["iterations"]),
                        "learning_rate": float(config["phase1d"]["learning_rate"]),
                        "checkpoint_every": int(config["phase1d"]["checkpoint_every"]),
                    },
                }
            )
    return jobs


def _phase1c_worker(root_text: str, jobs: list[dict[str, Any]], force: bool) -> list[dict[str, Any]]:
    root = Path(root_text)
    original = _original(root)
    pipe = load_instruct_pix2pix(_model_settings(root), _device())
    clip_scorer = _semantic_scorer(root)
    baseline = {
        row["prompt_slug"]: row
        for row in read_json(outputs_root(root) / "baselines" / "baseline_summary.json", {}).get("baselines", [])
    }
    rows: list[dict[str, Any]] = []
    for job in jobs:
        selection = job["selection"]
        if selection["prompt_slug"] not in baseline:
            raise RuntimeError(f"Missing clean baseline for {selection['prompt_slug']}. Run baselines first.")
        clean_edit = load_rgb(root / baseline[selection["prompt_slug"]]["path_original_edited"])
        config = job["config"]
        attack = _phase1c_attack_settings(
            config,
            job["objective"],
            job["budget"],
            int(job["start_seed_index"]),
        )
        best, _ = _run_phase1c_attack_start(
            root,
            pipe,
            clip_scorer,
            original,
            clean_edit,
            selection,
            attack,
            int(job["start"]),
            root / job["folder"],
            force,
            job["budget_name"],
            job["combo_id"],
            fallback_objective_scale=float(config.get("fallback_objective_scale", 1.0)),
            objective_timestep_indices_fallback=config.get("objective_timestep_indices_fallback"),
        )
        rows.append(best)
    return rows


def _read_completed_rows(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for result_path in output.rglob("start_result.json"):
        payload = read_json(result_path, {})
        if payload.get("best_row"):
            best_rows.append(payload["best_row"])
        checkpoint_rows.extend(payload.get("checkpoint_rows", []))
    return best_rows, checkpoint_rows


def _top_sheet(root: Path, rows: list[dict[str, Any]], destination: Path, score_key: str = "final_attack_score") -> None:
    sheet_rows: list[tuple[list[str], list[Image.Image]]] = []
    for row in rows[:8]:
        best = root / row.get("best_folder", "")
        required = [best / name for name in ("original.png", "original_edited.png", "perturbed.png", "perturbed_edited.png", "flow.png")]
        if not all(path.exists() for path in required):
            continue
        label_score = float(row.get(score_key, row.get("attack_score", 0.0)))
        sheet_rows.append((
            [
                f"Original\n{row['prompt_slug']}",
                "Clean edit",
                f"Perturbed\n{row['objective']} / {row['budget']}",
                f"Perturbed edit\n{score_key}={label_score:.3f}\n{row.get('decision_label', '')}",
                f"Flow\ninput SSIM={float(row['input_ssim']):.3f}",
            ],
            [load_rgb(path) for path in required],
        ))
    if sheet_rows:
        image_sheet(destination, sheet_rows, columns=5, cell_width=192, cell_height=192)


def _decision_report(stage_name: str, rows: list[dict[str, Any]]) -> str:
    labels = ["strong_candidate", "weak_candidate", "metric_only_candidate", "reject_input_damage", "reject_clean_failed"]
    counts = {label: 0 for label in labels}
    for row in rows:
        label = str(row.get("decision_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    lines = [
        f"# {stage_name} decision report",
        "",
        f"- Completed starts: {len(rows)}",
        f"- Strong semantic candidates: {counts.get('strong_candidate', 0)}",
        f"- Weak semantic candidates: {counts.get('weak_candidate', 0)}",
        f"- Metric-only candidates: {counts.get('metric_only_candidate', 0)}",
        f"- Rejected for input damage: {counts.get('reject_input_damage', 0)}",
        f"- Rejected because clean edit was weak: {counts.get('reject_clean_failed', 0)}",
        "",
        "Rows are ranked by `final_attack_score` for semantic decisions. The old `attack_score` is retained only as an output-disruption diagnostic.",
        "",
    ]
    for label in labels:
        subset = [row for row in rows if row.get("decision_label") == label]
        if not subset:
            continue
        lines.extend([f"## {label}", ""])
        for row in subset[:8]:
            lines.append(
                f"- {row['prompt_slug']} / {row['objective']} / {row['budget']} "
                f"start {row.get('start')} iter {row.get('iter')}: "
                f"final={float(row.get('final_attack_score', 0.0)):.4f}, "
                f"semantic_drop={float(row.get('semantic_drop', 0.0)):.4f}, "
                f"input_ssim={float(row.get('input_ssim', 0.0)):.4f}, "
                f"old_attack={float(row.get('attack_score', 0.0)):.4f}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _aggregate_outputs(root: Path, output: Path, prefix: str) -> list[dict[str, Any]]:
    best_rows, checkpoint_rows = _read_completed_rows(output)
    old_ranked = sorted(best_rows, key=lambda row: float(row.get("attack_score", 0.0)), reverse=True)
    semantic_ranked = sorted(best_rows, key=lambda row: float(row.get("final_attack_score", row.get("attack_score", 0.0))), reverse=True)
    checkpoint_ranked = sorted(
        checkpoint_rows,
        key=lambda row: float(row.get("final_attack_score", row.get("attack_score", 0.0))),
        reverse=True,
    )
    write_csv(output / f"{prefix}_all_candidates.csv", semantic_ranked)
    write_csv(output / f"{prefix}_all_checkpoints.csv", checkpoint_ranked)
    write_csv(output / f"{prefix}_top_candidates.csv", old_ranked[: min(24, len(old_ranked))])
    write_csv(output / f"{prefix}_semantic_top_candidates.csv", semantic_ranked[: min(24, len(semantic_ranked))])
    write_json(
        output / f"{prefix}_summary.json",
        {
            "best_rows": semantic_ranked,
            "checkpoint_count": len(checkpoint_rows),
            "decision_counts": _decision_counts(best_rows),
        },
    )
    _top_sheet(root, old_ranked, output / f"{prefix}_top_sheet.jpg", score_key="attack_score")
    _top_sheet(root, semantic_ranked, output / f"{prefix}_semantic_top_sheet.jpg", score_key="final_attack_score")
    (output / f"{prefix}_decision_report.md").write_text(_decision_report(prefix.upper(), semantic_ranked), encoding="utf-8")
    return semantic_ranked


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("decision_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def run_phase1c_parallel_smoke(root: Path, force: bool = False) -> list[dict[str, Any]]:
    output = outputs_root(root) / "phase1c_parallel_smoke"
    if is_complete(output) and not force:
        return read_json(output / "phase1c_parallel_smoke_summary.json", {}).get("best_rows", [])
    config = _load_phase1c_config(root)
    smoke_config = {
        **config,
        "prompts": [config["prompts"][0]],
        "objectives": config["objectives"][:2],
        "budgets": [config["budgets"][0]],
        "starts": 1,
        "iterations": 5,
        "checkpoint_every": 5,
    }
    parallel = {**_load_parallel_config(root), "parallel_experimental": True, "parallel_workers": 2}
    jobs = _make_screening_jobs(root, output, smoke_config)[:2]
    rows = run_serial_or_parallel(
        root=root,
        jobs=jobs,
        worker_entry=_phase1c_worker,
        force=force,
        parallel_config=parallel,
        notes_title="Phase 1C parallel smoke",
    )
    ranked = _aggregate_outputs(root, output, "phase1c_parallel_smoke")
    mark_done(output, {"starts": len(rows), "ranked": len(ranked)})
    return ranked


def run_phase1c_screening(root: Path, force: bool = False) -> list[dict[str, Any]]:
    output = outputs_root(root) / "phase1c_screening"
    if is_complete(output) and not force:
        return read_json(output / "phase1c_summary.json", {}).get("best_rows", [])
    config = _load_phase1c_config(root)
    parallel = _load_parallel_config(root)
    semantic = _load_semantic_config(root)
    if bool(semantic.get("require_clip_for_phase1c", True)):
        check_clip_semantic_preflight(root, require_available=True)
    if bool(parallel.get("parallel_experimental", False)):
        run_phase1c_parallel_smoke(root, force=force)
    jobs = _make_screening_jobs(root, output, config)
    rows = run_serial_or_parallel(
        root=root,
        jobs=jobs,
        worker_entry=_phase1c_worker,
        force=force,
        parallel_config=parallel,
        notes_title="Phase 1C screening",
    )
    ranked = _aggregate_outputs(root, output, "phase1c")
    mark_done(output, {"starts": len(rows), "ranked": len(ranked)})
    return ranked


def _unique_phase1d_combinations(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        label = row.get("decision_label")
        semantic_drop = float(row.get("semantic_drop", 0.0))
        if label != "strong_candidate" and not (label == "weak_candidate" and semantic_drop >= 0.03):
            continue
        key = (row["prompt_slug"], row["objective"], row["budget"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
        if len(output) >= limit:
            break
    return output


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_phase1d_deepen(root: Path, force: bool = False) -> list[dict[str, Any]]:
    output = outputs_root(root) / "phase1d_deepen"
    if is_complete(output) and not force:
        return read_json(output / "phase1d_summary.json", {}).get("best_rows", [])
    config = _load_phase1c_config(root)
    phase1c_rows = _read_csv(outputs_root(root) / "phase1c_screening" / "phase1c_semantic_top_candidates.csv")
    if not phase1c_rows:
        output.mkdir(parents=True, exist_ok=True)
        (output / "phase1d_decision_report.md").write_text(
            "# Phase 1D decision report\n\nPhase 1C semantic candidates are missing. Run Phase 1C first.\n",
            encoding="utf-8",
        )
        mark_done(output, {"status": "skipped", "reason": "missing_phase1c"})
        return []
    combinations = _unique_phase1d_combinations(phase1c_rows, int(config["phase1d"]["top_combinations"]))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "phase1d_selected_combinations.json", {"combinations": combinations})
    if len(combinations) < 2:
        (output / "phase1d_decision_report.md").write_text(
            "# Phase 1D decision report\n\n"
            "Phase 1C did not find at least two strong candidates, or weak candidates with high semantic drop. "
            "Deepening is skipped to avoid wasting A6000 time on metric-only failures.\n",
            encoding="utf-8",
        )
        write_json(output / "phase1d_summary.json", {"best_rows": [], "checkpoint_count": 0, "skipped": True})
        mark_done(output, {"status": "skipped", "reason": "insufficient_semantic_candidates"})
        return []

    jobs = _make_phase1d_jobs(root, output, config, combinations)
    parallel = _load_parallel_config(root)
    rows = run_serial_or_parallel(
        root=root,
        jobs=jobs,
        worker_entry=_phase1c_worker,
        force=force,
        parallel_config=parallel,
        notes_title="Phase 1D deepening",
    )
    ranked = _aggregate_outputs(root, output, "phase1d")
    _write_final_validation_phase1c(root, ranked)
    mark_done(output, {"starts": len(rows), "ranked": len(ranked)})
    return ranked


def _write_final_validation_phase1c(root: Path, candidates: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    output = outputs_root(root) / "final_validation_phase1c"
    output.mkdir(parents=True, exist_ok=True)
    strong = [row for row in candidates if row.get("decision_label") == "strong_candidate"]
    if not strong:
        write_json(output / "final_validation_phase1c_summary.json", {"rows": [], "skipped": True})
        (output / "final_report.md").write_text(
            "# Final validation Phase 1C\n\nNo strong Phase 1D candidates were available for identity validation.\n",
            encoding="utf-8",
        )
        mark_done(output, {"candidate_count": 0, "status": "skipped"})
        return []

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(strong[:limit], 1):
        target = output / f"candidate_{index:03d}"
        target.mkdir(parents=True, exist_ok=True)
        source = root / candidate["best_folder"]
        for filename in ("original.png", "original_edited.png", "perturbed.png", "perturbed_edited.png", "flow.png", "metrics.json", "semantic_score.json"):
            copy_if_exists(source / filename, target / filename)
        paths = {name: target / f"{name}.png" for name in ("original", "original_edited", "perturbed", "perturbed_edited")}
        panel = identity_panel(paths, models=("SFace", "Facenet512", "ArcFace"))
        write_json(target / "identity_panel.json", panel)
        row = {
            **candidate,
            "candidate_id": f"candidate_{index:03d}",
            "identity_panel_available": panel["available"],
            "path_validation": relative_path(target, root),
        }
        write_json(target / "record.json", row)
        report = [
            f"# Final validation Phase 1C: candidate {index:03d}",
            "",
            f"- Prompt: {candidate['prompt']}",
            f"- Objective: {candidate['objective']}",
            f"- Budget: {candidate['budget']}",
            f"- Decision label: {candidate.get('decision_label')}",
            f"- Final attack score: {float(candidate.get('final_attack_score', 0.0)):.5f}",
            f"- Semantic drop: {float(candidate.get('semantic_drop', 0.0)):.5f}",
            f"- Input SSIM: {float(candidate.get('input_ssim', 0.0)):.5f}",
            f"- Identity panel available: {panel['available']}",
        ]
        (target / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        attack_sheet(
            target / "sheet.jpg",
            load_rgb(paths["original"]),
            load_rgb(paths["original_edited"]),
            load_rgb(paths["perturbed"]),
            load_rgb(paths["perturbed_edited"]),
            load_rgb(target / "flow.png"),
            f"final={float(candidate.get('final_attack_score', 0.0)):.3f}",
        )
        mark_done(target, {"candidate_id": row["candidate_id"]})
        rows.append(row)

    write_csv(output / "final_top_candidates.csv", rows)
    write_json(output / "final_validation_phase1c_summary.json", {"rows": rows})
    _top_sheet(root, rows, output / "final_top_sheet.jpg", score_key="final_attack_score")
    mark_done(output, {"candidate_count": len(rows)})
    return rows


def rescore_legacy_phase1ab(root: Path, force: bool = False) -> dict[str, Any]:
    output = outputs_root(root) / "semantic_rescore"
    output.mkdir(parents=True, exist_ok=True)
    scorer = _semantic_scorer(root)
    summary: dict[str, Any] = {}
    for phase, rel in {
        "phase1a": "phase1a_screening/phase1a_all_candidates.csv",
        "phase1b": "phase1b_deepen/phase1b_all_candidates.csv",
    }.items():
        source_csv = outputs_root(root) / rel
        rows = _read_csv(source_csv)
        rescored: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for row in rows:
            best_folder = root / row.get("best_folder", "")
            paths = {
                "original": best_folder / "original.png",
                "original_edited": best_folder / "original_edited.png",
                "perturbed": best_folder / "perturbed.png",
                "perturbed_edited": best_folder / "perturbed_edited.png",
                "flow": best_folder / "flow.png",
            }
            absent = [name for name, path in paths.items() if not path.exists()]
            if absent:
                missing.append({"best_folder": row.get("best_folder"), "missing": absent})
                continue
            input_values = image_metrics(load_rgb(paths["original"]), load_rgb(paths["perturbed"]))
            output_values = image_metrics(load_rgb(paths["original_edited"]), load_rgb(paths["perturbed_edited"]))
            displacement = {
                "max_disp_px": row.get("max_disp_px", 0.0),
                "target_input_ssim": row.get("target_input_ssim", 0.90),
                "max_disp_px_budget": 6.0 if str(row.get("budget", "")).lower() == "strong" else 4.0,
            }
            for key in ("mean_disp_px", "p95_disp_px", "foldover_fraction_det_below_0", "jacobian_det_min"):
                if key in row:
                    displacement[key] = row[key]
            semantic = score_final_edit_case(
                row.get("prompt", row.get("prompt_slug", "")),
                load_rgb(paths["original"]),
                load_rgb(paths["original_edited"]),
                load_rgb(paths["perturbed"]),
                load_rgb(paths["perturbed_edited"]),
                input_values,
                output_values,
                displacement,
                optional_clip_model=scorer,
            )
            rescored.append({**row, **semantic})
        ranked = sorted(rescored, key=lambda item: float(item.get("final_attack_score", 0.0)), reverse=True)
        write_csv(output / f"{phase}_semantic_rescore.csv", ranked)
        if ranked:
            _top_sheet(root, ranked, output / f"{phase}_semantic_top_sheet.jpg", score_key="final_attack_score")
        summary[phase] = {
            "source": relative_path(source_csv, root),
            "rows_in": len(rows),
            "rows_scored": len(ranked),
            "rows_missing_images": len(missing),
            "decision_counts": _decision_counts(ranked),
            "missing": missing[:20],
            "top": ranked[:8],
        }
    write_json(output / "semantic_rescore_summary.json", summary)
    report_lines = [
        "# Semantic rescore of legacy Phase 1A/1B",
        "",
        "This pass re-ranks legacy internal-surrogate candidates by final-edit-aware semantic scoring.",
        "If CLIP is unavailable, rows are intentionally labeled metric-only rather than strong.",
        "",
    ]
    for phase, payload in summary.items():
        report_lines.extend([
            f"## {phase}",
            "",
            f"- Rows in source CSV: {payload['rows_in']}",
            f"- Rows semantically scored: {payload['rows_scored']}",
            f"- Rows missing images: {payload['rows_missing_images']}",
            f"- Decision counts: {payload['decision_counts']}",
            "",
        ])
        for row in payload.get("top", [])[:5]:
            report_lines.append(
                f"- {row['prompt_slug']} / {row['objective']} / {row['budget']}: "
                f"{row.get('decision_label')} final={float(row.get('final_attack_score', 0.0)):.4f}, "
                f"semantic_drop={float(row.get('semantic_drop', 0.0)):.4f}"
            )
        report_lines.append("")
    (output / "semantic_rescore_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    mark_done(output, {"phases": list(summary)})
    return summary


def summarize_phase1c(root: Path) -> dict[str, Any]:
    summary_root = outputs_root(root) / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    semantic_rescore = read_json(outputs_root(root) / "semantic_rescore" / "semantic_rescore_summary.json", {})
    phase1c = read_json(outputs_root(root) / "phase1c_screening" / "phase1c_summary.json", {})
    phase1d = read_json(outputs_root(root) / "phase1d_deepen" / "phase1d_summary.json", {})
    final = read_json(outputs_root(root) / "final_validation_phase1c" / "final_validation_phase1c_summary.json", {})
    payload = {
        "semantic_rescore_decisions": {key: value.get("decision_counts", {}) for key, value in semantic_rescore.items()},
        "phase1c_decision_counts": phase1c.get("decision_counts", {}),
        "phase1d_decision_counts": phase1d.get("decision_counts", {}),
        "final_validation_count": len(final.get("rows", [])),
        "status": "complete",
    }
    write_json(summary_root / "phase1c_final_summary.json", payload)
    report = [
        "# Phase 1C final-edit-aligned report",
        "",
        "## Interpretation rule",
        "",
        "A candidate is only convincing if the clean original edit succeeds, the perturbed input remains close enough, and the perturbed edit visibly weakens or fails. If final images look the same, this report treats the row as metric-only even when CSV scores improved.",
        "",
        "## Legacy Phase 1A/1B semantic rescore",
        "",
    ]
    if semantic_rescore:
        for phase, value in semantic_rescore.items():
            report.append(f"- {phase}: {value.get('decision_counts', {})}")
    else:
        report.append("- Not run yet.")
    report.extend([
        "",
        "## Phase 1C screening",
        "",
        f"- Decision counts: {phase1c.get('decision_counts', {})}",
        "",
        "## Phase 1D deepening",
        "",
        f"- Decision counts: {phase1d.get('decision_counts', {})}",
        f"- Final validation candidates: {len(final.get('rows', []))}",
        "",
        "## Next interpretation",
        "",
    ])
    strong_count = int(phase1c.get("decision_counts", {}).get("strong_candidate", 0)) + int(phase1d.get("decision_counts", {}).get("strong_candidate", 0))
    if strong_count:
        report.append("The best candidates are promising only if the semantic top sheets visually show clean edit success and perturbed edit weakening/failure.")
    else:
        report.append("The white-box internal objectives produced measurable output differences but not a convincing visible edit failure yet, unless manual inspection of the sheets says otherwise.")
    (summary_root / "phase1c_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    mark_done(summary_root, payload)
    return payload
