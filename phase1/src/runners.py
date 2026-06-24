"""Resumable Phase 1 stage implementations."""
from __future__ import annotations

import csv
import dataclasses
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .config import AttackSettings, ModelSettings
from .data import load_rgb, tensor_to_pil
from .geometry import flow_to_image
from .identity import identity_panel
from .instruct_pipeline import (
    assert_whitebox_contract,
    generate_edit,
    load_instruct_pix2pix,
    prepare_internal_reference,
)
from .metrics import attack_score, clean_baseline_metrics, image_metrics, prompt_quality
from .masks import resolved_face_mask
from .optimizer import optimize_geometry
from .reporting import attack_sheet, copy_if_exists, image_sheet, save_rgb, write_csv, write_json
from .utils import (
    append_run_note,
    is_complete,
    mark_done,
    mark_failed,
    outputs_root,
    read_json,
    relative_path,
    require_selected_prompts,
    slug,
)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 1. Run this workflow on the A6000.")
    return torch.device("cuda")


def _model_settings(root: Path) -> ModelSettings:
    raw = read_json(root / "phase1" / "configs" / "phase1_prompt_filtering.json")
    return ModelSettings.from_mapping(raw["model"])


def _original(root: Path) -> Image.Image:
    return load_rgb(root / "data" / "face_001" / "instruct_512.png", (512, 512))


def _float_key(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _prompt_folder(image_guidance_scale: float, guidance_scale: float) -> str:
    return f"igs_{_float_key(image_guidance_scale)}_gs_{_float_key(guidance_scale)}"


def _optional_sface(original_path: Path, edited_path: Path, enabled: bool) -> float | None:
    if not enabled:
        return None
    panel = identity_panel({"original": original_path, "edited": edited_path}, models=("SFace",))
    item = panel.get("models", {}).get("SFace", {})
    comparison = item.get("comparisons", {}).get("original_vs_edited", {})
    return comparison.get("similarity")


def _prompt_record(
    root: Path,
    prompt: str,
    prompt_group: str,
    settings: ModelSettings,
    folder: Path,
    metrics: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "prompt_slug": slug(prompt),
        "prompt_group": prompt_group,
        "guidance_scale": settings.guidance_scale,
        "image_guidance_scale": settings.image_guidance_scale,
        "num_inference_steps": settings.num_inference_steps,
        "seed": settings.seed,
        **metrics,
        **quality,
        "path_original": relative_path(folder / "original.png", root),
        "path_edited": relative_path(folder / "edited.png", root),
        "path_config": relative_path(folder / "config.json", root),
    }


def _run_prompt_case(
    root: Path,
    pipe,
    original: Image.Image,
    prompt: str,
    prompt_group: str,
    settings: ModelSettings,
    folder: Path,
    force: bool,
    identity_enabled: bool,
) -> dict[str, Any]:
    if is_complete(folder) and not force:
        return read_json(folder / "record.json")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        save_rgb(folder / "original.png", original)
        (folder / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        write_json(
            folder / "config.json",
            {"prompt": prompt, "prompt_group": prompt_group, **settings.payload()},
        )
        edited = generate_edit(pipe, original, prompt, settings, _device())
        save_rgb(folder / "edited.png", edited)
        identity_similarity = _optional_sface(folder / "original.png", folder / "edited.png", identity_enabled)
        metrics = clean_baseline_metrics(original, edited, identity_similarity)
        selection = read_json(root / "phase1" / "configs" / "phase1_prompt_filtering.json")["selection"]
        quality = prompt_quality(
            metrics,
            minimum_ssim=float(selection["minimum_ssim"]),
            minimum_contrast=float(selection["minimum_contrast"]),
            min_identity=float(selection["minimum_identity_similarity_when_available"]),
        )
        record = _prompt_record(root, prompt, prompt_group, settings, folder, metrics, quality)
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "record.json", record)
        mark_done(folder, {"prompt": prompt, "filter_status": quality["filter_status"]})
        return record
    except Exception as error:
        mark_failed(folder, error)
        return {
            "prompt": prompt,
            "prompt_slug": slug(prompt),
            "prompt_group": prompt_group,
            "guidance_scale": settings.guidance_scale,
            "image_guidance_scale": settings.image_guidance_scale,
            "num_inference_steps": settings.num_inference_steps,
            "seed": settings.seed,
            "filter_status": "failed",
            "filter_reasons": str(error),
            "clean_quality_score": -999.0,
            "path_original": relative_path(folder / "original.png", root),
            "path_edited": relative_path(folder / "edited.png", root),
        }


def _select_prompts(records: list[dict[str, Any]], maximum: int, final_inference_steps: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in sorted(records, key=lambda item: float(item.get("clean_quality_score", -999)), reverse=True):
        if record.get("filter_status") != "keep" or record["prompt"] in seen:
            continue
        seen.add(record["prompt"])
        selected.append(
            {
                "prompt": record["prompt"],
                "prompt_slug": record["prompt_slug"],
                "guidance_scale": record["guidance_scale"],
                "image_guidance_scale": record["image_guidance_scale"],
                "num_inference_steps": final_inference_steps,
                "discovery_num_inference_steps": record["num_inference_steps"],
                "seed": record["seed"],
                "baseline_output_path": record["path_edited"],
                "clean_quality_score": record["clean_quality_score"],
                "reason_selected": (
                    "Clean edit passed face/health/identity heuristics and ranked highly for "
                    "identity preservation plus visible localized edit."
                ),
            }
        )
        if len(selected) >= maximum:
            break
    return selected


def _discovery_sheet(root: Path, records: list[dict[str, Any]]) -> None:
    labels: list[str] = []
    images: list[Image.Image] = []
    for record in records:
        edited_path = root / record.get("path_edited", "")
        original_path = root / record.get("path_original", "")
        if not edited_path.exists() or not original_path.exists():
            continue
        caption = (
            f"{record['prompt_slug']} | igs={record['image_guidance_scale']} "
            f"| {record.get('filter_status')} | score={float(record.get('clean_quality_score', 0)):.2f}"
        )
        labels.extend([f"Original\n{caption}", "Edited"])
        images.extend([load_rgb(original_path), load_rgb(edited_path)])
    if images:
        image_sheet(
            outputs_root(root) / "prompt_discovery" / "prompt_discovery_sheet.jpg",
            [(labels, images)],
            columns=4,
            cell_width=256,
            cell_height=256,
        )


def run_prompt_discovery(root: Path, force: bool = False, identity_enabled: bool = False) -> list[dict[str, Any]]:
    output = outputs_root(root) / "prompt_discovery"
    marker = output / "DONE.json"
    selected_path = output / "selected_prompts.json"
    if marker.exists() and selected_path.exists() and not force:
        selected = read_json(selected_path, {}).get("selected_prompts", [])
        if len(selected) >= 3:
            return selected

    config = read_json(root / "phase1" / "configs" / "phase1_prompt_filtering.json")
    bank = read_json(root / "phase1" / "configs" / "phase1_prompt_bank.json")
    original = _original(root)
    base_model = _model_settings(root)
    pipe = load_instruct_pix2pix(base_model, _device())
    records: list[dict[str, Any]] = []
    discovery_steps = base_model.num_inference_steps
    first_case_elapsed: float | None = None

    def run_grid(prompts: list[str], group: str, grid: dict[str, Any]) -> None:
        nonlocal discovery_steps, first_case_elapsed
        for prompt in prompts:
            for image_guidance_scale in grid["image_guidance_scales"]:
                for guidance_scale in grid["guidance_scales"]:
                    settings = dataclasses.replace(
                        base_model,
                        image_guidance_scale=float(image_guidance_scale),
                        guidance_scale=float(guidance_scale),
                        num_inference_steps=discovery_steps,
                    )
                    folder = output / "all_runs" / slug(prompt) / _prompt_folder(
                        settings.image_guidance_scale, settings.guidance_scale
                    )
                    started = time.monotonic()
                    records.append(_run_prompt_case(root, pipe, original, prompt, group, settings, folder, force, identity_enabled))
                    if first_case_elapsed is None:
                        first_case_elapsed = time.monotonic() - started
                        if first_case_elapsed > float(config["discovery_slow_case_seconds"]) and discovery_steps != int(config["discovery_fallback_steps"]):
                            discovery_steps = int(config["discovery_fallback_steps"])
                            append_run_note(
                                root,
                                "Prompt discovery speed fallback",
                                f"First 20-step discovery edit took {first_case_elapsed:.1f}s; remaining discovery candidates use {discovery_steps} steps while baselines and validation remain at {base_model.num_inference_steps} steps.",
                            )

    run_grid(bank["primary_prompts"], "primary", config["primary_grid"])
    run_grid(bank["experimental_risky_prompts"], "experimental_risky", config["primary_grid"])
    selected = _select_prompts(records, int(config["selection"]["max_selected"]), base_model.num_inference_steps)

    if len(selected) < int(config["selection"]["min_selected"]):
        ordered = sorted(records, key=lambda item: float(item.get("clean_quality_score", -999)), reverse=True)
        fallback_prompts: list[str] = []
        for item in ordered:
            if item["prompt"] not in fallback_prompts:
                fallback_prompts.append(item["prompt"])
            if len(fallback_prompts) >= int(config["secondary_grid"]["top_weak_prompts"]):
                break
        append_run_note(
            root,
            "Prompt discovery secondary grid",
            f"Primary grid yielded {len(selected)} selected prompts; testing {len(fallback_prompts)} near-pass prompts at the secondary grid.",
        )
        run_grid(fallback_prompts, "secondary_near_pass", config["secondary_grid"])
        selected = _select_prompts(records, int(config["selection"]["max_selected"]), base_model.num_inference_steps)

    output.mkdir(parents=True, exist_ok=True)
    _discovery_sheet(root, records)
    write_csv(output / "prompt_discovery_summary.csv", records)
    write_json(output / "prompt_discovery_summary.json", {"records": records})
    write_csv(output / "selected_prompts.csv", selected)
    write_json(output / "selected_prompts.json", {"selected_prompts": selected})
    write_csv(
        output / "rejected_prompts.csv",
        [record for record in records if record.get("filter_status") != "keep"],
    )
    report = [
        "# Prompt filtering report",
        "",
        f"- Candidate settings evaluated: {len(records)}",
        f"- Selected usable prompt/settings: {len(selected)}",
        "",
        "Selection uses clean edit health, face detection where OpenCV is available, image similarity, and optional SFace similarity. The sheet remains the visual audit artifact.",
        "",
    ]
    for item in selected:
        report.append(
            f"- {item['prompt']} (IGS {item['image_guidance_scale']}, GS {item['guidance_scale']}): score {item['clean_quality_score']:.3f}"
        )
    (output / "prompt_filtering_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    selected_payload = {
        "selected_prompts": selected,
        "source": relative_path(output / "selected_prompts.json", root),
    }
    write_json(root / "phase1" / "configs" / "phase1_selected_prompts.json", selected_payload)

    if len(selected) >= int(config["selection"]["min_selected"]):
        mark_done(output, {"selected_count": len(selected)})
        append_run_note(root, "Prompt discovery", f"Selected {len(selected)} clean prompt/settings for Phase 1A.")
        return selected

    mark_failed(output, "Fewer than three usable clean prompt/settings after both discovery grids.")
    raise RuntimeError(
        "Prompt discovery found fewer than three usable clean prompt/settings after both grids. "
        "Inspect phase1/outputs/prompt_discovery/prompt_filtering_report.md and the discovery sheet before proceeding."
    )


def _baseline_sheet(root: Path, selected: list[dict[str, Any]]) -> None:
    labels: list[str] = []
    images: list[Image.Image] = []
    baseline_root = outputs_root(root) / "baselines" / "instruct_pix2pix"
    for item in selected:
        folder = baseline_root / item["prompt_slug"]
        original = folder / "original.png"
        edited = folder / "original_edited.png"
        if original.exists() and edited.exists():
            labels.extend([f"Original\n{item['prompt']}", "Clean edit"])
            images.extend([load_rgb(original), load_rgb(edited)])
    if images:
        image_sheet(
            outputs_root(root) / "baselines" / "baseline_sheet.jpg",
            [(labels, images)],
            columns=4,
            cell_width=256,
            cell_height=256,
        )


def run_clean_baselines(root: Path, force: bool = False, identity_enabled: bool = False) -> list[dict[str, Any]]:
    selected = require_selected_prompts(root)
    output = outputs_root(root) / "baselines"
    if is_complete(output) and not force:
        return read_json(output / "baseline_summary.json", {}).get("baselines", [])

    original = _original(root)
    pipe = load_instruct_pix2pix(_model_settings(root), _device())
    rows: list[dict[str, Any]] = []
    for item in selected:
        folder = output / "instruct_pix2pix" / item["prompt_slug"]
        if is_complete(folder) and not force:
            rows.append(read_json(folder / "record.json"))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        try:
            save_rgb(folder / "original.png", original)
            baseline_source = root / item["baseline_output_path"]
            if baseline_source.exists() and not force and item.get("discovery_num_inference_steps", item["num_inference_steps"]) == item["num_inference_steps"]:
                shutil.copy2(baseline_source, folder / "original_edited.png")
                edited = load_rgb(folder / "original_edited.png")
            else:
                edited = generate_edit(
                    pipe, original, item["prompt"], _settings_for_selection(root, item), _device()
                )
                save_rgb(folder / "original_edited.png", edited)
            identity_similarity = _optional_sface(folder / "original.png", folder / "original_edited.png", identity_enabled)
            metrics = clean_baseline_metrics(original, edited, identity_similarity)
            record = {
                **item,
                **metrics,
                "path_original": relative_path(folder / "original.png", root),
                "path_original_edited": relative_path(folder / "original_edited.png", root),
                "path_config": relative_path(folder / "config.json", root),
            }
            write_json(folder / "config.json", item)
            (folder / "prompt.txt").write_text(item["prompt"] + "\n", encoding="utf-8")
            write_json(folder / "metrics.json", metrics)
            write_json(folder / "record.json", record)
            mark_done(folder, {"prompt": item["prompt"]})
            rows.append(record)
        except Exception as error:
            mark_failed(folder, error)
            raise

    _baseline_sheet(root, selected)
    write_csv(output / "baseline_summary.csv", rows)
    write_json(output / "baseline_summary.json", {"baselines": rows})
    mark_done(output, {"baseline_count": len(rows)})
    return rows


def _settings_for_selection(root: Path, selection: dict[str, Any]) -> ModelSettings:
    base = _model_settings(root).payload()
    for key in ("guidance_scale", "image_guidance_scale", "num_inference_steps", "seed"):
        base[key] = selection[key]
    return ModelSettings(**base)


def _attack_output_record(
    root: Path,
    selection: dict[str, Any],
    attack: AttackSettings,
    start: int,
    iteration: int,
    input_values: dict[str, float],
    output_values: dict[str, float],
    displacement: dict[str, float],
    score: dict[str, float],
    folder: Path,
) -> dict[str, Any]:
    return {
        "prompt": selection["prompt"],
        "prompt_slug": selection["prompt_slug"],
        "objective": attack.objective,
        "budget": "custom",
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
        **score,
        "path_perturbed": relative_path(folder / "perturbed.png", root),
        "path_perturbed_edited": relative_path(folder / "perturbed_edited.png", root),
        "path_flow": relative_path(folder / "flow.png", root),
    }


def _run_attack_start(
    root: Path,
    pipe,
    original: Image.Image,
    clean_edit: Image.Image,
    selection: dict[str, Any],
    attack: AttackSettings,
    start: int,
    folder: Path,
    force: bool,
    fallback_objective_scale: float | None = None,
    budget_name: str = "custom",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = folder / "start_result.json"
    if is_complete(folder) and result_path.exists() and not force:
        payload = read_json(result_path)
        return payload["best_row"], payload["checkpoint_rows"]

    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        face_mask, face_mask_source = resolved_face_mask(original)
        attack = dataclasses.replace(attack, face_mask=face_mask)
        model_settings = _settings_for_selection(root, selection)
        original_tensor, reference = prepare_internal_reference(
            pipe,
            original,
            selection["prompt"],
            model_settings,
            _device(),
            attack.objective_timestep_index,
        )
        optimization = optimize_geometry(pipe, original_tensor, reference, attack)
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
                "face_mask_source": face_mask_source,
                "model": model_settings.payload(),
                "whitebox_contract": assert_whitebox_contract(pipe),
                "objective_scale_fallback_used": retry_used,
            },
        )
        write_csv(folder / "history.csv", optimization.history)

        checkpoint_rows: list[dict[str, Any]] = []
        snapshots = sorted(optimization.snapshots.items())
        for iteration, snapshot in snapshots:
            checkpoint = folder / "checkpoints" / f"iter_{iteration:03d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            perturbed = tensor_to_pil(snapshot.image)
            flow = flow_to_image(snapshot.displacement)
            perturbed_edit = generate_edit(pipe, perturbed, selection["prompt"], model_settings, _device())
            save_rgb(checkpoint / "perturbed.png", perturbed)
            save_rgb(checkpoint / "perturbed_edited.png", perturbed_edit)
            save_rgb(checkpoint / "flow.png", flow)
            input_values = image_metrics(original, perturbed)
            output_values = image_metrics(clean_edit, perturbed_edit)
            from .geometry import displacement_stats, jacobian_penalty
            _, jacobian = jacobian_penalty(snapshot.displacement)
            displacement = {**displacement_stats(snapshot.displacement), **jacobian}
            score = attack_score(
                input_values, output_values, displacement, attack.target_input_ssim, attack.max_disp_px
            )
            metrics = {"input": input_values, "output": output_values, "displacement": displacement}
            row = _attack_output_record(
                root, selection, attack, start, iteration, input_values, output_values, displacement, score, checkpoint
            )
            row["budget"] = budget_name
            write_json(checkpoint / "metrics.json", metrics)
            write_json(checkpoint / "score.json", score)
            checkpoint_rows.append(row)

        best_row = max(checkpoint_rows, key=lambda row: float(row["attack_score"]))
        best_checkpoint = root / best_row["path_perturbed"].replace("perturbed.png", "")
        best_folder = folder / "best"
        best_folder.mkdir(parents=True, exist_ok=True)
        for filename in ("perturbed.png", "perturbed_edited.png", "flow.png", "metrics.json", "score.json"):
            copy_if_exists(best_checkpoint / filename, best_folder / filename)
        save_rgb(best_folder / "original.png", original)
        save_rgb(best_folder / "original_edited.png", clean_edit)
        best_snapshot = optimization.snapshots[int(best_row["iter"])]
        torch.save(
            {"state_dict": best_snapshot.state_dict, "iteration": best_snapshot.iteration},
            best_folder / "theta.pt",
        )
        attack_sheet(
            best_folder / "sheet.jpg",
            original,
            clean_edit,
            load_rgb(best_folder / "perturbed.png"),
            load_rgb(best_folder / "perturbed_edited.png"),
            load_rgb(best_folder / "flow.png"),
            f"score={best_row['attack_score']:.3f}, input SSIM={best_row['input_ssim']:.3f}",
        )
        best_row = {
            **best_row,
            "elapsed_seconds": time.monotonic() - started,
            "best_folder": relative_path(best_folder, root),
            "objective_scale_used": attack.objective_scale,
        }
        write_json(result_path, {"best_row": best_row, "checkpoint_rows": checkpoint_rows})
        mark_done(folder, {"best_attack_score": best_row["attack_score"], "best_iter": best_row["iter"]})
        return best_row, checkpoint_rows
    except Exception as error:
        mark_failed(folder, error)
        raise


def _attack_settings(
    screen: dict[str, Any],
    objective: str,
    budget: dict[str, Any],
    start: int,
    checkpoint_every: int,
    iterations: int | None = None,
    learning_rate: float | None = None,
) -> AttackSettings:
    return AttackSettings(
        objective=objective,
        max_disp_px=float(budget["max_disp_px"]),
        target_input_ssim=float(budget["target_input_ssim"]),
        iterations=int(iterations or screen["iterations"]),
        learning_rate=float(learning_rate or screen["learning_rate"]),
        objective_scale=float(screen["objective_scale"]),
        lambda_visual=float(screen["lambda_visual"]),
        lambda_disp=float(screen["lambda_disp"]),
        lambda_smooth=float(screen["lambda_smooth"]),
        lambda_fold=float(screen["lambda_fold"]),
        checkpoint_every=int(checkpoint_every),
        objective_timestep_index=int(screen["objective_timestep_index"]),
        geometry_method=str(screen["geometry_method"]),
        dct_size=int(screen["dct_size"]),
        tps_size=int(screen["tps_size"]),
        face_mask=dict(screen["face_mask"]),
        seed=1234 + start * 1009,
    )


def _attack_top_sheet(root: Path, rows: list[dict[str, Any]], destination: Path) -> None:
    sheet_rows: list[tuple[list[str], list[Image.Image]]] = []
    for row in rows[:8]:
        best = root / row["best_folder"]
        required = [best / name for name in ("original.png", "original_edited.png", "perturbed.png", "perturbed_edited.png", "flow.png")]
        if not all(path.exists() for path in required):
            continue
        sheet_rows.append((
            [
                f"Original\n{row['prompt_slug']}",
                "Clean edit",
                f"Perturbed\n{row['objective']} / {row['budget']}",
                f"Perturbed edit\nscore={float(row['attack_score']):.3f}",
                f"Flow\ninput SSIM={float(row['input_ssim']):.3f}",
            ],
            [load_rgb(path) for path in required],
        ))
    if sheet_rows:
        image_sheet(destination, sheet_rows, columns=5, cell_width=192, cell_height=192)


def run_phase1a(root: Path, force: bool = False) -> list[dict[str, Any]]:
    selected = require_selected_prompts(root)
    output = outputs_root(root) / "phase1a_screening"
    if is_complete(output) and not force:
        return read_json(output / "phase1a_summary.json", {}).get("best_rows", [])

    screen = read_json(root / "phase1" / "configs" / "phase1_screening.json")
    baseline = {row["prompt_slug"]: row for row in read_json(outputs_root(root) / "baselines" / "baseline_summary.json", {}).get("baselines", [])}
    if len(baseline) < 3:
        raise RuntimeError("Selected clean baselines are missing. Run the baselines mode before Phase 1A.")
    original = _original(root)
    pipe = load_instruct_pix2pix(_model_settings(root), _device())
    best_rows: list[dict[str, Any]] = []
    all_checkpoint_rows: list[dict[str, Any]] = []
    effective_checkpoint_every = int(screen["checkpoint_every"])
    first_elapsed: float | None = None

    for selection in selected:
        clean_edit = load_rgb(root / baseline[selection["prompt_slug"]]["path_original_edited"])
        for objective in screen["objectives"]:
            for budget in screen["budgets"]:
                for start in range(int(screen["starts"])):
                    attack = _attack_settings(screen, objective, budget, start, effective_checkpoint_every)
                    folder = output / selection["prompt_slug"] / objective / budget["name"] / f"start_{start:02d}"
                    best, checkpoints = _run_attack_start(
                        root, pipe, original, clean_edit, selection, attack, start, folder, force,
                        fallback_objective_scale=float(screen["fallback_objective_scale"]),
                        budget_name=budget["name"],
                    )
                    best["budget"] = budget["name"]
                    for row in checkpoints:
                        row["budget"] = budget["name"]
                    best_rows.append(best)
                    all_checkpoint_rows.extend(checkpoints)
                    if first_elapsed is None:
                        first_elapsed = float(best["elapsed_seconds"])
                        if first_elapsed > float(screen["fallback_first_start_seconds"]):
                            effective_checkpoint_every = int(screen["fallback_checkpoint_every"])
                            append_run_note(
                                root,
                                "Phase 1A checkpoint fallback",
                                f"First screening start took {first_elapsed:.1f}s; remaining starts use checkpoints every {effective_checkpoint_every} iterations.",
                            )

    ranked = sorted(best_rows, key=lambda row: float(row["attack_score"]), reverse=True)
    top = ranked[: min(24, len(ranked))]
    write_csv(output / "phase1a_all_candidates.csv", ranked)
    write_csv(output / "phase1a_all_checkpoints.csv", all_checkpoint_rows)
    write_csv(output / "phase1a_top_candidates.csv", top)
    write_json(output / "phase1a_summary.json", {"best_rows": ranked, "checkpoint_count": len(all_checkpoint_rows)})
    _attack_top_sheet(root, top, output / "phase1a_top_sheet.jpg")
    decision = [
        "# Phase 1A decision report",
        "",
        f"- Completed starts: {len(best_rows)}",
        f"- Checkpoint rows: {len(all_checkpoint_rows)}",
        f"- Effective checkpoint interval after timing: {effective_checkpoint_every}",
        "",
        "Phase 1B will take the top four unique prompt/objective/budget combinations ranked by attack score.",
    ]
    (output / "phase1a_decision_report.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    mark_done(output, {"starts": len(best_rows)})
    return ranked


def _unique_phase1b_combinations(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["prompt_slug"], row["objective"], row["budget"])
        if key not in seen:
            seen.add(key)
            output.append(row)
        if len(output) >= limit:
            break
    return output


def run_phase1b(root: Path, force: bool = False) -> list[dict[str, Any]]:
    output = outputs_root(root) / "phase1b_deepen"
    if is_complete(output) and not force:
        return read_json(output / "phase1b_summary.json", {}).get("best_rows", [])

    phase1a = read_json(outputs_root(root) / "phase1a_screening" / "phase1a_summary.json", {})
    all_rows = phase1a.get("best_rows", [])
    if not all_rows:
        raise RuntimeError("Phase 1A results are missing. Run phase1a before phase1b.")
    deep = read_json(root / "phase1" / "configs" / "phase1_deepening.json")
    screen = read_json(root / "phase1" / "configs" / "phase1_screening.json")
    selected = {item["prompt_slug"]: item for item in require_selected_prompts(root)}
    baseline = {row["prompt_slug"]: row for row in read_json(outputs_root(root) / "baselines" / "baseline_summary.json", {}).get("baselines", [])}
    combinations = _unique_phase1b_combinations(all_rows, int(deep["top_combinations"]))
    write_json(output / "phase1b_selected_combinations.json", {"combinations": combinations})

    original = _original(root)
    pipe = load_instruct_pix2pix(_model_settings(root), _device())
    best_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    effective_starts = int(deep["starts"])
    effective_iterations = int(deep["iterations"])
    effective_checkpoint_every = int(deep["checkpoint_every"])

    budget_lookup = {item["name"]: item for item in screen["budgets"]}
    for combo_index, combo in enumerate(combinations):
        selection = selected[combo["prompt_slug"]]
        clean_edit = load_rgb(root / baseline[selection["prompt_slug"]]["path_original_edited"])
        budget = budget_lookup[combo["budget"]]
        combo_id = f"{combo_index + 1:02d}_{selection['prompt_slug']}__{combo['objective']}__{combo['budget']}"
        start = 0
        while start < effective_starts:
            attack = _attack_settings(
                screen, combo["objective"], budget, start + 100 * (combo_index + 1),
                effective_checkpoint_every, iterations=effective_iterations, learning_rate=float(deep["learning_rate"]),
            )
            folder = output / combo_id / f"start_{start:02d}"
            best, checkpoint_rows = _run_attack_start(
                root, pipe, original, clean_edit, selection, attack, start, folder, force,
                fallback_objective_scale=float(screen["fallback_objective_scale"]),
                budget_name=combo["budget"],
            )
            best["budget"] = combo["budget"]
            best["combo_id"] = combo_id
            for row in checkpoint_rows:
                row["budget"] = combo["budget"]
                row["combo_id"] = combo_id
            best_rows.append(best)
            checkpoints.extend(checkpoint_rows)
            if combo_index == 0 and start == 0 and best["elapsed_seconds"] > float(deep["fallback_first_start_seconds"]):
                effective_starts = int(deep["fallback_starts"])
                effective_iterations = int(deep["fallback_iterations"])
                effective_checkpoint_every = int(deep["fallback_checkpoint_every"])
                append_run_note(
                    root,
                    "Phase 1B timing fallback",
                    f"First deepening start took {best['elapsed_seconds']:.1f}s; remaining deepening starts use {effective_starts} starts, {effective_iterations} iterations, checkpoint every {effective_checkpoint_every}.",
                )
            start += 1

    ranked = sorted(best_rows, key=lambda row: float(row["attack_score"]), reverse=True)
    write_csv(output / "phase1b_all_candidates.csv", ranked)
    write_csv(output / "phase1b_all_checkpoints.csv", checkpoints)
    write_csv(output / "phase1b_top_candidates.csv", ranked[: min(24, len(ranked))])
    write_json(output / "phase1b_summary.json", {"best_rows": ranked, "checkpoint_count": len(checkpoints)})
    _attack_top_sheet(root, ranked, output / "phase1b_top_sheet.jpg")
    mark_done(output, {"starts": len(best_rows), "combinations": len(combinations)})
    return ranked


def run_final_validation(root: Path, force: bool = False) -> list[dict[str, Any]]:
    output = outputs_root(root) / "final_validation"
    if is_complete(output) and not force:
        return read_json(output / "final_validation_summary.json", {}).get("rows", [])

    candidates = read_json(outputs_root(root) / "phase1b_deepen" / "phase1b_summary.json", {}).get("best_rows", [])
    if not candidates:
        candidates = read_json(outputs_root(root) / "phase1a_screening" / "phase1a_summary.json", {}).get("best_rows", [])
    if not candidates:
        raise RuntimeError("No Phase 1A or Phase 1B candidates exist for final validation.")

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:8], 1):
        target = output / "top_candidates" / f"candidate_{index:03d}"
        if is_complete(target) and not force:
            rows.append(read_json(target / "record.json"))
            continue
        target.mkdir(parents=True, exist_ok=True)
        source = root / candidate["best_folder"]
        for filename in ("original.png", "original_edited.png", "perturbed.png", "perturbed_edited.png", "flow.png", "metrics.json"):
            copy_if_exists(source / filename, target / filename)
        paths = {name: target / f"{name}.png" for name in ("original", "original_edited", "perturbed", "perturbed_edited")}
        panel = identity_panel(paths)
        write_json(target / "identity_panel.json", panel)
        report = [
            f"# Final validation: candidate {index:03d}",
            "",
            f"- Prompt: {candidate['prompt']}",
            f"- Objective: {candidate['objective']}",
            f"- Budget: {candidate['budget']}",
            f"- Attack score: {float(candidate['attack_score']):.5f}",
            f"- Input SSIM: {float(candidate['input_ssim']):.5f}",
            f"- Output SSIM: {float(candidate['output_ssim']):.5f}",
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
            f"score={float(candidate['attack_score']):.3f}",
        )
        row = {
            **candidate,
            "candidate_id": f"candidate_{index:03d}",
            "identity_panel_available": panel["available"],
            "path_validation": relative_path(target, root),
        }
        write_json(target / "record.json", row)
        mark_done(target, {"candidate_id": row["candidate_id"]})
        rows.append(row)

    write_csv(output / "final_top_candidates.csv", rows)
    write_json(output / "final_validation_summary.json", {"rows": rows})
    _attack_top_sheet(root, rows, output / "final_top_sheet.jpg")
    mark_done(output, {"candidate_count": len(rows)})
    return rows


def run_summary(root: Path) -> dict[str, Any]:
    final = read_json(outputs_root(root) / "final_validation" / "final_validation_summary.json", {}).get("rows", [])
    prompt = read_json(outputs_root(root) / "prompt_discovery" / "selected_prompts.json", {}).get("selected_prompts", [])
    payload = {
        "selected_prompt_count": len(prompt),
        "final_candidate_count": len(final),
        "top_candidate": final[0] if final else None,
        "status": "complete" if final else "incomplete",
    }
    summary_root = outputs_root(root) / "summaries"
    write_json(summary_root / "phase1_final_summary.json", payload)
    report = [
        "# Phase 1 geometric white-box summary",
        "",
        f"- Selected clean prompt/settings: {len(prompt)}",
        f"- Final validated candidates: {len(final)}",
        f"- Status: {payload['status']}",
        "",
    ]
    if final:
        best = final[0]
        report.extend(
            [
                "## Best current candidate",
                "",
                f"- Prompt: {best['prompt']}",
                f"- Objective: {best['objective']}",
                f"- Budget: {best['budget']}",
                f"- Attack score: {float(best['attack_score']):.5f}",
                f"- Input SSIM: {float(best['input_ssim']):.5f}",
                f"- Output SSIM: {float(best['output_ssim']):.5f}",
            ]
        )
    (summary_root / "phase1_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    mark_done(summary_root, payload)
    return payload


def run_smoke(root: Path, force: bool = False) -> dict[str, Any]:
    output = outputs_root(root) / "smoke"
    if is_complete(output) and not force:
        return read_json(output / "smoke_summary.json")
    model = _model_settings(root)
    original = _original(root)
    pipe = load_instruct_pix2pix(model, _device())
    clean_edit = generate_edit(pipe, original, "add black sunglasses", model, _device())
    attack = AttackSettings(
        objective="edit_direction",
        max_disp_px=1.0,
        target_input_ssim=0.98,
        iterations=3,
        learning_rate=0.05,
        checkpoint_every=1,
        objective_timestep_index=min(2, model.num_inference_steps - 1),
        geometry_method="combined_tps_dct",
        dct_size=3,
        tps_size=4,
        seed=model.seed,
    )
    selection = {
        "prompt": "add black sunglasses",
        "prompt_slug": "add_black_sunglasses",
        "guidance_scale": model.guidance_scale,
        "image_guidance_scale": model.image_guidance_scale,
        "num_inference_steps": model.num_inference_steps,
        "seed": model.seed,
    }
    best, checkpoints = _run_attack_start(
        root, pipe, original, clean_edit, selection, attack, 0, output / "whitebox", force
    )
    history = list(csv.DictReader((output / "whitebox" / "history.csv").open(encoding="utf-8")))
    gradients_reached_geometry = any(float(row.get("grad_norm", 0.0)) > 0.0 for row in history)
    summary = {
        "status": "passed" if gradients_reached_geometry else "failed",
        "whitebox_contract": assert_whitebox_contract(pipe),
        "gradients_reached_geometry": gradients_reached_geometry,
        "optimizer_iterations": len(history),
        "best": best,
        "checkpoint_count": len(checkpoints),
    }
    write_json(output / "smoke_summary.json", summary)
    if not gradients_reached_geometry:
        mark_failed(output, "Geometry gradients were zero in the three-iteration smoke test.")
        raise RuntimeError("Smoke failed: geometry gradients did not reach the trainable parameters.")
    mark_done(output, summary)
    return summary
