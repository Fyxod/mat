"""Phase 2 final-edit CEM runner.

This phase is black-box with respect to InstructPix2Pix internals: every
candidate is judged by the actual final edited image.  The optimized variable
is still only a geometric coordinate warp.
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase1.src.config import ModelSettings
from phase1.src.data import load_rgb
from phase1.src.instruct_pipeline import generate_edit, load_instruct_pix2pix
from phase1.src.metrics import image_metrics
from phase1.src.reporting import save_rgb

from .cem import CEMState, CandidateSample
from .final_edit_scoring import load_clip_scorer, score_phase2_candidate
from .geometry_regions import GeometryCodec, apply_flow, displacement_stats, flow_to_image, regions_for_prompt
from .reporting import candidate_sheet, copy_candidate_best, write_aggregate_outputs
from .utils import (
    done_path,
    load_parallel_config,
    load_phase2_config,
    load_region_config,
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


def _candidate_name(sample: CandidateSample) -> str:
    return "cand_000_initial" if sample.initial else f"cand_{sample.candidate_index:03d}_g{sample.generation:02d}_m{sample.member:02d}"


def _model_payload(prompt_info: dict[str, Any]) -> dict[str, Any]:
    return ModelSettings.from_mapping(prompt_info).payload()


def _prepare_clean_images(
    *,
    root: Path,
    combo_folder: Path,
    prompt_info: dict[str, Any],
    input_image: Path,
    pipe: Any | None = None,
    device: Any | None = None,
) -> tuple[Path, Path]:
    settings = ModelSettings.from_mapping(prompt_info)
    original = load_rgb(input_image, size=(settings.width, settings.height))
    original_path = combo_folder / "original.png"
    clean_path = combo_folder / "original_edited.png"
    save_rgb(original_path, original)
    if clean_path.exists():
        return original_path, clean_path
    baseline = root / str(prompt_info.get("baseline_output_path", ""))
    if baseline.exists():
        clean = load_rgb(baseline, size=original.size)
    else:
        if pipe is None or device is None:
            device = _torch_device("cuda:0")
            pipe = load_instruct_pix2pix(settings, device)
        clean = generate_edit(pipe, original, str(prompt_info["prompt"]), settings, device)
    save_rgb(clean_path, clean)
    return original_path, clean_path


def _base_candidate_row(job: dict[str, Any], folder: Path, geometry_info: dict[str, Any], displacement: dict[str, float]) -> dict[str, Any]:
    return {
        "phase": job["phase"],
        "prompt": job["prompt"],
        "prompt_slug": job["prompt_slug"],
        "prompt_priority": job.get("prompt_priority", 0),
        "budget": job["budget_name"],
        "target_input_ssim": float(job["target_input_ssim"]),
        "max_disp_px_budget": float(job["max_disp_px"]),
        "seed": int(job["seed"]),
        "candidate_name": job["candidate_name"],
        "candidate_index": int(job["candidate_index"]),
        "generation": int(job["generation"]),
        "member": int(job["member"]),
        "initial_candidate": bool(job["initial_candidate"]),
        "region_key": job["region_key"],
        "geometry_method": job["geometry_method"],
        "candidate_folder_abs": str(folder),
        **geometry_info,
        **displacement,
    }


def _evaluate_candidate_with_pipe(job: dict[str, Any], pipe: Any, device: Any) -> dict[str, Any]:
    began = time.monotonic()
    folder = Path(job["candidate_folder_abs"])
    folder.mkdir(parents=True, exist_ok=True)
    original = load_rgb(Path(job["original_path"]))
    clean_edit = load_rgb(Path(job["clean_edit_path"]), size=original.size)
    codec = GeometryCodec.from_config(job["geometry_config"])
    flow, geometry_info = codec.decode(
        np.asarray(job["theta"], dtype=np.float32),
        method=str(job["geometry_method"]),
        height=original.height,
        width=original.width,
        max_disp_px=float(job["max_disp_px"]),
        regions=list(job["regions"]),
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
    save_rgb(folder / "original_edited.png", clean_edit)
    save_rgb(folder / "perturbed.png", perturbed)
    save_rgb(folder / "perturbed_edited.png", perturbed_edit)
    save_rgb(folder / "flow.png", flow_to_image(flow))
    row = {
        **_base_candidate_row(job, folder, geometry_info, displacement),
        **{f"input_{key}": value for key, value in input_values.items()},
        **{f"output_{key}": value for key, value in output_values.items()},
        "elapsed_seconds": float(time.monotonic() - began),
    }
    write_json(folder / "metrics.json", {"input": input_values, "output": output_values, "displacement": displacement})
    write_json(folder / "geometry_parameters.json", {"info": geometry_info, "regions": job["regions"]})
    write_json(folder / "theta.json", {"names": job["theta_names"], "values": job["theta"], "method": job["geometry_method"]})
    write_json(folder / "base_row.json", row)
    return row


def _evaluate_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_PIPE is None or _WORKER_DEVICE is None:
        raise RuntimeError("Phase 2 worker was not initialized with an InstructPix2Pix pipeline.")
    return _evaluate_candidate_with_pipe(job, _WORKER_PIPE, _WORKER_DEVICE)


class SerialEvaluator:
    def __init__(self, model_payload: dict[str, Any], device_name: str) -> None:
        self.device = _torch_device(device_name)
        self.pipe = load_instruct_pix2pix(ModelSettings.from_mapping(model_payload), self.device)

    def evaluate(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [_evaluate_candidate_with_pipe(job, self.pipe, self.device) for job in jobs]

    def close(self) -> None:
        del self.pipe
        _clear_cuda()


def _score_and_persist(root: Path, row: dict[str, Any], scoring_config: dict[str, Any], clip_scorer: Any) -> dict[str, Any]:
    folder = Path(row["candidate_folder_abs"])
    original = load_rgb(folder / "original.png")
    clean_edit = load_rgb(folder / "original_edited.png", size=original.size)
    perturbed = load_rgb(folder / "perturbed.png", size=original.size)
    perturbed_edit = load_rgb(folder / "perturbed_edited.png", size=original.size)
    metrics = read_json(folder / "metrics.json", {})
    displacement = dict(metrics.get("displacement", {}))
    semantic = score_phase2_candidate(
        prompt=str(row["prompt"]),
        original=original,
        clean_edit=clean_edit,
        perturbed=perturbed,
        perturbed_edit=perturbed_edit,
        displacement_metrics=displacement,
        scoring_config=scoring_config,
        clip_scorer=clip_scorer,
    )
    merged = {**row, **semantic}
    merged.pop("candidate_folder_abs", None)
    candidate_rel = relative_path(folder, root)
    merged["candidate_folder"] = candidate_rel
    merged["best_folder"] = candidate_rel
    for key, filename in (
        ("path_original", "original.png"),
        ("path_original_edited", "original_edited.png"),
        ("path_perturbed", "perturbed.png"),
        ("path_perturbed_edited", "perturbed_edited.png"),
        ("path_flow", "flow.png"),
    ):
        merged[key] = relative_path(folder / filename, root)
    write_json(folder / "semantic_score.json", semantic)
    write_json(folder / "score.json", {"phase2_final_score": merged["phase2_final_score"], "decision_label": merged["decision_label"]})
    write_json(folder / "candidate_row.json", merged)
    candidate_sheet(folder, merged)
    mark_done(folder, {"candidate_name": merged["candidate_name"], "phase2_final_score": merged["phase2_final_score"], "decision_label": merged["decision_label"]})
    return merged


def _load_completed_candidate(folder: Path) -> dict[str, Any] | None:
    row_path = folder / "candidate_row.json"
    if done_path(folder).exists() and row_path.exists():
        return read_json(row_path, None)
    return None


def _make_job(
    *,
    root: Path,
    phase: str,
    combo_folder: Path,
    sample: CandidateSample,
    prompt_info: dict[str, Any],
    prompt_priority: int,
    budget: dict[str, Any],
    original_path: Path,
    clean_edit_path: Path,
    codec: GeometryCodec,
    geometry_config: dict[str, Any],
    region_key: str,
    regions: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    candidate_name = _candidate_name(sample)
    return {
        "phase": phase,
        "root": str(root),
        "candidate_name": candidate_name,
        "candidate_index": sample.candidate_index,
        "generation": sample.generation,
        "member": sample.member,
        "initial_candidate": sample.initial,
        "candidate_folder_abs": str(combo_folder / "candidates" / candidate_name),
        "original_path": str(original_path),
        "clean_edit_path": str(clean_edit_path),
        "prompt": str(prompt_info["prompt"]),
        "prompt_slug": str(prompt_info["prompt_slug"]),
        "prompt_priority": prompt_priority,
        "budget_name": str(budget["name"]),
        "max_disp_px": float(budget["max_disp_px"]),
        "target_input_ssim": float(budget["target_input_ssim"]),
        "seed": int(seed),
        "model_settings": _model_payload(prompt_info),
        "geometry_config": geometry_config,
        "geometry_method": sample.method,
        "region_key": region_key,
        "regions": regions,
        "theta": [float(value) for value in sample.theta],
        "theta_names": codec.parameter_names,
    }


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
            raise RuntimeError("No Phase 2 evaluator is available.")
    scored = [(_score_and_persist(root, row, scoring_config, clip_scorer), row["candidate_name"]) for row in base_rows]
    completed = cached[:]
    for row, name in scored:
        completed.append((sample_by_name[name], row))
    completed.sort(key=lambda item: item[0].candidate_index)
    return completed


def _combo_seed(base_seed: int, prompt_index: int, budget_index: int, seed_index: int) -> int:
    return int(base_seed + prompt_index * 1000 + budget_index * 100 + seed_index)


def _run_combo(
    *,
    root: Path,
    phase: str,
    output: Path,
    prompt_info: dict[str, Any],
    prompt_priority: int,
    prompt_index: int,
    budget: dict[str, Any],
    budget_index: int,
    seed_index: int,
    cem_config: dict[str, Any],
    config: dict[str, Any],
    region_config: dict[str, Any],
    evaluator: SerialEvaluator | None,
    pool: Any | None,
    clip_scorer: Any,
    force: bool = False,
) -> list[dict[str, Any]]:
    prompt_slug = str(prompt_info["prompt_slug"])
    combo_folder = output / prompt_slug / str(budget["name"]) / f"seed_{seed_index:02d}"
    rows_path = combo_folder / "candidates.csv"
    if done_path(combo_folder).exists() and rows_path.exists() and not force:
        return [dict(row) for row in read_csv(rows_path)]
    combo_folder.mkdir(parents=True, exist_ok=True)
    input_image = root / str(config.get("input_image", "data/face_001/instruct_512.png"))
    original_path, clean_edit_path = _prepare_clean_images(
        root=root,
        combo_folder=combo_folder,
        prompt_info=prompt_info,
        input_image=input_image,
        pipe=evaluator.pipe if evaluator is not None else None,
        device=evaluator.device if evaluator is not None else None,
    )
    geometry_config = dict(config.get("geometry", {}))
    methods = list(geometry_config.get("methods", ["combined_tps_dct"]))
    codec = GeometryCodec.from_config(geometry_config)
    region_key, regions = regions_for_prompt(str(prompt_info["prompt"]), region_config)
    seed = _combo_seed(int(cem_config.get("seed", 22001)), prompt_index, budget_index, seed_index)
    state = CEMState(
        dimension=codec.dimension,
        methods=methods,
        seed=seed,
        sample_std=float(cem_config.get("sample_std", 0.65)),
        min_std=float(cem_config.get("min_std", 0.08)),
        std_smoothing=float(cem_config.get("std_smoothing", 0.65)),
    )
    rows: list[dict[str, Any]] = []
    index = 0
    job_kwargs = {
        "root": root,
        "phase": phase,
        "combo_folder": combo_folder,
        "prompt_info": prompt_info,
        "prompt_priority": prompt_priority,
        "budget": budget,
        "original_path": original_path,
        "clean_edit_path": clean_edit_path,
        "codec": codec,
        "geometry_config": geometry_config,
        "region_key": region_key,
        "regions": regions,
        "seed": seed,
    }
    if bool(cem_config.get("eval_initial", True)):
        initial = state.initial(candidate_index=index)
        rows.extend(row for _, row in _evaluate_samples(
            root=root,
            samples=[initial],
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
        ranked_pairs = sorted(evaluated, key=lambda item: float(item[1].get("phase2_final_score", -1e9)), reverse=True)
        elite_pairs = ranked_pairs[: int(cem_config["elite"])]
        state_payload = state.update([sample for sample, _ in elite_pairs])
        summary = {
            "generation": generation,
            "prompt": prompt_info["prompt"],
            "budget": budget["name"],
            "population": int(cem_config["population"]),
            "elite": int(cem_config["elite"]),
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


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("decision_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _benchmark_jobs(
    *,
    root: Path,
    output: Path,
    config: dict[str, Any],
    region_config: dict[str, Any],
    prompt_info: dict[str, Any],
    budget: dict[str, Any],
    count: int,
    namespace: str,
) -> list[dict[str, Any]]:
    geometry_config = dict(config.get("geometry", {}))
    codec = GeometryCodec.from_config(geometry_config)
    methods = list(geometry_config.get("methods", ["combined_tps_dct"]))
    state = CEMState(
        dimension=codec.dimension,
        methods=methods,
        seed=99901,
        sample_std=0.45,
        min_std=0.08,
    )
    combo_folder = output / "parallel_benchmark" / namespace
    input_image = root / str(config.get("input_image", "data/face_001/instruct_512.png"))
    original_path, clean_edit_path = _prepare_clean_images(
        root=root,
        combo_folder=combo_folder,
        prompt_info=prompt_info,
        input_image=input_image,
    )
    region_key, regions = regions_for_prompt(str(prompt_info["prompt"]), region_config)
    samples = state.sample_generation(1, count, 1)
    return [
        _make_job(
            root=root,
            phase=f"benchmark_{namespace}",
            combo_folder=combo_folder,
            sample=sample,
            prompt_info=prompt_info,
            prompt_priority=1,
            budget=budget,
            original_path=original_path,
            clean_edit_path=clean_edit_path,
            codec=codec,
            geometry_config=geometry_config,
            region_key=region_key,
            regions=regions,
            seed=99901,
        )
        for sample in samples
    ]


def _score_benchmark_rows(root: Path, rows: list[dict[str, Any]], config: dict[str, Any], clip_scorer: Any) -> list[dict[str, Any]]:
    return [_score_and_persist(root, row, dict(config.get("scoring", {})), clip_scorer) for row in rows]


def benchmark_parallel(root: Path, config: dict[str, Any], parallel_config: dict[str, Any], clip_scorer: Any) -> dict[str, Any]:
    output = outputs_root(root) / "phase2a_probe"
    bench_dir = output / "parallel_benchmark"
    result_path = bench_dir / "benchmark.json"
    if result_path.exists():
        return read_json(result_path, {})
    selected = selected_prompt_map(root)
    prompt_info = selected["add headphones"]
    budget = next(item for item in config["budgets"] if item["name"] == "medium")
    region_config = load_region_config(root)
    count = int(parallel_config.get("benchmark_candidates", 8))
    device_name = str(parallel_config.get("cuda_device", "cuda:0"))
    model_payload = _model_payload(prompt_info)
    result: dict[str, Any] = {"benchmark_candidates": count, "workers": int(parallel_config.get("parallel_workers", 2))}
    try:
        serial_jobs = _benchmark_jobs(root=root, output=output, config=config, region_config=region_config, prompt_info=prompt_info, budget=budget, count=count, namespace="serial")
        serial = SerialEvaluator(model_payload, device_name)
        began = time.monotonic()
        serial_rows = serial.evaluate(serial_jobs)
        serial_seconds = time.monotonic() - began
        _score_benchmark_rows(root, serial_rows, config, clip_scorer)
        serial.close()
        parallel_jobs = _benchmark_jobs(root=root, output=output, config=config, region_config=region_config, prompt_info=prompt_info, budget=budget, count=count, namespace="parallel_2")
        ctx = mp.get_context(str(parallel_config.get("worker_start_method", "spawn")))
        began = time.monotonic()
        with ctx.Pool(processes=int(parallel_config.get("parallel_workers", 2)), initializer=_worker_init, initargs=(model_payload, device_name)) as pool:
            parallel_rows = list(pool.map(_evaluate_candidate_job, parallel_jobs))
        parallel_seconds = time.monotonic() - began
        _score_benchmark_rows(root, parallel_rows, config, clip_scorer)
        speedup = serial_seconds / max(parallel_seconds, 1e-6)
        result.update({
            "status": "completed",
            "serial_seconds": float(serial_seconds),
            "parallel_seconds": float(parallel_seconds),
            "speedup": float(speedup),
            "use_parallel": bool(speedup >= float(parallel_config.get("minimum_speedup_for_parallel", 1.15))),
        })
    except Exception as error:
        result.update({
            "status": "failed",
            "use_parallel": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
    write_json(result_path, result)
    return result


def _phase_prompts(config: dict[str, Any], selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = [str(prompt) for prompt in config.get("prompts", [])]
    priority = [str(prompt) for prompt in config.get("prompt_priority", prompts)]
    ordered = [prompt for prompt in priority if prompt in prompts] + [prompt for prompt in prompts if prompt not in priority]
    return [selected[prompt] for prompt in ordered if prompt in selected]


def _run_matrix(
    *,
    root: Path,
    phase: str,
    output: Path,
    cem_config: dict[str, Any],
    config: dict[str, Any],
    prompts: list[dict[str, Any]],
    budgets: list[dict[str, Any]],
    evaluator: SerialEvaluator | None,
    pool: Any | None,
    clip_scorer: Any,
    force: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    region_config = load_region_config(root)
    seeds_per_combo = int(cem_config.get("seeds_per_prompt_budget", 1))
    for prompt_index, prompt_info in enumerate(prompts):
        for budget_index, budget in enumerate(budgets):
            for seed_index in range(seeds_per_combo):
                rows.extend(
                    _run_combo(
                        root=root,
                        phase=phase,
                        output=output,
                        prompt_info=prompt_info,
                        prompt_priority=prompt_index + 1,
                        prompt_index=prompt_index,
                        budget=budget,
                        budget_index=budget_index,
                        seed_index=seed_index,
                        cem_config=cem_config,
                        config=config,
                        region_config=region_config,
                        evaluator=evaluator,
                        pool=pool,
                        clip_scorer=clip_scorer,
                        force=force,
                    )
                )
    return rows


def run_phase2a_probe(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase2_config(root)
    parallel_config = load_parallel_config(root)
    output = outputs_root(root) / "phase2a_probe"
    if done_path(output).exists() and not force:
        rows = read_csv(output / "phase2a_all_candidates.csv")
        print(f"Phase 2A already complete: {len(rows)} candidates")
        return rows
    clip_scorer = load_clip_scorer(config)
    selected = selected_prompt_map(root)
    prompts = _phase_prompts(config, selected)
    budgets = list(config.get("budgets", []))
    benchmark = {"use_parallel": False, "status": "skipped"}
    if bool(parallel_config.get("benchmark_before_phase2a", True)) and bool(parallel_config.get("parallel_experimental", True)):
        benchmark = benchmark_parallel(root, config, parallel_config, clip_scorer)
    use_parallel = bool(benchmark.get("use_parallel")) and bool(parallel_config.get("parallel_experimental", True))
    device_name = str(parallel_config.get("cuda_device", "cuda:0"))
    model_payload = _model_payload(prompts[0])
    rows: list[dict[str, Any]]
    try:
        if use_parallel:
            ctx = mp.get_context(str(parallel_config.get("worker_start_method", "spawn")))
            with ctx.Pool(processes=int(parallel_config.get("parallel_workers", 2)), initializer=_worker_init, initargs=(model_payload, device_name)) as pool:
                rows = _run_matrix(
                    root=root,
                    phase="phase2a_probe",
                    output=output,
                    cem_config=dict(config["phase2a"]),
                    config=config,
                    prompts=prompts,
                    budgets=budgets,
                    evaluator=None,
                    pool=pool,
                    clip_scorer=clip_scorer,
                    force=force,
                )
        else:
            evaluator = SerialEvaluator(model_payload, device_name)
            try:
                rows = _run_matrix(
                    root=root,
                    phase="phase2a_probe",
                    output=output,
                    cem_config=dict(config["phase2a"]),
                    config=config,
                    prompts=prompts,
                    budgets=budgets,
                    evaluator=evaluator,
                    pool=None,
                    clip_scorer=clip_scorer,
                    force=force,
                )
            finally:
                evaluator.close()
    except Exception as error:
        if not use_parallel or not bool(parallel_config.get("retry_serial_on_parallel_failure", True)):
            mark_failed(output, error)
            raise
        write_text(
            outputs_root(root) / "summaries" / "phase2_run_notes.md",
            f"# Phase 2 run notes\n\nParallel Phase 2A failed at {utc_now()} and serial fallback was used.\n\n```\n{traceback.format_exc()}\n```\n",
        )
        evaluator = SerialEvaluator(model_payload, device_name)
        try:
            rows = _run_matrix(
                root=root,
                phase="phase2a_probe",
                output=output,
                cem_config=dict(config["phase2a"]),
                config=config,
                prompts=prompts,
                budgets=budgets,
                evaluator=evaluator,
                pool=None,
                clip_scorer=clip_scorer,
                force=False,
            )
        finally:
            evaluator.close()
    ranked = write_aggregate_outputs(root, output, "phase2a", rows)
    write_json(output / "parallel_benchmark_summary.json", benchmark)
    mark_done(output, {"candidate_count": len(ranked), "parallel_used": use_parallel, "benchmark": benchmark})
    print(f"Phase 2A probe rows: {len(ranked)}")
    print("Inspect phase2/outputs/phase2a_probe/phase2a_semantic_top_sheet.jpg")
    return ranked


def _select_phase2b_candidate(root: Path, config: dict[str, Any]) -> dict[str, str] | None:
    rows = read_csv(outputs_root(root) / "phase2a_probe" / "phase2a_semantic_top_candidates.csv")
    threshold = float(config.get("phase2b", {}).get("promising_weak_semantic_drop", 0.025))
    for row in rows:
        label = row.get("decision_label")
        semantic_drop = float(row.get("semantic_drop", 0.0))
        if label == "strong_candidate" or (label == "weak_candidate" and semantic_drop >= threshold):
            return row
    return None


def run_phase2b_cem(root_value: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    root = project_root(root_value)
    config = load_phase2_config(root)
    output = outputs_root(root) / "phase2b_cem"
    candidate = _select_phase2b_candidate(root, config)
    if candidate is None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "phase2b_summary.json", {"skipped": True, "reason": "phase2a_no_promising_candidate", "best_rows": [], "candidate_count": 0})
        write_text(
            output / "phase2b_decision_report.md",
            "# Phase 2B decision report\n\nPhase 2A found no strong candidates and no weak candidates above the configured semantic-drop threshold. Phase 2B is skipped.\n",
        )
        mark_done(output, {"status": "skipped", "reason": "phase2a_no_promising_candidate"})
        print("Phase 2B skipped: no promising Phase 2A candidate")
        return []
    if done_path(output).exists() and (output / "phase2b_all_candidates.csv").exists() and not force:
        rows = read_csv(output / "phase2b_all_candidates.csv")
        print(f"Phase 2B already complete: {len(rows)} candidates")
        return rows
    clip_scorer = load_clip_scorer(config)
    selected = selected_prompt_map(root)
    prompt_info = selected[str(candidate["prompt"])]
    budget = next(item for item in config["budgets"] if str(item["name"]) == str(candidate["budget"]))
    parallel_config = load_parallel_config(root)
    benchmark = read_json(outputs_root(root) / "phase2a_probe" / "parallel_benchmark_summary.json", {})
    use_parallel = bool(benchmark.get("use_parallel")) and bool(parallel_config.get("parallel_experimental", True))
    device_name = str(parallel_config.get("cuda_device", "cuda:0"))
    model_payload = _model_payload(prompt_info)
    cem_config = dict(config["phase2b"])
    cem_config["seeds_per_prompt_budget"] = 1
    try:
        if use_parallel:
            ctx = mp.get_context(str(parallel_config.get("worker_start_method", "spawn")))
            with ctx.Pool(processes=int(parallel_config.get("parallel_workers", 2)), initializer=_worker_init, initargs=(model_payload, device_name)) as pool:
                rows = _run_matrix(
                    root=root,
                    phase="phase2b_cem",
                    output=output,
                    cem_config=cem_config,
                    config=config,
                    prompts=[prompt_info],
                    budgets=[budget],
                    evaluator=None,
                    pool=pool,
                    clip_scorer=clip_scorer,
                    force=force,
                )
        else:
            evaluator = SerialEvaluator(model_payload, device_name)
            try:
                rows = _run_matrix(
                    root=root,
                    phase="phase2b_cem",
                    output=output,
                    cem_config=cem_config,
                    config=config,
                    prompts=[prompt_info],
                    budgets=[budget],
                    evaluator=evaluator,
                    pool=None,
                    clip_scorer=clip_scorer,
                    force=force,
                )
            finally:
                evaluator.close()
    except Exception as error:
        mark_failed(output, error)
        raise
    ranked = write_aggregate_outputs(root, output, "phase2b", rows)
    write_json(output / "phase2b_selected_from_phase2a.json", candidate)
    mark_done(output, {"candidate_count": len(ranked), "selected_prompt": prompt_info["prompt"], "selected_budget": budget["name"]})
    print(f"Phase 2B CEM rows: {len(ranked)}")
    print("Inspect phase2/outputs/phase2b_cem/phase2b_semantic_top_sheet.jpg")
    return ranked


def summarize_phase2(root_value: str | Path) -> dict[str, Any]:
    root = project_root(root_value)
    summary_dir = outputs_root(root) / "summaries"
    phase1_report = root / "phase1" / "outputs" / "summaries" / "phase1c_final_report.md"
    phase2a = read_json(outputs_root(root) / "phase2a_probe" / "phase2a_summary.json", {})
    phase2b = read_json(outputs_root(root) / "phase2b_cem" / "phase2b_summary.json", {})
    payload = {
        "phase1c_final_report_exists": phase1_report.exists(),
        "phase2a_decision_counts": phase2a.get("decision_counts", {}),
        "phase2b_decision_counts": phase2b.get("decision_counts", {}),
        "phase2b_skipped": bool(phase2b.get("skipped", False)),
        "updated_at": utc_now(),
    }
    lines = [
        "# Phase 2 final-edit geometric CEM report",
        "",
        "## Phase 1 handoff",
        "",
        "Phase 1A/1B/1C are preserved as diagnostic internal-objective results. The current conclusion is not to run more Phase 1C/1D with the same objective family.",
        "",
        "## Phase 2A probe",
        "",
    ]
    if phase2a:
        lines.append(f"- Decision counts: {phase2a.get('decision_counts', {})}")
        best = list(phase2a.get("best_rows", []))[:3]
        for row in best:
            lines.append(
                f"- {row.get('decision_label')}: {row.get('prompt')} / {row.get('geometry_method')} / {row.get('budget')} "
                f"score={float(row.get('phase2_final_score', 0.0)):.4f}, semantic_drop={float(row.get('semantic_drop', 0.0)):.4f}"
            )
    else:
        lines.append("- Phase 2A has not been run yet.")
    lines.extend(["", "## Phase 2B", ""])
    if phase2b:
        if phase2b.get("skipped"):
            lines.append(f"- Skipped: {phase2b.get('reason')}")
        else:
            lines.append(f"- Decision counts: {phase2b.get('decision_counts', {})}")
    else:
        lines.append("- Phase 2B has not been run yet.")
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "If the final clean and perturbed edited images look the same, treat the row as metric-only even if the CSV score is high.",
    ])
    write_json(summary_dir / "phase2_final_summary.json", payload)
    write_text(summary_dir / "phase2_final_report.md", "\n".join(lines) + "\n")
    mark_done(summary_dir, {"phase2_summary": True})
    print("Wrote phase2/outputs/summaries/phase2_final_report.md")
    return payload


__all__ = ["benchmark_parallel", "run_phase2a_probe", "run_phase2b_cem", "summarize_phase2"]

