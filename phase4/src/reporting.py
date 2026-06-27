"""Reporting helpers for Phase 4 landmark semantic geometry."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from phase1.src.data import load_rgb
from phase1.src.reporting import image_sheet

from .utils import relative_path, write_csv, write_json, write_text


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def candidate_sheet(folder: Path, row: dict[str, Any]) -> None:
    required = [
        folder / "original.png",
        folder / "clean_edit.png",
        folder / "perturbed.png",
        folder / "perturbed_edit.png",
        folder / "flow.png",
        folder / "landmarks_overlay_perturbed.jpg",
    ]
    if not all(path.exists() for path in required):
        return
    labels = [
        f"Original\n{row.get('face_id', '')}",
        f"Clean edit\n{row.get('prompt', '')}",
        f"Perturbed\n{row.get('geometry_method', '')}/{row.get('budget', '')}",
        f"Perturbed edit\nsem_drop={_float(row, 'semantic_drop'):.3f}\n{row.get('decision_label', '')}",
        f"Flow\ninput SSIM={_float(row, 'input_ssim'):.3f}",
        "Landmark overlay",
    ]
    image_sheet(folder / "sheet.jpg", [(labels, [load_rgb(path) for path in required])], columns=6, cell_width=180, cell_height=180)


def top_sheet(root: Path, rows: list[dict[str, Any]], destination: Path, *, score_key: str, limit: int = 8) -> None:
    sheet_rows: list[tuple[list[str], list[Image.Image]]] = []
    for row in rows[:limit]:
        folder = root / str(row.get("best_folder", row.get("candidate_folder", "")))
        required = [
            folder / "original.png",
            folder / "clean_edit.png",
            folder / "perturbed.png",
            folder / "perturbed_edit.png",
            folder / "flow.png",
            folder / "landmarks_overlay_perturbed.jpg",
        ]
        if not all(path.exists() for path in required):
            continue
        labels = [
            f"Original\n{row.get('face_id', '')}",
            f"Clean edit\n{row.get('prompt', '')}",
            f"Perturbed\n{row.get('geometry_method', '')}/{row.get('budget', '')}",
            f"Perturbed edit\n{score_key}={_float(row, score_key):.3f}\n{row.get('decision_label', '')}",
            f"Flow\ninput SSIM={_float(row, 'input_ssim'):.3f}",
            "Landmarks",
        ]
        sheet_rows.append((labels, [load_rgb(path) for path in required]))
    if sheet_rows:
        image_sheet(destination, sheet_rows, columns=6, cell_width=180, cell_height=180)
        return
    canvas = Image.new("RGB", (720, 180), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 72), f"No rows for {destination.name}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=93)


def copy_candidate_best(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        "original.png",
        "clean_edit.png",
        "original_edited.png",
        "perturbed.png",
        "perturbed_edit.png",
        "perturbed_edited.png",
        "flow.png",
        "landmarks_overlay_original.jpg",
        "landmarks_overlay_perturbed.jpg",
        "regions_overlay_original.jpg",
        "action_vector.json",
        "geometry_params.json",
        "metrics.json",
        "semantic_score.json",
        "score.json",
        "sheet.jpg",
    ):
        item = source / filename
        if item.exists():
            shutil.copy2(item, destination / filename)


def decision_report(title: str, rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("decision_label", "unknown")) for row in rows)
    lines = [
        f"# {title} decision report",
        "",
        f"- Total candidates: {len(rows)}",
        f"- Visible failure candidates: {counts.get('visible_failure_candidate', 0)}",
        f"- Existence success candidates: {counts.get('existence_success_candidate', 0)}",
        f"- Strong candidates: {counts.get('strong_candidate', 0)}",
        f"- Weak candidates: {counts.get('weak_candidate', 0)}",
        f"- Metric-only candidates: {counts.get('metric_only_candidate', 0)}",
        f"- Rejected for input damage: {counts.get('reject_input_damage', 0)}",
        "",
    ]
    if counts.get("visible_failure_candidate", 0) or counts.get("existence_success_candidate", 0):
        lines.append("Landmark-aware geometry can break at least one InstructPix2Pix edit at the tested budget. Next step is tightening only those cases.")
    else:
        lines.append("Landmark-aware geometry did not produce final-edit failures in this run.")
    lines.append("")
    for label in (
        "visible_failure_candidate",
        "existence_success_candidate",
        "strong_candidate",
        "weak_candidate",
        "metric_only_candidate",
        "reject_input_damage",
    ):
        subset = [row for row in rows if row.get("decision_label") == label][:12]
        if not subset:
            continue
        lines.extend([f"## {label}", ""])
        for row in subset:
            lines.append(
                "- "
                f"{row.get('face_id')} / {row.get('prompt')} / igs={row.get('image_guidance_scale')} / "
                f"{row.get('geometry_method')} / {row.get('budget')} / {row.get('candidate_name')}: "
                f"score={_float(row, 'phase4_final_score'):.4f}, "
                f"semantic_drop={_float(row, 'semantic_drop'):.4f}, "
                f"perturbed_margin={_float(row, 'perturbed_clip_positive_margin'):.4f}, "
                f"input_ssim={_float(row, 'input_ssim'):.4f}, "
                f"max_disp={_float(row, 'max_disp_px'):.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_aggregate_outputs(root: Path, output: Path, prefix: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _float(row, "phase4_final_score", -1e9), reverse=True)
    semantic_ranked = sorted(rows, key=lambda row: (_float(row, "semantic_drop", -1e9), _float(row, "phase4_final_score", -1e9)), reverse=True)
    visible = [row for row in semantic_ranked if row.get("decision_label") == "visible_failure_candidate"]
    existence = [row for row in semantic_ranked if row.get("decision_label") == "existence_success_candidate"]
    write_csv(output / f"{prefix}_all_candidates.csv", ranked)
    write_csv(output / f"{prefix}_top_candidates.csv", ranked[: min(40, len(ranked))])
    write_csv(output / f"{prefix}_semantic_top_candidates.csv", semantic_ranked[: min(40, len(semantic_ranked))])
    write_csv(output / f"{prefix}_visible_failure_candidates.csv", visible)
    write_csv(output / f"{prefix}_existence_success_candidates.csv", existence)
    summary = {
        "candidate_count": len(rows),
        "decision_counts": dict(Counter(str(row.get("decision_label", "unknown")) for row in rows)),
        "best_rows": ranked[: min(24, len(ranked))],
        "semantic_best_rows": semantic_ranked[: min(24, len(semantic_ranked))],
        "visible_failure_count": len(visible),
        "existence_success_count": len(existence),
    }
    write_json(output / f"{prefix}_summary.json", summary)
    write_text(output / f"{prefix}_decision_report.md", decision_report(prefix.upper(), ranked))
    top_sheet(root, ranked, output / f"{prefix}_top_sheet.jpg", score_key="phase4_final_score")
    top_sheet(root, semantic_ranked, output / f"{prefix}_semantic_top_sheet.jpg", score_key="semantic_drop")
    top_sheet(root, visible, output / f"{prefix}_visible_failure_sheet.jpg", score_key="semantic_drop")
    top_sheet(root, existence, output / f"{prefix}_existence_success_sheet.jpg", score_key="semantic_drop")
    return ranked


def attach_candidate_paths(root: Path, row: dict[str, Any], folder: Path) -> dict[str, Any]:
    payload = dict(row)
    payload["candidate_folder"] = relative_path(folder, root)
    payload.setdefault("best_folder", payload["candidate_folder"])
    for key, filename in (
        ("path_original", "original.png"),
        ("path_clean_edit", "clean_edit.png"),
        ("path_perturbed", "perturbed.png"),
        ("path_perturbed_edit", "perturbed_edit.png"),
        ("path_flow", "flow.png"),
        ("path_landmarks_overlay_original", "landmarks_overlay_original.jpg"),
        ("path_landmarks_overlay_perturbed", "landmarks_overlay_perturbed.jpg"),
        ("path_regions_overlay_original", "regions_overlay_original.jpg"),
    ):
        payload[key] = relative_path(folder / filename, root)
    return payload


__all__ = [
    "attach_candidate_paths",
    "candidate_sheet",
    "copy_candidate_best",
    "decision_report",
    "top_sheet",
    "write_aggregate_outputs",
]
