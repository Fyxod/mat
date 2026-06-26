"""Reporting helpers for Phase 2 CEM outputs."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from phase1.src.data import load_rgb
from phase1.src.reporting import attack_sheet, image_sheet, save_rgb

from .utils import relative_path, write_csv, write_json


def copy_candidate_best(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        "original.png",
        "original_edited.png",
        "perturbed.png",
        "perturbed_edited.png",
        "flow.png",
        "metrics.json",
        "semantic_score.json",
        "score.json",
        "geometry_parameters.json",
        "theta.json",
    ):
        item = source / filename
        if item.exists():
            shutil.copy2(item, destination / filename)


def candidate_sheet(folder: Path, row: dict[str, Any]) -> None:
    required = [folder / name for name in ("original.png", "original_edited.png", "perturbed.png", "perturbed_edited.png", "flow.png")]
    if not all(path.exists() for path in required):
        return
    attack_sheet(
        folder / "sheet.jpg",
        load_rgb(required[0]),
        load_rgb(required[1]),
        load_rgb(required[2]),
        load_rgb(required[3]),
        load_rgb(required[4]),
        f"score={float(row.get('phase2_final_score', 0.0)):.3f}, {row.get('decision_label', '')}",
    )


def top_sheet(root: Path, rows: list[dict[str, Any]], destination: Path, *, score_key: str = "phase2_final_score", limit: int = 8) -> None:
    sheet_rows: list[tuple[list[str], list[Image.Image]]] = []
    for row in rows[:limit]:
        folder = root / str(row.get("best_folder", row.get("candidate_folder", "")))
        if not folder.exists():
            folder = root / str(row.get("candidate_folder", ""))
        required = [folder / name for name in ("original.png", "original_edited.png", "perturbed.png", "perturbed_edited.png", "flow.png")]
        if not all(path.exists() for path in required):
            continue
        prompt = str(row.get("prompt", ""))
        labels = [
            f"Original\n{prompt}",
            "Clean edit",
            f"Perturbed\n{row.get('geometry_method', '')}/{row.get('budget', '')}",
            f"Perturbed edit\n{score_key}={float(row.get(score_key, 0.0)):.3f}\n{row.get('decision_label', '')}",
            f"Flow\ninput SSIM={float(row.get('input_ssim', 0.0)):.3f}",
        ]
        sheet_rows.append((labels, [load_rgb(path) for path in required]))
    if sheet_rows:
        image_sheet(destination, sheet_rows, columns=5, cell_width=192, cell_height=192)


def decision_report(title: str, rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("decision_label", "unknown")) for row in rows)
    lines = [
        f"# {title} decision report",
        "",
        f"- Total candidates: {len(rows)}",
        f"- Visible failure candidates: {counts.get('visible_failure_candidate', 0)}",
        f"- Strong semantic candidates: {counts.get('strong_candidate', 0)}",
        f"- Weak semantic candidates: {counts.get('weak_candidate', 0)}",
        f"- Metric-only candidates: {counts.get('metric_only_candidate', 0)}",
        f"- Rejected for input damage: {counts.get('reject_input_damage', 0)}",
        "",
        "Rows are ranked by `phase2_final_score`, which prioritizes semantic/final-edit failure over generic output difference.",
        "",
    ]
    for label in ("visible_failure_candidate", "strong_candidate", "weak_candidate", "metric_only_candidate", "reject_input_damage"):
        subset = [row for row in rows if row.get("decision_label") == label][:10]
        if not subset:
            continue
        lines.extend([f"## {label}", ""])
        for row in subset:
            lines.append(
                "- "
                f"{row.get('prompt')} / {row.get('geometry_method')} / {row.get('budget')} "
                f"{row.get('candidate_name')}: "
                f"score={float(row.get('phase2_final_score', 0.0)):.4f}, "
                f"semantic_drop={float(row.get('semantic_drop', 0.0)):.4f}, "
                f"input_ssim={float(row.get('input_ssim', 0.0)):.4f}, "
                f"output_ssim={float(row.get('output_ssim', 0.0)):.4f}"
            )
        lines.append("")
    if not rows:
        lines.append("No candidates were available.")
    return "\n".join(lines).rstrip() + "\n"


def write_aggregate_outputs(root: Path, output: Path, prefix: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row.get("phase2_final_score", row.get("final_attack_score", 0.0))), reverse=True)
    semantic_ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("semantic_drop", 0.0)),
            float(row.get("phase2_final_score", row.get("final_attack_score", 0.0))),
        ),
        reverse=True,
    )
    write_csv(output / f"{prefix}_all_candidates.csv", ranked)
    write_csv(output / f"{prefix}_top_candidates.csv", ranked[: min(32, len(ranked))])
    write_csv(output / f"{prefix}_semantic_top_candidates.csv", semantic_ranked[: min(32, len(semantic_ranked))])
    write_json(
        output / f"{prefix}_summary.json",
        {
            "candidate_count": len(rows),
            "decision_counts": dict(Counter(str(row.get("decision_label", "unknown")) for row in rows)),
            "best_rows": ranked[: min(24, len(ranked))],
        },
    )
    (output / f"{prefix}_decision_report.md").write_text(decision_report(prefix.upper(), ranked), encoding="utf-8")
    top_sheet(root, ranked, output / f"{prefix}_top_sheet.jpg", score_key="phase2_final_score")
    top_sheet(root, semantic_ranked, output / f"{prefix}_semantic_top_sheet.jpg", score_key="semantic_drop")
    return ranked


def attach_relative_paths(root: Path, row: dict[str, Any], candidate_folder: Path) -> dict[str, Any]:
    payload = dict(row)
    payload["candidate_folder"] = relative_path(candidate_folder, root)
    payload.setdefault("best_folder", payload["candidate_folder"])
    for key, filename in (
        ("path_original", "original.png"),
        ("path_original_edited", "original_edited.png"),
        ("path_perturbed", "perturbed.png"),
        ("path_perturbed_edited", "perturbed_edited.png"),
        ("path_flow", "flow.png"),
    ):
        payload[key] = relative_path(candidate_folder / filename, root)
    return payload


__all__ = [
    "attach_relative_paths",
    "candidate_sheet",
    "copy_candidate_best",
    "decision_report",
    "save_rgb",
    "top_sheet",
    "write_aggregate_outputs",
]
