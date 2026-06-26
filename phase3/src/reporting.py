"""Reporting helpers for Phase 3 clean discovery and breadth probe."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from phase1.src.data import load_rgb
from phase1.src.reporting import image_sheet
from phase2.src.reporting import top_sheet as phase2_top_sheet

from .utils import relative_path, write_csv, write_json, write_text


def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def clean_sheet(root: Path, rows: list[dict[str, Any]], destination: Path, *, limit: int = 48) -> None:
    sheet_rows: list[tuple[list[str], list[Image.Image]]] = []
    for row in rows[:limit]:
        original = root / str(row.get("image_path", ""))
        clean = root / str(row.get("clean_output_path", ""))
        if not original.exists() or not clean.exists():
            continue
        labels = [
            f"{row.get('face_id')}\nOriginal",
            f"{row.get('prompt')}\nigs={row.get('image_guidance_scale')}\nmargin={_safe_float(row, 'clean_clip_margin'):.3f}",
        ]
        sheet_rows.append((labels, [load_rgb(original), load_rgb(clean)]))
    if sheet_rows:
        image_sheet(destination, sheet_rows, columns=2, cell_width=192, cell_height=192)


def clean_report(
    *,
    detected_faces: list[dict[str, Any]],
    discovery_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> str:
    selected_by_prompt = Counter(str(row.get("prompt_type", "unknown")) for row in selected_rows)
    rejected_by_reason = Counter(str(row.get("reject_reason", "unknown")) for row in rejected_rows)
    lines = [
        "# Phase 3 clean discovery report",
        "",
        f"- Detected face folders: {len(detected_faces)}",
        f"- Clean discovery evaluations: {len(discovery_rows)}",
        f"- Selected for first breadth probe: {len(selected_rows)}",
        f"- Rejected clean cases: {len(rejected_rows)}",
        f"- Skipped incompatible image/prompt/settings rows: {len(skipped_rows)}",
        "",
        "## Face usage",
        "",
    ]
    for face in detected_faces:
        restrictions = []
        if face.get("existing_glasses"):
            restrictions.append("no glasses prompts")
        if face.get("existing_facial_hair"):
            restrictions.append("no beard/stubble prompts")
        lines.append(
            f"- {face.get('face_id')}: {face.get('priority_group')}; "
            f"{', '.join(restrictions) if restrictions else 'no prompt restriction detected'}"
        )
    lines.extend([
        "",
        "## Selected prompt-type counts",
        "",
        f"`{dict(selected_by_prompt)}`",
        "",
        "## Main clean rejection reasons",
        "",
        f"`{dict(rejected_by_reason)}`",
        "",
        "A clean case is selected only when the clean edit has positive CLIP margin and avoids coarse global-collapse checks. Visual audit is still required from the contact sheet.",
    ])
    if len(selected_rows) < 8:
        lines.extend([
            "",
            "## Breadth probe warning",
            "",
            "Fewer than 8 clean-success cases were selected. The Phase 3B breadth probe will skip the large attack stage until more images/prompts are added or the clean-discovery thresholds are revised.",
        ])
    return "\n".join(lines).rstrip() + "\n"


def breadth_decision_report(title: str, rows: list[dict[str, Any]]) -> str:
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
        "Ranking prioritizes final-edit semantic weakening over generic output pixel difference. If the clean and perturbed final edits visually still show the requested edit, treat the row as weak or metric-only.",
        "",
    ]
    for label in ("visible_failure_candidate", "strong_candidate", "weak_candidate", "metric_only_candidate", "reject_input_damage"):
        subset = [row for row in rows if row.get("decision_label") == label][:12]
        if not subset:
            continue
        lines.extend([f"## {label}", ""])
        for row in subset:
            lines.append(
                "- "
                f"{row.get('face_id')} / {row.get('prompt')} / igs={row.get('image_guidance_scale')} / "
                f"{row.get('geometry_method')} / {row.get('budget')} / {row.get('candidate_name')}: "
                f"score={_safe_float(row, 'phase2_final_score'):.4f}, "
                f"semantic_drop={_safe_float(row, 'semantic_drop'):.4f}, "
                f"clean_margin={_safe_float(row, 'clean_clip_positive_margin'):.4f}, "
                f"perturbed_margin={_safe_float(row, 'perturbed_clip_positive_margin'):.4f}, "
                f"input_ssim={_safe_float(row, 'input_ssim'):.4f}"
            )
        lines.append("")
    if counts.get("visible_failure_candidate", 0) or counts.get("strong_candidate", 0):
        lines.extend([
            "## Recommendation",
            "",
            "Phase 3 found at least one strong/visible candidate. Preserve these artifacts and run a narrow Phase 3C deepening only for those image/prompt/settings cases.",
        ])
    elif counts.get("weak_candidate", 0):
        lines.extend([
            "## Recommendation",
            "",
            "Phase 3 found weak candidates but no convincing strong final-edit failure. Inspect the top sheet before spending A6000 time on any deepening.",
        ])
    else:
        lines.extend([
            "## Recommendation",
            "",
            "No promising candidates were found in the tested breadth probe. Consider a larger image set or another edit model rather than deepening the same setup.",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_breadth_aggregate(root: Path, output: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _safe_float(row, "phase2_final_score", -1e9), reverse=True)
    semantic_ranked = sorted(
        rows,
        key=lambda row: (_safe_float(row, "semantic_drop", -1e9), _safe_float(row, "phase2_final_score", -1e9)),
        reverse=True,
    )
    strong = [
        row for row in semantic_ranked
        if row.get("decision_label") in {"visible_failure_candidate", "strong_candidate"}
    ]
    write_csv(output / "phase3_all_candidates.csv", ranked)
    write_csv(output / "phase3_top_candidates.csv", ranked[: min(32, len(ranked))])
    write_csv(output / "phase3_semantic_top_candidates.csv", semantic_ranked[: min(32, len(semantic_ranked))])
    write_csv(output / "phase3_strong_candidates.csv", strong)
    write_json(
        output / "phase3_summary.json",
        {
            "candidate_count": len(rows),
            "decision_counts": dict(Counter(str(row.get("decision_label", "unknown")) for row in rows)),
            "best_rows": ranked[: min(24, len(ranked))],
            "semantic_best_rows": semantic_ranked[: min(24, len(semantic_ranked))],
            "strong_candidate_count": len(strong),
        },
    )
    write_text(output / "phase3_decision_report.md", breadth_decision_report("Phase 3 breadth probe", ranked))
    phase2_top_sheet(root, ranked, output / "phase3_top_sheet.jpg", score_key="phase2_final_score")
    phase2_top_sheet(root, semantic_ranked, output / "phase3_semantic_top_sheet.jpg", score_key="semantic_drop")
    return ranked


def attach_phase3_paths(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    for key in ("candidate_folder", "best_folder"):
        if key in payload:
            payload[key] = str(payload[key]).replace("\\", "/")
    folder_value = payload.get("candidate_folder") or payload.get("best_folder")
    if folder_value:
        folder = root / str(folder_value)
        for key, filename in (
            ("path_original", "original.png"),
            ("path_original_edited", "original_edited.png"),
            ("path_perturbed", "perturbed.png"),
            ("path_perturbed_edited", "perturbed_edited.png"),
            ("path_flow", "flow.png"),
        ):
            payload[key] = relative_path(folder / filename, root)
    return payload


__all__ = [
    "attach_phase3_paths",
    "breadth_decision_report",
    "clean_report",
    "clean_sheet",
    "write_breadth_aggregate",
]

