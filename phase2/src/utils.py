"""Small Phase 2 helpers built on MAT's Phase 1 utilities."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from phase1.src.utils import (
    done_path,
    failed_path,
    mark_done,
    mark_failed,
    project_root,
    read_json,
    relative_path,
    require_selected_prompts,
    slug,
    utc_now,
    write_json,
    write_text,
)


def phase2_root(root: Path) -> Path:
    return root / "phase2"


def outputs_root(root: Path) -> Path:
    path = phase2_root(root) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path(root: Path, name: str) -> Path:
    return phase2_root(root) / "configs" / name


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = list(rows)
    keys = sorted({key for row in data for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def selected_prompt_map(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row["prompt"]): row for row in require_selected_prompts(root)}


def load_phase2_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase2_final_edit_cem.json"), {})


def load_phase2c_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase2c_headphone_failure_probe.json"), {})


def load_parallel_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase2_parallel.json"), {})


def load_region_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase2_prompt_regions.json"), {})


__all__ = [
    "config_path",
    "done_path",
    "failed_path",
    "load_parallel_config",
    "load_phase2_config",
    "load_phase2c_config",
    "load_region_config",
    "mark_done",
    "mark_failed",
    "outputs_root",
    "phase2_root",
    "project_root",
    "read_csv",
    "read_json",
    "relative_path",
    "selected_prompt_map",
    "slug",
    "utc_now",
    "write_csv",
    "write_json",
    "write_text",
]
