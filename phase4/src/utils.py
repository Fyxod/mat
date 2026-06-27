"""Small utilities for Phase 4 landmark semantic geometry."""
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
    slug,
    utc_now,
    write_json,
    write_text,
)


def phase4_root(root: Path) -> Path:
    return root / "phase4"


def outputs_root(root: Path) -> Path:
    path = phase4_root(root) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path(root: Path, name: str) -> Path:
    return phase4_root(root) / "configs" / name


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


REAL_LANDMARK_DETECTORS = {
    "mediapipe_solutions_face_mesh",
    "mediapipe_python_solutions_face_mesh",
    "mediapipe_tasks_face_landmarker",
}


def load_phase4_config(root: Path, name: str = "phase4_landmark_probe.json") -> dict[str, Any]:
    return read_json(config_path(root, name), {})


def load_action_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase4_landmark_actions.json"), {})


def load_parallel_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase4_parallel.json"), {})


def prompt_slug(prompt: str) -> str:
    return slug(prompt)


def setting_slug(image_guidance_scale: float) -> str:
    return f"igs_{float(image_guidance_scale):.2f}".replace(".", "p").replace("-", "m")


def face_ids_from_data(root: Path) -> list[str]:
    return sorted(path.name for path in (root / "data").glob("face_*") if path.is_dir())


def phase4_model_payload(config: dict[str, Any], *, image_guidance_scale: float) -> dict[str, Any]:
    payload = dict(config.get("model", {}))
    payload["image_guidance_scale"] = float(image_guidance_scale)
    return payload


def landmark_output_folder(root: Path, face_id: str) -> Path:
    return outputs_root(root) / "landmarks" / face_id


def landmark_status_path(root: Path, face_id: str) -> Path:
    return landmark_output_folder(root, face_id) / "status.json"


def landmark_json_path(root: Path, face_id: str) -> Path:
    return landmark_output_folder(root, face_id) / "landmarks.json"


def load_landmark_statuses(root: Path) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for face_id in face_ids_from_data(root):
        status = read_json(landmark_status_path(root, face_id), {})
        if status:
            statuses[face_id] = status
    return statuses


def status_success(status: dict[str, Any]) -> bool:
    success_value = status.get("success", False)
    return str(success_value).lower() not in {"", "0", "false", "none", "no"}


def status_has_real_landmarks(status: dict[str, Any]) -> bool:
    try:
        landmark_count = int(status.get("landmark_count", 0))
    except Exception:
        landmark_count = 0
    return (
        status_success(status)
        and str(status.get("detector", "")) in REAL_LANDMARK_DETECTORS
        and landmark_count >= 468
    )


def successful_landmark_faces(root: Path, *, require_real: bool = False) -> set[str]:
    return {
        face_id
        for face_id, status in load_landmark_statuses(root).items()
        if (status_has_real_landmarks(status) if require_real else status_success(status))
    }


__all__ = [
    "config_path",
    "done_path",
    "face_ids_from_data",
    "failed_path",
    "landmark_json_path",
    "landmark_output_folder",
    "landmark_status_path",
    "load_action_config",
    "load_landmark_statuses",
    "load_parallel_config",
    "load_phase4_config",
    "mark_done",
    "mark_failed",
    "outputs_root",
    "phase4_model_payload",
    "phase4_root",
    "project_root",
    "prompt_slug",
    "read_csv",
    "read_json",
    "relative_path",
    "setting_slug",
    "status_success",
    "status_has_real_landmarks",
    "successful_landmark_faces",
    "utc_now",
    "write_csv",
    "write_json",
    "write_text",
]
