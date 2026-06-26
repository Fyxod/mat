"""Utilities for Phase 3 breadth search.

The helpers here deliberately keep public-person metadata out of experiment
logic.  Reports and CSVs use stable `face_xxx` IDs; source names/URLs stay in
the original data metadata for attribution only.
"""
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


def phase3_root(root: Path) -> Path:
    return root / "phase3"


def outputs_root(root: Path) -> Path:
    path = phase3_root(root) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path(root: Path, name: str) -> Path:
    return phase3_root(root) / "configs" / name


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


def load_phase3_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase3_breadth_search.json"), {})


def load_prompt_bank(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase3_prompt_bank.json"), {})


def load_image_manifest_config(root: Path) -> dict[str, Any]:
    return read_json(config_path(root, "phase3_image_manifest.json"), {})


def prompt_slug(prompt: str) -> str:
    return slug(prompt)


def setting_slug(image_guidance_scale: float) -> str:
    text = f"igs_{float(image_guidance_scale):.2f}".replace(".", "p")
    return text.replace("-", "m")


def prompt_type(prompt: str, prompt_bank: dict[str, Any] | None = None) -> str:
    if prompt_bank:
        for item in prompt_bank.get("prompts", []):
            if str(item.get("prompt")) == prompt:
                return str(item.get("prompt_type", "other"))
    value = prompt.lower()
    if "headphone" in value:
        return "headphones"
    if "glasses" in value or "sunglasses" in value:
        return "glasses"
    if "beard" in value or "stubble" in value:
        return "beard"
    if any(token in value for token in ("jacket", "scarf", "hoodie", "shirt")):
        return "clothing"
    if "smile" in value:
        return "smile"
    return "other"


def _metadata_has_flag(metadata: dict[str, Any], *needles: str) -> bool:
    text = " ".join(str(metadata.get(key, "")) for key in ("why", "priority", "notes")).lower()
    return any(needle in text for needle in needles)


def _face_metadata(root: Path, face_id: str) -> dict[str, Any]:
    return read_json(root / "data" / face_id / "metadata.json", {})


def detect_phase3_faces(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return usable face records and a list of missing required files."""
    manifest_cfg = load_image_manifest_config(root)
    data_manifest_path = root / str(manifest_cfg.get("data_manifest", "data/phase3_image_manifest.json"))
    data_records = read_json(data_manifest_path, [])
    if not isinstance(data_records, list):
        data_records = []
    records_by_id = {str(item.get("face_id")): dict(item) for item in data_records if item.get("face_id")}

    face_ids = ["face_001"] + sorted(face_id for face_id in records_by_id if face_id != "face_001")
    if not records_by_id:
        # Keep the error precise for the user: Phase 3 must not silently collapse
        # back to face_001-only.
        face_ids = ["face_001"]
    canonical = str(manifest_cfg.get("canonical_input", "instruct_512.png"))
    manual = dict(manifest_cfg.get("manual_restrictions", {}))
    primary = set(str(x) for x in manifest_cfg.get("primary_face_ids", []))
    secondary = set(str(x) for x in manifest_cfg.get("secondary_prompt_restricted_face_ids", []))
    baseline = set(str(x) for x in manifest_cfg.get("baseline_face_ids", []))

    faces: list[dict[str, Any]] = []
    missing: list[str] = []
    for face_id in face_ids:
        folder = root / "data" / face_id
        metadata_path = folder / "metadata.json"
        input_path = folder / canonical
        master_path = folder / "master_1024.png"
        flux_path = folder / "flux_768.png"
        for path in (metadata_path, input_path, master_path, flux_path):
            if not path.exists():
                missing.append(path.as_posix())
        metadata = _face_metadata(root, face_id)
        restrictions = dict(manual.get(face_id, {}))
        existing_glasses = bool(restrictions.get("existing_glasses", False)) or _metadata_has_flag(metadata, "existing glasses", "with existing glasses")
        existing_facial_hair = (
            bool(restrictions.get("existing_facial_hair", False))
            or _metadata_has_flag(metadata, "existing beard", "facial hair", "with existing glasses/beard")
        )
        if face_id == "face_008":
            # User explicitly asked to treat face_008 as facial-hair-restricted
            # for beard/stubble prompts, even if the metadata emphasizes glasses.
            existing_facial_hair = True
        priority_group = "primary"
        if face_id in secondary:
            priority_group = "secondary_prompt_restricted"
        if face_id in baseline:
            priority_group = "baseline_reference"
        elif face_id not in primary and face_id not in secondary:
            priority_group = "additional"
        faces.append({
            "face_id": face_id,
            "image_path": relative_path(input_path, root),
            "metadata_path": relative_path(metadata_path, root),
            "priority_group": priority_group,
            "existing_glasses": existing_glasses,
            "existing_facial_hair": existing_facial_hair,
            "prompt_restricted": existing_glasses or existing_facial_hair or face_id in secondary,
        })
    return faces, missing


def prompt_is_compatible(face: dict[str, Any], prompt: str, prompt_bank: dict[str, Any]) -> tuple[bool, str]:
    value = prompt.lower()
    avoid_glasses = {str(x).lower() for x in prompt_bank.get("avoid_prompts_for_existing_glasses", [])}
    avoid_facial = {str(x).lower() for x in prompt_bank.get("avoid_prompts_for_existing_facial_hair", [])}
    if bool(face.get("existing_glasses", False)) and value in avoid_glasses:
        return False, "existing_glasses_invalidate_prompt"
    if bool(face.get("existing_facial_hair", False)) and value in avoid_facial:
        return False, "existing_facial_hair_invalidate_prompt"
    return True, "compatible"


def configured_prompts(root: Path) -> list[dict[str, Any]]:
    bank = load_prompt_bank(root)
    prompts = sorted(list(bank.get("prompts", [])), key=lambda item: int(item.get("priority", 999)))
    return [dict(item) for item in prompts]


def phase3_model_payload(config: dict[str, Any], *, image_guidance_scale: float) -> dict[str, Any]:
    payload = dict(config.get("model", {}))
    payload["image_guidance_scale"] = float(image_guidance_scale)
    return payload


__all__ = [
    "config_path",
    "configured_prompts",
    "detect_phase3_faces",
    "done_path",
    "failed_path",
    "load_image_manifest_config",
    "load_phase3_config",
    "load_prompt_bank",
    "mark_done",
    "mark_failed",
    "outputs_root",
    "phase3_model_payload",
    "phase3_root",
    "project_root",
    "prompt_is_compatible",
    "prompt_slug",
    "prompt_type",
    "read_csv",
    "read_json",
    "relative_path",
    "setting_slug",
    "slug",
    "utc_now",
    "write_csv",
    "write_json",
    "write_text",
]

