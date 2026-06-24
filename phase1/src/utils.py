"""Small, cross-platform persistence helpers used by every Phase 1 stage."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def project_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "data" / "face_001" / "instruct_512.png").exists():
        raise FileNotFoundError(
            f"{root} is not a MAT project root: data/face_001/instruct_512.png is missing."
        )
    return root


def phase_root(root: Path) -> Path:
    return root / "phase1"


def outputs_root(root: Path) -> Path:
    path = phase_root(root) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def append_run_note(root: Path, title: str, body: str) -> None:
    path = outputs_root(root) / "summaries" / "phase1_run_notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = utc_now()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## {title}\n\n- Time: {stamp}\n- {body}\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = list(rows)
    keys = sorted({key for row in data for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def slug(value: str, limit: int = 76) -> str:
    output = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return (output or "item")[:limit]


def done_path(folder: Path) -> Path:
    return folder / "DONE.json"


def failed_path(folder: Path) -> Path:
    return folder / "FAILED.json"


def is_complete(folder: Path) -> bool:
    return done_path(folder).exists()


def mark_done(folder: Path, payload: dict[str, Any] | None = None) -> None:
    data = {"status": "completed", "completed_at": utc_now(), **(payload or {})}
    write_json(done_path(folder), data)
    failed_path(folder).unlink(missing_ok=True)


def mark_failed(folder: Path, error: BaseException | str) -> None:
    data = {"status": "failed", "failed_at": utc_now(), "error": str(error)}
    write_json(failed_path(folder), data)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def require_selected_prompts(root: Path) -> list[dict[str, Any]]:
    path = phase_root(root) / "configs" / "phase1_selected_prompts.json"
    payload = read_json(path, {})
    selected = list(payload.get("selected_prompts", []))
    if len(selected) < 3:
        raise RuntimeError(
            "At least three selected prompts are required. Run prompt discovery first; "
            "if it completed with fewer than three usable prompts, inspect its report."
        )
    return selected
