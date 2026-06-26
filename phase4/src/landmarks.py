"""Face landmark detection and semantic region summaries for Phase 4."""
from __future__ import annotations

import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from phase1.src.data import load_rgb
from phase1.src.reporting import save_rgb

from .utils import relative_path, utc_now, write_json


MEDIAPIPE_REGION_INDICES: dict[str, list[int]] = {
    "left_eye_region": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye_region": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "eyes_region": [33, 133, 159, 145, 263, 362, 386, 374, 6, 168],
    "nose_bridge_region": [6, 168, 8, 9, 197, 195, 5, 4],
    "mouth_region": [61, 291, 13, 14, 17, 0, 78, 308, 81, 178, 402, 311],
    "lower_face_region": [152, 175, 199, 200, 18, 164, 57, 287, 61, 291],
    "jaw_chin_region": [152, 148, 176, 149, 150, 136, 172, 397, 365, 379, 378, 400, 377],
    "face_outline_region": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "head_side_region": [234, 127, 93, 132, 58, 454, 356, 323, 361, 288],
    "head_top_region": [10, 338, 297, 332, 109, 67, 103, 54],
}


TEMPLATE_POINTS: dict[str, tuple[float, float]] = {
    "face_left": (0.22, 0.52),
    "face_right": (0.78, 0.52),
    "face_top": (0.50, 0.17),
    "chin": (0.50, 0.82),
    "jaw_left": (0.34, 0.72),
    "jaw_right": (0.66, 0.72),
    "left_eye_outer": (0.36, 0.42),
    "left_eye_inner": (0.47, 0.43),
    "left_eye_top": (0.41, 0.39),
    "left_eye_bottom": (0.41, 0.46),
    "right_eye_inner": (0.53, 0.43),
    "right_eye_outer": (0.64, 0.42),
    "right_eye_top": (0.59, 0.39),
    "right_eye_bottom": (0.59, 0.46),
    "nose_bridge": (0.50, 0.47),
    "nose_tip": (0.50, 0.56),
    "mouth_left": (0.40, 0.65),
    "mouth_right": (0.60, 0.65),
    "mouth_top": (0.50, 0.62),
    "mouth_bottom": (0.50, 0.69),
    "lower_face_center": (0.50, 0.72),
    "head_left": (0.22, 0.45),
    "head_right": (0.78, 0.45),
    "head_top": (0.50, 0.18),
}


TEMPLATE_REGIONS: dict[str, list[str]] = {
    "left_eye_region": ["left_eye_outer", "left_eye_inner", "left_eye_top", "left_eye_bottom"],
    "right_eye_region": ["right_eye_inner", "right_eye_outer", "right_eye_top", "right_eye_bottom"],
    "eyes_region": ["left_eye_outer", "left_eye_inner", "right_eye_inner", "right_eye_outer", "nose_bridge"],
    "nose_bridge_region": ["nose_bridge", "nose_tip"],
    "mouth_region": ["mouth_left", "mouth_right", "mouth_top", "mouth_bottom"],
    "lower_face_region": ["mouth_left", "mouth_right", "lower_face_center", "chin"],
    "jaw_chin_region": ["jaw_left", "jaw_right", "chin", "lower_face_center"],
    "face_outline_region": ["face_left", "face_right", "face_top", "chin", "jaw_left", "jaw_right"],
    "head_side_region": ["head_left", "head_right", "face_left", "face_right"],
    "head_top_region": ["head_top", "face_top"],
}


def _point_payload(points: np.ndarray, width: int, height: int) -> list[dict[str, float | int]]:
    payload: list[dict[str, float | int]] = []
    for index, (x_norm, y_norm) in enumerate(points):
        payload.append({
            "index": int(index),
            "x_norm": float(x_norm),
            "y_norm": float(y_norm),
            "x": float(x_norm * width),
            "y": float(y_norm * height),
        })
    return payload


def _summarize_region(points: np.ndarray, indices: list[int], width: int, height: int) -> dict[str, Any]:
    valid = [index for index in indices if 0 <= int(index) < len(points)]
    if not valid:
        return {"indices": [], "center": [width * 0.5, height * 0.5], "bbox": [0, 0, width, height], "point_count": 0}
    coords = points[valid].astype(np.float32)
    xs = coords[:, 0] * width
    ys = coords[:, 1] * height
    return {
        "indices": [int(index) for index in valid],
        "center": [float(xs.mean()), float(ys.mean())],
        "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        "point_count": len(valid),
    }


def _semantic_regions(points: np.ndarray, width: int, height: int, detector: str) -> dict[str, dict[str, Any]]:
    if detector == "template":
        regions: dict[str, dict[str, Any]] = {}
        name_to_index = {name: index for index, name in enumerate(TEMPLATE_POINTS)}
        for region_name, names in TEMPLATE_REGIONS.items():
            regions[region_name] = _summarize_region(points, [name_to_index[name] for name in names], width, height)
        return regions
    return {
        region_name: _summarize_region(points, indices, width, height)
        for region_name, indices in MEDIAPIPE_REGION_INDICES.items()
    }


def _template_landmarks() -> np.ndarray:
    return np.asarray(list(TEMPLATE_POINTS.values()), dtype=np.float32)


def _detect_mediapipe(image: Image.Image) -> tuple[np.ndarray | None, str | None]:
    try:
        import mediapipe as mp
    except Exception as error:
        return None, f"mediapipe_import_failed:{type(error).__name__}:{error}"
    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
        )
        try:
            result = face_mesh.process(rgb)
        finally:
            face_mesh.close()
        if not result.multi_face_landmarks:
            return None, "mediapipe_no_face_detected"
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.asarray([[float(item.x), float(item.y)] for item in landmarks], dtype=np.float32)
        points = np.clip(points, 0.0, 1.0)
        return points, None
    except Exception as error:
        return None, f"mediapipe_detection_failed:{type(error).__name__}:{error}"


def detect_landmarks_for_image(image_path: Path, *, prefer_mediapipe: bool = True) -> tuple[dict[str, Any], str | None]:
    image = load_rgb(image_path)
    width, height = image.size
    warning = None
    detector = "template"
    points: np.ndarray | None = None
    if prefer_mediapipe:
        points, warning = _detect_mediapipe(image)
        if points is not None:
            detector = "mediapipe_face_mesh"
    if points is None:
        points = _template_landmarks()
        detector = "template"
    regions = _semantic_regions(points, width, height, "template" if detector == "template" else "mediapipe")
    record = {
        "image_path": str(image_path),
        "detector": detector,
        "detector_warning": warning,
        "image_width": width,
        "image_height": height,
        "landmark_count": int(len(points)),
        "points": _point_payload(points, width, height),
        "regions": regions,
        "created_at": utc_now(),
    }
    return record, warning


def _draw_points(draw: ImageDraw.ImageDraw, points: list[dict[str, Any]], *, color: str, radius: int = 2) -> None:
    for point in points:
        x = float(point["x"])
        y = float(point["y"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, fill=color)


def draw_landmarks_overlay(image: Image.Image, landmarks: dict[str, Any], path: Path, *, title: str | None = None) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    _draw_points(draw, list(landmarks.get("points", [])), color="#00ff88", radius=1)
    if title:
        draw.rectangle((0, 0, min(canvas.width, 420), 18), fill="white")
        draw.text((4, 3), title[:80], fill="black")
    save_rgb(path, canvas)


def draw_regions_overlay(image: Image.Image, landmarks: dict[str, Any], path: Path) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    colors = [
        "#ff4040",
        "#40ff40",
        "#4080ff",
        "#ffcc00",
        "#ff40ff",
        "#00cccc",
        "#ff8840",
        "#8840ff",
        "#00aa55",
        "#aa0055",
    ]
    for index, (name, region) in enumerate(dict(landmarks.get("regions", {})).items()):
        color = colors[index % len(colors)]
        bbox = region.get("bbox", [0, 0, canvas.width, canvas.height])
        left, top, right, bottom = [float(value) for value in bbox]
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        center = region.get("center", [(left + right) * 0.5, (top + bottom) * 0.5])
        x, y = float(center[0]), float(center[1])
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((left + 2, max(0, top - 11)), name.replace("_region", "")[:18], fill=color)
    save_rgb(path, canvas)


def transform_landmark_points(landmarks: dict[str, Any], flow: np.ndarray) -> list[dict[str, Any]]:
    height, width = flow.shape[:2]
    transformed: list[dict[str, Any]] = []
    for point in landmarks.get("points", []):
        x = int(max(0, min(width - 1, round(float(point["x"])))))
        y = int(max(0, min(height - 1, round(float(point["y"])))))
        # apply_flow samples from x+flow, so visible content roughly moves in
        # the opposite direction.  This overlay is diagnostic, not a geometry
        # objective.
        dx, dy = flow[y, x]
        moved = dict(point)
        moved["x"] = float(point["x"]) - float(dx)
        moved["y"] = float(point["y"]) - float(dy)
        moved["x_norm"] = float(moved["x"] / max(width, 1))
        moved["y_norm"] = float(moved["y"] / max(height, 1))
        transformed.append(moved)
    return transformed


def draw_transformed_landmarks_overlay(
    image: Image.Image,
    landmarks: dict[str, Any],
    flow: np.ndarray,
    path: Path,
    *,
    title: str | None = None,
) -> None:
    payload = dict(landmarks)
    payload["points"] = transform_landmark_points(landmarks, flow)
    draw_landmarks_overlay(image, payload, path, title=title)


def save_landmarks_for_face(
    *,
    root: Path,
    face_id: str,
    image_path: Path,
    output: Path,
    prefer_mediapipe: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "face_id": face_id,
        "image_path": relative_path(image_path, root),
        "success": False,
        "detector": None,
        "landmark_count": 0,
        "failure_reason": None,
        "updated_at": utc_now(),
    }
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist")
        record, warning = detect_landmarks_for_image(image_path, prefer_mediapipe=prefer_mediapipe)
        status.update({
            "success": True,
            "detector": record.get("detector"),
            "landmark_count": int(record.get("landmark_count", 0)),
            "failure_reason": warning if record.get("detector") == "template" else None,
        })
        if dry_run:
            return status
        output.mkdir(parents=True, exist_ok=True)
        image = load_rgb(image_path)
        write_json(output / "landmarks.json", record)
        draw_landmarks_overlay(image, record, output / "landmarks_overlay.jpg", title=f"{face_id}: {record.get('detector')}")
        draw_regions_overlay(image, record, output / "regions_overlay.jpg")
        write_json(output / "status.json", status)
        return status
    except Exception as error:
        status.update({
            "success": False,
            "failure_reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "status.json", status)
        return status


__all__ = [
    "detect_landmarks_for_image",
    "draw_landmarks_overlay",
    "draw_regions_overlay",
    "draw_transformed_landmarks_overlay",
    "save_landmarks_for_face",
    "transform_landmark_points",
]

