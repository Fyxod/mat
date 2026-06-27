"""Face landmark detection and semantic region summaries for Phase 4."""
from __future__ import annotations

import os
import sys
import traceback
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from phase1.src.data import load_rgb
from phase1.src.reporting import save_rgb

from .utils import relative_path, utc_now, write_json


REAL_LANDMARK_DETECTORS = {
    "mediapipe_solutions_face_mesh",
    "mediapipe_python_solutions_face_mesh",
    "mediapipe_tasks_face_landmarker",
}


MEDIAPIPE_REGION_INDICES: dict[str, list[int]] = {
    "left_eye_region": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye_region": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "eyes_region": [33, 133, 159, 145, 263, 362, 386, 374, 6, 168],
    "nose_bridge_region": [6, 168, 8, 9, 197, 195, 5, 4],
    "mouth_region": [61, 291, 13, 14, 17, 0, 78, 308, 81, 178, 402, 311],
    "lower_face_region": [152, 175, 199, 200, 18, 164, 57, 287, 61, 291],
    "jaw_chin_region": [152, 148, 176, 149, 150, 136, 172, 397, 365, 379, 378, 400, 377],
    "face_outline_region": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
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


def is_real_landmark_detector(detector: str | None) -> bool:
    return str(detector or "") in REAL_LANDMARK_DETECTORS


def is_real_landmark_record(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    try:
        count = int(record.get("landmark_count", 0))
    except Exception:
        count = 0
    success_value = record.get("success", True)
    success = str(success_value).lower() not in {"", "0", "false", "none", "no"}
    return success and is_real_landmark_detector(str(record.get("detector", ""))) and count >= 468


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


def mediapipe_backend_report() -> dict[str, Any]:
    """Return a side-effect-free report of usable MediaPipe Face Mesh backends."""
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "mediapipe_imported": False,
        "mediapipe_version": None,
        "mediapipe_file": None,
        "has_solutions": False,
        "has_solutions_face_mesh": False,
        "mp_solutions_face_mesh_ok": False,
        "direct_python_solutions_face_mesh_ok": False,
        "tasks_face_landmarker_import_ok": False,
        "tasks_model_asset_path": os.environ.get("MEDIAPIPE_FACE_LANDMARKER_TASK"),
        "tasks_model_asset_exists": False,
        "selected_backend": None,
        "errors": [],
    }
    try:
        import mediapipe as mp  # type: ignore

        report["mediapipe_imported"] = True
        report["mediapipe_version"] = getattr(mp, "__version__", None)
        report["mediapipe_file"] = getattr(mp, "__file__", None)
        report["has_solutions"] = hasattr(mp, "solutions")
        if hasattr(mp, "solutions"):
            try:
                face_mesh_module = mp.solutions.face_mesh
                report["has_solutions_face_mesh"] = True
                getattr(face_mesh_module, "FaceMesh")
                report["mp_solutions_face_mesh_ok"] = True
                report["selected_backend"] = report["selected_backend"] or "mediapipe_solutions_face_mesh"
            except Exception as error:
                report["errors"].append(f"mp.solutions.face_mesh_failed:{type(error).__name__}:{error}")
    except Exception as error:
        report["errors"].append(f"mediapipe_import_failed:{type(error).__name__}:{error}")

    try:
        module = import_module("mediapipe.python.solutions.face_mesh")
        getattr(module, "FaceMesh")
        report["direct_python_solutions_face_mesh_ok"] = True
        report["selected_backend"] = report["selected_backend"] or "mediapipe_python_solutions_face_mesh"
    except Exception as error:
        report["errors"].append(f"direct_python_solutions_face_mesh_failed:{type(error).__name__}:{error}")

    task_model = report.get("tasks_model_asset_path")
    if task_model:
        report["tasks_model_asset_exists"] = Path(str(task_model)).exists()
    try:
        from mediapipe.tasks import python as mp_tasks_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_tasks_vision  # type: ignore

        getattr(mp_tasks_python, "BaseOptions")
        getattr(mp_tasks_vision, "FaceLandmarker")
        getattr(mp_tasks_vision, "FaceLandmarkerOptions")
        report["tasks_face_landmarker_import_ok"] = True
        if report.get("tasks_model_asset_exists"):
            report["selected_backend"] = report["selected_backend"] or "mediapipe_tasks_face_landmarker"
    except Exception as error:
        report["errors"].append(f"tasks_face_landmarker_import_failed:{type(error).__name__}:{error}")
    return report


def _points_from_facemesh_result(result: Any) -> np.ndarray | None:
    if not getattr(result, "multi_face_landmarks", None):
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    raw = np.asarray([[float(item.x), float(item.y)] for item in landmarks], dtype=np.float32)
    return np.clip(raw, 0.0, 1.0)


def _detect_with_facemesh_class(image: Image.Image, face_mesh_cls: Any, detector: str) -> tuple[np.ndarray | None, str | None, str]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    face_mesh = face_mesh_cls(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.45,
    )
    try:
        result = face_mesh.process(rgb)
    finally:
        try:
            face_mesh.close()
        except Exception:
            pass
    points = _points_from_facemesh_result(result)
    if points is None:
        return None, f"{detector}:no_face_detected", detector
    return points, None, detector


def _detect_with_tasks_face_landmarker(image: Image.Image) -> tuple[np.ndarray | None, str | None, str]:
    task_model = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_TASK")
    if not task_model or not Path(task_model).exists():
        return None, "mediapipe_tasks_face_landmarker:model_asset_missing", "mediapipe_tasks_face_landmarker"
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_tasks_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_tasks_vision  # type: ignore

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    options = mp_tasks_vision.FaceLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=task_model),
        running_mode=mp_tasks_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp_tasks_vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)
    if not getattr(result, "face_landmarks", None):
        return None, "mediapipe_tasks_face_landmarker:no_face_detected", "mediapipe_tasks_face_landmarker"
    raw = np.asarray([[float(item.x), float(item.y)] for item in result.face_landmarks[0]], dtype=np.float32)
    return np.clip(raw, 0.0, 1.0), None, "mediapipe_tasks_face_landmarker"


def _detect_mediapipe(image: Image.Image) -> tuple[np.ndarray | None, str | None, str | None]:
    errors: list[str] = []
    try:
        import mediapipe as mp  # type: ignore

        try:
            face_mesh_cls = mp.solutions.face_mesh.FaceMesh
            points, warning, detector = _detect_with_facemesh_class(image, face_mesh_cls, "mediapipe_solutions_face_mesh")
            if points is not None:
                return points, warning, detector
            errors.append(str(warning))
        except Exception as error:
            errors.append(f"mediapipe_solutions_face_mesh_failed:{type(error).__name__}:{error}")
    except Exception as error:
        errors.append(f"mediapipe_import_failed:{type(error).__name__}:{error}")

    try:
        module = import_module("mediapipe.python.solutions.face_mesh")
        face_mesh_cls = getattr(module, "FaceMesh")
        points, warning, detector = _detect_with_facemesh_class(image, face_mesh_cls, "mediapipe_python_solutions_face_mesh")
        if points is not None:
            return points, warning, detector
        errors.append(str(warning))
    except Exception as error:
        errors.append(f"mediapipe_python_solutions_face_mesh_failed:{type(error).__name__}:{error}")

    try:
        points, warning, detector = _detect_with_tasks_face_landmarker(image)
        if points is not None:
            return points, warning, detector
        errors.append(str(warning))
    except Exception as error:
        errors.append(f"mediapipe_tasks_face_landmarker_failed:{type(error).__name__}:{error}")

    return None, "; ".join(error for error in errors if error), None


def _sanity_warnings(points: np.ndarray, regions: dict[str, dict[str, Any]], width: int, height: int, detector: str) -> list[str]:
    warnings: list[str] = []
    if len(points) == 0:
        return ["no_landmark_points"]
    inside = np.logical_and.reduce((points[:, 0] >= 0.0, points[:, 0] <= 1.0, points[:, 1] >= 0.0, points[:, 1] <= 1.0))
    inside_ratio = float(np.mean(inside))
    if inside_ratio < 0.95:
        warnings.append(f"landmarks_outside_image_ratio:{1.0 - inside_ratio:.3f}")
    if detector != "template":
        eyes = regions.get("eyes_region", {}).get("center")
        mouth = regions.get("mouth_region", {}).get("center")
        nose = regions.get("nose_bridge_region", {}).get("center")
        if eyes and mouth and float(eyes[1]) >= float(mouth[1]):
            warnings.append("eye_region_not_above_mouth_region")
        if nose and mouth and float(nose[1]) >= float(mouth[1]):
            warnings.append("nose_bridge_not_above_mouth_region")
        if eyes and not (0.20 * height <= float(eyes[1]) <= 0.65 * height):
            warnings.append("eye_region_y_unusual")
        if mouth and not (0.40 * height <= float(mouth[1]) <= 0.90 * height):
            warnings.append("mouth_region_y_unusual")
    return warnings


def detect_landmarks_for_image(
    image_path: Path,
    *,
    prefer_mediapipe: bool = True,
    require_real_landmarks: bool = False,
) -> tuple[dict[str, Any], str | None]:
    image = load_rgb(image_path)
    width, height = image.size
    warning = None
    detector = "template"
    points: np.ndarray | None = None
    if prefer_mediapipe:
        points, warning, backend = _detect_mediapipe(image)
        if points is not None and backend:
            detector = backend
    if points is None:
        if require_real_landmarks:
            raise RuntimeError(f"real_mediapipe_landmarks_required_but_unavailable:{warning}")
        points = _template_landmarks()
        detector = "template"
    real_landmarks = is_real_landmark_detector(detector) and len(points) >= 468
    if require_real_landmarks and not real_landmarks:
        raise RuntimeError(f"real_mediapipe_landmarks_required_but_got:{detector}:{len(points)}")
    regions = _semantic_regions(points, width, height, "template" if detector == "template" else "mediapipe")
    sanity = _sanity_warnings(points, regions, width, height, detector)
    record = {
        "image_path": str(image_path),
        "detector": detector,
        "detector_backend": detector,
        "detector_warning": warning,
        "real_landmarks": real_landmarks,
        "image_width": width,
        "image_height": height,
        "landmark_count": int(len(points)),
        "points": _point_payload(points, width, height),
        "regions": regions,
        "overlay_sanity_warnings": sanity,
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
    radius = 1 if int(landmarks.get("landmark_count", len(landmarks.get("points", [])))) >= 100 else 2
    _draw_points(draw, list(landmarks.get("points", [])), color="#00ff88", radius=radius)
    if title:
        draw.rectangle((0, 0, min(canvas.width, 460), 20), fill="white")
        draw.text((4, 4), title[:92], fill="black")
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
        # the opposite direction. This overlay is diagnostic, not an objective.
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
    require_real_landmarks: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "face_id": face_id,
        "image_path": relative_path(image_path, root),
        "success": False,
        "detector": None,
        "detector_backend": None,
        "real_landmarks": False,
        "landmark_count": 0,
        "failure_reason": None,
        "overlay_sanity_warnings": [],
        "updated_at": utc_now(),
    }
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist")
        record, warning = detect_landmarks_for_image(
            image_path,
            prefer_mediapipe=prefer_mediapipe,
            require_real_landmarks=require_real_landmarks,
        )
        status.update({
            "success": True,
            "detector": record.get("detector"),
            "detector_backend": record.get("detector_backend"),
            "real_landmarks": bool(record.get("real_landmarks", False)),
            "landmark_count": int(record.get("landmark_count", 0)),
            "failure_reason": warning if record.get("detector") == "template" else None,
            "overlay_sanity_warnings": list(record.get("overlay_sanity_warnings", [])),
        })
        if dry_run:
            return status
        output.mkdir(parents=True, exist_ok=True)
        image = load_rgb(image_path)
        write_json(output / "landmarks.json", record)
        draw_landmarks_overlay(image, record, output / "landmarks_overlay.jpg", title=f"{face_id}: {record.get('detector')} ({record.get('landmark_count')})")
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
    "MEDIAPIPE_REGION_INDICES",
    "REAL_LANDMARK_DETECTORS",
    "detect_landmarks_for_image",
    "draw_landmarks_overlay",
    "draw_regions_overlay",
    "draw_transformed_landmarks_overlay",
    "is_real_landmark_detector",
    "is_real_landmark_record",
    "mediapipe_backend_report",
    "save_landmarks_for_face",
    "transform_landmark_points",
]
