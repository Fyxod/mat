"""Landmark semantic coordinate warps for Phase 4.

This module produces only geometric coordinate transformations.  It never
modifies RGB values directly, adds patches, or trains model weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from phase2.src.geometry_regions import apply_flow, displacement_stats, flow_to_image

from .semantic_actions import action_parameter_names, decode_action_values


Array = np.ndarray


def _grid(height: int, width: int) -> tuple[Array, Array]:
    yy, xx = np.mgrid[0:height, 0:width]
    return xx.astype(np.float32), yy.astype(np.float32)


def _region_center(record: dict[str, Any], name: str, default: tuple[float, float]) -> np.ndarray:
    region = dict(record.get("regions", {})).get(name, {})
    center = region.get("center")
    if center and len(center) >= 2:
        return np.asarray([float(center[0]), float(center[1])], dtype=np.float32)
    return np.asarray(default, dtype=np.float32)


def _region_bbox(record: dict[str, Any], name: str, default: tuple[float, float, float, float]) -> np.ndarray:
    region = dict(record.get("regions", {})).get(name, {})
    bbox = region.get("bbox")
    if bbox and len(bbox) >= 4:
        return np.asarray([float(value) for value in bbox[:4]], dtype=np.float32)
    return np.asarray(default, dtype=np.float32)


def _bbox_anchors(bbox: Array) -> dict[str, np.ndarray]:
    left, top, right, bottom = [float(value) for value in bbox]
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    return {
        "center": np.asarray([cx, cy], dtype=np.float32),
        "left": np.asarray([left, cy], dtype=np.float32),
        "right": np.asarray([right, cy], dtype=np.float32),
        "top": np.asarray([cx, top], dtype=np.float32),
        "bottom": np.asarray([cx, bottom], dtype=np.float32),
    }


def semantic_anchors(record: dict[str, Any]) -> dict[str, np.ndarray]:
    width = float(record.get("image_width", 512))
    height = float(record.get("image_height", 512))
    left_eye = _bbox_anchors(_region_bbox(record, "left_eye_region", (0.34 * width, 0.39 * height, 0.48 * width, 0.47 * height)))
    right_eye = _bbox_anchors(_region_bbox(record, "right_eye_region", (0.52 * width, 0.39 * height, 0.66 * width, 0.47 * height)))
    mouth = _bbox_anchors(_region_bbox(record, "mouth_region", (0.40 * width, 0.61 * height, 0.60 * width, 0.70 * height)))
    jaw = _bbox_anchors(_region_bbox(record, "jaw_chin_region", (0.32 * width, 0.68 * height, 0.68 * width, 0.84 * height)))
    outline = _bbox_anchors(_region_bbox(record, "face_outline_region", (0.22 * width, 0.18 * height, 0.78 * width, 0.84 * height)))
    head_side = _bbox_anchors(_region_bbox(record, "head_side_region", (0.20 * width, 0.34 * height, 0.80 * width, 0.62 * height)))
    head_top = _bbox_anchors(_region_bbox(record, "head_top_region", (0.34 * width, 0.14 * height, 0.66 * width, 0.28 * height)))
    lower_face = _bbox_anchors(_region_bbox(record, "lower_face_region", (0.36 * width, 0.62 * height, 0.64 * width, 0.80 * height)))
    nose_bridge = _region_center(record, "nose_bridge_region", (0.50 * width, 0.47 * height))
    return {
        "left_eye_center": left_eye["center"],
        "left_eye_outer": left_eye["left"],
        "left_eye_inner": left_eye["right"],
        "left_eye_top": left_eye["top"],
        "left_eye_bottom": left_eye["bottom"],
        "right_eye_center": right_eye["center"],
        "right_eye_inner": right_eye["left"],
        "right_eye_outer": right_eye["right"],
        "right_eye_top": right_eye["top"],
        "right_eye_bottom": right_eye["bottom"],
        "nose_bridge": nose_bridge,
        "mouth_center": mouth["center"],
        "mouth_left": mouth["left"],
        "mouth_right": mouth["right"],
        "mouth_top": mouth["top"],
        "mouth_bottom": mouth["bottom"],
        "chin": jaw["bottom"],
        "jaw_left": jaw["left"],
        "jaw_right": jaw["right"],
        "lower_face_center": lower_face["center"],
        "face_left": outline["left"],
        "face_right": outline["right"],
        "head_left": head_side["left"],
        "head_right": head_side["right"],
        "head_top": head_top["top"],
    }


def _add(acc: dict[str, np.ndarray], key: str, dx: float = 0.0, dy: float = 0.0) -> None:
    acc.setdefault(key, np.zeros(2, dtype=np.float32))
    acc[key] += np.asarray([dx, dy], dtype=np.float32)


def action_displacements(
    *,
    record: dict[str, Any],
    action_values: dict[str, float],
    max_disp_px: float,
    strength: float,
) -> tuple[Array, Array, dict[str, Any]]:
    anchors = semantic_anchors(record)
    disp: dict[str, np.ndarray] = {}
    scale = float(max_disp_px) * float(strength)
    for action, value in action_values.items():
        amount = float(value) * scale
        if action == "eye_spread":
            _add(disp, "left_eye_center", -amount, 0)
            _add(disp, "left_eye_outer", -amount, 0)
            _add(disp, "right_eye_center", amount, 0)
            _add(disp, "right_eye_outer", amount, 0)
        elif action == "eye_compress":
            _add(disp, "left_eye_center", amount, 0)
            _add(disp, "left_eye_outer", amount, 0)
            _add(disp, "right_eye_center", -amount, 0)
            _add(disp, "right_eye_outer", -amount, 0)
        elif action == "eye_raise_left":
            _add(disp, "left_eye_center", 0, -amount)
            _add(disp, "left_eye_top", 0, -amount)
            _add(disp, "left_eye_bottom", 0, -0.5 * amount)
        elif action == "eye_raise_right":
            _add(disp, "right_eye_center", 0, -amount)
            _add(disp, "right_eye_top", 0, -amount)
            _add(disp, "right_eye_bottom", 0, -0.5 * amount)
        elif action == "eye_squint":
            _add(disp, "left_eye_top", 0, amount)
            _add(disp, "left_eye_bottom", 0, -amount)
            _add(disp, "right_eye_top", 0, amount)
            _add(disp, "right_eye_bottom", 0, -amount)
        elif action == "nose_bridge_shift_x":
            _add(disp, "nose_bridge", amount, 0)
        elif action == "mouth_corner_raise":
            _add(disp, "mouth_left", 0, -amount)
            _add(disp, "mouth_right", 0, -amount)
        elif action == "mouth_corner_lower":
            _add(disp, "mouth_left", 0, amount)
            _add(disp, "mouth_right", 0, amount)
        elif action == "mouth_width_expand":
            _add(disp, "mouth_left", -amount, 0)
            _add(disp, "mouth_right", amount, 0)
        elif action == "mouth_width_compress":
            _add(disp, "mouth_left", amount, 0)
            _add(disp, "mouth_right", -amount, 0)
        elif action == "chin_raise":
            _add(disp, "chin", 0, -amount)
            _add(disp, "lower_face_center", 0, -0.5 * amount)
        elif action == "chin_lower":
            _add(disp, "chin", 0, amount)
            _add(disp, "lower_face_center", 0, 0.5 * amount)
        elif action == "jaw_expand":
            _add(disp, "jaw_left", -amount, 0)
            _add(disp, "jaw_right", amount, 0)
        elif action == "jaw_compress":
            _add(disp, "jaw_left", amount, 0)
            _add(disp, "jaw_right", -amount, 0)
        elif action == "face_side_push_left":
            _add(disp, "head_left", -amount, 0)
            _add(disp, "face_left", -0.75 * amount, 0)
        elif action == "face_side_push_right":
            _add(disp, "head_right", amount, 0)
            _add(disp, "face_right", 0.75 * amount, 0)
        elif action == "head_top_raise":
            _add(disp, "head_top", 0, -amount)
        elif action == "head_top_lower":
            _add(disp, "head_top", 0, amount)
        elif action == "asymmetry_twist":
            _add(disp, "head_left", 0, -amount)
            _add(disp, "jaw_left", 0, -0.5 * amount)
            _add(disp, "head_right", 0, amount)
            _add(disp, "jaw_right", 0, 0.5 * amount)
        elif action == "lower_face_shift":
            _add(disp, "lower_face_center", amount, 0)
            _add(disp, "chin", 0.75 * amount, 0)

    source: list[np.ndarray] = []
    displacement: list[np.ndarray] = []
    for key, vector in disp.items():
        if key in anchors:
            source.append(anchors[key])
            displacement.append(vector)
    width = float(record.get("image_width", 512))
    height = float(record.get("image_height", 512))
    stabilizers = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (width * 0.5, 0),
        (width * 0.5, height - 1),
        (0, height * 0.5),
        (width - 1, height * 0.5),
    ]
    for x, y in stabilizers:
        source.append(np.asarray([x, y], dtype=np.float32))
        displacement.append(np.zeros(2, dtype=np.float32))
    if not source:
        source = [np.asarray([width * 0.5, height * 0.5], dtype=np.float32)]
        displacement = [np.zeros(2, dtype=np.float32)]
    info = {
        "active_anchor_count": len(disp),
        "control_point_count": len(source),
        "active_actions": {key: float(value) for key, value in action_values.items() if abs(float(value)) > 1e-5},
    }
    return np.stack(source).astype(np.float32), np.stack(displacement).astype(np.float32), info


def _cap_flow(flow: Array, max_disp_px: float) -> Array:
    magnitude = np.sqrt(np.sum(flow * flow, axis=2))
    scale = np.minimum(1.0, (0.999 * float(max_disp_px)) / np.maximum(magnitude, 1e-6))
    return (flow * scale[:, :, None]).astype(np.float32)


def _tps_kernel(distance: Array) -> Array:
    radius2 = distance * distance
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = radius2 * np.log(radius2)
    kernel[~np.isfinite(kernel)] = 0.0
    return kernel


def _tps_flow(height: int, width: int, source: Array, displacement: Array, regularization: float) -> Array:
    source64 = source.astype(np.float64)
    count = len(source64)
    distance = np.linalg.norm(source64[:, None, :] - source64[None, :, :], axis=2)
    system = np.zeros((count + 3, count + 3), dtype=np.float64)
    system[:count, :count] = _tps_kernel(distance) + np.eye(count) * float(regularization)
    polynomial = np.column_stack([np.ones(count), source64])
    system[:count, count:] = polynomial
    system[count:, :count] = polynomial.T
    xx, yy = _grid(height, width)
    query = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    query_distance = np.linalg.norm(query[:, None, :] - source64[None, :, :], axis=2)
    evaluation = np.column_stack([_tps_kernel(query_distance), np.ones(len(query)), query])
    channels: list[Array] = []
    for channel in range(2):
        rhs = np.concatenate([displacement[:, channel].astype(np.float64), np.zeros(3)])
        weights = np.linalg.solve(system, rhs)
        channels.append((evaluation @ weights).reshape(height, width))
    return np.stack(channels, axis=2).astype(np.float32)


def _piecewise_affine_flow(height: int, width: int, source: Array, displacement: Array) -> Array:
    from scipy.spatial import Delaunay

    target = (source + displacement).astype(np.float32)
    triangulation = Delaunay(target)
    xx, yy = _grid(height, width)
    map_x = xx.copy()
    map_y = yy.copy()
    for simplex in triangulation.simplices:
        dst_tri = target[simplex].astype(np.float32)
        src_tri = source[simplex].astype(np.float32)
        matrix = cv2.getAffineTransform(dst_tri, src_tri)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_tri).astype(np.int32), 1)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), np.ones(len(xs), dtype=np.float32)])
        mapped = coords @ matrix.T
        map_x[ys, xs] = mapped[:, 0]
        map_y[ys, xs] = mapped[:, 1]
    return np.stack([map_x - xx, map_y - yy], axis=2).astype(np.float32)


@dataclass(frozen=True)
class LandmarkGeometryCodec:
    action_names: tuple[str, ...]
    action_strength: float = 0.72
    tps_regularization: float = 1e-4

    @classmethod
    def from_config(cls, *, action_names: list[str], geometry_config: dict[str, Any]) -> "LandmarkGeometryCodec":
        return cls(
            action_names=tuple(action_names),
            action_strength=float(geometry_config.get("action_strength", 0.72)),
            tps_regularization=float(geometry_config.get("tps_regularization", 1e-4)),
        )

    @property
    def parameter_names(self) -> list[str]:
        return action_parameter_names(list(self.action_names))

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    def decode(
        self,
        theta: Array,
        *,
        method: str,
        landmarks: dict[str, Any],
        height: int,
        width: int,
        max_disp_px: float,
    ) -> tuple[Array, dict[str, Any]]:
        theta = np.asarray(theta, dtype=np.float32)
        scale_factor = float(0.20 + 0.80 * (np.tanh(float(theta[0])) * 0.5 + 0.5)) if len(theta) else 1.0
        action_values = decode_action_values(theta, list(self.action_names), start_index=1, scale=scale_factor)
        source, displacement, info = action_displacements(
            record=landmarks,
            action_values=action_values,
            max_disp_px=float(max_disp_px),
            strength=self.action_strength,
        )
        if method == "landmark_semantic_tps":
            flow = _tps_flow(height, width, source, displacement, self.tps_regularization)
        elif method == "landmark_piecewise_affine":
            flow = _piecewise_affine_flow(height, width, source, displacement)
        else:
            raise ValueError(f"Unsupported Phase 4 geometry method: {method}")
        flow = _cap_flow(flow.astype(np.float32), float(max_disp_px))
        info.update({
            "geometry_method": method,
            "scale_factor": scale_factor,
            "action_names": list(self.action_names),
            "action_values": action_values,
            "max_disp_px_budget": float(max_disp_px),
            "tps_regularization": self.tps_regularization,
            "action_strength": self.action_strength,
            "source_control_points": source.tolist(),
            "target_control_points": (source + displacement).tolist(),
        })
        return flow, info


__all__ = [
    "LandmarkGeometryCodec",
    "apply_flow",
    "displacement_stats",
    "flow_to_image",
    "semantic_anchors",
]

