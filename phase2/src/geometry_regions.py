"""Prompt-region geometry for Phase 2 final-edit search.

All perturbations produced here are geometric coordinate warps.  The module
does not add pixel noise, patches, learned weights, or content overlays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


Array = np.ndarray


def prompt_region_key(prompt: str) -> str:
    value = prompt.lower()
    if "headphone" in value:
        return "headphones"
    if "sunglasses" in value or "glasses" in value:
        return "glasses"
    if "beard" in value:
        return "beard"
    if any(token in value for token in ("jacket", "scarf", "hoodie", "shirt")):
        return "clothing"
    if "smile" in value:
        return "smile"
    return "default"


def _grid(height: int, width: int) -> tuple[Array, Array]:
    yy, xx = np.mgrid[0:height, 0:width]
    return xx.astype(np.float32), yy.astype(np.float32)


def _smoothstep(value: Array) -> Array:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def region_mask(height: int, width: int, regions: list[dict[str, Any]], edge_falloff: float = 0.12) -> Array:
    yy = (np.arange(height, dtype=np.float32) + 0.5) / max(height, 1)
    xx = (np.arange(width, dtype=np.float32) + 0.5) / max(width, 1)
    y, x = np.meshgrid(yy, xx, indexing="ij")
    mask = np.zeros((height, width), dtype=np.float32)
    edge = max(float(edge_falloff), 1e-4)
    for region in regions:
        cx = float(region.get("center_x", 0.5))
        cy = float(region.get("center_y", 0.5))
        rx = max(float(region.get("radius_x", 0.25)), 1e-4)
        ry = max(float(region.get("radius_y", 0.25)), 1e-4)
        radius = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        local = 1.0 - _smoothstep((radius - (1.0 - edge)) / edge)
        mask = np.maximum(mask, local.astype(np.float32))
    return np.clip(mask, 0.0, 1.0)


def border_mask(height: int, width: int, edge_falloff: float = 0.08) -> Array:
    yy = np.minimum(np.arange(height), np.arange(height - 1, -1, -1)).astype(np.float32)
    xx = np.minimum(np.arange(width), np.arange(width - 1, -1, -1)).astype(np.float32)
    distance = np.minimum(yy[:, None], xx[None, :])
    edge_px = max(4.0, min(height, width) * float(edge_falloff))
    return _smoothstep(distance / edge_px).astype(np.float32)


def _control_points(size: int) -> Array:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _tps_kernel(distances: Array) -> Array:
    radius2 = distances * distances
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = radius2 * np.log(radius2)
    kernel[~np.isfinite(kernel)] = 0.0
    return kernel


def _tps_field(height: int, width: int, control_values: Array, size: int) -> Array:
    points = _control_points(size)
    count = len(points)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    system = np.zeros((count + 3, count + 3), dtype=np.float64)
    system[:count, :count] = _tps_kernel(distance) + np.eye(count) * 1e-5
    polynomial = np.column_stack([np.ones(count), points])
    system[:count, count:] = polynomial
    system[count:, :count] = polynomial.T
    weights: list[Array] = []
    for channel in range(2):
        rhs = np.concatenate([control_values[:, channel], np.zeros(3)])
        weights.append(np.linalg.solve(system, rhs))
    xx, yy = _grid(height, width)
    query = np.column_stack([xx.ravel() / max(width - 1, 1), yy.ravel() / max(height - 1, 1)])
    query_distance = np.linalg.norm(query[:, None, :] - points[None, :, :], axis=2)
    evaluation = np.column_stack([_tps_kernel(query_distance), np.ones(len(query)), query])
    return np.column_stack([evaluation @ weights[0], evaluation @ weights[1]]).reshape(height, width, 2).astype(np.float32)


def _dct_field(height: int, width: int, modes: list[list[int]], coeffs: Array) -> Array:
    yy = (np.arange(height, dtype=np.float32) + 0.5) / max(height, 1)
    xx = (np.arange(width, dtype=np.float32) + 0.5) / max(width, 1)
    y, x = np.meshgrid(yy, xx, indexing="ij")
    field = np.zeros((height, width, 2), dtype=np.float32)
    for index, (ky, kx) in enumerate(modes):
        basis = np.cos(math.pi * int(ky) * y) * np.cos(math.pi * int(kx) * x)
        field[:, :, 0] += float(coeffs[index, 0]) * basis
        field[:, :, 1] += float(coeffs[index, 1]) * basis
    return field


def _mesh_field(height: int, width: int, controls: Array, size: int) -> Array:
    low = controls.reshape(size, size, 2).astype(np.float32)
    return cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _normalize(field: Array) -> Array:
    magnitude = np.sqrt(np.sum(field * field, axis=2))
    scale = max(float(np.max(magnitude)), 1e-6)
    return (field / scale).astype(np.float32)


def _cap_flow(flow: Array, max_disp_px: float) -> Array:
    magnitude = np.sqrt(np.sum(flow * flow, axis=2))
    scale = np.minimum(1.0, (0.999 * float(max_disp_px)) / np.maximum(magnitude, 1e-6))
    return (flow * scale[:, :, None]).astype(np.float32)


@dataclass(frozen=True)
class GeometryCodec:
    tps_size: int = 4
    mesh_size: int = 4
    dct_modes: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1))
    edge_falloff: float = 0.12

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeometryCodec":
        modes = tuple(tuple(int(v) for v in item) for item in config.get("dct_modes", [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 1]]))
        return cls(
            tps_size=int(config.get("tps_size", 4)),
            mesh_size=int(config.get("mesh_size", 4)),
            dct_modes=modes,
            edge_falloff=float(config.get("edge_falloff", 0.12)),
        )

    @property
    def parameter_names(self) -> list[str]:
        names = ["scale_logit"]
        for prefix, size in (("tps", self.tps_size), ("mesh", self.mesh_size)):
            for y in range(size):
                for x in range(size):
                    names.extend([f"{prefix}_{y}_{x}_dx", f"{prefix}_{y}_{x}_dy"])
        for ky, kx in self.dct_modes:
            names.extend([f"dct_{ky}_{kx}_dx", f"dct_{ky}_{kx}_dy"])
        return names

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    def decode(
        self,
        theta: Array,
        *,
        method: str,
        height: int,
        width: int,
        max_disp_px: float,
        regions: list[dict[str, Any]],
    ) -> tuple[Array, dict[str, Any]]:
        theta = np.asarray(theta, dtype=np.float32)
        index = 0
        scale_factor = float(0.15 + 0.85 * (np.tanh(theta[index]) * 0.5 + 0.5))
        index += 1
        tps_count = self.tps_size * self.tps_size
        tps_controls = np.tanh(theta[index:index + 2 * tps_count]).reshape(tps_count, 2).astype(np.float32)
        index += 2 * tps_count
        mesh_count = self.mesh_size * self.mesh_size
        mesh_controls = np.tanh(theta[index:index + 2 * mesh_count]).reshape(mesh_count, 2).astype(np.float32)
        index += 2 * mesh_count
        dct_coeffs = np.tanh(theta[index:index + 2 * len(self.dct_modes)]).reshape(len(self.dct_modes), 2).astype(np.float32)

        # Keep the outer control ring quiet so the local warp stays smooth.
        for controls, size in ((tps_controls, self.tps_size), (mesh_controls, self.mesh_size)):
            grid = controls.reshape(size, size, 2)
            grid[0, :, :] = 0.0
            grid[-1, :, :] = 0.0
            grid[:, 0, :] = 0.0
            grid[:, -1, :] = 0.0

        tps = _normalize(_tps_field(height, width, tps_controls, self.tps_size))
        mesh = _normalize(_mesh_field(height, width, mesh_controls, self.mesh_size))
        dct = _normalize(_dct_field(height, width, [list(mode) for mode in self.dct_modes], dct_coeffs))
        if method == "region_local_tps":
            raw = tps
        elif method == "region_local_dct":
            raw = dct
        elif method == "region_local_mesh":
            raw = mesh
        elif method == "combined_tps_dct":
            raw = 0.65 * tps + 0.45 * dct
        elif method == "combined_all":
            raw = 0.50 * tps + 0.35 * dct + 0.35 * mesh
        else:
            raise ValueError(f"Unsupported Phase 2 geometry method: {method}")

        local = region_mask(height, width, regions, self.edge_falloff) * border_mask(height, width, self.edge_falloff)
        flow = _normalize(raw) * (0.98 * float(max_disp_px) * scale_factor)
        flow = flow * local[:, :, None]
        flow = _cap_flow(flow, float(max_disp_px))
        info = {
            "geometry_method": method,
            "scale_factor": scale_factor,
            "region_count": len(regions),
            "tps_size": self.tps_size,
            "mesh_size": self.mesh_size,
            "dct_modes": [list(mode) for mode in self.dct_modes],
        }
        return flow.astype(np.float32), info


def apply_flow(image: Image.Image, flow: Array) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    xx, yy = _grid(height, width)
    map_x = (xx + flow[:, :, 0]).astype(np.float32)
    map_y = (yy + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(array, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped, mode="RGB")


def displacement_stats(flow: Array) -> dict[str, float]:
    magnitude = np.sqrt(np.sum(flow * flow, axis=2))
    return {
        "max_disp_px": float(np.max(magnitude)),
        "mean_disp_px": float(np.mean(magnitude)),
        "p95_disp_px": float(np.percentile(magnitude, 95.0)),
    }


def flow_to_image(flow: Array) -> Image.Image:
    magnitude = np.sqrt(np.sum(flow * flow, axis=2))
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((angle + math.pi) / (2.0 * math.pi) * 179.0).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(magnitude / max(float(np.max(magnitude)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mode="RGB")


def regions_for_prompt(prompt: str, region_config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    key = prompt_region_key(prompt)
    payload = region_config.get(key) or region_config.get("default") or {"regions": []}
    regions = list(payload.get("regions", []))
    if not regions:
        regions = [{"center_x": 0.50, "center_y": 0.52, "radius_x": 0.34, "radius_y": 0.42}]
    return key, regions


__all__ = [
    "GeometryCodec",
    "apply_flow",
    "displacement_stats",
    "flow_to_image",
    "prompt_region_key",
    "region_mask",
    "regions_for_prompt",
]
