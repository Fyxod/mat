"""Typed configuration for differentiable geometry optimization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    height: int = 512
    width: int = 512
    torch_dtype: str = "float16"
    seed: int = 1234

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ModelSettings":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometrySettings:
    method: str = "combined_tps_dct"
    max_disp_px: float = 4.0
    dct_size: int = 4
    tps_size: int = 5
    init_scale_px: float = 0.05
    center_x: float = 0.50
    center_y: float = 0.52
    radius_x: float = 0.34
    radius_y: float = 0.42
    edge_falloff: float = 0.12

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttackSettings:
    objective: str
    max_disp_px: float
    target_input_ssim: float
    iterations: int
    learning_rate: float
    objective_scale: float = 1.0
    objective_gradient_balance: float = 1.25
    objective_scale_min: float = 1e-6
    objective_scale_max: float = 1e6
    lambda_visual: float = 30.0
    lambda_disp: float = 0.2
    lambda_smooth: float = 0.4
    lambda_fold: float = 10.0
    checkpoint_every: int = 50
    objective_timestep_index: int = 10
    objective_timestep_indices: tuple[int, ...] | None = None
    geometry_method: str = "combined_tps_dct"
    dct_size: int = 4
    tps_size: int = 5
    face_mask: dict[str, float] | None = None
    seed: int = 1234

    def geometry(self) -> GeometrySettings:
        face = self.face_mask or {}
        return GeometrySettings(
            method=self.geometry_method,
            max_disp_px=self.max_disp_px,
            dct_size=self.dct_size,
            tps_size=self.tps_size,
            center_x=float(face.get("center_x", 0.50)),
            center_y=float(face.get("center_y", 0.52)),
            radius_x=float(face.get("radius_x", 0.34)),
            radius_y=float(face.get("radius_y", 0.42)),
            edge_falloff=float(face.get("edge_falloff", 0.12)),
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self)
