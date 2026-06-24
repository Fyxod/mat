"""Differentiable face-local TPS/DCT geometric warps; no pixel perturbation path."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import GeometrySettings


def _dct_basis(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    basis: list[torch.Tensor] = []
    for ky in range(size):
        for kx in range(size):
            if ky == 0 and kx == 0:
                continue
            value = torch.cos(math.pi * ky * (yy + 0.5) / height) * torch.cos(
                math.pi * kx * (xx + 0.5) / width
            )
            basis.append(value / value.square().mean().sqrt().clamp_min(1e-6))
    return torch.stack(basis)


def _tps_evaluator(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    control = torch.stack([x.flatten(), y.flatten()], dim=-1)
    count = control.shape[0]
    distances = (control[:, None] - control[None, :]).square().sum(dim=-1)
    kernel = distances * torch.log(distances.clamp_min(1e-8))
    kernel = kernel + torch.eye(count, device=device) * 1e-5
    polynomial = torch.cat([torch.ones(count, 1, device=device), control], dim=1)
    system = torch.cat(
        [
            torch.cat([kernel, polynomial], dim=1),
            torch.cat([polynomial.T, torch.zeros(3, 3, device=device)], dim=1),
        ],
        dim=0,
    )
    yq, xq = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    query = torch.stack([xq.flatten(), yq.flatten()], dim=-1)
    qdist = (query[:, None] - control[None, :]).square().sum(dim=-1)
    qkernel = qdist * torch.log(qdist.clamp_min(1e-8))
    evaluation = torch.cat([qkernel, torch.ones(len(query), 1, device=device), query], dim=1)
    return evaluation @ torch.linalg.inv(system)[:, :count]


class DifferentiableGeometry(torch.nn.Module):
    """A differentiable local flow field parameterized by TPS and/or low-frequency DCT."""

    def __init__(self, height: int, width: int, settings: GeometrySettings, device: torch.device, seed: int):
        super().__init__()
        self.height = height
        self.width = width
        self.settings = settings
        generator = torch.Generator(device=device).manual_seed(seed)

        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1).unsqueeze(0))

        cx = 2.0 * settings.center_x - 1.0
        cy = 2.0 * settings.center_y - 1.0
        rx = max(2.0 * settings.radius_x, 1e-4)
        ry = max(2.0 * settings.radius_y, 1e-4)
        radius = torch.sqrt(((xx - cx) / rx).square() + ((yy - cy) / ry).square())
        edge = max(settings.edge_falloff, 1e-3)
        transition = ((radius - (1.0 - edge)) / edge).clamp(0.0, 1.0)
        local = 1.0 - transition.square() * (3.0 - 2.0 * transition)
        self.register_buffer("local_mask", local.unsqueeze(0).unsqueeze(0))

        border_distance = torch.minimum(
            torch.minimum(torch.arange(width, device=device)[None], torch.arange(width - 1, -1, -1, device=device)[None]),
            torch.minimum(torch.arange(height, device=device)[:, None], torch.arange(height - 1, -1, -1, device=device)[:, None]),
        ).float()
        edge_px = max(4.0, min(height, width) * settings.edge_falloff)
        border = (border_distance / edge_px).clamp(0.0, 1.0)
        self.register_buffer("border_mask", (border.square() * (3.0 - 2.0 * border)).unsqueeze(0).unsqueeze(0))

        method = settings.method
        if method not in {"face_local_tps", "dct_lowfreq", "combined_tps_dct"}:
            raise ValueError(f"Unsupported geometry method: {method}")

        if method in {"face_local_tps", "combined_tps_dct"}:
            raw = torch.randn(
                1, 2, settings.tps_size, settings.tps_size, device=device, generator=generator
            ) * settings.init_scale_px
            boundary = torch.ones_like(raw)
            boundary[:, :, 0, :] = 0.0
            boundary[:, :, -1, :] = 0.0
            boundary[:, :, :, 0] = 0.0
            boundary[:, :, :, -1] = 0.0
            self.tps_raw = torch.nn.Parameter(raw * boundary)
            self.register_buffer("tps_boundary", boundary)
            self.register_buffer("tps_eval", _tps_evaluator(settings.tps_size, height, width, device))

        if method in {"dct_lowfreq", "combined_tps_dct"}:
            basis = _dct_basis(settings.dct_size, height, width, device)
            self.register_buffer("dct_basis", basis)
            self.dct_raw = torch.nn.Parameter(
                torch.randn(2, basis.shape[0], device=device, generator=generator) * settings.init_scale_px
            )

    def _bound(self, value: torch.Tensor) -> torch.Tensor:
        cap = max(self.settings.max_disp_px, 1e-6)
        return cap * torch.tanh(value / cap)

    def displacement(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fields: dict[str, torch.Tensor] = {}
        if hasattr(self, "tps_raw"):
            control = self._bound(self.tps_raw * self.tps_boundary).reshape(1, 2, -1)
            tps = torch.einsum("pn,bcn->bcp", self.tps_eval, control).reshape(
                1, 2, self.height, self.width
            )
            fields["face_local_tps"] = self._bound(tps)
        if hasattr(self, "dct_raw"):
            dct = torch.einsum("ck,khw->chw", self.dct_raw, self.dct_basis).unsqueeze(0)
            fields["dct_lowfreq"] = self._bound(dct)

        total = sum(fields.values())
        total = total * self.local_mask * self.border_mask
        # Add epsilon before sqrt rather than clamping after it. The latter has
        # an undefined 0 * infinity backward path at the masked border and
        # turns otherwise valid geometry gradients into NaNs.
        magnitude = torch.sqrt(total.square().sum(dim=1, keepdim=True) + 1e-12)
        total = total * torch.clamp(self.settings.max_disp_px / magnitude, max=1.0)
        return total, {name: value * self.local_mask * self.border_mask for name, value in fields.items()}

    def warp(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        displacement, fields = self.displacement()
        grid = self.base_grid.clone()
        grid[..., 0] += 2.0 * displacement[:, 0] / max(self.width - 1, 1)
        grid[..., 1] += 2.0 * displacement[:, 1] / max(self.height - 1, 1)
        warped = F.grid_sample(
            image, grid, mode="bilinear", padding_mode="reflection", align_corners=True
        ).clamp(0.0, 1.0)
        return warped, displacement, fields

    def project_(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.nan_to_num_(0.0).clamp_(-3.0 * self.settings.max_disp_px, 3.0 * self.settings.max_disp_px)

    def resolved_config(self) -> dict[str, Any]:
        return asdict(self.settings)


def total_variation(displacement: torch.Tensor) -> torch.Tensor:
    horizontal = (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]).abs().mean()
    vertical = (displacement[:, :, 1:, :] - displacement[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def jacobian_penalty(displacement: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    dxdx = F.pad((dx[:, :, 2:] - dx[:, :, :-2]) / 2.0, (1, 1))
    dxdy = F.pad((dx[:, 2:, :] - dx[:, :-2, :]) / 2.0, (0, 0, 1, 1))
    dydx = F.pad((dy[:, :, 2:] - dy[:, :, :-2]) / 2.0, (1, 1))
    dydy = F.pad((dy[:, 2:, :] - dy[:, :-2, :]) / 2.0, (0, 0, 1, 1))
    determinant = (1.0 + dxdx) * (1.0 + dydy) - dxdy * dydx
    penalty = F.relu(0.2 - determinant).square().mean()
    return penalty, {
        "jacobian_det_min": float(determinant.detach().float().min().cpu()),
        "jacobian_det_mean": float(determinant.detach().float().mean().cpu()),
        "foldover_fraction_det_below_0": float((determinant < 0).float().mean().detach().cpu()),
        "low_det_fraction_below_0p2": float((determinant < 0.2).float().mean().detach().cpu()),
    }


def displacement_stats(displacement: torch.Tensor) -> dict[str, float]:
    magnitude = displacement.detach().float().square().sum(dim=1).sqrt()
    return {
        "max_disp_px": float(magnitude.max().cpu()),
        "mean_disp_px": float(magnitude.mean().cpu()),
        "p95_disp_px": float(torch.quantile(magnitude.flatten(), 0.95).cpu()),
    }


def flow_to_image(displacement: torch.Tensor) -> Image.Image:
    flow = displacement.detach().float().cpu()[0]
    magnitude = flow.square().sum(dim=0).sqrt()
    scale = max(float(magnitude.max()), 1e-6)
    array = np.stack(
        [
            (flow[0] / scale * 0.5 + 0.5).numpy(),
            (flow[1] / scale * 0.5 + 0.5).numpy(),
            (magnitude / scale).numpy(),
        ],
        axis=-1,
    )
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
