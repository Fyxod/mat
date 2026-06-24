"""Adam optimization of only geometric transformation parameters."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from .config import AttackSettings
from .geometry import DifferentiableGeometry, displacement_stats, jacobian_penalty, total_variation
from .instruct_pipeline import InternalReference, internal_objective


@dataclass
class Snapshot:
    iteration: int
    image: torch.Tensor
    displacement: torch.Tensor
    state_dict: dict[str, torch.Tensor]
    objective: float


@dataclass
class OptimizationResult:
    geometry: DifferentiableGeometry
    history: list[dict[str, Any]]
    snapshots: dict[int, Snapshot]
    best: Snapshot


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _gradient_norm(parameters) -> float:
    pieces = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not pieces else float(torch.stack(pieces).sum().sqrt().cpu())


def optimize_geometry(
    pipe,
    original_tensor: torch.Tensor,
    reference: InternalReference,
    settings: AttackSettings,
) -> OptimizationResult:
    device = original_tensor.device
    geometry = DifferentiableGeometry(
        original_tensor.shape[-2],
        original_tensor.shape[-1],
        settings.geometry(),
        device,
        settings.seed,
    )
    optimizer = torch.optim.Adam(geometry.parameters(), lr=settings.learning_rate)
    history: list[dict[str, Any]] = []
    snapshots: dict[int, Snapshot] = {}
    best: Snapshot | None = None
    started = time.monotonic()
    blank_mse = original_tensor.square().mean().detach()

    def snapshot(iteration: int, objective_value: float) -> Snapshot:
        with torch.no_grad():
            image, displacement, _ = geometry.warp(original_tensor)
        return Snapshot(
            iteration=iteration,
            image=image.detach().clone(),
            displacement=displacement.detach().clone(),
            state_dict=_clone_state(geometry),
            objective=objective_value,
        )

    snapshots[0] = snapshot(0, 0.0)
    for iteration in range(1, settings.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        perturbed, displacement, fields = geometry.warp(original_tensor)
        objective, terms = internal_objective(pipe, perturbed, reference, settings.objective)
        visual_mse = (perturbed - original_tensor).square().mean()
        normalized_visual = visual_mse / blank_mse.clamp_min(1e-8)
        magnitude = displacement.square().sum(dim=1).sqrt()
        disp_penalty = magnitude.square().mean() / (settings.max_disp_px**2)
        smooth_penalty = total_variation(displacement) / settings.max_disp_px
        fold_penalty, jacobian = jacobian_penalty(displacement)
        scaled_objective = settings.objective_scale * objective
        loss = (
            -scaled_objective
            + settings.lambda_visual * visual_mse
            + settings.lambda_disp * disp_penalty
            + settings.lambda_smooth * smooth_penalty
            + settings.lambda_fold * fold_penalty
        )
        if not bool(torch.isfinite(loss).item()):
            history.append({"iter": iteration, "stopped": True, "reason": "non_finite_loss"})
            break

        loss.backward()
        grad_norm = _gradient_norm(geometry.parameters())
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in geometry.parameters()
        )
        if not gradients_finite:
            history.append({"iter": iteration, "stopped": True, "reason": "non_finite_gradient"})
            break
        torch.nn.utils.clip_grad_norm_(geometry.parameters(), 1.0)
        optimizer.step()
        geometry.project_()

        row = {
            "iter": iteration,
            "loss": float(loss.detach().float().cpu()),
            "objective": float(objective.detach().float().cpu()),
            "scaled_objective": float(scaled_objective.detach().float().cpu()),
            "edit_direction_mse": float(terms["edit_direction_mse"].detach().float().cpu()),
            "edit_direction_cosine": float(terms["edit_direction_cosine"].detach().float().cpu()),
            "unet_prediction_mse": float(terms["unet_prediction_mse"].detach().float().cpu()),
            "vae_conditioning_mse": float(terms["vae_conditioning_mse"].detach().float().cpu()),
            "visual_mse": float(visual_mse.detach().float().cpu()),
            "normalized_visual_mse": float(normalized_visual.detach().float().cpu()),
            "disp_penalty": float(disp_penalty.detach().float().cpu()),
            "smooth_penalty": float(smooth_penalty.detach().float().cpu()),
            "fold_penalty": float(fold_penalty.detach().float().cpu()),
            "grad_norm": grad_norm,
            "elapsed_seconds": time.monotonic() - started,
            **displacement_stats(displacement),
            **jacobian,
        }
        history.append(row)
        candidate = Snapshot(
            iteration=iteration,
            image=perturbed.detach().clone(),
            displacement=displacement.detach().clone(),
            state_dict=_clone_state(geometry),
            objective=row["objective"],
        )
        if best is None or candidate.objective > best.objective:
            best = candidate
        if iteration % settings.checkpoint_every == 0 or iteration == settings.iterations:
            snapshots[iteration] = snapshot(iteration, row["objective"])

    if best is None:
        raise RuntimeError("No finite geometry optimization step completed.")
    if settings.iterations not in snapshots:
        snapshots[settings.iterations] = best
    return OptimizationResult(geometry=geometry, history=history, snapshots=snapshots, best=best)
