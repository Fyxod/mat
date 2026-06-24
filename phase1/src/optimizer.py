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
    effective_objective_scale: float
    initial_objective_grad_norm: float
    initial_regularizer_grad_norm: float
    objective_gradient_probe_multiplier: float


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _gradient_norm(parameters) -> float:
    pieces = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not pieces else float(torch.stack(pieces).sum().sqrt().cpu())


def _gradient_norm_from_tensors(gradients: tuple[torch.Tensor | None, ...]) -> float:
    pieces = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
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
    effective_objective_scale: float | None = None
    initial_objective_grad_norm = 0.0
    initial_regularizer_grad_norm = 0.0
    objective_gradient_probe_multiplier = 1.0

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
        # The face/border masks intentionally create exact zero displacement
        # pixels. Add epsilon *inside* sqrt so their derivative remains finite;
        # clamping after sqrt leaves a 0 * infinity backward path and causes
        # every otherwise-valid geometry gradient to become NaN.
        magnitude = torch.sqrt(displacement.square().sum(dim=1) + 1e-12)
        disp_penalty = magnitude.square().mean() / (settings.max_disp_px**2)
        smooth_penalty = total_variation(displacement) / settings.max_disp_px
        fold_penalty, jacobian = jacobian_penalty(displacement)
        regularizer = (
            settings.lambda_visual * visual_mse
            + settings.lambda_disp * disp_penalty
            + settings.lambda_smooth * smooth_penalty
            + settings.lambda_fold * fold_penalty
        )
        if effective_objective_scale is None:
            # The three internal objectives differ by orders of magnitude.  A
            # fixed scalar therefore either erases the geometry (direction / UNet)
            # or drives it straight into the projection bound (VAE).  Match their
            # *initial geometry-gradient* to the regularizer once per start; the
            # configured balance then states the intended adversarial pressure in
            # a scale-invariant way.
            parameters = tuple(geometry.parameters())
            objective_gradients = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            initial_objective_grad_norm = _gradient_norm_from_tensors(objective_gradients)
            # The frozen VAE/UNet run in fp16.  For the small direction and
            # UNet MSE objectives, an otherwise-valid gradient can underflow
            # to zero *only while we measure it*.  Re-evaluate a scaled copy
            # of that scalar, then undo the factor in the reported norm.
            # This does not alter the optimization objective itself.
            if initial_objective_grad_norm <= 1e-12:
                objective_gradient_probe_multiplier = 1e6
                probed_gradients = torch.autograd.grad(
                    objective * objective_gradient_probe_multiplier,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                initial_objective_grad_norm = (
                    _gradient_norm_from_tensors(probed_gradients) / objective_gradient_probe_multiplier
                )
            regularizer_gradients = torch.autograd.grad(regularizer, parameters, retain_graph=True, allow_unused=True)
            initial_regularizer_grad_norm = _gradient_norm_from_tensors(regularizer_gradients)
            raw_scale = (
                settings.objective_scale
                * settings.objective_gradient_balance
                * initial_regularizer_grad_norm
                / max(initial_objective_grad_norm, 1e-12)
            )
            effective_objective_scale = min(
                max(raw_scale, settings.objective_scale_min), settings.objective_scale_max
            )
        scaled_objective = effective_objective_scale * objective
        loss = -scaled_objective + regularizer
        if not bool(torch.isfinite(loss).item()):
            finite_terms = {
                "perturbed": bool(torch.isfinite(perturbed).all().item()),
                "displacement": bool(torch.isfinite(displacement).all().item()),
                "objective": bool(torch.isfinite(objective).all().item()),
                "edit_direction_mse": bool(torch.isfinite(terms["edit_direction_mse"]).all().item()),
                "unet_prediction_mse": bool(torch.isfinite(terms["unet_prediction_mse"]).all().item()),
                "vae_conditioning_mse": bool(torch.isfinite(terms["vae_conditioning_mse"]).all().item()),
                "visual_mse": bool(torch.isfinite(visual_mse).all().item()),
                "disp_penalty": bool(torch.isfinite(disp_penalty).all().item()),
                "smooth_penalty": bool(torch.isfinite(smooth_penalty).all().item()),
                "fold_penalty": bool(torch.isfinite(fold_penalty).all().item()),
            }
            raise FloatingPointError(f"Non-finite white-box loss at iteration {iteration}: {finite_terms}")

        loss.backward()
        grad_norm = _gradient_norm(geometry.parameters())
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in geometry.parameters()
        )
        if not gradients_finite:
            bad_parameters = [
                name for name, parameter in geometry.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            ]
            raise FloatingPointError(
                f"Non-finite geometry gradient at iteration {iteration}; "
                f"parameters={bad_parameters}, loss={float(loss.detach().float().cpu())}, "
                f"objective={float(objective.detach().float().cpu())}"
            )
        torch.nn.utils.clip_grad_norm_(geometry.parameters(), 1.0)
        optimizer.step()
        geometry.project_()

        row = {
            "iter": iteration,
            "loss": float(loss.detach().float().cpu()),
            "objective": float(objective.detach().float().cpu()),
            "scaled_objective": float(scaled_objective.detach().float().cpu()),
            "effective_objective_scale": effective_objective_scale,
            "initial_objective_grad_norm": initial_objective_grad_norm,
            "initial_regularizer_grad_norm": initial_regularizer_grad_norm,
            "objective_gradient_probe_multiplier": objective_gradient_probe_multiplier,
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
    return OptimizationResult(
        geometry=geometry,
        history=history,
        snapshots=snapshots,
        best=best,
        effective_objective_scale=float(effective_objective_scale or settings.objective_scale),
        initial_objective_grad_norm=initial_objective_grad_norm,
        initial_regularizer_grad_norm=initial_regularizer_grad_norm,
        objective_gradient_probe_multiplier=objective_gradient_probe_multiplier,
    )
