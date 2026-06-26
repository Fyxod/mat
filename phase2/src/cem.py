"""Cross-entropy method state for Phase 2 geometry candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateSample:
    theta: np.ndarray
    method: str
    method_index: int
    generation: int
    member: int
    candidate_index: int
    initial: bool = False

    def theta_payload(self, names: list[str]) -> dict[str, Any]:
        return {
            "names": names,
            "values": [float(value) for value in self.theta],
            "method": self.method,
            "generation": self.generation,
            "member": self.member,
            "candidate_index": self.candidate_index,
            "initial": self.initial,
        }


class CEMState:
    """Continuous CEM over theta with a smoothed categorical method choice."""

    def __init__(
        self,
        *,
        dimension: int,
        methods: list[str],
        seed: int,
        sample_std: float,
        min_std: float,
        std_smoothing: float = 0.65,
    ) -> None:
        if dimension < 1:
            raise ValueError("CEM dimension must be positive")
        if not methods:
            raise ValueError("At least one geometry method is required")
        self.dimension = int(dimension)
        self.methods = list(methods)
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(self.dimension, dtype=np.float32)
        self.std = np.full(self.dimension, float(sample_std), dtype=np.float32)
        self.sample_std = float(sample_std)
        self.min_std = float(min_std)
        self.std_smoothing = float(std_smoothing)
        self.method_probs = np.full(len(self.methods), 1.0 / len(self.methods), dtype=np.float64)

    def initial(self, candidate_index: int = 0) -> CandidateSample:
        method = "combined_tps_dct" if "combined_tps_dct" in self.methods else self.methods[0]
        method_index = self.methods.index(method)
        return CandidateSample(
            theta=np.zeros(self.dimension, dtype=np.float32),
            method=method,
            method_index=method_index,
            generation=0,
            member=0,
            candidate_index=candidate_index,
            initial=True,
        )

    def sample_generation(self, generation: int, population: int, start_index: int) -> list[CandidateSample]:
        samples: list[CandidateSample] = []
        method_indices = self.rng.choice(len(self.methods), size=population, p=self.method_probs)
        for offset, method_index in enumerate(method_indices, 1):
            theta = self.mean + self.std * self.rng.normal(size=self.dimension)
            theta = np.clip(theta, -2.5, 2.5).astype(np.float32)
            samples.append(
                CandidateSample(
                    theta=theta,
                    method=self.methods[int(method_index)],
                    method_index=int(method_index),
                    generation=generation,
                    member=offset,
                    candidate_index=start_index + offset - 1,
                )
            )
        return samples

    def update(self, elite_samples: list[CandidateSample]) -> dict[str, Any]:
        if not elite_samples:
            return self.state_payload()
        elite_theta = np.stack([sample.theta for sample in elite_samples]).astype(np.float32)
        observed_mean = elite_theta.mean(axis=0)
        observed_std = elite_theta.std(axis=0) if len(elite_samples) > 1 else self.std * 0.70
        self.mean = observed_mean.astype(np.float32)
        self.std = np.clip(
            self.std * (1.0 - self.std_smoothing) + observed_std * self.std_smoothing,
            self.min_std,
            self.sample_std,
        ).astype(np.float32)
        counts = np.ones(len(self.methods), dtype=np.float64) * 0.10
        for sample in elite_samples:
            counts[sample.method_index] += 1.0
        elite_probs = counts / counts.sum()
        self.method_probs = 0.35 * self.method_probs + 0.65 * elite_probs
        self.method_probs = self.method_probs / self.method_probs.sum()
        return self.state_payload()

    def state_payload(self) -> dict[str, Any]:
        return {
            "mean_abs": float(np.mean(np.abs(self.mean))),
            "std_mean": float(np.mean(self.std)),
            "std_min": float(np.min(self.std)),
            "std_max": float(np.max(self.std)),
            "methods": self.methods,
            "method_probs": [float(value) for value in self.method_probs],
        }


__all__ = ["CEMState", "CandidateSample"]

