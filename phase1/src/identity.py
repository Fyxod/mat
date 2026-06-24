"""Optional DeepFace identity panel; failures are recorded rather than fatal."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _embedding(path: Path, model_name: str) -> np.ndarray:
    from deepface import DeepFace

    response = DeepFace.represent(
        img_path=str(path),
        model_name=model_name,
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
    )
    item = response[0] if isinstance(response, list) else response
    return np.asarray(item["embedding"], dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def identity_panel(paths: dict[str, Path], models: tuple[str, ...] = ("SFace", "Facenet512", "ArcFace")) -> dict[str, Any]:
    pairs = {
        "original_vs_perturbed": ("original", "perturbed"),
        "original_edited_vs_perturbed_edited": ("original_edited", "perturbed_edited"),
        "original_vs_original_edited": ("original", "original_edited"),
        "perturbed_vs_perturbed_edited": ("perturbed", "perturbed_edited"),
    }
    result: dict[str, Any] = {"available": False, "models": {}, "pairs": pairs}
    for model in models:
        try:
            vectors = {name: _embedding(path, model) for name, path in paths.items()}
            comparisons = {
                name: {"similarity": cosine_similarity(vectors[left], vectors[right]), "distance": float(1.0 - cosine_similarity(vectors[left], vectors[right]))}
                for name, (left, right) in pairs.items()
            }
            result["models"][model] = {"status": "ok", "comparisons": comparisons}
            result["available"] = True
        except Exception as error:
            result["models"][model] = {"status": "unavailable", "error": str(error)}
    return result
