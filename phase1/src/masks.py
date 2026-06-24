"""Face-local mask construction with a deterministic centered-face fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FALLBACK_FACE_MASK: dict[str, float] = {
    "center_x": 0.50,
    "center_y": 0.52,
    "radius_x": 0.34,
    "radius_y": 0.42,
    "edge_falloff": 0.12,
}


def detect_face_fraction(image: Image.Image) -> dict[str, float] | None:
    """Return a soft ellipse proposal, or None when OpenCV cannot find a face."""
    try:
        import cv2

        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        image_width, image_height = image.size
        return {
            "center_x": float((x + 0.5 * width) / image_width),
            "center_y": float((y + 0.50 * height) / image_height),
            "radius_x": float(min(0.48, max(0.22, 0.70 * width / image_width))),
            "radius_y": float(min(0.62, max(0.30, 0.82 * height / image_height))),
            "edge_falloff": 0.12,
        }
    except Exception:
        return None


def resolved_face_mask(image: Image.Image) -> tuple[dict[str, float], str]:
    detected = detect_face_fraction(image)
    return (detected, "opencv_haar") if detected is not None else (dict(FALLBACK_FACE_MASK), "centered_fallback")


def face_detected(image: Image.Image) -> bool | None:
    """None means detector unavailable, False means available but no face found."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return None
    return detect_face_fraction(image) is not None
