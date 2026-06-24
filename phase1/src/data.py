"""Image conversion helpers with no hard-coded platform paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def pil_to_tensor(image: Image.Image, device, dtype=None):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def tensor_to_pil(tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")
