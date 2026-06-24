"""Artifact writing and compact contact sheets for Phase 1."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .utils import write_csv, write_json


def save_rgb(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    kwargs = {"quality": 93} if fmt == "JPEG" else {}
    image.convert("RGB").save(path, format=fmt, **kwargs)


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def image_sheet(
    path: Path,
    rows: Iterable[tuple[list[str], list[Image.Image]]],
    columns: int | None = None,
    cell_width: int = 256,
    cell_height: int = 256,
) -> None:
    entries = list(rows)
    if not entries:
        return
    max_images = max(len(images) for _, images in entries)
    cols = columns or max_images
    row_blocks: list[tuple[list[str], list[Image.Image]]] = []
    for labels, images in entries:
        for index in range(0, len(images), cols):
            row_blocks.append((labels[index:index + cols], images[index:index + cols]))

    label_height = 38
    canvas = Image.new("RGB", (cols * cell_width, len(row_blocks) * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (labels, images) in enumerate(row_blocks):
        y = row_index * (cell_height + label_height)
        for column, (label, image) in enumerate(zip(labels, images)):
            x = column * cell_width
            canvas.paste(image.convert("RGB").resize((cell_width, cell_height), Image.Resampling.LANCZOS), (x, y))
            draw.multiline_text((x + 4, y + cell_height + 3), label[:95], fill="black", spacing=1)
    save_rgb(path, canvas)


def attack_sheet(
    path: Path,
    original: Image.Image,
    clean_edit: Image.Image,
    perturbed: Image.Image,
    perturbed_edit: Image.Image,
    flow: Image.Image,
    caption: str,
) -> None:
    image_sheet(
        path,
        [(
            [
                "Original",
                "Clean edit",
                "Perturbed input",
                "Perturbed edit",
                f"Flow\n{caption}",
            ],
            [original, clean_edit, perturbed, perturbed_edit, flow],
        )],
        columns=5,
    )


__all__ = ["attack_sheet", "copy_if_exists", "image_sheet", "save_rgb", "write_csv", "write_json"]
