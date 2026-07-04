from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def png_compress_level() -> int:
    raw = os.getenv("WM_BENCH_PNG_COMPRESS_LEVEL", "1")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(0, min(9, value))


def png_save_kwargs() -> dict[str, Any]:
    return {"format": "PNG", "compress_level": png_compress_level()}


def _register_saved_image(path: Path, image: Image.Image) -> None:
    try:
        from evaluator.image_protocol import invalidate_image_metadata, register_image_size

        invalidate_image_metadata(path)
        register_image_size(path, [int(image.size[0]), int(image.size[1])])
    except Exception:
        return


def save_png_image(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = image.convert("RGB")
    converted.save(path, **png_save_kwargs())
    _register_saved_image(path, converted)


def save_png_array(array: np.ndarray, path: str | Path) -> None:
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
    save_png_image(image, path)
