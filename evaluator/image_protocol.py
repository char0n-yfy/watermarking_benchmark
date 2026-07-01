from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.image_io import save_png_image


JsonDict = dict[str, Any]

CANONICAL_IMAGE_SIZE = (512, 512)
CANONICAL_PREPROCESS_POLICY = "center_cover_crop_512"
CANONICAL_OUTPUT_POLICY = "resize_bicubic_to_canonical_512"
QUALITY_ALIGNMENT_NONE = "none"
QUALITY_ALIGNMENT_RESIZE_TARGET = "resize_target_bicubic_to_reference"


def size_list(size: Any) -> list[int] | None:
    if size is None:
        return None
    if isinstance(size, Mapping):
        width = size.get("width") or size.get("w")
        height = size.get("height") or size.get("h")
        if width is None or height is None:
            return None
        try:
            return [int(width), int(height)]
        except (TypeError, ValueError):
            return None
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            return [int(size[0]), int(size[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(size, int):
        return [int(size), int(size)]
    return None


def image_size(path: str | Path) -> list[int] | None:
    try:
        with Image.open(path) as image:
            return [int(image.size[0]), int(image.size[1])]
    except Exception:
        return None


def first_metadata_size(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> list[int] | None:
    for key in keys:
        if key in metadata:
            parsed = size_list(metadata.get(key))
            if parsed is not None:
                return parsed
    return None


def canonical_preprocess_image(
    source_path: str | Path,
    target_path: str | Path,
    *,
    size: tuple[int, int] = CANONICAL_IMAGE_SIZE,
) -> JsonDict:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_w, target_h = size

    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    original_w, original_h = image.size
    if original_w <= 0 or original_h <= 0:
        raise ValueError(f"Invalid image dimensions for {source_path}: {image.size}")

    scale = max(target_w / original_w, target_h / original_h)
    resized_w = max(target_w, int(round(original_w * scale)))
    resized_h = max(target_h, int(round(original_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    right = left + target_w
    bottom = top + target_h
    cropped = resized.crop((left, top, right, bottom))
    save_png_image(cropped, target_path)

    return {
        "preprocessPolicy": CANONICAL_PREPROCESS_POLICY,
        "cropPolicy": "deterministic_center_cover_crop",
        "originalSize": [original_w, original_h],
        "canonicalSize": [target_w, target_h],
        "resizedContentSize": [resized_w, resized_h],
        "cropBox": [left, top, right, bottom],
        "cropMargins": {
            "left": left,
            "top": top,
            "right": resized_w - right,
            "bottom": resized_h - bottom,
        },
        "padding": None,
        "scale": scale,
        "paddingColor": None,
    }


def canonicalize_image_file_in_place(
    image_path: str | Path,
    *,
    size: tuple[int, int] = CANONICAL_IMAGE_SIZE,
) -> JsonDict:
    image_path = Path(image_path)
    target_size = [int(size[0]), int(size[1])]
    before = image_size(image_path)
    if before is None:
        return {
            "ok": False,
            "changed": False,
            "preCanonicalOutputSize": None,
            "outputSize": None,
            "outputSizePolicy": None,
        }
    if before == target_size:
        return {
            "ok": True,
            "changed": False,
            "preCanonicalOutputSize": before,
            "outputSize": before,
            "outputSizePolicy": "already_canonical",
        }
    with Image.open(image_path) as opened:
        image = opened.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    save_png_image(image, image_path)
    return {
        "ok": True,
        "changed": True,
        "preCanonicalOutputSize": before,
        "outputSize": target_size,
        "outputSizePolicy": CANONICAL_OUTPUT_POLICY,
    }


def quality_alignment_metadata(reference_path: str | Path, target_path: str | Path) -> JsonDict:
    reference_size = image_size(reference_path)
    target_size = image_size(target_path)
    aligned_size = reference_size
    alignment_policy = QUALITY_ALIGNMENT_NONE if reference_size == target_size else QUALITY_ALIGNMENT_RESIZE_TARGET
    return {
        "referenceSize": reference_size,
        "targetSize": target_size,
        "alignedSize": aligned_size,
        "alignmentPolicy": alignment_policy,
    }


def semantic_size_change_attack(
    attack_name: str,
    params: Mapping[str, Any],
    metadata: Mapping[str, Any],
    input_size: list[int] | None,
    output_size: list[int] | None,
) -> bool:
    if input_size is None or output_size is None or input_size == output_size:
        return False

    token = str(attack_name).lower()
    if token.startswith(("2x_", "4x_")) or token in {"2x_regen", "4x_regen"}:
        return True
    if token.startswith("cew_s"):
        return True
    if str(metadata.get("task_name") or "").lower() == "super_resolution":
        return True
    if str(metadata.get("backend") or "").lower() == "cew_composite_chain":
        for step in metadata.get("steps") or []:
            if isinstance(step, Mapping) and str(step.get("task_name") or "").lower() == "super_resolution":
                return True
            if isinstance(step, Mapping) and str(step.get("step") or "").lower().startswith("cew_s"):
                return True

    raw_scale = metadata.get("scale", params.get("scale"))
    try:
        scale = int(raw_scale)
    except (TypeError, ValueError):
        scale = 0
    return scale in {2, 4} and output_size == [input_size[0] * scale, input_size[1] * scale]
