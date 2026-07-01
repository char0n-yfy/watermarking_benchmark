from __future__ import annotations

import io
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.registry import register_attack
from evaluator.image_io import save_png_image


DEFAULT_STRENGTH_RANGES: dict[str, tuple[float, float]] = {
    "rotation": (0.0, 45.0),
    "resized_crop": (1.0, 0.5),
    "erasing": (0.0, 0.25),
    "brightness": (1.0, 2.0),
    "contrast": (1.0, 2.0),
    "gaussian_blur": (0.0, 20.0),
    "gaussian_noise": (0.0, 0.1),
    "jpeg": (90.0, 10.0),
}


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _save_png(image: Image.Image, path: Path) -> None:
    save_png_image(image, path)


def _rng(context: AttackContext) -> random.Random:
    return random.Random(context.seed)


def _resolve_strength(
    attack_key: str,
    strength: float | None,
    *,
    relative: bool,
    fallback: float | None = None,
) -> float:
    if strength is None:
        if fallback is None:
            raise ValueError("Either strength or fallback must be provided")
        return float(fallback)

    strength = float(strength)
    if not relative:
        return strength

    if not 0.0 <= strength <= 1.0:
        raise ValueError("relative strength must be in [0, 1]")
    start, end = DEFAULT_STRENGTH_RANGES[attack_key]
    value = start + strength * (end - start)
    return max(min(start, end), min(max(start, end), value))


def _normalize_ratio(value: float, field_name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return value


class _DistortionAttack(BaseAttack):
    thread_safe_parallel = True
    batch_capability = False


@register_attack
class IdentityAttack(_DistortionAttack):
    name = "identity"
    description = "No-op distortion. Copies the watermarked image unchanged."

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        try:
            if input_path.resolve() == output_path.resolve():
                return {
                    "operation": "same_path",
                    "noOp": True,
                    "encodedBytesPreserved": True,
                }
        except OSError:
            pass

        if output_path.exists():
            output_path.unlink()

        try:
            os.link(input_path, output_path)
            operation = "hardlink"
        except OSError:
            shutil.copy2(input_path, output_path)
            operation = "copy2"

        return {
            "operation": operation,
            "noOp": True,
            "encodedBytesPreserved": True,
        }


@register_attack
class RotationAttack(_DistortionAttack):
    name = "rotation"
    description = "Rotate image by a fixed or relative angle."

    def __init__(
        self,
        angle: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
        fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        angle = _resolve_strength(
            "rotation",
            strength,
            relative=relative_strength,
            fallback=0.0 if angle is None else angle,
        )
        super().__init__(
            angle=angle,
            strength=strength,
            relative_strength=relative_strength,
            fill=list(fill),
        )
        self.angle = angle
        self.fill = fill

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        attacked = image.rotate(self.angle, resample=Image.Resampling.BICUBIC, fillcolor=self.fill)
        _save_png(attacked, output_path)
        return {"angle": self.angle}


@register_attack
class ResizedCropAttack(_DistortionAttack):
    name = "resized_crop"
    description = "Crop a square region and resize it back to the original image size."

    def __init__(
        self,
        scale: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
    ) -> None:
        scale = _resolve_strength(
            "resized_crop",
            strength,
            relative=relative_strength,
            fallback=1.0 if scale is None else scale,
        )
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        super().__init__(scale=scale, strength=strength, relative_strength=relative_strength)
        self.scale = scale

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        rng = _rng(context)
        width, height = image.size
        crop_w = max(1, int(width * self.scale))
        crop_h = max(1, int(height * self.scale))
        left = rng.randint(0, width - crop_w)
        top = rng.randint(0, height - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        attacked = image.crop(box).resize((width, height), Image.Resampling.BICUBIC)
        _save_png(attacked, output_path)
        return {"scale": self.scale, "box": list(box)}


@register_attack
class ErasingAttack(_DistortionAttack):
    name = "erasing"
    description = "Randomly erase a square area and keep the original image size."

    def __init__(
        self,
        area_ratio: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
        value: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        area_ratio = _resolve_strength(
            "erasing",
            strength,
            relative=relative_strength,
            fallback=0.0 if area_ratio is None else area_ratio,
        )
        _normalize_ratio(area_ratio, "area_ratio")
        super().__init__(
            area_ratio=area_ratio,
            strength=strength,
            relative_strength=relative_strength,
            value=list(value),
        )
        self.area_ratio = area_ratio
        self.value = value

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        rng = _rng(context)
        width, height = image.size
        erase_w = max(1, int(width * (self.area_ratio**0.5)))
        erase_h = max(1, int(height * (self.area_ratio**0.5)))
        left = rng.randint(0, width - erase_w)
        top = rng.randint(0, height - erase_h)
        box = (left, top, left + erase_w, top + erase_h)
        patch = Image.new("RGB", (erase_w, erase_h), self.value)
        attacked = image.copy()
        attacked.paste(patch, box[:2])
        _save_png(attacked, output_path)
        return {"area_ratio": self.area_ratio, "box": list(box)}


@register_attack
class BrightnessAttack(_DistortionAttack):
    name = "brightness"
    description = "Brightness scaling distortion."

    def __init__(
        self,
        factor: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
    ) -> None:
        factor = _resolve_strength(
            "brightness",
            strength,
            relative=relative_strength,
            fallback=1.0 if factor is None else factor,
        )
        if factor < 0:
            raise ValueError("factor must be >= 0")
        super().__init__(factor=factor, strength=strength, relative_strength=relative_strength)
        self.factor = factor

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        attacked = ImageEnhance.Brightness(image).enhance(self.factor)
        _save_png(attacked, output_path)
        return {"factor": self.factor}


@register_attack
class ContrastAttack(_DistortionAttack):
    name = "contrast"
    description = "Contrast scaling distortion."

    def __init__(
        self,
        factor: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
    ) -> None:
        factor = _resolve_strength(
            "contrast",
            strength,
            relative=relative_strength,
            fallback=1.0 if factor is None else factor,
        )
        if factor < 0:
            raise ValueError("factor must be >= 0")
        super().__init__(factor=factor, strength=strength, relative_strength=relative_strength)
        self.factor = factor

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        attacked = ImageEnhance.Contrast(image).enhance(self.factor)
        _save_png(attacked, output_path)
        return {"factor": self.factor}


@register_attack
class GaussianBlurAttack(_DistortionAttack):
    name = "gaussian_blur"
    description = "Gaussian blur distortion."

    def __init__(
        self,
        radius: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
    ) -> None:
        radius = _resolve_strength(
            "gaussian_blur",
            strength,
            relative=relative_strength,
            fallback=0.0 if radius is None else radius,
        )
        if radius < 0:
            raise ValueError("radius must be >= 0")
        super().__init__(radius=radius, strength=strength, relative_strength=relative_strength)
        self.radius = radius

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        attacked = image.filter(ImageFilter.GaussianBlur(self.radius))
        _save_png(attacked, output_path)
        return {"radius": self.radius}


@register_attack
class GaussianNoiseAttack(_DistortionAttack):
    name = "gaussian_noise"
    description = "Add zero-mean Gaussian pixel noise."

    def __init__(
        self,
        sigma: float | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
    ) -> None:
        sigma = _resolve_strength(
            "gaussian_noise",
            strength,
            relative=relative_strength,
            fallback=0.0 if sigma is None else sigma,
        )
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        super().__init__(sigma=sigma, strength=strength, relative_strength=relative_strength)
        self.sigma = sigma

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        rng = np.random.default_rng(context.seed)
        pixels = np.asarray(image).astype(np.float32) / 255.0
        noise = rng.normal(0.0, self.sigma, size=pixels.shape).astype(np.float32)
        attacked_pixels = np.clip(pixels + noise, 0.0, 1.0)
        attacked = Image.fromarray((attacked_pixels * 255.0).round().astype(np.uint8), mode="RGB")
        _save_png(attacked, output_path)
        return {"sigma": self.sigma}


@register_attack
class JPEGCompressionAttack(_DistortionAttack):
    name = "jpeg"
    description = "JPEG compression distortion, saved back as PNG."

    def __init__(
        self,
        quality: int | None = None,
        strength: float | None = None,
        relative_strength: bool = True,
    ) -> None:
        quality_value = _resolve_strength(
            "jpeg",
            strength,
            relative=relative_strength,
            fallback=75.0 if quality is None else float(quality),
        )
        quality = int(round(quality_value))
        if not 1 <= quality <= 100:
            raise ValueError("quality must be in [1, 100]")
        super().__init__(quality=quality, strength=strength, relative_strength=relative_strength)
        self.quality = quality

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        attacked = Image.open(buffer).convert("RGB")
        _save_png(attacked, output_path)
        return {"quality": self.quality}


@register_attack
class ResizeAttack(_DistortionAttack):
    name = "resize"
    description = "Resize by a scale factor and resize back to original dimensions."

    def __init__(self, scale: float = 0.5) -> None:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("scale must be > 0")
        super().__init__(scale=scale)
        self.scale = scale

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = _load_rgb(input_path)
        width, height = image.size
        resized_size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
        attacked = image.resize(resized_size, Image.Resampling.BICUBIC)
        attacked = ImageOps.fit(attacked, (width, height), method=Image.Resampling.BICUBIC)
        _save_png(attacked, output_path)
        return {"scale": self.scale, "intermediate_size": list(resized_size)}
