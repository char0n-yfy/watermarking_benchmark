from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from evaluator.image_io import save_png_image

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


ImageFormat = Literal["jpeg", "webp", "png"]


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def save_png(image: Image.Image, path: Path) -> None:
    save_png_image(image, path)


def to_uint8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def from_uint8(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def resize_max_side(image: Image.Image, max_side: int, method: int = Image.Resampling.LANCZOS) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be > 0")
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image.copy()
    scale = max_side / current_max
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, method)


def encode_decode(
    image: Image.Image,
    fmt: ImageFormat = "jpeg",
    quality: int = 85,
    subsampling: int | str | None = None,
) -> tuple[Image.Image, str]:
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")

    normalized = fmt.lower()
    pil_format = {"jpeg": "JPEG", "webp": "WEBP", "png": "PNG"}.get(normalized)
    if pil_format is None:
        raise ValueError("fmt must be one of: jpeg, webp, png")

    buffer = io.BytesIO()
    try:
        kwargs = {"quality": quality}
        if pil_format == "JPEG" and subsampling is not None:
            kwargs["subsampling"] = subsampling
        image.convert("RGB").save(buffer, format=pil_format, **kwargs)
        used = normalized
    except Exception:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        used = "jpeg"
    buffer.seek(0)
    return Image.open(buffer).convert("RGB"), used


def unsharp(image: Image.Image, radius: float = 1.2, amount: float = 0.7, threshold: int = 2) -> Image.Image:
    percent = int(max(0.0, amount) * 100)
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def gamma_correct(image: Image.Image, gamma: float = 1.0) -> Image.Image:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    inv = 1.0 / gamma
    lut = [round(((i / 255.0) ** inv) * 255.0) for i in range(256)]
    return image.point(lut * 3)


def gray_world_white_balance(image: Image.Image, strength: float = 1.0) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))
    array = to_uint8(image).astype(np.float32)
    channel_means = array.reshape(-1, 3).mean(axis=0)
    gray = channel_means.mean()
    gains = gray / np.maximum(channel_means, 1e-6)
    gains = 1.0 + strength * (gains - 1.0)
    return from_uint8(array * gains.reshape(1, 1, 3))


def temperature_tint(image: Image.Image, temperature: float = 0.0, tint: float = 0.0, intensity: float = 1.0) -> Image.Image:
    temperature = float(np.clip(temperature, -1.0, 1.0))
    tint = float(np.clip(tint, -1.0, 1.0))
    intensity = float(np.clip(intensity, 0.0, 1.0))
    gains = np.array(
        [
            1.0 + 0.16 * temperature,
            1.0 + 0.08 * tint,
            1.0 - 0.16 * temperature,
        ],
        dtype=np.float32,
    )
    gains = 1.0 + intensity * (gains - 1.0)
    return from_uint8(to_uint8(image).astype(np.float32) * gains.reshape(1, 1, 3))


def auto_enhance(
    image: Image.Image,
    gamma: float = 1.03,
    contrast: float = 1.08,
    saturation: float = 1.08,
    white_balance: float = 0.5,
    autocontrast_cutoff: int = 1,
) -> Image.Image:
    result = ImageOps.autocontrast(image.convert("RGB"), cutoff=autocontrast_cutoff)
    result = gray_world_white_balance(result, white_balance)
    result = gamma_correct(result, gamma)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(saturation)
    return result


def denoise(image: Image.Image, strength: float = 0.5, method: str = "nlm") -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))
    if cv2 is not None and method in {"nlm", "auto"}:
        bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
        h = 3.0 + 12.0 * strength
        out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        return from_uint8(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    radius = 1 if strength < 0.6 else 2
    return image.filter(ImageFilter.MedianFilter(size=2 * radius + 1))


def edge_preserve_smooth(
    image: Image.Image,
    radius: int = 7,
    sigma_color: float = 35.0,
    sigma_space: float = 35.0,
) -> Image.Image:
    if cv2 is not None:
        array = to_uint8(image)
        out = cv2.bilateralFilter(array, int(radius), float(sigma_color), float(sigma_space))
        return from_uint8(out)
    return image.filter(ImageFilter.SMOOTH_MORE)


def deartifact(image: Image.Image, strength: float = 0.45, radius: int = 5) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))
    smoothed = edge_preserve_smooth(
        image,
        radius=max(3, int(radius)),
        sigma_color=20.0 + 50.0 * strength,
        sigma_space=20.0 + 50.0 * strength,
    )
    return Image.blend(image, smoothed, 0.35 + 0.45 * strength)


def despeckle(
    image: Image.Image,
    kernel_size: int = 3,
    component_threshold: int = 24,
    strength: float = 0.6,
    difference_threshold: int = 10,
) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if cv2 is None:
        filtered = image.filter(ImageFilter.MedianFilter(size=kernel_size))
        return Image.blend(image.convert("RGB"), filtered, 0.2 + 0.55 * strength)

    array = to_uint8(image)
    median = cv2.medianBlur(array, kernel_size)
    diff = cv2.absdiff(array, median)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, raw_mask = cv2.threshold(diff_gray, int(difference_threshold), 255, cv2.THRESH_BINARY)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    max_area = max(1, int(component_threshold))
    for label in range(1, labels_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            mask[labels == label] = 255

    if not np.any(mask):
        return Image.blend(image.convert("RGB"), from_uint8(median), 0.15 + 0.35 * strength)

    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * (0.35 + 0.55 * strength)
    out = array.astype(np.float32) * (1.0 - alpha) + median.astype(np.float32) * alpha
    return from_uint8(out)


def clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Image.Image:
    if cv2 is None:
        return ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    array = to_uint8(image)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l_channel = op.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return from_uint8(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def color_space_quantize(image: Image.Image, color_space: str = "ycbcr", bit_depth: int = 7) -> Image.Image:
    bit_depth = int(bit_depth)
    if not 4 <= bit_depth <= 8:
        raise ValueError("bit_depth must be in [4, 8]")
    levels = 2**bit_depth - 1

    if color_space.lower() == "lab" and cv2 is not None:
        array = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2LAB).astype(np.float32)
        quantized = np.round(array / 255.0 * levels) / levels * 255.0
        return from_uint8(cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_LAB2RGB))

    converted = image.convert("YCbCr" if color_space.lower() == "ycbcr" else "RGB")
    array = np.asarray(converted, dtype=np.float32)
    quantized = np.round(array / 255.0 * levels) / levels * 255.0
    out = Image.fromarray(quantized.astype(np.uint8), mode=converted.mode)
    return out.convert("RGB")


def apply_filter_lut(image: Image.Image, lut_type: str = "warm_film", alpha: float = 0.5, saturation: float = 1.0) -> Image.Image:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = image.convert("RGB")
    array = to_uint8(base).astype(np.float32)

    if lut_type == "warm_film":
        mapped = array * np.array([1.08, 1.02, 0.92], dtype=np.float32)
        mapped = mapped * 0.96 + 8.0
    elif lut_type == "cool_clean":
        mapped = array * np.array([0.94, 1.02, 1.10], dtype=np.float32)
    elif lut_type == "matte":
        mapped = array * 0.86 + 24.0
    elif lut_type == "vivid":
        tmp = ImageEnhance.Color(base).enhance(1.25)
        tmp = ImageEnhance.Contrast(tmp).enhance(1.12)
        mapped = to_uint8(tmp).astype(np.float32)
    elif lut_type == "mono_soft":
        gray = np.asarray(ImageOps.grayscale(base), dtype=np.float32)
        mapped = np.stack([gray, gray, gray], axis=-1) * 0.95 + 10.0
    else:
        raise ValueError("Unknown lut_type")

    filtered = from_uint8(mapped)
    filtered = ImageEnhance.Color(filtered).enhance(saturation)
    return Image.blend(base, filtered, alpha)


def hdr_like(
    image: Image.Image,
    tone_strength: float = 0.5,
    shadow_lift: float = 0.18,
    highlight_compression: float = 0.16,
    local_contrast: float = 0.25,
) -> Image.Image:
    tone_strength = float(np.clip(tone_strength, 0.0, 1.0))
    shadow_lift = float(np.clip(shadow_lift, 0.0, 1.0))
    highlight_compression = float(np.clip(highlight_compression, 0.0, 1.0))
    local_contrast = float(np.clip(local_contrast, 0.0, 1.0))

    array = to_uint8(image).astype(np.float32) / 255.0
    luma = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    shadow_mask = np.square(1.0 - luma)[:, :, None]
    highlight_mask = np.square(luma)[:, :, None]

    lifted = array + shadow_lift * tone_strength * shadow_mask * (1.0 - array)
    compressed = lifted - highlight_compression * tone_strength * highlight_mask * np.maximum(lifted - 0.45, 0.0)
    tone_mapped = from_uint8(compressed * 255.0)

    if local_contrast > 0:
        local = clahe(tone_mapped, clip_limit=1.2 + 2.4 * local_contrast, tile_grid_size=8)
        tone_mapped = Image.blend(tone_mapped, local, 0.2 + 0.35 * local_contrast)

    tone_mapped = ImageEnhance.Contrast(tone_mapped).enhance(1.0 + 0.08 * tone_strength)
    return ImageEnhance.Color(tone_mapped).enhance(1.0 + 0.05 * tone_strength)


def fade_matte(
    image: Image.Image,
    fade_strength: float = 0.5,
    black_lift: float = 0.12,
    contrast: float = 0.92,
    saturation: float = 0.92,
) -> Image.Image:
    fade_strength = float(np.clip(fade_strength, 0.0, 1.0))
    black_lift = float(np.clip(black_lift, 0.0, 0.5))
    base = image.convert("RGB")
    array = to_uint8(base).astype(np.float32) / 255.0

    lifted = array * (1.0 - black_lift * fade_strength) + black_lift * fade_strength
    lifted = (lifted - 0.5) * float(contrast) + 0.5
    matte = from_uint8(lifted * 255.0)
    matte = ImageEnhance.Color(matte).enhance(float(saturation))
    warm_matte = apply_filter_lut(matte, "matte", alpha=0.35 + 0.35 * fade_strength, saturation=1.0)
    return Image.blend(base, warm_matte, 0.45 + 0.4 * fade_strength)


def mono_style(image: Image.Image, desaturation_ratio: float = 0.85, contrast: float = 1.08, lift: float = 0.04) -> Image.Image:
    desaturation_ratio = float(np.clip(desaturation_ratio, 0.0, 1.0))
    base = image.convert("RGB")
    gray = ImageOps.grayscale(base).convert("RGB")
    mono = Image.blend(base, gray, desaturation_ratio)
    mono = ImageEnhance.Contrast(mono).enhance(float(contrast))
    if lift > 0:
        array = to_uint8(mono).astype(np.float32)
        mono = from_uint8(array * (1.0 - float(lift)) + 255.0 * float(lift))
    return mono


def restore_size(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.LANCZOS) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, method)
