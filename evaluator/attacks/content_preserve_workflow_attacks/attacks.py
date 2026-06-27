from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageEnhance, ImageOps

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.registry import register_attack

from .helpers import (
    apply_filter_lut,
    auto_enhance,
    clahe,
    color_space_quantize,
    deartifact,
    denoise,
    despeckle,
    edge_preserve_smooth,
    encode_decode,
    fade_matte,
    gamma_correct,
    gray_world_white_balance,
    hdr_like,
    load_rgb,
    mono_style,
    resize_max_side,
    restore_size,
    save_png,
    temperature_tint,
    unsharp,
)


PROFILE_PRESETS: dict[str, dict[str, float]] = {
    "light": {"strength": 0.25},
    "medium": {"strength": 0.5},
    "strong": {"strength": 0.75},
    "extreme": {"strength": 1.0},
}

DEFAULT_WEIGHT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "resources"
    / "weights"
    / "attacks"
    / "content_preserve_workflow_attacks"
)


def _profile_strength(profile: str, fallback: float) -> float:
    return PROFILE_PRESETS.get(profile, {}).get("strength", fallback)


def _safe_quality(value: int) -> int:
    return max(1, min(100, int(value)))


def _resolve_weight_path(
    task_name: str,
    model_name: str,
    weight_path: str | Path | None,
    weight_root: str | Path | None,
) -> Path:
    if weight_path is not None:
        return Path(weight_path).expanduser()
    root = DEFAULT_WEIGHT_ROOT if weight_root is None else Path(weight_root).expanduser()
    return root / task_name / model_name


def _weight_metadata(path: Path, task_name: str, model_name: str, allow_fallback: bool) -> dict[str, Any]:
    exists = path.exists()
    if not exists and not allow_fallback:
        raise FileNotFoundError(f"Weight path for {task_name}/{model_name} does not exist: {path}")

    files: list[str] = []
    if path.is_file():
        files = [path.name]
    elif path.is_dir():
        files = [item.name for item in sorted(path.iterdir()) if item.is_file()][:10]

    return {
        "backend": "heuristic_fallback",
        "fallback_reason": "deep model loader is not configured for this weight format yet",
        "task_name": task_name,
        "model_name": model_name,
        "weight_path": str(path),
        "weight_exists": exists,
        "weight_files": files,
        "allow_fallback": allow_fallback,
    }


def _torch_backend_metadata(weight_meta: Mapping[str, Any], backend: str) -> dict[str, Any]:
    return {
        **weight_meta,
        "backend": backend,
        "fallback_reason": None,
        "fallback_used": False,
        "model_loaded": True,
    }


def _fallback_metadata(weight_meta: Mapping[str, Any], exc: Exception, ops: list[str]) -> dict[str, Any]:
    return {
        **weight_meta,
        "backend": "heuristic_fallback",
        "fallback_reason": f"{type(exc).__name__}: {exc}",
        "fallback_used": True,
        "model_loaded": False,
        "fallback_ops": ops,
    }


def _restore_pipeline_weight_paths(path: Path) -> dict[str, Path]:
    manifest_path = path if path.is_file() else path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError(f"restore pipeline manifest must be a JSON object: {manifest_path}")
        return {key: (manifest_path.parent / value).resolve() for key, value in manifest.items()}

    return {
        "denoise": DEFAULT_WEIGHT_ROOT / "denoise" / "restormer_denoise",
        "jpeg_car": DEFAULT_WEIGHT_ROOT / "deartifact" / "swinir_car",
        "super_resolution": DEFAULT_WEIGHT_ROOT / "super_resolution" / "real_esrgan",
    }


@register_attack
class CPDenoiseAttack(BaseAttack):
    name = "cp_denoise"
    description = "Content-preserving denoise workflow using NLM or median fallback."

    def __init__(self, strength: float | None = None, profile: str = "medium", method: str = "nlm") -> None:
        strength = _profile_strength(profile, 0.5) if strength is None else float(strength)
        super().__init__(strength=strength, profile=profile, method=method)
        self.strength = strength
        self.method = method

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        attacked = denoise(image, self.strength, self.method)
        save_png(attacked, output_path)
        return {"strength": self.strength, "method": self.method}


@register_attack
class CPDeblockAttack(BaseAttack):
    name = "cp_deblock"
    description = "JPEG encode/decode followed by content-preserving deblocking."

    def __init__(
        self,
        source_quality: int = 75,
        deblock_strength: float | None = None,
        profile: str = "medium",
    ) -> None:
        deblock_strength = _profile_strength(profile, 0.5) if deblock_strength is None else float(deblock_strength)
        source_quality = _safe_quality(source_quality)
        super().__init__(source_quality=source_quality, deblock_strength=deblock_strength, profile=profile)
        self.source_quality = source_quality
        self.deblock_strength = deblock_strength

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        compressed, fmt = encode_decode(image, "jpeg", self.source_quality, subsampling=2)
        attacked = edge_preserve_smooth(
            compressed,
            radius=5,
            sigma_color=25.0 + 55.0 * self.deblock_strength,
            sigma_space=25.0 + 55.0 * self.deblock_strength,
        )
        attacked = Image.blend(compressed, attacked, 0.25 + 0.45 * self.deblock_strength)
        save_png(attacked, output_path)
        return {"source_quality": self.source_quality, "deblock_strength": self.deblock_strength, "format": fmt}


@register_attack
class CPDeartifactAttack(BaseAttack):
    name = "cp_deartifact"
    description = "Remove compression artifacts and ringing while preserving edges."

    def __init__(self, artifact_strength: float | None = None, profile: str = "medium", radius: int = 5) -> None:
        artifact_strength = _profile_strength(profile, 0.45) if artifact_strength is None else float(artifact_strength)
        super().__init__(artifact_strength=artifact_strength, profile=profile, radius=radius)
        self.artifact_strength = artifact_strength
        self.radius = int(radius)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        attacked = deartifact(image, self.artifact_strength, self.radius)
        save_png(attacked, output_path)
        return {"artifact_strength": self.artifact_strength, "radius": self.radius}


@register_attack
class CPDespeckleAttack(BaseAttack):
    name = "cp_despeckle"
    description = "Remove small speckles and tiny texture residues while preserving local structure."

    def __init__(
        self,
        kernel_size: int = 3,
        component_threshold: int = 24,
        strength: float | None = None,
        profile: str = "medium",
        difference_threshold: int = 10,
    ) -> None:
        strength = _profile_strength(profile, 0.6) if strength is None else float(strength)
        super().__init__(
            kernel_size=kernel_size,
            component_threshold=component_threshold,
            strength=strength,
            profile=profile,
            difference_threshold=difference_threshold,
        )
        self.kernel_size = int(kernel_size)
        self.component_threshold = int(component_threshold)
        self.strength = strength
        self.difference_threshold = int(difference_threshold)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = despeckle(
            load_rgb(input_path),
            kernel_size=self.kernel_size,
            component_threshold=self.component_threshold,
            strength=self.strength,
            difference_threshold=self.difference_threshold,
        )
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPEdgePreserveSmoothAttack(BaseAttack):
    name = "cp_edge_preserve_smooth"
    description = "Smooth weak texture/noise while preserving edges."

    def __init__(self, radius: int = 7, sigma_color: float = 45.0, sigma_space: float = 45.0) -> None:
        super().__init__(radius=radius, sigma_color=sigma_color, sigma_space=sigma_space)
        self.radius = int(radius)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = edge_preserve_smooth(load_rgb(input_path), self.radius, self.sigma_color, self.sigma_space)
        save_png(attacked, output_path)
        return {"radius": self.radius, "sigma_color": self.sigma_color, "sigma_space": self.sigma_space}


@register_attack
class CPDenoiseDeepAttack(BaseAttack):
    name = "cp_denoise_deep"
    description = "Deep-denoise workflow entrypoint with weight-path metadata and deterministic fallback."

    def __init__(
        self,
        model_name: str = "restormer_denoise",
        weight_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_fallback: bool = True,
        strength: float | None = None,
        profile: str = "medium",
    ) -> None:
        strength = _profile_strength(profile, 0.65) if strength is None else float(strength)
        super().__init__(
            model_name=model_name,
            weight_path=None if weight_path is None else str(weight_path),
            weight_root=None if weight_root is None else str(weight_root),
            allow_fallback=allow_fallback,
            strength=strength,
            profile=profile,
        )
        self.model_name = model_name
        self.weight_path = _resolve_weight_path("denoise", model_name, weight_path, weight_root)
        self.allow_fallback = bool(allow_fallback)
        self.strength = strength

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        weight_meta = _weight_metadata(self.weight_path, "denoise", self.model_name, self.allow_fallback)
        image = load_rgb(input_path)
        try:
            from .deep_models import run_restormer_denoise

            restored = run_restormer_denoise(image, self.weight_path, context.device)
            attacked = Image.blend(image, restored, 0.5 + 0.5 * min(1.0, self.strength))
            metadata = _torch_backend_metadata(weight_meta, "torch_restormer_denoise")
        except Exception as exc:
            if not self.allow_fallback:
                raise
            restored = denoise(image, max(0.55, self.strength), method="nlm")
            restored = edge_preserve_smooth(
                restored,
                radius=5,
                sigma_color=20.0 + 45.0 * self.strength,
                sigma_space=20.0 + 45.0 * self.strength,
            )
            attacked = Image.blend(image, restored, 0.45 + 0.4 * self.strength)
            metadata = _fallback_metadata(weight_meta, exc, ["nlm_denoise", "edge_preserve_smooth"])
        save_png(attacked, output_path)
        return {**self.params, **metadata}


@register_attack
class CPDeblockDeepAttack(BaseAttack):
    name = "cp_deblock_deep"
    description = "Deep-deblock workflow entrypoint with weight-path metadata and deterministic fallback."

    def __init__(
        self,
        model_name: str = "swinir_jpeg",
        weight_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_fallback: bool = True,
        source_quality: int = 72,
        restore_strength: float | None = None,
        profile: str = "medium",
    ) -> None:
        restore_strength = _profile_strength(profile, 0.65) if restore_strength is None else float(restore_strength)
        source_quality = _safe_quality(source_quality)
        super().__init__(
            model_name=model_name,
            weight_path=None if weight_path is None else str(weight_path),
            weight_root=None if weight_root is None else str(weight_root),
            allow_fallback=allow_fallback,
            source_quality=source_quality,
            restore_strength=restore_strength,
            profile=profile,
        )
        self.model_name = model_name
        self.weight_path = _resolve_weight_path("deblock", model_name, weight_path, weight_root)
        self.allow_fallback = bool(allow_fallback)
        self.source_quality = source_quality
        self.restore_strength = restore_strength

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        weight_meta = _weight_metadata(self.weight_path, "deblock", self.model_name, self.allow_fallback)
        image = load_rgb(input_path)
        compressed, fmt = encode_decode(image, "jpeg", self.source_quality, subsampling=2)
        try:
            from .deep_models import run_swinir_jpeg_car

            restored = run_swinir_jpeg_car(compressed, self.weight_path, context.device)
            restored = Image.blend(compressed, restored, 0.45 + 0.45 * min(1.0, self.restore_strength))
            metadata = _torch_backend_metadata(weight_meta, "torch_swinir_jpeg_car")
        except Exception as exc:
            if not self.allow_fallback:
                raise
            restored = deartifact(compressed, self.restore_strength, radius=7)
            restored = unsharp(restored, radius=0.8, amount=0.18 + 0.25 * self.restore_strength, threshold=3)
            metadata = _fallback_metadata(weight_meta, exc, ["jpeg_decode", "deartifact", "unsharp"])
        save_png(restored, output_path)
        return {
            **self.params,
            **metadata,
            "used_format": fmt,
        }


@register_attack
class CPDeartifactDeepAttack(BaseAttack):
    name = "cp_deartifact_deep"
    description = "Deep artifact-removal workflow entrypoint with weight-path metadata and fallback."

    def __init__(
        self,
        model_name: str = "swinir_car",
        weight_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_fallback: bool = True,
        artifact_strength: float | None = None,
        profile: str = "medium",
    ) -> None:
        artifact_strength = _profile_strength(profile, 0.6) if artifact_strength is None else float(artifact_strength)
        super().__init__(
            model_name=model_name,
            weight_path=None if weight_path is None else str(weight_path),
            weight_root=None if weight_root is None else str(weight_root),
            allow_fallback=allow_fallback,
            artifact_strength=artifact_strength,
            profile=profile,
        )
        self.model_name = model_name
        self.weight_path = _resolve_weight_path("deartifact", model_name, weight_path, weight_root)
        self.allow_fallback = bool(allow_fallback)
        self.artifact_strength = artifact_strength

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        weight_meta = _weight_metadata(self.weight_path, "deartifact", self.model_name, self.allow_fallback)
        image = load_rgb(input_path)
        try:
            from .deep_models import run_swinir_jpeg_car

            restored = run_swinir_jpeg_car(image, self.weight_path, context.device)
            attacked = Image.blend(image, restored, 0.45 + 0.45 * min(1.0, self.artifact_strength))
            metadata = _torch_backend_metadata(weight_meta, "torch_swinir_jpeg_car")
        except Exception as exc:
            if not self.allow_fallback:
                raise
            restored = deartifact(image, self.artifact_strength, radius=7)
            restored = denoise(restored, min(1.0, self.artifact_strength * 0.65), method="nlm")
            attacked = Image.blend(image, restored, 0.4 + 0.4 * self.artifact_strength)
            metadata = _fallback_metadata(weight_meta, exc, ["deartifact", "nlm_denoise"])
        save_png(attacked, output_path)
        return {**self.params, **metadata}


@register_attack
class CPSuperResolutionCoreAttack(BaseAttack):
    name = "cp_super_resolution"
    description = "Low-resolution save and bicubic/Lanczos restoration workflow."

    def __init__(self, downsample_scale: float = 0.6, upsampler: str = "lanczos", sharpen_amount: float = 0.25) -> None:
        if not 0.0 < float(downsample_scale) <= 1.0:
            raise ValueError("downsample_scale must be in (0, 1]")
        super().__init__(downsample_scale=downsample_scale, upsampler=upsampler, sharpen_amount=sharpen_amount)
        self.downsample_scale = float(downsample_scale)
        self.upsampler = upsampler
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        width, height = image.size
        small_size = (max(1, int(width * self.downsample_scale)), max(1, int(height * self.downsample_scale)))
        small = image.resize(small_size, Image.Resampling.BICUBIC)
        resample = Image.Resampling.BICUBIC if self.upsampler == "bicubic" else Image.Resampling.LANCZOS
        attacked = small.resize((width, height), resample)
        if self.sharpen_amount > 0:
            attacked = unsharp(attacked, radius=1.0, amount=self.sharpen_amount, threshold=2)
        save_png(attacked, output_path)
        return {"downsample_scale": self.downsample_scale, "small_size": list(small_size), "upsampler": self.upsampler}


@register_attack
class CPThumbnailRestoreAttack(BaseAttack):
    name = "cp_thumbnail_restore"
    description = "Platform thumbnail generation, compression, then restoration to original size."

    def __init__(self, thumbnail_max_side: int = 768, compression_quality: int = 82, restore_method: str = "lanczos") -> None:
        super().__init__(
            thumbnail_max_side=thumbnail_max_side,
            compression_quality=compression_quality,
            restore_method=restore_method,
        )
        self.thumbnail_max_side = int(thumbnail_max_side)
        self.compression_quality = _safe_quality(compression_quality)
        self.restore_method = restore_method

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        preview = resize_max_side(image, self.thumbnail_max_side)
        preview, fmt = encode_decode(preview, "jpeg", self.compression_quality, subsampling=2)
        method = Image.Resampling.BICUBIC if self.restore_method == "bicubic" else Image.Resampling.LANCZOS
        attacked = restore_size(preview, original_size, method)
        attacked = unsharp(attacked, radius=1.0, amount=0.2, threshold=3)
        save_png(attacked, output_path)
        return {
            "thumbnail_max_side": self.thumbnail_max_side,
            "compression_quality": self.compression_quality,
            "compressed_format": fmt,
            "preview_size": list(preview.size),
        }


@register_attack
class CPResampleRestoreAttack(BaseAttack):
    name = "cp_resample_restore"
    description = "Repeated resampling followed by mild restoration and sharpening."

    def __init__(
        self,
        scale_sequence: tuple[float, ...] = (0.85, 0.65, 1.0),
        interpolation: str = "mixed",
        sharpen_amount: float = 0.25,
    ) -> None:
        if not scale_sequence:
            raise ValueError("scale_sequence must not be empty")
        scales = tuple(float(scale) for scale in scale_sequence)
        if any(scale <= 0 for scale in scales):
            raise ValueError("all scale values must be > 0")
        super().__init__(scale_sequence=scales, interpolation=interpolation, sharpen_amount=sharpen_amount)
        self.scale_sequence = scales
        self.interpolation = interpolation
        self.sharpen_amount = float(sharpen_amount)

    def _resample_method(self, index: int) -> int:
        if self.interpolation == "bicubic":
            return Image.Resampling.BICUBIC
        if self.interpolation == "bilinear":
            return Image.Resampling.BILINEAR
        if self.interpolation == "nearest":
            return Image.Resampling.NEAREST
        return Image.Resampling.BICUBIC if index % 2 == 0 else Image.Resampling.LANCZOS

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        current = image
        visited_sizes: list[list[int]] = []
        for index, scale in enumerate(self.scale_sequence):
            target_size = (
                max(1, int(original_size[0] * scale)),
                max(1, int(original_size[1] * scale)),
            )
            visited_sizes.append([target_size[0], target_size[1]])
            current = current.resize(target_size, self._resample_method(index))
        current = restore_size(current, original_size, Image.Resampling.LANCZOS)
        current = edge_preserve_smooth(current, radius=3, sigma_color=18.0, sigma_space=18.0)
        if self.sharpen_amount > 0:
            current = unsharp(current, radius=0.9, amount=self.sharpen_amount, threshold=3)
        save_png(current, output_path)
        return {**self.params, "original_size": list(original_size), "visited_sizes": visited_sizes}


@register_attack
class CPSRDenoiseAttack(BaseAttack):
    name = "cp_sr_denoise"
    description = "Low-resolution restoration workflow that combines denoise and super-resolution."

    def __init__(self, downsample_scale: float = 0.55, denoise_strength: float = 0.45, sharpen_amount: float = 0.2) -> None:
        if not 0.0 < float(downsample_scale) <= 1.0:
            raise ValueError("downsample_scale must be in (0, 1]")
        super().__init__(
            downsample_scale=downsample_scale,
            denoise_strength=denoise_strength,
            sharpen_amount=sharpen_amount,
        )
        self.downsample_scale = float(downsample_scale)
        self.denoise_strength = float(denoise_strength)
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        width, height = image.size
        small_size = (max(1, int(width * self.downsample_scale)), max(1, int(height * self.downsample_scale)))
        current = image.resize(small_size, Image.Resampling.BICUBIC)
        current = denoise(current, self.denoise_strength, method="nlm")
        current = current.resize((width, height), Image.Resampling.LANCZOS)
        if self.sharpen_amount > 0:
            current = unsharp(current, radius=1.0, amount=self.sharpen_amount, threshold=3)
        save_png(current, output_path)
        return {**self.params, "small_size": list(small_size)}


@register_attack
class CPSuperResolutionDeepAttack(BaseAttack):
    name = "cp_super_resolution_deep"
    description = "Deep super-resolution workflow entrypoint with weight-path metadata and fallback."

    def __init__(
        self,
        model_name: str = "real_esrgan",
        weight_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_fallback: bool = True,
        downsample_scale: float = 0.5,
        sr_scale: int = 2,
        denoise_strength: float = 0.12,
        sharpen_amount: float = 0.28,
    ) -> None:
        if not 0.0 < float(downsample_scale) <= 1.0:
            raise ValueError("downsample_scale must be in (0, 1]")
        super().__init__(
            model_name=model_name,
            weight_path=None if weight_path is None else str(weight_path),
            weight_root=None if weight_root is None else str(weight_root),
            allow_fallback=allow_fallback,
            downsample_scale=downsample_scale,
            sr_scale=sr_scale,
            denoise_strength=denoise_strength,
            sharpen_amount=sharpen_amount,
        )
        self.model_name = model_name
        self.weight_path = _resolve_weight_path("super_resolution", model_name, weight_path, weight_root)
        self.allow_fallback = bool(allow_fallback)
        self.downsample_scale = float(downsample_scale)
        self.sr_scale = int(sr_scale)
        self.denoise_strength = float(denoise_strength)
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        weight_meta = _weight_metadata(self.weight_path, "super_resolution", self.model_name, self.allow_fallback)
        image = load_rgb(input_path)
        original_size = image.size
        small_size = (
            max(1, int(original_size[0] * self.downsample_scale)),
            max(1, int(original_size[1] * self.downsample_scale)),
        )
        current = image.resize(small_size, Image.Resampling.BICUBIC)
        try:
            from .deep_models import run_rrdbnet_x4

            if self.denoise_strength > 0:
                current = denoise(current, self.denoise_strength, method="nlm")
            current = run_rrdbnet_x4(current, self.weight_path, context.device)
            current = restore_size(current, original_size, Image.Resampling.LANCZOS)
            if self.sharpen_amount > 0:
                current = unsharp(current, radius=1.0, amount=self.sharpen_amount, threshold=3)
            metadata = _torch_backend_metadata(weight_meta, "torch_realesrgan_rrdbnet_x4")
        except Exception as exc:
            if not self.allow_fallback:
                raise
            if self.denoise_strength > 0:
                current = denoise(current, self.denoise_strength, method="nlm")
            current = current.resize(original_size, Image.Resampling.LANCZOS)
            current = unsharp(current, radius=1.0, amount=self.sharpen_amount, threshold=3)
            metadata = _fallback_metadata(exc=exc, weight_meta=weight_meta, ops=["bicubic_downsample", "nlm_denoise", "lanczos_restore", "unsharp"])
        save_png(current, output_path)
        return {
            **self.params,
            **metadata,
            "small_size": list(small_size),
        }


@register_attack
class CPThumbnailRestoreDeepAttack(BaseAttack):
    name = "cp_thumbnail_restore_deep"
    description = "Deep thumbnail-restoration workflow entrypoint with weight-path metadata and fallback."

    def __init__(
        self,
        model_name: str = "real_esrgan_thumbnail",
        weight_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_fallback: bool = True,
        thumbnail_max_side: int = 640,
        compression_quality: int = 78,
        sharpen_amount: float = 0.3,
    ) -> None:
        super().__init__(
            model_name=model_name,
            weight_path=None if weight_path is None else str(weight_path),
            weight_root=None if weight_root is None else str(weight_root),
            allow_fallback=allow_fallback,
            thumbnail_max_side=thumbnail_max_side,
            compression_quality=compression_quality,
            sharpen_amount=sharpen_amount,
        )
        self.model_name = model_name
        self.weight_path = _resolve_weight_path("thumbnail_restore", model_name, weight_path, weight_root)
        self.allow_fallback = bool(allow_fallback)
        self.thumbnail_max_side = int(thumbnail_max_side)
        self.compression_quality = _safe_quality(compression_quality)
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        weight_meta = _weight_metadata(self.weight_path, "thumbnail_restore", self.model_name, self.allow_fallback)
        image = load_rgb(input_path)
        original_size = image.size
        preview = resize_max_side(image, self.thumbnail_max_side)
        preview, fmt = encode_decode(preview, "jpeg", self.compression_quality, subsampling=2)
        try:
            from .deep_models import run_rrdbnet_x4

            preview = denoise(preview, 0.18, method="nlm")
            restored = run_rrdbnet_x4(preview, self.weight_path, context.device)
            restored = restore_size(restored, original_size, Image.Resampling.LANCZOS)
            if self.sharpen_amount > 0:
                restored = unsharp(restored, radius=1.0, amount=self.sharpen_amount, threshold=3)
            metadata = _torch_backend_metadata(weight_meta, "torch_realesrgan_rrdbnet_x4")
        except Exception as exc:
            if not self.allow_fallback:
                raise
            preview = denoise(preview, 0.18, method="nlm")
            restored = preview.resize(original_size, Image.Resampling.LANCZOS)
            restored = unsharp(restored, radius=1.0, amount=self.sharpen_amount, threshold=3)
            metadata = _fallback_metadata(weight_meta, exc, ["thumbnail_jpeg", "nlm_denoise", "lanczos_restore", "unsharp"])
        save_png(restored, output_path)
        return {
            **self.params,
            **metadata,
            "used_format": fmt,
            "preview_size": list(preview.size),
        }


@register_attack
class CPAutoEnhanceAttack(BaseAttack):
    name = "cp_auto_enhance"
    description = "One-click auto enhance: autocontrast, white balance, gamma, saturation."

    def __init__(
        self,
        gamma: float = 1.03,
        contrast: float = 1.08,
        saturation: float = 1.08,
        white_balance: float = 0.5,
    ) -> None:
        super().__init__(gamma=gamma, contrast=contrast, saturation=saturation, white_balance=white_balance)
        self.gamma = float(gamma)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.white_balance = float(white_balance)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = auto_enhance(load_rgb(input_path), self.gamma, self.contrast, self.saturation, self.white_balance)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPCLAHEAttack(BaseAttack):
    name = "cp_clahe"
    description = "Local contrast enhancement on luminance channel."

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        super().__init__(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = int(tile_grid_size)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = clahe(load_rgb(input_path), self.clip_limit, self.tile_grid_size)
        save_png(attacked, output_path)
        return {"clip_limit": self.clip_limit, "tile_grid_size": self.tile_grid_size}


@register_attack
class CPHDRLikeAttack(BaseAttack):
    name = "cp_hdr_like"
    description = "Phone-style HDR/tone-mapping workflow with shadow lift and highlight recovery."

    def __init__(
        self,
        tone_strength: float | None = None,
        profile: str = "medium",
        shadow_lift: float = 0.18,
        highlight_compression: float = 0.16,
        local_contrast: float = 0.25,
    ) -> None:
        tone_strength = _profile_strength(profile, 0.5) if tone_strength is None else float(tone_strength)
        super().__init__(
            tone_strength=tone_strength,
            profile=profile,
            shadow_lift=shadow_lift,
            highlight_compression=highlight_compression,
            local_contrast=local_contrast,
        )
        self.tone_strength = tone_strength
        self.shadow_lift = float(shadow_lift)
        self.highlight_compression = float(highlight_compression)
        self.local_contrast = float(local_contrast)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = hdr_like(
            load_rgb(input_path),
            tone_strength=self.tone_strength,
            shadow_lift=self.shadow_lift,
            highlight_compression=self.highlight_compression,
            local_contrast=self.local_contrast,
        )
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPSharpenAttack(BaseAttack):
    name = "cp_sharpen"
    description = "Content-preserving clarity enhancement using unsharp masking."

    def __init__(self, radius: float = 1.2, amount: float = 0.7, threshold: int = 2) -> None:
        super().__init__(radius=radius, amount=amount, threshold=threshold)
        self.radius = float(radius)
        self.amount = float(amount)
        self.threshold = int(threshold)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = unsharp(load_rgb(input_path), self.radius, self.amount, self.threshold)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPColorBalanceAttack(BaseAttack):
    name = "cp_color_balance"
    description = "Automatic white balance and optional temperature/tint correction."

    def __init__(self, white_balance: float = 0.8, temperature: float = 0.0, tint: float = 0.0) -> None:
        super().__init__(white_balance=white_balance, temperature=temperature, tint=tint)
        self.white_balance = float(white_balance)
        self.temperature = float(temperature)
        self.tint = float(tint)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = gray_world_white_balance(load_rgb(input_path), self.white_balance)
        attacked = temperature_tint(image, self.temperature, self.tint, 1.0)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPFilterLUTAttack(BaseAttack):
    name = "cp_filter_lut"
    description = "Deterministic social-app style LUT/filter workflow."

    def __init__(self, lut_type: str = "warm_film", alpha: float = 0.5, saturation: float = 1.0) -> None:
        super().__init__(lut_type=lut_type, alpha=alpha, saturation=saturation)
        self.lut_type = lut_type
        self.alpha = float(alpha)
        self.saturation = float(saturation)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = apply_filter_lut(load_rgb(input_path), self.lut_type, self.alpha, self.saturation)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPWarmColdToneAttack(BaseAttack):
    name = "cp_warm_cold_tone"
    description = "Temperature and tint workflow similar to lightweight app editing."

    def __init__(self, temperature: float = 0.35, tint: float = 0.0, intensity: float = 0.8) -> None:
        super().__init__(temperature=temperature, tint=tint, intensity=intensity)
        self.temperature = float(temperature)
        self.tint = float(tint)
        self.intensity = float(intensity)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = temperature_tint(load_rgb(input_path), self.temperature, self.tint, self.intensity)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPFadeMatteAttack(BaseAttack):
    name = "cp_fade_matte"
    description = "Faded matte film-style edit with lifted blacks and compressed contrast."

    def __init__(
        self,
        fade_strength: float | None = None,
        profile: str = "medium",
        black_lift: float = 0.12,
        contrast: float = 0.92,
        saturation: float = 0.92,
    ) -> None:
        fade_strength = _profile_strength(profile, 0.5) if fade_strength is None else float(fade_strength)
        super().__init__(
            fade_strength=fade_strength,
            profile=profile,
            black_lift=black_lift,
            contrast=contrast,
            saturation=saturation,
        )
        self.fade_strength = fade_strength
        self.black_lift = float(black_lift)
        self.contrast = float(contrast)
        self.saturation = float(saturation)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = fade_matte(
            load_rgb(input_path),
            fade_strength=self.fade_strength,
            black_lift=self.black_lift,
            contrast=self.contrast,
            saturation=self.saturation,
        )
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPVividBoostAttack(BaseAttack):
    name = "cp_vivid_boost"
    description = "Vivid color, local contrast, and clarity boosting workflow."

    def __init__(self, saturation: float = 1.25, contrast: float = 1.10, sharpen_amount: float = 0.35) -> None:
        super().__init__(saturation=saturation, contrast=contrast, sharpen_amount=sharpen_amount)
        self.saturation = float(saturation)
        self.contrast = float(contrast)
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        image = ImageEnhance.Color(image).enhance(self.saturation)
        image = ImageEnhance.Contrast(image).enhance(self.contrast)
        attacked = unsharp(image, radius=1.0, amount=self.sharpen_amount, threshold=2)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPMonoStyleAttack(BaseAttack):
    name = "cp_mono_style"
    description = "Black-and-white or low-saturation app style workflow."

    def __init__(self, desaturation_ratio: float = 0.85, contrast: float = 1.08, lift: float = 0.04) -> None:
        super().__init__(desaturation_ratio=desaturation_ratio, contrast=contrast, lift=lift)
        self.desaturation_ratio = float(desaturation_ratio)
        self.contrast = float(contrast)
        self.lift = float(lift)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        attacked = mono_style(load_rgb(input_path), self.desaturation_ratio, self.contrast, self.lift)
        save_png(attacked, output_path)
        return self.params


@register_attack
class CPPlatformPipelineAttack(BaseAttack):
    name = "cp_platform_pipeline"
    description = "Resize, color conversion, chroma subsampling, compression, and restore workflow."

    def __init__(
        self,
        max_side: int = 1600,
        export_format: str = "jpeg",
        quality: int = 84,
        subsampling: int = 2,
        color_space: str = "ycbcr",
        bit_depth: int = 8,
    ) -> None:
        super().__init__(
            max_side=max_side,
            export_format=export_format,
            quality=quality,
            subsampling=subsampling,
            color_space=color_space,
            bit_depth=bit_depth,
        )
        self.max_side = int(max_side)
        self.export_format = export_format
        self.quality = _safe_quality(quality)
        self.subsampling = subsampling
        self.color_space = color_space
        self.bit_depth = int(bit_depth)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        processed = resize_max_side(image, self.max_side)
        processed = color_space_quantize(processed, self.color_space, self.bit_depth)
        processed, used_format = encode_decode(processed, self.export_format, self.quality, self.subsampling)
        processed = restore_size(processed, original_size)
        save_png(processed, output_path)
        return {**self.params, "used_format": used_format, "original_size": list(original_size), "processed_size": list(processed.size)}


@register_attack
class CPSocialExportAttack(BaseAttack):
    name = "cp_social_export"
    description = "Social-app export: auto enhance, resize, compression, and restore."

    def __init__(self, max_side: int = 1440, quality: int = 82, filter_alpha: float = 0.2) -> None:
        super().__init__(max_side=max_side, quality=quality, filter_alpha=filter_alpha)
        self.max_side = int(max_side)
        self.quality = _safe_quality(quality)
        self.filter_alpha = float(filter_alpha)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        processed = auto_enhance(image, gamma=1.02, contrast=1.06, saturation=1.08, white_balance=0.4)
        processed = apply_filter_lut(processed, "vivid", alpha=self.filter_alpha, saturation=1.05)
        processed = resize_max_side(processed, self.max_side)
        processed, used_format = encode_decode(processed, "jpeg", self.quality, subsampling=2)
        processed = restore_size(processed, original_size)
        save_png(processed, output_path)
        return {**self.params, "used_format": used_format}


@register_attack
class CPIterativeExportAttack(BaseAttack):
    name = "cp_iterative_export"
    description = "Repeated edit-save-reload workflow simulating reposting and re-export."

    def __init__(self, rounds: int = 3, q_start: int = 88, q_decay: int = 5, enhance_strength: float = 0.12) -> None:
        super().__init__(rounds=rounds, q_start=q_start, q_decay=q_decay, enhance_strength=enhance_strength)
        self.rounds = int(rounds)
        self.q_start = _safe_quality(q_start)
        self.q_decay = int(q_decay)
        self.enhance_strength = float(enhance_strength)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        qualities = []
        for idx in range(max(1, self.rounds)):
            image = auto_enhance(
                image,
                gamma=1.0 + self.enhance_strength * 0.1,
                contrast=1.0 + self.enhance_strength,
                saturation=1.0 + self.enhance_strength * 0.5,
                white_balance=0.15,
            )
            q = _safe_quality(self.q_start - idx * self.q_decay)
            qualities.append(q)
            image, _ = encode_decode(image, "jpeg", q, subsampling=2)
        save_png(image, output_path)
        return {**self.params, "qualities": qualities}


@register_attack
class CPColorSpacePipelineAttack(BaseAttack):
    name = "cp_color_space_pipeline"
    description = "Color-management style conversion and quantization workflow."

    def __init__(self, color_space: str = "ycbcr", bit_depth: int = 7, quality: int | None = None) -> None:
        super().__init__(color_space=color_space, bit_depth=bit_depth, quality=quality)
        self.color_space = color_space
        self.bit_depth = int(bit_depth)
        self.quality = None if quality is None else _safe_quality(quality)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = color_space_quantize(load_rgb(input_path), self.color_space, self.bit_depth)
        used_format = None
        if self.quality is not None:
            image, used_format = encode_decode(image, "jpeg", self.quality, subsampling=2)
        save_png(image, output_path)
        return {**self.params, "used_format": used_format}


@register_attack
class CPMetadataStripExportAttack(BaseAttack):
    name = "cp_metadata_strip_export"
    description = "Metadata-stripping export workflow with optional profile conversion and compression."

    def __init__(
        self,
        export_format: str = "jpeg",
        quality: int = 88,
        color_space: str = "rgb",
        bit_depth: int = 8,
        subsampling: int = 2,
    ) -> None:
        super().__init__(
            export_format=export_format,
            quality=quality,
            color_space=color_space,
            bit_depth=bit_depth,
            subsampling=subsampling,
        )
        self.export_format = export_format
        self.quality = _safe_quality(quality)
        self.color_space = color_space
        self.bit_depth = int(bit_depth)
        self.subsampling = subsampling

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        image = color_space_quantize(image, self.color_space, self.bit_depth)
        image, used_format = encode_decode(image, self.export_format, self.quality, self.subsampling)
        save_png(image, output_path)
        return {**self.params, "used_format": used_format, "metadata_stripped": True}


@register_attack
class CPPreviewPipelineAttack(BaseAttack):
    name = "cp_preview_pipeline"
    description = "Preview/thumbnail generation, export, and restore workflow."

    def __init__(
        self,
        preview_max_side: int = 512,
        export_format: str = "jpeg",
        quality: int = 80,
        restore_method: str = "lanczos",
        sharpen_amount: float = 0.18,
    ) -> None:
        super().__init__(
            preview_max_side=preview_max_side,
            export_format=export_format,
            quality=quality,
            restore_method=restore_method,
            sharpen_amount=sharpen_amount,
        )
        self.preview_max_side = int(preview_max_side)
        self.export_format = export_format
        self.quality = _safe_quality(quality)
        self.restore_method = restore_method
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        preview = resize_max_side(image, self.preview_max_side)
        preview, used_format = encode_decode(preview, self.export_format, self.quality, subsampling=2)
        method = Image.Resampling.BICUBIC if self.restore_method == "bicubic" else Image.Resampling.LANCZOS
        restored = restore_size(preview, original_size, method)
        if self.sharpen_amount > 0:
            restored = unsharp(restored, radius=0.9, amount=self.sharpen_amount, threshold=3)
        save_png(restored, output_path)
        return {
            **self.params,
            "used_format": used_format,
            "original_size": list(original_size),
            "preview_size": list(preview.size),
        }


@register_attack
class CPRetouchPipelineCoreAttack(BaseAttack):
    name = "cp_retouch_pipeline_core"
    description = "Core retouch chain: denoise, auto enhance, sharpen, and clean export."

    def __init__(
        self,
        denoise_strength: float = 0.45,
        gamma: float = 1.03,
        sharpen_amount: float = 0.45,
        export_quality: int = 88,
    ) -> None:
        super().__init__(
            denoise_strength=denoise_strength,
            gamma=gamma,
            sharpen_amount=sharpen_amount,
            export_quality=export_quality,
        )
        self.denoise_strength = float(denoise_strength)
        self.gamma = float(gamma)
        self.sharpen_amount = float(sharpen_amount)
        self.export_quality = _safe_quality(export_quality)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = denoise(load_rgb(input_path), self.denoise_strength)
        image = auto_enhance(image, gamma=self.gamma, contrast=1.08, saturation=1.06, white_balance=0.45)
        image = unsharp(image, radius=1.1, amount=self.sharpen_amount, threshold=2)
        image, fmt = encode_decode(image, "jpeg", self.export_quality, subsampling=2)
        save_png(image, output_path)
        return {**self.params, "used_format": fmt}


@register_attack
class CPRestorePipelineDeepAttack(BaseAttack):
    name = "cp_restore_pipeline_deep"
    description = "Deep restoration pipeline entrypoint with weight-path metadata and fallback."

    def __init__(
        self,
        model_name: str = "restore_pipeline",
        weight_path: str | Path | None = None,
        weight_root: str | Path | None = None,
        allow_fallback: bool = True,
        restore_strength: float | None = None,
        profile: str = "medium",
        downsample_scale: float = 0.75,
        export_quality: int = 86,
    ) -> None:
        restore_strength = _profile_strength(profile, 0.6) if restore_strength is None else float(restore_strength)
        if not 0.0 < float(downsample_scale) <= 1.0:
            raise ValueError("downsample_scale must be in (0, 1]")
        super().__init__(
            model_name=model_name,
            weight_path=None if weight_path is None else str(weight_path),
            weight_root=None if weight_root is None else str(weight_root),
            allow_fallback=allow_fallback,
            restore_strength=restore_strength,
            profile=profile,
            downsample_scale=downsample_scale,
            export_quality=export_quality,
        )
        self.model_name = model_name
        self.weight_path = _resolve_weight_path("restore_pipeline", model_name, weight_path, weight_root)
        self.allow_fallback = bool(allow_fallback)
        self.restore_strength = restore_strength
        self.downsample_scale = float(downsample_scale)
        self.export_quality = _safe_quality(export_quality)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        weight_meta = _weight_metadata(self.weight_path, "restore_pipeline", self.model_name, self.allow_fallback)
        image = load_rgb(input_path)
        original_size = image.size
        try:
            from .deep_models import run_restormer_denoise, run_rrdbnet_x4, run_swinir_jpeg_car

            pipeline_weights = _restore_pipeline_weight_paths(self.weight_path)
            image = run_restormer_denoise(image, pipeline_weights["denoise"], context.device)
            image = run_swinir_jpeg_car(image, pipeline_weights["jpeg_car"], context.device)
            if self.downsample_scale < 1.0:
                small_size = (
                    max(1, int(original_size[0] * self.downsample_scale)),
                    max(1, int(original_size[1] * self.downsample_scale)),
                )
                image = image.resize(small_size, Image.Resampling.BICUBIC)
                image = run_rrdbnet_x4(image, pipeline_weights["super_resolution"], context.device)
                image = restore_size(image, original_size, Image.Resampling.LANCZOS)
            else:
                small_size = original_size
            image = auto_enhance(image, gamma=1.02, contrast=1.06, saturation=1.04, white_balance=0.35)
            image = unsharp(image, radius=1.0, amount=0.2 + 0.25 * self.restore_strength, threshold=3)
            image, fmt = encode_decode(image, "jpeg", self.export_quality, subsampling=2)
            metadata = {
                **_torch_backend_metadata(weight_meta, "torch_restore_pipeline"),
                "pipeline_weights": {key: str(value) for key, value in pipeline_weights.items()},
            }
        except Exception as exc:
            if not self.allow_fallback:
                raise
            image = deartifact(image, self.restore_strength, radius=7)
            image = denoise(image, min(1.0, self.restore_strength * 0.75), method="nlm")
            if self.downsample_scale < 1.0:
                small_size = (
                    max(1, int(original_size[0] * self.downsample_scale)),
                    max(1, int(original_size[1] * self.downsample_scale)),
                )
                image = image.resize(small_size, Image.Resampling.BICUBIC)
                image = image.resize(original_size, Image.Resampling.LANCZOS)
            else:
                small_size = original_size
            image = auto_enhance(image, gamma=1.02, contrast=1.06, saturation=1.04, white_balance=0.35)
            image = unsharp(image, radius=1.0, amount=0.2 + 0.25 * self.restore_strength, threshold=3)
            image, fmt = encode_decode(image, "jpeg", self.export_quality, subsampling=2)
            metadata = _fallback_metadata(weight_meta, exc, ["deartifact", "nlm_denoise", "sr_restore", "auto_enhance", "jpeg_export"])
        save_png(image, output_path)
        return {
            **self.params,
            **metadata,
            "used_format": fmt,
            "small_size": list(small_size),
        }


@register_attack
class CPAppEditPipelineAttack(BaseAttack):
    name = "cp_app_edit_pipeline"
    description = "App-style one-click edit: filter, enhance, resize, export."

    def __init__(self, lut_type: str = "warm_film", filter_alpha: float = 0.45, max_side: int = 1600, quality: int = 84) -> None:
        super().__init__(lut_type=lut_type, filter_alpha=filter_alpha, max_side=max_side, quality=quality)
        self.lut_type = lut_type
        self.filter_alpha = float(filter_alpha)
        self.max_side = int(max_side)
        self.quality = _safe_quality(quality)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        image = apply_filter_lut(image, self.lut_type, self.filter_alpha, saturation=1.05)
        image = auto_enhance(image, gamma=1.02, contrast=1.04, saturation=1.04, white_balance=0.25)
        image = resize_max_side(image, self.max_side)
        image, fmt = encode_decode(image, "jpeg", self.quality, subsampling=2)
        image = restore_size(image, original_size)
        save_png(image, output_path)
        return {**self.params, "used_format": fmt}


@register_attack
class CPPlatformRetouchAttack(BaseAttack):
    name = "cp_platform_retouch"
    description = "Platform compression followed by deblocking and sharpen restoration."

    def __init__(self, max_side: int = 1280, quality: int = 78, deblock_strength: float = 0.55, sharpen_amount: float = 0.35) -> None:
        super().__init__(max_side=max_side, quality=quality, deblock_strength=deblock_strength, sharpen_amount=sharpen_amount)
        self.max_side = int(max_side)
        self.quality = _safe_quality(quality)
        self.deblock_strength = float(deblock_strength)
        self.sharpen_amount = float(sharpen_amount)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = load_rgb(input_path)
        original_size = image.size
        image = resize_max_side(image, self.max_side)
        image, fmt = encode_decode(image, "jpeg", self.quality, subsampling=2)
        image = restore_size(image, original_size)
        image = deartifact(image, self.deblock_strength, radius=5)
        image = unsharp(image, radius=1.0, amount=self.sharpen_amount, threshold=3)
        save_png(image, output_path)
        return {**self.params, "used_format": fmt}


@register_attack
class CPCleanExportPipelineAttack(BaseAttack):
    name = "cp_clean_export_pipeline"
    description = "Clean export: denoise, deartifact, color-profile conversion, and compression."

    def __init__(self, denoise_strength: float = 0.35, artifact_strength: float = 0.4, bit_depth: int = 7, quality: int = 90) -> None:
        super().__init__(
            denoise_strength=denoise_strength,
            artifact_strength=artifact_strength,
            bit_depth=bit_depth,
            quality=quality,
        )
        self.denoise_strength = float(denoise_strength)
        self.artifact_strength = float(artifact_strength)
        self.bit_depth = int(bit_depth)
        self.quality = _safe_quality(quality)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        image = denoise(load_rgb(input_path), self.denoise_strength)
        image = deartifact(image, self.artifact_strength)
        image = color_space_quantize(image, "ycbcr", self.bit_depth)
        image, fmt = encode_decode(image, "jpeg", self.quality, subsampling=2)
        save_png(image, output_path)
        return {**self.params, "used_format": fmt}
