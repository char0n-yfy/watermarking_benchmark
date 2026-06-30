from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from evaluator.attacks.base import AttackContext, BaseAttack
from evaluator.attacks.registry import register_attack

from evaluator.attacks.consumer_enhancement_workflow_attacks.helpers import (
    apply_filter_lut,
    auto_enhance,
    clahe,
    denoise,
    edge_preserve_smooth,
    fade_matte,
    from_uint8,
    gamma_correct,
    gray_world_white_balance,
    hdr_like,
    load_rgb,
    save_png,
    temperature_tint,
    to_uint8,
    unsharp,
)


DEFAULT_WEIGHT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "resources"
    / "weights"
    / "attacks"
    / "consumer_enhancement_workflow_attacks"
)


def _resolve_weight_path(task_name: str, model_name: str, weight_root: str | Path | None = None) -> Path:
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
        "task_name": task_name,
        "model_name": model_name,
        "weight_path": str(path),
        "weight_exists": exists,
        "weight_files": files,
        "allow_fallback": allow_fallback,
    }


def _fallback_metadata(weight_meta: Mapping[str, Any], exc: Exception, ops: list[str]) -> dict[str, Any]:
    return {
        **weight_meta,
        "backend": "heuristic_fallback",
        "fallback_used": True,
        "model_loaded": False,
        "fallback_reason": f"{type(exc).__name__}: {exc}",
        "fallback_ops": ops,
    }


def _torch_metadata(weight_meta: Mapping[str, Any], backend: str) -> dict[str, Any]:
    return {
        **weight_meta,
        "backend": backend,
        "fallback_used": False,
        "model_loaded": True,
        "fallback_reason": None,
    }


def _exposure(image: Image.Image, ev: float) -> Image.Image:
    array = to_uint8(image).astype(np.float32)
    return from_uint8(array * (2.0 ** float(ev)))


def _black_lift(image: Image.Image, lift: float) -> Image.Image:
    lift = float(np.clip(lift, 0.0, 0.5))
    array = to_uint8(image).astype(np.float32)
    return from_uint8(array * (1.0 - lift) + 255.0 * lift)


def _tone_curve_s(image: Image.Image, amount: float) -> Image.Image:
    amount = float(np.clip(amount, -1.0, 1.0))
    array = to_uint8(image).astype(np.float32) / 255.0
    curved = array + amount * (array - 0.5) * (1.0 - np.abs(2.0 * array - 1.0))
    return from_uint8(curved * 255.0)


def _auto_tone(image: Image.Image, exposure_ev: float, contrast: float, vibrance: float, sharpen: float) -> Image.Image:
    result = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    result = _exposure(result, exposure_ev)
    result = ImageEnhance.Contrast(result).enhance(1.0 + contrast)
    result = ImageEnhance.Color(result).enhance(1.0 + vibrance)
    return unsharp(result, radius=1.0, amount=sharpen, threshold=2)


def _warm_vivid(
    image: Image.Image,
    temperature: float,
    saturation: float,
    contrast: float,
    tone_curve: float,
) -> Image.Image:
    result = temperature_tint(image, temperature=temperature, tint=0.02, intensity=1.0)
    result = ImageEnhance.Color(result).enhance(1.0 + saturation)
    result = ImageEnhance.Contrast(result).enhance(1.0 + contrast)
    return _tone_curve_s(result, tone_curve)


def _film_faded(
    image: Image.Image,
    contrast: float,
    saturation: float,
    black_lift: float,
    local_contrast: float,
) -> Image.Image:
    result = fade_matte(
        image,
        fade_strength=max(0.2, black_lift / 0.08),
        black_lift=black_lift,
        contrast=1.0 + contrast,
        saturation=1.0 + saturation,
    )
    if local_contrast < 0:
        smooth = edge_preserve_smooth(result, radius=5, sigma_color=30.0, sigma_space=30.0)
        result = Image.blend(result, smooth, min(0.45, abs(local_contrast) * 3.0))
    return result


def _local_clarity(
    image: Image.Image,
    local_contrast: float,
    sharpen: float,
    shadow_lift: float,
    tone_strength: float,
) -> Image.Image:
    result = hdr_like(
        image,
        tone_strength=tone_strength,
        shadow_lift=shadow_lift,
        highlight_compression=0.12 + 0.08 * tone_strength,
        local_contrast=local_contrast,
    )
    return unsharp(result, radius=1.0, amount=sharpen, threshold=2)


_EDIT_STRENGTH_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "edit_auto_tone": {
        "light": {"exposure_ev": 0.10, "contrast": 0.04, "vibrance": 0.03, "sharpen": 0.4},
        "medium": {"exposure_ev": 0.25, "contrast": 0.08, "vibrance": 0.06, "sharpen": 0.8},
        "strong": {"exposure_ev": 0.40, "contrast": 0.12, "vibrance": 0.10, "sharpen": 1.2},
    },
    "edit_warm_vivid": {
        "light": {"temperature": 0.22, "saturation": 0.05, "contrast": 0.04, "tone_curve": 0.05},
        "medium": {"temperature": 0.40, "saturation": 0.10, "contrast": 0.08, "tone_curve": 0.10},
        "strong": {"temperature": 0.70, "saturation": 0.16, "contrast": 0.12, "tone_curve": 0.18},
    },
    "edit_film_faded": {
        "light": {"contrast": -0.03, "saturation": -0.04, "black_lift": 0.01, "local_contrast": -0.02},
        "medium": {"contrast": -0.06, "saturation": -0.08, "black_lift": 0.03, "local_contrast": -0.05},
        "strong": {"contrast": -0.10, "saturation": -0.14, "black_lift": 0.06, "local_contrast": -0.08},
    },
    "edit_local_clarity": {
        "light": {"local_contrast": 0.08, "tone_strength": 0.20, "sharpen": 0.5, "shadow_lift": 0.03},
        "medium": {"local_contrast": 0.20, "tone_strength": 0.50, "sharpen": 1.0, "shadow_lift": 0.08},
        "strong": {"local_contrast": 0.35, "tone_strength": 0.78, "sharpen": 1.5, "shadow_lift": 0.12},
    },
}


_EDIT_STRENGTH_ALIASES = {
    "l": 0.0,
    "low": 0.0,
    "light": 0.0,
    "m": 0.5,
    "med": 0.5,
    "medium": 0.5,
    "default": 0.5,
    "s": 1.0,
    "high": 1.0,
    "strong": 1.0,
}


def _clamp_edit_strength(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_edit_strength(value: Any) -> float:
    if value is None:
        return 0.5
    if isinstance(value, (int, float)):
        return _clamp_edit_strength(float(value))
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _EDIT_STRENGTH_ALIASES:
        raise ValueError(f"Unsupported CEW edit strength: {value!r}. Use a 0..1 value, light, medium, or strong.")
    return _EDIT_STRENGTH_ALIASES[normalized]


def _piecewise_edit_value(strength: float, light: float, medium: float, strong: float) -> float:
    value = _clamp_edit_strength(strength)
    if value <= 0.5:
        return light + (medium - light) * (value / 0.5)
    return medium + (strong - medium) * ((value - 0.5) / 0.5)


def _continuous_edit_params(operation: str, strength: float) -> dict[str, Any]:
    presets = _EDIT_STRENGTH_PRESETS[operation]
    params = dict(presets["medium"])
    for key in params:
        light = presets["light"].get(key)
        medium = presets["medium"].get(key)
        strong = presets["strong"].get(key)
        if isinstance(light, (int, float)) and isinstance(medium, (int, float)) and isinstance(strong, (int, float)):
            params[key] = _piecewise_edit_value(strength, float(light), float(medium), float(strong))
    return params


def _edit_params(operation: str, config: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    strength = _normalize_edit_strength(config.get("strength", 0.5))
    params = _continuous_edit_params(operation, strength)
    for key in list(params):
        if key in config:
            params[key] = config[key]
    return strength, params


def _auto_light(image: Image.Image, strength: float = 0.65) -> Image.Image:
    result = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    result = clahe(result, clip_limit=1.4 + 1.8 * strength, tile_grid_size=8)
    result = gamma_correct(result, gamma=1.0 + 0.35 * strength)
    return Image.blend(image.convert("RGB"), result, 0.5 + 0.4 * strength)


def _auto_white_balance(image: Image.Image, strength: float = 0.85) -> Image.Image:
    balanced = gray_world_white_balance(image, strength=strength)
    return Image.blend(image.convert("RGB"), balanced, 0.85)


def _adaptive_color(image: Image.Image, alpha: float = 0.55) -> Image.Image:
    vivid = apply_filter_lut(image, "vivid", alpha=0.45 + 0.25 * alpha, saturation=1.04)
    warm = apply_filter_lut(vivid, "warm_film", alpha=0.18 + 0.15 * alpha, saturation=1.02)
    return _tone_curve_s(warm, 0.12)


def _detail_low_light(image: Image.Image, strength: float = 0.75) -> Image.Image:
    result = _auto_light(image, strength=0.6 + 0.25 * strength)
    result = hdr_like(result, tone_strength=0.45 + 0.35 * strength, shadow_lift=0.16, local_contrast=0.35)
    return unsharp(result, radius=1.1, amount=0.35 + 0.35 * strength, threshold=2)


def _ai_denoise(image: Image.Image, strength: float = 0.65) -> Image.Image:
    cleaned = denoise(image, strength=strength, method="nlm")
    cleaned = edge_preserve_smooth(cleaned, radius=5, sigma_color=22.0 + 30.0 * strength, sigma_space=22.0 + 30.0 * strength)
    return Image.blend(image.convert("RGB"), cleaned, 0.38 + 0.42 * strength)


def _fallback_sr(image: Image.Image, scale: int, sharpen: float = 0.28, denoise_strength: float = 0.0) -> Image.Image:
    if denoise_strength > 0:
        image = denoise(image, strength=denoise_strength, method="nlm")
    size = (max(1, image.size[0] * int(scale)), max(1, image.size[1] * int(scale)))
    result = image.resize(size, Image.Resampling.LANCZOS)
    if sharpen > 0:
        result = unsharp(result, radius=1.0, amount=sharpen, threshold=3)
    return result


_SR_VARIANTS: dict[str, dict[int, dict[str, Any]]] = {
    "realesrgan": {
        2: {"model_name": "realesrgan_x2plus", "sharpen": 0.25, "denoise_strength": 0.03},
        4: {"model_name": "realesrgan_x4plus", "sharpen": 0.30, "denoise_strength": 0.04},
    },
    "swinir": {
        2: {"model_name": "swinir_x2", "sharpen": 0.22, "denoise_strength": 0.02},
        4: {"model_name": "swinir_x4", "sharpen": 0.28, "denoise_strength": 0.03},
    },
    "bsrgan": {
        2: {"model_name": "bsrgan_x2", "sharpen": 0.24, "denoise_strength": 0.05},
        4: {"model_name": "bsrgan_x4", "sharpen": 0.28, "denoise_strength": 0.06},
    },
}

_SR_MODEL_TO_FAMILY_SCALE: dict[str, tuple[str, int]] = {
    variant["model_name"]: (family, scale)
    for family, variants in _SR_VARIANTS.items()
    for scale, variant in variants.items()
}


def _normalize_sr_scale(value: Any) -> int:
    try:
        scale = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported CEW-SR scale: {value!r}. Use 2 or 4.") from exc
    if scale not in {2, 4}:
        raise ValueError(f"Unsupported CEW-SR scale: {value!r}. Use 2 or 4.")
    return scale


def _resolve_sr_params(params: Mapping[str, Any]) -> dict[str, Any]:
    requested_model = params.get("model_name")
    if requested_model in _SR_MODEL_TO_FAMILY_SCALE:
        family, inferred_scale = _SR_MODEL_TO_FAMILY_SCALE[str(requested_model)]
    else:
        family = str(params.get("model_family") or params.get("model") or requested_model or "realesrgan").lower()
        inferred_scale = 2
    family = family.replace("-", "").replace("_", "")
    family_aliases = {
        "real_esrgan": "realesrgan",
        "realesrgan": "realesrgan",
        "swinir": "swinir",
        "bsrgan": "bsrgan",
    }
    if family not in family_aliases:
        raise ValueError(f"Unsupported CEW-SR model family: {family!r}. Use realesrgan, swinir, or bsrgan.")
    family = family_aliases[family]
    scale = _normalize_sr_scale(params.get("scale", inferred_scale))
    resolved = {"model_family": family, "scale": scale, **_SR_VARIANTS[family][scale]}
    for key in ("allow_fallback", "weight_root", "sharpen", "denoise_strength"):
        if key in params:
            resolved[key] = params[key]
    return resolved


def _deep_enhance(image: Image.Image, operation: str, params: Mapping[str, Any], context: AttackContext) -> tuple[Image.Image, dict[str, Any]]:
    model_name = str(params["model_name"])
    task_name = str(params["task_name"])
    allow_fallback = bool(params.get("allow_fallback", True))
    weight_path = _resolve_weight_path(task_name, model_name, params.get("weight_root"))
    weight_meta = _weight_metadata(weight_path, task_name, model_name, allow_fallback)
    try:
        if operation == "d1":
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.restoration_sr import run_zero_dce_plus_plus

            output = run_zero_dce_plus_plus(image, weight_path, context.device)
            return output, _torch_metadata(weight_meta, "torch_zero_dce_plus_plus")
        if operation == "d2":
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.deep_enhance import run_deepwb_awb

            output = run_deepwb_awb(
                image,
                weight_path,
                context.device,
                max_size=int(params.get("max_size", 656)),
            )
            return output, _torch_metadata(weight_meta, "torch_deepwb_awb")
        if operation == "d3":
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.deep_enhance import run_image_adaptive_3dlut

            output = run_image_adaptive_3dlut(
                image,
                weight_path,
                context.device,
                blend=float(params.get("blend", 1.0)),
            )
            return output, _torch_metadata(weight_meta, "torch_image_adaptive_3dlut")
        if operation == "d4":
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.deep_enhance import run_retinexformer_low_light

            output = run_retinexformer_low_light(
                image,
                weight_path,
                context.device,
                window_size=int(params.get("window_size", 4)),
            )
            return output, _torch_metadata(weight_meta, "torch_retinexformer_low_light")
        if operation == "d5":
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.restoration_sr import run_restormer_denoise

            output = run_restormer_denoise(image, weight_path, context.device)
            return output, _torch_metadata(weight_meta, "torch_restormer_denoise")
        raise NotImplementedError(f"No local torch backend is wired for {operation}/{model_name}")
    except Exception as exc:
        if not allow_fallback:
            raise
        if operation == "d1":
            output = _auto_light(image, float(params.get("strength", 0.65)))
            ops = ["autocontrast", "clahe", "gamma"]
        elif operation == "d2":
            output = _auto_white_balance(image, float(params.get("strength", 0.85)))
            ops = ["gray_world_white_balance"]
        elif operation == "d3":
            output = _adaptive_color(image, float(params.get("strength", 0.55)))
            ops = ["adaptive_lut", "tone_curve"]
        elif operation == "d4":
            output = _detail_low_light(image, float(params.get("strength", 0.75)))
            ops = ["low_light_gamma", "hdr_like", "unsharp"]
        elif operation == "d5":
            output = _ai_denoise(image, float(params.get("strength", 0.65)))
            ops = ["nlm_denoise", "edge_preserve_smooth"]
        else:
            raise
        return output, _fallback_metadata(weight_meta, exc, ops)


def _sr(image: Image.Image, params: Mapping[str, Any], context: AttackContext) -> tuple[Image.Image, dict[str, Any]]:
    params = _resolve_sr_params(params)
    model_name = str(params["model_name"])
    task_name = "super_resolution"
    scale = int(params["scale"])
    allow_fallback = bool(params.get("allow_fallback", True))
    weight_path = _resolve_weight_path(task_name, model_name, params.get("weight_root"))
    weight_meta = _weight_metadata(weight_path, task_name, model_name, allow_fallback)
    runtime_meta = {
        "model_family": params["model_family"],
        "model_name": model_name,
        "scale": scale,
        "sharpen": params.get("sharpen"),
        "denoise_strength": params.get("denoise_strength"),
    }
    try:
        if model_name in {"realesrgan_x2plus", "bsrgan_x2", "bsrgan_x2_rescaled"} and scale == 2:
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.restoration_sr import run_rrdbnet_x2

            output = run_rrdbnet_x2(image, weight_path, context.device)
            return output, {**_torch_metadata(weight_meta, "torch_rrdbnet_x2"), **runtime_meta}
        if model_name in {"realesrgan_x4plus", "bsrgan_x4"} and scale == 4:
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.restoration_sr import run_rrdbnet_x4

            output = run_rrdbnet_x4(image, weight_path, context.device)
            return output, {**_torch_metadata(weight_meta, "torch_rrdbnet_x4"), **runtime_meta}
        if model_name in {"swinir_x2", "swinir_x4"}:
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends.restoration_sr import run_swinir_classical_sr

            output = run_swinir_classical_sr(image, weight_path, scale, context.device)
            return output, {**_torch_metadata(weight_meta, f"torch_swinir_classical_sr_x{scale}"), **runtime_meta}
        raise NotImplementedError(f"No local torch backend is wired for {model_name} x{scale}")
    except Exception as exc:
        if not allow_fallback:
            raise
        output = _fallback_sr(
            image,
            scale=scale,
            sharpen=float(params.get("sharpen", 0.28)),
            denoise_strength=float(params.get("denoise_strength", 0.0)),
        )
        return output, {**_fallback_metadata(weight_meta, exc, ["lanczos_upscale", "unsharp"]), **runtime_meta}


def _apply_named_step(
    image: Image.Image,
    step: str | Mapping[str, Any],
    context: AttackContext,
    inherited_params: Mapping[str, Any] | None = None,
) -> tuple[Image.Image, str, dict[str, Any]]:
    if isinstance(step, str):
        step_name = step
        step_params: Mapping[str, Any] = {}
    else:
        step_name = str(step.get("name") or step.get("attack") or step.get("method"))
        step_params = step.get("params") if isinstance(step.get("params"), Mapping) else {
            key: value
            for key, value in step.items()
            if key not in {"name", "attack", "method"}
        }
    cls = ATTACK_CLASS_BY_NAME[step_name]
    inherited_params = inherited_params or {}
    overrides = {
        key: value
        for key, value in inherited_params.items()
        if key in getattr(cls, "default_params", {})
    }
    overrides.update(step_params)
    attack = cls(**overrides)
    output, metadata = attack._apply_image(image, context)
    return output, step_name, metadata


class CEWAttack(BaseAttack):
    operation: str = ""
    default_params: Mapping[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        merged = {**self.default_params, **params}
        super().__init__(**merged)
        self.config = merged

    def _apply_image(self, image: Image.Image, context: AttackContext) -> tuple[Image.Image, dict[str, Any]]:
        operation = self.operation
        if operation == "edit_auto_tone":
            strength, params = _edit_params(operation, self.config)
            output = _auto_tone(image, **params)
            return output, {"backend": "pil_darktable_style", "fallback_used": False, "strength": strength, **params}
        if operation == "edit_warm_vivid":
            strength, params = _edit_params(operation, self.config)
            output = _warm_vivid(image, **params)
            return output, {"backend": "pil_darktable_style", "fallback_used": False, "strength": strength, **params}
        if operation == "edit_film_faded":
            strength, params = _edit_params(operation, self.config)
            output = _film_faded(image, **params)
            return output, {"backend": "pil_darktable_style", "fallback_used": False, "strength": strength, **params}
        if operation == "edit_local_clarity":
            strength, params = _edit_params(operation, self.config)
            output = _local_clarity(image, **params)
            return output, {"backend": "pil_darktable_style", "fallback_used": False, "strength": strength, **params}
        if operation in {"d1", "d2", "d3", "d4", "d5"}:
            return _deep_enhance(image, operation, self.config, context)
        if operation == "sr":
            return _sr(image, self.config, context)
        if operation == "composite":
            current = image
            metadata_steps: list[dict[str, Any]] = []
            for step in self.config["chain"]:
                current, step_name, step_meta = _apply_named_step(current, step, context, self.config)
                metadata_steps.append({"step": step_name, **step_meta})
            return current, {"backend": "cew_composite_chain", "fallback_used": any(step.get("fallback_used") for step in metadata_steps), "steps": metadata_steps}
        raise ValueError(f"Unsupported CEW operation: {operation}")

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        output, metadata = self._apply_image(load_rgb(input_path), context)
        save_png(output, output_path)
        return metadata


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Attack"


def _register(name: str, description: str, operation: str, params: Mapping[str, Any]) -> type[CEWAttack]:
    cls = type(
        _class_name(name),
        (CEWAttack,),
        {
            "name": name,
            "description": description,
            "operation": operation,
            "default_params": dict(params),
            "__module__": __name__,
        },
    )
    registered = register_attack(cls)
    globals()[registered.__name__] = registered
    ATTACK_CLASS_BY_NAME[name] = registered
    return registered


ATTACK_CLASS_BY_NAME: dict[str, type[CEWAttack]] = {}


_EDIT_PRESETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("cew_e1", "CEW-E1 Auto-Tone. Smooth 0..1 strength interpolates light/medium/strong edit presets.", "edit_auto_tone", {"strength": 0.5}),
    ("cew_e2", "CEW-E2 Warm-Vivid. Smooth 0..1 strength interpolates light/medium/strong edit presets.", "edit_warm_vivid", {"strength": 0.5}),
    ("cew_e3", "CEW-E3 Film-Faded. Smooth 0..1 strength interpolates light/medium/strong edit presets.", "edit_film_faded", {"strength": 0.5}),
    ("cew_e4", "CEW-E4 Local-Clarity HDR. Smooth 0..1 strength interpolates light/medium/strong edit presets.", "edit_local_clarity", {"strength": 0.5}),
]

_DEEP_PRESETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("cew_d1", "CEW-D1 Auto-Light using Zero-DCE++ style fallback when weights are unavailable.", "d1", {"task_name": "deep_enhance", "model_name": "zero_dce_plus_plus", "strength": 0.65, "allow_fallback": True}),
    ("cew_d2", "CEW-D2 Auto-WhiteBalance using DeepWB AWB when weights are available.", "d2", {"task_name": "deep_enhance", "model_name": "deepwb_awb", "strength": 0.85, "allow_fallback": True}),
    ("cew_d3", "CEW-D3 Adaptive AI Color using Image-Adaptive 3D LUT when weights are available.", "d3", {"task_name": "deep_enhance", "model_name": "image_adaptive_3dlut_fivek", "strength": 0.55, "allow_fallback": True}),
    ("cew_d4", "CEW-D4 Detail Low-Light Enhance using Retinexformer when weights are available.", "d4", {"task_name": "deep_enhance", "model_name": "retinexformer_low_light", "strength": 0.75, "allow_fallback": True}),
    ("cew_d5", "CEW-D5 AI-Denoise Clean using NAFNet/Restormer style fallback.", "d5", {"task_name": "deep_enhance", "model_name": "restormer_or_nafnet_denoise", "strength": 0.65, "allow_fallback": True}),
]

_SR_PRESETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("cew_s1", "CEW-S1 RealESRGAN. Select x2/x4 with the scale parameter.", "sr", {"model_family": "realesrgan", "scale": 2, "allow_fallback": True}),
    ("cew_s2", "CEW-S2 SwinIR. Select x2/x4 with the scale parameter.", "sr", {"model_family": "swinir", "scale": 2, "allow_fallback": True}),
    ("cew_s3", "CEW-S3 BSRGAN. Select x2/x4 with the scale parameter.", "sr", {"model_family": "bsrgan", "scale": 2, "allow_fallback": True}),
]

_COMPOSITE_PRESETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("cew_c1", "CEW-C1 Basic Auto-Fix SR: D1 -> D2 -> D5 -> S1.", "composite", {"chain": ["cew_d1", "cew_d2", "cew_d5", "cew_s1"]}),
    ("cew_c2", "CEW-C2 Color Retouch SR: D2 -> D3 -> D5 -> SwinIR x2.", "composite", {"chain": ["cew_d2", "cew_d3", "cew_d5", {"name": "cew_s2", "params": {"scale": 2}}]}),
    ("cew_c3", "CEW-C3 Detail Enhance SR: D1 -> D4 -> D5 -> RealESRGAN x4.", "composite", {"chain": ["cew_d1", "cew_d4", "cew_d5", {"name": "cew_s1", "params": {"scale": 4}}]}),
    ("cew_c4", "CEW-C4 Full Enhancement Chain: D1 -> D2 -> D3 -> D4 -> D5 -> BSRGAN x4.", "composite", {"chain": ["cew_d1", "cew_d2", "cew_d3", "cew_d4", "cew_d5", {"name": "cew_s3", "params": {"scale": 4}}]}),
]

for _name, _description, _operation, _params in [*_EDIT_PRESETS, *_DEEP_PRESETS, *_SR_PRESETS, *_COMPOSITE_PRESETS]:
    _register(_name, _description, _operation, _params)


__all__ = sorted(name for name in globals() if name.startswith("Cew") and name.endswith("Attack"))
