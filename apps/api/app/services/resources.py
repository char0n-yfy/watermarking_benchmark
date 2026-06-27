from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.attacks import ATTACK_REGISTRY
from evaluator.watermarking import WATERMARK_REGISTRY


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class DatasetResource:
    id: str
    name: str
    sample_count: int
    version: str
    path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sampleCount": self.sample_count,
            "version": self.version,
            "path": str(self.path),
        }


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def scan_dataset_resources(resources_root: Path) -> list[DatasetResource]:
    datasets_root = resources_root / "datasets"
    if not datasets_root.exists():
        return []

    direct_images = iter_image_paths(datasets_root)
    children = sorted(path for path in datasets_root.iterdir() if path.is_dir())
    resources: list[DatasetResource] = []

    if direct_images:
        resources.append(
            DatasetResource(
                id="local-root",
                name="Local dataset root",
                sample_count=len(direct_images),
                version="local",
                path=datasets_root,
            )
        )

    for child in children:
        images = iter_image_paths(child)
        if not images:
            continue
        resources.append(
            DatasetResource(
                id=child.name,
                name=child.name.replace("_", " ").replace("-", " ").title(),
                sample_count=len(images),
                version="local",
                path=child,
            )
        )

    return resources


def get_dataset_by_id(resources_root: Path, dataset_id: str) -> DatasetResource:
    for dataset in scan_dataset_resources(resources_root):
        if dataset.id == dataset_id:
            return dataset
    raise KeyError(f"Unknown dataset id: {dataset_id}")


WATERMARK_DISPLAY_NAMES = {
    "blind_watermark": "Blind Watermark",
    "chunkyseal": "ChunkySeal",
    "cin": "CIN",
    "hidden": "HiDDeN",
    "invismark": "InvisMark",
    "invisible-watermark-dwtdct": "Invisible Watermark DWT-DCT",
    "invisible-watermark-dwtdctsvd": "Invisible Watermark DWT-DCT-SVD",
    "invisible-watermark-rivagan": "Invisible Watermark RivaGAN",
    "maskwm-d32": "MaskWM-D32",
    "mbrs": "MBRS",
    "pimog": "PIMoG",
    "pixelseal": "PixelSeal",
    "rawatermark": "RAWatermark",
    "ssl-watermarking": "SSL Watermarking",
    "stegastamp": "StegaStamp",
    "traditional-dct": "Traditional DCT",
    "traditional-haar": "Traditional Haar",
    "traditional-lsb": "Traditional LSB",
    "traditional-spread-dct": "Traditional Spread DCT",
    "trustmark": "TrustMark",
    "trustmark-c": "TrustMark-C",
    "trustmark-q": "TrustMark-Q",
    "videoseal": "VideoSeal",
    "wam": "WAM",
}

ATTACK_DISPLAY_NAMES = {
    "2x_regen": "2x Regeneration",
    "4x_regen": "4x Regeneration",
    "cp_app_edit_pipeline": "CP App Edit Pipeline",
    "cp_auto_enhance": "CP Auto Enhance",
    "cp_clahe": "CP CLAHE",
    "cp_clean_export_pipeline": "CP Clean Export Pipeline",
    "cp_color_balance": "CP Color Balance",
    "cp_color_space_pipeline": "CP Color Space Pipeline",
    "cp_deartifact": "CP Deartifact",
    "cp_deartifact_deep": "CP Deartifact Deep",
    "cp_deblock": "CP Deblock",
    "cp_deblock_deep": "CP Deblock Deep",
    "cp_denoise": "CP Denoise",
    "cp_denoise_deep": "CP Denoise Deep",
    "cp_despeckle": "CP Despeckle",
    "cp_edge_preserve_smooth": "CP Edge-Preserve Smooth",
    "cp_fade_matte": "CP Fade Matte",
    "cp_filter_lut": "CP Filter LUT",
    "cp_hdr_like": "CP HDR-Like",
    "cp_iterative_export": "CP Iterative Export",
    "cp_metadata_strip_export": "CP Metadata Strip Export",
    "cp_mono_style": "CP Mono Style",
    "cp_platform_pipeline": "CP Platform Pipeline",
    "cp_platform_retouch": "CP Platform Retouch",
    "cp_preview_pipeline": "CP Preview Pipeline",
    "cp_resample_restore": "CP Resample Restore",
    "cp_restore_pipeline_deep": "CP Restore Pipeline Deep",
    "cp_retouch_pipeline_core": "CP Retouch Pipeline Core",
    "cp_sharpen": "CP Sharpen",
    "cp_social_export": "CP Social Export",
    "cp_sr_denoise": "CP SR Denoise",
    "cp_super_resolution": "CP Super Resolution",
    "cp_super_resolution_deep": "CP Super Resolution Deep",
    "cp_thumbnail_restore": "CP Thumbnail Restore",
    "cp_thumbnail_restore_deep": "CP Thumbnail Restore Deep",
    "cp_vivid_boost": "CP Vivid Boost",
    "cp_warm_cold_tone": "CP Warm/Cold Tone",
    "gaussian_blur": "Gaussian Blur",
    "gaussian_noise": "Gaussian Noise",
    "jpeg": "JPEG Compression",
    "regen_diffusion": "Diffusion Regeneration",
    "regen_vae": "VAE Regeneration",
    "resized_crop": "Resized Crop",
}

WATERMARK_CPU_METHODS = {
    "blind_watermark",
    "invisible-watermark-dwtdct",
    "invisible-watermark-dwtdctsvd",
    "traditional-dct",
    "traditional-haar",
    "traditional-lsb",
    "traditional-spread-dct",
}

WATERMARK_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "traditional-dct": {"payload_bits": 16},
    "traditional-haar": {"payload_bits": 16},
    "traditional-lsb": {"payload_bits": 16},
    "traditional-spread-dct": {"payload_bits": 16},
}

RECOMMENDED_WATERMARKS = {"traditional-lsb"}

ATTACK_STRENGTH_SWEEPS: dict[str, list[float]] = {
    "brightness": [0.25, 0.5, 0.75],
    "contrast": [0.25, 0.5, 0.75],
    "cp_denoise": [0.5],
    "cp_denoise_deep": [0.5],
    "cp_despeckle": [0.5],
    "erasing": [0.25, 0.5, 0.75],
    "gaussian_blur": [0.2, 0.4, 0.6],
    "gaussian_noise": [0.25, 0.5, 0.75],
    "jpeg": [0.25, 0.5, 0.75],
    "resized_crop": [0.1, 0.3, 0.5],
    "rotation": [0.25, 0.5, 0.75],
}

LEGACY_ATTACK_PRESETS: dict[str, dict[str, Any]] = {
    "atk-jpeg-smoke": {
        "id": "atk-jpeg-smoke",
        "name": "JPEG smoke",
        "method": "jpeg",
        "strengths": [0.5],
        "strengthParam": "strength",
        "recommended": True,
        "params": {},
    },
    "atk-blur-smoke": {
        "id": "atk-blur-smoke",
        "name": "Blur smoke",
        "method": "gaussian_blur",
        "strengths": [0.2],
        "strengthParam": "strength",
        "recommended": False,
        "params": {},
    },
    "atk-jpeg-sweep": {
        "id": "atk-jpeg-sweep",
        "name": "JPEG sweep",
        "method": "jpeg",
        "strengths": [0.25, 0.5, 0.75],
        "strengthParam": "strength",
        "recommended": False,
        "params": {},
    },
    "atk-blur-sweep": {
        "id": "atk-blur-sweep",
        "name": "Blur sweep",
        "method": "gaussian_blur",
        "strengths": [0.2, 0.4, 0.6],
        "strengthParam": "strength",
        "recommended": False,
        "params": {},
    },
    "atk-crop-sweep": {
        "id": "atk-crop-sweep",
        "name": "Crop sweep",
        "method": "resized_crop",
        "strengths": [0.1, 0.3, 0.5],
        "strengthParam": "strength",
        "recommended": False,
        "params": {},
    },
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "resource"


def _display_name(method: str, overrides: dict[str, str]) -> str:
    if method in overrides:
        return overrides[method]
    return method.replace("_", " ").replace("-", " ").title()


def _watermark_category(method: str) -> str:
    if method.startswith("traditional") or method in WATERMARK_CPU_METHODS:
        return "classical"
    if method in {"videoseal", "pixelseal", "chunkyseal"}:
        return "videoseal-family"
    return "neural"


def _attack_category(method: str) -> str:
    if method.startswith("cp_"):
        return "content-preserving"
    if "regen" in method:
        return "regeneration"
    return "distortion"


def _has_explicit_init_param(cls: type[Any], parameter_name: str) -> bool:
    return parameter_name in inspect.signature(cls.__init__).parameters


def _attack_requires_gpu(method: str) -> bool:
    return "regen" in method or method.endswith("_deep")


def _build_watermark_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for method, cls in sorted(WATERMARK_REGISTRY.items()):
        resource_id = f"alg-{_slug(method)}"
        catalog[resource_id] = {
            "id": resource_id,
            "name": _display_name(method, WATERMARK_DISPLAY_NAMES),
            "method": method,
            "description": cls.description,
            "category": _watermark_category(method),
            "version": "local" if method.startswith("traditional") else "packaged",
            "status": "enabled",
            "requiresGpu": method not in WATERMARK_CPU_METHODS,
            "recommended": method in RECOMMENDED_WATERMARKS,
            "available": True,
            "params": dict(WATERMARK_PARAM_OVERRIDES.get(method, {})),
        }
    return catalog


def _base_attack_preset(method: str, cls: type[Any]) -> dict[str, Any]:
    strength_param = "strength" if _has_explicit_init_param(cls, "strength") else None
    strengths = ATTACK_STRENGTH_SWEEPS.get(method, [0.5] if strength_param else [0.0])
    return {
        "id": f"atk-{_slug(method)}",
        "name": _display_name(method, ATTACK_DISPLAY_NAMES),
        "method": method,
        "description": cls.description,
        "category": _attack_category(method),
        "strengths": strengths,
        "strengthParam": strength_param,
        "requiresGpu": _attack_requires_gpu(method),
        "recommended": method == "identity",
        "available": True,
        "params": {},
    }


def _build_attack_catalog() -> dict[str, dict[str, Any]]:
    catalog = {
        f"atk-{_slug(method)}": _base_attack_preset(method, cls)
        for method, cls in sorted(ATTACK_REGISTRY.items())
    }
    for preset_id, preset in LEGACY_ATTACK_PRESETS.items():
        if preset["method"] in ATTACK_REGISTRY:
            base = _base_attack_preset(preset["method"], ATTACK_REGISTRY[preset["method"]])
            catalog[preset_id] = {
                **base,
                **preset,
                "description": base["description"],
                "category": base["category"],
                "requiresGpu": base["requiresGpu"],
                "available": True,
            }
    return catalog


def list_watermark_resources() -> list[dict[str, Any]]:
    return list(_build_watermark_catalog().values())


def list_attack_resources() -> list[dict[str, Any]]:
    return list(_build_attack_catalog().values())


def get_watermark_catalog_item(algorithm_id: str) -> dict[str, Any]:
    catalog = _build_watermark_catalog()
    if algorithm_id in catalog:
        return catalog[algorithm_id]
    if algorithm_id in {item["method"] for item in catalog.values()}:
        for item in catalog.values():
            if item["method"] == algorithm_id:
                return item
    raise KeyError(f"Unknown watermark algorithm id: {algorithm_id}")


def get_attack_catalog_item(attack_preset_id: str) -> dict[str, Any]:
    catalog = _build_attack_catalog()
    if attack_preset_id in catalog:
        return catalog[attack_preset_id]
    if attack_preset_id in {item["method"] for item in catalog.values()}:
        for item in catalog.values():
            if item["method"] == attack_preset_id:
                return item
    raise KeyError(f"Unknown attack preset id: {attack_preset_id}")
