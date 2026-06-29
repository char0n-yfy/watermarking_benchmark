from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.attacks import ATTACK_REGISTRY
from evaluator.watermarking import WATERMARK_REGISTRY

from app.services.attack_weights import enrich_attack_resource
from app.services.object_storage import ObjectStorageClient
from app.services.watermark_weights import enrich_watermark_resource


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

    direct_images = sorted(
        path
        for path in datasets_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
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
    "cew_c1": "CEW-C1 Basic Auto-Fix SR",
    "cew_c2": "CEW-C2 Color Retouch SR",
    "cew_c3": "CEW-C3 Detail Enhance SR",
    "cew_c4": "CEW-C4 Full Enhancement Chain",
    "cew_d1": "CEW-D1 Auto-Light",
    "cew_d2": "CEW-D2 Auto-WhiteBalance",
    "cew_d3": "CEW-D3 Adaptive AI Color",
    "cew_d4": "CEW-D4 Detail Low-Light Enhance",
    "cew_d5": "CEW-D5 AI-Denoise Clean",
    "cew_e1": "CEW-E1 Auto-Tone",
    "cew_e2": "CEW-E2 Warm-Vivid",
    "cew_e3": "CEW-E3 Film-Faded",
    "cew_e4": "CEW-E4 Local-Clarity HDR",
    "cew_s1": "CEW-S1 RealESRGAN",
    "cew_s2": "CEW-S2 SwinIR",
    "cew_s3": "CEW-S3 BSRGAN",
    "2x_regen": "2x Regeneration",
    "4x_regen": "4x Regeneration",
    "combined_physical": "Combined Physical",
    "gaussian_blur": "Gaussian Blur",
    "gaussian_noise": "Gaussian Noise",
    "image_to_vedio": "NFPA Image-to-Vedio",
    "jpeg": "JPEG Compression",
    "noise_to_image": "CtrlRegen Noise-to-Image",
    "print_camera": "Print-Camera",
    "regen_diffusion": "Diffusion Regeneration",
    "regen_vae": "VAE Regeneration",
    "resized_crop": "Resized Crop",
    "screen_shoot": "Screen-Shoot",
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
    "2x_regen": [0.0, 0.5, 1.0],
    "4x_regen": [0.0, 0.5, 1.0],
    "brightness": [0.25, 0.5, 0.75],
    "combined_physical": [0.0, 0.5, 1.0],
    "contrast": [0.25, 0.5, 0.75],
    "cew_e1": [0.25, 0.5, 0.75],
    "cew_e2": [0.25, 0.5, 0.75],
    "cew_e3": [0.25, 0.5, 0.75],
    "cew_e4": [0.25, 0.5, 0.75],
    "cew_s1": [2.0, 4.0],
    "cew_s2": [2.0, 4.0],
    "cew_s3": [2.0, 4.0],
    "erasing": [0.25, 0.5, 0.75],
    "gaussian_blur": [0.2, 0.4, 0.6],
    "gaussian_noise": [0.25, 0.5, 0.75],
    "image_to_vedio": [20.0, 40.0, 60.0],
    "jpeg": [0.25, 0.5, 0.75],
    "noise_to_image": [0.25, 0.5, 0.75, 1.0],
    "print_camera": [0.0, 0.5, 1.0],
    "regen_diffusion": [0.0, 0.5, 1.0],
    "resized_crop": [0.1, 0.3, 0.5],
    "rotation": [0.25, 0.5, 0.75],
    "screen_shoot": [0.0, 0.5, 1.0],
}

ATTACK_PARAM_BY_METHOD = {
    "2x_regen": "strength",
    "4x_regen": "strength",
    "cew_e1": "strength",
    "cew_e2": "strength",
    "cew_e3": "strength",
    "cew_e4": "strength",
    "cew_s1": "scale",
    "cew_s2": "scale",
    "cew_s3": "scale",
    "combined_physical": "strength",
    "image_to_vedio": "xy",
    "noise_to_image": "step",
    "print_camera": "strength",
    "regen_diffusion": "strength",
    "screen_shoot": "strength",
}

PHYSICAL_CHANNEL_METHODS = {"screen_shoot", "print_camera", "combined_physical"}
VIEWPOINT_RERENDERING_PREFIX = "3d_viewpoint_rerendering_phase"
VIEWPOINT_RERENDERING_STRENGTHS = [0.0, 0.5, 1.0]

LEGACY_ATTACK_ALIASES: dict[str, str] = {
    "atk-jpeg-smoke": "atk-jpeg",
    "atk-blur-smoke": "atk-gaussian-blur",
    "atk-jpeg-sweep": "atk-jpeg",
    "atk-blur-sweep": "atk-gaussian-blur",
    "atk-crop-sweep": "atk-resized-crop",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "resource"


def _is_viewpoint_rerendering_variant(method: str) -> bool:
    return method.startswith(VIEWPOINT_RERENDERING_PREFIX)


def _viewpoint_display_name(method: str) -> str | None:
    match = re.fullmatch(r"3d_viewpoint_rerendering_phase(\d+)_(point|ahead)", method)
    if match is None:
        return None
    phase_index, lookat_mode = match.groups()
    return f"3D Viewpoint Phase {phase_index} ({lookat_mode})"


def _display_name(method: str, overrides: dict[str, str]) -> str:
    if method in overrides:
        return overrides[method]
    viewpoint_name = _viewpoint_display_name(method)
    if viewpoint_name is not None:
        return viewpoint_name
    return method.replace("_", " ").replace("-", " ").title()


def _watermark_category(method: str) -> str:
    if method.startswith("traditional") or method in WATERMARK_CPU_METHODS:
        return "classical"
    if method in {"videoseal", "pixelseal", "chunkyseal"}:
        return "videoseal-family"
    return "neural"


ATTACK_CATEGORY_LABELS = {
    "3d_viewpoint_rerendering": "3D viewpoint re-rendering",
    "adversarial_attacks": "Adversarial attacks",
    "consumer_enhancement_workflow_attacks": "Consumer enhancement workflow attacks",
    "distortion_attacks": "Distortion attacks",
    "identity": "Identity",
    "physical_channel_attacks": "Physical channel attacks",
    "regeneration_attacks": "Regeneration attacks",
}


def _attack_category_from_class(cls: type[Any]) -> str:
    prefix = "evaluator.attacks."
    module = cls.__module__
    if module.startswith(prefix):
        folder = module[len(prefix) :].split(".", 1)[0]
        if folder and folder not in {"base", "registry"}:
            return folder
    return "uncategorized_attacks"


def _attack_category(method: str, cls: type[Any]) -> str:
    if method == "identity":
        return "identity"
    return _attack_category_from_class(cls)


def _attack_category_path(method: str, category: str) -> str:
    if method == "identity":
        return "evaluator/attacks/distortion_attacks"
    return f"evaluator/attacks/{category}"


def _attack_category_label(category: str) -> str:
    return ATTACK_CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def _has_explicit_init_param(cls: type[Any], parameter_name: str) -> bool:
    return parameter_name in inspect.signature(cls.__init__).parameters


def _attack_requires_gpu(method: str) -> bool:
    return (
        "regen" in method
        or method in {"noise_to_image", "image_to_vedio"}
        or _is_viewpoint_rerendering_variant(method)
        or method.endswith("_deep")
        or method.startswith("cew_d")
        or method.startswith("cew_s")
        or method.startswith("cew_c")
    )


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
    strength_param = ATTACK_PARAM_BY_METHOD.get(method)
    if _is_viewpoint_rerendering_variant(method):
        strength_param = "strength"
    if strength_param is None and _has_explicit_init_param(cls, "strength"):
        strength_param = "strength"
    strengths = (
        VIEWPOINT_RERENDERING_STRENGTHS
        if _is_viewpoint_rerendering_variant(method)
        else ATTACK_STRENGTH_SWEEPS.get(method, [0.5] if strength_param else [0.0])
    )
    category = _attack_category(method, cls)
    return {
        "id": f"atk-{_slug(method)}",
        "name": _display_name(method, ATTACK_DISPLAY_NAMES),
        "method": method,
        "description": cls.description,
        "category": category,
        "categoryLabel": _attack_category_label(category),
        "categoryPath": _attack_category_path(method, category),
        "strengths": strengths,
        "strengthParam": strength_param,
        "requiresGpu": _attack_requires_gpu(method),
        "recommended": method == "identity",
        "available": True,
        "params": {},
    }


def _build_attack_catalog() -> dict[str, dict[str, Any]]:
    return {
        f"atk-{_slug(method)}": _base_attack_preset(method, cls)
        for method, cls in sorted(ATTACK_REGISTRY.items())
    }


def list_watermark_resources(
    resources_root: Path | None = None,
    *,
    oss: ObjectStorageClient | None = None,
    probe_remote: bool = False,
) -> list[dict[str, Any]]:
    items = list(_build_watermark_catalog().values())
    if resources_root is None:
        return items
    return [
        enrich_watermark_resource(item, resources_root, oss=oss, probe_remote=probe_remote)
        for item in items
    ]


def list_attack_resources(
    resources_root: Path | None = None,
    *,
    oss: ObjectStorageClient | None = None,
    probe_remote: bool = False,
) -> list[dict[str, Any]]:
    items = list(_build_attack_catalog().values())
    if resources_root is None:
        return items
    return [
        enrich_attack_resource(item, resources_root, oss=oss, probe_remote=probe_remote)
        for item in items
    ]


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
    if attack_preset_id in LEGACY_ATTACK_ALIASES:
        return catalog[LEGACY_ATTACK_ALIASES[attack_preset_id]]
    if attack_preset_id in {item["method"] for item in catalog.values()}:
        for item in catalog.values():
            if item["method"] == attack_preset_id:
                return item
    raise KeyError(f"Unknown attack preset id: {attack_preset_id}")
