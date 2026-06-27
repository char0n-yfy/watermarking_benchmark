from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.attacks import list_attacks
from evaluator.watermarking import list_watermarks


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


WATERMARK_CATALOG: dict[str, dict[str, Any]] = {
    "alg-traditional-lsb": {
        "id": "alg-traditional-lsb",
        "name": "Traditional LSB",
        "method": "traditional-lsb",
        "version": "local-smoke",
        "status": "enabled",
        "requiresGpu": False,
        "recommended": True,
        "params": {"payload_bits": 16},
    },
    "alg-traditional-dct": {
        "id": "alg-traditional-dct",
        "name": "Traditional DCT",
        "method": "traditional-dct",
        "version": "local-smoke",
        "status": "enabled",
        "requiresGpu": False,
        "recommended": False,
        "params": {"payload_bits": 16},
    },
    "alg-hidden": {
        "id": "alg-hidden",
        "name": "HiDDeN",
        "method": "hidden",
        "version": "packaged",
        "status": "enabled",
        "requiresGpu": True,
        "recommended": False,
        "params": {},
    },
    "alg-ssl-watermarking": {
        "id": "alg-ssl-watermarking",
        "name": "SSL Watermarking",
        "method": "ssl-watermarking",
        "version": "packaged",
        "status": "enabled",
        "requiresGpu": True,
        "recommended": False,
        "params": {},
    },
    "alg-stegastamp": {
        "id": "alg-stegastamp",
        "name": "StegaStamp",
        "method": "stegastamp",
        "version": "packaged",
        "status": "enabled",
        "requiresGpu": True,
        "recommended": False,
        "params": {},
    },
}


ATTACK_PRESET_CATALOG: dict[str, dict[str, Any]] = {
    "atk-identity": {
        "id": "atk-identity",
        "name": "Identity",
        "method": "identity",
        "strengths": [0.0],
        "recommended": True,
        "params": {},
    },
    "atk-jpeg-smoke": {
        "id": "atk-jpeg-smoke",
        "name": "JPEG smoke",
        "method": "jpeg",
        "strengths": [0.5],
        "recommended": True,
        "params": {},
    },
    "atk-blur-smoke": {
        "id": "atk-blur-smoke",
        "name": "Blur smoke",
        "method": "gaussian_blur",
        "strengths": [0.2],
        "recommended": False,
        "params": {},
    },
    "atk-jpeg-sweep": {
        "id": "atk-jpeg-sweep",
        "name": "JPEG sweep",
        "method": "jpeg",
        "strengths": [0.25, 0.5, 0.75],
        "recommended": False,
        "params": {},
    },
    "atk-blur-sweep": {
        "id": "atk-blur-sweep",
        "name": "Blur sweep",
        "method": "gaussian_blur",
        "strengths": [0.2, 0.4, 0.6],
        "recommended": False,
        "params": {},
    },
}


def list_watermark_resources() -> list[dict[str, Any]]:
    registered = {item["name"] for item in list_watermarks()}
    return [
        {**item, "available": item["method"] in registered}
        for item in WATERMARK_CATALOG.values()
    ]


def list_attack_resources() -> list[dict[str, Any]]:
    registered = {item["name"] for item in list_attacks()}
    return [
        {**item, "available": item["method"] in registered}
        for item in ATTACK_PRESET_CATALOG.values()
    ]


def get_watermark_catalog_item(algorithm_id: str) -> dict[str, Any]:
    if algorithm_id in WATERMARK_CATALOG:
        return WATERMARK_CATALOG[algorithm_id]
    if algorithm_id in {item["method"] for item in WATERMARK_CATALOG.values()}:
        for item in WATERMARK_CATALOG.values():
            if item["method"] == algorithm_id:
                return item
    raise KeyError(f"Unknown watermark algorithm id: {algorithm_id}")


def get_attack_catalog_item(attack_preset_id: str) -> dict[str, Any]:
    if attack_preset_id not in ATTACK_PRESET_CATALOG:
        raise KeyError(f"Unknown attack preset id: {attack_preset_id}")
    return ATTACK_PRESET_CATALOG[attack_preset_id]
