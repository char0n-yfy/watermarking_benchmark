from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.object_storage import ObjectStorageClient


WEIGHT_FILE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".json",
    ".onnx",
    ".pickle",
    ".pt",
    ".pth",
    ".pyt",
    ".safetensors",
    ".yaml",
    ".yml",
}

# method id -> resources/weights/watermarking/<dir>
WATERMARK_WEIGHT_DIRS: dict[str, str] = {
    "chunkyseal": "chunkyseal",
    "cin": "cin",
    "hidden": "hidden",
    "invismark": "invismark",
    "invisible-watermark-rivagan": "rivaGan",
    "maskwm-d32": "maskwm",
    "mbrs": "mbrs",
    "pimog": "pimog",
    "pixelseal": "pixelseal",
    "rawatermark": "rawatermark",
    "ssl-watermarking": "ssl_watermarking",
    "stegastamp": "stegastamp",
    "trustmark": "trustmark",
    "trustmark-c": "trustmark",
    "trustmark-q": "trustmark",
    "videoseal": "videoseal",
    "vine": "vine",
    "wam": "wam",
}


def weights_dir_name(method: str) -> str | None:
    return WATERMARK_WEIGHT_DIRS.get(method)


def weights_need_download(method: str) -> bool:
    return method in WATERMARK_WEIGHT_DIRS


def weights_install_dir(resources_root: Path, method: str) -> Path:
    directory = weights_dir_name(method)
    if directory is None:
        raise KeyError(f"Watermark method has no packaged weights: {method}")
    return resources_root / "weights" / "watermarking" / directory


def iter_weight_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_FILE_SUFFIXES
    )


def weights_installed(install_dir: Path) -> bool:
    if not install_dir.is_dir():
        return False
    if iter_weight_files(install_dir):
        return True
    # Allow directory-based checkpoints (e.g. diffusers layout under vine/).
    return any(path.is_file() and path.stat().st_size > 0 for path in install_dir.rglob("*") if not path.name.startswith("."))


def enrich_watermark_resource(
    item: dict[str, Any],
    resources_root: Path,
    *,
    oss: ObjectStorageClient | None = None,
    probe_remote: bool = True,
) -> dict[str, Any]:
    method = str(item["method"])
    directory = weights_dir_name(method)
    needs_weights = weights_need_download(method)
    install_dir = weights_install_dir(resources_root, method) if needs_weights else None
    installed = weights_installed(install_dir) if install_dir else True
    remote_available = False
    if needs_weights and oss and oss.enabled and probe_remote and directory:
        remote_available = oss.exists(oss.watermark_weights_key(directory))
    download_ready = needs_weights and (installed or remote_available or (oss.enabled if oss else False))
    return {
        **item,
        "weightsDir": directory,
        "weightsPath": str(install_dir) if install_dir else None,
        "weightsInstalled": installed,
        "weightsDownloadReady": download_ready,
        "remoteWeightsAvailable": remote_available,
        "weightsPackRequired": needs_weights,
    }


def resolve_watermark_method(algorithm_id: str, catalog: dict[str, dict[str, Any]]) -> str:
    if algorithm_id in catalog:
        return str(catalog[algorithm_id]["method"])
    for item in catalog.values():
        if item["method"] == algorithm_id:
            return str(item["method"])
    raise KeyError(f"Unknown watermark algorithm id: {algorithm_id}")
