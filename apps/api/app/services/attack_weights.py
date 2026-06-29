from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.object_storage import ObjectStorageClient
from app.services.watermark_weights import iter_weight_files


REGENERATION_ATTACK_METHODS = {
    "regen_vae",
    "regen_diffusion",
    "2x_regen",
    "4x_regen",
    "noise_to_image",
    "image_to_vedio",
    "3d_viewpoint_rerendering",
}

CEW_ATTACK_METHODS = {
    "cew_c1",
    "cew_c2",
    "cew_c3",
    "cew_c4",
    "cew_d3",
    "cew_d4",
    "cew_d5",
    "cew_s1",
    "cew_s2",
    "cew_s3",
}

# attack method -> resources/weights/attacks/<dir>
ATTACK_WEIGHT_DIRS: dict[str, str] = {
    **{method: "regeneration_attacks" for method in REGENERATION_ATTACK_METHODS},
    **{method: "consumer_enhancement_workflow_attacks" for method in CEW_ATTACK_METHODS},
}


def attack_weights_dir_name(method: str) -> str | None:
    return ATTACK_WEIGHT_DIRS.get(method)


def attack_weights_need_download(method: str) -> bool:
    return method in ATTACK_WEIGHT_DIRS


def attack_weights_install_dir(resources_root: Path, method: str) -> Path:
    directory = attack_weights_dir_name(method)
    if directory is None:
        raise KeyError(f"Attack method has no packaged weights: {method}")
    return resources_root / "weights" / "attacks" / directory


def attack_weights_installed(install_dir: Path) -> bool:
    if not install_dir.is_dir():
        return False
    if iter_weight_files(install_dir):
        return True
    return any(path.is_file() and path.stat().st_size > 0 for path in install_dir.rglob("*") if not path.name.startswith("."))


def enrich_attack_resource(
    item: dict[str, Any],
    resources_root: Path,
    *,
    oss: ObjectStorageClient | None = None,
    probe_remote: bool = True,
) -> dict[str, Any]:
    method = str(item["method"])
    directory = attack_weights_dir_name(method)
    needs_weights = attack_weights_need_download(method)
    install_dir = attack_weights_install_dir(resources_root, method) if needs_weights else None
    installed = attack_weights_installed(install_dir) if install_dir else True
    remote_available = False
    if needs_weights and oss and oss.enabled and probe_remote and directory:
        remote_available = oss.exists(oss.attack_weights_key(directory))
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
