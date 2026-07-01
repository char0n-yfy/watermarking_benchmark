from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.object_storage import ObjectStorageClient
from app.services.watermark_weights import iter_weight_files


VIEWPOINT_RERENDERING_METHOD_PATTERN = re.compile(
    r"3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_(point|ahead)"
)

DIFFUSION_WEIGHT_MARKER = "diffusion/sd2-1-base"
CEW_WEIGHT_ROOT = "consumer_enhancement_workflow_attacks"
INSTALLED_PACKS_DIR = ".wmbench/installed-packs"


@dataclass(frozen=True)
class AttackWeightSpec:
    pack_id: str
    storage_dir: str
    markers: tuple[str, ...]


ATTACK_WEIGHT_PACKS: dict[str, AttackWeightSpec] = {
    "regen_vae": AttackWeightSpec("regen_vae", "regeneration_attacks", ("vae",)),
    "regen_diffusion": AttackWeightSpec("regen_diffusion", "regeneration_attacks", (DIFFUSION_WEIGHT_MARKER,)),
    "2x_regen": AttackWeightSpec("2x_regen", "regeneration_attacks", (DIFFUSION_WEIGHT_MARKER,)),
    "4x_regen": AttackWeightSpec("4x_regen", "regeneration_attacks", (DIFFUSION_WEIGHT_MARKER,)),
    "noise_to_image": AttackWeightSpec("noise_to_image", "regeneration_attacks", ("noise_to_image",)),
    "image_to_vedio": AttackWeightSpec("image_to_vedio", "regeneration_attacks", (DIFFUSION_WEIGHT_MARKER,)),
    "3d_viewpoint_rerendering_swipe": AttackWeightSpec(
        "3d_viewpoint_rerendering_swipe",
        "3d_viewpoint_rerendering",
        ("checkpoints",),
    ),
    "3d_viewpoint_rerendering_shake": AttackWeightSpec(
        "3d_viewpoint_rerendering_shake",
        "3d_viewpoint_rerendering",
        ("checkpoints",),
    ),
    "3d_viewpoint_rerendering_rotate": AttackWeightSpec(
        "3d_viewpoint_rerendering_rotate",
        "3d_viewpoint_rerendering",
        ("checkpoints",),
    ),
    "3d_viewpoint_rerendering_rotate_forward": AttackWeightSpec(
        "3d_viewpoint_rerendering_rotate_forward",
        "3d_viewpoint_rerendering",
        ("checkpoints",),
    ),
    "cew_d3": AttackWeightSpec(
        "cew_d3",
        CEW_WEIGHT_ROOT,
        ("deep_enhance/image_adaptive_3dlut_fivek",),
    ),
    "cew_d4": AttackWeightSpec(
        "cew_d4",
        CEW_WEIGHT_ROOT,
        ("deep_enhance/retinexformer_low_light",),
    ),
    "cew_d5": AttackWeightSpec(
        "cew_d5",
        CEW_WEIGHT_ROOT,
        ("deep_enhance/restormer_or_nafnet_denoise",),
    ),
    "cew_s1": AttackWeightSpec(
        "cew_s1",
        CEW_WEIGHT_ROOT,
        ("super_resolution/realesrgan_x2plus", "super_resolution/realesrgan_x4plus"),
    ),
    "cew_s2": AttackWeightSpec(
        "cew_s2",
        CEW_WEIGHT_ROOT,
        ("super_resolution/swinir_x2", "super_resolution/swinir_x4"),
    ),
    "cew_s3": AttackWeightSpec(
        "cew_s3",
        CEW_WEIGHT_ROOT,
        ("super_resolution/bsrgan_x2", "super_resolution/bsrgan_x4"),
    ),
    "cew_c1": AttackWeightSpec(
        "cew_c1",
        CEW_WEIGHT_ROOT,
        (
            "deep_enhance/restormer_or_nafnet_denoise",
            "super_resolution/realesrgan_x2plus",
            "super_resolution/realesrgan_x4plus",
        ),
    ),
    "cew_c2": AttackWeightSpec(
        "cew_c2",
        CEW_WEIGHT_ROOT,
        (
            "deep_enhance/image_adaptive_3dlut_fivek",
            "deep_enhance/restormer_or_nafnet_denoise",
            "super_resolution/swinir_x2",
        ),
    ),
    "cew_c3": AttackWeightSpec(
        "cew_c3",
        CEW_WEIGHT_ROOT,
        (
            "deep_enhance/retinexformer_low_light",
            "deep_enhance/restormer_or_nafnet_denoise",
            "super_resolution/realesrgan_x4plus",
        ),
    ),
    "cew_c4": AttackWeightSpec(
        "cew_c4",
        CEW_WEIGHT_ROOT,
        (
            "deep_enhance/image_adaptive_3dlut_fivek",
            "deep_enhance/retinexformer_low_light",
            "deep_enhance/restormer_or_nafnet_denoise",
            "super_resolution/bsrgan_x4",
        ),
    ),
}

def attack_weight_pack_id(method: str) -> str | None:
    if method in ATTACK_WEIGHT_PACKS:
        return method
    match = VIEWPOINT_RERENDERING_METHOD_PATTERN.fullmatch(method)
    if match is not None:
        return f"3d_viewpoint_rerendering_{match.group(1)}"
    return None


def attack_weight_spec(method: str) -> AttackWeightSpec | None:
    pack_id = attack_weight_pack_id(method)
    if pack_id is None:
        return None
    return ATTACK_WEIGHT_PACKS.get(pack_id)


def attack_weights_dir_name(method: str) -> str | None:
    spec = attack_weight_spec(method)
    return spec.storage_dir if spec is not None else None


def methods_for_attack_weight_pack(pack_id: str) -> list[str]:
    spec = ATTACK_WEIGHT_PACKS.get(pack_id)
    if spec is None:
        return [pack_id]

    shared = {pack_id}
    marker_set = set(spec.markers)
    for other_id, other_spec in ATTACK_WEIGHT_PACKS.items():
        if other_spec.storage_dir != spec.storage_dir:
            continue
        if marker_set & set(other_spec.markers):
            shared.add(other_id)
    return sorted(shared)


def attack_weights_need_download(method: str) -> bool:
    return attack_weight_spec(method) is not None


def attack_weights_install_dir(resources_root: Path, method: str) -> Path:
    spec = attack_weight_spec(method)
    if spec is None:
        raise KeyError(f"Attack method has no packaged weights: {method}")
    return resources_root / "weights" / "attacks" / spec.storage_dir


def attack_weight_storage_root(resources_root: Path, method: str) -> Path:
    return attack_weights_install_dir(resources_root, method)


def pack_install_marker_path(storage_root: Path, pack_id: str) -> Path:
    return storage_root / INSTALLED_PACKS_DIR / pack_id


def is_attack_pack_marked_installed(storage_root: Path, pack_id: str) -> bool:
    marker = pack_install_marker_path(storage_root, pack_id)
    return marker.is_file() and marker.stat().st_size >= 0


def mark_attack_pack_installed(storage_root: Path, pack_id: str) -> None:
    marker = pack_install_marker_path(storage_root, pack_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("installed\n", encoding="utf-8")


def unmark_attack_pack_installed(storage_root: Path, pack_id: str) -> None:
    marker = pack_install_marker_path(storage_root, pack_id)
    if marker.exists():
        marker.unlink()


def packs_referencing_weight_marker(storage_dir: str, rel_marker: str) -> list[str]:
    return sorted(
        pack_id
        for pack_id, spec in ATTACK_WEIGHT_PACKS.items()
        if spec.storage_dir == storage_dir and rel_marker in spec.markers
    )


def other_installed_packs_using_marker(
    resources_root: Path,
    storage_dir: str,
    rel_marker: str,
    *,
    excluding_pack_id: str,
) -> list[str]:
    storage_root = resources_root / "weights" / "attacks" / storage_dir
    installed: list[str] = []
    for pack_id in packs_referencing_weight_marker(storage_dir, rel_marker):
        if pack_id == excluding_pack_id:
            continue
        if is_attack_pack_marked_installed(storage_root, pack_id):
            installed.append(pack_id)
    return installed


def _marker_installed(storage_root: Path, rel_path: str) -> bool:
    target = storage_root / rel_path
    if target.is_file() and target.stat().st_size > 0:
        return True
    if not target.is_dir():
        return False
    if iter_weight_files(target):
        return True
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in target.rglob("*")
        if not path.name.startswith(".")
    )


def attack_weight_files_ready(storage_root: Path, spec: AttackWeightSpec) -> bool:
    return all(_marker_installed(storage_root, marker) for marker in spec.markers)


def attack_method_weights_installed(resources_root: Path, method: str) -> bool:
    spec = attack_weight_spec(method)
    if spec is None:
        return True
    storage_root = attack_weight_storage_root(resources_root, method)
    if not is_attack_pack_marked_installed(storage_root, spec.pack_id):
        return False
    return attack_weight_files_ready(storage_root, spec)


def reconcile_stale_attack_pack_marker(storage_root: Path, spec: AttackWeightSpec) -> bool:
    """Drop install markers left behind when weight files were never extracted."""
    if not is_attack_pack_marked_installed(storage_root, spec.pack_id):
        return False
    if attack_weight_files_ready(storage_root, spec):
        return False
    unmark_attack_pack_installed(storage_root, spec.pack_id)
    return True


def attack_method_can_install(resources_root: Path, method: str, *, remote_available: bool) -> bool:
    spec = attack_weight_spec(method)
    if spec is None:
        return True
    storage_root = attack_weight_storage_root(resources_root, method)
    if attack_method_weights_installed(resources_root, method):
        return False
    if remote_available:
        return True
    return attack_weight_files_ready(storage_root, spec)


def attack_weights_installed(install_dir: Path, *, method: str | None = None, resources_root: Path | None = None) -> bool:
    if method is not None and resources_root is not None:
        return attack_method_weights_installed(resources_root, method)
    if not install_dir.is_dir():
        return False
    if iter_weight_files(install_dir):
        return True
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in install_dir.rglob("*")
        if not path.name.startswith(".")
    )


def uninstall_attack_weight_pack(resources_root: Path, method: str) -> dict[str, Any]:
    spec = attack_weight_spec(method)
    if spec is None:
        raise ValueError(f"Missing attack weights mapping for: {method}")

    storage_root = attack_weight_storage_root(resources_root, method)
    if not is_attack_pack_marked_installed(storage_root, spec.pack_id):
        raise FileNotFoundError(f"Attack weights are not installed for method: {method}")

    unmark_attack_pack_installed(storage_root, spec.pack_id)
    removed_paths: list[str] = []
    for marker in spec.markers:
        if other_installed_packs_using_marker(
            resources_root,
            spec.storage_dir,
            marker,
            excluding_pack_id=spec.pack_id,
        ):
            continue
        target = storage_root / marker
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed_paths.append(str(target))

    return {
        "method": method,
        "weightsDir": spec.storage_dir,
        "weightsPackId": spec.pack_id,
        "installed": False,
        "removedPath": str(storage_root),
        "removedWeightPaths": removed_paths,
        "sharedMethods": methods_for_attack_weight_pack(spec.pack_id),
        "message": "卸载完成",
    }


def enrich_attack_resource(
    item: dict[str, Any],
    resources_root: Path,
    *,
    oss: ObjectStorageClient | None = None,
    probe_remote: bool = True,
) -> dict[str, Any]:
    method = str(item["method"])
    spec = attack_weight_spec(method)
    needs_weights = attack_weights_need_download(method)
    install_dir = attack_weights_install_dir(resources_root, method) if needs_weights else None
    if needs_weights and spec is not None and install_dir is not None:
        reconcile_stale_attack_pack_marker(install_dir, spec)
    installed = attack_method_weights_installed(resources_root, method) if needs_weights else True
    remote_available = False
    if needs_weights and oss and oss.enabled and probe_remote and spec is not None:
        remote_available = oss.exists(oss.attack_weights_key(spec.pack_id))
        if not remote_available:
            remote_available = oss.exists(oss.attack_weights_legacy_key(spec.storage_dir))
    can_install = (
        attack_method_can_install(resources_root, method, remote_available=remote_available)
        if needs_weights
        else False
    )
    download_ready = needs_weights and (installed or can_install)
    return {
        **item,
        "weightsDir": spec.storage_dir if spec is not None else None,
        "weightsPackId": spec.pack_id if spec is not None else None,
        "weightsPath": str(install_dir) if install_dir else None,
        "weightsInstalled": installed,
        "weightsDownloadReady": download_ready,
        "remoteWeightsAvailable": remote_available,
        "weightsPackRequired": needs_weights,
    }
