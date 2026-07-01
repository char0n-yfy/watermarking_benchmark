from __future__ import annotations

import gc
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import (  # noqa: F401 - import registers default attacks
    consumer_enhancement_workflow_attacks,
    distortion_attacks,
    physical_channel_attacks,
    regeneration_attacks,
)
from .base import AttackContext, AttackResult, BaseAttack
from .registry import build_attack


INTERMEDIATE_ARTIFACT_DIR = "_intermediates"
_ATTACK_INSTANCE_CACHE: OrderedDict[str, BaseAttack] = OrderedDict()


def _cache_max_entries() -> int:
    raw = os.getenv("WM_BENCH_ATTACK_CACHE_MAX_ENTRIES", "8")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 8


def _release_attack_instance(attack: BaseAttack) -> None:
    del attack
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _prepend_path_once(directory: Path) -> None:
    directory_text = str(directory)
    if not directory_text or not directory.exists():
        return
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory_text not in entries:
        os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")


def _ensure_runtime_binaries_on_path() -> None:
    _prepend_path_once(Path(sys.executable).resolve().parent)
    try:
        import ninja

        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir:
            _prepend_path_once(Path(bin_dir))
    except Exception:
        pass


@dataclass(frozen=True)
class AttackJob:
    run_id: str
    attack_name: str
    params: dict[str, Any]
    input_dir: Path
    output_dir: Path
    device: str = "cpu"
    seed: int | None = 42
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _cache_key(name: str, params: dict[str, Any], device: str) -> str:
    payload = {
        "name": str(name).lower(),
        "params": params,
        "device": str(device),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def get_cached_attack(name: str, params: dict[str, Any], device: str = "cpu") -> BaseAttack:
    _ensure_runtime_binaries_on_path()
    key = _cache_key(name, params, device)
    max_entries = _cache_max_entries()
    if max_entries == 0:
        return build_attack(name, **params)

    attack = _ATTACK_INSTANCE_CACHE.get(key)
    if attack is not None:
        _ATTACK_INSTANCE_CACHE.move_to_end(key)
        return attack

    attack = build_attack(name, **params)
    _ATTACK_INSTANCE_CACHE[key] = attack
    while len(_ATTACK_INSTANCE_CACHE) > max_entries:
        _old_key, old_attack = _ATTACK_INSTANCE_CACHE.popitem(last=False)
        _release_attack_instance(old_attack)
    return attack


def clear_attack_cache() -> None:
    while _ATTACK_INSTANCE_CACHE:
        _old_key, old_attack = _ATTACK_INSTANCE_CACHE.popitem(last=False)
        _release_attack_instance(old_attack)


def iter_image_paths(input_dir: Path, image_exts: Iterable[str]) -> list[Path]:
    normalized_exts = {ext.lower() for ext in image_exts}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in normalized_exts
            and INTERMEDIATE_ARTIFACT_DIR not in path.relative_to(input_dir).parts
        )
    )


def run_attack_dir_with_attack(job: AttackJob, attack: BaseAttack) -> list[AttackResult]:
    image_paths = iter_image_paths(job.input_dir, job.image_exts)
    results: list[AttackResult] = []

    for index, input_path in enumerate(image_paths):
        relative = input_path.relative_to(job.input_dir)
        output_path = (job.output_dir / relative).with_suffix(attack.output_ext)
        context = AttackContext(
            run_id=job.run_id,
            sample_id=str(relative.with_suffix("")),
            attack_name=attack.name,
            params=attack.params,
            workspace_dir=job.output_dir,
            device=job.device,
            seed=None if job.seed is None else job.seed + index,
        )
        results.append(attack.attack(input_path, output_path, context))

    attack.write_manifest(job.output_dir / "attack_manifest.json", results)
    return results


def run_attack_dir(job: AttackJob) -> list[AttackResult]:
    attack = get_cached_attack(job.attack_name, job.params, job.device)
    return run_attack_dir_with_attack(job, attack)
