from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import (  # noqa: F401 - import registers default attacks
    consumer_enhancement_workflow_attacks,
    content_preserve_workflow_attacks,
    distortion_attacks,
    physical_channel_attacks,
    regeneration_attacks,
)
from .base import AttackContext, AttackResult
from .registry import build_attack


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


def iter_image_paths(input_dir: Path, image_exts: Iterable[str]) -> list[Path]:
    normalized_exts = {ext.lower() for ext in image_exts}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    )


def run_attack_dir(job: AttackJob) -> list[AttackResult]:
    attack = build_attack(job.attack_name, **job.params)
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
