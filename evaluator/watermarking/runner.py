from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import methods  # noqa: F401 - import registers default watermark methods
from .base import WatermarkContext, WatermarkEmbedResult, WatermarkExtractResult
from .registry import build_watermark


INTERMEDIATE_ARTIFACT_DIR = "_intermediates"


@dataclass(frozen=True)
class WatermarkEmbedJob:
    run_id: str
    method_name: str
    params: dict[str, Any]
    input_dir: Path
    output_dir: Path
    message: str | None = "test_watermark_001"
    device: str = "cpu"
    seed: int | None = 42
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class WatermarkExtractJob:
    run_id: str
    method_name: str
    params: dict[str, Any]
    input_dir: Path
    output_dir: Path
    message: str | None = None
    device: str = "cpu"
    seed: int | None = 42
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


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


def run_watermark_embed_dir(job: WatermarkEmbedJob) -> list[WatermarkEmbedResult]:
    method = build_watermark(job.method_name, **job.params)
    image_paths = iter_image_paths(job.input_dir, job.image_exts)
    results: list[WatermarkEmbedResult] = []

    for index, input_path in enumerate(image_paths):
        relative = input_path.relative_to(job.input_dir)
        output_path = (job.output_dir / relative).with_suffix(method.output_ext)
        context = WatermarkContext(
            run_id=job.run_id,
            sample_id=str(relative.with_suffix("")),
            method_name=method.name,
            params=method.params,
            workspace_dir=job.output_dir,
            device=job.device,
            seed=None if job.seed is None else job.seed + index,
            message=job.message,
        )
        results.append(method.embed(input_path, output_path, context))

    method.write_embed_manifest(job.output_dir / "watermark_embed_manifest.json", results)
    return results


def run_watermark_extract_dir(job: WatermarkExtractJob) -> list[WatermarkExtractResult]:
    method = build_watermark(job.method_name, **job.params)
    image_paths = iter_image_paths(job.input_dir, job.image_exts)
    results: list[WatermarkExtractResult] = []

    for index, input_path in enumerate(image_paths):
        relative = input_path.relative_to(job.input_dir)
        context = WatermarkContext(
            run_id=job.run_id,
            sample_id=str(relative.with_suffix("")),
            method_name=method.name,
            params=method.params,
            workspace_dir=job.output_dir,
            device=job.device,
            seed=None if job.seed is None else job.seed + index,
            message=job.message,
        )
        results.append(method.extract(input_path, context))

    method.write_extract_manifest(job.output_dir / "watermark_extract_manifest.json", results)
    return results
