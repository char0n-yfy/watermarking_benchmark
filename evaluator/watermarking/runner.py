from __future__ import annotations

import gc
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import methods  # noqa: F401 - import registers default watermark methods
from .base import BaseWatermark, WatermarkContext, WatermarkEmbedResult, WatermarkExtractResult
from .registry import build_watermark


INTERMEDIATE_ARTIFACT_DIR = "_intermediates"
_WATERMARK_INSTANCE_CACHE: OrderedDict[str, BaseWatermark] = OrderedDict()


def _cache_max_entries() -> int:
    raw = os.getenv("WM_BENCH_WATERMARK_CACHE_MAX_ENTRIES", "1")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1


def _release_watermark_instance(method: BaseWatermark) -> None:
    try:
        method.release()
    except Exception:
        pass
    del method
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def _cache_key(name: str, params: dict[str, Any], device: str) -> str:
    payload = {
        "name": str(name).lower(),
        "params": params,
        "device": str(device),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def get_cached_watermark(name: str, params: dict[str, Any], device: str = "cpu") -> BaseWatermark:
    key = _cache_key(name, params, device)
    max_entries = _cache_max_entries()
    if max_entries == 0:
        return build_watermark(name, **params)

    method = _WATERMARK_INSTANCE_CACHE.get(key)
    if method is not None:
        _WATERMARK_INSTANCE_CACHE.move_to_end(key)
        return method

    method = build_watermark(name, **params)
    _WATERMARK_INSTANCE_CACHE[key] = method
    while len(_WATERMARK_INSTANCE_CACHE) > max_entries:
        _old_key, old_method = _WATERMARK_INSTANCE_CACHE.popitem(last=False)
        _release_watermark_instance(old_method)
    return method


def clear_watermark_cache() -> None:
    while _WATERMARK_INSTANCE_CACHE:
        _old_key, old_method = _WATERMARK_INSTANCE_CACHE.popitem(last=False)
        _release_watermark_instance(old_method)


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


def run_watermark_embed_dir_with_method(job: WatermarkEmbedJob, method: BaseWatermark) -> list[WatermarkEmbedResult]:
    image_paths = iter_image_paths(job.input_dir, job.image_exts)
    jobs: list[tuple[Path, Path, WatermarkContext]] = []

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
        jobs.append((input_path, output_path, context))

    results = method.embed_many(jobs)
    method.write_embed_manifest(job.output_dir / "watermark_embed_manifest.json", results)
    return results


def run_watermark_embed_dir(job: WatermarkEmbedJob) -> list[WatermarkEmbedResult]:
    method = get_cached_watermark(job.method_name, job.params, job.device)
    return run_watermark_embed_dir_with_method(job, method)


def run_watermark_extract_dir_with_method(job: WatermarkExtractJob, method: BaseWatermark) -> list[WatermarkExtractResult]:
    image_paths = iter_image_paths(job.input_dir, job.image_exts)
    jobs: list[tuple[Path, WatermarkContext]] = []

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
        jobs.append((input_path, context))

    results = method.extract_many(jobs)
    method.write_extract_manifest(job.output_dir / "watermark_extract_manifest.json", results)
    return results


def run_watermark_extract_dir(job: WatermarkExtractJob) -> list[WatermarkExtractResult]:
    method = get_cached_watermark(job.method_name, job.params, job.device)
    return run_watermark_extract_dir_with_method(job, method)
