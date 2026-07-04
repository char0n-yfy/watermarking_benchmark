from __future__ import annotations

"""Local experiment executor implementation used by the public local_runner facade."""

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

from PIL import Image

from evaluator.attacks.base import AttackResult
from evaluator.attacks.registry import ATTACK_REGISTRY
from evaluator.attacks.runner import AttackJob, get_cached_attack, run_attack_dir_with_attack
from evaluator.image_protocol import (
    CANONICAL_IMAGE_SIZE,
    CANONICAL_OUTPUT_POLICY,
    CANONICAL_PREPROCESS_POLICY,
    image_size,
    quality_alignment_metadata,
)
from evaluator.execution import ExecutionProfile, summarize_execution_profiles
from evaluator.watermarking.base import WatermarkEmbedResult
from evaluator.watermarking.runner import WatermarkEmbedJob, get_cached_watermark, run_watermark_embed_dir_with_method

from app.core.storage import safe_segment
from app.services.experiment_schema import (
    IMAGE_ATTACK_SCHEMA,
    IMAGE_DETECTION_SCHEMA,
    IMAGE_QUALITY_SCHEMA,
    IMAGE_WATERMARK_EMBED_SCHEMA,
    RUNTIME_PROFILE_SCHEMA,
)
from app.services.experiment_stages import (
    AttackStage,
    CANONICAL_MANIFEST_NAME,
    DatasetStage,
    DetectionStage,
    ExtractStage,
    QualityStage,
    WatermarkStage,
    normalize_attack_params_for_runtime,
)
from app.services.resources import (
    get_dataset_by_id,
    get_watermark_catalog_item,
)
from app.services.local_plan import (
    _cell_key,
    build_run_plan_payload,
    estimate_selection,
    normalize_selection,
    pending_variant_groups,
)
from app.services.local_artifacts import (
    RunStateWriter,
    append_jsonl as _append_jsonl,
    artifact_paths as _artifact_paths,
    compact_result_units as _compact_result_units,
    compact_result_units_file as _compact_result_units_file,
    latest_result_unit_map as _latest_result_unit_map,
    progress as _progress,
    read_json_object as _read_json_object,
    read_jsonl as _read_jsonl,
    utc_timestamp as _utc_timestamp,
    write_json as _write_json,
    write_run_status as _write_run_status,
)
from app.services.runtime_resource_manager import RuntimeResourceManager, clear_transient_experiment_caches
from app.services.scoring import compute_image_quality_pairs_with_profile


JsonDict = dict[str, Any]
RunStateCallback = Callable[[JsonDict], None]
StopIntentCallback = Callable[[], Union[str, bool, None]]


@dataclass(frozen=True)
class LocalRunRequest:
    run_id: str
    selection: JsonDict
    resources_root: Path
    runs_root: Path
    device: str = "cpu"
    message: str = "1010101010101010"
    resume: bool = True


@dataclass
class MaterializedCellState:
    variant: JsonDict
    dataset_id: str
    algorithm_id: str
    algorithm: JsonDict
    algorithm_params: JsonDict
    seed: int
    canonical_input_dir: Path
    copied_samples: list[Path]
    watermarked_dir: Path
    embed_quality_records: list[JsonDict]
    embed_elapsed_ms: float
    embed_error: str | None
    cell_key: str
    attack_id: str
    attack: JsonDict
    attack_params: JsonDict
    strength: float
    cell_root: Path
    variant_key: str
    variant_root: Path
    attacked_dir: Path
    extracted_dir: Path
    negative_attack_key: str
    negative_attacked_dir: Path
    negative_extracted_dir: Path
    detection_manifest_path: Path
    cell_started: float
    detection_records: list[JsonDict] = field(default_factory=list)
    status: str = "succeeded"
    error: str | None = None
    positive_attack_results: list[Any] = field(default_factory=list)
    negative_attack_results: list[Any] = field(default_factory=list)
    positive_extract_results: list[Any] = field(default_factory=list)
    negative_extract_results: list[Any] = field(default_factory=list)


def _operation_error(results: list[Any]) -> str | None:
    errors = [
        str(getattr(result, "error"))
        for result in results
        if not getattr(result, "ok", False) and getattr(result, "error", None)
    ]
    return "; ".join(errors) or None


def _stage_status_and_error(
    results: list[Any],
    *,
    fallback_error: str | None = None,
    expected_count: int = 0,
) -> tuple[str, str | None]:
    error = _operation_error(results)
    if error is None and fallback_error and expected_count > 0 and not results:
        error = fallback_error
    return ("failed" if error else "succeeded", error)


def _mark_state_operation_result(state: MaterializedCellState, results: list[Any]) -> str | None:
    error = _operation_error(results)
    if error:
        state.status = "failed"
        state.error = error if state.error is None else f"{state.error}; {error}"
    return error


def _scene_cache_hit_count(results: list[Any]) -> int:
    total = 0
    for result in results:
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("scene_cache_hit") is True:
            total += 1
    return total


def _quality_pair_cache_hit_count(runtime_profiles: list[JsonDict]) -> int:
    total = 0
    for profile in runtime_profiles:
        metadata = profile.get("metadata") if isinstance(profile, dict) else None
        execution = metadata.get("execution") if isinstance(metadata, dict) else None
        if not isinstance(execution, dict):
            continue
        if isinstance(execution.get("cacheHits"), int):
            total += int(execution["cacheHits"])
            continue
        details = execution.get("details")
        if isinstance(details, dict) and isinstance(details.get("cacheHits"), int):
            total += int(details["cacheHits"])
    return total


@dataclass(frozen=True)
class MaterializedDatasetContext:
    dataset_id: str
    path_planner: "MaterializedPathPlanner"
    canonical_input_dir: Path
    copied_samples: list[Path]


def _run_dataset_root(run_root: Path, dataset_id: str) -> Path:
    return run_root / safe_segment(dataset_id)


def _run_canonical_root(run_root: Path, dataset_id: str) -> Path:
    return _run_dataset_root(run_root, dataset_id) / "canonical"


def _run_negative_control_root(run_root: Path, dataset_id: str, attack_id: str, variant_key: str) -> Path:
    return (
        _run_dataset_root(run_root, dataset_id)
        / "_negative_controls"
        / safe_segment(attack_id)
        / safe_segment(variant_key or "default")
    )


def _run_watermark_root(run_root: Path, dataset_id: str, algorithm_id: str, seed: int) -> Path:
    return (
        _run_dataset_root(run_root, dataset_id)
        / safe_segment(algorithm_id)
        / f"seed_{safe_segment(str(seed))}"
    )


def _run_variant_root(
    run_root: Path,
    *,
    dataset_id: str,
    algorithm_id: str,
    seed: int,
    attack_id: str,
    variant_key: str,
) -> Path:
    return (
        _run_watermark_root(run_root, dataset_id, algorithm_id, seed)
        / safe_segment(attack_id)
        / safe_segment(variant_key or "default")
    )


@dataclass(frozen=True)
class DeferredQualityResult:
    quality_records: list[JsonDict]
    runtime_profiles: list[JsonDict]
    error: str | None = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
INTERMEDIATE_ARTIFACT_DIR = "_intermediates"
STOP_INTENT_CANCEL = "cancel"
STOP_INTENT_PAUSE = "pause"
MATERIALIZED_CACHE_SCHEMA_VERSION = 5
_WEIGHT_FINGERPRINT_CACHE: dict[tuple[str, str, str], JsonDict | None] = {}


def _quality_pair_cache_enabled() -> bool:
    return os.getenv("WM_BENCH_QUALITY_PAIR_CACHE", "1") != "0"


def _quality_pair_cache_max_entries() -> int:
    raw = os.getenv("WM_BENCH_QUALITY_PAIR_CACHE_MAX_ENTRIES", "200000")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 200000
    return max(0, value)


def _file_fingerprint(path: Path) -> JsonDict:
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path.absolute())
    try:
        stat = path.stat()
    except OSError:
        return {"path": resolved, "exists": False}
    return {
        "path": resolved,
        "exists": True,
        "mtimeNs": int(stat.st_mtime_ns),
        "sizeBytes": int(stat.st_size),
    }


class QualityPairCache:
    def __init__(self) -> None:
        self.enabled = _quality_pair_cache_enabled()
        self.max_entries = _quality_pair_cache_max_entries()
        self._lock = threading.Lock()
        self._metrics: OrderedDict[str, JsonDict] = OrderedDict()

    def get(self, key: str) -> JsonDict | None:
        if not self.enabled or self.max_entries <= 0:
            return None
        with self._lock:
            cached = self._metrics.get(key)
            if cached is None:
                return None
            self._metrics.move_to_end(key)
            return dict(cached)

    def set(self, key: str, metrics: JsonDict) -> None:
        if not self.enabled or self.max_entries <= 0:
            return
        with self._lock:
            self._metrics[key] = dict(metrics)
            self._metrics.move_to_end(key)
            while len(self._metrics) > self.max_entries:
                self._metrics.popitem(last=False)

    def stats(self) -> JsonDict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "maxEntries": self.max_entries,
                "entryCount": len(self._metrics),
            }


def _stop_status_from_callback(callback: StopIntentCallback | None) -> str | None:
    if callback is None:
        return None
    value = callback()
    if value is True:
        return "cancelled"
    if not isinstance(value, str):
        return None
    normalized = value.lower()
    if normalized in {STOP_INTENT_PAUSE, "paused"}:
        return "paused"
    if normalized in {STOP_INTENT_CANCEL, "cancelled"}:
        return "cancelled"
    return None


def _image_sample_id(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).with_suffix("").as_posix()
    except ValueError:
        return path.with_suffix("").name


def _image_size(path: Path) -> tuple[int | None, int | None]:
    size = image_size(path)
    if size is None:
        return None, None
    return int(size[0]), int(size[1])


def _is_intermediate_artifact(path: Path) -> bool:
    return INTERMEDIATE_ARTIFACT_DIR in path.parts


def _total_megapixels(paths: list[Path]) -> float:
    total = 0.0
    for path in paths:
        width, height = _image_size(path)
        if width and height:
            total += (width * height) / 1_000_000.0
    return total


def _process_peak_memory_mb() -> float | None:
    try:
        import resource
    except Exception:
        return None
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak_rss <= 0:
        return None
    if sys.platform == "darwin":
        return peak_rss / (1024.0 * 1024.0)
    return peak_rss / 1024.0


def _reset_gpu_peak(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _gpu_peak_memory_mb(device: str) -> float | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return None


def _record_runtime_profile(
    paths: dict[str, Path],
    *,
    run_id: str,
    cell_key: str,
    stage: str,
    method: str,
    device: str,
    elapsed_ms: float,
    image_paths: list[Path],
    status: str,
    error: str | None = None,
    metadata: JsonDict | None = None,
) -> None:
    total_mp = _total_megapixels(image_paths)
    gpu_peak = _gpu_peak_memory_mb(device)
    if gpu_peak is not None:
        peak_memory_mb = gpu_peak
        peak_memory_source = "cuda_max_memory_allocated"
    else:
        peak_memory_mb = _process_peak_memory_mb()
        peak_memory_source = "process_peak_rss" if peak_memory_mb is not None else None
    _append_jsonl(
        paths["runtimeProfile"],
        RUNTIME_PROFILE_SCHEMA.apply(
            {
                "runId": run_id,
                "cellKey": cell_key,
                "stage": stage,
                "method": method,
                "device": device,
                "status": status,
                "imageCount": len(image_paths),
                "totalMegapixels": total_mp,
                "elapsedMs": elapsed_ms,
                "peakMemoryMB": peak_memory_mb,
                "peakMemorySource": peak_memory_source,
                "error": error,
                "metadata": metadata or {},
                "timestamp": _utc_timestamp(),
            }
        ),
    )


def _pair_images(reference_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    references = {
        path.relative_to(reference_dir).with_suffix("").as_posix(): path
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not _is_intermediate_artifact(path)
    }
    pairs: list[tuple[Path, Path]] = []
    for target in sorted(target_dir.rglob("*")):
        if (
            not target.is_file()
            or target.suffix.lower() not in IMAGE_EXTS
            or _is_intermediate_artifact(target)
        ):
            continue
        key = target.relative_to(target_dir).with_suffix("").as_posix()
        reference = references.get(key)
        if reference is not None:
            pairs.append((reference, target))
    return pairs


def _quality_pair_cache_key(reference_path: Path, target_path: Path) -> str:
    alignment = quality_alignment_metadata(reference_path, target_path)
    return json.dumps(
        {
            "schema": "quality-pair-v1",
            "reference": _file_fingerprint(reference_path),
            "target": _file_fingerprint(target_path),
            "alignmentPolicy": alignment.get("alignmentPolicy"),
            "referenceSize": alignment.get("referenceSize"),
            "targetSize": alignment.get("targetSize"),
            "alignedSize": alignment.get("alignedSize"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _compute_quality_metrics_with_cache(
    pairs: list[tuple[Path, Path]],
    *,
    quality_pair_cache: QualityPairCache | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    if not pairs or quality_pair_cache is None or not quality_pair_cache.enabled:
        return compute_image_quality_pairs_with_profile(pairs)

    metrics_by_pair: list[JsonDict | None] = [None] * len(pairs)
    misses: list[tuple[Path, Path]] = []
    miss_indexes: list[int] = []
    miss_keys: list[str] = []
    cache_hits = 0
    for index, (reference_path, target_path) in enumerate(pairs):
        key = _quality_pair_cache_key(reference_path, target_path)
        cached = quality_pair_cache.get(key)
        if cached is None:
            misses.append((reference_path, target_path))
            miss_indexes.append(index)
            miss_keys.append(key)
            continue
        cache_hits += 1
        metrics_by_pair[index] = cached

    if misses:
        miss_metrics, miss_profile = compute_image_quality_pairs_with_profile(misses)
        for index, key, metrics in zip(miss_indexes, miss_keys, miss_metrics):
            copied = dict(metrics)
            metrics_by_pair[index] = copied
            quality_pair_cache.set(key, copied)
    else:
        miss_profile = None

    if any(metrics is None for metrics in metrics_by_pair):
        raise RuntimeError("quality pair cache produced incomplete metrics")

    if misses and cache_hits:
        execution_profile = {
            "stage": "quality",
            "method": "image_quality",
            "mode": "hybrid_pair_cache",
            "jobCount": len(pairs),
            "cacheHits": cache_hits,
            "cacheMisses": len(misses),
            "cache": quality_pair_cache.stats(),
            "missExecution": miss_profile,
        }
    elif misses:
        execution_profile = {
            **dict(miss_profile or {}),
            "cacheHits": 0,
            "cacheMisses": len(misses),
            "cache": quality_pair_cache.stats(),
        }
    else:
        execution_profile = ExecutionProfile(
            stage="quality",
            method="image_quality",
            mode="pair_cache",
            job_count=len(pairs),
            details={
                "cacheHits": cache_hits,
                "cacheMisses": 0,
                "cache": quality_pair_cache.stats(),
            },
        ).to_json()

    return [dict(metrics or {}) for metrics in metrics_by_pair], execution_profile


def _list_image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not _is_intermediate_artifact(path)
    ]


def _stable_json_digest(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _dataset_fingerprint(dataset_path: Path, sample_paths: list[Path] | None = None) -> str:
    entries: list[JsonDict] = []
    selected_paths = sample_paths if sample_paths is not None else _list_image_files(dataset_path)
    for path in selected_paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            relative = path.relative_to(dataset_path).as_posix()
        except ValueError:
            relative = path.name
        entries.append(
            {
                "relative": relative,
                "size": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
            }
        )
    return _stable_json_digest(
        {
            "cacheSchemaVersion": MATERIALIZED_CACHE_SCHEMA_VERSION,
            "canonicalPreprocessPolicy": CANONICAL_PREPROCESS_POLICY,
            "canonicalOutputPolicy": CANONICAL_OUTPUT_POLICY,
            "canonicalImageSize": list(CANONICAL_IMAGE_SIZE),
            "samplePool": "selected_shard_samples" if sample_paths is not None else "all_sorted_images",
            "entries": entries,
        },
        length=16,
    )


def _sample_relatives_for_dataset(selection: JsonDict, dataset_id: str) -> list[str] | None:
    overrides = selection.get("_sampleRelativesByDataset")
    if not isinstance(overrides, dict):
        return None
    values = overrides.get(dataset_id)
    if not isinstance(values, list):
        return None
    return [str(value) for value in values if str(value)]


def _selected_dataset_image_files(dataset_path: Path, selection: JsonDict, dataset_id: str) -> list[Path]:
    relatives = _sample_relatives_for_dataset(selection, dataset_id)
    if relatives is None:
        return _list_image_files(dataset_path)[: int(selection["maxSamples"])]
    selected: list[Path] = []
    for relative in relatives:
        raw_path = Path(relative)
        path = raw_path if raw_path.is_absolute() else dataset_path / raw_path
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            selected.append(path)
    return selected[: int(selection["maxSamples"])]


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _resource_weight_fingerprint(resources_root: Path, resource: JsonDict) -> JsonDict | None:
    cache_key = (
        str(resources_root.resolve()),
        str(resource.get("id") or ""),
        str(resource.get("method") or ""),
    )
    if cache_key in _WEIGHT_FINGERPRINT_CACHE:
        return _WEIGHT_FINGERPRINT_CACHE[cache_key]
    candidates: list[Path] = []
    for key in (
        "weightsPath",
        "weightPath",
        "weightsDir",
        "weightDir",
        "modelPath",
        "modelDir",
        "checkpointPath",
    ):
        raw = resource.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (resources_root / path).resolve()
        candidates.append(path)
    candidates.extend(
        (resources_root / "weights" / safe_segment(str(segment))).resolve()
        for segment in (resource.get("id"), resource.get("method"))
        if isinstance(segment, str) and segment.strip()
    )

    entries: list[JsonDict] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        paths = [candidate] if candidate.is_file() else [path for path in candidate.rglob("*") if path.is_file()]
        file_count = 0
        total_size = 0
        latest_mtime_ns = 0
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            total_size += stat.st_size
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
        entries.append(
            {
                "path": str(candidate),
                "fileCount": file_count,
                "totalSize": total_size,
                "latestMtimeNs": latest_mtime_ns,
            }
        )
    if not entries:
        _WEIGHT_FINGERPRINT_CACHE[cache_key] = None
        return None
    fingerprint = {"digest": _stable_json_digest(entries, length=16), "entries": entries}
    _WEIGHT_FINGERPRINT_CACHE[cache_key] = fingerprint
    return fingerprint


def _runtime_cache_fingerprint(
    *,
    resources_root: Path,
    resource: JsonDict,
    method: str,
    device: str,
) -> JsonDict:
    return {
        "deviceClass": "cuda" if str(device).startswith("cuda") else "cpu",
        "device": str(device),
        "method": str(method),
        "python": sys.version.split()[0],
        "torch": _module_version("torch"),
        "diffusers": _module_version("diffusers"),
        "implementationVersion": MATERIALIZED_CACHE_SCHEMA_VERSION,
        "weights": _resource_weight_fingerprint(resources_root, resource),
    }


def _configure_3d_scene_cache_runtime_min(max_samples: int) -> JsonDict:
    runtime_min = max(0, int(max_samples) * 2)
    try:
        import importlib

        module = importlib.import_module("evaluator.attacks.3d_viewpoint_rerendering.attacks")
        configure = getattr(module, "set_sharp_scene_cache_runtime_min_entries", None)
        stats = getattr(module, "sharp_scene_cache_stats", None)
        if callable(configure):
            configure(runtime_min)
        if callable(stats):
            return dict(stats())
    except Exception as exc:
        return {"runtimeMinEntries": runtime_min, "error": f"{type(exc).__name__}: {exc}"}
    return {"runtimeMinEntries": runtime_min}


def _sample_count_dir_name(max_samples: int, digest: str) -> str:
    return safe_segment(f"samples_{max_samples}_{digest}")


def _parse_sample_count_dir_name(path: Path, digest: str) -> int:
    prefix = "samples_"
    suffix = f"_{digest}"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return -1
    raw = name[len(prefix) : -len(suffix)]
    try:
        return int(raw)
    except ValueError:
        return -1


def _parse_any_sample_count_dir_name(path: Path) -> int:
    parts = path.name.split("_", 2)
    if len(parts) != 3 or parts[0] != "samples":
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def _digest_from_sample_count_dir(path: Path) -> str:
    parts = path.name.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[0] == "samples" else ""


def _materialized_root(run_root: Path) -> Path:
    configured = os.getenv("WM_BENCH_CACHE_ROOT")
    if configured:
        root = Path(configured).expanduser()
    else:
        root = run_root / "materialized"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _async_quality_workers() -> int:
    raw = os.getenv("WM_BENCH_ASYNC_QUALITY_WORKERS", "1")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _runtime_profile_elapsed_ms(path: Path) -> float:
    total = 0.0
    for record in _read_jsonl(path):
        value = _float_or_none(record.get("elapsedMs"))
        if value is not None:
            total += value
    return total


def _summary_elapsed_ms(
    *,
    previous_summary: JsonDict,
    invocation_elapsed_ms: float,
    runtime_profile_path: Path,
    pending_cell_count: int,
    resume: bool,
) -> float:
    runtime_elapsed_ms = _runtime_profile_elapsed_ms(runtime_profile_path)
    previous_elapsed_ms = _float_or_none(previous_summary.get("elapsedMs")) if resume else None
    if previous_elapsed_ms is not None:
        previous_status = str(previous_summary.get("status") or "")
        if pending_cell_count > 0 or previous_status not in {"succeeded", "partially_failed"}:
            return max(
                previous_elapsed_ms + max(0.0, invocation_elapsed_ms),
                runtime_elapsed_ms,
            )
        return max(previous_elapsed_ms, runtime_elapsed_ms)
    return max(max(0.0, invocation_elapsed_ms), runtime_elapsed_ms)


def _sample_counts_by_dataset(
    sample_manifest_path: Path,
    expected_samples_by_dataset: dict[str, int],
) -> dict[str, int]:
    sample_ids_by_dataset: dict[str, set[str]] = {}
    for record in _read_jsonl(sample_manifest_path):
        dataset_id = record.get("datasetId")
        sample_id = record.get("sampleId")
        if dataset_id is None or sample_id is None:
            continue
        sample_ids_by_dataset.setdefault(str(dataset_id), set()).add(str(sample_id))
    counts = {dataset_id: len(sample_ids) for dataset_id, sample_ids in sample_ids_by_dataset.items()}
    for dataset_id, expected_count in expected_samples_by_dataset.items():
        counts.setdefault(str(dataset_id), max(0, int(expected_count)))
    return counts


def _result_unit_sample_count(unit: JsonDict, sample_counts_by_dataset: dict[str, int]) -> int:
    value = unit.get("sampleCount")
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return count
    dataset_id = unit.get("datasetId")
    if dataset_id is None:
        return 0
    return max(0, int(sample_counts_by_dataset.get(str(dataset_id), 0)))


def _negative_attack_group_key(unit: JsonDict) -> str:
    return _stable_json_digest(
        {
            "datasetId": unit.get("datasetId"),
            "attackPresetId": unit.get("attackPresetId"),
            "attackMethod": unit.get("attackMethod"),
            "attackStrength": unit.get("attackStrength"),
            "attackParams": unit.get("attackParams") or {},
            "variantKey": unit.get("variantKey") or "",
        },
        length=24,
    )


def _normalize_completed_phase_state(
    state_writer: RunStateWriter,
    *,
    selection: JsonDict,
    expected_samples_by_dataset: dict[str, int],
    result_units: list[JsonDict],
    expected_cells: int,
    failed_units: int,
    skipped_units: int,
) -> None:
    sample_counts_by_dataset = _sample_counts_by_dataset(
        state_writer.paths["sampleManifest"],
        expected_samples_by_dataset,
    )
    selected_datasets = [str(dataset_id) for dataset_id in selection.get("datasetIds") or []]
    selected_algorithms = [str(algorithm_id) for algorithm_id in selection.get("algorithmIds") or []]
    selected_seeds = list(selection.get("seeds") or [])
    dataset_count = len(selected_datasets)
    algorithm_count = len(selected_algorithms)
    seed_count = len(selected_seeds)
    canonical_total = sum(max(0, int(sample_counts_by_dataset.get(dataset_id, 0))) for dataset_id in selected_datasets)
    watermark_groups = dataset_count * algorithm_count * seed_count
    watermark_total = canonical_total * algorithm_count * seed_count
    unit_sample_counts = [_result_unit_sample_count(unit, sample_counts_by_dataset) for unit in result_units]
    positive_attack_total = sum(unit_sample_counts)
    negative_attack_counts: dict[str, int] = {}
    for unit, sample_count in zip(result_units, unit_sample_counts):
        key = _negative_attack_group_key(unit)
        negative_attack_counts[key] = max(negative_attack_counts.get(key, 0), sample_count)
    negative_attack_total = sum(negative_attack_counts.values())
    attack_total = positive_attack_total + negative_attack_total
    per_unit_pair_total = positive_attack_total * 2
    result_unit_count = len(result_units)
    completed_status = "succeeded"

    state_writer.phase_finish(
        "canonical",
        status=completed_status,
        current=canonical_total,
        total=canonical_total,
        current_item={"datasetCount": dataset_count, "sampleCount": canonical_total},
        counters={"datasetsDone": dataset_count, "imagesDone": canonical_total},
        replace_counters=True,
    )
    state_writer.phase_finish(
        "watermark_embed",
        status=completed_status,
        current=watermark_total,
        total=watermark_total,
        current_item={
            "datasetCount": dataset_count,
            "algorithmCount": algorithm_count,
            "seedCount": seed_count,
        },
        counters={
            "imagesDone": watermark_total,
            "groupsDone": watermark_groups,
            "resultUnitsDone": result_unit_count,
            "phaseCellsDone": result_unit_count,
            "phaseCellsTotal": expected_cells,
        },
        replace_counters=True,
    )
    state_writer.phase_finish(
        "attack",
        status=completed_status,
        current=attack_total,
        total=attack_total,
        current_item={
            "resultUnitCount": result_unit_count,
            "negativeControlGroupCount": len(negative_attack_counts),
        },
        counters={
            "imagesDone": attack_total,
            "positiveImagesDone": positive_attack_total,
            "negativeImagesDone": negative_attack_total,
            "negativeControlGroupsDone": len(negative_attack_counts),
            "resultUnitsDone": result_unit_count,
            "phaseCellsDone": result_unit_count,
            "phaseCellsTotal": expected_cells,
        },
        replace_counters=True,
    )
    state_writer.phase_finish(
        "watermark_extract",
        status=completed_status,
        current=per_unit_pair_total,
        total=per_unit_pair_total,
        current_item={"resultUnitCount": result_unit_count},
        counters={
            "imagesDone": per_unit_pair_total,
            "positiveImagesDone": positive_attack_total,
            "negativeImagesDone": positive_attack_total,
            "resultUnitsDone": result_unit_count,
            "phaseCellsDone": result_unit_count,
            "phaseCellsTotal": expected_cells,
        },
        replace_counters=True,
    )
    state_writer.phase_finish(
        "quality",
        status=completed_status,
        current=per_unit_pair_total,
        total=per_unit_pair_total,
        current_item={"resultUnitCount": result_unit_count},
        counters={
            "pairsDone": per_unit_pair_total,
            "failedUnits": failed_units,
            "resultUnitsDone": result_unit_count,
            "phaseCellsDone": result_unit_count,
            "phaseCellsTotal": expected_cells,
        },
        replace_counters=True,
    )
    state_writer.phase_finish(
        "summary",
        status=completed_status,
        current=result_unit_count,
        total=expected_cells,
        current_item={"resultUnitCount": result_unit_count},
        counters={
            "resultUnitsDone": result_unit_count,
            "failedUnits": failed_units,
            "skippedUnits": skipped_units,
            "phaseCellsDone": result_unit_count,
            "phaseCellsTotal": expected_cells,
        },
        artifact_refs={"summary": str(state_writer.paths["runSummary"])},
        replace_counters=True,
    )


def _attack_model_cache_key(attack_method: str, attack_params: JsonDict, device: str) -> str:
    method_key = str(attack_method).lower()
    attack_cls = ATTACK_REGISTRY.get(method_key)
    model_params = dict(attack_params)
    if attack_cls is not None:
        model_params = attack_cls.model_cache_params(attack_params)
    return json.dumps(
        {"method": method_key, "params": model_params, "device": str(device)},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class MaterializedPathPlanner:
    root: Path
    dataset_id: str
    dataset_digest: str
    max_samples: int

    @property
    def dataset_segment(self) -> str:
        return safe_segment(self.dataset_id)

    def canonical_parent_dir(self) -> Path:
        return self.root / "canonical" / self.dataset_segment

    def canonical_dir(self) -> Path:
        return self.canonical_parent_dir() / _sample_count_dir_name(self.max_samples, self.dataset_digest)

    def watermarked_dir(
        self,
        *,
        algorithm_id: str,
        algorithm_method: str,
        algorithm_params: JsonDict,
        runtime_fingerprint: JsonDict,
        seed: int,
        message: str,
    ) -> tuple[Path, str]:
        digest = _stable_json_digest(
            {
                "cacheSchemaVersion": MATERIALIZED_CACHE_SCHEMA_VERSION,
                "datasetDigest": self.dataset_digest,
                "algorithmId": algorithm_id,
                "algorithmMethod": algorithm_method,
                "algorithmParams": algorithm_params,
                "runtimeFingerprint": runtime_fingerprint,
                "seed": seed,
                "message": message,
            },
            length=16,
        )
        return (
            self.root
            / "watermarked"
            / self.dataset_segment
            / safe_segment(algorithm_id)
            / safe_segment(str(seed))
            / _sample_count_dir_name(self.max_samples, digest),
            digest,
        )

    def positive_attacked_dir(
        self,
        *,
        algorithm_id: str,
        seed: int,
        cell_key: str,
        watermarked_digest: str,
        attack_id: str,
        attack_method: str,
        attack_params: JsonDict,
        runtime_fingerprint: JsonDict,
        strength: float,
    ) -> Path:
        attack_digest = _stable_json_digest(
            {
                "cacheSchemaVersion": MATERIALIZED_CACHE_SCHEMA_VERSION,
                "datasetDigest": self.dataset_digest,
                "watermarkedDigest": watermarked_digest,
                "attackId": attack_id,
                "attackMethod": attack_method,
                "attackParams": attack_params,
                "runtimeFingerprint": runtime_fingerprint,
                "strength": strength,
            },
            length=16,
        )
        return (
            self.root
            / "positive_attacked"
            / self.dataset_segment
            / safe_segment(algorithm_id)
            / safe_segment(str(seed))
            / safe_segment(cell_key)
            / _sample_count_dir_name(self.max_samples, attack_digest)
        )

    def negative_attack_key(
        self,
        *,
        attack_id: str,
        attack_method: str,
        attack_params: JsonDict,
        runtime_fingerprint: JsonDict,
        strength: float,
        variant_key: str,
    ) -> str:
        return _stable_json_digest(
            {
                "cacheSchemaVersion": MATERIALIZED_CACHE_SCHEMA_VERSION,
                "datasetDigest": self.dataset_digest,
                "attackId": attack_id,
                "attackMethod": attack_method,
                "attackParams": attack_params,
                "runtimeFingerprint": runtime_fingerprint,
                "strength": strength,
                "variantKey": variant_key,
                "negativeSeed": 0,
            },
            length=16,
        )

    def negative_attacked_dir(self, negative_attack_key: str) -> Path:
        return (
            self.root
            / "negative_attacked"
            / self.dataset_segment
            / _sample_count_dir_name(self.max_samples, negative_attack_key)
        )


def _record_quality_pairs(
    paths: dict[str, Path],
    *,
    run_id: str,
    cell_key: str,
    scope: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str | None,
    attack_method: str | None,
    attack_strength: float | None,
    seed: int,
    reference_dir: Path,
    target_dir: Path,
    device: str = "cpu",
    quality_pair_cache: QualityPairCache | None = None,
) -> list[JsonDict]:
    pairs = _pair_images(reference_dir, target_dir)
    started = time.perf_counter()
    target_paths = [target_path for _reference_path, target_path in pairs]
    try:
        metrics_by_pair, execution_profile = _compute_quality_metrics_with_cache(
            pairs,
            quality_pair_cache=quality_pair_cache,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        _record_runtime_profile(
            paths,
            run_id=run_id,
            cell_key=cell_key,
            stage="quality",
            method="image_quality",
            device=device,
            elapsed_ms=elapsed_ms,
            image_paths=target_paths,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            metadata={"scope": scope},
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    _record_runtime_profile(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        stage="quality",
        method="image_quality",
        device=device,
        elapsed_ms=elapsed_ms,
        image_paths=target_paths,
        status="succeeded",
        metadata={"scope": scope, "execution": execution_profile},
    )
    records = [
        _quality_record(
            run_id=run_id,
            cell_key=cell_key,
            scope=scope,
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=attack_id,
            attack_method=attack_method,
            attack_strength=attack_strength,
            seed=seed,
            sample_id=_image_sample_id(reference_path, reference_dir),
            reference_path=reference_path,
            target_path=target_path,
            metrics=metrics,
        )
        for (reference_path, target_path), metrics in zip(pairs, metrics_by_pair)
    ]
    _append_quality_records(paths, records)
    return records


def _quality_record(
    *,
    run_id: str,
    cell_key: str,
    scope: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str | None,
    attack_method: str | None,
    attack_strength: float | None,
    seed: int,
    sample_id: str,
    reference_path: Path,
    target_path: Path,
    metrics: JsonDict,
) -> JsonDict:
    return IMAGE_QUALITY_SCHEMA.apply(
        {
            "runId": run_id,
            "cellKey": cell_key,
            "scope": scope,
            "datasetId": dataset_id,
            "algorithmId": algorithm_id,
            "attackPresetId": attack_id,
            "attackMethod": attack_method,
            "attackStrength": attack_strength,
            "seed": seed,
            "sampleId": sample_id,
            **quality_alignment_metadata(reference_path, target_path),
            "metrics": dict(metrics),
            "timestamp": _utc_timestamp(),
        }
    )


def _append_quality_records(paths: dict[str, Path], records: list[JsonDict]) -> None:
    for record in records:
        _append_jsonl(paths["imageQuality"], record)


def _retarget_quality_records(
    source_records: list[JsonDict],
    *,
    run_id: str,
    cell_key: str,
    scope: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str | None,
    attack_method: str | None,
    attack_strength: float | None,
    seed: int,
    source_scope: str,
    reuse_policy: str,
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for source in source_records:
        record = dict(source)
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            record["metrics"] = dict(metrics)
        record.update(
            {
                "runId": run_id,
                "cellKey": cell_key,
                "scope": scope,
                "datasetId": dataset_id,
                "algorithmId": algorithm_id,
                "attackPresetId": attack_id,
                "attackMethod": attack_method,
                "attackStrength": attack_strength,
                "seed": seed,
                "qualityComputation": "reused",
                "sourceScope": source_scope,
                "reusePolicy": reuse_policy,
                "timestamp": _utc_timestamp(),
            }
        )
        records.append(record)
    return records


def _identity_quality_metrics() -> JsonDict:
    return {
        "psnr": 60.0,
        "ssim": 1.0,
        "msSsim": 1.0,
        "nmi": 1.0,
        "lpips": 0.0,
        "dists": 0.0,
    }


def _bit_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return "".join(str(int(bit)) for bit in value)
        except (TypeError, ValueError):
            return None
    return None


def _detection_record(
    *,
    run_id: str,
    cell_key: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str,
    attack_method: str,
    attack_strength: float,
    seed: int,
    label: int,
    input_root: Path,
    result: Any,
) -> JsonDict:
    metadata = dict(getattr(result, "metadata", {}) or {})
    decoded_bits_metadata = metadata.pop("decoded_bits", None)
    decoded_bits = _bit_string(getattr(result, "bits", None)) or _bit_string(decoded_bits_metadata)
    expected_bits = _bit_string(metadata.pop("expected_bits", None))
    expected_message = metadata.pop("expected_message", None)
    metadata.pop("payload_bits", None)
    for derived_key in ("bit_accuracy", "bit_error_rate", "match", "matched"):
        metadata.pop(derived_key, None)

    input_path = Path(getattr(result, "input_path", ""))
    return IMAGE_DETECTION_SCHEMA.apply(
        {
            "runId": run_id,
            "cellKey": cell_key,
            "datasetId": dataset_id,
            "algorithmId": algorithm_id,
            "attackPresetId": attack_id,
            "attackMethod": attack_method,
            "attackStrength": attack_strength,
            "seed": seed,
            "label": label,
            "sampleId": _image_sample_id(input_path, input_root),
            "status": "succeeded" if getattr(result, "ok", False) else "failed",
            "decodedMessage": getattr(result, "message", None),
            "expectedMessage": expected_message,
            "decodedBits": decoded_bits,
            "expectedBits": expected_bits,
            "elapsedMs": getattr(result, "elapsed_ms", None),
            "error": getattr(result, "error", None),
            "metadata": metadata,
            "timestamp": _utc_timestamp(),
        }
    )


def _record_watermark_embed_results(
    paths: dict[str, Path],
    *,
    run_id: str,
    cell_key: str,
    dataset_id: str,
    algorithm_id: str,
    watermark_method: str,
    seed: int,
    input_root: Path,
    results: list[Any],
) -> None:
    for result in results:
        metadata = dict(getattr(result, "metadata", {}) or {})
        input_path = Path(getattr(result, "input_path", ""))
        output_path = Path(getattr(result, "output_path", ""))
        _append_jsonl(
            paths["imageWatermarkEmbed"],
            IMAGE_WATERMARK_EMBED_SCHEMA.apply(
                {
                    "runId": run_id,
                    "cellKey": cell_key,
                    "stage": "watermark_embed",
                    "datasetId": dataset_id,
                    "algorithmId": algorithm_id,
                    "watermarkMethod": watermark_method,
                    "seed": seed,
                    "sampleId": _image_sample_id(input_path, input_root),
                    "status": "succeeded" if getattr(result, "ok", False) else "failed",
                    "inputPath": str(input_path),
                    "outputPath": str(output_path),
                    "inputSize": metadata.get("inputSize"),
                    "internalSize": metadata.get("internalSize"),
                    "preCanonicalOutputSize": metadata.get("preCanonicalOutputSize"),
                    "outputSize": metadata.get("outputSize"),
                    "canonicalSize": metadata.get("canonicalSize"),
                    "outputSizePolicy": metadata.get("outputSizePolicy"),
                    "canonicalizedOutput": metadata.get("canonicalizedOutput"),
                    "elapsedMs": getattr(result, "elapsed_ms", None),
                    "error": getattr(result, "error", None),
                    "metadata": metadata,
                    "timestamp": _utc_timestamp(),
                }
            ),
        )


def _record_attack_results(
    paths: dict[str, Path],
    *,
    run_id: str,
    cell_key: str,
    stage: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str,
    attack_method: str,
    attack_strength: float,
    attack_params: JsonDict,
    seed: int,
    label: int,
    input_root: Path,
    results: list[Any],
    cache_hit: bool = False,
) -> None:
    for result in results:
        metadata = dict(getattr(result, "metadata", {}) or {})
        input_path = Path(getattr(result, "input_path", ""))
        output_path = Path(getattr(result, "output_path", ""))
        _append_jsonl(
            paths["imageAttack"],
            IMAGE_ATTACK_SCHEMA.apply(
                {
                    "runId": run_id,
                    "cellKey": cell_key,
                    "stage": stage,
                    "datasetId": dataset_id,
                    "algorithmId": algorithm_id,
                    "attackPresetId": attack_id,
                    "attackMethod": attack_method,
                    "attackStrength": attack_strength,
                    "attackParams": attack_params,
                    "seed": seed,
                    "label": label,
                    "sampleId": _image_sample_id(input_path, input_root),
                    "status": "succeeded" if getattr(result, "ok", False) else "failed",
                    "inputPath": str(input_path),
                    "outputPath": str(output_path),
                    "inputSize": metadata.get("inputSize"),
                    "preProtocolOutputSize": metadata.get("preProtocolOutputSize"),
                    "outputSize": metadata.get("outputSize"),
                    "protocolResizedOutput": metadata.get("protocolResizedOutput"),
                    "sizePreserving": metadata.get("sizePreserving"),
                    "sizeChangeSemantic": metadata.get("sizeChangeSemantic"),
                    "sizePolicy": metadata.get("sizePolicy"),
                    "cacheHit": cache_hit,
                    "elapsedMs": getattr(result, "elapsed_ms", None),
                    "error": getattr(result, "error", None),
                    "metadata": metadata,
                    "timestamp": _utc_timestamp(),
                }
            ),
        )


def _read_json_array_file(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _clean_output_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _write_json_array_file(path: Path, records: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _reflink_file(source: Path, target: Path) -> bool:
    if sys.platform != "linux":
        return False
    try:
        import fcntl

        ficlone = 0x40049409
        with source.open("rb") as src, target.open("wb") as dst:
            fcntl.ioctl(dst.fileno(), ficlone, src.fileno())
        shutil.copystat(source, target, follow_symlinks=True)
        return True
    except OSError:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(source)
    except OSError:
        if not _reflink_file(source, target):
            shutil.copy2(source, target)
    if source.suffix.lower() in IMAGE_EXTS:
        size = image_size(source)
        if size is not None:
            try:
                from evaluator.image_protocol import invalidate_image_metadata, register_image_size

                invalidate_image_metadata(target)
                register_image_size(target, size)
            except Exception:
                pass


def _compatible_sample_dirs(parent_dir: Path, digest: str, target_dir: Path) -> list[Path]:
    if not parent_dir.exists():
        return []
    dirs = [
        path
        for path in parent_dir.iterdir()
        if path.is_dir()
        and path != target_dir
        and _parse_any_sample_count_dir_name(path) > 0
    ]
    return sorted(
        dirs,
        key=lambda path: (
            _parse_sample_count_dir_name(path, digest) > 0,
            _parse_any_sample_count_dir_name(path),
        ),
        reverse=True,
    )


def _sample_relative_token(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("samples_") and _parse_any_sample_count_dir_name(Path(part)) > 0:
            return "/".join(parts[index + 1 :])
    return "/".join(parts[-2:]) if len(parts) >= 2 else path.name


def _record_input_matches(record: JsonDict, input_key: str | None, expected_path: Path) -> bool:
    if input_key is None:
        return True
    raw = record.get(input_key)
    if not raw:
        return False
    if "sourceSize" in record or "sourceMtimeNs" in record:
        try:
            expected_stat = expected_path.stat()
        except OSError:
            return False
        if record.get("sourceSize") != expected_stat.st_size:
            return False
        if record.get("sourceMtimeNs") != expected_stat.st_mtime_ns:
            return False
    source_path = Path(str(raw))
    if source_path == expected_path:
        return True
    return _sample_relative_token(source_path) == _sample_relative_token(expected_path)


def _normalized_json_value(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _record_params_match(record_params: Any, expected_params: Any) -> bool:
    return _normalized_json_value(record_params or {}) == _normalized_json_value(expected_params or {})


def _record_params_include_expected(record_params: Any, expected_params: Any) -> bool:
    if not isinstance(expected_params, dict) or not expected_params:
        return True
    if not isinstance(record_params, dict):
        return False
    for key, expected_value in expected_params.items():
        if key not in record_params:
            return False
        if _normalized_json_value(record_params.get(key)) != _normalized_json_value(expected_value):
            return False
    return True


def _watermark_embed_record_matches(
    record: JsonDict,
    *,
    method_name: str,
    params: JsonDict,
    message: str,
) -> bool:
    record_message = record.get("message")
    return (
        str(record.get("method_name") or "") == method_name
        and (record_message in {None, ""} or str(record_message) == message)
        and _record_params_include_expected(record.get("params"), params)
    )


def _attack_record_matches(record: JsonDict, *, attack_name: str, params: JsonDict) -> bool:
    return str(record.get("attack_name") or "") == attack_name and _record_params_match(record.get("params"), params)


def _valid_prefix_records(
    manifest_path: Path,
    *,
    expected_count: int,
    output_key: str,
    output_root: Path,
    record_matches: Callable[[JsonDict], bool] | None = None,
) -> list[JsonDict]:
    prefix: list[JsonDict] = []
    records = _read_json_array_file(manifest_path)
    for record in records[:expected_count]:
        if record_matches is not None and not record_matches(record):
            break
        if output_key not in record:
            break
        if "ok" in record and not bool(record.get("ok")):
            break
        output_path = Path(str(record[output_key]))
        if not output_path.is_file() or not _is_relative_to(output_path, output_root):
            break
        prefix.append(dict(record))
    return prefix


def _hydrate_prefix_records_from_compatible_dirs(
    *,
    parent_dir: Path,
    target_dir: Path,
    digest: str,
    manifest_name: str,
    expected_count: int,
    expected_input_paths: list[Path],
    output_key: str,
    input_key: str | None,
    record_matches: Callable[[JsonDict], bool] | None = None,
) -> list[JsonDict]:
    target_manifest = target_dir / manifest_name
    target_dir.mkdir(parents=True, exist_ok=True)
    current = _valid_prefix_records(
        target_manifest,
        expected_count=expected_count,
        output_key=output_key,
        output_root=target_dir,
        record_matches=record_matches,
    )
    if len(current) >= expected_count:
        return current[:expected_count]

    for source_dir in _compatible_sample_dirs(parent_dir, digest, target_dir):
        source_records = _valid_prefix_records(
            source_dir / manifest_name,
            expected_count=expected_count,
            output_key=output_key,
            output_root=source_dir,
            record_matches=record_matches,
        )
        matched_records: list[JsonDict] = []
        for index, record in enumerate(source_records):
            if index >= len(expected_input_paths):
                break
            if not _record_input_matches(record, input_key, expected_input_paths[index]):
                break
            matched_records.append(record)
        source_records = matched_records
        if len(source_records) <= len(current):
            continue
        for index in range(len(current), min(expected_count, len(source_records))):
            source = dict(source_records[index])
            source_output = Path(str(source[output_key]))
            try:
                relative = source_output.relative_to(source_dir)
            except ValueError:
                break
            target_output = target_dir / relative
            _link_or_copy_file(source_output, target_output)
            source[output_key] = str(target_output)
            if input_key is not None and index < len(expected_input_paths):
                source[input_key] = str(expected_input_paths[index])
            if "outputRelative" in source:
                source["outputRelative"] = relative.as_posix()
            current.append(source)
        if len(current) >= expected_count:
            break

    if current:
        _write_json_array_file(target_manifest, current)
    return current


def _prepare_subset_input_dir(input_root: Path, input_paths: list[Path], workspace_dir: Path) -> Path:
    _clean_output_dir(workspace_dir)
    for input_path in input_paths:
        relative = input_path.relative_to(input_root)
        _link_or_copy_file(input_path, workspace_dir / relative)
    return workspace_dir


def _retarget_embed_results(
    results: list[WatermarkEmbedResult],
    *,
    subset_input_dir: Path,
    original_input_root: Path,
) -> list[WatermarkEmbedResult]:
    retargeted: list[WatermarkEmbedResult] = []
    for result in results:
        try:
            relative = result.input_path.relative_to(subset_input_dir)
            input_path = original_input_root / relative
        except ValueError:
            input_path = result.input_path
        retargeted.append(
            WatermarkEmbedResult(
                input_path=input_path,
                output_path=result.output_path,
                method_name=result.method_name,
                message=result.message,
                params=dict(result.params),
                elapsed_ms=result.elapsed_ms,
                ok=result.ok,
                error=result.error,
                metadata=dict(result.metadata),
            )
        )
    return retargeted


def _retarget_attack_results(
    results: list[AttackResult],
    *,
    subset_input_dir: Path,
    original_input_root: Path,
) -> list[AttackResult]:
    retargeted: list[AttackResult] = []
    for result in results:
        try:
            relative = result.input_path.relative_to(subset_input_dir)
            input_path = original_input_root / relative
        except ValueError:
            input_path = result.input_path
        retargeted.append(
            AttackResult(
                input_path=input_path,
                output_path=result.output_path,
                attack_name=result.attack_name,
                params=dict(result.params),
                elapsed_ms=result.elapsed_ms,
                ok=result.ok,
                error=result.error,
                metadata=dict(result.metadata),
            )
        )
    return retargeted


def _write_embed_manifest(path: Path, results: list[WatermarkEmbedResult]) -> None:
    _write_json_array_file(path, [result.to_json() for result in results])


def _write_attack_manifest(path: Path, results: list[AttackResult]) -> None:
    _write_json_array_file(path, [result.to_json() for result in results])


def _watermark_embed_results_from_records(records: list[JsonDict]) -> list[WatermarkEmbedResult] | None:
    results: list[WatermarkEmbedResult] = []
    try:
        for record in records:
            results.append(
                WatermarkEmbedResult(
                    input_path=Path(str(record["input_path"])),
                    output_path=Path(str(record["output_path"])),
                    method_name=str(record["method_name"]),
                    message=record.get("message"),
                    params=dict(record.get("params") or {}),
                    elapsed_ms=float(record.get("elapsed_ms") or 0.0),
                    ok=bool(record.get("ok")),
                    error=record.get("error"),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
    except (KeyError, TypeError, ValueError):
        return None
    return results


def _watermark_embed_prefix_from_manifest(
    path: Path,
    *,
    expected_count: int,
    record_matches: Callable[[JsonDict], bool] | None = None,
) -> list[WatermarkEmbedResult]:
    records = _valid_prefix_records(
        path,
        expected_count=expected_count,
        output_key="output_path",
        output_root=path.parent,
        record_matches=record_matches,
    )
    results = _watermark_embed_results_from_records(records)
    return results or []


def _watermark_embed_results_from_manifest(
    path: Path,
    *,
    expected_count: int,
    record_matches: Callable[[JsonDict], bool] | None = None,
) -> list[WatermarkEmbedResult] | None:
    results = _watermark_embed_prefix_from_manifest(
        path,
        expected_count=expected_count,
        record_matches=record_matches,
    )
    if len(results) != expected_count:
        return None
    return results


def _attack_results_from_records(records: list[JsonDict]) -> list[AttackResult] | None:
    results: list[AttackResult] = []
    try:
        for record in records:
            results.append(
                AttackResult(
                    input_path=Path(str(record["input_path"])),
                    output_path=Path(str(record["output_path"])),
                    attack_name=str(record["attack_name"]),
                    params=dict(record.get("params") or {}),
                    elapsed_ms=float(record.get("elapsed_ms") or 0.0),
                    ok=bool(record.get("ok")),
                    error=record.get("error"),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
    except (KeyError, TypeError, ValueError):
        return None
    return results


def _attack_prefix_from_manifest(
    path: Path,
    *,
    expected_count: int,
    record_matches: Callable[[JsonDict], bool] | None = None,
) -> list[AttackResult]:
    records = _valid_prefix_records(
        path,
        expected_count=expected_count,
        output_key="output_path",
        output_root=path.parent,
        record_matches=record_matches,
    )
    results = _attack_results_from_records(records)
    return results or []


def _attack_results_from_manifest(
    path: Path,
    *,
    expected_count: int,
    record_matches: Callable[[JsonDict], bool] | None = None,
) -> list[AttackResult] | None:
    results = _attack_prefix_from_manifest(
        path,
        expected_count=expected_count,
        record_matches=record_matches,
    )
    if len(results) != expected_count:
        return None
    return results


def _record_reused_watermark_embed(
    *,
    paths: dict[str, Path],
    run_id: str,
    cell_key: str,
    dataset_id: str,
    algorithm_id: str,
    algorithm: JsonDict,
    seed: int,
    input_dir: Path,
    output_dir: Path,
    copied_samples: list[Path],
    results: list[WatermarkEmbedResult],
    device: str,
    quality_pair_cache: QualityPairCache | None = None,
) -> list[JsonDict]:
    _record_watermark_embed_results(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        watermark_method=algorithm["method"],
        seed=seed,
        input_root=input_dir,
        results=results,
    )
    _record_runtime_profile(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        stage="watermark_embed",
        method=algorithm["method"],
        device=device,
        elapsed_ms=0.0,
        image_paths=copied_samples,
        status="reused",
        metadata={"cacheHit": True, "materializedDir": str(output_dir), "execution": summarize_execution_profiles(results)},
    )
    return _record_quality_pairs(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        scope="original_vs_watermarked",
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        attack_id=None,
        attack_method=None,
        attack_strength=None,
        seed=seed,
        reference_dir=input_dir,
        target_dir=output_dir,
        device=device,
        quality_pair_cache=quality_pair_cache,
    )


def _record_reused_attack(
    *,
    paths: dict[str, Path],
    run_id: str,
    cell_key: str,
    runtime_stage: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str,
    attack: JsonDict,
    attack_params: JsonDict,
    strength: float,
    seed: int,
    label: int,
    input_dir: Path,
    output_dir: Path,
    results: list[AttackResult],
    device: str,
    cache_key: str,
    image_paths: list[Path] | None = None,
) -> None:
    _record_attack_results(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        stage=runtime_stage,
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        attack_id=attack_id,
        attack_method=attack["method"],
        attack_strength=strength,
        attack_params=attack_params,
        seed=seed,
        label=label,
        input_root=input_dir,
        results=results,
        cache_hit=True,
    )
    _record_runtime_profile(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        stage=runtime_stage,
        method=attack["method"],
        device=device,
        elapsed_ms=0.0,
        image_paths=image_paths if image_paths is not None else _list_image_files(input_dir),
        status="reused",
        metadata={
            "attackParams": attack_params,
            "cacheKey": cache_key,
            "cacheHit": True,
            "materializedDir": str(output_dir),
            "execution": summarize_execution_profiles(results),
        },
    )


def _quality_runtime_profile(
    *,
    cell_key: str,
    elapsed_ms: float,
    image_paths: list[Path],
    status: str,
    device: str,
    error: str | None = None,
    metadata: JsonDict | None = None,
) -> JsonDict:
    return {
        "cell_key": cell_key,
        "stage": "quality",
        "method": "image_quality",
        "device": device,
        "elapsed_ms": elapsed_ms,
        "image_paths": image_paths,
        "status": status,
        "error": error,
        "metadata": metadata or {},
    }


def _compute_quality_pairs_deferred(
    *,
    cell_key: str,
    scope: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str | None,
    attack_method: str | None,
    attack_strength: float | None,
    seed: int,
    reference_dir: Path,
    target_dir: Path,
    device: str,
    quality_pair_cache: QualityPairCache | None = None,
) -> DeferredQualityResult:
    pairs = _pair_images(reference_dir, target_dir)
    target_paths = [target_path for _reference_path, target_path in pairs]
    started = time.perf_counter()
    try:
        metrics_by_pair, execution_profile = _compute_quality_metrics_with_cache(
            pairs,
            quality_pair_cache=quality_pair_cache,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return DeferredQualityResult(
            quality_records=[],
            runtime_profiles=[
                _quality_runtime_profile(
                    cell_key=cell_key,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    image_paths=target_paths,
                    status="failed",
                    device=device,
                    error=error,
                    metadata={"scope": scope},
                )
            ],
            error=error,
        )

    records = [
        _quality_record(
            run_id="",
            cell_key=cell_key,
            scope=scope,
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=attack_id,
            attack_method=attack_method,
            attack_strength=attack_strength,
            seed=seed,
            sample_id=_image_sample_id(reference_path, reference_dir),
            reference_path=reference_path,
            target_path=target_path,
            metrics=metrics,
        )
        for (reference_path, target_path), metrics in zip(pairs, metrics_by_pair)
    ]
    return DeferredQualityResult(
        quality_records=records,
        runtime_profiles=[
            _quality_runtime_profile(
                cell_key=cell_key,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                image_paths=target_paths,
                status="succeeded",
                device=device,
                metadata={"scope": scope, "execution": execution_profile},
            )
        ],
    )


def _compute_reused_quality_deferred(
    *,
    cell_key: str,
    scope: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str | None,
    attack_method: str | None,
    attack_strength: float | None,
    seed: int,
    source_records: list[JsonDict],
    source_scope: str,
    target_dir: Path,
    device: str,
    reuse_policy: str,
) -> DeferredQualityResult:
    started = time.perf_counter()
    execution_profile = ExecutionProfile(
        stage="quality",
        method="image_quality",
        mode="reused",
        job_count=len(source_records),
        device=device,
        details={"sourceScope": source_scope, "reusePolicy": reuse_policy},
    ).to_json()
    records = _retarget_quality_records(
        source_records,
        run_id="",
        cell_key=cell_key,
        scope=scope,
        dataset_id=dataset_id,
        algorithm_id=algorithm_id,
        attack_id=attack_id,
        attack_method=attack_method,
        attack_strength=attack_strength,
        seed=seed,
        source_scope=source_scope,
        reuse_policy=reuse_policy,
    )
    return DeferredQualityResult(
        quality_records=records,
        runtime_profiles=[
            _quality_runtime_profile(
                cell_key=cell_key,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                image_paths=_list_image_files(target_dir),
                status="reused",
                device=device,
                metadata={
                    "scope": scope,
                    "sourceScope": source_scope,
                    "reusePolicy": reuse_policy,
                    "execution": execution_profile,
                },
            )
        ],
    )


def _compute_identity_quality_deferred(
    *,
    cell_key: str,
    scope: str,
    dataset_id: str,
    algorithm_id: str,
    attack_id: str | None,
    attack_method: str | None,
    attack_strength: float | None,
    seed: int,
    reference_dir: Path,
    target_dir: Path,
    device: str,
) -> DeferredQualityResult:
    started = time.perf_counter()
    metrics = _identity_quality_metrics()
    pairs = _pair_images(reference_dir, target_dir)
    execution_profile = ExecutionProfile(
        stage="quality",
        method="image_quality",
        mode="reused",
        job_count=len(pairs),
        device=device,
        details={"sourceScope": "identity_noop", "reusePolicy": "identity_noop_perfect"},
    ).to_json()
    records = [
        {
            **_quality_record(
                run_id="",
                cell_key=cell_key,
                scope=scope,
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                attack_id=attack_id,
                attack_method=attack_method,
                attack_strength=attack_strength,
                seed=seed,
                sample_id=_image_sample_id(reference_path, reference_dir),
                reference_path=reference_path,
                target_path=target_path,
                metrics=metrics,
            ),
            "qualityComputation": "reused",
            "sourceScope": "identity_noop",
            "reusePolicy": "identity_noop_perfect",
        }
        for reference_path, target_path in pairs
    ]
    return DeferredQualityResult(
        quality_records=records,
        runtime_profiles=[
            _quality_runtime_profile(
                cell_key=cell_key,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                image_paths=[target_path for _reference_path, target_path in pairs],
                status="reused",
                device=device,
                metadata={
                    "scope": scope,
                    "sourceScope": "identity_noop",
                    "reusePolicy": "identity_noop_perfect",
                    "execution": execution_profile,
                },
            )
        ],
    )


def _compute_attack_quality_deferred(
    state: MaterializedCellState,
    *,
    device: str,
    quality_pair_cache: QualityPairCache | None = None,
) -> DeferredQualityResult:
    results: list[DeferredQualityResult]
    if str(state.attack["method"]).lower() == "identity":
        results = [
            _compute_reused_quality_deferred(
                cell_key=state.cell_key,
                scope="original_vs_attacked_watermarked",
                dataset_id=state.dataset_id,
                algorithm_id=state.algorithm_id,
                attack_id=state.attack_id,
                attack_method=state.attack["method"],
                attack_strength=state.strength,
                seed=state.seed,
                source_records=state.embed_quality_records,
                source_scope="original_vs_watermarked",
                target_dir=state.attacked_dir,
                device=device,
                reuse_policy="identity_attack_watermarked_copy",
            ),
            _compute_identity_quality_deferred(
                cell_key=state.cell_key,
                scope="watermarked_vs_attacked_watermarked",
                dataset_id=state.dataset_id,
                algorithm_id=state.algorithm_id,
                attack_id=state.attack_id,
                attack_method=state.attack["method"],
                attack_strength=state.strength,
                seed=state.seed,
                reference_dir=state.watermarked_dir,
                target_dir=state.attacked_dir,
                device=device,
            ),
        ]
    else:
        results = [
            _compute_quality_pairs_deferred(
                cell_key=state.cell_key,
                scope="original_vs_attacked_watermarked",
                dataset_id=state.dataset_id,
                algorithm_id=state.algorithm_id,
                attack_id=state.attack_id,
                attack_method=state.attack["method"],
                attack_strength=state.strength,
                seed=state.seed,
                reference_dir=state.canonical_input_dir,
                target_dir=state.attacked_dir,
                device=device,
                quality_pair_cache=quality_pair_cache,
            ),
            _compute_quality_pairs_deferred(
                cell_key=state.cell_key,
                scope="watermarked_vs_attacked_watermarked",
                dataset_id=state.dataset_id,
                algorithm_id=state.algorithm_id,
                attack_id=state.attack_id,
                attack_method=state.attack["method"],
                attack_strength=state.strength,
                seed=state.seed,
                reference_dir=state.watermarked_dir,
                target_dir=state.attacked_dir,
                device=device,
                quality_pair_cache=quality_pair_cache,
            ),
        ]

    records: list[JsonDict] = []
    runtime_profiles: list[JsonDict] = []
    errors: list[str] = []
    for result in results:
        records.extend(result.quality_records)
        runtime_profiles.extend(result.runtime_profiles)
        if result.error:
            errors.append(result.error)
    return DeferredQualityResult(
        quality_records=records,
        runtime_profiles=runtime_profiles,
        error="; ".join(errors) or None,
    )


def run_local_experiment(
    request: LocalRunRequest,
    on_state: RunStateCallback | None = None,
    should_cancel: StopIntentCallback | None = None,
) -> JsonDict:
    selection = normalize_selection(request.selection, request.resources_root)
    if not selection["datasetIds"]:
        raise ValueError("No datasets found under resources/datasets")
    run_root = request.runs_root / safe_segment(request.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    materialized_root = _materialized_root(run_root)
    paths = _artifact_paths(run_root)
    previous_summary = _read_json_object(paths["runSummary"])
    latest_result_units = _latest_result_unit_map(paths["resultUnits"]) if request.resume else {}
    pending_groups, existing_completed, skipped_units = pending_variant_groups(
        selection,
        latest_result_units,
        resume=request.resume,
    )
    pending_cell_count = sum(len(variants) for variants in pending_groups.values())
    result_units: list[JsonDict] = list(existing_completed.values())
    emitted_result_unit_keys = {
        str(unit.get("resultUnitKey") or unit.get("cellKey"))
        for unit in result_units
        if unit.get("resultUnitKey") or unit.get("cellKey")
    }
    started = time.perf_counter()
    estimate = estimate_selection(selection, request.resources_root)
    expected_cells = int(estimate["cellCount"])
    expected_samples_by_dataset: dict[str, int] = {}
    for dataset_id_value in selection["datasetIds"]:
        dataset_id = str(dataset_id_value)
        dataset = get_dataset_by_id(request.resources_root, dataset_id)
        expected_samples_by_dataset[dataset_id] = len(_selected_dataset_image_files(dataset.path, selection, dataset_id))
    expected_sample_total = sum(expected_samples_by_dataset.values())
    expected_watermark_images = expected_sample_total * len(selection["algorithmIds"]) * len(selection["seeds"])
    scene_cache_runtime = _configure_3d_scene_cache_runtime_min(
        max(expected_samples_by_dataset.values(), default=int(selection["maxSamples"]))
    )
    stop_status: str | None = None
    negative_attack_cache: dict[str, dict[str, Any]] = {}
    attack_order = {
        str(attack_id): index
        for index, attack_id in enumerate(selection.get("attackPresetIds") or [])
    }

    _write_json(
        paths["runPlan"],
        build_run_plan_payload(
            run_id=request.run_id,
            selection=selection,
            artifact_paths=paths,
            expected_cells=expected_cells,
            pending_groups=pending_groups,
            skipped_cells=skipped_units,
            resume=request.resume,
            created_at=_utc_timestamp(),
        ),
    )
    _write_run_status(
        paths,
        run_id=request.run_id,
        status="running",
        result_units=len(result_units),
        expected_result_units=expected_cells,
    )
    state_writer = RunStateWriter(
        paths=paths,
        run_id=request.run_id,
        run_root=run_root,
        selection=selection,
        expected_result_units=expected_cells,
        materialized_root=materialized_root,
        on_state=on_state,
    )
    state_writer.write_initial(status="running")

    existing_sample_keys = {
        (str(record.get("datasetId")), str(record.get("sampleId")))
        for record in _read_jsonl(paths["sampleManifest"])
        if record.get("datasetId") is not None and record.get("sampleId") is not None
    }
    quality_pair_cache = QualityPairCache()

    def record_quality_pairs_cached(*args: Any, **kwargs: Any) -> list[JsonDict]:
        kwargs.setdefault("quality_pair_cache", quality_pair_cache)
        return _record_quality_pairs(*args, **kwargs)

    dataset_stage = DatasetStage(
        paths=paths,
        run_id=request.run_id,
        append_jsonl=_append_jsonl,
        image_sample_id=_image_sample_id,
        utc_timestamp=_utc_timestamp,
    )
    watermark_stage = WatermarkStage(
        paths=paths,
        run_id=request.run_id,
        device=request.device,
        message=request.message,
        reset_gpu_peak=_reset_gpu_peak,
        record_runtime_profile=_record_runtime_profile,
        record_watermark_embed_results=_record_watermark_embed_results,
        record_quality_pairs=record_quality_pairs_cached,
    )
    attack_stage = AttackStage(
        paths=paths,
        run_id=request.run_id,
        device=request.device,
        reset_gpu_peak=_reset_gpu_peak,
        list_image_files=_list_image_files,
        record_runtime_profile=_record_runtime_profile,
        record_attack_results=_record_attack_results,
    )
    extract_stage = ExtractStage(
        paths=paths,
        run_id=request.run_id,
        device=request.device,
        message=request.message,
        reset_gpu_peak=_reset_gpu_peak,
        list_image_files=_list_image_files,
        record_runtime_profile=_record_runtime_profile,
    )
    detection_stage = DetectionStage(
        paths=paths,
        run_id=request.run_id,
        append_jsonl=_append_jsonl,
        detection_record=_detection_record,
    )
    quality_stage = QualityStage(
        device=request.device,
        compute_attack_quality=_compute_attack_quality_deferred,
    )
    resource_manager = RuntimeResourceManager(
        paths=paths,
        run_id=request.run_id,
        device=request.device,
        append_jsonl=_append_jsonl,
    )
    run_cleanup_done = False

    def cleanup_run_once(*, reason: str) -> None:
        nonlocal run_cleanup_done
        if run_cleanup_done:
            return
        run_cleanup_done = True
        negative_attack_cache.clear()
        clear_transient_experiment_caches()
        resource_manager.cleanup(
            scope="run",
            reason=reason,
            cell_key="run",
            release_attacks=True,
            release_watermarks=True,
            release_perceptual=True,
            release_auxiliary=True,
        )

    def poll_stop() -> bool:
        nonlocal stop_status
        requested_stop_status = _stop_status_from_callback(should_cancel)
        if requested_stop_status is None:
            return False
        stop_status = requested_stop_status
        return True

    def emit_failed_variant(
        *,
        dataset_id: str,
        algorithm_id: str,
        algorithm: JsonDict,
        seed: int,
        copied_samples: list[Path],
        variant: JsonDict,
        error: str,
        elapsed_ms: float,
        canonical_input_dir: Path,
        watermarked_dir: Path,
    ) -> None:
        attack = variant["attack"]
        attack_params = normalize_attack_params_for_runtime(str(attack["method"]), dict(variant["attackParams"]))
        cell_key = str(variant["cellKey"])
        attack_id = str(variant["attackId"])
        variant_key = str(variant.get("variantKey") or "")
        variant_root = _run_variant_root(
            run_root,
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            seed=seed,
            attack_id=attack_id,
            variant_key=variant_key,
        )
        variant_root.mkdir(parents=True, exist_ok=True)
        detection_manifest = variant_root / "detection.json"
        _write_json(detection_manifest, [])
        result_unit = {
            "runId": request.run_id,
            "cellKey": cell_key,
            "resultUnitKey": cell_key,
            "status": "failed",
            "datasetId": dataset_id,
            "algorithmId": algorithm_id,
            "watermarkMethod": algorithm["method"],
            "attackPresetId": attack_id,
            "attackMethod": attack["method"],
            "attackStrength": variant["strength"],
            "seed": int(seed),
            "sampleCount": len(copied_samples),
            "attackParams": attack_params,
            "manifestPath": str(detection_manifest),
            "outputDir": str(variant_root),
            "variantKey": variant_key,
            "materialized": {
                "canonicalDir": str(canonical_input_dir),
                "watermarkedDir": str(watermarked_dir),
            },
            "error": error,
            "elapsedMs": elapsed_ms,
            "completedAt": _utc_timestamp(),
        }
        result_unit_manifest = variant_root / "result_unit.json"
        _write_json(result_unit_manifest, result_unit)
        _append_jsonl(paths["resultUnits"], result_unit)
        result_units.append(result_unit)
        emitted_result_unit_keys.add(cell_key)
        state_writer.upsert_tree_path(
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            seed=seed,
            attack_id=attack_id,
            variant_key=variant_key,
            refs={
                "dir": str(variant_root),
                "resultUnit": str(result_unit_manifest),
                "detection": str(detection_manifest),
                "status": "failed",
            },
        )

    def emit_completed_variant(state: MaterializedCellState) -> tuple[JsonDict | None, Path]:
        cell_key = state.cell_key
        result_unit_manifest = state.variant_root / "result_unit.json"
        if cell_key in emitted_result_unit_keys:
            existing = next(
                (
                    unit
                    for unit in result_units
                    if str(unit.get("resultUnitKey") or unit.get("cellKey") or "") == cell_key
                ),
                None,
            )
            return existing, result_unit_manifest

        elapsed_ms = (time.perf_counter() - state.cell_started) * 1000
        result_unit = {
            "runId": request.run_id,
            "cellKey": state.cell_key,
            "resultUnitKey": state.cell_key,
            "status": state.status,
            "datasetId": state.dataset_id,
            "algorithmId": state.algorithm_id,
            "watermarkMethod": state.algorithm["method"],
            "attackPresetId": state.attack_id,
            "attackMethod": state.attack["method"],
            "attackStrength": state.strength,
            "seed": state.seed,
            "sampleCount": len(state.copied_samples),
            "attackParams": state.attack_params,
            "manifestPath": str(state.detection_manifest_path),
            "outputDir": str(state.cell_root),
            "variantKey": state.variant_key,
            "materialized": {
                "canonicalDir": str(state.canonical_input_dir),
                "watermarkedDir": str(state.watermarked_dir),
                "attackedDir": str(state.attacked_dir),
                "negativeAttackedDir": str(state.negative_attacked_dir),
            },
            "error": state.error,
            "elapsedMs": elapsed_ms,
            "completedAt": _utc_timestamp(),
        }
        _write_json(result_unit_manifest, result_unit)
        _append_jsonl(paths["resultUnits"], result_unit)
        result_units.append(result_unit)
        emitted_result_unit_keys.add(cell_key)
        state_writer.upsert_tree_path(
            dataset_id=state.dataset_id,
            algorithm_id=state.algorithm_id,
            seed=state.seed,
            attack_id=state.attack_id,
            variant_key=state.variant_key,
            refs={
                "resultUnit": str(result_unit_manifest),
                "detection": str(state.detection_manifest_path),
                "status": state.status,
            },
        )
        return result_unit, result_unit_manifest

    dataset_contexts: list[MaterializedDatasetContext] = []
    try:
        all_states: list[MaterializedCellState] = []

        # Phase 1: materialize all canonical datasets first. This keeps the visible
        # experiment timeline aligned with the global pipeline instead of interleaving
        # dataset-level embed/attack/extract work.
        state_writer.phase_start(
            "canonical",
            total=expected_sample_total,
            current_item={"datasetCount": len(selection["datasetIds"])},
            counters={"datasetsDone": 0, "imagesDone": 0},
            artifact_refs={"root": str(run_root)},
        )
        canonical_images_done = 0
        canonical_datasets_done = 0
        for dataset_id_value in selection["datasetIds"]:
            dataset_id = str(dataset_id_value)
            if poll_stop():
                break
            if not any(group_key[0] == dataset_id for group_key in pending_groups):
                continue

            dataset = get_dataset_by_id(request.resources_root, dataset_id)
            max_samples = int(selection["maxSamples"])
            canonical_source_paths = _selected_dataset_image_files(dataset.path, selection, dataset_id)
            dataset_digest = _dataset_fingerprint(dataset.path, canonical_source_paths)
            path_planner = MaterializedPathPlanner(
                root=materialized_root,
                dataset_id=dataset_id,
                dataset_digest=dataset_digest,
                max_samples=max_samples,
            )
            canonical_input_dir = path_planner.canonical_dir()
            _hydrate_prefix_records_from_compatible_dirs(
                parent_dir=path_planner.canonical_parent_dir(),
                target_dir=canonical_input_dir,
                digest=dataset_digest,
                manifest_name=CANONICAL_MANIFEST_NAME,
                expected_count=len(canonical_source_paths),
                expected_input_paths=canonical_source_paths,
                output_key="outputPath",
                input_key="sourcePath",
            )
            dataset_stage_result = dataset_stage.prepare(
                dataset_id=dataset_id,
                dataset_path=dataset.path,
                input_dir=canonical_input_dir,
                max_samples=max_samples,
                existing_sample_keys=existing_sample_keys,
                sample_paths=canonical_source_paths,
            )
            dataset_contexts.append(
                MaterializedDatasetContext(
                    dataset_id=dataset_id,
                    path_planner=path_planner,
                    canonical_input_dir=canonical_input_dir,
                    copied_samples=dataset_stage_result.copied_samples,
                )
            )
            canonical_root = _run_canonical_root(run_root, dataset_id)
            canonical_root.mkdir(parents=True, exist_ok=True)
            canonical_manifest = canonical_root / "manifest.json"
            _write_json(
                canonical_manifest,
                {
                    "runId": request.run_id,
                    "datasetId": dataset_id,
                    "sampleCount": len(dataset_stage_result.copied_samples),
                    "materializedDir": str(canonical_input_dir),
                    "samples": [str(path) for path in dataset_stage_result.copied_samples],
                },
            )
            state_writer.upsert_tree_path(
                dataset_id=dataset_id,
                refs={
                    "canonical": {
                        "dir": str(canonical_root),
                        "manifest": str(canonical_manifest),
                        "materializedDir": str(canonical_input_dir),
                        "sampleCount": len(dataset_stage_result.copied_samples),
                    }
                },
            )
            canonical_images_done += len(dataset_stage_result.copied_samples)
            canonical_datasets_done += 1
            state_writer.phase_advance(
                "canonical",
                current=canonical_images_done,
                current_item={
                    "datasetId": dataset_id,
                    "sampleCount": len(dataset_stage_result.copied_samples),
                    "materializedDir": str(canonical_input_dir),
                },
                counters={"datasetsDone": canonical_datasets_done, "imagesDone": canonical_images_done},
                artifact_refs={"latestManifest": str(canonical_manifest)},
            )

        state_writer.phase_finish(
            "canonical",
            status=stop_status or "succeeded",
            counters={"datasetsDone": canonical_datasets_done, "imagesDone": canonical_images_done},
        )

        # Phase 2: embed every selected dataset for each watermark algorithm/seed.
        # The watermark cache is kept alive across datasets for the same algorithm/seed
        # and released only after that global embed group finishes.
        if stop_status is None:
            state_writer.phase_start(
                "watermark_embed",
                total=expected_watermark_images,
                current_item={"algorithmCount": len(selection["algorithmIds"]), "seedCount": len(selection["seeds"])},
                counters={
                    "imagesDone": 0,
                    "groupsDone": 0,
                    "cacheHits": 0,
                    "artifactCacheHits": 0,
                    "phaseCellsDone": len(result_units),
                    "phaseCellsTotal": expected_cells,
                },
            )
            watermark_images_done = 0
            watermark_groups_done = 0
            watermark_cache_hits = 0
            watermark_cells_done = len(result_units)
            for algorithm_id_value in selection["algorithmIds"]:
                if stop_status is not None or poll_stop():
                    break
                algorithm_id = str(algorithm_id_value)
                if not any(
                    group_key[1] == algorithm_id
                    for group_key in pending_groups
                ):
                    continue

                algorithm = get_watermark_catalog_item(algorithm_id)
                algorithm_params = dict(algorithm.get("params") or {})
                algorithm_runtime_fingerprint = _runtime_cache_fingerprint(
                    resources_root=request.resources_root,
                    resource=algorithm,
                    method=str(algorithm["method"]),
                    device=request.device,
                )
                for seed_value in selection["seeds"]:
                    if stop_status is not None or poll_stop():
                        break
                    seed = int(seed_value)
                    if not any(
                        pending_groups.get((context.dataset_id, algorithm_id, seed))
                        for context in dataset_contexts
                    ):
                        continue

                    embed_group_key = f"{safe_segment(algorithm_id)}__{seed}__watermark_embed"
                    try:
                        for context in dataset_contexts:
                            if stop_status is not None or poll_stop():
                                break
                            dataset_id = context.dataset_id
                            pending_variants = pending_groups.get((dataset_id, algorithm_id, seed), [])
                            if not pending_variants:
                                continue

                            path_planner = context.path_planner
                            canonical_input_dir = context.canonical_input_dir
                            copied_samples = context.copied_samples
                            embed_key = _cell_key(dataset_id, algorithm_id, "watermark_embed", 0.0, seed)
                            watermarked_dir, watermarked_digest = path_planner.watermarked_dir(
                                algorithm_id=algorithm_id,
                                algorithm_method=str(algorithm["method"]),
                                algorithm_params=algorithm_params,
                                runtime_fingerprint=algorithm_runtime_fingerprint,
                                seed=seed,
                                message=request.message,
                            )
                            embed_record_matches = lambda record, method=str(algorithm["method"]), params=algorithm_params: (
                                _watermark_embed_record_matches(
                                    record,
                                    method_name=method,
                                    params=params,
                                    message=request.message,
                                )
                            )
                            embed_error = None
                            embed_elapsed_ms = 0.0
                            embed_quality_records: list[JsonDict] = []
                            watermark_run_root = _run_watermark_root(run_root, dataset_id, algorithm_id, seed) / "watermark"
                            watermark_run_root.mkdir(parents=True, exist_ok=True)

                            def record_embed_artifact() -> None:
                                nonlocal watermark_cells_done, watermark_images_done, watermark_groups_done
                                embed_manifest = watermark_run_root / "manifest.json"
                                _write_json(
                                    embed_manifest,
                                    {
                                        "runId": request.run_id,
                                        "datasetId": dataset_id,
                                        "algorithmId": algorithm_id,
                                        "algorithmMethod": algorithm["method"],
                                        "algorithmParams": algorithm_params,
                                        "seed": seed,
                                        "sampleCount": len(copied_samples),
                                        "status": "failed" if embed_error else "succeeded",
                                        "materializedDir": str(watermarked_dir),
                                        "cacheDigest": watermarked_digest,
                                        "error": embed_error,
                                    },
                                )
                                state_writer.upsert_tree_path(
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    seed=seed,
                                    refs={
                                        "watermark": {
                                            "dir": str(watermark_run_root),
                                            "manifest": str(embed_manifest),
                                            "materializedDir": str(watermarked_dir),
                                            "method": algorithm["method"],
                                            "params": algorithm_params,
                                            "sampleCount": len(copied_samples),
                                            "status": "failed" if embed_error else "succeeded",
                                        }
                                    },
                                )
                                watermark_images_done += len(copied_samples)
                                watermark_groups_done += 1
                                watermark_cells_done += len(pending_variants)
                                state_writer.phase_advance(
                                    "watermark_embed",
                                    current=watermark_images_done,
                                    current_item={
                                        "datasetId": dataset_id,
                                        "algorithmId": algorithm_id,
                                        "algorithmMethod": algorithm["method"],
                                        "algorithmParams": algorithm_params,
                                        "seed": seed,
                                        "processedImages": len(copied_samples),
                                        "remainingImages": max(
                                            0,
                                            expected_watermark_images - watermark_images_done,
                                        ),
                                        "materializedDir": str(watermarked_dir),
                                    },
                                    counters={
                                        "imagesDone": watermark_images_done,
                                        "groupsDone": watermark_groups_done,
                                        "cacheHits": watermark_cache_hits,
                                        "artifactCacheHits": watermark_cache_hits,
                                        "phaseCellsDone": watermark_cells_done,
                                        "phaseCellsTotal": expected_cells,
                                    },
                                    artifact_refs={"latestManifest": str(embed_manifest)},
                                )

                            try:
                                _hydrate_prefix_records_from_compatible_dirs(
                                    parent_dir=watermarked_dir.parent,
                                    target_dir=watermarked_dir,
                                    digest=watermarked_digest,
                                    manifest_name="watermark_embed_manifest.json",
                                    expected_count=len(copied_samples),
                                    expected_input_paths=copied_samples,
                                    output_key="output_path",
                                    input_key="input_path",
                                    record_matches=embed_record_matches,
                                )
                                reused_embed_results = _watermark_embed_results_from_manifest(
                                    watermarked_dir / "watermark_embed_manifest.json",
                                    expected_count=len(copied_samples),
                                    record_matches=embed_record_matches,
                                )
                                if reused_embed_results is not None:
                                    watermark_cache_hits += len(reused_embed_results)
                                    embed_quality_records = _record_reused_watermark_embed(
                                        paths=paths,
                                        run_id=request.run_id,
                                        cell_key=embed_key,
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        algorithm=algorithm,
                                        seed=seed,
                                        input_dir=canonical_input_dir,
                                        output_dir=watermarked_dir,
                                        copied_samples=copied_samples,
                                        results=reused_embed_results,
                                        device=request.device,
                                        quality_pair_cache=quality_pair_cache,
                                    )
                                else:
                                    prefix_embed_results = _watermark_embed_prefix_from_manifest(
                                        watermarked_dir / "watermark_embed_manifest.json",
                                        expected_count=len(copied_samples),
                                        record_matches=embed_record_matches,
                                    )
                                    if prefix_embed_results:
                                        watermark_cache_hits += len(prefix_embed_results)
                                        embed_quality_records.extend(
                                            _record_reused_watermark_embed(
                                                paths=paths,
                                                run_id=request.run_id,
                                                cell_key=embed_key,
                                                dataset_id=dataset_id,
                                                algorithm_id=algorithm_id,
                                                algorithm=algorithm,
                                                seed=seed,
                                                input_dir=canonical_input_dir,
                                                output_dir=watermarked_dir,
                                                copied_samples=copied_samples[: len(prefix_embed_results)],
                                                results=prefix_embed_results,
                                                device=request.device,
                                                quality_pair_cache=quality_pair_cache,
                                            )
                                        )
                                        suffix_inputs = copied_samples[len(prefix_embed_results) :]
                                        suffix_input_dir = _prepare_subset_input_dir(
                                            canonical_input_dir,
                                            suffix_inputs,
                                            watermarked_dir / INTERMEDIATE_ARTIFACT_DIR / f"embed_suffix_{len(prefix_embed_results)}",
                                        )
                                        watermark_method = get_cached_watermark(
                                            algorithm["method"],
                                            algorithm_params,
                                            request.device,
                                        )
                                        _reset_gpu_peak(request.device)
                                        suffix_started = time.perf_counter()
                                        raw_suffix_results = run_watermark_embed_dir_with_method(
                                            WatermarkEmbedJob(
                                                run_id=request.run_id,
                                                method_name=algorithm["method"],
                                                params=algorithm_params,
                                                input_dir=suffix_input_dir,
                                                output_dir=watermarked_dir,
                                                device=request.device,
                                                seed=seed + len(prefix_embed_results),
                                                message=request.message,
                                            ),
                                            watermark_method,
                                        )
                                        embed_elapsed_ms = (time.perf_counter() - suffix_started) * 1000
                                        suffix_results = _retarget_embed_results(
                                            raw_suffix_results,
                                            subset_input_dir=suffix_input_dir,
                                            original_input_root=canonical_input_dir,
                                        )
                                        combined_embed_results = [*prefix_embed_results, *suffix_results]
                                        _write_embed_manifest(
                                            watermarked_dir / "watermark_embed_manifest.json",
                                            combined_embed_results,
                                        )
                                        _record_watermark_embed_results(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=embed_key,
                                            dataset_id=dataset_id,
                                            algorithm_id=algorithm_id,
                                            watermark_method=algorithm["method"],
                                            seed=seed,
                                            input_root=canonical_input_dir,
                                            results=suffix_results,
                                        )
                                        suffix_quality_records = _record_quality_pairs(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=embed_key,
                                            scope="original_vs_watermarked",
                                            dataset_id=dataset_id,
                                            algorithm_id=algorithm_id,
                                            attack_id=None,
                                            attack_method=None,
                                            attack_strength=None,
                                            seed=seed,
                                            reference_dir=suffix_input_dir,
                                            target_dir=watermarked_dir,
                                            device=request.device,
                                            quality_pair_cache=quality_pair_cache,
                                        )
                                        embed_quality_records.extend(suffix_quality_records)
                                        embed_error = "; ".join(
                                            result.error for result in suffix_results if getattr(result, "error", None)
                                        ) or None
                                        _record_runtime_profile(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=embed_key,
                                            stage="watermark_embed",
                                            method=algorithm["method"],
                                            device=request.device,
                                            elapsed_ms=embed_elapsed_ms,
                                            image_paths=suffix_inputs,
                                            status="failed" if embed_error else "succeeded",
                                            error=embed_error,
                                            metadata={
                                                "partialFill": True,
                                                "reusedSamples": len(prefix_embed_results),
                                                "pendingSamples": len(suffix_inputs),
                                                "materializedDir": str(watermarked_dir),
                                                "execution": summarize_execution_profiles(suffix_results),
                                            },
                                        )
                                        shutil.rmtree(suffix_input_dir, ignore_errors=True)
                                    else:
                                        _clean_output_dir(watermarked_dir)
                                        watermark_stage_result = watermark_stage.embed(
                                            embed_key=embed_key,
                                            dataset_id=dataset_id,
                                            algorithm_id=algorithm_id,
                                            algorithm=algorithm,
                                            algorithm_params=algorithm_params,
                                            seed=seed,
                                            input_dir=canonical_input_dir,
                                            output_dir=watermarked_dir,
                                            copied_samples=copied_samples,
                                        )
                                        embed_quality_records = watermark_stage_result.quality_records
                                        embed_elapsed_ms = watermark_stage_result.elapsed_ms
                                        embed_error = watermark_stage_result.error
                            except Exception as exc:
                                embed_error = f"{type(exc).__name__}: {exc}"
                                record_embed_artifact()
                                for variant in pending_variants:
                                    emit_failed_variant(
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        algorithm=algorithm,
                                        seed=seed,
                                        copied_samples=copied_samples,
                                        variant=variant,
                                        error=embed_error,
                                        elapsed_ms=embed_elapsed_ms,
                                        canonical_input_dir=canonical_input_dir,
                                        watermarked_dir=watermarked_dir,
                                    )
                                continue

                            record_embed_artifact()
                            for variant in pending_variants:
                                attack = variant["attack"]
                                attack_id = str(variant["attackId"])
                                strength = float(variant["strength"])
                                attack_params = normalize_attack_params_for_runtime(
                                    str(attack["method"]),
                                    dict(variant["attackParams"]),
                                )
                                attack_runtime_fingerprint = _runtime_cache_fingerprint(
                                    resources_root=request.resources_root,
                                    resource=attack,
                                    method=str(attack["method"]),
                                    device=request.device,
                                )
                                cell_key = str(variant["cellKey"])
                                variant_key = str(variant.get("variantKey") or "")
                                variant_root = _run_variant_root(
                                    run_root,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    seed=seed,
                                    attack_id=attack_id,
                                    variant_key=variant_key,
                                )
                                variant_root.mkdir(parents=True, exist_ok=True)
                                cell_root = variant_root
                                variant_key = str(variant.get("variantKey") or "")
                                negative_attack_key = path_planner.negative_attack_key(
                                    attack_id=attack_id,
                                    attack_method=str(attack["method"]),
                                    attack_params=attack_params,
                                    runtime_fingerprint=attack_runtime_fingerprint,
                                    strength=strength,
                                    variant_key=variant_key,
                                )
                                attacked_dir = path_planner.positive_attacked_dir(
                                    algorithm_id=algorithm_id,
                                    seed=seed,
                                    cell_key=cell_key,
                                    watermarked_digest=watermarked_digest,
                                    attack_id=attack_id,
                                    attack_method=str(attack["method"]),
                                    attack_params=attack_params,
                                    runtime_fingerprint=attack_runtime_fingerprint,
                                    strength=strength,
                                )
                                state = MaterializedCellState(
                                    variant=variant,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    algorithm=algorithm,
                                    algorithm_params=algorithm_params,
                                    seed=seed,
                                    canonical_input_dir=canonical_input_dir,
                                    copied_samples=copied_samples,
                                    watermarked_dir=watermarked_dir,
                                    embed_quality_records=embed_quality_records,
                                    embed_elapsed_ms=embed_elapsed_ms,
                                    embed_error=embed_error,
                                    cell_key=cell_key,
                                    attack_id=attack_id,
                                    attack=attack,
                                    attack_params=attack_params,
                                    strength=strength,
                                    cell_root=cell_root,
                                    variant_key=variant_key,
                                    variant_root=variant_root,
                                    attacked_dir=attacked_dir,
                                    extracted_dir=variant_root / "extracted_positive",
                                    negative_attack_key=negative_attack_key,
                                    negative_attacked_dir=path_planner.negative_attacked_dir(negative_attack_key),
                                    negative_extracted_dir=variant_root / "extracted_negative",
                                    detection_manifest_path=variant_root / "detection.json",
                                    cell_started=time.perf_counter(),
                                )
                                all_states.append(state)
                                variant_manifest = variant_root / "variant.json"
                                _write_json(
                                    variant_manifest,
                                    {
                                        "runId": request.run_id,
                                        "datasetId": dataset_id,
                                        "algorithmId": algorithm_id,
                                        "seed": seed,
                                        "attackPresetId": attack_id,
                                        "attackMethod": attack["method"],
                                        "attackStrength": strength,
                                        "attackParams": attack_params,
                                        "variantKey": variant_key,
                                        "cellKey": cell_key,
                                    },
                                )
                                state_writer.upsert_tree_path(
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    seed=seed,
                                    attack_id=attack_id,
                                    variant_key=variant_key,
                                    refs={
                                        "dir": str(variant_root),
                                        "variantManifest": str(variant_manifest),
                                        "status": "planned",
                                    },
                                )
                    finally:
                        resource_manager.cleanup(
                            scope="watermark_embed",
                            reason="watermark_embed_group_finished",
                            cell_key=embed_group_key,
                            release_watermarks=True,
                            metadata={
                                "algorithmId": algorithm_id,
                                "watermarkMethod": algorithm["method"],
                                "seed": seed,
                            },
                        )

            state_writer.phase_finish(
                "watermark_embed",
                status=stop_status or "succeeded",
                counters={
                    "imagesDone": watermark_images_done,
                    "groupsDone": watermark_groups_done,
                    "cacheHits": watermark_cache_hits,
                    "artifactCacheHits": watermark_cache_hits,
                    "phaseCellsDone": watermark_cells_done,
                    "phaseCellsTotal": expected_cells,
                },
            )

        # Phase 3: attack every materialized positive/negative image globally. Heavy
        # backends are grouped across all datasets/algorithms/seeds so each backend is
        # loaded once for the whole run.
        if stop_status is None and all_states:
            unique_negative_attack_keys = {state.negative_attack_key: state for state in all_states}
            attack_total_images = sum(len(state.copied_samples) for state in unique_negative_attack_keys.values())
            attack_total_images += sum(len(state.copied_samples) for state in all_states)
            attack_images_done = 0
            positive_attack_images_done = 0
            negative_attack_images_done = 0
            attack_cache_hits = 0
            attack_scene_cache_hits = 0
            attack_backend_done = 0
            attack_cells_done = len(result_units)
            counted_negative_attack_keys: set[str] = set()
            attack_model_groups: dict[str, list[MaterializedCellState]] = {}
            for state in all_states:
                model_key = _attack_model_cache_key(state.attack["method"], state.attack_params, request.device)
                attack_model_groups.setdefault(model_key, []).append(state)
            state_writer.phase_start(
                "attack",
                total=attack_total_images,
                current_item={
                    "attackBackendCount": len(attack_model_groups),
                    "variantCount": len(all_states),
                },
                counters={
                    "imagesDone": 0,
                    "backendDone": 0,
                    "cacheHits": 0,
                    "artifactCacheHits": 0,
                    "sceneCacheHits": 0,
                    "positiveImagesDone": 0,
                    "negativeImagesDone": 0,
                    "phaseCellsDone": attack_cells_done,
                    "phaseCellsTotal": expected_cells,
                },
            )

            def attack_model_group_order(item: tuple[str, list[MaterializedCellState]]) -> tuple[int, str, str]:
                model_key, states = item
                min_attack_order = min(
                    attack_order.get(state.attack_id, len(attack_order) + 1)
                    for state in states
                )
                first_method = str(states[0].attack["method"]) if states else ""
                return (min_attack_order, first_method, model_key)

            for attack_model_key, model_states in sorted(attack_model_groups.items(), key=attack_model_group_order):
                if stop_status is not None or poll_stop():
                    break
                model_digest = _stable_json_digest(attack_model_key, length=12)
                first_state = model_states[0]
                model_dataset_ids = sorted({state.dataset_id for state in model_states})
                attack_model_cell_key = f"attack_model__{model_digest}"
                runtime_groups: dict[str, list[MaterializedCellState]] = {}
                for state in model_states:
                    runtime_key = json.dumps(
                        {
                            "method": str(state.attack["method"]).lower(),
                            "params": state.attack_params,
                            "strength": state.strength,
                        },
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    )
                    runtime_groups.setdefault(runtime_key, []).append(state)

                def runtime_group_order(item: tuple[str, list[MaterializedCellState]]) -> tuple[int, float, str, str]:
                    runtime_key, states = item
                    first = states[0]
                    return (
                        attack_order.get(first.attack_id, len(attack_order) + 1),
                        first.strength,
                        str(first.attack["method"]),
                        runtime_key,
                    )

                try:
                    for runtime_key, runtime_states in sorted(runtime_groups.items(), key=runtime_group_order):
                        if stop_status is not None or poll_stop():
                            break
                        runtime_state = runtime_states[0]
                        try:
                            attack_instance = get_cached_attack(
                                runtime_state.attack["method"],
                                runtime_state.attack_params,
                                request.device,
                            )
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            for state in runtime_states:
                                state.status = "failed"
                                state.error = error
                            continue

                        for state in runtime_states:
                            if stop_status is not None or poll_stop():
                                break
                            try:
                                negative_record_matches = lambda record, method=str(state.attack["method"]), params=state.attack_params: (
                                    _attack_record_matches(record, attack_name=method, params=params)
                                )
                                _hydrate_prefix_records_from_compatible_dirs(
                                    parent_dir=state.negative_attacked_dir.parent,
                                    target_dir=state.negative_attacked_dir,
                                    digest=state.negative_attack_key,
                                    manifest_name="attack_manifest.json",
                                    expected_count=len(state.copied_samples),
                                    expected_input_paths=state.copied_samples,
                                    output_key="output_path",
                                    input_key="input_path",
                                    record_matches=negative_record_matches,
                                )
                                cached_negative = _attack_results_from_manifest(
                                    state.negative_attacked_dir / "attack_manifest.json",
                                    expected_count=len(state.copied_samples),
                                    record_matches=negative_record_matches,
                                )
                                if cached_negative is not None:
                                    attack_cache_hits += len(cached_negative)
                                    negative_attack_cache[state.negative_attack_key] = {
                                        "outputDir": state.negative_attacked_dir,
                                        "results": cached_negative,
                                        "error": None,
                                    }
                                    _record_reused_attack(
                                        paths=paths,
                                        run_id=request.run_id,
                                        cell_key=state.cell_key,
                                        runtime_stage="attack_negative_control",
                                        dataset_id=state.dataset_id,
                                        algorithm_id=state.algorithm_id,
                                        attack_id=state.attack_id,
                                        attack=state.attack,
                                        attack_params=state.attack_params,
                                        strength=state.strength,
                                        seed=0,
                                        label=0,
                                        input_dir=state.canonical_input_dir,
                                        output_dir=state.negative_attacked_dir,
                                        results=cached_negative,
                                        device=request.device,
                                        cache_key=state.negative_attack_key,
                                    )
                                    state.negative_attack_results = cached_negative
                                    _mark_state_operation_result(state, cached_negative)
                                else:
                                    prefix_negative = _attack_prefix_from_manifest(
                                        state.negative_attacked_dir / "attack_manifest.json",
                                        expected_count=len(state.copied_samples),
                                        record_matches=negative_record_matches,
                                    )
                                    if prefix_negative:
                                        prefix_count = len(prefix_negative)
                                        attack_cache_hits += prefix_count
                                        _record_reused_attack(
                                            paths=paths,
                                            run_id=request.run_id,
                                            cell_key=state.cell_key,
                                            runtime_stage="attack_negative_control",
                                            dataset_id=state.dataset_id,
                                            algorithm_id=state.algorithm_id,
                                            attack_id=state.attack_id,
                                            attack=state.attack,
                                            attack_params=state.attack_params,
                                            strength=state.strength,
                                            seed=0,
                                            label=0,
                                            input_dir=state.canonical_input_dir,
                                            output_dir=state.negative_attacked_dir,
                                            results=prefix_negative,
                                            device=request.device,
                                            cache_key=state.negative_attack_key,
                                            image_paths=state.copied_samples[:prefix_count],
                                        )
                                        suffix_inputs = state.copied_samples[prefix_count:]
                                        suffix_input_dir = _prepare_subset_input_dir(
                                            state.canonical_input_dir,
                                            suffix_inputs,
                                            state.negative_attacked_dir
                                            / INTERMEDIATE_ARTIFACT_DIR
                                            / f"attack_suffix_{prefix_count}",
                                        )
                                        _reset_gpu_peak(request.device)
                                        suffix_started = time.perf_counter()
                                        raw_suffix = run_attack_dir_with_attack(
                                            AttackJob(
                                                run_id=request.run_id,
                                                attack_name=state.attack["method"],
                                                params=state.attack_params,
                                                input_dir=suffix_input_dir,
                                                output_dir=state.negative_attacked_dir,
                                                device=request.device,
                                                seed=prefix_count,
                                            ),
                                            attack_instance,
                                        )
                                        suffix_elapsed_ms = (time.perf_counter() - suffix_started) * 1000
                                        suffix_results = _retarget_attack_results(
                                            raw_suffix,
                                            subset_input_dir=suffix_input_dir,
                                            original_input_root=state.canonical_input_dir,
                                        )
                                        combined_negative = [*prefix_negative, *suffix_results]
                                        _write_attack_manifest(
                                            state.negative_attacked_dir / "attack_manifest.json",
                                            combined_negative,
                                        )
                                        _record_attack_results(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=state.cell_key,
                                            stage="attack_negative_control",
                                            dataset_id=state.dataset_id,
                                            algorithm_id=state.algorithm_id,
                                            attack_id=state.attack_id,
                                            attack_method=state.attack["method"],
                                            attack_strength=state.strength,
                                            attack_params=state.attack_params,
                                            seed=0,
                                            label=0,
                                            input_root=state.canonical_input_dir,
                                            results=suffix_results,
                                            cache_hit=False,
                                        )
                                        suffix_error = "; ".join(
                                            result.error for result in suffix_results if getattr(result, "error", None)
                                        )
                                        _record_runtime_profile(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=state.cell_key,
                                            stage="attack_negative_control",
                                            method=state.attack["method"],
                                            device=request.device,
                                            elapsed_ms=suffix_elapsed_ms,
                                            image_paths=suffix_inputs,
                                            status="failed" if suffix_error else "succeeded",
                                            error=suffix_error or None,
                                            metadata={
                                                "attackParams": state.attack_params,
                                                "cacheKey": state.negative_attack_key,
                                                "partialFill": True,
                                                "reusedSamples": prefix_count,
                                                "pendingSamples": len(suffix_inputs),
                                                "materializedDir": str(state.negative_attacked_dir),
                                                "execution": summarize_execution_profiles(suffix_results),
                                            },
                                        )
                                        shutil.rmtree(suffix_input_dir, ignore_errors=True)
                                        negative_attack_cache[state.negative_attack_key] = {
                                            "outputDir": state.negative_attacked_dir,
                                            "results": combined_negative,
                                            "error": suffix_error or None,
                                        }
                                        state.negative_attack_results = combined_negative
                                        attack_scene_cache_hits += _scene_cache_hit_count(suffix_results)
                                        _mark_state_operation_result(state, combined_negative)
                                    else:
                                        _clean_output_dir(state.negative_attacked_dir)
                                        negative_attack = attack_stage.negative_control(
                                            cell_key=state.cell_key,
                                            dataset_id=state.dataset_id,
                                            algorithm_id=state.algorithm_id,
                                            attack_id=state.attack_id,
                                            attack=state.attack,
                                            attack_params=state.attack_params,
                                            strength=state.strength,
                                            seed=0,
                                            input_dir=state.canonical_input_dir,
                                            output_dir=state.negative_attacked_dir,
                                            copied_samples=state.copied_samples,
                                            cache_key=state.negative_attack_key,
                                            cache=negative_attack_cache,
                                            attack_instance=attack_instance,
                                        )
                                        state.negative_attack_results = negative_attack.results
                                        attack_scene_cache_hits += _scene_cache_hit_count(negative_attack.results)
                                        _mark_state_operation_result(state, negative_attack.results)
                                        state.negative_attacked_dir = negative_attack.output_dir
                            except Exception as exc:
                                state.status = "failed"
                                state.error = f"{type(exc).__name__}: {exc}"
                            negative_control_root = _run_negative_control_root(
                                run_root,
                                state.dataset_id,
                                state.attack_id,
                                state.variant_key,
                            )
                            negative_control_root.mkdir(parents=True, exist_ok=True)
                            negative_control_manifest = negative_control_root / "manifest.json"
                            negative_attack_status, negative_attack_error = _stage_status_and_error(
                                state.negative_attack_results,
                                fallback_error=state.error,
                                expected_count=len(state.copied_samples),
                            )
                            _write_json(
                                negative_control_manifest,
                                {
                                    "runId": request.run_id,
                                    "datasetId": state.dataset_id,
                                    "attackPresetId": state.attack_id,
                                    "attackMethod": state.attack["method"],
                                    "attackStrength": state.strength,
                                    "attackParams": state.attack_params,
                                    "variantKey": state.variant_key,
                                    "sampleCount": len(state.copied_samples),
                                    "materializedDir": str(state.negative_attacked_dir),
                                    "status": negative_attack_status,
                                    "error": negative_attack_error,
                                    "cellStatus": state.status,
                                    "cellError": state.error,
                                },
                            )
                            state_writer.upsert_tree_path(
                                dataset_id=state.dataset_id,
                                refs={
                                    "negativeControls": {
                                        f"{state.attack_id}:{state.variant_key}": {
                                            "dir": str(negative_control_root),
                                            "manifest": str(negative_control_manifest),
                                            "materializedDir": str(state.negative_attacked_dir),
                                        }
                                    }
                                },
                            )
                            negative_progress_images = 0
                            if state.negative_attack_key not in counted_negative_attack_keys:
                                counted_negative_attack_keys.add(state.negative_attack_key)
                                negative_progress_images = len(state.copied_samples)
                                attack_images_done += negative_progress_images
                                negative_attack_images_done += negative_progress_images
                            state_writer.phase_advance(
                                "attack",
                                current=attack_images_done,
                                current_item={
                                    "scope": "negative",
                                    "datasetId": state.dataset_id,
                                    "algorithmId": state.algorithm_id,
                                    "seed": state.seed,
                                    "attackPresetId": state.attack_id,
                                    "attackMethod": state.attack["method"],
                                    "attackStrength": state.strength,
                                    "attackParams": state.attack_params,
                                    "variantKey": state.variant_key,
                                    "processedImages": len(state.copied_samples),
                                    "countedImages": negative_progress_images,
                                    "reusedNegativeKey": negative_progress_images == 0,
                                    "remainingImages": max(0, attack_total_images - attack_images_done),
                                    "materializedDir": str(state.negative_attacked_dir),
                                },
                                counters={
                                    "imagesDone": attack_images_done,
                                    "backendDone": attack_backend_done,
                                    "cacheHits": attack_cache_hits,
                                    "artifactCacheHits": attack_cache_hits,
                                    "sceneCacheHits": attack_scene_cache_hits,
                                    "positiveImagesDone": positive_attack_images_done,
                                    "negativeImagesDone": negative_attack_images_done,
                                    "phaseCellsDone": attack_cells_done,
                                    "phaseCellsTotal": expected_cells,
                                },
                                artifact_refs={"latestManifest": str(negative_control_manifest)},
                            )

                        for state in runtime_states:
                            if stop_status is not None or poll_stop():
                                break
                            try:
                                watermarked_inputs = _list_image_files(state.watermarked_dir)
                                positive_attack_digest = _digest_from_sample_count_dir(state.attacked_dir)
                                positive_record_matches = lambda record, method=str(state.attack["method"]), params=state.attack_params: (
                                    _attack_record_matches(record, attack_name=method, params=params)
                                )
                                _hydrate_prefix_records_from_compatible_dirs(
                                    parent_dir=state.attacked_dir.parent,
                                    target_dir=state.attacked_dir,
                                    digest=positive_attack_digest,
                                    manifest_name="attack_manifest.json",
                                    expected_count=len(watermarked_inputs),
                                    expected_input_paths=watermarked_inputs,
                                    output_key="output_path",
                                    input_key="input_path",
                                    record_matches=positive_record_matches,
                                )
                                cached_positive = _attack_results_from_manifest(
                                    state.attacked_dir / "attack_manifest.json",
                                    expected_count=len(watermarked_inputs),
                                    record_matches=positive_record_matches,
                                )
                                if cached_positive is not None:
                                    attack_cache_hits += len(cached_positive)
                                    _record_reused_attack(
                                        paths=paths,
                                        run_id=request.run_id,
                                        cell_key=state.cell_key,
                                        runtime_stage="attack",
                                        dataset_id=state.dataset_id,
                                        algorithm_id=state.algorithm_id,
                                        attack_id=state.attack_id,
                                        attack=state.attack,
                                        attack_params=state.attack_params,
                                        strength=state.strength,
                                        seed=state.seed,
                                        label=1,
                                        input_dir=state.watermarked_dir,
                                        output_dir=state.attacked_dir,
                                        results=cached_positive,
                                        device=request.device,
                                        cache_key=state.cell_key,
                                    )
                                    state.positive_attack_results = cached_positive
                                    _mark_state_operation_result(state, cached_positive)
                                else:
                                    prefix_positive = _attack_prefix_from_manifest(
                                        state.attacked_dir / "attack_manifest.json",
                                        expected_count=len(watermarked_inputs),
                                        record_matches=positive_record_matches,
                                    )
                                    if prefix_positive:
                                        prefix_count = len(prefix_positive)
                                        attack_cache_hits += prefix_count
                                        _record_reused_attack(
                                            paths=paths,
                                            run_id=request.run_id,
                                            cell_key=state.cell_key,
                                            runtime_stage="attack",
                                            dataset_id=state.dataset_id,
                                            algorithm_id=state.algorithm_id,
                                            attack_id=state.attack_id,
                                            attack=state.attack,
                                            attack_params=state.attack_params,
                                            strength=state.strength,
                                            seed=state.seed,
                                            label=1,
                                            input_dir=state.watermarked_dir,
                                            output_dir=state.attacked_dir,
                                            results=prefix_positive,
                                            device=request.device,
                                            cache_key=state.cell_key,
                                            image_paths=watermarked_inputs[:prefix_count],
                                        )
                                        suffix_inputs = watermarked_inputs[prefix_count:]
                                        suffix_input_dir = _prepare_subset_input_dir(
                                            state.watermarked_dir,
                                            suffix_inputs,
                                            state.attacked_dir
                                            / INTERMEDIATE_ARTIFACT_DIR
                                            / f"attack_suffix_{prefix_count}",
                                        )
                                        _reset_gpu_peak(request.device)
                                        suffix_started = time.perf_counter()
                                        raw_suffix = run_attack_dir_with_attack(
                                            AttackJob(
                                                run_id=request.run_id,
                                                attack_name=state.attack["method"],
                                                params=state.attack_params,
                                                input_dir=suffix_input_dir,
                                                output_dir=state.attacked_dir,
                                                device=request.device,
                                                seed=state.seed + prefix_count,
                                            ),
                                            attack_instance,
                                        )
                                        suffix_elapsed_ms = (time.perf_counter() - suffix_started) * 1000
                                        suffix_results = _retarget_attack_results(
                                            raw_suffix,
                                            subset_input_dir=suffix_input_dir,
                                            original_input_root=state.watermarked_dir,
                                        )
                                        combined_positive = [*prefix_positive, *suffix_results]
                                        _write_attack_manifest(
                                            state.attacked_dir / "attack_manifest.json",
                                            combined_positive,
                                        )
                                        _record_attack_results(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=state.cell_key,
                                            stage="attack",
                                            dataset_id=state.dataset_id,
                                            algorithm_id=state.algorithm_id,
                                            attack_id=state.attack_id,
                                            attack_method=state.attack["method"],
                                            attack_strength=state.strength,
                                            attack_params=state.attack_params,
                                            seed=state.seed,
                                            label=1,
                                            input_root=state.watermarked_dir,
                                            results=suffix_results,
                                            cache_hit=False,
                                        )
                                        suffix_error = "; ".join(
                                            result.error for result in suffix_results if getattr(result, "error", None)
                                        )
                                        _record_runtime_profile(
                                            paths,
                                            run_id=request.run_id,
                                            cell_key=state.cell_key,
                                            stage="attack",
                                            method=state.attack["method"],
                                            device=request.device,
                                            elapsed_ms=suffix_elapsed_ms,
                                            image_paths=suffix_inputs,
                                            status="failed" if suffix_error else "succeeded",
                                            error=suffix_error or None,
                                            metadata={
                                                "attackParams": state.attack_params,
                                                "partialFill": True,
                                                "reusedSamples": prefix_count,
                                                "pendingSamples": len(suffix_inputs),
                                                "materializedDir": str(state.attacked_dir),
                                                "execution": summarize_execution_profiles(suffix_results),
                                            },
                                        )
                                        shutil.rmtree(suffix_input_dir, ignore_errors=True)
                                        state.positive_attack_results = combined_positive
                                        attack_scene_cache_hits += _scene_cache_hit_count(suffix_results)
                                        _mark_state_operation_result(state, combined_positive)
                                    else:
                                        _clean_output_dir(state.attacked_dir)
                                        _attack_instance, positive_attack = attack_stage.positive(
                                            cell_key=state.cell_key,
                                            dataset_id=state.dataset_id,
                                            algorithm_id=state.algorithm_id,
                                            attack_id=state.attack_id,
                                            attack=state.attack,
                                            attack_params=state.attack_params,
                                            strength=state.strength,
                                            seed=state.seed,
                                            input_dir=state.watermarked_dir,
                                            output_dir=state.attacked_dir,
                                        )
                                        state.positive_attack_results = positive_attack.results
                                        attack_scene_cache_hits += _scene_cache_hit_count(positive_attack.results)
                                        _mark_state_operation_result(state, positive_attack.results)
                            except Exception as exc:
                                state.status = "failed"
                                state.error = f"{type(exc).__name__}: {exc}"
                            positive_expected_count = len(_list_image_files(state.watermarked_dir))
                            positive_attack_status, positive_attack_error = _stage_status_and_error(
                                state.positive_attack_results,
                                fallback_error=state.error,
                                expected_count=positive_expected_count,
                            )
                            positive_attack_manifest = state.variant_root / "positive_attacked" / "manifest.json"
                            _write_json(
                                positive_attack_manifest,
                                {
                                    "runId": request.run_id,
                                    "datasetId": state.dataset_id,
                                    "algorithmId": state.algorithm_id,
                                    "seed": state.seed,
                                    "attackPresetId": state.attack_id,
                                    "attackMethod": state.attack["method"],
                                    "attackStrength": state.strength,
                                    "attackParams": state.attack_params,
                                    "variantKey": state.variant_key,
                                    "sampleCount": positive_expected_count,
                                    "materializedDir": str(state.attacked_dir),
                                    "negativeControlDir": str(state.negative_attacked_dir),
                                    "status": positive_attack_status,
                                    "error": positive_attack_error,
                                    "cellStatus": state.status,
                                    "cellError": state.error,
                                },
                            )
                            state_writer.upsert_tree_path(
                                dataset_id=state.dataset_id,
                                algorithm_id=state.algorithm_id,
                                seed=state.seed,
                                attack_id=state.attack_id,
                                variant_key=state.variant_key,
                                refs={
                                    "dir": str(state.variant_root),
                                    "positiveAttacked": {
                                        "manifest": str(positive_attack_manifest),
                                        "materializedDir": str(state.attacked_dir),
                                    },
                                    "negativeAttackedRef": str(state.negative_attacked_dir),
                                    "status": state.status,
                                },
                            )
                            positive_image_count = len(_list_image_files(state.watermarked_dir))
                            attack_images_done += positive_image_count
                            positive_attack_images_done += positive_image_count
                            attack_cells_done += 1
                            state_writer.phase_advance(
                                "attack",
                                current=attack_images_done,
                                current_item={
                                    "scope": "positive",
                                    "datasetId": state.dataset_id,
                                    "algorithmId": state.algorithm_id,
                                    "seed": state.seed,
                                    "attackPresetId": state.attack_id,
                                    "attackMethod": state.attack["method"],
                                    "attackStrength": state.strength,
                                    "attackParams": state.attack_params,
                                    "variantKey": state.variant_key,
                                    "processedImages": positive_image_count,
                                    "remainingImages": max(0, attack_total_images - attack_images_done),
                                    "materializedDir": str(state.attacked_dir),
                                },
                                counters={
                                    "imagesDone": attack_images_done,
                                    "backendDone": attack_backend_done,
                                    "cacheHits": attack_cache_hits,
                                    "artifactCacheHits": attack_cache_hits,
                                    "sceneCacheHits": attack_scene_cache_hits,
                                    "positiveImagesDone": positive_attack_images_done,
                                    "negativeImagesDone": negative_attack_images_done,
                                    "phaseCellsDone": attack_cells_done,
                                    "phaseCellsTotal": expected_cells,
                                },
                                artifact_refs={"latestManifest": str(positive_attack_manifest)},
                            )

                finally:
                    resource_manager.cleanup(
                        scope="attack_model",
                        reason="attack_model_finished",
                        cell_key=attack_model_cell_key,
                        release_attacks=True,
                        release_auxiliary=True,
                        metadata={
                            "datasetIds": model_dataset_ids,
                            "attackMethod": first_state.attack["method"],
                            "attackModelKey": model_digest,
                            "variantCount": len(model_states),
                        },
                    )
                    attack_backend_done += 1
                    state_writer.phase_advance(
                        "attack",
                        current=attack_images_done,
                        counters={
                            "imagesDone": attack_images_done,
                            "backendDone": attack_backend_done,
                            "cacheHits": attack_cache_hits,
                            "artifactCacheHits": attack_cache_hits,
                            "sceneCacheHits": attack_scene_cache_hits,
                            "positiveImagesDone": positive_attack_images_done,
                            "negativeImagesDone": negative_attack_images_done,
                            "phaseCellsDone": attack_cells_done,
                            "phaseCellsTotal": expected_cells,
                        },
                    )
            state_writer.phase_finish(
                "attack",
                status=stop_status or "succeeded",
                counters={
                    "imagesDone": attack_images_done,
                    "backendDone": attack_backend_done,
                    "cacheHits": attack_cache_hits,
                    "artifactCacheHits": attack_cache_hits,
                    "sceneCacheHits": attack_scene_cache_hits,
                    "positiveImagesDone": positive_attack_images_done,
                    "negativeImagesDone": negative_attack_images_done,
                    "phaseCellsDone": attack_cells_done,
                    "phaseCellsTotal": expected_cells,
                },
            )

        # Phase 4: extract all attacked positives and reused negatives by watermark
        # algorithm/seed, loading each watermark extractor once for the whole run.
        if stop_status is None and all_states:
            extract_total_images = sum(len(state.copied_samples) * 2 for state in all_states)
            extract_images_done = 0
            extract_groups_done = 0
            extract_cells_done = len(result_units)
            state_writer.phase_start(
                "watermark_extract",
                total=extract_total_images,
                current_item={"watermarkGroupCount": len({(state.algorithm_id, state.seed) for state in all_states})},
                counters={
                    "imagesDone": 0,
                    "groupsDone": 0,
                    "positiveImagesDone": 0,
                    "negativeImagesDone": 0,
                    "phaseCellsDone": extract_cells_done,
                    "phaseCellsTotal": expected_cells,
                },
            )
            extract_positive_images_done = 0
            extract_negative_images_done = 0
            extract_groups_by_watermark: dict[tuple[str, int], list[MaterializedCellState]] = {}
            for state in all_states:
                extract_groups_by_watermark.setdefault((state.algorithm_id, state.seed), []).append(state)

            for (_algorithm_id, _seed), extract_states in sorted(extract_groups_by_watermark.items()):
                if stop_status is not None or poll_stop():
                    break
                first_state = extract_states[0]
                extract_cell_key = (
                    f"{safe_segment(first_state.algorithm_id)}__{first_state.seed}__watermark_extract"
                )
                extract_error = None
                watermark_method = None
                extract_dataset_ids = sorted({state.dataset_id for state in extract_states})
                try:
                    watermark_method = get_cached_watermark(
                        first_state.algorithm["method"],
                        first_state.algorithm_params,
                        request.device,
                    )
                    extract_groups: list[JsonDict] = []
                    for state in extract_states:
                        extract_groups.append(
                            {
                                "id": f"{state.cell_key}:positive",
                                "cell_key": state.cell_key,
                                "runtime_stage": "watermark_extract_positive",
                                "input_dir": state.attacked_dir,
                                "output_dir": state.extracted_dir,
                                "seed": state.seed,
                            }
                        )
                        extract_groups.append(
                            {
                                "id": f"{state.cell_key}:negative",
                                "cell_key": state.cell_key,
                                "runtime_stage": "watermark_extract_negative",
                                "input_dir": state.negative_attacked_dir,
                                "output_dir": state.negative_extracted_dir,
                                "seed": state.seed,
                            }
                        )
                    extract_results_by_id = extract_stage.run_groups(
                        algorithm=first_state.algorithm,
                        algorithm_params=first_state.algorithm_params,
                        watermark_method=watermark_method,
                        groups=extract_groups,
                    )
                    for state in extract_states:
                        state.positive_extract_results = extract_results_by_id.get(
                            f"{state.cell_key}:positive",
                            [],
                        )
                        state.negative_extract_results = extract_results_by_id.get(
                            f"{state.cell_key}:negative",
                            [],
                        )
                        _mark_state_operation_result(state, state.positive_extract_results)
                        _mark_state_operation_result(state, state.negative_extract_results)
                        positive_extract_status, positive_extract_error = _stage_status_and_error(
                            state.positive_extract_results,
                            fallback_error=state.error,
                            expected_count=len(state.positive_attack_results),
                        )
                        negative_extract_status, negative_extract_error = _stage_status_and_error(
                            state.negative_extract_results,
                            fallback_error=state.error,
                            expected_count=len(state.negative_attack_results),
                        )
                        positive_extract_manifest = state.variant_root / "extracted_positive" / "manifest.json"
                        negative_extract_manifest = state.variant_root / "extracted_negative" / "manifest.json"
                        _write_json(
                            positive_extract_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "inputDir": str(state.attacked_dir),
                                "outputDir": str(state.extracted_dir),
                                "sampleCount": len(state.positive_extract_results),
                                "status": positive_extract_status,
                                "error": positive_extract_error,
                            },
                        )
                        _write_json(
                            negative_extract_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "inputDir": str(state.negative_attacked_dir),
                                "outputDir": str(state.negative_extracted_dir),
                                "sampleCount": len(state.negative_extract_results),
                                "status": negative_extract_status,
                                "error": negative_extract_error,
                            },
                        )
                        positive_count = len(state.positive_extract_results)
                        negative_count = len(state.negative_extract_results)
                        extract_images_done += positive_count + negative_count
                        extract_positive_images_done += positive_count
                        extract_negative_images_done += negative_count
                        extract_cells_done += 1
                        state_writer.upsert_tree_path(
                            dataset_id=state.dataset_id,
                            algorithm_id=state.algorithm_id,
                            seed=state.seed,
                            attack_id=state.attack_id,
                            variant_key=state.variant_key,
                            refs={
                                "extractedPositive": {
                                    "manifest": str(positive_extract_manifest),
                                    "dir": str(state.extracted_dir),
                                },
                                "extractedNegative": {
                                    "manifest": str(negative_extract_manifest),
                                    "dir": str(state.negative_extracted_dir),
                                },
                            },
                        )
                        state_writer.phase_advance(
                            "watermark_extract",
                            current=extract_images_done,
                            current_item={
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "attackMethod": state.attack["method"],
                                "attackStrength": state.strength,
                                "attackParams": state.attack_params,
                                "variantKey": state.variant_key,
                                "positiveImages": positive_count,
                                "negativeImages": negative_count,
                                "remainingImages": max(0, extract_total_images - extract_images_done),
                            },
                            counters={
                                "imagesDone": extract_images_done,
                                "groupsDone": extract_groups_done,
                                "positiveImagesDone": extract_positive_images_done,
                                "negativeImagesDone": extract_negative_images_done,
                                "phaseCellsDone": extract_cells_done,
                                "phaseCellsTotal": expected_cells,
                            },
                            artifact_refs={"latestManifest": str(positive_extract_manifest)},
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    extract_error = error
                    for state in extract_states:
                        state.status = "failed"
                        state.error = error if state.error is None else f"{state.error}; {error}"
                        positive_extract_manifest = state.variant_root / "extracted_positive" / "manifest.json"
                        negative_extract_manifest = state.variant_root / "extracted_negative" / "manifest.json"
                        _write_json(
                            positive_extract_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "inputDir": str(state.attacked_dir),
                                "outputDir": str(state.extracted_dir),
                                "sampleCount": 0,
                                "status": "failed",
                                "error": error,
                            },
                        )
                        _write_json(
                            negative_extract_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "inputDir": str(state.negative_attacked_dir),
                                "outputDir": str(state.negative_extracted_dir),
                                "sampleCount": 0,
                                "status": "failed",
                                "error": error,
                            },
                        )
                        extract_cells_done += 1
                finally:
                    resource_manager.cleanup(
                        scope="watermark_extract",
                        reason="watermark_extract_finished",
                        cell_key=extract_cell_key,
                        release_watermarks=True,
                        metadata={
                            "datasetIds": extract_dataset_ids,
                            "algorithmId": first_state.algorithm_id,
                            "watermarkMethod": first_state.algorithm["method"],
                            "seed": first_state.seed,
                        },
                    )
                    extract_groups_done += 1
                    state_writer.phase_advance(
                        "watermark_extract",
                        current=extract_images_done,
                        counters={
                            "imagesDone": extract_images_done,
                            "groupsDone": extract_groups_done,
                            "positiveImagesDone": extract_positive_images_done,
                            "negativeImagesDone": extract_negative_images_done,
                            "phaseCellsDone": extract_cells_done,
                            "phaseCellsTotal": expected_cells,
                        },
                    )
            state_writer.phase_finish(
                "watermark_extract",
                status=stop_status or "succeeded",
                counters={
                    "imagesDone": extract_images_done,
                    "groupsDone": extract_groups_done,
                    "positiveImagesDone": extract_positive_images_done,
                    "negativeImagesDone": extract_negative_images_done,
                    "phaseCellsDone": extract_cells_done,
                    "phaseCellsTotal": expected_cells,
                },
            )

        # Phase 5: quality evaluation and final cell emission happen after all attack
        # and extraction outputs exist.
        if stop_status is None and all_states:
            quality_total_pairs = sum(len(state.copied_samples) * 2 for state in all_states)
            quality_pairs_done = 0
            quality_failed_units = 0
            quality_pair_cache_hits = 0
            state_writer.phase_start(
                "quality",
                total=quality_total_pairs,
                current_item={"workerMode": "async", "resultUnitCount": len(all_states)},
                counters={
                    "pairsDone": 0,
                    "failedUnits": 0,
                    "qualityPairCacheHits": 0,
                    "phaseCellsDone": len(result_units),
                    "phaseCellsTotal": expected_cells,
                },
            )
            quality_states: list[MaterializedCellState] = []
            for state in all_states:
                if stop_status is not None or poll_stop():
                    break
                operation_results = [
                    *state.positive_attack_results,
                    *state.negative_attack_results,
                    *state.positive_extract_results,
                    *state.negative_extract_results,
                ]
                operation_errors = [
                    result.error
                    for result in operation_results
                    if not getattr(result, "ok", False) and getattr(result, "error", None)
                ]
                if operation_errors:
                    state.status = "failed"
                    existing_error = state.error or ""
                    fresh_errors = [str(item) for item in operation_errors if str(item) not in existing_error]
                    if fresh_errors:
                        error = "; ".join(fresh_errors)
                        state.error = error if state.error is None else f"{state.error}; {error}"

                try:
                    detection_stage.append_results(
                        detection_records=state.detection_records,
                        cell_key=state.cell_key,
                        dataset_id=state.dataset_id,
                        algorithm_id=state.algorithm_id,
                        attack_id=state.attack_id,
                        attack_method=state.attack["method"],
                        attack_strength=state.strength,
                        seed=state.seed,
                        label=1,
                        input_root=state.attacked_dir,
                        results=state.positive_extract_results,
                    )
                    detection_stage.append_results(
                        detection_records=state.detection_records,
                        cell_key=state.cell_key,
                        dataset_id=state.dataset_id,
                        algorithm_id=state.algorithm_id,
                        attack_id=state.attack_id,
                        attack_method=state.attack["method"],
                        attack_strength=state.strength,
                        seed=state.seed,
                        label=0,
                        input_root=state.negative_attacked_dir,
                        results=state.negative_extract_results,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    state.status = "failed"
                    state.error = error if state.error is None else f"{state.error}; {error}"
                finally:
                    _write_json(state.detection_manifest_path, state.detection_records)
                quality_states.append(state)

            if quality_states:
                quality_workers = min(_async_quality_workers(), len(quality_states))

                def record_quality_result(state: MaterializedCellState, quality_result: DeferredQualityResult) -> bool:
                    nonlocal quality_failed_units, quality_pair_cache_hits, quality_pairs_done
                    quality_manifest = state.variant_root / "quality.json"
                    if quality_result.quality_records:
                        records = []
                        for record in quality_result.quality_records:
                            updated = dict(record)
                            updated["runId"] = request.run_id
                            records.append(updated)
                        _append_quality_records(paths, records)
                        _write_json(
                            quality_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "recordCount": len(records),
                                "records": records,
                                "status": "succeeded",
                            },
                        )
                    for profile in quality_result.runtime_profiles:
                        _record_runtime_profile(
                            paths,
                            run_id=request.run_id,
                            cell_key=str(profile["cell_key"]),
                            stage=str(profile["stage"]),
                            method=str(profile["method"]),
                            device=str(profile.get("device") or request.device),
                            elapsed_ms=float(profile["elapsed_ms"]),
                            image_paths=list(profile.get("image_paths") or []),
                            status=str(profile["status"]),
                            error=profile.get("error"),
                            metadata=dict(profile.get("metadata") or {}),
                        )
                    quality_pair_cache_hits += _quality_pair_cache_hit_count(quality_result.runtime_profiles)
                    if quality_result.error:
                        state.status = "failed"
                        state.error = (
                            quality_result.error
                            if state.error is None
                            else f"{state.error}; {quality_result.error}"
                        )
                        quality_failed_units += 1
                        _write_json(
                            quality_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "recordCount": 0,
                                "records": [],
                                "status": "failed",
                                "error": quality_result.error,
                            },
                        )
                    elif not quality_result.quality_records:
                        _write_json(
                            quality_manifest,
                            {
                                "runId": request.run_id,
                                "datasetId": state.dataset_id,
                                "algorithmId": state.algorithm_id,
                                "seed": state.seed,
                                "attackPresetId": state.attack_id,
                                "variantKey": state.variant_key,
                                "recordCount": 0,
                                "records": [],
                                "status": state.status,
                            },
                        )
                    quality_pair_count = len(quality_result.quality_records)
                    quality_pairs_done += quality_pair_count
                    state_writer.upsert_tree_path(
                        dataset_id=state.dataset_id,
                        algorithm_id=state.algorithm_id,
                        seed=state.seed,
                        attack_id=state.attack_id,
                        variant_key=state.variant_key,
                        refs={"quality": {"manifest": str(quality_manifest), "status": state.status}},
                    )
                    state_writer.phase_advance(
                        "quality",
                        current=quality_pairs_done,
                        current_item={
                            "datasetId": state.dataset_id,
                            "algorithmId": state.algorithm_id,
                            "seed": state.seed,
                            "attackPresetId": state.attack_id,
                            "attackMethod": state.attack["method"],
                            "attackStrength": state.strength,
                            "attackParams": state.attack_params,
                            "variantKey": state.variant_key,
                            "pairCount": quality_pair_count,
                            "remainingPairs": max(0, quality_total_pairs - quality_pairs_done),
                        },
                        counters={
                            "pairsDone": quality_pairs_done,
                            "failedUnits": quality_failed_units,
                            "workerCount": quality_workers,
                            "qualityPairCacheHits": quality_pair_cache_hits,
                            "resultUnitsDone": len(emitted_result_unit_keys),
                            "phaseCellsDone": len(emitted_result_unit_keys),
                            "phaseCellsTotal": expected_cells,
                        },
                        artifact_refs={"latestManifest": str(quality_manifest)},
                    )
                    result_unit, result_unit_manifest = emit_completed_variant(state)
                    state_writer.phase_advance(
                        "quality",
                        current=quality_pairs_done,
                        current_item={
                            "datasetId": state.dataset_id,
                            "algorithmId": state.algorithm_id,
                            "seed": state.seed,
                            "attackPresetId": state.attack_id,
                            "attackMethod": state.attack["method"],
                            "attackStrength": state.strength,
                            "attackParams": state.attack_params,
                            "variantKey": state.variant_key,
                            "status": state.status,
                        },
                        counters={
                            "pairsDone": quality_pairs_done,
                            "failedUnits": sum(1 for item in result_units if item.get("status") != "succeeded"),
                            "workerCount": quality_workers,
                            "qualityPairCacheHits": quality_pair_cache_hits,
                            "resultUnitsDone": len(result_units),
                            "phaseCellsDone": len(result_units),
                            "phaseCellsTotal": expected_cells,
                        },
                        artifact_refs={"latestResultUnit": str(result_unit_manifest)},
                    )
                    return result_unit is not None

                if quality_workers <= 1:
                    for state in quality_states:
                        if stop_status is not None or poll_stop():
                            break
                        try:
                            quality_result = quality_stage.compute_for_cell(
                                state,
                                quality_pair_cache=quality_pair_cache,
                            )
                        except Exception as exc:
                            quality_result = DeferredQualityResult(
                                quality_records=[],
                                runtime_profiles=[],
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        if record_quality_result(state, quality_result) and poll_stop():
                            break
                else:
                    pending_futures = {}
                    state_iter = iter(quality_states)
                    stop_submitting = False

                    def submit_next_quality(executor: ThreadPoolExecutor) -> bool:
                        nonlocal stop_submitting
                        if stop_submitting or stop_status is not None or poll_stop():
                            stop_submitting = True
                            return False
                        try:
                            next_state = next(state_iter)
                        except StopIteration:
                            return False
                        pending_futures[
                            executor.submit(
                                quality_stage.compute_for_cell,
                                next_state,
                                quality_pair_cache=quality_pair_cache,
                            )
                        ] = next_state
                        return True

                    with ThreadPoolExecutor(max_workers=quality_workers) as executor:
                        for _index in range(quality_workers):
                            if not submit_next_quality(executor):
                                break
                        while pending_futures:
                            done_futures, _pending = wait(pending_futures, return_when=FIRST_COMPLETED)
                            for future in done_futures:
                                state = pending_futures.pop(future)
                                try:
                                    quality_result = future.result()
                                except Exception as exc:
                                    quality_result = DeferredQualityResult(
                                        quality_records=[],
                                        runtime_profiles=[],
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                record_quality_result(state, quality_result)
                                if poll_stop():
                                    stop_submitting = True
                            while len(pending_futures) < quality_workers and not stop_submitting:
                                if not submit_next_quality(executor):
                                    break
            state_writer.phase_finish(
                "quality",
                status=stop_status or "succeeded",
                counters={
                    "pairsDone": quality_pairs_done,
                    "failedUnits": quality_failed_units,
                    "qualityPairCacheHits": quality_pair_cache_hits,
                    "resultUnitsDone": len(result_units),
                    "phaseCellsDone": len(result_units),
                    "phaseCellsTotal": expected_cells,
                },
            )

            state_writer.phase_start(
                "summary",
                total=expected_cells,
                current_item={"resultUnitCount": len(result_units) + len(all_states)},
                counters={
                    "resultUnitsDone": len(result_units),
                    "failedUnits": sum(1 for item in result_units if item.get("status") != "succeeded"),
                    "phaseCellsDone": len(result_units),
                    "phaseCellsTotal": expected_cells,
                },
            )
            result_units_done = len(result_units)
            if result_units_done:
                state_writer.phase_advance(
                    "summary",
                    current=result_units_done,
                    counters={
                        "resultUnitsDone": result_units_done,
                        "failedUnits": sum(1 for item in result_units if item.get("status") != "succeeded"),
                        "phaseCellsDone": result_units_done,
                        "phaseCellsTotal": expected_cells,
                    },
                )

        cleanup_run_once(reason="run_finished")

        if stop_status is None:
            stop_status = _stop_status_from_callback(should_cancel)

        result_units = _compact_result_units(result_units)
        if paths["resultUnits"].exists():
            result_units = _compact_result_units_file(paths["resultUnits"])
        failed = sum(1 for item in result_units if item["status"] != "succeeded")
        status = (
            stop_status
            if stop_status is not None
            else "succeeded"
            if failed == 0
            else "partially_failed"
            if failed < len(result_units)
            else "failed"
        )
        run_progress = _progress(len(result_units), expected_cells)
        invocation_elapsed_ms = (time.perf_counter() - started) * 1000
        elapsed_ms = _summary_elapsed_ms(
            previous_summary=previous_summary,
            invocation_elapsed_ms=invocation_elapsed_ms,
            runtime_profile_path=paths["runtimeProfile"],
            pending_cell_count=pending_cell_count,
            resume=request.resume,
        )
        summary = {
            "runId": request.run_id,
            "status": status,
            "selection": selection,
            "artifactRoot": str(run_root),
            "materializedRoot": str(materialized_root),
            "artifactFiles": {key: str(path) for key, path in paths.items()},
            "artifactTreePath": str(paths["artifactTree"]),
            "phaseStatePath": str(paths["phaseState"]),
            "runStatePath": str(paths["runState"]),
            "resultUnitCount": len(result_units),
            "failedResultUnits": failed,
            "skippedResultUnits": skipped_units,
            "progress": run_progress,
            "completedProgress": run_progress,
            "succeededProgress": _progress(len(result_units) - failed, expected_cells),
            "progressKind": "phaseOperations",
            "elapsedMs": elapsed_ms,
            "invocationElapsedMs": invocation_elapsed_ms,
            "resultUnits": result_units,
        }

        if result_units or all_states:
            state_writer.phase_finish(
                "summary",
                status="succeeded" if status in {"succeeded", "partially_failed"} else status,
                counters={
                    "resultUnitsDone": len(result_units),
                    "failedUnits": failed,
                    "skippedUnits": skipped_units,
                },
                artifact_refs={"summary": str(paths["runSummary"])},
            )
        if status in {"succeeded", "partially_failed"} and (
            expected_cells <= 0 or len(result_units) >= expected_cells
        ):
            _normalize_completed_phase_state(
                state_writer,
                selection=selection,
                expected_samples_by_dataset=expected_samples_by_dataset,
                result_units=result_units,
                expected_cells=expected_cells,
                failed_units=failed,
                skipped_units=skipped_units,
            )
        state_writer.set_status(status)
        _write_json(paths["runSummary"], summary)
        _write_run_status(
            paths,
            run_id=request.run_id,
            status=status,
            result_units=len(result_units),
            expected_result_units=expected_cells,
        )
        return summary
    finally:
        cleanup_run_once(reason="run_finally")
