from __future__ import annotations

import json
import hashlib
import math
import sys
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from evaluator.image_protocol import (
    CANONICAL_IMAGE_SIZE,
    CANONICAL_OUTPUT_POLICY,
    CANONICAL_PREPROCESS_POLICY,
    canonical_preprocess_image,
    quality_alignment_metadata,
)
from evaluator.attacks.runner import AttackJob, get_cached_attack, run_attack_dir_with_attack
from evaluator.watermarking.runner import (
    WatermarkEmbedJob,
    WatermarkExtractJob,
    get_cached_watermark,
    run_watermark_embed_dir_with_method,
    run_watermark_extract_dir_with_method,
)

from app.core.storage import safe_segment
from app.services.resources import (
    get_attack_catalog_item,
    get_dataset_by_id,
    get_watermark_catalog_item,
    iter_image_paths,
    scan_dataset_resources,
)
from app.services.scoring import compute_image_quality_pairs


JsonDict = dict[str, Any]
CellCallback = Callable[[JsonDict], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class LocalRunRequest:
    run_id: str
    selection: JsonDict
    resources_root: Path
    runs_root: Path
    device: str = "cpu"
    message: str = "1010101010101010"
    resume: bool = True


@dataclass(frozen=True)
class StagedSample:
    path: Path
    source_path: Path
    metadata: JsonDict


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
INTERMEDIATE_ARTIFACT_DIR = "_intermediates"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")


def _write_jsonl(path: Path, records: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    records: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _artifact_paths(run_root: Path) -> dict[str, Path]:
    return {
        "runPlan": run_root / "run_plan.json",
        "runStatus": run_root / "run_status.json",
        "sampleManifest": run_root / "sample_manifest.jsonl",
        "cellManifest": run_root / "cell_manifest.jsonl",
        "cellManifestLatest": run_root / "cell_manifest_latest.jsonl",
        "cellSummaryLatest": run_root / "cell_summary_latest.json",
        "imageQuality": run_root / "image_quality.jsonl",
        "imageWatermarkEmbed": run_root / "image_watermark_embed.jsonl",
        "imageAttack": run_root / "image_attack.jsonl",
        "imageDetection": run_root / "image_detection.jsonl",
        "imageDetectionLatest": run_root / "image_detection_latest.jsonl",
        "runtimeProfile": run_root / "runtime_profile.jsonl",
        "stageEvents": run_root / "stage_events.jsonl",
        "runSummary": run_root / "run_summary.json",
    }


def _stage_event(paths: dict[str, Path], run_id: str, stage: str, status: str, **payload: Any) -> None:
    _append_jsonl(
        paths["stageEvents"],
        {
            "runId": run_id,
            "stage": stage,
            "status": status,
            "timestamp": _utc_timestamp(),
            **payload,
        },
    )


def _write_run_status(
    paths: dict[str, Path],
    *,
    run_id: str,
    status: str,
    completed_cells: int,
    expected_cells: int,
    error: str | None = None,
) -> None:
    _write_json(
        paths["runStatus"],
        {
            "runId": run_id,
            "status": status,
            "completedCells": completed_cells,
            "expectedCells": expected_cells,
            "progress": _progress(completed_cells, expected_cells),
            "completedProgress": _progress(completed_cells, expected_cells),
            "progressKind": "completedCells",
            "error": error,
            "updatedAt": _utc_timestamp(),
        },
    )


def _completed_cell_rows(cell_manifest_path: Path) -> dict[str, JsonDict]:
    completed: dict[str, JsonDict] = {}
    for record in _read_jsonl(cell_manifest_path):
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str) and record.get("status") == "succeeded":
            completed[cell_key] = record
    return completed


def _cell_attempt_counts(cell_manifest_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in _read_jsonl(cell_manifest_path):
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str):
            counts[cell_key] = counts.get(cell_key, 0) + 1
    return counts


def _latest_cell_rows(cell_manifest_path: Path) -> list[JsonDict]:
    latest: dict[str, JsonDict] = {}
    attempt_counts: dict[str, int] = {}
    for record in _read_jsonl(cell_manifest_path):
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str):
            attempt_counts[cell_key] = attempt_counts.get(cell_key, 0) + 1
            enriched = dict(record)
            enriched.setdefault("attemptIndex", attempt_counts[cell_key])
            enriched.setdefault("supersedesPreviousAttempt", attempt_counts[cell_key] > 1)
            latest[cell_key] = enriched
    return list(latest.values())


def _json_record_has_intermediate_artifact(record: JsonDict) -> bool:
    for key in ("inputPath", "sampleId", "referencePath", "targetPath"):
        value = record.get(key)
        if isinstance(value, str) and INTERMEDIATE_ARTIFACT_DIR in Path(value).parts:
            return True
    return False


def _read_json_array(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _latest_image_detection_rows(latest_cells: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for cell in latest_cells:
        manifest_path = cell.get("manifestPath")
        if not isinstance(manifest_path, str):
            continue
        for record in _read_json_array(Path(manifest_path)):
            if not _json_record_has_intermediate_artifact(record):
                rows.append(record)
    return rows


def _write_latest_cell_artifacts(paths: dict[str, Path], *, run_id: str, expected_cells: int) -> None:
    latest_cells = _latest_cell_rows(paths["cellManifest"])
    status_counts: dict[str, int] = {}
    for cell in latest_cells:
        status = str(cell.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    attempted_cells = len(latest_cells)
    succeeded_cells = status_counts.get("succeeded", 0)
    failed_cells = status_counts.get("failed", 0)

    _write_jsonl(paths["cellManifestLatest"], latest_cells)
    _write_jsonl(paths["imageDetectionLatest"], _latest_image_detection_rows(latest_cells))
    _write_json(
        paths["cellSummaryLatest"],
        {
            "runId": run_id,
            "cellCount": attempted_cells,
            "attemptedCells": attempted_cells,
            "succeededCells": succeeded_cells,
            "failedCells": failed_cells,
            "expectedCells": expected_cells,
            "progress": _progress(attempted_cells, expected_cells),
            "completedProgress": _progress(attempted_cells, expected_cells),
            "progressKind": "completedCells",
            "attemptedProgress": _progress(attempted_cells, expected_cells),
            "succeededProgress": _progress(succeeded_cells, expected_cells),
            "statusCounts": status_counts,
            "updatedAt": _utc_timestamp(),
        },
    )


def _image_sample_id(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).with_suffix("").as_posix()
    except ValueError:
        return path.with_suffix("").name


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


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
    _append_jsonl(
        paths["runtimeProfile"],
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
            "peakMemoryMB": _gpu_peak_memory_mb(device) or _process_peak_memory_mb(),
            "error": error,
            "metadata": metadata or {},
            "timestamp": _utc_timestamp(),
        },
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


def _list_image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not _is_intermediate_artifact(path)
    ]


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
) -> list[JsonDict]:
    pairs = _pair_images(reference_dir, target_dir)
    started = time.perf_counter()
    target_paths = [target_path for _reference_path, target_path in pairs]
    try:
        metrics_by_pair = compute_image_quality_pairs(pairs)
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
        metadata={"scope": scope},
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
    return {
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


def _record_reused_quality_records(
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
    source_records: list[JsonDict],
    source_scope: str,
    target_dir: Path,
    device: str = "cpu",
    reuse_policy: str = "identity_attack",
) -> list[JsonDict]:
    started = time.perf_counter()
    records = _retarget_quality_records(
        source_records,
        run_id=run_id,
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
    _append_quality_records(paths, records)
    _record_runtime_profile(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        stage="quality",
        method="image_quality",
        device=device,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        image_paths=_list_image_files(target_dir),
        status="reused",
        metadata={"scope": scope, "sourceScope": source_scope, "reusePolicy": reuse_policy},
    )
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


def _record_identity_quality_pairs(
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
) -> list[JsonDict]:
    started = time.perf_counter()
    metrics = _identity_quality_metrics()
    pairs = _pair_images(reference_dir, target_dir)
    records = [
        {
            **_quality_record(
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
            ),
            "qualityComputation": "reused",
            "sourceScope": "identity_noop",
            "reusePolicy": "identity_noop_perfect",
        }
        for reference_path, target_path in pairs
    ]
    _append_quality_records(paths, records)
    _record_runtime_profile(
        paths,
        run_id=run_id,
        cell_key=cell_key,
        stage="quality",
        method="image_quality",
        device=device,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        image_paths=[target_path for _reference_path, target_path in pairs],
        status="reused",
        metadata={"scope": scope, "sourceScope": "identity_noop", "reusePolicy": "identity_noop_perfect"},
    )
    return records


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
    return {
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
            },
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
            },
        )


def _ensure_list(value: Any, fallback: list[Any]) -> list[Any]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return fallback


def _normalize_strength_overrides(value: Any, valid_attack_ids: list[str]) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        return {}

    valid_ids = set(valid_attack_ids)
    normalized: dict[str, list[float]] = {}
    for attack_id, strengths in value.items():
        normalized_attack_id = str(attack_id)
        if normalized_attack_id not in valid_ids:
            continue
        strength_values: list[float] = []
        for strength in _ensure_list(strengths, []):
            try:
                parsed = float(strength)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                strength_values.append(parsed)
        deduped = sorted(set(strength_values))
        if deduped:
            normalized[normalized_attack_id] = deduped
    return normalized


def _normalize_param_overrides(value: Any, valid_attack_ids: list[str]) -> dict[str, list[JsonDict]]:
    if not isinstance(value, dict):
        return {}

    valid_ids = set(valid_attack_ids)
    normalized: dict[str, list[JsonDict]] = {}
    for attack_id, variants in value.items():
        normalized_attack_id = str(attack_id)
        if normalized_attack_id not in valid_ids:
            continue
        cleaned_variants: list[JsonDict] = []
        seen: set[str] = set()
        for variant in _ensure_list(variants, []):
            if not isinstance(variant, dict):
                continue
            cleaned: JsonDict = {}
            for key, raw_value in variant.items():
                if not isinstance(key, str) or raw_value is None:
                    continue
                if isinstance(raw_value, (str, bool)):
                    cleaned[key] = raw_value
                elif isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
                    cleaned[key] = raw_value
            if not cleaned:
                continue
            marker = json.dumps(cleaned, sort_keys=True, ensure_ascii=True)
            if marker in seen:
                continue
            seen.add(marker)
            cleaned_variants.append(cleaned)
        if cleaned_variants:
            normalized[normalized_attack_id] = cleaned_variants
    return normalized


def normalize_selection(selection: JsonDict, resources_root: Path) -> JsonDict:
    datasets = scan_dataset_resources(resources_root)
    default_dataset_ids = [datasets[0].id] if datasets else []
    dataset_ids = _ensure_list(selection.get("datasetIds"), default_dataset_ids)
    algorithm_ids = _ensure_list(selection.get("algorithmIds"), ["alg-invisible-watermark-dwtdct"])
    attack_ids = _ensure_list(selection.get("attackPresetIds"), ["atk-identity", "atk-jpeg"])
    normalized_attack_ids = [str(value) for value in attack_ids]
    seeds = [int(seed) for seed in _ensure_list(selection.get("seeds"), [42])]
    max_samples = int(selection.get("maxSamples") or 1)

    return {
        "datasetIds": [str(value) for value in dataset_ids],
        "algorithmIds": [str(value) for value in algorithm_ids],
        "attackPresetIds": normalized_attack_ids,
        "attackStrengthOverrides": _normalize_strength_overrides(
            selection.get("attackStrengthOverrides"), normalized_attack_ids
        ),
        "attackParamOverrides": _normalize_param_overrides(
            selection.get("attackParamOverrides"), normalized_attack_ids
        ),
        "seeds": seeds,
        "maxSamples": max(1, max_samples),
    }


def _strengths_for_attack(selection: JsonDict, attack_id: str, attack: JsonDict) -> list[float]:
    overrides = selection.get("attackStrengthOverrides") or {}
    override_strengths = overrides.get(attack_id) if isinstance(overrides, dict) else None
    if isinstance(override_strengths, list) and override_strengths:
        return [float(strength) for strength in override_strengths]
    return [float(strength) for strength in (attack["strengths"] or [0.0])]


def _param_overrides_for_attack(selection: JsonDict, attack_id: str) -> list[JsonDict]:
    overrides = selection.get("attackParamOverrides") or {}
    override_params = overrides.get(attack_id) if isinstance(overrides, dict) else None
    if not isinstance(override_params, list):
        return []
    return [dict(params) for params in override_params if isinstance(params, dict)]


def _variant_strength(attack: JsonDict, params: JsonDict, fallback: float = 0.0) -> float:
    strength_param = attack.get("strengthParam")
    if strength_param and str(strength_param) in params:
        try:
            return float(params[str(strength_param)])
        except (TypeError, ValueError):
            return fallback
    return fallback


def _params_digest(params: JsonDict) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _attack_variants_for_attack(selection: JsonDict, attack_id: str, attack: JsonDict) -> list[tuple[float, JsonDict, str]]:
    param_overrides = _param_overrides_for_attack(selection, attack_id)
    if param_overrides:
        variants: list[tuple[float, JsonDict, str]] = []
        for index, override_params in enumerate(param_overrides):
            strength = _variant_strength(attack, override_params, fallback=0.0)
            params = _attack_params(attack, strength)
            params.update(override_params)
            variants.append((float(strength), params, _params_digest({"index": index, **params})))
        return variants

    return [
        (float(strength), _attack_params(attack, float(strength)), f"{float(strength):g}")
        for strength in _strengths_for_attack(selection, attack_id, attack)
    ]


def estimate_selection(selection: JsonDict, resources_root: Path) -> JsonDict:
    normalized = normalize_selection(selection, resources_root)
    sample_count = 0
    for dataset_id in normalized["datasetIds"]:
        try:
            dataset = get_dataset_by_id(resources_root, dataset_id)
        except KeyError:
            continue
        sample_count += min(dataset.sample_count, normalized["maxSamples"])

    strength_count = 0
    for attack_id in normalized["attackPresetIds"]:
        try:
            attack = get_attack_catalog_item(attack_id)
        except KeyError:
            continue
        strength_count += max(1, len(_attack_variants_for_attack(normalized, attack_id, attack)))

    cell_count = (
        len(normalized["datasetIds"])
        * len(normalized["algorithmIds"])
        * max(1, strength_count)
        * len(normalized["seeds"])
    )
    return {
        "selection": normalized,
        "cellCount": cell_count,
        "sampleCount": sample_count,
        "imageOperationCount": cell_count * max(1, sample_count),
    }


def _canonical_target_path(output_dir: Path, relative: Path, index: int) -> Path:
    target = (output_dir / relative).with_suffix(".png")
    if not target.exists():
        return target
    return target.with_name(f"{target.stem}_{index:04d}.png")


def _copy_samples(dataset_path: Path, output_dir: Path, max_samples: int) -> list[StagedSample]:
    sample_paths = iter_image_paths(dataset_path)[:max_samples]
    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[StagedSample] = []

    for index, sample_path in enumerate(sample_paths, start=1):
        try:
            relative = sample_path.relative_to(dataset_path)
        except ValueError:
            relative = Path(f"sample_{index:04d}{sample_path.suffix.lower()}")
        if relative.name.startswith("."):
            relative = Path(f"sample_{index:04d}{sample_path.suffix.lower()}")
        target = _canonical_target_path(output_dir, relative, index)
        metadata = canonical_preprocess_image(sample_path, target)
        staged.append(StagedSample(path=target, source_path=sample_path, metadata=metadata))

    return staged


def _cell_key(
    dataset_id: str,
    algorithm_id: str,
    attack_id: str,
    strength: float,
    seed: int,
    variant_key: str | None = None,
) -> str:
    suffix = f"__{variant_key}" if variant_key else ""
    raw = f"{dataset_id}__{algorithm_id}__{attack_id}__{strength:g}__{seed}{suffix}"
    return safe_segment(raw)


def _attack_params(attack: JsonDict, strength: float) -> JsonDict:
    params = dict(attack.get("params") or {})
    strength_param = attack.get("strengthParam")
    if strength_param:
        value: float | int = float(strength)
        if str(strength_param) in {"scale", "xy"} and float(strength).is_integer():
            value = int(strength)
        params[str(strength_param)] = value
    return params


def _progress(completed_cells: int, total_cells: int) -> int:
    if total_cells <= 0:
        return 0
    return int(round((completed_cells / total_cells) * 100))


def run_local_experiment(
    request: LocalRunRequest,
    on_cell: CellCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> JsonDict:
    selection = normalize_selection(request.selection, request.resources_root)
    if not selection["datasetIds"]:
        raise ValueError("No datasets found under resources/datasets")
    run_root = request.runs_root / safe_segment(request.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(run_root)
    existing_completed = _completed_cell_rows(paths["cellManifest"]) if request.resume else {}
    attempt_counts = _cell_attempt_counts(paths["cellManifest"]) if request.resume else {}
    cells: list[JsonDict] = list(existing_completed.values())
    started = time.perf_counter()
    estimate = estimate_selection(selection, request.resources_root)
    expected_cells = int(estimate["cellCount"])
    cancelled = False
    skipped_cells = 0
    negative_attack_cache: dict[str, dict[str, Any]] = {}

    attack_plan: list[JsonDict] = []
    for attack_id in selection["attackPresetIds"]:
        attack = get_attack_catalog_item(attack_id)
        attack_plan.append(
            {
                "id": attack_id,
                "method": attack["method"],
                "variants": [
                    {
                        "strength": float(strength),
                        "params": params,
                        "variantKey": variant_key,
                    }
                    for strength, params, variant_key in _attack_variants_for_attack(selection, attack_id, attack)
                ],
            }
        )

    _write_json(
        paths["runPlan"],
        {
            "runId": request.run_id,
            "selection": selection,
            "expectedCells": expected_cells,
            "artifactFiles": {key: str(path) for key, path in paths.items()},
            "datasets": selection["datasetIds"],
            "watermarkAlgorithms": selection["algorithmIds"],
            "attacks": attack_plan,
            "imageSizeProtocol": {
                "canonicalSize": list(CANONICAL_IMAGE_SIZE),
                "preprocessPolicy": CANONICAL_PREPROCESS_POLICY,
                "watermarkOutputPolicy": CANONICAL_OUTPUT_POLICY,
                "qualityAlignmentPolicy": "resize target to reference only when sizes differ",
            },
            "resume": request.resume,
            "createdAt": _utc_timestamp(),
        },
    )
    _write_run_status(
        paths,
        run_id=request.run_id,
        status="running",
        completed_cells=len(cells),
        expected_cells=expected_cells,
    )
    _write_latest_cell_artifacts(paths, run_id=request.run_id, expected_cells=expected_cells)
    _stage_event(
        paths,
        request.run_id,
        "run",
        "started",
        expectedCells=expected_cells,
        resumedCells=len(cells),
    )

    existing_sample_keys = {
        (str(record.get("datasetId")), str(record.get("sampleId")))
        for record in _read_jsonl(paths["sampleManifest"])
        if record.get("datasetId") is not None and record.get("sampleId") is not None
    }

    def emit_cell(cell: JsonDict) -> None:
        record = dict(cell)
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str):
            attempt_counts[cell_key] = attempt_counts.get(cell_key, 0) + 1
            record["attemptIndex"] = attempt_counts[cell_key]
            record["supersedesPreviousAttempt"] = attempt_counts[cell_key] > 1
        record["completedAt"] = _utc_timestamp()

        cells.append(record)
        _append_jsonl(paths["cellManifest"], record)
        _write_latest_cell_artifacts(paths, run_id=request.run_id, expected_cells=expected_cells)
        _write_run_status(
            paths,
            run_id=request.run_id,
            status="running",
            completed_cells=len(cells),
            expected_cells=expected_cells,
        )
        if on_cell is not None:
            on_cell(record)

    for dataset_id in selection["datasetIds"]:
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        dataset = get_dataset_by_id(request.resources_root, dataset_id)
        cell_input_dir = run_root / "staging" / "samples" / safe_segment(dataset_id)
        _stage_event(paths, request.run_id, "dataset", "started", datasetId=dataset_id)
        try:
            staged_samples = _copy_samples(dataset.path, cell_input_dir, selection["maxSamples"])
            copied_samples = [sample.path for sample in staged_samples]
            if not staged_samples:
                raise ValueError(f"Dataset has no supported image files: {dataset.path}")

            for staged_sample in staged_samples:
                sample_path = staged_sample.path
                sample_id = _image_sample_id(sample_path, cell_input_dir)
                sample_key = (dataset_id, sample_id)
                if sample_key in existing_sample_keys:
                    continue
                original_size = staged_sample.metadata.get("originalSize") or [None, None]
                canonical_size = staged_sample.metadata.get("canonicalSize") or [None, None]
                _append_jsonl(
                    paths["sampleManifest"],
                    {
                        "runId": request.run_id,
                        "datasetId": dataset_id,
                        "sampleId": sample_id,
                        "sourcePath": str(staged_sample.source_path),
                        "width": original_size[0],
                        "height": original_size[1],
                        "originalSize": original_size,
                        "canonicalSize": canonical_size,
                        "canonicalWidth": canonical_size[0],
                        "canonicalHeight": canonical_size[1],
                        "preprocessPolicy": staged_sample.metadata.get("preprocessPolicy"),
                        "cropPolicy": staged_sample.metadata.get("cropPolicy"),
                        "resizedContentSize": staged_sample.metadata.get("resizedContentSize"),
                        "cropBox": staged_sample.metadata.get("cropBox"),
                        "cropMargins": staged_sample.metadata.get("cropMargins"),
                        "padding": staged_sample.metadata.get("padding"),
                        "scale": staged_sample.metadata.get("scale"),
                        "paddingColor": staged_sample.metadata.get("paddingColor"),
                        "timestamp": _utc_timestamp(),
                    },
                )
                existing_sample_keys.add(sample_key)

            for algorithm_id in selection["algorithmIds"]:
                if cancelled:
                    break
                algorithm = get_watermark_catalog_item(algorithm_id)
                algorithm_params = dict(algorithm.get("params") or {})
                for seed in selection["seeds"]:
                    if should_cancel is not None and should_cancel():
                        cancelled = True
                        break

                    pending_variants: list[JsonDict] = []
                    for attack_id in selection["attackPresetIds"]:
                        attack = get_attack_catalog_item(attack_id)
                        for strength, attack_params, variant_key in _attack_variants_for_attack(
                            selection, attack_id, attack
                        ):
                            cell_key = _cell_key(
                                dataset_id,
                                algorithm_id,
                                attack_id,
                                float(strength),
                                int(seed),
                                variant_key,
                            )
                            if cell_key in existing_completed:
                                skipped_cells += 1
                                _stage_event(
                                    paths,
                                    request.run_id,
                                    "cell",
                                    "skipped",
                                    cellKey=cell_key,
                                    datasetId=dataset_id,
                                    algorithmId=algorithm_id,
                                    attackPresetId=attack_id,
                                    attackStrength=float(strength),
                                    attackParams=attack_params,
                                    reason="resume_completed",
                                )
                                continue
                            pending_variants.append(
                                {
                                    "cellKey": cell_key,
                                    "attackId": attack_id,
                                    "attack": attack,
                                    "strength": float(strength),
                                    "attackParams": attack_params,
                                    "variantKey": variant_key,
                                }
                            )

                    if not pending_variants:
                        continue

                    embed_key = _cell_key(dataset_id, algorithm_id, "watermark_embed", 0.0, int(seed))
                    watermarked_dir = (
                        run_root
                        / "staging"
                        / "watermarked"
                        / safe_segment(dataset_id)
                        / safe_segment(algorithm_id)
                        / safe_segment(str(seed))
                    )
                    embed_results: list[Any] = []
                    embed_error = None
                    embed_elapsed_ms = 0.0

                    try:
                        _stage_event(
                            paths,
                            request.run_id,
                            "watermark_embed",
                            "started",
                            cellKey=embed_key,
                            datasetId=dataset_id,
                            algorithmId=algorithm_id,
                            seed=int(seed),
                        )
                        _reset_gpu_peak(request.device)
                        embed_started = time.perf_counter()
                        watermark_method = get_cached_watermark(
                            algorithm["method"],
                            algorithm_params,
                            request.device,
                        )
                        embed_results = run_watermark_embed_dir_with_method(
                            WatermarkEmbedJob(
                                run_id=request.run_id,
                                method_name=algorithm["method"],
                                params=algorithm_params,
                                input_dir=cell_input_dir,
                                output_dir=watermarked_dir,
                                message=request.message,
                                device=request.device,
                                seed=int(seed),
                            ),
                            watermark_method,
                        )
                        embed_elapsed_ms = (time.perf_counter() - embed_started) * 1000
                        _record_watermark_embed_results(
                            paths,
                            run_id=request.run_id,
                            cell_key=embed_key,
                            dataset_id=dataset_id,
                            algorithm_id=algorithm_id,
                            watermark_method=algorithm["method"],
                            seed=int(seed),
                            input_root=cell_input_dir,
                            results=embed_results,
                        )
                        embed_errors = [result.error for result in embed_results if getattr(result, "error", None)]
                        if not all(result.ok for result in embed_results):
                            embed_error = "; ".join(embed_errors) or "one or more watermark embed operations failed"
                        _record_runtime_profile(
                            paths,
                            run_id=request.run_id,
                            cell_key=embed_key,
                            stage="watermark_embed",
                            method=algorithm["method"],
                            device=request.device,
                            elapsed_ms=embed_elapsed_ms,
                            image_paths=copied_samples,
                            status="failed" if embed_error else "succeeded",
                            error=embed_error,
                        )
                        if embed_error:
                            raise RuntimeError(embed_error)
                        embed_quality_records = _record_quality_pairs(
                            paths,
                            run_id=request.run_id,
                            cell_key=embed_key,
                            scope="original_vs_watermarked",
                            dataset_id=dataset_id,
                            algorithm_id=algorithm_id,
                            attack_id=None,
                            attack_method=None,
                            attack_strength=None,
                            seed=int(seed),
                            reference_dir=cell_input_dir,
                            target_dir=watermarked_dir,
                            device=request.device,
                        )
                        _stage_event(
                            paths,
                            request.run_id,
                            "watermark_embed",
                            "succeeded",
                            cellKey=embed_key,
                            elapsedMs=embed_elapsed_ms,
                        )
                    except Exception as exc:
                        embed_error = f"{type(exc).__name__}: {exc}"
                        _stage_event(
                            paths,
                            request.run_id,
                            "watermark_embed",
                            "failed",
                            cellKey=embed_key,
                            error=embed_error,
                        )
                        for variant in pending_variants:
                            failed_cell_root = run_root / "cells" / str(variant["cellKey"])
                            failed_detection_manifest = failed_cell_root / "cell_detection_manifest.json"
                            _write_json(failed_detection_manifest, [])
                            emit_cell(
                                {
                                    "runId": request.run_id,
                                    "cellKey": variant["cellKey"],
                                    "status": "failed",
                                    "datasetId": dataset_id,
                                    "algorithmId": algorithm_id,
                                    "watermarkMethod": algorithm["method"],
                                    "attackPresetId": variant["attackId"],
                                    "attackMethod": variant["attack"]["method"],
                                    "attackStrength": variant["strength"],
                                    "seed": int(seed),
                                    "sampleCount": len(copied_samples),
                                    "attackParams": variant["attackParams"],
                                    "manifestPath": str(failed_detection_manifest),
                                    "outputDir": str(failed_cell_root),
                                    "error": embed_error,
                                    "elapsedMs": embed_elapsed_ms,
                                }
                            )
                            _stage_event(
                                paths,
                                request.run_id,
                                "cell",
                                "failed",
                                cellKey=variant["cellKey"],
                                datasetId=dataset_id,
                                algorithmId=algorithm_id,
                                attackPresetId=variant["attackId"],
                                attackStrength=variant["strength"],
                                attackParams=variant["attackParams"],
                                error=embed_error,
                            )
                        shutil.rmtree(watermarked_dir, ignore_errors=True)
                        continue

                    try:
                        for variant in pending_variants:
                            if should_cancel is not None and should_cancel():
                                cancelled = True
                                break

                            attack_id = str(variant["attackId"])
                            attack = variant["attack"]
                            strength = float(variant["strength"])
                            attack_params = dict(variant["attackParams"])
                            cell_key = str(variant["cellKey"])
                            cell_root = run_root / "cells" / cell_key
                            attacked_dir = cell_root / "attacked"
                            extracted_dir = cell_root / "extracted"
                            negative_attack_key = _cell_key(
                                dataset_id,
                                "negative_control",
                                attack_id,
                                strength,
                                int(seed),
                                str(variant.get("variantKey") or ""),
                            )
                            negative_attacked_dir = (
                                run_root
                                / "staging"
                                / "negative_attacked"
                                / safe_segment(dataset_id)
                                / safe_segment(negative_attack_key)
                            )
                            negative_extracted_dir = cell_root / "negative_extracted"
                            cell_detection_manifest_path = cell_root / "cell_detection_manifest.json"
                            detection_records: list[JsonDict] = []
                            status = "succeeded"
                            error = None
                            cell_started = time.perf_counter()
                            elapsed_ms = 0.0

                            _stage_event(
                                paths,
                                request.run_id,
                                "cell",
                                "started",
                                cellKey=cell_key,
                                datasetId=dataset_id,
                                algorithmId=algorithm_id,
                                attackPresetId=attack_id,
                                attackStrength=strength,
                                attackParams=attack_params,
                            )

                            try:
                                attack_instance = get_cached_attack(attack["method"], attack_params, request.device)

                                _reset_gpu_peak(request.device)
                                attack_started = time.perf_counter()
                                attack_results = run_attack_dir_with_attack(
                                    AttackJob(
                                        run_id=request.run_id,
                                        attack_name=attack["method"],
                                        params=attack_params,
                                        input_dir=watermarked_dir,
                                        output_dir=attacked_dir,
                                        device=request.device,
                                        seed=int(seed),
                                    ),
                                    attack_instance,
                                )
                                attack_elapsed_ms = (time.perf_counter() - attack_started) * 1000
                                _record_attack_results(
                                    paths,
                                    run_id=request.run_id,
                                    cell_key=cell_key,
                                    stage="attack",
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    attack_id=attack_id,
                                    attack_method=attack["method"],
                                    attack_strength=strength,
                                    attack_params=attack_params,
                                    seed=int(seed),
                                    label=1,
                                    input_root=watermarked_dir,
                                    results=attack_results,
                                )
                                attack_error = "; ".join(
                                    result.error for result in attack_results if getattr(result, "error", None)
                                )
                                _record_runtime_profile(
                                    paths,
                                    run_id=request.run_id,
                                    cell_key=cell_key,
                                    stage="attack",
                                    method=attack["method"],
                                    device=request.device,
                                    elapsed_ms=attack_elapsed_ms,
                                    image_paths=_list_image_files(watermarked_dir),
                                    status="failed" if attack_error else "succeeded",
                                    error=attack_error or None,
                                    metadata={"attackParams": attack_params},
                                )

                                _reset_gpu_peak(request.device)
                                extract_started = time.perf_counter()
                                extract_results = run_watermark_extract_dir_with_method(
                                    WatermarkExtractJob(
                                        run_id=request.run_id,
                                        method_name=algorithm["method"],
                                        params=algorithm_params,
                                        input_dir=attacked_dir,
                                        output_dir=extracted_dir,
                                        message=request.message,
                                        device=request.device,
                                        seed=int(seed),
                                    ),
                                    watermark_method,
                                )
                                extract_elapsed_ms = (time.perf_counter() - extract_started) * 1000
                                extract_error = "; ".join(
                                    result.error for result in extract_results if getattr(result, "error", None)
                                )
                                _record_runtime_profile(
                                    paths,
                                    run_id=request.run_id,
                                    cell_key=cell_key,
                                    stage="watermark_extract_positive",
                                    method=algorithm["method"],
                                    device=request.device,
                                    elapsed_ms=extract_elapsed_ms,
                                    image_paths=_list_image_files(attacked_dir),
                                    status="failed" if extract_error else "succeeded",
                                    error=extract_error or None,
                                )

                                cached_negative_attack = negative_attack_cache.get(negative_attack_key)
                                if cached_negative_attack is not None:
                                    negative_attack_results = cached_negative_attack["results"]
                                    negative_attack_elapsed_ms = 0.0
                                    negative_attack_error = cached_negative_attack.get("error")
                                    _record_attack_results(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        stage="attack_negative_control",
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        attack_params=attack_params,
                                        seed=int(seed),
                                        label=0,
                                        input_root=cell_input_dir,
                                        results=negative_attack_results,
                                        cache_hit=True,
                                    )
                                    _record_runtime_profile(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        stage="attack_negative_control",
                                        method=attack["method"],
                                        device=request.device,
                                        elapsed_ms=negative_attack_elapsed_ms,
                                        image_paths=copied_samples,
                                        status="reused" if not negative_attack_error else "failed",
                                        error=negative_attack_error or None,
                                        metadata={
                                            "attackParams": attack_params,
                                            "cacheKey": negative_attack_key,
                                            "cacheHit": True,
                                        },
                                    )
                                else:
                                    _reset_gpu_peak(request.device)
                                    negative_attack_started = time.perf_counter()
                                    negative_attack_results = run_attack_dir_with_attack(
                                        AttackJob(
                                            run_id=request.run_id,
                                            attack_name=attack["method"],
                                            params=attack_params,
                                            input_dir=cell_input_dir,
                                            output_dir=negative_attacked_dir,
                                            device=request.device,
                                            seed=int(seed),
                                        ),
                                        attack_instance,
                                    )
                                    negative_attack_elapsed_ms = (time.perf_counter() - negative_attack_started) * 1000
                                    _record_attack_results(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        stage="attack_negative_control",
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        attack_params=attack_params,
                                        seed=int(seed),
                                        label=0,
                                        input_root=cell_input_dir,
                                        results=negative_attack_results,
                                        cache_hit=False,
                                    )
                                    negative_attack_error = "; ".join(
                                        result.error
                                        for result in negative_attack_results
                                        if getattr(result, "error", None)
                                    )
                                    if not negative_attack_error:
                                        negative_attack_cache[negative_attack_key] = {
                                            "outputDir": negative_attacked_dir,
                                            "results": negative_attack_results,
                                            "error": None,
                                        }
                                    _record_runtime_profile(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        stage="attack_negative_control",
                                        method=attack["method"],
                                        device=request.device,
                                        elapsed_ms=negative_attack_elapsed_ms,
                                        image_paths=copied_samples,
                                        status="failed" if negative_attack_error else "succeeded",
                                        error=negative_attack_error or None,
                                        metadata={
                                            "attackParams": attack_params,
                                            "cacheKey": negative_attack_key,
                                            "cacheHit": False,
                                        },
                                    )

                                _reset_gpu_peak(request.device)
                                negative_extract_started = time.perf_counter()
                                negative_extract_results = run_watermark_extract_dir_with_method(
                                    WatermarkExtractJob(
                                        run_id=request.run_id,
                                        method_name=algorithm["method"],
                                        params=algorithm_params,
                                        input_dir=negative_attacked_dir,
                                        output_dir=negative_extracted_dir,
                                        message=request.message,
                                        device=request.device,
                                        seed=int(seed),
                                    ),
                                    watermark_method,
                                )
                                negative_extract_elapsed_ms = (time.perf_counter() - negative_extract_started) * 1000
                                negative_extract_error = "; ".join(
                                    result.error
                                    for result in negative_extract_results
                                    if getattr(result, "error", None)
                                )
                                _record_runtime_profile(
                                    paths,
                                    run_id=request.run_id,
                                    cell_key=cell_key,
                                    stage="watermark_extract_negative",
                                    method=algorithm["method"],
                                    device=request.device,
                                    elapsed_ms=negative_extract_elapsed_ms,
                                    image_paths=_list_image_files(negative_attacked_dir),
                                    status="failed" if negative_extract_error else "succeeded",
                                    error=negative_extract_error or None,
                                )

                                operation_results = [
                                    *attack_results,
                                    *extract_results,
                                    *negative_attack_results,
                                    *negative_extract_results,
                                ]
                                if not all(result.ok for result in operation_results):
                                    status = "failed"
                                    errors = [
                                        result.error
                                        for result in operation_results
                                        if getattr(result, "error", None)
                                    ]
                                    error = "; ".join(errors) or "one or more image operations failed"

                                if str(attack["method"]).lower() == "identity":
                                    _record_reused_quality_records(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        scope="original_vs_attacked_watermarked",
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        seed=int(seed),
                                        source_records=embed_quality_records,
                                        source_scope="original_vs_watermarked",
                                        target_dir=attacked_dir,
                                        device=request.device,
                                        reuse_policy="identity_attack_watermarked_copy",
                                    )
                                    _record_identity_quality_pairs(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        scope="watermarked_vs_attacked_watermarked",
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        seed=int(seed),
                                        reference_dir=watermarked_dir,
                                        target_dir=attacked_dir,
                                        device=request.device,
                                    )
                                else:
                                    _record_quality_pairs(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        scope="original_vs_attacked_watermarked",
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        seed=int(seed),
                                        reference_dir=cell_input_dir,
                                        target_dir=attacked_dir,
                                        device=request.device,
                                    )
                                    _record_quality_pairs(
                                        paths,
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        scope="watermarked_vs_attacked_watermarked",
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        seed=int(seed),
                                        reference_dir=watermarked_dir,
                                        target_dir=attacked_dir,
                                        device=request.device,
                                    )

                                for result in extract_results:
                                    record = _detection_record(
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        seed=int(seed),
                                        label=1,
                                        input_root=attacked_dir,
                                        result=result,
                                    )
                                    detection_records.append(record)
                                    _append_jsonl(
                                        paths["imageDetection"],
                                        record,
                                    )
                                for result in negative_extract_results:
                                    record = _detection_record(
                                        run_id=request.run_id,
                                        cell_key=cell_key,
                                        dataset_id=dataset_id,
                                        algorithm_id=algorithm_id,
                                        attack_id=attack_id,
                                        attack_method=attack["method"],
                                        attack_strength=strength,
                                        seed=int(seed),
                                        label=0,
                                        input_root=negative_attacked_dir,
                                        result=result,
                                    )
                                    detection_records.append(record)
                                    _append_jsonl(
                                        paths["imageDetection"],
                                        record,
                                    )

                                elapsed_ms = (time.perf_counter() - cell_started) * 1000
                            except Exception as exc:
                                status = "failed"
                                error = f"{type(exc).__name__}: {exc}"
                                elapsed_ms = (time.perf_counter() - cell_started) * 1000
                            finally:
                                shutil.rmtree(attacked_dir, ignore_errors=True)
                                shutil.rmtree(extracted_dir, ignore_errors=True)
                                shutil.rmtree(negative_extracted_dir, ignore_errors=True)
                                _write_json(cell_detection_manifest_path, detection_records)

                            cell = {
                                "runId": request.run_id,
                                "cellKey": cell_key,
                                "status": status,
                                "datasetId": dataset_id,
                                "algorithmId": algorithm_id,
                                "watermarkMethod": algorithm["method"],
                                "attackPresetId": attack_id,
                                "attackMethod": attack["method"],
                                "attackStrength": strength,
                                "seed": int(seed),
                                "sampleCount": len(copied_samples),
                                "attackParams": attack_params,
                                "manifestPath": str(cell_detection_manifest_path),
                                "outputDir": str(cell_root),
                                "error": error,
                                "elapsedMs": elapsed_ms,
                            }
                            emit_cell(cell)
                            _stage_event(
                                paths,
                                request.run_id,
                                "cell",
                                status,
                                cellKey=cell_key,
                                datasetId=dataset_id,
                                algorithmId=algorithm_id,
                                attackPresetId=attack_id,
                                attackStrength=strength,
                                attackParams=attack_params,
                                elapsedMs=elapsed_ms,
                                error=error,
                            )
                    finally:
                        shutil.rmtree(watermarked_dir, ignore_errors=True)

                if cancelled:
                    break
        finally:
            shutil.rmtree(cell_input_dir, ignore_errors=True)
            shutil.rmtree(
                run_root / "staging" / "negative_attacked" / safe_segment(dataset_id),
                ignore_errors=True,
            )
            negative_attack_cache.clear()
            _stage_event(paths, request.run_id, "dataset", "finished", datasetId=dataset_id)

    failed = sum(1 for cell in cells if cell["status"] != "succeeded")
    status = (
        "cancelled"
        if cancelled
        else "succeeded"
        if failed == 0
        else "partially_failed"
        if failed < len(cells)
        else "failed"
    )
    summary = {
        "runId": request.run_id,
        "status": status,
        "selection": selection,
        "artifactRoot": str(run_root),
        "artifactFiles": {key: str(path) for key, path in paths.items()},
        "cellCount": len(cells),
        "completedCells": len(cells),
        "failedCells": failed,
        "skippedCells": skipped_cells,
        "progress": _progress(len(cells), expected_cells),
        "completedProgress": _progress(len(cells), expected_cells),
        "succeededProgress": _progress(len(cells) - failed, expected_cells),
        "progressKind": "completedCells",
        "elapsedMs": (time.perf_counter() - started) * 1000,
        "cells": cells,
    }

    _write_latest_cell_artifacts(paths, run_id=request.run_id, expected_cells=expected_cells)
    _write_json(paths["runSummary"], summary)
    _write_run_status(
        paths,
        run_id=request.run_id,
        status=status,
        completed_cells=len(cells),
        expected_cells=expected_cells,
    )
    _stage_event(
        paths,
        request.run_id,
        "run",
        status,
        completedCells=len(cells),
        failedCells=failed,
        skippedCells=skipped_cells,
        elapsedMs=summary["elapsedMs"],
    )
    return summary
