from __future__ import annotations

"""Local experiment executor implementation used by the public local_runner facade."""

import sys
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from evaluator.image_protocol import quality_alignment_metadata
from evaluator.execution import ExecutionProfile

from app.core.storage import safe_segment
from app.services.experiment_schema import (
    CELL_MANIFEST_SCHEMA,
    IMAGE_ATTACK_SCHEMA,
    IMAGE_DETECTION_SCHEMA,
    IMAGE_QUALITY_SCHEMA,
    IMAGE_WATERMARK_EMBED_SCHEMA,
    RUNTIME_PROFILE_SCHEMA,
)
from app.services.experiment_stages import (
    AttackStage,
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
    append_jsonl as _append_jsonl,
    artifact_paths as _artifact_paths,
    cell_attempt_counts as _cell_attempt_counts,
    latest_cell_row_map as _latest_cell_row_map,
    progress as _progress,
    read_jsonl as _read_jsonl,
    stage_event as _stage_event,
    utc_timestamp as _utc_timestamp,
    write_json as _write_json,
    write_latest_cell_artifacts as _write_latest_cell_artifacts,
    write_run_status as _write_run_status,
)
from app.services.runtime_resource_manager import RuntimeResourceManager
from app.services.scoring import compute_image_quality_pairs_with_profile


JsonDict = dict[str, Any]
CellCallback = Callable[[JsonDict], None]
StopIntentCallback = Callable[[], str | bool | None]


@dataclass(frozen=True)
class LocalRunRequest:
    run_id: str
    selection: JsonDict
    resources_root: Path
    runs_root: Path
    device: str = "cpu"
    message: str = "1010101010101010"
    resume: bool = True


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
INTERMEDIATE_ARTIFACT_DIR = "_intermediates"
STOP_INTENT_CANCEL = "cancel"
STOP_INTENT_PAUSE = "pause"


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
        metrics_by_pair, execution_profile = compute_image_quality_pairs_with_profile(pairs)
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
        metadata={
            "scope": scope,
            "sourceScope": source_scope,
            "reusePolicy": reuse_policy,
            "execution": execution_profile,
        },
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
        metadata={
            "scope": scope,
            "sourceScope": "identity_noop",
            "reusePolicy": "identity_noop_perfect",
            "execution": execution_profile,
        },
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


def run_local_experiment(
    request: LocalRunRequest,
    on_cell: CellCallback | None = None,
    should_cancel: StopIntentCallback | None = None,
) -> JsonDict:
    selection = normalize_selection(request.selection, request.resources_root)
    if not selection["datasetIds"]:
        raise ValueError("No datasets found under resources/datasets")
    run_root = request.runs_root / safe_segment(request.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(run_root)
    latest_cells = _latest_cell_row_map(paths["cellManifest"]) if request.resume else {}
    pending_groups, existing_completed, skipped_cells = pending_variant_groups(
        selection,
        latest_cells,
        resume=request.resume,
    )
    attempt_counts = _cell_attempt_counts(paths["cellManifest"]) if request.resume else {}
    cells: list[JsonDict] = list(existing_completed.values())
    started = time.perf_counter()
    estimate = estimate_selection(selection, request.resources_root)
    expected_cells = int(estimate["cellCount"])
    stop_status: str | None = None
    negative_attack_cache: dict[str, dict[str, Any]] = {}

    _write_json(
        paths["runPlan"],
        build_run_plan_payload(
            run_id=request.run_id,
            selection=selection,
            artifact_paths=paths,
            expected_cells=expected_cells,
            pending_groups=pending_groups,
            skipped_cells=skipped_cells,
            resume=request.resume,
            created_at=_utc_timestamp(),
        ),
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
        pendingCells=sum(len(variants) for variants in pending_groups.values()),
        skippedCells=skipped_cells,
        resumeMode="pending_cells_only",
    )

    existing_sample_keys = {
        (str(record.get("datasetId")), str(record.get("sampleId")))
        for record in _read_jsonl(paths["sampleManifest"])
        if record.get("datasetId") is not None and record.get("sampleId") is not None
    }
    dataset_stage = DatasetStage(
        paths=paths,
        run_id=request.run_id,
        append_jsonl=_append_jsonl,
        stage_event=_stage_event,
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
        record_quality_pairs=_record_quality_pairs,
        stage_event=_stage_event,
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
    quality_stage = QualityStage(
        paths=paths,
        run_id=request.run_id,
        device=request.device,
        record_quality_pairs=_record_quality_pairs,
        record_reused_quality_records=_record_reused_quality_records,
        record_identity_quality_pairs=_record_identity_quality_pairs,
    )
    detection_stage = DetectionStage(
        paths=paths,
        run_id=request.run_id,
        append_jsonl=_append_jsonl,
        detection_record=_detection_record,
    )
    resource_manager = RuntimeResourceManager(
        paths=paths,
        run_id=request.run_id,
        device=request.device,
        append_jsonl=_append_jsonl,
        stage_event=_stage_event,
    )

    def emit_cell(cell: JsonDict) -> None:
        record = dict(cell)
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str):
            attempt_counts[cell_key] = attempt_counts.get(cell_key, 0) + 1
            record["attemptIndex"] = attempt_counts[cell_key]
            record["supersedesPreviousAttempt"] = attempt_counts[cell_key] > 1
        record["completedAt"] = _utc_timestamp()
        record = CELL_MANIFEST_SCHEMA.apply(record)

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
        requested_stop_status = _stop_status_from_callback(should_cancel)
        if requested_stop_status is not None:
            stop_status = requested_stop_status
            break
        if not any(group_key[0] == str(dataset_id) for group_key in pending_groups):
            continue
        dataset = get_dataset_by_id(request.resources_root, dataset_id)
        cell_input_dir = run_root / "staging" / "samples" / safe_segment(dataset_id)
        try:
            dataset_stage_result = dataset_stage.prepare(
                dataset_id=dataset_id,
                dataset_path=dataset.path,
                input_dir=cell_input_dir,
                max_samples=selection["maxSamples"],
                existing_sample_keys=existing_sample_keys,
            )
            copied_samples = dataset_stage_result.copied_samples

            for algorithm_id in selection["algorithmIds"]:
                if stop_status is not None:
                    break
                if not any(
                    group_key[0] == str(dataset_id) and group_key[1] == str(algorithm_id)
                    for group_key in pending_groups
                ):
                    continue
                algorithm = get_watermark_catalog_item(algorithm_id)
                algorithm_params = dict(algorithm.get("params") or {})
                for seed in selection["seeds"]:
                    requested_stop_status = _stop_status_from_callback(should_cancel)
                    if requested_stop_status is not None:
                        stop_status = requested_stop_status
                        break

                    pending_variants = pending_groups.get((str(dataset_id), str(algorithm_id), int(seed)), [])
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
                    embed_quality_records: list[JsonDict] = []
                    watermark_method = None

                    try:
                        watermark_stage_result = watermark_stage.embed(
                            embed_key=embed_key,
                            dataset_id=dataset_id,
                            algorithm_id=algorithm_id,
                            algorithm=algorithm,
                            algorithm_params=algorithm_params,
                            seed=int(seed),
                            input_dir=cell_input_dir,
                            output_dir=watermarked_dir,
                            copied_samples=copied_samples,
                        )
                        watermark_method = watermark_stage_result.method
                        embed_results = watermark_stage_result.results
                        embed_quality_records = watermark_stage_result.quality_records
                        embed_elapsed_ms = watermark_stage_result.elapsed_ms
                        embed_error = watermark_stage_result.error
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
                        for variant_index, variant in enumerate(pending_variants):
                            requested_stop_status = _stop_status_from_callback(should_cancel)
                            if requested_stop_status is not None:
                                stop_status = requested_stop_status
                                break

                            attack_id = str(variant["attackId"])
                            attack = variant["attack"]
                            strength = float(variant["strength"])
                            attack_params = normalize_attack_params_for_runtime(
                                str(attack["method"]),
                                dict(variant["attackParams"]),
                            )
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
                                attack_instance, positive_attack = attack_stage.positive(
                                    cell_key=cell_key,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    attack_id=attack_id,
                                    attack=attack,
                                    attack_params=attack_params,
                                    strength=strength,
                                    seed=int(seed),
                                    input_dir=watermarked_dir,
                                    output_dir=attacked_dir,
                                )
                                attack_results = positive_attack.results
                                if request.device.startswith("cuda") and algorithm["method"] == "invisible-watermark-rivagan":
                                    resource_manager.cleanup(
                                        scope="pre_extract",
                                        reason="positive_attack_finished",
                                        cell_key=cell_key,
                                        metadata={
                                            "datasetId": dataset_id,
                                            "algorithmId": algorithm_id,
                                            "watermarkMethod": algorithm["method"],
                                            "attackMethod": attack["method"],
                                            "attackPresetId": attack_id,
                                            "label": 1,
                                            "seed": int(seed),
                                        },
                                    )

                                positive_extract = extract_stage.run(
                                    cell_key=cell_key,
                                    runtime_stage="watermark_extract_positive",
                                    algorithm=algorithm,
                                    algorithm_params=algorithm_params,
                                    watermark_method=watermark_method,
                                    seed=int(seed),
                                    input_dir=attacked_dir,
                                    output_dir=extracted_dir,
                                )
                                extract_results = positive_extract.results

                                negative_attack = attack_stage.negative_control(
                                    cell_key=cell_key,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    attack_id=attack_id,
                                    attack=attack,
                                    attack_params=attack_params,
                                    strength=strength,
                                    seed=int(seed),
                                    input_dir=cell_input_dir,
                                    output_dir=negative_attacked_dir,
                                    copied_samples=copied_samples,
                                    cache_key=negative_attack_key,
                                    cache=negative_attack_cache,
                                    attack_instance=attack_instance,
                                )
                                negative_attack_results = negative_attack.results
                                negative_attacked_dir = negative_attack.output_dir
                                if request.device.startswith("cuda") and algorithm["method"] == "invisible-watermark-rivagan":
                                    resource_manager.cleanup(
                                        scope="pre_extract",
                                        reason="negative_attack_finished",
                                        cell_key=cell_key,
                                        metadata={
                                            "datasetId": dataset_id,
                                            "algorithmId": algorithm_id,
                                            "watermarkMethod": algorithm["method"],
                                            "attackMethod": attack["method"],
                                            "attackPresetId": attack_id,
                                            "label": 0,
                                            "seed": int(seed),
                                        },
                                    )

                                negative_extract = extract_stage.run(
                                    cell_key=cell_key,
                                    runtime_stage="watermark_extract_negative",
                                    algorithm=algorithm,
                                    algorithm_params=algorithm_params,
                                    watermark_method=watermark_method,
                                    seed=int(seed),
                                    input_dir=negative_attacked_dir,
                                    output_dir=negative_extracted_dir,
                                )
                                negative_extract_results = negative_extract.results

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

                                quality_stage.record_attack_quality(
                                    is_identity=str(attack["method"]).lower() == "identity",
                                    cell_key=cell_key,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    attack_id=attack_id,
                                    attack_method=attack["method"],
                                    attack_strength=strength,
                                    seed=int(seed),
                                    canonical_input_dir=cell_input_dir,
                                    watermarked_dir=watermarked_dir,
                                    attacked_dir=attacked_dir,
                                    embed_quality_records=embed_quality_records,
                                )

                                detection_stage.append_results(
                                    detection_records=detection_records,
                                    cell_key=cell_key,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    attack_id=attack_id,
                                    attack_method=attack["method"],
                                    attack_strength=strength,
                                    seed=int(seed),
                                    label=1,
                                    input_root=attacked_dir,
                                    results=extract_results,
                                )
                                detection_stage.append_results(
                                    detection_records=detection_records,
                                    cell_key=cell_key,
                                    dataset_id=dataset_id,
                                    algorithm_id=algorithm_id,
                                    attack_id=attack_id,
                                    attack_method=attack["method"],
                                    attack_strength=strength,
                                    seed=int(seed),
                                    label=0,
                                    input_root=negative_attacked_dir,
                                    results=negative_extract_results,
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
                            next_variant = (
                                pending_variants[variant_index + 1]
                                if variant_index + 1 < len(pending_variants)
                                else None
                            )
                            next_attack_method = (
                                str(next_variant["attack"]["method"])
                                if isinstance(next_variant, dict) and isinstance(next_variant.get("attack"), dict)
                                else None
                            )
                            if next_attack_method != str(attack["method"]):
                                resource_manager.cleanup(
                                    scope="attack_method",
                                    reason="attack_method_finished",
                                    cell_key=cell_key,
                                    release_attacks=True,
                                    release_auxiliary=True,
                                    metadata={
                                        "datasetId": dataset_id,
                                        "algorithmId": algorithm_id,
                                        "attackMethod": attack["method"],
                                        "attackPresetId": attack_id,
                                        "seed": int(seed),
                                    },
                                )
                    finally:
                        shutil.rmtree(watermarked_dir, ignore_errors=True)

                resource_manager.cleanup(
                    scope="watermark_algorithm",
                    reason="watermark_algorithm_finished",
                    cell_key=f"{safe_segment(dataset_id)}__{safe_segment(algorithm_id)}__watermark_algorithm",
                    release_watermarks=True,
                    metadata={
                        "datasetId": dataset_id,
                        "algorithmId": algorithm_id,
                        "watermarkMethod": algorithm["method"],
                    },
                )

                if stop_status is not None:
                    break
        finally:
            shutil.rmtree(cell_input_dir, ignore_errors=True)
            shutil.rmtree(
                run_root / "staging" / "negative_attacked" / safe_segment(dataset_id),
                ignore_errors=True,
            )
            negative_attack_cache.clear()
            resource_manager.cleanup(
                scope="dataset",
                reason="dataset_finished",
                cell_key=f"{safe_segment(dataset_id)}__dataset",
                release_attacks=True,
                release_watermarks=True,
                release_perceptual=True,
                release_auxiliary=True,
                metadata={"datasetId": dataset_id},
            )
            _stage_event(paths, request.run_id, "dataset", "finished", datasetId=dataset_id)

    resource_manager.cleanup(
        scope="run",
        reason="run_finished",
        cell_key="run",
        release_attacks=True,
        release_watermarks=True,
        release_perceptual=True,
        release_auxiliary=True,
    )

    if stop_status is None:
        stop_status = _stop_status_from_callback(should_cancel)

    failed = sum(1 for cell in cells if cell["status"] != "succeeded")
    status = (
        stop_status
        if stop_status is not None
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
