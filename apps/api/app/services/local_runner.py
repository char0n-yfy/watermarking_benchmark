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

from evaluator.attacks.runner import AttackJob, run_attack_dir
from evaluator.watermarking.runner import (
    WatermarkEmbedJob,
    WatermarkExtractJob,
    run_watermark_embed_dir,
    run_watermark_extract_dir,
)

from app.core.storage import safe_segment
from app.services.resources import (
    get_attack_catalog_item,
    get_dataset_by_id,
    get_watermark_catalog_item,
    iter_image_paths,
    scan_dataset_resources,
)
from app.services.scoring import (
    aggregate_benchmark_score,
    compute_image_quality_pair,
    compute_quality_summary,
    score_cell,
)


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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
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
        "imageQuality": run_root / "image_quality.jsonl",
        "imageDetection": run_root / "image_detection.jsonl",
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
    seconds = elapsed_ms / 1000.0
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
            "msPerImage": None if not image_paths else elapsed_ms / len(image_paths),
            "msPerMP": None if total_mp <= 0 else elapsed_ms / total_mp,
            "throughputImagesPerSecond": None if seconds <= 0 else len(image_paths) / seconds,
            "throughputMPPerSecond": None if seconds <= 0 else total_mp / seconds,
            "peakMemoryMB": _gpu_peak_memory_mb(device) or _process_peak_memory_mb(),
            "macs": None,
            "flops": None,
            "error": error,
            "metadata": metadata or {},
            "timestamp": _utc_timestamp(),
        },
    )


def _pair_images(reference_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    references = {
        path.relative_to(reference_dir).with_suffix("").as_posix(): path
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    pairs: list[tuple[Path, Path]] = []
    for target in sorted(target_dir.rglob("*")):
        if not target.is_file() or target.suffix.lower() not in IMAGE_EXTS:
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
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
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
) -> None:
    for reference_path, target_path in _pair_images(reference_dir, target_dir):
        metrics = compute_image_quality_pair(reference_path, target_path)
        width, height = _image_size(reference_path)
        _append_jsonl(
            paths["imageQuality"],
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
                "sampleId": _image_sample_id(reference_path, reference_dir),
                "width": width,
                "height": height,
                "referencePath": str(reference_path),
                "targetPath": str(target_path),
                "metrics": metrics,
                "timestamp": _utc_timestamp(),
            },
        )


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
    metadata = getattr(result, "metadata", {}) or {}
    bit_accuracy = metadata.get("bit_accuracy")
    try:
        parsed_bit_accuracy = None if bit_accuracy is None else float(bit_accuracy)
    except (TypeError, ValueError):
        parsed_bit_accuracy = None
    detection_score = metadata.get("detection_score")
    if detection_score is None:
        detection_score = parsed_bit_accuracy
    try:
        parsed_detection_score = None if detection_score is None else float(detection_score)
    except (TypeError, ValueError):
        parsed_detection_score = None
    bits = getattr(result, "bits", None)
    matched = metadata.get("matched")
    if matched is None:
        matched = metadata.get("match")
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
        "inputPath": str(input_path),
        "status": "succeeded" if getattr(result, "ok", False) else "failed",
        "detectionScore": parsed_detection_score,
        "bitAccuracy": parsed_bit_accuracy,
        "bitErrorRate": _bit_error_rate(parsed_bit_accuracy),
        "bitLength": len(bits) if isinstance(bits, list) else None,
        "matched": matched,
        "elapsedMs": getattr(result, "elapsed_ms", None),
        "error": getattr(result, "error", None),
        "metadata": metadata,
        "timestamp": _utc_timestamp(),
    }


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


def _copy_samples(dataset_path: Path, output_dir: Path, max_samples: int) -> list[Path]:
    sample_paths = iter_image_paths(dataset_path)[:max_samples]
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for index, sample_path in enumerate(sample_paths, start=1):
        try:
            relative = sample_path.relative_to(dataset_path)
        except ValueError:
            relative = Path(f"sample_{index:04d}{sample_path.suffix.lower()}")
        if relative.name.startswith("."):
            relative = Path(f"sample_{index:04d}{sample_path.suffix.lower()}")
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample_path, target)
        copied.append(target)

    return copied


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


def _average_bit_accuracy(extract_results: list[Any]) -> float | None:
    values: list[float] = []
    for result in extract_results:
        value = result.metadata.get("bit_accuracy")
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _bit_error_rate(bit_accuracy: float | None) -> float | None:
    if bit_accuracy is None:
        return None
    return max(0.0, min(1.0, 1.0 - bit_accuracy))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_cells(cells: list[JsonDict]) -> list[JsonDict]:
    groups: dict[tuple[str, str, float], list[JsonDict]] = {}
    for cell in cells:
        key = (
            str(cell["algorithmId"]),
            str(cell["attackPresetId"]),
            float(cell["attackStrength"]),
        )
        groups.setdefault(key, []).append(cell)

    summaries: list[JsonDict] = []
    for (algorithm_id, attack_preset_id, attack_strength), group_cells in sorted(groups.items()):
        bit_accuracies = [
            float(cell["bitAccuracy"])
            for cell in group_cells
            if cell.get("bitAccuracy") is not None
        ]
        bit_error_rates = [
            float(cell["bitErrorRate"])
            for cell in group_cells
            if cell.get("bitErrorRate") is not None
        ]
        succeeded = sum(1 for cell in group_cells if cell["status"] == "succeeded")
        summaries.append(
            {
                "algorithmId": algorithm_id,
                "attackPresetId": attack_preset_id,
                "attackStrength": attack_strength,
                "cellCount": len(group_cells),
                "succeededCells": succeeded,
                "failedCells": len(group_cells) - succeeded,
                "meanBitAccuracy": _mean(bit_accuracies),
                "meanBitErrorRate": _mean(bit_error_rates),
            }
        )
    return summaries


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
    cells: list[JsonDict] = list(existing_completed.values())
    started = time.perf_counter()
    estimate = estimate_selection(selection, request.resources_root)
    expected_cells = int(estimate["cellCount"])
    cancelled = False
    skipped_cells = 0

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
        cells.append(cell)
        _append_jsonl(paths["cellManifest"], cell)
        _write_run_status(
            paths,
            run_id=request.run_id,
            status="running",
            completed_cells=len(cells),
            expected_cells=expected_cells,
        )
        if on_cell is not None:
            on_cell(cell)

    for dataset_id in selection["datasetIds"]:
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        dataset = get_dataset_by_id(request.resources_root, dataset_id)
        cell_input_dir = run_root / "staging" / "samples" / safe_segment(dataset_id)
        _stage_event(paths, request.run_id, "dataset", "started", datasetId=dataset_id)
        try:
            copied_samples = _copy_samples(dataset.path, cell_input_dir, selection["maxSamples"])
            if not copied_samples:
                raise ValueError(f"Dataset has no supported image files: {dataset.path}")

            for sample_path in copied_samples:
                sample_id = _image_sample_id(sample_path, cell_input_dir)
                sample_key = (dataset_id, sample_id)
                if sample_key in existing_sample_keys:
                    continue
                width, height = _image_size(sample_path)
                _append_jsonl(
                    paths["sampleManifest"],
                    {
                        "runId": request.run_id,
                        "datasetId": dataset_id,
                        "sampleId": sample_id,
                        "sourcePath": str(dataset.path / sample_path.relative_to(cell_input_dir)),
                        "stagedPath": str(sample_path),
                        "width": width,
                        "height": height,
                        "timestamp": _utc_timestamp(),
                    },
                )
                existing_sample_keys.add(sample_key)

            for algorithm_id in selection["algorithmIds"]:
                if cancelled:
                    break
                algorithm = get_watermark_catalog_item(algorithm_id)
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
                    clean_quality_summary: JsonDict = {"sampleCount": 0}
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
                        embed_results = run_watermark_embed_dir(
                            WatermarkEmbedJob(
                                run_id=request.run_id,
                                method_name=algorithm["method"],
                                params=dict(algorithm.get("params") or {}),
                                input_dir=cell_input_dir,
                                output_dir=watermarked_dir,
                                message=request.message,
                                device=request.device,
                                seed=int(seed),
                            )
                        )
                        embed_elapsed_ms = (time.perf_counter() - embed_started) * 1000
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
                        clean_quality_summary = compute_quality_summary(cell_input_dir, watermarked_dir)
                        _record_quality_pairs(
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
                                    "bitAccuracy": None,
                                    "bitErrorRate": None,
                                    "attackParams": variant["attackParams"],
                                    "manifestPath": str(failed_detection_manifest),
                                    "negativeManifestPath": str(failed_detection_manifest),
                                    "outputDir": str(failed_cell_root),
                                    "error": embed_error,
                                    "elapsedMs": embed_elapsed_ms,
                                    "scoring": None,
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
                            negative_attacked_dir = cell_root / "negative_attacked"
                            negative_extracted_dir = cell_root / "negative_extracted"
                            cell_detection_manifest_path = cell_root / "cell_detection_manifest.json"
                            detection_records: list[JsonDict] = []
                            status = "succeeded"
                            error = None
                            bit_accuracy = None
                            bit_error_rate = None
                            scoring = None
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
                                _reset_gpu_peak(request.device)
                                attack_started = time.perf_counter()
                                attack_results = run_attack_dir(
                                    AttackJob(
                                        run_id=request.run_id,
                                        attack_name=attack["method"],
                                        params=attack_params,
                                        input_dir=watermarked_dir,
                                        output_dir=attacked_dir,
                                        device=request.device,
                                        seed=int(seed),
                                    )
                                )
                                attack_elapsed_ms = (time.perf_counter() - attack_started) * 1000
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
                                extract_results = run_watermark_extract_dir(
                                    WatermarkExtractJob(
                                        run_id=request.run_id,
                                        method_name=algorithm["method"],
                                        params=dict(algorithm.get("params") or {}),
                                        input_dir=attacked_dir,
                                        output_dir=extracted_dir,
                                        message=request.message,
                                        device=request.device,
                                        seed=int(seed),
                                    )
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

                                _reset_gpu_peak(request.device)
                                negative_attack_started = time.perf_counter()
                                negative_attack_results = run_attack_dir(
                                    AttackJob(
                                        run_id=request.run_id,
                                        attack_name=attack["method"],
                                        params=attack_params,
                                        input_dir=cell_input_dir,
                                        output_dir=negative_attacked_dir,
                                        device=request.device,
                                        seed=int(seed),
                                    )
                                )
                                negative_attack_elapsed_ms = (time.perf_counter() - negative_attack_started) * 1000
                                negative_attack_error = "; ".join(
                                    result.error
                                    for result in negative_attack_results
                                    if getattr(result, "error", None)
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
                                    status="failed" if negative_attack_error else "succeeded",
                                    error=negative_attack_error or None,
                                    metadata={"attackParams": attack_params},
                                )

                                _reset_gpu_peak(request.device)
                                negative_extract_started = time.perf_counter()
                                negative_extract_results = run_watermark_extract_dir(
                                    WatermarkExtractJob(
                                        run_id=request.run_id,
                                        method_name=algorithm["method"],
                                        params=dict(algorithm.get("params") or {}),
                                        input_dir=negative_attacked_dir,
                                        output_dir=negative_extracted_dir,
                                        message=request.message,
                                        device=request.device,
                                        seed=int(seed),
                                    )
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

                                bit_accuracy = _average_bit_accuracy(extract_results)
                                bit_error_rate = _bit_error_rate(bit_accuracy)
                                elapsed_ms = (time.perf_counter() - cell_started) * 1000
                                quality_summary = compute_quality_summary(cell_input_dir, attacked_dir)
                                scoring = score_cell(
                                    algorithm_id=algorithm_id,
                                    attack_preset_id=attack_id,
                                    attack_method=attack["method"],
                                    attack_strength=strength,
                                    sample_count=len(copied_samples),
                                    positive_extract_results=extract_results,
                                    negative_extract_results=negative_extract_results,
                                    quality_summary=quality_summary,
                                    clean_quality_summary=clean_quality_summary,
                                    elapsed_ms=elapsed_ms,
                                )
                            except Exception as exc:
                                status = "failed"
                                error = f"{type(exc).__name__}: {exc}"
                                elapsed_ms = (time.perf_counter() - cell_started) * 1000
                            finally:
                                shutil.rmtree(attacked_dir, ignore_errors=True)
                                shutil.rmtree(extracted_dir, ignore_errors=True)
                                shutil.rmtree(negative_attacked_dir, ignore_errors=True)
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
                                "bitAccuracy": bit_accuracy,
                                "bitErrorRate": bit_error_rate,
                                "attackParams": attack_params,
                                "manifestPath": str(cell_detection_manifest_path),
                                "negativeManifestPath": str(cell_detection_manifest_path),
                                "outputDir": str(cell_root),
                                "error": error,
                                "elapsedMs": elapsed_ms,
                                "scoring": scoring,
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
        "elapsedMs": (time.perf_counter() - started) * 1000,
        "aggregates": _aggregate_cells(cells),
        "score": aggregate_benchmark_score(cells),
        "cells": cells,
    }

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
