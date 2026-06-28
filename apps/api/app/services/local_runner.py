from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
from app.services.scoring import aggregate_benchmark_score, compute_quality_summary, score_cell


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


def _ensure_list(value: Any, fallback: list[Any]) -> list[Any]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return fallback


def normalize_selection(selection: JsonDict, resources_root: Path) -> JsonDict:
    datasets = scan_dataset_resources(resources_root)
    default_dataset_ids = [datasets[0].id] if datasets else []
    dataset_ids = _ensure_list(selection.get("datasetIds"), default_dataset_ids)
    algorithm_ids = _ensure_list(selection.get("algorithmIds"), ["alg-traditional-lsb"])
    attack_ids = _ensure_list(selection.get("attackPresetIds"), ["atk-identity", "atk-jpeg-smoke"])
    seeds = [int(seed) for seed in _ensure_list(selection.get("seeds"), [42])]
    max_samples = int(selection.get("maxSamples") or 1)

    return {
        "datasetIds": [str(value) for value in dataset_ids],
        "algorithmIds": [str(value) for value in algorithm_ids],
        "attackPresetIds": [str(value) for value in attack_ids],
        "seeds": seeds,
        "maxSamples": max(1, max_samples),
    }


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
        strength_count += max(1, len(attack["strengths"]))

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
) -> str:
    raw = f"{dataset_id}__{algorithm_id}__{attack_id}__{strength:g}__{seed}"
    return safe_segment(raw)


def _attack_params(attack: JsonDict, strength: float) -> JsonDict:
    params = dict(attack.get("params") or {})
    strength_param = attack.get("strengthParam")
    if strength_param:
        params[str(strength_param)] = float(strength)
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

    cells: list[JsonDict] = []
    started = time.perf_counter()
    estimate = estimate_selection(selection, request.resources_root)
    expected_cells = int(estimate["cellCount"])
    cancelled = False

    for dataset_id in selection["datasetIds"]:
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        dataset = get_dataset_by_id(request.resources_root, dataset_id)
        cell_input_dir = run_root / "input" / safe_segment(dataset_id)
        copied_samples = _copy_samples(dataset.path, cell_input_dir, selection["maxSamples"])
        if not copied_samples:
            raise ValueError(f"Dataset has no supported image files: {dataset.path}")

        for algorithm_id in selection["algorithmIds"]:
            algorithm = get_watermark_catalog_item(algorithm_id)
            for attack_id in selection["attackPresetIds"]:
                attack = get_attack_catalog_item(attack_id)
                for strength in attack["strengths"] or [0.0]:
                    for seed in selection["seeds"]:
                        if should_cancel is not None and should_cancel():
                            cancelled = True
                            break
                        cell_key = _cell_key(dataset_id, algorithm_id, attack_id, float(strength), int(seed))
                        cell_root = run_root / "cells" / cell_key
                        watermarked_dir = cell_root / "watermarked"
                        attacked_dir = cell_root / "attacked"
                        extracted_dir = cell_root / "extracted"
                        negative_attacked_dir = cell_root / "negative_attacked"
                        negative_extracted_dir = cell_root / "negative_extracted"
                        attack_params = _attack_params(attack, float(strength))
                        status = "succeeded"
                        error = None
                        bit_accuracy = None
                        bit_error_rate = None
                        scoring = None
                        cell_started = time.perf_counter()

                        try:
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

                            operation_results = [
                                *embed_results,
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
                                    if result.error
                                ]
                                error = "; ".join(errors) or "one or more image operations failed"
                            bit_accuracy = _average_bit_accuracy(extract_results)
                            bit_error_rate = _bit_error_rate(bit_accuracy)
                            elapsed_ms = (time.perf_counter() - cell_started) * 1000
                            quality_summary = compute_quality_summary(cell_input_dir, attacked_dir)
                            clean_quality_summary = compute_quality_summary(cell_input_dir, watermarked_dir)
                            scoring = score_cell(
                                algorithm_id=algorithm_id,
                                attack_preset_id=attack_id,
                                attack_method=attack["method"],
                                attack_strength=float(strength),
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

                        cell = {
                            "cellKey": cell_key,
                            "status": status,
                            "datasetId": dataset_id,
                            "algorithmId": algorithm_id,
                            "watermarkMethod": algorithm["method"],
                            "attackPresetId": attack_id,
                            "attackMethod": attack["method"],
                            "attackStrength": float(strength),
                            "seed": int(seed),
                            "sampleCount": len(copied_samples),
                            "bitAccuracy": bit_accuracy,
                            "bitErrorRate": bit_error_rate,
                            "attackParams": attack_params,
                            "manifestPath": str(extracted_dir / "watermark_extract_manifest.json"),
                            "negativeManifestPath": str(negative_extracted_dir / "watermark_extract_manifest.json"),
                            "outputDir": str(cell_root),
                            "error": error,
                            "elapsedMs": elapsed_ms,
                            "scoring": scoring,
                        }
                        cells.append(cell)
                        if on_cell is not None:
                            on_cell(cell)
                    if cancelled:
                        break
                if cancelled:
                    break
            if cancelled:
                break

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
        "cellCount": len(cells),
        "completedCells": len(cells),
        "failedCells": failed,
        "progress": _progress(len(cells), expected_cells),
        "elapsedMs": (time.perf_counter() - started) * 1000,
        "aggregates": _aggregate_cells(cells),
        "score": aggregate_benchmark_score(cells),
        "cells": cells,
    }

    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
