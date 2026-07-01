from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evaluator.attacks.runner import AttackJob, get_cached_attack, run_attack_dir_with_attack
from evaluator.execution import summarize_execution_profiles
from evaluator.image_protocol import canonical_preprocess_image
from evaluator.watermarking.runner import (
    WatermarkEmbedJob,
    WatermarkExtractJob,
    get_cached_watermark,
    run_watermark_embed_dir_with_method,
    run_watermark_extract_dir_with_method,
)

from app.services.experiment_schema import SAMPLE_MANIFEST_SCHEMA
from app.services.resources import iter_image_paths


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class StagedSample:
    path: Path
    source_path: Path
    metadata: JsonDict


@dataclass(frozen=True)
class DatasetStageResult:
    staged_samples: list[StagedSample]
    copied_samples: list[Path]


@dataclass(frozen=True)
class WatermarkStageResult:
    method: Any
    results: list[Any]
    quality_records: list[JsonDict]
    elapsed_ms: float
    error: str | None


@dataclass(frozen=True)
class AttackStageResult:
    results: list[Any]
    output_dir: Path
    elapsed_ms: float
    error: str | None
    cache_hit: bool = False


@dataclass(frozen=True)
class ExtractStageResult:
    results: list[Any]
    elapsed_ms: float
    error: str | None


def canonical_target_path(output_dir: Path, relative: Path, index: int) -> Path:
    target = (output_dir / relative).with_suffix(".png")
    if not target.exists():
        return target
    return target.with_name(f"{target.stem}_{index:04d}.png")


def copy_canonical_samples(dataset_path: Path, output_dir: Path, max_samples: int) -> list[StagedSample]:
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
        target = canonical_target_path(output_dir, relative, index)
        metadata = canonical_preprocess_image(sample_path, target)
        staged.append(StagedSample(path=target, source_path=sample_path, metadata=metadata))

    return staged


@dataclass
class DatasetStage:
    paths: dict[str, Path]
    run_id: str
    append_jsonl: Callable[..., None]
    stage_event: Callable[..., None]
    image_sample_id: Callable[[Path, Path], str]
    utc_timestamp: Callable[[], str]

    def prepare(
        self,
        *,
        dataset_id: str,
        dataset_path: Path,
        input_dir: Path,
        max_samples: int,
        existing_sample_keys: set[tuple[str, str]],
    ) -> DatasetStageResult:
        self.stage_event(self.paths, self.run_id, "dataset", "started", datasetId=dataset_id)
        staged_samples = copy_canonical_samples(dataset_path, input_dir, max_samples)
        copied_samples = [sample.path for sample in staged_samples]
        if not staged_samples:
            raise ValueError(f"Dataset has no supported image files: {dataset_path}")

        for staged_sample in staged_samples:
            sample_path = staged_sample.path
            sample_id = self.image_sample_id(sample_path, input_dir)
            sample_key = (dataset_id, sample_id)
            if sample_key in existing_sample_keys:
                continue
            original_size = staged_sample.metadata.get("originalSize") or [None, None]
            canonical_size = staged_sample.metadata.get("canonicalSize") or [None, None]
            self.append_jsonl(
                self.paths["sampleManifest"],
                SAMPLE_MANIFEST_SCHEMA.apply(
                    {
                        "runId": self.run_id,
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
                        "timestamp": self.utc_timestamp(),
                    }
                ),
            )
            existing_sample_keys.add(sample_key)

        return DatasetStageResult(staged_samples=staged_samples, copied_samples=copied_samples)


@dataclass
class WatermarkStage:
    paths: dict[str, Path]
    run_id: str
    device: str
    message: str
    reset_gpu_peak: Callable[[str], None]
    record_runtime_profile: Callable[..., None]
    record_watermark_embed_results: Callable[..., None]
    record_quality_pairs: Callable[..., list[JsonDict]]
    stage_event: Callable[..., None]

    def embed(
        self,
        *,
        embed_key: str,
        dataset_id: str,
        algorithm_id: str,
        algorithm: JsonDict,
        algorithm_params: JsonDict,
        seed: int,
        input_dir: Path,
        output_dir: Path,
        copied_samples: list[Path],
    ) -> WatermarkStageResult:
        self.stage_event(
            self.paths,
            self.run_id,
            "watermark_embed",
            "started",
            cellKey=embed_key,
            datasetId=dataset_id,
            algorithmId=algorithm_id,
            seed=seed,
        )
        self.reset_gpu_peak(self.device)
        started = time.perf_counter()
        watermark_method = get_cached_watermark(algorithm["method"], algorithm_params, self.device)
        embed_results = run_watermark_embed_dir_with_method(
            WatermarkEmbedJob(
                run_id=self.run_id,
                method_name=algorithm["method"],
                params=algorithm_params,
                input_dir=input_dir,
                output_dir=output_dir,
                message=self.message,
                device=self.device,
                seed=seed,
            ),
            watermark_method,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.record_watermark_embed_results(
            self.paths,
            run_id=self.run_id,
            cell_key=embed_key,
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            watermark_method=algorithm["method"],
            seed=seed,
            input_root=input_dir,
            results=embed_results,
        )
        embed_errors = [result.error for result in embed_results if getattr(result, "error", None)]
        embed_error = None
        if not all(result.ok for result in embed_results):
            embed_error = "; ".join(embed_errors) or "one or more watermark embed operations failed"
        self.record_runtime_profile(
            self.paths,
            run_id=self.run_id,
            cell_key=embed_key,
            stage="watermark_embed",
            method=algorithm["method"],
            device=self.device,
            elapsed_ms=elapsed_ms,
            image_paths=copied_samples,
            status="failed" if embed_error else "succeeded",
            error=embed_error,
            metadata={"execution": summarize_execution_profiles(embed_results)},
        )
        if embed_error:
            raise RuntimeError(embed_error)
        quality_records = self.record_quality_pairs(
            self.paths,
            run_id=self.run_id,
            cell_key=embed_key,
            scope="original_vs_watermarked",
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=None,
            attack_method=None,
            attack_strength=None,
            seed=seed,
            reference_dir=input_dir,
            target_dir=output_dir,
            device=self.device,
        )
        self.stage_event(
            self.paths,
            self.run_id,
            "watermark_embed",
            "succeeded",
            cellKey=embed_key,
            elapsedMs=elapsed_ms,
        )
        return WatermarkStageResult(
            method=watermark_method,
            results=embed_results,
            quality_records=quality_records,
            elapsed_ms=elapsed_ms,
            error=embed_error,
        )


@dataclass
class AttackStage:
    paths: dict[str, Path]
    run_id: str
    device: str
    reset_gpu_peak: Callable[[str], None]
    list_image_files: Callable[[Path], list[Path]]
    record_runtime_profile: Callable[..., None]
    record_attack_results: Callable[..., None]

    def positive(
        self,
        *,
        cell_key: str,
        dataset_id: str,
        algorithm_id: str,
        attack_id: str,
        attack: JsonDict,
        attack_params: JsonDict,
        strength: float,
        seed: int,
        input_dir: Path,
        output_dir: Path,
    ) -> tuple[Any, AttackStageResult]:
        attack_instance = get_cached_attack(attack["method"], attack_params, self.device)
        self.reset_gpu_peak(self.device)
        started = time.perf_counter()
        results = run_attack_dir_with_attack(
            AttackJob(
                run_id=self.run_id,
                attack_name=attack["method"],
                params=attack_params,
                input_dir=input_dir,
                output_dir=output_dir,
                device=self.device,
                seed=seed,
            ),
            attack_instance,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.record_attack_results(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            stage="attack",
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=attack_id,
            attack_method=attack["method"],
            attack_strength=strength,
            attack_params=attack_params,
            seed=seed,
            label=1,
            input_root=input_dir,
            results=results,
        )
        error = "; ".join(result.error for result in results if getattr(result, "error", None))
        self.record_runtime_profile(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            stage="attack",
            method=attack["method"],
            device=self.device,
            elapsed_ms=elapsed_ms,
            image_paths=self.list_image_files(input_dir),
            status="failed" if error else "succeeded",
            error=error or None,
            metadata={"attackParams": attack_params, "execution": summarize_execution_profiles(results)},
        )
        return attack_instance, AttackStageResult(
            results=results,
            output_dir=output_dir,
            elapsed_ms=elapsed_ms,
            error=error or None,
        )

    def negative_control(
        self,
        *,
        cell_key: str,
        dataset_id: str,
        algorithm_id: str,
        attack_id: str,
        attack: JsonDict,
        attack_params: JsonDict,
        strength: float,
        seed: int,
        input_dir: Path,
        output_dir: Path,
        copied_samples: list[Path],
        cache_key: str,
        cache: dict[str, dict[str, Any]],
        attack_instance: Any,
    ) -> AttackStageResult:
        cached = cache.get(cache_key)
        if cached is not None:
            results = cached["results"]
            error = cached.get("error")
            self.record_attack_results(
                self.paths,
                run_id=self.run_id,
                cell_key=cell_key,
                stage="attack_negative_control",
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                attack_id=attack_id,
                attack_method=attack["method"],
                attack_strength=strength,
                attack_params=attack_params,
                seed=seed,
                label=0,
                input_root=input_dir,
                results=results,
                cache_hit=True,
            )
            self.record_runtime_profile(
                self.paths,
                run_id=self.run_id,
                cell_key=cell_key,
                stage="attack_negative_control",
                method=attack["method"],
                device=self.device,
                elapsed_ms=0.0,
                image_paths=copied_samples,
                status="reused" if not error else "failed",
                error=error or None,
                metadata={
                    "attackParams": attack_params,
                    "cacheKey": cache_key,
                    "cacheHit": True,
                    "execution": summarize_execution_profiles(results),
                },
            )
            return AttackStageResult(
                results=results,
                output_dir=Path(cached["outputDir"]),
                elapsed_ms=0.0,
                error=error or None,
                cache_hit=True,
            )

        self.reset_gpu_peak(self.device)
        started = time.perf_counter()
        results = run_attack_dir_with_attack(
            AttackJob(
                run_id=self.run_id,
                attack_name=attack["method"],
                params=attack_params,
                input_dir=input_dir,
                output_dir=output_dir,
                device=self.device,
                seed=seed,
            ),
            attack_instance,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.record_attack_results(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            stage="attack_negative_control",
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=attack_id,
            attack_method=attack["method"],
            attack_strength=strength,
            attack_params=attack_params,
            seed=seed,
            label=0,
            input_root=input_dir,
            results=results,
            cache_hit=False,
        )
        error = "; ".join(result.error for result in results if getattr(result, "error", None))
        if not error:
            cache[cache_key] = {"outputDir": output_dir, "results": results, "error": None}
        self.record_runtime_profile(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            stage="attack_negative_control",
            method=attack["method"],
            device=self.device,
            elapsed_ms=elapsed_ms,
            image_paths=copied_samples,
            status="failed" if error else "succeeded",
            error=error or None,
            metadata={
                "attackParams": attack_params,
                "cacheKey": cache_key,
                "cacheHit": False,
                "execution": summarize_execution_profiles(results),
            },
        )
        return AttackStageResult(
            results=results,
            output_dir=output_dir,
            elapsed_ms=elapsed_ms,
            error=error or None,
            cache_hit=False,
        )


@dataclass
class ExtractStage:
    paths: dict[str, Path]
    run_id: str
    device: str
    message: str
    reset_gpu_peak: Callable[[str], None]
    list_image_files: Callable[[Path], list[Path]]
    record_runtime_profile: Callable[..., None]

    def run(
        self,
        *,
        cell_key: str,
        runtime_stage: str,
        algorithm: JsonDict,
        algorithm_params: JsonDict,
        watermark_method: Any,
        seed: int,
        input_dir: Path,
        output_dir: Path,
    ) -> ExtractStageResult:
        self.reset_gpu_peak(self.device)
        started = time.perf_counter()
        results = run_watermark_extract_dir_with_method(
            WatermarkExtractJob(
                run_id=self.run_id,
                method_name=algorithm["method"],
                params=algorithm_params,
                input_dir=input_dir,
                output_dir=output_dir,
                message=self.message,
                device=self.device,
                seed=seed,
            ),
            watermark_method,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        error = "; ".join(result.error for result in results if getattr(result, "error", None))
        self.record_runtime_profile(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            stage=runtime_stage,
            method=algorithm["method"],
            device=self.device,
            elapsed_ms=elapsed_ms,
            image_paths=self.list_image_files(input_dir),
            status="failed" if error else "succeeded",
            error=error or None,
            metadata={"execution": summarize_execution_profiles(results)},
        )
        return ExtractStageResult(results=results, elapsed_ms=elapsed_ms, error=error or None)


@dataclass
class QualityStage:
    paths: dict[str, Path]
    run_id: str
    device: str
    record_quality_pairs: Callable[..., list[JsonDict]]
    record_reused_quality_records: Callable[..., list[JsonDict]]
    record_identity_quality_pairs: Callable[..., list[JsonDict]]

    def record_attack_quality(
        self,
        *,
        is_identity: bool,
        cell_key: str,
        dataset_id: str,
        algorithm_id: str,
        attack_id: str,
        attack_method: str,
        attack_strength: float,
        seed: int,
        canonical_input_dir: Path,
        watermarked_dir: Path,
        attacked_dir: Path,
        embed_quality_records: list[JsonDict],
    ) -> None:
        if is_identity:
            self.record_reused_quality_records(
                self.paths,
                run_id=self.run_id,
                cell_key=cell_key,
                scope="original_vs_attacked_watermarked",
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                attack_id=attack_id,
                attack_method=attack_method,
                attack_strength=attack_strength,
                seed=seed,
                source_records=embed_quality_records,
                source_scope="original_vs_watermarked",
                target_dir=attacked_dir,
                device=self.device,
                reuse_policy="identity_attack_watermarked_copy",
            )
            self.record_identity_quality_pairs(
                self.paths,
                run_id=self.run_id,
                cell_key=cell_key,
                scope="watermarked_vs_attacked_watermarked",
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                attack_id=attack_id,
                attack_method=attack_method,
                attack_strength=attack_strength,
                seed=seed,
                reference_dir=watermarked_dir,
                target_dir=attacked_dir,
                device=self.device,
            )
            return

        self.record_quality_pairs(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            scope="original_vs_attacked_watermarked",
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=attack_id,
            attack_method=attack_method,
            attack_strength=attack_strength,
            seed=seed,
            reference_dir=canonical_input_dir,
            target_dir=attacked_dir,
            device=self.device,
        )
        self.record_quality_pairs(
            self.paths,
            run_id=self.run_id,
            cell_key=cell_key,
            scope="watermarked_vs_attacked_watermarked",
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            attack_id=attack_id,
            attack_method=attack_method,
            attack_strength=attack_strength,
            seed=seed,
            reference_dir=watermarked_dir,
            target_dir=attacked_dir,
            device=self.device,
        )


@dataclass
class DetectionStage:
    paths: dict[str, Path]
    run_id: str
    append_jsonl: Callable[..., None]
    detection_record: Callable[..., JsonDict]

    def append_results(
        self,
        *,
        detection_records: list[JsonDict],
        cell_key: str,
        dataset_id: str,
        algorithm_id: str,
        attack_id: str,
        attack_method: str,
        attack_strength: float,
        seed: int,
        label: int,
        input_root: Path,
        results: list[Any],
    ) -> None:
        for result in results:
            record = self.detection_record(
                run_id=self.run_id,
                cell_key=cell_key,
                dataset_id=dataset_id,
                algorithm_id=algorithm_id,
                attack_id=attack_id,
                attack_method=attack_method,
                attack_strength=attack_strength,
                seed=seed,
                label=label,
                input_root=input_root,
                result=result,
            )
            detection_records.append(record)
            self.append_jsonl(self.paths["imageDetection"], record)
