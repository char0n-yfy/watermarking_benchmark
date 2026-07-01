from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluator.image_protocol import (
    CANONICAL_IMAGE_SIZE,
    canonicalize_image_file_in_place,
    first_metadata_size,
    image_size,
)
from evaluator.execution import (
    ExecutionProfile,
    attach_execution_metadata,
    replace_result_execution,
    resolve_named_batch_size,
    resolve_named_cpu_workers,
)


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class WatermarkContext:
    """Runtime metadata passed to watermarking methods.

    Images are exchanged through files. The context carries reproducibility
    and model-selection metadata without exposing model internals.
    """

    run_id: str
    sample_id: str
    method_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    workspace_dir: Path | None = None
    device: str = "cpu"
    seed: int | None = None
    message: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WatermarkEmbedResult:
    input_path: Path
    output_path: Path
    method_name: str
    message: str | None
    params: Mapping[str, Any]
    elapsed_ms: float
    ok: bool = True
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        data = asdict(self)
        data["input_path"] = str(self.input_path)
        data["output_path"] = str(self.output_path)
        return data


@dataclass(frozen=True)
class WatermarkExtractResult:
    input_path: Path
    method_name: str
    params: Mapping[str, Any]
    elapsed_ms: float
    ok: bool = True
    error: str | None = None
    message: str | None = None
    bits: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        data = asdict(self)
        data["input_path"] = str(self.input_path)
        return data


class BaseWatermark:
    """Base class for image watermark embedders/extractors.

    Contract:
    - embed reads input_path and writes a watermarked image to output_path.
    - extract reads one image and returns JSON-serializable decoded metadata.
    - heavy model loading and checkpoint paths stay behind this interface.
    """

    name = "base"
    description = ""
    output_ext = ".png"
    thread_safe_parallel = False

    def __init__(self, **params: Any) -> None:
        self.params: JsonDict = dict(params)

    def embed_impl(
        self,
        input_path: Path,
        output_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def extract_impl(
        self,
        input_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError

    def _supports_embed_batch(self) -> bool:
        return type(self).embed_batch_impl is not BaseWatermark.embed_batch_impl

    def _supports_extract_batch(self) -> bool:
        return type(self).extract_batch_impl is not BaseWatermark.extract_batch_impl

    def _batch_config(self, stage: str):
        stage_key = str(stage).lower()
        if stage_key in {"embed", "extract"}:
            specific = resolve_named_batch_size(
                str(self.name),
                params=self.params,
                param_key=f"watermark_{stage_key}_batch_size",
                overrides_env=f"WM_BENCH_WATERMARK_{stage_key.upper()}_BATCH_SIZES",
                global_env=f"WM_BENCH_WATERMARK_{stage_key.upper()}_BATCH_SIZE",
                default=0,
            )
            if specific.raw is not None or specific.source != "default":
                return specific
        return resolve_named_batch_size(
            str(self.name),
            params=self.params,
            param_key="watermark_batch_size",
            overrides_env="WM_BENCH_WATERMARK_BATCH_SIZES",
            global_env="WM_BENCH_WATERMARK_BATCH_SIZE",
            default=4,
        )

    def _cpu_worker_config(self, job_count: int):
        return resolve_named_cpu_workers(
            str(self.name),
            overrides_env="WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD",
            global_env="WM_BENCH_WATERMARK_CPU_WORKERS",
            job_count=job_count,
            enabled=bool(self.thread_safe_parallel),
            default_cap=8,
        )

    def _execution_profile(
        self,
        *,
        stage: str,
        mode: str,
        job_count: int,
        device: str | None = None,
        cpu_workers: int = 1,
        configured_batch_size: int | None = None,
        actual_batch_size: int | None = None,
        supports_batch: bool | None = None,
        fallback: bool = False,
        fallback_reason: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> ExecutionProfile:
        return ExecutionProfile(
            stage=stage,
            method=str(self.name),
            mode=mode,
            job_count=job_count,
            device=device,
            cpu_workers=cpu_workers,
            configured_batch_size=configured_batch_size,
            actual_batch_size=actual_batch_size,
            batch_stage=stage,
            supports_batch=supports_batch,
            thread_safe_parallel=bool(self.thread_safe_parallel),
            fallback=fallback,
            fallback_reason=fallback_reason,
            config=config or {},
        )

    def _embed_protocol_metadata(
        self,
        input_path: Path,
        output_path: Path,
        metadata: Mapping[str, Any],
    ) -> JsonDict:
        enriched = dict(metadata)
        input_size = image_size(input_path)
        pre_output_size = image_size(output_path)
        if input_size is None and pre_output_size is None:
            return enriched

        canonicalized = canonicalize_image_file_in_place(output_path)
        output_size = canonicalized.get("outputSize") or image_size(output_path)
        internal_size = first_metadata_size(
            enriched,
            ("internalSize", "internal_size", "image_size", "output_size", "outputSize"),
        )
        if internal_size is None:
            internal_size = pre_output_size or output_size

        enriched.update(
            {
                "inputSize": input_size,
                "internalSize": internal_size,
                "preCanonicalOutputSize": pre_output_size,
                "outputSize": output_size,
                "canonicalSize": list(CANONICAL_IMAGE_SIZE),
                "outputSizePolicy": canonicalized.get("outputSizePolicy"),
                "canonicalizedOutput": bool(canonicalized.get("changed")),
            }
        )
        return enriched

    def _extract_protocol_metadata(
        self,
        input_path: Path,
        metadata: Mapping[str, Any],
    ) -> JsonDict:
        enriched = dict(metadata)
        decode_input_size = image_size(input_path)
        if decode_input_size is None:
            return enriched
        decode_internal_size = first_metadata_size(
            enriched,
            (
                "decodeInternalSize",
                "decode_internal_size",
                "internalSize",
                "internal_size",
                "image_size",
            ),
        )
        enriched.update(
            {
                "decodeInputSize": decode_input_size,
                "decodeInternalSize": decode_internal_size or decode_input_size,
            }
        )
        return enriched

    def embed_many(
        self,
        jobs: Iterable[tuple[str | Path, str | Path, WatermarkContext]],
    ) -> list[WatermarkEmbedResult]:
        normalized = [(Path(input_path), Path(output_path), context) for input_path, output_path, context in jobs]
        if not self._supports_embed_batch():
            worker_config = self._cpu_worker_config(len(normalized))
            workers = worker_config.value
            mode = "threadpool" if workers > 1 else "serial"
            profile = self._execution_profile(
                stage="watermark_embed",
                mode=mode,
                job_count=len(normalized),
                device=normalized[0][2].device if normalized else None,
                cpu_workers=workers,
                supports_batch=False,
                config={"cpuWorkers": worker_config.to_json()},
            )
            if workers > 1:
                def run_one(job: tuple[Path, Path, WatermarkContext]) -> WatermarkEmbedResult:
                    input_path, output_path, context = job
                    return self.embed(input_path, output_path, context)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    return [replace_result_execution(result, profile) for result in executor.map(run_one, normalized)]
            return [
                replace_result_execution(self.embed(input_path, output_path, context), profile)
                for input_path, output_path, context in normalized
            ]

        results: list[WatermarkEmbedResult] = []
        batch_config = self._batch_config("embed")
        batch_size = batch_config.value
        for offset in range(0, len(normalized), batch_size):
            chunk = normalized[offset : offset + batch_size]
            for _input_path, output_path, _context in chunk:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            profile = self._execution_profile(
                stage="watermark_embed",
                mode="batch",
                job_count=len(normalized),
                device=chunk[0][2].device if chunk else None,
                configured_batch_size=batch_size,
                actual_batch_size=len(chunk),
                supports_batch=True,
                config={"batchSize": batch_config.to_json()},
            )
            started = time.perf_counter()
            try:
                metadatas = [dict(metadata) for metadata in self.embed_batch_impl(chunk)]
                if len(metadatas) != len(chunk):
                    raise ValueError(
                        f"embed_batch_impl returned {len(metadatas)} results for {len(chunk)} jobs"
                    )
            except Exception as exc:
                fallback_profile = self._execution_profile(
                    stage="watermark_embed",
                    mode="batch_fallback_serial",
                    job_count=len(normalized),
                    device=chunk[0][2].device if chunk else None,
                    configured_batch_size=batch_size,
                    actual_batch_size=len(chunk),
                    supports_batch=True,
                    fallback=True,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    config={"batchSize": batch_config.to_json()},
                )
                results.extend(
                    replace_result_execution(self.embed(input_path, output_path, context), fallback_profile)
                    for input_path, output_path, context in chunk
                )
                continue

            result_payloads: list[tuple[Path, Path, WatermarkContext, JsonDict]] = []
            for (input_path, output_path, context), metadata in zip(chunk, metadatas):
                metadata = self._embed_protocol_metadata(input_path, output_path, metadata)
                metadata = attach_execution_metadata(metadata, profile)
                result_payloads.append((input_path, output_path, context, metadata))
            elapsed_ms = ((time.perf_counter() - started) * 1000) / max(1, len(chunk))
            for input_path, output_path, context, metadata in result_payloads:
                results.append(
                    WatermarkEmbedResult(
                        input_path=input_path,
                        output_path=output_path,
                        method_name=self.name,
                        message=context.message,
                        params=self.params,
                        elapsed_ms=elapsed_ms,
                        ok=True,
                        error=None,
                        metadata=metadata,
                    )
                )
        return results

    def extract_many(
        self,
        jobs: Iterable[tuple[str | Path, WatermarkContext]],
    ) -> list[WatermarkExtractResult]:
        normalized = [(Path(input_path), context) for input_path, context in jobs]
        if not self._supports_extract_batch():
            worker_config = self._cpu_worker_config(len(normalized))
            workers = worker_config.value
            mode = "threadpool" if workers > 1 else "serial"
            profile = self._execution_profile(
                stage="watermark_extract",
                mode=mode,
                job_count=len(normalized),
                device=normalized[0][1].device if normalized else None,
                cpu_workers=workers,
                supports_batch=False,
                config={"cpuWorkers": worker_config.to_json()},
            )
            if workers > 1:
                def run_one(job: tuple[Path, WatermarkContext]) -> WatermarkExtractResult:
                    input_path, context = job
                    return self.extract(input_path, context)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    return [replace_result_execution(result, profile) for result in executor.map(run_one, normalized)]
            return [
                replace_result_execution(self.extract(input_path, context), profile)
                for input_path, context in normalized
            ]

        results: list[WatermarkExtractResult] = []
        batch_config = self._batch_config("extract")
        batch_size = batch_config.value
        for offset in range(0, len(normalized), batch_size):
            chunk = normalized[offset : offset + batch_size]
            profile = self._execution_profile(
                stage="watermark_extract",
                mode="batch",
                job_count=len(normalized),
                device=chunk[0][1].device if chunk else None,
                configured_batch_size=batch_size,
                actual_batch_size=len(chunk),
                supports_batch=True,
                config={"batchSize": batch_config.to_json()},
            )
            started = time.perf_counter()
            try:
                metadatas = [dict(metadata) for metadata in self.extract_batch_impl(chunk)]
                if len(metadatas) != len(chunk):
                    raise ValueError(
                        f"extract_batch_impl returned {len(metadatas)} results for {len(chunk)} jobs"
                    )
            except Exception as exc:
                fallback_profile = self._execution_profile(
                    stage="watermark_extract",
                    mode="batch_fallback_serial",
                    job_count=len(normalized),
                    device=chunk[0][1].device if chunk else None,
                    configured_batch_size=batch_size,
                    actual_batch_size=len(chunk),
                    supports_batch=True,
                    fallback=True,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    config={"batchSize": batch_config.to_json()},
                )
                results.extend(
                    replace_result_execution(self.extract(input_path, context), fallback_profile)
                    for input_path, context in chunk
                )
                continue

            elapsed_ms = ((time.perf_counter() - started) * 1000) / max(1, len(chunk))
            for (input_path, _context), metadata in zip(chunk, metadatas):
                message = metadata.pop("message", None)
                bits = metadata.pop("bits", None)
                metadata = self._extract_protocol_metadata(input_path, metadata)
                metadata = attach_execution_metadata(metadata, profile)
                results.append(
                    WatermarkExtractResult(
                        input_path=input_path,
                        method_name=self.name,
                        params=self.params,
                        elapsed_ms=elapsed_ms,
                        ok=True,
                        error=None,
                        message=message,
                        bits=bits,
                        metadata=metadata,
                    )
                )
        return results

    def embed(
        self,
        input_path: str | Path,
        output_path: str | Path,
        context: WatermarkContext,
    ) -> WatermarkEmbedResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        profile = self._execution_profile(
            stage="watermark_embed",
            mode="serial",
            job_count=1,
            device=context.device,
            cpu_workers=1,
            supports_batch=self._supports_embed_batch(),
        )
        try:
            metadata = dict(self.embed_impl(input_path, output_path, context))
            metadata = self._embed_protocol_metadata(input_path, output_path, metadata)
            ok = True
            error = None
        except Exception as exc:
            metadata = {}
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        metadata = attach_execution_metadata(metadata, profile)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return WatermarkEmbedResult(
            input_path=input_path,
            output_path=output_path,
            method_name=self.name,
            message=context.message,
            params=self.params,
            elapsed_ms=elapsed_ms,
            ok=ok,
            error=error,
            metadata=metadata,
        )

    def extract(
        self,
        input_path: str | Path,
        context: WatermarkContext,
    ) -> WatermarkExtractResult:
        input_path = Path(input_path)
        started = time.perf_counter()
        profile = self._execution_profile(
            stage="watermark_extract",
            mode="serial",
            job_count=1,
            device=context.device,
            cpu_workers=1,
            supports_batch=self._supports_extract_batch(),
        )
        try:
            metadata = dict(self.extract_impl(input_path, context))
            message = metadata.pop("message", None)
            bits = metadata.pop("bits", None)
            metadata = self._extract_protocol_metadata(input_path, metadata)
            ok = True
            error = None
        except Exception as exc:
            metadata = {}
            message = None
            bits = None
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        metadata = attach_execution_metadata(metadata, profile)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return WatermarkExtractResult(
            input_path=input_path,
            method_name=self.name,
            params=self.params,
            elapsed_ms=elapsed_ms,
            ok=ok,
            error=error,
            message=message,
            bits=bits,
            metadata=metadata,
        )

    @staticmethod
    def write_embed_manifest(path: str | Path, results: list[WatermarkEmbedResult]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([result.to_json() for result in results], f, ensure_ascii=False, indent=2)

    @staticmethod
    def write_extract_manifest(path: str | Path, results: list[WatermarkExtractResult]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([result.to_json() for result in results], f, ensure_ascii=False, indent=2)
