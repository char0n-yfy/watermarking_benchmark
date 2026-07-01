from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonDict = dict[str, Any]



def _parse_batch_size_overrides(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    overrides: dict[str, int] = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().lower()
        try:
            parsed = int(value.strip())
        except ValueError:
            continue
        if key and parsed > 0:
            overrides[key] = parsed
    return overrides


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

    def _batch_size(self) -> int:
        overrides = _parse_batch_size_overrides(os.getenv("WM_BENCH_WATERMARK_BATCH_SIZES"))
        raw = (
            self.params.get("watermark_batch_size")
            or overrides.get(str(self.name).lower())
            or os.getenv("WM_BENCH_WATERMARK_BATCH_SIZE", "4")
        )
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 4

    def embed_many(
        self,
        jobs: Iterable[tuple[str | Path, str | Path, WatermarkContext]],
    ) -> list[WatermarkEmbedResult]:
        normalized = [(Path(input_path), Path(output_path), context) for input_path, output_path, context in jobs]
        if not self._supports_embed_batch():
            return [self.embed(input_path, output_path, context) for input_path, output_path, context in normalized]

        results: list[WatermarkEmbedResult] = []
        batch_size = self._batch_size()
        for offset in range(0, len(normalized), batch_size):
            chunk = normalized[offset : offset + batch_size]
            for _input_path, output_path, _context in chunk:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                metadatas = [dict(metadata) for metadata in self.embed_batch_impl(chunk)]
                if len(metadatas) != len(chunk):
                    raise ValueError(
                        f"embed_batch_impl returned {len(metadatas)} results for {len(chunk)} jobs"
                    )
            except NotImplementedError:
                results.extend(self.embed(input_path, output_path, context) for input_path, output_path, context in chunk)
                continue
            except Exception:
                results.extend(self.embed(input_path, output_path, context) for input_path, output_path, context in chunk)
                continue

            elapsed_ms = ((time.perf_counter() - started) * 1000) / max(1, len(chunk))
            for (input_path, output_path, context), metadata in zip(chunk, metadatas):
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
            return [self.extract(input_path, context) for input_path, context in normalized]

        results: list[WatermarkExtractResult] = []
        batch_size = self._batch_size()
        for offset in range(0, len(normalized), batch_size):
            chunk = normalized[offset : offset + batch_size]
            started = time.perf_counter()
            try:
                metadatas = [dict(metadata) for metadata in self.extract_batch_impl(chunk)]
                if len(metadatas) != len(chunk):
                    raise ValueError(
                        f"extract_batch_impl returned {len(metadatas)} results for {len(chunk)} jobs"
                    )
            except NotImplementedError:
                results.extend(self.extract(input_path, context) for input_path, context in chunk)
                continue
            except Exception:
                results.extend(self.extract(input_path, context) for input_path, context in chunk)
                continue

            elapsed_ms = ((time.perf_counter() - started) * 1000) / max(1, len(chunk))
            for (input_path, _context), metadata in zip(chunk, metadatas):
                message = metadata.pop("message", None)
                bits = metadata.pop("bits", None)
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
        try:
            metadata = dict(self.embed_impl(input_path, output_path, context))
            ok = True
            error = None
        except Exception as exc:
            metadata = {}
            ok = False
            error = f"{type(exc).__name__}: {exc}"
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
        try:
            metadata = dict(self.extract_impl(input_path, context))
            message = metadata.pop("message", None)
            bits = metadata.pop("bits", None)
            ok = True
            error = None
        except Exception as exc:
            metadata = {}
            message = None
            bits = None
            ok = False
            error = f"{type(exc).__name__}: {exc}"
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
