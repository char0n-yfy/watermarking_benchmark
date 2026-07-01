from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _setup_imports() -> None:
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_setup_imports()

from evaluator.image_protocol import canonical_preprocess_image  # noqa: E402
from evaluator.watermarking import WATERMARK_REGISTRY  # noqa: E402
from evaluator.watermarking.base import BaseWatermark  # noqa: E402
from evaluator.watermarking.registry import build_watermark  # noqa: E402
from evaluator.watermarking.runner import (  # noqa: E402
    WatermarkEmbedJob,
    WatermarkExtractJob,
    run_watermark_embed_dir_with_method,
    run_watermark_extract_dir_with_method,
)


def _configure_runtime() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _prepare_inputs(source_root: Path, target_dir: Path, count: int) -> list[Path]:
    if target_dir.exists():
        existing = _image_paths(target_dir)
        if len(existing) >= count:
            return existing[:count]
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    sources = _image_paths(source_root)
    if len(sources) < count:
        raise ValueError(f"Need at least {count} images under {source_root}, found {len(sources)}")

    outputs: list[Path] = []
    for index, source in enumerate(sources[:count]):
        output = target_dir / f"sample_{index:04d}.png"
        canonical_preprocess_image(source, output)
        outputs.append(output)
    return outputs


def _subset_dir(source_dir: Path, target_dir: Path, count: int) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(_image_paths(source_dir)[:count]):
        shutil.copy2(source, target_dir / f"sample_{index:04d}{source.suffix.lower()}")
    return target_dir


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _set_stage_batch(method: str, stage: str, batch_size: int) -> None:
    os.environ[f"WM_BENCH_WATERMARK_{stage.upper()}_BATCH_SIZES"] = f"{method}={batch_size}"
    os.environ[f"WM_BENCH_WATERMARK_{stage.upper()}_BATCH_SIZE"] = str(batch_size)


def _clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _result_error(results: list[Any]) -> str | None:
    errors = [str(getattr(result, "error", "")) for result in results if getattr(result, "error", None)]
    if errors:
        return "; ".join(errors[:3])
    return None


def _run_embed(
    *,
    method_obj: BaseWatermark,
    method: str,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    sample_count: int,
) -> dict[str, Any]:
    _set_stage_batch(method, "embed", batch_size)
    _clean_dir(output_dir)
    started = time.perf_counter()
    results = run_watermark_embed_dir_with_method(
        WatermarkEmbedJob(
            run_id="watermark_batch_tuning",
            method_name=method,
            params=method_obj.params,
            input_dir=input_dir,
            output_dir=output_dir,
            message="test_watermark_001",
            device=device,
            seed=2026,
        ),
        method_obj,
    )
    elapsed = time.perf_counter() - started
    ok = len(results) == sample_count and all(result.ok for result in results)
    error = _result_error(results)
    return {
        "batchSize": batch_size,
        "elapsedSeconds": elapsed,
        "imagesPerSecond": (sample_count / elapsed) if ok and elapsed > 0 else None,
        "ok": ok,
        "resultCount": len(results),
        "error": error,
        "outputDir": str(output_dir),
    }


def _run_extract(
    *,
    method_obj: BaseWatermark,
    method: str,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    sample_count: int,
) -> dict[str, Any]:
    _set_stage_batch(method, "extract", batch_size)
    _clean_dir(output_dir)
    started = time.perf_counter()
    results = run_watermark_extract_dir_with_method(
        WatermarkExtractJob(
            run_id="watermark_batch_tuning",
            method_name=method,
            params=method_obj.params,
            input_dir=input_dir,
            output_dir=output_dir,
            message="test_watermark_001",
            device=device,
            seed=2026,
        ),
        method_obj,
    )
    elapsed = time.perf_counter() - started
    ok = len(results) == sample_count and all(result.ok for result in results)
    error = _result_error(results)
    return {
        "batchSize": batch_size,
        "elapsedSeconds": elapsed,
        "imagesPerSecond": (sample_count / elapsed) if ok and elapsed > 0 else None,
        "ok": ok,
        "resultCount": len(results),
        "error": error,
        "outputDir": str(output_dir),
    }


def _best(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [entry for entry in entries if entry.get("ok") and entry.get("imagesPerSecond") is not None]
    if not valid:
        return None
    return max(valid, key=lambda entry: float(entry["imagesPerSecond"]))


def _oom_like(error: str) -> bool:
    lowered = error.lower()
    return "out of memory" in lowered or "cuda error" in lowered or "cudnn" in lowered


def _benchmark_method(
    *,
    method: str,
    input_dir: Path,
    warmup_dir: Path,
    work_dir: Path,
    candidates: list[int],
    device: str,
    sample_count: int,
) -> dict[str, Any]:
    cls = WATERMARK_REGISTRY[method]
    supports_embed = cls.embed_batch_impl is not BaseWatermark.embed_batch_impl
    supports_extract = cls.extract_batch_impl is not BaseWatermark.extract_batch_impl
    record: dict[str, Any] = {
        "method": method,
        "supportsEmbedBatch": supports_embed,
        "supportsExtractBatch": supports_extract,
        "embed": [],
        "extract": [],
        "status": "pending",
    }

    try:
        method_obj = build_watermark(method)
    except Exception as exc:
        record["status"] = "build_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=8)
        return record

    try:
        if supports_embed:
            try:
                _set_stage_batch(method, "embed", 1)
                warmup_output = work_dir / method / "warmup_embed"
                _run_embed(
                    method_obj=method_obj,
                    method=method,
                    input_dir=warmup_dir,
                    output_dir=warmup_output,
                    batch_size=1,
                    device=device,
                    sample_count=len(_image_paths(warmup_dir)),
                )
            except Exception:
                _clear_torch_cache()

            for batch_size in candidates:
                output_dir = work_dir / method / "embed" / f"batch_{batch_size}"
                try:
                    entry = _run_embed(
                        method_obj=method_obj,
                        method=method,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        batch_size=batch_size,
                        device=device,
                        sample_count=sample_count,
                    )
                except Exception as exc:
                    entry = {
                        "batchSize": batch_size,
                        "elapsedSeconds": None,
                        "imagesPerSecond": None,
                        "ok": False,
                        "resultCount": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                        "outputDir": str(output_dir),
                    }
                record["embed"].append(entry)
                print(
                    f"{method} embed batch={batch_size} "
                    f"ips={entry.get('imagesPerSecond')} ok={entry.get('ok')}",
                    flush=True,
                )
                _clear_torch_cache()
                if entry.get("error") and _oom_like(str(entry["error"])):
                    break

        best_embed = _best(record["embed"])
        extract_input_dir = input_dir
        if best_embed and best_embed.get("outputDir"):
            extract_input_dir = Path(str(best_embed["outputDir"]))

        if supports_extract:
            try:
                _set_stage_batch(method, "extract", 1)
                warmup_extract_input = (
                    _subset_dir(extract_input_dir, work_dir / method / "warmup_extract_input", len(_image_paths(warmup_dir)))
                    if extract_input_dir != input_dir
                    else warmup_dir
                )
                _run_extract(
                    method_obj=method_obj,
                    method=method,
                    input_dir=warmup_extract_input,
                    output_dir=work_dir / method / "warmup_extract",
                    batch_size=1,
                    device=device,
                    sample_count=len(_image_paths(warmup_extract_input)),
                )
            except Exception:
                _clear_torch_cache()

            for batch_size in candidates:
                output_dir = work_dir / method / "extract" / f"batch_{batch_size}"
                try:
                    entry = _run_extract(
                        method_obj=method_obj,
                        method=method,
                        input_dir=extract_input_dir,
                        output_dir=output_dir,
                        batch_size=batch_size,
                        device=device,
                        sample_count=sample_count,
                    )
                except Exception as exc:
                    entry = {
                        "batchSize": batch_size,
                        "elapsedSeconds": None,
                        "imagesPerSecond": None,
                        "ok": False,
                        "resultCount": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                        "outputDir": str(output_dir),
                    }
                record["extract"].append(entry)
                print(
                    f"{method} extract batch={batch_size} "
                    f"ips={entry.get('imagesPerSecond')} ok={entry.get('ok')}",
                    flush=True,
                )
                _clear_torch_cache()
                if entry.get("error") and _oom_like(str(entry["error"])):
                    break

        record["bestEmbed"] = _best(record["embed"])
        record["bestExtract"] = _best(record["extract"])
        record["status"] = "ok"
    finally:
        del method_obj
        _clear_torch_cache()

    return record


def _parse_candidates(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    return [value for value in values if value > 0]


def main() -> int:
    _configure_runtime()
    parser = argparse.ArgumentParser(description="Tune watermark embed/extract batch sizes by throughput.")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "resources" / "datasets")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "watermark_batch_tuning")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--candidates", default="1,2,4,8,16,32")
    parser.add_argument("--device", default=os.getenv("WM_BENCH_DEVICE", "cuda:0"))
    parser.add_argument("--methods", default="")
    args = parser.parse_args()

    candidates = _parse_candidates(args.candidates)
    output_root = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    input_dir = output_root / "canonical_inputs"
    warmup_dir = output_root / "warmup_inputs"
    work_dir = output_root / "work"
    result_path = output_root / "watermark_batch_tuning_results.json"
    output_root.mkdir(parents=True, exist_ok=True)

    _prepare_inputs(args.source_root, input_dir, args.sample_count)
    _subset_dir(input_dir, warmup_dir, min(args.warmup_count, args.sample_count))

    if args.methods.strip():
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    else:
        methods = sorted(WATERMARK_REGISTRY)

    results: list[dict[str, Any]] = []
    for method in methods:
        if method not in WATERMARK_REGISTRY:
            print(f"skip unknown method: {method}", flush=True)
            continue
        print(f"=== {method} ===", flush=True)
        record = _benchmark_method(
            method=method,
            input_dir=input_dir,
            warmup_dir=warmup_dir,
            work_dir=work_dir,
            candidates=[candidate for candidate in candidates if candidate <= args.sample_count],
            device=args.device,
            sample_count=args.sample_count,
        )
        results.append(record)
        result_path.write_text(
            json.dumps(
                {
                    "sampleCount": args.sample_count,
                    "candidates": candidates,
                    "device": args.device,
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    embed_overrides = []
    extract_overrides = []
    for record in results:
        best_embed = record.get("bestEmbed")
        best_extract = record.get("bestExtract")
        method = record["method"]
        if isinstance(best_embed, dict):
            embed_overrides.append(f"{method}={best_embed['batchSize']}")
        if isinstance(best_extract, dict):
            extract_overrides.append(f"{method}={best_extract['batchSize']}")

    summary = {
        "outputRoot": str(output_root),
        "resultPath": str(result_path),
        "embedOverrides": ",".join(embed_overrides),
        "extractOverrides": ",".join(extract_overrides),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
