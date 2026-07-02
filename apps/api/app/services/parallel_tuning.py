from __future__ import annotations

import gc
import json
import os
import re
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from uuid import uuid4

from evaluator.attacks import ATTACK_REGISTRY
from evaluator.attacks.runner import AttackJob, run_attack_dir_with_attack
from evaluator.image_protocol import canonical_preprocess_image
from evaluator.watermarking import WATERMARK_REGISTRY
from evaluator.watermarking.base import BaseWatermark
from evaluator.watermarking.registry import build_watermark
from evaluator.watermarking.runner import (
    WatermarkEmbedJob,
    WatermarkExtractJob,
    run_watermark_embed_dir_with_method,
    run_watermark_extract_dir_with_method,
)

from app.services.resources import iter_image_paths, scan_dataset_resources
from app.services.scoring import (
    _compute_cpu_quality_metrics_batch_with_profile,
    _compute_perceptual_metrics_batch_with_profile,
)
from app.services.runtime_parallel_config import (
    PARALLEL_TUNING_ENV_KEYS,
    apply_parallel_env_updates,
    write_runtime_parallel_env,
)


JsonDict = dict[str, Any]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIEWPOINT_RERENDERING_METHOD_PATTERN = re.compile(
    r"3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_(point|ahead)"
)
VIEWPOINT_RERENDERING_PRIMARY_METHOD = "3d_viewpoint_rerendering_rotate_point"
FIXED_ATTACK_BATCH_OVERRIDES = {
    "2x_regen": 8,
}
TERMINAL_TUNING_STATUSES = {"succeeded", "failed", "cancelled"}


class TuningCancelled(Exception):
    pass


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _positive_candidates(values: Iterable[Any]) -> list[int]:
    parsed: set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            parsed.add(number)
    return sorted(parsed)


def _candidates_up_to(candidates: list[int], max_value: int) -> list[int]:
    return [candidate for candidate in candidates if candidate <= max(1, max_value)] or [1]


def _next_power_candidate(current: int, max_value: int) -> int | None:
    if current >= max_value:
        return None
    return min(max_value, max(current + 1, current * 2))


def _median_number(values: Iterable[Any]) -> float | None:
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return float(median(parsed))


def _clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _images_per_second(sample_count: int, elapsed_seconds: float, ok: bool) -> float | None:
    if not ok or elapsed_seconds <= 0:
        return None
    return sample_count / elapsed_seconds


def _best_by_throughput(entries: list[JsonDict]) -> JsonDict | None:
    valid = [entry for entry in entries if entry.get("ok") and entry.get("imagesPerSecond") is not None]
    if not valid:
        return None
    return max(valid, key=lambda entry: float(entry["imagesPerSecond"]))


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _oom_like(error: str | None) -> bool:
    lowered = (error or "").lower()
    return "out of memory" in lowered or "cuda error" in lowered or "cudnn" in lowered


def _set_named_override(env_name: str, name: str, value: int) -> None:
    os.environ[env_name] = f"{name}={value}"


def _set_stage_watermark_batch(method: str, stage: str, batch_size: int) -> None:
    os.environ[f"WM_BENCH_WATERMARK_{stage.upper()}_BATCH_SIZES"] = f"{method}={batch_size}"
    os.environ[f"WM_BENCH_WATERMARK_{stage.upper()}_BATCH_SIZE"] = str(batch_size)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def _mark_output_cleaned(entry: JsonDict, output_dir: Path | str | None) -> None:
    if output_dir is None:
        return
    entry["outputDir"] = str(output_dir)
    entry["outputDirCleaned"] = _remove_tree(Path(output_dir))


def _update_dotenv(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    next_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            next_lines.append(line)
    missing = [key for key in updates if key not in seen]
    if missing and next_lines and next_lines[-1].strip():
        next_lines.append("")
    for key in missing:
        next_lines.append(f"{key}={updates[key]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


@dataclass
class TuningRequest:
    mode: str = "quick"
    sample_count: int = 16
    warmup_count: int = 2
    batch_candidates: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    worker_candidates: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 24, 32])
    repeat_count: int = 1
    auto_expand_candidates: bool = False
    max_batch_size: int = 16
    max_worker_count: int = 32
    min_improvement_ratio: float = 0.03
    boundary_patience: int = 1
    tune_watermarks: bool = True
    tune_attacks: bool = True
    tune_quality: bool = True
    watermark_methods: list[str] = field(default_factory=list)
    attack_methods: list[str] = field(default_factory=list)
    include_viewpoint_3d_attacks: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "TuningRequest":
        payload = payload or {}
        mode = str(payload.get("mode") or "quick")
        full_mode = mode == "full"
        default_samples = 64 if full_mode else 16
        default_batches = [1, 2, 4, 8, 16, 32, 64] if full_mode else [1, 2, 4, 8, 16]
        default_workers = [1, 2, 4, 8, 16, 24, 32, 48, 64] if full_mode else [1, 2, 4, 8, 16, 24, 32]
        batch_candidates = _positive_candidates(payload.get("batchCandidates") or default_batches)
        worker_candidates = _positive_candidates(payload.get("workerCandidates") or default_workers)
        max_batch_size = max(max(batch_candidates or [1]), int(payload.get("maxBatchSize") or max(batch_candidates or [1])))
        max_worker_count = max(max(worker_candidates or [1]), int(payload.get("maxWorkerCount") or max(worker_candidates or [1])))
        auto_expand = bool(payload.get("autoExpandCandidates", full_mode))
        sample_count = max(2, int(payload.get("sampleCount") or default_samples), max(batch_candidates or [1]))
        if auto_expand:
            sample_count = max(sample_count, max_batch_size)
        return cls(
            mode=mode,
            sample_count=sample_count,
            warmup_count=max(1, int(payload.get("warmupCount") or 2)),
            batch_candidates=batch_candidates,
            worker_candidates=worker_candidates,
            repeat_count=max(1, int(payload.get("repeatCount") or (3 if full_mode else 1))),
            auto_expand_candidates=auto_expand,
            max_batch_size=max_batch_size,
            max_worker_count=max_worker_count,
            min_improvement_ratio=max(0.0, float(payload.get("minImprovementRatio") or 0.03)),
            boundary_patience=max(1, int(payload.get("boundaryPatience") or (2 if full_mode else 1))),
            tune_watermarks=bool(payload.get("tuneWatermarks", True)),
            tune_attacks=bool(payload.get("tuneAttacks", True)),
            tune_quality=bool(payload.get("tuneQuality", True)),
            watermark_methods=[str(item) for item in payload.get("watermarkMethods") or [] if str(item)],
            attack_methods=[str(item) for item in payload.get("attackMethods") or [] if str(item)],
            include_viewpoint_3d_attacks=bool(payload.get("includeViewpoint3dAttacks", False)),
        )

    def to_json(self) -> JsonDict:
        return {
            "mode": self.mode,
            "sampleCount": self.sample_count,
            "warmupCount": self.warmup_count,
            "batchCandidates": self.batch_candidates,
            "workerCandidates": self.worker_candidates,
            "repeatCount": self.repeat_count,
            "autoExpandCandidates": self.auto_expand_candidates,
            "maxBatchSize": self.max_batch_size,
            "maxWorkerCount": self.max_worker_count,
            "minImprovementRatio": self.min_improvement_ratio,
            "boundaryPatience": self.boundary_patience,
            "tuneWatermarks": self.tune_watermarks,
            "tuneAttacks": self.tune_attacks,
            "tuneQuality": self.tune_quality,
            "watermarkMethods": self.watermark_methods,
            "attackMethods": self.attack_methods,
            "includeViewpoint3dAttacks": self.include_viewpoint_3d_attacks,
        }


class ParallelTuningService:
    def __init__(self, *, resources_root: Path, runs_root: Path, device: str = "cpu") -> None:
        self.resources_root = resources_root
        self.runs_root = runs_root
        self.device = device
        self.root = runs_root / "parallel_tuning"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def start(self, payload: dict[str, Any] | None = None) -> JsonDict:
        request = TuningRequest.from_payload(payload)
        with self._lock:
            running = [job for job in self.list_jobs() if job.get("status") == "running"]
            if running:
                raise ValueError(f"Parallel tuning job already running: {running[0]['id']}")
            job_id = f"tune_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
            job_dir = self.root / job_id
            state = {
                "id": job_id,
                "status": "running",
                "progress": 0,
                "message": "preparing tuning inputs",
                "request": request.to_json(),
                "artifactRoot": str(job_dir),
                "createdAt": _utc_timestamp(),
                "updatedAt": _utc_timestamp(),
                "events": [],
                "report": None,
                "summary": None,
            }
            self._write_state(job_id, state)
            thread = threading.Thread(target=self._run_job, args=(job_id, request), daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return self.get(job_id)

    def list_jobs(self) -> list[JsonDict]:
        jobs: list[JsonDict] = []
        for state_path in sorted(self.root.glob("*/state.json"), reverse=True):
            try:
                jobs.append(json.loads(state_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return jobs

    def latest(self) -> JsonDict | None:
        jobs = self.list_jobs()
        return jobs[0] if jobs else None

    def get(self, job_id: str) -> JsonDict:
        state_path = self.root / job_id / "state.json"
        if not state_path.exists():
            raise KeyError(f"Unknown tuning job id: {job_id}")
        return json.loads(state_path.read_text(encoding="utf-8"))

    def save_parameters(self, job_id: str, env_path: Path | None = None) -> JsonDict:
        state = self.get(job_id)
        if state.get("status") != "succeeded":
            raise ValueError("Only succeeded tuning jobs can be saved")
        summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
        updates = summary.get("envUpdates") if isinstance(summary, dict) else None
        if not isinstance(updates, dict) or not updates:
            raise ValueError("Tuning job did not produce environment updates")
        target = env_path or (Path.cwd() / ".env.autodl")
        cleaned_updates = apply_parallel_env_updates(updates)
        _update_dotenv(target, cleaned_updates)
        runtime = write_runtime_parallel_env(self.runs_root, cleaned_updates, job_id=job_id, env_path=target)
        saved = {
            "jobId": job_id,
            "envPath": str(target),
            "runtimePath": runtime["path"],
            "savedKeys": sorted(cleaned_updates),
            "appliedKeys": sorted(cleaned_updates),
        }
        self._event(job_id, "saved", "saved recommended parameters", saved)
        return saved

    def cancel(self, job_id: str) -> JsonDict:
        with self._lock:
            state = self.get(job_id)
            if state.get("status") in TERMINAL_TUNING_STATUSES:
                return state
            state["status"] = "cancelled"
            state["cancelRequested"] = True
            state["finishedAt"] = state.get("finishedAt") or _utc_timestamp()
            state["message"] = "tuning cancelled"
            events = list(state.get("events") or [])
            events.append(
                {
                    "timestamp": _utc_timestamp(),
                    "stage": "cancel",
                    "message": "tuning cancellation requested",
                }
            )
            state["events"] = events[-200:]
            self._write_state(job_id, state)
            return state

    def _write_state(self, job_id: str, state: JsonDict) -> None:
        state["updatedAt"] = _utc_timestamp()
        _write_json(self.root / job_id / "state.json", state)

    def _mutate_state(self, job_id: str, **updates: Any) -> JsonDict:
        with self._lock:
            state = self.get(job_id)
            state.update(updates)
            self._write_state(job_id, state)
            return state

    def _event(self, job_id: str, stage: str, message: str, payload: JsonDict | None = None) -> None:
        with self._lock:
            state = self.get(job_id)
            events = list(state.get("events") or [])
            events.append(
                {
                    "timestamp": _utc_timestamp(),
                    "stage": stage,
                    "message": message,
                    **(payload or {}),
                }
            )
            state["events"] = events[-200:]
            state["message"] = message
            self._write_state(job_id, state)

    def _progress(self, job_id: str, completed: int, total: int, message: str) -> None:
        progress = int(round((completed / max(1, total)) * 100))
        self._mutate_state(job_id, progress=max(0, min(99, progress)), message=message)

    def _is_cancel_requested(self, job_id: str) -> bool:
        try:
            state = self.get(job_id)
        except KeyError:
            return False
        return bool(state.get("cancelRequested")) or state.get("status") == "cancelled"

    def _ensure_not_cancelled(self, job_id: str) -> None:
        if self._is_cancel_requested(job_id):
            raise TuningCancelled()

    def _viewpoint_rerendering_methods(self) -> list[str]:
        return sorted(method for method in ATTACK_REGISTRY if VIEWPOINT_RERENDERING_METHOD_PATTERN.fullmatch(method))

    def _attack_methods_for_tuning(self, request: TuningRequest) -> list[str]:
        methods = [
            method
            for method in (request.attack_methods or sorted(ATTACK_REGISTRY))
            if method not in FIXED_ATTACK_BATCH_OVERRIDES
        ]
        has_viewpoint = any(VIEWPOINT_RERENDERING_METHOD_PATTERN.fullmatch(method) for method in methods)
        if not has_viewpoint:
            return methods
        if not request.include_viewpoint_3d_attacks:
            return [method for method in methods if not VIEWPOINT_RERENDERING_METHOD_PATTERN.fullmatch(method)]

        normalized: list[str] = []
        primary_inserted = False
        for method in methods:
            if VIEWPOINT_RERENDERING_METHOD_PATTERN.fullmatch(method):
                if not primary_inserted and VIEWPOINT_RERENDERING_PRIMARY_METHOD in ATTACK_REGISTRY:
                    normalized.append(VIEWPOINT_RERENDERING_PRIMARY_METHOD)
                    primary_inserted = True
                continue
            normalized.append(method)
        return normalized

    def _cleanup_job_artifacts(self, job_id: str) -> None:
        job_dir = self.root / job_id
        for relative in ("work", "canonical_inputs", "warmup_inputs"):
            _remove_tree(job_dir / relative)

    def _cleanup_entry_outputs(self, entries: Iterable[JsonDict], *, keep_output_dirs: set[str] | None = None) -> None:
        keep = keep_output_dirs or set()
        for entry in entries:
            output_dir = entry.get("outputDir")
            if not output_dir or str(output_dir) in keep:
                continue
            _mark_output_cleaned(entry, str(output_dir))

    def _run_job(self, job_id: str, request: TuningRequest) -> None:
        job_dir = self.root / job_id
        env_snapshot = {key: os.environ.get(key) for key in PARALLEL_TUNING_ENV_KEYS}
        try:
            os.environ.setdefault("WANDB_DISABLED", "true")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            needs_inputs = request.tune_watermarks or request.tune_attacks or request.tune_quality
            if needs_inputs:
                input_dir = self._prepare_inputs(job_id, request)
                warmup_dir = self._subset_dir(input_dir, job_dir / "warmup_inputs", min(request.warmup_count, request.sample_count))
            else:
                input_dir = job_dir / "canonical_inputs"
                warmup_dir = job_dir / "warmup_inputs"
                input_dir.mkdir(parents=True, exist_ok=True)
                warmup_dir.mkdir(parents=True, exist_ok=True)
                self._event(job_id, "prepare", "no tuning stages selected; skipped input preparation")
            self._ensure_not_cancelled(job_id)
            estimated_steps = self._estimate_steps(request)
            completed_steps = 0
            report: JsonDict = {
                "jobId": job_id,
                "device": self.device,
                "request": request.to_json(),
                "inputDir": str(input_dir),
                "watermarks": [],
                "attacks": [],
                "quality": {},
                "startedAt": _utc_timestamp(),
            }

            if request.tune_watermarks:
                self._ensure_not_cancelled(job_id)
                for record in self._benchmark_watermarks(job_id, request, input_dir, warmup_dir):
                    self._ensure_not_cancelled(job_id)
                    completed_steps += int(record.get("_stepCount", 1))
                    record.pop("_stepCount", None)
                    report["watermarks"].append(record)
                    _write_json(job_dir / "report.json", report)
                    self._progress(job_id, completed_steps, estimated_steps, f"watermark {record.get('method')} tuned")

            if request.tune_attacks:
                self._ensure_not_cancelled(job_id)
                for record in self._benchmark_attacks(job_id, request, input_dir):
                    self._ensure_not_cancelled(job_id)
                    completed_steps += int(record.get("_stepCount", 1))
                    record.pop("_stepCount", None)
                    report["attacks"].append(record)
                    _write_json(job_dir / "report.json", report)
                    self._progress(job_id, completed_steps, estimated_steps, f"attack {record.get('method')} tuned")

            if request.tune_quality:
                self._ensure_not_cancelled(job_id)
                quality_record = self._benchmark_quality(job_id, request, input_dir)
                self._ensure_not_cancelled(job_id)
                completed_steps += int(quality_record.get("_stepCount", 1))
                quality_record.pop("_stepCount", None)
                report["quality"] = quality_record
                _write_json(job_dir / "report.json", report)
                self._progress(job_id, completed_steps, estimated_steps, "quality metrics tuned")

            summary = self._build_summary(report)
            report["summary"] = summary
            report["finishedAt"] = _utc_timestamp()
            _write_json(job_dir / "report.json", report)
            self._ensure_not_cancelled(job_id)
            self._mutate_state(
                job_id,
                status="succeeded",
                progress=100,
                message="tuning completed",
                report=report,
                summary=summary,
            )
        except TuningCancelled:
            self.cancel(job_id)
        except Exception as exc:
            self._mutate_state(
                job_id,
                status="failed",
                message=_error_text(exc),
                error=_error_text(exc),
                traceback=traceback.format_exc(limit=12),
            )
        finally:
            self._cleanup_job_artifacts(job_id)
            for key, value in env_snapshot.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _prepare_inputs(self, job_id: str, request: TuningRequest) -> Path:
        output_dir = self.root / job_id / "canonical_inputs"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = scan_dataset_resources(self.resources_root)
        sources: list[Path] = []
        for dataset in datasets:
            sources.extend(iter_image_paths(dataset.path))
            if len(sources) >= request.sample_count:
                break
        if len(sources) < request.sample_count:
            raise ValueError(
                f"Need at least {request.sample_count} local images for tuning, found {len(sources)}"
            )
        for index, source in enumerate(sources[: request.sample_count]):
            canonical_preprocess_image(source, output_dir / f"sample_{index:04d}.png")
        self._event(job_id, "prepare", f"prepared {request.sample_count} canonical images", {"inputDir": str(output_dir)})
        return output_dir

    def _subset_dir(self, source_dir: Path, target_dir: Path, count: int) -> Path:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(iter_image_paths(source_dir)[:count]):
            shutil.copy2(source, target_dir / f"sample_{index:04d}{source.suffix.lower()}")
        return target_dir

    def _estimate_steps(self, request: TuningRequest) -> int:
        total = 1
        batch_candidates = _candidates_up_to(request.batch_candidates, request.max_batch_size)
        worker_candidates = _candidates_up_to(request.worker_candidates, request.max_worker_count)
        multiplier = max(1, request.repeat_count)
        if request.tune_watermarks:
            methods = request.watermark_methods or sorted(WATERMARK_REGISTRY)
            total += len(methods) * max(1, len(batch_candidates)) * multiplier
        if request.tune_attacks:
            methods = self._attack_methods_for_tuning(request)
            total += len(methods) * max(1, len(batch_candidates)) * multiplier
        if request.tune_quality:
            total += (len(worker_candidates) + len(batch_candidates)) * multiplier
        return max(1, total)

    def _measure_repeated(self, repeat_count: int, value_key: str, run_once: Any, should_cancel: Any | None = None) -> JsonDict:
        repetitions: list[JsonDict] = []
        for repeat_index in range(repeat_count):
            if should_cancel is not None and should_cancel():
                raise TuningCancelled()
            entry = dict(run_once())
            if should_cancel is not None and should_cancel():
                raise TuningCancelled()
            entry["repeatIndex"] = repeat_index + 1
            repetitions.append(entry)
            if _oom_like(entry.get("error")) or not entry.get("ok"):
                break

        last = dict(repetitions[-1]) if repetitions else {}
        ok_entries = [entry for entry in repetitions if entry.get("ok") and entry.get("imagesPerSecond") is not None]
        ips = _median_number(entry.get("imagesPerSecond") for entry in ok_entries)
        elapsed = _median_number(entry.get("elapsedSeconds") for entry in ok_entries)
        errors = [str(entry.get("error")) for entry in repetitions if entry.get("error")]
        aggregate: JsonDict = {
            **last,
            "ok": len(ok_entries) == repeat_count,
            "repeatCount": len(repetitions),
            "targetRepeatCount": repeat_count,
            "stable": len(ok_entries) == repeat_count,
            "imagesPerSecond": ips,
            "elapsedSeconds": elapsed,
            "repetitions": repetitions,
        }
        if value_key in last:
            aggregate[value_key] = last[value_key]
        if errors:
            aggregate["error"] = "; ".join(errors)
        return aggregate

    def _search_numeric_candidates(
        self,
        *,
        job_id: str,
        request: TuningRequest,
        initial_candidates: list[int],
        max_value: int,
        value_key: str,
        run_once: Any,
        on_entry: Any,
    ) -> list[JsonDict]:
        schedule = _candidates_up_to(initial_candidates, max_value)
        entries: list[JsonDict] = []
        index = 0
        while index < len(schedule):
            self._ensure_not_cancelled(job_id)
            candidate = schedule[index]
            previous_best = _best_by_throughput(entries)
            entry = self._measure_repeated(
                request.repeat_count,
                value_key,
                lambda candidate=candidate: run_once(candidate),
                should_cancel=lambda: self._is_cancel_requested(job_id),
            )
            self._ensure_not_cancelled(job_id)
            entries.append(entry)
            on_entry(entry)
            if _oom_like(entry.get("error")) or not entry.get("ok"):
                break

            current_ips = float(entry.get("imagesPerSecond") or 0)
            previous_best_ips = float(previous_best.get("imagesPerSecond") or 0) if previous_best else 0.0
            meaningful_gain = previous_best is None or current_ips >= previous_best_ips * (1 + request.min_improvement_ratio)
            best_now = _best_by_throughput(entries)
            best_value = int(best_now.get(value_key) or 0) if best_now else 0

            if request.auto_expand_candidates and index == len(schedule) - 1 and best_value == candidate and meaningful_gain:
                next_candidate = _next_power_candidate(candidate, max_value)
                if next_candidate is not None and next_candidate not in schedule:
                    schedule.append(next_candidate)

            if request.auto_expand_candidates and best_now and best_value != candidate:
                best_ips = float(best_now.get("imagesPerSecond") or 0)
                tail = 0
                for tail_entry in reversed(entries):
                    tail_ips = float(tail_entry.get("imagesPerSecond") or 0)
                    if tail_entry.get("ok") and tail_ips < best_ips * (1 + request.min_improvement_ratio):
                        tail += 1
                    else:
                        break
                if tail >= request.boundary_patience:
                    break

            index += 1
        return entries

    def _benchmark_watermarks(
        self,
        job_id: str,
        request: TuningRequest,
        input_dir: Path,
        warmup_dir: Path,
    ) -> Iterable[JsonDict]:
        candidates = _candidates_up_to(request.batch_candidates, request.max_batch_size)
        worker_candidates = _candidates_up_to(request.worker_candidates, request.max_worker_count)
        methods = request.watermark_methods or sorted(WATERMARK_REGISTRY)
        for method in methods:
            if method not in WATERMARK_REGISTRY:
                yield {"method": method, "status": "unknown_method", "_stepCount": 1}
                continue
            cls = WATERMARK_REGISTRY[method]
            supports_embed = cls.embed_batch_impl is not BaseWatermark.embed_batch_impl
            supports_extract = cls.extract_batch_impl is not BaseWatermark.extract_batch_impl
            thread_safe = bool(getattr(cls, "thread_safe_parallel", False))
            record: JsonDict = {
                "method": method,
                "supportsEmbedBatch": supports_embed,
                "supportsExtractBatch": supports_extract,
                "threadSafeParallel": thread_safe,
                "embed": [],
                "extract": [],
                "cpuWorkers": [],
                "status": "pending",
                "_stepCount": 0,
            }
            try:
                method_obj = build_watermark(method)
            except Exception as exc:
                record.update({"status": "build_failed", "error": _error_text(exc), "_stepCount": 1})
                yield record
                continue
            try:
                if supports_embed:
                    self._warmup_watermark_embed(method_obj, method, warmup_dir, job_id)

                    def remember_embed(entry: JsonDict) -> None:
                        entry["method"] = method
                        entry["stage"] = "watermark_embed"
                        record["embed"].append(entry)
                        record["_stepCount"] += int(entry.get("repeatCount") or 1)
                        best_embed = _best_by_throughput(record["embed"])
                        keep = {str(best_embed["outputDir"])} if best_embed and best_embed.get("outputDir") else set()
                        self._cleanup_entry_outputs(record["embed"], keep_output_dirs=keep)
                        self._event(job_id, "watermark_embed", f"{method} embed batch={entry.get('batchSize')}", entry)

                    self._search_numeric_candidates(
                        job_id=job_id,
                        request=request,
                        initial_candidates=candidates,
                        max_value=request.max_batch_size,
                        value_key="batchSize",
                        run_once=lambda batch_size: self._run_watermark_embed(method_obj, method, input_dir, batch_size, job_id),
                        on_entry=remember_embed,
                    )
                if supports_extract:
                    best_embed = _best_by_throughput(record["embed"])
                    extract_input = Path(str(best_embed["outputDir"])) if best_embed and best_embed.get("outputDir") else input_dir
                    self._warmup_watermark_extract(method_obj, method, extract_input, warmup_dir, job_id)

                    def remember_extract(entry: JsonDict) -> None:
                        entry["method"] = method
                        entry["stage"] = "watermark_extract"
                        record["extract"].append(entry)
                        record["_stepCount"] += int(entry.get("repeatCount") or 1)
                        _mark_output_cleaned(entry, entry.get("outputDir"))
                        self._event(job_id, "watermark_extract", f"{method} extract batch={entry.get('batchSize')}", entry)

                    self._search_numeric_candidates(
                        job_id=job_id,
                        request=request,
                        initial_candidates=candidates,
                        max_value=request.max_batch_size,
                        value_key="batchSize",
                        run_once=lambda batch_size: self._run_watermark_extract(method_obj, method, extract_input, batch_size, job_id),
                        on_entry=remember_extract,
                    )
                if thread_safe and not supports_embed and not supports_extract:

                    def remember_cpu(entry: JsonDict) -> None:
                        entry["method"] = method
                        entry["stage"] = "watermark_cpu"
                        record["cpuWorkers"].append(entry)
                        record["_stepCount"] += int(entry.get("repeatCount") or 1)
                        self._event(job_id, "watermark_cpu", f"{method} workers={entry.get('workers')}", entry)

                    self._search_numeric_candidates(
                        job_id=job_id,
                        request=request,
                        initial_candidates=worker_candidates,
                        max_value=request.max_worker_count,
                        value_key="workers",
                        run_once=lambda workers: self._run_watermark_cpu(method_obj, method, input_dir, workers, job_id),
                        on_entry=remember_cpu,
                    )
                record["bestEmbed"] = _best_by_throughput(record["embed"])
                record["bestExtract"] = _best_by_throughput(record["extract"])
                record["bestCpuWorkers"] = _best_by_throughput(record["cpuWorkers"])
                record["status"] = "ok"
            finally:
                self._cleanup_entry_outputs(record["embed"])
                self._cleanup_entry_outputs(record["extract"])
                self._cleanup_entry_outputs(record["cpuWorkers"])
                _remove_tree(self.root / job_id / "work" / "watermark" / method)
                del method_obj
                _clear_torch_cache()
            yield record

    def _run_watermark_embed(
        self,
        method_obj: BaseWatermark,
        method: str,
        input_dir: Path,
        batch_size: int,
        job_id: str,
        *,
        keep_output: bool = True,
    ) -> JsonDict:
        output_dir = self.root / job_id / "work" / "watermark" / method / "embed" / f"batch_{batch_size}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _set_stage_watermark_batch(method, "embed", batch_size)
        started = time.perf_counter()
        try:
            results = run_watermark_embed_dir_with_method(
                WatermarkEmbedJob(
                    run_id=job_id,
                    method_name=method,
                    params=method_obj.params,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    message="test_watermark_001",
                    device=self.device,
                    seed=2026,
                ),
                method_obj,
            )
            elapsed = time.perf_counter() - started
            ok = len(results) == len(iter_image_paths(input_dir)) and all(result.ok for result in results)
            error = "; ".join(str(result.error) for result in results if getattr(result, "error", None)) or None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            ok = False
            error = _error_text(exc)
        entry = {
            "batchSize": batch_size,
            "elapsedSeconds": elapsed,
            "imagesPerSecond": _images_per_second(len(iter_image_paths(input_dir)), elapsed, ok),
            "ok": ok,
            "error": error,
            "outputDir": str(output_dir),
        }
        if not keep_output:
            _mark_output_cleaned(entry, output_dir)
        return entry

    def _run_watermark_extract(self, method_obj: BaseWatermark, method: str, input_dir: Path, batch_size: int, job_id: str) -> JsonDict:
        output_dir = self.root / job_id / "work" / "watermark" / method / "extract" / f"batch_{batch_size}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _set_stage_watermark_batch(method, "extract", batch_size)
        started = time.perf_counter()
        try:
            results = run_watermark_extract_dir_with_method(
                WatermarkExtractJob(
                    run_id=job_id,
                    method_name=method,
                    params=method_obj.params,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    message="test_watermark_001",
                    device=self.device,
                    seed=2026,
                ),
                method_obj,
            )
            elapsed = time.perf_counter() - started
            ok = len(results) == len(iter_image_paths(input_dir)) and all(result.ok for result in results)
            error = "; ".join(str(result.error) for result in results if getattr(result, "error", None)) or None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            ok = False
            error = _error_text(exc)
        return {
            "batchSize": batch_size,
            "elapsedSeconds": elapsed,
            "imagesPerSecond": _images_per_second(len(iter_image_paths(input_dir)), elapsed, ok),
            "ok": ok,
            "error": error,
            "outputDir": str(output_dir),
        }

    def _run_watermark_cpu(self, method_obj: BaseWatermark, method: str, input_dir: Path, workers: int, job_id: str) -> JsonDict:
        os.environ["WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD"] = f"{method}={workers}"
        output_dir = self.root / job_id / "work" / "watermark" / method / "cpu" / f"workers_{workers}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        started = time.perf_counter()
        try:
            results = run_watermark_embed_dir_with_method(
                WatermarkEmbedJob(
                    run_id=job_id,
                    method_name=method,
                    params=method_obj.params,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    message="test_watermark_001",
                    device=self.device,
                    seed=2026,
                ),
                method_obj,
            )
            elapsed = time.perf_counter() - started
            ok = len(results) == len(iter_image_paths(input_dir)) and all(result.ok for result in results)
            error = "; ".join(str(result.error) for result in results if getattr(result, "error", None)) or None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            ok = False
            error = _error_text(exc)
        entry = {
            "workers": workers,
            "elapsedSeconds": elapsed,
            "imagesPerSecond": _images_per_second(len(iter_image_paths(input_dir)), elapsed, ok),
            "ok": ok,
            "error": error,
            "outputDir": str(output_dir),
        }
        _mark_output_cleaned(entry, output_dir)
        return entry

    def _warmup_watermark_embed(self, method_obj: BaseWatermark, method: str, warmup_dir: Path, job_id: str) -> None:
        try:
            self._run_watermark_embed(method_obj, method, warmup_dir, 1, job_id, keep_output=False)
        except Exception:
            pass
        _clear_torch_cache()

    def _warmup_watermark_extract(self, method_obj: BaseWatermark, method: str, extract_input: Path, warmup_dir: Path, job_id: str) -> None:
        try:
            input_dir = self._subset_dir(extract_input, self.root / job_id / "work" / "warmup_extract_input" / method, len(iter_image_paths(warmup_dir)))
            self._run_watermark_extract(method_obj, method, input_dir, 1, job_id)
        except Exception:
            pass
        finally:
            _remove_tree(self.root / job_id / "work" / "warmup_extract_input" / method)
        _clear_torch_cache()

    def _benchmark_attacks(self, job_id: str, request: TuningRequest, input_dir: Path) -> Iterable[JsonDict]:
        candidates = _candidates_up_to(request.batch_candidates, request.max_batch_size)
        worker_candidates = _candidates_up_to(request.worker_candidates, request.max_worker_count)
        methods = self._attack_methods_for_tuning(request)
        viewpoint_methods = self._viewpoint_rerendering_methods()
        for method in methods:
            if method not in ATTACK_REGISTRY:
                yield {"method": method, "status": "unknown_method", "_stepCount": 1}
                continue
            record: JsonDict = {
                "method": method,
                "batch": [],
                "cpuWorkers": [],
                "status": "pending",
                "_stepCount": 0,
            }
            try:
                attack_kwargs = (
                    {"save_intermediates": False}
                    if VIEWPOINT_RERENDERING_METHOD_PATTERN.fullmatch(method)
                    else {}
                )
                attack = ATTACK_REGISTRY[method](**attack_kwargs)
                capability = attack.batch_capability_info()
                record["supportsBatch"] = capability.supported
                record["threadSafeParallel"] = bool(getattr(attack, "thread_safe_parallel", False))
                if method == VIEWPOINT_RERENDERING_PRIMARY_METHOD and viewpoint_methods:
                    record["tuningRepresentativeFor"] = viewpoint_methods
                    record["tuningPolicy"] = "3d_viewpoint_rerendering_primary_rotate_point"
                if capability.supported and not record["threadSafeParallel"]:

                    def remember_batch(entry: JsonDict) -> None:
                        entry["method"] = method
                        entry["stage"] = "attack_batch"
                        record["batch"].append(entry)
                        record["_stepCount"] += int(entry.get("repeatCount") or 1)
                        self._event(job_id, "attack_batch", f"{method} batch={entry.get('batchSize')}", entry)

                    self._search_numeric_candidates(
                        job_id=job_id,
                        request=request,
                        initial_candidates=candidates,
                        max_value=request.max_batch_size,
                        value_key="batchSize",
                        run_once=lambda batch_size: self._run_attack_batch(attack, method, input_dir, batch_size, job_id),
                        on_entry=remember_batch,
                    )
                elif record["threadSafeParallel"]:

                    def remember_cpu(entry: JsonDict) -> None:
                        entry["method"] = method
                        entry["stage"] = "attack_cpu"
                        record["cpuWorkers"].append(entry)
                        record["_stepCount"] += int(entry.get("repeatCount") or 1)
                        self._event(job_id, "attack_cpu", f"{method} workers={entry.get('workers')}", entry)

                    self._search_numeric_candidates(
                        job_id=job_id,
                        request=request,
                        initial_candidates=worker_candidates,
                        max_value=request.max_worker_count,
                        value_key="workers",
                        run_once=lambda workers: self._run_attack_cpu(attack, method, input_dir, workers, job_id),
                        on_entry=remember_cpu,
                    )
                else:
                    record["status"] = "serial_only"
                record["bestBatch"] = _best_by_throughput(record["batch"])
                record["bestCpuWorkers"] = _best_by_throughput(record["cpuWorkers"])
                record["status"] = "ok" if record["status"] == "pending" else record["status"]
            except Exception as exc:
                record.update({"status": "failed", "error": _error_text(exc), "traceback": traceback.format_exc(limit=8)})
            finally:
                self._cleanup_entry_outputs(record["batch"])
                self._cleanup_entry_outputs(record["cpuWorkers"])
                _remove_tree(self.root / job_id / "work" / "attack" / method)
                _clear_torch_cache()
            yield record

    def _run_attack_batch(self, attack: Any, method: str, input_dir: Path, batch_size: int, job_id: str) -> JsonDict:
        _set_named_override("WM_BENCH_ATTACK_BATCH_SIZES", method, batch_size)
        output_dir = self.root / job_id / "work" / "attack" / method / f"batch_{batch_size}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        started = time.perf_counter()
        try:
            results = run_attack_dir_with_attack(
                AttackJob(job_id, method, dict(attack.params), input_dir, output_dir, device=self.device, seed=2026),
                attack,
            )
            elapsed = time.perf_counter() - started
            ok = len(results) == len(iter_image_paths(input_dir)) and all(result.ok for result in results)
            error = "; ".join(str(result.error) for result in results if getattr(result, "error", None)) or None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            ok = False
            error = _error_text(exc)
        entry = {
            "batchSize": batch_size,
            "elapsedSeconds": elapsed,
            "imagesPerSecond": _images_per_second(len(iter_image_paths(input_dir)), elapsed, ok),
            "ok": ok,
            "error": error,
            "outputDir": str(output_dir),
        }
        _mark_output_cleaned(entry, output_dir)
        return entry

    def _run_attack_cpu(self, attack: Any, method: str, input_dir: Path, workers: int, job_id: str) -> JsonDict:
        _set_named_override("WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD", method, workers)
        output_dir = self.root / job_id / "work" / "attack" / method / f"workers_{workers}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        started = time.perf_counter()
        try:
            results = run_attack_dir_with_attack(
                AttackJob(job_id, method, dict(attack.params), input_dir, output_dir, device=self.device, seed=2026),
                attack,
            )
            elapsed = time.perf_counter() - started
            ok = len(results) == len(iter_image_paths(input_dir)) and all(result.ok for result in results)
            error = "; ".join(str(result.error) for result in results if getattr(result, "error", None)) or None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            ok = False
            error = _error_text(exc)
        entry = {
            "workers": workers,
            "elapsedSeconds": elapsed,
            "imagesPerSecond": _images_per_second(len(iter_image_paths(input_dir)), elapsed, ok),
            "ok": ok,
            "error": error,
            "outputDir": str(output_dir),
        }
        _mark_output_cleaned(entry, output_dir)
        return entry

    def _benchmark_quality(self, job_id: str, request: TuningRequest, input_dir: Path) -> JsonDict:
        images = iter_image_paths(input_dir)
        pairs = [(path, path) for path in images]
        record: JsonDict = {"cpuWorkers": [], "perceptualBatch": [], "_stepCount": 0}

        def run_quality_cpu(workers: int) -> JsonDict:
            os.environ["WM_BENCH_QUALITY_CPU_WORKERS"] = str(workers)
            started = time.perf_counter()
            try:
                metrics, profile = _compute_cpu_quality_metrics_batch_with_profile(pairs)
                elapsed = time.perf_counter() - started
                ok = len(metrics) == len(pairs)
                error = None
            except Exception as exc:
                elapsed = time.perf_counter() - started
                ok = False
                profile = {}
                error = _error_text(exc)
            return {
                "workers": workers,
                "elapsedSeconds": elapsed,
                "imagesPerSecond": _images_per_second(len(pairs), elapsed, ok),
                "ok": ok,
                "error": error,
                "profile": profile,
            }

        def remember_quality_cpu(entry: JsonDict) -> None:
            entry["method"] = "quality_cpu"
            entry["stage"] = "quality_cpu"
            record["cpuWorkers"].append(entry)
            record["_stepCount"] += int(entry.get("repeatCount") or 1)
            self._event(job_id, "quality_cpu", f"quality CPU workers={entry.get('workers')}", entry)

        self._search_numeric_candidates(
            job_id=job_id,
            request=request,
            initial_candidates=_candidates_up_to(request.worker_candidates, request.max_worker_count),
            max_value=request.max_worker_count,
            value_key="workers",
            run_once=run_quality_cpu,
            on_entry=remember_quality_cpu,
        )

        def run_perceptual(batch_size: int) -> JsonDict:
            os.environ["WM_BENCH_PERCEPTUAL_BATCH_SIZE"] = str(batch_size)
            started = time.perf_counter()
            try:
                metrics, profile = _compute_perceptual_metrics_batch_with_profile(pairs)
                elapsed = time.perf_counter() - started
                ok = len(metrics) == len(pairs)
                error = None
            except Exception as exc:
                elapsed = time.perf_counter() - started
                ok = False
                profile = {}
                error = _error_text(exc)
            return {
                "batchSize": batch_size,
                "elapsedSeconds": elapsed,
                "imagesPerSecond": _images_per_second(len(pairs), elapsed, ok),
                "ok": ok,
                "error": error,
                "profile": profile,
            }

        def remember_perceptual(entry: JsonDict) -> None:
            entry["method"] = "quality_perceptual"
            entry["stage"] = "quality_perceptual"
            record["perceptualBatch"].append(entry)
            record["_stepCount"] += int(entry.get("repeatCount") or 1)
            self._event(job_id, "quality_perceptual", f"perceptual batch={entry.get('batchSize')}", entry)

        self._search_numeric_candidates(
            job_id=job_id,
            request=request,
            initial_candidates=_candidates_up_to(request.batch_candidates, request.max_batch_size),
            max_value=request.max_batch_size,
            value_key="batchSize",
            run_once=run_perceptual,
            on_entry=remember_perceptual,
        )
        record["bestCpuWorkers"] = _best_by_throughput(record["cpuWorkers"])
        record["bestPerceptualBatch"] = _best_by_throughput(record["perceptualBatch"])
        return record

    def _build_summary(self, report: JsonDict) -> JsonDict:
        embed_overrides: list[str] = []
        extract_overrides: list[str] = []
        watermark_cpu_overrides: list[str] = []
        attack_batch_overrides: list[str] = []
        attack_cpu_overrides: list[str] = []
        inherited_attack_batch_overrides: list[str] = []
        seen_attack_batch_methods: set[str] = set()

        def add_attack_batch_override(method: str, batch_size: Any, *, inherited: bool = False) -> None:
            if method in seen_attack_batch_methods:
                return
            override = f"{method}={batch_size}"
            seen_attack_batch_methods.add(method)
            attack_batch_overrides.append(override)
            if inherited:
                inherited_attack_batch_overrides.append(override)

        for method, batch_size in FIXED_ATTACK_BATCH_OVERRIDES.items():
            add_attack_batch_override(method, batch_size)
        for record in report.get("watermarks") or []:
            method = record.get("method")
            best_embed = record.get("bestEmbed")
            best_extract = record.get("bestExtract")
            best_cpu = record.get("bestCpuWorkers")
            if method and isinstance(best_embed, dict):
                embed_overrides.append(f"{method}={best_embed['batchSize']}")
            if method and isinstance(best_extract, dict):
                extract_overrides.append(f"{method}={best_extract['batchSize']}")
            if method and isinstance(best_cpu, dict):
                watermark_cpu_overrides.append(f"{method}={best_cpu['workers']}")
        for record in report.get("attacks") or []:
            method = record.get("method")
            best_batch = record.get("bestBatch")
            best_cpu = record.get("bestCpuWorkers")
            if method and isinstance(best_batch, dict):
                if method == VIEWPOINT_RERENDERING_PRIMARY_METHOD:
                    viewpoint_methods = self._viewpoint_rerendering_methods()
                    for viewpoint_method in viewpoint_methods:
                        add_attack_batch_override(
                            viewpoint_method,
                            best_batch["batchSize"],
                            inherited=viewpoint_method != VIEWPOINT_RERENDERING_PRIMARY_METHOD,
                        )
                else:
                    add_attack_batch_override(str(method), best_batch["batchSize"])
            if method and isinstance(best_cpu, dict):
                attack_cpu_overrides.append(f"{method}={best_cpu['workers']}")
        quality = report.get("quality") if isinstance(report.get("quality"), dict) else {}
        env_updates: dict[str, str] = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "WM_BENCH_PNG_COMPRESS_LEVEL": "1",
        }
        if embed_overrides:
            env_updates["WM_BENCH_WATERMARK_EMBED_BATCH_SIZES"] = ",".join(embed_overrides)
        if extract_overrides:
            env_updates["WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZES"] = ",".join(extract_overrides)
        if watermark_cpu_overrides:
            env_updates["WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD"] = ",".join(watermark_cpu_overrides)
        if attack_batch_overrides:
            env_updates["WM_BENCH_ATTACK_BATCH_SIZES"] = ",".join(attack_batch_overrides)
        if attack_cpu_overrides:
            env_updates["WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD"] = ",".join(attack_cpu_overrides)
        best_quality_cpu = quality.get("bestCpuWorkers") if isinstance(quality, dict) else None
        best_perceptual = quality.get("bestPerceptualBatch") if isinstance(quality, dict) else None
        if isinstance(best_quality_cpu, dict):
            env_updates["WM_BENCH_QUALITY_CPU_WORKERS"] = str(best_quality_cpu["workers"])
        if isinstance(best_perceptual, dict):
            env_updates["WM_BENCH_PERCEPTUAL_BATCH_SIZE"] = str(best_perceptual["batchSize"])
        return {
            "envUpdates": env_updates,
            "watermarkEmbedOverrides": embed_overrides,
            "watermarkExtractOverrides": extract_overrides,
            "watermarkCpuWorkerOverrides": watermark_cpu_overrides,
            "attackBatchOverrides": attack_batch_overrides,
            "fixedAttackBatchOverrides": [f"{method}={batch_size}" for method, batch_size in FIXED_ATTACK_BATCH_OVERRIDES.items()],
            "inheritedAttackBatchOverrides": inherited_attack_batch_overrides,
            "attackCpuWorkerOverrides": attack_cpu_overrides,
            "viewpointRerenderingTuningPolicy": {
                "primaryMethod": VIEWPOINT_RERENDERING_PRIMARY_METHOD,
                "appliesTo": self._viewpoint_rerendering_methods(),
            },
            "qualityBestCpuWorkers": best_quality_cpu,
            "qualityBestPerceptualBatch": best_perceptual,
            "reportPath": str(self.root / str(report["jobId"]) / "report.json"),
        }
