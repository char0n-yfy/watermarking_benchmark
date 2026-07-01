from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import (  # noqa: F401 - import registers default attacks
    consumer_enhancement_workflow_attacks,
    distortion_attacks,
    physical_channel_attacks,
    regeneration_attacks,
)
from .base import AttackContext, AttackResult, BaseAttack
from .registry import build_attack
from evaluator.execution import (
    ExecutionProfile,
    attach_execution_metadata,
    replace_result_execution,
    resolve_named_batch_size,
    resolve_named_cpu_workers,
)


INTERMEDIATE_ARTIFACT_DIR = "_intermediates"
_ATTACK_INSTANCE_CACHE: OrderedDict[str, BaseAttack] = OrderedDict()


def _cache_max_entries() -> int:
    raw = os.getenv("WM_BENCH_ATTACK_CACHE_MAX_ENTRIES", "8")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 8


def _release_attack_instance(attack: BaseAttack) -> None:
    del attack
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _prepend_path_once(directory: Path) -> None:
    directory_text = str(directory)
    if not directory_text or not directory.exists():
        return
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory_text not in entries:
        os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")


def _ensure_runtime_binaries_on_path() -> None:
    _prepend_path_once(Path(sys.executable).resolve().parent)
    try:
        import ninja

        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir:
            _prepend_path_once(Path(bin_dir))
    except Exception:
        pass


@dataclass(frozen=True)
class AttackJob:
    run_id: str
    attack_name: str
    params: dict[str, Any]
    input_dir: Path
    output_dir: Path
    device: str = "cpu"
    seed: int | None = 42
    image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _cache_key(name: str, params: dict[str, Any], device: str) -> str:
    payload = {
        "name": str(name).lower(),
        "params": params,
        "device": str(device),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def get_cached_attack(name: str, params: dict[str, Any], device: str = "cpu") -> BaseAttack:
    _ensure_runtime_binaries_on_path()
    key = _cache_key(name, params, device)
    max_entries = _cache_max_entries()
    if max_entries == 0:
        return build_attack(name, **params)

    attack = _ATTACK_INSTANCE_CACHE.get(key)
    if attack is not None:
        _ATTACK_INSTANCE_CACHE.move_to_end(key)
        return attack

    attack = build_attack(name, **params)
    _ATTACK_INSTANCE_CACHE[key] = attack
    while len(_ATTACK_INSTANCE_CACHE) > max_entries:
        _old_key, old_attack = _ATTACK_INSTANCE_CACHE.popitem(last=False)
        _release_attack_instance(old_attack)
    return attack


def clear_attack_cache() -> None:
    while _ATTACK_INSTANCE_CACHE:
        _old_key, old_attack = _ATTACK_INSTANCE_CACHE.popitem(last=False)
        _release_attack_instance(old_attack)


def iter_image_paths(input_dir: Path, image_exts: Iterable[str]) -> list[Path]:
    normalized_exts = {ext.lower() for ext in image_exts}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in normalized_exts
            and INTERMEDIATE_ARTIFACT_DIR not in path.relative_to(input_dir).parts
        )
    )


def run_attack_dir_with_attack(job: AttackJob, attack: BaseAttack) -> list[AttackResult]:
    image_paths = iter_image_paths(job.input_dir, job.image_exts)
    tasks: list[tuple[Path, Path, AttackContext]] = []

    for index, input_path in enumerate(image_paths):
        relative = input_path.relative_to(job.input_dir)
        output_path = (job.output_dir / relative).with_suffix(attack.output_ext)
        context = AttackContext(
            run_id=job.run_id,
            sample_id=str(relative.with_suffix("")),
            attack_name=attack.name,
            params=attack.params,
            workspace_dir=job.output_dir,
            device=job.device,
            seed=None if job.seed is None else job.seed + index,
        )
        tasks.append((input_path, output_path, context))

    def run_one(task: tuple[Path, Path, AttackContext]) -> AttackResult:
        input_path, output_path, context = task
        return attack.attack(input_path, output_path, context)

    thread_safe_parallel = bool(getattr(attack, "thread_safe_parallel", False))
    batch_capability = attack.batch_capability_info()
    supports_batch = batch_capability.supported
    if supports_batch and not thread_safe_parallel:
        batch_config = resolve_named_batch_size(
            str(attack.name),
            params=attack.params,
            param_key="attack_batch_size",
            overrides_env="WM_BENCH_ATTACK_BATCH_SIZES",
            global_env="WM_BENCH_ATTACK_BATCH_SIZE",
            default=1,
        )
        batch_size = batch_config.value
        results: list[AttackResult] = []
        for offset in range(0, len(tasks), batch_size):
            chunk = tasks[offset : offset + batch_size]
            for _input_path, output_path, _context in chunk:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            execution = ExecutionProfile(
                stage="attack",
                method=str(attack.name),
                mode="batch",
                job_count=len(tasks),
                device=job.device,
                cpu_workers=1,
                configured_batch_size=batch_size,
                actual_batch_size=len(chunk),
                batch_stage=batch_capability.stage,
                supports_batch=True,
                thread_safe_parallel=thread_safe_parallel,
                config={
                    "batchSize": batch_config.to_json(),
                    "batchCapability": batch_capability.to_json(),
                },
            )
            started = time.perf_counter()
            try:
                metadatas = [dict(metadata) for metadata in attack.apply_batch_impl(chunk)]
                if len(metadatas) != len(chunk):
                    raise ValueError(
                        f"apply_batch_impl returned {len(metadatas)} results for {len(chunk)} jobs"
                    )
            except Exception as exc:
                fallback = ExecutionProfile(
                    stage="attack",
                    method=str(attack.name),
                    mode="batch_fallback_serial",
                    job_count=len(tasks),
                    device=job.device,
                    cpu_workers=1,
                    configured_batch_size=batch_size,
                    actual_batch_size=len(chunk),
                    batch_stage=batch_capability.stage,
                    supports_batch=True,
                    thread_safe_parallel=thread_safe_parallel,
                    fallback=True,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    config={
                        "batchSize": batch_config.to_json(),
                        "batchCapability": batch_capability.to_json(),
                    },
                )
                results.extend(replace_result_execution(run_one(task), fallback) for task in chunk)
                continue

            elapsed_ms = ((time.perf_counter() - started) * 1000) / max(1, len(chunk))
            for (input_path, output_path, _context), metadata in zip(chunk, metadatas):
                try:
                    metadata = attack._protocol_metadata(input_path, output_path, metadata)
                    metadata = attach_execution_metadata(metadata, execution)
                    results.append(
                        AttackResult(
                            input_path=input_path,
                            output_path=output_path,
                            attack_name=attack.name,
                            params=attack.params,
                            elapsed_ms=elapsed_ms,
                            ok=True,
                            error=None,
                            metadata=metadata,
                        )
                    )
                except Exception as exc:
                    results.append(
                        AttackResult(
                            input_path=input_path,
                            output_path=output_path,
                            attack_name=attack.name,
                            params=attack.params,
                            elapsed_ms=elapsed_ms,
                            ok=False,
                            error=f"{type(exc).__name__}: {exc}",
                            metadata=attach_execution_metadata({}, execution),
                        )
                    )

        attack.write_manifest(job.output_dir / "attack_manifest.json", results)
        return results

    worker_config = resolve_named_cpu_workers(
        str(attack.name),
        overrides_env="WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD",
        global_env="WM_BENCH_ATTACK_CPU_WORKERS",
        job_count=len(tasks),
        enabled=thread_safe_parallel,
        default_cap=8,
    )
    workers = worker_config.value
    execution = ExecutionProfile(
        stage="attack",
        method=str(attack.name),
        mode="threadpool" if workers > 1 else "serial",
        job_count=len(tasks),
        device=job.device,
        cpu_workers=workers,
        batch_stage=batch_capability.stage,
        supports_batch=supports_batch,
        thread_safe_parallel=thread_safe_parallel,
        config={
            "cpuWorkers": worker_config.to_json(),
            "batchCapability": batch_capability.to_json(),
        },
    )
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = [replace_result_execution(result, execution) for result in executor.map(run_one, tasks)]
    else:
        results = [replace_result_execution(run_one(task), execution) for task in tasks]

    attack.write_manifest(job.output_dir / "attack_manifest.json", results)
    return results


def run_attack_dir(job: AttackJob) -> list[AttackResult]:
    attack = get_cached_attack(job.attack_name, job.params, job.device)
    return run_attack_dir_with_attack(job, attack)
