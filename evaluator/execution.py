from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping


JsonDict = dict[str, Any]

EXECUTION_ENV_VARS = (
    "APP_ENV",
    "WM_BENCH_DEVICE",
    "WM_BENCH_DATA_ROOT",
    "WM_BENCH_RESOURCES_ROOT",
    "WM_BENCH_RUNS_ROOT",
    "WM_BENCH_DB_PATH",
    "WM_BENCH_DOTENV_PATH",
    "WM_BENCH_VENV",
    "WM_BENCH_VENV_SYSTEM_SITE_PACKAGES",
    "WM_BENCH_WORKER_ID",
    "WM_BENCH_WORKER_POLL_SECONDS",
    "WM_BENCH_RUN_TIMEOUT_SECONDS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "WM_BENCH_PNG_COMPRESS_LEVEL",
    "WM_BENCH_WATERMARK_CACHE_MAX_ENTRIES",
    "WM_BENCH_WATERMARK_EMBED_BATCH_SIZE",
    "WM_BENCH_WATERMARK_EMBED_BATCH_SIZES",
    "WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZE",
    "WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZES",
    "WM_BENCH_WATERMARK_BATCH_SIZE",
    "WM_BENCH_WATERMARK_BATCH_SIZES",
    "WM_BENCH_WATERMARK_CPU_WORKERS",
    "WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD",
    "WM_BENCH_ATTACK_CACHE_MAX_ENTRIES",
    "WM_BENCH_ATTACK_CPU_WORKERS",
    "WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD",
    "WM_BENCH_ATTACK_BATCH_SIZE",
    "WM_BENCH_ATTACK_BATCH_SIZES",
    "WM_BENCH_QUALITY_CPU_WORKERS",
    "WM_BENCH_PERCEPTUAL_DEVICE",
    "WM_BENCH_PERCEPTUAL_BATCH_SIZE",
    "WM_BENCH_PERCEPTUAL_BATCH_SIZES",
    "WM_BENCH_DISABLE_PERCEPTUAL_METRICS",
    "WM_BENCH_LOG_DIR",
)


@dataclass(frozen=True)
class ResolvedInt:
    value: int
    source: str
    raw: Any = None

    def to_json(self) -> JsonDict:
        return {"value": self.value, "source": self.source, "raw": self.raw}


@dataclass(frozen=True)
class ExecutionProfile:
    stage: str
    method: str
    mode: str
    job_count: int
    device: str | None = None
    cpu_workers: int = 1
    configured_batch_size: int | None = None
    actual_batch_size: int | None = None
    batch_stage: str | None = None
    supports_batch: bool | None = None
    thread_safe_parallel: bool | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        return _drop_none(
            {
                "stage": self.stage,
                "method": self.method,
                "mode": self.mode,
                "jobCount": self.job_count,
                "device": self.device,
                "cpuWorkers": self.cpu_workers,
                "configuredBatchSize": self.configured_batch_size,
                "actualBatchSize": self.actual_batch_size,
                "batchStage": self.batch_stage,
                "supportsBatch": self.supports_batch,
                "threadSafeParallel": self.thread_safe_parallel,
                "fallback": self.fallback,
                "fallbackReason": self.fallback_reason,
                "config": dict(self.config),
                "details": dict(self.details),
            }
        )


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def parse_positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, value)


def parse_named_int_overrides(raw: str | None) -> dict[str, int]:
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


def default_worker_count(job_count: int, *, cap: int = 8) -> int:
    if job_count <= 1:
        return 1
    return max(1, min(int(cap), int(job_count), os.cpu_count() or 1))


def resolve_cpu_workers(
    env_name: str,
    job_count: int,
    *,
    enabled: bool = True,
    default_cap: int = 8,
) -> ResolvedInt:
    if not enabled:
        return ResolvedInt(1, "disabled")
    if job_count <= 1:
        return ResolvedInt(1, "single_job")

    raw = os.getenv(env_name)
    if raw is None or raw.strip() == "":
        return ResolvedInt(default_worker_count(job_count, cap=default_cap), "default", None)
    try:
        configured = int(raw)
    except (TypeError, ValueError):
        return ResolvedInt(default_worker_count(job_count, cap=default_cap), "invalid_env_default", raw)
    return ResolvedInt(max(1, min(job_count, configured)), f"env:{env_name}", raw)


def resolve_named_cpu_workers(
    name: str,
    *,
    overrides_env: str,
    global_env: str,
    job_count: int,
    enabled: bool = True,
    default_cap: int = 8,
) -> ResolvedInt:
    if not enabled:
        return ResolvedInt(1, "disabled")
    if job_count <= 1:
        return ResolvedInt(1, "single_job")

    normalized_name = str(name).lower()
    raw_overrides = os.getenv(overrides_env)
    overrides = parse_named_int_overrides(raw_overrides)
    if normalized_name in overrides:
        return ResolvedInt(
            max(1, min(job_count, overrides[normalized_name])),
            f"env:{overrides_env}:{normalized_name}",
            raw_overrides,
        )
    return resolve_cpu_workers(global_env, job_count, enabled=True, default_cap=default_cap)


def resolve_named_batch_size(
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    param_key: str | None = None,
    overrides_env: str,
    global_env: str,
    default: int,
) -> ResolvedInt:
    if params is not None and param_key and params.get(param_key) is not None:
        raw = params.get(param_key)
        return ResolvedInt(parse_positive_int(raw, default), f"param:{param_key}", raw)

    normalized_name = str(name).lower()
    raw_overrides = os.getenv(overrides_env)
    overrides = parse_named_int_overrides(raw_overrides)
    if normalized_name in overrides:
        return ResolvedInt(overrides[normalized_name], f"env:{overrides_env}:{normalized_name}", raw_overrides)

    raw_global = os.getenv(global_env)
    if raw_global is not None and raw_global.strip() != "":
        return ResolvedInt(parse_positive_int(raw_global, default), f"env:{global_env}", raw_global)
    return ResolvedInt(max(1, int(default)), "default", None)


def attach_execution_metadata(metadata: Mapping[str, Any], profile: ExecutionProfile) -> JsonDict:
    enriched = dict(metadata)
    execution = profile.to_json()
    enriched["execution"] = execution
    enriched["executionMode"] = profile.mode
    return enriched


def replace_result_execution(result: Any, profile: ExecutionProfile) -> Any:
    metadata = attach_execution_metadata(getattr(result, "metadata", {}) or {}, profile)
    return replace(result, metadata=metadata)


def summarize_execution_profiles(results: Iterable[Any]) -> JsonDict:
    profiles: list[JsonDict] = []
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        execution = metadata.get("execution") if isinstance(metadata, Mapping) else None
        if isinstance(execution, Mapping):
            profiles.append(dict(execution))

    if not profiles:
        return {"profileCount": 0}

    mode_counts = Counter(str(profile.get("mode") or "unknown") for profile in profiles)
    stage_counts = Counter(str(profile.get("stage") or "unknown") for profile in profiles)
    method_counts = Counter(str(profile.get("method") or "unknown") for profile in profiles)
    cpu_workers = sorted(
        {
            int(profile["cpuWorkers"])
            for profile in profiles
            if isinstance(profile.get("cpuWorkers"), int)
        }
    )
    configured_batch_sizes = sorted(
        {
            int(profile["configuredBatchSize"])
            for profile in profiles
            if isinstance(profile.get("configuredBatchSize"), int)
        }
    )
    actual_batch_sizes = sorted(
        {
            int(profile["actualBatchSize"])
            for profile in profiles
            if isinstance(profile.get("actualBatchSize"), int)
        }
    )
    fallback_count = sum(1 for profile in profiles if bool(profile.get("fallback")))

    unique_profiles: Counter[str] = Counter()
    profile_lookup: dict[str, JsonDict] = {}
    for profile in profiles:
        key_parts = {
            key: profile.get(key)
            for key in (
                "stage",
                "method",
                "mode",
                "device",
                "cpuWorkers",
                "configuredBatchSize",
                "actualBatchSize",
                "batchStage",
                "supportsBatch",
                "threadSafeParallel",
                "fallback",
                "fallbackReason",
            )
            if key in profile
        }
        key = repr(sorted(key_parts.items()))
        unique_profiles[key] += 1
        profile_lookup[key] = dict(key_parts)

    return {
        "profileCount": len(profiles),
        "modes": dict(sorted(mode_counts.items())),
        "stages": dict(sorted(stage_counts.items())),
        "methods": dict(sorted(method_counts.items())),
        "cpuWorkers": cpu_workers,
        "configuredBatchSizes": configured_batch_sizes,
        "actualBatchSizes": actual_batch_sizes,
        "fallbackCount": fallback_count,
        "profiles": [
            {"count": count, **profile_lookup[key]}
            for key, count in sorted(unique_profiles.items())
        ],
    }


def execution_environment_snapshot() -> JsonDict:
    return {name: os.getenv(name) for name in EXECUTION_ENV_VARS if os.getenv(name) is not None}
