from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_PARALLEL_ENV_PATH = Path("parallel_tuning") / "active_env.json"

PARALLEL_TUNING_ENV_KEYS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "WM_BENCH_PNG_COMPRESS_LEVEL",
    "WM_BENCH_WATERMARK_EMBED_BATCH_SIZE",
    "WM_BENCH_WATERMARK_EMBED_BATCH_SIZES",
    "WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZE",
    "WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZES",
    "WM_BENCH_WATERMARK_BATCH_SIZE",
    "WM_BENCH_WATERMARK_BATCH_SIZES",
    "WM_BENCH_WATERMARK_CPU_WORKERS",
    "WM_BENCH_WATERMARK_CPU_WORKERS_BY_METHOD",
    "WM_BENCH_ATTACK_CPU_WORKERS",
    "WM_BENCH_ATTACK_CPU_WORKERS_BY_METHOD",
    "WM_BENCH_ATTACK_BATCH_SIZE",
    "WM_BENCH_ATTACK_BATCH_SIZES",
    "WM_BENCH_QUALITY_CPU_WORKERS",
    "WM_BENCH_PERCEPTUAL_BATCH_SIZE",
    "WM_BENCH_PERCEPTUAL_BATCH_SIZES",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_parallel_env_path(runs_root: Path) -> Path:
    return runs_root / RUNTIME_PARALLEL_ENV_PATH


def clean_parallel_env_updates(updates: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in updates.items()
        if str(key) in PARALLEL_TUNING_ENV_KEYS and value is not None
    }


def apply_parallel_env_updates(updates: dict[str, Any]) -> dict[str, str]:
    cleaned = clean_parallel_env_updates(updates)
    for key, value in cleaned.items():
        os.environ[key] = value
    return cleaned


def write_runtime_parallel_env(
    runs_root: Path,
    updates: dict[str, Any],
    *,
    job_id: str | None = None,
    env_path: Path | None = None,
) -> dict[str, Any]:
    cleaned = clean_parallel_env_updates(updates)
    payload = {
        "schemaVersion": 1,
        "jobId": job_id,
        "savedAt": _utc_timestamp(),
        "envPath": str(env_path) if env_path is not None else None,
        "envUpdates": cleaned,
    }
    path = runtime_parallel_env_path(runs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path), **payload}


def read_runtime_parallel_env(runs_root: Path) -> dict[str, Any] | None:
    path = runtime_parallel_env_path(runs_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def apply_runtime_parallel_env(runs_root: Path) -> dict[str, Any] | None:
    payload = read_runtime_parallel_env(runs_root)
    if not payload:
        return None
    updates = payload.get("envUpdates")
    if not isinstance(updates, dict):
        return None
    applied = apply_parallel_env_updates(updates)
    return {
        "path": str(runtime_parallel_env_path(runs_root)),
        "jobId": payload.get("jobId"),
        "savedAt": payload.get("savedAt"),
        "envPath": payload.get("envPath"),
        "applied": applied,
    }
