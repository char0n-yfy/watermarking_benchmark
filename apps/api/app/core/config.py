from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .env_loader import PROJECT_ROOT, load_project_env


def _resolve_repo_path(raw: str | None, default: Path) -> Path:
    path = Path(raw or str(default)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


@dataclass(frozen=True)
class Settings:
    environment: str
    project_root: Path
    resources_root: Path
    data_root: Path
    runs_root: Path
    database_path: Path
    api_host: str
    api_port: int
    device: str
    worker_poll_seconds: float
    run_timeout_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_project_env(override=False)
    resolved_runs_root = _resolve_repo_path(
        os.getenv("WM_BENCH_RUNS_ROOT"),
        PROJECT_ROOT / "runs" / "local",
    )
    return Settings(
        environment=os.getenv("APP_ENV", "development"),
        project_root=PROJECT_ROOT,
        resources_root=_resolve_repo_path(
            os.getenv("WM_BENCH_RESOURCES_ROOT"),
            PROJECT_ROOT / "resources",
        ),
        data_root=_resolve_repo_path(
            os.getenv("WM_BENCH_DATA_ROOT"),
            PROJECT_ROOT,
        ),
        runs_root=resolved_runs_root,
        database_path=_resolve_repo_path(
            os.getenv("WM_BENCH_DB_PATH"),
            resolved_runs_root / "wmbench.sqlite",
        ),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        device=os.getenv("WM_BENCH_DEVICE", "cpu"),
        worker_poll_seconds=float(os.getenv("WM_BENCH_WORKER_POLL_SECONDS", "2")),
        run_timeout_seconds=int(os.getenv("WM_BENCH_RUN_TIMEOUT_SECONDS", "3600")),
    )
