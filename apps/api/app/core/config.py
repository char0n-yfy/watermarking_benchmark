from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
    data_root = Path(os.getenv("WM_BENCH_DATA_ROOT", str(PROJECT_ROOT))).expanduser()
    runs_root = Path(os.getenv("WM_BENCH_RUNS_ROOT", str(PROJECT_ROOT / "runs" / "local"))).expanduser()
    database_path = Path(
        os.getenv("WM_BENCH_DB_PATH", str(runs_root / "wmbench.sqlite"))
    ).expanduser()
    return Settings(
        environment=os.getenv("APP_ENV", "development"),
        project_root=PROJECT_ROOT,
        resources_root=Path(
            os.getenv("WM_BENCH_RESOURCES_ROOT", str(PROJECT_ROOT / "resources"))
        ).expanduser(),
        data_root=data_root,
        runs_root=runs_root,
        database_path=database_path,
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        device=os.getenv("WM_BENCH_DEVICE", "cpu"),
        worker_poll_seconds=float(os.getenv("WM_BENCH_WORKER_POLL_SECONDS", "2")),
        run_timeout_seconds=int(os.getenv("WM_BENCH_RUN_TIMEOUT_SECONDS", "3600")),
    )
