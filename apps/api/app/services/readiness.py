from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.experiment_service import ExperimentService
from app.services.resources import list_attack_resources, list_watermark_resources, scan_dataset_resources


CheckStatus = str


def collect_readiness(settings: Settings, service: ExperimentService) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        label: str,
        status: CheckStatus,
        detail: str,
        *,
        required: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "label": label,
                "status": status,
                "detail": detail,
                "required": required,
                "meta": meta or {},
            }
        )

    add_path_check(checks, "project_root", "Project root", settings.project_root, required=True)
    add_path_check(checks, "resources_root", "Resources root", settings.resources_root, required=True)

    datasets_root = settings.resources_root / "datasets"
    weights_root = settings.resources_root / "weights"
    add_path_check(checks, "datasets_root", "Datasets directory", datasets_root, required=False)
    add_path_check(checks, "weights_root", "Weights directory", weights_root, required=False)
    add_weight_files_check(checks, weights_root)
    add_writable_check(checks, "runs_root_writable", "Runs directory writable", settings.runs_root)
    add_writable_check(checks, "database_parent_writable", "SQLite directory writable", settings.database_path.parent)

    try:
        datasets = scan_dataset_resources(settings.resources_root)
        if datasets:
            add_check(
                "datasets_present",
                "Datasets indexed",
                "ok",
                f"{len(datasets)} dataset(s) are available.",
                required=False,
                meta={"count": len(datasets)},
            )
        else:
            add_check(
                "datasets_present",
                "Datasets indexed",
                "warn",
                "No image datasets were found under resources/datasets.",
                required=False,
                meta={"count": 0},
            )
    except Exception as exc:
        add_check(
            "datasets_present",
            "Datasets indexed",
            "error",
            f"{type(exc).__name__}: {exc}",
            required=False,
        )

    try:
        with service.database.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        add_check("sqlite", "SQLite metadata database", "ok", str(settings.database_path))
    except Exception as exc:
        add_check("sqlite", "SQLite metadata database", "error", f"{type(exc).__name__}: {exc}")

    try:
        watermarks = list_watermark_resources()
        attacks = list_attack_resources()
        status = "ok" if watermarks and attacks else "error"
        detail = f"{len(watermarks)} watermark algorithm(s), {len(attacks)} attack preset(s)."
        add_check(
            "resource_catalog",
            "Algorithm and attack catalog",
            status,
            detail,
            meta={"watermarks": len(watermarks), "attacks": len(attacks)},
        )
    except Exception as exc:
        add_check("resource_catalog", "Algorithm and attack catalog", "error", f"{type(exc).__name__}: {exc}")

    workers = service.list_worker_heartbeats()
    fresh_workers = [worker for worker in workers if is_fresh_worker(worker, settings.worker_poll_seconds)]
    if fresh_workers:
        add_check(
            "worker_heartbeat",
            "Worker heartbeat",
            "ok",
            f"{len(fresh_workers)} active worker(s), {len(workers)} known worker(s).",
            required=False,
            meta={"active": len(fresh_workers), "known": len(workers)},
        )
    elif workers:
        add_check(
            "worker_heartbeat",
            "Worker heartbeat",
            "warn",
            f"{len(workers)} worker heartbeat(s) exist, but none are fresh.",
            required=False,
            meta={"active": 0, "known": len(workers)},
        )
    else:
        add_check(
            "worker_heartbeat",
            "Worker heartbeat",
            "warn",
            "No worker heartbeat found. Runs can be queued, but will not execute until a worker starts.",
            required=False,
            meta={"active": 0, "known": 0},
        )

    has_required_error = any(check["required"] and check["status"] == "error" for check in checks)
    has_warning = any(check["status"] in {"warn", "error"} for check in checks)
    return {
        "status": "not_ready" if has_required_error else "degraded" if has_warning else "ready",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment,
        "device": settings.device,
        "checks": checks,
    }


def add_path_check(
    checks: list[dict[str, Any]],
    check_id: str,
    label: str,
    path: Path,
    *,
    required: bool,
) -> None:
    exists = path.exists()
    is_dir = path.is_dir()
    status = "ok" if exists and is_dir else "error" if required else "warn"
    detail = str(path) if exists and is_dir else f"Missing directory: {path}"
    checks.append(
        {
            "id": check_id,
            "label": label,
            "status": status,
            "detail": detail,
            "required": required,
            "meta": {"path": str(path)},
        }
    )


def add_writable_check(checks: list[dict[str, Any]], check_id: str, label: str, path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".wmbench-check-", dir=path) as handle:
            handle.write(b"ok")
            handle.flush()
        status = "ok"
        detail = str(path)
    except Exception as exc:
        status = "error"
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "id": check_id,
            "label": label,
            "status": status,
            "detail": detail,
            "required": True,
            "meta": {"path": str(path)},
        }
    )


def add_weight_files_check(checks: list[dict[str, Any]], weights_root: Path) -> None:
    if not weights_root.exists():
        return
    ignored_names = {"readme", "readme.md", ".gitkeep", ".gitignore"}
    files = [
        path
        for path in weights_root.rglob("*")
        if path.is_file() and path.name.lower() not in ignored_names
    ]
    if files:
        status = "ok"
        detail = f"{len(files)} weight/artifact file(s) found under {weights_root}."
    else:
        status = "warn"
        detail = f"No weight/artifact files found under {weights_root}; only placeholder files may be present."
    checks.append(
        {
            "id": "weights_present",
            "label": "Weight artifacts present",
            "status": status,
            "detail": detail,
            "required": False,
            "meta": {"count": len(files), "path": str(weights_root)},
        }
    )


def is_fresh_worker(worker: dict[str, Any], poll_seconds: float) -> bool:
    raw_last_seen = worker.get("lastSeenAt")
    if not isinstance(raw_last_seen, str):
        return False
    try:
        last_seen = datetime.fromisoformat(raw_last_seen)
    except ValueError:
        return False
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    max_age_seconds = max(30.0, poll_seconds * 5)
    return (datetime.now(timezone.utc) - last_seen).total_seconds() <= max_age_seconds
