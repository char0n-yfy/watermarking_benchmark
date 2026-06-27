from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import get_settings
from .core.local_db import LocalDatabase
from .core.status import RunStatus
from .schemas.experiments import ExperimentConfigCreatePayload, ExperimentConfigRenamePayload, RunCreatePayload
from .services.experiment_service import ExperimentService
from .services.resources import (
    list_attack_resources,
    list_watermark_resources,
    scan_dataset_resources,
)
from .services.system_metrics import collect_system_metrics


def create_app() -> FastAPI:
    settings = get_settings()
    service = ExperimentService(
        database=LocalDatabase(settings.database_path),
        resources_root=settings.resources_root,
        runs_root=settings.runs_root,
    )

    app = FastAPI(
        title="Watermark Benchmark API",
        version="0.1.0",
        description="Local-first service for experiment metadata and small run orchestration.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:6006",
            "http://127.0.0.1:6006",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "data_root": str(settings.data_root),
            "resources_root": str(settings.resources_root),
            "runs_root": str(settings.runs_root),
            "database_path": str(settings.database_path),
        }

    @app.get("/system/runtime")
    def runtime() -> dict[str, object]:
        return {
            "environment": settings.environment,
            "device": settings.device,
            "dataRoot": str(settings.data_root),
            "resourcesRoot": str(settings.resources_root),
            "runsRoot": str(settings.runs_root),
            "databasePath": str(settings.database_path),
            "apiHost": settings.api_host,
            "apiPort": settings.api_port,
            "workerPollSeconds": settings.worker_poll_seconds,
            "workers": service.list_worker_heartbeats(),
        }

    @app.get("/system/metrics")
    def system_metrics() -> dict[str, object]:
        return collect_system_metrics(data_root=settings.data_root, device=settings.device)

    @app.get("/status-values")
    def status_values() -> dict[str, list[str]]:
        return {"run_statuses": [status.value for status in RunStatus]}

    @app.get("/resources/datasets")
    def datasets() -> list[dict[str, object]]:
        return [dataset.to_json() for dataset in scan_dataset_resources(settings.resources_root)]

    @app.get("/resources/watermarks")
    def watermarks() -> list[dict[str, object]]:
        return list_watermark_resources()

    @app.get("/resources/attacks")
    def attacks() -> list[dict[str, object]]:
        return list_attack_resources()

    @app.post("/experiment-configs")
    def create_config(payload: ExperimentConfigCreatePayload) -> dict[str, object]:
        try:
            return service.create_config(payload.name, payload.selection.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/experiment-configs")
    def list_configs() -> list[dict[str, object]]:
        return service.list_configs()

    @app.patch("/experiment-configs/{config_id}")
    def rename_config(config_id: str, payload: ExperimentConfigRenamePayload) -> dict[str, object]:
        try:
            return service.rename_config(config_id, payload.name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/experiment-configs/{config_id}")
    def delete_config(config_id: str) -> dict[str, str]:
        try:
            return service.delete_config(config_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs")
    def create_run(payload: RunCreatePayload) -> dict[str, object]:
        try:
            return service.create_run(payload.resolved_config_id(), execute=payload.execute)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, object]:
        try:
            return service.cancel_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs")
    def list_runs() -> list[dict[str, object]]:
        return service.list_runs()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        try:
            run = service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {**run, "cellsDetail": service.list_run_cells(run_id)}

    @app.get("/runs/{run_id}/results")
    def get_run_results(run_id: str) -> dict[str, object]:
        try:
            return service.get_run_results(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/logs")
    def get_run_logs(run_id: str) -> dict[str, object]:
        try:
            return service.get_run_logs(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    web_out = Path(settings.project_root) / "apps" / "web" / "out"
    if web_out.exists():
        app.mount("/", StaticFiles(directory=web_out, html=True), name="web")

    return app


app = create_app()
