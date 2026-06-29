from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import get_settings
from .core.local_db import LocalDatabase
from .core.status import RunStatus
from .schemas.datasets import DatasetDownloadCreatePayload
from .schemas.experiments import ExperimentConfigCreatePayload, ExperimentConfigRenamePayload, RunCreatePayload
from .services.dataset_catalog import build_catalog_item, get_catalog_entry, list_categories, list_dataset_catalog
from .services.dataset_download import DatasetDownloadService
from .services.experiment_service import ExperimentService
from .services.object_storage import get_object_storage_client
from .services.attack_weight_download import AttackWeightDownloadService
from .services.resources import (
    get_attack_catalog_item,
    get_watermark_catalog_item,
    list_attack_resources,
    list_watermark_resources,
    scan_dataset_resources,
)
from .services.weight_download import WeightDownloadService
from .services.system_metrics import collect_system_metrics


def create_app() -> FastAPI:
    settings = get_settings()
    service = ExperimentService(
        database=LocalDatabase(settings.database_path),
        resources_root=settings.resources_root,
        runs_root=settings.runs_root,
    )
    oss_client = get_object_storage_client()
    download_service = DatasetDownloadService(settings.resources_root, oss=oss_client)
    weight_download_service = WeightDownloadService(settings.resources_root, oss=oss_client)
    attack_weight_download_service = AttackWeightDownloadService(settings.resources_root, oss=oss_client)

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

    @app.get("/resources/storage/status")
    def storage_status() -> dict[str, object]:
        return oss_client.status()

    @app.get("/resources/datasets/catalog")
    def dataset_catalog(remote: bool = False) -> dict[str, object]:
        return {
            "categories": list_categories(),
            "items": list_dataset_catalog(
                settings.resources_root,
                oss=oss_client,
                probe_remote=remote,
            ),
        }

    @app.get("/resources/datasets/downloads/{job_id}")
    def get_dataset_download(job_id: str) -> dict[str, object]:
        try:
            return download_service.get_job(job_id).to_json()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/resources/datasets/downloads/{job_id}/archive")
    def download_dataset_archive(job_id: str) -> FileResponse:
        try:
            job = download_service.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.status != "succeeded" or not job.archive_path:
            raise HTTPException(status_code=409, detail="Download job is not ready")
        archive_path = Path(job.archive_path)
        if not archive_path.exists():
            raise HTTPException(status_code=404, detail="Archive file not found")
        filename = f"{job.dataset_id}-{job.mode}-{job.id}.zip"
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=filename,
        )

    @app.get("/resources/datasets/{dataset_id}")
    def dataset_detail(dataset_id: str) -> dict[str, object]:
        try:
            entry = get_catalog_entry(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return build_catalog_item(settings.resources_root, entry, oss=oss_client)

    @app.post("/resources/datasets/{dataset_id}/downloads")
    def start_dataset_download(dataset_id: str, payload: DatasetDownloadCreatePayload) -> dict[str, object]:
        try:
            get_catalog_entry(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            job = download_service.start_download(
                dataset_id,
                mode=payload.mode,
                seed=payload.seed,
                sample_count=payload.sample_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job.to_json()

    @app.get("/resources/datasets/{dataset_id}/downloads")
    def list_dataset_downloads(dataset_id: str) -> list[dict[str, object]]:
        return [job.to_json() for job in download_service.list_jobs(dataset_id)]

    @app.get("/resources/watermarks")
    def watermarks(remote: bool = False) -> list[dict[str, object]]:
        return list_watermark_resources(
            settings.resources_root,
            oss=oss_client,
            probe_remote=remote,
        )

    @app.post("/resources/watermarks/{identifier}/downloads")
    def start_weight_download(identifier: str) -> dict[str, object]:
        try:
            item = get_watermark_catalog_item(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        method = str(item["method"])
        try:
            job = weight_download_service.start_download(method)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job.to_json()

    @app.get("/resources/watermarks/downloads/{job_id}")
    def get_weight_download(job_id: str) -> dict[str, object]:
        try:
            return weight_download_service.get_job(job_id).to_json()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/resources/attacks")
    def attacks(remote: bool = False) -> list[dict[str, object]]:
        return list_attack_resources(
            settings.resources_root,
            oss=oss_client,
            probe_remote=remote,
        )

    @app.post("/resources/attacks/{identifier}/downloads")
    def start_attack_weight_download(identifier: str) -> dict[str, object]:
        try:
            item = get_attack_catalog_item(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        method = str(item["method"])
        try:
            job = attack_weight_download_service.start_download(method)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job.to_json()

    @app.get("/resources/attacks/downloads/{job_id}")
    def get_attack_weight_download(job_id: str) -> dict[str, object]:
        try:
            return attack_weight_download_service.get_job(job_id).to_json()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.get("/runs/{run_id}/score")
    def get_run_score(run_id: str) -> dict[str, object]:
        try:
            return service.get_run_score(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/logs")
    def get_run_logs(run_id: str) -> dict[str, object]:
        try:
            return service.get_run_logs(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/benchmark-protocols")
    def get_benchmark_protocols() -> list[dict[str, object]]:
        return service.list_benchmark_protocols()

    web_out = Path(settings.project_root) / "apps" / "web" / "out"

    @app.get("/leaderboard")
    def get_leaderboard(
        protocol_id: Optional[str] = Query(default=None),
    ) -> object:
        if protocol_id is None and web_out.exists():
            page = web_out / "leaderboard.html"
            fallback = web_out / "leaderboard" / "index.html"
            if page.exists():
                return FileResponse(page)
            if fallback.exists():
                return FileResponse(fallback)
        try:
            return service.list_leaderboard(protocol_id or "waves-official-detection-v1")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if web_out.exists():
        app.mount("/", StaticFiles(directory=web_out, html=True), name="web")

    return app


app = create_app()
