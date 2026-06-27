from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.local_db import (
    LocalDatabase,
    dumps_json,
    row_to_cell,
    row_to_config,
    row_to_run,
)
from app.core.planner import ExperimentCell, ExperimentSpec, materialize_cells
from app.core.storage import safe_segment
from app.services.local_runner import LocalRunRequest, estimate_selection, run_local_experiment


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "partially_failed"}


@dataclass(frozen=True)
class MaterializedExperiment:
    spec: ExperimentSpec
    cells: list[ExperimentCell]


class ExperimentService:
    """Local experiment service backed by SQLite and project-local artifacts."""

    def __init__(
        self,
        database: LocalDatabase | None = None,
        resources_root: Path | None = None,
        runs_root: Path | None = None,
    ) -> None:
        if database is None or resources_root is None or runs_root is None:
            from app.core.config import get_settings

            settings = get_settings()
            database = database or LocalDatabase(settings.database_path)
            resources_root = resources_root or settings.resources_root
            runs_root = runs_root or settings.runs_root
        self.database = database
        self.resources_root = resources_root
        self.runs_root = runs_root
        self.database.initialize()

    def materialize(self, spec: ExperimentSpec) -> MaterializedExperiment:
        return MaterializedExperiment(spec=spec, cells=materialize_cells(spec))

    def create_config(self, name: str, selection: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        config_id = f"cfg-{uuid4().hex[:12]}"
        estimate = estimate_selection(selection, self.resources_root)
        normalized = estimate["selection"]
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_configs (
                  id, name, selection_json, cell_count, sample_count,
                  image_operation_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config_id,
                    name.strip() or "Untitled experiment config",
                    dumps_json(normalized),
                    estimate["cellCount"],
                    estimate["sampleCount"],
                    estimate["imageOperationCount"],
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM experiment_configs WHERE id = ?",
                (config_id,),
            ).fetchone()
        return row_to_config(row)

    def list_configs(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiment_configs ORDER BY created_at DESC"
            ).fetchall()
        return [row_to_config(row) for row in rows]

    def get_config(self, config_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_configs WHERE id = ?",
                (config_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown config id: {config_id}")
        return row_to_config(row)

    def create_run(self, config_id: str, *, execute: bool = False) -> dict[str, Any]:
        config = self.get_config(config_id)
        now = utc_now()
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        artifact_root = self.runs_root / safe_segment(run_id)
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_runs (
                  id, config_id, config_name, status, cells, progress,
                  artifact_root, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config["id"],
                    config["name"],
                    "queued",
                    config["cellCount"],
                    0,
                    str(artifact_root),
                    now,
                    now,
                ),
            )

        if execute:
            self.execute_run(run_id)
        return self.get_run(run_id)

    def claim_next_run(self, worker_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM experiment_runs
                WHERE status = ? AND cancel_requested = 0
                ORDER BY created_at ASC
                LIMIT 1
                """,
                ("queued",),
            ).fetchone()
            if row is None:
                return None

            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, worker_id = ?, started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE id = ? AND status = ?
                """,
                ("running", worker_id, now, now, row["id"], "queued"),
            )
            claimed = connection.execute(
                "SELECT * FROM experiment_runs WHERE id = ?",
                (row["id"],),
            ).fetchone()
        return row_to_run(claimed)

    def execute_run(
        self,
        run_id: str,
        *,
        worker_id: str | None = None,
        device: str = "cpu",
        log_path: Path | str | None = None,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] in TERMINAL_STATUSES:
            return run
        if run["cancelRequested"]:
            return self._finish_cancelled_run(run_id)

        config = self.get_config(run["configId"])
        now = utc_now()
        log_path_value = str(log_path) if log_path is not None else run.get("logPath")
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, progress = ?, worker_id = COALESCE(?, worker_id),
                    log_path = COALESCE(?, log_path),
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?
                """,
                ("running", 0, worker_id, log_path_value, now, now, run_id),
            )

        completed = 0

        def should_cancel() -> bool:
            try:
                return bool(self.get_run(run_id)["cancelRequested"])
            except KeyError:
                return True

        def record_cell(cell: dict[str, Any]) -> None:
            nonlocal completed
            completed += 1
            progress = int(round((completed / max(1, run["cells"])) * 100))
            cell_id = f"{run_id}:{cell['cellKey']}"
            timestamp = utc_now()
            with self.database.connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO experiment_cells (
                      id, run_id, cell_key, status, dataset_id, algorithm_id,
                      watermark_method, attack_preset_id, attack_method,
                      attack_strength, seed, sample_count, bit_accuracy,
                      bit_error_rate, elapsed_ms, manifest_path, output_dir,
                      error, summary_json, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cell_id,
                        run_id,
                        cell["cellKey"],
                        cell["status"],
                        cell["datasetId"],
                        cell["algorithmId"],
                        cell["watermarkMethod"],
                        cell["attackPresetId"],
                        cell["attackMethod"],
                        cell["attackStrength"],
                        cell["seed"],
                        cell["sampleCount"],
                        cell["bitAccuracy"],
                        cell.get("bitErrorRate"),
                        cell.get("elapsedMs"),
                        cell["manifestPath"],
                        cell["outputDir"],
                        cell["error"],
                        dumps_json(cell),
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    UPDATE experiment_runs
                    SET progress = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (progress, timestamp, run_id),
                )

        try:
            summary = run_local_experiment(
                LocalRunRequest(
                    run_id=run_id,
                    selection=config["selection"],
                    resources_root=self.resources_root,
                    runs_root=self.runs_root,
                    device=device,
                ),
                on_cell=record_cell,
                should_cancel=should_cancel,
            )
            status = summary["status"]
            error = None
        except Exception as exc:
            summary = None
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        finished = utc_now()
        final_progress = summary["progress"] if summary is not None else 0
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, progress = ?, error = ?, finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, final_progress, error, finished, finished, run_id),
            )
        return self.get_run(run_id)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        now = utc_now()
        if run["status"] in TERMINAL_STATUSES:
            return run
        if run["status"] == "queued":
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE experiment_runs
                    SET status = ?, cancel_requested = 1, finished_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    ("cancelled", now, now, run_id),
                )
            return self.get_run(run_id)

        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET cancel_requested = 1, updated_at = ?
                WHERE id = ?
                """,
                (now, run_id),
            )
        return self.get_run(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiment_runs ORDER BY created_at DESC"
            ).fetchall()
        return [row_to_run(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run id: {run_id}")
        return row_to_run(row)

    def list_run_cells(self, run_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiment_cells WHERE run_id = ? ORDER BY cell_key",
                (run_id,),
            ).fetchall()
        return [row_to_cell(row) for row in rows]

    def get_run_results(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        cells = self.list_run_cells(run_id)
        summary_path = Path(run["artifactRoot"]) / "run_summary.json"
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "run": run,
            "cells": cells,
            "summaryPath": str(summary_path),
            "summaryExists": summary_path.exists(),
            "summary": summary,
            "aggregates": summary.get("aggregates", []) if isinstance(summary, dict) else [],
        }

    def get_run_logs(self, run_id: str, *, max_lines: int = 200) -> dict[str, Any]:
        run = self.get_run(run_id)
        log_path = run.get("logPath") or str(Path(run["artifactRoot"]) / "worker.log")
        path = Path(log_path)
        exists = path.exists()
        lines: list[str] = []
        if exists:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[-max_lines:]
        return {
            "runId": run_id,
            "logPath": log_path,
            "exists": exists,
            "lines": lines,
        }

    def update_worker_heartbeat(
        self,
        *,
        worker_id: str,
        status: str,
        pid: int,
        device: str,
        current_run_id: str | None = None,
        message: str | None = None,
    ) -> None:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO worker_heartbeats (
                  worker_id, status, pid, device, current_run_id, message, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (worker_id, status, pid, device, current_run_id, message, now),
            )

    def list_worker_heartbeats(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM worker_heartbeats ORDER BY last_seen_at DESC"
            ).fetchall()
        return [
            {
                "workerId": row["worker_id"],
                "status": row["status"],
                "pid": row["pid"],
                "device": row["device"],
                "currentRunId": row["current_run_id"],
                "message": row["message"],
                "lastSeenAt": row["last_seen_at"],
            }
            for row in rows
        ]

    def _finish_cancelled_run(self, run_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, cancel_requested = 1, finished_at = COALESCE(finished_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                ("cancelled", now, now, run_id),
            )
        return self.get_run(run_id)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
