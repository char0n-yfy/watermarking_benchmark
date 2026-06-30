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
from app.services.resources import get_attack_catalog_item, get_dataset_by_id, get_watermark_catalog_item
from app.services.scoring import PROTOCOL_ID, aggregate_benchmark_score, benchmark_protocols


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "partially_failed"}
HIDDEN_BASELINE_ATTACK_ID = "atk-identity"


@dataclass(frozen=True)
class MaterializedExperiment:
    spec: ExperimentSpec
    cells: list[ExperimentCell]


def with_hidden_baseline_attack(selection: dict[str, Any]) -> dict[str, Any]:
    next_selection = dict(selection)
    attack_ids = [str(attack_id) for attack_id in next_selection.get("attackPresetIds") or []]
    if HIDDEN_BASELINE_ATTACK_ID not in attack_ids:
        attack_ids.append(HIDDEN_BASELINE_ATTACK_ID)
    next_selection["attackPresetIds"] = attack_ids
    return next_selection


def _selection_id_list(selection: dict[str, Any], field: str) -> list[str]:
    value = selection.get(field) or []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    return [str(item) for item in value]


def _selection_override_keys(selection: dict[str, Any], field: str) -> list[str]:
    value = selection.get(field) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return [str(item) for item in value.keys()]


def validate_selection_resource_ids(selection: dict[str, Any], resources_root: Path) -> None:
    for dataset_id in _selection_id_list(selection, "datasetIds"):
        get_dataset_by_id(resources_root, dataset_id)
    for algorithm_id in _selection_id_list(selection, "algorithmIds"):
        get_watermark_catalog_item(algorithm_id)

    attack_ids = _selection_id_list(selection, "attackPresetIds")
    for attack_id in attack_ids:
        get_attack_catalog_item(attack_id)

    selected_attack_ids = set(attack_ids)
    for field in ("attackStrengthOverrides", "attackParamOverrides"):
        unknown_override_ids = sorted(
            attack_id
            for attack_id in _selection_override_keys(selection, field)
            if attack_id not in selected_attack_ids
        )
        if unknown_override_ids:
            raise ValueError(
                f"{field} contains ids that are not selected in attackPresetIds: "
                + ", ".join(unknown_override_ids)
            )


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
        selection_with_baseline = with_hidden_baseline_attack(selection)
        validate_selection_resource_ids(selection_with_baseline, self.resources_root)
        estimate = estimate_selection(selection_with_baseline, self.resources_root)
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
                "SELECT * FROM experiment_configs WHERE deleted_at IS NULL ORDER BY created_at DESC"
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

    def rename_config(self, config_id: str, name: str) -> dict[str, Any]:
        next_name = name.strip()
        if not next_name:
            raise ValueError("Config name cannot be empty")
        now = utc_now()
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_configs WHERE id = ? AND deleted_at IS NULL",
                (config_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown config id: {config_id}")
            connection.execute(
                """
                UPDATE experiment_configs
                SET name = ?, updated_at = ?
                WHERE id = ?
                """,
                (next_name, now, config_id),
            )
            updated = connection.execute(
                "SELECT * FROM experiment_configs WHERE id = ?",
                (config_id,),
            ).fetchone()
        return row_to_config(updated)

    def delete_config(self, config_id: str) -> dict[str, str]:
        now = utc_now()
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_configs WHERE id = ? AND deleted_at IS NULL",
                (config_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown config id: {config_id}")
            connection.execute(
                """
                UPDATE experiment_configs
                SET deleted_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, config_id),
            )
        return {"id": config_id, "status": "deleted"}

    def create_run(self, config_id: str, *, execute: bool = False, name: str | None = None) -> dict[str, Any]:
        config = self.get_config(config_id)
        now = utc_now()
        run_name = (name or "").strip() or config["name"]
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        artifact_root = self.runs_root / safe_segment(run_id)
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_runs (
                  id, config_id, config_name, run_name, status, cells, progress,
                  artifact_root, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config["id"],
                    config["name"],
                    run_name,
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
        existing_completed = sum(
            1 for cell in self.list_run_cells(run_id) if cell.get("status") == "succeeded"
        )
        initial_progress = int(round((existing_completed / max(1, run["cells"])) * 100))
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, progress = ?, worker_id = COALESCE(?, worker_id),
                    log_path = COALESCE(?, log_path),
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?
                """,
                ("running", initial_progress, worker_id, log_path_value, now, now, run_id),
            )

        completed = existing_completed

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
                        cell.get("bitAccuracy"),
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
                    resume=True,
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

    def resume_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] in {"queued", "running"}:
            return run
        if run["status"] == "succeeded":
            return run
        existing_completed = sum(
            1 for cell in self.list_run_cells(run_id) if cell.get("status") == "succeeded"
        )
        progress = int(round((existing_completed / max(1, run["cells"])) * 100))
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, progress = ?, cancel_requested = 0, error = NULL,
                    worker_id = NULL, finished_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                ("queued", progress, now, run_id),
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

    def list_runs(self, *, scope: str | None = None) -> list[dict[str, Any]]:
        active_statuses = ["queued", "running", "failed", "cancelled", "partially_failed"]
        with self.database.connect() as connection:
            if scope == "active":
                rows = connection.execute(
                    """
                    SELECT * FROM experiment_runs
                    WHERE status IN (?, ?, ?, ?, ?)
                    ORDER BY
                      CASE status
                        WHEN 'running' THEN 0
                        WHEN 'queued' THEN 1
                        WHEN 'partially_failed' THEN 2
                        WHEN 'failed' THEN 3
                        WHEN 'cancelled' THEN 4
                        ELSE 5
                      END,
                      updated_at DESC
                    """,
                    tuple(active_statuses),
                ).fetchall()
                return [row_to_run(row) for row in rows]
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
            "score": self._score_from_summary_or_cells(summary, cells),
        }

    def get_run_score(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        results = self.get_run_results(run_id)
        return {
            "run": run,
            "score": results["score"],
            "summaryPath": results["summaryPath"],
            "summaryExists": results["summaryExists"],
        }

    def list_benchmark_protocols(self) -> list[dict[str, Any]]:
        return benchmark_protocols()

    def list_leaderboard(self, protocol_id: str = PROTOCOL_ID) -> dict[str, Any]:
        if protocol_id != PROTOCOL_ID:
            raise KeyError(f"Unknown benchmark protocol: {protocol_id}")
        rows: list[dict[str, Any]] = []
        for run in self.list_runs():
            if run["status"] not in {"succeeded", "partially_failed"}:
                continue
            score_response = self.get_run_score(run["id"])
            score = score_response["score"]
            for row in score.get("leaderboardRows", []):
                rows.append(
                    {
                        **row,
                        "runId": run["id"],
                        "runStatus": run["status"],
                        "configId": run["configId"],
                        "configName": run["configName"],
                        "updatedAt": run["updatedAt"],
                    }
                )
        rows.sort(key=lambda row: (row["officialEligible"], row["wrs"] is not None, row["wrs"] or -1), reverse=True)
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return {
            "protocol": benchmark_protocols()[0],
            "rows": rows,
            "officialRows": [row for row in rows if row.get("officialEligible")],
            "provisionalRows": [row for row in rows if not row.get("officialEligible")],
            "generatedAt": utc_now(),
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

    def get_run_events(self, run_id: str, *, max_events: int = 80) -> dict[str, Any]:
        run = self.get_run(run_id)
        event_path = Path(run["artifactRoot"]) / "stage_events.jsonl"
        exists = event_path.exists()
        events: list[dict[str, Any]] = []
        if exists:
            for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        return {
            "runId": run_id,
            "eventPath": str(event_path),
            "exists": exists,
            "events": events[-max_events:],
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

    def _score_from_summary_or_cells(
        self,
        summary: dict[str, Any] | None,
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(summary, dict) and isinstance(summary.get("score"), dict):
            return summary["score"]
        if isinstance(summary, dict) and isinstance(summary.get("cells"), list):
            return aggregate_benchmark_score(summary["cells"])
        hydrated_cells: list[dict[str, Any]] = []
        for cell in cells:
            cell_summary = cell.get("summary") if isinstance(cell.get("summary"), dict) else {}
            hydrated_cells.append({**cell, **cell_summary})
        return aggregate_benchmark_score(hydrated_cells)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
