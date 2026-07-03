from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.local_db import (
    LocalDatabase,
    dumps_json,
    row_to_config,
    row_to_run,
)
from app.core.storage import safe_segment
from app.services.local_artifacts import default_phase_states, read_json_object, read_jsonl
from app.services.local_runner import LocalRunRequest, estimate_selection, run_local_experiment
from app.services.resources import get_attack_catalog_item, get_dataset_by_id, get_watermark_catalog_item
from app.services.runtime_resource_manager import release_runtime_resources
from app.services.runtime_parallel_config import apply_runtime_parallel_env
from app.services.scoring import PROTOCOL_ID, aggregate_benchmark_score, benchmark_protocols, score_cell_from_records


TERMINAL_STATUSES = {"succeeded", "failed", "paused", "cancelled", "partially_failed"}
RESUMABLE_STATUSES = {"paused", "failed", "partially_failed"}
STOP_INTENT_CANCEL = "cancel"
STOP_INTENT_PAUSE = "pause"
HIDDEN_BASELINE_ATTACK_ID = "atk-identity"
WORKER_HEARTBEAT_RETENTION_SECONDS = 3600


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


def _tail_lines(path: Path, max_lines: int, *, chunk_size: int = 64 * 1024) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            chunks: list[bytes] = []
            newline_count = 0
            while position > 0 and newline_count <= max_lines:
                read_size = min(chunk_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
            data = b"".join(reversed(chunks))
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


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
        self._cache_lock = threading.Lock()
        self._result_units_cache: dict[str, tuple[tuple[Any, ...], list[dict[str, Any]]]] = {}
        self._score_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

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
        self.reconcile_stale_runs()
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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

            cursor = connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, worker_id = ?, started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE id = ? AND status = ?
                """,
                ("running", worker_id, now, now, row["id"], "queued"),
            )
            if cursor.rowcount != 1:
                return None
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
            return self._finish_stopped_run(run_id, self._stop_intent(run))

        apply_runtime_parallel_env(self.runs_root)
        config = self.get_config(run["configId"])
        now = utc_now()
        log_path_value = str(log_path) if log_path is not None else run.get("logPath")
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, worker_id = COALESCE(?, worker_id),
                    log_path = COALESCE(?, log_path),
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ?
                """,
                ("running", worker_id, log_path_value, now, now, run_id),
            )

        heartbeat_stop: threading.Event | None = None
        heartbeat_thread: threading.Thread | None = None
        if worker_id:
            heartbeat_stop, heartbeat_thread = self._start_run_heartbeat(
                worker_id=worker_id,
                device=device,
                run_id=run_id,
            )

        def stop_intent() -> str | None:
            try:
                current_run = self.get_run(run_id)
            except KeyError:
                return STOP_INTENT_CANCEL
            if current_run["cancelRequested"]:
                return self._stop_intent(current_run)
            return None

        def record_run_state(state: dict[str, Any]) -> None:
            progress = int(state.get("overallProgress") or state.get("progress") or 0)
            timestamp = utc_now()
            with self.database.connect() as connection:
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
                on_state=record_run_state,
                should_cancel=stop_intent,
            )
            status = self._run_status_from_summary(run_id, summary["status"])
            error = None
        except Exception as exc:
            summary = None
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            cleanup_errors = release_runtime_resources()
            if cleanup_errors:
                error = f"{error}; cleanup errors: {'; '.join(cleanup_errors)}"
        finally:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2.0)

        finished = utc_now()
        final_progress = (
            summary["progress"]
            if summary is not None
            else self.get_run(run_id).get("progress", 0)
        )
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

    def _start_run_heartbeat(
        self,
        *,
        worker_id: str,
        device: str,
        run_id: str,
    ) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()
        interval = max(1.0, min(10.0, _worker_poll_seconds()))

        def beat() -> None:
            while not stop.is_set():
                try:
                    self.update_worker_heartbeat(
                        worker_id=worker_id,
                        status="running",
                        pid=os.getpid(),
                        device=device,
                        current_run_id=run_id,
                        message=f"executing {run_id}",
                    )
                except Exception:
                    pass
                stop.wait(interval)

        thread = threading.Thread(target=beat, name=f"wmbench-heartbeat-{run_id}", daemon=True)
        thread.start()
        return stop, thread

    def resume_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] in {"queued", "running"}:
            return run
        if run["status"] not in RESUMABLE_STATUSES:
            raise ValueError(f"Run cannot be resumed from status: {run['status']}")
        now = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, progress = ?, cancel_requested = 0, error = NULL,
                    stop_intent = NULL, worker_id = NULL, finished_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                ("queued", run["progress"], now, run_id),
            )
        return self.get_run(run_id)

    def pause_run(self, run_id: str) -> dict[str, Any]:
        return self._request_stop_run(run_id, STOP_INTENT_PAUSE)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._request_stop_run(run_id, STOP_INTENT_CANCEL)

    def _request_stop_run(self, run_id: str, intent: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        now = utc_now()
        if run["status"] in TERMINAL_STATUSES:
            return run
        if run["status"] == "queued":
            status = "paused" if intent == STOP_INTENT_PAUSE else "cancelled"
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE experiment_runs
                    SET status = ?, cancel_requested = 1, stop_intent = ?, finished_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, intent, now, now, run_id),
                )
            return self.get_run(run_id)

        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET cancel_requested = 1, stop_intent = ?, updated_at = ?
                WHERE id = ?
                """,
                (intent, now, run_id),
            )
        return self.get_run(run_id)

    def list_runs(self, *, scope: str | None = None) -> list[dict[str, Any]]:
        self.reconcile_stale_runs()
        active_statuses = ["queued", "running"]
        unfinished_statuses = ["queued", "running", "paused", "failed", "partially_failed"]
        with self.database.connect() as connection:
            if scope == "active":
                rows = connection.execute(
                    """
                    SELECT * FROM experiment_runs
                    WHERE status IN (?, ?)
                    ORDER BY
                      CASE status
                        WHEN 'running' THEN 0
                        WHEN 'queued' THEN 1
                        ELSE 2
                      END,
                      updated_at DESC
                    """,
                    tuple(active_statuses),
                ).fetchall()
                runs = [self._enrich_run_with_state(row_to_run(row)) for row in rows]
                return self._merge_file_backed_runs(runs, scope=scope)
            if scope == "unfinished":
                rows = connection.execute(
                    """
                    SELECT * FROM experiment_runs
                    WHERE status IN (?, ?, ?, ?, ?)
                    ORDER BY
                      CASE status
                        WHEN 'running' THEN 0
                        WHEN 'queued' THEN 1
                        WHEN 'paused' THEN 2
                        WHEN 'failed' THEN 3
                        WHEN 'partially_failed' THEN 4
                        ELSE 5
                      END,
                      updated_at DESC
                    """,
                    tuple(unfinished_statuses),
                ).fetchall()
                runs = [self._enrich_run_with_state(row_to_run(row)) for row in rows]
                return self._merge_file_backed_runs(runs, scope=scope)
            rows = connection.execute(
                "SELECT * FROM experiment_runs ORDER BY created_at DESC"
            ).fetchall()
        runs = [self._enrich_run_with_state(row_to_run(row)) for row in rows]
        return self._merge_file_backed_runs(runs, scope=scope)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            file_run = self._file_backed_run(run_id)
            if file_run is not None:
                return file_run
            raise KeyError(f"Unknown run id: {run_id}")
        return self._enrich_run_with_state(row_to_run(row))

    def _merge_file_backed_runs(self, runs: list[dict[str, Any]], *, scope: str | None) -> list[dict[str, Any]]:
        existing_ids = {str(run.get("id")) for run in runs}
        merged = list(runs)
        for run_dir in self._iter_file_run_dirs():
            if run_dir.name in existing_ids:
                continue
            file_run = self._file_backed_run(run_dir.name)
            if file_run is None or not self._run_matches_scope(file_run, scope):
                continue
            merged.append(file_run)
        return sorted(
            merged,
            key=lambda run: str(run.get("createdAt") or run.get("updatedAt") or ""),
            reverse=True,
        )

    def _iter_file_run_dirs(self) -> list[Path]:
        if not self.runs_root.exists():
            return []
        run_dirs = [
            path
            for path in self.runs_root.iterdir()
            if path.is_dir()
            and path.name.startswith("run_")
            and (
                (path / "run_summary.json").exists()
                or (path / "run_state.json").exists()
                or (path / "result_units.jsonl").exists()
            )
        ]
        return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)

    def _file_backed_run(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self.runs_root / safe_segment(run_id)
        if not run_dir.is_dir():
            return None
        summary = read_json_object(run_dir / "run_summary.json")
        state = read_json_object(run_dir / "run_state.json")
        status_doc = read_json_object(run_dir / "run_status.json")
        plan = read_json_object(run_dir / "run_plan.json")
        if not any((summary, state, status_doc, plan, (run_dir / "result_units.jsonl").exists())):
            return None

        selection = self._first_dict(summary.get("selection"), state.get("selection"), plan.get("selection"))
        created_at = self._first_string(
            plan.get("createdAt"),
            summary.get("createdAt"),
            state.get("createdAt"),
            self._path_timestamp(run_dir),
        )
        updated_at = self._first_string(
            status_doc.get("updatedAt"),
            state.get("updatedAt"),
            summary.get("updatedAt"),
            self._path_timestamp(run_dir),
        )
        status = self._first_string(status_doc.get("status"), state.get("status"), summary.get("status"), "succeeded")
        cells = self._first_int(
            status_doc.get("expectedResultUnits"),
            state.get("expectedResultUnits"),
            plan.get("expectedCells"),
            summary.get("resultUnitCount"),
            0,
        )
        progress = self._first_int(
            status_doc.get("progress"),
            state.get("overallProgress"),
            state.get("progress"),
            summary.get("progress"),
            100 if status in TERMINAL_STATUSES else 0,
        )
        config_name = self._first_string(
            summary.get("configName"),
            state.get("configName"),
            plan.get("configName"),
            self._imported_run_name(selection),
        )
        log_path = run_dir / "worker.log"
        run = {
            "id": run_id,
            "taskName": self._first_string(summary.get("taskName"), summary.get("runName"), config_name),
            "configId": self._first_string(summary.get("configId"), state.get("configId"), plan.get("configId"), f"imported-{run_id}"),
            "configName": config_name,
            "status": status,
            "cells": cells,
            "progress": progress,
            "completedProgress": progress,
            "progressKind": self._first_string(status_doc.get("progressKind"), state.get("progressKind"), summary.get("progressKind"), "phaseOperations"),
            "artifactRoot": str(run_dir),
            "logPath": str(log_path) if log_path.exists() else None,
            "workerId": None,
            "cancelRequested": False,
            "stopIntent": None,
            "error": status_doc.get("error") or state.get("error") or summary.get("error"),
            "createdAt": created_at,
            "updatedAt": updated_at,
            "startedAt": self._first_string(summary.get("startedAt"), state.get("startedAt"), plan.get("startedAt"), None),
            "finishedAt": self._first_string(summary.get("finishedAt"), state.get("finishedAt"), status_doc.get("finishedAt"), None),
        }
        return self._enrich_run_with_state(run)

    def _run_matches_scope(self, run: dict[str, Any], scope: str | None) -> bool:
        status = str(run.get("status") or "")
        if scope == "active":
            return status in {"queued", "running"}
        if scope == "unfinished":
            return status in {"queued", "running", "paused", "failed", "partially_failed"}
        return True

    def _imported_run_name(self, selection: dict[str, Any]) -> str:
        dataset_count = len(selection.get("datasetIds") or [])
        algorithm_count = len(selection.get("algorithmIds") or [])
        attack_count = len(selection.get("attackPresetIds") or [])
        return f"Imported run ({dataset_count} datasets, {algorithm_count} algorithms, {attack_count} attacks)"

    def _path_timestamp(self, path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()

    def _first_string(self, *values: Any) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _first_int(self, *values: Any) -> int:
        for value in values:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _first_dict(self, *values: Any) -> dict[str, Any]:
        for value in values:
            if isinstance(value, dict):
                return value
        return {}

    def _run_state_path(self, run: dict[str, Any]) -> Path:
        return Path(str(run["artifactRoot"])) / "run_state.json"

    def _phase_state_path(self, run: dict[str, Any]) -> Path:
        return Path(str(run["artifactRoot"])) / "phase_state.json"

    def _artifact_tree_path(self, run: dict[str, Any]) -> Path:
        return Path(str(run["artifactRoot"])) / "artifact_tree.json"

    def _enrich_run_with_state(self, run: dict[str, Any]) -> dict[str, Any]:
        state = read_json_object(self._run_state_path(run))
        if not state:
            current_phase = "summary" if run["status"] in TERMINAL_STATUSES else "canonical"
            return {
                **run,
                "progressKind": "phaseOperations",
                "currentPhase": current_phase,
                "phases": default_phase_states(),
                "runStatePath": str(self._run_state_path(run)),
                "phaseStatePath": str(self._phase_state_path(run)),
                "artifactTreePath": str(self._artifact_tree_path(run)),
            }
        progress = int(state.get("overallProgress") or state.get("progress") or run.get("progress") or 0)
        return {
            **run,
            "progress": progress,
            "completedProgress": progress,
            "progressKind": str(state.get("progressKind") or "phaseOperations"),
            "currentPhase": state.get("currentPhase"),
            "phases": state.get("phases") if isinstance(state.get("phases"), list) else [],
            "runStatePath": str(self._run_state_path(run)),
            "phaseStatePath": str(self._phase_state_path(run)),
            "artifactTreePath": str(self._artifact_tree_path(run)),
        }

    def get_run_state(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        state = read_json_object(self._run_state_path(run))
        if not state:
            phases = default_phase_states()
            current_phase = "summary" if run["status"] in TERMINAL_STATUSES else "canonical"
            state = {
                "runId": run_id,
                "status": run["status"],
                "currentPhase": current_phase,
                "overallProgress": run["progress"],
                "progress": run["progress"],
                "progressKind": "phaseOperations",
                "expectedResultUnits": run["cells"],
                "artifactRoot": run["artifactRoot"],
                "phaseStatePath": str(self._phase_state_path(run)),
                "artifactTreePath": str(self._artifact_tree_path(run)),
                "summaryPath": str(Path(run["artifactRoot"]) / "run_summary.json"),
                "phases": phases,
                "updatedAt": run.get("updatedAt"),
            }
        return {**state, "run": run}

    def get_run_tree(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        tree = read_json_object(self._artifact_tree_path(run))
        if not tree:
            return {
                "runId": run_id,
                "artifactRoot": run["artifactRoot"],
                "datasets": {},
                "exists": False,
            }
        return {**tree, "exists": True, "artifactRoot": run["artifactRoot"]}

    def list_run_result_units(self, run_id: str) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        return self._list_run_result_units_for_run(run)

    def _list_run_result_units_for_run(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        cache_key = str(run["id"])
        signature = self._run_artifact_signature(run, include_summary=False)
        with self._cache_lock:
            cached = self._result_units_cache.get(cache_key)
            if cached is not None and cached[0] == signature:
                return cached[1]

        result_units = read_jsonl(Path(run["artifactRoot"]) / "result_units.jsonl")
        latest: dict[str, dict[str, Any]] = {}
        for unit in result_units:
            key = unit.get("resultUnitKey") or unit.get("cellKey")
            if isinstance(key, str):
                latest[key] = unit
        units = sorted(latest.values(), key=lambda unit: str(unit.get("resultUnitKey") or unit.get("cellKey") or ""))
        scored_units = self._attach_scoring_from_run_records(run, units)
        with self._cache_lock:
            self._result_units_cache[cache_key] = (signature, scored_units)
        return scored_units

    def _attach_scoring_from_run_records(self, run: dict[str, Any], result_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not result_units or all(isinstance(unit.get("scoring"), dict) for unit in result_units):
            return result_units
        artifact_root = Path(str(run["artifactRoot"]))
        detection_records = read_jsonl(artifact_root / "image_detection.jsonl")
        quality_records = read_jsonl(artifact_root / "image_quality.jsonl")
        if not detection_records and not quality_records:
            return result_units

        detections_by_cell: dict[str, list[dict[str, Any]]] = {}
        quality_by_cell: dict[str, list[dict[str, Any]]] = {}
        clean_quality_by_algorithm: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

        for record in detection_records:
            key = record.get("cellKey")
            if isinstance(key, str):
                detections_by_cell.setdefault(key, []).append(record)

        for record in quality_records:
            scope = record.get("scope")
            if scope == "original_vs_watermarked":
                clean_key = (
                    str(record.get("datasetId") or ""),
                    str(record.get("algorithmId") or ""),
                    str(record.get("seed") or ""),
                )
                clean_quality_by_algorithm.setdefault(clean_key, []).append(record)
                continue
            if scope != "original_vs_attacked_watermarked":
                continue
            key = record.get("cellKey")
            if isinstance(key, str):
                quality_by_cell.setdefault(key, []).append(record)

        scored_units: list[dict[str, Any]] = []
        for unit in result_units:
            if isinstance(unit.get("scoring"), dict):
                scored_units.append(unit)
                continue
            key = str(unit.get("resultUnitKey") or unit.get("cellKey") or "")
            clean_key = (
                str(unit.get("datasetId") or ""),
                str(unit.get("algorithmId") or ""),
                str(unit.get("seed") or ""),
            )
            try:
                attack_strength = float(unit.get("attackStrength") or 0.0)
            except (TypeError, ValueError):
                attack_strength = 0.0
            try:
                sample_count = int(unit.get("sampleCount") or 0)
            except (TypeError, ValueError):
                sample_count = 0
            attack_params = unit.get("attackParams") if isinstance(unit.get("attackParams"), dict) else {}
            scoring = score_cell_from_records(
                algorithm_id=str(unit.get("algorithmId") or ""),
                attack_preset_id=str(unit.get("attackPresetId") or ""),
                attack_method=str(unit.get("attackMethod") or unit.get("attackPresetId") or ""),
                attack_strength=attack_strength,
                sample_count=sample_count,
                detection_records=detections_by_cell.get(key, []),
                quality_records=quality_by_cell.get(key, []),
                clean_quality_records=clean_quality_by_algorithm.get(clean_key, []),
                elapsed_ms=float(unit.get("elapsedMs") or 0.0),
                attack_params=attack_params,
            )
            scored_units.append({**unit, "scoring": scoring})
        return scored_units

    def get_run_results(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        summary_path, summary_exists, summary = self._read_run_summary(run)
        result_units = self._list_run_result_units_for_run(run)
        score = self._score_for_run(run, summary, result_units)
        response_summary = self._summary_for_response(summary)
        return {
            "run": run,
            "resultUnits": result_units,
            "summaryPath": str(summary_path),
            "summaryExists": summary_exists,
            "summary": response_summary,
            "aggregates": response_summary.get("aggregates", []) if isinstance(response_summary, dict) else [],
            "score": score,
        }

    def get_run_score(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        summary_path = Path(run["artifactRoot"]) / "run_summary.json"
        summary_exists = summary_path.exists()
        signature = self._run_artifact_signature(run, include_summary=True)
        with self._cache_lock:
            cached = self._score_cache.get(str(run["id"]))
            if cached is not None and cached[0] == signature:
                return {
                    "run": run,
                    "score": cached[1],
                    "summaryPath": str(summary_path),
                    "summaryExists": summary_exists,
                }
        summary_path, summary_exists, summary = self._read_run_summary(run)
        return {
            "run": run,
            "score": self._score_for_run(run, summary),
            "summaryPath": str(summary_path),
            "summaryExists": summary_exists,
        }

    def _read_run_summary(self, run: dict[str, Any]) -> tuple[Path, bool, dict[str, Any] | None]:
        summary_path = Path(run["artifactRoot"]) / "run_summary.json"
        if not summary_path.exists():
            return summary_path, False, None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            return summary_path, True, None
        return (
            summary_path,
            True,
            {
                **summary,
                "runId": run["id"],
                "status": run["status"],
                "progress": run["progress"],
                "completedProgress": run["completedProgress"],
                "progressKind": summary.get("progressKind") or run["progressKind"],
            },
        )

    @staticmethod
    def _summary_for_response(summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(summary, dict):
            return summary
        result_units = summary.get("resultUnits")
        if not isinstance(result_units, list):
            return summary
        response_summary = dict(summary)
        response_summary["resultUnitCount"] = len(result_units)
        response_summary.pop("resultUnits", None)
        return response_summary

    def _score_for_run(
        self,
        run: dict[str, Any],
        summary: dict[str, Any] | None,
        result_units: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cache_key = str(run["id"])
        signature = self._run_artifact_signature(run, include_summary=True)
        with self._cache_lock:
            cached = self._score_cache.get(cache_key)
            if cached is not None and cached[0] == signature:
                return cached[1]

        if isinstance(summary, dict) and isinstance(summary.get("score"), dict):
            score = summary["score"]
        else:
            if result_units is None:
                result_units = self._list_run_result_units_for_run(run)
            score = self._score_from_summary_or_result_units(summary, result_units)
        with self._cache_lock:
            self._score_cache[cache_key] = (signature, score)
        return score

    def _run_artifact_signature(self, run: dict[str, Any], *, include_summary: bool) -> tuple[Any, ...]:
        artifact_root = Path(str(run["artifactRoot"]))
        filenames = ["result_units.jsonl", "image_detection.jsonl", "image_quality.jsonl"]
        if include_summary:
            filenames.append("run_summary.json")
        parts: list[Any] = [str(artifact_root)]
        for filename in filenames:
            path = artifact_root / filename
            try:
                stat = path.stat()
            except OSError:
                parts.append((filename, None, None))
            else:
                parts.append((filename, stat.st_mtime_ns, stat.st_size))
        return tuple(parts)

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
            lines = _tail_lines(path, max_lines)
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
            self._prune_worker_heartbeats(connection, now=now)
            connection.execute(
                """
                INSERT OR REPLACE INTO worker_heartbeats (
                  worker_id, status, pid, device, current_run_id, message, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (worker_id, status, pid, device, current_run_id, message, now),
            )

    def _prune_worker_heartbeats(self, connection: Any, *, now: str) -> None:
        try:
            now_dt = datetime.fromisoformat(now)
        except ValueError:
            return
        cutoff = now_dt.timestamp() - WORKER_HEARTBEAT_RETENTION_SECONDS
        rows = connection.execute("SELECT worker_id, last_seen_at FROM worker_heartbeats").fetchall()
        stale_ids: list[str] = []
        for row in rows:
            try:
                last_seen = datetime.fromisoformat(row["last_seen_at"])
            except (TypeError, ValueError):
                stale_ids.append(row["worker_id"])
                continue
            if last_seen.timestamp() < cutoff:
                stale_ids.append(row["worker_id"])
        if stale_ids:
            connection.executemany(
                "DELETE FROM worker_heartbeats WHERE worker_id = ?",
                [(worker_id,) for worker_id in stale_ids],
            )

    def _prune_dead_worker_heartbeats(self, connection: Any) -> None:
        if os.name != "posix" or not Path("/proc").exists():
            return
        rows = connection.execute("SELECT worker_id, pid FROM worker_heartbeats").fetchall()
        dead_ids = [
            row["worker_id"]
            for row in rows
            if isinstance(row["pid"], int) and row["pid"] > 0 and not Path(f"/proc/{row['pid']}").exists()
        ]
        if dead_ids:
            connection.executemany(
                "DELETE FROM worker_heartbeats WHERE worker_id = ?",
                [(worker_id,) for worker_id in dead_ids],
            )

    def list_worker_heartbeats(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            self._prune_dead_worker_heartbeats(connection)
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

    def reconcile_stale_runs(self, *, stale_seconds: float | None = None) -> int:
        """Mark orphaned running tasks as paused when no fresh worker is executing them."""
        poll_seconds = _worker_poll_seconds()
        threshold = stale_seconds if stale_seconds is not None else max(120.0, poll_seconds * 15)
        workers = self.list_worker_heartbeats()
        active_run_ids = {
            worker["currentRunId"]
            for worker in workers
            if worker.get("currentRunId") and _is_fresh_worker(worker, poll_seconds)
        }

        now = datetime.now(timezone.utc)
        reconciled = 0
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, updated_at FROM experiment_runs WHERE status = ?",
                ("running",),
            ).fetchall()
            for row in rows:
                run_id = row["id"]
                if run_id in active_run_ids:
                    continue
                try:
                    updated_at = datetime.fromisoformat(str(row["updated_at"]))
                except ValueError:
                    updated_at = now
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                if (now - updated_at).total_seconds() < threshold:
                    continue

                timestamp = utc_now()
                connection.execute(
                    """
                    UPDATE experiment_runs
                    SET status = ?, progress = progress, error = ?, worker_id = NULL,
                        stop_intent = ?, cancel_requested = 1,
                        finished_at = COALESCE(finished_at, ?), updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        "paused",
                        "Worker stopped before completion; run auto-paused for resume.",
                        STOP_INTENT_PAUSE,
                        timestamp,
                        timestamp,
                        run_id,
                        "running",
                    ),
                )
                reconciled += 1
        return reconciled

    def _finish_stopped_run(self, run_id: str, intent: str) -> dict[str, Any]:
        now = utc_now()
        status = "paused" if intent == STOP_INTENT_PAUSE else "cancelled"
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, cancel_requested = 1, stop_intent = ?, finished_at = COALESCE(finished_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (status, intent, now, now, run_id),
            )
        return self.get_run(run_id)

    def _run_status_from_summary(self, run_id: str, summary_status: str) -> str:
        run = self.get_run(run_id)
        if summary_status == "cancelled" and run["cancelRequested"]:
            return "paused" if self._stop_intent(run) == STOP_INTENT_PAUSE else "cancelled"
        return summary_status

    def _stop_intent(self, run: dict[str, Any]) -> str:
        intent = run.get("stopIntent")
        if intent in {STOP_INTENT_CANCEL, STOP_INTENT_PAUSE}:
            return str(intent)
        return STOP_INTENT_PAUSE

    def _score_from_summary_or_result_units(
        self,
        summary: dict[str, Any] | None,
        result_units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(summary, dict) and isinstance(summary.get("score"), dict):
            return summary["score"]
        if any(isinstance(unit.get("scoring"), dict) for unit in result_units):
            return aggregate_benchmark_score(result_units)
        if isinstance(summary, dict) and isinstance(summary.get("resultUnits"), list):
            return aggregate_benchmark_score(summary["resultUnits"])
        return aggregate_benchmark_score(result_units)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_poll_seconds() -> float:
    import os

    return float(os.getenv("WM_BENCH_WORKER_POLL_SECONDS", "2"))


def _is_fresh_worker(worker: dict[str, Any], poll_seconds: float) -> bool:
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
