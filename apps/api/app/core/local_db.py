from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


JsonDict = dict[str, Any]


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_configs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  selection_json TEXT NOT NULL,
  cell_count INTEGER NOT NULL,
  sample_count INTEGER NOT NULL,
  image_operation_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
  id TEXT PRIMARY KEY,
  config_id TEXT NOT NULL,
  config_name TEXT NOT NULL,
  run_name TEXT,
  status TEXT NOT NULL,
  cells INTEGER NOT NULL,
  progress INTEGER NOT NULL,
  artifact_root TEXT NOT NULL,
  log_path TEXT,
  worker_id TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  stop_intent TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  FOREIGN KEY (config_id) REFERENCES experiment_configs(id)
);

CREATE TABLE IF NOT EXISTS experiment_cells (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  cell_key TEXT NOT NULL,
  status TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  algorithm_id TEXT NOT NULL,
  watermark_method TEXT NOT NULL,
  attack_preset_id TEXT NOT NULL,
  attack_method TEXT NOT NULL,
  attack_strength REAL NOT NULL,
  seed INTEGER NOT NULL,
  sample_count INTEGER NOT NULL,
  bit_accuracy REAL,
  bit_error_rate REAL,
  elapsed_ms REAL,
  manifest_path TEXT,
  output_dir TEXT,
  error TEXT,
  summary_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS worker_heartbeats (
  worker_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  pid INTEGER NOT NULL,
  device TEXT NOT NULL,
  current_run_id TEXT,
  message TEXT,
  last_seen_at TEXT NOT NULL
);
"""


MIGRATION_COLUMNS = {
    "experiment_runs": {
        "run_name": "TEXT",
        "log_path": "TEXT",
        "worker_id": "TEXT",
        "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
        "stop_intent": "TEXT",
    },
    "experiment_cells": {
        "bit_error_rate": "REAL",
        "elapsed_ms": "REAL",
    },
    "experiment_configs": {
        "deleted_at": "TEXT",
    },
}


class LocalDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            for table_name, columns in MIGRATION_COLUMNS.items():
                existing = {
                    row["name"]
                    for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
                }
                for column_name, column_spec in columns.items():
                    if column_name not in existing:
                        connection.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}"
                        )


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads_json(value: str | None) -> Any:
    if not value:
        return {}
    return json.loads(value)


def row_to_config(row: sqlite3.Row) -> JsonDict:
    return {
        "id": row["id"],
        "name": row["name"],
        "selection": loads_json(row["selection_json"]),
        "cellCount": row["cell_count"],
        "sampleCount": row["sample_count"],
        "imageOperationCount": row["image_operation_count"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def row_to_run(row: sqlite3.Row) -> JsonDict:
    return {
        "id": row["id"],
        "taskName": row["run_name"] or row["config_name"],
        "configId": row["config_id"],
        "configName": row["config_name"],
        "status": row["status"],
        "cells": row["cells"],
        "progress": row["progress"],
        "completedProgress": row["progress"],
        "progressKind": "completedCells",
        "artifactRoot": row["artifact_root"],
        "logPath": row["log_path"],
        "workerId": row["worker_id"],
        "cancelRequested": bool(row["cancel_requested"]),
        "stopIntent": row["stop_intent"],
        "error": row["error"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
    }


def row_to_cell(row: sqlite3.Row) -> JsonDict:
    return {
        "id": row["id"],
        "runId": row["run_id"],
        "cellKey": row["cell_key"],
        "status": row["status"],
        "datasetId": row["dataset_id"],
        "algorithmId": row["algorithm_id"],
        "watermarkMethod": row["watermark_method"],
        "attackPresetId": row["attack_preset_id"],
        "attackMethod": row["attack_method"],
        "attackStrength": row["attack_strength"],
        "seed": row["seed"],
        "sampleCount": row["sample_count"],
        "bitAccuracy": row["bit_accuracy"],
        "bitErrorRate": row["bit_error_rate"],
        "elapsedMs": row["elapsed_ms"],
        "manifestPath": row["manifest_path"],
        "outputDir": row["output_dir"],
        "error": row["error"],
        "summary": loads_json(row["summary_json"]),
        "updatedAt": row["updated_at"],
    }
