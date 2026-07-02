from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.experiment_schema import STAGE_EVENT_SCHEMA


JsonDict = dict[str, Any]
INTERMEDIATE_ARTIFACT_DIR = "_intermediates"


def write_json(path: Path, payload: JsonDict | list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")


def write_jsonl(path: Path, records: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    records: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def artifact_paths(run_root: Path) -> dict[str, Path]:
    return {
        "runPlan": run_root / "run_plan.json",
        "runStatus": run_root / "run_status.json",
        "sampleManifest": run_root / "sample_manifest.jsonl",
        "cellManifest": run_root / "cell_manifest.jsonl",
        "cellManifestLatest": run_root / "cell_manifest_latest.jsonl",
        "cellSummaryLatest": run_root / "cell_summary_latest.json",
        "imageQuality": run_root / "image_quality.jsonl",
        "imageWatermarkEmbed": run_root / "image_watermark_embed.jsonl",
        "imageAttack": run_root / "image_attack.jsonl",
        "imageDetection": run_root / "image_detection.jsonl",
        "imageDetectionLatest": run_root / "image_detection_latest.jsonl",
        "runtimeProfile": run_root / "runtime_profile.jsonl",
        "stageEvents": run_root / "stage_events.jsonl",
        "runSummary": run_root / "run_summary.json",
    }


def stage_event(paths: dict[str, Path], run_id: str, stage: str, status: str, **payload: Any) -> None:
    append_jsonl(
        paths["stageEvents"],
        STAGE_EVENT_SCHEMA.apply(
            {
                "runId": run_id,
                "stage": stage,
                "status": status,
                "timestamp": utc_timestamp(),
                **payload,
            }
        ),
    )


def progress(completed_cells: int, total_cells: int) -> int:
    if total_cells <= 0:
        return 0
    return int(round((completed_cells / total_cells) * 100))


def write_run_status(
    paths: dict[str, Path],
    *,
    run_id: str,
    status: str,
    completed_cells: int,
    expected_cells: int,
    error: str | None = None,
) -> None:
    write_json(
        paths["runStatus"],
        {
            "runId": run_id,
            "status": status,
            "completedCells": completed_cells,
            "expectedCells": expected_cells,
            "progress": progress(completed_cells, expected_cells),
            "completedProgress": progress(completed_cells, expected_cells),
            "progressKind": "completedCells",
            "error": error,
            "updatedAt": utc_timestamp(),
        },
    )


def cell_attempt_counts(cell_manifest_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in read_jsonl(cell_manifest_path):
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str):
            counts[cell_key] = counts.get(cell_key, 0) + 1
    return counts


def latest_cell_rows(cell_manifest_path: Path) -> list[JsonDict]:
    latest: dict[str, JsonDict] = {}
    attempt_counts: dict[str, int] = {}
    for record in read_jsonl(cell_manifest_path):
        cell_key = record.get("cellKey")
        if isinstance(cell_key, str):
            attempt_counts[cell_key] = attempt_counts.get(cell_key, 0) + 1
            enriched = dict(record)
            enriched.setdefault("attemptIndex", attempt_counts[cell_key])
            enriched.setdefault("supersedesPreviousAttempt", attempt_counts[cell_key] > 1)
            latest[cell_key] = enriched
    return list(latest.values())


def latest_cell_row_map(cell_manifest_path: Path) -> dict[str, JsonDict]:
    return {
        str(record["cellKey"]): record
        for record in latest_cell_rows(cell_manifest_path)
        if isinstance(record.get("cellKey"), str)
    }


def _json_record_has_intermediate_artifact(record: JsonDict) -> bool:
    for key in ("inputPath", "sampleId", "referencePath", "targetPath"):
        value = record.get(key)
        if isinstance(value, str) and INTERMEDIATE_ARTIFACT_DIR in Path(value).parts:
            return True
    return False


def _read_json_array(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _latest_image_detection_rows(latest_cells: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for cell in latest_cells:
        manifest_path = cell.get("manifestPath")
        if not isinstance(manifest_path, str):
            continue
        for record in _read_json_array(Path(manifest_path)):
            if not _json_record_has_intermediate_artifact(record):
                rows.append(record)
    return rows


def write_latest_cell_artifacts(paths: dict[str, Path], *, run_id: str, expected_cells: int) -> None:
    latest_cells = latest_cell_rows(paths["cellManifest"])
    status_counts: dict[str, int] = {}
    for cell in latest_cells:
        status = str(cell.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    attempted_cells = len(latest_cells)
    succeeded_cells = status_counts.get("succeeded", 0)
    failed_cells = status_counts.get("failed", 0)

    write_jsonl(paths["cellManifestLatest"], latest_cells)
    write_jsonl(paths["imageDetectionLatest"], _latest_image_detection_rows(latest_cells))
    write_json(
        paths["cellSummaryLatest"],
        {
            "runId": run_id,
            "cellCount": attempted_cells,
            "attemptedCells": attempted_cells,
            "succeededCells": succeeded_cells,
            "failedCells": failed_cells,
            "expectedCells": expected_cells,
            "progress": progress(attempted_cells, expected_cells),
            "completedProgress": progress(attempted_cells, expected_cells),
            "progressKind": "completedCells",
            "attemptedProgress": progress(attempted_cells, expected_cells),
            "succeededProgress": progress(succeeded_cells, expected_cells),
            "statusCounts": status_counts,
            "updatedAt": utc_timestamp(),
        },
    )
