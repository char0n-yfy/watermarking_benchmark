from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from app.core.storage import safe_segment
from app.services.experiment_stages import copy_canonical_samples
from app.services.local_artifacts import (
    PHASE_LABELS,
    PHASE_ORDER,
    RunStateWriter,
    append_jsonl,
    artifact_paths,
    progress as _progress,
    read_json_object,
    read_jsonl,
    utc_timestamp,
    write_json,
    write_jsonl,
    write_run_status,
)
from app.services.local_executor import LocalRunRequest, run_local_experiment as run_single_local_experiment
from app.services.local_plan import estimate_selection, normalize_selection
from app.services.resources import get_dataset_by_id, iter_image_paths


JsonDict = dict[str, Any]
StopIntentCallback = Any
RunStateCallback = Any
SHARD_STATE_POLL_SECONDS = 1.0
SHARDS_DIR_NAME = "shards"
PARENT_PAUSE_REQUEST_NAME = "pause_requested"
PARENT_CANCEL_REQUEST_NAME = "cancel_requested"


def run_sharded_experiment(
    request: LocalRunRequest,
    on_state: RunStateCallback | None = None,
    should_cancel: StopIntentCallback | None = None,
) -> JsonDict:
    devices = _resolve_devices(request.device)
    if len(devices) <= 1:
        return run_single_local_experiment(request, on_state=on_state, should_cancel=should_cancel)

    selection = normalize_selection(request.selection, request.resources_root)
    run_root = request.runs_root / safe_segment(request.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    parent_paths = artifact_paths(run_root)
    materialized_root = _parent_materialized_root(run_root)
    logical_estimate = estimate_selection(selection, request.resources_root)
    previous_summary = read_json_object(parent_paths["runSummary"])

    parent_samples = _materialize_parent_canonical(
        request=request,
        selection=selection,
        run_root=run_root,
        materialized_root=materialized_root,
        paths=parent_paths,
    )
    shard_specs = _build_shard_specs(
        request=request,
        selection=selection,
        devices=devices,
        samples_by_dataset=parent_samples,
    )
    if not shard_specs:
        raise ValueError("No shard has input samples")

    total_expanded_cells = sum(int(shard["expectedCells"]) for shard in shard_specs)
    started = time.perf_counter()
    state_writer = RunStateWriter(
        paths=parent_paths,
        run_id=request.run_id,
        run_root=run_root,
        selection=selection,
        expected_result_units=total_expanded_cells,
        materialized_root=materialized_root,
        on_state=on_state,
    )
    _write_parent_plan(
        request=request,
        selection=selection,
        paths=parent_paths,
        logical_cells=int(logical_estimate["cellCount"]),
        total_expanded_cells=total_expanded_cells,
        shard_specs=shard_specs,
    )
    write_run_status(
        parent_paths,
        run_id=request.run_id,
        status="running",
        result_units=0,
        expected_result_units=total_expanded_cells,
    )
    _write_parent_state(
        state_writer,
        status="running",
        shard_specs=shard_specs,
        total_expanded_cells=total_expanded_cells,
        logical_cells=int(logical_estimate["cellCount"]),
    )

    pause_path = run_root / PARENT_PAUSE_REQUEST_NAME
    cancel_path = run_root / PARENT_CANCEL_REQUEST_NAME
    pause_path.unlink(missing_ok=True)
    cancel_path.unlink(missing_ok=True)
    cache_root = Path(os.getenv("WM_BENCH_CACHE_ROOT") or str(materialized_root)).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    worker_payloads = [
        {
            "parentRunId": request.run_id,
            "shardId": shard["id"],
            "device": shard["device"],
            "selection": shard["selection"],
            "resourcesRoot": str(request.resources_root),
            "runsRoot": str(run_root / SHARDS_DIR_NAME),
            "message": request.message,
            "resume": request.resume,
            "pausePath": str(pause_path),
            "cancelPath": str(cancel_path),
            "cacheRoot": str(cache_root),
        }
        for shard in shard_specs
    ]

    summaries: list[JsonDict] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=len(worker_payloads)) as executor:
        future_map = {executor.submit(_run_shard_worker, payload): payload for payload in worker_payloads}
        pending = set(future_map)
        while pending:
            stop_intent = _stop_status_from_callback(should_cancel)
            if stop_intent == "cancel":
                cancel_path.write_text(utc_timestamp(), encoding="utf-8")
            elif stop_intent == "pause":
                pause_path.write_text(utc_timestamp(), encoding="utf-8")

            done = [future for future in pending if future.done()]
            for future in done:
                pending.remove(future)
                payload = future_map[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    errors.append(f"{payload['shardId']}: {type(exc).__name__}: {exc}")
                    _write_failed_shard_state(run_root, payload, str(errors[-1]))
            _write_parent_state(
                state_writer,
                status="running" if pending else "summarizing",
                shard_specs=shard_specs,
                total_expanded_cells=total_expanded_cells,
                logical_cells=int(logical_estimate["cellCount"]),
            )
            if pending:
                time.sleep(SHARD_STATE_POLL_SECONDS)

    summary = _aggregate_shard_outputs(
        request=request,
        selection=selection,
        parent_paths=parent_paths,
        run_root=run_root,
        materialized_root=materialized_root,
        shard_specs=shard_specs,
        total_expanded_cells=total_expanded_cells,
        logical_cells=int(logical_estimate["cellCount"]),
        previous_summary=previous_summary,
        invocation_elapsed_ms=(time.perf_counter() - started) * 1000,
        errors=errors,
    )
    _write_parent_state(
        state_writer,
        status=str(summary["status"]),
        shard_specs=shard_specs,
        total_expanded_cells=total_expanded_cells,
        logical_cells=int(logical_estimate["cellCount"]),
        summary_path=str(parent_paths["runSummary"]),
    )
    write_run_status(
        parent_paths,
        run_id=request.run_id,
        status=str(summary["status"]),
        result_units=int(summary["resultUnitCount"]),
        expected_result_units=total_expanded_cells,
        error="; ".join(errors) or None,
    )
    return summary


def _run_shard_worker(payload: JsonDict) -> JsonDict:
    os.environ["WM_BENCH_CACHE_ROOT"] = str(payload["cacheRoot"])
    pause_path = Path(str(payload["pausePath"]))
    cancel_path = Path(str(payload["cancelPath"]))

    def should_cancel() -> str | None:
        if cancel_path.exists():
            return "cancel"
        if pause_path.exists():
            return "pause"
        return None

    request = LocalRunRequest(
        run_id=str(payload["shardId"]),
        selection=dict(payload["selection"]),
        resources_root=Path(str(payload["resourcesRoot"])),
        runs_root=Path(str(payload["runsRoot"])),
        device=str(payload["device"]),
        message=str(payload["message"]),
        resume=bool(payload["resume"]),
    )
    summary = run_single_local_experiment(request, should_cancel=should_cancel)
    shard_root = Path(str(payload["runsRoot"])) / safe_segment(str(payload["shardId"]))
    state = read_json_object(shard_root / "run_state.json")
    state.update(
        {
            "parentRunId": payload["parentRunId"],
            "shardId": payload["shardId"],
            "device": payload["device"],
            "status": summary.get("status", state.get("status", "succeeded")),
            "updatedAt": utc_timestamp(),
        }
    )
    write_json(shard_root / "shard_state.json", state)
    return {**summary, "shardId": payload["shardId"], "device": payload["device"]}


def _resolve_devices(request_device: str) -> list[str]:
    configured = os.getenv("WM_BENCH_DEVICES")
    if configured:
        devices = [item.strip() for item in configured.split(",") if item.strip()]
        return devices or [request_device]
    if not str(request_device).startswith("cuda"):
        return [request_device]
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() and visible.strip() != "-1":
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if ids:
            return [f"cuda:{index}" for index in range(len(ids))]
    count = _nvidia_smi_device_count()
    if count <= 0:
        count = _torch_cuda_device_count()
    if count <= 1:
        return [request_device]
    return [f"cuda:{index}" for index in range(count)]


def _nvidia_smi_device_count() -> int:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return 0
    if completed.returncode != 0:
        return 0
    return sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("GPU "))


def _torch_cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _parent_materialized_root(run_root: Path) -> Path:
    configured = os.getenv("WM_BENCH_CACHE_ROOT")
    root = Path(configured).expanduser() if configured else run_root / "materialized"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _materialize_parent_canonical(
    *,
    request: LocalRunRequest,
    selection: JsonDict,
    run_root: Path,
    materialized_root: Path,
    paths: dict[str, Path],
) -> dict[str, list[Path]]:
    samples_by_dataset: dict[str, list[Path]] = {}
    existing_sample_keys = {
        (str(record.get("datasetId")), str(record.get("sampleId")))
        for record in read_jsonl(paths["sampleManifest"])
        if record.get("datasetId") is not None and record.get("sampleId") is not None
    }
    for dataset_id in selection["datasetIds"]:
        dataset = get_dataset_by_id(request.resources_root, str(dataset_id))
        source_paths = iter_image_paths(dataset.path)[: int(selection["maxSamples"])]
        samples_by_dataset[str(dataset_id)] = source_paths
        canonical_dir = materialized_root / "canonical_parent" / safe_segment(str(dataset_id))
        staged = copy_canonical_samples(
            dataset.path,
            canonical_dir,
            int(selection["maxSamples"]),
            sample_paths=source_paths,
        )
        manifest = run_root / safe_segment(str(dataset_id)) / "canonical" / "manifest.json"
        write_json(
            manifest,
            {
                "runId": request.run_id,
                "datasetId": str(dataset_id),
                "sampleCount": len(staged),
                "materializedDir": str(canonical_dir),
                "samples": [str(sample.path) for sample in staged],
            },
        )
        for sample in staged:
            try:
                sample_id = sample.path.relative_to(canonical_dir).as_posix()
            except ValueError:
                sample_id = sample.path.name
            sample_key = (str(dataset_id), sample_id)
            if sample_key in existing_sample_keys:
                continue
            metadata = sample.metadata
            append_jsonl(
                paths["sampleManifest"],
                {
                    "runId": request.run_id,
                    "datasetId": str(dataset_id),
                    "sampleId": sample_id,
                    "sourcePath": str(sample.source_path),
                    "width": (metadata.get("originalSize") or [None, None])[0],
                    "height": (metadata.get("originalSize") or [None, None])[1],
                    "originalSize": metadata.get("originalSize"),
                    "canonicalSize": metadata.get("canonicalSize"),
                    "canonicalWidth": (metadata.get("canonicalSize") or [None, None])[0],
                    "canonicalHeight": (metadata.get("canonicalSize") or [None, None])[1],
                    "preprocessPolicy": metadata.get("preprocessPolicy"),
                    "cropPolicy": metadata.get("cropPolicy"),
                    "resizedContentSize": metadata.get("resizedContentSize"),
                    "cropBox": metadata.get("cropBox"),
                    "cropMargins": metadata.get("cropMargins"),
                    "padding": metadata.get("padding"),
                    "scale": metadata.get("scale"),
                    "paddingColor": metadata.get("paddingColor"),
                    "timestamp": utc_timestamp(),
                },
            )
            existing_sample_keys.add(sample_key)
    return samples_by_dataset


def _build_shard_specs(
    *,
    request: LocalRunRequest,
    selection: JsonDict,
    devices: list[str],
    samples_by_dataset: dict[str, list[Path]],
) -> list[JsonDict]:
    shard_count = max(1, len(devices))
    sample_relatives: list[dict[str, list[str]]] = [dict() for _ in range(shard_count)]
    for dataset_id, paths in samples_by_dataset.items():
        dataset = get_dataset_by_id(request.resources_root, dataset_id)
        for index, sample_path in enumerate(paths):
            shard_index = index % shard_count
            try:
                relative = sample_path.relative_to(dataset.path).as_posix()
            except ValueError:
                relative = sample_path.name
            sample_relatives[shard_index].setdefault(dataset_id, []).append(relative)

    specs: list[JsonDict] = []
    for index, device in enumerate(devices):
        overrides = {dataset_id: values for dataset_id, values in sample_relatives[index].items() if values}
        if not overrides:
            continue
        shard_selection = dict(selection)
        shard_selection["datasetIds"] = [dataset_id for dataset_id in selection["datasetIds"] if dataset_id in overrides]
        shard_selection["maxSamples"] = max(len(values) for values in overrides.values())
        shard_selection["_sampleRelativesByDataset"] = overrides
        shard_id = f"shard_{index:03d}_{safe_segment(device)}"
        shard_selection["_shard"] = {"id": shard_id, "index": index, "device": device}
        estimate = estimate_selection(shard_selection, request.resources_root)
        specs.append(
            {
                "id": shard_id,
                "index": index,
                "device": device,
                "selection": estimate["selection"],
                "sampleCount": estimate["sampleCount"],
                "expectedCells": estimate["cellCount"],
                "statePath": str(request.runs_root / safe_segment(request.run_id) / SHARDS_DIR_NAME / shard_id / "run_state.json"),
            }
        )
    return specs


def _write_parent_plan(
    *,
    request: LocalRunRequest,
    selection: JsonDict,
    paths: dict[str, Path],
    logical_cells: int,
    total_expanded_cells: int,
    shard_specs: list[JsonDict],
) -> None:
    write_json(
        paths["runPlan"],
        {
            "runId": request.run_id,
            "selection": selection,
            "executionMode": "gpu_sharded",
            "logicalCells": logical_cells,
            "expectedCells": total_expanded_cells,
            "totalExpandedCells": total_expanded_cells,
            "artifactFiles": {key: str(path) for key, path in paths.items()},
            "shards": [
                {
                    "id": shard["id"],
                    "index": shard["index"],
                    "device": shard["device"],
                    "sampleCount": shard["sampleCount"],
                    "expectedCells": shard["expectedCells"],
                    "statePath": shard["statePath"],
                }
                for shard in shard_specs
            ],
            "createdAt": utc_timestamp(),
        },
    )


def _write_parent_state(
    state_writer: RunStateWriter,
    *,
    status: str,
    shard_specs: list[JsonDict],
    total_expanded_cells: int,
    logical_cells: int,
    summary_path: str | None = None,
) -> None:
    shard_states = [_read_shard_progress(shard) for shard in shard_specs]
    phases = _aggregate_phase_states(shard_states, status=status, summary_path=summary_path)
    extra = {
        "executionMode": "gpu_sharded",
        "shards": shard_states,
        "phaseTotals": {str(phase["key"]): dict(phase) for phase in phases},
        "logicalCells": logical_cells,
        "totalExpandedCells": total_expanded_cells,
        "expectedResultUnits": total_expanded_cells,
    }
    state_writer.replace_phase_states(phases, status=status, extra_state=extra)


def _read_shard_progress(shard: JsonDict) -> JsonDict:
    state_path = Path(str(shard["statePath"]))
    state = read_json_object(state_path)
    if not state:
        state = read_json_object(state_path.parent / "shard_state.json")
    if not state:
        state = {
            "runId": shard["id"],
            "status": "queued",
            "currentPhase": "canonical",
            "overallProgress": 0,
            "progress": 0,
            "phases": [],
        }
    current_phase = str(state.get("currentPhase") or "canonical")
    current_phase_doc = next(
        (
            phase
            for phase in state.get("phases", [])
            if isinstance(phase, dict) and str(phase.get("key")) == current_phase
        ),
        {},
    )
    return {
        "id": shard["id"],
        "index": shard["index"],
        "device": shard["device"],
        "status": state.get("status") or "queued",
        "currentPhase": current_phase,
        "progress": state.get("progress") or state.get("overallProgress") or 0,
        "sampleCount": shard.get("sampleCount", 0),
        "expectedCells": shard.get("expectedCells", 0),
        "statePath": str(state_path),
        "logPath": str(state_path.parent / "worker.log"),
        "currentItem": current_phase_doc.get("currentItem") if isinstance(current_phase_doc, dict) else {},
        "imageProgress": _phase_image_progress(current_phase_doc, shard),
        "cellProgress": _phase_cell_progress(current_phase_doc, shard),
        "phases": state.get("phases", []),
        "updatedAt": state.get("updatedAt"),
    }


def _phase_image_progress(phase: JsonDict, shard: JsonDict) -> JsonDict:
    item = phase.get("currentItem") if isinstance(phase, dict) else {}
    phase_key = str(phase.get("key") or "") if isinstance(phase, dict) else ""
    current = 0
    total = _safe_int(shard.get("sampleCount"))
    if isinstance(item, dict):
        if item.get("processedImages") is not None:
            current = _safe_int(item.get("processedImages"))
        elif item.get("sampleCount") is not None:
            current = _safe_int(item.get("sampleCount"))
        else:
            paired_current = max(_safe_int(item.get("positiveImages")), _safe_int(item.get("negativeImages")))
            if paired_current:
                current = paired_current
            elif item.get("pairCount") is not None:
                current = _safe_int(item.get("pairCount"))
                if phase_key == "quality":
                    total = max(total * 2, current)
    if not current and str(phase.get("status") or "") == "succeeded":
        current = total
    current_int = _safe_int(current)
    total_int = max(total, current_int)
    if total_int:
        current_int = min(current_int, total_int)
    return {"current": current_int, "total": total_int}


def _phase_cell_progress(phase: JsonDict, shard: JsonDict) -> JsonDict:
    counters = phase.get("counters") if isinstance(phase, dict) else {}
    phase_key = str(phase.get("key") or "")
    if phase_key == "canonical":
        current = 0
        if isinstance(counters, dict):
            current = _safe_int(counters.get("imagesDone") or phase.get("current"))
        total = max(_safe_int(shard.get("sampleCount")), current)
        if not current and str(phase.get("status") or "") == "succeeded":
            current = total
        return {"current": min(current, total) if total else current, "total": total}

    current = 0
    if isinstance(counters, dict):
        current = _safe_int(
            counters.get("phaseCellsDone")
            or counters.get("cellsDone")
            or counters.get("resultUnitsDone")
            or 0
        )
    if not current and phase_key == "summary":
        current = _safe_int(phase.get("current"))
    total = _safe_int(counters.get("phaseCellsTotal") if isinstance(counters, dict) else 0) or _safe_int(shard.get("expectedCells"))
    if not current and str(phase.get("status") or "") == "succeeded":
        current = total
    return {"current": min(current, total) if total else current, "total": total}


def _aggregate_phase_cell_progress(key: str, shard_states: list[JsonDict]) -> JsonDict:
    current = 0
    total = 0
    for shard_state in shard_states:
        phase_doc = next(
            (
                phase
                for phase in shard_state.get("phases", [])
                if isinstance(phase, dict) and str(phase.get("key")) == key
            ),
            {},
        )
        if not phase_doc:
            phase_doc = {"key": key, "status": "pending", "counters": {}}
        progress_doc = _phase_cell_progress(phase_doc, shard_state)
        current += _safe_int(progress_doc.get("current"))
        total += _safe_int(progress_doc.get("total"))
    return {"current": min(current, total) if total else current, "total": total, "percent": _progress(current, total)}


def _aggregate_phase_states(shard_states: list[JsonDict], *, status: str, summary_path: str | None) -> list[JsonDict]:
    now = utc_timestamp()
    phases: list[JsonDict] = []
    for key in PHASE_ORDER:
        shard_phase_docs = []
        for shard_state in shard_states:
            phase_doc = next(
                (
                    phase
                    for phase in shard_state.get("phases", [])
                    if isinstance(phase, dict) and str(phase.get("key")) == key
                ),
                None,
            )
            if isinstance(phase_doc, dict):
                shard_phase_docs.append(phase_doc)
        total = sum(_safe_int(phase.get("total")) for phase in shard_phase_docs)
        current = sum(_safe_int(phase.get("current")) for phase in shard_phase_docs)
        phase_status = _aggregate_status(key, shard_states, shard_phase_docs, status)
        counters = _sum_counters(shard_phase_docs)
        cell_progress = _aggregate_phase_cell_progress(key, shard_states)
        phase = {
            "key": key,
            "label": PHASE_LABELS[key],
            "status": phase_status,
            "current": current,
            "total": total,
            "percent": _progress(current, total),
            "currentItem": _latest_current_item(shard_phase_docs),
            "counters": counters,
            "cellProgress": cell_progress,
            "artifactRefs": {},
            "updatedAt": now,
        }
        if key == "summary" and summary_path:
            phase["artifactRefs"] = {"summary": summary_path}
        phases.append(phase)
    return phases


def _aggregate_status(key: str, shard_states: list[JsonDict], shard_phases: list[JsonDict], parent_status: str) -> str:
    if parent_status in {"paused", "cancelled", "failed", "succeeded", "partially_failed"} and key == "summary":
        return "succeeded" if parent_status in {"succeeded", "partially_failed", "paused"} else parent_status
    if any(str(shard.get("currentPhase")) == key and str(shard.get("status")) == "running" for shard in shard_states):
        return "running"
    statuses = [str(phase.get("status") or "pending") for phase in shard_phases]
    if statuses and all(item == "succeeded" for item in statuses):
        return "succeeded"
    if any(item == "running" for item in statuses):
        return "running"
    if any(item in {"failed", "partially_failed"} for item in statuses):
        return "failed"
    stopped_status = next((item for item in statuses if item in {"paused", "cancelled"}), None)
    if stopped_status is not None:
        return stopped_status
    return "pending"


def _sum_counters(phases: list[JsonDict]) -> JsonDict:
    totals: JsonDict = {}
    for phase in phases:
        counters = phase.get("counters")
        if not isinstance(counters, dict):
            continue
        for key, value in counters.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return totals


def _latest_current_item(phases: list[JsonDict]) -> JsonDict:
    for phase in reversed(phases):
        current_item = phase.get("currentItem")
        if isinstance(current_item, dict) and current_item:
            return current_item
    return {}


def _aggregate_shard_outputs(
    *,
    request: LocalRunRequest,
    selection: JsonDict,
    parent_paths: dict[str, Path],
    run_root: Path,
    materialized_root: Path,
    shard_specs: list[JsonDict],
    total_expanded_cells: int,
    logical_cells: int,
    previous_summary: JsonDict,
    invocation_elapsed_ms: float,
    errors: list[str],
) -> JsonDict:
    merged_result_units: list[JsonDict] = []
    failed = 0
    for key in ("resultUnits", "imageDetection", "imageQuality", "imageAttack", "imageWatermarkEmbed", "runtimeProfile"):
        parent_paths[key].parent.mkdir(parents=True, exist_ok=True)
        parent_paths[key].write_text("", encoding="utf-8")

    shard_summaries: list[JsonDict] = []
    for shard in shard_specs:
        shard_context = {**shard, "parentRunId": request.run_id}
        shard_root = run_root / SHARDS_DIR_NAME / str(shard["id"])
        shard_summary = read_json_object(shard_root / "run_summary.json")
        shard_progress = _read_shard_progress(shard)
        shard_summaries.append(
            {
                "id": shard["id"],
                "device": shard["device"],
                "status": shard_summary.get("status") or shard_progress.get("status"),
                "currentPhase": shard_progress.get("currentPhase"),
                "cellProgress": shard_progress.get("cellProgress"),
                "phaseCellProgress": {
                    key: _aggregate_phase_cell_progress(key, [shard_progress])
                    for key in PHASE_ORDER
                },
                "sampleCount": shard.get("sampleCount", 0),
                "expectedCells": shard.get("expectedCells", 0),
                "resultUnitCount": shard_summary.get("resultUnitCount", 0),
                "artifactRoot": str(shard_root),
                "statePath": str(shard_root / "run_state.json"),
            }
        )
        for artifact_key, filename in (
            ("imageDetection", "image_detection.jsonl"),
            ("imageQuality", "image_quality.jsonl"),
            ("imageAttack", "image_attack.jsonl"),
            ("imageWatermarkEmbed", "image_watermark_embed.jsonl"),
            ("runtimeProfile", "runtime_profile.jsonl"),
        ):
            records = [
                _with_shard_record_fields(record, shard_context)
                for record in read_jsonl(shard_root / filename)
            ]
            for record in records:
                append_jsonl(parent_paths[artifact_key], record)
        for unit in read_jsonl(shard_root / "result_units.jsonl"):
            merged = _with_shard_result_unit(unit, shard_context)
            if merged.get("status") != "succeeded":
                failed += 1
            merged_result_units.append(merged)

    write_jsonl(parent_paths["resultUnits"], merged_result_units)
    status = _parent_status(shard_summaries, merged_result_units, total_expanded_cells, errors)
    shard_progress_states = [_read_shard_progress(shard) for shard in shard_specs]
    phase_cell_progress = {
        key: _aggregate_phase_cell_progress(key, shard_progress_states)
        for key in PHASE_ORDER
    }
    elapsed_ms = max(
        _safe_float(previous_summary.get("elapsedMs")) + invocation_elapsed_ms,
        invocation_elapsed_ms,
    )
    summary = {
        "runId": request.run_id,
        "status": status,
        "selection": selection,
        "executionMode": "gpu_sharded",
        "artifactRoot": str(run_root),
        "materializedRoot": str(materialized_root),
        "artifactFiles": {key: str(path) for key, path in parent_paths.items()},
        "artifactTreePath": str(parent_paths["artifactTree"]),
        "phaseStatePath": str(parent_paths["phaseState"]),
        "runStatePath": str(parent_paths["runState"]),
        "logicalCells": logical_cells,
        "totalExpandedCells": total_expanded_cells,
        "resultUnitCount": len(merged_result_units),
        "failedResultUnits": failed,
        "skippedResultUnits": max(0, total_expanded_cells - len(merged_result_units)),
        "completedCells": len(merged_result_units),
        "failedCells": failed,
        "remainingCells": max(0, total_expanded_cells - len(merged_result_units)),
        "phaseCellProgress": phase_cell_progress,
        "completedCellsByPhase": {
            key: _safe_int(progress_doc.get("current"))
            for key, progress_doc in phase_cell_progress.items()
        },
        "remainingCellsByPhase": {
            key: max(0, _safe_int(progress_doc.get("total")) - _safe_int(progress_doc.get("current")))
            for key, progress_doc in phase_cell_progress.items()
        },
        "progress": _progress(len(merged_result_units), total_expanded_cells),
        "completedProgress": _progress(len(merged_result_units), total_expanded_cells),
        "succeededProgress": _progress(len(merged_result_units) - failed, total_expanded_cells),
        "progressKind": "phaseOperations",
        "elapsedMs": elapsed_ms,
        "invocationElapsedMs": invocation_elapsed_ms,
        "shards": shard_summaries,
        "errors": errors,
        "resultUnits": merged_result_units,
    }
    write_json(
        parent_paths["artifactTree"],
        {
            "runId": request.run_id,
            "executionMode": "gpu_sharded",
            "shards": shard_summaries,
            "datasets": {},
            "updatedAt": utc_timestamp(),
        },
    )
    write_json(parent_paths["runSummary"], summary)
    return summary


def _with_shard_record_fields(record: JsonDict, shard: JsonDict) -> JsonDict:
    updated = dict(record)
    updated["parentRunId"] = updated.get("runId")
    updated["runId"] = str(shard.get("parentRunId") or updated.get("runId") or "")
    updated["shardId"] = shard["id"]
    updated["device"] = shard["device"]
    logical = updated.get("cellKey") or updated.get("cell_key")
    if logical:
        updated["logicalCellKey"] = str(logical)
        unique = _shard_cell_key(str(shard["id"]), str(logical))
        if "cellKey" in updated:
            updated["cellKey"] = unique
        if "cell_key" in updated:
            updated["cell_key"] = unique
    return updated


def _with_shard_result_unit(unit: JsonDict, shard: JsonDict) -> JsonDict:
    updated = _with_shard_record_fields(unit, shard)
    logical = str(unit.get("resultUnitKey") or unit.get("cellKey") or "")
    unique = _shard_cell_key(str(shard["id"]), logical)
    updated["logicalCellKey"] = logical
    updated["cellKey"] = unique
    updated["resultUnitKey"] = unique
    updated["shardId"] = shard["id"]
    updated["device"] = shard["device"]
    sample_overrides = shard.get("selection", {}).get("_sampleRelativesByDataset")
    dataset_id = str(unit.get("datasetId") or "")
    if isinstance(sample_overrides, dict) and isinstance(sample_overrides.get(dataset_id), list):
        updated["sampleIds"] = list(sample_overrides[dataset_id])
    return updated


def _shard_cell_key(shard_id: str, logical_cell_key: str) -> str:
    return safe_segment(f"{shard_id}__{logical_cell_key}")


def _parent_status(shard_summaries: list[JsonDict], result_units: list[JsonDict], expected: int, errors: list[str]) -> str:
    statuses = {str(summary.get("status") or "") for summary in shard_summaries}
    if "cancelled" in statuses:
        return "cancelled"
    if "paused" in statuses:
        return "paused"
    if errors:
        return "partially_failed" if result_units else "failed"
    failed = sum(1 for unit in result_units if unit.get("status") != "succeeded")
    if expected > 0 and len(result_units) < expected:
        return "partially_failed" if result_units else "failed"
    if failed == 0:
        return "succeeded"
    return "partially_failed" if failed < len(result_units) else "failed"


def _write_failed_shard_state(run_root: Path, payload: JsonDict, error: str) -> None:
    shard_root = run_root / SHARDS_DIR_NAME / safe_segment(str(payload["shardId"]))
    shard_root.mkdir(parents=True, exist_ok=True)
    write_json(
        shard_root / "shard_state.json",
        {
            "runId": payload["shardId"],
            "parentRunId": payload["parentRunId"],
            "shardId": payload["shardId"],
            "device": payload["device"],
            "status": "failed",
            "currentPhase": "summary",
            "overallProgress": 0,
            "progress": 0,
            "error": error,
            "updatedAt": utc_timestamp(),
        },
    )


def _stop_status_from_callback(callback: StopIntentCallback | None) -> str | None:
    if callback is None:
        return None
    value = callback()
    if value == "cancel":
        return "cancel"
    if value == "pause" or value is True:
        return "pause"
    return None


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
