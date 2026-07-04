from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


JsonDict = dict[str, Any]
RunStateHook = Callable[[JsonDict], None]
PHASE_ORDER = [
    "canonical",
    "watermark_embed",
    "attack",
    "watermark_extract",
    "quality",
    "summary",
]
PHASE_LABELS = {
    "canonical": "采样 canonical 数据集",
    "watermark_embed": "嵌入水印",
    "attack": "攻击",
    "watermark_extract": "提取",
    "quality": "评估质量",
    "summary": "汇总",
}
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
        "runState": run_root / "run_state.json",
        "phaseState": run_root / "phase_state.json",
        "artifactTree": run_root / "artifact_tree.json",
        "sampleManifest": run_root / "sample_manifest.jsonl",
        "resultUnits": run_root / "result_units.jsonl",
        "imageQuality": run_root / "image_quality.jsonl",
        "imageWatermarkEmbed": run_root / "image_watermark_embed.jsonl",
        "imageAttack": run_root / "image_attack.jsonl",
        "imageDetection": run_root / "image_detection.jsonl",
        "runtimeProfile": run_root / "runtime_profile.jsonl",
        "runSummary": run_root / "run_summary.json",
    }


def progress(current: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round((current / total) * 100))


def _phase_percent(current: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round((max(0, min(current, total)) / total) * 100))


def _result_unit_key(record: JsonDict) -> str | None:
    for key in ("resultUnitKey", "cellKey"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def compact_result_units(records: list[JsonDict]) -> list[JsonDict]:
    latest_by_key: dict[str, JsonDict] = {}
    key_order: list[str] = []
    unkeyed_records: list[JsonDict] = []
    for record in records:
        key = _result_unit_key(record)
        if key is None:
            unkeyed_records.append(dict(record))
            continue
        if key not in latest_by_key:
            key_order.append(key)
        latest_by_key[key] = dict(record)
    return [latest_by_key[key] for key in key_order] + unkeyed_records


def compact_result_units_file(result_units_path: Path) -> list[JsonDict]:
    compacted = compact_result_units(read_jsonl(result_units_path))
    write_jsonl(result_units_path, compacted)
    return compacted


def default_phase_states() -> list[JsonDict]:
    return [
        {
            "key": key,
            "label": PHASE_LABELS[key],
            "status": "pending",
            "current": 0,
            "total": 0,
            "percent": 0,
            "currentItem": {},
            "counters": {},
            "artifactRefs": {},
        }
        for key in PHASE_ORDER
    ]


def read_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class RunStateWriter:
    """Writes the run's authoritative phase state.

    JSONL files remain useful for debugging and scoring, but the frontend should
    use this state instead of inferring progress from logs or events.
    """

    def __init__(
        self,
        *,
        paths: dict[str, Path],
        run_id: str,
        run_root: Path,
        selection: JsonDict,
        expected_result_units: int,
        materialized_root: Path,
        on_state: RunStateHook | None = None,
    ) -> None:
        self.paths = paths
        self.run_id = run_id
        self.run_root = run_root
        self.selection = selection
        self.expected_result_units = expected_result_units
        self.materialized_root = materialized_root
        self.on_state = on_state
        existing = read_json_object(paths["phaseState"])
        phase_values = existing.get("phases") if isinstance(existing.get("phases"), list) else None
        self.phases: dict[str, JsonDict] = {
            str(phase["key"]): dict(phase)
            for phase in (phase_values or default_phase_states())
            if isinstance(phase, dict) and isinstance(phase.get("key"), str)
        }
        for phase in default_phase_states():
            self.phases.setdefault(str(phase["key"]), phase)
        tree = read_json_object(paths["artifactTree"])
        self.artifact_tree: JsonDict = tree if tree else {"runId": run_id, "datasets": {}, "updatedAt": utc_timestamp()}
        self.status = "running"
        self.extra_state: JsonDict = {}

    def write_initial(self, *, status: str = "running") -> None:
        self.status = status
        self._write()

    def phase_start(
        self,
        key: str,
        *,
        total: int | None = None,
        current_item: JsonDict | None = None,
        counters: JsonDict | None = None,
        artifact_refs: JsonDict | None = None,
    ) -> None:
        phase = self._phase(key)
        now = utc_timestamp()
        phase["status"] = "running"
        phase.setdefault("startedAt", now)
        phase["updatedAt"] = now
        if total is not None:
            phase["total"] = max(0, int(total))
        if current_item is not None:
            phase["currentItem"] = current_item
        if counters:
            phase["counters"] = {**dict(phase.get("counters") or {}), **counters}
        if artifact_refs:
            phase["artifactRefs"] = {**dict(phase.get("artifactRefs") or {}), **artifact_refs}
        phase["percent"] = _phase_percent(int(phase.get("current") or 0), int(phase.get("total") or 0))
        self._write()

    def phase_advance(
        self,
        key: str,
        *,
        delta: int = 1,
        current: int | None = None,
        total: int | None = None,
        current_item: JsonDict | None = None,
        counters: JsonDict | None = None,
        artifact_refs: JsonDict | None = None,
    ) -> None:
        phase = self._phase(key)
        now = utc_timestamp()
        phase["status"] = "running"
        phase.setdefault("startedAt", now)
        phase["updatedAt"] = now
        if total is not None:
            phase["total"] = max(0, int(total))
        if current is None:
            phase["current"] = max(0, int(phase.get("current") or 0) + int(delta))
        else:
            phase["current"] = max(0, int(current))
        total_value = int(phase.get("total") or 0)
        if total_value > 0:
            phase["current"] = min(int(phase.get("current") or 0), total_value)
        if current_item is not None:
            phase["currentItem"] = current_item
        if counters:
            phase["counters"] = {**dict(phase.get("counters") or {}), **counters}
        if artifact_refs:
            phase["artifactRefs"] = {**dict(phase.get("artifactRefs") or {}), **artifact_refs}
        phase["percent"] = _phase_percent(int(phase.get("current") or 0), int(phase.get("total") or 0))
        self._write()

    def phase_finish(
        self,
        key: str,
        *,
        status: str = "succeeded",
        current: int | None = None,
        total: int | None = None,
        current_item: JsonDict | None = None,
        counters: JsonDict | None = None,
        artifact_refs: JsonDict | None = None,
        error: str | None = None,
        replace_counters: bool = False,
    ) -> None:
        phase = self._phase(key)
        now = utc_timestamp()
        phase["status"] = status
        phase["updatedAt"] = now
        phase["finishedAt"] = now
        if total is not None:
            phase["total"] = max(0, int(total))
        if current is not None:
            phase["current"] = max(0, int(current))
        elif int(phase.get("total") or 0) > 0 and status == "succeeded":
            phase["current"] = int(phase["total"])
        total_value = int(phase.get("total") or 0)
        if total_value > 0:
            phase["current"] = min(int(phase.get("current") or 0), total_value)
        if current_item is not None:
            phase["currentItem"] = current_item
        if counters is not None:
            phase["counters"] = (
                dict(counters)
                if replace_counters
                else {**dict(phase.get("counters") or {}), **counters}
            )
        if artifact_refs:
            phase["artifactRefs"] = {**dict(phase.get("artifactRefs") or {}), **artifact_refs}
        if error:
            phase["error"] = error
        phase["percent"] = _phase_percent(int(phase.get("current") or 0), int(phase.get("total") or 0))
        self._write()

    def set_status(self, status: str) -> None:
        self.status = status
        self._write()

    def set_extra_state(self, extra_state: JsonDict) -> None:
        self.extra_state = dict(extra_state)
        self._write()

    def replace_phase_states(
        self,
        phases: list[JsonDict],
        *,
        status: str | None = None,
        extra_state: JsonDict | None = None,
    ) -> None:
        next_phases = {
            str(phase["key"]): dict(phase)
            for phase in phases
            if isinstance(phase, dict) and isinstance(phase.get("key"), str)
        }
        for phase in default_phase_states():
            next_phases.setdefault(str(phase["key"]), phase)
        self.phases = next_phases
        if status is not None:
            self.status = status
        if extra_state is not None:
            self.extra_state = dict(extra_state)
        self._write()

    def upsert_tree_path(
        self,
        *,
        dataset_id: str,
        algorithm_id: str | None = None,
        seed: int | None = None,
        attack_id: str | None = None,
        variant_key: str | None = None,
        refs: JsonDict,
    ) -> None:
        datasets = self.artifact_tree.setdefault("datasets", {})
        dataset_node = datasets.setdefault(dataset_id, {"id": dataset_id})
        if algorithm_id is None:
            dataset_node.update(refs)
        elif attack_id is None:
            algorithms = dataset_node.setdefault("watermarks", {})
            algorithm_node = algorithms.setdefault(algorithm_id, {"id": algorithm_id, "seeds": {}})
            if seed is None:
                algorithm_node.update(refs)
            else:
                seed_node = algorithm_node.setdefault("seeds", {}).setdefault(str(seed), {"seed": seed})
                seed_node.update(refs)
        else:
            algorithms = dataset_node.setdefault("watermarks", {})
            algorithm_node = algorithms.setdefault(algorithm_id, {"id": algorithm_id, "seeds": {}})
            seed_node = algorithm_node.setdefault("seeds", {}).setdefault(str(seed or 0), {"seed": seed})
            attacks = seed_node.setdefault("attacks", {})
            attack_node = attacks.setdefault(attack_id, {"id": attack_id, "variants": {}})
            variant_node = attack_node.setdefault("variants", {}).setdefault(variant_key or "default", {"variantKey": variant_key or "default"})
            variant_node.update(refs)
        self.artifact_tree["updatedAt"] = utc_timestamp()
        self._write_tree()

    def phases_list(self) -> list[JsonDict]:
        return [self._phase(key) for key in PHASE_ORDER]

    def run_state(self) -> JsonDict:
        phases = self.phases_list()
        active = next((phase for phase in phases if phase.get("status") == "running"), None)
        current_phase = str(active.get("key")) if active else self._latest_reached_phase(phases)
        overall = self._overall_progress(phases)
        return {
            "runId": self.run_id,
            "status": self.status,
            "currentPhase": current_phase,
            "overallProgress": overall,
            "progress": overall,
            "progressKind": "phaseOperations",
            "expectedResultUnits": self.expected_result_units,
            "selection": self.selection,
            "artifactRoot": str(self.run_root),
            "materializedRoot": str(self.materialized_root),
            "phaseStatePath": str(self.paths["phaseState"]),
            "artifactTreePath": str(self.paths["artifactTree"]),
            "summaryPath": str(self.paths["runSummary"]),
            "phases": phases,
            "updatedAt": utc_timestamp(),
            **self.extra_state,
        }

    def _phase(self, key: str) -> JsonDict:
        if key not in self.phases:
            raise KeyError(f"Unknown phase key: {key}")
        return self.phases[key]

    def _latest_reached_phase(self, phases: list[JsonDict]) -> str:
        latest = "canonical"
        for phase in phases:
            if phase.get("status") != "pending":
                latest = str(phase.get("key") or latest)
        return latest

    def _overall_progress(self, phases: list[JsonDict]) -> int:
        if not phases:
            return 0
        score = 0.0
        for phase in phases:
            status = str(phase.get("status") or "pending")
            if status == "succeeded":
                score += 1.0
            elif status in {"running", "failed", "paused", "cancelled"}:
                score += max(0.0, min(1.0, float(phase.get("percent") or 0) / 100.0))
        return int(round((score / len(phases)) * 100))

    def _write_tree(self) -> None:
        write_json(self.paths["artifactTree"], self.artifact_tree)

    def _write(self) -> None:
        phase_payload = {
            "runId": self.run_id,
            "phases": self.phases_list(),
            "updatedAt": utc_timestamp(),
        }
        write_json(self.paths["phaseState"], phase_payload)
        self._write_tree()
        state = self.run_state()
        write_json(self.paths["runState"], state)
        if self.on_state is not None:
            self.on_state(state)


def write_run_status(
    paths: dict[str, Path],
    *,
    run_id: str,
    status: str,
    result_units: int,
    expected_result_units: int,
    error: str | None = None,
) -> None:
    write_json(
        paths["runStatus"],
        {
            "runId": run_id,
            "status": status,
            "resultUnitCount": result_units,
            "expectedResultUnits": expected_result_units,
            "progress": progress(result_units, expected_result_units),
            "completedProgress": progress(result_units, expected_result_units),
            "progressKind": "phaseOperations",
            "error": error,
            "updatedAt": utc_timestamp(),
        },
    )


def latest_result_unit_map(result_units_path: Path) -> dict[str, JsonDict]:
    return {
        str(key): dict(record)
        for record in read_jsonl(result_units_path)
        if (key := _result_unit_key(record)) is not None
    }
