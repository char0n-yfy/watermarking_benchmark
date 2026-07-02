from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from evaluator.execution import execution_environment_snapshot
from evaluator.image_protocol import (
    CANONICAL_IMAGE_SIZE,
    CANONICAL_OUTPUT_POLICY,
    CANONICAL_PREPROCESS_POLICY,
)

from app.core.storage import safe_segment
from app.services.experiment_stages import normalize_attack_params_for_runtime
from app.services.resources import (
    get_attack_catalog_item,
    get_dataset_by_id,
    scan_dataset_resources,
)


JsonDict = dict[str, Any]


def _ensure_list(value: Any, fallback: list[Any]) -> list[Any]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return fallback


def _normalize_strength_overrides(value: Any, valid_attack_ids: list[str]) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        return {}

    valid_ids = set(valid_attack_ids)
    normalized: dict[str, list[float]] = {}
    for attack_id, strengths in value.items():
        normalized_attack_id = str(attack_id)
        if normalized_attack_id not in valid_ids:
            continue
        strength_values: list[float] = []
        for strength in _ensure_list(strengths, []):
            try:
                parsed = float(strength)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                strength_values.append(parsed)
        deduped = sorted(set(strength_values))
        if deduped:
            normalized[normalized_attack_id] = deduped
    return normalized


def _normalize_param_overrides(value: Any, valid_attack_ids: list[str]) -> dict[str, list[JsonDict]]:
    if not isinstance(value, dict):
        return {}

    valid_ids = set(valid_attack_ids)
    normalized: dict[str, list[JsonDict]] = {}
    for attack_id, variants in value.items():
        normalized_attack_id = str(attack_id)
        if normalized_attack_id not in valid_ids:
            continue
        cleaned_variants: list[JsonDict] = []
        seen: set[str] = set()
        for variant in _ensure_list(variants, []):
            if not isinstance(variant, dict):
                continue
            cleaned: JsonDict = {}
            for key, raw_value in variant.items():
                if not isinstance(key, str) or raw_value is None:
                    continue
                if isinstance(raw_value, (str, bool)):
                    cleaned[key] = raw_value
                elif isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
                    cleaned[key] = raw_value
            if not cleaned:
                continue
            marker = json.dumps(cleaned, sort_keys=True, ensure_ascii=True)
            if marker in seen:
                continue
            seen.add(marker)
            cleaned_variants.append(cleaned)
        if cleaned_variants:
            normalized[normalized_attack_id] = cleaned_variants
    return normalized


def normalize_selection(selection: JsonDict, resources_root: Path) -> JsonDict:
    datasets = scan_dataset_resources(resources_root)
    default_dataset_ids = [datasets[0].id] if datasets else []
    dataset_ids = _ensure_list(selection.get("datasetIds"), default_dataset_ids)
    algorithm_ids = _ensure_list(selection.get("algorithmIds"), ["alg-invisible-watermark-dwtdct"])
    attack_ids = _ensure_list(selection.get("attackPresetIds"), ["atk-identity", "atk-jpeg"])
    normalized_attack_ids = [str(value) for value in attack_ids]
    seeds = [int(seed) for seed in _ensure_list(selection.get("seeds"), [42])]
    max_samples = int(selection.get("maxSamples") or 1)

    return {
        "datasetIds": [str(value) for value in dataset_ids],
        "algorithmIds": [str(value) for value in algorithm_ids],
        "attackPresetIds": normalized_attack_ids,
        "attackStrengthOverrides": _normalize_strength_overrides(
            selection.get("attackStrengthOverrides"), normalized_attack_ids
        ),
        "attackParamOverrides": _normalize_param_overrides(
            selection.get("attackParamOverrides"), normalized_attack_ids
        ),
        "seeds": seeds,
        "maxSamples": max(1, max_samples),
    }


def _strengths_for_attack(selection: JsonDict, attack_id: str, attack: JsonDict) -> list[float]:
    overrides = selection.get("attackStrengthOverrides") or {}
    override_strengths = overrides.get(attack_id) if isinstance(overrides, dict) else None
    if isinstance(override_strengths, list) and override_strengths:
        return [float(strength) for strength in override_strengths]
    return [float(strength) for strength in (attack["strengths"] or [0.0])]


def _param_overrides_for_attack(selection: JsonDict, attack_id: str) -> list[JsonDict]:
    overrides = selection.get("attackParamOverrides") or {}
    override_params = overrides.get(attack_id) if isinstance(overrides, dict) else None
    if not isinstance(override_params, list):
        return []
    return [dict(params) for params in override_params if isinstance(params, dict)]


def _variant_strength(attack: JsonDict, params: JsonDict, fallback: float = 0.0) -> float:
    strength_param = attack.get("strengthParam")
    if strength_param and str(strength_param) in params:
        try:
            return float(params[str(strength_param)])
        except (TypeError, ValueError):
            return fallback
    return fallback


def _params_digest(params: JsonDict) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _attack_params(attack: JsonDict, strength: float) -> JsonDict:
    params = dict(attack.get("params") or {})
    strength_param = attack.get("strengthParam")
    if strength_param:
        value: float | int = float(strength)
        if str(strength_param) in {"scale", "xy"} and float(strength).is_integer():
            value = int(strength)
        params[str(strength_param)] = value
    return normalize_attack_params_for_runtime(str(attack["method"]), params)


def _attack_variants_for_attack(selection: JsonDict, attack_id: str, attack: JsonDict) -> list[tuple[float, JsonDict, str]]:
    param_overrides = _param_overrides_for_attack(selection, attack_id)
    if param_overrides:
        variants: list[tuple[float, JsonDict, str]] = []
        for index, override_params in enumerate(param_overrides):
            strength = _variant_strength(attack, override_params, fallback=0.0)
            params = _attack_params(attack, strength)
            params.update(override_params)
            variants.append((float(strength), params, _params_digest({"index": index, **params})))
        return variants

    return [
        (float(strength), _attack_params(attack, float(strength)), f"{float(strength):g}")
        for strength in _strengths_for_attack(selection, attack_id, attack)
    ]


def estimate_selection(selection: JsonDict, resources_root: Path) -> JsonDict:
    normalized = normalize_selection(selection, resources_root)
    sample_count = 0
    for dataset_id in normalized["datasetIds"]:
        try:
            dataset = get_dataset_by_id(resources_root, dataset_id)
        except KeyError:
            continue
        sample_count += min(dataset.sample_count, normalized["maxSamples"])

    strength_count = 0
    for attack_id in normalized["attackPresetIds"]:
        try:
            attack = get_attack_catalog_item(attack_id)
        except KeyError:
            continue
        strength_count += max(1, len(_attack_variants_for_attack(normalized, attack_id, attack)))

    cell_count = (
        len(normalized["datasetIds"])
        * len(normalized["algorithmIds"])
        * max(1, strength_count)
        * len(normalized["seeds"])
    )
    return {
        "selection": normalized,
        "cellCount": cell_count,
        "sampleCount": sample_count,
        "imageOperationCount": cell_count * max(1, sample_count),
    }


def _cell_key(
    dataset_id: str,
    algorithm_id: str,
    attack_id: str,
    strength: float,
    seed: int,
    variant_key: str | None = None,
) -> str:
    suffix = f"__{variant_key}" if variant_key else ""
    raw = f"{dataset_id}__{algorithm_id}__{attack_id}__{strength:g}__{seed}{suffix}"
    return safe_segment(raw)


def pending_variant_groups(
    selection: JsonDict,
    latest_cells: dict[str, JsonDict],
    *,
    resume: bool,
) -> tuple[dict[tuple[str, str, int], list[JsonDict]], dict[str, JsonDict], int]:
    completed: dict[str, JsonDict] = {}
    groups: dict[tuple[str, str, int], list[JsonDict]] = {}
    skipped = 0

    for dataset_id in selection["datasetIds"]:
        for algorithm_id in selection["algorithmIds"]:
            for seed in selection["seeds"]:
                seed_int = int(seed)
                group_key = (str(dataset_id), str(algorithm_id), seed_int)
                variants: list[JsonDict] = []
                for attack_id in selection["attackPresetIds"]:
                    attack = get_attack_catalog_item(str(attack_id))
                    for strength, attack_params, variant_key in _attack_variants_for_attack(
                        selection,
                        str(attack_id),
                        attack,
                    ):
                        cell_key = _cell_key(
                            str(dataset_id),
                            str(algorithm_id),
                            str(attack_id),
                            float(strength),
                            seed_int,
                            variant_key,
                        )
                        latest = latest_cells.get(cell_key)
                        if resume and latest is not None and latest.get("status") == "succeeded":
                            completed[cell_key] = latest
                            skipped += 1
                            continue
                        variants.append(
                            {
                                "cellKey": cell_key,
                                "attackId": str(attack_id),
                                "attack": attack,
                                "strength": float(strength),
                                "attackParams": attack_params,
                                "variantKey": variant_key,
                            }
                        )
                if variants:
                    groups[group_key] = variants
    return groups, completed, skipped


def build_attack_plan(selection: JsonDict) -> list[JsonDict]:
    attack_plan: list[JsonDict] = []
    for attack_id in selection["attackPresetIds"]:
        attack = get_attack_catalog_item(attack_id)
        attack_plan.append(
            {
                "id": attack_id,
                "method": attack["method"],
                "variants": [
                    {
                        "strength": float(strength),
                        "params": params,
                        "variantKey": variant_key,
                    }
                    for strength, params, variant_key in _attack_variants_for_attack(selection, attack_id, attack)
                ],
            }
        )
    return attack_plan


def build_run_plan_payload(
    *,
    run_id: str,
    selection: JsonDict,
    artifact_paths: dict[str, Path],
    expected_cells: int,
    pending_groups: dict[tuple[str, str, int], list[JsonDict]],
    skipped_cells: int,
    resume: bool,
    created_at: str,
) -> JsonDict:
    return {
        "runId": run_id,
        "selection": selection,
        "expectedCells": expected_cells,
        "artifactFiles": {key: str(path) for key, path in artifact_paths.items()},
        "datasets": selection["datasetIds"],
        "watermarkAlgorithms": selection["algorithmIds"],
        "attacks": build_attack_plan(selection),
        "imageSizeProtocol": {
            "canonicalSize": list(CANONICAL_IMAGE_SIZE),
            "preprocessPolicy": CANONICAL_PREPROCESS_POLICY,
            "watermarkOutputPolicy": CANONICAL_OUTPUT_POLICY,
            "qualityAlignmentPolicy": "resize target to reference only when sizes differ",
        },
        "executionPolicy": execution_environment_snapshot(),
        "resume": resume,
        "resumeMode": "pending_cells_only",
        "resumePendingCells": sum(len(variants) for variants in pending_groups.values()),
        "resumeSkippedSucceededCells": skipped_cells,
        "createdAt": created_at,
    }
