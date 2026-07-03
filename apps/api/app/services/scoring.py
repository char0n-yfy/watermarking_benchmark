from __future__ import annotations

import math
import os
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from evaluator.execution import ExecutionProfile, resolve_cpu_workers, resolve_named_batch_size


JsonDict = dict[str, Any]

PROTOCOL_ID = "wrs-v2-detection-v1"
LEGACY_PROTOCOL_IDS = {"waves-official-detection-v1"}
PROTOCOL_NAME = "WRS-v2 Physical-Aware Detection"
FPR_TARGET = 0.001
PRACTICAL_NQD_THRESHOLD = 0.8
OFFICIAL_MIN_SAMPLES = 5000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PERCEPTUAL_RESIZE_SHORT_SIDE = 256
PERFORMANCE_THRESHOLDS = (0.95, 0.7)
SCORE_TIE_BUFFER = 0.01
STRENGTH_PARAM_KEYS = (
    "strength",
    "quality",
    "step",
    "steps",
    "xy",
    "severity",
    "level",
    "amount",
    "scale",
)


@dataclass(frozen=True)
class AttackCategory:
    key: str
    label: str
    description: str


WRS_ATTACK_CATEGORIES = [
    AttackCategory("distortion-single", "Distortion Single", "Single image distortion attacks."),
    AttackCategory("distortion-combination", "Distortion Combination", "Combined distortion pipelines."),
    AttackCategory("content-preserving-workflow", "Content-Preserving Workflow", "Realistic editing/export workflows that preserve image semantics."),
    AttackCategory("consumer-enhancement-workflow", "Consumer Enhancement Workflow", "Consumer enhancement and restoration workflows."),
    AttackCategory("regeneration", "Regeneration", "VAE, diffusion, video, 3D re-rendering, and rinsing-style regeneration attacks."),
    AttackCategory("physical-screen", "Physical Screen", "Screen-shooting physical channel attacks."),
    AttackCategory("physical-print", "Physical Print", "Print-camera physical channel attacks."),
    AttackCategory("physical-combined", "Physical Combined", "Multi-hop print-camera and screen-shooting combined physical attacks."),
    AttackCategory("adversarial", "Adversarial", "Adversarial embedding or surrogate-detector attacks."),
]

WAVES_ATTACK_CATEGORIES = WRS_ATTACK_CATEGORIES
WAVES_CATEGORY_KEYS = [category.key for category in WRS_ATTACK_CATEGORIES]

# The official WAVES paper calibrates quality metrics with corpus-level 10% and
# 90% quantiles. These anchors make local smoke runs scoreable before the project
# has an official calibration corpus; the summary marks such runs provisional.
QUALITY_BOUNDS = {
    "psnr_degradation": (0.02, 0.45),
    "ssim_degradation": (0.001, 0.25),
    "ms_ssim_degradation": (0.001, 0.25),
    "nmi_degradation": (0.05, 0.55),
}


def benchmark_protocols() -> list[JsonDict]:
    return [
        {
            "id": PROTOCOL_ID,
            "name": PROTOCOL_NAME,
            "task": "detection",
            "rankMethod": "WRS-v2: mean attack-family AUC over normalized strength in practical NQD range",
            "fprTarget": FPR_TARGET,
            "officialMinSamples": OFFICIAL_MIN_SAMPLES,
            "practicalNqdThreshold": PRACTICAL_NQD_THRESHOLD,
            "performanceThresholds": list(PERFORMANCE_THRESHOLDS),
            "requiredCategories": [category.__dict__ for category in WRS_ATTACK_CATEGORIES],
            "qualityMetrics": [
                "PSNR",
                "SSIM",
                "MS-SSIM",
                "NMI",
                "LPIPS",
                "DISTS",
            ],
            "status": "provisional-local-calibration",
        }
    ]


def attack_category(method: str, preset_id: str | None = None) -> str:
    token = f"{preset_id or ''} {method}".lower()
    parts = [part for part in token.replace("-", "_").split() if part]
    if "identity" in token:
        return "clean-control"
    if "combined_physical" in token or "combined-physical" in token:
        return "physical-combined"
    if "print_camera" in token or "print-camera" in token:
        return "physical-print"
    if "screen_shoot" in token or "screen-shoot" in token:
        return "physical-screen"
    if "adv" in token or "surrogate" in token:
        return "adversarial"
    if any(part.startswith("cp_") for part in parts) or "content_preserve" in token:
        return "content-preserving-workflow"
    if any(part.startswith("cew_") for part in parts) or "consumer_enhancement" in token:
        return "consumer-enhancement-workflow"
    if (
        "2x" in token
        or "4x" in token
        or "rinse" in token
        or "rinsing" in token
        or "rerender" in token
        or "regen" in token
        or "vae" in token
        or "diffusion" in token
        or "noise_to_image" in token
        or "image_to_vedio" in token
        or "3d_viewpoint_rerendering" in token
    ):
        return "regeneration"
    if "combo" in token or "distcom" in token:
        return "distortion-combination"
    return "distortion-single"


def attack_param_strength(attack_strength: float, attack_params: Mapping[str, Any] | None = None) -> JsonDict:
    params = attack_params if isinstance(attack_params, Mapping) else {}
    for key in STRENGTH_PARAM_KEYS:
        value = params.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return {"name": key, "value": float(value)}
    return {"name": "strength", "value": float(attack_strength)}


def attack_variant_summary(attack_params: Mapping[str, Any] | None = None) -> JsonDict:
    if not isinstance(attack_params, Mapping):
        return {"key": "default", "label": "default"}
    variant_items: dict[str, Any] = {}
    for key, value in attack_params.items():
        key_text = str(key)
        if key_text in STRENGTH_PARAM_KEYS or value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            variant_items[key_text] = value
    if not variant_items:
        return {"key": "default", "label": "default"}
    key = json.dumps(variant_items, sort_keys=True, ensure_ascii=True, default=str)
    label = ", ".join(f"{name}={_format_variant_value(value)}" for name, value in sorted(variant_items.items()))
    return {"key": key, "label": label}


def _format_variant_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def score_cell(
    *,
    algorithm_id: str,
    attack_preset_id: str,
    attack_method: str,
    attack_strength: float,
    sample_count: int,
    positive_extract_results: Iterable[Any],
    negative_extract_results: Iterable[Any],
    quality_summary: JsonDict,
    clean_quality_summary: JsonDict,
    elapsed_ms: float,
    attack_params: Mapping[str, Any] | None = None,
) -> JsonDict:
    positive_scores = [_detection_score(result) for result in positive_extract_results]
    negative_scores = [_detection_score(result) for result in negative_extract_results]
    positive_scores = [score for score in positive_scores if score is not None]
    negative_scores = [score for score in negative_scores if score is not None]
    threshold = _quantile(negative_scores, 1.0 - FPR_TARGET)
    tpr = None
    empirical_fpr = None
    if threshold is not None and positive_scores:
        tpr = sum(1 for score in positive_scores if score > threshold) / len(positive_scores)
        empirical_fpr = (
            sum(1 for score in negative_scores if score > threshold) / len(negative_scores)
            if negative_scores
            else None
        )

    nqd = quality_summary.get("normalizedQualityDegradation")
    clean_nqd = clean_quality_summary.get("normalizedQualityDegradation")
    category = attack_category(attack_method, attack_preset_id)
    variant = attack_variant_summary(attack_params)
    param_strength = attack_param_strength(attack_strength, attack_params)
    return {
        "protocolId": PROTOCOL_ID,
        "algorithmId": algorithm_id,
        "attackPresetId": attack_preset_id,
        "attackMethod": attack_method,
        "attackCategory": category,
        "attackStrength": attack_strength,
        "attackParams": dict(attack_params or {}),
        "attackVariantKey": variant["key"],
        "attackVariantLabel": variant["label"],
        "attackParamStrengthName": param_strength["name"],
        "attackParamStrength": param_strength["value"],
        "sampleCount": sample_count,
        "fprTarget": FPR_TARGET,
        "detectionThreshold": threshold,
        "tprAtFpr": tpr,
        "empiricalFpr": empirical_fpr,
        "meanPositiveDetectionScore": _mean(positive_scores),
        "meanNegativeDetectionScore": _mean(negative_scores),
        "positiveScoreCount": len(positive_scores),
        "negativeScoreCount": len(negative_scores),
        "normalizedQualityDegradation": nqd,
        "cleanNormalizedQualityDegradation": clean_nqd,
        "cleanFidelity": None if clean_nqd is None else max(0.0, min(1.0, 1.0 - float(clean_nqd))),
        "quality": quality_summary,
        "cleanQuality": clean_quality_summary,
        "elapsedMs": elapsed_ms,
        "practicalForWrs": category in WAVES_CATEGORY_KEYS
        and tpr is not None
        and nqd is not None
        and float(nqd) < PRACTICAL_NQD_THRESHOLD,
    }


def score_cell_from_records(
    *,
    algorithm_id: str,
    attack_preset_id: str,
    attack_method: str,
    attack_strength: float,
    sample_count: int,
    detection_records: Iterable[Mapping[str, Any]],
    quality_records: Iterable[Mapping[str, Any]],
    clean_quality_records: Iterable[Mapping[str, Any]],
    elapsed_ms: float,
    attack_params: Mapping[str, Any] | None = None,
) -> JsonDict:
    detections = list(detection_records)
    positive_records = [record for record in detections if int(record.get("label", 1) or 0) == 1]
    negative_records = [record for record in detections if int(record.get("label", 1) or 0) == 0]
    positive_scores = [_detection_score(record) for record in positive_records]
    negative_scores = [_detection_score(record) for record in negative_records]
    positive_scores = [score for score in positive_scores if score is not None]
    negative_scores = [score for score in negative_scores if score is not None]
    threshold = _quantile(negative_scores, 1.0 - FPR_TARGET)
    tpr = None
    empirical_fpr = None
    if threshold is not None and positive_scores:
        tpr = sum(1 for score in positive_scores if score > threshold) / len(positive_scores)
        empirical_fpr = (
            sum(1 for score in negative_scores if score > threshold) / len(negative_scores)
            if negative_scores
            else None
        )

    quality_summary = summarize_quality_records(quality_records)
    clean_quality_summary = summarize_quality_records(clean_quality_records)
    nqd = quality_summary.get("normalizedQualityDegradation")
    clean_nqd = clean_quality_summary.get("normalizedQualityDegradation")
    category = attack_category(attack_method, attack_preset_id)
    variant = attack_variant_summary(attack_params)
    param_strength = attack_param_strength(attack_strength, attack_params)
    return {
        "protocolId": PROTOCOL_ID,
        "algorithmId": algorithm_id,
        "attackPresetId": attack_preset_id,
        "attackMethod": attack_method,
        "attackCategory": category,
        "attackStrength": attack_strength,
        "attackParams": dict(attack_params or {}),
        "attackVariantKey": variant["key"],
        "attackVariantLabel": variant["label"],
        "attackParamStrengthName": param_strength["name"],
        "attackParamStrength": param_strength["value"],
        "sampleCount": sample_count,
        "fprTarget": FPR_TARGET,
        "detectionThreshold": threshold,
        "tprAtFpr": tpr,
        "empiricalFpr": empirical_fpr,
        "meanPositiveDetectionScore": _mean(positive_scores),
        "meanNegativeDetectionScore": _mean(negative_scores),
        "positiveScoreCount": len(positive_scores),
        "negativeScoreCount": len(negative_scores),
        "normalizedQualityDegradation": nqd,
        "cleanNormalizedQualityDegradation": clean_nqd,
        "cleanFidelity": None if clean_nqd is None else max(0.0, min(1.0, 1.0 - float(clean_nqd))),
        "quality": quality_summary,
        "cleanQuality": clean_quality_summary,
        "elapsedMs": elapsed_ms,
        "practicalForWrs": category in WAVES_CATEGORY_KEYS
        and tpr is not None
        and nqd is not None
        and float(nqd) < PRACTICAL_NQD_THRESHOLD,
    }


def aggregate_benchmark_score(cells: list[JsonDict]) -> JsonDict:
    scored_cells = [cell for cell in cells if isinstance(cell.get("scoring"), dict)]
    score = _aggregate_benchmark_core(scored_cells)
    score["leaderboardRows"] = rank_algorithm_scores(scored_cells)
    score["curvePoints"] = build_curve_points(scored_cells)
    score["attackLeaderboard"] = rank_attack_scores(scored_cells)
    return score


def _aggregate_benchmark_core(scored_cells: list[JsonDict]) -> JsonDict:
    category_scores: list[JsonDict] = []
    for category in WRS_ATTACK_CATEGORIES:
        category_cells = [
            cell
            for cell in scored_cells
            if cell["scoring"].get("attackCategory") == category.key
            and cell["scoring"].get("practicalForWrs")
        ]
        attack_scores = [
            _attack_curve_auc(cells_for_attack)
            for cells_for_attack in _group_cells(category_cells, "attackPresetId").values()
        ]
        attack_scores = [score for score in attack_scores if score is not None]
        performances = [_cell_performance(cell) for cell in category_cells]
        performances = [value for value in performances if value is not None]
        nqds = [
            float(cell["scoring"]["normalizedQualityDegradation"])
            for cell in category_cells
            if cell["scoring"].get("normalizedQualityDegradation") is not None
        ]
        category_scores.append(
            {
                "key": category.key,
                "label": category.label,
                "score": _mean(attack_scores),
                "meanPerformance": _mean(performances),
                "meanNqd": _mean(nqds),
                "cellCount": len(category_cells),
                "attackCount": len(attack_scores),
                "covered": bool(category_cells),
            }
        )

    covered = [item for item in category_scores if item["covered"]]
    missing = [item["key"] for item in category_scores if not item["covered"]]
    sample_counts = [
        int(cell["scoring"].get("sampleCount") or 0)
        for cell in scored_cells
        if cell["scoring"].get("attackCategory") in WAVES_CATEGORY_KEYS
    ]
    meets_sample_floor = bool(sample_counts) and min(sample_counts) >= OFFICIAL_MIN_SAMPLES
    official_eligible = len(missing) == 0 and meets_sample_floor
    wrs_values = [float(item["score"]) for item in covered if item["score"] is not None]
    wrs = None if not wrs_values else 100.0 * sum(wrs_values) / len(wrs_values)

    return {
        "protocolId": PROTOCOL_ID,
        "protocolName": PROTOCOL_NAME,
        "status": "official" if official_eligible else "provisional",
        "officialEligible": official_eligible,
        "wrs": wrs,
        "wrsLabel": "WRS-v2" if official_eligible else "Provisional WRS-v2",
        "rankMethod": "mean category AUC over normalized attack strength",
        "fprTarget": FPR_TARGET,
        "practicalNqdThreshold": PRACTICAL_NQD_THRESHOLD,
        "performanceThresholds": list(PERFORMANCE_THRESHOLDS),
        "officialMinSamples": OFFICIAL_MIN_SAMPLES,
        "categoryScores": category_scores,
        "coverage": {
            "requiredCategories": WAVES_CATEGORY_KEYS,
            "coveredCategories": [item["key"] for item in covered],
            "missingCategories": missing,
            "coveredCategoryCount": len(covered),
            "requiredCategoryCount": len(WAVES_ATTACK_CATEGORIES),
            "coverageRatio": len(covered) / len(WAVES_ATTACK_CATEGORIES),
            "minSampleCount": min(sample_counts) if sample_counts else 0,
            "meetsSampleFloor": meets_sample_floor,
        },
    }


def rank_algorithm_scores(scored_cells: list[JsonDict]) -> list[JsonDict]:
    algorithms = sorted({str(cell["algorithmId"]) for cell in scored_cells})
    rows: list[JsonDict] = []
    for algorithm_id in algorithms:
        algorithm_cells = [cell for cell in scored_cells if cell["algorithmId"] == algorithm_id]
        scoring_items = [cell["scoring"] for cell in algorithm_cells]
        nqds = [
            float(item["normalizedQualityDegradation"])
            for item in scoring_items
            if item.get("normalizedQualityDegradation") is not None
        ]
        clean_fidelity = [
            float(item["cleanFidelity"])
            for item in scoring_items
            if item.get("cleanFidelity") is not None
        ]
        elapsed = [
            float(item["elapsedMs"])
            for item in scoring_items
            if item.get("elapsedMs") is not None
        ]
        score = _aggregate_benchmark_core(algorithm_cells)
        physical_values = [
            item["score"]
            for item in score["categoryScores"]
            if item["key"].startswith("physical-") and item.get("score") is not None
        ]
        covered_scores = [
            item
            for item in score["categoryScores"]
            if item["covered"] and item.get("score") is not None
        ]
        worst = min(covered_scores, key=lambda item: float(item["score"])) if covered_scores else None
        rows.append(
            {
                "rank": 0,
                "algorithmId": algorithm_id,
                "protocolId": PROTOCOL_ID,
                "protocolStatus": score["status"],
                "officialEligible": score["officialEligible"],
                "wrs": score["wrs"],
                "physicalScore": _mean(physical_values),
                "worstCategory": worst,
                "profileTags": _profile_tags(score["categoryScores"], _mean(clean_fidelity), _mean(elapsed)),
                "cleanFidelity": _mean(clean_fidelity),
                "avgNqd": _mean(nqds),
                "runtimeMs": _mean(elapsed),
                "coverage": score["coverage"],
                "categoryScores": score["categoryScores"],
                "cellCount": len(algorithm_cells),
            }
        )

    rows.sort(key=lambda row: (row["officialEligible"], row["wrs"] is not None, row["wrs"] or -1), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def build_curve_points(scored_cells: list[JsonDict]) -> list[JsonDict]:
    points: list[JsonDict] = []
    for cell in scored_cells:
        scoring = cell["scoring"]
        if scoring.get("tprAtFpr") is None or scoring.get("normalizedQualityDegradation") is None:
            continue
        attack_params = cell.get("attackParams")
        if not isinstance(attack_params, Mapping):
            attack_params = scoring.get("attackParams") if isinstance(scoring.get("attackParams"), Mapping) else {}
        variant = attack_variant_summary(attack_params)
        param_strength = attack_param_strength(float(cell["attackStrength"]), attack_params)
        points.append(
            {
                "datasetId": cell.get("datasetId") or "",
                "algorithmId": cell["algorithmId"],
                "attackPresetId": cell["attackPresetId"],
                "attackMethod": cell["attackMethod"],
                "attackCategory": scoring.get("attackCategory"),
                "attackStrength": cell["attackStrength"],
                "attackParams": dict(attack_params),
                "attackVariantKey": str(scoring.get("attackVariantKey") or variant["key"]),
                "attackVariantLabel": str(scoring.get("attackVariantLabel") or variant["label"]),
                "attackParamStrengthName": str(scoring.get("attackParamStrengthName") or param_strength["name"]),
                "attackParamStrength": scoring.get("attackParamStrength", param_strength["value"]),
                "sampleCount": int(scoring.get("sampleCount") or cell.get("sampleCount") or 0),
                "xStrength": _normalized_strength(float(cell["attackStrength"])),
                "xNqd": scoring["normalizedQualityDegradation"],
                "yTprAtFpr": scoring["tprAtFpr"],
                "yPerformance": _cell_performance(cell),
            }
        )
    return points


def rank_attack_scores(scored_cells: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    grouped = _group_cells(scored_cells, "algorithmId", "attackPresetId")
    for (algorithm_id, attack_preset_id), cells_for_attack in grouped.items():
        scoring_items = [cell["scoring"] for cell in cells_for_attack]
        category = str(scoring_items[0].get("attackCategory") or "unknown") if scoring_items else "unknown"
        method = str(cells_for_attack[0].get("attackMethod") or attack_preset_id) if cells_for_attack else str(attack_preset_id)
        practical_cells = [
            cell
            for cell in cells_for_attack
            if cell["scoring"].get("tprAtFpr") is not None
            and cell["scoring"].get("normalizedQualityDegradation") is not None
        ]
        rows.append(
            {
                "rank": 0,
                "algorithmId": algorithm_id,
                "attackPresetId": attack_preset_id,
                "attackMethod": method,
                "attackCategory": category,
                "qAtP95": _json_threshold_value(_quality_at_performance(practical_cells, 0.95)),
                "qAtP70": _json_threshold_value(_quality_at_performance(practical_cells, 0.70)),
                "avgPerformance": _mean(_cell_performance(cell) for cell in practical_cells),
                "avgNqd": _mean(
                    float(cell["scoring"]["normalizedQualityDegradation"])
                    for cell in practical_cells
                    if cell["scoring"].get("normalizedQualityDegradation") is not None
                ),
                "auc": _attack_curve_auc(practical_cells),
                "cellCount": len(cells_for_attack),
            }
        )

    rows.sort(key=_attack_rank_key)
    current_rank = 0
    previous_key: tuple[float, float, float, float] | None = None
    for index, row in enumerate(rows, start=1):
        key = _attack_rank_key(row)
        if previous_key is None or _rank_key_distance(previous_key, key) > SCORE_TIE_BUFFER:
            current_rank = index
            previous_key = key
        row["rank"] = current_rank
    return rows


def _group_cells(cells: Iterable[JsonDict], *keys: str) -> dict[Any, list[JsonDict]]:
    grouped: dict[Any, list[JsonDict]] = {}
    for cell in cells:
        key_values = tuple(str(cell.get(key, "")) for key in keys)
        group_key: Any = key_values[0] if len(key_values) == 1 else key_values
        grouped.setdefault(group_key, []).append(cell)
    return grouped


def _cell_performance(cell: JsonDict) -> float | None:
    scoring = cell.get("scoring") if isinstance(cell.get("scoring"), dict) else {}
    value = scoring.get("tprAtFpr")
    if value is None:
        value = cell.get("bitAccuracy")
    if value is None and cell.get("bitErrorRate") is not None:
        try:
            value = 1.0 - float(cell["bitErrorRate"])
        except (TypeError, ValueError):
            value = None
    try:
        return max(0.0, min(1.0, float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_strength(strength: float) -> float:
    return max(0.0, min(1.0, float(strength)))


def _attack_curve_auc(cells_for_attack: Iterable[JsonDict]) -> float | None:
    raw_points: list[tuple[float, float]] = []
    for cell in cells_for_attack:
        performance = _cell_performance(cell)
        if performance is None:
            continue
        try:
            strength = float(cell.get("attackStrength", cell["scoring"].get("attackStrength", 0.0)))
        except (TypeError, ValueError):
            continue
        if math.isfinite(strength):
            raw_points.append((strength, performance))
    if not raw_points:
        return None

    strengths = [point[0] for point in raw_points]
    min_strength = min(strengths)
    max_strength = max(strengths)
    if 0.0 <= min_strength and max_strength <= 1.0:
        normalized_points = [(_normalized_strength(strength), performance) for strength, performance in raw_points]
    elif max_strength > min_strength:
        span = max_strength - min_strength
        normalized_points = [((strength - min_strength) / span, performance) for strength, performance in raw_points]
    else:
        normalized_points = [(0.5, performance) for _strength, performance in raw_points]

    by_strength: dict[float, list[float]] = {}
    for strength, performance in normalized_points:
        by_strength.setdefault(strength, []).append(performance)
    points = sorted((strength, float(_mean(values) or 0.0)) for strength, values in by_strength.items())
    if len(points) == 1:
        return points[0][1]
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += max(0.0, x1 - x0) * (y0 + y1) / 2.0
    return max(0.0, min(1.0, area))


def _quality_at_performance(cells_for_attack: Iterable[JsonDict], threshold: float) -> float | None:
    points: list[tuple[float, float]] = []
    for cell in cells_for_attack:
        performance = _cell_performance(cell)
        scoring = cell.get("scoring") if isinstance(cell.get("scoring"), dict) else {}
        nqd = scoring.get("normalizedQualityDegradation")
        if performance is None or nqd is None:
            continue
        try:
            points.append((float(nqd), performance))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    points.sort(key=lambda point: point[0])
    performances = [performance for _q, performance in points]
    if min(performances) > threshold:
        return math.inf
    if max(performances) < threshold:
        return -math.inf
    previous_q, previous_p = points[0]
    if previous_p <= threshold:
        return previous_q
    for q, performance in points[1:]:
        if performance == threshold:
            return q
        crossed = (previous_p - threshold) * (performance - threshold) <= 0
        if crossed:
            denominator = performance - previous_p
            if abs(denominator) <= 1e-12:
                return min(previous_q, q)
            ratio = (threshold - previous_p) / denominator
            return previous_q + ratio * (q - previous_q)
        previous_q, previous_p = q, performance
    return None


def _json_threshold_value(value: float | None) -> float | str | None:
    if value is None:
        return None
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    return float(value)


def _threshold_sort_value(value: Any) -> float:
    if value == "-inf":
        return -1_000_000.0
    if value == "inf" or value is None:
        return 1_000_000.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1_000_000.0


def _attack_rank_key(row: JsonDict) -> tuple[float, float, float, float]:
    return (
        _threshold_sort_value(row.get("qAtP95")),
        _threshold_sort_value(row.get("qAtP70")),
        _threshold_sort_value(row.get("avgPerformance")),
        _threshold_sort_value(row.get("avgNqd")),
    )


def _rank_key_distance(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def _profile_tags(
    category_scores: list[JsonDict],
    clean_fidelity: float | None,
    runtime_ms: float | None,
) -> list[str]:
    scores = {
        str(item["key"]): float(item["score"])
        for item in category_scores
        if item.get("score") is not None
    }
    tags: list[str] = []
    physical_values = [
        scores[key]
        for key in ("physical-screen", "physical-print", "physical-combined")
        if key in scores
    ]
    if len(physical_values) == 3 and min(physical_values) >= 0.75:
        tags.append("physical-robust")
    if scores.get("physical-screen", 0.0) >= 0.75 and scores.get("physical-print", 1.0) < 0.65:
        tags.append("screen-specialist")
    if scores.get("physical-print", 0.0) >= 0.75 and scores.get("physical-screen", 1.0) < 0.65:
        tags.append("print-specialist")
    if (
        "physical-combined" in scores
        and max(scores.get("physical-screen", 0.0), scores.get("physical-print", 0.0))
        - scores["physical-combined"]
        >= 0.20
    ):
        tags.append("combined-fragile")
    if clean_fidelity is not None and clean_fidelity >= 0.85:
        tags.append("quality-first")
    if runtime_ms is not None and runtime_ms < 1000.0:
        tags.append("fast-lightweight")
    if scores.get("physical-screen", 1.0) < 0.45 or scores.get("physical-print", 1.0) < 0.45:
        tags.append("geometry-fragile")
    return tags[:4]


def compute_quality_summary(reference_dir: Path, target_dir: Path) -> JsonDict:
    pairs = _pair_images(reference_dir, target_dir)
    metrics_by_pair = compute_image_quality_pairs(pairs)
    return summarize_quality_metrics(metrics_by_pair)


def summarize_quality_records(records: Iterable[Mapping[str, Any]]) -> JsonDict:
    metrics_by_pair: list[JsonDict] = []
    for record in records:
        metrics = record.get("metrics")
        if isinstance(metrics, Mapping):
            metrics_by_pair.append(dict(metrics))
    return summarize_quality_metrics(metrics_by_pair)


def summarize_quality_metrics(metrics_by_pair: Iterable[JsonDict]) -> JsonDict:
    metrics_list = list(metrics_by_pair)
    metric_values: dict[str, list[float]] = {
        "psnr": [],
        "ssim": [],
        "msSsim": [],
        "nmi": [],
        "lpips": [],
        "dists": [],
        "psnr_degradation": [],
        "ssim_degradation": [],
        "ms_ssim_degradation": [],
        "nmi_degradation": [],
    }
    for metrics in metrics_list:
        psnr = metrics["psnr"]
        ssim = metrics["ssim"]
        ms_ssim = metrics["msSsim"]
        nmi = metrics["nmi"]
        metric_values["psnr"].append(psnr)
        metric_values["ssim"].append(ssim)
        metric_values["msSsim"].append(ms_ssim)
        metric_values["nmi"].append(nmi)
        if metrics.get("lpips") is not None:
            metric_values["lpips"].append(float(metrics["lpips"]))
        if metrics.get("dists") is not None:
            metric_values["dists"].append(float(metrics["dists"]))
        metric_values["psnr_degradation"].append(max(0.0, 60.0 - min(psnr, 60.0)) / 60.0)
        metric_values["ssim_degradation"].append(max(0.0, 1.0 - ssim))
        metric_values["ms_ssim_degradation"].append(max(0.0, 1.0 - ms_ssim))
        metric_values["nmi_degradation"].append(max(0.0, 1.0 - nmi))

    raw_degradation = {
        key: _mean(values)
        for key, values in metric_values.items()
        if key.endswith("_degradation")
    }
    normalized = {
        key: _normalize_quality_metric(key, value)
        for key, value in raw_degradation.items()
        if value is not None
    }
    nqd = _mean([value for value in normalized.values() if value is not None])
    return {
        "sampleCount": len(metrics_list),
        "metrics": {
            "psnr": _mean(metric_values["psnr"]),
            "ssim": _mean(metric_values["ssim"]),
            "msSsim": _mean(metric_values["msSsim"]),
            "nmi": _mean(metric_values["nmi"]),
            "fid": None,
            "clipFid": None,
            "lpips": _mean(metric_values["lpips"]),
            "dists": _mean(metric_values["dists"]),
            "aestheticDelta": None,
            "artifactDelta": None,
        },
        "rawDegradation": raw_degradation,
        "normalizedMetrics": normalized,
        "normalizedQualityDegradation": nqd,
        "qualityCompleteness": {
            "availableMetrics": len([value for value in normalized.values() if value is not None]),
            "targetMetrics": 6,
            "mode": "local-lightweight-batched",
        },
    }


def compute_image_quality_pair(reference_path: Path, target_path: Path) -> JsonDict:
    return compute_image_quality_pairs([(reference_path, target_path)])[0]


def compute_image_quality_pairs(pairs: Iterable[tuple[Path, Path]]) -> list[JsonDict]:
    metrics, _profile = compute_image_quality_pairs_with_profile(pairs)
    return metrics


def compute_image_quality_pairs_with_profile(pairs: Iterable[tuple[Path, Path]]) -> tuple[list[JsonDict], JsonDict]:
    pair_list = [(Path(reference), Path(target)) for reference, target in pairs]
    if not pair_list:
        profile = ExecutionProfile(
            stage="quality",
            method="image_quality",
            mode="empty",
            job_count=0,
        ).to_json()
        return [], profile
    cpu_metrics, cpu_profile = _compute_cpu_quality_metrics_batch_with_profile(pair_list)
    perceptual_metrics, perceptual_profile = _compute_perceptual_metrics_batch_with_profile(pair_list)
    profile = {
        "stage": "quality",
        "method": "image_quality",
        "mode": "hybrid",
        "jobCount": len(pair_list),
        "cpu": cpu_profile,
        "perceptual": perceptual_profile,
    }
    return [{**cpu, **perceptual} for cpu, perceptual in zip(cpu_metrics, perceptual_metrics)], profile


def _compute_cpu_quality_metrics_batch(pairs: list[tuple[Path, Path]]) -> list[JsonDict]:
    metrics, _profile = _compute_cpu_quality_metrics_batch_with_profile(pairs)
    return metrics


def _compute_cpu_quality_metrics_batch_with_profile(pairs: list[tuple[Path, Path]]) -> tuple[list[JsonDict], JsonDict]:
    worker_config = resolve_cpu_workers(
        "WM_BENCH_QUALITY_CPU_WORKERS",
        len(pairs),
        enabled=True,
        default_cap=32,
    )
    workers = worker_config.value
    profile = ExecutionProfile(
        stage="quality_cpu",
        method="psnr_ssim_ms_ssim_nmi",
        mode="threadpool" if workers > 1 else "serial",
        job_count=len(pairs),
        cpu_workers=workers,
        batch_stage="quality_cpu",
        config={"cpuWorkers": worker_config.to_json()},
    ).to_json()
    if workers <= 1 or len(pairs) <= 1:
        return [_compute_cpu_quality_metrics(reference, target) for reference, target in pairs], profile
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda pair: _compute_cpu_quality_metrics(*pair), pairs)), profile


def _compute_cpu_quality_metrics(reference_path: Path, target_path: Path) -> JsonDict:
    ref, target = _load_pair(reference_path, target_path)
    return {
        "psnr": _psnr(ref, target),
        "ssim": _ssim(ref, target),
        "msSsim": _ms_ssim(ref, target),
        "nmi": _nmi(ref, target),
    }


def _torch_home_checkpoints() -> list[Path]:
    roots: list[Path] = []
    torch_home = os.getenv("TORCH_HOME")
    if torch_home:
        roots.append(Path(torch_home).expanduser() / "hub" / "checkpoints")
    roots.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    return roots


def _has_torchvision_checkpoint(filename: str, min_size_bytes: int) -> bool:
    for root in _torch_home_checkpoints():
        candidate = root / filename
        try:
            if candidate.is_file() and candidate.stat().st_size >= min_size_bytes:
                return True
        except OSError:
            continue
    return False


@lru_cache(maxsize=1)
def _perceptual_backend() -> JsonDict:
    if os.getenv("WM_BENCH_DISABLE_PERCEPTUAL_METRICS", "0") == "1":
        return {"device": "cpu", "models": {}, "errors": {"disabled": "WM_BENCH_DISABLE_PERCEPTUAL_METRICS=1"}}

    try:
        import torch
    except Exception as exc:
        return {"device": "cpu", "models": {}, "errors": {"torch": f"{type(exc).__name__}: {exc}"}}

    requested_device = os.getenv("WM_BENCH_PERCEPTUAL_DEVICE")
    if not requested_device:
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    models: dict[str, Any] = {}
    errors: dict[str, str] = {}

    if _has_torchvision_checkpoint("alexnet-owt-7be5be79.pth", 200_000_000):
        try:
            import warnings

            import lpips

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
                warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")
                models["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:
            errors["lpips"] = f"{type(exc).__name__}: {exc}"
    else:
        errors["lpips"] = "missing torchvision AlexNet weights: alexnet-owt-7be5be79.pth"

    if _has_torchvision_checkpoint("vgg16-397923af.pth", 500_000_000):
        try:
            import warnings

            from evaluator.watermarking.algorithms.videoseal.videoseal.losses.dists import DISTS

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
                warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")
                models["dists"] = DISTS().to(device).eval()
        except Exception as exc:
            errors["dists"] = f"{type(exc).__name__}: {exc}"
    else:
        errors["dists"] = "missing torchvision VGG16 weights: vgg16-397923af.pth"

    return {"device": str(device), "models": models, "errors": errors}


def clear_perceptual_backend() -> None:
    cached = None
    cache_info = _perceptual_backend.cache_info()
    if cache_info.currsize > 0:
        try:
            cached = _perceptual_backend()
        except Exception:
            cached = None
    if isinstance(cached, dict):
        models = cached.get("models") or {}
        for model in list(models.values()):
            try:
                to_fn = getattr(model, "to", None)
                if callable(to_fn):
                    to_fn("cpu")
            except Exception:
                pass
        try:
            models.clear()
        except Exception:
            pass
    _perceptual_backend.cache_clear()
    try:
        from evaluator.runtime_cleanup import torch_cleanup

        torch_cleanup()
    except Exception:
        pass


def _resize_for_perceptual(image: Image.Image) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    if short_side <= 0 or short_side <= PERCEPTUAL_RESIZE_SHORT_SIDE:
        return image
    scale = PERCEPTUAL_RESIZE_SHORT_SIDE / short_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


def _load_perceptual_pair(reference_path: Path, target_path: Path) -> tuple[Any, Any]:
    from torchvision.transforms import functional as TF

    reference = Image.open(reference_path).convert("RGB")
    target = Image.open(target_path).convert("RGB").resize(reference.size, Image.Resampling.BICUBIC)
    reference = _resize_for_perceptual(reference)
    target = target.resize(reference.size, Image.Resampling.BICUBIC)
    ref_tensor = TF.to_tensor(reference).unsqueeze(0)
    target_tensor = TF.to_tensor(target).unsqueeze(0)
    return ref_tensor, target_tensor


def _move_perceptual_batch(batch: Any, device: Any) -> Any:
    import torch

    device_text = str(device)
    non_blocking = False
    if device_text.startswith("cuda") and torch.cuda.is_available():
        try:
            batch = batch.pin_memory()
            non_blocking = True
        except Exception:
            non_blocking = False
    return batch.to(device, non_blocking=non_blocking)


def _perceptual_batch_config(metric: str | None = None):
    return resolve_named_batch_size(
        metric or "default",
        overrides_env="WM_BENCH_PERCEPTUAL_BATCH_SIZES",
        global_env="WM_BENCH_PERCEPTUAL_BATCH_SIZE",
        default=4,
    )


def _perceptual_batch_size(metric: str | None = None) -> int:
    return _perceptual_batch_config(metric).value


def _compute_perceptual_metrics_batch(pairs: list[tuple[Path, Path]]) -> list[JsonDict]:
    metrics, _profile = _compute_perceptual_metrics_batch_with_profile(pairs)
    return metrics


def _compute_perceptual_metrics_batch_with_profile(pairs: list[tuple[Path, Path]]) -> tuple[list[JsonDict], JsonDict]:
    backend = _perceptual_backend()
    models = backend.get("models") or {}
    results: list[JsonDict] = [{"lpips": None, "dists": None} for _pair in pairs]
    base_details = {
        "models": sorted(str(name) for name in models.keys()),
        "errors": dict(backend.get("errors") or {}),
    }
    if not pairs:
        return results, ExecutionProfile(
            stage="quality_perceptual",
            method="lpips_dists",
            mode="empty",
            job_count=0,
            device=str(backend.get("device") or "cpu"),
            batch_stage="quality_perceptual",
            details=base_details,
        ).to_json()
    if not models:
        return results, ExecutionProfile(
            stage="quality_perceptual",
            method="lpips_dists",
            mode="unavailable",
            job_count=len(pairs),
            device=str(backend.get("device") or "cpu"),
            batch_stage="quality_perceptual",
            details=base_details,
        ).to_json()

    try:
        import torch

        device = backend.get("device", "cpu")
        grouped: dict[tuple[int, int], list[tuple[int, Any, Any]]] = {}
        for index, (reference_path, target_path) in enumerate(pairs):
            ref_tensor, target_tensor = _load_perceptual_pair(reference_path, target_path)
            shape = tuple(ref_tensor.shape[-2:])
            grouped.setdefault(shape, []).append((index, ref_tensor, target_tensor))

        lpips_model = models.get("lpips")
        dists_model = models.get("dists")
        batch_configs: dict[str, JsonDict] = {}
        actual_batches: dict[str, list[int]] = {"lpips": [], "dists": []}
        with torch.no_grad():
            for items in grouped.values():
                if lpips_model is not None:
                    batch_config = _perceptual_batch_config("lpips")
                    batch_configs["lpips"] = batch_config.to_json()
                    batch_size = batch_config.value
                    for offset in range(0, len(items), batch_size):
                        chunk = items[offset : offset + batch_size]
                        actual_batches["lpips"].append(len(chunk))
                        indexes = [item[0] for item in chunk]
                        ref_batch = _move_perceptual_batch(torch.cat([item[1] for item in chunk], dim=0), device)
                        target_batch = _move_perceptual_batch(torch.cat([item[2] for item in chunk], dim=0), device)
                        values = lpips_model(ref_batch, target_batch, normalize=True).reshape(len(chunk), -1).mean(dim=1)
                        for result_index, value in zip(indexes, values.detach().cpu().tolist()):
                            results[result_index]["lpips"] = float(value)
                if dists_model is not None:
                    batch_config = _perceptual_batch_config("dists")
                    batch_configs["dists"] = batch_config.to_json()
                    batch_size = batch_config.value
                    for offset in range(0, len(items), batch_size):
                        chunk = items[offset : offset + batch_size]
                        actual_batches["dists"].append(len(chunk))
                        indexes = [item[0] for item in chunk]
                        ref_batch = _move_perceptual_batch(torch.cat([item[1] for item in chunk], dim=0), device)
                        target_batch = _move_perceptual_batch(torch.cat([item[2] for item in chunk], dim=0), device)
                        values = dists_model(ref_batch, target_batch).reshape(len(chunk), -1).mean(dim=1)
                        for result_index, value in zip(indexes, values.detach().cpu().tolist()):
                            results[result_index]["dists"] = float(value)
        actual_batch_values = [size for sizes in actual_batches.values() for size in sizes]
        max_actual_batch = max(actual_batch_values) if actual_batch_values else None
        max_configured_batch = max(
            (int(config["value"]) for config in batch_configs.values() if isinstance(config.get("value"), int)),
            default=None,
        )
        profile = ExecutionProfile(
            stage="quality_perceptual",
            method="lpips_dists",
            mode="batch" if max_actual_batch and max_actual_batch > 1 else "serial",
            job_count=len(pairs),
            device=str(backend.get("device") or "cpu"),
            configured_batch_size=max_configured_batch,
            actual_batch_size=max_actual_batch,
            batch_stage="quality_perceptual",
            supports_batch=True,
            details={
                **base_details,
                "groupCount": len(grouped),
                "batchConfigs": batch_configs,
                "actualBatches": {key: value for key, value in actual_batches.items() if value},
                "transferPolicy": "cpu_stack_then_batch_to_device",
            },
        ).to_json()
    except Exception as exc:
        profile = ExecutionProfile(
            stage="quality_perceptual",
            method="lpips_dists",
            mode="failed_fallback_none",
            job_count=len(pairs),
            device=str(backend.get("device") or "cpu"),
            batch_stage="quality_perceptual",
            supports_batch=True,
            fallback=True,
            fallback_reason=f"{type(exc).__name__}: {exc}",
            details=base_details,
        ).to_json()
    return results, profile


def _compute_perceptual_metrics(reference_path: Path, target_path: Path) -> JsonDict:
    return _compute_perceptual_metrics_batch([(reference_path, target_path)])[0]


def _detection_score(result: Any) -> float | None:
    if isinstance(result, Mapping):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
        for key in ("detection_score", "bit_accuracy", "confidence", "score"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        bits_score = _bit_accuracy_from_values(result.get("expectedBits"), result.get("decodedBits"))
        if bits_score is not None:
            return bits_score
        message_score = _message_match_score(result.get("expectedMessage"), result.get("decodedMessage"))
        if message_score is not None:
            return message_score
        return None

    metadata = getattr(result, "metadata", {}) or {}
    value = metadata.get("detection_score")
    if value is None:
        value = metadata.get("bit_accuracy")
    if value is None:
        value = metadata.get("confidence")
    if value is None:
        value = metadata.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bit_accuracy_from_values(expected: Any, decoded: Any) -> float | None:
    expected_bits = _coerce_bit_string(expected)
    decoded_bits = _coerce_bit_string(decoded)
    if not expected_bits or not decoded_bits:
        return None
    n = max(len(expected_bits), len(decoded_bits))
    expected_bits = expected_bits.ljust(n, "0")
    decoded_bits = decoded_bits.ljust(n, "0")
    return sum(left == right for left, right in zip(expected_bits, decoded_bits)) / max(1, n)


def _coerce_bit_string(value: Any) -> str | None:
    if isinstance(value, str):
        bits = "".join(ch for ch in value if ch in {"0", "1"})
        return bits or None
    if isinstance(value, (list, tuple)):
        try:
            return "".join(str(int(bit)) for bit in value)
        except (TypeError, ValueError):
            return None
    return None


def _message_match_score(expected: Any, decoded: Any) -> float | None:
    if expected is None or decoded is None:
        return None
    expected_text = str(expected)
    decoded_text = str(decoded)
    if not expected_text:
        return None
    if expected_text == decoded_text:
        return 1.0
    n = max(len(expected_text), len(decoded_text), 1)
    expected_text = expected_text.ljust(n, "\0")
    decoded_text = decoded_text.ljust(n, "\0")
    return sum(left == right for left, right in zip(expected_text, decoded_text)) / n


def _pair_images(reference_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    references = {
        path.relative_to(reference_dir).with_suffix("").as_posix(): path
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    pairs: list[tuple[Path, Path]] = []
    for target in sorted(target_dir.rglob("*")):
        if not target.is_file() or target.suffix.lower() not in IMAGE_EXTS:
            continue
        key = target.relative_to(target_dir).with_suffix("").as_posix()
        reference = references.get(key)
        if reference is not None:
            pairs.append((reference, target))
    return pairs


def _load_pair(reference_path: Path, target_path: Path):
    import numpy as np

    reference = Image.open(reference_path).convert("RGB")
    target = Image.open(target_path).convert("RGB").resize(reference.size, Image.Resampling.BICUBIC)
    return (
        np.asarray(reference, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )


def _psnr(reference: Any, target: Any) -> float:
    import numpy as np

    mse = float(np.mean((reference - target) ** 2))
    if mse <= 1e-12:
        return 60.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim(reference: Any, target: Any) -> float:
    import numpy as np

    ref = reference.mean(axis=2)
    tgt = target.mean(axis=2)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_ref = float(ref.mean())
    mu_tgt = float(tgt.mean())
    var_ref = float(((ref - mu_ref) ** 2).mean())
    var_tgt = float(((tgt - mu_tgt) ** 2).mean())
    cov = float(((ref - mu_ref) * (tgt - mu_tgt)).mean())
    numerator = (2 * mu_ref * mu_tgt + c1) * (2 * cov + c2)
    denominator = (mu_ref**2 + mu_tgt**2 + c1) * (var_ref + var_tgt + c2)
    if denominator <= 1e-12:
        return 1.0
    return max(-1.0, min(1.0, numerator / denominator))


def _ms_ssim(reference: Any, target: Any) -> float:
    import numpy as np

    values = [_ssim(reference, target)]
    ref = reference
    tgt = target
    for _scale in range(3):
        if ref.shape[0] < 32 or ref.shape[1] < 32:
            break
        ref = _downsample2x(ref)
        tgt = _downsample2x(tgt)
        values.append(_ssim(ref, tgt))
    clipped = [max(0.0, min(1.0, float(value))) for value in values]
    if not clipped:
        return 1.0
    return float(np.prod(np.asarray(clipped, dtype=np.float64) ** (1.0 / len(clipped))))


def _downsample2x(image: Any) -> Any:
    height = image.shape[0] - (image.shape[0] % 2)
    width = image.shape[1] - (image.shape[1] % 2)
    if height <= 0 or width <= 0:
        return image
    cropped = image[:height, :width]
    return (
        cropped[0::2, 0::2]
        + cropped[1::2, 0::2]
        + cropped[0::2, 1::2]
        + cropped[1::2, 1::2]
    ) / 4.0


def _nmi(reference: Any, target: Any) -> float:
    import numpy as np

    ref = reference.mean(axis=2).reshape(-1)
    tgt = target.mean(axis=2).reshape(-1)
    hist_2d, _, _ = np.histogram2d(ref, tgt, bins=64)
    total = float(hist_2d.sum())
    if total <= 0:
        return 0.0
    pxy = hist_2d / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    if hx <= 1e-12 or hy <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, mi / math.sqrt(hx * hy)))


def _normalize_quality_metric(metric_key: str, value: float | None) -> float | None:
    if value is None:
        return None
    low, high = QUALITY_BOUNDS[metric_key]
    if high <= low:
        return None
    return 0.1 + 0.8 * ((float(value) - low) / (high - low))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(max(0.0, min(1.0, q)) * len(ordered))) - 1))
    return ordered[index]


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)
