from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


JsonDict = dict[str, Any]

PROTOCOL_ID = "waves-official-detection-v1"
PROTOCOL_NAME = "WAVES Official Detection v1"
FPR_TARGET = 0.001
PRACTICAL_NQD_THRESHOLD = 0.8
OFFICIAL_MIN_SAMPLES = 5000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class AttackCategory:
    key: str
    label: str
    description: str


WAVES_ATTACK_CATEGORIES = [
    AttackCategory("distortion-single", "Distortion Single", "Single image distortion attacks."),
    AttackCategory("distortion-combination", "Distortion Combination", "Combined distortion pipelines."),
    AttackCategory("regeneration-single", "Regeneration Single", "Single VAE or diffusion regeneration."),
    AttackCategory("regeneration-rinsing", "Regeneration Rinsing", "Repeated regeneration or rinsing attacks."),
    AttackCategory("adv-embedding-grey-box", "Adv Embedding Grey-box", "Embedding attacks with partial model knowledge."),
    AttackCategory("adv-embedding-black-box", "Adv Embedding Black-box", "Embedding attacks with substitute encoders."),
    AttackCategory("adv-surrogate", "Adv Surrogate Detector", "Attacks using trained surrogate detectors."),
]

WAVES_CATEGORY_KEYS = [category.key for category in WAVES_ATTACK_CATEGORIES]

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
            "fprTarget": FPR_TARGET,
            "officialMinSamples": OFFICIAL_MIN_SAMPLES,
            "practicalNqdThreshold": PRACTICAL_NQD_THRESHOLD,
            "requiredCategories": [category.__dict__ for category in WAVES_ATTACK_CATEGORIES],
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
    if "identity" in token:
        return "clean-control"
    if "advcl" in token or "surrogate" in token:
        return "adv-surrogate"
    if "advembg" in token or "grey" in token or "gray" in token:
        return "adv-embedding-grey-box"
    if "advembb" in token or "black" in token:
        return "adv-embedding-black-box"
    if "2x" in token or "4x" in token or "rinse" in token:
        return "regeneration-rinsing"
    if (
        "regen" in token
        or "vae" in token
        or "diffusion" in token
        or "noise_to_image" in token
        or "image_to_vedio" in token
        or "3d_viewpoint_rerendering" in token
    ):
        return "regeneration-single"
    if "combo" in token or "distcom" in token:
        return "distortion-combination"
    return "distortion-single"


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
    return {
        "protocolId": PROTOCOL_ID,
        "algorithmId": algorithm_id,
        "attackPresetId": attack_preset_id,
        "attackMethod": attack_method,
        "attackCategory": category,
        "attackStrength": attack_strength,
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
    return score


def _aggregate_benchmark_core(scored_cells: list[JsonDict]) -> JsonDict:
    category_scores: list[JsonDict] = []
    for category in WAVES_ATTACK_CATEGORIES:
        category_cells = [
            cell
            for cell in scored_cells
            if cell["scoring"].get("attackCategory") == category.key
            and cell["scoring"].get("practicalForWrs")
        ]
        tprs = [float(cell["scoring"]["tprAtFpr"]) for cell in category_cells]
        nqds = [
            float(cell["scoring"]["normalizedQualityDegradation"])
            for cell in category_cells
            if cell["scoring"].get("normalizedQualityDegradation") is not None
        ]
        category_scores.append(
            {
                "key": category.key,
                "label": category.label,
                "score": _mean(tprs),
                "meanNqd": _mean(nqds),
                "cellCount": len(category_cells),
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
        "wrsLabel": "WRS" if official_eligible else "Provisional WRS",
        "fprTarget": FPR_TARGET,
        "practicalNqdThreshold": PRACTICAL_NQD_THRESHOLD,
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
        rows.append(
            {
                "rank": 0,
                "algorithmId": algorithm_id,
                "protocolId": PROTOCOL_ID,
                "protocolStatus": score["status"],
                "officialEligible": score["officialEligible"],
                "wrs": score["wrs"],
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
        points.append(
            {
                "algorithmId": cell["algorithmId"],
                "attackPresetId": cell["attackPresetId"],
                "attackMethod": cell["attackMethod"],
                "attackCategory": scoring.get("attackCategory"),
                "attackStrength": cell["attackStrength"],
                "xNqd": scoring["normalizedQualityDegradation"],
                "yTprAtFpr": scoring["tprAtFpr"],
            }
        )
    return points


def compute_quality_summary(reference_dir: Path, target_dir: Path) -> JsonDict:
    pairs = _pair_images(reference_dir, target_dir)
    metric_values: dict[str, list[float]] = {
        "psnr": [],
        "ssim": [],
        "msSsim": [],
        "nmi": [],
        "psnr_degradation": [],
        "ssim_degradation": [],
        "ms_ssim_degradation": [],
        "nmi_degradation": [],
    }
    for reference_path, target_path in pairs:
        metrics = compute_image_quality_pair(reference_path, target_path)
        psnr = metrics["psnr"]
        ssim = metrics["ssim"]
        ms_ssim = metrics["msSsim"]
        nmi = metrics["nmi"]
        metric_values["psnr"].append(psnr)
        metric_values["ssim"].append(ssim)
        metric_values["msSsim"].append(ms_ssim)
        metric_values["nmi"].append(nmi)
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
        "sampleCount": len(pairs),
        "metrics": {
            "psnr": _mean(metric_values["psnr"]),
            "ssim": _mean(metric_values["ssim"]),
            "msSsim": _mean(metric_values["msSsim"]),
            "nmi": _mean(metric_values["nmi"]),
            "fid": None,
            "clipFid": None,
            "lpips": None,
            "dists": None,
            "aestheticDelta": None,
            "artifactDelta": None,
        },
        "rawDegradation": raw_degradation,
        "normalizedMetrics": normalized,
        "normalizedQualityDegradation": nqd,
        "qualityCompleteness": {
            "availableMetrics": len([value for value in normalized.values() if value is not None]),
            "targetMetrics": 6,
            "mode": "local-lightweight",
        },
    }


def compute_image_quality_pair(reference_path: Path, target_path: Path) -> JsonDict:
    ref, target = _load_pair(reference_path, target_path)
    return {
        "psnr": _psnr(ref, target),
        "ssim": _ssim(ref, target),
        "msSsim": _ms_ssim(ref, target),
        "nmi": _nmi(ref, target),
        "lpips": None,
        "dists": None,
        "perceptualBackend": "not_configured",
    }


def _detection_score(result: Any) -> float | None:
    metadata = getattr(result, "metadata", {}) or {}
    value = metadata.get("detection_score")
    if value is None:
        value = metadata.get("bit_accuracy")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    target = Image.open(target_path).convert("RGB").resize(reference.size)
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
