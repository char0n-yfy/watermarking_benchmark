from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.scoring import (
    PROTOCOL_ID,
    _compute_cpu_quality_metrics_batch_with_profile,
    aggregate_benchmark_score,
    attack_category,
    attack_resource_category,
    build_curve_points,
    compute_image_quality_pairs_with_profile,
    compute_quality_summary,
    score_cell,
    score_cell_from_records,
)


class ScoringTest(unittest.TestCase):
    def test_ctrlregen_and_nfpa_are_regeneration_scoring_categories(self) -> None:
        self.assertEqual(attack_category("noise_to_image"), "regeneration")
        self.assertEqual(attack_category("image_to_vedio"), "regeneration")
        self.assertEqual(attack_category("3d_viewpoint_rerendering_rotate_point"), "regeneration")

    def test_physical_attacks_have_separate_wrs_v2_categories(self) -> None:
        self.assertEqual(attack_category("screen_shoot", "screen_shoot_strong"), "physical-screen")
        self.assertEqual(attack_category("print_camera", "print_camera_medium"), "physical-print")
        self.assertEqual(attack_category("combined_physical", "combined_physical_strong"), "physical-combined")
        self.assertEqual(attack_category("cp_resize_export"), "content-preserving-workflow")
        self.assertEqual(attack_category("cew_restore"), "consumer-enhancement-workflow")

    def test_resource_attack_categories_match_overview_taxonomy(self) -> None:
        self.assertEqual(attack_resource_category("jpeg"), "distortion_attacks")
        self.assertEqual(attack_resource_category("screen_shoot"), "physical_channel_attacks")
        self.assertEqual(
            attack_resource_category("3d_viewpoint_rerendering_rotate_point"),
            "3d_viewpoint_rerendering",
        )
        self.assertEqual(attack_resource_category("regen_vae"), "regeneration_attacks")
        self.assertEqual(attack_resource_category("cew_restore"), "consumer_enhancement_workflow_attacks")
        self.assertIsNone(attack_resource_category("identity"))

    def test_score_cell_uses_negative_quantile_for_low_fpr_threshold(self) -> None:
        scoring = score_cell(
            algorithm_id="alg-demo",
            attack_preset_id="atk-jpeg",
            attack_method="jpeg",
            attack_strength=0.5,
            sample_count=3,
            positive_extract_results=[
                SimpleNamespace(metadata={"detection_score": 0.95}),
                SimpleNamespace(metadata={"detection_score": 0.85}),
            ],
            negative_extract_results=[
                SimpleNamespace(metadata={"detection_score": 0.25}),
                SimpleNamespace(metadata={"detection_score": 0.4}),
            ],
            quality_summary={"normalizedQualityDegradation": 0.2},
            clean_quality_summary={"normalizedQualityDegradation": 0.05},
            elapsed_ms=12.0,
        )

        self.assertEqual(scoring["protocolId"], PROTOCOL_ID)
        self.assertEqual(scoring["attackCategory"], "distortion-single")
        self.assertAlmostEqual(scoring["tprAtFpr"], 1.0)
        self.assertLess(scoring["empiricalFpr"], 0.01)
        self.assertTrue(scoring["practicalForWrs"])

    def test_missing_categories_produce_provisional_wrs(self) -> None:
        cell = {
            "algorithmId": "alg-demo",
            "attackPresetId": "atk-jpeg",
            "attackMethod": "jpeg",
            "attackStrength": 0.5,
            "scoring": {
                "attackCategory": "distortion-single",
                "practicalForWrs": True,
                "tprAtFpr": 0.8,
                "normalizedQualityDegradation": 0.2,
                "sampleCount": 5,
                "cleanFidelity": 0.95,
                "elapsedMs": 10.0,
            },
        }

        score = aggregate_benchmark_score([cell])

        self.assertEqual(score["status"], "provisional")
        self.assertFalse(score["officialEligible"])
        self.assertAlmostEqual(score["wrs"], 80.0)
        self.assertEqual(score["coverage"]["requiredCategoryCount"], 5)
        self.assertIn("regeneration_attacks", score["coverage"]["missingCategories"])

    def test_score_cell_from_records_accepts_bit_strings(self) -> None:
        metrics = {"psnr": 35.0, "ssim": 0.92, "msSsim": 0.93, "nmi": 0.96}
        scoring = score_cell_from_records(
            algorithm_id="alg-demo",
            attack_preset_id="screen_shoot_mild",
            attack_method="screen_shoot",
            attack_strength=0.2,
            sample_count=2,
            detection_records=[
                {"label": 1, "expectedBits": "1111", "decodedBits": "1110"},
                {"label": 0, "expectedBits": "1111", "decodedBits": "0001"},
            ],
            quality_records=[{"metrics": metrics}],
            clean_quality_records=[{"metrics": {"psnr": 58.0, "ssim": 0.99, "msSsim": 0.99, "nmi": 0.99}}],
            elapsed_ms=8.0,
        )

        self.assertEqual(scoring["protocolId"], PROTOCOL_ID)
        self.assertEqual(scoring["attackCategory"], "physical-screen")
        self.assertAlmostEqual(scoring["meanPositiveDetectionScore"], 0.75)
        self.assertAlmostEqual(scoring["meanNegativeDetectionScore"], 0.25)
        self.assertAlmostEqual(scoring["tprAtFpr"], 1.0)
        self.assertTrue(scoring["practicalForWrs"])

    def test_curve_points_preserve_attack_variant_params(self) -> None:
        points = build_curve_points(
            [
                {
                    "datasetId": "dataset-demo",
                    "algorithmId": "alg-demo",
                    "attackPresetId": "atk-regen-vae",
                    "attackMethod": "regen_vae",
                    "attackStrength": 0.0,
                    "attackParams": {"quality": 3, "vae_model_name": "bmshj2018-factorized"},
                    "scoring": {
                        "attackCategory": "regeneration",
                        "tprAtFpr": 0.75,
                        "normalizedQualityDegradation": 0.42,
                        "sampleCount": 12,
                    },
                }
            ]
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["datasetId"], "dataset-demo")
        self.assertEqual(points[0]["attackParamStrengthName"], "quality")
        self.assertEqual(points[0]["attackParamStrength"], 3.0)
        self.assertEqual(points[0]["attackVariantLabel"], "vae_model_name=bmshj2018-factorized")
        self.assertEqual(points[0]["attackParams"]["quality"], 3)
        self.assertEqual(points[0]["sampleCount"], 12)

    def test_quality_summary_returns_lightweight_nqd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            target = root / "target"
            reference.mkdir()
            target.mkdir()
            Image.new("RGB", (32, 32), (120, 160, 200)).save(reference / "sample.png")
            Image.new("RGB", (32, 32), (122, 158, 198)).save(target / "sample.png")

            summary = compute_quality_summary(reference, target)

            self.assertEqual(summary["sampleCount"], 1)
            self.assertIsNotNone(summary["metrics"]["psnr"])
            self.assertIsNotNone(summary["normalizedQualityDegradation"])

    def test_quality_pairs_report_execution_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.png"
            target = root / "target.png"
            Image.new("RGB", (32, 32), (120, 160, 200)).save(reference)
            Image.new("RGB", (32, 32), (122, 158, 198)).save(target)

            metrics, profile = compute_image_quality_pairs_with_profile([(reference, target)])

            self.assertEqual(len(metrics), 1)
            self.assertEqual(profile["mode"], "hybrid")
            self.assertEqual(profile["jobCount"], 1)
            self.assertIn("cpu", profile)
            self.assertIn("perceptual", profile)
            self.assertIn(profile["cpu"]["mode"], {"serial", "threadpool"})

    def test_cpu_quality_metric_workers_can_be_split_by_metric(self) -> None:
        previous = os.environ.get("WM_BENCH_QUALITY_CPU_WORKERS_BY_METRIC")
        os.environ["WM_BENCH_QUALITY_CPU_WORKERS_BY_METRIC"] = "psnr=1,ssim=1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference = root / "reference.png"
                target = root / "target.png"
                Image.new("RGB", (32, 32), (120, 160, 200)).save(reference)
                Image.new("RGB", (32, 32), (122, 158, 198)).save(target)

                metrics, profile = _compute_cpu_quality_metrics_batch_with_profile(
                    [(reference, target)],
                    metrics=("psnr",),
                )

                self.assertEqual(profile["mode"], "split_serial")
                self.assertEqual(profile["config"]["metrics"], ["psnr"])
                self.assertIn("psnr", metrics[0])
                self.assertNotIn("ssim", metrics[0])
        finally:
            if previous is None:
                os.environ.pop("WM_BENCH_QUALITY_CPU_WORKERS_BY_METRIC", None)
            else:
                os.environ["WM_BENCH_QUALITY_CPU_WORKERS_BY_METRIC"] = previous


if __name__ == "__main__":
    unittest.main()
