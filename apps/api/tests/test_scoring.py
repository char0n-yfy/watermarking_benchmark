from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.scoring import (
    PROTOCOL_ID,
    aggregate_benchmark_score,
    attack_category,
    compute_quality_summary,
    score_cell,
)


class ScoringTest(unittest.TestCase):
    def test_ctrlregen_and_nfpa_are_regeneration_scoring_categories(self) -> None:
        self.assertEqual(attack_category("noise_to_image"), "regeneration-single")
        self.assertEqual(attack_category("image_to_vedio"), "regeneration-single")
        self.assertEqual(attack_category("3d_viewpoint_rerendering_phase0_point"), "regeneration-single")

    def test_score_cell_uses_negative_quantile_for_low_fpr_threshold(self) -> None:
        scoring = score_cell(
            algorithm_id="alg-demo",
            attack_preset_id="atk-jpeg",
            attack_method="jpeg",
            attack_strength=0.5,
            sample_count=3,
            positive_extract_results=[
                SimpleNamespace(metadata={"bit_accuracy": 0.95}),
                SimpleNamespace(metadata={"bit_accuracy": 0.85}),
            ],
            negative_extract_results=[
                SimpleNamespace(metadata={"bit_accuracy": 0.25}),
                SimpleNamespace(metadata={"bit_accuracy": 0.4}),
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
        self.assertIn("regeneration-single", score["coverage"]["missingCategories"])

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


if __name__ == "__main__":
    unittest.main()
