from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.local_runner import LocalRunRequest, run_local_experiment


class LocalRunnerTest(unittest.TestCase):
    def test_smoke_run_extracts_default_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")

            summary = run_local_experiment(
                LocalRunRequest(
                    run_id="run_smoke",
                    selection={
                        "datasetIds": ["smoke"],
                        "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                    resources_root=root / "resources",
                    runs_root=runs_root,
                )
            )

            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["cellCount"], 1)
            self.assertEqual(summary["cells"][0]["status"], "succeeded")
            extract_manifest = json.loads(Path(summary["cells"][0]["manifestPath"]).read_text())
            self.assertEqual(extract_manifest[0]["decodedBits"], extract_manifest[0]["expectedBits"])
            self.assertNotIn("bitAccuracy", summary["cells"][0])
            self.assertNotIn("bitErrorRate", summary["cells"][0])
            self.assertNotIn("scoring", summary["cells"][0])
            self.assertNotIn("score", summary)
            self.assertNotIn("aggregates", summary)
            self.assertNotIn("bit_accuracy", extract_manifest[0]["metadata"])
            self.assertNotIn("match", extract_manifest[0]["metadata"])
            self.assertNotIn("inputPath", extract_manifest[0])
            self.assertNotIn("payloadBits", extract_manifest[0])
            self.assertNotIn("detectionScore", extract_manifest[0])
            self.assertNotIn("negativeManifestPath", summary["cells"][0])
            run_root = runs_root / "run_smoke"
            self.assertTrue((run_root / "run_summary.json").exists())
            self.assertTrue((run_root / "run_plan.json").exists())
            self.assertTrue((run_root / "cell_manifest.jsonl").exists())
            self.assertTrue((run_root / "image_quality.jsonl").exists())
            self.assertTrue((run_root / "image_detection.jsonl").exists())
            self.assertTrue((run_root / "runtime_profile.jsonl").exists())
            self.assertTrue((run_root / "stage_events.jsonl").exists())
            run_status = json.loads((run_root / "run_status.json").read_text())
            cell_summary = json.loads((run_root / "cell_summary_latest.json").read_text())
            self.assertEqual(run_status["progress"], 100)
            self.assertEqual(run_status["completedProgress"], 100)
            self.assertEqual(run_status["progressKind"], "completedCells")
            self.assertEqual(cell_summary["progress"], 100)
            self.assertEqual(cell_summary["completedProgress"], 100)
            self.assertEqual(cell_summary["succeededProgress"], 100)
            self.assertEqual(cell_summary["progressKind"], "completedCells")
            self.assertEqual(summary["completedProgress"], 100)
            self.assertEqual(summary["succeededProgress"], 100)
            self.assertEqual(summary["progressKind"], "completedCells")
            sample_record = json.loads((run_root / "sample_manifest.jsonl").read_text().splitlines()[0])
            quality_record = json.loads((run_root / "image_quality.jsonl").read_text().splitlines()[0])
            runtime_record = json.loads((run_root / "runtime_profile.jsonl").read_text().splitlines()[0])
            self.assertNotIn("stagedPath", sample_record)
            for field in ("width", "height", "referencePath", "targetPath"):
                self.assertNotIn(field, quality_record)
            for field in ("perceptualBackend", "perceptualDevice", "perceptualErrors"):
                self.assertNotIn(field, quality_record["metrics"])
            for field in ("msPerImage", "msPerMP", "throughputImagesPerSecond", "throughputMPPerSecond", "macs", "flops"):
                self.assertNotIn(field, runtime_record)
            self.assertFalse((Path(summary["cells"][0]["outputDir"]) / "attacked").exists())

            resumed = run_local_experiment(
                LocalRunRequest(
                    run_id="run_smoke",
                    selection={
                        "datasetIds": ["smoke"],
                        "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                    resources_root=root / "resources",
                    runs_root=runs_root,
                )
            )
            self.assertEqual(resumed["status"], "succeeded")
            self.assertEqual(resumed["cellCount"], 1)
            self.assertEqual(resumed["skippedCells"], 1)

    def test_negative_attack_outputs_are_reused_across_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (90, 140, 210)).save(dataset_dir / "sample.png")

            summary = run_local_experiment(
                LocalRunRequest(
                    run_id="run_negative_reuse",
                    selection={
                        "datasetIds": ["smoke"],
                        "algorithmIds": [
                            "alg-invisible-watermark-dwtdct",
                            "alg-invisible-watermark-dwtdctsvd",
                        ],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                    resources_root=root / "resources",
                    runs_root=runs_root,
                )
            )

            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["cellCount"], 2)

            run_root = runs_root / "run_negative_reuse"
            profiles = [
                json.loads(line)
                for line in (run_root / "runtime_profile.jsonl").read_text().splitlines()
                if line.strip()
            ]
            negative_profiles = [
                record
                for record in profiles
                if record["stage"] == "attack_negative_control"
            ]
            self.assertEqual(len(negative_profiles), 2)
            self.assertEqual(
                sum(1 for record in negative_profiles if record.get("metadata", {}).get("cacheHit") is True),
                1,
            )
            self.assertEqual(
                sum(1 for record in negative_profiles if record.get("status") == "reused"),
                1,
            )
            self.assertFalse((run_root / "staging" / "negative_attacked" / "smoke").exists())


if __name__ == "__main__":
    unittest.main()
