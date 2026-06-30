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
            run_root = runs_root / "run_smoke"
            self.assertTrue((run_root / "run_summary.json").exists())
            self.assertTrue((run_root / "run_plan.json").exists())
            self.assertTrue((run_root / "cell_manifest.jsonl").exists())
            self.assertTrue((run_root / "image_quality.jsonl").exists())
            self.assertTrue((run_root / "image_detection.jsonl").exists())
            self.assertTrue((run_root / "runtime_profile.jsonl").exists())
            self.assertTrue((run_root / "stage_events.jsonl").exists())
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


if __name__ == "__main__":
    unittest.main()
