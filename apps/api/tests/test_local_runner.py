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
            self.assertTrue(extract_manifest[0]["metadata"]["match"])
            self.assertIsNone(summary["cells"][0]["bitAccuracy"])
            self.assertEqual(summary["score"]["status"], "provisional")
            self.assertEqual(summary["score"]["protocolId"], "waves-official-detection-v1")
            self.assertIn("leaderboardRows", summary["score"])
            self.assertIsNone(summary["cells"][0]["scoring"]["tprAtFpr"])
            self.assertIsNone(summary["aggregates"][0]["meanBitErrorRate"])
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
