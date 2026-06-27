from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.local_runner import LocalRunRequest, run_local_experiment


class LocalRunnerTest(unittest.TestCase):
    def test_smoke_run_extracts_lsb_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            runs_root = root / "runs"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), (120, 160, 200)).save(dataset_dir / "sample.png")

            summary = run_local_experiment(
                LocalRunRequest(
                    run_id="run_smoke",
                    selection={
                        "datasetIds": ["smoke"],
                        "algorithmIds": ["alg-traditional-lsb"],
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
            self.assertEqual(summary["cells"][0]["bitAccuracy"], 1.0)
            self.assertEqual(summary["cells"][0]["bitErrorRate"], 0.0)
            self.assertEqual(summary["aggregates"][0]["meanBitErrorRate"], 0.0)
            self.assertTrue((runs_root / "run_smoke" / "run_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
