from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.local_db import LocalDatabase
from app.services.experiment_service import ExperimentService


class ExperimentServiceTest(unittest.TestCase):
    def test_create_run_queues_then_executes_local_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), (120, 160, 200)).save(dataset_dir / "sample.png")
            service = ExperimentService(
                database=LocalDatabase(root / "state.sqlite"),
                resources_root=root / "resources",
                runs_root=root / "runs",
            )

            config = service.create_config(
                "Smoke",
                {
                    "datasetIds": ["smoke"],
                    "algorithmIds": ["alg-traditional-lsb"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            self.assertEqual(run["status"], "queued")
            finished = service.execute_run(run["id"])
            results = service.get_run_results(run["id"])
            score = service.get_run_score(run["id"])

            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(len(results["cells"]), 1)
            self.assertEqual(results["cells"][0]["bitAccuracy"], 1.0)
            self.assertEqual(results["cells"][0]["bitErrorRate"], 0.0)
            self.assertEqual(results["aggregates"][0]["meanBitAccuracy"], 1.0)
            self.assertEqual(results["score"]["protocolId"], "waves-official-detection-v1")
            self.assertEqual(score["score"]["status"], "provisional")
            self.assertEqual(service.list_benchmark_protocols()[0]["id"], "waves-official-detection-v1")
            self.assertTrue(results["summaryExists"])

    def test_cancel_queued_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), (120, 160, 200)).save(dataset_dir / "sample.png")
            service = ExperimentService(
                database=LocalDatabase(root / "state.sqlite"),
                resources_root=root / "resources",
                runs_root=root / "runs",
            )
            config = service.create_config(
                "Smoke",
                {
                    "datasetIds": ["smoke"],
                    "algorithmIds": ["alg-traditional-lsb"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            cancelled = service.cancel_run(run["id"])

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertTrue(cancelled["cancelRequested"])


if __name__ == "__main__":
    unittest.main()
