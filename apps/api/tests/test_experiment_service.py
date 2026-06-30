from __future__ import annotations

import json
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
    def test_create_config_rejects_unknown_resource_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")
            service = ExperimentService(
                database=LocalDatabase(root / "state.sqlite"),
                resources_root=root / "resources",
                runs_root=root / "runs",
            )
            valid_selection = {
                "datasetIds": ["smoke"],
                "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                "attackPresetIds": ["atk-jpeg"],
                "seeds": [42],
                "maxSamples": 1,
            }

            invalid_cases = [
                {"datasetIds": ["missing-dataset"]},
                {"algorithmIds": ["alg-missing"]},
                {"attackPresetIds": ["atk-missing"]},
                {"attackStrengthOverrides": {"atk-missing": [0.5]}},
                {"attackParamOverrides": {"atk-missing": [{"strength": 0.5}]}},
            ]
            for override in invalid_cases:
                with self.subTest(override=override):
                    selection = {**valid_selection, **override}
                    with self.assertRaises((KeyError, ValueError)):
                        service.create_config("Invalid", selection)

    def test_create_config_adds_hidden_identity_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")
            service = ExperimentService(
                database=LocalDatabase(root / "state.sqlite"),
                resources_root=root / "resources",
                runs_root=root / "runs",
            )

            config = service.create_config(
                "Smoke",
                {
                    "datasetIds": ["smoke"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                    "attackPresetIds": ["atk-jpeg"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )

            self.assertEqual(config["selection"]["attackPresetIds"], ["atk-jpeg", "atk-identity"])
            self.assertEqual(config["cellCount"], 4)

    def test_create_run_queues_then_executes_local_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")
            service = ExperimentService(
                database=LocalDatabase(root / "state.sqlite"),
                resources_root=root / "resources",
                runs_root=root / "runs",
            )

            config = service.create_config(
                "Smoke",
                {
                    "datasetIds": ["smoke"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
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
            self.assertEqual(results["cells"][0]["status"], "succeeded")
            extract_manifest = json.loads(Path(results["cells"][0]["manifestPath"]).read_text())
            self.assertTrue(extract_manifest[0]["metadata"]["match"])
            self.assertIsNone(results["cells"][0]["bitAccuracy"])
            self.assertIsNone(results["aggregates"][0]["meanBitAccuracy"])
            self.assertEqual(results["score"]["protocolId"], "waves-official-detection-v1")
            self.assertEqual(score["score"]["status"], "provisional")
            self.assertEqual(service.list_benchmark_protocols()[0]["id"], "waves-official-detection-v1")
            self.assertTrue(results["summaryExists"])

    def test_cancel_queued_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")
            service = ExperimentService(
                database=LocalDatabase(root / "state.sqlite"),
                resources_root=root / "resources",
                runs_root=root / "runs",
            )
            config = service.create_config(
                "Smoke",
                {
                    "datasetIds": ["smoke"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
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
