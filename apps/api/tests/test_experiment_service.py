from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                "algorithmIds": ["alg-traditional-spread-dct"],
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            self.assertEqual(run["status"], "queued")
            self.assertEqual(service.list_runs(scope="active")[0]["id"], run["id"])
            finished = service.execute_run(run["id"])
            results = service.get_run_results(run["id"])
            score = service.get_run_score(run["id"])
            events = service.get_run_events(run["id"])

            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["completedProgress"], 100)
            self.assertEqual(finished["progressKind"], "completedCells")
            self.assertNotIn(run["id"], [item["id"] for item in service.list_runs(scope="active")])
            self.assertEqual(len(results["cells"]), 1)
            self.assertEqual(results["cells"][0]["status"], "succeeded")
            self.assertTrue(events["exists"])
            self.assertGreater(len(events["events"]), 0)
            extract_manifest = json.loads(Path(results["cells"][0]["manifestPath"]).read_text())
            self.assertEqual(extract_manifest[0]["decodedBits"], extract_manifest[0]["expectedBits"])
            self.assertIsNone(results["cells"][0]["bitAccuracy"])
            self.assertEqual(results["aggregates"], [])
            self.assertEqual(results["score"]["protocolId"], "waves-official-detection-v1")
            self.assertEqual(score["score"]["status"], "provisional")
            self.assertNotIn("score", results["summary"])
            self.assertNotIn("aggregates", results["summary"])
            self.assertEqual(service.list_benchmark_protocols()[0]["id"], "waves-official-detection-v1")
            self.assertTrue(results["summaryExists"])

    def test_create_run_records_task_name(self) -> None:
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
                "Smoke Config",
                {
                    "datasetIds": ["smoke"],
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )

            named_run = service.create_run(config["id"], name="Task A")
            default_run = service.create_run(config["id"])

            self.assertEqual(named_run["taskName"], "Task A")
            self.assertEqual(default_run["taskName"], "Smoke Config")

    def test_claim_next_run_only_allows_one_worker(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            claimed = service.claim_next_run("worker-a")
            duplicate = service.claim_next_run("worker-b")
            current = service.get_run(run["id"])

            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["id"], run["id"])
            self.assertIsNone(duplicate)
            self.assertEqual(current["status"], "running")
            self.assertEqual(current["workerId"], "worker-a")

    def test_pause_queued_run(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            paused = service.pause_run(run["id"])

            self.assertEqual(paused["status"], "paused")
            self.assertTrue(paused["cancelRequested"])
            self.assertNotIn(run["id"], [item["id"] for item in service.list_runs(scope="active")])
            self.assertEqual([item["id"] for item in service.list_runs(scope="unfinished")], [run["id"]])

            resumed = service.resume_run(run["id"])

            self.assertEqual(resumed["status"], "queued")
            self.assertFalse(resumed["cancelRequested"])
            self.assertEqual(service.list_runs(scope="active")[0]["id"], run["id"])

    def test_cancel_queued_run_is_not_resumable(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            cancelled = service.cancel_run(run["id"])

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertTrue(cancelled["cancelRequested"])
            self.assertNotIn(run["id"], [item["id"] for item in service.list_runs(scope="active")])
            self.assertNotIn(run["id"], [item["id"] for item in service.list_runs(scope="unfinished")])
            with self.assertRaises(ValueError):
                service.resume_run(run["id"])

    def test_running_pause_request_finishes_as_paused(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            def fake_runner(request, on_cell, should_cancel):
                service.pause_run(run["id"])
                self.assertEqual(should_cancel(), "pause")
                return {"status": "paused", "progress": 0}

            with patch("app.services.experiment_service.run_local_experiment", side_effect=fake_runner):
                finished = service.execute_run(run["id"])

            self.assertEqual(finished["status"], "paused")
            self.assertTrue(finished["cancelRequested"])

    def test_running_cancel_request_finishes_as_cancelled(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            def fake_runner(request, on_cell, should_cancel):
                service.cancel_run(run["id"])
                self.assertEqual(should_cancel(), "cancel")
                return {"status": "cancelled", "progress": 0}

            with patch("app.services.experiment_service.run_local_experiment", side_effect=fake_runner):
                finished = service.execute_run(run["id"])

            self.assertEqual(finished["status"], "cancelled")
            self.assertTrue(finished["cancelRequested"])
            self.assertNotIn(run["id"], [item["id"] for item in service.list_runs(scope="unfinished")])

    def test_runner_exception_preserves_recorded_progress(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])

            def fake_runner(request, on_cell, should_cancel):
                manifest = root / "cell_detection_manifest.json"
                manifest.write_text("[]", encoding="utf-8")
                on_cell(
                    {
                        "runId": run["id"],
                        "cellKey": "smoke__alg-traditional-spread-dct__atk-identity__0__42",
                        "status": "succeeded",
                        "datasetId": "smoke",
                        "algorithmId": "alg-traditional-spread-dct",
                        "watermarkMethod": "traditional-spread-dct",
                        "attackPresetId": "atk-identity",
                        "attackMethod": "identity",
                        "attackStrength": 0.0,
                        "seed": 42,
                        "sampleCount": 1,
                        "manifestPath": str(manifest),
                        "outputDir": str(root / "cell"),
                        "error": None,
                        "elapsedMs": 1.0,
                    }
                )
                raise RuntimeError("boom")

            with patch("app.services.experiment_service.run_local_experiment", side_effect=fake_runner):
                finished = service.execute_run(run["id"])

            self.assertEqual(finished["status"], "failed")
            self.assertEqual(finished["progress"], 100)
            self.assertIn("RuntimeError: boom", finished["error"])

    def test_run_results_use_sqlite_lifecycle_over_artifact_status(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])
            paused = service.pause_run(run["id"])
            summary_path = Path(paused["artifactRoot"]) / "run_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "runId": paused["id"],
                        "status": "cancelled",
                        "progress": 0,
                        "completedProgress": 0,
                        "progressKind": "completedCells",
                        "cells": [],
                    }
                ),
                encoding="utf-8",
            )

            results = service.get_run_results(run["id"])

            self.assertEqual(results["run"]["status"], "paused")
            self.assertEqual(results["summary"]["status"], "paused")
            self.assertEqual(results["summary"]["progress"], paused["progress"])

    def test_reconcile_stale_runs_marks_orphaned_running_as_paused(self) -> None:
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
                    "algorithmIds": ["alg-traditional-spread-dct"],
                    "attackPresetIds": ["atk-identity"],
                    "seeds": [42],
                    "maxSamples": 1,
                },
            )
            run = service.create_run(config["id"])
            stale_time = "2020-01-01T00:00:00+00:00"
            with service.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE experiment_runs
                    SET status = ?, worker_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    ("running", "dead-worker", stale_time, run["id"]),
                )

            reconciled = service.reconcile_stale_runs(stale_seconds=1)
            refreshed = service.get_run(run["id"])

            self.assertEqual(reconciled, 1)
            self.assertEqual(refreshed["status"], "paused")
            self.assertIn("auto-paused", refreshed["error"] or "")


if __name__ == "__main__":
    unittest.main()
