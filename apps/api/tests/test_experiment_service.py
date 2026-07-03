from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.local_db import LocalDatabase
from app.services.experiment_service import ExperimentService
from app.services.local_artifacts import (
    RunStateWriter,
    artifact_paths,
    compact_result_units_file,
    write_jsonl,
)


class ExperimentServiceTest(unittest.TestCase):
    def test_result_unit_jsonl_compaction_keeps_latest_record_per_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result_units.jsonl"
            write_jsonl(
                path,
                [
                    {"resultUnitKey": "unit-a", "status": "failed", "error": "timeout"},
                    {"resultUnitKey": "unit-b", "status": "succeeded"},
                    {"resultUnitKey": "unit-a", "status": "succeeded", "error": None},
                ],
            )

            compacted = compact_result_units_file(path)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(compacted), 2)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["resultUnitKey"], "unit-a")
            self.assertEqual(records[0]["status"], "succeeded")
            self.assertEqual(records[1]["resultUnitKey"], "unit-b")

    def test_phase_progress_is_clamped_to_declared_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_clamp"
            paths = artifact_paths(run_root)
            writer = RunStateWriter(
                paths=paths,
                run_id="run_clamp",
                run_root=run_root,
                selection={},
                expected_result_units=1,
                materialized_root=run_root / "materialized",
            )

            writer.phase_start("attack", total=10)
            writer.phase_advance("attack", current=12, counters={"imagesDone": 12})
            state = json.loads(paths["runState"].read_text(encoding="utf-8"))
            attack_phase = next(phase for phase in state["phases"] if phase["key"] == "attack")

            self.assertEqual(attack_phase["current"], 10)
            self.assertEqual(attack_phase["percent"], 100)

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

            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["completedProgress"], 100)
            self.assertEqual(finished["progressKind"], "phaseOperations")
            self.assertEqual(finished["currentPhase"], "summary")
            state = service.get_run_state(run["id"])
            tree = service.get_run_tree(run["id"])
            self.assertEqual(state["progressKind"], "phaseOperations")
            self.assertEqual(state["currentPhase"], "summary")
            self.assertEqual([phase["key"] for phase in state["phases"]], [
                "canonical",
                "watermark_embed",
                "attack",
                "watermark_extract",
                "quality",
                "summary",
            ])
            self.assertIn("smoke", tree["datasets"])
            self.assertNotIn(run["id"], [item["id"] for item in service.list_runs(scope="active")])
            self.assertEqual(len(results["resultUnits"]), 1)
            self.assertEqual(results["resultUnits"][0]["status"], "succeeded")
            extract_manifest = json.loads(Path(results["resultUnits"][0]["manifestPath"]).read_text())
            self.assertEqual(extract_manifest[0]["decodedBits"], extract_manifest[0]["expectedBits"])
            self.assertNotIn("bitAccuracy", results["resultUnits"][0])
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

            def fake_runner(request, on_state, should_cancel):
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

            def fake_runner(request, on_state, should_cancel):
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

            def fake_runner(request, on_state, should_cancel):
                on_state({"overallProgress": 100, "currentPhase": "attack", "progressKind": "phaseOperations"})
                raise RuntimeError("boom")

            with patch("app.services.experiment_service.run_local_experiment", side_effect=fake_runner):
                finished = service.execute_run(run["id"])

            self.assertEqual(finished["status"], "failed")
            self.assertEqual(finished["progress"], 100)
            self.assertIn("RuntimeError: boom", finished["error"])

    def test_run_result_units_are_read_from_authoritative_jsonl(self) -> None:
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
            manifest = root / "cell_detection_manifest.json"
            manifest.write_text("[]", encoding="utf-8")
            cell = {
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

            result_unit = {
                **cell,
                "resultUnitKey": cell["cellKey"],
            }

            def fake_runner(request, on_state, should_cancel):
                result_units_path = Path(request.runs_root) / request.run_id / "result_units.jsonl"
                result_units_path.parent.mkdir(parents=True, exist_ok=True)
                result_units_path.write_text(json.dumps(result_unit) + "\n", encoding="utf-8")
                on_state({"overallProgress": 100, "currentPhase": "summary", "progressKind": "phaseOperations"})
                return {
                    "status": "succeeded",
                    "progress": 100,
                    "progressKind": "phaseOperations",
                    "resultUnits": [result_unit],
                    "resultUnitCount": 1,
                }

            with patch("app.services.experiment_service.run_local_experiment", side_effect=fake_runner):
                finished = service.execute_run(run["id"])

            result_units = service.list_run_result_units(run["id"])
            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(len(result_units), 1)
            self.assertEqual(result_units[0]["resultUnitKey"], cell["cellKey"])

    def test_execute_run_heartbeat_prevents_stale_reconcile_during_long_stage(self) -> None:
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

            def fake_runner(request, on_state, should_cancel):
                for _ in range(20):
                    workers = service.list_worker_heartbeats()
                    if any(worker["currentRunId"] == run["id"] for worker in workers):
                        break
                    time.sleep(0.05)
                with service.database.connect() as connection:
                    connection.execute(
                        """
                        UPDATE experiment_runs
                        SET updated_at = ?
                        WHERE id = ?
                        """,
                        ("2020-01-01T00:00:00+00:00", run["id"]),
                    )
                self.assertEqual(service.reconcile_stale_runs(stale_seconds=0), 0)
                on_state({"overallProgress": 100, "currentPhase": "summary", "progressKind": "phaseOperations"})
                return {"status": "succeeded", "progress": 100, "progressKind": "phaseOperations", "resultUnits": []}

            with patch("app.services.experiment_service.run_local_experiment", side_effect=fake_runner):
                finished = service.execute_run(run["id"], worker_id="worker-a")

            self.assertEqual(finished["status"], "succeeded")

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
                        "progressKind": "phaseOperations",
                        "resultUnits": [],
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
