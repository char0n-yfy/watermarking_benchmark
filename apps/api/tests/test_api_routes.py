from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "FastAPI is not installed")
class ApiRoutesTest(unittest.TestCase):
    def test_resource_weight_download_jobs_can_be_listed_by_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            os.environ["WM_BENCH_RESOURCES_ROOT"] = str(root / "resources")
            os.environ["WM_BENCH_RUNS_ROOT"] = str(root / "runs")
            os.environ["WM_BENCH_DB_PATH"] = str(root / "runs" / "wmbench.sqlite")

            from app.core.config import get_settings

            get_settings.cache_clear()
            from app.main import create_app
            from fastapi.testclient import TestClient

            client = TestClient(create_app())

            watermark_job = client.post("/resources/watermarks/alg-hidden/downloads")
            self.assertEqual(watermark_job.status_code, 200)
            watermark_jobs = client.get("/resources/watermarks/alg-hidden/downloads")
            self.assertEqual(watermark_jobs.status_code, 200)
            self.assertIn(watermark_job.json()["id"], [job["id"] for job in watermark_jobs.json()])

            attack_job = client.post("/resources/attacks/atk-regen-diffusion/downloads")
            self.assertEqual(attack_job.status_code, 200)
            attack_jobs = client.get("/resources/attacks/atk-regen-diffusion/downloads")
            self.assertEqual(attack_jobs.status_code, 200)
            self.assertIn(attack_job.json()["id"], [job["id"] for job in attack_jobs.json()])

    def test_dataset_download_jobs_can_be_listed_by_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            os.environ["WM_BENCH_RESOURCES_ROOT"] = str(root / "resources")
            os.environ["WM_BENCH_RUNS_ROOT"] = str(root / "runs")
            os.environ["WM_BENCH_DB_PATH"] = str(root / "runs" / "wmbench.sqlite")

            from app.core.config import get_settings

            get_settings.cache_clear()
            from app.main import create_app
            from fastapi.testclient import TestClient

            client = TestClient(create_app())

            created = client.post(
                "/resources/datasets/ms-coco/downloads",
                json={"mode": "custom", "seed": 7, "sampleCount": 2},
            )
            self.assertEqual(created.status_code, 200)
            listed = client.get("/resources/datasets/ms-coco/downloads")
            self.assertEqual(listed.status_code, 200)
            self.assertIn(created.json()["id"], [job["id"] for job in listed.json()])

    def test_post_run_queues_and_runtime_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")
            Image.new("RGB", (320, 240), (90, 110, 160)).save(dataset_dir / "sample_2.png")

            os.environ["WM_BENCH_RESOURCES_ROOT"] = str(root / "resources")
            os.environ["WM_BENCH_RUNS_ROOT"] = str(root / "runs")
            os.environ["WM_BENCH_DB_PATH"] = str(root / "runs" / "wmbench.sqlite")
            os.environ["WM_BENCH_DOTENV_PATH"] = str(root / ".env.autodl")

            from app.core.config import get_settings

            get_settings.cache_clear()
            from app.main import create_app
            from fastapi.testclient import TestClient

            client = TestClient(create_app())
            config_response = client.post(
                "/experiment-configs",
                json={
                    "name": "Smoke",
                    "selection": {
                        "datasetIds": ["smoke"],
                        "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                },
            )
            self.assertEqual(config_response.status_code, 200)
            self.assertIn("charset=utf-8", config_response.headers.get("content-type", "").lower())
            config_id = config_response.json()["id"]

            rename_response = client.patch(
                f"/experiment-configs/{config_id}",
                json={"name": "Renamed smoke"},
            )
            self.assertEqual(rename_response.status_code, 200)
            self.assertEqual(rename_response.json()["name"], "Renamed smoke")

            run_response = client.post(
                "/runs",
                json={"configId": config_id},
            )
            self.assertEqual(run_response.status_code, 200)
            self.assertIn("charset=utf-8", run_response.headers.get("content-type", "").lower())
            self.assertEqual(run_response.json()["status"], "queued")
            run_id = run_response.json()["id"]

            state_response = client.get(f"/runs/{run_id}/state")
            self.assertEqual(state_response.status_code, 200)
            self.assertEqual(
                [phase["key"] for phase in state_response.json()["phases"]],
                ["canonical", "watermark_embed", "attack", "watermark_extract", "quality", "summary"],
            )
            tree_response = client.get(f"/runs/{run_id}/tree")
            self.assertEqual(tree_response.status_code, 200)
            self.assertEqual(tree_response.json()["datasets"], {})

            if (Path(__file__).resolve().parents[3] / "apps" / "web" / "out").exists():
                runs_page_response = client.get("/runs", headers={"accept": "text/html"})
                self.assertEqual(runs_page_response.status_code, 200)
                self.assertIn("text/html", runs_page_response.headers.get("content-type", "").lower())

            runtime_response = client.get("/system/runtime")
            self.assertEqual(runtime_response.status_code, 200)
            self.assertIn("charset=utf-8", runtime_response.headers.get("content-type", "").lower())
            self.assertEqual(runtime_response.json()["device"], "cpu")

            readiness_response = client.get("/system/readiness")
            self.assertEqual(readiness_response.status_code, 200)
            readiness = readiness_response.json()
            self.assertIn(readiness["status"], {"ready", "degraded"})
            check_ids = {check["id"] for check in readiness["checks"]}
            self.assertIn("sqlite", check_ids)
            self.assertIn("resource_catalog", check_ids)
            self.assertIn("worker_heartbeat", check_ids)

            protocols_response = client.get("/benchmark-protocols")
            self.assertEqual(protocols_response.status_code, 200)
            self.assertEqual(protocols_response.json()[0]["id"], "waves-official-detection-v1")

            leaderboard_response = client.get("/leaderboard?protocol_id=waves-official-detection-v1")
            self.assertEqual(leaderboard_response.status_code, 200)
            self.assertEqual(leaderboard_response.json()["protocol"]["id"], "waves-official-detection-v1")

            tuning_response = client.post(
                "/system/parallel-tuning",
                json={
                    "sampleCount": 2,
                    "tuneWatermarks": False,
                    "tuneAttacks": False,
                    "tuneQuality": False,
                },
            )
            self.assertEqual(tuning_response.status_code, 200)
            tuning_id = tuning_response.json()["id"]
            tuning_state = tuning_response.json()
            for _attempt in range(20):
                if tuning_state["status"] != "running":
                    break
                time.sleep(0.05)
                tuning_state = client.get(f"/system/parallel-tuning/{tuning_id}").json()
            self.assertEqual(tuning_state["status"], "succeeded")
            self.assertIn("summary", tuning_state)
            latest_tuning_response = client.get("/system/parallel-tuning/latest")
            self.assertEqual(latest_tuning_response.status_code, 200)
            self.assertEqual(latest_tuning_response.json()["id"], tuning_id)
            save_tuning_response = client.post(f"/system/parallel-tuning/{tuning_id}/save")
            self.assertEqual(save_tuning_response.status_code, 200)
            saved_tuning = save_tuning_response.json()
            self.assertEqual(saved_tuning["envPath"], str(root / ".env.autodl"))
            self.assertEqual(
                Path(saved_tuning["runtimePath"]).resolve(),
                (root / "runs" / "parallel_tuning" / "active_env.json").resolve(),
            )
            self.assertIn("WM_BENCH_PNG_COMPRESS_LEVEL", saved_tuning["savedKeys"])
            self.assertEqual(os.environ["WM_BENCH_PNG_COMPRESS_LEVEL"], "1")
            active_runtime = json.loads(Path(saved_tuning["runtimePath"]).read_text(encoding="utf-8"))
            self.assertEqual(active_runtime["jobId"], tuning_id)
            self.assertEqual(active_runtime["envUpdates"]["WM_BENCH_PNG_COMPRESS_LEVEL"], "1")

            pause_response = client.post(f"/runs/{run_id}/pause")
            self.assertEqual(pause_response.status_code, 200)
            self.assertEqual(pause_response.json()["status"], "paused")

            unfinished_response = client.get("/runs?scope=unfinished")
            self.assertEqual(unfinished_response.status_code, 200)
            self.assertIn(run_id, [item["id"] for item in unfinished_response.json()])

            cancel_run_response = client.post(
                "/runs",
                json={"configId": config_id},
            )
            self.assertEqual(cancel_run_response.status_code, 200)
            cancel_response = client.post(f"/runs/{cancel_run_response.json()['id']}/cancel")
            self.assertEqual(cancel_response.status_code, 200)
            self.assertEqual(cancel_response.json()["status"], "cancelled")
            resume_cancelled_response = client.post(f"/runs/{cancel_run_response.json()['id']}/resume")
            self.assertEqual(resume_cancelled_response.status_code, 400)

            delete_response = client.delete(f"/experiment-configs/{config_id}")
            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual(delete_response.json()["status"], "deleted")
            list_response = client.get("/experiment-configs")
            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(list_response.json(), [])


if __name__ == "__main__":
    unittest.main()
