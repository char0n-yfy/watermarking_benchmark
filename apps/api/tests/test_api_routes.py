from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "FastAPI is not installed")
class ApiRoutesTest(unittest.TestCase):
    def test_post_run_queues_and_runtime_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), (120, 160, 200)).save(dataset_dir / "sample.png")

            os.environ["WM_BENCH_RESOURCES_ROOT"] = str(root / "resources")
            os.environ["WM_BENCH_RUNS_ROOT"] = str(root / "runs")
            os.environ["WM_BENCH_DB_PATH"] = str(root / "runs" / "wmbench.sqlite")

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
                        "algorithmIds": ["alg-traditional-lsb"],
                        "attackPresetIds": ["atk-identity"],
                        "seeds": [42],
                        "maxSamples": 1,
                    },
                },
            )
            self.assertEqual(config_response.status_code, 200)
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
            self.assertEqual(run_response.json()["status"], "queued")

            runtime_response = client.get("/system/runtime")
            self.assertEqual(runtime_response.status_code, 200)
            self.assertEqual(runtime_response.json()["device"], "cpu")

            cancel_response = client.post(f"/runs/{run_response.json()['id']}/cancel")
            self.assertEqual(cancel_response.status_code, 200)
            self.assertEqual(cancel_response.json()["status"], "cancelled")

            delete_response = client.delete(f"/experiment-configs/{config_id}")
            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual(delete_response.json()["status"], "deleted")
            list_response = client.get("/experiment-configs")
            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(list_response.json(), [])


if __name__ == "__main__":
    unittest.main()
