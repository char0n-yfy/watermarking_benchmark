from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.config import get_settings
from app.core.local_db import LocalDatabase
from app.services.experiment_service import ExperimentService
from app.services.runtime_parallel_config import write_runtime_parallel_env
from apps.worker.local_worker import run_once


class LocalWorkerTest(unittest.TestCase):
    def test_worker_claims_and_finishes_queued_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "resources" / "datasets" / "smoke"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (300, 300), (120, 160, 200)).save(dataset_dir / "sample.png")

            os.environ["WM_BENCH_RESOURCES_ROOT"] = str(root / "resources")
            os.environ["WM_BENCH_RUNS_ROOT"] = str(root / "runs")
            os.environ["WM_BENCH_DB_PATH"] = str(root / "runs" / "wmbench.sqlite")
            os.environ["WM_BENCH_DEVICE"] = "cpu"
            get_settings.cache_clear()

            service = ExperimentService(
                database=LocalDatabase(root / "runs" / "wmbench.sqlite"),
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
            queued = service.create_run(config["id"])
            previous_png_level = os.environ.pop("WM_BENCH_PNG_COMPRESS_LEVEL", None)
            write_runtime_parallel_env(
                root / "runs",
                {"WM_BENCH_PNG_COMPRESS_LEVEL": "3"},
                job_id="test-tuning",
                env_path=root / ".env.autodl",
            )

            applied_png_level = None
            try:
                processed = run_once("test-worker")
                finished = service.get_run(queued["id"])
                applied_png_level = os.environ.get("WM_BENCH_PNG_COMPRESS_LEVEL")
            finally:
                if previous_png_level is None:
                    os.environ.pop("WM_BENCH_PNG_COMPRESS_LEVEL", None)
                else:
                    os.environ["WM_BENCH_PNG_COMPRESS_LEVEL"] = previous_png_level

            self.assertEqual(processed, 1)
            self.assertEqual(finished["status"], "succeeded")
            self.assertTrue(Path(finished["logPath"]).exists())
            self.assertEqual(applied_png_level, "3")


if __name__ == "__main__":
    unittest.main()
