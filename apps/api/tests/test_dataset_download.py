from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.dataset_catalog import COMPACT_SAMPLE_COUNT, list_dataset_catalog
from app.services.dataset_download import DatasetDownloadService


class DatasetDownloadServiceTest(unittest.TestCase):
    def test_compact_download_creates_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets" / "shopee-product-matching"
            dataset_dir.mkdir(parents=True)
            for index in range(5):
                Image.new("RGB", (32, 32), (index * 10, 80, 120)).save(dataset_dir / f"img_{index:03d}.png")

            catalog = list_dataset_catalog(root)
            shopee = next(item for item in catalog if item["id"] == "shopee-product-matching")
            self.assertTrue(shopee["compactAvailable"])

            service = DatasetDownloadService(root)
            job = service.start_download("shopee-product-matching", mode="compact", sample_count=COMPACT_SAMPLE_COUNT)
            for _ in range(100):
                job = service.get_job(job.id)
                if job.status in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(0.05)

            self.assertEqual(job.status, "succeeded")
            self.assertTrue(job.archive_path)
            self.assertTrue(Path(job.archive_path).exists())

    def test_custom_download_samples_with_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets" / "shopee-product-matching" / "full"
            dataset_dir.mkdir(parents=True)
            for index in range(20):
                Image.new("RGB", (32, 32), (index * 5, 90, 140)).save(dataset_dir / f"full_{index:03d}.png")

            service = DatasetDownloadService(root)
            job = service.start_download(
                "shopee-product-matching",
                mode="custom",
                seed=7,
                sample_count=8,
            )
            for _ in range(100):
                job = service.get_job(job.id)
                if job.status in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(0.05)

            self.assertEqual(job.status, "succeeded")
            output_dir = Path(job.output_dir or "")
            self.assertTrue(output_dir.exists())
            self.assertEqual(len(list(output_dir.glob("*.png"))), 8)


if __name__ == "__main__":
    unittest.main()
