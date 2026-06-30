from __future__ import annotations

import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.weight_download import WeightDownloadService


class WeightDownloadServiceTest(unittest.TestCase):
    def _wait_for_job(self, service: WeightDownloadService, job_id: str) -> None:
        for _ in range(100):
            job = service.get_job(job_id)
            if job.status in {"succeeded", "failed"}:
                return
            time.sleep(0.05)
        self.fail(f"Weight download job did not finish: {job_id}")

    def test_extracts_cached_archive_into_weights_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = WeightDownloadService(root)
            cache_path = service.cache_root / "hidden__weights.zip"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cache_path, "w", compression=zipfile.ZIP_STORED) as archive:
                checkpoint = root / "decoder.pt"
                checkpoint.write_bytes(b"fake-checkpoint" + b"x" * 2048)
                archive.write(checkpoint, arcname="decoder.pt")

            job = service.start_download("hidden")
            self._wait_for_job(service, job.id)
            job = service.get_job(job.id)

            self.assertEqual(job.status, "succeeded")
            install_dir = Path(job.output_dir or "")
            self.assertTrue(install_dir.exists())
            self.assertEqual(install_dir, root / "weights" / "watermarking" / "hidden")
            self.assertTrue((install_dir / "decoder.pt").is_file())
            self.assertEqual(str(job.archive_path), str(cache_path))

    def test_returns_succeeded_immediately_when_already_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "weights" / "watermarking" / "hidden"
            install_dir.mkdir(parents=True)
            (install_dir / "decoder.pt").write_bytes(b"existing")

            service = WeightDownloadService(root)
            job = service.start_download("hidden")

            self.assertEqual(job.status, "succeeded")
            self.assertGreaterEqual(job.progress, 1)

    def test_skips_when_already_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "weights" / "watermarking" / "hidden"
            install_dir.mkdir(parents=True)
            (install_dir / "decoder.pt").write_bytes(b"existing")

            service = WeightDownloadService(root)
            job = service.start_download("hidden")
            self._wait_for_job(service, job.id)
            job = service.get_job(job.id)

            self.assertEqual(job.status, "succeeded")
            self.assertGreaterEqual(job.progress, 1)

    def test_rejects_methods_without_packaged_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = WeightDownloadService(Path(tmp))
            with self.assertRaises(ValueError):
                service.start_download("invisible-watermark-dwtdct")

    def test_uninstall_removes_installed_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "weights" / "watermarking" / "hidden"
            install_dir.mkdir(parents=True)
            (install_dir / "decoder.pt").write_bytes(b"existing")

            service = WeightDownloadService(root)
            result = service.uninstall("hidden")

            self.assertFalse(result["installed"])
            self.assertEqual(result["weightsDir"], "hidden")
            self.assertTrue(install_dir.exists())
            self.assertFalse(any(install_dir.iterdir()))

    def test_uninstall_raises_when_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = WeightDownloadService(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                service.uninstall("hidden")


if __name__ == "__main__":
    unittest.main()
