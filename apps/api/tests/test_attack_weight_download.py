from __future__ import annotations

import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.attack_weight_download import AttackWeightDownloadService


class AttackWeightDownloadServiceTest(unittest.TestCase):
    def _wait_for_job(self, service: AttackWeightDownloadService, job_id: str) -> None:
        for _ in range(100):
            job = service.get_job(job_id)
            if job.status in {"succeeded", "failed"}:
                return
            time.sleep(0.05)
        self.fail(f"Attack weight download job did not finish: {job_id}")

    def test_extracts_cached_archive_into_attack_weights_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = AttackWeightDownloadService(root)
            cache_path = service.cache_root / "regeneration_attacks__attack-weights.zip"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cache_path, "w", compression=zipfile.ZIP_STORED) as archive:
                checkpoint = root / "diffusion" / "sd2-1-base" / "model.safetensors"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"fake-checkpoint" + b"x" * 2048)
                archive.write(checkpoint, arcname="diffusion/sd2-1-base/model.safetensors")

            job = service.start_download("regen_diffusion")
            self._wait_for_job(service, job.id)
            job = service.get_job(job.id)

            self.assertEqual(job.status, "succeeded")
            install_dir = Path(job.output_dir or "")
            self.assertTrue(install_dir.exists())
            self.assertEqual(install_dir, root / "weights" / "attacks" / "regeneration_attacks")
            self.assertTrue((install_dir / "diffusion" / "sd2-1-base" / "model.safetensors").is_file())

    def test_rejects_attacks_without_packaged_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = AttackWeightDownloadService(Path(tmp))
            with self.assertRaises(ValueError):
                service.start_download("jpeg")


if __name__ == "__main__":
    unittest.main()
