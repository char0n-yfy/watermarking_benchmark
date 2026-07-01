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
from app.services.attack_weights import (
    attack_method_weights_installed,
    is_attack_pack_marked_installed,
    mark_attack_pack_installed,
)


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
            cache_path = service.cache_root / "regen_diffusion__attack-weights.zip"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cache_path, "w", compression=zipfile.ZIP_STORED) as archive:
                checkpoint = root / "diffusion" / "sd2-1-base" / "model.safetensors"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"fake-checkpoint" + b"x" * 2048)
                archive.write(checkpoint, arcname="diffusion/sd2-1-base/model.safetensors")

            job = service.start_download("regen_diffusion")
            self._wait_for_job(service, job.id)
            job = service.get_job(job.id)

            install_dir = root / "weights" / "attacks" / "regeneration_attacks"
            self.assertEqual(job.status, "succeeded")
            self.assertEqual(job.weights_pack_id, "regen_diffusion")
            self.assertTrue(install_dir.exists())
            self.assertTrue((install_dir / "diffusion" / "sd2-1-base" / "model.safetensors").is_file())
            self.assertTrue(attack_method_weights_installed(root, "regen_diffusion"))
            self.assertFalse(attack_method_weights_installed(root, "2x_regen"))

    def test_merge_extract_does_not_remove_other_method_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = AttackWeightDownloadService(root)
            install_dir = root / "weights" / "attacks" / "regeneration_attacks"
            vae_checkpoint = install_dir / "vae" / "bmshj2018-factorized" / "checkpoint.pth.tar"
            vae_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            vae_checkpoint.write_bytes(b"vae" + b"x" * 2048)
            mark_attack_pack_installed(install_dir, "regen_vae")

            cache_path = service.cache_root / "regen_diffusion__attack-weights.zip"
            with zipfile.ZipFile(cache_path, "w", compression=zipfile.ZIP_STORED) as archive:
                checkpoint = root / "diffusion" / "sd2-1-base" / "model.safetensors"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"fake-checkpoint" + b"x" * 2048)
                archive.write(checkpoint, arcname="diffusion/sd2-1-base/model.safetensors")

            job = service.start_download("regen_diffusion")
            self._wait_for_job(service, job.id)
            job = service.get_job(job.id)

            self.assertEqual(job.status, "succeeded")
            self.assertTrue(vae_checkpoint.is_file())
            self.assertTrue(attack_method_weights_installed(root, "regen_vae"))
            self.assertTrue(attack_method_weights_installed(root, "regen_diffusion"))

    def test_shared_weights_can_be_enabled_for_second_pack_without_reextract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = AttackWeightDownloadService(root)
            install_dir = root / "weights" / "attacks" / "regeneration_attacks"
            diffusion_checkpoint = install_dir / "diffusion" / "sd2-1-base" / "model.safetensors"
            diffusion_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            diffusion_checkpoint.write_bytes(b"diffusion" + b"x" * 2048)
            mark_attack_pack_installed(install_dir, "regen_diffusion")

            job = service.start_download("2x_regen")
            self._wait_for_job(service, job.id)
            job = service.get_job(job.id)

            self.assertEqual(job.status, "succeeded")
            self.assertTrue(attack_method_weights_installed(root, "2x_regen"))
            self.assertTrue(diffusion_checkpoint.is_file())

    def test_uninstall_only_removes_method_markers_and_unrefed_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = AttackWeightDownloadService(root)
            install_dir = root / "weights" / "attacks" / "regeneration_attacks"
            vae_checkpoint = install_dir / "vae" / "bmshj2018-factorized" / "checkpoint.pth.tar"
            vae_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            vae_checkpoint.write_bytes(b"vae" + b"x" * 2048)
            mark_attack_pack_installed(install_dir, "regen_vae")

            diffusion_checkpoint = install_dir / "diffusion" / "sd2-1-base" / "model.safetensors"
            diffusion_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            diffusion_checkpoint.write_bytes(b"diffusion" + b"x" * 2048)
            mark_attack_pack_installed(install_dir, "regen_diffusion")

            service.uninstall("regen_diffusion")

            self.assertFalse(attack_method_weights_installed(root, "regen_diffusion"))
            self.assertTrue(attack_method_weights_installed(root, "regen_vae"))
            self.assertFalse(diffusion_checkpoint.exists())

    def test_uninstall_keeps_shared_weights_when_other_pack_still_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = AttackWeightDownloadService(root)
            install_dir = root / "weights" / "attacks" / "regeneration_attacks"
            diffusion_checkpoint = install_dir / "diffusion" / "sd2-1-base" / "model.safetensors"
            diffusion_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            diffusion_checkpoint.write_bytes(b"diffusion" + b"x" * 2048)
            mark_attack_pack_installed(install_dir, "regen_diffusion")
            mark_attack_pack_installed(install_dir, "2x_regen")

            service.uninstall("regen_diffusion")

            self.assertFalse(is_attack_pack_marked_installed(install_dir, "regen_diffusion"))
            self.assertTrue(attack_method_weights_installed(root, "2x_regen"))
            self.assertTrue(diffusion_checkpoint.is_file())

    def test_rejects_attacks_without_packaged_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = AttackWeightDownloadService(Path(tmp))
            with self.assertRaises(ValueError):
                service.start_download("jpeg")


if __name__ == "__main__":
    unittest.main()
