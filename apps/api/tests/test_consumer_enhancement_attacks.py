from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.attacks.runner import AttackJob, run_attack_dir


class ConsumerEnhancementAttackTest(unittest.TestCase):
    def test_cew_weight_manifest_tracks_verified_downloads(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[3]
            / "resources"
            / "weights"
            / "attacks"
            / "consumer_enhancement_workflow_attacks"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {entry["resourceId"]: entry for entry in manifest["entries"]}
        download_files = []
        for entry in entries.values():
            if entry.get("downloadUrl"):
                download_files.append(entry)
            download_files.extend(entry.get("files", []))

        self.assertEqual(manifest["schemaVersion"], 3)
        self.assertEqual(len(entries), 11)
        self.assertEqual(len(download_files), 12)
        self.assertIn("downloadUrl", entries["cew_d2"])
        self.assertEqual(entries["cew_d3"]["files"][0]["filename"], "classifier.pth")
        self.assertEqual(entries["cew_d3"]["files"][1]["filename"], "LUTs.pth")
        self.assertEqual(entries["cew_s3_scale2"]["attackId"], "cew_s3")
        self.assertEqual(entries["cew_s3_scale2"]["params"], {"scale": 2})
        self.assertEqual(entries["cew_s3_scale2"]["model"], "bsrgan_x2")
        self.assertEqual(entries["cew_d4"]["filename"], "LOL_v1.pth")
        for file_entry in download_files:
            self.assertGreater(file_entry["expectedSizeBytes"], 0)
            self.assertRegex(file_entry["sha256"], r"^[0-9a-f]{64}$")

    def test_representative_cew_attacks_run_through_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()

            array = np.zeros((32, 40, 3), dtype=np.uint8)
            array[:, :, 0] = np.linspace(20, 220, 40, dtype=np.uint8)[None, :]
            array[:, :, 1] = np.linspace(30, 200, 32, dtype=np.uint8)[:, None]
            array[:, :, 2] = 90
            Image.fromarray(array, mode="RGB").save(input_dir / "sample.png")
            empty_weight_root = root / "empty_weights"

            expected_sizes = {
                "cew_e1": (40, 32),
                "cew_d5": (40, 32),
                "cew_s1": (80, 64),
                "cew_c1": (80, 64),
            }

            for attack_name, expected_size in expected_sizes.items():
                with self.subTest(attack_name=attack_name):
                    output_dir = root / "out" / attack_name
                    params = (
                        {"weight_root": str(empty_weight_root)}
                        if attack_name.startswith(("cew_d", "cew_s", "cew_c"))
                        else {}
                    )
                    if attack_name == "cew_e1":
                        params["strength"] = "medium"
                    if attack_name == "cew_s1":
                        params["scale"] = 2
                    results = run_attack_dir(
                        AttackJob(
                            run_id="cew_test",
                            attack_name=attack_name,
                            params=params,
                            input_dir=input_dir,
                            output_dir=output_dir,
                            device="cpu",
                            seed=7,
                        )
                    )

                    self.assertEqual(len(results), 1)
                    self.assertTrue(results[0].ok, results[0].error)
                    self.assertTrue(results[0].output_path.exists())
                    self.assertEqual(results[0].metadata["inputSize"], [40, 32])
                    self.assertEqual(results[0].metadata["outputSize"], list(expected_size))
                    if expected_size == (40, 32):
                        self.assertTrue(results[0].metadata["sizePreserving"])
                        self.assertEqual(results[0].metadata["sizePolicy"], "preserve_input_size")
                    else:
                        self.assertFalse(results[0].metadata["sizePreserving"])
                        self.assertTrue(results[0].metadata["sizeChangeSemantic"])
                        self.assertEqual(results[0].metadata["sizePolicy"], "semantic_size_change")
                    with Image.open(results[0].output_path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertEqual(image.size, expected_size)

                    manifest = json.loads((output_dir / "attack_manifest.json").read_text())
                    self.assertEqual(manifest[0]["attack_name"], attack_name)
                    self.assertTrue(manifest[0]["ok"])
                    self.assertEqual(manifest[0]["metadata"]["inputSize"], [40, 32])
                    self.assertEqual(manifest[0]["metadata"]["outputSize"], list(expected_size))
                    self.assertEqual(manifest[0]["metadata"]["execution"]["stage"], "attack")
                    self.assertIn(manifest[0]["metadata"]["execution"]["mode"], {"serial", "threadpool", "batch"})

    def test_edit_strength_and_sr_scale_are_runtime_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.fromarray(np.full((16, 20, 3), 96, dtype=np.uint8), mode="RGB").save(input_dir / "sample.png")
            empty_weight_root = root / "empty_weights"

            edit_result = run_attack_dir(
                AttackJob(
                    run_id="cew_edit_strength_test",
                    attack_name="cew_e2",
                    params={"strength": "strong"},
                    input_dir=input_dir,
                    output_dir=root / "edit",
                    device="cpu",
                )
            )[0]
            self.assertTrue(edit_result.ok, edit_result.error)
            self.assertEqual(edit_result.metadata["strength"], 1.0)

            sr_result = run_attack_dir(
                AttackJob(
                    run_id="cew_sr_scale_test",
                    attack_name="cew_s3",
                    params={"scale": 4, "weight_root": str(empty_weight_root)},
                    input_dir=input_dir,
                    output_dir=root / "sr",
                    device="cpu",
                )
            )[0]
            self.assertTrue(sr_result.ok, sr_result.error)
            self.assertEqual(sr_result.metadata["model_name"], "bsrgan_x4")
            self.assertEqual(sr_result.metadata["scale"], 4)
            self.assertEqual(sr_result.metadata["inputSize"], [20, 16])
            self.assertEqual(sr_result.metadata["outputSize"], [80, 64])
            self.assertTrue(sr_result.metadata["sizeChangeSemantic"])
            with Image.open(sr_result.output_path) as image:
                self.assertEqual(image.size, (80, 64))

    def test_downloaded_d2_d3_weights_run_without_fallback(self) -> None:
        weight_root = (
            Path(__file__).resolve().parents[3]
            / "resources"
            / "weights"
            / "attacks"
            / "consumer_enhancement_workflow_attacks"
        )
        required = [
            weight_root / "deep_enhance" / "deepwb_awb" / "net_awb.pth",
            weight_root / "deep_enhance" / "image_adaptive_3dlut_fivek" / "classifier.pth",
            weight_root / "deep_enhance" / "image_adaptive_3dlut_fivek" / "LUTs.pth",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("downloaded CEW-D2/D3 checkpoints are not present")
        if importlib.util.find_spec("einops") is None:
            self.skipTest("einops is required for downloaded CEW-D2/D3 torch backends")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.fromarray(np.full((16, 20, 3), 96, dtype=np.uint8), mode="RGB").save(input_dir / "sample.png")

            checks = [
                ("cew_d2", {"allow_fallback": False, "max_size": 64}),
                ("cew_d3", {"allow_fallback": False}),
            ]
            for attack_name, params in checks:
                with self.subTest(attack_name=attack_name):
                    result = run_attack_dir(
                        AttackJob(
                            run_id="cew_d_real_backend_test",
                            attack_name=attack_name,
                            params=params,
                            input_dir=input_dir,
                            output_dir=root / attack_name,
                            device="cpu",
                        )
                    )[0]
                    self.assertTrue(result.ok, result.error)
                    self.assertFalse(result.metadata["fallback_used"])
                    self.assertTrue(str(result.metadata["backend"]).startswith("torch_"))
                    with Image.open(result.output_path) as image:
                        self.assertEqual(image.size, (20, 16))


if __name__ == "__main__":
    unittest.main()
