from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.attack_weights import (
    attack_method_can_install,
    attack_method_weights_installed,
    enrich_attack_resource,
    is_attack_pack_marked_installed,
    mark_attack_pack_installed,
)


class AttackWeightsInstallStateTest(unittest.TestCase):
    def test_viewpoint_motion_install_state_is_per_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "weights" / "attacks" / "3d_viewpoint_rerendering"
            checkpoint = storage / "checkpoints" / "sharp_2572gikvuh.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"x" * 2048)
            mark_attack_pack_installed(storage, "3d_viewpoint_rerendering_swipe")

            swipe = "3d_viewpoint_rerendering_swipe_phase0_point"
            shake = "3d_viewpoint_rerendering_shake_phase0_point"

            self.assertTrue(attack_method_weights_installed(root, swipe))
            self.assertFalse(attack_method_weights_installed(root, shake))

    def test_enrich_reports_installed_only_for_marked_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "weights" / "attacks" / "3d_viewpoint_rerendering"
            checkpoint = storage / "checkpoints" / "sharp_2572gikvuh.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"x" * 2048)
            mark_attack_pack_installed(storage, "3d_viewpoint_rerendering_swipe")

            swipe_item = enrich_attack_resource(
                {"method": "3d_viewpoint_rerendering_swipe_phase0_point"},
                root,
                oss=None,
                probe_remote=False,
            )
            shake_item = enrich_attack_resource(
                {"method": "3d_viewpoint_rerendering_shake_phase0_point"},
                root,
                oss=None,
                probe_remote=False,
            )

            self.assertTrue(swipe_item["weightsInstalled"])
            self.assertFalse(shake_item["weightsInstalled"])
            self.assertTrue(shake_item["weightsDownloadReady"])

    def test_stale_marker_does_not_block_remote_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "weights" / "attacks" / "consumer_enhancement_workflow_attacks"
            storage.mkdir(parents=True, exist_ok=True)
            mark_attack_pack_installed(storage, "cew_c1")

            self.assertFalse(attack_method_weights_installed(root, "cew_c1"))
            self.assertTrue(is_attack_pack_marked_installed(storage, "cew_c1"))
            self.assertTrue(attack_method_can_install(root, "cew_c1", remote_available=True))

            enrich_attack_resource({"method": "cew_c1"}, root, oss=None, probe_remote=False)
            self.assertFalse(is_attack_pack_marked_installed(storage, "cew_c1"))


if __name__ == "__main__":
    unittest.main()
