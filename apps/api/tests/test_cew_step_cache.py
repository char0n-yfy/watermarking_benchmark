from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.attacks.base import AttackContext
from evaluator.attacks.consumer_enhancement_workflow_attacks import attacks


class _DummyStepAttack:
    operation = "dummy"
    config = {"strength": 0.5}

    def __init__(self) -> None:
        self.calls = 0

    def _apply_images(self, images, contexts):
        self.calls += len(images)
        return [(image.copy(), {"backend": "dummy"}) for image in images]


class CewStepCacheTest(unittest.TestCase):
    def test_apply_step_batch_cached_skips_cached_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = AttackContext(
                run_id="run",
                sample_id="sample",
                attack_name="cew_c1",
                workspace_dir=root,
                device="cpu",
            )
            image = Image.new("RGB", (8, 8), (10, 20, 30))
            attack = _DummyStepAttack()
            with patch.dict(
                os.environ,
                {"WM_BENCH_CEW_STEP_CACHE": "1", "WM_BENCH_CEW_STEP_CACHE_MAX_ENTRIES": "4"},
                clear=False,
            ):
                attacks.clear_cew_step_cache()
                first = attacks._apply_step_batch_cached(
                    step_name="dummy",
                    attack=attack,
                    images=[image],
                    contexts=[context],
                )
                second = attacks._apply_step_batch_cached(
                    step_name="dummy",
                    attack=attack,
                    images=[image],
                    contexts=[context],
                )

            self.assertEqual(attack.calls, 1)
            self.assertFalse(first[0][1]["cew_step_cache_hit"])
            self.assertTrue(second[0][1]["cew_step_cache_hit"])
            self.assertEqual(attacks.cew_step_cache_stats()["entryCount"], 1)


if __name__ == "__main__":
    unittest.main()
